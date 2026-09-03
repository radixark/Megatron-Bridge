# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM-5.3-Flash flavour of megatron-core's mHC hyper-connections.

Two deviations from the megatron-core defaults, both taken from the GLM-5.3 reference
implementation (radixark/miles ``miles_plugins/models/glm5_next/glm5_next.py``):

* the per-layer mapping projection normalizes its input with the model's ``rms_norm_eps``
  (1e-5) instead of the hard-coded 1e-6, using the plain (unfused) ``proj_rms`` formula;
* the block-level output contraction is a plain mean over the residual streams -- GLM-5.3
  has no learned ``hc_head_*`` contraction parameters.

The contraction plugs into the ``hc_head_contraction`` slot of
``TransformerBlockSubmodules`` (radixark/Megatron-LM#89). On a megatron-core without that
slot :func:`install_mean_output_contract` patches the block instead.
"""

import logging

import torch
from megatron.core.transformer import transformer_block as _tb
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.module import MegatronModule


logger = logging.getLogger(__name__)

# megatron-core's _MHC_SINKHORN_EPS / _MHC_COMPUTE_H_EPS; the HF config's hc_eps must match.
MHC_EPS = 1e-6

_HC_HEAD_PARAM_NAMES = ("hc_head_fn", "hc_head_base", "hc_head_scale")


def _reference_proj_rms(x: torch.Tensor, weight: torch.Tensor, eps: float):
    proj = torch.matmul(x, weight.t())
    r = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return proj, r


class Glm5NextHyperConnectionModule(HyperConnectionModule):
    """``HyperConnectionModule`` with GLM-5.3's normalization eps and reference ``proj_rms``."""

    def __init__(self, config, layer_number: int):
        super().__init__(config, layer_number)
        assert not config.use_fused_mhc, "GLM-5.3 mHC requires the native (unfused) proj_rms path"
        self.norm_eps = config.layernorm_epsilon
        self._proj_rms_op = torch.compile(_reference_proj_rms)


class MeanStreamContraction(MegatronModule):
    """Parameter-free ``[s, b, n*C] -> [s, b, C]`` mean over the ``n`` residual streams."""

    def __init__(self, config):
        super().__init__(config)
        self.num_residual_streams = config.num_residual_streams

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return HyperConnectionModule.output_contract(hidden_states, self.num_residual_streams)


def supports_head_contraction_slot() -> bool:
    """True when megatron-core exposes ``TransformerBlockSubmodules.hc_head_contraction``."""
    return "hc_head_contraction" in getattr(_tb.TransformerBlockSubmodules, "__dataclass_fields__", {})


def install_mean_output_contract() -> None:
    """Fallback for megatron-core without the ``hc_head_contraction`` slot.

    Replaces the learned block-level contraction with the stream mean and demotes the
    ``hc_head_*`` parameters that ``TransformerBlock`` still allocates to plain tensors, so
    they neither train nor appear in state dicts. Idempotent.
    """
    if getattr(_tb, "_glm5_next_mean_contract_patched", False):
        return
    logger.warning(
        "megatron-core has no TransformerBlockSubmodules.hc_head_contraction slot; patching "
        "learned_output_contract to the GLM-5.3 stream mean (see radixark/Megatron-LM#89)."
    )

    def _mean_output_contract(hidden_states, head_fn, base, scale, n, eps):
        return HyperConnectionModule.output_contract(hidden_states, n)

    _tb.learned_output_contract = _mean_output_contract

    original_init = _tb.TransformerBlock.__init__

    def _init_and_demote_hc_head(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for param_name in _HC_HEAD_PARAM_NAMES:
            param = self._parameters.pop(param_name, None)
            if param is not None:
                setattr(self, param_name, param.data)

    _tb.TransformerBlock.__init__ = _init_and_demote_hc_head
    _tb._glm5_next_mean_contract_patched = True


__all__ = [
    "MHC_EPS",
    "Glm5NextHyperConnectionModule",
    "MeanStreamContraction",
    "install_mean_output_contract",
    "supports_head_contraction_slot",
]
