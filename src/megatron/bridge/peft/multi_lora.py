# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Multi-adapter LoRA model transform.

:class:`MultiLoRA` wraps target modules with multi-adapter LoRA layers.
All per-adapter state (alpha, rank, weights, routing) lives on the layers
and is managed by standalone functions in :mod:`multi_lora_layers`.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
import torch.nn as nn
from megatron.core.transformer.moe.router import TopKRouter

from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.canonical_lora import CanonicalLoRA
from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.peft.multi_lora_layers import (
    MultiLoRAGroupedExpertLinear,
    MultiLoRALinear,
    install_moe_slot_routing,
)
from megatron.bridge.peft.utils import is_expert_linear, is_grouped_expert_linear


logger = logging.getLogger(__name__)

# One-shot flag so the "sequential expert modules are skipped" warning fires once
# per process rather than once per (many) expert linear.
_EXPERT_SKIP_WARNED = False


@dataclass
class MultiLoRA(PEFT, ModuleMatcher):
    """Multi-adapter LoRA transform.

    Args:
        target_modules: Module names or wildcard patterns to apply multi-LoRA to.
        n_adapters: Maximum number of concurrent adapter slots.
        dim: LoRA max rank (bottleneck dimension for weight allocation).
        alpha: Default LoRA scaling parameter.
        dropout: Dropout probability for the adapter.
        dropout_position: Where to apply dropout.
        lora_A_init_method: Initialisation method for the A matrix.
        lora_B_init_method: Initialisation method for the B matrix.
        a2a_experimental: Enable experimental all-to-all communication.
        lora_dtype: Data type for adapter weights.
        normalize_moe_lora: Unsupported for multi-LoRA; see :meth:`__call__`.
        share_expert_adapters: Unsupported for multi-LoRA; see :meth:`__call__`.
        experts_shared_outer_loras: Share the hidden-side expert factor within each adapter slot.
    """

    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    n_adapters: int = 2
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"
    a2a_experimental: bool = False
    lora_dtype: Optional[torch.dtype] = None
    # Accepted (rather than rejected as unknown kwargs) so callers that share an
    # argument surface with single-LoRA get an explicit error instead of a
    # silently different adapter layout. Validated in __call__.
    normalize_moe_lora: bool = False
    share_expert_adapters: bool = False
    experts_shared_outer_loras: bool = False

    def __call__(self, model, training: bool = True):
        """Apply multi-LoRA, then install MoE slot routing for wrapped expert linears."""
        # Every module in a slot uses the same rank; only outer expert factors may be shared.
        for unsupported in ("normalize_moe_lora", "share_expert_adapters"):
            if getattr(self, unsupported):
                raise NotImplementedError(
                    f"MultiLoRA does not support {unsupported}=True; expert adapters use the "
                    f"same max rank as every other target module and only support sharing the outer factors."
                )
        # Unlike single-LoRA, the grouped-GEMM path never casts adapter weights:
        # the field is accepted for argument-surface parity but honoring it needs
        # an explicit activation/weight-cast contract. Reject rather than ignore.
        if self.lora_dtype is not None:
            raise NotImplementedError(
                f"MultiLoRA does not support lora_dtype={self.lora_dtype}; adapters run in the "
                f"base model's dtype on the grouped-GEMM path."
            )

        model = super().__call__(model, training=training)
        hooked = install_moe_slot_routing(model)
        if hooked:
            logger.info("Installed multi-lora MoE slot routing on %d MoE layer(s)", hooked)
        return model

    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        if isinstance(module, MultiLoRALinear):
            return module

        if (ans := self.match(module, name, prefix)) is not None:
            (match, full_name) = ans
            components = self.canonical_mapping.get(match, set())
            projection_targets = components & {"linear_q", "linear_k", "linear_v", "linear_fc1_gate", "linear_fc1_up"}

            if is_expert_linear(full_name):
                if not is_grouped_expert_linear(full_name):
                    # SequentialMLP keeps one linear per expert
                    # (mlp.experts.local_experts.N.linear_fc*), each seeing only
                    # its own routed tokens in dispatcher order. Neither the
                    # dense per-slot spans nor the grouped (slot, expert)
                    # routing segments that, so skip with a one-shot warning.
                    global _EXPERT_SKIP_WARNED
                    if not _EXPERT_SKIP_WARNED:
                        logger.warning(
                            "MultiLoRA does not support sequential MoE expert linears; skipping "
                            "them (e.g. %s). Use a grouped expert implementation "
                            "(moe_grouped_gemm=True) to put adapters on experts.",
                            full_name,
                        )
                        _EXPERT_SKIP_WARNED = True
                    return module

                logger.info(f"Adding multi-lora ({self.n_adapters} adapters) to expert: {full_name}")

                return MultiLoRAGroupedExpertLinear(
                    to_wrap=module,
                    n_adapters=self.n_adapters,
                    dim=self.dim,
                    alpha=self.alpha,
                    full_name=full_name,
                    projection_targets=projection_targets,
                    num_local_experts=module.num_gemms,
                    experts_shared_outer_loras=self.experts_shared_outer_loras,
                    column_init_method=self.lora_A_init_method,
                    row_init_method=self.lora_B_init_method,
                    dropout=self.dropout,
                    dropout_position=self.dropout_position,
                )
            if isinstance(module, TopKRouter):
                return module

            logger.info(f"Adding multi-lora ({self.n_adapters} adapters) to: {full_name}")

            return MultiLoRALinear(
                to_wrap=module,
                n_adapters=self.n_adapters,
                dim=self.dim,
                alpha=self.alpha,
                full_name=full_name,
                projection_targets=projection_targets,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                a2a_experimental=self.a2a_experimental,
            )

        return module

    def adapter_key_filter(self, key) -> bool:
        if isinstance(key, tuple):
            return key[1].requires_grad
        return ".adapters." in key or ".weight_A." in key or ".weight_B." in key


@dataclass
class CanonicalMultiLoRA(MultiLoRA, CanonicalLoRA):
    """Multi-LoRA with independent Q/K/V and gate/up factors in each slot."""

    target_modules: List[str] = field(default_factory=lambda: [
        "linear_q", "linear_k", "linear_v", "linear_proj", "linear_fc1_gate", "linear_fc1_up", "linear_fc2"
    ])
