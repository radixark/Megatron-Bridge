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

"""Per-layer KDA / DSA block spec for GLM-5.3-Flash (``glm5_next``)."""

import copy

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from megatron.bridge.models.glm5_next.mhc import (
    Glm5NextHyperConnectionModule,
    MeanStreamContraction,
    install_mean_output_contract,
    supports_head_contraction_slot,
)


def glm5_next_dsa_attention_spec(backend: TESpecProvider) -> ModuleSpec:
    """DSA (NoPE absorbed sparse-MLA + kpool indexer) self-attention spec."""
    # Imported here, not at module level: dsa.py pulls in the TileLang kernels (and kpool_indexer the
    # triton ones), which must stay optional for `import megatron.bridge.models` -- same pattern as
    # glm5_bridge.py's lazy TileLangMLASelfAttention import.
    from megatron.bridge.models.glm5_next.dsa import Glm5NextDSAAttention, Glm5NextDSASubmodules

    return ModuleSpec(
        module=Glm5NextDSAAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=Glm5NextDSASubmodules(
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_layer_norm_linear(),
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=backend.column_parallel_layer_norm_linear(),
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            kv_layernorm=IdentityOp,
            wq_b=backend.linear(),
            wk=backend.linear(),
            k_norm=backend.layer_norm(),
            weights_proj=backend.linear(),
        ),
    )


def glm5_next_kda_attention_spec(backend: TESpecProvider) -> ModuleSpec:
    """KDA linear-attention self-attention spec (TP-sharded by head)."""
    from megatron.bridge.models.glm5_next.kda import Glm5NextKDAAttention, Glm5NextKDASubmodules

    return ModuleSpec(
        module=Glm5NextKDAAttention,
        submodules=Glm5NextKDASubmodules(
            linear_q=backend.column_parallel_linear(),
            linear_k=backend.column_parallel_linear(),
            linear_v=backend.column_parallel_linear(),
            linear_b=backend.column_parallel_linear(),
            linear_f_a=backend.linear(),
            linear_f_b=backend.column_parallel_linear(),
            linear_g_a=backend.linear(),
            linear_g_b=backend.column_parallel_linear(),
            linear_proj=backend.row_parallel_linear(),
        ),
    )


def glm5_next_layer_spec(config, vp_stage=None) -> TransformerBlockSubmodules:
    """Block spec: GPT/MLA decoder block with each layer's ``self_attention`` swapped for KDA or DSA.

    Starts from ``get_gpt_decoder_block_spec`` so the dense/MoE MLP pattern (``moe_layer_freq``),
    the separate MLA input layernorm and the mHC layer wiring stay megatron-core's own; then the
    attention of layer ``i`` (global 0-based id) becomes KDA if ``i in config.kda_layers`` and DSA
    otherwise. With hyper connections on, the per-layer modules are GLM-5.3's and the block-level
    contraction is the stream mean.
    """
    block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage)
    backend = TESpecProvider()
    dsa_spec = glm5_next_dsa_attention_spec(backend)
    kda_spec = glm5_next_kda_attention_spec(backend)

    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    kda_layers = set(int(i) for i in config.kda_layers)

    for local_id in range(num_layers_to_build):
        layer_spec = copy.deepcopy(block_spec.layer_specs[local_id])
        layer_spec.submodules.self_attention = kda_spec if (local_id + offset) in kda_layers else dsa_spec
        if config.enable_hyper_connections:
            layer_spec.submodules.self_attention_hyper_connection = Glm5NextHyperConnectionModule
            layer_spec.submodules.mlp_hyper_connection = Glm5NextHyperConnectionModule
        block_spec.layer_specs[local_id] = layer_spec

    if config.enable_hyper_connections:
        if supports_head_contraction_slot():
            block_spec.hc_head_contraction = ModuleSpec(module=MeanStreamContraction)
        else:
            install_mean_output_contract()
    return block_spec


__all__ = ["glm5_next_dsa_attention_spec", "glm5_next_kda_attention_spec", "glm5_next_layer_spec"]
