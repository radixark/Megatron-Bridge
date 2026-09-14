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

"""GLM-5.3-Flash (``glm5_next``) model provider."""

from dataclasses import dataclass
from typing import Callable, Tuple, Union

from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.glm5.glm5_provider import GLM5ModelProvider
from megatron.bridge.models.glm5_next.layer_specs import glm5_next_layer_spec


@dataclass
class Glm5NextModelProvider(GLM5ModelProvider):
    """GLM-5.3-Flash: KDA + DSA(kpool) hybrid, MoE, mHC hyper-connections, NoPE MLA.

    Extends the GLM-5 (DSA) provider with the linear-attention layout and the kpool indexer.
    ``kda_layers`` are the 0-based global ids of the KDA layers; every other layer is DSA.
    """

    transformer_layer_spec: Union[ModuleSpec, Callable] = glm5_next_layer_spec

    # Only the fused TileLang sparse-MLA path is implemented for GLM-5.3.
    dsa_attention_backend: str = "tilelang"
    # kpool-compressed indexer: keys pooled `dsa_index_kpool` at a time before the top-k.
    dsa_index_kpool: int = 4

    # KDA linear attention (HF `linear_attn_config`).
    kda_layers: Tuple[int, ...] = ()
    kda_num_heads: int = 64
    kda_head_dim: int = 128
    kda_conv_kernel_size: int = 4
    kda_gate_lower_bound: float = -5.0


__all__ = ["Glm5NextModelProvider"]
