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

"""GLM5 model provider: MLAModelProvider plus the GLM-5 DSA configuration fields."""

from dataclasses import dataclass

from megatron.bridge.models.mla_provider import MLAModelProvider


@dataclass
class GLM5ModelProvider(MLAModelProvider):
    """GLM-5 (glm_moe_dsa) provider: MLA plus DeepSeek Sparse Attention."""

    # DSA sparse-MLA kernel backend; set from the miles --dsa-attention-backend arg. Declared
    # as a real field (rather than an ad-hoc attribute) so it survives any fields-based config
    # copy and reaches every module's config without caller-side propagation.
    dsa_attention_backend: str = "megatron"


__all__ = ["GLM5ModelProvider"]
