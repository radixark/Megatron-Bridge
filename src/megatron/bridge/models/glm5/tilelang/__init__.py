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

"""Fused TileLang DSA kernels for GLM ``glm_moe_dsa`` (vendored from THUDM/slime).

Provides the ``tilelang`` sparse-attention backend: ``SparseMLA`` (sparse-MLA attention) and
``lighting_indexer`` (the DSA indexer), both with fwd+bwd TileLang kernels. Importing this
package pulls in the optional ``tilelang`` dependency, so
``cross_layer_dsa_dispatch.py`` imports it lazily — only when ``config.dsa_attention_backend == "tilelang"``
— keeping the default unfused (``megatron``) path dependency-free.
"""
from .indexer import generate_varlen_mask_params, lighting_indexer
from .sparse_mla import SparseMLA


__all__ = ["SparseMLA", "generate_varlen_mask_params", "lighting_indexer"]
