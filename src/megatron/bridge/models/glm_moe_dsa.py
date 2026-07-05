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

"""Deprecated import shim for the old ``megatron.bridge.models.glm_moe_dsa`` package.

The GLM-5.x (``glm_moe_dsa``) model was restructured into
``megatron.bridge.models.glm5`` with an explicit tilelang/megatron backend split. The HF-facing
identity is unchanged (``model_type="glm_moe_dsa"``, ``GlmMoeDsaForCausalLM``); only the Python
import path moved. This module re-exports the public API from the new location so existing
``import megatron.bridge.models.glm_moe_dsa`` / ``from ... import GLM5Bridge`` code keeps working,
with a DeprecationWarning pointing at the new path. Remove after downstream imports migrate.
"""

import warnings

from megatron.bridge.models.glm5 import GLM5Bridge


warnings.warn(
    "megatron.bridge.models.glm_moe_dsa has moved to megatron.bridge.models.glm5; "
    "import from megatron.bridge.models.glm5 instead. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "GLM5Bridge",
]
