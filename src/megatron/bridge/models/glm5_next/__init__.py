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

"""GLM-5.3-Flash (``glm5_next``): KDA + DSA(kpool) hybrid MoE with mHC hyper-connections.

Bridge + provider for the HF ``Glm5NextForConditionalGeneration`` language model. Requires a
megatron-core with mHC hyper-connections (``megatron.core.transformer.hyper_connection``),
``flash-linear-attention`` (KDA kernels) and ``tilelang`` (fused sparse MLA / indexer scoring).

LoRA notes: every KDA / MLA / MLP projection is a Megatron-TE linear, so ``megatron.bridge.peft``
adapters wrap them directly (``linear_q``/``linear_k``/``linear_v``/``linear_b``/``linear_f_a``/
``linear_f_b``/``linear_g_a``/``linear_g_b``/``linear_proj`` on KDA layers; ``linear_q_down_proj``/
``linear_q_up_proj``/``linear_kv_down_proj``/``linear_kv_up_proj``/``linear_proj`` on DSA layers;
``linear_fc1``/``linear_fc2`` incl. grouped experts and ``shared_experts``). Not adaptable:
``conv1d``, ``A_log``, ``dt_bias``, ``o_norm``, the kpool ``index_kpool_compress_*`` tensors and
the mHC ``mapping_proj``/``alpha_*``/``bias`` (all frozen base parameters). The DSA indexer
linears (``wq_b``/``wk``/``weights_proj``) are wrappable but receive no gradient on the fused
top-k path, so leave them out of ``target_modules``.
"""

from megatron.bridge.models.glm5_next.glm5_next_bridge import Glm5NextBridge, register_glm5_next_hf_config_alias
from megatron.bridge.models.glm5_next.glm5_next_provider import Glm5NextModelProvider


__all__ = ["Glm5NextBridge", "Glm5NextModelProvider", "register_glm5_next_hf_config_alias"]
