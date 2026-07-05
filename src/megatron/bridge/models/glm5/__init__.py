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

"""GLM-5.x (``glm_moe_dsa``: MoE + MLA + DeepSeek Sparse Attention) Megatron-Bridge model.

Two DSA sparse-MLA kernel backends are selectable via the provider attribute
``config.dsa_attention_backend`` (set from the miles ``--dsa-attention-backend`` arg under
``--megatron-to-hf-mode bridge``). The choice is **orthogonal to the model version and to LoRA** --
BOTH backends support GLM-5.1 *and* GLM-5.2 (DSA cross-layer index sharing), full or LoRA:

  * ``"megatron"`` (default) -- the portable *unfused* megatron-core DSA kernels
    (``DSAttention`` / ``CrossLayerDSAttention`` in ``cross_layer_dsa_dispatch.py``). No extra dependencies.
    Works with both the ``bshd`` and ``thd`` query layouts; ``thd`` is the preferred,
    activation-recompute-safe carrier, while ``bshd`` + activation recompute is rejected at forward
    time (the ``cross_layer_dsa_dispatch.py`` forward guard).

  * ``"tilelang"`` -- the vendored *fused* TileLang kernels (``SparseMLA`` + ``lighting_indexer`` under
    ``tilelang/``, driven by ``TileLangMLASelfAttention`` in ``tilelang_mla.py``). Matches slime's rollout
    kernels for rollout<->train numerical parity, including R3 indexer replay. Requires the optional
    ``tilelang`` dependency (imported lazily, so the default path stays dependency-free) and the
    ``thd`` (packed) layout (``--qkv-format thd``). **Training/forward-only**: it asserts
    ``inference_context is None`` (no KV cache) and cannot serve generation -- the rollout is always
    served by sglang; tilelang only matches sglang numerically on the *train* side.

Both backends are LoRA-capable. The DSA indexer (``wq_b`` / ``wk`` / ``weights_proj``) is excluded
from LoRA by default in the miles launcher. On the tilelang backend the indexer adapters get no
gradient (the fused ``lighting_indexer`` returns only discrete top-k and computes no indexer loss),
so training it there is a genuine no-op; on the default/unfused backend the indexer adapter *would*
get a tiny aux-loss gradient (~1e-5, ``dsa_indexer_loss_coeff=0.001``), so excluding it there is a
deliberate choice. GLM-5.2's cross-layer index sharing (``index_topk_freq`` /
``index_skip_topk_offset``) is read from the HF config by ``GLM5Bridge`` and honored by both
backends -- no extra CLI args.
"""

from megatron.bridge.models.glm5.glm5_bridge import GLM5Bridge


__all__ = [
    "GLM5Bridge",
]
