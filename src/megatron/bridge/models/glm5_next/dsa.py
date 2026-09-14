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

"""GLM-5.3-Flash DSA layer: NoPE absorbed sparse-MLA with the kpool-compressed lightning indexer.

Built on :class:`~megatron.bridge.models.glm5.tilelang.tilelang_mla.TileLangMLASelfAttention`
(the GLM-5 / 5.2 fused ``SparseMLA`` path) with three GLM-5.3 changes, mirroring
radixark/miles ``miles_plugins/models/glm5_next/dsa.py``:

* **NoPE**: ``qk_pos_emb_head_dim == 0``; no rotary embedding anywhere. Queries/keys are padded
  with 64 zero dims so the ``SparseMLA`` kernel keeps its ``kv_lora_rank + 64`` layout.
* **kpool indexer**: the indexer projections (``wq_b`` / ``wk`` / ``k_norm`` / ``weights_proj``)
  live on this module, and the keys are pooled ``kpool`` at a time with the learned
  ``index_kpool_compress_gate`` / ``index_kpool_compress_ape`` before the top-k
  (:mod:`~megatron.bridge.models.glm5_next.kpool_indexer`).
* The indexer receives no gradient (fused top-k is discrete); its inputs are detached.

LoRA: the MLA projections are ordinary Megatron/TE linears; ``linear_kv_up_proj`` is read as a
weight for the absorb, so its adapter delta is folded in through the base class'
``_kv_up_proj_weight_and_norm``.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.moe.moe_utils import RouterGatingLinearFunction
from megatron.core.transformer.multi_latent_attention import MLASelfAttentionSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from megatron.bridge.models.glm5.tilelang.tilelang_mla import TileLangMLASelfAttention
from megatron.bridge.models.glm5_next.kpool_indexer import build_pooled_keys, kpool_select_topk, pool_boundaries


_SPARSE_MLA_TAIL_DIM = 64


@dataclass
class Glm5NextDSASubmodules(MLASelfAttentionSubmodules):
    """MLA submodules plus the DSA indexer projections."""

    wq_b: Union[ModuleSpec, type] = None
    wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    weights_proj: Union[ModuleSpec, type] = None


def _base_linear(module):
    """Unwrap a PEFT ``LoRALinear`` (``to_wrap``) to the underlying Megatron linear."""
    return getattr(module, "to_wrap", module)


class Glm5NextDSAAttention(TileLangMLASelfAttention):
    """GLM-5.3 DSA attention (fused TileLang backend only)."""

    def __init__(
        self,
        config,
        submodules: Glm5NextDSASubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection=None,
        pp_layer_offset: Optional[int] = None,
        is_mtp_layer: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ):
        assert config.qk_pos_emb_head_dim == 0, "GLM-5.3 DSA is NoPE; qk_pos_emb_head_dim must be 0"
        assert getattr(config, "dsa_attention_backend", "tilelang") == "tilelang", (
            "GLM-5.3 DSA only implements the fused tilelang backend"
        )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )
        self.softmax_scale = self.q_head_dim**-0.5
        self.index_n_heads = int(config.dsa_indexer_n_heads)
        self.index_head_dim = int(config.dsa_indexer_head_dim)
        self.index_topk = int(config.dsa_indexer_topk)
        self.index_kpool = int(config.dsa_index_kpool)
        assert self.index_kpool > 1, "GLM-5.3 kpool indexer expects index_kpool > 1"

        indexer_linear_kwargs = dict(
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            parallel_mode="duplicated",
            skip_weight_param_allocation=False,
        )
        self.wq_b = build_module(
            submodules.wq_b,
            input_size=config.q_lora_rank,
            output_size=self.index_n_heads * self.index_head_dim,
            tp_comm_buffer_name="wq_b",
            **indexer_linear_kwargs,
        )
        self.wk = build_module(
            submodules.wk,
            input_size=config.hidden_size,
            output_size=self.index_head_dim,
            tp_comm_buffer_name="wk",
            **indexer_linear_kwargs,
        )
        # k_norm is a LayerNorm with bias applied in fp32; TENorm follows config.normalization.
        saved_normalization = config.normalization
        config.normalization = "LayerNorm"
        try:
            self.k_norm = build_module(submodules.k_norm, hidden_size=self.index_head_dim, config=config, eps=1e-6)
        finally:
            config.normalization = saved_normalization
        self.weights_proj = build_module(
            submodules.weights_proj,
            input_size=config.hidden_size,
            output_size=self.index_n_heads,
            tp_comm_buffer_name="weights_proj",
            **indexer_linear_kwargs,
        )

        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self.index_kpool_compress_gate = nn.Parameter(
            torch.zeros(self.index_head_dim, config.hidden_size, dtype=config.params_dtype, device=device)
        )
        self.index_kpool_compress_ape = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(self.index_kpool, self.index_head_dim, dtype=torch.float32, device=device))
        )
        self.index_kpool_compress_gate.requires_grad_(False)
        self.index_kpool_compress_ape.requires_grad_(False)
        if getattr(config, "freeze_indexer", False):
            for module in (self.wq_b, self.wk, self.k_norm, self.weights_proj):
                for param in module.parameters():
                    param.requires_grad_(False)

        # miles R3 indexer replay streams are indexed over DSA layers only.
        replay = getattr(self, "indexer_replay", None)
        if replay is not None:
            dsa_layers = [i for i in range(config.num_layers) if i not in set(config.kda_layers)]
            replay.stream_idx = dsa_layers.index(self.layer_number - 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ):
        """``hidden_states`` [s, 1, h] (SP slice) -> (output, bias); always the fused kpool path."""
        assert rotary_pos_emb is None, "GLM-5.3 DSA is NoPE; no rotary embedding expected"
        assert attention_bias is None, "attention_bias is not supported by SparseMLA"
        assert inference_context is None and inference_params is None
        assert packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", "thd") == "thd"

        from megatron.bridge.models.glm5.tilelang import SparseMLA

        query, key, w_vc, index_q, index_k, head_weights, gate_score = self._absorb_query_key_value_tensors(
            hidden_states, packed_seq_params
        )
        topk_indices = self._kpool_select(index_q, index_k, head_weights, gate_score, packed_seq_params)

        query = F.pad(query, (0, _SPARSE_MLA_TAIL_DIM)).contiguous()
        key = F.pad(key, (0, _SPARSE_MLA_TAIL_DIM)).contiguous()
        core_attn_out, _ = SparseMLA.apply(query, key, topk_indices, self.softmax_scale)
        core_attn_out = torch.einsum("thm,hdm->thd", core_attn_out, w_vc)
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        output, bias = self.linear_proj(core_attn_out)
        return output, bias

    # ------------------------------------------------------------------
    def _absorb_query_key_value_tensors(self, hidden_states, packed_seq_params):
        """Absorbed q / latent k for SparseMLA plus the (detached) indexer inputs."""
        assert hidden_states.ndim == 3, f"hidden_states should be [s, b, h], got {hidden_states.shape}"
        config = self.config

        q_compressed, _ = self.linear_q_down_proj(hidden_states)
        q_compressed = q_compressed.squeeze(1)

        kv_compressed, _ = self.linear_kv_down_proj(hidden_states)
        if config.sequence_parallel:
            kv_compressed = gather_from_sequence_parallel_region(kv_compressed)
        kv_compressed = self.kv_layernorm(kv_compressed)

        q_compressed = self.q_layernorm(q_compressed)
        q, _ = self.linear_q_up_proj(q_compressed)
        q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

        kv_up_weight, kv_up_ln_weight = self._kv_up_proj_weight_and_norm()
        w_kc, w_vc = kv_up_weight.unflatten(0, (-1, config.qk_head_dim + config.v_head_dim)).split(
            [config.qk_head_dim, config.v_head_dim], dim=1
        )
        query = torch.einsum("thd,hdm->thm", q, w_kc)

        kv_compressed = torch.nn.functional.rms_norm(
            kv_compressed.float(),
            normalized_shape=(kv_compressed.shape[-1],),
            weight=kv_up_ln_weight.float(),
            eps=config.layernorm_epsilon,
        ).to(kv_compressed.dtype)
        kv_compressed = gather_from_sequence_parallel_region(
            kv_compressed, group=parallel_state.get_context_parallel_group()
        )
        query = query.contiguous()
        key = kv_compressed.contiguous()

        # Indexer: no gradient flows back into the trunk (the fused top-k is discrete).
        q_compressed = q_compressed.detach()
        hidden_states = hidden_states.detach()

        index_q, _ = self.wq_b(q_compressed)
        index_q = index_q.view(*index_q.size()[:-1], self.index_n_heads, self.index_head_dim)
        if config.sequence_parallel:
            index_q = gather_from_sequence_parallel_region(index_q)

        index_k, _ = self.wk(hidden_states)
        index_k = self.k_norm(index_k.squeeze(1).float()).bfloat16()
        if config.sequence_parallel:
            index_k = gather_from_sequence_parallel_region(index_k)
        index_k = gather_from_sequence_parallel_region(index_k, group=parallel_state.get_context_parallel_group())

        gate_score = F.linear(hidden_states.squeeze(1), self.index_kpool_compress_gate)
        if config.sequence_parallel:
            gate_score = gather_from_sequence_parallel_region(gate_score)
        gate_score = gather_from_sequence_parallel_region(
            gate_score, group=parallel_state.get_context_parallel_group()
        )

        weights_proj_weight = _base_linear(self.weights_proj).weight
        head_weights = RouterGatingLinearFunction.apply(hidden_states, weights_proj_weight, None, torch.float32)
        head_weights = head_weights.squeeze(1) * ((self.index_n_heads**-0.5) * (self.index_head_dim**-0.5))
        if config.sequence_parallel:
            head_weights = gather_from_sequence_parallel_region(head_weights)

        return query, key, w_vc, index_q, index_k, head_weights, gate_score

    def _kpool_select(self, index_q, index_k, head_weights, gate_score, packed_seq_params):
        if parallel_state.get_context_parallel_world_size() > 1:
            raise NotImplementedError("GLM-5.3 kpool indexer selection does not support context parallelism.")
        cu_seqlens = packed_seq_params.cu_seqlens_kv
        pool_cu_seqlens = pool_boundaries(cu_seqlens, self.index_kpool)
        pooled_k = build_pooled_keys(index_k, gate_score, self.index_kpool_compress_ape, cu_seqlens, self.index_kpool)
        return kpool_select_topk(
            index_q=index_q,
            pooled_k=pooled_k,
            head_weights=head_weights,
            cu_seqlens=cu_seqlens,
            pool_cu_seqlens=pool_cu_seqlens,
            index_topk=self.index_topk,
            kpool=self.index_kpool,
        )


__all__ = ["Glm5NextDSAAttention", "Glm5NextDSASubmodules"]
