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

"""``MLASelfAttention`` subclass that runs slime's fused DSA forward (the ``tilelang`` backend).

The default (``megatron``) backend stays byte-identical: :meth:`TileLangMLASelfAttention.forward`
delegates to ``super().forward`` unless ``config.dsa_attention_backend == "tilelang"``. Only the slime
branch runs the fused TileLang kernels (``SparseMLA`` + ``lighting_indexer``), which are imported
lazily so the default path stays free of the optional ``tilelang`` dependency.

Why subclass ``MLASelfAttention`` (and not just dispatch inside ``CrossLayerDSAttention``): slime's
``SparseMLA`` consumes the *absorbed-latent* q/kv (q ``[t, heads, kv_lora_rank + qk_pos_emb_head_dim]``,
kv ``[t, 1, kv_lora_rank + qk_pos_emb_head_dim]``) plus the absorb weights ``w_kc`` / ``w_vc`` from
``linear_kv_up_proj``. Those live *upstream* in ``MLASelfAttention``; ``CrossLayerDSAttention`` (the
``core_attention`` submodule) only sees the already-expanded per-head ``query`` / ``key`` / ``value``.

NUMERICS -- this forward replicates slime's ``get_absorb_query_key_value_tensors`` *exactly*, starting
from the (bit-identical) projection layers, rather than reusing megatron-core's
``get_query_key_value_tensors``. The MLA RoPE is the one place the two frameworks diverge at bf16: both
interleave the same way and use a non-interleaved emb (``rotary_interleaved`` is mutually exclusive with
``multi_latent_attention`` in megatron-core, so MLA always uses the manual-interleave path), but slime
applies RoPE with apex's fused ``fused_apply_rotary_pos_emb_thd`` kernel, whereas megatron-core's
``_apply_rotary_pos_emb_thd`` uses a Python baseline. That kernel-precision difference is the entire
~7e-4 attention-output residual the MoE top-8 router amplifies ~130x downstream. We therefore mirror
slime's ``fuse_rope`` (apex, interleaved input, CP-aware) here so the absorbed q/kv -- and hence the
attention output -- bit-match slime. The indexer top-k still goes through the bridge indexer
(``core_attention.indexer.forward_before_topk``, which already carries the rope-half-swap fix and gives
>0.99 top-k overlap with slime) feeding the fused ``lighting_indexer``. Mirrors slime
``miles_plugins/models/glm5/glm5.py`` (``DSAMultiLatentAttention.forward`` +
``get_absorb_query_key_value_tensors``).
"""

import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.utils import deprecate_inference_params

from megatron.bridge.models.glm5.cross_layer_dsa_dispatch import _holder


class TileLangMLASelfAttention(MLASelfAttention):
    """``MLASelfAttention`` with an optional slime fused-DSA forward path.

    Default backend (``megatron``) -> ``super().forward`` (unchanged, regression-safe).
    ``tilelang`` backend -> the fused ``SparseMLA`` + ``lighting_indexer`` TileLang kernels, with the
    q/kv absorb + RoPE numerics matched to slime so the attention output bit-matches slime.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # R3 indexer replay: register this attention's per-layer stream so the fused lighting_indexer
        # top-k is recorded/replayed against the rollout's DSA selection (arxiv 2510.11370). The
        # megatron-core DSAIndexer only self-registers in DeepSeek-V4 mode (dsv4_mode); for GLM
        # (dsv4_mode=False) it does not, so -- exactly like slime's glm5.py -- we register here.
        # Gated on the tilelang backend (the unfused path has no fused indexer top-k to replay) and a
        # no-op unless --use-indexer-replay enabled the manager before the build (register_to_module
        # returns early while disabled). Guarded so the package still imports without miles.
        if getattr(self.config, "dsa_attention_backend", "megatron") == "tilelang":
            try:
                from miles.utils.replay_base import indexer_replay_manager

                indexer_replay_manager.register_to_module(self, "indexer_replay", stream_idx=self.layer_number - 1)
            except ImportError:
                pass

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
    ):
        """Forward pass; delegates to the base class unless the slime backend is selected."""
        if getattr(self.config, "dsa_attention_backend", "megatron") != "tilelang":
            return super().forward(
                hidden_states,
                attention_mask,
                key_value_states=key_value_states,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                position_ids=position_ids,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
            )
        return self._tilelang_forward(
            hidden_states,
            attention_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            position_ids=position_ids,
            inference_params=inference_params,
        )

    def _tilelang_forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        *,
        inference_params=None,
    ):
        # Lazy import: only the slime branch pulls in tilelang, so the package imports fine without
        # it (the default backend stays dependency-free). Guard the import so a missing optional dep
        # gives a clear, actionable error rather than a deep ImportError from the vendored kernels.
        try:
            from megatron.bridge.models.glm5.tilelang import (
                SparseMLA,
                generate_varlen_mask_params,
                lighting_indexer,
            )
        except ImportError as e:
            raise ImportError(
                "dsa_attention_backend='tilelang' needs the optional fused-kernel dependency "
                "tilelang, which is not installed. Install it, or "
                "select the default backend (--dsa-attention-backend megatron-bridge)."
            ) from e

        assert attention_bias is None, "Attention bias should not be passed into MLA."
        # The slime fused kernels require packed (thd) inputs: lighting_indexer / SparseMLA index by
        # cu_seqlens, and the absorbed q/kv layout is [t, heads, dim]. bshd has no cu_seqlens.
        assert packed_seq_params is not None and packed_seq_params.qkv_format == "thd", (
            "The tilelang DSA backend (dsa_attention_backend='tilelang') requires the thd layout "
            "(packed_seq_params with qkv_format='thd'); got "
            f"packed_seq_params={'None' if packed_seq_params is None else packed_seq_params.qkv_format}. "
            "Use --qkv-format thd, or select the 'megatron' backend."
        )

        inference_context = deprecate_inference_params(inference_context, inference_params)
        assert inference_context is None, "The slime DSA backend is training/forward-only (no inference cache)."

        core_attention = self.core_attention  # CrossLayerDSAttention

        # =====================================================================
        # Absorbed-latent q/kv -- slime's get_absorb_query_key_value_tensors, faithfully.
        #   query: [t, np, kv_lora_rank + qk_pos_emb_head_dim]   (q_no_pe absorbed via w_kc | roped pe)
        #   key:   [t, 1,  kv_lora_rank + qk_pos_emb_head_dim]   (rms-normed kv | roped k_pos_emb)
        #   w_vc:  [np, v_head_dim, kv_lora_rank]                (de-absorb weight for the output)
        # q_compressed (post q_layernorm, pre up-proj) is reused by the indexer below.
        # =====================================================================
        query, key, w_vc, q_compressed = self._absorb_query_key_value_tensors(hidden_states, packed_seq_params)

        # =====================================================================
        # Top-k via the (bridge) indexer + the fused lighting_indexer kernel.
        # forward_before_topk returns q [t, 1, index_n_heads, index_head_dim],
        #   k [t, 1, index_head_dim], weights [t, 1, index_n_heads]; already projected, roped
        #   (rope-half-swap fix), Hadamard-rotated, and scaled (weights *= n_heads**-.5 * softmax_scale).
        #   The unfused path feeds the SAME q/k/weights to its top-k, so the fused top-k matches it.
        # Skip layers (GLM-5.2 cross-layer sharing) reuse the anchor's top-k from the holder.
        # =====================================================================
        topk_indices = self._tilelang_topk(
            core_attention,
            hidden_states,
            q_compressed,
            packed_seq_params,
            lighting_indexer,
            generate_varlen_mask_params,
        )

        # =====================================================================
        # Sparse-MLA attention (fused) + de-absorb with w_vc.
        #   SparseMLA: q [t, np, dim+tail], kv [t, 1, dim+tail], indices [t, 1, topk] -> out [t, np, kv_lora_rank]
        #   then einsum("thm,hdm->thd", out, w_vc) -> [t, np, v_head_dim].
        # BF16 for the TileLang kernel: the absorb chain stays bf16 (base linears, the bf16 LoRA
        # adapters, and the folded kv-up delta are all bf16), but a non-bf16 LoRA dtype or an fp32
        # autocast region could promote query/key. The kernel only accepts bf16, so cast defensively
        # (a no-op on the bf16 fast path; keeps the adapter contributions in the kernel's dtype).
        if query.dtype != torch.bfloat16:
            query = query.to(torch.bfloat16)
        if key.dtype != torch.bfloat16:
            key = key.to(torch.bfloat16)
        core_attn_out, _ = SparseMLA.apply(query, key, topk_indices, self.softmax_scale)
        core_attn_out = torch.einsum("thm,hdm->thd", core_attn_out, w_vc)
        # [t, np, v_head_dim] -> [t, 1, np * v_head_dim]
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [t, 1, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)
        return output, bias

    def _absorb_query_key_value_tensors(self, hidden_states, packed_seq_params):
        """Derive absorbed-latent ``query`` / ``key`` (+ de-absorb ``w_vc`` + ``q_compressed``).

        Faithful port of slime ``get_absorb_query_key_value_tensors`` using the bridge module objects
        (which are bit-identical to slime's): the projections / norms are the same LayerNorm-Linear
        kernels, and the RoPE is slime's apex ``fuse_rope`` (interleaved input, non-interleaved emb,
        CP-aware) instead of megatron-core's Python baseline -- this is the numeric match.
        """
        assert hidden_states.ndim == 3, f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"

        config = self.config
        qk_head_dim = config.qk_head_dim
        qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        kv_lora_rank = config.kv_lora_rank
        v_head_dim = config.v_head_dim

        # ---- RoPE freqs (same call megatron-core MLA makes) ----
        # YarnRotaryEmbedding returns (emb, mscale); plain RotaryEmbedding returns emb. Slime's
        # fuse_rope uses only the freqs and lets the mscale ride on softmax_scale (which megatron's
        # MultiLatentAttention.__init__ already bakes in identically), so we drop mscale here too.
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(None, None, hidden_states, config, packed_seq_params)
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq_params is not None)
        if isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb[0]

        cu_seqlens_q = packed_seq_params.cu_seqlens_q
        cu_seqlens_kv = packed_seq_params.cu_seqlens_kv

        # ---- QKV down projection + (Identity) layernorm. thd: outputs are [t, 1, *]. ----
        q_compressed, _ = self.linear_q_down_proj(hidden_states)
        q_compressed = q_compressed.squeeze(1)

        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if config.sequence_parallel:
            kv_combined = gather_from_sequence_parallel_region(kv_combined)
        kv_compressed, k_pos_emb = torch.split(kv_combined, [kv_lora_rank, qk_pos_emb_head_dim], dim=-1)
        # kv_layernorm is IdentityOp in the DSA spec (the norm is fused into linear_kv_up_proj); the
        # absorb path applies the rms-norm explicitly below.
        kv_compressed = self.kv_layernorm(kv_compressed)

        # ---- q up projection + split. q_layernorm is IdentityOp; the norm is in linear_q_up_proj. ----
        q_compressed = self.q_layernorm(q_compressed)
        q, _ = self.linear_q_up_proj(q_compressed)
        q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)
        q_no_pe, q_pos_emb = torch.split(q, [qk_head_dim, qk_pos_emb_head_dim], dim=-1)

        # ---- absorb weights from linear_kv_up_proj ----
        # The absorb consumes the up-proj WEIGHT MATRIX directly (it cannot be expressed as a
        # forward call), so we read .weight / .layer_norm_weight off the module. Under LoRA,
        # ``linear_kv_up_proj`` is wrapped (``LoRALinear``: base at ``.to_wrap``, adapter at
        # ``.adapter``); :meth:`_kv_up_proj_weight_and_norm` unwraps it and folds the adapter
        # delta into the effective weight so a LoRA on ``kv_b_proj`` is trained on the fused path
        # too (gradients flow back to its adapter), matching the unfused backend. See its docstring.
        kv_up_weight, kv_up_ln_weight = self._kv_up_proj_weight_and_norm()
        # weight [np * (qk_head_dim + v_head_dim), kv_lora_rank] -> per head [np, qk_head_dim+v_head_dim, kv_lora_rank]
        w_kc, w_vc = kv_up_weight.unflatten(0, (-1, qk_head_dim + v_head_dim)).split([qk_head_dim, v_head_dim], dim=1)

        # absorbed q content: [t, np, kv_lora_rank]
        q_no_pe = torch.einsum("thd,hdm->thm", q_no_pe, w_kc)

        # rms-norm the compressed kv in fp32 with the fused up-proj layernorm weight (matches slime
        # and the unfused LayerNorm-Linear precision), then cast back.
        kv_compressed = torch.nn.functional.rms_norm(
            kv_compressed.float(),
            normalized_shape=(kv_compressed.shape[-1],),
            weight=kv_up_ln_weight.float(),
            eps=config.layernorm_epsilon,
        ).to(kv_compressed.dtype)

        # CP-gather the kv latent + k_pos_emb (no-op at CP=1; matches slime's gathered=True path for k).
        k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb, group=parallel_state.get_context_parallel_group())
        kv_compressed = gather_from_sequence_parallel_region(
            kv_compressed, group=parallel_state.get_context_parallel_group()
        )

        # ---- RoPE: slime's apex fuse_rope (interleaved input, CP-aware) on the pe halves ----
        q_pos_emb = self._fuse_rope(q_pos_emb, cu_seqlens_q, rotary_pos_emb, gathered=False)
        k_pos_emb = self._fuse_rope(k_pos_emb, cu_seqlens_kv, rotary_pos_emb, gathered=True)

        query = torch.cat([q_no_pe, q_pos_emb], dim=-1).contiguous()
        # kv_compressed / k_pos_emb already carry the kv_group=1 dim ([t, 1, *]) in the thd layout
        # (only q_compressed is squeezed, mirroring slime), so the cat is already [t, 1, dim+tail].
        key = torch.cat([kv_compressed, k_pos_emb], dim=-1).contiguous()
        assert key.shape[-1] == kv_lora_rank + qk_pos_emb_head_dim
        return query, key, w_vc, q_compressed

    def _kv_up_proj_weight_and_norm(self):
        """Return ``(weight, layer_norm_weight)`` for the absorb, LoRA-aware.

        The slime absorb reads the ``linear_kv_up_proj`` weight matrix directly to build ``w_kc`` /
        ``w_vc`` (an absorb cannot be a forward call). When LoRA targets ``kv_b_proj`` the module is
        wrapped (``megatron.bridge.peft.lora_layers.LoRALinear``: base linear under ``to_wrap``,
        :class:`ParallelLinearAdapter` under ``adapter``); the wrapper exposes no ``.weight``, so a
        naive read raises ``AttributeError``. We unwrap to the base linear and **fold the LoRA delta
        into the effective weight** so the adapter on ``kv_b_proj`` is genuinely trained on the fused
        path (its gradient flows back through the einsum -> ``SparseMLA`` -> ``w_vc`` chain), keeping
        LoRA semantics identical to the unfused backend (where ``linear_kv_up_proj(kv)`` is a forward
        and the adapter applies natively).

        The adapter is LoRA (identity activation, no bias) so its weight-space delta is exactly
        ``(alpha/dim) * (linear_out.weight @ linear_in.weight)`` -- the same expression
        :class:`~megatron.bridge.peft.lora.LoRAMerge` uses. ``linear_kv_up_proj`` is a
        column-parallel ``LayerNormColumnParallelLinear``: the base weight shard, ``linear_out``
        (column-parallel), and the absorb's per-head unflatten are all on dim 0 (heads), so folding
        the local shard is correct at any TP. ``linear_in`` (the LoRA-A) is column-parallel along the
        rank dim, so we all-gather it across TP to reconstruct the full ``dim`` before the matmul,
        mirroring ``LoRAMerge.merge`` Case 1. (No LoRA -> just the base weight, byte-identical.)
        """
        module = self.linear_kv_up_proj
        base = getattr(module, "to_wrap", module)
        weight = base.weight
        ln_weight = base.layer_norm_weight
        adapter = getattr(module, "adapter", None)
        if adapter is None or not getattr(module, "_adapter_enabled", True):
            return weight, ln_weight

        # Fold the LoRA delta: weight_eff = weight + (alpha/dim) * (linear_out @ linear_in).
        linear_in = adapter.linear_in.weight  # [dim (maybe /TP), in_features]
        linear_out = adapter.linear_out.weight  # [out_features (/TP), dim]
        dim = adapter.dim
        scale = adapter.alpha / dim

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        # Column-parallel base (input_is_parallel=False): linear_in is sharded along `dim` (dim 0),
        # so all-gather it to the full rank before the matmul. linear_out already matches the local
        # output shard of the base weight.
        if tp_size > 1 and not getattr(adapter, "input_is_parallel", False) and linear_in.shape[0] * tp_size == dim:
            # Differentiable all-gather so the LoRA-A (linear_in) gradient flows back. The gathered
            # full linear_in is used in EVERY TP rank's local delta, so its adjoint is reduce-scatter
            # (sum grads across TP ranks, then scatter each rank's shard). Plain torch.distributed
            # all_gather has no autograd -> it silently detached linear_in and froze LoRA-A on the
            # fused path; the autograd-aware variant restores the gradient.
            from torch.distributed.nn.functional import all_gather as _diff_all_gather

            gathered = _diff_all_gather(
                linear_in.contiguous(), group=parallel_state.get_tensor_model_parallel_group()
            )
            linear_in = torch.cat(gathered, dim=0)

        delta = scale * (linear_out.to(weight.dtype) @ linear_in.to(weight.dtype))
        return weight + delta, ln_weight

    @staticmethod
    def _fuse_rope(t, cu_seqlens, rotary_pos_emb, *, gathered):
        """slime ``fuse_rope``: interleave the input (MLA convention) then apply apex's thd RoPE.

        ``gathered=False`` (q): the input is the CP-local shard, so replicate across CP, apply RoPE
        over the full packed sequence, then slice this rank's segment back out. ``gathered=True`` (k):
        the input is already CP-gathered, so apply directly. apex is bit-for-bit what slime runs; the
        megatron-core baseline differs at bf16 (this is the residual being closed).
        """
        from apex.transformer.functional import fused_apply_rotary_pos_emb_thd

        # MLA rope interleave: [x0,x1,x2,x3,...] -> [x0,x2,...,x1,x3,...] (even then odd halves).
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)
        freqs = rotary_pos_emb.squeeze(0)
        if gathered:
            return fused_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs)
        seq_len = t.shape[0]
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()
        t = t.repeat(cp_size, 1, 1)
        out = fused_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs)
        return out[cp_rank * seq_len : (cp_rank + 1) * seq_len]

    def _tilelang_topk(
        self,
        core_attention,
        hidden_states,
        q_compressed,
        packed_seq_params,
        lighting_indexer,
        generate_varlen_mask_params,
    ):
        """Compute (or reuse, for GLM-5.2 skip layers) the sparse top-k via the fused indexer kernel.

        LoRA on the indexer: ``linear_wq_b`` / ``linear_wk`` are applied as forwards in
        :meth:`_tilelang_index_qkw`, so an adapter on them is *used* and differentiable. BUT the fused
        ``lighting_indexer`` returns only the (discrete, non-differentiable) top-k indices -- it does
        NOT compute the ``FusedDSAIndexerLoss`` the unfused path uses (cross_layer_dsa_dispatch.py). The
        indexer projections therefore receive a gradient ONLY from that auxiliary loss, so on the
        fused path their LoRA adapters get NO gradient (verified: ``grad=None``), whereas the unfused
        path gives them a tiny aux-loss gradient (~1e-5). This matches slime's own fused forward
        (no indexer loss). Consequently LoRA on the indexer is a no-op on the fused backend; the
        e2e / CI target list deliberately EXCLUDES the indexer (wq_b/wk/weights_proj). Adding the
        fused indexer loss is Step-6 / future work.
        """
        # GLM-5.2 cross-layer index sharing: skip layers reuse the most recent computing layer's top-k.
        index_share = getattr(core_attention, "_index_share", False)
        if index_share and getattr(core_attention, "_skip_topk", False):
            holder = _holder(packed_seq_params)
            src = core_attention._source_layer
            if src not in holder:
                raise AssertionError(
                    f"DSA index-share (slime): skip layer (layer_number={core_attention.layer_number}) needs "
                    f"the top-k of its source computing layer (layer_number={src}), which did not run in this "
                    f"pipeline stage's forward. Holder has {sorted(holder)}."
                )
            return holder[src]

        # ---- compute this layer's top-k ----
        # index_q/index_k/weights built to match slime EXACTLY (apex rope, NO Hadamard rotation):
        # the megatron-core indexer's forward_before_topk applies a Hadamard rotate_activation +
        # baseline rope that slime does not, so its index_q/index_k differ from slime by ~rel 1.4
        # (orthogonal Hadamard => logits ~preserved, hence the old 0.999 top-k overlap, but the extra
        # rotation + baseline-rope rounding flips ~0.1% of the top-k, which the MoE router amplifies).
        index_q, index_k, weights = self._tilelang_index_qkw(
            core_attention.indexer, hidden_states, q_compressed, packed_seq_params
        )

        index_q = index_q.to(torch.bfloat16).contiguous()
        index_k = index_k.to(torch.bfloat16).contiguous()
        # lighting_indexer's fwd kernel declares Weights as float32 (slime feeds fp32 here too).
        weights = weights.float().contiguous()

        # cu_seqlens_q spans the FULL packed sequence, so generate_varlen_mask_params yields
        # per-query starts/ends of length S_full. The fused lighting_indexer kernel derives its
        # query count from index_q.shape[0] and requires starts/ends (CuSeqLenKS/KE) to have the
        # SAME first dim. After _tilelang_index_qkw, index_q/index_k/weights are SP-gathered to the
        # full-over-TP, CP-local token dim (S_full/CP per rank); the FULL-length starts/ends must
        # be brought to the same token dim. Mirror slime's glm5.py exactly: scatter starts/ends
        # over the CONTEXT-PARALLEL group (CP=1 -> no-op, so SP-only runs keep the full length to
        # match the SP-gathered index_q; CP>1 -> each rank keeps its S_full/CP slice). Without this
        # the kernel aborts with "CuSeqLenKS shape[0] expected <S> but got <S/TP>".
        starts, ends = generate_varlen_mask_params(packed_seq_params.cu_seqlens_q)
        cp_group = parallel_state.get_context_parallel_group()
        starts = scatter_to_sequence_parallel_region(starts, group=cp_group)
        ends = scatter_to_sequence_parallel_region(ends, group=cp_group)
        starts = starts.to(torch.int32)
        ends = ends.to(torch.int32)

        index_topk = core_attention.indexer.index_topk
        # R3 indexer replay: select THIS layer's stream (registered on self in __init__, mirroring
        # slime's glm5.py -- the megatron-core DSAIndexer only self-registers in DeepSeek-V4 mode,
        # not for GLM, so we own the registration). register_to_module also installs a forward-pre-hook
        # that does this, but the fused path computes the top-k in this helper rather than in the
        # indexer module's own ``forward``, so we set_current explicitly instead of relying on hook
        # ordering. No-op when --use-indexer-replay is off: ``self.indexer_replay`` is then unset
        # (register_to_module returns early while the manager is disabled) -> ``_replay`` is None.
        _replay = getattr(self, "indexer_replay", None)
        if _replay is not None:
            from miles.utils.replay_base import indexer_replay_manager

            indexer_replay_manager.set_current(_replay)
        _, topk_indices = lighting_indexer(index_q, index_k, weights, starts, ends, index_topk)
        # lighting_indexer returns [S, topk]; SparseMLA wants indices [t, kv_group=1, topk].
        topk_indices = topk_indices.unsqueeze(1)

        # Anchor / plain-DSA layer: publish for any skip layers sharing this top-k.
        # ACTIVATION RECOMPUTE: the slime path is thd-only (asserted in _tilelang_forward), so the
        # holder rides on packed_seq_params -- the per-microbatch carrier that megatron's
        # activation-checkpoint custom_forward closure-captures, making this write recompute-safe
        # (the same property the unfused thd path relies on; see cross_layer_dsa_dispatch._holder). The
        # bshd thread-local fallback that the unfused guard rejects under recompute is unreachable
        # here because slime never runs bshd. The rest of _tilelang_forward is functionally pure
        # (projections + RoPE + the differentiable SparseMLA / lighting_indexer kernels), so full /
        # selective recompute re-executes it correctly.
        if index_share:
            _holder(packed_seq_params)[core_attention.layer_number] = topk_indices
        return topk_indices

    def _tilelang_index_qkw(self, indexer, hidden_states, q_compressed, packed_seq_params):
        """Build the indexer index_q / index_k / head_weights to match slime's inline path exactly.

        Faithful port of slime ``get_absorb_query_key_value_tensors`` (indexer section): same
        ``linear_wq_b`` / ``linear_wk`` / ``k_norm`` / ``linear_weights_proj`` weights (which are
        loaded with the same rope-half swap as slime), but with apex ``fuse_rope`` and *no* Hadamard
        ``rotate_activation`` -- so index_q / index_k bit-match slime and the top-k bit-matches.

        Returns index_q ``[t, index_n_heads, index_head_dim]``, index_k ``[t, index_head_dim]``,
        head_weights ``[t, index_n_heads]``.
        """
        from megatron.core.transformer.moe.moe_utils import RouterGatingLinearFunction

        config = self.config
        index_n_heads = indexer.index_n_heads
        index_head_dim = indexer.index_head_dim
        qk_pos_emb_head_dim = config.qk_pos_emb_head_dim

        # RoPE freqs from the indexer's own RotaryEmbedding (qk_pos_emb_head_dim), same call slime makes.
        rotary_seq_len = indexer.rotary_pos_emb.get_rotary_seq_len(None, None, hidden_states, config, None)
        rotary_pos_emb = indexer.rotary_pos_emb(rotary_seq_len, packed_seq=False)
        if isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb[0]
        cu_seqlens_q = packed_seq_params.cu_seqlens_q
        cu_seqlens_kv = packed_seq_params.cu_seqlens_kv

        # SP/CP token-dim contract (mirrors slime glm5.py get_absorb_query_key_value_tensors,
        # indexer section). The indexer projections run on the SP-LOCAL hidden / q_compressed
        # (S_full/TP tokens per rank). The fused lighting_indexer needs them reconciled to the
        # full-over-TP, CP-local token dim:
        #   * index_q: SP all-gather (over the TP group) -> S_full-over-TP (CP-local). The RoPE
        #     below (_fuse_rope gathered=False) then does the CP repeat/slice, so index_q must
        #     already be SP-full here for the cu_seqlens_q (FULL) RoPE to line up. (slime l.538-539)
        #   * index_k: SP all-gather AND CP all-gather -> FULL S_full (keys are dense across CP;
        #     RoPE gathered=True). (slime l.544-546)
        #   * head_weights: SP all-gather -> matches index_q's token dim. (slime l.554-555)
        # At CP=1 the CP gathers are no-ops; SP=1 makes the SP gathers no-ops. starts/ends are
        # reconciled symmetrically in _tilelang_topk (CP-scatter of the FULL-length starts/ends).

        # index_q: wq_b on the (post-q_layernorm) compressed q.
        index_q, _ = indexer.linear_wq_b(q_compressed)
        index_q = index_q.view(*index_q.size()[:-1], index_n_heads, index_head_dim)
        if config.sequence_parallel:
            index_q = gather_from_sequence_parallel_region(index_q)

        # index_k: wk on hidden, k_norm in fp32 (slime uses eps=1e-6), back to bf16.
        index_k, _ = indexer.linear_wk(hidden_states)
        index_k = index_k.squeeze(1)
        index_k = torch.nn.functional.layer_norm(
            index_k.float(),
            normalized_shape=(index_head_dim,),
            weight=indexer.k_norm.weight.float(),
            bias=indexer.k_norm.bias.float() if getattr(indexer.k_norm, "bias", None) is not None else None,
            eps=1e-6,
        ).to(torch.bfloat16)
        if config.sequence_parallel:
            index_k = gather_from_sequence_parallel_region(index_k)
        index_k = gather_from_sequence_parallel_region(index_k, group=parallel_state.get_context_parallel_group())
        index_k = index_k.unsqueeze(1)  # [t, 1, index_head_dim]

        # head_weights: fp32 router-gating linear, scaled by n_heads**-.5 * head_dim**-.5.
        # The fused RouterGatingLinearFunction needs the weight matrix directly; unwrap any LoRA
        # wrapper (``LoRALinear.to_wrap``) so this does not crash if ``weights_proj`` is targeted.
        # The e2e excludes the indexer from LoRA, so no adapter delta is folded here (the base
        # weight is used, matching slime's indexer); revisit if indexer LoRA on the fused path is
        # required (would need folding the delta like ``_kv_up_proj_weight_and_norm``).
        weights_proj = getattr(indexer.linear_weights_proj, "to_wrap", indexer.linear_weights_proj)
        head_weights = RouterGatingLinearFunction.apply(hidden_states, weights_proj.weight, None, torch.float32)
        head_weights = head_weights.squeeze(1) * ((index_n_heads**-0.5) * (index_head_dim**-0.5))
        if config.sequence_parallel:
            head_weights = gather_from_sequence_parallel_region(head_weights)

        # RoPE (interleaved branch -- indexer_rope_interleave=True for GLM-5.2): split [no_pe | pe],
        # apex-rope the pe half, re-cat. The wq_b/wk rope-half swap at load makes the bridge's
        # last-half split rotate the same dims slime does.
        rope_dim = qk_pos_emb_head_dim
        iq_no_pe, iq_pe = torch.split(index_q, [index_head_dim - rope_dim, rope_dim], dim=-1)
        iq_pe = self._fuse_rope(iq_pe, cu_seqlens_q, rotary_pos_emb, gathered=False)
        index_q = torch.cat([iq_no_pe, iq_pe], dim=-1)

        ik_no_pe, ik_pe = torch.split(index_k, [index_head_dim - rope_dim, rope_dim], dim=-1)
        ik_pe = self._fuse_rope(ik_pe, cu_seqlens_kv, rotary_pos_emb, gathered=True)
        index_k = torch.cat([ik_no_pe, ik_pe], dim=-1)

        return index_q, index_k.squeeze(1), head_weights
