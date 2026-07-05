# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""GLM-5.2 DSA *cross-layer index sharing* for the Megatron-Bridge path.

GLM-5.2 keeps GLM-5.1's ``glm_moe_dsa`` skeleton (MLA + lightning indexer + MoE) but adds
DSA cross-layer index sharing: only "computing"/anchor layers carry indexer weights and
compute the sparse top-k; "skip" layers reuse the most recent computing layer's top-k.
Activated by the HF config field ``index_topk_freq > 1`` (+ ``index_skip_topk_offset``);
GLM-5.1 lacks these fields (freq defaults to 1 -> every layer computes -> plain DSA).

Implemented entirely on the Megatron-Bridge side (no megatron-core edits): a ``DSAttention``
subclass that, on skip layers, drops its indexer and reuses the anchor's top-k; plus a cloned
spec builder pointing ``module=`` at the subclass. Mirrors the slime reference
(``slime_plugins/models/glm5/glm5.py``: ``is_skip_topk_layer`` / ``source_compute_layer`` /
the per-microbatch top-k holder / the skip-layer ``delattr``).
"""

import threading

import torch
from megatron.core.transformer.enums import AttnMaskType
from megatron.bridge.models.glm5.megatron import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    DSAttention,
    FusedDSAIndexerLoss,
    unfused_dsa_fn,
)


# ---- computing-layer schedule (mirrors slime glm5.py:37-52) ----
def is_skip_topk_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> bool:
    """1-indexed Megatron ``layer_number`` reuses a previous layer's top-k when True.

    A layer *computes* its own top-k iff ``max(layer_number - offset, 0) % freq == 0``.
    """
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


def source_compute_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> int:
    """The computing layer whose ``topk_indices`` a skip layer reuses (walk downward)."""
    layer = layer_number
    while is_skip_topk_layer(layer, skip_topk_offset, topk_freq):
        layer -= 1
    return layer


def assert_pp_stage_starts_on_computing_layer(config, vp_stage=None) -> None:
    """Build-time guard: a (virtual) pipeline stage must not START on a skip layer.

    The per-microbatch top-k holder does NOT cross pipeline boundaries, so a skip layer's source
    computing layer must live in the same PP stage. If a stage's first layer is a skip layer, its
    source is on a previous stage -> cross-PP top-k sharing (unsupported). Mirrors slime's
    ``get_glm5_spec`` build-time check so a bad PP layout fails at model construction with a clear
    message, instead of only at the first forward (the runtime guard in ``CrossLayerDSAttention``).

    No-op unless cross-layer sharing is active (``dsa_index_topk_freq > 1``). If the layer layout
    cannot be determined (e.g. parallel state not yet initialised), this silently returns and
    leaves the runtime guard as the backstop.
    """
    freq = getattr(config, "dsa_index_topk_freq", 1) or 1
    if freq <= 1:
        return
    offset = getattr(config, "dsa_index_skip_topk_offset", 0) or 0
    try:
        from megatron.core.transformer.transformer_block import get_transformer_layer_offset

        layer_offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    except Exception:  # noqa: BLE001 - layout not determinable; runtime guard still applies
        return
    first_layer_number = layer_offset + 1  # Megatron layer_number is 1-indexed
    if is_skip_topk_layer(first_layer_number, offset, freq):
        src = source_compute_layer(first_layer_number, offset, freq)
        raise AssertionError(
            "DSA cross-layer index-share: this pipeline stage starts at global "
            f"layer_number={first_layer_number}, which is a SKIP layer whose source computing "
            f"layer={src} is on a previous pipeline stage. The per-microbatch top-k holder does "
            "not cross PP boundaries -- choose a pipeline layout where every (virtual) stage "
            f"begins on a computing layer (dsa_index_topk_freq={freq}, "
            f"dsa_index_skip_topk_offset={offset})."
        )


# Per-microbatch top-k holder. Preferred carrier is the ``packed_seq_params`` object (thd:
# fresh per microbatch + closure-captured by activation-checkpoint custom_forward => recompute
# safe under PP 1F1B), matching slime. With ``--qkv-format bshd`` packed_seq_params is None, so
# we fall back to a thread-local dict keyed by layer_number. The fallback is correct for
# sequentially-executed micro-batches WITHOUT full activation recompute (each micro-batch's
# forward writes the anchor before the in-stage skip layer reads it, and the next micro-batch
# overwrites before its own skip reads). bshd + activation recompute is UNSAFE and is rejected
# at forward time (see the recompute guard in CrossLayerDSAttention.forward) -- use thd there.
# (Skip layers always have their source anchor in the same PP stage; see the runtime assert.)
_HOLDER_ATTR = "_dsa_index_share_topk_holder"
_TLS = threading.local()


def _holder(packed_seq_params):
    if packed_seq_params is not None:
        h = getattr(packed_seq_params, _HOLDER_ATTR, None)
        if h is None:
            h = {}
            setattr(packed_seq_params, _HOLDER_ATTR, h)
        return h
    h = getattr(_TLS, "holder", None)
    if h is None:
        h = {}
        _TLS.holder = h
    return h


class CrossLayerDSAttention(DSAttention):
    """``DSAttention`` with GLM-5.2 cross-layer index sharing.

    Anchor (computing) layers behave like the base class but also publish their ``topk_indices``
    to a per-microbatch holder. Skip layers carry no indexer (dropped in ``__init__``) and reuse
    the most recent computing layer's ``topk_indices`` for the sparse-attention kernel.
    """

    def __init__(self, config, submodules, layer_number, *args, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        self._index_topk_freq = getattr(config, "dsa_index_topk_freq", 1) or 1
        self._skip_topk_offset = getattr(config, "dsa_index_skip_topk_offset", 0) or 0
        self._index_share = self._index_topk_freq > 1
        self._skip_topk = self._index_share and is_skip_topk_layer(
            layer_number, self._skip_topk_offset, self._index_topk_freq
        )
        self._source_layer = (
            source_compute_layer(layer_number, self._skip_topk_offset, self._index_topk_freq)
            if self._index_share
            else layer_number
        )
        # Skip layers carry NO indexer params: drop the module the base class built so the
        # parameter set matches the GLM-5.2 checkpoint (indexer weights only on computing
        # layers) and HF export / LoRA target-matching naturally omit them here.
        if self._skip_topk and hasattr(self, "indexer"):
            del self.indexer
        # The bshd holder fallback (thread-local) is NOT recompute-safe (see ``forward``); the
        # thd carrier on ``packed_seq_params`` is. Remember whether activation recompute is on so
        # the forward can reject the unsafe bshd + recompute + cross-layer combination loudly.
        self._recompute_active = self._index_share and (getattr(config, "recompute_granularity", None) is not None)

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        x,
        qr,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        backend = getattr(self.config, "dsa_attention_backend", "megatron")

        # GLM-5.1 / no cross-layer sharing. With the default (unfused) backend this stays byte-for-
        # byte the base class. With the ``tilelang`` backend we run the single-layer logic in-class so
        # the sparse-attention kernel is dispatchable -- the base ``DSAttention.forward`` calls the
        # unfused kernel directly and lives in megatron-core, so it cannot be intercepted there.
        if not self._index_share:
            if backend != "tilelang":
                return super().forward(
                    query,
                    key,
                    value,
                    attention_mask,
                    x,
                    qr,
                    attn_mask_type,
                    attention_bias,
                    packed_seq_params,
                )
            return self._compute_layer_forward(
                query, key, value, attention_mask, x, qr, attn_mask_type, packed_seq_params, holder=None
            )

        # bshd (``packed_seq_params is None``) uses the thread-local holder fallback, which is
        # NOT recompute-safe: under activation recompute a skip layer's recompute can read a stale
        # anchor top-k (the thread-local dict is not closure-captured per microbatch the way
        # ``packed_seq_params`` is). Fail loudly instead of silently producing wrong gradients.
        if packed_seq_params is None and self._recompute_active:
            raise AssertionError(
                "DSA cross-layer index-share is not recompute-safe in the bshd layout: "
                "packed_seq_params is None, so the per-microbatch top-k holder falls back to a "
                "thread-local dict that activation recompute can read stale. Use --qkv-format thd "
                "(the holder rides on packed_seq_params and is recompute-safe), or disable "
                f"activation recompute (recompute_granularity="
                f"{getattr(self.config, 'recompute_granularity', None)})."
            )

        holder = _holder(packed_seq_params)

        # ---- skip layer: reuse the anchor's top-k, no indexer compute, no indexer loss ----
        if self._skip_topk:
            if self._source_layer not in holder:
                raise AssertionError(
                    f"DSA index-share: skip layer (layer_number={self.layer_number}) needs the "
                    f"top-k of its source computing layer (layer_number={self._source_layer}), "
                    f"which did not run in this pipeline stage's forward. Ensure every PP stage "
                    f"starts on a computing layer (index_topk_freq={self._index_topk_freq}, "
                    f"index_skip_topk_offset={self._skip_topk_offset}). Holder has {sorted(holder)}."
                )
            topk_indices = holder[self._source_layer]
            return self._sparse_attention(query, key, value, topk_indices)

        # ---- anchor layer: compute top-k (base-class logic) + publish to holder ----
        return self._compute_layer_forward(
            query, key, value, attention_mask, x, qr, attn_mask_type, packed_seq_params, holder=holder
        )

    def _sparse_attention(self, query, key, value, topk_indices):
        """Dispatch the sparse-MLA attention kernel to the configured backend.

        Both branches currently call the unfused megatron-core kernel; the ``tilelang`` fused TileLang
        ``SparseMLA`` path is wired in a later step. Centralising the call here keeps the GLM-5.1 and
        GLM-5.2 forwards on a single, backend-agnostic dispatch point.
        """
        if getattr(self.config, "dsa_attention_backend", "megatron") == "tilelang":
            # TODO(fused backend, Step 4): call the TileLang SparseMLA kernel here. Until it is
            # wired the tilelang backend uses the unfused kernel so the plumbing is regression-safe.
            return unfused_dsa_fn(query, key, value, topk_indices, self.softmax_scale)
        return unfused_dsa_fn(query, key, value, topk_indices, self.softmax_scale)

    def _compute_layer_forward(
        self, query, key, value, attention_mask, x, qr, attn_mask_type, packed_seq_params, holder
    ):
        """Compute this layer's indexer top-k and run sparse attention (anchor / plain-DSA path).

        Mirrors the base ``DSAttention.forward`` but (a) routes the sparse-attention kernel through
        :meth:`_sparse_attention` so the backend is dispatchable, and (b) optionally publishes the
        top-k to ``holder`` for cross-layer sharing (``holder=None`` for GLM-5.1, which has no skip
        layers, so it behaves exactly like the base class apart from the kernel dispatch).
        """
        sq, b, np, hn = query.size()
        skv = key.size(0)
        x = x.detach()
        qr = qr.detach()

        if attn_mask_type is not None:
            assert attn_mask_type == AttnMaskType.causal, "Only causal mask is supported for now"
            float_mask = torch.triu(
                torch.full((sq, skv), float("-inf"), dtype=torch.float32, device=x.device),
                diagonal=1,
            )
        else:
            assert attention_mask.shape == (b, 1, sq, skv), "attention_mask shape mismatch"
            mask = attention_mask.squeeze()
            float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(mask, float("-inf"))

        if self.training and torch.is_grad_enabled():
            q, k, weights = self.indexer.forward_before_topk(x, qr, packed_seq_params)
            indexer_loss_coeff = getattr(self.config, "dsa_indexer_loss_coeff", 0.0)
            topk_indices, indexer_loss = FusedDSAIndexerLoss.apply(
                q,
                weights,
                k,
                query.detach(),
                key.detach(),
                self.softmax_scale,
                self.indexer.index_topk,
                indexer_loss_coeff,
                float_mask,
                getattr(self.config, "dsa_indexer_use_sparse_loss", False),
                self.indexer.pg_collection,
            )
            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_layers,
                )
            if holder is not None:
                holder[self.layer_number] = topk_indices
            output = self._sparse_attention(query, key, value, topk_indices)
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        else:
            _, topk_indices = self.indexer.forward_with_scores(
                x, qr, mask=float_mask, packed_seq_params=packed_seq_params
            )
            if holder is not None:
                holder[self.layer_number] = topk_indices
            output = self._sparse_attention(query, key, value, topk_indices)

        return output


def get_glm5_crosslayer_dsa_spec(config, backend=None):
    """megatron-core's *exact* DSA MLA spec, with the core-attention module swapped to
    :class:`CrossLayerDSAttention`.

    Rather than hand-clone ``get_dsa_module_spec_for_backend`` (which is easy to get subtly
    wrong -- e.g. it fuses the qk-layernorm into the q/kv up-projections via
    ``column_parallel_layer_norm_linear`` and sets ``q_layernorm = kv_layernorm = IdentityOp``,
    so the MLA tensor dims match the checkpoint), we call it and mutate only the one thing that
    differs: ``submodules.core_attention.module``. The indexer ModuleSpec is uniform across
    layers; skip layers drop it in ``CrossLayerDSAttention.__init__``.

    ``metainfo['fuse_input_layernorm']=False`` is set here because this path bypasses the
    dispatcher (``get_experimental_attention_variant_module_spec``) that would otherwise set it;
    this mirrors the GLM-5.1 fallback in ``glm5_bridge._build_glm5_dsa_block_spec``.
    """
    from megatron.core.models.gpt import experimental_attention_variant_module_specs as _eav

    if backend is None:
        backend = _eav._get_backend_spec_provider(config=config)
    spec = _eav.get_dsa_module_spec_for_backend(config=config, backend=backend)
    spec.submodules.core_attention.module = CrossLayerDSAttention
    # Point the MLA self-attention module at TileLangMLASelfAttention so the fused (tilelang) backend is
    # dispatchable from the MLA level (where the absorbed-latent q/kv live). With the default
    # "megatron" backend its forward delegates to MLASelfAttention.forward -> unchanged.
    from megatron.bridge.models.glm5.tilelang.tilelang_mla import TileLangMLASelfAttention

    spec.module = TileLangMLASelfAttention
    if spec.metainfo is None:
        spec.metainfo = {}
    spec.metainfo.setdefault("fuse_input_layernorm", False)
    return spec
