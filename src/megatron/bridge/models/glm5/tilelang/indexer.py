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
#
# ruff: noqa: D101, D103, E741, F401
#
# Vendored from THUDM/slime (miles_plugins/models/glm5/ops) for the GLM DSA fused attention
# backend. Kept close to upstream; the only change is removing the miles-only indexer replay
# hook. Imported lazily by glm5/tilelang/tilelang_mla.py only when dsa_attention_backend="tilelang".

import torch

from .tilelang_indexer_bwd import indexer_bwd_interface
from .tilelang_indexer_fwd import indexer_fwd_interface


def pytorch_extract_topk_scores(logits, topk_indices, dim=-1):
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=dim, index=safe_indices)
    scores = torch.where(valid_mask, scores, float("-inf"))
    return scores


def _original_topk(logits, topk):
    # Short sequence (seq_len_kv < index_topk): the indexer degenerates to dense. torch.topk cannot
    # select more entries than exist ("selected index k out of range"), so cap k and pad the
    # selection back out to the fixed `topk` width with -1 (invalid). This matches the
    # rollout-captured indexer top-k shape that R3 replay asserts ([n_tokens, topk]) and that
    # SparseMLA expects; -1 is the same sentinel masked_fill uses for out-of-window picks, which
    # downstream ignores. The long-sequence path (k == topk) is unchanged.
    k = min(topk, logits.shape[-1])
    score, indices = torch.topk(logits, k, dim=-1)
    indices = indices.to(torch.int32).masked_fill(score == -torch.inf, -1)
    if k < topk:
        pad = torch.full((*indices.shape[:-1], topk - k), -1, dtype=torch.int32, device=indices.device)
        indices = torch.cat([indices, pad], dim=-1)
    return indices


class IndexerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        logits: torch.Tensor,
        topk_indices: torch.Tensor,
    ):
        index_score = pytorch_extract_topk_scores(logits, topk_indices)
        ctx.save_for_backward(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices)
        return index_score

    @staticmethod
    def backward(ctx, grad_scores):
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices = ctx.saved_tensors
        grad_q, grad_w, grad_k = indexer_bwd_interface(index_q, weights, index_k, topk_indices, grad_scores)
        return grad_q, grad_k, grad_w, None, None, None, None


def lighting_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    topk_indices: torch.Tensor | None = None,
):
    weights_2d = weights.squeeze(-1)
    logits = indexer_fwd_interface(index_q, index_k, weights_2d, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True)

    if topk_indices is None:
        # R3 indexer replay (matched DSA top-k between rollout & training, arxiv 2510.11370): route
        # the selection through miles' indexer_replay_manager when it is present + enabled, mirroring
        # slime's indexer.py so the fused backend records/replays the indexer top-k instead of
        # recomputing it. Guarded so the vendored kernel still imports + runs standalone (no miles),
        # in which case it falls back to a plain top-k -- identical to the previous behaviour.
        try:
            from miles.utils.replay_base import indexer_replay_manager

            topk_fn = indexer_replay_manager.get_topk_fn(_original_topk, return_probs=False)
        except ImportError:
            topk_fn = _original_topk
        topk_indices = topk_fn(logits, topk)

    index_score = IndexerFunction.apply(index_q, index_k, weights_2d, cu_seqlen_ks, cu_seqlen_ke, logits, topk_indices)
    return index_score, topk_indices


def generate_varlen_mask_params(cu_seqlens):
    seq_len = cu_seqlens[-1].item()
    q_indices = torch.arange(0, seq_len, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    assert torch.all((ends - starts) > 0)
    return starts, ends
