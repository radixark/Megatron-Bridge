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

"""Slot dispatch for multi-adapter LoRA on grouped MoE experts.

This module is deliberately free of ``megatron`` / ``transformer_engine`` imports
so the routing algorithm can be unit-tested on CPU with plain tensors. The
megatron-coupled wiring (weight allocation, expert-TP collectives, checkpointing)
lives in :class:`megatron.bridge.peft.multi_lora_layers.MultiLoRAGroupedExpertLinear`,
which calls :func:`dispatch_expert_lora_by_slot` with its per-slot adapters.

Background. On an MoE expert linear the tokens arriving at the grouped GEMM have
been permuted by the MoE token dispatcher into *expert-major* order and (under
expert parallelism) shuffled across ranks. A single-adapter LoRA only needs the
expert grouping (``tokens_per_expert``), which the dispatcher already provides.
Multi-adapter LoRA additionally needs, per token, *which adapter slot* it belongs
to — a second, orthogonal grouping. That per-token slot id (``permuted_slot_ids``)
is threaded through the dispatcher alongside the hidden states; given it, this
routine applies each slot's grouped-expert adapter to just that slot's tokens.
"""

from typing import Callable, Optional, Sequence

import torch


def per_expert_slot_counts(
    expert_ids: torch.Tensor,
    permuted_slot_ids: torch.Tensor,
    num_local_experts: int,
    slot: int,
) -> torch.Tensor:
    """Count, per local expert, how many permuted tokens belong to ``slot``.

    Args:
        expert_ids: ``[T]`` local-expert id of each permuted token (expert-major).
        permuted_slot_ids: ``[T]`` adapter slot id of each permuted token.
        num_local_experts: number of local experts ``E`` (the returned length).
        slot: the adapter slot to count for.

    Returns:
        ``[E]`` int64 tensor whose entries sum to ``(permuted_slot_ids == slot).sum()``.
    """
    mask = permuted_slot_ids == slot
    selected_experts = expert_ids[mask]
    return torch.bincount(selected_experts, minlength=num_local_experts).to(torch.int64)


def dispatch_expert_lora_by_slot(
    permuted_hidden: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    permuted_slot_ids: Optional[torch.Tensor],
    num_local_experts: int,
    out_features: int,
    adapter_fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
    active_slots: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Compute the summed multi-adapter LoRA delta for one grouped expert linear.

    For each active adapter slot ``s`` present in ``permuted_slot_ids``:

    1. gather that slot's permuted tokens with a *stable* mask (so they stay
       expert-major, i.e. contiguous per expert);
    2. compute the slot's per-expert token counts ``tpe_s`` (length ``E``);
    3. run ``adapter_fn(s, x_s, tpe_s)`` — the slot's single-adapter grouped-expert
       adapter — to get that slot's delta for its tokens;
    4. scatter the delta back to the slot's original permuted positions.

    Because each permuted token belongs to exactly one adapter slot (all top-k
    expansions of an input token inherit that token's adapter), the per-slot masks
    *partition* the permuted tokens, so the scatter is a permutation with no
    accumulation.

    Inert behavior: when ``permuted_slot_ids`` is ``None`` a zero delta is returned,
    so a wrapped expert linear falls back to base-only output whenever no slot ids
    were threaded through the dispatcher (e.g. the Megatron-LM companion channel is
    absent). Positions with a negative slot id (dispatcher drop/pad sentinels) get no
    delta.

    Args:
        permuted_hidden: ``[T, H_in]`` expert-major permuted input to the grouped linear.
        tokens_per_expert: ``[E]`` (tensor or list) token count per local expert; sums to ``T``.
        permuted_slot_ids: ``[T]`` int adapter slot per permuted token, or ``None``.
        num_local_experts: number of local experts ``E``.
        out_features: output width of the grouped linear (shape of the returned delta).
        adapter_fn: ``(slot, x_s, tokens_per_expert_s) -> delta_s`` with
            ``x_s: [n_s, H_in]`` and ``delta_s: [n_s, out_features]``.
        active_slots: optional explicit slot ids to consider; defaults to the distinct
            non-negative slot ids present in ``permuted_slot_ids``.

    Returns:
        ``[T, out_features]`` delta to add to the base grouped-linear output.
    """
    total_tokens = permuted_hidden.shape[0]
    delta = permuted_hidden.new_zeros((total_tokens, out_features))
    if permuted_slot_ids is None:
        return delta
    # Co-locate the routing tensors with the hidden states. Callers legitimately hand
    # tokens_per_expert / slot ids as CPU tensors or Python lists (Megatron converts
    # tokens_per_expert to a CPU list before the grouped GEMM), so coerce rather than
    # assume same-device inputs.
    permuted_slot_ids = permuted_slot_ids.to(permuted_hidden.device)
    if permuted_slot_ids.shape[0] != total_tokens:
        raise ValueError(
            f"permuted_slot_ids has {permuted_slot_ids.shape[0]} entries but there are "
            f"{total_tokens} permuted tokens"
        )

    tpe = tokens_per_expert
    if not isinstance(tpe, torch.Tensor):
        tpe = torch.tensor(list(tpe))
    tpe = tpe.to(device=permuted_hidden.device, dtype=torch.int64)
    if int(tpe.sum().item()) != total_tokens:
        raise ValueError(
            f"tokens_per_expert sums to {int(tpe.sum().item())} but there are {total_tokens} tokens"
        )
    if tpe.shape[0] != num_local_experts:
        raise ValueError(
            f"tokens_per_expert has {tpe.shape[0]} entries, expected num_local_experts={num_local_experts}"
        )

    # expert-major id per permuted token, e.g. [0,0,1,1,1,2,...]
    expert_ids = torch.repeat_interleave(
        torch.arange(num_local_experts, device=permuted_hidden.device), tpe
    )

    if active_slots is None:
        present = torch.unique(permuted_slot_ids)
        slots = [int(s) for s in present.tolist() if s >= 0]
    else:
        # De-duplicate while preserving order so a slot is never processed (and its
        # positions overwritten) twice.
        slots = list(dict.fromkeys(int(s) for s in active_slots))

    for slot in slots:
        mask = permuted_slot_ids == slot
        if not bool(mask.any()):
            continue
        idx = mask.nonzero(as_tuple=True)[0]
        x_s = permuted_hidden.index_select(0, idx)
        tpe_s = per_expert_slot_counts(expert_ids, permuted_slot_ids, num_local_experts, slot)
        out_s = adapter_fn(slot, x_s, tpe_s)
        if out_s.shape != (x_s.shape[0], out_features):
            raise ValueError(
                f"adapter_fn for slot {slot} returned shape {tuple(out_s.shape)}, "
                f"expected {(x_s.shape[0], out_features)}"
            )
        delta.index_copy_(0, idx, out_s.to(device=delta.device, dtype=delta.dtype))

    return delta
