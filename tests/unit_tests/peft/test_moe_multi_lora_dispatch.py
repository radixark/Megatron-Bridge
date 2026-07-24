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

"""Unit tests for the pure-torch multi-adapter grouped-expert slot dispatch.

The routine under test is deliberately megatron-free, so these run on CPU. They
validate the (slot x expert) routing against an independent per-token reference,
plus the correctness-critical guarantees: inert (zero delta) when no slot ids are
threaded, drop/pad sentinel handling, and that the per-slot masks partition the
permuted tokens (no double counting).
"""

import torch

from megatron.bridge.peft.moe_multi_lora_dispatch import (
    dispatch_expert_lora_by_slot,
    per_expert_slot_counts,
)


H_IN, OUT, E, N_AD = 4, 3, 2, 3


def _fixture():
    torch.manual_seed(0)
    tokens_per_expert = torch.tensor([3, 4])  # T = 7
    total = int(tokens_per_expert.sum())
    x = torch.randn(total, H_IN, dtype=torch.float64)
    weights = {(s, e): torch.randn(H_IN, OUT, dtype=torch.float64) for s in range(N_AD) for e in range(E)}
    expert_ids = torch.repeat_interleave(torch.arange(E), tokens_per_expert)
    return tokens_per_expert, total, x, weights, expert_ids


def _adapter_fn(weights):
    """A grouped-expert adapter that applies weights[(slot, e)] to expert-e's block."""

    def fn(slot, x_s, tpe_s):
        outs = []
        start = 0
        for e, n in enumerate(tpe_s.tolist()):
            outs.append(x_s[start : start + n] @ weights[(slot, e)])
            start += n
        return torch.cat(outs, dim=0) if outs else x_s.new_zeros((0, OUT))

    return fn


def _oracle(x, weights, expert_ids, slot_ids, total):
    """Independent reference: token i -> x_i @ W[slot_i, expert_i], 0 when slot_i < 0."""
    out = torch.zeros(total, OUT, dtype=torch.float64)
    for i in range(total):
        s = int(slot_ids[i])
        if s < 0:
            continue
        out[i] = x[i] @ weights[(s, int(expert_ids[i]))]
    return out


def test_mixed_slots_match_independent_oracle():
    tpe, total, x, w, expert_ids = _fixture()
    slot_ids = torch.tensor([0, 1, 0, 2, 0, 1, 1])  # interleaved within each expert block
    got = dispatch_expert_lora_by_slot(x, tpe, slot_ids, E, OUT, _adapter_fn(w))
    assert torch.allclose(got, _oracle(x, w, expert_ids, slot_ids, total))


def test_none_slot_ids_is_inert():
    tpe, total, x, w, _ = _fixture()
    got = dispatch_expert_lora_by_slot(x, tpe, None, E, OUT, _adapter_fn(w))
    assert got.shape == (total, OUT)
    assert torch.count_nonzero(got).item() == 0


def test_negative_sentinel_gets_no_delta():
    tpe, total, x, w, expert_ids = _fixture()
    slot_ids = torch.tensor([0, -1, 0, 2, -1, 1, 1])
    got = dispatch_expert_lora_by_slot(x, tpe, slot_ids, E, OUT, _adapter_fn(w))
    assert torch.allclose(got, _oracle(x, w, expert_ids, slot_ids, total))
    assert torch.count_nonzero(got[1]).item() == 0
    assert torch.count_nonzero(got[4]).item() == 0


def test_single_slot():
    tpe, total, x, w, expert_ids = _fixture()
    slot_ids = torch.ones(total, dtype=torch.long)
    got = dispatch_expert_lora_by_slot(x, tpe, slot_ids, E, OUT, _adapter_fn(w))
    assert torch.allclose(got, _oracle(x, w, expert_ids, slot_ids, total))


def test_active_slots_restricts_contribution():
    tpe, total, x, w, expert_ids = _fixture()
    slot_ids = torch.tensor([0, 1, 0, 2, 0, 1, 1])
    got = dispatch_expert_lora_by_slot(x, tpe, slot_ids, E, OUT, _adapter_fn(w), active_slots=[0])
    masked = torch.where(slot_ids == 0, slot_ids, torch.full_like(slot_ids, -1))
    assert torch.allclose(got, _oracle(x, w, expert_ids, masked, total))


def test_per_expert_slot_counts_partition():
    tpe, total, _, _, expert_ids = _fixture()
    slot_ids = torch.tensor([0, 1, 0, 2, 0, 1, 1])
    # counts per slot sum to that slot's token count, and across slots sum to total.
    per_slot = [per_expert_slot_counts(expert_ids, slot_ids, E, s) for s in range(N_AD)]
    for s in range(N_AD):
        assert int(per_slot[s].sum()) == int((slot_ids == s).sum())
    assert sum(int(c.sum()) for c in per_slot) == total


def test_shape_mismatch_raises():
    tpe, total, x, w, _ = _fixture()
    bad = torch.zeros(total + 1, dtype=torch.long)
    try:
        dispatch_expert_lora_by_slot(x, tpe, bad, E, OUT, _adapter_fn(w))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_list_tokens_per_expert():
    # Megatron converts tokens_per_expert to a Python list before the grouped GEMM.
    _, total, x, w, expert_ids = _fixture()
    slot_ids = torch.tensor([0, 1, 0, 2, 0, 1, 1])
    got = dispatch_expert_lora_by_slot(x, [3, 4], slot_ids, E, OUT, _adapter_fn(w))
    assert torch.allclose(got, _oracle(x, w, expert_ids, slot_ids, total))


def test_duplicate_active_slots_deduped():
    tpe, total, x, w, expert_ids = _fixture()
    slot_ids = torch.tensor([0, 1, 0, 2, 0, 1, 1])
    got = dispatch_expert_lora_by_slot(x, tpe, slot_ids, E, OUT, _adapter_fn(w), active_slots=[0, 0])
    masked = torch.where(slot_ids == 0, slot_ids, torch.full_like(slot_ids, -1))
    assert torch.allclose(got, _oracle(x, w, expert_ids, masked, total))


def test_int32_slot_ids():
    tpe, total, x, w, expert_ids = _fixture()
    slot_ids = torch.tensor([0, 1, 0, 2, 0, 1, 1], dtype=torch.int32)
    got = dispatch_expert_lora_by_slot(x, tpe, slot_ids, E, OUT, _adapter_fn(w))
    assert torch.allclose(got, _oracle(x, w, expert_ids, slot_ids, total))


def test_empty_batch():
    got = dispatch_expert_lora_by_slot(
        torch.zeros(0, H_IN, dtype=torch.float64),
        torch.tensor([0, 0]),
        torch.zeros(0, dtype=torch.long),
        E,
        OUT,
        lambda s, x_s, tpe_s: x_s.new_zeros((0, OUT)),
    )
    assert got.shape == (0, OUT)
