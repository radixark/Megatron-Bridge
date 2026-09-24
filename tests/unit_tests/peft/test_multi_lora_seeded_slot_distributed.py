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

"""Two-rank tests for seeded multi-LoRA slot initialisation (``init_adapter_slot(..., seed=)``).

The seeded path rebuilds the RNG-tracker streams from the seed. It must keep
Megatron-Core's topology semantics: tensor-parallel and expert-parallel shards
draw distinct values (they assemble one logical weight) while data-parallel
replicas stay bitwise identical.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/peft/test_multi_lora_seeded_slot_distributed.py
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TERowParallelLinear
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
    get_expert_parallel_rng_tracker_name,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.peft import multi_lora_layers as multi_lora_layers_module
from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear, init_adapter_slot, set_tokens_per_adapter_slot
from megatron.bridge.peft.utils import init_method_normal


_WORLD_SIZE = 2


@pytest.fixture(scope="module")
def two_rank_process_group() -> Iterator[None]:
    """Provide the two-rank NCCL process group; the model-parallel layout is set per test."""
    if int(os.environ.get("WORLD_SIZE", "1")) != _WORLD_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(backend="nccl")
    try:
        yield
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


@contextmanager
def _model_parallel(*, tp: int, ep: int) -> Iterator[None]:
    """Initialize a two-rank layout (TP=2, DP=2 via TP=1, or EP=2) with a freshly seeded tracker."""
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(2026, force_reset_rng=True)
    try:
        yield
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()


def _all_gather(tensor: torch.Tensor) -> list[torch.Tensor]:
    tensor = tensor.detach().contiguous().cuda()
    gathered = [torch.empty_like(tensor) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, tensor)
    return gathered


def _build_multi_lora(tp: int) -> MultiLoRALinear:
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,  # divisible by tensor_model_parallel_size=2
        sequence_parallel=False,
        tensor_model_parallel_size=tp,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    base = ColumnParallelLinear(
        16,
        16,
        config=config,
        init_method=init_method_normal(0.02),
        bias=False,
        gather_output=False,
    ).cuda()
    mlora = MultiLoRALinear(
        to_wrap=base,
        n_adapters=2,
        dim=8,
        alpha=16,
        full_name="linear_qkv",
        column_init_method="xavier",
        row_init_method="zero",
        dropout=0.0,
    )
    mlora.adapters.to(device="cuda", dtype=torch.bfloat16)
    return mlora


@pytest.mark.gpu
def test_seeded_slot_init_keeps_tp_shards_distinct_and_reproducible(two_rank_process_group) -> None:
    """Under TP=2 each rank owns a different slice of LoRA-A: distinct draws, reproducible per rank."""
    with _model_parallel(tp=_WORLD_SIZE, ep=1):
        mlora = _build_multi_lora(tp=_WORLD_SIZE)
        init_adapter_slot([mlora], 0, rank=8, alpha=16, seed=99)
        first = mlora.adapters[0].linear_in.weight.detach().clone()

        shards = _all_gather(first)
        assert not torch.equal(shards[0], shards[1])

        mlora.clear_adapter_slot(0)  # arbitrary re-init from the live tracker state
        init_adapter_slot([mlora], 0, rank=8, alpha=16, seed=99)
        torch.testing.assert_close(mlora.adapters[0].linear_in.weight, first, rtol=0, atol=0)


@pytest.mark.gpu
def test_seeded_slot_init_is_identical_across_dp(two_rank_process_group) -> None:
    """Under TP=1 the two ranks are DP replicas and must end up bitwise identical."""
    with _model_parallel(tp=1, ep=1):
        mlora = _build_multi_lora(tp=1)
        init_adapter_slot([mlora], 0, rank=8, alpha=16, seed=99)

        shards = _all_gather(mlora.adapters[0].linear_in.weight)
        torch.testing.assert_close(shards[1], shards[0], rtol=0, atol=0)


@pytest.mark.gpu
def test_seeded_reseed_keeps_expert_stream_distinct_across_ep(two_rank_process_group) -> None:
    """Under EP=2 the expert stream differs per EP rank while the data-parallel stream matches."""
    with _model_parallel(tp=1, ep=_WORLD_SIZE):
        tracker = get_cuda_rng_tracker()
        saved_states = tracker.get_states()
        saved_default_rng_state = torch.cuda.get_rng_state()
        try:
            multi_lora_layers_module._reseed_rng_tracker(99)
            with tracker.fork(get_expert_parallel_rng_tracker_name()):
                expert_draw = torch.randn(8, device="cuda")
            with tracker.fork(get_data_parallel_rng_tracker_name()):
                data_parallel_draw = torch.randn(8, device="cuda")
        finally:
            tracker.set_states(saved_states)
            torch.cuda.set_rng_state(saved_default_rng_state)

        expert_draws = _all_gather(expert_draw)
        data_parallel_draws = _all_gather(data_parallel_draw)
        assert not torch.equal(expert_draws[0], expert_draws[1])
        torch.testing.assert_close(data_parallel_draws[1], data_parallel_draws[0], rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("base_cls", [TERowParallelLinear, RowParallelLinear])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_row_parallel_multi_lora_output_and_gradients(
    two_rank_process_group, base_cls, sequence_parallel, bias
):
    with _model_parallel(tp=2, ep=1):
        rank = parallel_state.get_tensor_model_parallel_rank()
        config = TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=2,
            tensor_model_parallel_size=2, sequence_parallel=sequence_parallel,
            params_dtype=torch.float32, gradient_accumulation_fusion=False,
        )
        base = base_cls(
            32, 32, config=config, init_method=init_method_normal(0.02),
            bias=bias, is_expert=False, input_is_parallel=True, skip_bias_add=False,
        )
        base.requires_grad_(False)
        layer = MultiLoRALinear(base, n_adapters=2, dim=8, alpha=8, full_name="linear_proj")
        for slot in range(2):
            init_adapter_slot([layer], slot, rank=8, alpha=8)
        set_tokens_per_adapter_slot([layer], torch.tensor([3, 5], device="cuda", dtype=torch.int32))
        # Dyadic values isolate collective/layout errors from the TE base GEMM's rounding floor.
        x = ((torch.arange(8 * 32, device="cuda").reshape(8, 32) % 7 - 3).float() / 8).requires_grad_()
        w = (torch.arange(32 * 32, device="cuda").reshape(32, 32) % 5 - 2).float() / 16
        a = ((torch.arange(2 * 8 * 32, device="cuda").reshape(2, 8, 32) % 3 - 1).float() / 32).requires_grad_()
        b = ((torch.arange(2 * 32 * 8, device="cuda").reshape(2, 32, 8) % 5 - 2).float() / 32).requires_grad_()
        with torch.no_grad():
            base.weight.copy_(w.chunk(2, dim=1)[rank])
            if bias:
                base.bias.fill_(0.125)
            for slot, adapter in enumerate(layer.adapters):
                adapter.linear_in.weight.copy_(a[slot].chunk(2, dim=1)[rank])
                adapter.linear_out.weight.copy_(b[slot].chunk(2, dim=0)[rank])
        local_x = x.detach().chunk(2, dim=1)[rank].contiguous().requires_grad_()
        expected_base = F.linear(x, w) + (0.125 if bias else 0)
        layer.disable_adapter_layers()
        actual_base, output_bias = layer(local_x)
        assert output_bias is None
        torch.testing.assert_close(
            actual_base, expected_base.chunk(2, dim=0)[rank] if sequence_parallel else expected_base,
            atol=1e-6, rtol=1e-6,
        )
        layer.enable_adapter_layers()
        actual, output_bias = layer(local_x)
        assert output_bias is None
        expected = expected_base + torch.cat([
            F.linear(F.linear(tokens, aa), bb) for tokens, aa, bb in zip(x.split((3, 5)), a, b)
        ])
        torch.testing.assert_close(
            actual, expected.chunk(2, dim=0)[rank] if sequence_parallel else expected,
            atol=1e-6, rtol=1e-6,
        )
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(local_x.grad, x.grad.chunk(2, dim=1)[rank], atol=1e-6, rtol=1e-6)
        for slot, adapter in enumerate(layer.adapters):
            torch.testing.assert_close(adapter.linear_in.weight.grad, a.grad[slot].chunk(2, dim=1)[rank], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(adapter.linear_out.weight.grad, b.grad[slot].chunk(2, dim=0)[rank], atol=1e-6, rtol=1e-6)
