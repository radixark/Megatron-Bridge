"""CPU coverage for packed CP ownership and remote latent-key gradients.

The helper has no Bridge/Megatron runtime dependencies, so it can also be run in
isolation with --confcutdir=tests/unit_tests/models on a CPU-only installation.
"""

import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


_HELPER = Path(__file__).resolve().parents[3] / "src/megatron/bridge/models/glm5_next/context_parallel.py"
_spec = importlib.util.spec_from_file_location("glm5_next_cp_helpers", _HELPER)
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize("layout", ["contiguous", "zigzag"])
def test_query_ownership_is_a_partition(size, layout):
    cu = torch.tensor([0, 16, 48, 56])
    positions = [cp.cp_token_positions(cu, cp_size=size, cp_rank=rank, layout=layout) for rank in range(size)]
    assert all(len(pos) == 56 // size for pos in positions)
    torch.testing.assert_close(torch.cat(positions).sort().values, torch.arange(56))


def test_zigzag_is_per_packed_sequence():
    cu = torch.tensor([0, 8, 20])
    torch.testing.assert_close(
        cp.cp_token_positions(cu, cp_size=2, cp_rank=0, layout="zigzag"),
        torch.tensor([0, 1, 6, 7, 8, 9, 10, 17, 18, 19]),
    )
    torch.testing.assert_close(
        cp.cp_token_positions(cu, cp_size=2, cp_rank=1, layout="zigzag"),
        torch.tensor([2, 3, 4, 5, 11, 12, 13, 14, 15, 16]),
    )


@pytest.mark.parametrize("layout", ["contiguous", "zigzag"])
def test_kda_segments_preserve_sample_boundaries(layout):
    cu = torch.tensor([0, 16, 48])
    segments = cp.kda_cp_segments(cu, cp_size=2, layout=layout)
    if layout == "contiguous":
        assert len(segments) == 1
        assert segments[0][0] == slice(None)
        torch.testing.assert_close(segments[0][1], cu)
    else:
        assert [rows for rows, _ in segments] == [slice(0, 8), slice(8, 24)]
        assert [boundaries.tolist() for _, boundaries in segments] == [[0, 16], [0, 32]]


@pytest.mark.parametrize(
    "cu,size,rank,layout",
    [
        ([0, 7], 2, 0, "contiguous"),
        ([0, 6, 16], 2, 0, "zigzag"),
        ([0, 8], 2, 2, "zigzag"),
        ([0, 8], 2, 0, "typo"),
        ([1, 8], 2, 0, "contiguous"),
        ([0, 0, 8], 2, 0, "contiguous"),
    ],
)
def test_invalid_packing_fails_before_collectives(cu, size, rank, layout):
    with pytest.raises(ValueError):
        cp.cp_token_positions(torch.tensor(cu), cp_size=size, cp_rank=rank, layout=layout)


def _gather_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        for layout, boundaries in [("contiguous", [0, 5, 16]), ("zigzag", [0, 8, 24])]:
            cu = torch.tensor(boundaries)
            positions = cp.cp_token_positions(cu, cp_size=2, cp_rank=rank, layout=layout)
            reference = torch.arange(boundaries[-1] * 3, dtype=torch.float64).reshape(-1, 3)
            local = reference[positions].clone().requires_grad_()
            gathered = cp.gather_cp_tokens(local, cu, group=dist.group.WORLD, layout=layout)
            torch.testing.assert_close(gathered, reference)
            # Distinct per-rank losses catch a silently detached or unsummed KV gradient.
            (gathered.square().sum() * (rank + 1)).backward()
            torch.testing.assert_close(local.grad, 6 * reference[positions])
            with torch.no_grad():
                frozen = cp.gather_cp_tokens(local, cu, group=dist.group.WORLD, layout=layout)
            assert not frozen.requires_grad
            torch.testing.assert_close(frozen, reference)
    finally:
        dist.destroy_process_group()


def test_distributed_gather_order_and_gradients(tmp_path):
    mp.spawn(_gather_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
