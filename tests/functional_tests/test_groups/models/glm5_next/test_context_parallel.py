"""Two-GPU GLM CP numerics, without downloading a model checkpoint.

Run with torch.distributed.run --nproc_per_node=2 -m pytest on this file.
Exercises the production KDA forward with small projection modules and the real
FLA convolution/KDA kernels, plus the real pooled-key and SparseMLA kernels.
"""

import copy
import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2,
        reason="requires a two-GPU torch.distributed.run launch",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def cp_group():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    yield
    dist.destroy_process_group()


class Projection(nn.Module):
    """Small projection preserving the Megatron (tensor, bias) contract."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x), None


def _relative_error(actual, expected, limit):
    error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8)
    assert error.item() < limit, f"relative L2 error {error.item()} >= {limit}"


def _kda_layer(cp_size, layout):
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.cp import build_cp_context

    from megatron.bridge.models.glm5_next.kda import Glm5NextKDAAttention

    # Use the production forward without requiring a full TE/MoE transformer.
    layer = Glm5NextKDAAttention.__new__(Glm5NextKDAAttention)
    nn.Module.__init__(layer)
    layer.cp_size, layer.cp_layout = cp_size, layout
    layer.pg_collection = SimpleNamespace(cp=dist.group.WORLD if cp_size > 1 else None)
    layer.build_cp_context = build_cp_context if cp_size > 1 else None
    layer.local_num_heads, layer.head_dim = 2, 64
    layer.local_projection_size = 128
    layer.conv_kernel_size, layer.gate_lower_bound = 4, -5.0
    for name in ("q", "k", "v", "f_b", "g_b"):
        setattr(layer, f"linear_{name}", Projection(64, 128))
    for name in ("f_a", "g_a"):
        setattr(layer, f"linear_{name}", Projection(64, 64))
    layer.linear_b = Projection(64, 2)
    layer.linear_proj = Projection(128, 64)
    layer.conv1d = ShortConvolution(384, kernel_size=4, activation="silu", bias=False)
    layer.A_log = nn.Parameter(torch.zeros(2, dtype=torch.float32, device="cuda"))
    layer.dt_bias = nn.Parameter(torch.zeros(128, dtype=torch.float32, device="cuda"))
    layer.o_norm = FusedRMSNormGated(64, eps=1e-5, activation="sigmoid")
    layer = layer.to(device="cuda", dtype=torch.bfloat16)
    layer.A_log.data = layer.A_log.data.float()
    layer.dt_bias.data = layer.dt_bias.data.float()
    return layer


@pytest.mark.parametrize("layout", ["contiguous", "zigzag"])
@pytest.mark.parametrize("recompute", [False, True])
def test_kda_forward_and_backward(layout, recompute):
    from torch.utils.checkpoint import checkpoint

    from megatron.bridge.models.glm5_next.context_parallel import cp_token_positions

    torch.manual_seed(321)
    reference = _kda_layer(1, layout)
    parallel = copy.deepcopy(reference)
    parallel.cp_size = 2
    parallel.pg_collection = SimpleNamespace(cp=dist.group.WORLD)
    from fla.ops.cp import build_cp_context

    parallel.build_cp_context = build_cp_context
    # Contiguous rank 0's shard ends inside the second packed sequence.
    cu = torch.tensor([0, 256, 1024], device="cuda", dtype=torch.int32)
    packed = SimpleNamespace(cu_seqlens_q=cu, qkv_format="thd")
    x = torch.randn(1024, 1, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn_like(x)
    expected = reference(x, packed_seq_params=packed)[0]
    (expected.float() * upstream.float()).sum().backward()
    positions = cp_token_positions(cu, cp_size=2, cp_rank=dist.get_rank(), layout=layout)
    local = x.detach()[positions].requires_grad_()

    def forward(inp):
        return parallel(inp, packed_seq_params=packed)[0]

    actual = checkpoint(forward, local, use_reentrant=False) if recompute else forward(local)
    (actual.float() * upstream[positions].float()).sum().backward()
    _relative_error(actual, expected[positions], 0.025)
    _relative_error(local.grad, x.grad[positions], 0.04)
    for (name, param), (_, ref_param) in zip(parallel.named_parameters(), reference.named_parameters(), strict=True):
        assert param.grad is not None, name
        dist.all_reduce(param.grad)
        _relative_error(param.grad, ref_param.grad, 0.06)


@pytest.mark.parametrize("layout", ["contiguous", "zigzag"])
def test_pooled_selection_matches_global_queries(layout, monkeypatch):
    from megatron.bridge.models.glm5_next import kpool_indexer as indexer
    from megatron.bridge.models.glm5_next.context_parallel import cp_token_positions, gather_cp_tokens

    monkeypatch.setattr(indexer, "_indexer_replay_manager", lambda: None)
    torch.manual_seed(654)
    cu = torch.tensor([0, 260, 1032], device="cuda", dtype=torch.int32)
    q = torch.randn(1032, 64, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(1032, 128, device="cuda", dtype=torch.bfloat16)
    gates = torch.randn_like(keys)
    ape = torch.randn(4, 128, device="cuda")
    weights = torch.randn(1032, 64, device="cuda")
    positions = cp_token_positions(cu, cp_size=2, cp_rank=dist.get_rank(), layout=layout)
    global_keys = gather_cp_tokens(keys[positions], cu, group=dist.group.WORLD, layout=layout)
    global_gates = gather_cp_tokens(gates[positions], cu, group=dist.group.WORLD, layout=layout)
    torch.testing.assert_close(global_keys, keys)
    pooled = indexer.build_pooled_keys(global_keys, global_gates, ape, cu, 4)
    kwargs = dict(
        pooled_k=pooled, cu_seqlens=cu, pool_cu_seqlens=indexer.pool_boundaries(cu, 4), index_topk=64, kpool=4
    )
    expected = indexer.kpool_select_topk(q, head_weights=weights, **kwargs)
    actual = indexer.kpool_select_topk(
        q[positions], head_weights=weights[positions], query_positions=positions, **kwargs
    )
    torch.testing.assert_close(actual, expected[positions], rtol=0, atol=0)
    # Force several query blocks, including a final partial block.
    monkeypatch.setattr(indexer, "_QUERY_BLOCK", 128)
    blocked = indexer.kpool_select_topk(
        q[positions], head_weights=weights[positions], query_positions=positions, **kwargs
    )
    torch.testing.assert_close(blocked, actual, rtol=0, atol=0)
    seq = torch.searchsorted(cu, positions, right=True) - 1
    for row, pos in enumerate(positions):
        selected = actual[row, 0]
        selected = selected[selected >= 0]
        assert bool(((selected >= cu[seq[row]]) & (selected <= pos)).all())


@pytest.mark.parametrize("layout", ["contiguous", "zigzag"])
def test_sparse_attention_remote_key_gradients(layout):
    from megatron.bridge.models.glm5.tilelang import SparseMLA
    from megatron.bridge.models.glm5_next.context_parallel import cp_token_positions, gather_cp_tokens

    torch.manual_seed(987)
    cu = torch.tensor([0, 128, 512], device="cuda", dtype=torch.int32)
    positions = cp_token_positions(cu, cp_size=2, cp_rank=dist.get_rank(), layout=layout)
    q = torch.randn(512, 8, 576, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(512, 1, 576, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    base = cu[torch.searchsorted(cu, torch.arange(512, device="cuda"), right=True) - 1]
    indices = torch.arange(64, device="cuda")[None, :] + base[:, None]
    indices = torch.where(indices <= torch.arange(512, device="cuda")[:, None], indices, -1).int().unsqueeze(1)
    upstream = torch.randn(512, 8, 512, device="cuda", dtype=torch.bfloat16)
    expected = SparseMLA.apply(q, kv, indices, 0.0625)[0]
    (expected.float() * upstream.float()).sum().backward()
    local_q, local_kv = (value.detach()[positions].requires_grad_() for value in (q, kv))
    keys = gather_cp_tokens(local_kv, cu, group=dist.group.WORLD, layout=layout)
    actual = SparseMLA.apply(local_q, keys, indices[positions], 0.0625)[0]
    (actual.float() * upstream[positions].float()).sum().backward()
    _relative_error(actual, expected[positions], 0.01)
    _relative_error(local_q.grad, q.grad[positions], 0.02)
    _relative_error(local_kv.grad, kv.grad[positions], 0.025)
