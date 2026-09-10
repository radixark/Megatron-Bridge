"""Regression coverage for consumers that absorb a LoRA projection matrix."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from megatron.bridge.peft.lora_layers import LoRALinear


class _Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight), None


class _Adapter(nn.Module):
    def __init__(self, *, dtype, dropout=0.0):
        super().__init__()
        self.dim = 4
        self.alpha = 7
        self.linear_in = nn.Linear(6, self.dim, bias=False, dtype=dtype)
        self.linear_out = nn.Linear(self.dim, 8, bias=False, dtype=dtype)
        self.activation = nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear_out(self.activation(self.linear_in(self.dropout(x)))) * (self.alpha / self.dim)


def _layer(*, dtype=torch.float64, dropout=0.0):
    base = _Linear(6, 8, bias=False, dtype=dtype)
    base.weight.requires_grad_(False)
    return LoRALinear(base, _Adapter(dtype=dtype, dropout=dropout))


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_absorbed_weight_matches_forward_and_both_factor_gradients(dtype):
    torch.manual_seed(37)
    absorbed = _layer(dtype=dtype)
    ordinary = deepcopy(absorbed)
    x_absorbed = torch.randn(5, 6, dtype=dtype, requires_grad=True)
    x_ordinary = x_absorbed.detach().clone().requires_grad_()

    actual = F.linear(x_absorbed, absorbed.weight)
    expected, _ = ordinary(x_ordinary)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x_absorbed.grad, x_ordinary.grad)
    for factor in ("linear_in", "linear_out"):
        actual_grad = getattr(absorbed.adapter, factor).weight.grad
        assert actual_grad is not None and torch.count_nonzero(actual_grad) > 0
        torch.testing.assert_close(actual_grad, getattr(ordinary.adapter, factor).weight.grad)
    assert absorbed.to_wrap.weight.grad is None


@pytest.mark.unit
def test_weight_follows_optimizer_steps_and_adapter_enablement_without_mutating_base():
    layer = _layer()
    base = layer.to_wrap.weight.detach().clone()
    before = layer.weight.detach().clone()
    optimizer = torch.optim.SGD(layer.adapter.parameters(), lr=0.01)
    layer.weight.square().sum().backward()
    optimizer.step()
    assert not torch.equal(layer.weight, before)
    torch.testing.assert_close(layer.to_wrap.weight, base, rtol=0, atol=0)
    layer.disable_adapter_layers()
    assert layer.weight is layer.to_wrap.weight
    layer.enable_adapter_layers()
    assert not torch.equal(layer.weight, base)
    assert "weight" not in layer._parameters
    torch.testing.assert_close(layer.state_dict()["weight"], base, rtol=0, atol=0)


@pytest.mark.unit
def test_training_dropout_is_rejected_but_eval_and_disabled_adapters_work():
    layer = _layer(dropout=0.2)
    with pytest.raises(ValueError, match="zero LoRA dropout"):
        _ = layer.weight
    layer.disable_adapter_layers()
    assert layer.weight is layer.to_wrap.weight
    layer.enable_adapter_layers()
    layer.eval()
    x = torch.randn(5, 6, dtype=torch.float64)
    torch.testing.assert_close(F.linear(x, layer.weight), layer(x)[0])


@pytest.mark.unit
def test_nonlinear_adapter_cannot_be_absorbed():
    layer = _layer()
    layer.adapter.activation = nn.ReLU()
    with pytest.raises(ValueError, match="identity adapter activation"):
        _ = layer.weight


@pytest.mark.unit
@pytest.mark.parametrize("fused_norm", [False, True])
@pytest.mark.parametrize("adapter_enabled", [False, True])
def test_tilelang_absorb_normalizes_kv_once_and_uses_effective_weight(monkeypatch, fused_norm, adapter_enabled):
    mla = pytest.importorskip("megatron.bridge.models.glm5.tilelang.tilelang_mla")
    torch.manual_seed(37)
    projection = _layer(dtype=torch.float32)
    if not adapter_enabled:
        projection.disable_adapter_layers()
    norm = nn.RMSNorm(6, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 2, 6))
    if fused_norm:
        projection.to_wrap.layer_norm_weight = norm.weight
    config = SimpleNamespace(
        qk_head_dim=2,
        qk_pos_emb_head_dim=2,
        kv_lora_rank=6,
        v_head_dim=2,
        sequence_parallel=False,
        layernorm_epsilon=1e-6,
    )
    attention = SimpleNamespace(
        config=config,
        num_attention_heads_per_partition=2,
        q_head_dim=4,
        linear_q_down_proj=_Linear(6, 4, bias=False),
        linear_kv_down_proj=_Linear(6, 8, bias=False),
        linear_q_up_proj=_Linear(4, 8, bias=False),
        linear_kv_up_proj=projection,
        q_layernorm=nn.Identity(),
        kv_layernorm=nn.Identity() if fused_norm else norm,
        rotary_pos_emb=Mock(return_value=torch.empty(0)),
        _fuse_rope=lambda x, *_args, **_kwargs: x,
    )
    attention._kv_up_proj_weight_and_norm = lambda: mla.TileLangMLASelfAttention._kv_up_proj_weight_and_norm(attention)
    monkeypatch.setattr(mla, "gather_from_sequence_parallel_region", lambda x, **_kwargs: x)
    monkeypatch.setattr(mla.parallel_state, "get_context_parallel_group", lambda: None)
    monkeypatch.setattr(mla.parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
    hidden = torch.randn(4, 1, 6)
    packed = SimpleNamespace(cu_seqlens_q=None, cu_seqlens_kv=None)
    query, key, w_vc, _ = mla.TileLangMLASelfAttention._absorb_query_key_value_tensors(attention, hidden, packed)

    raw_kv, rope_key = attention.linear_kv_down_proj(hidden)[0].split([6, 2], dim=-1)
    expected_key = torch.cat((F.rms_norm(raw_kv, (6,), norm.weight, 1e-6), rope_key), dim=-1)
    torch.testing.assert_close(key, expected_key)
    effective_weight = projection.to_wrap.weight
    if adapter_enabled:
        effective_weight = effective_weight + (
            projection.adapter.linear_out.weight @ projection.adapter.linear_in.weight
        ) * (projection.adapter.alpha / projection.adapter.dim)
    per_head = effective_weight.view(2, 4, 6)
    torch.testing.assert_close(w_vc, per_head[:, 2:])
    raw_q = attention.linear_q_up_proj(attention.linear_q_down_proj(hidden)[0].squeeze(1))[0].view(4, 2, 4)
    expected_query = torch.cat((torch.einsum("thd,hdm->thm", raw_q[..., :2], per_head[:, :2]), raw_q[..., 2:]), dim=-1)
    torch.testing.assert_close(query, expected_query)
    if adapter_enabled:
        (query.square().sum() + w_vc.square().sum()).backward()
        for factor in (projection.adapter.linear_in, projection.adapter.linear_out):
            assert factor.weight.grad is not None and torch.count_nonzero(factor.weight.grad) > 0


def _tensor_parallel_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        for layout in ("column", "row", "replicated"):
            torch.manual_seed(37)
            reference = _layer()
            actual = deepcopy(reference)
            a = reference.adapter.linear_in.weight
            b = reference.adapter.linear_out.weight
            base = reference.to_wrap.weight
            inputs = torch.randn(5, 6, dtype=torch.float64)
            a_slice = (
                (slice(None), slice(rank * 3, (rank + 1) * 3))
                if layout == "row"
                else (slice(rank * 2, (rank + 1) * 2), slice(None))
            )
            b_slice = (slice(rank * 4, (rank + 1) * 4), slice(None))
            actual.adapter.linear_in.weight = nn.Parameter(a.detach()[a_slice].clone())
            actual.adapter.linear_out.weight = nn.Parameter(b.detach()[b_slice].clone())
            actual.adapter.linear_in.tp_group = dist.group.WORLD
            if layout == "column":
                actual.to_wrap.weight = nn.Parameter(base.detach()[b_slice].clone(), requires_grad=False)
            elif layout == "row":
                actual.to_wrap.weight = nn.Parameter(
                    base.detach()[:, rank * 3 : (rank + 1) * 3].clone(), requires_grad=False
                )

            full_weight = base + (b @ a) * (reference.adapter.alpha / reference.adapter.dim)
            losses = []
            for peer in range(2):
                x = inputs * (peer + 1)
                weight = full_weight
                if layout == "column":
                    weight = weight[peer * 4 : (peer + 1) * 4]
                elif layout == "row":
                    x = x[:, peer * 3 : (peer + 1) * 3]
                    weight = weight[:, peer * 3 : (peer + 1) * 3]
                expected = F.linear(x, weight)
                losses.append(expected.square().sum())
                if peer == rank:
                    output = F.linear(x, actual.weight)
                    torch.testing.assert_close(output, expected)
            output.square().sum().backward()
            sum(losses).backward()
            torch.testing.assert_close(actual.adapter.linear_in.weight.grad, a.grad[a_slice])
            torch.testing.assert_close(actual.adapter.linear_out.weight.grad, b.grad[b_slice])
            assert torch.count_nonzero(actual.adapter.linear_in.weight.grad) > 0
            assert torch.count_nonzero(actual.adapter.linear_out.weight.grad) > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_tensor_parallel_column_row_and_replicated_weight_gradients(tmp_path):
    mp.spawn(_tensor_parallel_worker, args=(f"file://{tmp_path}/rendezvous",), nprocs=2, join=True)
