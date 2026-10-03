import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.peft.multi_lora import MultiLoRA
from megatron.bridge.peft.multi_lora_layers import init_adapter_slot, set_tokens_per_adapter_slot
from tests.unit_tests.peft.test_multi_lora_seeded_slot_distributed import (
    _model_parallel,
    two_rank_process_group,  # noqa: F401
)


def _gather(weight, dim=0):
    weights = [torch.empty_like(weight) for _ in range(2)]
    dist.all_gather(weights, weight.detach().contiguous())
    return torch.cat(weights, dim=dim).requires_grad_()


def _assert_close(actual, expected):
    errors = [None, None]
    error = None
    try:
        # TP reductions change FP32 summation order.
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
    except AssertionError as exc:
        error = str(exc)
    dist.all_gather_object(errors, error)
    assert not any(errors), errors


@pytest.mark.usefixtures("two_rank_process_group")
@pytest.mark.parametrize("row_parallel", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_fused_tp_gradients(row_parallel, sequence_parallel):
    with _model_parallel(tp=2, ep=1):
        config = TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=4, tensor_model_parallel_size=2,
            sequence_parallel=sequence_parallel, params_dtype=torch.float32, gradient_accumulation_fusion=False,
        )
        linear_cls = RowParallelLinear if row_parallel else ColumnParallelLinear
        base = linear_cls(
            32, 32, config=config, init_method=torch.nn.init.zeros_, bias=False,
            **({"input_is_parallel": True, "skip_bias_add": True} if row_parallel else {"gather_output": False}),
        ).cuda().requires_grad_(False)
        name = "linear_proj" if row_parallel else "linear_qkv"
        layer = MultiLoRA(target_modules=[name], n_adapters=1, dim=8).transform(base, name=name)
        init_adapter_slot(layer, 0, rank=8, alpha=8, seed=11)
        with torch.no_grad():
            layer.adapters[0].linear_out.weight.normal_(std=0.2)
        set_tokens_per_adapter_slot(layer, torch.tensor([8], device="cuda"))
        torch.manual_seed(17)
        full_x = torch.randn(8, 1, 32, device="cuda", requires_grad=True)
        rank = dist.get_rank()
        local = full_x.detach()
        if row_parallel:
            local = local.chunk(2, dim=-1)[rank]
        elif sequence_parallel:
            local = local.chunk(2)[rank]
        x = local.clone().requires_grad_()
        actual, _ = layer(x)
        adapter = layer.adapters[0]
        a = _gather(adapter.linear_in.weight, dim=-1 if row_parallel else 0)
        b = _gather(adapter.linear_out.weight)
        expected = F.linear(F.linear(full_x, a), b)
        expected_local = expected.chunk(2)[rank] if row_parallel and sequence_parallel else expected
        if not row_parallel:
            expected_local = expected.chunk(2, dim=-1)[rank]
        _assert_close(actual, expected_local)
        actual.square().sum().backward()
        expected.square().sum().backward()
        _assert_close(adapter.linear_in.weight.grad, a.grad.chunk(2, dim=-1 if row_parallel else 0)[rank])
        _assert_close(adapter.linear_out.weight.grad, b.grad.chunk(2)[rank])
        expected_dx = full_x.grad.chunk(2, dim=-1)[rank] if row_parallel else full_x.grad
        if not row_parallel and sequence_parallel:
            expected_dx = expected_dx.chunk(2)[rank]
        _assert_close(x.grad, expected_dx)


