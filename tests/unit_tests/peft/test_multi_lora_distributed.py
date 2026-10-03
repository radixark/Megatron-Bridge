# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.peft.multi_lora import CanonicalMultiLoRA, MultiLoRA
from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot, init_adapter_slot, set_tokens_per_adapter_slot
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
@pytest.mark.parametrize(
    "name,targets,widths",
    [
        ("linear_qkv", ["linear_q", "linear_k", "linear_v"], [32, 16, 16]),
        ("linear_qkv", ["linear_q"], [64, 16, 16]),
        ("linear_fc1", ["linear_fc1_gate", "linear_fc1_up"], [32, 32]),
    ],
)
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("fused_layernorm", [False, True])
def test_projection_forward_backward_tp(name, targets, widths, sequence_parallel, fused_layernorm):
    with _model_parallel(tp=2, ep=1):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=8,
            gated_linear_unit=True,
            tensor_model_parallel_size=2,
            sequence_parallel=sequence_parallel,
            params_dtype=torch.float32,
            gradient_accumulation_fusion=False,
        )
        if fused_layernorm:
            from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

            base = TELayerNormColumnParallelLinear(
                32,
                sum(widths),
                config=config,
                init_method=torch.nn.init.zeros_,
                bias=False,
                skip_bias_add=True,
                tp_comm_buffer_name="qkv",
                gather_output=False,
                is_expert=False,
            )
        else:
            base = ColumnParallelLinear(
                32,
                sum(widths),
                config=config,
                init_method=torch.nn.init.zeros_,
                bias=False,
                gather_output=False,
            )
        base = base.cuda().requires_grad_(False)
        transform = CanonicalMultiLoRA(
            target_modules=targets, n_adapters=3, dim=8, alpha=16, lora_B_init_method="normal"
        )
        layer = transform.transform(base, name=name, prefix="decoder.layers.0.self_attention")
        for slot, rank in enumerate((8, 4, 8)):
            init_adapter_slot(layer, slot, rank=rank, alpha=rank * 2, seed=10 + slot)
        set_tokens_per_adapter_slot(layer, torch.tensor([4, 4, 0], device="cuda"))
        torch.manual_seed(100)
        full_x = torch.randn(8, 1, 32, device="cuda", requires_grad=True)
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        x = (full_x.detach().chunk(2)[tp_rank] if sequence_parallel else full_x.detach()).clone().requires_grad_()
        actual, _ = layer(x)
        ref_x = F.layer_norm(full_x, (32,), eps=config.layernorm_epsilon) if fused_layernorm else full_x
        references = []
        leaves = []
        for slot in range(2):
            parts = []
            for key, width in zip(layer._projection_sizes, widths):
                if key not in layer.adapters[slot]:
                    parts.append(full_x.new_zeros(4, 1, width))
                    continue
                adapter = layer.adapters[slot][key]
                a, b = _gather(adapter.linear_in.weight), _gather(adapter.linear_out.weight)
                leaves.append((adapter, a, b))
                parts.append(F.linear(F.linear(ref_x[slot * 4 : (slot + 1) * 4], a), b) * 2)
            if name == "linear_qkv":
                parts = [p.reshape(4, 1, 2, -1) for p in parts]
                references.append(torch.cat(parts, dim=-1).flatten(-2))
            else:
                references.append(torch.cat(parts, dim=-1))
        expected = torch.cat(references)
        if name == "linear_qkv":
            expected_local = expected.chunk(2, dim=-1)[tp_rank]
        else:
            gate, up = expected.chunk(2, dim=-1)
            expected_local = torch.cat([p.chunk(2, dim=-1)[tp_rank] for p in (gate, up)], dim=-1)
        _assert_close(actual, expected_local)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for adapter, a, b in leaves:
            _assert_close(adapter.linear_in.weight.grad, a.grad.chunk(2)[tp_rank])
            _assert_close(adapter.linear_out.weight.grad, b.grad.chunk(2)[tp_rank])
        expected_dx = full_x.grad.chunk(2)[tp_rank] if sequence_parallel else full_x.grad
        _assert_close(x.grad, expected_dx)
        assert all(p.grad is not None and p.grad.count_nonzero() == 0 for p in layer.adapters[2].parameters())
        with expose_adapter_slot(layer, 1):
            assert all(adapter.alpha / adapter.dim == 2 for adapter in layer.adapter.values())


@pytest.mark.usefixtures("two_rank_process_group")
@pytest.mark.parametrize("shared_outer", [False, True])
@pytest.mark.parametrize("packed_checkpoint", [False, True])
def test_expert_projections_export(shared_outer, packed_checkpoint):
    from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear

    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.param_mapping import FusedGatedExpertMapping
    from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
    from megatron.bridge.peft.multi_lora_layers import ExpertSlotRouting

    with _model_parallel(tp=1, ep=2):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_moe_experts=4,
            gated_linear_unit=True,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=False,
            params_dtype=torch.bfloat16,
            bf16=True,
            gradient_accumulation_fusion=False,
        )
        base = (
            TEColumnParallelGroupedLinear(
                num_gemms=2,
                input_size=32,
                output_size=64,
                config=config,
                init_method=torch.nn.init.zeros_,
                bias=False,
                skip_bias_add=True,
                is_expert=True,
            )
            .cuda()
            .requires_grad_(False)
        )
        transform = CanonicalMultiLoRA(
            target_modules=["linear_fc1_gate", "linear_fc1_up"],
            n_adapters=3,
            dim=8,
            experts_shared_outer_loras=shared_outer,
            lora_B_init_method="normal",
        )
        layer = transform.transform(base, name="linear_fc1", prefix="decoder.layers.0.mlp.experts")
        for slot, rank in enumerate((8, 4, 8)):
            init_adapter_slot(layer, slot, rank=rank, alpha=rank, seed=20 + slot)
        indices = torch.arange(16, device="cuda")
        slots, experts = indices % 2, indices // 8
        keys = slots * 2 + experts
        order = keys.argsort(stable=True)
        counts = torch.bincount(keys, minlength=6)
        layer.expert_slot_routing = ExpertSlotRouting(
            order, order.argsort(), counts.cumsum(0, dtype=torch.int32), counts.view(3, 2).sum(1), 16
        )
        x = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
        actual, _ = layer(x, [8, 8])
        expected = []
        for row in range(16):
            parts = []
            for adapter in layer.adapters[row % 2].values():
                a = adapter.linear_in.weight if shared_outer else adapter.linear_in.weight[row // 8]
                b = adapter.linear_out.weight[row // 8]
                parts.append(F.linear(F.linear(x[row], a), b))
            expected.append(torch.cat(parts))
        torch.testing.assert_close(actual, torch.stack(expected), atol=0.03125, rtol=0.02)
        actual.float().square().mean().backward()
        assert all(p.grad is not None and p.grad.count_nonzero() == 0 for p in layer.adapters[2].parameters())
        assert all(p.grad is not None and p.grad.isfinite().all() for p in layer.adapters.parameters())

        model = torch.nn.Module()
        model.config = config
        model.decoder = torch.nn.Module()
        model.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
        model.decoder.layers[0].mlp = torch.nn.Module()
        model.decoder.layers[0].mlp.experts = torch.nn.Module()
        model.decoder.layers[0].mlp.experts.linear_fc1 = layer
        bridge = Qwen3MoEBridge()
        if packed_checkpoint:
            bridge.mapping_registry = lambda: MegatronMappingRegistry(
                FusedGatedExpertMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    hf_param="model.layers.*.mlp.experts.gate_up_proj",
                )
            )
        with expose_adapter_slot(model, 0):
            exported = dict(bridge.stream_adapter_weights_megatron_to_hf([model], cpu=False, show_progress=False))
        for key, projection in (("adapter_gate", "gate_proj"), ("adapter_up", "up_proj")):
            adapter = layer.adapters[0][key]
            for local_expert in range(2):
                expert = dist.get_rank() * 2 + local_expert
                prefix = f"model.layers.0.mlp.experts.{expert}.{projection}"
                a = (
                    exported[f"model.layers.0.mlp.experts.{projection}.lora_A.weight"][0]
                    if shared_outer
                    else exported[prefix + ".lora_A.weight"]
                )
                b = exported[prefix + ".lora_B.weight"]
                torch.testing.assert_close(
                    a,
                    adapter.linear_in.weight if shared_outer else adapter.linear_in.weight[local_expert],
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(b, adapter.linear_out.weight[local_expert], atol=0, rtol=0)


@pytest.mark.usefixtures("two_rank_process_group")
@pytest.mark.parametrize("row_parallel", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("fused_layernorm", [False, True])
def test_fused_tp_gradients(row_parallel, sequence_parallel, fused_layernorm):
    if row_parallel and fused_layernorm:
        pytest.skip("TE fuses the layernorm into column-parallel layers only")
    with _model_parallel(tp=2, ep=1):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            sequence_parallel=sequence_parallel,
            params_dtype=torch.float32,
            gradient_accumulation_fusion=False,
        )
        if fused_layernorm:
            from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

            base = TELayerNormColumnParallelLinear(
                32,
                32,
                config=config,
                init_method=torch.nn.init.zeros_,
                bias=False,
                skip_bias_add=True,
                tp_comm_buffer_name="qkv",
                gather_output=False,
                is_expert=False,
            )
        else:
            linear_cls = RowParallelLinear if row_parallel else ColumnParallelLinear
            base = linear_cls(
                32,
                32,
                config=config,
                init_method=torch.nn.init.zeros_,
                bias=False,
                **({"input_is_parallel": True, "skip_bias_add": True} if row_parallel else {"gather_output": False}),
            )
        base = base.cuda().requires_grad_(False)
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
        # the adapter reads the layernorm output, whose gather it must own under sequence parallelism
        ref_x = F.layer_norm(full_x, (32,), eps=config.layernorm_epsilon) if fused_layernorm else full_x
        expected = F.linear(F.linear(ref_x, a), b)
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
