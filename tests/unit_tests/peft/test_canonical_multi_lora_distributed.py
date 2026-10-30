# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.conversion.param_mapping import merge_qkv_weights
from megatron.bridge.peft.canonical_lora import CanonicalLoRA, LoRALinearSplitFC1UpGate
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.lora_layers import LoRALinear
from megatron.bridge.peft.multi_lora import CanonicalMultiLoRA
from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot, init_adapter_slot, set_tokens_per_adapter_slot
from megatron.bridge.peft.utils import ParallelLinearAdapter, _apply_grouped_expert_swiglu_sharded_factory
from tests.unit_tests.peft.test_multi_lora_seeded_slot_distributed import (
    _model_parallel,
    two_rank_process_group,  # noqa: F401
)
from tests.unit_tests.peft.test_multi_lora_tp_distributed import _assert_close, _gather


@pytest.mark.usefixtures("two_rank_process_group")
@pytest.mark.parametrize(
    "name,targets,widths,output_gate",
    [
        ("linear_qkv", ["linear_q", "linear_k", "linear_v"], [32, 16, 16], False),
        ("linear_qkv", ["linear_q", "linear_k", "linear_v"], [64, 16, 16], True),
        ("linear_qkv", ["linear_q"], [64, 16, 16], True),
        ("linear_fc1", ["linear_fc1_gate", "linear_fc1_up"], [32, 32], False),
    ],
)
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("fused_layernorm", [False, True])
def test_projection_forward_backward_tp(name, targets, widths, output_gate, sequence_parallel, fused_layernorm):
    with _model_parallel(tp=2, ep=1):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=8,
            gated_linear_unit=True,
            attention_output_gate=output_gate,
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
            deltas = []
            for key, width in zip(layer._projection_sizes, widths):
                if key not in layer.adapters[slot]:
                    deltas.append(full_x.new_zeros(width, 32))
                    continue
                adapter = layer.adapters[slot][key]
                a, b = _gather(adapter.linear_in.weight), _gather(adapter.linear_out.weight)
                leaves.append((adapter, a, b))
                deltas.append(b @ a * 2)
            # HF projections placed by the checkpoint conversion, which export inverts
            merged = merge_qkv_weights(config, *deltas) if name == "linear_qkv" else torch.cat(deltas)
            references.append(F.linear(ref_x[slot * 4 : (slot + 1) * 4], merged))
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
@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("name", ["linear_qkv", "linear_fc1"])
@pytest.mark.parametrize("sequence_parallel", [False])
def test_single_adapter_layernorm_input_grad_tp(canonical, name, sequence_parallel):
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

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
        is_qkv = name == "linear_qkv"
        base = TELayerNormColumnParallelLinear(
            32,
            64,
            config=config,
            init_method=torch.nn.init.zeros_,
            bias=False,
            skip_bias_add=True,
            tp_comm_buffer_name="qkv" if is_qkv else "fc1",
            gather_output=False,
            is_expert=False,
        )
        base = base.cuda().requires_grad_(False)
        kwargs = dict(dim=8, alpha=16, lora_B_init_method="normal")
        if canonical:
            targets = ["linear_q", "linear_k", "linear_v"] if is_qkv else ["linear_fc1_gate", "linear_fc1_up"]
            transform = CanonicalLoRA(target_modules=targets, **kwargs)
        else:
            transform = LoRA(target_modules=[name], **kwargs)
        layer = transform.transform(base, name=name, prefix="decoder.layers.0.self_attention")
        if canonical:
            adapters = [layer.adapter[f"adapter_{key}"] for key in (("q", "k", "v") if is_qkv else ("gate", "up"))]
        else:
            adapters = [layer.adapter]
        leaves = [(adapter, _gather(adapter.linear_in.weight), _gather(adapter.linear_out.weight)) for adapter in adapters]
        deltas = [b @ a * 2 for _, a, b in leaves]
        if canonical and is_qkv:
            merged = merge_qkv_weights(config, *deltas)
        elif canonical:
            # rank-local [gate; up] halves, matching the base's fc1 shards
            merged = torch.cat([torch.cat([d.chunk(2)[r] for d in deltas]) for r in range(2)])
        else:
            merged = deltas[0]
        torch.manual_seed(100)
        full_x = torch.randn(8, 1, 32, device="cuda", requires_grad=True)
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        x = (full_x.detach().chunk(2)[tp_rank] if sequence_parallel else full_x.detach()).clone().requires_grad_()
        actual, _ = layer(x)
        expected = F.linear(F.layer_norm(full_x, (32,), eps=config.layernorm_epsilon), merged)
        _assert_close(actual, expected.chunk(2, dim=-1)[tp_rank])
        actual.square().sum().backward()
        expected.square().sum().backward()
        for adapter, a, b in leaves:
            _assert_close(adapter.linear_in.weight.grad, a.grad.chunk(2)[tp_rank])
            _assert_close(adapter.linear_out.weight.grad, b.grad.chunk(2)[tp_rank])
        # the base weight is zero, so the whole input gradient flows through the adapter
        _assert_close(x.grad, full_x.grad.chunk(2)[tp_rank] if sequence_parallel else full_x.grad)


def _expert_fc1_config():
    return TransformerConfig(
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


def _expert_fc1(config):
    from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear

    return (
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


def _expert_factors(adapter, layout, local_expert):
    """This rank's (lora_A, lora_B) for one local expert: `shared` adapters are 2D, `shared_outer` shares only A."""
    a = adapter.linear_in.weight if layout != "per_expert" else adapter.linear_in.weight[local_expert]
    b = adapter.linear_out.weight if layout == "shared" else adapter.linear_out.weight[local_expert]
    return a, b


def _assert_expert_projections_exported(layer, adapters, config, *, layout, packed_checkpoint, slot=None):
    """Each rank's gate/up factors land under their own HF projection names, the shared factor once."""
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.param_mapping import FusedGatedExpertMapping
    from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge

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
    with expose_adapter_slot(model, slot) if slot is not None else nullcontext():
        exported = dict(bridge.stream_adapter_weights_megatron_to_hf([model], cpu=False, show_progress=False))
    for key, projection in (("adapter_gate", "gate_proj"), ("adapter_up", "up_proj")):
        adapter = adapters[key]
        for local_expert in range(2):
            expert = dist.get_rank() * 2 + local_expert
            prefix = f"model.layers.0.mlp.experts.{expert}.{projection}"
            a = (
                exported[f"model.layers.0.mlp.experts.{projection}.lora_A.weight"][0]
                if layout == "shared_outer"
                else exported[prefix + ".lora_A.weight"]
            )
            b = exported[prefix + ".lora_B.weight"]
            expected_a, expected_b = _expert_factors(adapter, layout, local_expert)
            torch.testing.assert_close(a, expected_a, atol=0, rtol=0)
            torch.testing.assert_close(b, expected_b, atol=0, rtol=0)


def _assert_gate_up_checkpoint_not_swiglu_split(adapters):
    """Each half holds one projection; splitting it again as fused [gate; up] permutes rows when ETP changes."""
    with (
        patch(
            "megatron.bridge.peft.utils._apply_grouped_expert_swiglu_sharded_factory",
            wraps=_apply_grouped_expert_swiglu_sharded_factory,
        ) as grouped_swiglu_split,
        patch.object(
            ParallelLinearAdapter,
            "_apply_expert_axis_factory",
            autospec=True,
            side_effect=ParallelLinearAdapter._apply_expert_axis_factory,
        ) as expert_axis_factory,
    ):
        metadata = {"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)}
        for key in ("adapter_gate", "adapter_up"):
            adapters[key].sharded_state_dict(prefix=f"{key}.", metadata=metadata)
    grouped_swiglu_split.assert_not_called()
    assert not any(call.kwargs.get("split_swiglu") for call in expert_axis_factory.call_args_list)


@pytest.mark.usefixtures("two_rank_process_group")
@pytest.mark.parametrize("shared_outer", [False, True])
@pytest.mark.parametrize("packed_checkpoint", [False, True])
def test_expert_projections_export(shared_outer, packed_checkpoint):
    from megatron.bridge.peft.multi_lora_layers import ExpertSlotRouting

    with _model_parallel(tp=1, ep=2):
        config = _expert_fc1_config()
        transform = CanonicalMultiLoRA(
            target_modules=["linear_fc1_gate", "linear_fc1_up"],
            n_adapters=3,
            dim=8,
            experts_shared_outer_loras=shared_outer,
            lora_B_init_method="normal",
        )
        layer = transform.transform(_expert_fc1(config), name="linear_fc1", prefix="decoder.layers.0.mlp.experts")
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

        _assert_expert_projections_exported(
            layer,
            layer.adapters[0],
            config,
            layout="shared_outer" if shared_outer else "per_expert",
            packed_checkpoint=packed_checkpoint,
            slot=0,
        )
        _assert_gate_up_checkpoint_not_swiglu_split(layer.adapters[0])


@pytest.mark.usefixtures("two_rank_process_group")
@pytest.mark.parametrize("layout", ["shared", "per_expert", "shared_outer"])
@pytest.mark.parametrize("packed_checkpoint", [False, True])
def test_single_adapter_expert_projections_export(layout, packed_checkpoint):
    with _model_parallel(tp=1, ep=2):
        config = _expert_fc1_config()
        transform = CanonicalLoRA(
            target_modules=["linear_fc1_gate", "linear_fc1_up"],
            dim=8,
            alpha=8,
            share_expert_adapters=layout == "shared",
            experts_shared_outer_loras=layout == "shared_outer",
            lora_B_init_method="normal",
        )
        layer = transform.transform(_expert_fc1(config), name="linear_fc1", prefix="decoder.layers.0.mlp.experts")
        assert isinstance(layer, LoRALinearSplitFC1UpGate)
        adapters = layer.adapter
        x = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
        actual, _ = layer(x, [8, 8])
        expected = []
        for row in range(16):
            parts = []
            for key in ("adapter_gate", "adapter_up"):
                a, b = _expert_factors(adapters[key], layout, row // 8)
                parts.append(F.linear(F.linear(x[row], a), b))
            expected.append(torch.cat(parts))
        torch.testing.assert_close(actual, torch.stack(expected), atol=0.03125, rtol=0.02)
        actual.float().square().mean().backward()
        assert all(p.grad is not None and p.grad.count_nonzero() > 0 for p in adapters.parameters())

        _assert_expert_projections_exported(
            layer, adapters, config, layout=layout, packed_checkpoint=packed_checkpoint
        )
        _assert_gate_up_checkpoint_not_swiglu_split(adapters)


@pytest.mark.usefixtures("two_rank_process_group")
def test_single_adapter_non_gated_expert_fc1_skips_swiglu_checkpoint_split():
    """A non-gated fc1 holds one projection, so its full-width shared-outer adapter must not be split as SwiGLU."""
    with _model_parallel(tp=1, ep=2):
        config = _expert_fc1_config()
        config.gated_linear_unit = False
        transform = CanonicalLoRA(
            target_modules=["linear_fc1_gate", "linear_fc1_up"], dim=8, alpha=8, experts_shared_outer_loras=True
        )
        layer = transform.transform(_expert_fc1(config), name="linear_fc1", prefix="decoder.layers.0.mlp.experts")
        assert type(layer) is LoRALinear
        with patch(
            "megatron.bridge.peft.utils._apply_grouped_expert_swiglu_sharded_factory",
            wraps=_apply_grouped_expert_swiglu_sharded_factory,
        ) as grouped_swiglu_split:
            layer.adapter.sharded_state_dict(
                prefix="adapter.",
                metadata={"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)},
            )
        grouped_swiglu_split.assert_not_called()


def _assert_gdn_components_exported(layer, adapters, config, slot=None):
    """Every targeted component exports this layer's gathered factors under its own HF component name."""
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.param_mapping import GDNLinearMappingSeparate
    from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge

    model = torch.nn.Module()
    model.config = config
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.decoder.layers[0].self_attention = torch.nn.Module()
    model.decoder.layers[0].self_attention.in_proj = layer
    bridge = Qwen3MoEBridge()
    bridge.mapping_registry = lambda: MegatronMappingRegistry(
        GDNLinearMappingSeparate(
            megatron_param="decoder.layers.*.self_attention.in_proj.weight",
            qkv="model.layers.*.linear_attn.in_proj_qkv.weight",
            z="model.layers.*.linear_attn.in_proj_z.weight",
            b="model.layers.*.linear_attn.in_proj_b.weight",
            a="model.layers.*.linear_attn.in_proj_a.weight",
        )
    )
    with expose_adapter_slot(model, slot) if slot is not None else nullcontext():
        exported = dict(bridge.stream_adapter_weights_megatron_to_hf([model], cpu=False, show_progress=False))
    expected = {}
    for key, adapter in adapters.items():
        if adapter is None:
            continue
        prefix = "model.layers.0.linear_attn." + key.removeprefix("adapter_")
        expected[prefix + ".lora_A.weight"] = _gather(adapter.linear_in.weight).detach()
        expected[prefix + ".lora_B.weight"] = _gather(adapter.linear_out.weight).detach()
    assert exported.keys() == expected.keys()
    for name, tensor in expected.items():
        torch.testing.assert_close(exported[name], tensor, atol=0, rtol=0)


@pytest.mark.usefixtures("two_rank_process_group")
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize(
    "targets",
    [
        ["in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z", "in_proj_b", "in_proj_a"],
        ["in_proj_k", "in_proj_z", "in_proj_a"],
    ],
)
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_gdn_in_proj_components_tp(multi, targets, sequence_parallel):
    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

    from megatron.bridge.models.conversion.param_mapping import (
        _fuse_gdn_separate_to_grouped,
        merge_gdn_linear_weights,
    )
    from megatron.bridge.peft.canonical_lora import LoRALinearSplitGDNInProj, gdn_in_proj_sizes

    with _model_parallel(tp=2, ep=1):
        # q/k: 2 key heads x 4, v/z: 4 value heads x 4, b/a: 4
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            linear_key_head_dim=4,
            linear_num_key_heads=2,
            linear_value_head_dim=4,
            linear_num_value_heads=4,
            tensor_model_parallel_size=2,
            sequence_parallel=sequence_parallel,
            params_dtype=torch.float32,
            gradient_accumulation_fusion=False,
        )
        sizes = gdn_in_proj_sizes(config)
        base = TELayerNormColumnParallelLinear(
            32,
            sum(sizes.values()),
            config=config,
            init_method=torch.nn.init.zeros_,
            bias=False,
            skip_bias_add=True,
            tp_comm_buffer_name="fc1",
            gather_output=False,
            is_expert=False,
        )
        base = base.cuda().requires_grad_(False)
        kwargs = dict(target_modules=targets, dim=8, alpha=16, lora_B_init_method="normal")
        transform = CanonicalMultiLoRA(n_adapters=3, **kwargs) if multi else CanonicalLoRA(**kwargs)
        layer = transform.transform(base, name="in_proj", prefix="decoder.layers.0.self_attention")
        if multi:
            for slot, rank in enumerate((8, 4, 8)):
                init_adapter_slot(layer, slot, rank=rank, alpha=rank * 2, seed=10 + slot)
            set_tokens_per_adapter_slot(layer, torch.tensor([4, 4, 0], device="cuda"))
            slot_adapters = [layer.adapters[slot] for slot in range(2)]
        else:
            assert isinstance(layer, LoRALinearSplitGDNInProj)
            slot_adapters = [layer.adapter, layer.adapter]
        torch.manual_seed(100)
        full_x = torch.randn(8, 1, 32, device="cuda", requires_grad=True)
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        x = (full_x.detach().chunk(2)[tp_rank] if sequence_parallel else full_x.detach()).clone().requires_grad_()
        actual, _ = layer(x)
        ref_x = F.layer_norm(full_x, (32,), eps=config.layernorm_epsilon)
        references = []
        # adapter -> its gathered (lora_A, lora_B); single-adapter LoRA serves both token halves with one adapter
        leaves = {}
        for slot, adapters in enumerate(slot_adapters):
            deltas = {}
            for target, width in sizes.items():
                adapter = adapters[f"adapter_{target}"] if f"adapter_{target}" in adapters else None
                if adapter is None:
                    deltas[target] = full_x.new_zeros(width, 32)
                    continue
                if adapter not in leaves:
                    leaves[adapter] = (_gather(adapter.linear_in.weight), _gather(adapter.linear_out.weight))
                a, b = leaves[adapter]
                deltas[target] = b @ a * 2
            # HF projections packed by the checkpoint conversion, which export inverts
            qkv = torch.cat([deltas["in_proj_q"], deltas["in_proj_k"], deltas["in_proj_v"]])
            qkvz, ba = _fuse_gdn_separate_to_grouped(
                config, qkv, deltas["in_proj_z"], deltas["in_proj_b"], deltas["in_proj_a"]
            )
            merged = merge_gdn_linear_weights(config, qkvz, ba, tp_size=2)
            references.append(F.linear(ref_x[slot * 4 : (slot + 1) * 4], merged))
        expected = torch.cat(references)
        _assert_close(actual, expected.chunk(2, dim=-1)[tp_rank])
        actual.square().sum().backward()
        expected.square().sum().backward()
        for adapter, (a, b) in leaves.items():
            _assert_close(adapter.linear_in.weight.grad, a.grad.chunk(2)[tp_rank])
            _assert_close(adapter.linear_out.weight.grad, b.grad.chunk(2)[tp_rank])
        # Single-adapter LoRA under SP takes TE's gathered LN output, whose grad TE drops, for every
        # wrapper (fused LoRA included); MultiLoRA owns that gather.
        if multi or not sequence_parallel:
            expected_dx = full_x.grad.chunk(2)[tp_rank] if sequence_parallel else full_x.grad
            _assert_close(x.grad, expected_dx)
        if multi:
            assert all(p.grad is not None and p.grad.count_nonzero() == 0 for p in layer.adapters[2].parameters())

        _assert_gdn_components_exported(layer, slot_adapters[0], config, slot=0 if multi else None)
