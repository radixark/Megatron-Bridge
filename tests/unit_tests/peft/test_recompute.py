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

"""Unit tests for PEFT-specific recompute helpers."""

from types import SimpleNamespace

import torch

from megatron.bridge.peft import recompute as recompute_mod
from megatron.bridge.peft.recompute import maybe_enable_recompute_inputs_grad


class DummyAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class DummyTransformerBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input_requires_grad = None

    def forward(self, hidden_states, *args, **kwargs):
        self.last_input_requires_grad = hidden_states.requires_grad
        return hidden_states


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(recompute_method="uniform")
        self.block = DummyTransformerBlock()

        # Frozen base parameter (not trainable)
        self.base = torch.nn.Linear(1, 1, bias=False)
        self.base.weight.requires_grad = False

        # Trainable adapter parameter whose name contains ".adapter."
        # Use a ModuleDict with key "adapter" so that the full parameter
        # name includes the expected substring (".adapter.") used by
        # maybe_enable_recompute_inputs_grad.
        self.adapter = torch.nn.ModuleDict({"adapter": DummyAdapter()})

    def modules(self):
        for module in super().modules():
            yield module


class DummyMultiLoRAModel(torch.nn.Module):
    """Model whose only trainable params are multi-LoRA slots (".adapters.<i>.")."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(recompute_method="uniform")
        self.block = DummyTransformerBlock()

        # Frozen base parameter (not trainable)
        self.base = torch.nn.Linear(1, 1, bias=False)
        self.base.weight.requires_grad = False

        # Multi-LoRA wrappers hold one adapter per slot in an ``adapters``
        # ModuleList, so parameter names contain ".adapters.<i>." rather than
        # ".adapter.". Nest under a ModuleDict so the full name carries the
        # leading dot (e.g. "linear_fc1.adapters.0.weight").
        self.linear_fc1 = torch.nn.ModuleDict({"adapters": torch.nn.ModuleList([DummyAdapter(), DummyAdapter()])})


def _patch_transformer_block(monkeypatch):
    import megatron.core.transformer.transformer_block as transformer_block

    monkeypatch.setattr(
        transformer_block,
        "TransformerBlock",
        DummyTransformerBlock,
        raising=False,
    )


def test_maybe_enable_recompute_inputs_grad_patches_block(monkeypatch):
    _patch_transformer_block(monkeypatch)
    recompute_mod.PEFT_RECOMPUTE_PATCHED.clear()

    model = DummyModel()
    patched_registry = maybe_enable_recompute_inputs_grad(model, set())

    assert id(model) in patched_registry

    patched_forward = model.block.forward

    input_tensor = torch.zeros(2, 2)
    assert input_tensor.requires_grad is False

    model.block(input_tensor)
    assert model.block.last_input_requires_grad is True

    # Second invocation should be a no-op (no duplicate patch)
    maybe_enable_recompute_inputs_grad(model, patched_registry)
    assert model.block.forward is patched_forward


def test_maybe_enable_recompute_inputs_grad_patches_block_multi_lora(monkeypatch):
    """Multi-LoRA slot params (".adapters.<i>.") must be recognized as adapters.

    Regression test: they used to be classified as trainable base weights, the
    patch was skipped, and under full activation recompute the checkpointed
    region never replayed in backward — every adapter grad stayed zero.
    """
    _patch_transformer_block(monkeypatch)
    recompute_mod.PEFT_RECOMPUTE_PATCHED.clear()

    model = DummyMultiLoRAModel()
    param_names = [n for n, _ in model.named_parameters()]
    assert any(".adapters." in n for n in param_names)
    assert not any(".adapter." in n for n in param_names)

    patched_registry = maybe_enable_recompute_inputs_grad(model, set())

    assert id(model) in patched_registry

    input_tensor = torch.zeros(2, 2)
    assert input_tensor.requires_grad is False

    model.block(input_tensor)
    assert model.block.last_input_requires_grad is True


def test_maybe_enable_recompute_inputs_grad_skips_trainable_base(monkeypatch):
    """A genuinely trainable base weight must still disable the patch."""
    _patch_transformer_block(monkeypatch)
    recompute_mod.PEFT_RECOMPUTE_PATCHED.clear()

    model = DummyMultiLoRAModel()
    model.base.weight.requires_grad = True

    patched_registry = maybe_enable_recompute_inputs_grad(model, set())

    assert id(model) not in patched_registry

    input_tensor = torch.zeros(2, 2)
    model.block(input_tensor)
    assert model.block.last_input_requires_grad is False
