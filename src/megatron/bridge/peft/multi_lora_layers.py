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

"""Multi-adapter LoRA layer for Megatron parallel linears.

:class:`MultiLoRALinear` wraps a single Megatron parallel linear module with
*N* concurrent LoRA adapters.  The active adapter is selected at forward time
via per-layer ``tokens_per_adapter`` set by :func:`set_tokens_per_adapter_slot`.

Forward stacks the raw weights of all adapters and uses ``torch._grouped_mm``
for a single fused kernel; TP/SP collectives are issued once around the two
GEMMs to match the layout of the wrapped base linear.
"""

import math
from typing import Any, Dict, Iterator, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.peft.moe_multi_lora_dispatch import dispatch_expert_lora_by_slot
from megatron.bridge.peft.utils import (
    GroupedExpertLinearAdapter,
    ParallelLinearAdapter,
    SharedOuterGroupedExpertAdapter,
    align_expert_dim_for_tp,
    all2all_hp2sp,
    get_adapter_attributes_from_linear,
    get_effective_lora_dim,
)


class MultiLoRALinear(AdapterWrapper):
    """Megatron parallel linear wrapped with *N* concurrent LoRA adapters.

    Each adapter slot is a :class:`ParallelLinearAdapter` stored in an
    ``nn.ModuleList``. Forward uses grouped GEMM with a single set of
    TP/SP comms for efficiency.

    For bridge export compatibility, use :func:`expose_adapter_slot` to
    temporarily expose one slot as ``.adapter``.
    """

    def __init__(
        self,
        to_wrap: nn.Module,
        n_adapters: int,
        dim: int,
        alpha: float,
        full_name: str,
        column_init_method: str = "xavier",
        row_init_method: str = "zero",
        dropout: float = 0.0,
        dropout_position: str = "pre",
        a2a_experimental: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        # The grouped-GEMM forward below never runs each adapter's own
        # ParallelLinearAdapter.forward, so adapter dropout would be silently
        # dropped. Reject dropout>0 loudly instead of pretending to apply it.
        assert dropout == 0.0, (
            f"MultiLoRALinear grouped-GEMM path does not apply adapter dropout "
            f"(got dropout={dropout}); set dropout/--lora-dropout to 0."
        )
        self.to_wrap = to_wrap
        self._adapter_enabled = True
        self.n_adapters = n_adapters
        self.max_rank = dim
        # Kept so a slot re-init (reset_adapter) mirrors the construction-time
        # init methods instead of hardcoding xavier/zero.
        self._column_init_method = column_init_method
        self._row_init_method = row_init_method

        attrs = get_adapter_attributes_from_linear(to_wrap)

        # input_is_parallel distinguishes column-parallel base (False, e.g. linear_qkv,
        # linear_fc1) from row-parallel base (True, e.g. linear_proj, linear_fc2).
        # It controls which TP collective runs between the two grouped GEMMs and
        # whether the second GEMM's output needs to be all-gathered to match the
        # wrapped base linear's output layout.
        self.input_is_parallel = attrs.input_is_parallel
        self.disable_sequence_parallel_comm = attrs.disable_sequence_parallel_comm
        self.use_a2a = a2a_experimental
        self._gather_output = attrs.input_is_parallel

        # ModuleList of ParallelLinearAdapters gives per-adapter optimizer state
        # isolation, clean checkpoint serialization, and bridge export compatibility.
        # Adapter kwargs mirror the single-LoRA path (LoRA.transform).
        self.adapters = nn.ModuleList([
            ParallelLinearAdapter(
                in_features=attrs.in_features,
                out_features=attrs.out_features,
                dim=dim,
                base_linear_name=full_name,
                activation="identity",
                alpha=alpha,
                input_is_parallel=attrs.input_is_parallel,
                column_init_method=column_init_method,
                row_init_method=row_init_method,
                model_parallel_config=getattr(to_wrap, "config", None),
                disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
                disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
                base_linear_is_parallel=attrs.base_linear_is_parallel,
                a2a_experimental=a2a_experimental,
                dropout=dropout,
                dropout_position=dropout_position,
            )
            for _ in range(n_adapters)
        ])

        self.tokens_per_adapter: Optional[torch.Tensor] = None
        device = next(to_wrap.parameters()).device
        dtype = next(to_wrap.parameters()).dtype
        # Non-persistent: slot lifecycle is externally managed, not checkpointed.
        self.register_buffer("alpha_values", torch.ones(n_adapters, dtype=dtype, device=device), persistent=False)
        self.register_buffer(
            "rank_values", torch.full((n_adapters,), dim, dtype=dtype, device=device), persistent=False
        )

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)

        if not self._adapter_enabled:
            return linear_output, bias

        tokens_per_adapter = self.tokens_per_adapter
        x = layernorm_output.contiguous()

        # SP gather (once) — for column-parallel base layers without an LN-fused
        # gather, the layernorm output is still SP-sharded and must be gathered
        # to full sequence before the adapter matmul.
        if not self.disable_sequence_parallel_comm and not self.input_is_parallel:
            x = gather_from_sequence_parallel_region(x)

        x_flat = x.reshape(-1, x.shape[-1])
        offsets = tokens_per_adapter.cumsum(dim=0, dtype=torch.int32)

        stacked_A = torch.stack([a.linear_in.weight for a in self.adapters])
        stacked_B = torch.stack([a.linear_out.weight for a in self.adapters])

        mid = torch._grouped_mm(x_flat, stacked_A.transpose(-2, -1), offsets)

        # TP collective between A and B: row-parallel base needs an all-reduce
        # of the partial sums; column-parallel base needs an all-gather of the
        # rank-sharded output to a full [tokens, dim] for the second GEMM.
        if self.input_is_parallel:
            mid = reduce_from_tensor_model_parallel_region(mid)
        else:
            mid = gather_from_tensor_model_parallel_region(mid)

        out = torch._grouped_mm(mid, stacked_B.transpose(-2, -1), offsets)

        # Per-token scaling is applied *before* the output-side TP/SP comms.
        # ``per_token_scaling`` is indexed by the full token count
        # (``tokens_per_adapter`` sums to it); doing it after a sequence-parallel
        # scatter would leave ``out`` with ``tokens/tp`` rows and crash here.
        # The ratio is computed and applied in the activation dtype: a ratio not
        # exactly representable there (e.g. alpha/rank = 32/24 in bf16) is
        # rounded, while the rollout engine (sglang) multiplies the exact fp32
        # ratio into an fp32 accumulator. Where train/rollout parity at that
        # level matters, keep alpha/rank ratios exactly representable; closing
        # the gap entirely would require applying the scaling in fp32.
        scaling = self.alpha_values / self.rank_values
        per_token_scaling = torch.repeat_interleave(scaling, tokens_per_adapter).unsqueeze(-1)
        out = out * per_token_scaling

        # Match the wrapped base linear's output layout: row-parallel base
        # produces a fully-summed [tokens, h_out] tensor (which we then SP
        # scatter); column-parallel base keeps the [tokens, h_out/tp] shard.
        if self._gather_output:
            out = gather_from_tensor_model_parallel_region(out)

        if not self.disable_sequence_parallel_comm and self.input_is_parallel:
            if self.use_a2a:
                out = all2all_hp2sp(out)
            else:
                out = scatter_to_sequence_parallel_region(out)

        return linear_output + out.reshape(linear_output.shape), bias

    def reset_adapter(self, idx: int) -> None:
        # Re-init through the model-parallel RNG tracker so every DP replica
        # produces identical weights regardless of how far the global RNG has
        # advanced since model build. A bare nn.init here diverges replicas on
        # slot reuse (breaking the DP-equal invariant the weight checker relies
        # on). Mirror the construction-time init methods rather than hardcoding.
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        from megatron.bridge.peft.utils import ParallelLinearAdapter

        adapter = self.adapters[idx]
        col_fn = ParallelLinearAdapter._get_init_fn(None, self._column_init_method)
        row_fn = ParallelLinearAdapter._get_init_fn(None, self._row_init_method)
        with get_cuda_rng_tracker().fork():
            col_fn(adapter.linear_in.weight.data)
            row_fn(adapter.linear_out.weight.data)

    def init_adapter_slot(self, idx: int, rank: int, alpha: float) -> None:
        """Claim slot ``idx`` for an adapter: bind ``rank``/``alpha`` and apply the rank mask."""
        assert 0 < rank <= self.max_rank, (
            f"Adapter rank {rank} must be in (0, {self.max_rank}]"
        )
        self.alpha_values[idx] = alpha
        self.rank_values[idx] = rank
        self._apply_rank_mask(idx)

    def clear_adapter_slot(self, idx: int) -> None:
        """Free slot ``idx``: zero alpha, restore max rank, re-init weights."""
        self.alpha_values[idx] = 0
        self.rank_values[idx] = self.max_rank
        self.reset_adapter(idx)

    def _apply_rank_mask(self, idx: int) -> None:
        """Zero padded rows of A and padded cols of B for slot ``idx``.

        For column-parallel base layers (``linear_qkv``, ``linear_fc1``)
        ``linear_in.weight`` is sharded across TP — rank ``r`` owns global
        rows ``[r*L : (r+1)*L]`` where ``L = max_rank/tp``. For row-parallel
        base it is replicated. Map the global cutoff ``actual_rank`` into
        the local shard before zeroing.

        With both sides zero in the padded region, the autograd chain through
        the two GEMMs keeps the gradient zero there too — no periodic
        re-masking needed during training.
        """
        actual_rank = int(self.rank_values[idx].item())
        if actual_rank >= self.max_rank:
            return
        adapter = self.adapters[idx]
        local_rank_dim = adapter.linear_in.weight.shape[0]
        if local_rank_dim < self.max_rank:
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            shard_start = tp_rank * local_rank_dim
            local_start = max(0, actual_rank - shard_start)
        else:
            local_start = actual_rank
        with torch.no_grad():
            if local_start < local_rank_dim:
                adapter.linear_in.weight.data[local_start:].zero_()
            adapter.linear_out.weight.data[:, actual_rank:].zero_()

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = {}
        self.to_wrap.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        self.adapters.state_dict(destination=destination, prefix=f"{prefix}adapters.", keep_vars=keep_vars)
        return destination

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sharded_sd: Dict[str, Any] = {}
        sharded_sd.update(self.to_wrap.sharded_state_dict(prefix, sharded_offsets, metadata))
        for i, adapter in enumerate(self.adapters):
            sharded_sd.update(adapter.sharded_state_dict(f"{prefix}adapters.{i}.", sharded_offsets, metadata))
        return sharded_sd


class MultiLoRAGroupedExpertLinear(AdapterWrapper):
    """Grouped MoE expert linear wrapped with *N* concurrent LoRA adapters.

    The multi-adapter analogue of :class:`MultiLoRALinear` for grouped expert
    linears (``TE*ParallelGroupedLinear``). It holds *N* single-adapter
    grouped-expert adapters in an ``nn.ModuleList`` — one of
    :class:`~megatron.bridge.peft.utils.GroupedExpertLinearAdapter` (per-expert),
    :class:`~megatron.bridge.peft.utils.SharedOuterGroupedExpertAdapter`
    (shared-outer), or :class:`~megatron.bridge.peft.utils.ParallelLinearAdapter`
    (``share_expert_adapters``) — reusing their tested grouped-GEMM /
    expert-TP / sharded-checkpoint math rather than re-implementing it.

    Routing. On an expert linear the tokens are permuted into expert-major order
    (and shuffled across ranks under EP) before the grouped GEMM, so the
    contiguous per-slot spans that :class:`MultiLoRALinear` relies on no longer
    exist. Instead, the active adapter for each *permuted* token is selected by a
    per-token slot id threaded through the MoE token dispatcher and exposed on the
    wrapped grouped linear as ``._companion_slot_ids`` (see the Megatron-LM
    companion-channel change). The forward delegates the (slot, expert) routing to
    :func:`~megatron.bridge.peft.moe_multi_lora_dispatch.dispatch_expert_lora_by_slot`.

    Inert by default. When ``._companion_slot_ids`` is absent (e.g. the Megatron-LM
    companion channel is not installed, or the base model is not being driven by the
    multi-LoRA trainer) the layer returns the base output unchanged — so wrapping an
    expert linear never changes the base-model result on its own.
    """

    def __init__(
        self,
        to_wrap: nn.Module,
        n_adapters: int,
        dim: int,
        alpha: float,
        full_name: str,
        num_local_experts: int,
        experts_shared_outer_loras: bool = False,
        share_expert_adapters: bool = False,
        normalize_moe_lora: bool = False,
        column_init_method: str = "xavier",
        row_init_method: str = "zero",
        dropout: float = 0.0,
        dropout_position: str = "pre",
    ) -> None:
        nn.Module.__init__(self)
        self.to_wrap = to_wrap
        self._adapter_enabled = True
        self.n_adapters = n_adapters
        self.full_name = full_name
        self.num_local_experts = num_local_experts
        self._column_init_method = column_init_method
        self._row_init_method = row_init_method

        # Expert linears live on the expert-TP group, so attributes/rank sizing must
        # be resolved with is_expert=True (mirrors LoRA.transform for grouped experts).
        attrs = get_adapter_attributes_from_linear(to_wrap, is_expert=True)
        effective_dim = get_effective_lora_dim(
            to_wrap, dim=dim, normalize_moe_lora=normalize_moe_lora, is_expert=True
        )
        effective_dim = align_expert_dim_for_tp(
            to_wrap,
            effective_dim,
            normalize_moe_lora=normalize_moe_lora,
            is_expert=True,
            input_is_parallel=attrs.input_is_parallel,
        )
        self.max_rank = effective_dim
        self.input_is_parallel = attrs.input_is_parallel

        use_shared_outer = experts_shared_outer_loras
        use_per_expert = not share_expert_adapters and not use_shared_outer
        use_grouped_expert_adapter = use_shared_outer or use_per_expert
        if use_shared_outer:
            adapter_cls = SharedOuterGroupedExpertAdapter
        elif use_per_expert:
            adapter_cls = GroupedExpertLinearAdapter
        else:
            adapter_cls = ParallelLinearAdapter

        adapter_kwargs = dict(
            base_linear_name=full_name,
            activation="identity",
            column_init_method=column_init_method,
            row_init_method=row_init_method,
            input_is_parallel=attrs.input_is_parallel,
            dropout=dropout,
            dropout_position=dropout_position,
            model_parallel_config=to_wrap.config,
            alpha=alpha,
            base_linear_is_parallel=attrs.base_linear_is_parallel,
        )
        if use_grouped_expert_adapter:
            first_param = next(to_wrap.parameters())
            adapter_kwargs.update(
                num_local_experts=num_local_experts,
                params_device=first_param.device,
                params_dtype=first_param.dtype,
            )
        else:
            adapter_kwargs.update(is_expert=True)

        # One single-adapter grouped-expert adapter per slot (per-slot optimizer-state
        # isolation, clean checkpoint keys, bridge-export compatibility — same rationale
        # as MultiLoRALinear's ModuleList).
        self.adapters = nn.ModuleList(
            [adapter_cls(attrs.in_features, attrs.out_features, effective_dim, **adapter_kwargs) for _ in range(n_adapters)]
        )

        # set_tokens_per_adapter_slot() sets this on every multi-LoRA module; it is the
        # routing signal for dense layers but unused here (expert routing rides
        # ._companion_slot_ids through the dispatcher instead). Kept for interface parity.
        self.tokens_per_adapter: Optional[torch.Tensor] = None
        device = next(to_wrap.parameters()).device
        dtype = next(to_wrap.parameters()).dtype
        self.register_buffer("alpha_values", torch.ones(n_adapters, dtype=dtype, device=device), persistent=False)
        self.register_buffer(
            "rank_values", torch.full((n_adapters,), effective_dim, dtype=dtype, device=device), persistent=False
        )

    def _extract_expert_splits(self, args: Any, kwargs: Any):
        """Pull the per-expert token counts the base grouped linear was called with."""
        splits = kwargs.get("m_splits")
        if splits is None:
            splits = kwargs.get("tokens_per_expert")
        if splits is None and args:
            splits = args[0]
        if splits is None:
            raise ValueError(
                f"MultiLoRAGroupedExpertLinear on {self.full_name} requires grouped expert token splits."
            )
        return splits

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        if not self._adapter_enabled:
            return linear_output, bias

        # Per-permuted-token adapter slot id, threaded through the MoE dispatcher.
        # Absent -> the layer is inert (base-only), so wrapping never changes the
        # base result on its own.
        permuted_slot_ids = getattr(self.to_wrap, "_companion_slot_ids", None)
        if permuted_slot_ids is None:
            return linear_output, bias

        tokens_per_expert = self._extract_expert_splits(args, kwargs)
        x_in = layernorm_output.contiguous()
        x_flat = x_in.reshape(-1, x_in.shape[-1])
        out_features = linear_output.shape[-1]

        def adapter_fn(slot: int, x_s: torch.Tensor, tpe_s: torch.Tensor) -> torch.Tensor:
            # ParallelLinearAdapter ignores the extra split arg; the grouped adapters
            # consume it via their own _extract_expert_splits.
            return self.adapters[slot](x_s, tpe_s)

        delta = dispatch_expert_lora_by_slot(
            x_flat,
            tokens_per_expert,
            permuted_slot_ids,
            self.num_local_experts,
            out_features,
            adapter_fn,
        )
        return linear_output + delta.reshape(linear_output.shape), bias

    def init_adapter_slot(self, idx: int, rank: int, alpha: float) -> None:
        """Claim slot ``idx``: bind ``alpha`` (and ``rank``) for this adapter.

        v1 constraint: expert slots share one rank (the constructed ``max_rank``);
        variable-rank per-expert masking is deferred. The per-slot scaling is applied
        inside the slot's grouped adapter via its ``alpha/dim`` factor.
        """
        assert rank == self.max_rank, (
            f"MultiLoRAGroupedExpertLinear currently requires all expert slots to share "
            f"rank == max_rank ({self.max_rank}); got rank={rank}. Variable per-expert rank "
            f"is not implemented yet."
        )
        self.alpha_values[idx] = alpha
        self.rank_values[idx] = rank
        self.adapters[idx].alpha = alpha

    def clear_adapter_slot(self, idx: int) -> None:
        """Free slot ``idx``: zero its contribution and re-init its weights."""
        self.alpha_values[idx] = 0
        self.rank_values[idx] = self.max_rank
        self.adapters[idx].alpha = 0.0
        self.reset_adapter(idx)

    def reset_adapter(self, idx: int) -> None:
        """Re-init slot ``idx`` weights through the model-parallel RNG tracker so every
        DP replica produces identical weights on slot reuse (mirrors MultiLoRALinear)."""
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        adapter = self.adapters[idx]
        col_fn = ParallelLinearAdapter._get_init_fn(None, self._column_init_method)
        row_fn = ParallelLinearAdapter._get_init_fn(None, self._row_init_method)
        with get_cuda_rng_tracker().fork():
            col_fn(adapter.linear_in.weight.data)
            row_fn(adapter.linear_out.weight.data)

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = {}
        self.to_wrap.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        self.adapters.state_dict(destination=destination, prefix=f"{prefix}adapters.", keep_vars=keep_vars)
        return destination

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sharded_sd: Dict[str, Any] = {}
        sharded_sd.update(self.to_wrap.sharded_state_dict(prefix, sharded_offsets, metadata))
        for i, adapter in enumerate(self.adapters):
            sharded_sd.update(adapter.sharded_state_dict(f"{prefix}adapters.{i}.", sharded_offsets, metadata))
        return sharded_sd


# ==================================================================
# Standalone functions
# ==================================================================

_MULTI_LORA_TYPES = (MultiLoRALinear, MultiLoRAGroupedExpertLinear)


def _iter_multi_lora_modules(model):
    models = model if isinstance(model, list) else [model]
    for model_chunk in models:
        for module in model_chunk.modules():
            if isinstance(module, _MULTI_LORA_TYPES):
                yield module


def set_tokens_per_adapter_slot(model, tokens_per_adapter: torch.Tensor) -> None:
    """Route a packed micro-batch to its per-slot token spans.

    ``tokens_per_adapter[i]`` is the number of contiguous tokens in the
    upcoming forward that belong to adapter slot ``i``. Must sum to the total
    token count of the micro-batch.
    """
    for module in _iter_multi_lora_modules(model):
        module.tokens_per_adapter = tokens_per_adapter


def init_adapter_slot(model, idx: int, rank: int, alpha: float) -> None:
    """Claim slot ``idx`` across every multi-LoRA layer for an adapter.

    A model-wide adapter is the set of slot-``idx`` chunks across all layers;
    this initialises that set with the given ``rank``/``alpha``. Thin iterator
    over the model — per-slot setup (rank/alpha bookkeeping + rank-mask
    invariant) lives on the layer itself in
    :meth:`MultiLoRALinear.init_adapter_slot` /
    """
    for module in _iter_multi_lora_modules(model):
        module.init_adapter_slot(idx, rank, alpha)


def clear_adapter_slot(model, idx: int) -> None:
    """Release slot ``idx`` across every multi-LoRA layer (zero alpha, re-init weights)."""
    for module in _iter_multi_lora_modules(model):
        module.clear_adapter_slot(idx)


def load_adapter(model, idx: int, state_dict: Dict[str, torch.Tensor]) -> int:
    """Load Megatron-shard format adapter weights into slot ``idx``.

    ``state_dict`` must use the *Megatron-native* names produced by saving
    while ``expose_adapter_slot(model, idx)`` is active — i.e. the same
    layout this function constructs to look them up. Each tensor is the
    local TP/PP shard, copied straight into the slot parameter with no
    gather, scatter, or rank-padding logic.

    Saving from slot A and loading into slot B is fine because the slot
    index is stripped from the name (``...adapter.linear_in.weight``)
    while ``expose_adapter_slot`` is active.

    Returns the number of tensors loaded (for logging / sanity checks).
    Raises ``KeyError`` when the checkpoint and the model's adapter params do
    not match exactly in either direction (missing or unconsumed tensors).
    """
    loaded = 0
    missing = []
    seen = set()
    with expose_adapter_slot(model, idx):
        models = model if isinstance(model, list) else [model]
        for chunk in models:
            for name, param in chunk.named_parameters():
                if ".adapter." not in name:
                    continue
                seen.add(name)
                if name not in state_dict:
                    missing.append(name)
                    continue
                src = state_dict[name].to(device=param.device, dtype=param.dtype)
                param.data.copy_(src)
                loaded += 1
    # A partial load silently leaves the unmatched slots at random init (e.g.
    # resuming after target_modules changed) — a zero-delta / wrong adapter with
    # no error. Fail loud instead.
    if missing:
        raise KeyError(
            f"load_adapter(slot={idx}): {len(missing)} adapter param(s) absent from the "
            f"checkpoint (e.g. {missing[0]}); they would stay at random init. "
            f"Loaded {loaded}. Did target_modules change since the checkpoint was saved?"
        )
    # The reverse mismatch is just as silent: checkpoint tensors no module
    # consumed (e.g. target_modules shrank since the save) would drop part of
    # the trained adapter without an error.
    unused = [key for key in state_dict if ".adapter." in key and key not in seen]
    if unused:
        raise KeyError(
            f"load_adapter(slot={idx}): {len(unused)} checkpoint tensor(s) matched no "
            f"adapter param (e.g. {unused[0]}); part of the saved adapter would be "
            f"silently dropped. Did target_modules change since the checkpoint was saved?"
        )
    return loaded


def expose_adapter_slot(model, idx: int):
    """Context manager that temporarily exposes one adapter slot as ``.adapter``.

    Used by two consumers:

    * The bridge's ``export_adapter_weights`` looks for ``.adapter.linear_in.weight``
      (single-LoRA layout) on each wrapped module.
    * Megatron-native save/load walk ``model.named_parameters()`` and want names
      that don't contain the slot index, so saving from slot ``A`` and loading into
      slot ``B`` produces matching keys.

    Export contract: tensors are exported max-rank padded with ``.dim == max_rank``,
    so the exposed ``.alpha`` is set to ``alpha * max_rank / rank`` — consumers
    computing ``alpha / dim`` apply the slot's runtime scaling. Restored on exit.

    ``MultiLoRALinear`` is handled via the
    common ``.adapters`` ModuleList — duck-typed rather than ``isinstance``-checked
    so future multi-LoRA module types are picked up automatically.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        modules = list(_iter_multi_lora_modules(model))
        saved = {}
        saved_alphas = {}
        for m in modules:
            if "adapters" in m._modules:
                saved[id(m)] = m._modules.pop("adapters")
                adapter = saved[id(m)][idx]
                saved_alphas[id(m)] = adapter.alpha
                adapter.alpha = float(m.alpha_values[idx]) * m.max_rank / float(m.rank_values[idx])
                m.adapter = adapter

        # try/finally: an exception in the body (e.g. an export/save error, which
        # happens on the weight-push path) must still restore the ModuleList,
        # otherwise `adapters` stays detached and every later forward/save/
        # named_parameters is silently corrupted.
        try:
            yield
        finally:
            for m in modules:
                if id(m) in saved:
                    if "adapter" in m._modules:
                        del m._modules["adapter"]
                    m._modules["adapters"] = saved[id(m)]
                    saved[id(m)][idx].alpha = saved_alphas[id(m)]

    return _ctx()


def hide_adapters(model):
    """Context manager that temporarily hides all adapter params from the model.

    Used during base checkpoint loading so the bridge doesn't try to map
    adapter parameters to HF weights.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        modules = list(_iter_multi_lora_modules(model))
        saved = {}
        for m in modules:
            # Duck-typed on the ``.adapters`` ModuleList (like ``expose_adapter_slot``) so
            # every multi-LoRA layer type — MultiLoRALinear and MultiLoRAGroupedExpertLinear —
            # is hidden during base-checkpoint loading, not just the dense one.
            if "adapters" in m._modules:
                saved[id(m)] = m._modules.pop("adapters")
        # try/finally: restore even if base-checkpoint loading raises, else the
        # adapters stay hidden from the model permanently.
        try:
            yield
        finally:
            for m in modules:
                if id(m) in saved:
                    m._modules["adapters"] = saved[id(m)]

    return _ctx()
