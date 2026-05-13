# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import re
from functools import partial
from typing import Dict, List, Mapping

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.kimi_vl.kimi_k25_vl_provider import KimiK25VLModelProvider
from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel
from megatron.bridge.models.kimi_vl.utils import (
    dequantize_int4,
    quantize_to_int4,
)
from megatron.bridge.peft.utils import is_expert_linear


try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


@MegatronModelBridge.register_bridge(
    source="KimiK25ForConditionalGeneration",
    target=KimiK25VLModel,
    provider=KimiK25VLModelProvider,
)
class KimiK25VLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Kimi K2.5 VL.

    Converts HuggingFace Kimi K2.5 VL models (KimiK25ForConditionalGeneration)
    to Megatron format (KimiK25VLModel) and vice versa.

    The language backbone shares the same architecture as Kimi K2 (MoE with MLA).
    """

    # ------------------------------------------------------------------
    # Shared-outer grouped expert LoRA export
    # ------------------------------------------------------------------
    # K2.5 uses SharedOuterGroupedExpertAdapter (see peft/utils.py):
    #   fc1: linear_in is 2D (shared A, hidden->rank);
    #        linear_out is 3D [N_local, 2*intermediate, rank] (per-expert B).
    #   fc2: linear_in is 3D [N_local, rank, intermediate] (per-expert A);
    #        linear_out is 2D (shared B, rank->hidden).
    # The shared side emits ONCE as [1, ..., ...] under the .weight0 HF name.
    # The per-expert side emits N times, one 2D slice per expert, under
    # .weight0..weight{N-1} HF names. The base MegatronPeftBridge emission
    # can't handle this mixed case (treats both sides through a single
    # per-suffix loop and asserts weight.ndim<3), so we override below.
    # Follows SGLang PR #21466's ``experts_shared_outer_loras=True`` contract.

    def _is_fused_fc1_gate_up(
        self,
        base_hf_weight_names,
        linear_out_tensor: torch.Tensor,
        base_weight_shape=None,
    ) -> bool:
        """Accept 3D per-expert ``linear_out`` in addition to the upstream 2D case.

        SharedOuterGroupedExpertAdapter stores fc1's linear_out as
        ``[N_local, 2*intermediate_per_tp, rank]`` (packed per-expert). The
        upstream check rejects any ``ndim != 2``; fall through to the per-
        expert dim check instead.
        """

        names = list(base_hf_weight_names)
        has_gate_up = (
            bool(names)
            and len(names) % 2 == 0
            and all(("gate_proj" in n or "up_proj" in n) for n in names)
            and any("gate_proj" in n for n in names)
            and any("up_proj" in n for n in names)
        )
        if not has_gate_up:
            return False
        if linear_out_tensor.ndim == 2:
            out_dim = linear_out_tensor.shape[0]
        elif linear_out_tensor.ndim == 3:
            # [N_local, 2*intermediate_per_tp, rank] — per-expert out dim is shape[1].
            out_dim = linear_out_tensor.shape[1]
        else:
            return False
        if out_dim % 2 != 0:
            return False
        if base_weight_shape is not None and out_dim != 2 * base_weight_shape[0]:
            return False
        return True

    def _gather_expert_adapter_weight(self, weight):
        # Relaxed: accepts 2D (shared across experts) and 3D (per-expert
        # packed [N_local, out, in]) tensors. EP all-gather returns one
        # tensor per EP rank; shape per entry matches the input.
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        if ep_size <= 1:
            return None
        gathered = [torch.empty_like(weight) for _ in range(ep_size)]
        torch.distributed.all_gather(
            gathered, weight, group=parallel_state.get_expert_model_parallel_group()
        )
        return gathered

    def _select_expert_adapter_weight(self, weight, gathered, expert_idx, num_experts):
        """Return the 2D slice for ``expert_idx`` across EP ranks.

        - 2D weight (shared): same tensor returned regardless of ``expert_idx``.
        - 3D weight (per-expert): ``gathered[rank][local_expert_idx]`` where
          the expert lives on EP rank ``rank``.
        """
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        if ep_size <= 1:
            return weight[expert_idx] if weight.ndim == 3 else weight
        num_local = num_experts // ep_size
        rank = expert_idx // num_local
        local_e = expert_idx % num_local
        per_rank = gathered[rank]
        return per_rank[local_e] if per_rank.ndim == 3 else per_rank

    def stream_adapter_weights_megatron_to_hf(
        self,
        megatron_model,
        cpu: bool = True,
        show_progress: bool = True,
    ):
        """Stream adapter weights with mixed-side emission for shared-outer grouped expert LoRA."""
        from megatron.bridge.models.conversion.model_bridge import HFWeightTuple

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        num_moe_experts = megatron_model[0].config.num_moe_experts
        adapter_tasks_by_base = self.build_adapter_conversion_tasks(megatron_model)
        adapter_tasks = list(itertools.chain.from_iterable(adapter_tasks_by_base.values()))
        if not adapter_tasks:
            return

        assert hasattr(self, "mapping_registry"), "MegatronModelBridge must define mapping_registry"
        mapping_registry = self.mapping_registry()

        for adapter_task in self._with_progress_tracking(
            adapter_tasks, "Streaming adapter weights", show_progress
        ):
            adapter_weight = self.materialize_adapter_weights([adapter_task])[0]

            linear_in_tensor = adapter_weight.linear_in_weight.weight
            linear_out_tensor = adapter_weight.linear_out_weight.weight
            megatron_linear_in_name = adapter_weight.linear_in_weight.param_name
            megatron_linear_out_name = adapter_weight.linear_out_weight.param_name
            is_expert = is_expert_linear(adapter_task.global_base_prefix)
            is_grouped_expert = (
                is_expert and ".local_experts." not in adapter_task.global_base_prefix
            )

            if not is_grouped_expert:
                # Non-grouped adapter: delegate to the default per-suffix loop.
                # Falls back to parent behavior (single ".weight" suffix).
                yield from self._emit_default_adapter(
                    adapter_task=adapter_task,
                    linear_in_tensor=linear_in_tensor,
                    linear_out_tensor=linear_out_tensor,
                    megatron_linear_in_name=megatron_linear_in_name,
                    megatron_linear_out_name=megatron_linear_out_name,
                    megatron_model=megatron_model,
                    mapping_registry=mapping_registry,
                    cpu=cpu,
                )
                continue

            # Grouped expert: mixed 2D (shared) / 3D (per-expert) emission.
            # Only gather across EP for per-expert sides; the shared-side
            # emission path reads ``side_tensor`` directly and never touches
            # ``side_gathered``, so allocating ep_size empty_like buffers +
            # running an all_gather for shared sides is pure waste.
            linear_in_shared = linear_in_tensor.ndim == 2
            linear_out_shared = linear_out_tensor.ndim == 2
            linear_in_gathered = (
                None if linear_in_shared else self._gather_expert_adapter_weight(linear_in_tensor)
            )
            linear_out_gathered = (
                None if linear_out_shared else self._gather_expert_adapter_weight(linear_out_tensor)
            )

            # Emit linear_in side.
            yield from self._emit_grouped_adapter_side(
                side_tensor=linear_in_tensor,
                side_gathered=linear_in_gathered,
                side_is_shared=linear_in_shared,
                side_suffix=".linear_in.weight",
                megatron_side_name=megatron_linear_in_name,
                adapter_task=adapter_task,
                num_moe_experts=num_moe_experts,
                megatron_model=megatron_model,
                mapping_registry=mapping_registry,
                cpu=cpu,
                is_linear_out=False,
            )
            # Emit linear_out side (may be fused — gate/up split under linear_fc1).
            yield from self._emit_grouped_adapter_side(
                side_tensor=linear_out_tensor,
                side_gathered=linear_out_gathered,
                side_is_shared=linear_out_shared,
                side_suffix=".linear_out.weight",
                megatron_side_name=megatron_linear_out_name,
                adapter_task=adapter_task,
                num_moe_experts=num_moe_experts,
                megatron_model=megatron_model,
                mapping_registry=mapping_registry,
                cpu=cpu,
                is_linear_out=True,
            )

    def _emit_grouped_adapter_side(
        self,
        *,
        side_tensor,
        side_gathered,
        side_is_shared,
        side_suffix,
        megatron_side_name,
        adapter_task,
        num_moe_experts,
        megatron_model,
        mapping_registry,
        cpu,
        is_linear_out,
    ):
        """Emit one adapter side (linear_in or linear_out) for a grouped expert.

        Shared (2D): emit ONCE at base_suffix ".weight0" with ``unsqueeze(0)``
        to produce [1, ..., ...] — SGLang detects expert_dim=1 as shared.

        Per-expert (3D): emit ``num_moe_experts`` times, one 2D slice per
        expert under ``.weight0..weight{N-1}``.
        """
        from megatron.bridge.models.conversion.model_bridge import HFWeightTuple

        if side_is_shared:
            suffixes = [".weight0"]
        else:
            suffixes = [f".weight{i}" for i in range(num_moe_experts)]

        # The shared side is a fully-replicated ReplicatedSharedLinear
        # (see peft/utils.py); every rank holds the full [out, in] tensor.
        # No TP gather is needed — emit ``side_tensor`` directly.
        for base_suffix in suffixes:
            if side_is_shared:
                current = side_tensor
            else:
                expert_idx = int(base_suffix[len(".weight"):])
                current = self._select_expert_adapter_weight(
                    side_tensor, side_gathered, expert_idx, num_moe_experts
                )
            if cpu:
                current = current.cpu()
            if side_is_shared:
                current = current.unsqueeze(0)  # 2D -> [1, ..., ...]

            base_hf_names = self._get_base_hf_param_names_for_adapter(
                mapping_registry,
                adapter_task.global_base_prefix,
                adapter_task.adapter_key,
                base_suffix,
            )
            if side_is_shared:
                # Shared-outer (PR #21466) is a single ``[1, ..., ...]`` tensor,
                # not "expert 0's weight". The mapping registry resolved
                # ``.weight0`` through the per-expert wildcard and produced an
                # HF name containing ``experts.N.``; strip it so SGLang's
                # ``_process_weight`` routes the tensor through the direct 3D
                # branch (``"experts" in name and weights.dim() == 3``) rather
                # than the per-expert dict path (``experts\.(\d+)\.``).
                base_hf_names = [
                    re.sub(r"\bexperts\.\d+\.", "experts.", n) for n in base_hf_names
                ]
            side_hf_names = [
                self._make_lora_param_name(bn, side_suffix) for bn in base_hf_names
            ]

            # Handle fused adapters (e.g., linear_fc1 splitting into
            # gate_proj/up_proj). Only applies to linear_out side.
            if is_linear_out and adapter_task.adapter_key is None:
                per_base = self._get_fused_adapter_linear_out_slices(
                    megatron_model,
                    base_hf_names,
                    current,
                    is_expert=is_expert_linear(adapter_task.global_base_prefix),
                )
                if per_base is not None:
                    for index, base_name in enumerate(base_hf_names):
                        chunk = per_base.get(base_name)
                        assert chunk is not None, "unknown projection name"
                        yield HFWeightTuple(side_hf_names[index], chunk, megatron_side_name)
                    continue

            # Shared side of a fused adapter (Megatron ``LoRA`` class, as
            # opposed to ``CanonicalLoRA``): one ``A`` feeds all fused
            # projections (gate AND up through the single ``linear_fc1``).
            # HF PEFT expects the same A at BOTH gate_proj.lora_A AND
            # up_proj.lora_A so that SGLang's ``normalize_gate_up_proj``
            # stacks ``[A, A]`` along the rank dim. If we only emit the
            # first name, the other half gets zero-filled and the up-side
            # LoRA delta is silently 0. Mirrors the dense-MLP path in
            # ``_emit_default_adapter``.
            if side_is_shared and len(side_hf_names) > 1:
                for hf_name in side_hf_names:
                    yield HFWeightTuple(hf_name, current, megatron_side_name)
                continue

            yield HFWeightTuple(side_hf_names[0], current, megatron_side_name)

    def _emit_default_adapter(
        self,
        *,
        adapter_task,
        linear_in_tensor,
        linear_out_tensor,
        megatron_linear_in_name,
        megatron_linear_out_name,
        megatron_model,
        mapping_registry,
        cpu,
    ):
        """Non-grouped-expert adapter: single-suffix emission (parent default)."""
        from megatron.bridge.models.conversion.model_bridge import HFWeightTuple

        if cpu:
            linear_in_tensor = linear_in_tensor.cpu()
            linear_out_tensor = linear_out_tensor.cpu()

        base_hf_names = self._get_base_hf_param_names_for_adapter(
            mapping_registry,
            adapter_task.global_base_prefix,
            adapter_task.adapter_key,
            ".weight",
        )
        linear_in_hf_names = [
            self._make_lora_param_name(bn, ".linear_in.weight") for bn in base_hf_names
        ]
        linear_out_hf_names = [
            self._make_lora_param_name(bn, ".linear_out.weight") for bn in base_hf_names
        ]

        if adapter_task.adapter_key is None:
            per_base = self._get_fused_adapter_linear_out_slices(
                megatron_model,
                base_hf_names,
                linear_out_tensor,
                is_expert=is_expert_linear(adapter_task.global_base_prefix),
            )
            if per_base is not None:
                for index, base_name in enumerate(base_hf_names):
                    chunk = per_base.get(base_name)
                    assert chunk is not None, "unknown projection name"
                    yield HFWeightTuple(linear_in_hf_names[index], linear_in_tensor, megatron_linear_in_name)
                    yield HFWeightTuple(linear_out_hf_names[index], chunk, megatron_linear_out_name)
                return

        yield HFWeightTuple(linear_in_hf_names[0], linear_in_tensor, megatron_linear_in_name)
        yield HFWeightTuple(linear_out_hf_names[0], linear_out_tensor, megatron_linear_out_name)

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> KimiK25VLModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = hf_config.vision_config

        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        mla_rope_params = provider_kwargs.pop("_mla_rope_params", None)
        valid_fields = KimiK25VLModelProvider.__dataclass_fields__
        provider = KimiK25VLModelProvider(**{k: v for k, v in provider_kwargs.items() if k in valid_fields})

        # --- Language model architecture defaults (MoE + MLA) ---
        provider.transformer_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
        provider.multi_latent_attention = True
        provider.position_embedding_type = "rope"

        # Apply MLA rope params, otherwise rope scaling factor will be wrong.
        if mla_rope_params:
            for key, value in mla_rope_params.items():
                setattr(provider, key, value)

        # MoE settings
        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_shared_expert_overlap = True
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_router_bias_update_rate = 1e-3
        provider.moe_router_topk_scaling_factor = getattr(text_config, "routed_scaling_factor", 2.827)
        provider.moe_shared_expert_intermediate_size = text_config.moe_intermediate_size * text_config.n_shared_experts
        provider.moe_layer_freq = [0] * text_config.first_k_dense_replace + [1] * (
            text_config.num_hidden_layers - text_config.first_k_dense_replace
        )

        # Fusions
        provider.apply_rope_fusion = False
        provider.bias_activation_fusion = True
        provider.bias_dropout_fusion = True
        provider.cross_entropy_fusion_impl = "te"
        provider.cross_entropy_loss_fusion = True
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True
        provider.gradient_accumulation_fusion = True

        # Misc
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.attention_softmax_in_fp32 = False
        provider.make_vocab_size_divisible_by = 1280
        provider.seq_length = 4096
        provider.async_tensor_model_parallel_allreduce = True

        # Precision
        dtype = self.dtype_from_hf(hf_config, default=torch.float32)
        provider.fp16 = dtype == torch.float16
        provider.bf16 = dtype == torch.bfloat16
        provider.params_dtype = dtype

        # VL-specific overrides
        provider.vision_config = vision_config
        provider.hf_model_path = hf_pretrained._model_name_or_path
        provider.generation_config = hf_pretrained.generation_config

        # media_placeholder_token_id is on the top-level KimiK25Config, not on text_config
        media_placeholder_token_id = getattr(hf_config, "media_placeholder_token_id", 163605)
        provider.bos_token_id = getattr(text_config, "bos_token_id", 163584)
        provider.eos_token_id = getattr(text_config, "eos_token_id", 163585)
        provider.image_token_id = media_placeholder_token_id
        provider.media_placeholder_token_id = media_placeholder_token_id
        provider.pad_token_id = getattr(hf_config, "pad_token_id", 163839)
        provider.ignore_index = getattr(hf_config, "ignore_index", -100)

        return provider

    def _load_and_dequantize(self, key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Load a weight, dequantizing INT4 packed tensors when present."""
        base = key[:-7] if key.endswith(".weight") else key
        packed_key = f"{base}.weight_packed"
        if packed_key in hf_state_dict:
            assert f"{base}.weight_scale" in hf_state_dict and f"{base}.weight_shape" in hf_state_dict, (
                f"Missing weight scale or shape for quantized weight {key}"
            )
            weight = dequantize_int4(
                hf_state_dict[packed_key],
                hf_state_dict[f"{base}.weight_scale"],
                hf_state_dict[f"{base}.weight_shape"],
                device=hf_state_dict[packed_key].device,
            )
        else:
            weight = hf_state_dict[key]
        return weight

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Load HF weights, dequantizing INT4 quantized tensors when present."""
        if isinstance(hf_param, str):
            return self._load_and_dequantize(hf_param, hf_state_dict)
        return {k: self._load_and_dequantize(v, hf_state_dict) for k, v in hf_param.items()}

    def _is_quantized_expert_key(self, key: str) -> bool:
        if "mlp.experts." in key and ".weight" in key:
            if "shared_experts" in key:
                return False
            if ".layers.0." in key:
                return False
            return True
        return False

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Re-quantize converted expert weights to INT4 format."""
        result = {}
        for fqn, tensor in converted_weights_dict.items():
            if self._is_quantized_expert_key(fqn):
                base = fqn[:-7] if fqn.endswith(".weight") else fqn
                # Preserve the original scale dtype from the HF checkpoint
                orig_scale_key = f"{base}.weight_scale"
                scale_dtype = (
                    hf_state_dict[orig_scale_key].dtype if orig_scale_key in hf_state_dict else torch.bfloat16
                )
                packed, scale, shape = quantize_to_int4(tensor, scale_dtype=scale_dtype)
                result[f"{base}.weight_packed"] = packed
                result[f"{base}.weight_scale"] = scale
                result[f"{base}.weight_shape"] = shape
            else:
                result[fqn] = tensor
        return result

    def build_conversion_tasks(
        self,
        hf_pretrained,
        megatron_model,
    ) -> List:
        """Override to synthesize virtual weight keys from INT4 quantized triplets.

        The HF checkpoint stores quantized expert weights as triplets
        (weight_packed, weight_scale, weight_shape) without a plain 'weight' key.
        We synthesize virtual 'weight' keys so the mapping registry can find them,
        then maybe_modify_loaded_hf_weight handles dequantization at load time.
        """
        original_get_all_keys = hf_pretrained.state.source.get_all_keys

        def _get_all_keys_with_virtual():
            keys = original_get_all_keys()
            all_keys_set = set(keys)
            virtual_keys = []
            for key in keys:
                if key.endswith("_packed"):
                    base = key[:-7]  # e.g. "...weight_packed" -> "...weight"
                    if f"{base}_scale" in all_keys_set and f"{base}_shape" in all_keys_set:
                        virtual_keys.append(base)
            return keys + virtual_keys

        hf_pretrained.state.source.get_all_keys = _get_all_keys_with_virtual
        try:
            return super().build_conversion_tasks(hf_pretrained, megatron_model)
        finally:
            hf_pretrained.state.source.get_all_keys = original_get_all_keys

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = get_common_mapping_list()
        param_mappings = {
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
        }

        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # In HF Kimi K2.5 VL models, the language component is nested under
        # "language_model.model" instead of just "model", so we need to add the prefix.
        for mapping in mapping_list:
            if isinstance(mapping, AutoMapping):
                mapping.hf_param = "language_model." + mapping.hf_param
                mapping.megatron_param = "language_model." + mapping.megatron_param
            elif isinstance(mapping, GatedMLPMapping):
                mapping.megatron_param = mapping.megatron_param.replace("decoder", "language_model.decoder")
                mapping.hf_param["gate"] = mapping.hf_param["gate"].replace("model", "language_model.model")
                mapping.hf_param["up"] = mapping.hf_param["up"].replace("model", "language_model.model")

        # Vision Tower and MM Projector use ReplicatedMapping because
        # vision components are not sharded across tensor parallel ranks.
        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="vision_tower.**",
                    hf_param="vision_tower.**",
                ),
                ReplicatedMapping(
                    megatron_param="mm_projector.**",
                    hf_param="mm_projector.**",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)
