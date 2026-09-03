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

"""Megatron Bridge for GLM-5.3-Flash (HF ``Glm5NextForConditionalGeneration``, model_type ``glm5_next``).

GLM-5.3-Flash is a 45-layer hybrid: 34 KDA linear-attention layers + 11 DSA sparse-MLA layers
(NoPE, kpool-compressed lightning indexer), a 288-expert sigmoid-routed MoE (first 3 layers dense,
one shared expert) and mHC hyper-connections on every block. Only the language model is bridged;
the vision tower and the MTP head of the HF checkpoint are ignored.

The public checkpoint is block-FP8 (128x128 ``weight_scale_inv`` companions): the FP8 tensors are
dequantized to bf16 while loading (:meth:`Glm5NextBridge.maybe_modify_loaded_hf_weight`).
"""

import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.glm5_next.glm5_next_provider import Glm5NextModelProvider
from megatron.bridge.models.glm5_next.mhc import MHC_EPS
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


logger = logging.getLogger(__name__)

GLM5_NEXT_ARCHITECTURE = "Glm5NextForConditionalGeneration"
_FP8_BLOCK = 128
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


# --------------------------------------------------------------------------------------
# HF config compatibility: transformers < 5.16 does not know ``glm5_next``. The GLM-5.3 text
# config is Glm4vMoe-shaped plus extra fields, so alias it (exactly what radixark/miles does).
# --------------------------------------------------------------------------------------
def register_glm5_next_hf_config_alias() -> None:
    """Register ``glm5_next`` with ``AutoConfig`` when the installed transformers lacks it."""
    try:
        from transformers import AutoConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except ImportError:  # pragma: no cover
        return
    if "glm5_next" in CONFIG_MAPPING_NAMES:
        return
    try:
        from transformers.models.glm4v_moe.configuration_glm4v_moe import Glm4vMoeConfig
    except ImportError:  # pragma: no cover
        return
    compat = type("Glm5NextConfig", (Glm4vMoeConfig,), {"model_type": "glm5_next", "__module__": __name__})
    try:
        AutoConfig.register("glm5_next", compat, exist_ok=True)
    except (TypeError, ValueError):  # pragma: no cover - older transformers without exist_ok
        try:
            AutoConfig.register("glm5_next", compat)
        except ValueError:
            pass


register_glm5_next_hf_config_alias()


def _text_config(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


def _kda_layers(text_config) -> Tuple[int, ...]:
    linear_attn_config = getattr(text_config, "linear_attn_config", None)
    if isinstance(linear_attn_config, dict) and linear_attn_config.get("kda_layers") is not None:
        return tuple(int(i) for i in linear_attn_config["kda_layers"])
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types:
        return tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")
    # GLM-5.3 default: every 4th layer is DSA, the rest KDA.
    return tuple(i for i in range(text_config.num_hidden_layers) if i % 4 != 3)


def _linear_attn_field(text_config, key: str, attr: str, default):
    linear_attn_config = getattr(text_config, "linear_attn_config", None)
    if isinstance(linear_attn_config, dict) and linear_attn_config.get(key) is not None:
        return linear_attn_config[key]
    return getattr(text_config, attr, default)


def dequant_fp8_blockwise(weight: torch.Tensor, scale_inv: torch.Tensor, block: int = _FP8_BLOCK) -> torch.Tensor:
    """Block-wise FP8 -> bf16: ``w[i, j] * scale_inv[i // block, j // block]``."""
    rows, cols = weight.shape
    scale = scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(block, dim=0)[:rows].repeat_interleave(block, dim=1)[:, :cols]
    return (weight.to(torch.float32) * scale).to(torch.bfloat16)


# --------------------------------------------------------------------------------------
# Custom parameter mappings
# --------------------------------------------------------------------------------------
class KDAConv1dMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """HF ``q_conv1d`` / ``k_conv1d`` / ``v_conv1d`` (each ``[proj, 1, K]``) <-> Megatron ``conv1d.weight``.

    The Megatron conv runs over the TP-local ``q|k|v`` channels, so rank ``r`` holds
    ``[q_r; k_r; v_r]`` -- the three HF tensors are chunked per rank and interleaved before the
    column-parallel scatter (the exact inverse on export).
    """

    def __init__(self, megatron_param: str, q: str, k: str, v: str):
        super().__init__(megatron_param, {"q": q, "k": k, "v": v})
        self._tp_mapping = ColumnParallelMapping(megatron_param, megatron_param)

    def hf_to_megatron(self, hf_weights: Dict[str, torch.Tensor], megatron_module) -> torch.Tensor:
        merged = None
        if self.tp_rank == 0:
            parts = [hf_weights[name].chunk(self.tp_size, dim=0) for name in ("q", "k", "v")]
            merged = torch.cat([torch.cat([p[r] for p in parts], dim=0) for r in range(self.tp_size)], dim=0)
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(self, megatron_weights: Optional[torch.Tensor], megatron_module) -> Dict[str, torch.Tensor]:
        packed = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not packed:
            return {}
        full = next(iter(packed.values()))
        per_rank = [chunk.chunk(3, dim=0) for chunk in full.chunk(self.tp_size, dim=0)]
        return {
            self.hf_param[name]: torch.cat([r[i] for r in per_rank], dim=0) for i, name in enumerate(("q", "k", "v"))
        }

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param, resolved_hf_param["q"], resolved_hf_param["k"], resolved_hf_param["v"]
        )


class HCScaleMapping(ReplicatedMapping):
    """HF ``hc_*_scale`` ``[3] = (alpha_pre, alpha_post, alpha_res)`` <-> one Megatron ``alpha_*`` ``[1]``.

    Import slices the HF tensor. Export emits the full ``[3]`` tensor from the ``alpha_pre`` mapping
    (reading its siblings off the owning ``HyperConnectionModule``); the other two export nothing.
    """

    _ORDER = ("alpha_pre", "alpha_post", "alpha_res")

    def __init__(self, megatron_param: str, hf_param: str, index: int):
        super().__init__(megatron_param, hf_param)
        self.index = index

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module) -> torch.Tensor:
        sliced = hf_weights.reshape(-1)[self.index : self.index + 1].clone()
        return super().hf_to_megatron(sliced, megatron_module)

    def megatron_to_hf(self, megatron_weights: Optional[torch.Tensor], megatron_module) -> Dict[str, torch.Tensor]:
        if self.index != 0:
            return {}
        if megatron_weights is not None and megatron_module is not None:
            alphas = [megatron_weights.reshape(1)]
            for name in self._ORDER[1:]:
                alphas.append(getattr(megatron_module, name).detach().reshape(1).to(megatron_weights.device))
            megatron_weights = torch.cat(alphas)
        return super().megatron_to_hf(megatron_weights, megatron_module)

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(resolved_megatron_param, resolved_hf_param, self.index)


# Direct attributes of these modules: A_log / dt_bias are head-sharded, the kpool tensors replicated.
AutoMapping.register_module_type("Glm5NextKDAAttention", "column")
AutoMapping.register_module_type("Glm5NextDSAAttention", "replicated")
AutoMapping.register_module_type("Glm5NextHyperConnectionModule", "replicated")


@MegatronModelBridge.register_bridge(
    source=GLM5_NEXT_ARCHITECTURE, target=GPTModel, provider=Glm5NextModelProvider, model_type="glm5_next"
)
class Glm5NextBridge(MegatronModelBridge):
    """HF ``Glm5NextForConditionalGeneration`` (language model) <-> Megatron ``GPTModel``."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Glm5NextModelProvider:
        hf_config = hf_pretrained.config
        text_config = _text_config(hf_config)

        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        provider_kwargs.pop("_mla_rope_params", None)
        valid_fields = Glm5NextModelProvider.__dataclass_fields__
        provider = Glm5NextModelProvider(**{k: v for k, v in provider_kwargs.items() if k in valid_fields})

        dtype = self.dtype_from_hf(text_config, default=torch.bfloat16)
        provider.params_dtype = dtype
        provider.bf16 = dtype == torch.bfloat16
        provider.fp16 = dtype == torch.float16

        # Trunk
        provider.normalization = "RMSNorm"
        provider.activation_func = F.silu
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.make_vocab_size_divisible_by = 16
        provider.mtp_num_layers = None  # MTP head dropped for training
        provider.hetereogenous_dist_checkpoint = True

        # MLA (NoPE): the positional half of the head is empty; no rotary anywhere.
        provider.multi_latent_attention = True
        provider.q_lora_rank = text_config.q_lora_rank
        provider.kv_lora_rank = text_config.kv_lora_rank
        provider.qk_head_dim = text_config.qk_nope_head_dim
        provider.qk_pos_emb_head_dim = int(getattr(text_config, "qk_rope_head_dim", 0) or 0)
        provider.v_head_dim = text_config.v_head_dim
        provider.kv_channels = text_config.qk_nope_head_dim
        provider.num_query_groups = text_config.num_attention_heads
        provider.position_embedding_type = "rope"
        provider.rope_type = "rope"
        provider.rotary_base = 10000
        provider.rotary_scaling_factor = 1.0
        provider.mscale = 1.0
        provider.mscale_all_dim = 1.0
        provider.apply_rope_fusion = False
        assert provider.qk_pos_emb_head_dim == 0, "GLM-5.3 DSA is NoPE (qk_rope_head_dim must be 0)"

        # MoE: sigmoid top-8 with expert bias, first_k_dense_replace dense layers, one shared expert.
        first_k_dense = int(text_config.first_k_dense_replace)
        provider.moe_layer_freq = [0] * first_k_dense + [1] * (text_config.num_hidden_layers - first_k_dense)
        provider.num_moe_experts = text_config.n_routed_experts
        provider.moe_ffn_hidden_size = text_config.moe_intermediate_size
        provider.moe_shared_expert_intermediate_size = text_config.moe_intermediate_size * text_config.n_shared_experts
        provider.moe_router_topk = text_config.num_experts_per_tok
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_pre_softmax = True
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_router_topk_scaling_factor = text_config.routed_scaling_factor
        provider.moe_router_dtype = "fp32"
        provider.moe_router_load_balancing_type = "none"
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_grouped_gemm = True
        provider.moe_permute_fusion = True
        provider.moe_token_dispatcher_type = "alltoall"

        # mHC hyper-connections
        hc_eps = getattr(text_config, "hc_eps", MHC_EPS)
        assert hc_eps == MHC_EPS, f"GLM-5.3 hc_eps={hc_eps} but megatron-core's mHC eps constants are {MHC_EPS}"
        provider.enable_hyper_connections = bool(getattr(text_config, "mhc", True))
        provider.num_residual_streams = int(getattr(text_config, "hc_mult", 4))
        provider.mhc_sinkhorn_iterations = int(getattr(text_config, "hc_sinkhorn_iters", 20))
        provider.use_fused_mhc = False

        # DSA kpool indexer
        provider.dsa_attention_backend = "tilelang"
        provider.dsa_indexer_n_heads = text_config.index_n_heads
        provider.dsa_indexer_head_dim = text_config.index_head_dim
        provider.dsa_indexer_topk = text_config.index_topk
        provider.dsa_index_kpool = int(getattr(text_config, "index_kpool", 4))
        assert (
            provider.dsa_index_kpool > 1
            and getattr(text_config, "index_kpool_compress", True)
            and getattr(text_config, "index_kpool_always_select_tail", True)
        ), "GLM-5.3 kpool indexer expects index_kpool>1 with compress and always_select_tail"

        # KDA linear attention
        gate_lower_bound = _linear_attn_field(text_config, "gate_lower_bound", "gate_lower_bound", None)
        if gate_lower_bound is None:
            gate_lower_bound = getattr(text_config, "linear_lower_bound", None)
        if gate_lower_bound is None:
            raise ValueError("GLM-5.3 KDA requires gate_lower_bound (safe gate) in the HF config.")
        provider.kda_layers = _kda_layers(text_config)
        provider.kda_num_heads = int(_linear_attn_field(text_config, "num_heads", "linear_num_heads", 64))
        provider.kda_head_dim = int(_linear_attn_field(text_config, "head_dim", "linear_head_dim", 128))
        provider.kda_conv_kernel_size = int(
            _linear_attn_field(text_config, "short_conv_kernel_size", "linear_conv_kernel_dim", 4)
        )
        provider.kda_gate_lower_bound = float(gate_lower_bound)
        return provider

    # ------------------------------------------------------------------
    # FP8 checkpoint support
    # ------------------------------------------------------------------
    def maybe_modify_loaded_hf_weight(self, hf_param: Union[str, Dict[str, str]], hf_state_dict):
        """Dequantize block-FP8 tensors (``<name>_scale_inv`` companions) to bf16 on load."""
        if isinstance(hf_param, str):
            return self._load_dequantized(hf_param, hf_state_dict)
        return {key: self._load_dequantized(name, hf_state_dict) for key, name in hf_param.items()}

    @staticmethod
    def _load_dequantized(name: str, hf_state_dict) -> torch.Tensor:
        weight = hf_state_dict[name]
        if weight.dtype not in _FP8_DTYPES:
            return weight
        scale_name = name + "_scale_inv"
        if weight.ndim == 2 and scale_name in hf_state_dict:
            return dequant_fp8_blockwise(weight, hf_state_dict[scale_name])
        return weight.to(torch.bfloat16)

    # ------------------------------------------------------------------
    def mapping_registry(self) -> MegatronMappingRegistry:
        hf = "model.language_model."
        L = hf + "layers.*."
        M = "decoder.layers.*."

        param_mappings = {
            # Embedding / head
            "embedding.word_embeddings.weight": hf + "embed_tokens.weight",
            "decoder.final_layernorm.weight": hf + "norm.weight",
            "output_layer.weight": "lm_head.weight",
            # Norms (MLA-style separate input layernorm; MoE pre_mlp vs dense fused fc1 layernorm)
            M + "input_layernorm.weight": L + "input_layernorm.weight",
            M + "pre_mlp_layernorm.weight": L + "post_attention_layernorm.weight",
            M + "mlp.linear_fc1.layer_norm_weight": L + "post_attention_layernorm.weight",
            # Attention output (KDA and DSA layers alike)
            M + "self_attention.linear_proj.weight": L + "self_attn.o_proj.weight",
            # DSA: MLA projections
            M + "self_attention.linear_q_down_proj.weight": L + "self_attn.q_a_proj.weight",
            M + "self_attention.linear_q_up_proj.weight": L + "self_attn.q_b_proj.weight",
            M + "self_attention.linear_q_up_proj.layer_norm_weight": L + "self_attn.q_a_layernorm.weight",
            M + "self_attention.linear_kv_down_proj.weight": L + "self_attn.kv_a_proj_with_mqa.weight",
            M + "self_attention.linear_kv_up_proj.weight": L + "self_attn.kv_b_proj.weight",
            M + "self_attention.linear_kv_up_proj.layer_norm_weight": L + "self_attn.kv_a_layernorm.weight",
            # DSA: kpool lightning indexer
            M + "self_attention.wq_b.weight": L + "self_attn.indexer.wq_b.weight",
            M + "self_attention.wk.weight": L + "self_attn.indexer.wk.weight",
            M + "self_attention.k_norm.weight": L + "self_attn.indexer.k_norm.weight",
            M + "self_attention.k_norm.bias": L + "self_attn.indexer.k_norm.bias",
            M + "self_attention.weights_proj.weight": L + "self_attn.indexer.weights_proj.weight",
            M + "self_attention.index_kpool_compress_gate": L + "self_attn.indexer.index_kpool_compress_gate",
            M + "self_attention.index_kpool_compress_ape": L + "self_attn.indexer.index_kpool_compress_ape",
            # KDA: projections, gates, recurrence parameters
            M + "self_attention.linear_q.weight": L + "self_attn.q_proj.weight",
            M + "self_attention.linear_k.weight": L + "self_attn.k_proj.weight",
            M + "self_attention.linear_v.weight": L + "self_attn.v_proj.weight",
            M + "self_attention.linear_b.weight": L + "self_attn.b_proj.weight",
            M + "self_attention.linear_f_a.weight": L + "self_attn.f_a_proj.weight",
            M + "self_attention.linear_f_b.weight": L + "self_attn.f_b_proj.weight",
            M + "self_attention.linear_g_a.weight": L + "self_attn.g_a_proj.weight",
            M + "self_attention.linear_g_b.weight": L + "self_attn.g_b_proj.weight",
            M + "self_attention.A_log": L + "self_attn.A_log",
            M + "self_attention.dt_bias": L + "self_attn.dt_bias",
            M + "self_attention.o_norm.weight": L + "self_attn.o_norm.weight",
            # Dense MLP down / MoE router / shared expert down
            M + "mlp.linear_fc2.weight": L + "mlp.down_proj.weight",
            M + "mlp.router.weight": L + "mlp.gate.weight",
            M + "mlp.router.expert_bias": L + "mlp.gate.e_score_correction_bias",
            M + "mlp.shared_experts.linear_fc2.weight": L + "mlp.shared_experts.down_proj.weight",
        }
        mapping_list = [AutoMapping(megatron_param=k, hf_param=v) for k, v in param_mappings.items()]

        # mHC mapping projections (plain nn.Linear, replicated across TP) and their biases.
        for site, hf_fn, hf_base in (
            ("self_attention_hyper_connection", "hc_attn_fn", "hc_attn_base"),
            ("mlp_hyper_connection", "hc_ffn_fn", "hc_ffn_base"),
        ):
            mapping_list.append(
                ReplicatedMapping(megatron_param=M + f"{site}.mapping_proj.weight", hf_param=L + hf_fn)
            )
            mapping_list.append(ReplicatedMapping(megatron_param=M + f"{site}.bias", hf_param=L + hf_base))

        # mHC gating scales: one HF [3] tensor per site -> three Megatron [1] parameters.
        for site, hf_name in (
            ("self_attention_hyper_connection", "hc_attn_scale"),
            ("mlp_hyper_connection", "hc_ffn_scale"),
        ):
            for index, alpha in enumerate(HCScaleMapping._ORDER):
                mapping_list.append(
                    HCScaleMapping(megatron_param=M + f"{site}.{alpha}", hf_param=L + hf_name, index=index)
                )

        mapping_list.extend(
            [
                # KDA short conv: three HF tensors interleaved per TP rank
                KDAConv1dMapping(
                    megatron_param=M + "self_attention.conv1d.weight",
                    q=L + "self_attn.q_conv1d.weight",
                    k=L + "self_attn.k_conv1d.weight",
                    v=L + "self_attn.v_conv1d.weight",
                ),
                # Dense MLP gate+up -> fc1
                GatedMLPMapping(
                    megatron_param=M + "mlp.linear_fc1.weight",
                    gate=L + "mlp.gate_proj.weight",
                    up=L + "mlp.up_proj.weight",
                ),
                # Shared expert gate+up -> fc1
                GatedMLPMapping(
                    megatron_param=M + "mlp.shared_experts.linear_fc1.weight",
                    gate=L + "mlp.shared_experts.gate_proj.weight",
                    up=L + "mlp.shared_experts.up_proj.weight",
                ),
                # Routed experts (per-expert HF format)
                GatedMLPMapping(
                    megatron_param=M + "mlp.experts.linear_fc1.weight*",
                    gate=L + "mlp.experts.*.gate_proj.weight",
                    up=L + "mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param=M + "mlp.experts.linear_fc2.weight*",
                    hf_param=L + "mlp.experts.*.down_proj.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)


__all__ = ["GLM5_NEXT_ARCHITECTURE", "Glm5NextBridge", "KDAConv1dMapping", "HCScaleMapping", "dequant_fp8_blockwise"]
