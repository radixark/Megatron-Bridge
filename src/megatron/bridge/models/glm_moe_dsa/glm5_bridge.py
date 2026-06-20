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

import logging

from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import GlmMoeDsaForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider


logger = logging.getLogger(__name__)


def _build_glm5_dsa_block_spec(config, *args, **kwargs):
    """``transformer_layer_spec`` for GLM-5 / GLM-5.1 DSA (feature-detected, self-disabling).

    Older megatron-core (e.g. 0.16.0rc0): its experimental-attention dispatcher
    (``get_experimental_attention_variant_module_spec``) only natively wires
    ``"gated_delta_net"`` and raises ``ValueError`` for ``"dsa"``, and its DSA builder
    (``get_dsa_module_spec_for_backend``) omits the ``metainfo`` the variant
    layer-builder reads. Newer megatron-core handles ``"dsa"`` natively (the dispatcher
    gained a ``== "dsa"`` branch and the DSA builder sets ``metainfo`` itself).

    So this wraps the dispatcher to PREFER megatron-core's own handling, and only when it
    raises for ``"dsa"`` (old megatron-core) back-fills via the shipped DSA builder + sets
    ``metainfo["fuse_input_layernorm"]=False`` (MLA-based DSA keeps a separate, non-fused
    input layernorm, like the DeepSeek-V4 ``dsv4`` spec; ``gated_delta_net`` uses ``True``).
    => On newer megatron-core this is a transparent no-op; once the runtime's megatron-core
    handles ``"dsa"``, this whole helper can be deleted. No megatron-core source change.
    """
    # GLM-5.2 cross-layer: fail early at build time if this (virtual) pipeline stage would start
    # on a skip layer -- the per-microbatch top-k holder does not cross PP boundaries. No-op for
    # GLM-5.1 (index_topk_freq=1) and when the layout can't be determined (runtime guard backs it).
    if getattr(config, "experimental_attention_variant", None) == "dsa" and (
        (getattr(config, "dsa_index_topk_freq", 1) or 1) > 1
    ):
        from megatron.bridge.models.glm_moe_dsa.cross_layer_dsa import (
            assert_pp_stage_starts_on_computing_layer,
        )

        assert_pp_stage_starts_on_computing_layer(config, vp_stage=kwargs.get("vp_stage"))

    from megatron.core.models.gpt import experimental_attention_variant_module_specs as _eav

    _orig = _eav.get_experimental_attention_variant_module_spec

    def _patched(config, backend=None):
        # GLM-5.2 DSA cross-layer index sharing: when index_topk_freq>1, build our own
        # CrossLayerDSAttention spec (megatron-core's DSA -- native or shimmed -- is per-layer
        # only and cannot share top-k across layers). GLM-5.1 (no freq) falls through below.
        if getattr(config, "experimental_attention_variant", None) == "dsa" and (
            (getattr(config, "dsa_index_topk_freq", 1) or 1) > 1
        ):
            if backend is None:
                backend = _eav._get_backend_spec_provider(config=config)
            from megatron.bridge.models.glm_moe_dsa.cross_layer_dsa import (
                get_glm5_crosslayer_dsa_spec,
            )

            return get_glm5_crosslayer_dsa_spec(config, backend)
        # Prefer megatron-core's native handling (works as-is on newer megatron-core).
        try:
            return _orig(config, backend)
        except ValueError:
            # Old megatron-core: dispatcher doesn't know "dsa". Don't mask genuine errors
            # for other variants -- only back-fill the dsa case.
            if getattr(config, "experimental_attention_variant", None) != "dsa":
                raise
            if backend is None:
                backend = _eav._get_backend_spec_provider(config=config)
            spec = _eav.get_dsa_module_spec_for_backend(config=config, backend=backend)
            if spec.metainfo is None:
                spec.metainfo = {}
            spec.metainfo.setdefault("fuse_input_layernorm", False)
            return spec

    _eav.get_experimental_attention_variant_module_spec = _patched
    try:
        return _eav.get_transformer_block_with_experimental_attention_variant_spec(config, *args, **kwargs)
    finally:
        _eav.get_experimental_attention_variant_module_spec = _orig


@MegatronModelBridge.register_bridge(
    source=GlmMoeDsaForCausalLM, target=GPTModel, provider=MLAModelProvider, model_type="glm_moe_dsa"
)
class GLM5Bridge(MegatronModelBridge):
    """
    Megatron Bridge for GLM-5 / GLM-5.1 (MoE + MLA + DSA).

    This bridge handles conversion between HuggingFace GlmMoeDsaForCausalLM
    and Megatron-Core GPTModel formats. GLM-5 and GLM-5.1 share the same
    architecture and configuration shape, so both ``zai-org/GLM-5`` and
    ``zai-org/GLM-5.1`` are auto-detected through this bridge.

    The architecture uses Multi-Latent Attention (MLA), Dynamic Sparse Attention
    (DSA) indexer layers, and Mixture-of-Experts (MoE).
    Requires transformers>=5.2.0.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-5.1")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Use experimental-attention spec for DSA. megatron-core's dispatcher raises for
        # "dsa", so route it through _build_glm5_dsa_block_spec (which makes the DSA
        # variant buildable + supplies the metainfo). This makes the GLM-5/5.1 bridge
        # self-contained for both LoRA and full-FT builds (no caller-side monkey-patch).
        try:
            import megatron.core.models.gpt.experimental_attention_variant_module_specs  # noqa: F401

            provider.transformer_layer_spec = _build_glm5_dsa_block_spec
        except (ImportError, ModuleNotFoundError):
            logger.warning("DSA spec not available; falling back to standard GPT decoder block spec.")

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
        provider.multi_latent_attention = True

        # Disable MTP (Multi-Token Prediction) — HF config has num_nextn_predict_layers=1
        # but Bridge does not yet have MTP weight mappings for GLM-5.
        provider.mtp_num_layers = None

        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_shared_expert_overlap = True
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True

        provider.hidden_dropout = 0.0
        provider.attention_softmax_in_fp32 = False

        provider.make_vocab_size_divisible_by = 1280

        # GLM5-specific: computed fields not in CONFIG_MAPPING
        provider.moe_layer_freq = [0] * hf_config.first_k_dense_replace + [1] * (
            hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        )
        provider.moe_shared_expert_intermediate_size = hf_config.moe_intermediate_size * hf_config.n_shared_experts

        # GLM5-specific: rope_theta is nested in rope_parameters (transformers 5.x) or flat
        # (older / GLM-5.2 = 8e6). Handle both shapes robustly.
        _rope_params = getattr(hf_config, "rope_parameters", None)
        provider.rotary_base = (
            (_rope_params.get("rope_theta") if isinstance(_rope_params, dict) else None)
            or getattr(hf_config, "rope_theta", None)
            or 10000
        )
        # GLM5 uses default rope (no YaRN scaling)
        provider.rotary_scaling_factor = 1.0
        provider.mscale = 1.0
        provider.mscale_all_dim = 1.0

        # GLM-5.2 / transformers>=5.12 mis-parses qk_rope_head_dim as head_dim (192) rather than
        # the config.json value (64); the base config-mapping then sizes MLA's decoupled-rope key
        # by 192, giving linear_kv_down_proj = kv_lora_rank + 192 = 704. The checkpoint is ground
        # truth: kv_a_proj_with_mqa = kv_lora_rank + qk_rope_head_dim = 576 = 512 + 64, and MLA
        # applies rotary over qk_pos_emb_head_dim. Read the rope dim straight from config.json so
        # the dims match the weights for both GLM-5.1 (64) and GLM-5.2 (64). No-op when correct.
        import json as _json
        import os as _os

        _cfg_dir = getattr(hf_config, "_name_or_path", "") or ""
        _cfg_json = _os.path.join(_cfg_dir, "config.json")
        if _os.path.isfile(_cfg_json):
            _rope = _json.load(open(_cfg_json)).get("qk_rope_head_dim")
            if _rope and _rope != provider.qk_pos_emb_head_dim:
                logger.info(
                    "GLM5 bridge: overriding qk_pos_emb_head_dim %s -> %s from config.json "
                    "(transformers mis-parse of qk_rope_head_dim)",
                    provider.qk_pos_emb_head_dim,
                    _rope,
                )
                provider.qk_pos_emb_head_dim = _rope

        # DSA indexer params
        provider.experimental_attention_variant = "dsa"
        provider.dsa_indexer_head_dim = hf_config.index_head_dim
        provider.dsa_indexer_n_heads = hf_config.index_n_heads
        provider.dsa_indexer_topk = hf_config.index_topk
        provider.dsa_indexer_loss_coeff = 0.001
        provider.dsa_indexer_use_sparse_loss = True
        # GLM-5.2 DSA cross-layer index sharing. Absent in GLM-5.1 (-> freq=1 -> every layer
        # computes its own top-k = plain DSA). When >1, CrossLayerDSAttention builds the indexer
        # only on computing layers and skip layers reuse the most recent computing layer's top-k.
        provider.dsa_index_topk_freq = getattr(hf_config, "index_topk_freq", 1) or 1
        provider.dsa_index_skip_topk_offset = getattr(hf_config, "index_skip_topk_offset", 0) or 0

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        param_mappings = {
            # Embed
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # LM Head
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
            # Attention layernorm
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            # Attention output
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Post-attention layernorm — MoE layers use pre_mlp_layernorm, dense layers use layer_norm_weight
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            # MLA weights
            "decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            # For non-MLA attention (fallback)
            "decoder.layers.*.self_attention.linear_q_proj.weight": "model.layers.*.self_attn.q_proj.weight",
            # DSA indexer
            "decoder.layers.*.self_attention.core_attention.indexer.linear_wq_b.weight": "model.layers.*.self_attn.indexer.wq_b.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.linear_wk.weight": "model.layers.*.self_attn.indexer.wk.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.k_norm.weight": "model.layers.*.self_attn.indexer.k_norm.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.k_norm.bias": "model.layers.*.self_attn.indexer.k_norm.bias",
            "decoder.layers.*.self_attention.core_attention.indexer.linear_weights_proj.weight": "model.layers.*.self_attn.indexer.weights_proj.weight",
            # Dense MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # MoE router
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            # MoE shared experts
            "decoder.layers.*.mlp.shared_experts.router.weight": "model.layers.*.mlp.shared_experts.gate.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
        }

        mapping_list = [AutoMapping(megatron_param=k, hf_param=v) for k, v in param_mappings.items()]

        # Attention (non-MLA fallback: combined QKV)
        mapping_list.extend(
            [
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                ),
                # Dense MLP gate+up → fc1
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                # Shared expert gate+up → fc1
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
            ]
        )

        # MoE expert weights (per-expert format: experts.N.gate_proj / up_proj / down_proj)
        mapping_list.extend(
            [
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
