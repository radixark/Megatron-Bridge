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

import torch
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
from megatron.bridge.models.glm5.glm5_provider import GLM5ModelProvider


logger = logging.getLogger(__name__)

# Canonical GlmMoeDsa DSA-indexer split (dsa_indexer_head_dim, qk_pos_emb_head_dim). Single source
# of truth for the config-less PP-broadcast export fallback in _IndexerRopeHalfSwapMapping._dims.
_GLM_DSA_INDEXER_FALLBACK_DIMS = (128, 64)


class _IndexerRopeHalfSwapMapping(AutoMapping):
    """HF<->Megatron mapping for the DSA-indexer ``wq_b``/``wk`` with a rope-half swap.

    megatron-core's ``DSAIndexer`` applies RoPE to the **last** ``qk_pos_emb_head_dim`` of every
    index head (``split([D-rope, rope])``), but the HF/DeepSeek checkpoint stores the rope dims in
    the **first** half. Without compensation the indexer rotates the wrong dimensions (confirmed:
    HF<->bridge index-score correlation ~0.48 vs slime ~0.70). So swap the two halves of each
    index head's ``dsa_indexer_head_dim`` at load (self-inverse on export). Mirrors
    THUDM/slime#2093 ``slime_plugins/mbridge/deepseek_v32.py``:
    ``wq_b = cat([wq_b[:, rope:], wq_b[:, :rope]])`` ; ``wk = cat([wk[rope:], wk[:rope]])``.
    """

    @staticmethod
    def _swap(w: torch.Tensor, head_dim: int, rope_dim: int) -> torch.Tensor:
        # swap [first rope | rest] -> [rest | first rope] along each head's head_dim.
        # wq_b [n_heads*head_dim, in] / wk [head_dim, in] (2-D); k_norm weight/bias [head_dim] (1-D).
        # Runtime guard: every indexer param folds dim-0 into head_dim-sized blocks. If it does not
        # tile exactly, the HF layout is not the per-head [rope | rest] shape this swap assumes --
        # refuse rather than silently mis-convert (e.g. a future GLM-5.x indexer layout change).
        if w.shape[0] % head_dim != 0:
            raise ValueError(
                f"DSA-indexer rope-half swap: param leading dim {w.shape[0]} is not a multiple of "
                f"dsa_indexer_head_dim={head_dim}. The HF indexer weight layout does not match what "
                f"this bridge assumes (per-head blocks of dsa_indexer_head_dim with RoPE in the "
                f"first qk_pos_emb_head_dim={rope_dim} dims). Refusing to silently mis-convert -- "
                f"re-check the HF checkpoint's indexer layout and the index_head_dim / rope config."
            )
        if w.dim() == 1:
            w2 = w.reshape(-1, head_dim)
            w2 = torch.cat([w2[:, rope_dim:], w2[:, :rope_dim]], dim=1)
            return w2.reshape(w.shape)
        w3 = w.reshape(-1, head_dim, w.shape[-1])
        w3 = torch.cat([w3[:, rope_dim:], w3[:, :rope_dim]], dim=1)
        return w3.reshape(w.shape)

    @staticmethod
    def _dims(megatron_module) -> tuple[int, int]:
        # Read the head/rope split from config. On the OWNING rank (and the entire import path)
        # build_conversion_tasks back-fills megatron_module.config (model_bridge.py ~1596), so this
        # resolves the real dims and the range guard below catches a genuinely changed indexer
        # layout. On EXPORT under pipeline_model_parallel_size > 1, params this PP rank does not own
        # get a broadcast-only "fill" task with megatron_module=None (model_bridge.py ~1626);
        # ColumnParallelMapping.megatron_to_hf still returns the PP-broadcast full (un-swapped)
        # tensor on that rank, so megatron_to_hf() must run the SAME swap to stay consistent across
        # ranks. With no config to read there we fall back to the known GlmMoeDsa indexer split
        # instead of crashing (this matches the pre-guard behavior); the owning rank still validates
        # the real dims, so a future layout change is still caught there.
        cfg = getattr(megatron_module, "config", None)
        head_dim = getattr(cfg, "dsa_indexer_head_dim", None)
        rope_dim = getattr(cfg, "qk_pos_emb_head_dim", None)
        if head_dim is None or rope_dim is None:
            # Only reached on the config-less PP-broadcast (non-owning) rank during export under
            # pipeline_model_parallel_size > 1. Fall back to the canonical GlmMoeDsa indexer split;
            # the swap must be IDENTICAL on every rank, and this is correct for every current GLM-5.x
            # model. If a future GLM indexer ever uses a different split this rank would diverge from
            # the owning rank silently, so warn -- the owning rank still validates the real dims via
            # the range guard below. Update _GLM_DSA_INDEXER_FALLBACK_DIMS if that day comes.
            head_dim, rope_dim = _GLM_DSA_INDEXER_FALLBACK_DIMS
            logger.warning(
                "DSA-indexer rope-half swap: no config on this rank (PP-broadcast export); assuming "
                "GlmMoeDsa indexer dims head_dim=%s, qk_pos_emb_head_dim=%s. Correct for all current "
                "GLM-5.x models; a different indexer layout would need _GLM_DSA_INDEXER_FALLBACK_DIMS.",
                head_dim,
                rope_dim,
            )
        if not 0 < rope_dim < head_dim:
            raise ValueError(
                f"DSA-indexer rope-half swap: expected 0 < qk_pos_emb_head_dim ({rope_dim}) < "
                f"dsa_indexer_head_dim ({head_dim}). A degenerate split means the swap would be a "
                "no-op or out-of-range -- the assumed indexer layout does not hold for this model."
            )
        return head_dim, rope_dim

    def hf_to_megatron(self, hf_weights, megatron_module):
        hd, rd = self._dims(megatron_module)
        return super().hf_to_megatron(self._swap(hf_weights, hd, rd), megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        out = super().megatron_to_hf(megatron_weights, megatron_module)
        key = str(self.hf_param)
        if key in out and out[key] is not None:
            hd, rd = self._dims(megatron_module)
            out[key] = self._swap(out[key], hd, rd)
        return out


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
        from megatron.bridge.models.glm5.cross_layer_dsa_dispatch import (
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
            from megatron.bridge.models.glm5.cross_layer_dsa_dispatch import (
                get_glm5_crosslayer_dsa_spec,
            )

            return get_glm5_crosslayer_dsa_spec(config, backend)
        # Prefer megatron-core's native handling (works as-is on newer megatron-core).
        try:
            spec = _orig(config, backend)
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
        # GLM-5.1 (no cross-layer sharing): point the MLA self-attention at TileLangMLASelfAttention so
        # the fused (tilelang) backend is dispatchable. Only meaningful when this is the DSA MLA spec.
        # With the default "megatron" backend its forward delegates to the base class ->
        # byte-identical. Guarded so we never re-wrap a non-DSA / non-MLA spec.
        if (
            getattr(config, "experimental_attention_variant", None) == "dsa"
            and getattr(getattr(spec, "module", None), "__name__", "") == "MLASelfAttention"
        ):
            from megatron.bridge.models.glm5.tilelang.tilelang_mla import TileLangMLASelfAttention

            spec.module = TileLangMLASelfAttention
        return spec

    _eav.get_experimental_attention_variant_module_spec = _patched
    try:
        return _eav.get_transformer_block_with_experimental_attention_variant_spec(config, *args, **kwargs)
    finally:
        _eav.get_experimental_attention_variant_module_spec = _orig


@MegatronModelBridge.register_bridge(
    source=GlmMoeDsaForCausalLM, target=GPTModel, provider=GLM5ModelProvider, model_type="glm_moe_dsa"
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

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GLM5ModelProvider:
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
        # applies rotary over qk_pos_emb_head_dim. Read the true rope dim so the dims match the
        # weights for both GLM-5.1 (64) and GLM-5.2 (64). No-op when already correct.
        #
        # We MUST read the RAW config.json value, not hf_config.qk_rope_head_dim: transformers
        # overwrites that in-memory attribute with the mis-parsed 192 (== provider.qk_pos_emb_head_dim),
        # so trusting the attribute would silently skip the correction. Resolve config.json for BOTH
        # load styles: a local dir on disk, else the HF cache (cached_file) for a repo-id load -- the
        # previous local-only os.path.join(_name_or_path, "config.json") no-op'd for repo ids
        # (_name_or_path == "zai-org/GLM-5.2" is not a real file), leaving the mis-parsed 192.
        import json as _json

        _cfg_json = None
        _cfg_dir = getattr(hf_config, "_name_or_path", "") or ""
        if _cfg_dir:
            import os as _os

            _local = _os.path.join(_cfg_dir, "config.json")
            if _os.path.isfile(_local):
                _cfg_json = _local
            else:
                try:  # resolve from the HF cache for repo-id loads
                    from transformers.utils import cached_file

                    _cfg_json = cached_file(_cfg_dir, "config.json")
                except Exception:
                    _cfg_json = None
        _rope = _json.load(open(_cfg_json)).get("qk_rope_head_dim") if _cfg_json else None
        if _rope and _rope != provider.qk_pos_emb_head_dim:
            logger.info(
                "GLM5 bridge: overriding qk_pos_emb_head_dim %s -> %s (raw qk_rope_head_dim; "
                "transformers mis-parse of the derived head_dim)",
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
            # DSA indexer (wq_b / wk / k_norm are mapped below with the rope-half swap)
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

        # DSA indexer wq_b/wk: rope-half swap (megatron ropes the last half, HF stores rope first).
        # Mirrors THUDM/slime#2093 slime_plugins/mbridge/deepseek_v32.py.
        mapping_list.extend(
            [
                _IndexerRopeHalfSwapMapping(
                    megatron_param="decoder.layers.*.self_attention.core_attention.indexer.linear_wq_b.weight",
                    hf_param="model.layers.*.self_attn.indexer.wq_b.weight",
                ),
                _IndexerRopeHalfSwapMapping(
                    megatron_param="decoder.layers.*.self_attention.core_attention.indexer.linear_wk.weight",
                    hf_param="model.layers.*.self_attn.indexer.wk.weight",
                ),
                # k_norm is applied to the (swapped) key BEFORE rope -> its per-dim scale/bias must be
                # swapped the same way (slime's mbridge swaps k_norm too).
                _IndexerRopeHalfSwapMapping(
                    megatron_param="decoder.layers.*.self_attention.core_attention.indexer.k_norm.weight",
                    hf_param="model.layers.*.self_attn.indexer.k_norm.weight",
                ),
                _IndexerRopeHalfSwapMapping(
                    megatron_param="decoder.layers.*.self_attention.core_attention.indexer.k_norm.bias",
                    hf_param="model.layers.*.self_attn.indexer.k_norm.bias",
                ),
            ]
        )

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
