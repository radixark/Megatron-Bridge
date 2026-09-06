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

"""Unit tests for the DeepSeek-V4 bridge mapping registry.

Locks in the MTP mapping layout: per-MTP-layer HC head, separate ``e_proj``
and ``h_proj`` mappings, and no deprecated concatenated ``eh_proj`` path.
"""

from types import SimpleNamespace

import pytest

from megatron.bridge.models.conversion.param_mapping import AutoMapping, ReplicatedMapping
from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    DeepSeekV4Bridge,
    _dsv4_compress_ratios,
    _dsv4_num_hash_layers,
)


@pytest.fixture
def bridge_with_mtp():
    """A DSv4 bridge with hf_config stubbed for a single MTP layer."""
    bridge = DeepSeekV4Bridge()
    # mapping_registry only reads num_nextn_predict_layers from hf_config.
    bridge.hf_config = SimpleNamespace(num_nextn_predict_layers=1)
    return bridge


@pytest.fixture
def bridge_without_mtp():
    """A DSv4 bridge with hf_config that has zero MTP layers."""
    bridge = DeepSeekV4Bridge()
    bridge.hf_config = SimpleNamespace(num_nextn_predict_layers=0)
    return bridge


def _by_megatron(registry):
    """Index mappings by megatron_param for quick lookup in assertions."""
    return {m.megatron_param: m for m in registry.mappings}


class TestNativeDeepSeekV4ConfigTranslation:
    """Native Transformers DSv4 config fields must map back to MCore fields."""

    def test_compress_ratios_from_native_layer_types(self):
        hf_config = SimpleNamespace(
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "compressed_sparse_attention",
                "heavily_compressed_attention",
            ],
            compress_rates={
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 128,
            },
        )

        assert _dsv4_compress_ratios(hf_config) == [0, 0, 4, 128, 0]

    def test_legacy_compress_ratios_still_work(self):
        hf_config = SimpleNamespace(
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            compress_ratios=[0, 0, 4, 128, 0],
        )

        assert _dsv4_compress_ratios(hf_config) == [0, 0, 4, 128, 0]

    def test_hash_layers_from_native_mlp_layer_types(self):
        hf_config = SimpleNamespace(
            mlp_layer_types=["hash_moe", "hash_moe", "hash_moe", "moe", "moe"],
        )

        assert _dsv4_num_hash_layers(hf_config) == 3

    def test_hash_layers_must_be_prefix(self):
        hf_config = SimpleNamespace(mlp_layer_types=["hash_moe", "moe", "hash_moe"])

        with pytest.raises(ValueError, match="contiguous prefix"):
            _dsv4_num_hash_layers(hf_config)

    def test_export_hash_layers_from_mcore_field(self):
        from unittest.mock import patch

        from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

        provider = SimpleNamespace(
            moe_n_hash_layers=3,
            num_layers=5,
            mtp_num_layers=None,
            activation_func_clamp_value=10.0,
            csa_compress_ratios=None,
            csa_window_size=128,
            num_residual_streams=4,
            mhc_sinkhorn_iterations=20,
            moe_shared_expert_intermediate_size=2048,
        )
        # Stub the generic export to isolate the DSV4 fields without constructing a full Megatron provider.
        with patch.object(
            MegatronModelBridge,
            "megatron_to_hf_config",
            return_value={"num_hidden_layers": 5, "moe_intermediate_size": 2048},
        ):
            hf_config = DeepSeekV4Bridge.megatron_to_hf_config(provider)

        assert hf_config["num_hash_layers"] == 3
        assert hf_config["mlp_layer_types"] == ["hash_moe", "hash_moe", "hash_moe", "moe", "moe"]


class TestDecoderHCHeadMappings:
    """The global decoder HC-head triplet must be replicated mappings."""

    @pytest.mark.parametrize(
        "name",
        ["decoder.hc_head_fn", "decoder.hc_head_base", "decoder.hc_head_scale"],
    )
    def test_decoder_hc_head_replicated(self, bridge_with_mtp, name):
        registry = bridge_with_mtp.mapping_registry()
        mapping = _by_megatron(registry).get(name)
        assert mapping is not None, f"missing decoder HC-head mapping: {name}"
        assert isinstance(mapping, ReplicatedMapping)
        # HF side drops the 'decoder.' prefix.
        assert mapping.hf_param == name.removeprefix("decoder.")


class TestMTPHCHeadMappings:
    """Per-MTP-layer HC head must mirror the decoder pattern."""

    @pytest.mark.parametrize(
        "suffix",
        ["hc_head_fn", "hc_head_base", "hc_head_scale"],
    )
    def test_mtp_hc_head_replicated(self, bridge_with_mtp, suffix):
        registry = bridge_with_mtp.mapping_registry()
        mapping = _by_megatron(registry).get(f"mtp.layers.0.{suffix}")
        assert mapping is not None, f"missing MTP HC-head mapping: mtp.layers.0.{suffix}"
        assert isinstance(mapping, ReplicatedMapping)
        assert mapping.hf_param == f"mtp.0.{suffix}"

    def test_mtp_hc_head_absent_when_no_mtp(self, bridge_without_mtp):
        registry = bridge_without_mtp.mapping_registry()
        names = _by_megatron(registry)
        for suffix in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            assert f"mtp.layers.0.{suffix}" not in names


class TestMTPEHProjSplit:
    """MTP e_proj and h_proj are separate ColumnParallelLinear modules.

    The bridge must use two AutoMappings (which auto-detect column parallelism),
    not the deprecated concatenated eh_proj path.
    """

    @pytest.mark.parametrize("name", ["e_proj", "h_proj"])
    def test_split_proj_automapping(self, bridge_with_mtp, name):
        registry = bridge_with_mtp.mapping_registry()
        mapping = _by_megatron(registry).get(f"mtp.layers.0.{name}.weight")
        assert mapping is not None, f"missing MTP projection: {name}"
        assert isinstance(mapping, AutoMapping)
        assert mapping.hf_param == f"mtp.0.{name}.weight"

    def test_eh_proj_not_in_registry(self, bridge_with_mtp):
        registry = bridge_with_mtp.mapping_registry()
        for mapping in registry.mappings:
            assert "eh_proj" not in mapping.megatron_param, (
                f"deprecated eh_proj reference found in megatron_param: {mapping.megatron_param}"
            )
            hf_param = mapping.hf_param
            if isinstance(hf_param, str):
                assert "eh_proj" not in hf_param, f"deprecated eh_proj reference found in hf_param: {hf_param}"
            elif isinstance(hf_param, dict):
                for v in hf_param.values():
                    assert "eh_proj" not in v, f"deprecated eh_proj reference found in hf_param dict value: {v}"


class TestDeepSeekV4RotaryPercent:
    """Regression: HF partial_rotary_factor (relative to head_dim=512) must not shrink
    the Megatron rope cache — qk_pos_emb_head_dim (64) already encodes the rope split.
    rotary_percent=0.125 yields an 8-dim cos/sin cache: the unfused path silently
    rotates 8/64 dims and the fused MLA rope kernel reads cos/sin out of bounds (SFT NaN)."""

    def test_provider_bridge_forces_full_rotary_percent(self):
        from unittest.mock import MagicMock, patch

        from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
        from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge

        hf_config = SimpleNamespace(
            head_dim=512,
            qk_rope_head_dim=64,
            q_lora_rank=1024,
            o_groups=8,
            o_lora_rank=1024,
            rope_theta=10000,
            compress_rope_theta=160000,
            rope_scaling={"factor": 16, "original_max_position_embeddings": 65536},
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            num_hash_layers=3,
            compress_ratios=[0, 4, 128, 4, 0],
            sliding_window=128,
            index_n_heads=64,
            index_head_dim=128,
            index_topk=512,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            scoring_func="sqrtsoftplus",
            num_experts_per_tok=6,
            norm_topk_prob=True,
            routed_scaling_factor=1.5,
            vocab_size=129280,
            swiglu_limit=10.0,
            moe_intermediate_size=1024,
            n_shared_experts=1,
            tie_word_embeddings=False,
        )
        hf_pretrained = MagicMock()
        hf_pretrained.config = hf_config
        provider = MagicMock()
        # what the generic partial_rotary_factor -> rotary_percent mapping produces
        provider.rotary_percent = 0.125

        bridge = DeepSeekV4Bridge.__new__(DeepSeekV4Bridge)
        with patch.object(MegatronModelBridge, "provider_bridge", return_value=provider):
            out = bridge.provider_bridge(hf_pretrained)

        assert out.rotary_percent == 1.0
        assert out.moe_n_hash_layers == 3
