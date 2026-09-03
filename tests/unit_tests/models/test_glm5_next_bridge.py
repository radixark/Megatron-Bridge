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

"""Unit tests for the GLM-5.3-Flash (glm5_next) bridge helpers that need no GPU or model."""

import pytest
import torch

from megatron.bridge.models.glm5_next.glm5_next_bridge import (
    GLM5_NEXT_ARCHITECTURE,
    HCScaleMapping,
    KDAConv1dMapping,
    _kda_layers,
    dequant_fp8_blockwise,
)


class TestDequantFp8Blockwise:
    def test_scales_apply_per_128_block_and_handle_ragged_edges(self):
        rows, cols, block = 200, 300, 128
        base = torch.randn(rows, cols) * 0.05
        weight = base.to(torch.float8_e4m3fn)
        scale_inv = torch.rand((rows + block - 1) // block, (cols + block - 1) // block) + 0.5
        out = dequant_fp8_blockwise(weight, scale_inv, block=block)
        assert out.dtype == torch.bfloat16 and out.shape == (rows, cols)
        expected = weight.to(torch.float32).clone()
        for bi in range(scale_inv.shape[0]):
            for bj in range(scale_inv.shape[1]):
                expected[bi * block : (bi + 1) * block, bj * block : (bj + 1) * block] *= scale_inv[bi, bj]
        torch.testing.assert_close(out.float(), expected.to(torch.bfloat16).float())


class TestKdaLayers:
    def test_prefers_explicit_kda_layers_then_layer_types_then_default(self):
        class Cfg:
            num_hidden_layers = 8

        cfg = Cfg()
        cfg.linear_attn_config = {"kda_layers": [0, 1, 4]}
        assert _kda_layers(cfg) == (0, 1, 4)
        cfg.linear_attn_config = None
        cfg.layer_types = ["linear_attention", "deepseek_sparse_attention"] * 4
        assert _kda_layers(cfg) == (0, 2, 4, 6)
        cfg.layer_types = None
        assert _kda_layers(cfg) == (0, 1, 2, 4, 5, 6)  # every 4th layer is DSA


class TestCustomMappings:
    def test_kda_conv1d_mapping_round_trip_tp1(self):
        proj, kernel = 16, 4
        q, k, v = (torch.randn(proj, 1, kernel) for _ in range(3))
        mapping = KDAConv1dMapping(
            megatron_param="decoder.layers.0.self_attention.conv1d.weight",
            q="model.language_model.layers.0.self_attn.q_conv1d.weight",
            k="model.language_model.layers.0.self_attn.k_conv1d.weight",
            v="model.language_model.layers.0.self_attn.v_conv1d.weight",
        )
        # tp_size == 1 outside a model-parallel group: the fused weight is [q; k; v]
        assert mapping.tp_size == 1

        class Conv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(3 * proj, 1, kernel))

        module = Conv()
        merged = mapping.hf_to_megatron({"q": q, "k": k, "v": v}, module)
        torch.testing.assert_close(merged, torch.cat([q, k, v], dim=0))
        back = mapping.megatron_to_hf(merged, module)
        torch.testing.assert_close(back[mapping.hf_param["q"]], q)
        torch.testing.assert_close(back[mapping.hf_param["k"]], k)
        torch.testing.assert_close(back[mapping.hf_param["v"]], v)

    def test_kda_conv1d_mapping_resolves_wildcards(self):
        mapping = KDAConv1dMapping(
            megatron_param="decoder.layers.*.self_attention.conv1d.weight",
            q="model.language_model.layers.*.self_attn.q_conv1d.weight",
            k="model.language_model.layers.*.self_attn.k_conv1d.weight",
            v="model.language_model.layers.*.self_attn.v_conv1d.weight",
        )
        resolved = mapping.resolve(("7",))
        assert resolved.megatron_param == "decoder.layers.7.self_attention.conv1d.weight"
        assert resolved.hf_param["v"] == "model.language_model.layers.7.self_attn.v_conv1d.weight"

    def test_hc_scale_mapping_slices_and_reassembles(self):
        scale = torch.tensor([0.25, 0.5, 0.75])

        class HC(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha_pre = torch.nn.Parameter(torch.zeros(1))
                self.alpha_post = torch.nn.Parameter(torch.full((1,), 0.5))
                self.alpha_res = torch.nn.Parameter(torch.full((1,), 0.75))

        module = HC()
        mappings = [
            HCScaleMapping(
                megatron_param=f"decoder.layers.0.self_attention_hyper_connection.{alpha}",
                hf_param="model.language_model.layers.0.hc_attn_scale",
                index=i,
            )
            for i, alpha in enumerate(HCScaleMapping._ORDER)
        ]
        sliced = [m.hf_to_megatron(scale, module) for m in mappings]
        assert [s.item() for s in sliced] == [0.25, 0.5, 0.75]
        # export: only alpha_pre emits the [3] tensor, rebuilt from its siblings on the module
        exported = mappings[0].megatron_to_hf(sliced[0], module)
        torch.testing.assert_close(exported["model.language_model.layers.0.hc_attn_scale"], scale)
        assert mappings[1].megatron_to_hf(sliced[1], module) == {}
        assert mappings[2].megatron_to_hf(sliced[2], module) == {}

    def test_architecture_name(self):
        assert GLM5_NEXT_ARCHITECTURE == "Glm5NextForConditionalGeneration"


if __name__ == "__main__":
    pytest.main([__file__])
