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

"""Tensor-parallel KDA (Kimi Delta Attention) layer for GLM-5.3-Flash (``glm5_next``).

GLM-5.3 runs 34 of its 45 layers as KDA linear attention:

    q, k, v = W_q x, W_k x, W_v x                (each num_heads * head_dim)
    q, k, v = silu(short_conv(q|k|v))            (depthwise causal conv, kernel 4)
    beta    = sigmoid(W_b x)                     (one scalar per head)
    g       = kda_gate(W_fb W_fa x, A_log, dt_bias, lower_bound)   (forget gate, per channel)
    o       = chunk_kda(l2norm(q), l2norm(k), v, g, beta)          (flash-linear-attention)
    o       = RMSNormGated(o, sigmoid(W_gb W_ga x))
    y       = W_o o

The module is sharded by head across the tensor-parallel group, mirroring the serving engine:
``linear_q/k/v``, ``linear_b``, ``linear_f_b``, ``linear_g_b`` are column-parallel (output
dim = heads), ``linear_f_a`` / ``linear_g_a`` are replicated (``head_dim`` outputs), and
``linear_proj`` is row-parallel; ``conv1d`` (channels), ``A_log`` (heads) and ``dt_bias``
(channels) are TP-local like megatron-core's ``GatedDeltaNet``. Every projection is a
Megatron/TE linear so the PEFT adapters wrap them like any other attention linear.

Numerics follow the GLM-5.3 reference (radixark/miles ``miles_plugins/models/glm5_next/kda.py``),
which is what the rollout engine matches. Context parallelism is not supported.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)


try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.gate import fused_kda_gate

    HAVE_FLA_KDA = True
except ImportError:  # pragma: no cover - depends on the runtime image
    FusedRMSNormGated = ShortConvolution = chunk_kda = fused_kda_gate = None
    HAVE_FLA_KDA = False


def _ensure_metadata_has_dp_cp_group(metadata):
    """Match megatron-core's ``ensure_metadata_has_dp_cp_group`` across versions."""
    try:
        from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group

        return ensure_metadata_has_dp_cp_group(metadata)
    except ImportError:
        return metadata if metadata is not None else {}


@dataclass
class Glm5NextKDASubmodules:
    """Linear projections of :class:`Glm5NextKDAAttention` (all Megatron/TE linears)."""

    linear_q: Union[ModuleSpec, type] = IdentityOp
    linear_k: Union[ModuleSpec, type] = IdentityOp
    linear_v: Union[ModuleSpec, type] = IdentityOp
    linear_b: Union[ModuleSpec, type] = IdentityOp
    linear_f_a: Union[ModuleSpec, type] = IdentityOp
    linear_f_b: Union[ModuleSpec, type] = IdentityOp
    linear_g_a: Union[ModuleSpec, type] = IdentityOp
    linear_g_b: Union[ModuleSpec, type] = IdentityOp
    linear_proj: Union[ModuleSpec, type] = IdentityOp


class Glm5NextKDAAttention(MegatronModule):
    """GLM-5.3 KDA linear attention; drop-in for ``self_attention`` in a transformer layer."""

    def __init__(
        self,
        config,
        submodules: Glm5NextKDASubmodules,
        layer_number: int,
        attn_mask_type=None,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
        is_mtp_layer: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(config=config)
        if not HAVE_FLA_KDA:
            raise ImportError("GLM-5.3 KDA requires flash-linear-attention >= 0.4.2 (fla.ops.kda).")
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.layer_number = layer_number
        self.tp_size = pg_collection.tp.size()
        assert config.context_parallel_size == 1, "GLM-5.3 KDA does not support context parallelism"

        self.hidden_size = config.hidden_size
        self.num_heads = int(config.kda_num_heads)
        self.head_dim = int(config.kda_head_dim)
        self.conv_kernel_size = int(config.kda_conv_kernel_size)
        self.gate_lower_bound = float(config.kda_gate_lower_bound)
        assert self.num_heads % self.tp_size == 0, (self.num_heads, self.tp_size)
        self.local_num_heads = self.num_heads // self.tp_size
        self.projection_size = self.num_heads * self.head_dim
        self.local_projection_size = self.local_num_heads * self.head_dim

        tp_group = pg_collection.tp

        def column(spec, in_features, out_features, buffer_name):
            return build_module(
                spec,
                in_features,
                out_features,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=buffer_name,
                tp_group=tp_group,
            )

        def replicated(spec, in_features, out_features, buffer_name):
            return build_module(
                spec,
                in_features,
                out_features,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=buffer_name,
                parallel_mode="duplicated",
                skip_weight_param_allocation=False,
            )

        self.linear_q = column(submodules.linear_q, self.hidden_size, self.projection_size, "kda_q")
        self.linear_k = column(submodules.linear_k, self.hidden_size, self.projection_size, "kda_k")
        self.linear_v = column(submodules.linear_v, self.hidden_size, self.projection_size, "kda_v")
        self.linear_b = column(submodules.linear_b, self.hidden_size, self.num_heads, "kda_b")
        self.linear_f_a = replicated(submodules.linear_f_a, self.hidden_size, self.head_dim, "kda_f_a")
        self.linear_f_b = column(submodules.linear_f_b, self.head_dim, self.projection_size, "kda_f_b")
        self.linear_g_a = replicated(submodules.linear_g_a, self.hidden_size, self.head_dim, "kda_g_a")
        self.linear_g_b = column(submodules.linear_g_b, self.head_dim, self.projection_size, "kda_g_b")

        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        # Depthwise causal conv over the local q|k|v channels; weight [3 * local_proj, 1, K].
        self.conv1d = ShortConvolution(
            hidden_size=3 * self.local_projection_size,
            kernel_size=self.conv_kernel_size,
            bias=False,
            activation="silu",
            device=device,
            dtype=config.params_dtype,
        )
        setattr(self.conv1d.weight, "tensor_model_parallel", True)
        setattr(self.conv1d.weight, "partition_dim", 0)

        self.A_log = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(self.local_num_heads, dtype=torch.float32, device=device))
        )
        self.dt_bias = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(self.local_projection_size, dtype=torch.float32, device=device))
        )
        for param in (self.A_log, self.dt_bias):
            setattr(param, "tensor_model_parallel", True)
            setattr(param, "partition_dim", 0)

        self.o_norm = FusedRMSNormGated(
            self.head_dim,
            eps=config.layernorm_epsilon,
            activation="sigmoid",
            device=device,
            dtype=config.params_dtype,
        )

        self.linear_proj = build_module(
            submodules.linear_proj,
            self.projection_size,
            self.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
            tp_group=tp_group,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ):
        """``hidden_states`` [s, 1, h] (sequence-parallel slice when SP is on) -> (output, bias)."""
        assert inference_context is None and inference_params is None, "KDA inference is not supported here"
        assert packed_seq_params is not None, "GLM-5.3 KDA needs packed (thd) sequences with cu_seqlens"
        assert hidden_states.ndim == 3 and hidden_states.shape[1] == 1, hidden_states.shape
        cu_seqlens = packed_seq_params.cu_seqlens_q

        # Column-parallel projections gather the sequence-parallel input internally, so
        # q/k/v/beta/gates below are [s_full, 1, local features].
        q, _ = self.linear_q(hidden_states)
        k, _ = self.linear_k(hidden_states)
        v, _ = self.linear_v(hidden_states)
        beta_logits, _ = self.linear_b(hidden_states)
        f_a, _ = self.linear_f_a(hidden_states)
        forget, _ = self.linear_f_b(f_a)
        g_a, _ = self.linear_g_a(hidden_states)
        norm_gate, _ = self.linear_g_b(g_a)

        # fla works batch-first: [1, T, ...]
        mixed_qkv = torch.cat((q, k, v), dim=-1).transpose(0, 1)
        mixed_qkv, _ = self.conv1d(x=mixed_qkv, cu_seqlens=cu_seqlens)
        query, key, value = torch.split(mixed_qkv, [self.local_projection_size] * 3, dim=-1)
        query = query.unflatten(-1, (self.local_num_heads, self.head_dim))
        key = key.unflatten(-1, (self.local_num_heads, self.head_dim))
        value = value.unflatten(-1, (self.local_num_heads, self.head_dim))

        beta = torch.sigmoid(beta_logits.transpose(0, 1).float())
        g = fused_kda_gate(
            forget.transpose(0, 1).unflatten(-1, (self.local_num_heads, self.head_dim)),
            self.A_log,
            self.dt_bias,
            lower_bound=self.gate_lower_bound,
        )

        core_attn_out, _ = chunk_kda(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

        out_shape = core_attn_out.shape  # [1, T, h_local, d]
        core_attn_out = self.o_norm(
            core_attn_out.reshape(-1, self.head_dim),
            norm_gate.transpose(0, 1).reshape(-1, self.head_dim),
        )
        core_attn_out = core_attn_out.reshape(out_shape[0], out_shape[1], -1).transpose(0, 1)  # [T, 1, h_local*d]

        output, bias = self.linear_proj(core_attn_out)
        return output, bias

    # ------------------------------------------------------------------
    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata=None) -> ShardedStateDict:
        """TP-aware sharded state dict: ``A_log`` / ``dt_bias`` / ``conv1d.weight`` shard on dim 0."""
        metadata = _ensure_metadata_has_dp_cp_group(metadata)
        tp_group = self.pg_collection.tp
        dp_cp_group = metadata.get("dp_cp_group") if isinstance(metadata, dict) else None

        state_dict = {}
        self._save_to_state_dict(state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            tensor_parallel_layers_axis_map={"A_log": 0, "dt_bias": 0},
            sharded_offsets=sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=dp_cp_group,
        )
        for name, module in self.named_children():
            if name == "conv1d":
                module_sd = module.state_dict(prefix="", keep_vars=True)
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    {"weight": 0},
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=dp_cp_group,
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )
            sharded_state_dict.update(module_sharded_sd)
        return sharded_state_dict


__all__ = ["Glm5NextKDAAttention", "Glm5NextKDASubmodules", "HAVE_FLA_KDA"]
