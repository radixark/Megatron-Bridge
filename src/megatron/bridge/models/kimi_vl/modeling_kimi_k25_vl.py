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

import types
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from torch import Tensor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
from megatron.bridge.utils.import_utils import safe_import_from
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core import InferenceParams, tensor_parallel, parallel_state


TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")


def _get_image_merge_stats(
    input_ids: torch.Tensor, image_features: list[torch.Tensor], image_token_index: int
) -> tuple[int, int]:
    num_placeholders = (input_ids == image_token_index).sum().item()
    total_image_features = sum(x.shape[0] for x in image_features)
    return num_placeholders, total_image_features


def _get_cp_group():
    """Return Context Parallel (CP) group if available, otherwise None."""
    if not parallel_state.is_initialized():
        return None
    try:
        return parallel_state.get_context_parallel_group()
    except Exception:
        return None


def _cp_all_gather_unzigzag(tensor: torch.Tensor, cp_group, seq_dim: int = 1):
    """All-gather and restore original sequence order from slime's zigzag CP distribution.

    slime's slice_with_cp distributes tokens in a zigzag pattern:
        rank r holds [chunk_r || chunk_{2*cp_size - r - 1}] along seq_dim.

    After all_gather we have cp_size tensors, each containing two chunks.
    This function reassembles them into the original sequential order:
        [chunk_0, chunk_1, ..., chunk_{2*cp_size - 1}]
    """
    if cp_group is None or dist.get_world_size(cp_group) <= 1:
        return tensor
    cp_size = dist.get_world_size(cp_group)
    gathered = [torch.empty_like(tensor) for _ in range(cp_size)]
    dist.all_gather(gathered, tensor, group=cp_group)
    # Each rank r contributes: chunk_r (first half) and chunk_{2N-r-1} (second half)
    chunk_size = tensor.size(seq_dim) // 2
    if tensor.size(seq_dim) % 2 != 0:
        raise RuntimeError(
            f"CP local sequence length must be even for zigzag gather, got {tensor.size(seq_dim)} on dim {seq_dim}"
        )
    ordered = [None] * (2 * cp_size)
    for r, data in enumerate(gathered):
        ordered[r] = data.narrow(seq_dim, 0, chunk_size)
        ordered[2 * cp_size - r - 1] = data.narrow(seq_dim, chunk_size, chunk_size)
    return torch.cat(ordered, dim=seq_dim)


def _pad_along_dim(tensor: torch.Tensor, target_len: int, dim: int, pad_value: int | float) -> torch.Tensor:
    if tensor.size(dim) >= target_len:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_len - tensor.size(dim)
    pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=dim)


def _cp_slice_zigzag(tensor: torch.Tensor, cp_group, seq_dim: int = 1, pad_value: int | float = 0):
    """Re-distribute a full-sequence tensor to this rank's zigzag slice.

    Mirrors slime's slice_with_cp: rank r receives
        [chunk_r || chunk_{2*cp_size - r - 1}] along seq_dim.
    """
    if cp_group is None or dist.get_world_size(cp_group) <= 1:
        return tensor
    cp_size = dist.get_world_size(cp_group)
    rank = dist.get_rank(cp_group)
    seq_len = tensor.size(seq_dim)
    chunk_size = (seq_len + 2 * cp_size - 1) // (2 * cp_size)
    padded_seq_len = 2 * cp_size * chunk_size
    if padded_seq_len != seq_len:
        tensor = _pad_along_dim(tensor, padded_seq_len, seq_dim, pad_value)
    start_1 = rank * chunk_size
    start_2 = (2 * cp_size - rank - 1) * chunk_size
    return torch.cat([
        tensor.narrow(seq_dim, start_1, chunk_size),
        tensor.narrow(seq_dim, start_2, chunk_size),
    ], dim=seq_dim).contiguous()


class KimiK25VLModel(MegatronModule):
    """
    Kimi K2.5 Vision-Language (VL) model wrapper for Megatron.
    Args:
        config (GPTModelProvider): Model provider containing configuration for language and vision modules.
        pre_process (bool, optional): Whether to construct the vision tower and projector. Default: True.
        post_process (bool, optional): Whether to apply post-processing. Default: True.
        vp_stage (Optional[int], optional): Pipeline stage for model parallelism. Default: None.

    Attributes:
        pre_process (bool): If True, enables vision and multimodal components.
        post_process (bool): If True, enables post-processing.
        vp_stage (Optional[int]): Pipeline stage for model parallelism.
        vision_tower (nn.Module): Vision encoder (MoonViT3d vision backbone).
        mm_projector (nn.Module): PatchMergerMLP that projects vision features to language model space.
        language_model (nn.Module): The underlying DeepSeek V3 language model.
        get_image_features (callable): Method to extract and project image features.

    Forward Inputs:
        input_ids (torch.LongTensor, optional): Tokenized input ids for the language model.
        attention_mask (torch.Tensor, optional): Attention mask for the language model.
        position_ids (torch.LongTensor, optional): Position ids for the language model.
        inputs_embeds (torch.FloatTensor, optional): Precomputed input embeddings.
        pixel_values (torch.Tensor, optional): Image tensor(s) for the vision tower.
        labels (torch.Tensor, optional): Target labels for supervised training.
        runtime_gather_output (bool, optional): If True, gather outputs across pipeline stages.
        loss_mask (Tensor, optional): Mask for loss computation.

    Returns:
        Tensor: Model output (e.g., logits or loss, depending on mode).

    Note:
        - If `pre_process` is False, only the language model is constructed.
        - The vision tower and projector are only active if `pre_process` is True.
        - This class is intended for use within the Megatron-LM framework.
    """

    def __init__(
        self,
        config: GPTModelProvider,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        if config.hf_model_path is None:
            raise ValueError("hf_model_path must be set.")

        self.config.image_token_index: int = getattr(config, "image_token_index", 163605)
        self.config.pad_token_id: int = getattr(config, "pad_token_id", 163839)
        self.config.ignore_index: int = getattr(config, "ignore_index", -100)

        KimiK25ForConditionalGeneration = get_class_from_dynamic_module(
            "modeling_kimi_k25.KimiK25ForConditionalGeneration",
            config.hf_model_path,
        )

        if pre_process:
            # Load vision tower and projector classes from the custom HuggingFace model code
            MoonViT3dPretrainedModel = get_class_from_dynamic_module(
                "modeling_kimi_k25.MoonViT3dPretrainedModel",
                config.hf_model_path,
            )
            PatchMergerMLP = get_class_from_dynamic_module(
                "modeling_kimi_k25.PatchMergerMLP",
                config.hf_model_path,
            )
            ProjectorConfig = get_class_from_dynamic_module(
                "modeling_kimi_k25.ProjectorConfig",
                config.hf_model_path,
            )
            VisionTowerConfig = get_class_from_dynamic_module(
                "modeling_kimi_k25.VisionTowerConfig",
                config.hf_model_path,
            )
            self.vision_tower_config = VisionTowerConfig(config.vision_config)
            self.projector_config = ProjectorConfig(config.vision_config)
            self.vision_tower = MoonViT3dPretrainedModel(self.vision_tower_config)
            self.mm_projector = PatchMergerMLP(self.projector_config)
            # Ensure HF visual tower params are marked for TP grad sync and future assignments are hooked.
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_tower)
            hook_hf_module_setattr_for_tp_grad_sync(self.mm_projector)

        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Finalize grad requires these to be bound with module
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        self._extract_image_features = types.MethodType(KimiK25ForConditionalGeneration._extract_image_features, self)

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        self.language_model.set_input_tensor(input_tensor)

    def _merge_input_ids_with_image_features(
        self,
        image_features: list[torch.Tensor],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        target_seq_length: Optional[int] = None,
    ):
        """Merge image features into input embeddings.

        Supports two modes:
        1. Pre-expanded (PP mode): input_ids already has N placeholder tokens per image,
           where N = number of image features. Does simple 1:1 replacement.
        2. Dynamic expansion: input_ids has 1 placeholder per image, expands to N tokens.

        Args:
            image_features: List of image feature tensors, one per image
            inputs_embeds: Text embeddings (batch_size, seq_len, embed_dim)
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Optional labels for training
            loss_mask: Optional per-token loss mask aligned with labels
            target_seq_length: Optional fixed output length for pipeline parallelism.
        """
        _, embed_dim = image_features[0].shape
        feature_lengths = [x.shape[0] for x in image_features]
        image_features_cat = torch.cat(image_features, dim=0)

        image_token_index: int = self.config.media_placeholder_token_id
        pad_token_id: int = self.config.pad_token_id
        ignore_index: int = self.config.ignore_index

        batch_size, sequence_length = input_ids.shape

        num_placeholders, total_image_features = _get_image_merge_stats(input_ids, image_features, image_token_index)

        # Check if tokens are pre-expanded (PP mode with collate-time expansion)
        if num_placeholders == total_image_features:
            # Pre-expanded mode: simple 1:1 replacement, no sequence length change
            final_embedding = inputs_embeds.clone()
            image_mask = input_ids == image_token_index

            # Replace placeholder embeddings with image features
            final_embedding[image_mask] = image_features_cat.to(inputs_embeds.dtype)

            # Attention mask and labels stay the same (no expansion)
            final_attention_mask = attention_mask
            position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_((attention_mask == 0), 1)

            if labels is not None:
                # Mask out image positions in labels (don't compute loss on image tokens)
                final_labels = labels.clone()
                final_labels[image_mask] = ignore_index
            else:
                final_labels = None

            if loss_mask is not None:
                final_loss_mask = loss_mask.clone()
                final_loss_mask[image_mask] = 0
            else:
                final_loss_mask = None

            return final_embedding, final_attention_mask, final_labels, final_loss_mask, position_ids

        # Dynamic expansion mode (original behavior)
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id))

        # Create token occupation table
        _token_occupation_table = torch.ones_like(input_ids.flatten())
        _token_occupation_table[input_ids.flatten() == image_token_index] = torch.tensor(
            feature_lengths, dtype=torch.long, device=input_ids.device
        )
        _token_occupation_table = _token_occupation_table.reshape(input_ids.shape)

        # Calculate natural expanded length, but use target if provided (for PP)
        natural_max_embed_dim = _token_occupation_table.sum(-1).max().item()
        max_embed_dim = target_seq_length if target_seq_length is not None else natural_max_embed_dim

        batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

        # Compute new positions for text tokens
        new_token_positions = torch.cumsum(_token_occupation_table, -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # Create final embeddings (with target_seq_length for PP consistency)
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        if loss_mask is not None:
            final_loss_mask = torch.zeros(
                batch_size, max_embed_dim, dtype=loss_mask.dtype, device=inputs_embeds.device
            )

        target_device = inputs_embeds.device
        batch_indices = batch_indices.to(target_device)
        non_image_indices = non_image_indices.to(target_device)
        text_to_overwrite = text_to_overwrite.to(target_device)
        attention_mask = attention_mask.to(target_device)
        if loss_mask is not None:
            loss_mask = loss_mask.to(target_device)

        # Fill text embeddings
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]
        if loss_mask is not None:
            final_loss_mask[batch_indices, text_to_overwrite] = loss_mask[batch_indices, non_image_indices]

        # Fill image embeddings
        image_to_overwrite = torch.full((batch_size, max_embed_dim), True, dtype=torch.bool, device=target_device)
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        final_embedding[image_to_overwrite] = image_features_cat.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # Mask out padding positions
        batch_indices_pad, pad_indices = torch.where(input_ids == pad_token_id)
        indices_to_mask = new_token_positions[batch_indices_pad, pad_indices]
        final_embedding[batch_indices_pad, indices_to_mask] = 0

        if labels is None:
            final_labels = None
        if loss_mask is None:
            final_loss_mask = None

        return final_embedding, final_attention_mask, final_labels, final_loss_mask, position_ids
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        grid_thws: torch.Tensor = None,
        image_input_mask: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        loss_mask: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        target_seq_length: Optional[int] = None,
    ) -> Tensor:
        r"""
        Args:
            input_ids: Tokenized input ids for the language model.
            attention_mask: Attention mask for the language model.
            position_ids: Position ids for the language model.
            inputs_embeds: Precomputed input embeddings.
            pixel_values: Image tensor for the vision tower.
            grid_thws: Tensor of shape (num_images, 3) containing [temporal, height, width]
                for each image's grid dimensions in the LLM.
            labels: Target labels for supervised training.
            runtime_gather_output: If True, gather outputs across pipeline stages.
            loss_mask: Mask for loss computation.
            target_seq_length: Optional fixed output length for pipeline parallelism.
        """
        cp_group = _get_cp_group()

        if self.pre_process:
            if position_ids is None:
                seq_len = input_ids.size(1)
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=input_ids.device)

            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                ).clone()

            inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()

            if pixel_values is not None:
                image_features = self._extract_image_features(pixel_values.to(self.vision_tower.dtype), grid_thws)
                image_features = self.mm_projector(image_features)
                inputs_embeds = inputs_embeds.to(image_features[0].dtype)

                if cp_group is not None and dist.get_world_size(cp_group) > 1:
                    # After line 365 transpose, inputs_embeds is [1, T_cp, H] (batch-first, seq on dim=1).
                    # Gather along seq_dim=1 and unzigzag to restore original token order → [1, full_T, H].
                    full_inputs_embeds = _cp_all_gather_unzigzag(inputs_embeds, cp_group, seq_dim=1)

                    # input_ids / labels / loss_mask are [1, T_cp] (batch-first, seq on dim=1).
                    full_input_ids = _cp_all_gather_unzigzag(input_ids, cp_group, seq_dim=1)
                    full_labels = _cp_all_gather_unzigzag(labels, cp_group, seq_dim=1) if labels is not None else None
                    full_loss_mask = _cp_all_gather_unzigzag(loss_mask, cp_group, seq_dim=1) if loss_mask is not None else None

                    num_placeholders, total_image_features = _get_image_merge_stats(
                        full_input_ids, image_features, self.config.media_placeholder_token_id
                    )
                    if packed_seq_params is not None and num_placeholders != total_image_features:
                        raise RuntimeError(
                            "KimiK25VL packed/CP path requires pre-expanded image placeholders, "
                            f"but got dynamic expansion (num_placeholders={num_placeholders}, "
                            f"total_image_features={total_image_features}). "
                            "This makes packed_seq_params and slime rollout total_lengths/response_lengths stale."
                        )

                    merge_attention_mask = torch.ones_like(full_input_ids, dtype=torch.long)

                    # merged_* are all batch-first: [1, merged_T, H] / [1, merged_T].
                    merged_embeds, _merged_mask, merged_labels, merged_loss_mask, merged_posids = (
                        self._merge_input_ids_with_image_features(
                            image_features,
                            full_inputs_embeds,
                            full_input_ids,
                            merge_attention_mask,
                            full_labels,
                            full_loss_mask,
                            target_seq_length=target_seq_length,
                        )
                    )

                    # Re-distribute using zigzag so the attention layer sees its own slice.
                    # Results are batch-first [1, 2*chunk, H/1]; line 380 will transpose to seq-first.
                    inputs_embeds = _cp_slice_zigzag(merged_embeds, cp_group, seq_dim=1, pad_value=0)
                    if labels is not None and merged_labels is not None:
                        labels = _cp_slice_zigzag(
                            merged_labels, cp_group, seq_dim=1, pad_value=self.config.ignore_index
                        )
                    if merged_posids is not None:
                        position_ids = _cp_slice_zigzag(merged_posids, cp_group, seq_dim=1, pad_value=1)
                    if merged_loss_mask is not None:
                        loss_mask = _cp_slice_zigzag(merged_loss_mask, cp_group, seq_dim=1, pad_value=0)

                    attention_mask = None
                else:
                    num_placeholders, total_image_features = _get_image_merge_stats(
                        input_ids, image_features, self.config.media_placeholder_token_id
                    )
                    if packed_seq_params is not None and num_placeholders != total_image_features:
                        raise RuntimeError(
                            "KimiK25VL packed path requires pre-expanded image placeholders, "
                            f"but got dynamic expansion (num_placeholders={num_placeholders}, "
                            f"total_image_features={total_image_features}). "
                            "This makes packed_seq_params and slime rollout total_lengths/response_lengths stale."
                        )
                    inputs_embeds, attention_mask, labels, loss_mask, position_ids = (
                        self._merge_input_ids_with_image_features(
                            image_features,
                            inputs_embeds,
                            input_ids,
                            attention_mask,
                            labels,
                            loss_mask,
                            target_seq_length=target_seq_length,
                        )
                    )

            inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()
            if self.config.sequence_parallel:
                inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)


        outputs = self.language_model.forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds,
            labels=labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
            **(extra_block_kwargs or {}),
        )
        return outputs

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module (patch_embed and blocks).
            freeze_vision_projection (bool): Freeze the vision projection module (merger).
        """
        modules = []

        if freeze_language_model and hasattr(self, "language_model") and self.language_model is not None:
            modules.append(self.language_model)

        if freeze_vision_model and hasattr(self, "vision_tower") and self.vision_tower is not None:
            # Vision model consists of patch_embed and blocks
            modules.append(self.vision_tower)

        if freeze_vision_projection and hasattr(self, "mm_projector") and self.mm_projector is not None:
            # Vision projection is the merger module
            modules.append(self.mm_projector)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False