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


from dataclasses import dataclass
from typing import Optional, Any

from megatron.core.models.gpt import GPTModel

from megatron.bridge.models.kimi.kimi_provider import KimiK2Provider
from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel
from megatron.bridge.models.kimi_vl.modelling_kimi_vl.transfomer_config import DeepseekV3Config
from transformers.dynamic_module_utils import get_class_from_dynamic_module



@dataclass
class KimiK25VLModelProvider(KimiK2Provider):
    """ """

    hf_model_path: Optional[str] = None
    vision_config: Any = None

    hf_text_config: Optional[DeepseekV3Config] = None
    pretrained_model_name: str = "moonshotai/Kimi-K2.5"

    bos_token_id: int = 163584
    eos_token_id: int = 163585
    media_placeholder_token_id: int = 163605
    freeze_language_model: bool = False
    # Whether to freeze vision encoder weights
    freeze_vision_model: bool = True
    # Whether to freeze vision-to-language projection weights
    freeze_vision_projection: bool = True
    scatter_embedding_sequence_parallel: bool = False

    variable_seq_lengths: bool = True
    moe_token_dispatcher_type: str = "alltoall"

    def finalize(self) -> None:
        if self.tensor_model_parallel_size > 1:
            self.sequence_parallel = True

        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        model = KimiK25VLModel(
            self,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

        # Apply freeze options if any are enabled for fine-tuning
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> GPTModel:
        """
        Provide just the language model component without vision.

        Args:
            pre_process: Whether this is the first stage in pipeline parallelism
            post_process: Whether this is the last stage in pipeline parallelism
            vp_stage: Virtual pipeline stage number

        Returns:
            GPTModel instance (language model only)
        """
        # Use parent class to create standard language model
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
