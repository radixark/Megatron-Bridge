# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""GLM-5 UNFUSED DSA backend (goes through megatron-core / megatron-bridge).

Selected when ``dsa_attention_backend != "tilelang"`` (the default). The unfused
sparse-MLA path reuses megatron-core's experimental ``DSAttention`` (lightning
indexer + ``unfused_dsa_fn``); GLM-5.2 cross-layer index-sharing is layered on
top by ``CrossLayerDSAttention`` in ``../cross_layer_dsa_dispatch.py``.

Centralising the megatron-core DSA imports here marks the unfused backend's
single entry point (mirrors how ``../tilelang/`` is the tilelang entry point).
"""

from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    DSAttention,
    FusedDSAIndexerLoss,
    unfused_dsa_fn,
)

__all__ = [
    "DSAttention",
    "FusedDSAIndexerLoss",
    "unfused_dsa_fn",
    "DSAIndexerLossAutoScaler",
    "DSAIndexerLossLoggingHelper",
]
