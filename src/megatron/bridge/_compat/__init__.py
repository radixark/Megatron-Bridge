"""Backports of newer megatron-core APIs.

Used as fallback imports when the runtime's Megatron-LM predates symbols
that newer megatron-bridge versions import (e.g. the radixark/miles:dev image's
bundled Megatron-LM is missing megatron.core.models.mimo.config.role).
"""
