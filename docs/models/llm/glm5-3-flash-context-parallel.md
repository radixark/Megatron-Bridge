# GLM-5.3 Flash context parallelism

GLM-5.3 Flash can split packed training sequences across a context-parallel
(CP) group while retaining head sharding across a tensor-parallel (TP) group.
KDA uses FLA's recurrent-state and causal-convolution halo communication.
DSA keeps queries local and gathers latent keys into global token order.
The pooled-key indexer selects global token indices for those local queries.

## Runtime requirements

The CP path requires FLA with `build_cp_context(layout=...)`, KDA CP backward,
and causal-convolution CP support. The tested revision is
`9c102739e348968753d86bb729c57dd6efa1c413`:

```bash
pip install --no-deps --no-build-isolation \
  'git+https://github.com/fla-org/flash-linear-attention.git@9c102739e348968753d86bb729c57dd6efa1c413'
```

Older FLA installations remain usable for CP1. The KDA constructor rejects
missing CP layout support when CP is enabled.

## Packed input layout

Set the provider's `glm5_next_cp_layout` before model construction:

- `"zigzag"` (default): matches Megatron packed THD input and Miles' default CP
  packing. Rank `r` owns chunks `r` and `2 * CP - 1 - r` of each sequence.
- `"contiguous"`: matches Miles' `--allgather-cp`. Concatenate the sequences,
  then assign each rank one contiguous slice of the entire packed stream.

Global packed boundaries must include padding used by the packer. Zigzag
requires every padded sequence length to be divisible by `2 * CP`; each local
half must have at least `conv_kernel_size - 1` tokens for convolution halos.
KDA runs packed zigzag sequences separately because FLA's zigzag context splits
the entire stream rather than each packed sequence independently.

TP4 with CP2 uses eight ranks without a data-parallel replica. The appropriate
balance depends on memory and communication cost; these tests do not establish
a performance advantage over TP8.

## Memory and replay

KDA token activations and DSA queries remain CP-local. DSA replicates latent
keys and indexer keys across CP ranks. The latent-key gather sums gradients
from all query shards in backward.

The indexer scores at most 32,768 local queries at once when indexer replay is
disabled. Indexer replay retains the complete local-query score path because
its manager consumes a complete stream. Expert-routing replay is separate.

## Tests

CPU packing and distributed key-gradient tests:

```bash
uv run python -m pytest tests/unit_tests/models/test_glm5_next_context_parallel.py
```

Two-GPU kernel tests, without model weights:

```bash
uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m pytest --confcutdir=tests/functional_tests/test_groups/models/glm5_next \
  tests/functional_tests/test_groups/models/glm5_next/test_context_parallel.py -v
```

The GPU tests compare CP2 with CP1 for both layouts. They cover the production
KDA forward using small projection modules, real FLA convolution and recurrence
backward, recomputation, pooled-key selection, query blocking, and SparseMLA
remote-key gradients. They do not cover a full model, TP+SP projection
collectives, optimizer updates, or sampler replay.
