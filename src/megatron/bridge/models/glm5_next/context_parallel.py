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

"""Token ordering and differentiable key gathering for packed GLM-5.3 CP."""

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


def cp_token_positions(cu_seqlens: torch.Tensor, *, cp_size: int, cp_rank: int, layout: str) -> torch.Tensor:
    """Map CP-local query rows to positions in the global packed token stream.

    ``contiguous`` shards the entire packed stream. ``zigzag`` shards each
    sequence separately, as Megatron's packed THD input and Miles do.
    Boundaries must include any padding used by the input packer.
    """
    if layout not in ("contiguous", "zigzag"):
        raise ValueError(f"Unknown GLM-5.3 CP layout: {layout!r}")
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"Invalid CP rank/size: {cp_rank}/{cp_size}")
    boundaries = cu_seqlens.tolist()
    if len(boundaries) < 2 or boundaries[0] != 0:
        raise ValueError("cu_seqlens must start at zero and contain at least one sequence")
    if any(end <= start for start, end in zip(boundaries, boundaries[1:])):
        raise ValueError("cu_seqlens must be strictly increasing")
    total = boundaries[-1]
    if total % cp_size:
        raise ValueError("Packed token count must be divisible by CP size")
    if cp_size == 1:
        return torch.arange(total, device=cu_seqlens.device)
    if layout == "contiguous":
        local = total // cp_size
        return torch.arange(cp_rank * local, (cp_rank + 1) * local, device=cu_seqlens.device)
    pieces = []
    for start, end in zip(boundaries, boundaries[1:]):
        if (end - start) % (2 * cp_size):
            raise ValueError("Each zigzag packed sequence must be divisible by 2 * CP size")
        half = (end - start) // (2 * cp_size)
        for slot in (cp_rank, 2 * cp_size - 1 - cp_rank):
            pieces.append(torch.arange(start + slot * half, start + (slot + 1) * half, device=cu_seqlens.device))
    return torch.cat(pieces)


def gather_cp_tokens(
    tensor: torch.Tensor, cu_seqlens: torch.Tensor, *, group: dist.ProcessGroup, layout: str
) -> torch.Tensor:
    """Gather local keys in natural token order, summing remote key gradients.

    Sparse attention on every CP rank uses the gathered latent keys. Its adjoint
    must sum contributions from all query shards before returning local gradients.
    The frozen indexer uses the same ordering without retaining an autograd graph.
    """
    size = dist.get_world_size(group)
    if size == 1:
        return tensor
    positions = torch.cat(
        [cp_token_positions(cu_seqlens, cp_size=size, cp_rank=rank, layout=layout) for rank in range(size)]
    )
    if positions.numel() != size * tensor.shape[0]:
        raise ValueError("CP-local key count does not match global packed boundaries")
    if torch.is_grad_enabled() and tensor.requires_grad:
        shards = all_gather(tensor.contiguous(), group=group)  # type: ignore[no-untyped-call]  # Untyped PyTorch API.
    else:
        shards = [torch.empty_like(tensor) for _ in range(size)]
        dist.all_gather(shards, tensor.contiguous(), group=group)
    gathered = torch.cat(shards, dim=0)
    if layout == "contiguous":
        return gathered
    return torch.empty_like(gathered).index_copy(0, positions, gathered)


def kda_cp_segments(cu_seqlens: torch.Tensor, *, cp_size: int, layout: str) -> list[tuple[slice, torch.Tensor]]:
    """Return local KDA slices and global boundaries for FLA CP contexts.

    FLA's zigzag layout splits the whole stream into two parts per rank.
    Megatron's packed zigzag layout instead splits every sequence. Run those
    sequences separately so recurrent state and convolution halos reset at the
    correct boundaries. Contiguous CP can process the whole packed stream once.
    """
    # Validate the packing contract before launching any CP collectives.
    cp_token_positions(cu_seqlens, cp_size=cp_size, cp_rank=0, layout=layout)
    if layout == "contiguous":
        return [(slice(None), cu_seqlens)]
    boundaries = cu_seqlens.tolist()
    segments = []
    local_start = 0
    for start, end in zip(boundaries, boundaries[1:]):
        local_end = local_start + (end - start) // cp_size
        segment_cu = cu_seqlens.new_tensor([0, end - start])
        segments.append((slice(local_start, local_end), segment_cu))
        local_start = local_end
    return segments
