# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Buffers of the context-parallel decode indexer (TP4, gfx950).

Rank ``me`` holds index head ``me`` and scores a quarter of every request's
blocks for all four heads; one fused kernel exchanges the index-q rows, scores
and stores head h's scores into rank h's receive buffer
(``nvfp4_index_decode.kernel_cp_push``), and rank h's top-k reads it.

``IndexCP`` is created once per rank (a collective: every TP rank calls it at
the same point, outside graph capture). It owns, in uncached memory mapped by
the peers: the score receive buffer [max_rows, max_width] fp32 (this rank's
head, every block), the index-q receive buffer [4, max_rows, 128] e4m3 and the
flag array. No tickets or atomics on a shared word: workgroup w stores its call
count into its own flag word on every receiver once its stores are released, and
the waiters spin on the words they need. The call counts live on the device, so
the handshake replays in CUDA graphs; every rank issues the same calls.
"""

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.models.minimax_m3.amd.ops.nvfp4_index_decode import (
    kernel_cp_push as _kernel,
)

WORLD = _kernel.WORLD


class _DevicePointer:
    """A raw device allocation as a tensor (``__cuda_array_interface__``)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def _exchange(obj, rank: int, group) -> list:
    out = [[None] for _ in range(WORLD)]
    out[rank][0] = obj
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


class IndexCP:
    def __init__(
        self, rank: int, group, device: torch.device, max_rows: int, max_width: int
    ):
        self.rank, self.device = rank, device
        self.max_rows, self.max_width = max_rows, max_width
        score_bytes = max_rows * max_width * 4
        iq_bytes = WORLD * max_rows * _kernel.D
        mine = [
            ops.allocate_shared_buffer_and_handle(n)
            for n in (score_bytes, iq_bytes, _kernel.FLAG_BYTES)
        ]
        peers = _exchange([h for _, h in mine], rank, group)
        ptrs = [
            [p if r == rank else ops.open_mem_handle(peers[r][k]) for r in range(WORLD)]
            for k, (p, _) in enumerate(mine)
        ]
        self.score_ptrs, self.iq_ptrs, self.flag_ptrs = ptrs
        self.score_rcv = torch.as_tensor(
            _DevicePointer(self.score_ptrs[rank], score_bytes), device=device
        ).view(torch.float32)
        flags = torch.as_tensor(
            _DevicePointer(self.flag_ptrs[rank], _kernel.FLAG_BYTES), device=device
        )
        flags.zero_()
        # per-workgroup call counts (the flag values), equal on every rank
        self.counts = torch.zeros(_kernel.CNT_WORDS, dtype=torch.int32, device=device)
        torch.accelerator.synchronize(device)
        dist.barrier(group=group)

    def score_push(
        self,
        iq: torch.Tensor,
        index_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        init_blocks: int,
        local_blocks: int,
        width: int,
        grid: int = 0,
    ) -> torch.Tensor:
        """From this rank's index-q rows ``iq`` [rows, 128] (its head), scores
        this rank's blocks of every head straight into the heads' ranks, the
        heads' rows exchanged on the way (``nvfp4_index_decode.kernel_cp_push``);
        returns this rank's head over every block, [1, rows, width] (a view of
        the receive buffer, valid until the next call)."""
        from vllm.models.minimax_m3.amd.ops.nvfp4_index_decode.host import (
            minimax_m3_nvfp4_index_score_push_cp,
        )

        rows = iq.shape[0]
        assert rows <= self.max_rows and width <= self.max_width
        minimax_m3_nvfp4_index_score_push_cp(
            iq,
            index_cache,
            self.score_ptrs,
            width,
            rows * width * 4,
            block_table,
            seq_lens,
            init_blocks,
            local_blocks,
            self.rank,
            self.counts,
            self.flag_ptrs,
            self.iq_ptrs,
            self.max_rows,
            grid=grid,
        )
        return self.score_rcv[: rows * width].view(1, rows, width)


# The rank's IndexCPs, one per DBO microbatch (each microbatch's calls keep
# their own order on every rank), once ``init`` made them; empty = the
# single-head decode path.
_CP: list[IndexCP] = []


def init(device: torch.device, max_width: int) -> None:
    """Create this rank's ``IndexCP``s (a collective: every TP rank calls it at
    the same point, once, outside graph capture). TP4 on gfx950 only."""
    if _CP:
        return
    from vllm.distributed import get_tp_group
    from vllm.platforms.rocm import on_gfx950

    tp = get_tp_group()
    if tp.world_size != WORLD or not on_gfx950():
        return
    for _ in range(2):
        _CP.append(
            IndexCP(tp.rank_in_group, tp.cpu_group, device, _kernel.MAX_ROWS, max_width)
        )


def get() -> IndexCP | None:
    if not _CP:
        return None
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id

    return _CP[dbo_current_ubatch_id()]
