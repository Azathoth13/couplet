"""Distribute SC Decode's RhombusMV plaintext groups across physical GPUs.

Each rank owns ``p = rank, rank + world_size, ...`` of a projection's ``pc``
groups.  A rank computes only those plaintext products, inserts zero groups
before the linear PackRLWEs, and sends one packed coefficient ciphertext.
Rank zero adds the packed partials and performs the sole HalfBoot.  Its slot
ciphertext is then broadcast so the unchanged DecodeMVLayer can continue on
every rank.  All ranks must use the same CKKS secret/evaluation keys and call
``project`` in the same order.  This module does not initialize CUDA at import.

``make_distributed_projector(DeviceProjector, runtime)`` is a drop-in class
factory for a runner that replaces its imported ``DeviceProjector`` symbol.
The runtime is ``baselines.sc.parallel.runtime.RankRuntime``.
"""

from __future__ import annotations

from itertools import count

import numpy as np


_PROJECT_SEQUENCE = count()


def pc_groups(plan, rank: int, world_size: int) -> np.ndarray:
    """Global ``pc`` group indices owned by one rank, in ascending order."""

    rank, world_size = int(rank), int(world_size)
    if world_size not in (1, 2, 4, 8) or not 0 <= rank < world_size:
        raise ValueError("SC DecodeMV supports rank within W1/2/4/8")
    groups = np.arange(rank, int(plan.pc), world_size, dtype=np.int64)
    if not len(groups):
        raise ValueError(f"MVPlan.pc={plan.pc} has no group for rank {rank} at W{world_size}")
    return groups


def _build_partial_middle(rh, plan, weights, row_scale, groups: np.ndarray):
    """Build only this rank's ``[local_pc * a, N]`` GPU middle table.

    This follows ``RhombusMV.build`` exactly, including its global ``1/m``
    factor.  Local table rows are ordered ``[local p, automorphism index]``;
    the uncompressed source index is ``i0 * plan.pc + p``.  The global factor
    matters because packed partials are added, not averaged, at rank zero.
    """

    import easyfhe as torch

    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (int(plan.rows), int(plan.cols)):
        raise ValueError(f"weight shape {w.shape} differs from MVPlan {(plan.rows, plan.cols)}")
    if row_scale is not None:
        scale = np.asarray(row_scale, dtype=np.float64)
        if scale.shape != (int(plan.rows),):
            raise ValueError("row_scale does not match MVPlan rows")
        w = w * scale[:, None]
    top = float(np.max(np.abs(w))) if w.size else 0.0

    n, m, a, pc = int(plan.n), int(plan.m), int(plan.a), int(plan.pc)
    local_pc = len(groups)
    dev_index = lambda v: torch.as_tensor(np.ascontiguousarray(v, dtype=np.int64), device="cuda")
    row_to_feature = np.full(int(plan.padded_rows), -1, dtype=np.int64)
    row_to_feature[np.asarray(plan.out_row, dtype=np.int64)] = np.arange(int(plan.rows))
    local_source = (np.arange(a, dtype=np.int64)[:, None] * pc + groups[None, :]).reshape(-1)
    pw = torch.zeros((a * local_pc, n), dtype=torch.float64, device="cuda")
    w_dev = torch.as_tensor(np.ascontiguousarray(w), device="cuda")

    # Each plan row is j*m + i0*pc + p.  Select the actual output features
    # mapped to our p groups, then fill precisely their negacyclic weights.
    for j in range(n // int(plan.span)):
        features = row_to_feature[j * m + local_source]
        selected = np.flatnonzero(features >= 0)
        if not len(selected):
            continue
        pos = j * int(plan.span) - np.asarray(plan.in_pos, dtype=np.int64)
        sign = np.where(pos < 0, -1.0, 1.0)
        # Use the same row-index and column-slice tensor primitives as the
        # production RhombusMV.build; EasyFHE has exercised both on GPU.
        block = torch.zeros((a * local_pc, int(plan.cols)), dtype=torch.float64, device="cuda")
        block[dev_index(selected)] = w_dev[dev_index(features[selected])]
        pw[:, dev_index(pos % n)] = block * torch.as_tensor(sign, device="cuda")
        del block
    del w_dev

    out = torch.zeros((local_pc, a, n), dtype=torch.float64, device="cuda")
    for idx, automorphism in enumerate(plan.autos):
        src, sign = plan.automorphism_tables(automorphism)
        src_dev = dev_index(src)
        sign_dev = torch.as_tensor(sign, device="cuda")
        for i0 in range(a):
            block = pw[i0 * local_pc:(i0 + 1) * local_pc][:, src_dev] * sign_dev
            shift = (int(plan.shift) * i0) % (2 * n)
            negative = shift >= n
            if negative:
                shift -= n
            if shift:
                block = torch.cat([-block[:, n - shift:], block[:, :n - shift]], 1)
            if negative:
                block = -block
            out[:, idx] += block
            del block
    del pw
    middle = (out.reshape(local_pc * a, n) * (1.0 / m))[:, rh._inter].contiguous()
    return middle, top * a / m


def _zero_like_cipher(cipher):
    import easyfhe as torch

    return cipher.cipher_like([torch.zeros(part.shape, dtype=part.dtype, device="cuda")
                               for part in cipher.cv], batch_size=1)


def _partial_packed(projector, coeff, name: str):
    """Compute this rank's packed coefficient partial; never HalfBoot here."""

    rh, kit, plan = projector.rh, projector.kit, projector.geo.plans[name]
    middle, mx, groups = projector.dev[name]
    copies = rh.inputs(coeff, plan)
    limbs = int(copies.state.cur_limbs)
    adjust = kit.raise_adjustment(int(coeff.state.cur_limbs))
    local_pc = len(groups)
    batch = int(projector.groups) if 0 < int(projector.groups) < local_pc else local_pc
    chunks = []
    for first in range(0, local_pc, batch):
        last = min(local_pc, first + batch)
        plaintexts = rh.encode(middle[first * plan.a:last * plan.a], mx, limbs, adjust)
        rh._count("mul_pt", (last - first) * plan.a)
        chunks.append(rh.arithmetic.grouped_pairwise_mac(copies, plaintexts, last - first, rh.ctx))
        del plaintexts
    local_acc = kit.stack(chunks) if len(chunks) > 1 else chunks[0]
    local_items = kit.unstack(local_acc)
    zero = _zero_like_cipher(local_items[0])
    full = [zero] * int(plan.pc)
    for p, item in zip(groups, local_items, strict=True):
        full[int(p)] = item
    # PackRLWEs is linear, so sum(rank-packed partial) equals packing all pc
    # products at once.  Zero slots cost rotations; the costly plaintext MAC
    # and table residency are partitioned.
    return rh._pack(kit.stack(full), plan) if plan.pc > 1 else rh.alignment.normalize_scale(full[0], rh.ctx)


def make_distributed_projector(DeviceProjector, runtime):
    """Return ``DeviceProjector`` subclass with a distributed ``project``.

    ``runtime`` provides ``rank``, ``world_size``, ``pack``, ``unpack``,
    ``all_gather(tag, packet)`` and ``broadcast_packet(tag, packet, root=0)``.
    W1 delegates to the unchanged source implementation for exact baseline
    parity; W2/4/8 split pc plaintext groups and keep one HalfBoot on rank 0.
    """

    class DistributedDeviceProjector(DeviceProjector):
        def __init__(self, eng, geo):
            super().__init__(eng, geo)
            self.runtime = runtime
            self.rank = int(runtime.rank)
            self.world_size = int(runtime.world_size)
            if self.world_size not in (1, 2, 4, 8) or not 0 <= self.rank < self.world_size:
                raise ValueError("invalid SC multi-GPU DecodeMV rank/world")

        def prepare(self, name, w, row_scale):
            if self.world_size == 1:
                return super().prepare(name, w, row_scale)
            plan = self.geo.plans[name]
            groups = pc_groups(plan, self.rank, self.world_size)
            middle, mx = _build_partial_middle(self.rh, plan, w, row_scale, groups)
            self.dev[name] = (middle, mx, groups)

        def project(self, coeff, name):
            if self.world_size == 1:
                return super().project(coeff, name)
            seq = next(_PROJECT_SEQUENCE)
            tag = f"sc_decode_mv/{seq:06d}/{name}"
            partial = _partial_packed(self, coeff, name)
            host_packet = not bool(getattr(runtime.transport, "device_payloads", False))
            packets = runtime.all_gather(
                f"{tag}/partials", runtime.pack(partial, tag=f"{tag}/partial", host=host_packet))
            if len(packets) != self.world_size:
                raise RuntimeError("DecodeMV partial gather returned the wrong rank count")
            slot = None
            if self.rank == 0:
                parts = [partial if rank == 0 else runtime.unpack(packet, template=partial)
                         for rank, packet in enumerate(packets)]
                packed = self.b.add_many(parts)
                slot = self.kit.halfboots([packed], folded=True, real_only=True)[0][0]
            packet = runtime.broadcast_packet(
                f"{tag}/slot", runtime.pack(slot, tag=f"{tag}/slot", host=host_packet)
                if slot is not None else None,
                root=0,
            )
            return slot if self.rank == 0 else runtime.unpack(packet)

    DistributedDeviceProjector.__name__ = "DistributedDeviceProjector"
    return DistributedDeviceProjector


__all__ = ["make_distributed_projector", "pc_groups"]
