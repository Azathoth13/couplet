"""RMSNorm feature reduction with one real inverse half on each of two GPUs.

Rank 0 performs the original encrypted feature/slot sum. Ranks 0 and 1 each
apply the unchanged inverse_sqrt to one 64-token half, then share the factors
in one all-gather. The original output order and local multiplication remain.
This is an isolated candidate; baseline/sharded modes delegate to the old A/B.
"""
from __future__ import annotations

import time
from typing import Any


def make_rmsnorm_factory(original_factory, *, mode: str = "two_roots",
                         audit: list[dict] | None = None):
    if mode in {"baseline", "sharded"}:
        from .rmsnorm_shard import (
            make_rmsnorm_factory as old_factory,
        )
        return old_factory(original_factory, mode=mode, audit=audit)
    if mode != "two_roots":
        raise ValueError("RMSNorm mode must be baseline, sharded or two_roots")

    def factory(base_class: type, runtime: Any, **kwargs: Any) -> type:
        from .prefill_shard import block_range
        from sylph_easy.layer.nonlinear import inverse_sqrt
        import easyfhe as torch

        distributed = original_factory(base_class, runtime, **kwargs)
        original_rmsnorm = distributed.rmsnorm

        class RMSNormLayer(distributed):
            def rmsnorm(self, x):
                b = self.b
                bucket = str(self._bucket)
                call_index = getattr(self, "_rmsnorm_ab_calls", 0)
                self._rmsnorm_ab_calls = call_index + 1
                identity = {
                    "layer": int(self.layer_index), "bucket": bucket,
                    "rank": int(runtime.rank), "world_size": int(runtime.world_size),
                    "call_index": call_index,
                    "warmup": int(self.layer_index) == 0 and call_index < 2,
                }
                if runtime.world_size == 1:
                    before_sum = self.times.get(f"{bucket}_sum", 0.0)
                    before_inv = self.times.get(f"{bucket}_inv", 0.0)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    result = original_rmsnorm(self, x)
                    torch.cuda.synchronize()
                    row = {
                        "mode": "baseline", **identity,
                        "ciphertexts": int(x.batch_size),
                        "square_reduce_sum_s": self.times.get(f"{bucket}_sum", 0.0) - before_sum,
                        "inverse_and_mul_s": self.times.get(f"{bucket}_inv", 0.0) - before_inv,
                        "total_s": time.perf_counter() - started,
                        "communication_s": 0.0,
                    }
                    if audit is not None:
                        audit.append(row)
                    return result

                groups = int(x.batch_size) // 2
                if int(x.batch_size) != 2 * groups or int(self.hidden) != 512 * groups:
                    raise ValueError("RMSNorm expects two SC ciphertexts per 512-feature group")
                first, last = block_range(groups, runtime.rank, runtime.world_size)
                tag = self._tag(f"{bucket}.rmsnorm")
                phases: dict[str, float] = {}
                torch.cuda.synchronize()
                started = time.perf_counter()

                def measured(name, operation):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    value = operation()
                    torch.cuda.synchronize()
                    phases[name] = time.perf_counter() - t0
                    return value

                local_x = b.take(x, 2 * first, 2 * last) if first < last else None

                def square_sum():
                    if local_x is None:
                        return None
                    sq = b.unstack(b.square(local_x))
                    # Keep degree 2 through the encrypted cross-rank reduction,
                    # exactly as the original keeps degree 2 across feature groups.
                    return b.stack([b.add_many(sq[0::2]), b.add_many(sq[1::2])])

                partial = measured("local_square_sum_s", square_sum)
                packet = measured("partial_pack_s", lambda: None if partial is None else runtime.pack(
                    partial, tag=f"{tag}.partial", host=not runtime.transport.device_payloads))
                owned = (first, last, packet)
                # all_to_all uses the existing pinned/shared-memory transport;
                # only destination 0 receives ciphertexts, other payloads are None.
                outgoing = [None] * runtime.world_size
                outgoing[0] = owned
                received = measured("partial_exchange_s", lambda: runtime.transport.all_to_all(
                    f"{tag}.reduce", outgoing))

                summed = None
                if runtime.rank == 0:
                    def unpack_partials():
                        parts = []
                        for rank, item in enumerate(received):
                            expected = block_range(groups, rank, runtime.world_size)
                            if item is None or tuple(item[:2]) != expected:
                                raise RuntimeError(f"RMSNorm partial ownership mismatch at rank {rank}")
                            begin, end, remote = item
                            if begin == end:
                                if remote is not None:
                                    raise RuntimeError("idle RMSNorm rank emitted a partial")
                                continue
                            part = partial if rank == 0 else runtime.unpack(remote)
                            if int(part.batch_size) != 2:
                                raise RuntimeError("RMSNorm partial must contain both token halves")
                            parts.append(part)
                        return parts
                    parts = measured("partial_unpack_s", unpack_partials)
                    total = measured("encrypted_reduce_s", lambda: b.add_many(parts))
                    # Normalize/rescale once only after all groups were summed.
                    def slot_sum():
                        lo, hi = b.unstack(total)
                        return b.sum_slots(b.stack_aligned([lo, hi]), 64, 512)
                    summed = measured("sum_slots_s", slot_sum)

                # Share the same two encrypted token-half sums. No plaintext
                # values leave CKKS, and no extra inverse is computed by idle
                # ranks. Keep the original plan, epsilon, factor and levels.
                sum_packet = measured("sum_pack_s", lambda: runtime.pack(
                    summed, tag=f"{tag}.summed", host=not runtime.transport.device_payloads)
                    if runtime.rank == 0 else None)
                shared_sum = measured("sum_broadcast_s", lambda: runtime.broadcast_packet(
                    f"{tag}.summed", sum_packet, root=0))
                owner = int(runtime.rank) if int(runtime.rank) < 2 else None
                local_inv = None
                if owner is not None:
                    if runtime.rank != 0:
                        summed = measured("sum_unpack_s", lambda: runtime.unpack(shared_sum))
                    half = measured("select_inverse_half_s", lambda: b.take(summed, owner, owner + 1))
                    local_inv = measured("inverse_sqrt_s", lambda: inverse_sqrt(
                        b, half, self.rms_plan, 1.0 / self.hidden, self.eps, output_levels=1))
                inv_packet = measured("inverse_pack_s", lambda: runtime.pack(
                    local_inv, tag=f"{tag}.inverse{owner}",
                    host=not runtime.transport.device_payloads)
                    if owner is not None else None)
                factor_packets = measured("inverse_all_gather_s", lambda: runtime.all_gather(
                    f"{tag}.inverse_halves", (owner, inv_packet)))

                def restore_inverse():
                    if len(factor_packets) != int(runtime.world_size):
                        raise RuntimeError("RMSNorm inverse gather has missing ranks")
                    factors = []
                    for half_owner in (0, 1):
                        rank_owner, packet = factor_packets[half_owner]
                        if rank_owner != half_owner or packet is None:
                            raise RuntimeError("RMSNorm inverse-half ownership mismatch")
                        factor = local_inv if runtime.rank == half_owner else runtime.unpack(packet)
                        if int(factor.batch_size) != 1:
                            raise RuntimeError("RMSNorm inverse half must have batch size one")
                        factors.append(factor)
                    if any(item != (None, None) for item in factor_packets[2:]):
                        raise RuntimeError("idle RMSNorm rank computed an inverse half")
                    return b.stack(factors)
                inv = measured("inverse_unpack_stack_s", restore_inverse)

                def apply_inverse():
                    if local_x is None:
                        return None
                    lo, hi = b.unstack(inv)
                    return b.mul(local_x, b.stack([lo, hi] * (last-first)), rescale=True)
                local_y = measured("local_mul_s", apply_inverse)
                y_packet = measured("output_pack_s", lambda: None if local_y is None else runtime.pack(
                    local_y, tag=f"{tag}.output", host=not runtime.transport.device_payloads))
                gathered = measured("output_gather_s", lambda: runtime.all_gather(
                    f"{tag}.output", (first, last, y_packet)))

                def restore_order():
                    parts = []
                    for rank, item in enumerate(gathered):
                        expected = block_range(groups, rank, runtime.world_size)
                        if item is None or tuple(item[:2]) != expected:
                            raise RuntimeError(f"RMSNorm output ownership mismatch at rank {rank}")
                        begin, end, remote = item
                        if begin == end:
                            continue
                        part = local_y if rank == runtime.rank else runtime.unpack(remote)
                        if int(part.batch_size) != 2 * (end-begin):
                            raise RuntimeError("RMSNorm output batch size differs from its group range")
                        parts.append(part)
                    return b.stack(parts)
                result = measured("output_unpack_stack_s", restore_order)
                total_s = time.perf_counter() - started
                for name, seconds in phases.items():
                    key = f"{bucket}_rms_two_roots_{name}"
                    self.times[key] = self.times.get(key, 0.0) + seconds
                if audit is not None:
                    audit.append({
                        "mode": "two_roots", **identity,
                        "inverse_owner": owner,
                        "inverse_input_batch": 1 if owner is not None else 0,
                        "factor_order": [0, 1],
                        "feature_groups": groups, "owned_groups": [first,last],
                        "total_s": total_s, **phases,
                    })
                return result

        return RMSNormLayer

    return factory


def make_two_root_rmsnorm_factory(original_factory, *, audit=None):
    return make_rmsnorm_factory(original_factory, mode="two_roots", audit=audit)


__all__ = ["make_rmsnorm_factory", "make_two_root_rmsnorm_factory"]
