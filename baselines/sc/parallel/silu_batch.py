"""SC SiLU feature partition with one packet per identical ciphertext state.

Polynomial evaluation, multiplication, output ordering and the single final
global stack_aligned are the established sharded implementation. Only the
wire representation changes: compatible atomic outputs are stacked together.
"""
from __future__ import annotations

import time
from typing import Any


def cipher_signature(cipher: Any) -> tuple:
    """Batch-independent CKKS state required by a lossless raw stack."""
    state = cipher.state
    factor = getattr(state, 'scaling_factor', None)
    return (len(cipher.cv), tuple(str(x.dtype) for x in cipher.cv),
            int(cipher.slots), bool(getattr(cipher, 'is_ext', False)),
            int(state.cur_limbs), int(state.scale_degree),
            None if factor is None else float(factor))


def group_local_outputs(local: list[Any], first_group: int) -> list[tuple[tuple[int, ...], list[Any]]]:
    """Keep original global positions while grouping equal states in order."""
    grouped: dict[tuple, tuple[list[int], list[Any]]] = {}
    for offset, ct in enumerate(local):
        if int(ct.batch_size) != 1:
            raise ValueError('SiLU state grouping expects atomic ciphertexts')
        indices, values = grouped.setdefault(cipher_signature(ct), ([], []))
        indices.append(2 * first_group + offset)
        values.append(ct)
    return [(tuple(indices), values) for indices, values in grouped.values()]


def make_silu_batched_decoder_layer(base_class: type, runtime: Any, *,
                                    tag_prefix: str = 'sc-prefill',
                                    original_factory: Any = None,
                                    audit: list[dict] | None = None) -> type:
    from .prefill_shard import block_range, make_distributed_decoder_layer
    from sylph_easy.layer.approx import silu_coefficients
    import easyfhe as torch

    factory = original_factory or make_distributed_decoder_layer
    distributed_base = factory(base_class, runtime, tag_prefix=tag_prefix)

    class SiLUBatchedDecoderLayer(distributed_base):
        def silu_gate(self, gate: Any, up: Any) -> Any:
            if runtime.world_size == 1:
                return super().silu_gate(gate, up)
            b, plans = self.b, self.silu_plans
            groups = len(plans)
            first, last = self._owned['gate']
            if (self._owned['up'] != (first, last)
                    or int(gate.batch_size) != 2 * groups
                    or int(up.batch_size) != 2 * groups
                    or (first, last) != block_range(groups, runtime.rank, runtime.world_size)):
                raise RuntimeError('SiLU ownership disagrees with the feature-group layout')
            torch.cuda.synchronize()
            started = time.perf_counter()
            local: list[Any] = []
            if first < last:
                g = b.unstack(b.take(gate, 2 * first, 2 * last))
                u = b.unstack(b.take(up, 2 * first, 2 * last))
                ordered: list[Any | None] = [None] * (2 * (last - first))
                local_plans = plans[first:last]
                for degree, domain in sorted(set(local_plans)):
                    items = [i for s, plan in enumerate(local_plans)
                             if plan == (degree, domain) for i in (2 * s, 2 * s + 1)]
                    s = b.poly(b.stack([g[i] for i in items]), silu_coefficients(domain, degree))
                    y = b.mul(b.ready(s, 1, 2 * domain),
                              b.stack([u[i] for i in items]), rescale=True)
                    for i, ct in zip(items, b.unstack(y), strict=True):
                        ordered[i] = ct
                if any(ct is None for ct in ordered):
                    raise RuntimeError('SiLU left an owned group uncomputed')
                local = ordered
            torch.cuda.synchronize()
            compute_s = time.perf_counter() - started

            tag = self._tag('silu.batched_output')
            t_pack = time.perf_counter()
            batches = [(indices, b.stack(cts))
                       for indices, cts in group_local_outputs(local, first)]
            packets = tuple((indices, runtime.pack(batch, tag=f'{tag}.{n}',
                              host=not runtime.transport.device_payloads))
                            for n, (indices, batch) in enumerate(batches))
            torch.cuda.synchronize()
            pack_s = time.perf_counter() - t_pack
            t_exchange = time.perf_counter()
            gathered = runtime.all_gather(tag, (first, last, packets))
            torch.cuda.synchronize()
            exchange_s = time.perf_counter() - t_exchange
            t_restore = time.perf_counter()
            output: list[Any | None] = [None] * (2 * groups)
            for rank, item in enumerate(gathered):
                expected = block_range(groups, rank, runtime.world_size)
                if item is None or tuple(item[:2]) != expected:
                    raise RuntimeError(f'SiLU ownership mismatch from rank {rank}')
                begin, end, received = item
                for packet_index, (indices, packet) in enumerate(received):
                    # A local batch remains live until the collective completes.
                    batch = batches[packet_index][1] if rank == runtime.rank else runtime.unpack(packet)
                    values = b.unstack(batch)
                    if len(indices) != len(values):
                        raise RuntimeError('SiLU packet positions do not match its batch size')
                    for index, ct in zip(indices, values, strict=True):
                        if not 2 * begin <= index < 2 * end or output[index] is not None:
                            raise RuntimeError('SiLU packet has duplicated or non-owned output positions')
                        output[index] = ct
                if any(output[i] is None for i in range(2 * begin, 2 * end)):
                    raise RuntimeError('SiLU packet omitted an owned output')
            if any(ct is None for ct in output):
                raise RuntimeError('SiLU batched gather did not cover the output')
            # Do not locally align/drop/rescale: the original global alignment
            # happens once after restoring every ciphertext's original position.
            result = b.stack_aligned(output)
            torch.cuda.synchronize()
            restore_s = time.perf_counter() - t_restore
            phases = dict(silu_local_compute=compute_s, silu_pack=pack_s,
                          silu_exchange=exchange_s, silu_restore=restore_s,
                          silu_gather=pack_s + exchange_s + restore_s)
            for name, value in phases.items():
                self.times[name] = self.times.get(name, 0.0) + value
            if audit is not None:
                audit.append(dict(layer=int(self.layer_index), rank=int(runtime.rank),
                    owned_groups=[first, last], local_ciphertexts=len(local),
                    packets=len(packets), received_packets=sum(len(x[2]) for x in gathered),
                    **phases))
            return result

    return SiLUBatchedDecoderLayer


make_silu_batch_decoder_layer = make_silu_batched_decoder_layer

__all__ = ['cipher_signature', 'group_local_outputs',
           'make_silu_batched_decoder_layer', 'make_silu_batch_decoder_layer']
