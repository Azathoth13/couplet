"""Output-block parallel Sylph Prefill projections.

The factory leaves upstream ``sylph_easy`` untouched.  All ranks execute the
same decoder layer.  Rank 0 performs each slot-to-coefficient conversion and
broadcasts its ciphertext blocks; every rank computes its own contiguous
512-output-feature Rhombus blocks and HalfBoots; an all-gather reconstructs
the original batch order on every rank.  RMSNorm, attention and SiLU still run
on every rank, so this is a correct projection-parallel baseline rather than
a fully partitioned attention implementation.

The runtime protocol is supplied by ``runtime.py`` in this directory:
``rank``, ``world_size``, ``all_gather(tag, payload)``, ``pack(cipher, tag=...)``
and ``unpack(packet)``.  The runtime must create key-compatible EasyFHE
contexts on every rank before ciphertext packets cross ranks.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np


PROJECTIONS = ("q", "k", "v", "o", "gate", "up", "down")


def verify_shared_context(runtime: Any, client: Any, *, slots: int = 1 << 15,
                          atol: float = 1e-3) -> list[float]:
    """Check that every rank can decrypt rank 0's encrypted probe.

    Call this once after all ranks have made their Sylph contexts and before
    a real Prefill collective.  It catches incompatible secret keys or prime
    chains before an expensive full-model run.  The probe is outside timing.
    """

    reference = np.linspace(-0.25, 0.25, int(slots), dtype=np.float64)
    probe = client.encrypt(reference, slots=int(slots)) if runtime.rank == 0 else None
    packet = runtime.broadcast_packet("sc-prefill/context-probe",
                                      runtime.pack(probe, host=not runtime.transport.device_payloads)
                                      if probe is not None else None)
    received = probe if probe is not None else runtime.unpack(packet)
    actual = client.decrypt(received).cpu().numpy()[:int(slots)].real
    error = float(np.max(np.abs(actual - reference)))
    errors = [float(v) for v in runtime.all_gather("sc-prefill/context-probe-errors", error)]
    if len(errors) != runtime.world_size or any(not np.isfinite(v) or v > atol for v in errors):
        raise RuntimeError(f"SC ranks do not share a compatible CKKS context: errors={errors}")
    return errors


def block_range(blocks: int, rank: int, world_size: int) -> tuple[int, int]:
    """Contiguous balanced block ownership; ranks may own no block."""

    blocks, rank, world_size = int(blocks), int(rank), int(world_size)
    if blocks < 0 or world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("invalid block ownership")
    quotient, remainder = divmod(blocks, world_size)
    start = rank * quotient + min(rank, remainder)
    return start, start + quotient + int(rank < remainder)


def _row_selection(layer: Any, name: str, start: int, stop: int) -> np.ndarray:
    """Rows in the *original* weight matrix for one output block interval."""

    positions = np.arange(start, stop, dtype=np.int64)
    if name in ("gate", "up") and layer.ffn_order is not None:
        return np.asarray(layer.ffn_order, dtype=np.int64)[positions]
    return positions


def make_distributed_decoder_layer(base_class: type, runtime: Any, *, tag_prefix: str = "sc-prefill") -> type:
    """Return a drop-in ``DecoderLayer`` subclass bound to one GPU rank.

    ``run_chain.py`` imports ``DecoderLayer`` at module load, so a linked runner
    can temporarily replace ``sylph_easy.layer.decoder.DecoderLayer`` with the
    returned class before ``runpy.run_path``.  Constructor arguments remain
    identical to the upstream class.  Calling ``forward`` on all ranks in the
    same order is required by the collective protocol.
    """

    from sylph_easy.layer.decoder import CoeffInput

    class DistributedDecoderLayer(base_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # The base constructor calls upload() when requested.  Runtime
            # fields must already exist so our upload override can shard rows.
            self.runtime = runtime
            self.tag_prefix = str(tag_prefix)
            self._forward_index = 0
            self._owned: dict[str, tuple[int, int]] = {}
            super().__init__(*args, **kwargs)
            self.layer_index = int(self.cal.get("layer", -1))
            self._attention_shard = None
            if os.environ.get("SC_PREFILL_ATTENTION_SHARD", "0") == "1" and runtime.world_size > 1:
                if (self.heads, self.kv_heads, self.head_dim) not in ((24, 8, 128), (32, 8, 128), (32, 8, 64)):
                    raise ValueError("SC attention shard trial supports 1B head64, 3B compact24 or Llama-3-8B")
                first, last = block_range(self.heads * self.head_dim // self.kit.k,
                                          runtime.rank, runtime.world_size)
                self._attention_shard = (first, last)
                self._full_attention_for_cache = self.attention
                if first < last:
                    # Keep the original geometry and global D-item positions;
                    # only Q shards owned by this rank enter QK/Softmax/SV.
                    from sylph_easy.layer.attention import Attention

                    m = self.cal.get("model", {})
                    self.attention = Attention(
                        self.b, public_keys=self.w["public_keys"], public_values=self.w["public_values"],
                        logit_bound=float(self.cal["logit_bound"]),
                        centered_bound=float(self.cal["centered_logit_bound"]),
                        rounds=int(self.cal["softmax_rounds"]),
                        rope_theta=float(m.get("rope_theta", 500000.0)),
                        heads=self.heads, kv_heads=self.kv_heads, head_dim=self.head_dim,
                        rope_scaling=m.get("rope_scaling"), inverse_degree=127,
                        intervals=self.cal.get("softmax_intervals"),
                        head_intervals=self.cal.get("softmax_head_intervals"),
                        active_shards=tuple(range(first, last)),
                    )
                local_forward = self.attention.forward if first < last else None

                def distributed_attention_forward(q: Any, k: Any, v: Any) -> Any:
                    local = local_forward(q, k, v) if local_forward is not None else None
                    started = time.perf_counter()
                    tag = self._tag("attention.output")
                    payload = (first, last, None if local is None else runtime.pack(
                        local, tag=tag, host=not runtime.transport.device_payloads))
                    gathered = runtime.all_gather(tag, payload)
                    if len(gathered) != runtime.world_size:
                        raise RuntimeError("attention output gather returned wrong rank count")
                    parts = []
                    covered = 0
                    state_signature = None
                    for rank, item in enumerate(gathered):
                        expected = block_range(self.heads * self.head_dim // self.kit.k,
                                               rank, runtime.world_size)
                        if item is None or tuple(item[:2]) != expected or item[0] != covered:
                            raise RuntimeError(f"attention output ownership mismatch at rank {rank}")
                        start, stop, packet = item
                        if start < stop:
                            if packet is None:
                                raise RuntimeError(f"attention shard {rank} did not emit ciphertext")
                            meta = packet.metadata
                            signature = (meta.component_count, meta.component_dtypes, meta.cur_limbs,
                                         meta.scale_degree, meta.scaling_factor, meta.slots, meta.is_ext)
                            if state_signature is None:
                                state_signature = signature
                            elif signature != state_signature:
                                raise RuntimeError(f"attention ciphertext state differs at rank {rank}")
                            part = local if rank == runtime.rank else runtime.unpack(packet)
                            if int(part.batch_size) != 2 * (stop - start):
                                raise RuntimeError(f"attention shard {rank} has wrong batch size")
                            parts.append(part)
                        elif packet is not None:
                            raise RuntimeError(f"idle attention shard {rank} emitted ciphertext")
                        covered = stop
                    if covered != self.heads * self.head_dim // self.kit.k:
                        raise RuntimeError("attention output does not cover all Q shards")
                    out = self.b.stack(parts)
                    self.times["attention_gather"] = self.times.get("attention_gather", 0.0) + (
                        time.perf_counter() - started)
                    return out

                self.attention.forward = distributed_attention_forward

        def _tag(self, suffix: str) -> str:
            return f"{self.tag_prefix}.layer{self.layer_index}.pass{self._forward_index}.{suffix}"

        def upload(self) -> None:
            """Upload just this rank's output rows, preserving FFN row order."""

            rank, world = int(runtime.rank), int(runtime.world_size)
            if world == 1:
                # The paper's W1 denominator must use the upstream Sylph
                # upload path, including its exact weight staging and caches.
                return super().upload()
            k = int(self.kit.k)
            self.dev = {}
            self._owned = {}
            for name in PROJECTIONS:
                weights = np.asarray(self.w[name], dtype=np.float64)
                rows = int(weights.shape[0])
                if rows % k:
                    raise ValueError(f"{name} output width {rows} is not divisible by {k}")
                first, last = block_range(rows // k, rank, world)
                self._owned[name] = (first, last)
                if first == last:
                    self.dev[name] = None
                    continue
                selected = _row_selection(self, name, first * k, last * k)
                local_weights = weights[selected]
                row_div = self.gate_domains[first * k:last * k] if name == "gate" else None
                cols = self.ffn_order if name == "down" else None
                self.dev[name] = self.rhombus.upload(local_weights, cols=cols, row_div=row_div)

        def forward(self, x: Any, probe: Any = None) -> Any:
            if runtime.world_size == 1:
                return super().forward(x, probe)
            # The single-rank chain encrypts the same plaintext separately on
            # every process.  Sync the *ciphertext* to avoid divergent residuals.
            self._forward_index += 1
            payload = (runtime.pack(x, tag=self._tag("input"), host=not runtime.transport.device_payloads)
                       if runtime.rank == 0 else None)
            gathered = runtime.all_gather(self._tag("input"), payload)
            if len(gathered) != runtime.world_size or gathered[0] is None:
                raise RuntimeError("rank-0 Prefill input ciphertext was not broadcast")
            if runtime.rank != 0:
                x = runtime.unpack(gathered[0])
            return super().forward(x, probe)

        def coefficients(self, x: Any, out_features: int) -> Any:
            """Convert on rank 0 once and broadcast coefficient ciphertexts."""

            if runtime.world_size == 1:
                return super().coefficients(x, out_features)
            stage = str(self._bucket)
            tag = self._tag(f"{stage}.coeff")
            started = time.perf_counter()
            coeff = super().coefficients(x, out_features) if runtime.rank == 0 else None
            payload = None
            if coeff is not None:
                # Couplet's host-staging transport recursively stages tuples and
                # dataclasses, but not mappings.  Keep the nested packets in a
                # tuple so their large arrays use shared memory.
                payload = (int(coeff.split), int(out_features),
                           tuple(runtime.pack(c, tag=f"{tag}.{i}", host=not runtime.transport.device_payloads)
                                 for i, c in enumerate(coeff.blocks)))
            received = runtime.all_gather(tag, payload)
            if len(received) != runtime.world_size or received[0] is None:
                raise RuntimeError(f"rank-0 coefficient broadcast failed at {tag}")
            source_split, source_features, source_blocks = received[0]
            if int(source_features) != int(out_features) or not source_blocks:
                raise RuntimeError(f"coefficient geometry mismatch at {tag}")
            if coeff is None:
                blocks = [runtime.unpack(packet) for packet in source_blocks]
                coeff = CoeffInput(self.rhombus, blocks, int(out_features) // int(self.kit.k))
            if int(coeff.split) != int(source_split):
                raise RuntimeError(f"Rhombus split differs across ranks at {tag}")
            self.times[f"{stage}_coeff_broadcast"] = self.times.get(f"{stage}_coeff_broadcast", 0.0) + (
                time.perf_counter() - started
            )
            return coeff

        def project(self, coeff: Any, name: str, bound: float | None = None) -> Any:
            """Run local Rhombus/BS blocks, then restore the full ordered batch."""

            if runtime.world_size == 1:
                return super().project(coeff, name, bound)
            if self.dev is None or name not in self._owned:
                raise RuntimeError("distributed projection weights have not been uploaded")
            b, kit = self.b, self.kit
            bound = self.bounds[name] if bound is None else bound
            first, last = self._owned[name]
            tag = self._tag(f"{self._bucket}.{name}.output")
            local = None
            if first < last:
                adj = kit.raise_adjustment(coeff.blocks[0].state.cur_limbs)
                t0 = time.time()
                packed = self.rhombus.project(coeff.inputs(), self.dev[name], coeff.split, scale=adj / bound)
                t0 = self._part("pcmm", t0)
                out = [ct for pair in kit.halfboots(packed, folded=True) for ct in pair]
                local = b.stack(out)
                local = b.smul(local, bound) if bound != 1 else local
                self._part("boot", t0)
                if int(local.batch_size) != 2 * (last - first):
                    raise RuntimeError(f"{name} local HalfBoot batch has wrong size")
            started = time.perf_counter()
            payload = (first, last, None if local is None else
                       runtime.pack(local, tag=tag, host=not runtime.transport.device_payloads))
            gathered = runtime.all_gather(tag, payload)
            if len(gathered) != runtime.world_size:
                raise RuntimeError(f"projection gather failed at {tag}")
            output = []
            covered = 0
            state_signature = None
            total_blocks = int(np.shape(self.w[name])[0]) // int(kit.k)
            for rank, item in enumerate(gathered):
                expected = block_range(total_blocks, rank, runtime.world_size)
                if item is None or tuple(item[:2]) != expected or int(item[0]) != covered:
                    raise RuntimeError(f"{name} output block ownership mismatch at rank {rank}")
                start, stop, packet = item
                if start < stop:
                    if packet is None:
                        raise RuntimeError(f"{name} output block missing from rank {rank}")
                    meta = packet.metadata
                    signature = (meta.component_count, meta.component_dtypes, meta.cur_limbs,
                                 meta.scale_degree, meta.scaling_factor, meta.slots, meta.is_ext)
                    if state_signature is None:
                        state_signature = signature
                    elif signature != state_signature:
                        raise RuntimeError(f"{name} ciphertext state differs across ranks")
                    part = local if rank == runtime.rank else runtime.unpack(packet)
                    if int(part.batch_size) != 2 * (stop - start):
                        raise RuntimeError(f"{name} output batch shape mismatch at rank {rank}")
                    output.append(part)
                elif packet is not None:
                    raise RuntimeError(f"idle rank {rank} emitted {name} ciphertext")
                covered = stop
            if covered != total_blocks or not output:
                raise RuntimeError(f"{name} output blocks do not cover the projection")
            result = b.stack(output)
            self.times[f"{self._bucket}_gather"] = self.times.get(f"{self._bucket}_gather", 0.0) + (
                time.perf_counter() - started
            )
            return result

    DistributedDecoderLayer.__name__ = "DistributedDecoderLayer"
    return DistributedDecoderLayer


__all__ = ["PROJECTIONS", "block_range", "make_distributed_decoder_layer", "verify_shared_context"]
