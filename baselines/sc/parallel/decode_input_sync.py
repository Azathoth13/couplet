"""Use rank 0's encrypted Decode state as the input to every SC rank.

The original linked driver encrypts the same query separately on every rank.
Sharing the first-layer ciphertext makes the initial residual stream identical.
Later inputs can either be broadcast again (the established baseline) or carried
locally from the previous layer (``SC_DECODE_INPUT_SYNC=first``).  The latter
avoids one full-ciphertext collective per layer.  It requires deterministic
local operations between the already shared projection outputs; the optional
``SC_DECODE_CHECK_STATE=1`` compares full ciphertext digests between ranks.
"""

from __future__ import annotations

import hashlib
import os
import time


def _cipher_digest(runtime, cipher, tag):
    """Exact digest of CKKS state and residues, used only by the audit path."""

    packet = runtime.pack(cipher, tag=tag, host=True)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(repr(packet.metadata.without_devices()).encode("ascii"))
    for component in packet.components:
        digest.update(memoryview(component).cast("B"))
    return digest.hexdigest()


def make_distributed_decode_layer(base_class, runtime):
    mode = os.environ.get("SC_DECODE_INPUT_SYNC", "every").lower()
    if mode not in {"every", "first"}:
        raise ValueError("SC_DECODE_INPUT_SYNC must be 'every' or 'first'")
    check_state = os.environ.get("SC_DECODE_CHECK_STATE", "0") == "1"

    class DistributedDecodeMVLayer(base_class):
        def forward(self, x, cache, probe=None):
            if runtime.world_size == 1:
                return super().forward(x, cache, probe)
            layer = int(self.cal.get("layer", -1))
            if mode == "every" or layer == 0:
                started = time.perf_counter()
                tag = f"sc_decode_mv/layer{layer}/input"
                payload = runtime.pack(
                    x, tag=tag, host=not bool(getattr(runtime.transport, "device_payloads", False))
                ) if runtime.rank == 0 else None
                shared = runtime.broadcast_packet(tag, payload)
                if runtime.rank != 0:
                    x = runtime.unpack(shared)
                self.times["input_sync"] = time.perf_counter() - started
            out = super().forward(x, cache, probe)
            if check_state:
                started = time.perf_counter()
                tag = f"sc_decode_mv/layer{layer}/state"
                fingerprint = _cipher_digest(runtime, out, tag)
                fingerprints = runtime.all_gather(tag, fingerprint)
                if any(item != fingerprint for item in fingerprints):
                    raise RuntimeError(f"Decode ciphertext differs across ranks after layer {layer}")
                self.times["state_check"] = time.perf_counter() - started
            return out

    DistributedDecodeMVLayer.__name__ = "DistributedDecodeMVLayer"
    return DistributedDecodeMVLayer


__all__ = ["make_distributed_decode_layer"]
