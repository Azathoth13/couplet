from __future__ import annotations

"""Cross-token cache of encoded linear weight plaintexts (``LLAMA_FHE_LINEAR_PLAIN_RESIDENT=1``).

Decode projections can reuse encoded weight plaintexts across tokens because
weights and the level schedule are fixed. Entries are keyed by weight, plan,
chunk, limb count and scale, and are retained on the selected device.

:class:`ResidentPlaintexts` is that process-wide store.  It is only consulted inside a *scope* (the
Decode station wrapper opens ``RESIDENT.scope("decode/L{idx}/{station}")``; the Prefill path opens
none and is untouched) and keys an entry by

  ``(scope, weight fingerprint, plan.cache_key, chunk.index, cur_limbs, scaling_factor, is_ext,
  raw_weight_policy)``

where the fingerprint hashes the weight *descriptor*: for a dense page the shape / dtype and a strided
sample grid of up to 256 x 256 values; for a :class:`~.structured_weight.StructuredLinearWeight` the
full index / scale vectors plus the same sample grid of every source matrix.  Within one scope (one
layer, one station) a weight is constant across tokens, so the scope + role (the sample) identify it;
two different matrices of the same station cannot share a sample.  A miss encodes as before and stores
the result while the byte budget (``LLAMA_FHE_LINEAR_PLAIN_RESIDENT_GIB``, default 16) allows; a full
store keeps what it has and lets later layers encode per token (the W=1 32-layer case: only the first
layers fit).  Hits skip the pack, both encode stages and the ``ODIN_FAST_ENCODE`` range check (the
stored plaintext passed it when it was made).
"""

import hashlib
import os
import threading
from contextlib import contextmanager

import numpy as np

_ENV = "LLAMA_FHE_LINEAR_PLAIN_RESIDENT"
_ENV_GIB = "LLAMA_FHE_LINEAR_PLAIN_RESIDENT_GIB"
_SAMPLE = 256  # sample grid per axis


def _sample_bytes(array: np.ndarray) -> bytes:
    """Strided sample grid (<= _SAMPLE x _SAMPLE values) of a 2-D array, without copying the matrix."""

    array = np.asarray(array)
    if array.ndim != 2 or array.size == 0:
        return np.ascontiguousarray(array).tobytes()
    rows = max(1, -(-int(array.shape[0]) // _SAMPLE))
    cols = max(1, -(-int(array.shape[1]) // _SAMPLE))
    return np.ascontiguousarray(array[::rows, ::cols]).tobytes()


def weight_fingerprint(weight) -> str:
    """Hex digest identifying a dense page or a structured descriptor (see the module docstring)."""

    from .structured_weight import StructuredLinearWeight

    digest = hashlib.blake2b(digest_size=16)
    if isinstance(weight, StructuredLinearWeight):
        digest.update(b"structured")
        digest.update(str(int(weight.dimension)).encode())
        for source in weight.sources:
            source = np.asarray(source)
            digest.update(str((tuple(int(v) for v in source.shape), str(source.dtype))).encode())
            digest.update(_sample_bytes(source))
        for name in ("real_source", "real_row", "real_scale", "imag_source", "imag_row", "imag_scale",
                     "column", "column_scale", "real_column", "imag_column"):
            value = getattr(weight, name, None)
            digest.update(name.encode())
            if value is None:
                digest.update(b"-")
            else:
                digest.update(np.ascontiguousarray(np.asarray(value)).tobytes())
    else:
        array = np.asarray(weight)
        digest.update(b"dense")
        digest.update(str((tuple(int(v) for v in array.shape), str(array.dtype))).encode())
        digest.update(_sample_bytes(array))
    return digest.hexdigest()


def _plaintext_nbytes(plaintext) -> int:
    total = 0
    for component in getattr(plaintext, "cv", ()):
        numel = getattr(component, "numel", None)
        size = getattr(component, "element_size", None)
        if callable(numel) and callable(size):
            total += int(numel()) * int(size())
    return total


class ResidentPlaintexts:
    """Process-wide ``key -> encoded plaintext`` store with a byte budget (see module docstring)."""

    def __init__(self) -> None:
        self._entries: dict[tuple, tuple[object, int]] = {}
        self._bytes = 0
        self._lock = threading.Lock()
        self._scope = threading.local()
        self.hits = 0
        self.misses = 0
        self.skipped = 0  # misses whose plaintext did not fit the budget

    # -- configuration ------------------------------------------------------
    @staticmethod
    def enabled() -> bool:
        return os.environ.get(_ENV, "0") not in ("", "0", "false", "no")

    @staticmethod
    def budget_bytes() -> int:
        raw = os.environ.get(_ENV_GIB, "16").strip()
        try:
            return int(float(raw) * (1 << 30))
        except ValueError:
            return 16 << 30

    @property
    def current_scope(self) -> str | None:
        return getattr(self._scope, "name", None)

    def active(self) -> bool:
        return self.enabled() and self.current_scope is not None

    @contextmanager
    def scope(self, name: str):
        """Run a block whose linear weights may be stored / looked up under ``name``."""

        previous = self.current_scope
        self._scope.name = str(name)
        try:
            yield self
        finally:
            self._scope.name = previous

    # -- store --------------------------------------------------------------
    def key(self, fingerprint: str, plan, chunk, *, cur_limbs: int, scaling_factor, is_ext: bool,
            operator_config) -> tuple:
        return (
            self.current_scope,
            str(fingerprint),
            tuple(plan.cache_key),
            int(chunk.index),
            int(cur_limbs),
            float(scaling_factor),
            bool(is_ext),
            str(operator_config.raw_weight_policy),
        )

    def peek(self, key: tuple) -> bool:
        with self._lock:
            return key in self._entries

    def get(self, key: tuple):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
            return entry[0]

    def put(self, key: tuple, plaintext) -> bool:
        nbytes = _plaintext_nbytes(plaintext)
        with self._lock:
            if key in self._entries:
                return True
            if self._bytes + nbytes > self.budget_bytes():
                self.skipped += 1
                return False
            self._entries[key] = (plaintext, nbytes)
            self._bytes += nbytes
            return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def info(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": self.enabled(),
                "entries": len(self._entries),
                "bytes": int(self._bytes),
                "gib": round(self._bytes / (1 << 30), 3),
                "budget_gib": round(self.budget_bytes() / (1 << 30), 3),
                "hits": int(self.hits),
                "misses": int(self.misses),
                "skipped": int(self.skipped),
            }

    def reset_counters(self) -> None:
        with self._lock:
            self.hits = self.misses = self.skipped = 0


#: The process-wide store consulted by :meth:`LinearWeight.plaintext`.
RESIDENT = ResidentPlaintexts()
