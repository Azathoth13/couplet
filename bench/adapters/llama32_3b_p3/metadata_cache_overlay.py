"""Opt-in reuse of immutable structured linear weight metadata on one GPU.

The model-weight sources stay per layer/page and follow the original
``LinearWeight.prepare`` path.  Only the small, public index/scale vectors are
    shared when all ten vectors match byte for byte.  The cache is scoped to a
single P3 worker and never affects the default runner.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from typing import Iterator


_METADATA_FIELDS = (
    "real_source", "real_row", "real_scale", "imag_source", "imag_row",
    "imag_scale", "column", "column_scale", "real_column", "imag_column",
)


def _release_cache_value(value) -> None:
    """Match the optional-handle cleanup used by LinearWeight.close()."""

    release = getattr(value, "release", None)
    if callable(release):
        release()
        return
    clear_cache = getattr(getattr(value, "constants", None), "clear_cache", None)
    if callable(clear_cache):
        clear_cache()


def _metadata_key(weight) -> tuple:
    """Exact descriptor identity; different layer-specific scales cannot alias."""

    descriptor = weight._structured_weight
    return (
        id(weight.crypto_context),
        str(weight.crypto_context.device),
        tuple(
            (name, array.dtype.str, tuple(int(v) for v in array.shape), array.tobytes())
            for name in _METADATA_FIELDS
            if (array := getattr(descriptor, name)) is not None
        ),
    )


@contextmanager
def structured_metadata_cache(linear_weight_class=None) -> Iterator[dict]:
    """Patch one worker's ``LinearWeight`` lifetime; restore methods on exit.

    ``linear_weight_class`` permits a CPU-only fake in the unit test.  On a
    cache hit, the original ``prepare`` still resolves this page's model-weight
    source and GPU ready event.  Borrowers skip metadata release in ``close``;
    the cache owns those tensors until the worker finishes.
    """

    if linear_weight_class is None:
        from llama3fhe.operators.linear.weight import LinearWeight

        linear_weight_class = LinearWeight
    cls = linear_weight_class
    original_prepare = cls.prepare
    original_close = cls.close
    templates: dict[tuple, dict] = {}
    stats = {
        "hits": 0, "misses": 0, "entries": 0, "key_bytes": 0,
        "resident_sources": 0, "source_uploads": 0,
    }

    def record_source(self):
        name = "resident_sources" if getattr(self, "_resident_device_source", False) else "source_uploads"
        stats[name] += 1

    def prepare(self, operator_config):
        if (
            self._structured_weight is None
            or str(operator_config.raw_weight_policy) != "device"
            or not str(self.crypto_context.device).startswith("cuda")
            or self._device_structured_metadata
        ):
            return original_prepare(self, operator_config)

        key = _metadata_key(self)
        template = templates.get(key)
        if template is not None:
            # The original method still obtains the correct per-page sources.
            self._device_structured_metadata = template
            self._p3_metadata_cache_borrowed = True
            stats["hits"] += 1
            elapsed = original_prepare(self, operator_config)
            record_source(self)
            return elapsed

        elapsed = original_prepare(self, operator_config)
        record_source(self)
        if self._device_structured_metadata:
            templates[key] = self._device_structured_metadata
            self._p3_metadata_cache_borrowed = True
            stats["misses"] += 1
            stats["entries"] = len(templates)
            stats["key_bytes"] += sum(len(field[3]) for field in key[2])
        return elapsed

    def close(self):
        if getattr(self, "_p3_metadata_cache_borrowed", False):
            # Shared metadata belongs to ``templates``, not this weight.
            self._device_structured_metadata = {}
            self._p3_metadata_cache_borrowed = False
        return original_close(self)

    cls.prepare = prepare
    cls.close = close
    try:
        yield stats
    finally:
        cls.prepare = original_prepare
        cls.close = original_close
        # Mirror LinearWeight.close() for backends with explicit release().
        # Current EasyFHE tensors use Python ownership, but a cached template
        # must be released exactly once if the backend exposes a handle.
        released: set[int] = set()
        for template in templates.values():
            for value in template.values():
                if id(value) not in released:
                    _release_cache_value(value)
                    released.add(id(value))
        templates.clear()


def print_summary(stats: dict) -> None:
    print("[p3-structured-metadata-cache] " + json.dumps(stats, sort_keys=True), flush=True)
