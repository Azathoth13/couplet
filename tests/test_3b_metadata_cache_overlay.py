"""CPU-only checks for the opt-in structured metadata cache."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from bench.adapters.llama32_3b_p3.metadata_cache_overlay import _metadata_key, structured_metadata_cache


def _structured_weight_function():
    from llama3fhe.operators.linear.structured_weight import structured_linear_weight

    return structured_linear_weight


@dataclass
class _Config:
    raw_weight_policy: str = "device"


@dataclass
class _Context:
    device: str = "cuda:0"


class _Tensor:
    def __init__(self):
        self.released = False

    def release(self):
        if self.released:
            raise AssertionError("metadata released twice")
        self.released = True


class _FakeLinearWeight:
    upload_count = 0

    def __init__(self, descriptor, context):
        self._structured_weight = descriptor
        self.crypto_context = context
        self._device_structured_metadata = {}
        self.source = descriptor.sources[0]
        self.source_seen = None
        self.closed = False

    def prepare(self, config):
        self.source_seen = self.source
        if not self._device_structured_metadata:
            self._device_structured_metadata = {"real_row": _Tensor()}
            _FakeLinearWeight.upload_count += 1
        return 1.0

    def close(self):
        for value in self._device_structured_metadata.values():
            value.release()
        self.closed = True


class MetadataCacheOverlayTest(unittest.TestCase):
    def setUp(self):
        self.structured = _structured_weight_function()
        _FakeLinearWeight.upload_count = 0

    def _descriptor(self, source, *, width=8, scale=1.0):
        rows = np.arange(width, dtype=np.int64)
        return self.structured(
            (source,), dimension=width,
            real_source=np.zeros(width, dtype=np.int64),
            real_row=rows, real_scale=0.5,
            column=rows, column_scale=np.full(width, scale),
        )

    def test_full_pages_share_metadata_but_sources_are_distinct(self):
        # Matches two full 4096-wide MLP pages; only the test width is smaller.
        source0 = np.arange(64, dtype=np.float32).reshape(8, 8)
        source1 = np.arange(64, 128, dtype=np.float32).reshape(8, 8)
        context = _Context()
        first = _FakeLinearWeight(self._descriptor(source0), context)
        second = _FakeLinearWeight(self._descriptor(source1), context)
        self.assertEqual(_metadata_key(first), _metadata_key(second))
        original_prepare = _FakeLinearWeight.prepare
        original_close = _FakeLinearWeight.close
        with structured_metadata_cache(_FakeLinearWeight) as stats:
            first.prepare(_Config())
            template = first._device_structured_metadata["real_row"]
            first.close()
            self.assertFalse(template.released)
            second.prepare(_Config())
            self.assertIs(second._device_structured_metadata["real_row"], template)
            self.assertIs(second.source_seen, source1)
            second.close()
            self.assertEqual(stats["hits"], 1)
            self.assertEqual(stats["misses"], 1)
            self.assertEqual(_FakeLinearWeight.upload_count, 1)
        self.assertIs(_FakeLinearWeight.prepare, original_prepare)
        self.assertIs(_FakeLinearWeight.close, original_close)
        self.assertTrue(template.released)

    def test_changed_scale_or_context_cannot_alias(self):
        source = np.eye(8, dtype=np.float32)
        first_context = _Context()
        weights = [
            _FakeLinearWeight(self._descriptor(source), first_context),
            _FakeLinearWeight(self._descriptor(source, scale=2.0), first_context),
            _FakeLinearWeight(self._descriptor(source), _Context()),
        ]
        with structured_metadata_cache(_FakeLinearWeight) as stats:
            for weight in weights:
                weight.prepare(_Config())
                weight.close()
            self.assertEqual(stats["hits"], 0)
            self.assertEqual(stats["misses"], 3)
            self.assertEqual(_FakeLinearWeight.upload_count, 3)


if __name__ == "__main__":
    unittest.main()
