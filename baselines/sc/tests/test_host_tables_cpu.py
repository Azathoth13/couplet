"""Small CPU fake verifies GPU table lifetimes and unchanged table tuples."""
import json
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baselines.sc.parallel.host_tables import HostTableSession


class HostArray:
    nbytes = 48

    def __init__(self, value):
        self.value = value.value if isinstance(value, HostArray) else value

    def copy(self):
        return HostArray(self.value)


class DeviceArray:
    live = weakref.WeakSet()
    peak = 0

    def __init__(self, value):
        self.value = HostArray(value)
        self.live.add(self)
        type(self).peak = max(type(self).peak, len(self.live))

    def cpu(self):
        return SimpleNamespace(numpy=lambda: self.value)


class Projector:
    def __init__(self, eng, geo):
        self.dev = {}

    def release(self):
        self.dev = {}


class Layer:
    built = 0
    world = 1

    def __init__(self, eng, weights, calibration, geo, projector, position, *, upload=True):
        type(self).built += 1
        self.P, self.index = projector, weights['layer']
        for name in ('qkv', 'o', 'gu', 'down'):
            tail = (float(self.index),) if self.world == 1 else (float(self.index), [0, 1])
            self.P.dev[name] = (DeviceArray(self.index), *tail)

    def forward(self, x, cache):
        assert len(self.P.dev) == 4
        for value in self.P.dev.values():
            assert value[0].value.value == self.index
            assert value[1] == float(self.index)
            if self.world > 1:
                assert value[2] == [0, 1]
        return x + self.index


class Base:
    DecodeMVLayer = Layer
    DeviceProjector = Projector

    def load(self, entry, staged=False):
        return {'layer': entry['layer']}, {'layer': entry['layer']}

    def decode_pass(self, eng, geo, entries, **kwargs):
        # Oracle's early cached_load calls must not cause GPU table uploads.
        for entry in entries:
            self.load(entry, staged=True)
        total = 0
        for entry in entries:
            weights, calibration = self.load(entry, staged=True)
            layer = self.DecodeMVLayer(eng, weights, calibration, geo,
                                       self.DeviceProjector(eng, geo), 128, upload=True)
            total = layer.forward(total, None)
            layer.P.release()
        return total


class LifecycleTests(unittest.TestCase):
    def test_w1_and_multirank_keep_only_current_gpu_layer_for_two_passes(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(empty_cache=lambda: None),
            as_tensor=lambda value, device: DeviceArray(value))
        for world in (1, 4):
            with self.subTest(world=world), tempfile.TemporaryDirectory() as directory:
                DeviceArray.live = weakref.WeakSet()
                DeviceArray.peak = 0
                Layer.built, Layer.world = 0, world
                base, eng, geo = Base(), object(), object()
                runtime = SimpleNamespace(rank=0, world_size=world,
                    all_gather=lambda tag, row: [row] * world)
                entries = [{'layer': index} for index in range(32)]
                session = HostTableSession(base, runtime, eng, geo, entries, Path(directory),
                    memory_reader=lambda: dict(used_mib=100, total_mib=100000, free_mib=99900),
                    host_reader=lambda: 1000000, sync=lambda: None)
                original_load = base.load
                original_layer = base.DecodeMVLayer
                with patch.dict(sys.modules, {'easyfhe': fake_torch}):
                    setup = session.prepare()
                    self.assertTrue(setup['passed'])
                    self.assertEqual(setup['completed_layers'], 32)
                    self.assertEqual(Layer.built, 32)
                    self.assertEqual(len(DeviceArray.live), 0)
                    self.assertTrue(all(not layer.P.dev for layer in session.layers))
                    for index in (0, 1):
                        actual = session.run_pass(base.decode_pass, eng, geo, entries,
                                                  index=index, measured=index > 0)
                        self.assertEqual(actual, sum(range(32)))
                        self.assertEqual(Layer.built, 32)  # no repeated table construction
                        self.assertEqual(len(DeviceArray.live), 0)
                        report = json.loads((Path(directory) /
                            f'decode_host_table_upload_pass{index}.json').read_text())
                        self.assertEqual([r['layer'] for r in report['layers']], list(range(32)))
                        self.assertEqual(report['total_h2d_bytes'], setup['host_table_bytes'])
                        self.assertEqual(report['includes_first_reference_setup'], index == 0)
                        self.assertEqual(base.load, original_load)
                        self.assertIs(base.DecodeMVLayer, original_layer)
                    self.assertLessEqual(DeviceArray.peak, 4)  # four current-layer plans only
                    session.release()
                    self.assertFalse(session.host_tables)
                    self.assertFalse(session.weights)


if __name__ == '__main__':
    unittest.main()
