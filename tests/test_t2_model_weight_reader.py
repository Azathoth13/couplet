"""The T2 model runner's direct npz readers (``preadv`` and file-backed mmap)."""

from __future__ import annotations

import gc
import os
import weakref

import numpy as np
import pytest

from bench.t2_model import WEIGHT_KEYS, npz_stored_members, read_into

SHAPES = {
    "input_layernorm": (64,), "q_proj": (64, 64), "k_proj": (16, 64), "v_proj": (16, 64), "o_proj": (64, 64),
    "post_attention_layernorm": (64,), "gate_proj": (224, 64), "up_proj": (224, 64), "down_proj": (64, 224),
}


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, shape in SHAPES.items()}


def test_stored_members_read_back_identically(tmp_path, arrays):
    path = tmp_path / "layer3_weights.npz"
    np.savez(path, **arrays)
    members = npz_stored_members(path)
    assert members is not None and set(members) == set(WEIGHT_KEYS)
    fd = os.open(path, os.O_RDONLY)
    try:
        for name, (offset, shape) in members.items():
            assert shape == SHAPES[name]
            buffer = np.empty(shape, dtype=np.float32)
            read_into(fd, memoryview(buffer).cast("B"), offset)
            np.testing.assert_array_equal(buffer, arrays[name])
    finally:
        os.close(fd)


def test_compressed_or_wrong_dtype_falls_back(tmp_path, arrays):
    compressed = tmp_path / "layer4_weights.npz"
    np.savez_compressed(compressed, **arrays)
    assert npz_stored_members(compressed) is None
    half = tmp_path / "layer5_weights.npz"
    np.savez(half, **{name: (value.astype(np.float16) if name == "q_proj" else value) for name, value in arrays.items()})
    assert npz_stored_members(half) is None
    missing = tmp_path / "layer6_weights.npz"
    np.savez(missing, **{name: value for name, value in arrays.items() if name != "down_proj"})
    assert npz_stored_members(missing) is None


def test_read_into_loops_over_short_reads(tmp_path):
    path = tmp_path / "blob.bin"
    payload = np.arange(1 << 16, dtype=np.uint8)
    path.write_bytes(b"xx" + payload.tobytes())
    fd = os.open(path, os.O_RDONLY)
    try:
        out = np.empty(payload.shape, dtype=np.uint8)
        read_into(fd, memoryview(out), 2)
        np.testing.assert_array_equal(out, payload)
        with pytest.raises(OSError):
            read_into(fd, memoryview(np.empty(8, dtype=np.uint8)), 2 + payload.nbytes)  # past EOF -> short read
    finally:
        os.close(fd)


@pytest.mark.parametrize("retain", ["none", "host"])
def test_weight_prefetcher_retain_reuses_the_first_read(tmp_path, arrays, monkeypatch, retain):
    """retain="host": the second pass over a layer does not touch the disk (CPU path, npload reader)."""

    from bench.t2_model import WeightPrefetcher

    monkeypatch.setenv("T2_WEIGHT_READER", "npload")  # no pinned memory on a CPU-only host
    np.savez(tmp_path / "layer3_weights.npz", **arrays)
    prefetcher = WeightPrefetcher(tmp_path, [3], enabled=True, device=None, retain=retain)
    first = prefetcher.get(3)
    assert np.array_equal(first["q_proj"], arrays["q_proj"])
    first.release()
    (tmp_path / "layer3_weights.npz").unlink()  # a re-read would now fail
    if retain == "none":
        with pytest.raises(FileNotFoundError):
            prefetcher.get(3)
        return
    prefetcher.start(3)
    second = prefetcher.get(3)
    assert prefetcher.reuploads == 1
    assert np.array_equal(second["down_proj"], arrays["down_proj"])
    second.release()
    prefetcher.close()
    with pytest.raises(FileNotFoundError):
        prefetcher.get(3)


def test_weight_prefetcher_rejects_unknown_retain(tmp_path):
    from bench.t2_model import WeightPrefetcher

    with pytest.raises(ValueError):
        WeightPrefetcher(tmp_path, [0], retain="gpu")


def test_mmap_reader_shares_file_pages_without_changing_fixture(tmp_path, arrays, monkeypatch):
    from bench.t2_model import WeightPrefetcher

    path = tmp_path / "layer3_weights.npz"
    np.savez(path, **arrays)
    monkeypatch.setenv("T2_WEIGHT_READER", "mmap")
    first_reader = WeightPrefetcher(tmp_path, [3], device=None)
    second_reader = WeightPrefetcher(tmp_path, [3], device=None)
    first_reader.start(3)
    first = first_reader.get(3)
    second = second_reader.get(3)
    for name in WEIGHT_KEYS:
        assert isinstance(first[name], np.memmap)
        np.testing.assert_array_equal(first[name], arrays[name])
        np.testing.assert_array_equal(second[name], arrays[name])
    first["q_proj"][0, 0] = -123.0  # private COW page; neither disk nor the other mapping changes
    assert second["q_proj"][0, 0] == arrays["q_proj"][0, 0]
    with np.load(path) as fixture:
        assert fixture["q_proj"][0, 0] == arrays["q_proj"][0, 0]
    mapping = weakref.ref(first["q_proj"])
    first.release()
    second.release()
    first_reader.close()
    second_reader.close()
    del first, second
    gc.collect()
    assert mapping() is None


@pytest.mark.parametrize("invalid", ["compressed", "half"])
def test_mmap_reader_rejects_unsupported_npz(tmp_path, arrays, monkeypatch, invalid):
    from bench.t2_model import WeightPrefetcher

    path = tmp_path / "layer3_weights.npz"
    if invalid == "compressed":
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **{name: (array.astype(np.float16) if name == "q_proj" else array)
                          for name, array in arrays.items()})
    monkeypatch.setenv("T2_WEIGHT_READER", "mmap")
    with pytest.raises(ValueError, match="ZIP_STORED C-order float32"):
        WeightPrefetcher(tmp_path, [3], device=None).get(3)
