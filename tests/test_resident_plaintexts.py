"""Cross-token linear plaintext store (``LLAMA_FHE_LINEAR_PLAIN_RESIDENT``)."""
from __future__ import annotations

import numpy as np
import pytest

from llama3fhe.operators.linear.resident_plaintexts import RESIDENT, ResidentPlaintexts, weight_fingerprint
from llama3fhe.operators.linear.structured_weight import structured_linear_weight


class _Tensor:
    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n

    def element_size(self):
        return 1


def test_fingerprint_is_content_based_and_cheap_on_views() -> None:
    rng = np.random.default_rng(1)
    root = rng.normal(size=(512, 64)).astype(np.float32)
    page = root[128:256]
    same = np.array(page)  # a copy with the same values
    assert weight_fingerprint(page) == weight_fingerprint(same)
    other = root[256:384]
    assert weight_fingerprint(page) != weight_fingerprint(other)
    # a strided column view (the ``down[:, a:b]`` pages) hashes without materialising the matrix
    column_view = root[:, 16:48]
    assert weight_fingerprint(column_view) == weight_fingerprint(np.ascontiguousarray(column_view))


def test_structured_fingerprint_sees_sources_indices_and_scales() -> None:
    rng = np.random.default_rng(2)
    gate = rng.normal(size=(32, 64)).astype(np.float32)
    up = rng.normal(size=(32, 64)).astype(np.float32)
    rows = np.arange(64)
    active = rows < 32
    kwargs = dict(dimension=64, real_source=np.where(active, 0, -1), real_row=np.where(active, rows, -1),
                  real_scale=0.25, imag_source=np.where(active, 1, -1), imag_row=np.where(active, rows, -1),
                  imag_scale=0.25)
    a = structured_linear_weight((gate, up), **kwargs)
    b = structured_linear_weight((gate, up), **kwargs)
    assert weight_fingerprint(a) == weight_fingerprint(b)
    c = structured_linear_weight((gate, up), **{**kwargs, "real_scale": 0.5})
    assert weight_fingerprint(a) != weight_fingerprint(c)
    d = structured_linear_weight((up, gate), **kwargs)
    assert weight_fingerprint(a) != weight_fingerprint(d)


def test_store_needs_env_and_scope_and_honours_budget(monkeypatch) -> None:
    store = ResidentPlaintexts()
    monkeypatch.delenv("LLAMA_FHE_LINEAR_PLAIN_RESIDENT", raising=False)
    assert not store.active()
    monkeypatch.setenv("LLAMA_FHE_LINEAR_PLAIN_RESIDENT", "1")
    monkeypatch.setenv("LLAMA_FHE_LINEAR_PLAIN_RESIDENT_GIB", str(3 / (1 << 30)))  # 3 bytes
    assert store.enabled() and not store.active()
    with store.scope("decode/L10/mlp"):
        assert store.active()
        key_a = ("decode/L10/mlp", "fp", (64,), 0, 4, 2.0**40, False, "device")
        plain = type("P", (), {"cv": [_Tensor(2)]})()
        assert store.get(key_a) is None
        assert store.put(key_a, plain)
        assert store.get(key_a) is plain
        assert store.peek(key_a)
        big = type("P", (), {"cv": [_Tensor(2)]})()
        assert not store.put(("decode/L10/mlp", "other", (64,), 0, 4, 2.0**40, False, "device"), big)
    info = store.info()
    assert info["entries"] == 1 and info["bytes"] == 2 and info["hits"] == 1 and info["misses"] == 1
    assert info["skipped"] == 1
    assert not store.active()  # scope closed
    store.clear()
    assert store.info()["entries"] == 0


@pytest.mark.gpu
def test_linear_operator_reuses_resident_plaintexts_across_calls(monkeypatch) -> None:
    from llama3fhe.backend import create_sim_context
    from llama3fhe.config import SimulatorConfig
    from llama3fhe.operators.linear import LinearOperator, LinearOperatorConfig, linear_rotations

    monkeypatch.setenv("LLAMA_FHE_LINEAR_PLAIN_RESIDENT", "1")
    monkeypatch.setenv("LLAMA_FHE_LINEAR_PLAIN_RESIDENT_GIB", "1")
    RESIDENT.clear()
    RESIDENT.reset_counters()
    config = LinearOperatorConfig(baby_steps=8, baby_anchor_step=-1, max_plaintext_rows=64, raw_weight_policy="device")
    simulator = SimulatorConfig(logN=14, maxLevelsRemaining=3, dnum=1, dcrtBits=59, firstMod=60, device="cuda")
    context = create_sim_context(
        simulator, input_dim=64, block_out=128,
        rotations=linear_rotations(dimension=64, token_lanes=128, operator_config=config),
        include_default_rotations=False,
    )
    rng = np.random.default_rng(26081431)
    values = rng.normal(scale=0.02, size=(128, 64))
    weight = rng.normal(scale=0.02, size=(64, 64))
    cipher = context.encrypt(values.T.reshape(-1), level=0, slots=8192)
    expected = values @ weight.T
    with LinearOperator(crypto_context=context, token_lanes=128, config=config) as operator:
        # no scope: the store stays out of the way
        out0, prof0 = operator.project((cipher,), weight, dtype=np.float64)
        assert RESIDENT.info()["entries"] == 0 and "resident_chunks" not in prof0
        with RESIDENT.scope("test/L0/linear"):
            out1, prof1 = operator.project((cipher,), weight, dtype=np.float64)  # transient weight #1: misses
            info1 = RESIDENT.info()
            out2, prof2 = operator.project((cipher,), weight, dtype=np.float64)  # transient weight #2: hits
            info2 = RESIDENT.info()
    assert info1["entries"] >= 1 and info1["hits"] == 0
    assert info2["entries"] == info1["entries"] and info2["hits"] == info1["entries"]
    assert prof2.get("resident_chunks") == float(info1["entries"]) and prof2["weight_upload"] == 0.0
    for output in (out0[0], out1[0], out2[0]):
        actual = np.asarray(context.decrypt(output)).real.reshape(64, 128).T
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-8)
    RESIDENT.clear()
