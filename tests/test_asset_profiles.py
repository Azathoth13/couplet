"""Frozen assets have no orphan tuning candidates or unresolved profile inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llama3fhe.application.args import (
    DEFAULT_INPUT_IDS,
    JOINT_TOL_V1_RMS,
    JOINT_TOL_V1_SILU,
    JOINT_TOL_V1_SOFTMAX,
    PREFILL_256_SOFTMAX,
    PREFIX_K2_SOFTMAX,
    model_poly_paths,
)
from llama3fhe.parallel.presets import PRESETS, polynomial_table_path

ROOT = Path(__file__).resolve().parents[1]


def test_coefficient_manifest_is_exact_runtime_dependency_closure():
    manifest = json.loads((ROOT / "assets/polynomials/manifest.json").read_text())
    declared = {ROOT / name for name in manifest["files"]}
    used = {JOINT_TOL_V1_RMS, JOINT_TOL_V1_SILU, JOINT_TOL_V1_SOFTMAX,
            PREFILL_256_SOFTMAX, PREFIX_K2_SOFTMAX}
    for model in ("llama3-8b", "llama3.2-1b", "llama3.2-3b-padded"):
        used.update(model_poly_paths(model))
    for preset in PRESETS.values():
        for field in ("rmsnorm_coeffs", "silu_coeffs", "softmax_coeffs"):
            spec = getattr(preset, field)
            if spec:
                used.add(polynomial_table_path(spec))
    for model in ("1b", "3b", "8b"):
        profile = json.loads((ROOT / "configs/models" / f"{model}.json").read_text())
        for family in ("polynomials", "fixture_polynomials"):
            used.update(ROOT / path for path in profile[family].values())
    assert used == declared
    assert len(declared) == 20
    for relative, expected in manifest["files"].items():
        payload = (ROOT / relative).read_bytes()
        assert len(payload) == expected["bytes"]
        assert hashlib.sha256(payload).hexdigest() == expected["sha256"]


def test_only_referenced_canonical_input_fixture_is_shipped():
    fixtures = set((ROOT / "assets").glob("prefill*_input_ids.json"))
    assert fixtures == {DEFAULT_INPUT_IDS}
    payload = json.loads(DEFAULT_INPUT_IDS.read_text())
    tokens = payload["input_ids"] if isinstance(payload, dict) else payload
    assert len(tokens) == 128
    assert all(isinstance(token, int) for token in tokens)
