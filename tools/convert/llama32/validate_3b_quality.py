#!/usr/bin/env python3
"""Fail closed on the plaintext gates needed before a 3B ciphertext comparison.

The fitter records out-of-range layers and exact/poly terminal predictions but
does not make them process failures.  The fixture builder likewise records
range diagnostics.  This script checks those records and, after fixtures are
built, checks the actual saved trajectory and KV arrays for finite values.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    _finite_tree(data, str(path))
    return data


def _finite_tree(value: object, where: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{where}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where} is not finite: {value!r}")


def _finite_number(value: object, where: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where} is not finite: {value!r}")
    return number


def validate_fit(poly: Path, *, model: str, layers: int) -> int:
    summary = _load_json(poly / "fit_summary.json")
    table = summary.get("table")
    if not isinstance(table, list) or [row.get("layer") for row in table] != list(range(layers)):
        raise ValueError(f"fit_summary.json must describe layers 0..{layers - 1}")
    rounds = summary.get("meta", {}).get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("fit_summary.json has no measured polynomial trajectory round")
    final = rounds[-1]
    if final.get("oob_layers") != []:
        raise ValueError(f"last measured polynomial trajectory has OOB layers: {final.get('oob_layers')!r}")
    exact = int(final["exact_top1"]["token_id"])
    poly_top = int(final["poly_top1"]["token_id"])
    if poly_top != exact:
        raise ValueError(f"polynomial Prefill top-1 {poly_top} differs from exact top-1 {exact}")
    for kind in ("exact_top1", "poly_top1"):
        _finite_number(final[kind]["logit"], f"fit final {kind}.logit")
    for key in ("logit_max_abs_diff",):
        _finite_number(final[key], f"fit final {key}")
    for family in ("rmsnorm", "silu", "softmax"):
        artifact = _load_json(poly / f"{family}_poly_coeffs.json")
        if artifact.get("model") != model:
            raise ValueError(f"{family} polynomial pack was built for {artifact.get('model')!r}, expected {model!r}")
    accuracy = summary.get("meta", {}).get("softmax_accuracy")
    if not isinstance(accuracy, list) or [row.get("layer") for row in accuracy] != list(range(layers)):
        raise ValueError("softmax accuracy report is incomplete")
    for row in accuracy:
        _finite_number(row["softmax_max_abs_err"], f"layer {row['layer']} softmax error")
    return exact


def validate_fixtures(fixtures: Path, *, model: str, layers: int, fit_top1: int,
                      hidden: int, kv: int, vocab: int) -> None:
    manifest = _load_json(fixtures / "manifest.json")
    if manifest.get("model") != model or int(manifest.get("layers", -1)) != layers:
        raise ValueError("fixture model or layer count differs from the fitted model")
    if int(manifest.get("prefix_k", -1)) != 0:
        raise ValueError("matched 3B comparison requires Prefix=0 fixtures")
    rows = manifest.get("per_layer")
    if not isinstance(rows, list) or [row.get("layer") for row in rows] != list(range(layers)):
        raise ValueError("fixture per-layer diagnostics are incomplete")
    for row in rows:
        index = row["layer"]
        for key in ("max_abs", "rms", "rmsn1_oob_frac", "rmsn2_oob_frac", "silu_oob_frac"):
            value = _finite_number(row[key], f"fixture layer {index} {key}")
            if key.endswith("_oob_frac") and value != 0.0:
                raise ValueError(f"fixture layer {index} {key}={value} is out of range")
        for key in ("softmax_exp_in_range", "softmax_invsqrt_in_range"):
            if row.get(key) is not True:
                raise ValueError(f"fixture layer {index} {key} is not true")
    top = manifest.get("poly_top_k") or []
    if not top or int(top[0]["token_id"]) != fit_top1:
        raise ValueError("fixture Prefill top-1 differs from the fitter's exact/poly top-1")
    with np.load(fixtures / "hidden_trajectory.npz", allow_pickle=False) as archive:
        ids = archive["input_ids"]
        if ids.shape != (128,) or ids.dtype.kind not in "iu":
            raise ValueError("fixture input_ids must be 128 integers")
        for index in range(layers + 1):
            name = f"hidden_{index}"
            array = archive[name]
            if array.shape != (128, hidden) or not np.isfinite(array).all():
                raise ValueError(f"fixture {name} has wrong shape or nonfinite values")
        logits = archive["poly_logits"]
        if logits.shape != (vocab,) or not np.isfinite(logits).all():
            raise ValueError("fixture poly_logits has wrong shape or nonfinite values")
        if int(np.argmax(logits)) != fit_top1:
            raise ValueError("saved polynomial logits disagree with fixture top-1")
    with np.load(fixtures / "prefill_kv_poly.npz", allow_pickle=False) as archive:
        for name in ("keys", "values"):
            array = archive[name]
            if array.shape != (layers, 128, kv) or not np.isfinite(array).all():
                raise ValueError(f"fixture {name} has wrong shape or nonfinite values")
        if int(archive["token_id"]) != fit_top1:
            raise ValueError("saved polynomial KV token_id differs from the fitted top-1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poly", required=True, type=Path)
    parser.add_argument("--fixtures", type=Path, help="also validate fixture trajectory and KV")
    parser.add_argument("--model", default="llama3.2-3b-padded")
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--kv", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=128256)
    args = parser.parse_args()
    try:
        top1 = validate_fit(args.poly, model=args.model, layers=args.layers)
        if args.fixtures is not None:
            validate_fixtures(args.fixtures, model=args.model, layers=args.layers, fit_top1=top1,
                              hidden=args.hidden, kv=args.kv, vocab=args.vocab)
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        parser.exit(1, f"3B plaintext quality gate failed: {exc}\n")
    print(json.dumps({"passed": True, "model": args.model, "layers": args.layers,
                      "prefill_top1": top1, "fixtures_checked": args.fixtures is not None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
