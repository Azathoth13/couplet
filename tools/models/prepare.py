#!/usr/bin/env python3
"""Prepare model assets in a relocatable data directory.

The 8B historical QuaRot converter uses standard CUDA PyTorch. The other
preparation mathematics uses NumPy, but fixture/SC generation and the 1B
conversion verifier import the EasyFHE-backed package: select the FHE Python
for those stages. No encrypted benchmark is started by this command.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("1b", "3b", "8b"))
    parser.add_argument("--stage", required=True, choices=("download", "convert", "fixtures", "client-rows", "linear", "sc", "all"))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--python", default=sys.executable,
                        help="preparation interpreter; all/fixtures/linear/sc and verified 1B conversion need EasyFHE imports")
    parser.add_argument("--rotate-python", default=None, help="8B standard PyTorch interpreter; not EasyFHE's torch shim")
    parser.add_argument("--download-source", choices=("official", "frozen-mirror"), default="official")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    profile = json.loads((ROOT / "configs/models" / f"{args.model}.json").read_text())
    data = args.data_root.expanduser().resolve()
    raw = (args.raw or data / profile["paths"]["raw"]).expanduser().resolve()
    checkpoint = (args.checkpoint or data / profile["paths"]["checkpoint"]).expanduser().resolve()
    fixtures = data / profile["paths"]["fixtures"]
    poly_kv = data / profile["paths"]["poly_kv"]
    chain = data / profile["paths"]["sc_chain"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    commands: dict[str, list[str]] = {}
    commands["download"] = [args.python, str(ROOT / "tools/models/download.py"),
                            "--model", args.model, "--output", str(raw), "--source", args.download_source]
    if args.model == "3b":
        commands["convert"] = [args.python, str(ROOT / "tools/convert/llama32/prepare_llama32_3b_padded.py"),
                               str(raw), str(checkpoint), "--seed", "0", "--dtype", "float32", "--verify-tokens", "2"]
    elif args.model == "1b":
        commands["convert"] = [args.python, str(ROOT / "tools/convert/llama32/quarot_rotate_numpy.py"),
                               str(raw), str(checkpoint), "--seed", "0", "--dtype", "float32",
                               "--no-embed-mean", "--verify-tokens", "2"]
    else:
        commands["convert"] = ["bash", str(ROOT / "tools/convert/prepare_quarot_weights.sh"), str(raw)]
        env["QUAROT_OUT_DIR"] = str(checkpoint)
        env["QUAROT_FP16_DIR"] = str(checkpoint.with_name(checkpoint.name + "-fp16"))
        env["ROTATE_PYTHON"] = args.rotate_python or args.python
        env["EXPORT_PYTHON"] = env["ROTATE_PYTHON"]
        env["ODIN_CONVERT_ARTIFACTS_DIR"] = str(data / "conversion/8b/artifacts")
        env["ODIN_CONVERT_LOGS_DIR"] = str(data / "conversion/8b/logs")
    commands["fixtures"] = [args.python, str(ROOT / "bench/make_model_fixtures.py"),
                             str(ROOT), str(fixtures), "--model", profile["model"],
                             "--model-dir", str(checkpoint), "--layers", str(profile["num_layers"]),
                             "--poly-kv-out", str(poly_kv)]
    for family, flag in (("rmsnorm", "--poly-rms"), ("silu", "--poly-silu"), ("softmax", "--poly-softmax")):
        commands["fixtures"] += [flag, str(ROOT / profile["fixture_polynomials"][family])]
    commands["client-rows"] = [args.python, str(ROOT / "tools/models/extract_client_rows.py"),
                               "--checkpoint", str(checkpoint), "--fixtures", str(fixtures),
                               "--output", str(data / profile["paths"]["client_embedding_rows"])]
    commands["linear"] = [args.python, str(ROOT / "tools/models/regenerate_linear_assets.py"),
                           "--odin-root", str(ROOT), "--model-dir", str(checkpoint),
                           "--fixture-dir", str(fixtures), "--sc-chain", str(chain),
                           "--layers", str(profile["num_layers"])]
    if args.model == "8b":
        # SC8B uses an additional residual and per-head rotation. Its own asset
        # builder performs that transform; raw Couplet matrices are not SC weights.
        commands["linear"].append("--skip-sc")
    commands["sc"] = [args.python, str(ROOT / "baselines/sc/install_assets.py"), "--model", args.model,
                       "--data-root", str(data), "--checkpoint", str(checkpoint)]
    stages = list(commands) if args.stage == "all" else [args.stage]
    for stage in stages:
        command = commands[stage]
        print(f"{stage}: {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
