#!/usr/bin/env python3
"""Launch Couplet or SC Llama profiles; dry runs require only Python."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_plan(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    if args.scheme == "sc":
        command = [sys.executable, str(ROOT / "baselines/sc/run.py"),
                   "--model", args.model, "--phase", args.phase,
                   "--world-size", str(args.world_size),
                   "--assets-root", str(args.data_root.resolve()),
                   "--output", str(args.output.resolve())]
        if args.dry_run:
            command.append("--dry-run")
        return command, {}

    profile_path = ROOT / "configs/models" / f"{args.model}.json"
    profile = json.loads(profile_path.read_text())
    odin = profile["odin"]
    paths = profile["paths"]
    data = args.data_root.resolve()

    def location(name: str) -> str:
        overrides = {"checkpoint": "model_dir", "fixtures": "fixtures", "poly_kv": "poly_kv"}
        override = getattr(args, overrides.get(name, name), None)
        if override is not None:
            return str(override.resolve())
        return str((data / paths[name]).resolve())

    environment = {
        name: str(value).replace("{repo}", str(ROOT)).replace("{data}", str(data))
        for name, value in odin["environment"].items()
    }
    environment.setdefault("COUPLET_KERNEL_ROOT", str(ROOT / "llama3fhe/acceleration/fhe_ext"))
    environment["COUPLET_CUDA_ARCH"] = (
        args.cuda_arch or os.environ.get("COUPLET_CUDA_ARCH")
        or os.environ.get("SYLPH_CUDA_ARCH", "sm_120")
    )
    if os.environ.get("COUPLET_PROFILE_LAYERS"):
        environment["COUPLET_PROFILE_LAYERS"] = os.environ["COUPLET_PROFILE_LAYERS"]
    command = [sys.executable, str(ROOT / odin["entrypoint"]),
               "--model", profile["model"], "--world-size", str(args.world_size),
               "--fixtures", location("fixtures"),
               "--model-dir", location("checkpoint"),
               "--poly-kv", location("poly_kv"), "--layers", "all",
               "--decode-tokens", "0" if args.phase == "prefill" else str(args.decode_tokens),
               "--output", str(args.output.resolve())]
    command.extend(odin.get("arguments", []))
    command.extend(odin.get("world_arguments", {}).get(str(args.world_size), []))
    if paths.get("client_embedding_rows") and (args.decode_tokens == 1 or args.hot_replay):
        command.extend(["--client-embedding-rows", location("client_embedding_rows")])
    if args.devices:
        command.extend(["--devices", *map(str, args.devices)])
    if args.hot_replay:
        command.append("--replay-decode-same-token")
    return command, environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("1b", "3b", "8b"), required=True)
    parser.add_argument("--scheme", choices=("couplet", "odin", "sc"), required=True,
                        help="couplet is the public name; odin remains a compatibility alias")
    parser.add_argument("--phase", choices=("prefill", "decode", "linked"), default="linked",
                        help="Decode also runs Prefill to obtain the actual ciphertext KV cache")
    parser.add_argument("--world-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, help="override the converted Couplet checkpoint")
    parser.add_argument("--fixtures", type=Path, help="override the Couplet fixture directory")
    parser.add_argument("--poly-kv", type=Path, help="override Couplet's polynomial check/reference file")
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--hot-replay", action="store_true",
                        help="Couplet: replay the same query and original KV; requires at least 2 calls")
    parser.add_argument("--cuda-arch", help="CUDA extension target (default COUPLET_CUDA_ARCH or sm_120)")
    parser.add_argument("--devices", type=int, nargs="+", help="physical GPU IDs (Couplet)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    public_scheme = "sc" if args.scheme == "sc" else "couplet"
    if args.scheme == "couplet":
        args.scheme = "odin"
    if args.decode_tokens < 1:
        parser.error("--decode-tokens must be positive; use --phase prefill for no Decode")
    if args.scheme == "sc" and args.devices:
        parser.error("For SC select physical GPUs with CUDA_VISIBLE_DEVICES")
    if args.hot_replay and (args.scheme != "odin" or args.decode_tokens < 2 or args.phase == "prefill"):
        parser.error("--hot-replay requires Couplet Decode and --decode-tokens >= 2")
    if args.devices and len(args.devices) != args.world_size:
        parser.error("Provide one physical GPU ID per rank")
    command, updates = build_plan(args)
    print(json.dumps({"scheme": public_scheme, "implementation": args.scheme, "model": args.model,
                      "phase": args.phase, "world_size": args.world_size,
                      "environment": updates, "command": command,
                      "shell_command": shlex.join(command)}, indent=2), flush=True)
    if args.dry_run and args.scheme == "odin":
        return 0
    environment = os.environ.copy()
    if args.scheme == "sc":
        environment["SYLPH_CUDA_ARCH"] = (
            args.cuda_arch or os.environ.get("COUPLET_CUDA_ARCH")
            or os.environ.get("SYLPH_CUDA_ARCH", "sm_120")
        )
    if args.scheme == "odin":
        # Campaign flags are intentionally isolated from an earlier experiment's
        # shell. Backend/source/compiler and ordinary system variables are retained.
        for name in list(environment):
            if name.startswith(("ODIN_", "T2_", "LLAMA_FHE_", "SYLPH_")):
                del environment[name]
        environment.update(updates)
        environment.setdefault("OMP_NUM_THREADS", "4")
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return subprocess.call(command, cwd=ROOT, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
