#!/usr/bin/env python3
"""Save a selected Couplet layer's existing Decode attention/MLP substage timers.

Pass normal run_benchmark.py arguments after --. This uses the actual linked
runner and frozen model profile, rather than a copied profiling fork.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=16,
                        help="zero-based target layer; use 0-15 for the 16-layer 1B model")
    args, forwarded = parser.parse_known_args()
    if args.layer < 0:
        parser.error("--layer must be nonnegative")
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    os.environ["COUPLET_PROFILE_LAYERS"] = str(args.layer)
    from run_benchmark import main as benchmark_main

    sys.argv = [str(ROOT / "run_benchmark.py"), *forwarded]
    return int(benchmark_main())


if __name__ == "__main__":
    raise SystemExit(main())
