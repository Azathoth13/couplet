#!/usr/bin/env python3
"""Report backend prerequisites without creating a context or running GPU work."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = {"python": sys.version.split()[0], "checks": {}, "errors": []}
    result["checks"]["python_312"] = sys.version_info[:2] == (3, 12)
    nvcc_request = os.environ.get("NVCC", "nvcc")
    nvcc = shutil.which(nvcc_request) or (nvcc_request if Path(nvcc_request).is_file() else None)
    result["nvcc"] = nvcc
    result["checks"]["nvcc"] = bool(nvcc and Path(nvcc).is_file())
    result["cuda_arch"] = os.environ.get("COUPLET_CUDA_ARCH", os.environ.get("SYLPH_CUDA_ARCH", "sm_120"))
    root = Path(__file__).resolve().parents[2]
    shared = Path(os.environ.get("COUPLET_KERNEL_ROOT", root / "llama3fhe/acceleration/fhe_ext"))
    result["checks"]["shared_acceleration"] = all(
        (shared / name).is_file()
        for name in ("keyswitch.py", "evalmod.py", "hostcache.py", "kernels/keyswitch.cu")
    )
    try:
        backend = importlib.import_module("easyfhe")
        fhe = importlib.import_module("easyfhe.fhe")
        try:
            result["easyfhe"] = importlib.metadata.version("easyfhe")
        except importlib.metadata.PackageNotFoundError:
            result["easyfhe"] = getattr(backend, "__version__", "source checkout; inspect its revision")
        result["easyfhe_module"] = backend.__file__
        result["checks"]["easyfhe"] = True
        for name in ("homo_mul_i", "homo_mul_no_relin", "homo_relinearize"):
            result["checks"][name] = callable(getattr(fhe, name, None))
        result["checks"]["bootstrap_api"] = importlib.util.find_spec("easyfhe.bs.openfhe") is not None
    except (ImportError, OSError, importlib.metadata.PackageNotFoundError) as exc:
        result["checks"]["easyfhe"] = False
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["ok"] = all(result["checks"].values())
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for name, passed in result["checks"].items():
            print(f"{'OK' if passed else 'MISSING'} {name}")
        for message in result["errors"]:
            print(message)
        print("No FHE context was created. See docs/BACKEND.md for installation.")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
