"""Compile a CUDA source under kernels/ with nvcc on first use; cached by content hash."""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

_DIR = os.path.join(os.path.dirname(__file__), "kernels")


def load(name):
    src = os.path.join(_DIR, f"{name}.cu")
    with open(src, "rb") as f:
        tag = hashlib.sha1(f.read()).hexdigest()[:12]
    so = os.path.join(_DIR, f"lib{name}_{tag}.so")
    if not os.path.exists(so):
        nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
        arch = os.environ.get("COUPLET_CUDA_ARCH", os.environ.get("SYLPH_CUDA_ARCH", "sm_120"))
        tmp = so + f".{os.getpid()}.tmp"
        subprocess.run([nvcc, "-O3", f"-arch={arch}", "-shared", "-Xcompiler", "-fPIC", "-o", tmp, src], check=True)
        os.replace(tmp, so)
    return ctypes.CDLL(so)


def check(code, name):
    if code:
        raise RuntimeError(f"{name} failed with CUDA error {code}")
