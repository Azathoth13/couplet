"""Fused subring-packing butterfly (kernels/packing.cu).

One packing level turns p = items [0, h) and q = items [h, 2h) into a = p + X^h q and
b = p - X^h q. EasyFHE does this with mul_by_monomial (iNTT, shift, NTT) on a copy of q,
then homo_add and homo_sub. In the NTT domain X^h q is the pointwise product of q with
NTT(X^h), so one kernel over p and q writes a and b with the same residues.
``SYLPH_PACK=0`` disables it.
"""
from __future__ import annotations

import ctypes
import os

import easyfhe as torch
from easyfhe.fhe.ops import kernels as F

from . import cudalib

_lib = None
_mono = {}


def enabled():
    return os.environ.get("SYLPH_PACK", "1") != "0"


def _load():
    global _lib
    if _lib is None:
        lib = cudalib.load("packing")
        p, i, ll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
        lib.sylph_pack_butterfly.argtypes = [p, p, ll, p, p, p, p, p, i, i, p, p, p, p]
        lib.sylph_pack_butterfly.restype = i
        _lib = lib
    return _lib


def monomial_ntt(ctx, degree):
    """NTT(X^degree) on all L limbs: EasyFHE's mul_by_monomial applied to the constant 1,
    whose NTT is 1 at every evaluation point."""
    key = (id(ctx), int(degree))
    m = _mono.get(key)
    if m is None:
        limbs = int(ctx.L)
        m = torch.ones((1, limbs, int(ctx.N)), dtype=torch.uint64, device="cuda")
        F.cv_mul_by_monomial(m, l=limbs, monomialDeg=int(degree), context=ctx)
        m = _mono[key] = m[0].contiguous()
    return m


def butterfly(cur, h, ctx, degree=None):
    """(p + X^d q, p - X^d q) for p = cur[:h], q = cur[h:2h] (two-component batch cipher),
    d = ``degree`` (default h)."""
    limbs, n = int(cur.state.cur_limbs), int(ctx.N)
    x0, x1 = cur.cv
    for x in (x0, x1):
        if x.dim() != 3 or x.size(0) < 2 * h or x.size(1) < limbs or x.size(2) != n\
                or x.stride(2) != 1 or x.stride(1) != n:
            raise ValueError(f"unexpected component layout {tuple(x.shape)} / {tuple(x.stride())}")
    if x1.stride(0) != x0.stride(0):
        raise ValueError("components must share batch layout")
    mono = monomial_ntt(ctx, h if degree is None else degree)
    out = torch.empty((4, h, limbs, n), dtype=torch.uint64, device=x0.device)
    stream = torch.cuda.current_stream().cuda_stream
    cudalib.check(_load().sylph_pack_butterfly(
        x0.data_ptr(), x1.data_ptr(), int(x0.stride(0)), out[0].data_ptr(), out[1].data_ptr(),
        out[2].data_ptr(), out[3].data_ptr(), mono.data_ptr(), int(h), limbs,
        ctx.primes.data_ptr(), ctx.barret_ratio.data_ptr(), ctx.barret_k.data_ptr(), stream), "pack_butterfly")
    a = cur.cipher_like([out[0], out[1]], batch_size=h)
    b = cur.cipher_like([out[2], out[3]], batch_size=h)
    return a, b
