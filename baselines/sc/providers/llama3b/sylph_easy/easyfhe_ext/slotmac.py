"""sum_t ct_t * pt_t in one kernel pass (kernels/slotmac.cu).

The slot engine's ``add_many([cmul(ct, v) ...])`` runs one homo_mul_pt per term and one
homo_add between terms, each a full read and write of the batch; this reads every operand once
and writes the result once. Residues are identical to the sequential path (same products and
sums mod q, fully reduced).
"""
from __future__ import annotations

import ctypes
import os

import easyfhe as torch
from easyfhe.fhe.ciphertext import CipherState

from . import cudalib

MAX_TERMS = 32
_lib = None


class _Args(ctypes.Structure):
    _fields_ = [("c0", ctypes.c_longlong * MAX_TERMS), ("c1", ctypes.c_longlong * MAX_TERMS),
                ("pt", ctypes.c_longlong * MAX_TERMS), ("cbs", ctypes.c_longlong * MAX_TERMS),
                ("pbs", ctypes.c_longlong * MAX_TERMS), ("n", ctypes.c_int)]


def enabled():
    return os.environ.get("SYLPH_SLOTMAC", "1") != "0"


def _load():
    global _lib
    if _lib is None:
        lib = cudalib.load("slotmac")
        p, i, ll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
        lib.sylph_pt_mac.argtypes = [p, p, ll, _Args, i, i, p, p, p, p]
        lib.sylph_pt_mac.restype = i
        _lib = lib
    return _lib


def _plane(t, limbs, n, batch):
    """(pointer, batch stride) of a [1 or batch, >= limbs, N] contiguous-plane tensor."""
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.dim() != 3 or t.size(1) < limbs or t.size(2) != n or t.stride(2) != 1 or t.stride(1) != n:
        raise ValueError(f"unexpected component layout {tuple(t.shape)} / {tuple(t.stride())}")
    if t.size(0) == batch:
        return t.data_ptr(), int(t.stride(0))
    if t.size(0) == 1:
        return t.data_ptr(), 0
    raise ValueError("batch mismatch")


def pt_mac(cts, pts, ctx):
    """Ciphertexts and plaintexts of equal cur_limbs and scale; batches 1 or B. Returns a Cipher."""
    if not 1 <= len(cts) <= MAX_TERMS or len(cts) != len(pts):
        raise ValueError("term count")
    n = int(ctx.N)
    st = cts[0].state
    limbs = int(st.cur_limbs)
    batch = max(c.batch_size for c in cts)
    args = _Args()
    for t, (c, p) in enumerate(zip(cts, pts)):
        if int(c.state.cur_limbs) != limbs or c.state.scale_degree != st.scale_degree\
                or c.state.scaling_factor != st.scaling_factor or int(p.state.cur_limbs) != limbs\
                or p.state.scale_degree != pts[0].state.scale_degree or len(c.cv) != 2:
            raise ValueError("terms must share limbs and scale")
        a0, s0 = _plane(c.cv[0], limbs, n, batch)
        a1, s1 = _plane(c.cv[1], limbs, n, batch)
        if s0 != s1:
            raise ValueError("components must share batch layout")
        ap, sp = _plane(p.cv[0], limbs, n, batch)
        args.c0[t], args.c1[t], args.pt[t], args.cbs[t], args.pbs[t] = a0, a1, ap, s0, sp
    args.n = len(cts)
    out = torch.empty((2, batch, limbs, n), dtype=torch.uint64, device=cts[0].cv[0].device)
    cudalib.check(_load().sylph_pt_mac(
        out[0].data_ptr(), out[1].data_ptr(), limbs * n, args, batch, limbs, ctx.primes.data_ptr(),
        ctx.barret_ratio.data_ptr(), ctx.barret_k.data_ptr(), torch.cuda.current_stream().cuda_stream), "pt_mac")
    p0 = pts[0].state
    state = CipherState(limbs, st.scale_degree + p0.scale_degree, st.scaling_factor * p0.scaling_factor)
    return cts[0].cipher_like([out[0], out[1]], state=state, batch_size=batch)
