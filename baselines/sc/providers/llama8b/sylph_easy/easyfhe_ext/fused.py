"""ctypes binding of kernels/rhombus_pcmm.cu (fused Rhombus plaintext build + NTT + grouped MAC).

The library is compiled with nvcc on first use and cached next to the source by content hash.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

import easyfhe as torch

from . import cudalib

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    lib = cudalib.load("rhombus_pcmm")
    p, i, ll, d = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_double
    lib.sylph_pcmm.argtypes = [p, p, p, ll, i, i, i, i, i, i, p, p, p, p, d, i, d, i, p, p, ll, p, p, p, p, p, p, p, i, i, i]
    lib.sylph_pcmm.restype = i
    _lib = lib
    return lib


_check = cudalib.check
CHUNK_POLYS = int(os.environ.get("SYLPH_PCMM_CHUNK", "0"))
PARTS = int(os.environ.get("SYLPH_PCMM_PARTS", "3"))      # benchmarking: 1 = build only, 2 = MAC only
PT = int(os.environ.get("SYLPH_PCMM_PT", "0"))
PCC = int(os.environ.get("SYLPH_PCMM_PCC", "8"))
# j chunks per call pipelined over two streams (build of chunk c + 1 overlaps the MAC of chunk c)
PIPE = int(os.environ.get("SYLPH_PCMM_PIPE", "0"))


_TERMS = {}

class FusedMac:
    """acc_p = sum_{c, i} sigma_{g_i}(ct_c) * z_{c, p, i} without materializing the plaintexts."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.n = int(ctx.N)
        if self.n != 1 << 16:
            raise ValueError("fused Rhombus kernels are built for N = 2^16")
        self._terms = _TERMS                                 # depends only on (autos, shift, K)

    def _terms_dev(self, plan, k):
        """Per automorphism i, the chain terms (i0, q) of kernel A (see rhombus_pcmm.cu): CSR
        offsets [A, N / 8 + 1] over slots and packed entries [A, A * K / 8] (int32, increasing i0
        within a chain), the chain m' of every slot (int16 [A, N / 8]; the slots of a group v
        hold its chains by decreasing term count) and the automorphisms (int32 [A])."""
        key = (tuple(plan["autos"]), int(plan["shift"]), int(k))
        if key not in self._terms:
            n, a, shift = self.n, plan["a"], int(plan["shift"])
            if shift % 8 or k % 8 or k > 512 or a >= 1 << 17 or any(int(g) % 256 != 1 for g in plan["autos"]):
                raise ValueError("chain terms need 8 | shift, 8 | K <= 512, i0 < 2^17 and g = 1 mod 256")
            kq = k // 8
            i0 = np.repeat(np.arange(a, dtype=np.int64), kq)
            q = np.tile(np.arange(kq, dtype=np.int64), a)
            kk = np.arange(8, dtype=np.int64)
            offs = np.empty((a, n // 8 + 1), dtype=np.int32)
            ents = np.empty((a, a * kq), dtype=np.int32)
            perm = np.empty((a, n // 8), dtype=np.int16)
            for idx, g in enumerate(plan["autos"]):
                g = int(g)
                c = (shift * i0 - 8 * q * g) % n                       # chain head C (0 mod 8)
                c8 = (c // 8 - 1) % (n // 8)                           # 8 m' + v
                chain = (c8 % 8) * 1024 + c8 // 8                      # v * 1024 + m'
                e = (shift * i0[:, None] - (8 * q[:, None] + kk[None, :]) * g) % (2 * n)
                signs = ((e >= n).astype(np.int64) << kk[None, :]).sum(axis=1)
                code = (i0 << 14) | (q << 8) | signs
                counts = np.bincount(chain, minlength=n // 8).reshape(8, 1024)
                # slot order: within each group v, chains by decreasing term count
                rank = np.argsort(-counts, axis=1, kind="stable")      # [8, 1024] slot -> m'
                perm[idx] = rank.ravel().astype(np.int16)
                slot_of = np.empty_like(rank)
                np.put_along_axis(slot_of, rank, np.arange(1024)[None, :].repeat(8, 0), axis=1)
                slot = (chain // 1024) * 1024 + slot_of.ravel()[chain]  # global slot v * 1024 + s
                order = np.argsort(slot, kind="stable")              # i0-major input keeps i0 order
                ents[idx] = code[order].astype(np.int32)
                offs[idx, 0] = 0
                offs[idx, 1:] = np.cumsum(np.take_along_axis(counts, rank, axis=1).ravel())
                if idx == 0:   # the chains partition the coefficients (bijection (chain, k) -> n)
                    heads = 8 * ((np.arange(n // 8) + 1) % (n // 8))
                    cov = (heads[:, None] - kk[None, :] * g) % n
                    assert np.array_equal(np.sort(cov.ravel()), np.arange(n))
            autos = np.asarray(plan["autos"], dtype=np.int32)
            self._terms[key] = tuple(torch.as_tensor(t, device="cuda") for t in (offs, ents, perm, autos))
        return self._terms[key]

    def mac(self, cts, dev, start, c0, plan, k, scale, sf):
        """cts: batch of chunk * A automorphic input copies (c-major); dev: [D_out, D_in] device
        weights (already / K); rows start..start+K, input blocks c0..c0+chunk."""
        ctx, n = self.ctx, self.n
        lib = _load()
        a, pc = plan["a"], plan["pc"]
        nc = int(cts.batch_size)
        chunk = nc // a
        limbs = int(cts.state.cur_limbs)
        bx, ax = cts.cv[0], cts.cv[1]
        if bx.dim() == 2:
            bx, ax = bx.unsqueeze(0), ax.unsqueeze(0)
        bx, ax = bx.contiguous(), ax.contiguous()
        if bx.stride(1) != n or ax.stride(0) != bx.stride(0):
            raise ValueError("unexpected ciphertext layout")
        if dev.dtype != torch.float64 or not dev.is_contiguous():
            raise ValueError("weights must be contiguous float64")
        row_stride = int(dev.stride(0))
        w_ptr = dev.data_ptr() + 8 * (start * row_stride + c0 * k)
        primes = ctx.QplusP_map[limbs]
        diffs = ctx.QmaxdiffplusPmaxdiff_map[limbs]
        ratios = ctx.QbarretRatioplusPbarretRatio_map[limbs]
        ks = ctx.QbarretKplusPbarretK_map[limbs]
        roots, roots_shoup = ctx.power_of_roots, ctx.power_of_roots_shoup
        stream = torch.cuda.current_stream().cuda_stream
        # chunks of pcc output groups x jc ciphertexts (CHUNK_POLYS plaintext polynomials of
        # 512 KB per chunk); PIPE >= 2 builds on a side stream into PIPE buffers
        nbuf = 1
        if CHUNK_POLYS > 0:
            pcc = min(pc, PCC)
            jc = max(1, min(nc, CHUNK_POLYS // (pcc * limbs)))
            if PIPE >= 2 and nc > jc:
                nbuf = PIPE
        else:                       # one launch pair: all plaintexts materialized
            pcc, jc = pc, nc
        inter = torch.empty((nbuf * pcc * jc, limbs, n), dtype=torch.uint64, device="cuda")
        offs, ents, perm, autos = self._terms_dev(plan, k)
        res = torch.empty((2, pc, limbs, n), dtype=torch.uint64, device="cuda")
        _check(lib.sylph_pcmm(res.data_ptr(), inter.data_ptr(), w_ptr, row_stride, pc, pcc, nc, jc, a, k,
                              offs.data_ptr(), ents.data_ptr(), perm.data_ptr(), autos.data_ptr(), float(scale), int(scale != 1.0), float(sf), limbs,
                              bx.data_ptr(), ax.data_ptr(), int(bx.stride(0)), primes.data_ptr(), diffs.data_ptr(),
                              ratios.data_ptr(), ks.data_ptr(), roots.data_ptr(), roots_shoup.data_ptr(), stream,
                              PARTS, PT, nbuf), "pcmm")
        del inter
        st = cts.state
        state = st.replace(scale_degree=st.scale_degree + 1, scaling_factor=st.scaling_factor * float(sf))
        cv = [res[0], res[1]]
        return cts.cipher_like(cv, state=state, batch_size=pc)
