"""Coefficient PCMM with Rhombus Algorithm 2 on EasyFHE (application logic after TurboFHE
apps/sylph/rhombus.py, generalized from one token to the token-interleaved layout).

Input coefficient K t + r holds feature r of token t (K = N / tokens). For one block of K
outputs, the plain method multiplies by K weight polynomials and merges them with a K-way
PackLWEs (K - 1 key switches). Rhombus moves the first log2(A) PackLWEs levels onto the
input: A = 2**(split - h) automorphic copies of each input ciphertext (shared by every
output block and by every projection reading that input), and the plaintexts absorb the
automorphisms and monomial shifts. The output pack then needs only K / A - 1 key switches.
Plaintext count is unchanged. Output coefficient o + K t holds output o of token t.
"""
from __future__ import annotations

import math
import os

import numpy as np
import easyfhe as torch
from easyfhe.fhe.ops import alignment, arithmetic, rotation
from .rotate import rotate as fast_rotate
from easyfhe.fhe.ops.encoding import PreparedPlaintext, encode_stage2

from .coeff import rotation_for_automorphism
from . import packing
from .fused import FusedMac

# Fused plaintext build + NTT + MAC (kernels/rhombus_pcmm.cu); 0 runs encode + grouped MAC.
FUSED = os.environ.get("SYLPH_FUSED", "1") != "0"


class DeviceWeights:
    """Projection weights resident on the GPU, divided by K; ``top`` = max |weight|."""

    def __init__(self, dev, top):
        self.dev, self.top = dev, top


_PLANS = {}

class Rhombus:
    def __init__(self, kit, max_batch=64):
        self.kit, self.ctx = kit, kit.ctx
        self.n, self.k = kit.n, kit.k
        self.h = int(math.log2(self.n // self.k))
        self.max_batch = max_batch
        self._plans = _PLANS.setdefault((self.n, self.k), {})     # shared by all layers' Rhombus
        self.fused = FusedMac(self.ctx) if FUSED else None

    # ---- public plan --------------------------------------------------------------------
    def plan(self, split):
        """Gather tables and rotation steps for ``split``; depends only on (N, K, split)."""
        if split in self._plans:
            return self._plans[split]
        n, k, h = self.n, self.k, self.h
        if not h <= split <= int(math.log2(n)):
            raise ValueError("split outside [log2(N/K), log2(N)]")
        autos = [1]
        for level in range(h + 1, split + 1):
            autos += [g * ((1 << level) + 1) % (2 * n) for g in autos[:]]
        a, pc, shift = len(autos), n >> split, n >> split
        r = np.arange(k)
        i0 = np.arange(a)
        # Plaintext z_{p,i} = sum_i0 X^{shift i0} sigma_g(w_{pc i0 + p}),  w_o = sum_r W[o, r] X^{-r}.
        # Term (i0, r) lands on exponent e, PreparedPlaintext position mid(e mod N), sign -1 when
        # e >= N. Per automorphism the first term of every covered position is assigned and the
        # remaining terms are accumulated; a term's source code indexes [vals, -vals]
        # (code A*K + q is -vals[q]). Uncovered positions stay zero.
        first_cols, first_src, rest_cols, rest_src = [], [], [], []
        q = (i0[:, None] * k + r[None, :]).reshape(-1)
        for idx, g in enumerate(autos):
            e = ((-(r[None, :] * g) + shift * i0[:, None]) % (2 * n)).reshape(-1)     # [A*K]
            j = e % n
            mid = np.where(j < n // 2, 2 * j, 2 * (j - n // 2) + 1)
            code = np.where(e >= n, q + a * k, q)
            order = np.argsort(mid, kind="stable")
            ms, cs = mid[order], code[order]
            lead = np.r_[True, ms[1:] != ms[:-1]]
            first_cols.append(idx * n + ms[lead]); first_src.append(cs[lead])
            rest_cols.append(idx * n + ms[~lead]); rest_src.append(cs[~lead])
        dev = lambda v: torch.as_tensor(np.concatenate(v).astype(np.int64), device="cuda")
        gather = (dev(first_cols), dev(first_src), dev(rest_cols), dev(rest_src))
        steps = [rotation_for_automorphism((1 << level) + 1, n) for level in range(h + 1, split + 1)]
        pack = []
        s = 1
        while (1 << s) <= pc:
            l = 1 << s
            pack.append((n // (l << split), rotation_for_automorphism((l << split) + 1, n)))
            s += 1
        p = dict(split=split, autos=autos, a=a, pc=pc, shift=shift, gather=gather, input_steps=steps, pack=pack)
        self._plans[split] = p
        return p

    @staticmethod
    def best_split(h, log_k, blocks, packs):
        """split minimizing input automorphisms (blocks * (A - 1)) + output key switches."""
        cost = lambda a: blocks * ((1 << a) - 1) + packs * ((1 << (log_k - a)) - 1)
        return h + min(range(log_k + 1), key=cost)

    # ---- ciphertext side ----------------------------------------------------------------
    def _rotate(self, ct, step):
        return fast_rotate(ct, step, self.ctx)

    def inputs(self, coeff_blocks, split):
        """Per input block: a batch of A automorphic copies sigma_g(ct), g in plan order."""
        plan = self.plan(split)
        out = []
        for ct in coeff_blocks:
            cur = alignment.normalize_scale(ct, self.ctx)
            for step in plan["input_steps"]:
                cur = self.kit.stack(self.kit.unstack(cur) + self.kit.unstack(self._rotate(cur, step)))
            out.append(cur)
        return out

    def upload(self, weights, rows=None, cols=None, row_div=None):
        """Host [D_out, D_in] weights -> DeviceWeights (on the GPU, already / K), optionally
        ``(weights / row_div[:, None])[rows][:, cols]``. The permutation runs on the GPU and
        the per-row division on the host in the original row order (an elementwise op commutes
        with the permutation, so the values are bit-identical). Uses a side stream so a
        prefetching thread does not queue behind the running layer."""
        w = np.asarray(weights, dtype=np.float64)
        if row_div is not None:
            w = w / np.asarray(row_div, dtype=np.float64)[:, None]
        w = np.ascontiguousarray(w)
        top = float(np.max(np.abs(w))) if w.size else 0.0      # permutation invariant
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            dev = torch.as_tensor(w, device="cuda")
            if rows is not None:
                dev = dev[torch.as_tensor(rows, device="cuda")]
            if cols is not None:
                dev = dev[:, torch.as_tensor(cols, device="cuda")]
            dev = (dev * (1.0 / self.k)).contiguous()           # K is a power of two: exact
        stream.synchronize()
        return DeviceWeights(dev, top)

    def _plaintexts(self, block, plan, limbs, mx, buffers):
        """block: [K outputs, C * K inputs] device weights (already / K) -> pc * C * A
        plaintexts, item (p, c, i) = z_{c, p, i}. ``buffers`` holds this projection's zeroed
        PreparedPlaintext buffers by row count: every call assigns the same covered positions
        (fixed per plan) before accumulating, so the uncovered ones stay zero."""
        a, pc, k, n = plan["a"], plan["pc"], self.k, self.n
        chunk = block.shape[1] // k
        first_cols, first_src, rest_cols, rest_src = plan["gather"]
        # rows o = pc * i0 + p, columns c * K + r  ->  [p, c, i0 * K + r]
        vals = block.contiguous().reshape(a, pc, chunk, k).permute(1, 2, 0, 3).reshape(pc * chunk, a * k)
        ext = torch.cat([vals, -vals], 1)
        middle = buffers.get(pc * chunk)
        if middle is None:
            middle = buffers[pc * chunk] = torch.zeros((pc * chunk, a * n), dtype=torch.float64, device="cuda")
        middle[:, first_cols] = ext[:, first_src]
        if rest_cols.numel():
            middle.index_add_(1, rest_cols, ext[:, rest_src])
        prep = PreparedPlaintext(None, self.kit.slots, middle.reshape(pc * chunk * a, n), mx, packed=True)
        return encode_stage2(prep, None, self.kit.slots, False, self.ctx, cur_limbs=limbs)

    def _pack(self, cur, plan):
        ctx, kit = self.ctx, self.kit
        cur = alignment.normalize_scale(cur, ctx)
        for degree, step in plan["pack"]:
            half = cur.batch_size // 2
            if packing.enabled() and len(cur.cv) == 2:
                a, b = packing.butterfly(cur, half, ctx, degree)
                cur = kit._rotate_add(b, step, a)
                continue
            p = cur.cipher_like([x[:half] for x in cur.cv], batch_size=half)
            q = cur.cipher_like([x[half:] for x in cur.cv], batch_size=half)
            q = kit._monomial(q, degree)
            cur = kit._rotate_add(arithmetic.homo_sub(p, q, ctx), step, arithmetic.homo_add(p, q, ctx))
        return cur.cipher_like(cur.cv, batch_size=1)

    def project(self, inputs, weights, split, budget=3 << 30, scale=1.0):
        """inputs: from ``inputs`` (same split); weights [D_out, blocks * K] host array or
        DeviceWeights from ``upload``; the projection uses ``scale * weights``.
        Returns one packed coefficient ciphertext per K outputs (as CoefficientKit.project).

        acc_p = sum_{c, i} sigma_{g_i}(ct_c) * z_{c, p, i} runs as one grouped MAC per
        chunk of input blocks (groups = pc), so ciphertexts are never replicated."""
        plan = self.plan(split)
        ctx, k, a, pc, n = self.ctx, self.k, plan["a"], plan["pc"], self.n
        if not isinstance(weights, DeviceWeights):
            weights = self.upload(weights)
        rows, cols = weights.dev.shape
        if cols != len(inputs) * k or rows % k:
            raise ValueError("weights must be [multiple of K, blocks * K]")
        limbs = inputs[0].state.cur_limbs
        chunk = max(1, min(len(inputs), budget // (k * limbs * n * 8)))
        scale = float(scale)
        mx = float(a * weights.top * abs(scale)) / k
        dev = weights.dev
        # the fused kernels take the scaled weights (one multiply per weight here instead of
        # one per term; the same double either way)
        dev_fused = None
        if self.fused is not None and os.environ.get("SYLPH_FUSED", "1") != "0":
            dev_fused = dev if scale == 1.0 else (dev * scale).contiguous()
        groups = [self.kit.stack([x for src in inputs[c:c + chunk] for x in self.kit.unstack(src)])
                  for c in range(0, len(inputs), chunk)]
        out, buffers = [], {}
        for start in range(0, rows, k):
            acc = None
            for g, cts in enumerate(groups):
                c0 = g * chunk
                block = dev[start:start + k, c0 * k:c0 * k + cts.batch_size // a * k]
                if self.fused is not None and os.environ.get("SYLPH_FUSED", "1") != "0":
                    if mx * float(ctx.scale_at(limbs)) >= 2.0 ** 62:
                        raise ValueError("projection plaintext too large for encoding")
                    part = self.fused.mac(cts, dev_fused, start, c0, plan, k, 1.0, ctx.scale_at(limbs))
                else:
                    pts = self._plaintexts(block * scale if scale != 1.0 else block, plan, limbs, mx, buffers)
                    part = arithmetic.grouped_pairwise_mac(cts, pts, pc, ctx)
                    del pts
                acc = part if acc is None else arithmetic.homo_add(acc, part, ctx)
            out.append(self._pack(acc, plan) if pc > 1 else alignment.normalize_scale(acc, ctx))
            del acc
        del groups, buffers, dev_fused
        return out
