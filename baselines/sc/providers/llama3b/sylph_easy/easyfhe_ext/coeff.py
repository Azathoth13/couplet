"""Coefficient-domain CKKS tools on EasyFHE.

Layout used throughout (N = ring degree, K = N // TOKENS components):
coefficient i = r + K * t holds feature r of token t (t < TOKENS).
After EasyFHE's SlotsToCoeffs, coefficient i (< N/2) is Re(slot bitrev(i)) and
coefficient i + N/2 is Im(slot bitrev(i)).  A real slot ciphertext pair
(lo, hi) for tokens [0, T/2) and [T/2, T) is therefore packed as lo + i*hi.
"""
from __future__ import annotations

import math
import os

import numpy as np
import easyfhe as torch
import easyfhe.fhe as fhe
from easyfhe.fhe.ciphertext import Cipher, CipherState
from easyfhe.fhe.ops import alignment, arithmetic, rotation
from .rotate import rotate as fast_rotate
from . import bsgs, evalmod, packing
from easyfhe.fhe.ops import kernels as F
from easyfhe.fhe.ops.encoding import PreparedPlaintext, encode_stage2
from easyfhe.bs.openfhe.runtime.transforms import eval_coeffs_to_slots, eval_slots_to_coeffs
from easyfhe.fhe.constants import encode_scalar
from easyfhe.bs.openfhe.runtime import approx as bs_approx
from easyfhe.bs.openfhe.runtime.scaling import scale_after_approx, scale_to_original_message
from .automorphism import rotation_for_automorphism


def bitrev(n_bits):
    idx = np.arange(1 << n_bits)
    out = np.zeros_like(idx)
    for b in range(n_bits):
        out |= ((idx >> b) & 1) << (n_bits - 1 - b)
    return out


def packing_automorphisms(n, components):
    """Automorphisms X -> X^(1 + 2N/m) for m = components, ..., 2."""
    out = []
    m = components
    while m >= 2:
        out.append((m, 1 + 2 * n // m))
        m //= 2
    return out


def packing_rotations(n, components):
    return [rotation_for_automorphism(g, n) for _, g in packing_automorphisms(n, components)]


def slot_layout(n, tokens):
    """slot index (in a real half ciphertext) of (token t < T/2, feature r)."""
    k = n // tokens
    half = tokens // 2
    log_slots = int(math.log2(n // 2))
    rev = bitrev(log_slots)
    i = np.arange(k)[None, :] + k * np.arange(half)[:, None]
    return rev[i]  # [T/2, K]


class CoefficientKit:
    HALFBOOT_BATCH = 16     # C2S workspace ~0.2 GB per ciphertext
    def __init__(self, ctx, program, *, tokens=128):
        self.ctx = ctx
        self.program = program
        self.plan = program._runtime_plan
        self.consts = program.constants
        self.n = int(ctx.N)
        self.slots = self.n // 2
        self.tokens = tokens
        self.k = self.n // tokens
        # S2C output keeps two limbs: q0 plus the one rescale that also applies the bootstrap
        # raise correction (folded into the projection weights or the refresh's raise scalar).
        self.s2c_start = int(self.plan.level_budget[1]) + 2
        self.pack_levels = packing_automorphisms(self.n, self.k)
        self.pack_steps = [rotation_for_automorphism(g, self.n) for _, g in self.pack_levels]
        self.layout = slot_layout(self.n, tokens)
        r = np.arange(1, self.k)
        self._weight_cols = torch.as_tensor(np.concatenate([[0], 2 * (self.slots - r) + 1]), device="cuda")

    # ---- slots -> coefficients -------------------------------------------------
    def complex_pack(self, lo, hi):
        hi_i = arithmetic.homo_mul_i(hi, self.ctx)
        return arithmetic.homo_add(lo, hi_i, self.ctx)

    def to_coefficients(self, z):
        ctx = self.ctx
        z = alignment.normalize_scale(z, ctx)
        if z.state.cur_limbs < self.s2c_start:
            raise ValueError(f"S2C needs {self.s2c_start} limbs, got {z.state.cur_limbs}")
        if z.state.cur_limbs > self.s2c_start:
            z = alignment.align_to(z, z.state.replace(cur_limbs=self.s2c_start, scale_degree=1), ctx)
        c = eval_slots_to_coeffs(z, ctx, self.consts, self.plan)
        return alignment.normalize_scale(c, ctx)

    # ---- plaintext weight polynomials ---------------------------------------
    def weight_plaintexts(self, block, cur_limbs, scaling_factor=None):
        """block: [B, K] float -> plaintexts of w_b(X) = sum_r block[b, r] X^{-r} / K.

        The 1/K factor cancels the factor K introduced by subring packing, so
        the packed result keeps EasyFHE's nominal (fixed-mode) scale.
        """
        # EasyFHE's CUDA ATen subset lacks float division/reductions: prepare on host.
        block = np.asarray(block, dtype=np.float64) / self.k
        b = int(block.shape[0])
        mx = float(np.max(np.abs(block))) if b else 0.0
        dev = torch.as_tensor(np.concatenate([block[:, :1], -block[:, 1:]], axis=1), device="cuda")
        # coefficient 0 -> middle[0]; coefficient N - r (r >= 1, >= N/2) -> middle[2(N/2 - r) + 1]
        middle = torch.zeros((b, self.n), dtype=torch.float64, device="cuda")
        middle[:, self._weight_cols] = dev
        prep = PreparedPlaintext(None, self.slots, middle, mx, packed=True)
        return encode_stage2(prep, None, self.slots, False, self.ctx,
                             scaling_factor=scaling_factor, cur_limbs=cur_limbs)

    # ---- coefficient PCMM ------------------------------------------------------
    def accumulate(self, coeff_blocks, weights, *, weight_scale=None, chunk=128):
        """weights: [D_out, len(coeff_blocks) * K]; returns list of batch ciphers (chunks)."""
        ctx = self.ctx
        d_out = weights.shape[0]
        out = []
        for start in range(0, d_out, chunk):
            stop = min(d_out, start + chunk)
            acc = None
            for c, ct in enumerate(coeff_blocks):
                pts = self.weight_plaintexts(weights[start:stop, c * self.k:(c + 1) * self.k],
                                             ct.state.cur_limbs, weight_scale)
                rep = ct.cipher_like([x.expand(stop - start, -1, -1).contiguous() for x in ct.cv],
                                     batch_size=stop - start)
                prod = arithmetic.homo_mul_pt(rep, pts, ctx)
                acc = prod if acc is None else arithmetic.homo_add(acc, prod, ctx)
                del pts, rep, prod
            out.append(acc)
        return out

    # ---- subring packing -------------------------------------------------------
    def _monomial(self, ct, degree):
        res = ct.deep_copy()
        for comp in res.cv:
            F.cv_mul_by_monomial(comp, l=res.state.cur_limbs, monomialDeg=degree, context=self.ctx)
        return res

    def _rotate_add(self, b, step, a):
        return fast_rotate(b, step, self.ctx, addend=a)

    def pack(self, accs):
        """accs: batch cipher of K outputs (valid on positions = 0 mod K) -> one cipher.

        Output coefficient r + K t = K * acc_r[K t]; weight_plaintexts pre-divides by K.
        """
        ctx = self.ctx
        # rescale first: packing key switches then run on one limb fewer
        cur = alignment.normalize_scale(accs, ctx)
        assert cur.batch_size == self.k
        for (m, g), step in zip(self.pack_levels, self.pack_steps):
            h = m // 2
            if packing.enabled() and len(cur.cv) == 2:
                a, b = packing.butterfly(cur, h, ctx)
            else:
                p = cur.cipher_like([x[:h] for x in cur.cv], batch_size=h)
                q = cur.cipher_like([x[h:m] for x in cur.cv], batch_size=h)
                q = self._monomial(q, h)
                a = arithmetic.homo_add(p, q, ctx)
                b = arithmetic.homo_sub(p, q, ctx)
            cur = self._rotate_add(b, step, a)
        return cur.cipher_like(cur.cv, batch_size=1)

    def project(self, coeff_blocks, weights):
        """Coefficient PCMM + packing; returns one packed coefficient cipher per K outputs."""
        out = []
        for start in range(0, weights.shape[0], self.k):
            acc = self.accumulate(coeff_blocks, weights[start:start + self.k], chunk=self.k)[0]
            out.append(self.pack(acc))
            del acc
        return out

    # ---- batching helpers ------------------------------------------------------
    @staticmethod
    def stack(cts):
        if len(cts) == 1:
            return cts[0]
        cv = [torch.stack([c.cv[i] if c.cv[i].dim() == 2 else c.cv[i][0] for c in cts], 0)
              for i in range(len(cts[0].cv))]
        return cts[0].cipher_like(cv, batch_size=len(cts))

    @staticmethod
    def unstack(ct):
        if ct.batch_size == 1:
            return [ct]
        return [ct.cipher_like([x[i] for x in ct.cv], batch_size=1) for i in range(ct.batch_size)]

    # ---- HalfBoot: coefficients -> (real, imag) slot ciphertexts ----------------
    def raise_adjustment(self, limbs):
        """Raise correction scalar for a ``limbs``-limb coefficient ciphertext at the nominal
        scale (EasyFHE raise_ciphertext: multiply, rescale one limb, then ModRaise)."""
        ctx, program = self.ctx, self.program
        source = float(ctx.scale_at(limbs))
        target = float(ctx.scale_at(program.raise_to_limbs))
        divisor = float(ctx.physical_rescale_divisor_for_limbs(limbs, 1))
        return target / source * divisor / source * program.constants._scalar_value("correction_scale")

    def _raise(self, c, factor=1.0, folded=False):
        """ModRaise of a coefficient ciphertext. ``factor`` scales the message inside the
        correction scalar (no extra level). ``folded``: the correction was already applied by
        the producer (projection weights times ``raise_adjustment``) and rescaled away."""
        ctx = self.ctx
        consts = self.consts
        raise_to = int(self.program.raise_to_limbs)
        r = alignment.normalize_scale(c, ctx)
        if not folded:
            scalar = encode_scalar(self.raise_adjustment(r.state.cur_limbs) * factor, cur_limbs=r.state.cur_limbs,
                                   scale_degree=1, context=ctx, scaling_factor=float(ctx.scale_at(r.state.cur_limbs)))
            r = alignment.rescale(arithmetic.homo_mul_scalar(r, scalar, ctx), ctx)
        elif r.state.cur_limbs != 1:
            raise ValueError(f"folded raise expects one limb, got {r.state.cur_limbs}")
        r = r.cipher_like(r.cv, state=r.state.replace(scaling_factor=float(ctx.scale_at(raise_to))))
        r = r.cipher_like([F.cv_mod_raise(x, raise_to, ctx) for x in r.cv], state=r.state.replace(cur_limbs=raise_to))
        scalar = consts.encoded_scalars("constant_eval_mult", cur_limbs=raise_to, scale_degree=1, context=ctx,
                                        mode="scaled", scaling_factor=r.state.scaling_factor)[0]
        return alignment.normalize_scale(arithmetic.homo_mul_scalar(r, scalar, ctx), ctx)

    def to_coefficients_many(self, zs):
        """[to_coefficients(z) for z in zs]; one batched S2C when the inputs share a state.

        ``SYLPH_S2C_BATCH`` caps the batch (0 = one shot).  A 24 GB card cannot
        hold the full SlotsToCoeffs workspace of a wide refresh.
        """
        limit = int(os.environ.get("SYLPH_S2C_BATCH", "0") or "0")
        if limit > 0 and len(zs) > limit:
            out = []
            for start in range(0, len(zs), limit):
                out.extend(self.to_coefficients_many(zs[start:start + limit]))
            return out
        if not bsgs.enabled() or len(zs) == 1:
            return [self.to_coefficients(z) for z in zs]
        ctx = self.ctx
        zs = [self._s2c_input(z) for z in zs]
        if any(z.state != zs[0].state or z.batch_size != 1 for z in zs):
            return [self.to_coefficients(z) for z in zs]
        c = bsgs.slots_to_coeffs(self.stack(zs), ctx, self.consts, self.plan)
        return self.unstack(alignment.normalize_scale(c, ctx))

    def _s2c_input(self, z):
        ctx = self.ctx
        z = alignment.normalize_scale(z, ctx)
        if z.state.cur_limbs < self.s2c_start:
            raise ValueError(f"S2C needs {self.s2c_start} limbs, got {z.state.cur_limbs}")
        if z.state.cur_limbs > self.s2c_start:
            z = alignment.align_to(z, z.state.replace(cur_limbs=self.s2c_start, scale_degree=1), ctx)
        return z

    def halfboots(self, cs, *, factors=None, folded=False, real_only=False):
        """[halfboot(c, factor=f) for c, f in zip(cs, factors)]: the C2S, conjugation and EvalMod
        of all inputs run batched (same residues as one at a time)."""
        if os.environ.get("SYLPH_HALFBOOT_BATCH") and hasattr(self.ctx, "clear_cuda_rotation_cache"):
            from . import hostcache
            hostcache.clear_device_keys()
            self.ctx.clear_cuda_rotation_cache()
        factors = [1.0] * len(cs) if factors is None else list(factors)
        batch = int(os.environ.get("SYLPH_HALFBOOT_BATCH", "") or self.HALFBOOT_BATCH)
        if batch < 1:
            raise ValueError("SYLPH_HALFBOOT_BATCH must be positive")
        if len(cs) > batch:
            return [x for i in range(0, len(cs), batch)
                    for x in self.halfboots(cs[i:i + batch], factors=factors[i:i + batch], folded=folded, real_only=real_only)]
        if not bsgs.enabled() or len(cs) == 1:
            return [self.halfboot(c, factor=f, folded=folded, real_only=real_only) for c, f in zip(cs, factors)]
        ctx = self.ctx
        consts, plan = self.consts, self.plan
        raise_to = self.program.raise_to_limbs
        rs = [self._raise(c, f, folded) for c, f in zip(cs, factors)]
        if any(r.state != rs[0].state or r.batch_size != 1 for r in rs):
            return [self.halfboot(c, factor=f, folded=folded, real_only=real_only) for c, f in zip(cs, factors)]
        e = bsgs.coeffs_to_slots(self.stack(rs), ctx, consts, plan)
        conj = fast_rotate(e, 2 * self.n - 1, ctx)
        parts = [arithmetic.homo_add(e, conj, ctx)]
        if not real_only:
            parts.append(self._monomial(arithmetic.homo_sub(e, conj, ctx), 3 * int(ctx.M) // 4))
        outs = []
        for part in parts:
            part = alignment.normalize_scale(part, ctx)
            if evalmod.enabled():
                part = evalmod.eval_mod(part, ctx, consts, plan)
                part = scale_after_approx(part, ctx, consts, raise_to_limbs=raise_to)
                part = alignment.normalize_scale(part, ctx)
                outs.append(self.unstack(scale_to_original_message(part, ctx, consts)))
                continue
            done = []
            for item in self.unstack(part):     # EasyFHE's EvalMod takes one ciphertext
                item = bs_approx.eval_bootstrap_approx_mod(item, ctx, consts, plan)
                item = scale_after_approx(item, ctx, consts, raise_to_limbs=raise_to)
                item = alignment.normalize_scale(item, ctx)
                done.append(scale_to_original_message(item, ctx, consts))
            outs.append(done)
        if real_only:
            return [(re, None) for re in outs[0]]
        return list(zip(outs[0], outs[1]))

    def halfboot(self, c, *, factor=1.0, folded=False, real_only=False):
        """Coefficient ciphertext -> (real, imag) slot ciphertexts. ``real_only``: the imag
        part is not needed (single-item refresh), so its EvalMod is skipped (imag is None)."""
        # A one-ciphertext refresh is the 24 GB peak.  Drop rotation keys the caller
        # cached (Chebyshev basis) and hand the fragmented pool back to the driver
        # before CoeffToSlots allocates its workspace.  Keys reload on demand.
        if os.environ.get("SYLPH_HALFBOOT_BATCH") and hasattr(self.ctx, "clear_cuda_rotation_cache"):
            from . import hostcache
            hostcache.clear_device_keys()
            self.ctx.clear_cuda_rotation_cache()
        ctx = self.ctx
        consts, plan = self.consts, self.plan
        raise_to = self.program.raise_to_limbs
        r = self._raise(c, factor, folded)
        # _raise reloads the keys it needs and leaves the caching allocator in holes.
        # CoeffToSlots then misses a ~50 MiB key by a few dozen MiB on a 24 GB card.
        if os.environ.get("SYLPH_HALFBOOT_BATCH"):
            from . import hostcache
            hostcache.clear_device_keys()
            if hasattr(self.ctx, "clear_cuda_rotation_cache"):
                self.ctx.clear_cuda_rotation_cache()
            torch.cuda.empty_cache()
        e = eval_coeffs_to_slots(r, ctx, consts, plan)
        conj = fast_rotate(e, 2 * self.n - 1, ctx)
        real = arithmetic.homo_add(e, conj, ctx)
        parts = [real]
        if not real_only:
            parts.append(self._monomial(arithmetic.homo_sub(e, conj, ctx), 3 * int(ctx.M) // 4))
        outs = []
        for part in parts:
            part = alignment.normalize_scale(part, ctx)
            if os.environ.get("SYLPH_HALFBOOT_BATCH"):
                torch.cuda.empty_cache()
            part = bs_approx.eval_bootstrap_approx_mod(part, ctx, consts, plan)
            part = scale_after_approx(part, ctx, consts, raise_to_limbs=raise_to)
            part = alignment.normalize_scale(part, ctx)
            part = scale_to_original_message(part, ctx, consts)
            outs.append(part)
        return outs if not real_only else (outs[0], None)
