"""Slot-domain helper engine on EasyFHE's fixed-scale, manual-rescale CKKS ops.

Every public method takes and returns EasyFHE ``Cipher`` objects (possibly batched).
Outputs of multiplications are rescaled lazily: inputs are normalized to scale
degree 1 before each product, so a value's depth is ``top - cur_limbs``.
"""
from __future__ import annotations

import math
import os

import numpy as np
import easyfhe as torch
from easyfhe.fhe.ciphertext import CipherState, Plaintext
from easyfhe.fhe.ops import kernels as F
from .hostcache import upload
from easyfhe.fhe.constants import encode_scalar
from easyfhe.fhe.ops import alignment, arithmetic, rotation
from .rotate import rotate as fast_rotate, rotate_hoisted
from easyfhe.fhe.ops.encoding import encode_stage1, encode_stage2
from . import slotmac

MAX_ROTATE_BATCH = 64


class Engine:
    def __init__(self, ctx, program, kit):
        self.ctx, self.program, self.kit = ctx, program, kit
        self.slots = kit.slots
        self.counts = {}
        self._pt_cache = {}

    def clear_cache(self):
        """Drop cached layer-specific plaintexts (cache keys starting with 'layer').

        Masks, RoPE tables and polynomial leaves are shared by all layers and stay cached."""
        def head(key):
            while isinstance(key, tuple) and key and isinstance(key[0], tuple):
                key = key[0]
            return key[0] if isinstance(key, tuple) and key else None
        for key in [k for k in self._pt_cache if head(k) == 'layer']:
            del self._pt_cache[key]

    # ---- bookkeeping --------------------------------------------------------
    def count(self, name, k=1):
        self.counts[name] = self.counts.get(name, 0) + k

    @staticmethod
    def limbs(ct):
        return int(ct.state.cur_limbs)

    def level(self, ct):
        """Multiplicative levels left before the S2C/refresh floor."""
        return self.limbs(ct) - (1 if ct.state.scale_degree > 1 else 0)

    def norm(self, ct):
        if ct.state.scale_degree == 1:
            return ct
        return alignment.normalize_scale(ct, self.ctx)

    def drop(self, ct, limbs):
        ct = self.norm(ct)
        if self.limbs(ct) == limbs:
            return ct
        if self.limbs(ct) < limbs:
            raise ValueError(f"cannot raise {self.limbs(ct)} limbs to {limbs}")
        return self.compact(alignment.align_to(ct, ct.state.replace(cur_limbs=limbs, scale_degree=1), self.ctx))

    @staticmethod
    def compact(ct):
        """Trim physical limbs to cur_limbs: EasyFHE's batched rescale assumes they match."""
        l = int(ct.state.cur_limbs)
        if all(x.shape[-2] == l for x in ct.cv):
            return ct
        return ct.cipher_like([x[..., :l, :].contiguous() for x in ct.cv], batch_size=ct.batch_size)

    def _pair(self, a, b):
        if a.state.scale_degree != b.state.scale_degree:
            a, b = self.norm(a), self.norm(b)
        if self.limbs(a) != self.limbs(b):
            l = min(self.limbs(a), self.limbs(b))
            if a.state.scale_degree > 1:
                a, b = self.norm(a), self.norm(b)
                l = min(self.limbs(a), self.limbs(b))
            a, b = self.drop(a, l) if self.limbs(a) != l else a, self.drop(b, l) if self.limbs(b) != l else b
        return a, b

    # ---- batches --------------------------------------------------------------
    @staticmethod
    def stack(cts):
        cts = list(cts)
        if len(cts) == 1:
            return cts[0]
        comps = []
        for i in range(len(cts[0].cv)):
            parts = [Engine.compact(c).cv[i].reshape(c.batch_size, -1, c.cv[i].shape[-1]) for c in cts]
            comps.append(torch.cat(parts, 0))
        return cts[0].cipher_like(comps, batch_size=sum(c.batch_size for c in cts))

    @staticmethod
    def unstack(ct):
        if ct.batch_size == 1:
            return [ct]
        return [ct.cipher_like([x[i] for x in ct.cv], batch_size=1) for i in range(ct.batch_size)]

    @staticmethod
    def take(ct, start, stop):
        if ct.batch_size == 1 and start == 0 and stop == 1:
            return ct
        return ct.cipher_like([x[start:stop] for x in ct.cv], batch_size=stop - start)

    def stack_aligned(self, cts):
        cts = [self.norm(c) for c in cts]
        l = min(self.limbs(c) for c in cts)
        return self.stack([self.drop(c, l) for c in cts])

    # ---- arithmetic -----------------------------------------------------------
    def add(self, a, b):
        a, b = self._pair(a, b)
        return arithmetic.homo_add(a, b, self.ctx)

    def sub(self, a, b):
        a, b = self._pair(a, b)
        return arithmetic.homo_sub(a, b, self.ctx)

    def add_many(self, cts):
        cts = list(cts)
        out = cts[0]
        for c in cts[1:]:
            out = self.add(out, c)
        return out

    def mul(self, a, b, *, rescale=False):
        """a * b (degree 2); ``rescale=True`` returns norm(a * b) through the fused
        multiply-relinearize-rescale kernel (same residues, one pass over the product)."""
        a, b = self.norm(a), self.norm(b)
        a, b = self._pair(a, b)
        if a.batch_size != b.batch_size:
            a, b = self._broadcast(a, b)
        self.count("mul", a.batch_size)
        if rescale and os.environ.get("SYLPH_MULRS", "1") != "0":
            return arithmetic.homo_mul_relin_rescale_postop(a, b, self.ctx)
        return arithmetic.homo_mul_relin(a, b, self.ctx)

    def square(self, a, *, rescale=False):
        return self.mul(a, a, rescale=rescale)

    @staticmethod
    def _broadcast(a, b):
        if a.batch_size == 1:
            a = a.cipher_like([x.reshape(1, *x.shape[-2:]).expand(b.batch_size, -1, -1).contiguous() for x in a.cv],
                              batch_size=b.batch_size)
        elif b.batch_size == 1:
            b = b.cipher_like([x.reshape(1, *x.shape[-2:]).expand(a.batch_size, -1, -1).contiguous() for x in b.cv],
                              batch_size=a.batch_size)
        else:
            raise ValueError("batch sizes differ")
        return a, b

    def encode(self, values, limbs, *, cache_key=None):
        """values: [slots] or [B, slots] real numpy -> plaintext at ``limbs`` (nominal scale)."""
        key = None if cache_key is None else (cache_key, limbs)
        if key is not None and key in self._pt_cache:
            return self._pt_cache[key]
        values = np.asarray(values, dtype=np.float64)
        if os.environ.get("SYLPH_FASTENC", "1") == "0":
            prep = encode_stage1(values, self.slots, context=self.ctx)
            pt = encode_stage2(prep, None, self.slots, False, self.ctx, cur_limbs=limbs)
        else:
            pt = self._encode_async(values, limbs)
        if key is not None:
            self._pt_cache[key] = pt
        return pt

    def _encode_async(self, values, limbs):
        """encode_stage1 + encode_stage2 (fixed scale, batch form) without the two host syncs:
        the pageable upload of the slot values and the ``max_value.cpu()`` range check."""
        ctx, slots = self.ctx, self.slots
        mat = values.reshape(1, -1) if values.ndim < 2 else values.reshape(values.shape[0], -1)
        if mat.shape[1] > slots:
            raise ValueError(f"{mat.shape[1]} values do not fit in {slots} slots")
        stage1 = np.zeros((mat.shape[0], slots), dtype=np.complex128)
        stage1[:, :mat.shape[1]] = mat
        encoded, _ = torch.fhe_pre_encode_stage1(input=upload(stage1, ctx.device), slots=slots, M=ctx.M,
                                                 rotGroup=ctx.encode_params_rotGroup,
                                                 ksiPows=ctx.encode_params_ksiPows,
                                                 bitrev=ctx.encode_bitrev_indices[int(math.log2(slots))])
        batch = int(mat.shape[0])
        sf = float(ctx.scale_at(limbs))
        body = F.cv_encode(encoded.reshape(batch, 2 * slots), ctx.N, limbs, slots, sf, False, ctx)
        return Plaintext([body], CipherState(limbs, 1, sf), slots, False, batch_size=batch)

    def cmul(self, a, values, *, cache_key=None):
        """Multiply by a public slot vector (one per batch item or shared); +1 level."""
        a = self.norm(a)
        values = np.asarray(values, dtype=np.float64)
        top = float(np.max(np.abs(values))) if values.size else 0.0
        if top > 2.0:   # encoder range: scale into [-2, 2], restore by an exact integer
            k = 2.0 ** math.ceil(math.log2(top / 2.0))
            out = self.cmul(a, values / k, cache_key=None if cache_key is None else (cache_key, 'k', k))
            return self.smul(out, k)
        pt = self.encode(values, self.limbs(a), cache_key=cache_key)
        if pt.batch_size != a.batch_size:
            if pt.batch_size != 1:
                raise ValueError("plaintext batch differs from cipher batch")
            pt = self._expand_pt(pt, a.batch_size)
        self.count("cmul", a.batch_size)
        return arithmetic.homo_mul_pt(a, pt, self.ctx)

    def cmul_sum(self, terms):
        """add_many([cmul(ct, values, cache_key=key) for ct, values, key in terms]) in one pass
        when every term shares limbs, scale and batch (SYLPH_SLOTMAC=0 keeps the sequential path)."""
        terms = [(self.norm(ct), np.asarray(v, dtype=np.float64), key) for ct, v, key in terms]
        a0 = terms[0][0]
        batch = max(ct.batch_size for ct, _, _ in terms)
        fast = slotmac.enabled() and len(terms) <= slotmac.MAX_TERMS and all(
            self.limbs(ct) == self.limbs(a0) and ct.state.scale_degree == a0.state.scale_degree
            and ct.batch_size in (1, batch) and float(np.max(np.abs(v))) <= 2.0 and v.ndim <= 2
            and (v.ndim < 2 or v.shape[0] in (1, batch)) for ct, v, _ in terms)
        if not fast:
            return self.add_many([self.cmul(ct, v, cache_key=key) for ct, v, key in terms])
        limbs = self.limbs(a0)
        cts = [ct for ct, _, _ in terms]
        pts = [self.encode(v, limbs, cache_key=key) for _, v, key in terms]
        self.count("cmul", batch * len(terms))
        return slotmac.pt_mac(cts, pts, self.ctx)

    @staticmethod
    def _expand_pt(pt, n):
        cv = [x.reshape(1, *x.shape[-2:]).expand(n, -1, -1).contiguous() for x in pt.cv]
        out = pt.__class__(cv, pt.state, pt.slots, pt.is_ext, batch_size=n)
        return out

    def smul(self, a, value):
        """Multiply by a public real scalar. Integers cost no level."""
        value = float(value)
        if value == 0:
            raise ValueError("multiply by zero")
        if value.is_integer() and abs(value) < 2 ** 30:
            s = encode_scalar(int(value), cur_limbs=self.limbs(a), scale_degree=0, context=self.ctx, mode="integer")
            return arithmetic.homo_mul_scalar(a, s, self.ctx)
        if abs(value) > 2 ** 20:   # keep the encoded fraction small; the power of two is exact
            k = 2.0 ** math.floor(math.log2(abs(value)) - 10)
            return self.smul(self.smul(a, value / k), k)
        a = self.norm(a)
        s = encode_scalar(value, cur_limbs=self.limbs(a), scale_degree=1, context=self.ctx)
        return arithmetic.homo_mul_scalar(a, s, self.ctx)

    def add_scalar(self, a, value):
        value = float(value)
        if value == 0:
            return a
        deg = int(a.state.scale_degree)
        s = encode_scalar(value, cur_limbs=self.limbs(a), scale_degree=deg, context=self.ctx,
                          **({} if deg == 1 else {"scaling_factor": a.state.scaling_factor}))
        return arithmetic.homo_add_scalar(a, s, self.ctx)

    def add_vector(self, a, values, *, cache_key=None):
        a = self.norm(a)
        values = np.asarray(values, dtype=np.float64)
        top = float(np.max(np.abs(values))) if values.size else 0.0
        if top > 2.0:   # encoder range (+-2 at the nominal scale): add the vector in pieces
            k = int(math.ceil(top / 2.0))
            part = values / k
            for i in range(k):
                a = self.add_vector(a, part, cache_key=None if cache_key is None else (cache_key, 'part', k))
            return a
        pt = self.encode(values, self.limbs(a), cache_key=cache_key)
        if pt.batch_size != a.batch_size:
            pt = self._expand_pt(pt, a.batch_size)
        return arithmetic.homo_add_pt(a, pt, self.ctx)

    # ---- rotations --------------------------------------------------------------
    def rotate(self, a, step):
        step = int(step) % self.slots
        if step == 0:
            return a
        a = self.norm(a)
        self.count("rotate", a.batch_size)
        return fast_rotate(a, step, self.ctx)

    def rotate_add(self, addend, a, step):
        """addend + rotate(a, step) with the addition folded into the key-switch epilogue
        (same modular sums as add(addend, rotate(a, step)); same batch required)."""
        step = int(step) % self.slots
        if step == 0:
            return self.add(addend, a)
        a = self.norm(a)
        addend, a = self._pair(addend, a)
        if addend.batch_size != a.batch_size or os.environ.get("SYLPH_ROTADD", "1") == "0":
            return self.add(addend, self.rotate(a, step))
        self.count("rotate", a.batch_size)
        return fast_rotate(a, step, self.ctx, addend=addend)

    def rotations(self, a, steps):
        """Hoisted rotations of one cipher (batch 1) by several steps -> list."""
        a = self.norm(a)
        steps = [int(s) % self.slots for s in steps]
        out = {}
        nz = sorted(set(s for s in steps if s))
        if nz:
            self.count("rotate_hoisted", len(nz))
            rot = []
            for i in range(0, len(nz), MAX_ROTATE_BATCH):   # key-list path: <= 64 keys
                part = rotation.fast_rotate(a, nz[i:i + MAX_ROTATE_BATCH], self.ctx)
                rot += list(part) if isinstance(part, (list, tuple)) else self.unstack(part)
            out = dict(zip(nz, rot))
        return [a if s == 0 else out[s] for s in steps]

    def rotate_many(self, a, steps):
        """[rotate(a, s) for s in steps], one shared key-switch decomposition (any batch)."""
        a = self.norm(a)
        steps = [int(s) % self.slots for s in steps]
        if a.batch_size == 1:
            return self.rotations(a, steps)
        for s in set(steps) - {0}:
            self.count("rotate", a.batch_size)
        return rotate_hoisted(a, steps, self.ctx)

    def sum_slots(self, a, stride, count):
        """Sum of ``count`` rotations by multiples of ``stride`` (count a power of two)."""
        step = stride
        while step < stride * count:
            a = self.rotate_add(a, a, step)
            step *= 2
        return a

    # ---- levels and refresh -------------------------------------------------------
    @property
    def floor(self):
        """Limbs a value must keep for a later refresh or projection (S2C input)."""
        return self.kit.s2c_start

    def avail(self, ct):
        return self.limbs(ct) - self.floor - (1 if ct.state.scale_degree > 1 else 0)

    def trim(self, ct, levels):
        """Drop limbs so that at most ``levels`` levels stay above the floor: the rotations and
        products that follow then run on fewer limbs (the message is unchanged)."""
        ct = self.norm(ct)
        return self.drop(ct, self.floor + levels) if self.avail(ct) > levels else ct

    def ready(self, ct, levels, bound=1.0, *, center=None, fine=False):
        """Refresh ``ct`` when fewer than ``levels`` levels remain (TurboFHE ``_ready``)."""
        if self.avail(ct) >= levels:
            return ct
        out = self.refresh(ct, bound, center=center, fine=fine)
        if self.avail(out) < levels:
            raise ValueError(f"refresh restores {self.avail(out)} levels, {levels} needed")
        return out

    def refresh(self, ct, bound=1.0, *, center=None, fine=False):
        """Refresh every batch item; items are paired into one complex S2C + HalfBoot.

        ``bound`` is rounded up to a power of two >= 1. The 1/bound scaling runs on the
        coefficient ciphertext (no slot level) and the restore is an exact integer.
        """
        if fine:   # any power of two; bound < 1 scales up by an exact integer
            bound = float(2.0 ** math.ceil(math.log2(float(np.max(bound)))))
        else:
            bound = float(2 ** max(0, math.ceil(math.log2(max(1.0, float(np.max(bound)))))))
        if center is not None:
            ct = self.add_scalar(ct, -center)
        items = self.unstack(self.norm(ct))
        kit = self.kit
        zs = []
        for j in range(0, len(items), 2):
            z = kit.complex_pack(items[j], items[j + 1]) if j + 1 < len(items) else items[j]
            zs.append(self._drop_for_s2c(z))
        cs = kit.to_coefficients_many(zs)
        if bound < 1:     # 1 / bound is an exact integer: no level
            cs = [self.smul(c, 1.0 / bound) for c in cs]
        factor = 1.0 / bound if bound > 1 else 1.0
        pairs = len(items) // 2
        out = [x for pair in kit.halfboots(cs[:pairs], factors=[factor] * pairs) for x in pair] if pairs else []
        if len(items) % 2:
            out.append(kit.halfboot(cs[pairs], factor=factor, real_only=True)[0])
        self.count("refresh", len(cs))
        res = self.stack(out)
        if bound != 1:
            res = self.smul(res, bound)
        if center is not None:
            res = self.add_scalar(res, center)
        return res

    def _drop_for_s2c(self, z):
        z = self.norm(z)
        return self.drop(z, self.floor) if self.limbs(z) > self.floor else z

    # ---- Chebyshev polynomials --------------------------------------------------------
    def poly(self, x, coeffs, *, mask=None, mask_key=None):
        """sum_i coeffs[i] T_i(x) for x in [-1, 1]; depth ceil(log2(deg + 1)) + 1.

        ``mask`` (a public slot vector, or one per batch item) multiplies the result at no
        extra level: it is folded into every leaf combination.
        ``SYLPH_POLY_BATCH`` evaluates the series on at most that many batch items at a
        time (0 = the whole batch).  The Chebyshev basis is one ciphertext per item, so
        a wide score batch does not fit a 24 GB card.
        """
        limit = int(os.environ.get("SYLPH_POLY_BATCH", "0") or "0")
        if limit > 0 and int(x.batch_size) > limit:
            if hasattr(self.ctx, "clear_cuda_rotation_cache"):
                self.ctx.clear_cuda_rotation_cache()
            items = []
            for part in self.unstack(x):
                items.append(part.cipher_like(
                    [t.contiguous() for t in part.cv], batch_size=1))
            # The caller still holds the batched ciphertext.  Park it so a 24 GB card
            # keeps one score, not the whole batch, under the Chebyshev basis.
            x.cv = [t.detach().to("cpu") for t in x.cv]
            masks = None
            if mask is not None:
                mask_arr = np.asarray(mask)
                if mask_arr.ndim >= 2 and int(mask_arr.shape[0]) == len(items):
                    masks = [mask_arr[i] for i in range(len(items))]
                else:
                    masks = [mask_arr] * len(items)
            parts = []
            for start in range(0, len(items), limit):
                chunk = items[start:start + limit]
                parked = []
                for j, it in enumerate(items):
                    if j < start or j >= start + limit:
                        if any(t.is_cuda for t in it.cv):
                            it.cv = [t.detach().to("cpu") for t in it.cv]
                            parked.append(it)
                chunk_mask = None if masks is None else (
                    masks[start] if len(chunk) == 1 else np.stack(masks[start:start + limit]))
                # Leaf plaintexts are cached by mask_key.  A fresh key per chunk keeps
                # one chunk's rows from being reused as another's.
                chunk_key = None if mask_key is None else (mask_key, "poly-batch", start)
                try:
                    parts.append(self.poly(
                        self.stack(chunk), coeffs, mask=chunk_mask, mask_key=chunk_key))
                finally:
                    for it in parked:
                        it.cv = [t.to("cuda") for t in it.cv]
            return self.stack(parts)
        coeffs = np.asarray(coeffs, dtype=np.float64)
        deg = len(coeffs) - 1
        while deg > 0 and coeffs[deg] == 0:
            deg -= 1
        coeffs = coeffs[:deg + 1]
        m = max(1, math.ceil(math.log2(deg + 1)))
        baby = 1 << max(1, m // 2)
        T = {1: self.norm(x)}
        def t(i, levels=0):
            """T_i with at least ``levels`` levels left; basis items refresh on demand
            (|T_i| <= 1, refreshed at bound 2) so any degree fits the level budget."""
            if i not in T:
                p = 1 << (i - 1).bit_length() - 1      # largest power of two < i
                if 2 * p == i:
                    T[i] = self.add_scalar(self.smul(self.square(t(p, 1)), 2), -1.0)
                else:
                    T[i] = self.sub(self.smul(self.mul(t(p, 1), t(i - p, 1)), 2), t(2 * p - i))
                T[i] = self.norm(T[i])
            if self.avail(T[i]) < levels:
                if os.environ.get("SYLPH_POLY_BATCH"):
                    parked = [T[k] for k in list(T) if k != i and any(t.is_cuda for t in T[k].cv)]
                    for c in parked:
                        c.cv = [t.detach().to("cpu") for t in c.cv]
                    try:
                        T[i] = self.refresh(T[i], 2.0)
                    finally:
                        for c in parked:
                            c.cv = [t.to("cuda") for t in c.cv]
                else:
                    T[i] = self.refresh(T[i], 2.0)
            return T[i]
        def operand(ct, bound):
            if self.avail(ct) >= 1:
                return ct
            if not os.environ.get("SYLPH_POLY_BATCH"):
                return self.refresh(ct, 2 * bound)
            self._pt_cache.clear()
            parked = [c for c in T.values() if c is not ct and any(t.is_cuda for t in c.cv)]
            for c in parked:
                c.cv = [t.detach().to("cpu") for t in c.cv]
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            try:
                return self.refresh(ct, 2 * bound)
            finally:
                for c in parked:
                    c.cv = [t.to("cuda") for t in c.cv]
        tag = [0]
        def leaf(c):
            nz = [i for i in range(1, len(c)) if c[i] != 0]
            if not nz:
                raise ValueError("constant leaf")
            basis = self.stack_aligned_list([t(i, 1) for i in nz])
            if mask is None:
                acc = self.add_many([self.smul(b, c[i]) for i, b in zip(nz, basis)])
            else:
                terms = []
                for i, b in zip(nz, basis):
                    tag[0] += 1
                    terms.append((b, c[i] * np.asarray(mask), None if mask_key is None else (mask_key, 'leaf', tag[0])))
                acc = self.cmul_sum(terms)
            acc = self.norm(acc)
            if c[0] != 0:
                acc = self.add_scalar(acc, c[0]) if mask is None else self.add_vector(
                    acc, c[0] * np.asarray(mask),
                    cache_key=None if mask_key is None else (mask_key, 'c0', tag[0]))
            return acc
        def split(c, n):
            q = np.zeros(len(c) - n)
            q[0] = c[n]
            q[1:] = 2 * c[n + 1:]
            r = c[:n].copy()
            for j in range(1, len(c) - n):
                r[n - j] -= c[n + j]
            return q, r
        def evaluate(c):
            if len(c) <= baby:
                return leaf(c)
            n = 1 << (len(c) - 1).bit_length() - 1
            n = max(n, baby)
            q, r = split(c, n)
            if len(q) == 1:
                hi = self.smul(t(n, 1), q[0]) if mask is None else self.cmul(t(n, 1), q[0] * np.asarray(mask))
            else:
                hi = self.mul(operand(evaluate(q), float(np.abs(q).sum())), t(n, 1))
            if np.any(r[1:] != 0):
                return self.norm(self.add(self.norm(hi), evaluate(r)))
            hi = self.norm(hi)
            if r[0] != 0:
                hi = self.add_scalar(hi, r[0]) if mask is None else self.add_vector(hi, r[0] * np.asarray(mask))
            return hi
        self.count("poly", x.batch_size)
        return evaluate(coeffs)

    def stack_aligned_list(self, cts):
        cts = [self.norm(c) for c in cts]
        l = min(self.limbs(c) for c in cts)
        return [self.drop(c, l) for c in cts]
