"""Single-token coefficient PCMV (Rhombus Algorithm 2 + PackRLWEs) on EasyFHE.

Port of TurboFHE ``apps/sylph/rhombus.py::RhombusPlan`` (one token, N = ring degree) with two
generalisations used by the decode layer (``layer/decode_mv.py``):

* ``in_pos[r]``: the coefficient holding input feature ``r`` (any injection into ``[0, span)``,
  ``span`` a power of two; a strided input ``2 r`` is the natural layout of the SiLU output);
* ``out_row[f]``: the plan row of output feature ``f`` (any injection into ``[0, padded_rows)``).
  Output ``f`` lands on coefficient ``out_row[f] * stride`` with ``stride = N / padded_rows``,
  so the row assignment chooses the slot layout after HalfBoot (coefficient c -> slot bitrev(c)).

Plaintext count ``m = padded_rows * span / N`` (independent of the split); ``A = 2**(split - h)``
automorphic input copies, ``pc = padded_rows >> split`` partial sums, ``pc - 1`` pack key switches.
``simulate`` runs the exact algorithm on numpy polynomials (test oracle for the device path).
"""
from __future__ import annotations

import math

import numpy as np

from .automorphism import rotation_for_automorphism


def _pow2(v):
    return v >= 1 and v & (v - 1) == 0


def _ceil_pow2(v):
    return 1 << (int(v) - 1).bit_length() if v > 1 else 1


class MVPlan:
    def __init__(self, n, rows, cols, in_pos, out_row, *, padded_rows=None, split=None):
        self.n = int(n)
        in_pos = np.asarray(in_pos, dtype=np.int64)
        out_row = np.asarray(out_row, dtype=np.int64)
        if in_pos.shape != (cols,) or len(np.unique(in_pos)) != cols or in_pos.min() < 0:
            raise ValueError("in_pos must inject the inputs into the coefficients")
        if out_row.shape != (rows,) or len(np.unique(out_row)) != rows or out_row.min() < 0:
            raise ValueError("out_row must inject the outputs into the plan rows")
        self.rows, self.cols = int(rows), int(cols)
        self.in_pos, self.out_row = in_pos, out_row
        self.span = _ceil_pow2(int(in_pos.max()) + 1)
        if self.span > self.n // 2:
            raise ValueError("input span must fit in N/2 coefficients (the input automorphisms need h >= 1)")
        natural = max(_ceil_pow2(int(out_row.max()) + 1), self.n // self.span)
        self.padded_rows = natural if padded_rows is None else int(padded_rows)
        if not _pow2(self.padded_rows) or self.padded_rows < natural or self.padded_rows > self.n:
            raise ValueError("padded_rows must be a power of two in [natural, N]")
        self.m = self.padded_rows * self.span // self.n          # plaintext products
        if self.m < 1:
            raise ValueError("padded_rows * span must reach N")
        self.h = int(math.log2(self.n // self.span))
        self.split = self.h + (self.m.bit_length() - 1) // 2 if split is None else int(split)
        if not self.h <= self.split <= int(math.log2(self.padded_rows)):
            raise ValueError("split must lie in [log2(N/span), log2(padded_rows)]")
        self.a = 1 << (self.split - self.h)
        self.pc = self.padded_rows >> self.split
        assert self.pc * self.a == self.m
        self.shift = self.n >> self.split
        self.stride = self.n // self.padded_rows
        autos = [1]
        for level in range(self.h + 1, self.split + 1):
            autos += [g * ((1 << level) + 1) % (2 * self.n) for g in autos[:]]
        self.autos = autos
        self.input_steps = [rotation_for_automorphism((1 << level) + 1, self.n)
                            for level in range(self.h + 1, self.split + 1)]
        self.pack = []
        l = 2
        while l <= self.pc:
            self.pack.append((self.n // (l << self.split), rotation_for_automorphism((l << self.split) + 1, self.n)))
            l *= 2
        self.out_coeff = self.out_row * self.stride

    def rotations(self):
        return set(self.input_steps) | {s for _, s in self.pack}

    def describe(self):
        return dict(rows=self.rows, cols=self.cols, span=self.span, padded_rows=self.padded_rows, m=self.m,
                    h=self.h, split=self.split, a=self.a, pc=self.pc, stride=self.stride)

    # ---- plaintext construction (numpy; the device path mirrors this) ------------------
    def sorted_weights(self, w):
        """[rows, cols] -> [padded_rows, cols] with row ``out_row[f]`` = w[f]."""
        w = np.asarray(w, dtype=np.float64)
        out = np.zeros((self.padded_rows, self.cols))
        out[self.out_row] = w
        return out

    def packed_weights(self, w):
        """pw[i_lo, (j span - in_pos[r]) mod N] = +-W[plan row i_lo + m j, r]."""
        ws = self.sorted_weights(w)
        n, m, span = self.n, self.m, self.span
        pw = np.zeros((m, n))
        for j in range(n // span):
            pos = j * span - self.in_pos
            sign = np.where(pos < 0, -1.0, 1.0)
            pw[:, pos % n] = ws[j * m:(j + 1) * m] * sign
        return pw

    def automorphism_tables(self, g):
        """(src, sign): sigma_g(v)[c] = sign[c] * v[src[c]] for coefficient vectors v."""
        n = self.n
        c = np.arange(n, dtype=np.int64)
        e = (c * int(g)) % (2 * n)
        dst, sign = e % n, np.where(e >= n, -1.0, 1.0)
        src = np.empty(n, dtype=np.int64)
        src[dst] = c
        return src, sign[src]

    @staticmethod
    def monomial_shift(v, s, n):
        """v * X^s on [..., n] coefficient arrays (negacyclic)."""
        s %= 2 * n
        sign = 1.0
        if s >= n:
            s, sign = s - n, -1.0
        if s == 0:
            return sign * v
        return sign * np.concatenate([-v[..., n - s:], v[..., :n - s]], axis=-1)

    def plaintexts(self, w):
        """[pc * A, N] coefficient vectors, item p * A + idx = z_{p, idx} (already / m)."""
        pw = self.packed_weights(w)
        n, a, pc = self.n, self.a, self.pc
        out = np.zeros((pc, a, n))
        for idx, g in enumerate(self.autos):
            src, sign = self.automorphism_tables(g)
            aut = pw[:, src] * sign                             # [m, N]
            blocks = aut.reshape(a, pc, n)                      # row pc * i0 + p -> [i0, p]
            for i0 in range(a):
                out[:, idx] += self.monomial_shift(blocks[i0], self.shift * i0, n)
        return out.reshape(pc * a, n) / self.m

    # ---- numpy oracle ------------------------------------------------------------------
    def simulate(self, w, x):
        """W @ x via the algorithm on plaintext polynomials; returns the packed polynomial."""
        n = self.n
        poly = np.zeros(n)
        poly[self.in_pos] = np.asarray(x, dtype=np.float64)
        copies = []
        for g in self.autos:
            src, sign = self.automorphism_tables(g)
            copies.append(poly[src] * sign)
        pts = self.plaintexts(w).reshape(self.pc, self.a, n)
        acc = [sum(_negacyclic(copies[i], pts[p, i]) for i in range(self.a)) for p in range(self.pc)]
        cur = acc
        for degree, step in self.pack:
            g = pow(5, step, 2 * n)
            src, sign = self.automorphism_tables(g)
            half = len(cur) // 2
            nxt = []
            for i in range(half):
                p, q = cur[i], self.monomial_shift(cur[i + half], degree, n)
                d = p - q
                nxt.append(p + q + d[src] * sign)
            cur = nxt
        return cur[0]

    def outputs(self, packed):
        return packed[self.out_coeff]


def _negacyclic(a, b):
    n = len(a)
    full = np.convolve(a, b)
    out = full[:n].copy()
    out[:n - 1] -= full[n:]
    return out


# ---- device evaluator ----------------------------------------------------------------------
class RhombusMV:
    """MVPlan on EasyFHE ciphertexts (coefficient domain, 2-limb inputs after S2C).

    ``build`` mirrors ``MVPlan.plaintexts`` on the GPU (weights are uploaded once per layer),
    ``encode`` turns the coefficient tensor into plaintexts at the input's limbs, ``project``
    runs the automorphic copies, the grouped MAC and the pack; the result is one packed
    coefficient ciphertext (one limb, nominal scale) ready for the folded-raise HalfBoot.
    """

    def __init__(self, kit):
        import easyfhe as torch
        from easyfhe.fhe.ops import alignment, arithmetic
        from easyfhe.fhe.ops.encoding import PreparedPlaintext, encode_stage2
        from . import packing
        self.torch, self.alignment, self.arithmetic, self.packing = torch, alignment, arithmetic, packing
        self.PreparedPlaintext, self.encode_stage2 = PreparedPlaintext, encode_stage2
        self.kit, self.ctx, self.n = kit, kit.ctx, kit.n
        j = np.arange(self.n // 2, dtype=np.int64)
        inter = np.empty(self.n, dtype=np.int64)
        inter[0::2], inter[1::2] = j, j + self.n // 2          # middle column 2j <- coeff j, 2j+1 <- coeff j + N/2
        self._inter = torch.as_tensor(inter, device="cuda")
        self.counts = {}

    def _count(self, k, v=1):
        self.counts[k] = self.counts.get(k, 0) + v

    # ---- plaintexts --------------------------------------------------------------------
    def build(self, plan, w, row_scale=None):
        """w: [rows, cols] host array (or cuda float64 tensor); ``row_scale`` [rows] host factors.
        Returns (middle [m, N] cuda float64 in the packed column order, max-abs bound).

        Peak device memory is two [m, N] float64 arrays (the sorted plaintexts and the output):
        the automorphisms and monomial shifts run on one ``pc``-row block at a time."""
        torch, n, m, a, pc = self.torch, self.n, plan.m, plan.a, plan.pc
        dev = lambda v: torch.as_tensor(np.ascontiguousarray(np.asarray(v, dtype=np.int64)), device="cuda")
        if isinstance(w, np.ndarray) or not hasattr(w, "device"):
            w = np.asarray(w, dtype=np.float64)
            if row_scale is not None:
                w = w * np.asarray(row_scale, dtype=np.float64)[:, None]
            top = float(np.max(np.abs(w))) if w.size else 0.0
            w = torch.as_tensor(np.ascontiguousarray(w), device="cuda")
        else:
            if row_scale is not None:
                w = w * torch.as_tensor(np.asarray(row_scale, dtype=np.float64)[:, None], device="cuda")
            top = float(np.max(np.abs(w.cpu().numpy())))
        pw = torch.zeros((m, n), dtype=torch.float64, device="cuda")
        rows = dev(plan.out_row)
        # sorted row i (= out_row[f]) is plan row i: block j = i // m, item i % m of pw
        for j in range(n // plan.span):
            pos = j * plan.span - plan.in_pos
            sign = torch.as_tensor(np.where(pos < 0, -1.0, 1.0), device="cuda")
            sel = (plan.out_row >= j * m) & (plan.out_row < (j + 1) * m)
            if not sel.any():
                continue
            idx = np.nonzero(sel)[0]
            blk = torch.zeros((m, plan.cols), dtype=torch.float64, device="cuda")
            blk[dev(plan.out_row[idx] - j * m)] = w[dev(idx)]
            pw[:, dev(pos % n)] = blk * sign
            del blk
        del w
        out = torch.zeros((pc, a, n), dtype=torch.float64, device="cuda")
        for idx, g in enumerate(plan.autos):
            src, sign = plan.automorphism_tables(g)
            src_d, sign_d = dev(src), torch.as_tensor(sign, device="cuda")
            for i0 in range(a):
                blk = pw[i0 * pc:(i0 + 1) * pc][:, src_d] * sign_d          # rows pc * i0 + p -> [p]
                s = (plan.shift * i0) % (2 * n)
                neg = False
                if s >= n:
                    s, neg = s - n, True
                if s:
                    blk = torch.cat([-blk[:, n - s:], blk[:, :n - s]], 1)
                if neg:
                    blk = -blk
                out[:, idx] += blk
                del blk
        del pw
        middle = (out.reshape(m, n) * (1.0 / m))[:, self._inter].contiguous()
        del out
        return middle, top * a / m

    def encode(self, middle, mx, limbs, scale=1.0):
        """Plaintexts of ``scale * middle`` at ``limbs`` (nominal scale)."""
        ctx = self.ctx
        if float(mx * abs(scale)) * float(ctx.scale_at(limbs)) >= 2.0 ** 62:
            raise ValueError("projection plaintext too large for encoding")
        vals = middle if scale == 1.0 else (middle * float(scale)).contiguous()
        prep = self.PreparedPlaintext(None, self.kit.slots, vals, float(mx * abs(scale)), packed=True)
        return self.encode_stage2(prep, None, self.kit.slots, False, ctx, cur_limbs=limbs)

    # ---- ciphertexts -------------------------------------------------------------------
    def inputs(self, ct, plan):
        """Batch of the A automorphic copies sigma_g(ct) in ``plan.autos`` order."""
        from .rotate import rotate as fast_rotate
        kit = self.kit
        cur = self.alignment.normalize_scale(ct, self.ctx)
        for step in plan.input_steps:
            self._count("rotate", cur.batch_size)
            cur = kit.stack(kit.unstack(cur) + kit.unstack(fast_rotate(cur, step, self.ctx)))
        return cur

    def _pack(self, cur, plan):
        ctx, kit, packing = self.ctx, self.kit, self.packing
        cur = self.alignment.normalize_scale(cur, ctx)
        for degree, step in plan.pack:
            half = cur.batch_size // 2
            self._count("rotate", half)
            if packing.enabled() and len(cur.cv) == 2:
                p, q = packing.butterfly(cur, half, ctx, degree)
                cur = kit._rotate_add(q, step, p)
                continue
            p = cur.cipher_like([x[:half] for x in cur.cv], batch_size=half)
            q = cur.cipher_like([x[half:] for x in cur.cv], batch_size=half)
            q = kit._monomial(q, degree)
            cur = kit._rotate_add(self.arithmetic.homo_sub(p, q, ctx), step, self.arithmetic.homo_add(p, q, ctx))
        return cur.cipher_like(cur.cv, batch_size=1)

    def project(self, copies, pts, plan):
        """copies: ``inputs`` batch (A); pts: ``encode`` output (pc * A) -> packed ciphertext."""
        if copies.batch_size != plan.a or pts.batch_size != plan.pc * plan.a:
            raise ValueError("inputs / plaintexts do not match the plan")
        self._count("mul_pt", plan.pc * plan.a)
        acc = self.arithmetic.grouped_pairwise_mac(copies, pts, plan.pc, self.ctx)
        return self._pack(acc, plan) if plan.pc > 1 else self.alignment.normalize_scale(acc, self.ctx)

    def project_middle(self, copies, middle, mx, plan, scale=1.0, groups=0):
        """``project`` with the plaintexts encoded on the fly, ``groups`` partial sums at a time
        (0 = all): the plaintext tensor never exceeds ``groups * A`` items."""
        if copies.batch_size != plan.a:
            raise ValueError("inputs do not match the plan")
        limbs = int(copies.state.cur_limbs)
        if groups <= 0 or groups >= plan.pc:
            return self.project(copies, self.encode(middle, mx, limbs, scale), plan)
        accs = []
        for p0 in range(0, plan.pc, groups):
            p1 = min(plan.pc, p0 + groups)
            pts = self.encode(middle[p0 * plan.a:p1 * plan.a], mx, limbs, scale)
            self._count("mul_pt", (p1 - p0) * plan.a)
            accs.append(self.arithmetic.grouped_pairwise_mac(copies, pts, p1 - p0, self.ctx))
            del pts
        acc = self.kit.stack(accs) if len(accs) > 1 else accs[0]
        return self._pack(acc, plan) if plan.pc > 1 else self.alignment.normalize_scale(acc, self.ctx)
