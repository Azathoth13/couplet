"""GQA attention with HF split-half RoPE and the joint causal SoftMax on EasyFHE slots.

Follows TurboFHE apps/sylph attention.py / nonlinear.softmax: RoPE with public
cos/sin vectors, optional public prefix keys/values sharing the private SoftMax
denominator, centered exponential, and normalize-and-square rounds with an
encrypted inverse square root. The QK/SV products use lane diagonals of the
projection layout (an EasyFHE slot implementation in place of SinC CCMM).

The slot geometry comes from ``layout.HeadGeometry`` (``g`` below).  Score layout: one
ciphertext per private *entry* (Q shard s, query half qh, key halves held).  Slot
``(64 * kh_off + d) * g.block + 64 * hl + p`` holds the logit of query lane p (token
``64 qh + LANE_TOKEN[p]``) of local head hl against key lane ``(p + d) % 64`` of key half
kh.  Llama-3-8B (head_dim 128) keeps both key halves in one ciphertext (16 entries);
Llama-3.2-1B (head_dim 64) has 64 key blocks per ciphertext, so each visible
(qh, kh) stream gets its own (12 entries).  Public logits share one ciphertext:
slot ``g.item_period * m + g.block * i + 64 hl + p`` (Q item ``i = 2 s + qh``).  Row
values (denominators) live in the "D" layout: slot ``g.block * i + 64 hl + p``, period
``g.item_period`` (4096 for both models).
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from .approx import InverseSqrtPlan, exp_coefficients, scaled_exp_plan
from .nonlinear import inverse_sqrt
from .layout import LANES, LANE_TOKEN, SLOTS, HeadGeometry, rope_frequencies

SLOT = np.arange(SLOTS)
GROUP = 4                                   # diagonals merged per channel reduction
STREAMS = ((0, 0), (1, 0), (1, 1))          # (query half, key half) pairs that can be visible


def _low_memory():
    """Small-card mode: free device-side caches between attention phases."""
    return bool(os.environ.get("SYLPH_LOWMEM") or os.environ.get("SYLPH_HALFBOOT_BATCH")
                or os.environ.get("SYLPH_POLY_BATCH"))


def attention_rotations(geometry=None):
    g = HeadGeometry() if geometry is None else geometry
    B = g.block
    r = {B << k for k in range(int(math.log2(SLOTS // B)))}          # block reductions
    r |= {LANES * k for k in range(1, g.hps)} | {SLOTS - LANES * k for k in range(1, g.hps)}   # phase moves
    r |= {SLOTS - B, SLOTS - 2 * B, 2 * B}                           # RoPE partner / SV shifts
    r |= set(range(1, LANES)) | {SLOTS - LANES + d for d in range(1, LANES)}   # lane diagonals
    r |= {g.item_period}                                             # public row fold
    return r


class Attention:
    def __init__(self, eng, *, public_keys, public_values, logit_bound, centered_bound, rounds,
                 rope_theta=500000.0, heads=32, kv_heads=8, head_dim=128, rope_scaling=None,
                 inverse_degree=127, intervals=None, head_intervals=None, active_shards=None):
        self.b = eng
        g = self.g = HeadGeometry(head_dim, heads, kv_heads)
        self.pk = np.asarray(public_keys, dtype=np.float64).reshape(-1, kv_heads, head_dim)   # [P, kv, d]
        self.pv = np.asarray(public_values, dtype=np.float64).reshape(-1, kv_heads, head_dim)
        self.public = self.pk.shape[0]
        if self.public > 2:
            raise ValueError("at most two public prefix tokens are supported")
        self.logit_bound, self.centered_bound, self.rounds = logit_bound, centered_bound, rounds
        self.inverse_degree = inverse_degree
        self.intervals = None if intervals is None else [tuple(map(float, iv)) for iv in intervals]
        self.head_intervals = None if head_intervals is None else np.asarray(head_intervals, dtype=np.float64)
        self.active_shards = tuple(range(g.shards)) if active_shards is None else tuple(active_shards)
        if (not self.active_shards or tuple(sorted(set(self.active_shards))) != self.active_shards
                or any(not 0 <= s < g.shards for s in self.active_shards)):
            raise ValueError("active_shards must be nonempty, sorted, unique and in range")
        self.shards = len(self.active_shards)
        self.entries = tuple(e for e in g.entries if e[0] in self.active_shards)
        self.n_priv = len(self.entries)
        # entries are shard-major: entry index = shard * len(classes) + class; a class is the
        # (query half, key halves) pattern shared by every shard (batched over shards below)
        self.classes = tuple(dict.fromkeys((qh, khs) for _, qh, khs in self.entries))
        self.max_keys = self.public + 128
        self._build_vectors(rope_theta, rope_scaling)
        self.sm = self._softmax_setup()
        self.times = None

    def _mark(self, name, t0):
        if self.times is None:
            return time.time()
        import easyfhe as torch
        torch.cuda.synchronize()
        self.times[name] = self.times.get(name, 0.0) + time.time() - t0
        return time.time()

    # ---- public vectors -------------------------------------------------------------
    def _build_vectors(self, theta, rope_scaling):
        g, P = self.g, self.public
        B, D, half = g.block, g.head_dim, g.head_dim // 2
        freq = rope_frequencies(D, theta, rope_scaling)
        c = g.slot_channel
        self.rope = {}
        for h in range(2):
            pos = P + 64 * h + LANE_TOKEN[g.slot_lane]
            ang = pos * freq[c % half]
            sign = np.where(c < half, -1.0, 1.0)
            even = g.slot_block % 2 == 0            # partner (c xor half) is one block up
            self.rope[h] = (np.cos(ang), np.where(even, sign * np.sin(ang), 0), np.where(even, 0, sign * np.sin(ang)))
        self.phase_mask = [(g.slot_phase == g.phase[j]).astype(float) for j in range(g.hps)]
        self.lane_lo = {d: (g.slot_lane < LANES - d).astype(float) for d in range(1, LANES)}
        self.lane_hi = {d: 1 - self.lane_lo[d] for d in range(1, LANES)}
        self.residue = [(g.slot_block % GROUP == q).astype(float) for q in range(GROUP)]
        qtok = LANE_TOKEN[g.slot_lane]
        # visibility of each score slot for stream (qh, kh) placed at key-block offset ``off``
        self.visible = {}
        for qh, kh in STREAMS:
            for off in (0, 64) if g.kh_per_ct == 2 else (0,):
                d = g.slot_block - off
                key = 64 * kh + LANE_TOKEN[(g.slot_lane + d) % LANES]
                ok = (key <= 64 * qh + qtok) & (d >= 0) & (d < 64)
                self.visible[qh, kh, off] = ok.astype(float)
        self.entry_mask = []
        for e in self.entries:
            s, qh, khs = e
            m = sum(self.visible[qh, kh, g.kh_offset(e, kh)] for kh in khs if (qh, kh) in STREAMS)
            self.entry_mask.append(np.asarray(m, dtype=float))
        valid_item = (g.slot_block % g.item_capacity) < g.items
        self.public_mask = ((SLOT < g.item_period * P) & valid_item).astype(float)
        # D layout
        i = g.slot_block % g.item_capacity
        self.d_count = 1.0 + P + 64 * (i % 2) + qtok
        self.d_block = [(g.slot_block == i_).astype(float) for i_ in range(g.items)]
        self.d_res = [(g.slot_block % g.item_capacity == i_).astype(float) for i_ in range(g.items)]
        # A 24-head layer uses 12 of the 16 items in each power-of-two D period.
        # Those unused slots still pass through the inverse-sqrt circuit: zero is
        # outside its calibrated domain, and can diverge during Newton steps.
        active_items = {g.item(entry) for entry in self.entries}
        self.d_dummy = np.isin(g.slot_block % g.item_capacity, list(active_items), invert=True).astype(float)
        self.has_dummy_rows = len(active_items) < g.item_capacity
        self.private_count = {qh: 1.0 + P + 64 * qh + qtok for qh in range(2)}
        self.public_count = self.d_count
        # query head of every score slot: shard s carries heads hps s + j, local head j at slot
        # phase phase[j]; public items sit at block ``items * m + i``
        local = g.slot_local
        self.private_head = [g.hps * e[0] + local for e in self.entries]
        # The unused tail items of a row period need a harmless in-range head
        # index even though their public masks are zero.
        self.public_head = np.minimum(g.hps * (i // 2) + local, g.heads - 1)
        self.item_sel = [(g.slot_block == m * g.item_capacity + i_).astype(float)
                         for m in range(max(1, P)) for i_ in range(g.items)]

    # ---- helpers -----------------------------------------------------------------------
    def _halves(self, ct):
        """batch (2s + h) -> (lo batch, hi batch) over shards."""
        b = self.b
        items = b.unstack(ct)
        return b.stack(items[0::2]), b.stack(items[1::2])

    def _interleave(self, lo, hi):
        b = self.b
        lo, hi = b.unstack(lo), b.unstack(hi)
        return b.stack_aligned([x for pair in zip(lo, hi) for x in pair])

    def _rope(self, x, half, scale=1.0):
        b, B = self.b, self.g.block
        cos, sp, sm = self.rope[half]
        key = ('rope', half, scale)
        x = b.norm(x)
        xp, xm = b.rotate_many(x, (B, -B))
        out = b.cmul_sum([(x, cos * scale, key + ('c',)), (xp, sp * scale, key + ('p',)),
                          (xm, sm * scale, key + ('m',))])
        return b.norm(out)

    def _expand(self, kv):
        """kv batch (2 ks + kh) over KV shards -> (E_lo, E_hi): batch over Q shards, every phase
        of Q shard s holding the KV head its local head reads (GQA expansion)."""
        b, g = self.b, self.g
        items = b.unstack(kv)
        per = {}
        for head in range(g.kv_heads):
            dests = [(s, j) for s in self.active_shards for j in range(g.hps) if g.kv_head(s, j) == head]
            if not dests:
                continue
            ks, jj = divmod(head, g.hps)
            src = b.stack([items[2 * ks], items[2 * ks + 1]])
            m = b.norm(b.cmul(src, self.phase_mask[jj], cache_key=('phase', jj)))
            shards = sorted({s for s, _ in dests})
            for s in shards:
                shifts = [(g.phase[jj] - g.phase[j]) * LANES for s_, j in dests if s_ == s]
                acc = b.add_many(b.rotate_many(m, shifts))
                for kh, item in enumerate(b.unstack(acc)):
                    per[s, kh] = item if (s, kh) not in per else b.add(per[s, kh], item)
        return (b.stack_aligned([per[s, 0] for s in self.active_shards]),
                b.stack_aligned([per[s, 1] for s in self.active_shards]))

    def _lane_rotate(self, x, d):
        """Lane p of the result holds lane (p + d) % 64 of x (inside every 64-lane run)."""
        return self._lane_rotates(x, [d])[0]

    def _lane_rotates(self, x, ds):
        """[_lane_rotate(x, d) for d in ds], all rotations of x hoisted together."""
        b = self.b
        x = b.norm(x)
        nz = [d for d in ds if d]
        rots = b.rotate_many(x, [s for d in nz for s in (d, d - LANES)])
        rot = {d: (rots[2 * i], rots[2 * i + 1]) for i, d in enumerate(nz)}
        out = []
        for d in ds:
            if d == 0:
                out.append(x)
                continue
            out.append(b.norm(b.cmul_sum([(rot[d][0], self.lane_lo[d], ('lane_lo', d)),
                                           (rot[d][1], self.lane_hi[d], ('lane_hi', d))])))
        return out

    def _class_of(self, st):
        """(class index, key-block offset) of stream ``st`` inside its entry ciphertext."""
        qh, kh = st
        for c, (qh_, khs) in enumerate(self.classes):
            if qh_ == qh and kh in khs:
                return c, 64 * khs.index(kh)
        raise KeyError(st)

    def _by_class(self, items):
        """Entry-ordered items -> one batch per class over the shards."""
        b, C = self.b, len(self.classes)
        return [b.stack([items[s * C + c] for s in range(self.shards)]) for c in range(C)]

    def _entry_order(self, per_class):
        """Per-class batches over the shards -> entry-ordered batch."""
        b = self.b
        cols = [b.unstack(x) for x in per_class]
        return b.stack_aligned([cols[c][s] for s in range(self.shards) for c in range(len(cols))])

    # ---- D layout reductions ---------------------------------------------------------------
    def reduce_rows(self, priv, pub, scale=None):
        """Sum over keys -> one D ciphertext (rows replicated with period item_period)."""
        b, g = self.b, self.g
        r = b.sum_slots(priv, g.block, g.head_dim)
        terms = []
        for e, it in zip(self.entries, b.unstack(r)):
            i = g.item(e)
            terms.append((it, self.d_block[i] if scale is None else self.d_block[i] * scale,
                          None if scale is not None else ('dblock', i)))
        if pub is not None:
            u = b.rotate_add(pub, pub, g.item_period) if self.public == 2 else pub
            vec = ((SLOT < g.item_period) & ((g.slot_block % g.item_capacity) < g.items)).astype(float)
            terms.append((u, vec if scale is None else vec * scale, None if scale is not None else ('d0',)))
        acc = b.norm(b.cmul_sum(terms))
        return b.sum_slots(acc, g.item_period, SLOTS // g.item_period)

    def expand_rows(self, D):
        """D -> (private batch replicated over keys, public ciphertext)."""
        b, g = self.b, self.g
        rep = b.stack([D] * self.n_priv)
        vec = np.stack([self.d_res[g.item(e)] for e in self.entries])
        x = b.norm(b.cmul(rep, vec, cache_key=('dres',)))
        return b.sum_slots(x, g.block, g.item_capacity), D

    # ---- SoftMax -------------------------------------------------------------------------------
    def _softmax_setup(self):
        """Public SoftMax plan (masks, counts, inverse-sqrt plans, exponential); calibration only."""
        R, eb, P = self.rounds, self.centered_bound, self.public
        masks = list(self.entry_mask) + ([self.public_mask] if P else [])
        counts = [self.private_count[e[1]] for e in self.entries] + ([self.public_count] if P else [])
        mask, cnt = np.stack(masks), np.stack(counts)
        first_lower = 0.9 * math.exp(-2 * eb / 2 ** R)
        first_upper = 1.02 * (1 + math.exp(-4 * eb / 2 ** R)) / 2
        # Calibrated per-round intervals (TurboFHE natural_attention.softmax), else the analytic ones.
        intervals = self.intervals or [(first_lower, first_upper)] + [(0.9 / self.max_keys, 1.25)] * (R - 1)
        scaled = None if self.head_intervals is None else scaled_exp_plan(eb, R, self.max_keys)
        sm = dict(mask=mask, cnt=cnt, scaled=scaled)
        if scaled is None:
            # Round-0 exponentials are far below one: refresh at sqrt(hi * count), which bounds every
            # value of a row, to keep their relative precision (sub-unit refresh bound).
            sm['first_bound'] = min(1.01, 1.05 * math.sqrt(intervals[0][1] * float(np.max(cnt))))
        else:
            # TurboFHE natural_attention head-scaled exponential: f_h = 1 / sqrt(hi_h) moves every
            # head's round-0 mean squares onto [lo_h / hi_h, 1] (later layers reach 1e-10, which
            # the fixed scale cannot hold). f_h exp(t) = exp(t + ln f_h): a public input shift.
            lo_h, hi_h = self.head_intervals[:, 0], self.head_intervals[:, 1]
            heads = np.stack(self.private_head + ([self.public_head] if P else []))
            shift = 2 ** R * np.log(1 / np.sqrt(hi_h)) / eb
            f = (1 / np.sqrt(hi_h))[heads]
            sm['shift'] = shift[heads]
            intervals = [(float(np.min(lo_h / hi_h)), 1.0)] + list(intervals[1:])
            sm['first_bound'] = 1.05 * float(np.max(np.minimum(f, np.sqrt(cnt))))
        sm['plans'] = [InverseSqrtPlan.create(lo, hi, self.inverse_degree) for lo, hi in intervals]
        return sm

    def _join(self, priv, pub):
        """(private batch, public ciphertext) -> one batch (private only without a public prefix)."""
        b = self.b
        return b.stack_aligned([priv, pub]) if self.public else b.norm(priv)

    def softmax(self, priv, pub):
        """TurboFHE Nonlinear.softmax with centering, on (private batch, public) logits."""
        b = self.b
        n = self.n_priv
        R, eb = self.rounds, self.centered_bound
        sm = self.sm
        mask, cnt, plans, first_bound = sm['mask'], sm['cnt'], sm['plans'], sm['first_bound']
        both = lambda t: (b.take(t, 0, n), b.take(t, n, n + 1) if self.public else None)
        if sm['scaled'] is None:
            x = b.ready(self._join(priv, pub), 8, self.logit_bound)
            lm = b.cmul(x, mask / (cnt * eb), cache_key=('sm_mean', eb))
            mean_p, mean_q = self.expand_rows(self.reduce_rows(*both(b.norm(lm))))
            normalized = b.sub(b.smul(x, 1 / eb), self._join(mean_p, mean_q))
            y = b.poly(normalized, exp_coefficients(eb, R), mask=mask, mask_key=('sm_exp', eb, R))
        else:
            coeffs, alpha, _ = sm['scaled']
            x = b.ready(self._join(priv, pub), 8, self.logit_bound)
            lm = b.cmul(x, alpha * mask / (cnt * eb), cache_key=('layer', 'sm_mean'))
            mean_p, mean_q = self.expand_rows(self.reduce_rows(*both(b.norm(lm))))
            mean = b.cmul(self._join(mean_p, mean_q), mask, cache_key=('sm_mask',))
            # visible: z = alpha (y + shift_h) + alpha - 1; masked slots: z = alpha - 1 (in domain)
            z = b.sub(b.smul(x, alpha / eb), mean)
            z = b.add_vector(z, mask * alpha * sm['shift'] + (alpha - 1), cache_key=('layer', 'sm_shift'))
            y = b.poly(z, coeffs, mask=mask, mask_key=('layer', 'sm_exp'))
        for step in range(R):
            y = b.ready(y, 5, 2 * first_bound, fine=True) if step == 0 else b.ready(y, 3, 1.01)
            sq = b.square(y, rescale=True)
            D = self.reduce_rows(*both(sq), scale=1 / self.d_count if step == 0 else None)
            if self.has_dummy_rows:
                # Fill inactive D slots with an in-domain public value.  Expansion
                # selects only active items, so this cannot affect model outputs.
                lo, hi = plans[step].lower, plans[step].upper
                D = b.add_vector(D, self.d_dummy * math.sqrt(lo * hi),
                                 cache_key=('layer', 'sm_d_dummy', step, lo, hi))
            inv = inverse_sqrt(b, D, plans[step], scale=1 / np.sqrt(self.d_count) if step == 0 else 1.0)
            inv_p, inv_q = self.expand_rows(inv)
            y = b.square(b.mul(y, self._join(inv_p, inv_q), rescale=True))
        y = b.ready(y, 4, 1.01)
        y = b.norm(b.cmul(y, mask, cache_key=('sm_mask',)))
        return both(y)

    # ---- forward -----------------------------------------------------------------------------------
    def forward(self, q, k, v):
        """Full Q/K/V layout in; active Q shards out in global shard order."""
        b, g = self.b, self.g
        B, D = g.block, g.head_dim
        n = self.shards
        tick = time.time()
        if n != g.shards:
            q_items = b.unstack(q)
            q = b.stack([q_items[i] for s in self.active_shards for i in (2 * s, 2 * s + 1)])
        # The logits are refreshed at the SoftMax input (ready(x, 8)), so QK only needs its own
        # depth: Q rope, product, residue mask, pack mask (4); K adds the phase and lane masks (6).
        q, k = b.trim(q, 4), b.trim(k, 6)
        q_lo, q_hi = self._halves(q)
        q_lo, q_hi = self._rope(q_lo, 0, 1 / math.sqrt(D)), self._rope(q_hi, 1, 1 / math.sqrt(D))
        k_lo, k_hi = self._halves(k)
        kr = self._interleave(self._rope(k_lo, 0), self._rope(k_hi, 1))      # (ks, kh) order
        e_lo, e_hi = self._expand(kr)
        # ---- QK^T, GROUP diagonals per channel reduction ----
        Q = {(0, 0): q_lo, (1, 0): q_hi, (1, 1): q_hi}
        E = {0: e_lo, 1: e_hi}
        scores = [None] * len(self.classes)
        for a in range(LANES // GROUP):
            prods = {st: [] for st in STREAMS}
            ds = [GROUP * a + qq for qq in range(GROUP)]
            lanes = {kh: self._lane_rotates(E[kh], ds) for kh in (0, 1)}
            for qq in range(GROUP):
                kd = {kh: lanes[kh][qq] for kh in (0, 1)}
                for st in STREAMS:
                    prods[st].append(b.mul(Q[st], kd[st[1]], rescale=True))
            # merged = sum_qq residue_qq * sum_t rot(prod_qq, t BLOCK)
            #        = sum_t rot(U_t, t BLOCK),  U_t = sum_qq residue_{(qq + t) % 4} * prod_qq
            merged = {}
            for st in STREAMS:
                U = [b.cmul_sum([(prods[st][qq], self.residue[(qq + t) % GROUP], ('res', (qq + t) % GROUP))
                                 for qq in range(GROUP)]) for t in range(GROUP)]
                inner = b.rotate_add(U[1], U[3], 2 * B)
                merged[st] = b.rotate_add(b.rotate_add(U[0], inner, B), U[2], 2 * B)
            for st in STREAMS:
                qh, kh = st
                m = b.sum_slots(b.norm(merged[st]), GROUP * B, D // GROUP)
                c, off = self._class_of(st)
                sel = ((g.slot_block >= off + GROUP * a) & (g.slot_block < off + GROUP * a + GROUP)).astype(float)
                part = b.cmul(m, sel * self.visible[qh, kh, off], cache_key=('pack', a, st))
                scores[c] = part if scores[c] is None else b.add(scores[c], part)
        priv = self._entry_order([b.norm(x) for x in scores])
        # public keys
        pub = None
        if self.public:
            qs = self._interleave(q_lo, q_hi)
            for m in range(self.public):
                vec = np.stack([self._kv_vector(self.pk[m], s)
                                for s in self.active_shards for _ in range(2)])
                t = b.sum_slots(b.norm(b.cmul(qs, vec, cache_key=('layer', 'pk', m))), B, D)
                sel = np.stack([self.item_sel[2 * s + h + g.items * m]
                                for s in self.active_shards for h in range(2)])
                t = b.norm(b.cmul(t, sel, cache_key=('pksel', m)))
                for it in b.unstack(t):
                    pub = it if pub is None else b.add(pub, it)
            pub = b.norm(pub)
        # Q and K are consumed.  Drop them and, on a small card (SYLPH_LOWMEM=1 or the
        # 24 GB batching knobs), the rotation keys the QK diagonals cached on the device,
        # before the softmax refresh.  The keys stay on the host and reload on demand; a
        # 24 GB card cannot hold that cache plus the HalfBoot workspace.  On a large card
        # the reload costs ~0.9 s per layer (Llama-3-8B), so the cache is kept.
        del scores, q, k, q_lo, q_hi, k_lo, k_hi, kr, e_lo, e_hi, Q, E
        if _low_memory() and hasattr(self.b.ctx, "clear_cuda_rotation_cache"):
            self.b.ctx.clear_cuda_rotation_cache()
        tick = self._mark("qk", tick)
        p_priv, p_pub = self.softmax(priv, pub)
        tick = self._mark("softmax", tick)
        out = self._sv(p_priv, p_pub, v)
        self._mark("sv", tick)
        return out

    def _kv_vector(self, values, s):
        """[kv heads, head_dim] public vector -> slots of Q shard s (the KV head each phase reads)."""
        g = self.g
        return values[g.kv_head_of_slot(s), g.slot_channel]

    def _sv(self, p_priv, p_pub, v):
        b, g = self.b, self.g
        B, D = g.block, g.head_dim
        n = self.shards
        # The output feeds the next projection's S2C at the floor: SV needs three levels
        # (P: select, residue mask, product; V: phase mask, lane mask, product).
        p_priv = b.trim(b.ready(p_priv, 3, 1.01), 3)
        v = b.trim(v, 3)
        by_class = self._by_class(b.unstack(p_priv))
        P = {st: by_class[self._class_of(st)[0]] for st in STREAMS}
        v_lo, v_hi = self._halves(v)
        V = dict(zip((0, 1), self._expand(self._interleave(v_lo, v_hi))))
        out = {0: None, 1: None}
        for a in range(LANES // GROUP):
            rep = {}
            for st in STREAMS:
                qh, kh = st
                off = self._class_of(st)[1]
                sel = ((g.slot_block >= off + GROUP * a) & (g.slot_block < off + GROUP * a + GROUP)).astype(float)
                x = b.norm(b.cmul(P[st], sel, cache_key=('svsel', a, st)))
                rep[st] = b.sum_slots(x, GROUP * B, D // GROUP)
            # sum_t rot(residue_qq * rep, -t BLOCK) = sum_t residue_{(qq + t) % 4} * rot(rep, -t BLOCK)
            shifted = {}
            for st in STREAMS:
                r1, r2 = b.rotate_many(rep[st], (-B, -2 * B))
                shifted[st] = [rep[st], r1, r2, b.rotate(r1, -2 * B)]
            ds = [GROUP * a + qq for qq in range(GROUP)]
            lanes = {kh: self._lane_rotates(V[kh], ds) for kh in (0, 1)}
            for qq in range(GROUP):
                vd = {kh: lanes[kh][qq] for kh in (0, 1)}
                for st in STREAMS:
                    x = b.cmul_sum([(shifted[st][t], self.residue[(qq + t) % GROUP], ('res', (qq + t) % GROUP))
                                    for t in range(GROUP)])
                    t = b.mul(x, vd[st[1]])
                    out[st[0]] = t if out[st[0]] is None else b.add(out[st[0]], t)
        priv = self._interleave(b.norm(out[0]), b.norm(out[1]))
        if not self.public:
            return b.norm(priv)
        # public values
        p_pub = b.trim(b.ready(p_pub, 3, 1.01), 3)
        pub_terms = None
        for m in range(self.public):
            rep = b.stack([p_pub] * (2 * n))
            sel = np.stack([self.item_sel[2 * s + h + g.items * m]
                            for s in self.active_shards for h in range(2)])
            x = b.sum_slots(b.norm(b.cmul(rep, sel, cache_key=('pksel', m))), B, D)
            vec = np.stack([self._kv_vector(self.pv[m], s)
                            for s in self.active_shards for _ in range(2)])
            t = b.cmul(x, vec, cache_key=('layer', 'pv', m))
            pub_terms = t if pub_terms is None else b.add(pub_terms, t)
        return b.norm(b.add(priv, b.norm(pub_terms)))
