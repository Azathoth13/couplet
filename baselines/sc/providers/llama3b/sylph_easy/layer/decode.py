"""One-token decode step of the Sylph (slot-coeff) layer on the *prefill* slot layout.

Design (mirrors ``decoder.DecoderLayer`` / ``attention.Attention`` as closely as possible):

* The decode token is replicated over the 64 token lanes of the ``lo`` ciphertexts (the ``hi``
  half, tokens 64..127, is empty and never materialized).  Every lane-wise stage (RMSNorm, the
  Rhombus coefficient projections, SiLU, residual adds) therefore runs the unchanged prefill code
  on half the prefill batch; the projections keep only the real (``lo``) HalfBoot output.
  The coefficient PCMM itself does not get cheaper for one token: Rhombus multiplies
  ``K`` plaintexts per (input block, output block) whatever the token count.
* The KV cache lives in the GQA-expanded slot layout of ``Attention._expand`` (one ciphertext
  per Q shard and token half; slot ``(block, phase, lane)`` holds channel ``channel_of_block``
  of the KV head read by local head ``phase`` for token ``64 half + LANE_TOKEN[lane]``).
  QK^T of one query against 64 cached keys is a slot product followed by the channel-block
  rotate-sum (no lane diagonals), SV a lane rotate-sum.  The new key / the public prefix keys
  form an "extra" stream that occupies lanes 0, 1, 2 of every run.
* Softmax is the prefill centered exponential (normalize-and-square, encrypted inverse square
  root).  The channel sum replicates every score across blocks, so each shard's lo, hi and extra
  streams occupy three blocks.  Those shards are then packed into one ciphertext (shard ``s`` at
  blocks ``4s``, ``4s+1``, ``4s+2``, with ``4s+3`` left empty so the block fold cannot cross a
  shard).  8B uses 32 of 128 blocks; 1B uses 16 of 64.  Empty slots are a public in-range constant
  for the duration of inverse square root, and the row expansion masks them back out.

Lane sums: ``rotate(x, d)[i] = x[i + d]``.  Summing rotations by +1, +2, .. +32 leaves lane 0 of
every 64-lane run with the sum of its run (other lanes mix in the next run); summing rotations by
-1, -2, .. -32 of a vector that is non-zero only at lane 0 replicates lane 0 over the run.
"""
from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING

import numpy as np

from .attention import Attention, SLOT
from .layout import LANES, LANE_TOKEN, SLOTS, pack, rope_frequencies
from .nonlinear import inverse_sqrt
from .approx import InverseSqrtPlan, exp_coefficients, silu_coefficients

if TYPE_CHECKING:
    from .decoder import DecoderLayer

STREAMS = 3           # lo (tokens 0..63), hi (64..127), ex (new key, public prefix keys)
LANE_BITS = int(math.log2(LANES))


def replicate_token(vector, hidden):
    """[hidden] -> the ``lo`` slot vectors (one per 512-feature shard) with the token in every lane."""
    x = np.zeros((2 * LANES, hidden))
    x[:LANES] = np.asarray(vector, dtype=np.float64)[None, :]
    return pack(x)[0::2]


def expanded_cache(values, g):
    """[128 tokens, kv_heads, head_dim] (K after RoPE, or V) -> slot vectors ``[half][Q shard]``
    of the GQA-expanded cache layout (``Attention._expand`` output)."""
    values = np.asarray(values, dtype=np.float64)
    out = []
    for half in range(2):
        tok = LANES * half + LANE_TOKEN[g.slot_lane]
        out.append([values[tok, g.kv_head_of_slot(s), g.slot_channel] for s in range(g.shards)])
    return out


class DecodeAttention:
    """Attention of one query (all lanes) against an expanded KV cache + new key + public prefix."""

    def __init__(self, att: Attention, position, rope_theta=500000.0, rope_scaling=None):
        self.att, self.b, self.g = att, att.b, att.g
        g, P = self.g, att.public
        self.P = P
        self.n = g.shards
        # RoPE vectors of the new token (same construction as Attention._build_vectors)
        half = g.head_dim // 2
        c = g.slot_channel
        freq = rope_frequencies(g.head_dim, rope_theta, rope_scaling)
        ang = float(position) * freq[c % half]
        sign = np.where(c < half, -1.0, 1.0)
        even = g.slot_block % 2 == 0
        att.rope['dec', int(position)] = (np.cos(ang), np.where(even, sign * np.sin(ang), 0),
                                          np.where(even, 0, sign * np.sin(ang)))
        self.rope_key = ('dec', int(position))
        lane = g.slot_lane
        self.lane0 = (lane == 0).astype(float)
        self.ex_lane = [(lane == m).astype(float) for m in range(1 + P)]
        self.times = None
        n = self.n
        block = g.slot_block
        # QK still emits one ciphertext per shard with the three streams in blocks 0, 1, 2.
        # Softmax packs shard s at blocks 4s .. 4s+2 (4s+3 is a guard).
        self.block0 = (block == 0).astype(float)
        self.scatter_mask = np.stack([
            (block == 0).astype(float),
            (block == 1).astype(float),
            ((block == 2) & (lane < 1 + P)).astype(float),
        ])
        nblocks = SLOTS // g.block
        if 4 * n >= nblocks or nblocks & (nblocks - 1):
            raise ValueError("decode score packing needs 4 blocks per shard inside a power-of-two ring")
        owned = block < 4 * n
        row = owned & (block % 4 == 0)
        self.row_sel = (row & (lane == 0)).astype(float)
        self.row_blocks = row.astype(float)
        self.empty_rows = (1.0 - self.row_blocks)
        self.mask = (owned & ((block % 4 == 0) | (block % 4 == 1)
                              | ((block % 4 == 2) & (lane < 1 + P)))).astype(float)
        self.count = float(1 + P + 2 * LANES)         # keys per row: public + cached 128 + new
        # softmax plan on the single packed ciphertext (same calibration as the prefill plan)
        sm = att.sm
        self.sm = dict(plans=sm['plans'], first_bound=sm['first_bound'], scaled=sm['scaled'])
        if sm['scaled'] is not None:
            shard = np.minimum(block // 4, n - 1)
            heads = shard * g.hps + g.slot_local
            lo_h, hi_h = att.head_intervals[:, 0], att.head_intervals[:, 1]
            # Decode head slack: the calibration's per-head round-0 upper bounds hi_h come from the
            # 128-token prefill visible sets; a decode row sees 129 keys (the new token included) and
            # its row mean square can exceed hi_h (1B chain layer 3: 1.026 x, which breaks the
            # inverse-sqrt domain).  Widening hi_h by `slack` moves round 0 onto [lo_h/(slack hi_h), 1]
            # at no cost (same degree, same bootstraps); 1.5 suffices for the 1B chain, 2 is the default.
            slack = float(os.environ.get('SYLPH_DEC_HEAD_SLACK', '2'))
            if slack != 1.0:
                hi_h = hi_h * slack
                f = 1 / np.sqrt(hi_h)
                first = InverseSqrtPlan.create(float(np.min(lo_h / hi_h)), 1.0, att.inverse_degree)
                self.sm['plans'] = [first] + list(sm['plans'][1:])
                self.sm['first_bound'] = 1.05 * float(np.max(np.minimum(f, math.sqrt(self.count))))
            shift = 2 ** att.rounds * np.log(1 / np.sqrt(hi_h)) / att.centered_bound
            self.sm['shift'] = shift[heads]                                          # phases repeat every block
        # public prefix vectors in the expanded layout
        self.pk_vec = [np.stack([att._kv_vector(att.pk[m], s) for s in range(n)]) for m in range(P)]
        self.pv_ex = np.stack([sum(self.ex_lane[1 + m] * att._kv_vector(att.pv[m], s) for m in range(P))
                               if P else np.zeros(SLOTS) for s in range(n)])

    # ---- lane sums ----------------------------------------------------------------
    def _mark(self, name, t0):
        if self.times is None:
            return time.time()
        import easyfhe as torch
        torch.cuda.synchronize()
        self.times[name] = self.times.get(name, 0.0) + time.time() - t0
        return time.time()

    def _broadcast_block0(self, z):
        """``z`` nonzero only on block 0 -> every block holds it.

        Positive block rotations are the ones the prefill key set carries (``B << k``).  Parking
        block 0 on the last block and summing toward lower indices fills the ring.
        """
        b, step = self.b, self.g.block
        z = b.rotate(z, step)
        while step < SLOTS:
            z = b.rotate_add(z, z, step)
            step *= 2
        return z

    def lane_sum0(self, x):
        """Lane 0 of every run <- sum of the run's 64 lanes (other lanes hold mixed sums)."""
        for k in range(LANE_BITS):
            x = self.b.rotate_add(x, x, 1 << k)
        return x

    def lane_bcast0(self, x):
        """x non-zero only at lane 0 -> every lane of the run holds lane 0."""
        for k in range(LANE_BITS):
            x = self.b.rotate_add(x, x, -(1 << k))
        return x

    def lane_total(self, x):
        """Every lane <- sum of its run (one level: lane-0 select)."""
        x = self.b.norm(self.b.cmul(self.lane_sum0(x), self.lane0, cache_key=('dec_lane0',)))
        return self.lane_bcast0(x)

    # ---- GQA expansion of one (lo) kv batch -------------------------------------------
    def expand_lo(self, kv):
        """kv batch over KV shards (tokens in the lanes) -> batch over Q shards (expanded layout)."""
        b, g, att = self.b, self.g, self.att
        items = b.unstack(kv)
        per = {}
        for head in range(g.kv_heads):
            ks, jj = divmod(head, g.hps)
            m = b.norm(b.cmul(items[ks], att.phase_mask[jj], cache_key=('phase', jj)))
            dests = [(s, j) for s in range(g.shards) for j in range(g.hps) if g.kv_head(s, j) == head]
            for s in sorted({s for s, _ in dests}):
                shifts = [(g.phase[jj] - g.phase[j]) * LANES for s_, j in dests if s_ == s]
                acc = b.add_many(b.rotate_many(m, shifts))
                per[s] = acc if s not in per else b.add(per[s], acc)
        return b.stack_aligned([per[s] for s in range(g.shards)])

    # ---- one-ciphertext score layout -------------------------------------------------
    def pack_scores(self, x):
        """Batch of ``n`` shard scores (blocks 0, 1, 2) -> one ciphertext, shard ``s`` at blocks ``4s``.

        The move is a wrap by ``nblocks - 4s`` blocks toward lower indices, decomposed into the
        positive block keys ``B << k``.  Negative steps past ``-2B`` are not in the key set.
        """
        b, B, n = self.b, self.g.block, self.n
        nblocks = SLOTS // B
        x = b.norm(x)
        items = b.unstack(x)
        offset = [(nblocks - 4 * i) % nblocks for i in range(n)]
        for k in range(int(math.log2(nblocks))):
            grp = [i for i in range(n) if (offset[i] >> k) & 1]
            if not grp:
                continue
            moved = b.unstack(b.rotate(b.stack([items[i] for i in grp]), B << k))
            for i, it in zip(grp, moved):
                items[i] = it
        return b.add_many(items)

    def _unshard(self, y):
        """Inverse of ``pack_scores``: shard ``s`` returns to blocks 0, 1, 2 (batch ``n``)."""
        b, B, n = self.b, self.g.block, self.n
        y = b.norm(y)
        items = [y] * n
        k = 0
        while (1 << k) < n:
            grp = [s for s in range(n) if s & (1 << k)]
            moved = b.unstack(b.rotate(b.stack([items[s] for s in grp]), B << (k + 2)))
            for s, it in zip(grp, moved):
                items[s] = it
            k += 1
        return b.stack(items)

    def reduce_rows(self, y, scale=None):
        """Sum blocks ``4s``, ``4s+1``, ``4s+2`` of the one score ciphertext.

        The kept value is lane 0 of block ``4s`` (then lane-broadcast).  The guard block stops
        the fold from reading the next shard, and the selector drops the blocks the fold dirties.
        """
        b, B = self.b, self.g.block
        s = self.lane_sum0(y)
        s = b.rotate_add(s, s, B)
        s = b.rotate_add(s, s, 2 * B)
        sel = self.row_sel if scale is None else self.row_sel * scale
        s = b.norm(b.cmul(s, sel, cache_key=None if scale is not None else ('dec_rowsel',)))
        return self.lane_bcast0(s)

    def expand_rows(self, D):
        """Copy the row value from block ``4s`` onto ``4s+1`` and ``4s+2``.

        The source is masked first, so a domain fill on the other slots cannot leak, and both
        copies are rotations of that masked source (not of the updated sum).
        """
        b, B = self.b, self.g.block
        base = b.norm(b.cmul(D, self.row_blocks, cache_key=('dec_rowblk',)))
        return b.add(base, b.add(b.rotate(base, -B), b.rotate(base, -2 * B)))

    def _cover_domain(self, D, plan):
        """Put an in-range constant on every slot that does not hold a row sum."""
        lo, hi = float(plan.lower), float(plan.upper)
        fill = math.sqrt(lo * hi) if lo > 0.0 else 0.5 * (lo + hi)
        return self.b.add_vector(D, self.empty_rows * fill, cache_key=('dec_dom', lo, hi))

    def scatter(self, y):
        """Packed probabilities -> the three full-ring streams SV multiplies (block 0 copied everywhere)."""
        b, n = self.b, self.n
        B = self.g.block
        y = self._unshard(y)
        masks = np.concatenate([np.broadcast_to(self.scatter_mask[i], (n, SLOTS)) for i in range(STREAMS)], 0)
        z = b.norm(b.cmul(b.stack([y, y, y]), masks, cache_key=('dec_scatter',)))
        lo = self._broadcast_block0(b.take(z, 0, n))
        hi = self._broadcast_block0(b.rotate(b.take(z, n, 2 * n), B))
        ex = self._broadcast_block0(b.rotate(b.take(z, 2 * n, 3 * n), 2 * B))
        L, H, E = b.unstack(lo), b.unstack(hi), b.unstack(ex)
        return b.stack_aligned([c for s in range(n) for c in (L[s], H[s], E[s])])

    # ---- SoftMax (Attention.softmax on the decode batch) -------------------------------------
    def softmax(self, x):
        b, att = self.b, self.att
        x = self.pack_scores(x)
        R, eb = att.rounds, att.centered_bound
        mask, cnt = self.mask, self.count
        sm = self.sm
        plans, first_bound = sm['plans'], sm['first_bound']
        if sm['scaled'] is None:
            x = b.ready(x, 8, att.logit_bound)
            lm = b.cmul(x, mask / (cnt * eb), cache_key=('dec_sm_mean', eb))
            mean = self.expand_rows(self.reduce_rows(b.norm(lm)))
            normalized = b.sub(b.smul(x, 1 / eb), mean)
            y = b.poly(normalized, exp_coefficients(eb, R), mask=mask, mask_key=('dec_sm_exp', eb, R))
        else:
            coeffs, alpha, _ = sm['scaled']
            x = b.ready(x, 8, att.logit_bound)
            lm = b.cmul(x, alpha * mask / (cnt * eb), cache_key=('layer', 'dec_sm_mean'))
            mean = b.cmul(self.expand_rows(self.reduce_rows(b.norm(lm))), mask, cache_key=('dec_sm_mask',))
            z = b.sub(b.smul(x, alpha / eb), mean)
            z = b.add_vector(z, mask * alpha * sm['shift'] + (alpha - 1), cache_key=('layer', 'dec_sm_shift'))
            y = b.poly(z, coeffs, mask=mask, mask_key=('layer', 'dec_sm_exp'))
        for step in range(R):
            y = b.ready(y, 5, 2 * first_bound, fine=True) if step == 0 else b.ready(y, 3, 1.01)
            sq = b.square(y, rescale=True)
            D = self.reduce_rows(sq, scale=1 / cnt if step == 0 else None)
            inv = inverse_sqrt(b, self._cover_domain(D, plans[step]), plans[step],
                               scale=1 / math.sqrt(cnt) if step == 0 else 1.0)
            y = b.square(b.mul(y, self.expand_rows(inv), rescale=True))
        y = b.ready(y, 4, 1.01)
        return self.scatter(y)

    # ---- forward -----------------------------------------------------------------------
    def forward(self, q, k, v, cache, lane_total=True):
        """q: batch shards, k/v: batch kv_shards (the new token in every lane);
        cache: dict(k=(E_lo, E_hi), v=(V_lo, V_hi)) batches over Q shards -> attention output
        (batch shards, replicated over the lanes).  ``lane_total=False`` returns the lane sums
        instead (lane 0 of every run valid, one level higher; ``decode_mv`` masks its own lanes)."""
        b, g, att, n = self.b, self.g, self.att, self.n
        B, D = g.block, g.head_dim
        t = time.time()
        # Q: rope, product, extra-stream select.  K also pays the phase mask inside
        # expand_lo, same budget as prefill (Q 4, K 6).  The cache is stored at
        # floor+4 so the product is not aligned down onto a shorter ciphertext;
        # one short operand used to land the logits at 4 limbs, below S2C.
        q, k = b.trim(q, 4), b.trim(k, 6)
        qr = att._rope(q, self.rope_key, 1 / math.sqrt(D))
        kr = self.expand_lo(att._rope(k, self.rope_key))
        e_lo, e_hi = cache['k']
        # Prefill multiplies the three (query, key) streams in one batch.  The channel
        # sum replicates the scores, so the block-0 mask below (one level, the level the
        # old stack-alignment threw away) keeps a single copy of each stream.
        prod = b.sum_slots(b.mul(b.stack([qr, qr, qr]), b.stack_aligned([e_lo, e_hi, kr]), rescale=True), B, D)
        lo, hi, nw = (b.take(prod, i * n, (i + 1) * n) for i in range(3))
        kept = b.cmul(b.stack_aligned([lo, hi]), self.block0, cache_key=('dec_block0',))
        lo, hi = b.norm(b.take(kept, 0, n)), b.rotate(b.take(kept, n, 2 * n), -B)
        terms = [(nw, self.ex_lane[0] * self.block0, ('dec_exlane', 0))]
        for m in range(self.P):
            pub = b.sum_slots(b.norm(b.cmul(qr, self.pk_vec[m], cache_key=('layer', 'dec_pk', m))), B, D)
            terms.append((pub, self.ex_lane[1 + m] * self.block0, ('dec_exlane', 1 + m)))
        ex = b.rotate(b.norm(b.cmul_sum(terms)), -2 * B)
        x = b.add(lo, b.add(hi, ex))
        del prod, qr, kr, q, k
        t = self._mark("qk", t)
        p = self.softmax(x)
        t = self._mark("softmax", t)
        # ---- SV ----
        # SV output goes into the O projection's S2C, which needs the floor.
        # Product rescale and the lane-0 select each take one level, so P and V
        # enter the product two levels above the floor (V pays two masks first).
        p = b.trim(b.ready(p, 2, 1.01), 2)
        items = b.unstack(p)
        P = [b.stack([items[STREAMS * s + st] for s in range(n)]) for st in range(STREAMS)]
        v_lo, v_hi = cache['v']
        v_ex = b.norm(b.cmul(self.expand_lo(b.trim(v, 4)), self.lane0, cache_key=('dec_lane0',)))
        if self.P:
            v_ex = b.add_vector(v_ex, self.pv_ex, cache_key=('layer', 'dec_pv'))
        prod = b.mul(b.stack_aligned(P), b.stack_aligned([b.trim(v_lo, 2), b.trim(v_hi, 2), v_ex]), rescale=True)
        out = b.add(b.add(b.take(prod, 0, n), b.take(prod, n, 2 * n)), b.take(prod, 2 * n, 3 * n))
        out = b.norm(self.lane_total(out)) if lane_total else b.norm(self.lane_sum0(out))
        self._mark("sv", t)
        return out


class DecodeLayer:
    """``DecoderLayer`` pieces on the lo-only (replicated token) batch + ``DecodeAttention``."""

    def __init__(self, layer: DecoderLayer, position):
        self.layer, self.b, self.kit = layer, layer.b, layer.kit
        m = layer.cal.get("model", {})
        self.attention = DecodeAttention(layer.attention, position, float(m.get("rope_theta", 500000.0)),
                                         m.get("rope_scaling"))
        self.times = {}
        self.attention.times = self.times
        self._bucket = "proj"
        self.progress = layer.progress

    def _part(self, name, t0):
        import easyfhe as torch
        torch.cuda.synchronize()
        key = f"{self._bucket}_{name}"
        self.times[key] = self.times.get(key, 0.0) + time.time() - t0
        return time.time()

    def _t(self, name, t0):
        import easyfhe as torch
        torch.cuda.synchronize()
        self.times[name] = self.times.get(name, 0) + time.time() - t0
        if self.progress:
            print(f"  [{name}] {time.time() - t0:.2f}s", flush=True)
        return time.time()

    def rmsnorm(self, x):
        b, L = self.b, self.layer
        t0 = time.time()
        sq = b.unstack(b.square(x))
        s = b.sum_slots(b.add_many(sq), LANES, 512)
        t0 = self._part("sum", t0)
        inv = inverse_sqrt(b, s, L.rms_plan, 1.0 / L.hidden, L.eps, output_levels=1)
        out = b.mul(x, b.stack([inv] * x.batch_size), rescale=True)
        self._part("inv", t0)
        return out

    def coefficients(self, x, out_features):
        from .decoder import CoeffInput
        b, kit, L = self.b, self.kit, self.layer
        t0 = time.time()
        zs = [b._drop_for_s2c(it) for it in b.unstack(b.norm(x))]      # real S2C: the hi half is empty
        out = kit.to_coefficients_many(zs)
        self._part("s2c", t0)
        return CoeffInput(L.rhombus, out, out_features // kit.k)

    def project(self, coeff, name, bound=None):
        b, kit, L = self.b, self.kit, self.layer
        bound = L.bounds[name] if bound is None else bound
        adj = kit.raise_adjustment(coeff.blocks[0].state.cur_limbs)
        t0 = time.time()
        packed = L.rhombus.project(coeff.inputs(), L.dev[name], coeff.split, scale=adj / bound)
        t0 = self._part("pcmm", t0)
        out = [pair[0] for pair in kit.halfboots(packed, folded=True, real_only=True)]
        res = b.stack(out)
        res = b.smul(res, bound) if bound != 1 else res
        self._part("boot", t0)
        return res

    def silu_gate(self, gate, up):
        b, L = self.b, self.layer
        g, u = b.unstack(gate), b.unstack(up)
        out = [None] * len(g)
        for plan in sorted(set(L.silu_plans)):
            degree, domain = plan
            items = [s for s, p in enumerate(L.silu_plans) if p == plan]
            s = b.poly(b.stack([g[i] for i in items]), silu_coefficients(domain, degree))
            y = b.mul(b.ready(s, 1, 2 * domain), b.stack([u[i] for i in items]), rescale=True)
            for i, c in zip(items, b.unstack(y)):
                out[i] = c
        return b.stack_aligned(out)

    def forward(self, x, cache, probe=None):
        b, L, w = self.b, self.layer, self.layer.w
        probe = probe or (lambda name, ct: None)
        t = time.time()
        self._bucket = "attn_norm"
        n1 = self.rmsnorm(x); probe("attn_norm", n1); t = self._t("attn_norm", t)
        self._bucket = "qkv"
        cb = self.coefficients(n1, sum(np.shape(w[n])[0] for n in ("q", "k", "v")))
        q = self.project(cb, "q"); k = self.project(cb, "k"); v = self.project(cb, "v")
        del cb
        q, k = self.layer.qk_norm(q, "q"), self.layer.qk_norm(k, "k")
        probe("q", q); probe("k", k); probe("v", v); t = self._t("qkv", t)
        att = self.attention.forward(q, k, v, cache); probe("attention", att); t = self._t("attention", t)
        self._bucket = "o"
        o = self.project(self.coefficients(att, np.shape(w["o"])[0]), "o")
        h = b.add(x, o); probe("residual", h); t = self._t("o", t)
        self._bucket = "ffn_norm"
        n2 = self.rmsnorm(h); probe("ffn_norm", n2); t = self._t("ffn_norm", t)
        self._bucket = "gate_up"
        cb = self.coefficients(n2, 2 * np.shape(w["up"])[0])
        gate = self.project(cb, "gate", bound=2.0)
        up = self.project(cb, "up")
        del cb
        t = self._t("gate_up", t)
        gated = self.silu_gate(gate, up); probe("gated", gated); t = self._t("silu", t)
        self._bucket = "down"
        down = self.project(self.coefficients(gated, np.shape(w["down"])[0]), "down")
        out = b.add(h, down); probe("output", out); t = self._t("down", t)
        return out
