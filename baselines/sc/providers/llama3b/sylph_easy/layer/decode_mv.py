"""Single-token decode layer of the Sylph (slot-coeff) scheme: one ciphertext per activation,
coefficient matrix-vector products (``easyfhe_ext.rhombus_mv``) instead of the prefill PCMM.

Layouts (N = 65536 ring, 32768 real slots; S2C maps slot j -> coefficient bitrev15(j), HalfBoot
maps coefficient c < N/2 -> slot bitrev15(c)):

* residual stream ``x``: feature f at coefficient ``XS * f`` (``XS`` = 1: slot 16 bitrev11(f) for a
  2048-wide model) - compact coefficients, so the Q/K/V and gate/up products read ``span = XS * hidden``;
* Q/K/V: one product over the concatenated [q; k; v] rows, output row f' at coefficient f'
  (stride 1), i.e. shard i (512 features) of the concatenation lands on lane ``bitrev6(i)`` of the
  prefill attention layout (feature r at slot bitrev9(r) * 64 + lane).  Lane masks + a lane rotation
  + the lane broadcast turn the one ciphertext into the per-shard, lane-replicated q/k/v batches
  ``layer.decode.DecodeAttention`` consumes;
* attention output: the lane-0 sums of the shards are placed on lanes bitrev6(s) of one ciphertext
  (coefficient r + 512 s after S2C: span = heads * head_dim), the o product writes the x layout;
* gate/up: one product over [gate; up] (stride 1), up row f at plan row P + f (P = next power of two
  of the intermediate width) so SiLU(gate) * up is a single slot rotation away; the gated product is
  then at coefficient f (span = intermediate width) for the down product.

Plaintext products per layer: span / stride of each product (Llama-3.2-1B: 2048 + 2048 + 2048 + 8192;
Llama-3-8B: 4096 + 4096 + 4096 + 16384), one HalfBoot per product.  Every HalfBoot output carries
garbage (partial sums) outside the plan's output coefficients; it is masked before any S2C or sum.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from .approx import InverseSqrtPlan, exp_coefficients, silu_coefficients, silu_plan
from .attention import Attention
from .decode import DecodeAttention, LANE_BITS
from .layout import LANES, SLOTS, bitrev, head_rms_gamma
from .nonlinear import inverse_sqrt
from .qk_norm import apply_qk_norm
from ..easyfhe_ext.rhombus_mv import MVPlan

N = 2 * SLOTS
LOG_SLOTS = int(math.log2(SLOTS))


def _ceil_pow2(v):
    return 1 << (int(v) - 1).bit_length() if v > 1 else 1


class MVGeometry:
    """Plans, slot layouts and masks of the single-token layer (model geometry only)."""

    def __init__(self, hidden, inter, heads, kv_heads, head_dim, *, xs=None):
        self.hidden, self.inter = int(hidden), int(inter)
        self.heads, self.kv_heads, self.head_dim = int(heads), int(kv_heads), int(head_dim)
        self.xs = int(os.environ.get("SYLPH_MV_XS", "1")) if xs is None else int(xs)
        H, I = self.hidden, self.inter
        q_rows, kv_rows = self.heads * self.head_dim, self.kv_heads * self.head_dim
        self.q_rows, self.kv_rows = q_rows, kv_rows
        # residual stream
        self.x_coeff = self.xs * np.arange(H)
        self.x_slot = bitrev(self.x_coeff, LOG_SLOTS)
        self.x_mask = np.zeros(SLOTS); self.x_mask[self.x_slot] = 1.0
        self.x_gap = SLOTS // _ceil_pow2(self.xs * H)         # slot spacing of the x layout
        self.sum_steps = [self.x_gap << k for k in range(int(math.log2(SLOTS // self.x_gap)))]
        # q/k/v concatenation: shard i -> lane bitrev6(i)
        self.n_q, self.n_kv = q_rows // 512, kv_rows // 512
        self.shards = self.n_q + 2 * self.n_kv
        if self.shards > LANES:
            raise ValueError("too many q/k/v shards for the 64 lanes")
        self.shard_lane = bitrev(np.arange(self.shards), LANE_BITS)
        lane = np.arange(SLOTS) % LANES
        self.lane_masks = np.stack([(lane == l).astype(float) for l in self.shard_lane])
        # attention output: q shard s on lane bitrev6(s) (the first n_q lanes above)
        self.att_lane = self.shard_lane[:self.n_q]
        self.att_masks = self.lane_masks[:self.n_q]
        # gate/up
        P = _ceil_pow2(I)
        if P + I > N // 2:
            raise ValueError("gate/up rows do not fit N/2 coefficients")
        self.gu_up_base = P
        self.gu_offset = bitrev(np.asarray([P]), LOG_SLOTS)[0]    # slot distance up - gate
        self.gate_slot = bitrev(np.arange(I), LOG_SLOTS)
        self.gate_mask = np.zeros(SLOTS); self.gate_mask[self.gate_slot] = 1.0
        # plans
        self.plans = {
            "qkv": MVPlan(N, q_rows + 2 * kv_rows, H, self.x_coeff, np.arange(q_rows + 2 * kv_rows), padded_rows=N),
            "o": MVPlan(N, H, q_rows, np.arange(q_rows), np.arange(H), padded_rows=N // self.xs),
            "gu": MVPlan(N, 2 * I, H, self.x_coeff, np.concatenate([np.arange(I), P + np.arange(I)]), padded_rows=N),
            "down": MVPlan(N, H, I, np.arange(I), np.arange(H), padded_rows=N // self.xs),
        }
        for name in ("o", "down"):
            assert np.array_equal(self.plans[name].out_coeff, self.x_coeff)
        want = (bitrev(np.arange(512), 9) * 64)[None, :] + self.att_lane[:, None]
        assert np.array_equal(bitrev(self.plans["qkv"].out_coeff[:q_rows], LOG_SLOTS), want.ravel())
        assert np.array_equal(bitrev(self.plans["gu"].out_coeff[I:], LOG_SLOTS), self.gate_slot + self.gu_offset)

    def rotations(self):
        rots = set()
        for plan in self.plans.values():
            rots |= plan.rotations()
        rots |= set(int(s) for s in self.sum_steps)
        rots |= {int(l) for l in self.shard_lane} | {(-int(l)) % SLOTS for l in self.att_lane}
        rots |= {1 << k for k in range(LANE_BITS)} | {(-(1 << k)) % SLOTS for k in range(LANE_BITS)}
        rots.add(int(self.gu_offset))
        rots.discard(0)
        return rots

    def describe(self):
        return {k: p.describe() for k, p in self.plans.items()}

    @property
    def plaintexts(self):
        return {k: p.m for k, p in self.plans.items()}

    # ---- host-side layout helpers -------------------------------------------------------------
    def encode_x(self, vector):
        v = np.zeros(SLOTS)
        v[self.x_slot] = np.asarray(vector, dtype=np.float64)
        return v

    def decode_x(self, slots):
        return np.asarray(slots)[self.x_slot]


class DeviceProjector:
    """S2C + RhombusMV + HalfBoot on EasyFHE for the plans of an ``MVGeometry``."""

    def __init__(self, eng, geo: MVGeometry):
        from ..easyfhe_ext.rhombus_mv import RhombusMV
        self.b, self.kit, self.geo = eng, eng.kit, geo
        self.rh = RhombusMV(self.kit)
        self.dev = {}
        self.groups = int(os.environ.get("SYLPH_MV_GROUPS", "0") or "0")

    def prepare(self, name, w, row_scale):
        """Weights of one plan to the GPU (untimed, like the prefill upload)."""
        middle, mx = self.rh.build(self.geo.plans[name], w, row_scale)
        self.dev[name] = (middle, mx)

    def release(self):
        self.dev = {}

    def coefficients(self, ct):
        b, kit = self.b, self.kit
        return kit.to_coefficients(b._drop_for_s2c(b.norm(ct)))

    def project(self, coeff, name):
        """Coefficient ciphertext -> HalfBoot slot ciphertext of ``plan`` (garbage outside out_coeff)."""
        kit, rh, plan = self.kit, self.rh, self.geo.plans[name]
        middle, mx = self.dev[name]
        adj = kit.raise_adjustment(int(coeff.state.cur_limbs))
        copies = rh.inputs(coeff, plan)
        packed = rh.project_middle(copies, middle, mx, plan, scale=adj, groups=self.groups)
        del copies
        return kit.halfboots([packed], folded=True, real_only=True)[0][0]

    @property
    def counts(self):
        return self.rh.counts


class DecodeMVLayer:
    """One decoder layer on the single-token layouts.  ``projector`` is a ``DeviceProjector`` (or
    the simulator in ``probes.check_decode_mv_sim``)."""

    def __init__(self, eng, weights, calibration, geo: MVGeometry, projector, position, *, progress=False,
                 upload=True):
        from .decoder import model_geometry, refresh_bound
        self.b, self.w, self.cal, self.geo, self.P = eng, weights, calibration, geo, projector
        m = calibration.get("model", {})
        self.eps = float(m.get("epsilon", 1e-5))
        rope_theta = float(m.get("rope_theta", 500000.0))
        heads, kv_heads, head_dim, hidden = model_geometry(weights, calibration)
        if (heads, kv_heads, head_dim, hidden) != (geo.heads, geo.kv_heads, geo.head_dim, geo.hidden):
            raise ValueError("layer geometry differs from the plan geometry")
        sb = calibration["stage_bounds"]
        self.bounds = {name: refresh_bound(sb[name]) for name in ("q", "k", "v", "o", "up", "down")}
        lo, hi = calibration["rms_interval"]
        self.rms_plan = InverseSqrtPlan.create(float(lo), float(hi), 127)
        self.silu = silu_plan(float(calibration["silu_bound"]))       # (degree, domain), one plan
        self.attention = Attention(eng, public_keys=weights["public_keys"], public_values=weights["public_values"],
                                   logit_bound=float(calibration["logit_bound"]),
                                   centered_bound=float(calibration["centered_logit_bound"]),
                                   rounds=int(calibration["softmax_rounds"]), rope_theta=rope_theta,
                                   heads=heads, kv_heads=kv_heads, head_dim=head_dim,
                                   rope_scaling=m.get("rope_scaling"), inverse_degree=127,
                                   intervals=calibration.get("softmax_intervals"),
                                   head_intervals=calibration.get("softmax_head_intervals"))
        self.qk_plan = None
        if "q_norm" in weights and "qk_rms_interval" in calibration:
            qlo, qhi = calibration["qk_rms_interval"]
            self.qk_plan = InverseSqrtPlan.create(float(qlo), float(qhi), 127)
            q_b, k_b = calibration.get("qk_output_bound", (None, None))
            self.qk_output_bound = {"q": None if q_b is None else float(q_b),
                                    "k": None if k_b is None else float(k_b)}
            self.qk_gamma = {
                "q": head_rms_gamma(weights["q_norm"], self.attention.g),
                "k": head_rms_gamma(weights["k_norm"], self.attention.g),
            }
        self.dec = DecodeAttention(self.attention, position, rope_theta, m.get("rope_scaling"))
        self.position = position
        self.times = {}
        self.dec.times = self.times
        self.progress = progress
        if upload:
            self.upload()

    # ---- weights ------------------------------------------------------------------------------
    def upload(self):
        w, geo, P = self.w, self.geo, self.P
        q_rows, kv_rows, I = geo.q_rows, geo.kv_rows, geo.inter
        _, domain = self.silu
        P.prepare("qkv", np.concatenate([w["q"], w["k"], w["v"]], 0),
                  np.concatenate([np.full(q_rows, 1 / self.bounds["q"]), np.full(kv_rows, 1 / self.bounds["k"]),
                                  np.full(kv_rows, 1 / self.bounds["v"])]))
        P.prepare("o", w["o"], np.full(geo.hidden, 1 / self.bounds["o"]))
        P.prepare("gu", np.concatenate([w["gate"], w["up"]], 0),
                  np.concatenate([np.full(I, 1 / (2.0 * domain)), np.full(I, 1 / self.bounds["up"])]))
        P.prepare("down", w["down"], np.full(geo.hidden, 1 / self.bounds["down"]))

    # ---- timing -------------------------------------------------------------------------------
    def _sync(self):
        try:
            import easyfhe as torch
            torch.cuda.synchronize()
        except Exception:
            pass

    def _t(self, name, t0):
        self._sync()
        self.times[name] = self.times.get(name, 0.0) + time.time() - t0
        if self.progress:
            print(f"  [{name}] {time.time() - t0:.3f}s", flush=True)
        return time.time()

    # ---- pieces -------------------------------------------------------------------------------
    def rmsnorm(self, x):
        """x in the x layout (zero elsewhere) -> x / sqrt(mean(x^2) + eps)."""
        b, geo, plan = self.b, self.geo, self.rms_plan
        s = b.norm(b.square(x))
        # The empty slot classes (slot mod x_gap != 0) must end on an in-range sum so the inverse
        # square root stays on its domain there: add the mid-interval mean per slot before the
        # cyclic sums (the encoder takes values up to 2; the summed constant would not fit).
        fill = min(max(math.sqrt(plan.lower * plan.upper), plan.lower), 1.5) - self.eps
        s = b.add_vector(s, (1.0 - geo.x_mask) * fill, cache_key=("mv_rms_fill", float(fill)))
        for step in geo.sum_steps:               # every x slot <- the full sum (cyclic cover)
            s = b.rotate_add(s, s, step)
        s = b.norm(s)
        if fill + self.eps < plan.lower:
            # The pre-sum fill is capped at 1.5 for encoding, but later layers can have a
            # larger RMS lower bound. Correct empty classes after scaling the sum down.
            s = b.norm(b.smul(s, 1.0 / (geo.hidden * plan.upper)))
            correction = (math.sqrt(plan.lower * plan.upper) - fill - self.eps) / plan.upper
            s = b.add_vector(s, (1.0 - geo.x_mask) * correction,
                             cache_key=("mv_rms_domain", float(correction)))
            inv = inverse_sqrt(b, s, plan, plan.upper, self.eps, output_levels=1)
        else:
            inv = inverse_sqrt(b, s, plan, 1.0 / geo.hidden, self.eps, output_levels=1)
        return b.mul(x, inv, rescale=True)

    def _project(self, ct, name):
        b, P = self.b, self.P
        t0 = time.time()
        c = P.coefficients(ct)
        t0 = self._t(f"{name}_s2c", t0)
        y = P.project(c, name)
        self._t(f"{name}_mv", t0)
        return y

    def split_qkv(self, y):
        """HalfBoot output of the q/k/v product -> (q batch n_q, k batch n_kv, v batch n_kv), every
        shard on its own ciphertext with the token replicated over the 64 lanes."""
        b, geo = self.b, self.geo
        n = geo.shards
        parts = b.cmul(b.stack([y] * n), geo.lane_masks, cache_key=("mv_lane_masks", n))
        parts = b.norm(parts)
        items = b.unstack(parts)
        items = [b.rotate(it, int(l)) for it, l in zip(items, geo.shard_lane)]      # lane bitrev6(i) -> lane 0
        x = b.stack(items)
        x = self.dec.lane_bcast0(x)
        q = b.smul(b.take(x, 0, geo.n_q), self.bounds["q"])
        k = b.smul(b.take(x, geo.n_q, geo.n_q + geo.n_kv), self.bounds["k"])
        v = b.smul(b.take(x, geo.n_q + geo.n_kv, n), self.bounds["v"])
        return q, k, v

    def qk_norm(self, x, which):
        """Normalize each Q/K head before DecodeAttention applies RoPE."""
        if self.qk_plan is None:
            return x
        return apply_qk_norm(
            self.b, x, self.qk_gamma[which], self.qk_plan,
            eps=self.eps, block=self.attention.g.block, head_dim=self.geo.head_dim,
            name=which, bound=self.qk_output_bound[which],
        )

    def combine_attention(self, out):
        """Batch over q shards with the attention output on lane 0 -> one ciphertext, shard s on
        lane bitrev6(s) (one level: the lane masks)."""
        b, geo = self.b, self.geo
        items = b.unstack(b.norm(out))
        items = [b.rotate(it, (-int(l)) % SLOTS) for it, l in zip(items, geo.att_lane)]
        x = b.cmul(b.stack(items), geo.att_masks, cache_key=("mv_att_masks", geo.n_q))
        return b.norm(b.add_many(b.unstack(x)))

    def silu_gate(self, y):
        """HalfBoot output of the gate/up product -> SiLU(gate) * up on the gate slots (zero elsewhere)."""
        b, geo = self.b, self.geo
        degree, domain = self.silu
        # gate / domain on the gate slots, zero elsewhere (the garbage must not enter the polynomial)
        g = b.norm(b.cmul(y, 2.0 * geo.gate_mask, cache_key=("mv_gate_mask2",)))
        u = b.smul(b.rotate(y, int(geo.gu_offset)), self.bounds["up"] * 2.0 * domain)   # up * 2 domain
        # the leaf plaintexts (coefficient x mask) are cached by key and survive clear_cache: key them
        # by the plan so a layer with another (degree, domain) does not reuse them
        # SiLU(gate) / (2 domain): keeps every leaf plaintext inside the encoder range (+-2);
        # the 2 domain factor is folded into the up operand
        coeffs = np.asarray(silu_coefficients(domain, degree)) / (2.0 * domain)
        s = b.poly(g, coeffs, mask=geo.gate_mask, mask_key=("mv_gate_mask", int(degree), round(float(domain), 9)))
        return b.mul(b.ready(s, 1, 1.0), u, rescale=True)

    # ---- forward ------------------------------------------------------------------------------
    def forward(self, x, cache, probe=None):
        b, geo = self.b, self.geo
        probe = probe or (lambda name, ct: None)
        t = time.time()
        n1 = self.rmsnorm(x); probe("attn_norm", n1); t = self._t("attn_norm", t)
        y = self._project(n1, "qkv")
        q, k, v = self.split_qkv(y); del y
        q, k = self.qk_norm(q, "q"), self.qk_norm(k, "k")
        probe("q", q); probe("k", k); probe("v", v); t = self._t("qkv", t)
        att = self.dec.forward(q, k, v, cache, lane_total=False); t = self._t("attention", t)
        a = self.combine_attention(att); probe("attention", a); del att
        o = self._project(a, "o")
        o = b.cmul(o, geo.x_mask, cache_key=("mv_x_mask",))
        o = b.norm(b.smul(o, self.bounds["o"]))
        h = b.add(x, o); probe("residual", h); t = self._t("o", t)
        n2 = self.rmsnorm(h); probe("ffn_norm", n2); t = self._t("ffn_norm", t)
        y = self._project(n2, "gu"); t = self._t("gate_up", t)
        gated = self.silu_gate(y); probe("gated", gated); t = self._t("silu", t)
        d = self._project(gated, "down")
        d = b.cmul(d, geo.x_mask, cache_key=("mv_x_mask",))
        d = b.norm(b.smul(d, self.bounds["down"]))
        out = b.add(h, d); probe("output", out); t = self._t("down", t)
        return out
