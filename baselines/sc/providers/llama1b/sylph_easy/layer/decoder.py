"""One Llama-3 decoder layer on EasyFHE, following TurboFHE apps/sylph model.DecoderLayer.forward:
RMSNorm -> Q/K/V -> RoPE/GQA/joint SoftMax/SV -> o + residual -> RMSNorm -> SiLU(gate) * up -> down + residual.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .approx import InverseSqrtPlan, range_grouped_order, silu_coefficients, silu_plan
from .layout import WIDTH, head_rms_gamma
from .attention import Attention
from .nonlinear import inverse_sqrt
from .qk_norm import apply_qk_norm
from ..easyfhe_ext.rhombus import Rhombus

SILU_HEADROOM = 3.0


def model_geometry(weights, calibration):
    """(heads, kv_heads, head_dim, hidden) from the calibration's ``model`` entry (1B chains) or
    the Llama-3-8B defaults (the upstream 8B chain carries no geometry)."""
    hidden = int(np.shape(weights["q"])[1])
    m = calibration.get("model", {})
    head_dim = int(m.get("head_dim", 128))
    heads = int(m.get("heads", np.shape(weights["q"])[0] // head_dim))
    kv_heads = int(m.get("kv_heads", np.shape(weights["k"])[0] // head_dim))
    if heads * head_dim != np.shape(weights["q"])[0] or kv_heads * head_dim != np.shape(weights["k"])[0]:
        raise ValueError("calibration model geometry disagrees with the projection weights")
    return heads, kv_heads, head_dim, hidden


def refresh_bound(value):
    """Power-of-two projection output bound with 2x headroom over a calibrated stage bound."""
    return float(2 ** max(0, math.ceil(math.log2(2 * value))))


class CoeffInput:
    """Coefficient blocks of one activation plus their Rhombus automorphic copies per split
    (shared by every projection reading the activation)."""

    def __init__(self, rhombus, blocks, out_blocks):
        self.rh, self.blocks, self.copies = rhombus, blocks, {}
        self.split = Rhombus.best_split(rhombus.h, int(math.log2(rhombus.k)), len(blocks), out_blocks)

    def inputs(self):
        if self.split not in self.copies:
            self.copies[self.split] = self.rh.inputs(self.blocks, self.split)
        return self.copies[self.split]


class DecoderLayer:
    def __init__(self, eng, weights, calibration, *, epsilon=None, rope_theta=None, progress=True, upload=True):
        self.b, self.kit = eng, eng.kit
        self.w, self.cal = weights, calibration
        m = calibration.get("model", {})
        self.eps = float(m.get("epsilon", 1e-5)) if epsilon is None else epsilon
        rope_theta = float(m.get("rope_theta", 500000.0)) if rope_theta is None else rope_theta
        self.heads, self.kv_heads, self.head_dim, self.hidden = model_geometry(weights, calibration)
        self.progress = progress
        sb = calibration["stage_bounds"]
        self.bounds = {name: refresh_bound(sb[name]) for name in ("q", "k", "v", "o", "up", "down")}
        self._group_ffn(calibration.get("stage_feature_bounds", {}).get("gate"))
        lo, hi = calibration["rms_interval"]
        self.rms_plan = InverseSqrtPlan.create(float(lo), float(hi), 127)
        self.qk_plan = None
        self.qk_gamma = None
        if "q_norm" in weights and "qk_rms_interval" in calibration:
            qlo, qhi = calibration["qk_rms_interval"]
            self.qk_plan = InverseSqrtPlan.create(float(qlo), float(qhi), 127)
            q_b, k_b = calibration.get("qk_output_bound", (None, None))
            self.qk_output_bound = {"q": None if q_b is None else float(q_b),
                                    "k": None if k_b is None else float(k_b)}
        self.attention = Attention(eng, public_keys=weights["public_keys"], public_values=weights["public_values"],
                                   logit_bound=float(calibration["logit_bound"]),
                                   centered_bound=float(calibration["centered_logit_bound"]),
                                   rounds=int(calibration["softmax_rounds"]), rope_theta=rope_theta,
                                   heads=self.heads, kv_heads=self.kv_heads, head_dim=self.head_dim,
                                   rope_scaling=m.get("rope_scaling"),
                                   inverse_degree=127, intervals=calibration.get("softmax_intervals"),
                                   head_intervals=calibration.get("softmax_head_intervals"))
        if self.qk_plan is not None:
            self.qk_gamma = {
                "q": head_rms_gamma(weights["q_norm"], self.attention.g),
                "k": head_rms_gamma(weights["k_norm"], self.attention.g),
            }
        self.rhombus = Rhombus(self.kit)
        # projection weights go to the GPU once per layer (runtime factors scale them there)
        self.dev = None
        if upload:
            self.upload()
        self.times = {}
        self._bucket = "proj"

    def _part(self, name, t0):
        import easyfhe as torch
        torch.cuda.synchronize()
        key = f"{self._bucket}_{name}"
        self.times[key] = self.times.get(key, 0.0) + time.time() - t0
        return time.time()

    def upload(self):
        """Projection weights to the GPU once per layer (ffn permutation and gate domains applied there)."""
        o = self.ffn_order
        div = self.gate_domains
        if o is not None:                  # gate_domains is in permuted order; divide before permuting
            div = np.empty_like(div)
            div[o] = self.gate_domains
        spec = dict(gate=dict(rows=o, row_div=div), up=dict(rows=o), down=dict(cols=o))
        self.dev = {name: self.rhombus.upload(self.w[name], **spec.get(name, {}))
                    for name in ("q", "k", "v", "o", "gate", "up", "down")}

    def _group_ffn(self, gate_bounds):
        """TurboFHE ffn_range_groups: permute the intermediate axis by calibrated gate range
        (polynomials.range_grouped_order) so each 512-feature shard gets its own SiLU plan.
        gate/up rows and down columns share the permutation, so the layer output is unchanged."""
        inter = np.shape(self.w["gate"])[0]
        self.ffn_order = None              # position f of the intermediate axis holds feature ffn_order[f]
        if gate_bounds is None:
            maxima = [float(self.cal["silu_bound"])] * (inter // WIDTH)
        else:
            order, maxima = range_grouped_order(gate_bounds, WIDTH)
            self.ffn_order = order         # gate/up rows and down columns are permuted on the GPU
        # Per-feature public maxima rest on far fewer samples than the layer-wide bound (the Paris
        # input exceeds them by up to 2x in layers 20-31): plan each shard at 3x its maximum,
        # capped at the layer-wide bound.
        cap = float(self.cal["silu_bound"])
        self.silu_plans = [silu_plan(min(SILU_HEADROOM * float(m), cap)) for m in maxima]
        domains = np.repeat([dom for _, dom in self.silu_plans], WIDTH)
        self.gate_domains = domains

    def silu_gate(self, gate, up):
        """SiLU(gate) * up per shard group with that group's (degree, domain) (TurboFHE model.silu)."""
        b = self.b
        g, u = b.unstack(gate), b.unstack(up)
        out = [None] * len(g)
        for plan in sorted(set(self.silu_plans)):
            degree, domain = plan
            items = [i for s, p in enumerate(self.silu_plans) if p == plan for i in (2 * s, 2 * s + 1)]
            s = b.poly(b.stack([g[i] for i in items]), silu_coefficients(domain, degree))
            y = b.mul(b.ready(s, 1, 2 * domain), b.stack([u[i] for i in items]), rescale=True)
            for i, c in zip(items, b.unstack(y)):
                out[i] = c
        return b.stack_aligned(out)

    def _t(self, name, t0):
        import easyfhe as torch
        torch.cuda.synchronize()
        self.times[name] = self.times.get(name, 0) + time.time() - t0
        if self.progress:
            print(f"  [{name}] {time.time() - t0:.2f}s", flush=True)
        return time.time()

    # ---- pieces ----------------------------------------------------------------------
    def rmsnorm(self, x):
        b = self.b
        t0 = time.time()
        sq = b.unstack(b.square(x))          # summed at degree 2, rescaled once in sum_slots
        lo, hi = b.add_many(sq[0::2]), b.add_many(sq[1::2])
        s = b.sum_slots(b.stack_aligned([lo, hi]), 64, 512)       # sum over 512 features per ciphertext
        t0 = self._part("sum", t0)
        inv = inverse_sqrt(b, s, self.rms_plan, 1.0 / self.hidden, self.eps, output_levels=1)
        lo, hi = b.unstack(inv)
        out = b.mul(x, b.stack([lo, hi] * (x.batch_size // 2)), rescale=True)
        self._part("inv", t0)
        return out

    def qk_norm(self, x, which):
        """Head RMS on a Q or K batch.  No-op unless this layer was calibrated with QK-Norm."""

        if self.qk_plan is None:
            return x
        t0 = time.time()
        out = apply_qk_norm(
            self.b, x, self.qk_gamma[which], self.qk_plan,
            eps=self.eps, block=self.attention.g.block, head_dim=self.head_dim,
            name=which, bound=self.qk_output_bound[which],
        )
        self._part("qk_norm", t0)
        return out

    def coefficients(self, x, out_features):
        b, kit = self.b, self.kit
        t0 = time.time()
        items = b.unstack(b.norm(x))
        zs = [kit.complex_pack(b._drop_for_s2c(items[j]), b._drop_for_s2c(items[j + 1]))
              for j in range(0, len(items), 2)]
        out = kit.to_coefficients_many(zs)
        self._part("s2c", t0)
        return CoeffInput(self.rhombus, out, out_features // self.kit.k)

    def project(self, coeff, name, bound=None):
        b = self.b
        kit = self.kit
        bound = self.bounds[name] if bound is None else bound
        # the HalfBoot raise correction rides on the weights: the PCMM rescale is the raise's
        adj = kit.raise_adjustment(coeff.blocks[0].state.cur_limbs)
        t0 = time.time()
        packed = self.rhombus.project(coeff.inputs(), self.dev[name], coeff.split, scale=adj / bound)
        t0 = self._part("pcmm", t0)
        out = [x for pair in kit.halfboots(packed, folded=True) for x in pair]
        res = b.stack(out)
        res = b.smul(res, bound) if bound != 1 else res
        self._part("boot", t0)
        return res

    # ---- forward ------------------------------------------------------------------------
    def forward(self, x, probe=None):
        b = self.b
        probe = probe or (lambda name, ct: None)
        self.attention.times = self.times
        t = time.time()
        self._bucket = "attn_norm"
        n1 = self.rmsnorm(x); probe("attn_norm", n1); t = self._t("attn_norm", t)
        self._bucket = "qkv"
        cb = self.coefficients(n1, sum(np.shape(self.w[n])[0] for n in ("q", "k", "v")))
        q = self.project(cb, "q"); k = self.project(cb, "k"); v = self.project(cb, "v")
        del cb
        q, k = self.qk_norm(q, "q"), self.qk_norm(k, "k")
        probe("q", q); probe("k", k); probe("v", v); t = self._t("qkv", t)
        att = self.attention.forward(q, k, v); probe("attention", att); t = self._t("attention", t)
        self._bucket = "o"
        o = self.project(self.coefficients(att, np.shape(self.w["o"])[0]), "o")
        h = b.add(x, o); probe("residual", h); t = self._t("o", t)
        self._bucket = "ffn_norm"
        n2 = self.rmsnorm(h); probe("ffn_norm", n2); t = self._t("ffn_norm", t)
        self._bucket = "gate_up"
        cb = self.coefficients(n2, 2 * np.shape(self.w["up"])[0])
        gate = self.project(cb, "gate", bound=2.0)
        up = self.project(cb, "up")
        del cb
        t = self._t("gate_up", t)
        gated = self.silu_gate(gate, up); probe("gated", gated); t = self._t("silu", t)
        self._bucket = "down"
        down = self.project(self.coefficients(gated, np.shape(self.w["down"])[0]), "down")
        out = b.add(h, down); probe("output", out); t = self._t("down", t)
        return out
