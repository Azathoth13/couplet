"""Plaintext numpy reference of one decoder layer in the rotated residual basis (for checks)."""
from __future__ import annotations

import numpy as np

from .layout import rope_frequencies


def rmsnorm(x, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def qk_norm(x, gamma, eps, head_dim):
    """RMS over each head (``head_dim`` channels), then the per-channel gamma.

    ``x`` is ``[tokens, heads * head_dim]`` and ``gamma`` is ``[head_dim]``,
    shared by every head.  Absent on Llama; Qwen3 applies it to Q and K
    before RoPE.
    """

    x = np.asarray(x, dtype=np.float64)
    heads = x.reshape(x.shape[0], -1, int(head_dim))
    inv = np.asarray(gamma, dtype=np.float64) / np.sqrt(
        np.mean(heads * heads, axis=-1, keepdims=True) + float(eps)
    )
    return (heads * inv).reshape(x.shape)


def _maybe_qk_norm(q, k, w, eps, head_dim):
    if "q_norm" not in w:
        return q, k
    return (qk_norm(q, w["q_norm"], eps, head_dim),
            qk_norm(k, w["k_norm"], eps, head_dim))


def rope(x, positions, theta, head_dim=128, rope_scaling=None):
    t, f = x.shape
    x = x.reshape(t, -1, head_dim)
    freq = rope_frequencies(head_dim, theta, rope_scaling)
    ang = positions[:, None] * freq[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :head_dim // 2], x[..., head_dim // 2:]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], -1).reshape(t, f)


def layer(x, w, *, eps=1e-5, theta=500000.0, heads=32, kv_heads=8, head_dim=128, rope_scaling=None):
    stages = {}
    pk = np.asarray(w["public_keys"], dtype=np.float64).reshape(-1, kv_heads, head_dim)
    pv = np.asarray(w["public_values"], dtype=np.float64).reshape(-1, kv_heads, head_dim)
    P = pk.shape[0]
    T = x.shape[0]
    n1 = rmsnorm(x, eps) * w["attention_norm"]
    stages["attn_norm"] = n1
    q, k, v = n1 @ w["q"].T, n1 @ w["k"].T, n1 @ w["v"].T
    stages.update(q=q, k=k, v=v)
    q, k = _maybe_qk_norm(q, k, w, eps, head_dim)
    pos = P + np.arange(T, dtype=np.float64)
    q, k = rope(q, pos, theta, head_dim, rope_scaling), rope(k, pos, theta, head_dim, rope_scaling)
    q = q.reshape(T, heads, head_dim)
    k = k.reshape(T, kv_heads, head_dim)
    v = v.reshape(T, kv_heads, head_dim)
    out = np.zeros((T, heads, head_dim))
    for h in range(heads):
        g = h // (heads // kv_heads)
        keys = np.concatenate([pk[:, g], k[:, g]])
        vals = np.concatenate([pv[:, g], v[:, g]])
        s = q[:, h] @ keys.T / np.sqrt(head_dim)
        mask = np.concatenate([np.ones((T, P), bool), np.tril(np.ones((T, T), bool))], 1)
        s = np.where(mask, s, -np.inf)
        s = np.exp(s - s.max(1, keepdims=True))
        s /= s.sum(1, keepdims=True)
        out[:, h] = s @ vals
    att = out.reshape(T, -1)
    stages["attention"] = att
    hdn = x + att @ w["o"].T
    stages["residual"] = hdn
    n2 = rmsnorm(hdn, eps) * w["ffn_norm"]
    stages["ffn_norm"] = n2
    gate, up = n2 @ w["gate"].T, n2 @ w["up"].T
    gated = gate / (1 + np.exp(-gate)) * up
    stages["gated"] = gated
    y = hdn + gated @ w["down"].T
    stages["output"] = y
    return y, stages


def layer_from_calibration(x, w, cal):
    """``layer`` with the geometry / RoPE of a chain calibration (8B defaults without ``model``)."""
    m = cal.get("model", {})
    head_dim = int(m.get("head_dim", 128))
    return layer(x, w, eps=float(m.get("epsilon", 1e-5)), theta=float(m.get("rope_theta", 500000.0)),
                 heads=int(m.get("heads", np.shape(w["q"])[0] // head_dim)),
                 kv_heads=int(m.get("kv_heads", np.shape(w["k"])[0] // head_dim)),
                 head_dim=head_dim, rope_scaling=m.get("rope_scaling"))


def kv_cache(x_ctx, w, *, eps=1e-5, theta=500000.0, kv_heads=8, head_dim=128, rope_scaling=None):
    """(K after RoPE, V) of the 128 context tokens at positions P .. P + 127, each [128, kv_heads, head_dim]."""
    P = np.asarray(w["public_keys"]).reshape(-1, kv_heads, head_dim).shape[0]
    T = x_ctx.shape[0]
    n = rmsnorm(x_ctx, eps) * w["attention_norm"]
    k = n @ w["k"].T
    if "k_norm" in w:
        k = qk_norm(k, w["k_norm"], eps, head_dim)
    k = rope(k, P + np.arange(T, dtype=np.float64), theta, head_dim, rope_scaling)
    v = n @ w["v"].T
    return k.reshape(T, kv_heads, head_dim), v.reshape(T, kv_heads, head_dim)


def decode_layer(x, cache, w, *, position, eps=1e-5, theta=500000.0, heads=32, kv_heads=8, head_dim=128,
                 rope_scaling=None):
    """One decode token ``x`` [hidden] at sequence position ``position`` (= P + 128) against
    ``cache = (K_rope, V)`` from ``kv_cache`` plus the public prefix; every key is visible."""
    stages = {}
    pk = np.asarray(w["public_keys"], dtype=np.float64).reshape(-1, kv_heads, head_dim)
    pv = np.asarray(w["public_values"], dtype=np.float64).reshape(-1, kv_heads, head_dim)
    kc, vc = cache
    x = np.asarray(x, dtype=np.float64)[None, :]
    n1 = rmsnorm(x, eps) * w["attention_norm"]
    stages["attn_norm"] = n1
    q, k, v = n1 @ w["q"].T, n1 @ w["k"].T, n1 @ w["v"].T
    stages.update(q=q, k=k, v=v)
    q, k = _maybe_qk_norm(q, k, w, eps, head_dim)
    pos = np.asarray([float(position)])
    q = rope(q, pos, theta, head_dim, rope_scaling).reshape(heads, head_dim)
    k = rope(k, pos, theta, head_dim, rope_scaling).reshape(kv_heads, head_dim)
    v = v.reshape(kv_heads, head_dim)
    out = np.zeros((heads, head_dim))
    for h in range(heads):
        g = h // (heads // kv_heads)
        keys = np.concatenate([pk[:, g], kc[:, g], k[None, g]])
        vals = np.concatenate([pv[:, g], vc[:, g], v[None, g]])
        s = keys @ q[h] / np.sqrt(head_dim)
        s = np.exp(s - s.max())
        s /= s.sum()
        out[h] = s @ vals
    att = out.reshape(1, -1)
    stages["attention"] = att
    hdn = x + att @ w["o"].T
    stages["residual"] = hdn
    n2 = rmsnorm(hdn, eps) * w["ffn_norm"]
    stages["ffn_norm"] = n2
    gate, up = n2 @ w["gate"].T, n2 @ w["up"].T
    gated = gate / (1 + np.exp(-gate)) * up
    stages["gated"] = gated
    y = hdn + gated @ w["down"].T
    stages["output"] = y
    return y[0], stages


def decode_geometry(cal, w):
    """kwargs (eps, theta, heads, kv_heads, head_dim, rope_scaling) of a chain calibration."""
    m = cal.get("model", {})
    head_dim = int(m.get("head_dim", 128))
    return dict(eps=float(m.get("epsilon", 1e-5)), theta=float(m.get("rope_theta", 500000.0)),
                heads=int(m.get("heads", np.shape(w["q"])[0] // head_dim)),
                kv_heads=int(m.get("kv_heads", np.shape(w["k"])[0] // head_dim)),
                head_dim=head_dim, rope_scaling=m.get("rope_scaling"))
