"""Head-wise RMSNorm on Q and K, in the coefficient-projection slot layout.

Qwen3 normalises each head over its ``head_dim`` channels before RoPE.
Inside one ciphertext a channel is one block (``hps * 64`` slots) and the
``hps`` phases of that block are the shard's heads, so a power-of-two
block sum is the per-head sum of squares, already replicated onto every
channel.  ``inverse_sqrt`` then supplies ``1/sqrt(mean + eps)`` and the
public gamma is a slot multiply.
"""
from __future__ import annotations

import math

import numpy as np

from .nonlinear import inverse_sqrt


def apply_qk_norm(b, x, gamma_slots, plan, *, eps, block, head_dim, name="qk", bound=None):
    """``x * gamma / sqrt(mean(x^2) + eps)`` with the mean taken over one head.

    The product and the gamma each spend a level, which leaves fewer levels than
    attention's QK depth (Q keeps 4, K keeps 6, then softmax refreshes).  The
    closing refresh restores a full ciphertext before that trim.  ``bound`` is
    the public maximum of this output; without one, a head of RMS 1 is at most
    ``sqrt(head_dim) * |gamma|``.
    """

    summed = b.sum_slots(b.square(x, rescale=True), int(block), int(head_dim))
    inv = inverse_sqrt(
        b, summed, plan,
        a=1.0 / int(head_dim), c=float(eps), scale=1.0, output_levels=3,
    )
    scaled = b.mul(x, inv, rescale=True)
    # The key has to carry the gamma itself: every layer shares the name "q"/"k",
    # and a limb-only cache would reuse the previous layer's table.
    gamma = np.ascontiguousarray(gamma_slots, dtype=np.float64)
    y = b.norm(b.cmul(scaled, gamma, cache_key=("qk_gamma", str(name), hash(gamma.tobytes()))))
    if bound is None:
        gamma_max = float(np.max(np.abs(np.asarray(gamma_slots, dtype=np.float64))))
        bound = math.sqrt(int(head_dim)) * max(gamma_max, 1e-6) * 1.5
    return b.ready(y, 6, float(bound))
