"""Encrypted inverse square root (TurboFHE Nonlinear.inverse_sqrt, Chebyshev initializer)."""
from __future__ import annotations

import math

import numpy as np


def inverse_sqrt(b, ct, plan, a=1.0, c=0.0, *, scale=1.0, output_levels=3):
    """scale / sqrt(a * ct + c) for a * ct + c in [plan.lower, plan.upper].

    u = (2 z - 1 - r) / (1 - r), z = (a ct + c) / upper, is the Chebyshev variable of the
    Lawson initializer; Newton y <- 1.5 y - (z / 2) y^3 refines it; the closing refresh is
    centered at the public midpoint of 1/sqrt(z).
    """
    r, up = plan.ratio, plan.upper
    u = b.add_scalar(b.smul(ct, 2 * a / ((1 - r) * up)), (2 * c / up - 1 - r) / (1 - r))
    half_z = b.smul(ct, 0.5 * a / up)
    if c:
        half_z = b.add_scalar(half_z, 0.5 * c / up)
    depth = math.ceil(math.log2(len(plan.coefficients))) + 1
    u = b.ready(u, depth, 1.0)
    y = b.poly(u, plan.coefficients)
    for step in range(plan.newton_steps):
        y = b.ready(y, 2, plan.refresh_bound)
        half_z = b.ready(half_z, 2, 1.0)
        cubic = b.mul(b.mul(half_z, y, rescale=True), b.square(y, rescale=True))
        y = b.norm(b.sub(b.smul(y, 1.5), cubic))
    if b.avail(y) < output_levels + 1:
        # Closing refresh of y is centered; one corrective Newton step y(1 - e/2) with the
        # small residual e = z y^2 - 1 refreshed on its own fine bound removes the refresh
        # noise of y (which is absolute, so large relative to small y).
        y_bound = 2.0 ** math.ceil(math.log2(plan.refresh_bound))
        y = b.refresh(y, plan.refresh_bound, center=plan.midpoint)
        half_z = b.ready(half_z, 2, 1.0)
        e = b.add_scalar(b.smul(b.mul(half_z, b.square(y, rescale=True)), 2), -1.0)
        e = b.refresh(e, 4 * y_bound * 3e-5, fine=True)
        y = b.norm(b.sub(y, b.mul(b.smul(y, 0.5), e)))
    s = np.asarray(scale, dtype=float) / math.sqrt(up)
    y = b.smul(y, float(s)) if s.ndim == 0 else b.cmul(y, s)
    b.count("inverse_sqrt")
    return b.norm(y)
