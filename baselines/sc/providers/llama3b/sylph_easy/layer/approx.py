"""Public approximation planning for the Sylph layer (pure numpy).

The math follows TurboFHE apps/sylph (polynomials.py, inverse_sqrt.py and the
Newton planners in nonlinear.py): SiLU/exp Chebyshev interpolants, the Lawson
inverse-square-root initializer with its interval certificate, and the Newton
step count derived from that certificate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

cheb = np.polynomial.chebyshev


# ---- SiLU ---------------------------------------------------------------------
def silu_coefficients(bound, degree=31):
    return cheb.chebinterpolate(lambda y: bound * y / (1 + np.exp(-bound * y)), degree)


SILU_DEGREES = (15, 31, 63, 127, 255, 511, 1023)
SILU_TOLERANCE = 2e-5


def silu_error(bound, degree):
    grid = np.linspace(-1, 1, 16385)
    expected = bound * grid / (1 + np.exp(-np.clip(bound * grid, -700, 700)))
    return float(np.max(np.abs(cheb.chebval(grid, silu_coefficients(bound, degree)) - expected)))


@lru_cache(maxsize=None)
def silu_domain(degree, tolerance=SILU_TOLERANCE):
    """Widest public bound whose degree-``degree`` SiLU stays within ``tolerance``."""
    lo, hi = 1e-3, 512.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if silu_error(mid, degree) <= tolerance else (lo, mid)
    return lo


def silu_plan(bound, tolerance=SILU_TOLERANCE):
    """(degree, domain) of the lowest degree whose certified domain covers ``bound``."""
    for degree in SILU_DEGREES:
        domain = silu_domain(degree, tolerance)
        if bound <= domain:
            return degree, domain
    raise ValueError(f"SiLU bound {bound} exceeds every planned degree")


def range_grouped_order(bounds, width):
    """TurboFHE polynomials.range_grouped_order: features sorted by descending bound fill
    consecutive groups of ``width``; returns (order, per-group maxima)."""
    bounds = np.asarray(bounds, dtype=float)
    order = np.argsort(-bounds, kind='stable')
    maxima = tuple(float(bounds[order[f:f + width]].max()) for f in range(0, len(bounds), width))
    return order, maxima


# ---- exponential -----------------------------------------------------------------
def exp_coefficients(bound, rounds):
    """exp((B*y - B) / 2**rounds) on y in [-1, 1], degree 15."""
    return cheb.chebinterpolate(lambda y: np.exp((bound * y - bound) / 2 ** rounds), 15)


SCALED_EXP_DEGREES = (15, 31)


def scaled_exp_plan(bound, rounds, max_count, tolerance=1e-6):
    """TurboFHE polynomials.scaled_exp_plan: the exponential over the head-scaled domain.

    Per-head values f_h * exp((B y - B) / 2**R) (f_h >= 1, applied as an input shift) are
    covered for y' in [-1, 1 + c], mapped onto [-1, 1] by z = alpha y' + alpha - 1.
    Returns (coefficients, alpha, c) for the lowest degree within ``tolerance`` relative
    to the domain's largest value, or None."""
    c = 2 ** rounds * math.log(1.05 * math.sqrt(max_count)) / bound
    alpha = 2 / (2 + c)
    f = lambda z: np.exp((bound * ((z + 1) / alpha - 1) - bound) / 2 ** rounds)
    grid = np.linspace(-1, 1, 16385)
    top = float(f(1.0))
    for degree in SCALED_EXP_DEGREES:
        coeffs = cheb.chebinterpolate(f, degree)
        if np.max(np.abs(cheb.chebval(grid, coeffs) - f(grid))) / top <= tolerance:
            return coeffs, alpha, c
    return None


def softmax_rounds(bound, max_keys, auxiliary_bound=128):
    rounds = 2
    while math.log(max_keys) + 4 * bound / 2 ** rounds > 2 * math.log(auxiliary_bound) - math.log(1.02 / 0.9):
        rounds += 1
    return rounds


# ---- inverse square root -----------------------------------------------------------
INVERSE_SQRT_DEGREES = (2, 4, 8, 15, 16, 31, 32, 63, 64, 127, 128)


def _inverse_sqrt_point(u, ratio):
    return ((1 - ratio) * u + 1 + ratio) / 2


@lru_cache(maxsize=64)
def inverse_sqrt_coefficients(ratio, degree):
    """Chebyshev p(u) ~ z**-1/2, z in [ratio, 1] (Lawson IRLS on relative error)."""
    samples = 4 * (degree + 1)
    u = np.cos((np.arange(samples) + 0.5) * (math.pi / samples))
    basis = cheb.chebvander(u, degree) * np.sqrt(_inverse_sqrt_point(u, ratio))[:, None]
    weights = np.full(samples, 1 / samples)
    best, best_error = None, math.inf
    for _ in range(100):
        root = np.sqrt(weights)
        coefficients = np.linalg.lstsq(basis * root[:, None], root, rcond=None)[0]
        residual = np.abs(1 - basis @ coefficients)
        error = float(np.max(residual))
        if error < best_error:
            best, best_error = coefficients, error
        total = float(np.sum(weights * residual))
        if not total > 0:
            break
        weights = weights * residual / total
    return np.asarray(best, dtype=np.float64)


def inverse_sqrt_error_bound(ratio, c, samples=1 << 18):
    """Certified max|1 - sqrt(z) p(u)| over z in [ratio, 1]."""
    k = np.arange(len(c), dtype=np.float64)
    q0, q1, q2 = (float(np.sum(k ** p * np.abs(c))) for p in (0, 1, 2))
    u = np.cos(np.arange(samples + 1) * (math.pi / samples))
    sampled = float(np.max(np.abs(1 - np.sqrt(_inverse_sqrt_point(u, ratio)) * cheb.chebval(u, c))))
    curvature = q0 / (2 * math.sqrt(ratio)) + q1 + q2
    slack = (math.pi / samples) ** 2 / 8 * curvature
    return math.nextafter(sampled + slack + 64 * np.finfo(float).eps * (1 + q0 + q1), math.inf)


def newton_steps(initial, tolerance):
    error, steps = initial, 0
    while error > tolerance and steps < 32:
        error = math.nextafter(error * error * ((3 + error) if steps == 0 else (3 - error)) / 2, math.inf)
        steps += 1
    if error > tolerance:
        raise ValueError("inverse square root did not converge")
    return steps, error


def newton_interval_steps(initial, tolerance, ceiling=1.65, limit=48):
    gain = min(1.0, ceiling / (1 + initial))
    low, high = gain * (1 - initial), gain * (1 + initial)
    g = lambda t: t * (3 - t * t) / 2
    pad = lambda v, d: math.nextafter(v * (1 + d * 4e-16), d * math.inf)
    for steps in range(1, limit + 1):
        values = (g(low), g(high))
        low, high = pad(min(values), -1), pad(1.0 if low <= 1 <= high else max(values), 1)
        error = max(1 - low, high - 1)
        if error <= tolerance:
            return gain, steps, error
    raise ValueError("interval Newton did not reach the tolerance")


@dataclass(frozen=True)
class InverseSqrtPlan:
    lower: float
    upper: float
    coefficients: np.ndarray      # Chebyshev series in u, already scaled by the Newton gain
    newton_steps: int
    refresh_bound: float
    initial_error: float
    final_error: float

    @property
    def ratio(self):
        return self.lower / self.upper

    @property
    def midpoint(self):
        return (1 + math.sqrt(self.upper / self.lower)) / 2

    @classmethod
    @lru_cache(maxsize=64)
    def create(cls, lower, upper, degree=63, tolerance=1e-12):
        ratio = lower / upper
        c = inverse_sqrt_coefficients(ratio, degree)
        initial = inverse_sqrt_error_bound(ratio, c)
        if initial < 0.5:
            steps, error = newton_steps(initial, tolerance)
            return cls(lower, upper, c, steps, math.sqrt(upper / lower) * 1.01 * (1 + initial), initial, error)
        gain, steps, error = newton_interval_steps(initial, tolerance)
        return cls(lower, upper, gain * c, steps, math.sqrt(upper / lower) * 1.01 * gain * (1 + initial),
                   initial, error)
