"""RMSNorm as plaintext mathematics: exact reference and fitted coefficients.

``rmsnorm_numpy`` is the exact function; ``is_rmsn_weight`` recognizes the
gamma vector QuaRot fusion zeroes out.

``rmsnorm_poly_numpy`` is the reference the encrypted RMSNorm is compared
against: it evaluates the same ``poly(mean(x**2))`` composition the ciphertext
path does, so a mismatch isolates an FHE scheduling bug from an approximation
error. The loaders read the release coefficient artifact under
``assets/polynomials/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .chebyshev import eval_chebyshev_series_direct


def rmsnorm_token_prescale(entry: Dict[str, Any] | None, seq_len: int) -> np.ndarray | None:
    """Per-token public pre-scale ``c[t]`` of the mean square (``None`` = all ones).

    The fitted entry may carry

    * ``"mean_square_prescale": K`` -- one public factor on every token, and/or
    * ``"token_prescale": {"<position>": c, ...}`` -- extra factors on known
      prompt positions (the Llama attention-sink / BOS token at position 0 has
      a mean square ~1e5x the other tokens; scaling it into the common range
      keeps the rsqrt Chebyshev degree small).

    The polynomial is then evaluated on ``c[t] * s[t]`` and its output is
    multiplied back by ``sqrt(c[t])`` (``rsqrt(c s) * sqrt(c) = rsqrt(s)``),
    so the result is mathematically the plain RMSNorm; only the fit interval
    (and hence the degree) changes.  Both factors are public, position-based
    constants fused into plaintext masks the graph already applies.
    """

    if not entry:
        return None
    uniform = float(entry.get("mean_square_prescale", 1.0))
    positions = entry.get("token_prescale") or {}
    if not positions and uniform == 1.0:
        return None
    if not np.isfinite(uniform) or uniform <= 0.0:
        raise ValueError(f"mean_square_prescale must be finite and positive, got {uniform}.")
    scale = np.full((int(seq_len),), uniform, dtype=np.float64)
    for key, value in dict(positions).items():
        position = int(key)
        factor = float(value)
        if not 0 <= position < int(seq_len):
            raise ValueError(f"token_prescale position {position} is outside [0, {seq_len}).")
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError(f"token_prescale[{position}] must be finite and positive, got {factor}.")
        scale[position] *= factor
    return scale


def rmsnorm_poly_numpy(
    x: np.ndarray,
    coefficients: Sequence[float],
    fit_interval: Sequence[float],
    *,
    skip_clamp: bool = True,
    token_prescale: np.ndarray | None = None,
) -> np.ndarray:
    """NumPy polynomial RMSNorm reference.

    Parameters
    ----------
    x : np.ndarray  shape [seq, hidden]
    coefficients : Chebyshev series for ``rsqrt(mean(x**2) + eps)`` — the
        stability epsilon (1e-5) is baked into the fitted target.
    fit_interval : [lo, hi] for the Chebyshev domain.
    skip_clamp : if True (FHE mode), skip clamping ``s`` before poly eval.
    token_prescale : optional per-token ``c[t]`` (``rmsnorm_token_prescale``):
        the polynomial sees ``c[t] * mean(x**2)`` and its output is scaled by
        ``sqrt(c[t])``.
    """
    x = np.asarray(x, dtype=np.float32)
    coeffs = np.asarray(coefficients, dtype=np.float64)
    fit_lo = float(fit_interval[0])
    fit_hi = float(fit_interval[1])

    variance = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True)
    undo = None
    if token_prescale is not None:
        prescale = np.asarray(token_prescale, dtype=np.float64).reshape(-1, 1)
        if prescale.shape[0] != variance.shape[0]:
            raise ValueError(
                f"token_prescale must have one factor per token ({variance.shape[0]}), got {prescale.shape[0]}."
            )
        variance = (variance.astype(np.float64) * prescale)
        undo = np.sqrt(prescale)
    if not skip_clamp:
        variance = np.clip(variance, fit_lo, fit_hi)

    scale = eval_chebyshev_series_direct(variance, coeffs, fit_lo, fit_hi)
    if undo is not None:
        scale = np.asarray(scale, dtype=np.float64) * undo
    return (x * scale).astype(np.float32, copy=False)


def load_rmsnorm_poly_coeffs(path: str | Path) -> Dict[str, Any]:
    """Load the RMSNorm polynomial coefficients JSON artifact.

    Returns the full dict (``branches.attn["0"]``, ``branches.ffn["0"]``, ...).
    """
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def get_layer_coeffs(
    artifact: Dict[str, Any],
    branch: str,
    layer_idx: int | str,
) -> Dict[str, Any]:
    """Extract per-layer coefficient entry from the artifact.

    *branch* is ``"attn"``, ``"ffn"``, or ``"final"``.
    *layer_idx* is ``"0"``, ``"1"``, ... or ``"final"`` for the final norm.
    """
    branches = artifact.get("branches", {})
    if branch not in branches:
        available = list(branches.keys())
        raise KeyError(f"Branch {branch!r} not found in artifact (available: {available}).")
    layer_map = branches[branch]
    key = str(layer_idx)
    if key not in layer_map:
        available = list(layer_map.keys())
        raise KeyError(f"Layer {key!r} not found in branch {branch!r} (available: {available}).")
    return layer_map[key]


def is_rmsn_weight(weight: np.ndarray, *, atol: float = 0.0) -> bool:
    """Return True when QuaRot fuse_layer_norms zeroed out the RMSNorm gamma vector.

    QuaRot checkpoints may store a scalar dummy ``weight`` of shape ``(1,)`` after fusion.
    """

    weight = np.asarray(weight, dtype=np.float32).reshape(-1)
    if weight.size == 0:
        return True
    if weight.size == 1:
        return float(np.abs(weight[0])) <= float(atol)
    return float(np.max(np.abs(weight))) <= float(atol)


def rmsnorm_numpy(values: np.ndarray, weight: np.ndarray, *, eps: float) -> np.ndarray:
    """Row RMSNorm reference used by baseline and QuaRot (RMSN) paths.

    Baseline HF: ``out = x / RMS(x) * gamma``.
    QuaRot RMSN: ``gamma == 0`` after ``fuse_layer_norms``; output is ``x / RMS(x)`` only.
    """

    values = np.asarray(values, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32).reshape(-1)
    if values.ndim != 2:
        raise ValueError(f"RMSNorm values must be [seq, width], got {values.shape}.")
    rmsn = is_rmsn_weight(weight)
    if not rmsn and weight.shape[0] != values.shape[1]:
        raise ValueError(f"RMSNorm weight width {weight.shape[0]} does not match values {values.shape[1]}.")
    variance = np.mean(np.square(values, dtype=np.float32), axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(variance + np.float32(eps))
    normalized = values * inv_rms
    if rmsn:
        return normalized.astype(np.float32, copy=False)
    return (normalized * weight).astype(np.float32, copy=False)


__all__ = [
    "get_layer_coeffs",
    "is_rmsn_weight",
    "load_rmsnorm_poly_coeffs",
    "rmsnorm_numpy",
    "rmsnorm_poly_numpy",
    "rmsnorm_token_prescale",
]
