"""SC residual and per-head value rotations used by the frozen 8B baseline.

Adapted from fheForge/TurboFHE, commit
0e1ce0a2d9c51f22fecfab2c25ed31e039343319,
`turbofhe/apps/sylph/calibration_artifact.py` (BSD-3-Clause upstream declaration).
This CPU-only subset has no dependency on the TurboFHE execution backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def _fwht(value):
    out = np.array(value, dtype=np.float64, copy=True, order='C')
    size = out.shape[-1]
    if size < 1 or size & (size - 1):
        raise ValueError('orthogonal rotation needs a power-of-two axis')
    for bit in range(size.bit_length() - 1):
        step = 1 << bit
        blocks = out.reshape(-1, size // (2 * step), 2, step)
        left, right = blocks[:, :, 0].copy(), blocks[:, :, 1].copy()
        blocks[:, :, 0], blocks[:, :, 1] = left + right, left - right
    return out / math.sqrt(size)


@dataclass(frozen=True)
class OrthogonalRotation:
    signs: np.ndarray
    permutation: np.ndarray

    @classmethod
    def generate(cls, size, seed):
        if size < 1 or size & (size - 1):
            raise ValueError('rotation dimension must be a power of two')
        rng = np.random.default_rng(seed)
        return cls(2 * rng.integers(0, 2, size) - 1, rng.permutation(size))

    def apply(self, values, *, inverse=False):
        values = np.asarray(values)
        if values.shape[-1] != len(self.signs):
            raise ValueError('rotation and value dimensions disagree')
        if inverse:
            return _fwht(values[..., np.argsort(self.permutation)]) * self.signs
        return _fwht(values * self.signs)[..., self.permutation]


def convert_weights(weights, residual, value, *, hidden, heads, kv_heads, head_dim):
    """Fuse norm gamma, rotate the residual basis and the shared per-head V basis."""
    w, result = weights, {}
    for name in ('q', 'k', 'v', 'gate', 'up'):
        gamma = w['attention_norm' if name in ('q', 'k', 'v') else 'ffn_norm']
        result[name] = residual.apply(w[name] * gamma)
    v = result['v'].reshape(kv_heads, head_dim, hidden)
    result['v'] = value.apply(v.swapaxes(-1, -2)).swapaxes(-1, -2).reshape(w['v'].shape).copy()
    o = value.apply(w['o'].reshape(hidden, heads, head_dim)).reshape(w['o'].shape)
    result['o'] = residual.apply(o.T).T.copy()
    result['down'] = residual.apply(w['down'].T).T.copy()
    result['attention_norm'] = result['ffn_norm'] = np.ones(hidden)
    return result
