"""CPU algebra check for W1/2/4/8 SC DecodeMV pc-group partitioning.

The test uses the production MVPlan's plaintext and PackRLWE algebra with a
small ring.  It needs NumPy but does not import EasyFHE or initialize a GPU.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from baselines.sc.providers.llama3b.sylph_easy.easyfhe_ext.rhombus_mv import MVPlan, _negacyclic
from baselines.sc.parallel.sc_decode_mv_pc import _build_partial_middle, pc_groups


def local_plaintexts(plan: MVPlan, weights: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Independent NumPy spelling of the rank-local middle builder."""

    n, m, a, pc = plan.n, plan.m, plan.a, plan.pc
    local_pc = len(groups)
    local_source = (np.arange(a)[:, None] * pc + groups[None, :]).reshape(-1)
    row_to_feature = np.full(plan.padded_rows, -1, dtype=np.int64)
    row_to_feature[plan.out_row] = np.arange(plan.rows)
    packed_weights = np.zeros((a * local_pc, n), dtype=np.float64)
    for j in range(n // plan.span):
        features = row_to_feature[j * m + local_source]
        selected = np.flatnonzero(features >= 0)
        pos = j * plan.span - plan.in_pos
        sign = np.where(pos < 0, -1.0, 1.0)
        packed_weights[selected[:, None], (pos % n)[None, :]] = weights[features[selected]] * sign
    out = np.zeros((local_pc, a, n), dtype=np.float64)
    for idx, automorphism in enumerate(plan.autos):
        src, sign = plan.automorphism_tables(automorphism)
        for i0 in range(a):
            block = packed_weights[i0 * local_pc:(i0 + 1) * local_pc][:, src] * sign
            out[:, idx] += plan.monomial_shift(block, plan.shift * i0, n)
    return out / m


def packed_from_partial(plan: MVPlan, x: np.ndarray, groups: np.ndarray,
                        plaintexts: np.ndarray) -> np.ndarray:
    """MAC local groups, pad missing groups with zero, then run PackRLWEs."""

    poly = np.zeros(plan.n)
    poly[plan.in_pos] = x
    copies = []
    for automorphism in plan.autos:
        src, sign = plan.automorphism_tables(automorphism)
        copies.append(poly[src] * sign)
    partials = [np.zeros(plan.n) for _ in range(plan.pc)]
    for local, p in enumerate(groups):
        partials[int(p)] = sum((_negacyclic(copies[i], plaintexts[local, i])
                                for i in range(plan.a)), np.zeros(plan.n))
    cur = partials
    for degree, step in plan.pack:
        automorphism = pow(5, step, 2 * plan.n)
        src, sign = plan.automorphism_tables(automorphism)
        half = len(cur) // 2
        next_partials = []
        for i in range(half):
            left = cur[i]
            right = plan.monomial_shift(cur[i + half], degree, plan.n)
            diff = left - right
            next_partials.append(left + right + diff[src] * sign)
        cur = next_partials
    return cur[0]


class SCDecodeMVPCAlgebraTest(unittest.TestCase):
    def test_actual_middle_builder_with_numpy_tensor_emulation(self):
        # Exercise the production indexing/automorphism loop itself, replacing
        # only its CUDA tensor primitives with NumPy arrays for this tiny ring.
        class Array(np.ndarray):
            def contiguous(self):
                return np.ascontiguousarray(self).view(Array)

        fake_torch = SimpleNamespace(
            as_tensor=lambda x, device=None: np.asarray(x).view(Array),
            zeros=lambda shape, dtype=None, device=None: np.zeros(shape, dtype=dtype).view(Array),
            cat=lambda parts, dim=0: np.concatenate(parts, axis=dim).view(Array),
            float64=np.float64,
        )
        plan = MVPlan(128, 32, 64, np.arange(64), np.r_[np.arange(16), 64 + np.arange(16)],
                      padded_rows=128)
        rng = np.random.default_rng(713)
        weights = rng.normal(0.0, 0.1, (plan.rows, plan.cols))
        row_scale = rng.uniform(0.75, 1.25, plan.rows)
        full = plan.plaintexts(weights * row_scale[:, None]).reshape(plan.pc, plan.a, plan.n)
        j = np.arange(plan.n // 2)
        inter = np.empty(plan.n, dtype=np.int64)
        inter[0::2], inter[1::2] = j, j + plan.n // 2
        rh = SimpleNamespace(_inter=inter)
        with patch.dict(sys.modules, {"easyfhe": fake_torch}):
            for world in (1, 2, 4, 8):
                for rank in range(world):
                    groups = pc_groups(plan, rank, world)
                    middle, bound = _build_partial_middle(rh, plan, weights, row_scale, groups)
                    np.testing.assert_allclose(middle.reshape(len(groups), plan.a, plan.n),
                                               full[groups][:, :, inter], rtol=0, atol=1e-12)
                    self.assertAlmostEqual(bound, np.max(np.abs(weights * row_scale[:, None])) *
                                           plan.a / plan.m)

    def test_rank_local_pc_tables_and_packed_addition(self):
        # N=128, span=64, padded_rows=128 gives m=64, a=8, pc=8: every
        # requested W partitions nonempty groups.  The weights have nonzero
        # rows in all eight p groups and use a nontrivial row-scale vector.
        plan = MVPlan(128, 32, 64, np.arange(64), np.arange(32), padded_rows=128)
        self.assertEqual(plan.pc, 8)
        rng = np.random.default_rng(20261115)
        weights = rng.normal(0.0, 0.12, (plan.rows, plan.cols))
        row_scale = rng.uniform(0.5, 1.5, plan.rows)
        scaled = weights * row_scale[:, None]
        x = rng.normal(0.0, 0.2, plan.cols)
        full_plaintexts = plan.plaintexts(scaled).reshape(plan.pc, plan.a, plan.n)
        expected_packed = plan.simulate(scaled, x)
        expected_output = scaled @ x

        for world in (1, 2, 4, 8):
            with self.subTest(world=world):
                groups = [pc_groups(plan, rank, world) for rank in range(world)]
                self.assertEqual(sorted(int(p) for owned in groups for p in owned), list(range(plan.pc)))
                packed = np.zeros(plan.n)
                for owned in groups:
                    local = local_plaintexts(plan, scaled, owned)
                    np.testing.assert_allclose(local, full_plaintexts[owned], rtol=0, atol=1e-12)
                    packed += packed_from_partial(plan, x, owned, local)
                np.testing.assert_allclose(packed, expected_packed, rtol=1e-11, atol=1e-10)
                np.testing.assert_allclose(plan.outputs(packed), expected_output, rtol=1e-10, atol=1e-10)

    def test_sparse_gate_up_output_rows(self):
        # Decode gate/up places the up rows at a power-of-two offset.  Those
        # output rows are noncontiguous inside the global padded MV plan.
        rows = np.r_[np.arange(16), 64 + np.arange(16)]
        plan = MVPlan(128, 32, 64, np.arange(64), rows, padded_rows=128)
        rng = np.random.default_rng(12095)
        weights = rng.normal(0.0, 0.1, (plan.rows, plan.cols))
        x = rng.normal(0.0, 0.2, plan.cols)
        full_plaintexts = plan.plaintexts(weights).reshape(plan.pc, plan.a, plan.n)
        packed = np.zeros(plan.n)
        for rank in range(8):
            owned = pc_groups(plan, rank, 8)
            local = local_plaintexts(plan, weights, owned)
            np.testing.assert_allclose(local, full_plaintexts[owned], rtol=0, atol=1e-12)
            packed += packed_from_partial(plan, x, owned, local)
        np.testing.assert_allclose(packed, plan.simulate(weights, x), rtol=1e-11, atol=1e-10)
        np.testing.assert_allclose(plan.outputs(packed), weights @ x, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
