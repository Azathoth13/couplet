"""PairFoldLayout (general geometry) reproduces the Llama-3-8B PairFold128 constants and is
self-consistent on the Llama-3.2-1B geometry (head_dim 64)."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.pairfold128 import (
    PairFold128Layout,
    PairFoldLayout,
    pairfold_layout_for,
)

EIGHT_B = PairFold128Layout()
ONE_B_G2 = PairFoldLayout(seq_len=128, head_dim=64, query_heads=32, key_value_heads=8,
                          slots=32768, native_token_lanes=8, key_value_groups=2)
ONE_B_G1 = PairFoldLayout(seq_len=128, head_dim=64, query_heads=32, key_value_heads=8,
                          slots=32768, native_token_lanes=16, key_value_groups=1)


def test_8b_constants_are_the_historical_ones():
    lay = EIGHT_B
    assert (lay.pair_blocks, lay.head_lanes, lay.fold_dim, lay.gqa_ratio) == (2, 2, 64, 4)
    assert (lay.residues_per_group, lay.active_residue_count, lay.residue_slots) == (6, 12, 16)
    assert (lay.channel_row, lay.page_block, lay.carrier_dimension) == (32, 16384, 4096)
    assert lay.active_channel_count == 3072
    assert lay.query_residue_count == 8 and lay.cache_cipher_count == 4
    for group in range(2):
        assert lay.key_residue(group) == 6 * group
        assert lay.value_residue(group) == 6 * group + 1
        for qo in range(4):
            assert lay.query_residue(group, qo) == 6 * group + 2 + qo
            assert lay.query_cipher_index(group, qo) == group * 4 + qo
            for pair in range(2):
                assert lay.score_pair_index(group, pair, qo) == 8 * group + 4 * pair + qo
                assert lay.page_to_score_rotation(group, pair, qo) == pair * 16384 - 256 * (8 * group + 4 * pair + qo)
            for lane in range(2):
                assert lay.page_to_output_rotation(group, qo, lane) == -16 * (6 * group + 2 + qo) - 7 * lane
    for pair, feature, token, lane in itertools.product(range(2), range(0, 64, 7), range(0, 128, 11), range(2)):
        assert lay.page_slot(pair, feature, token, lane) == pair * 16384 + 2 * (feature * 128 + token) + lane
    for pair, feature, residue, lane in itertools.product(range(2), range(0, 64, 5), range(16), range(2)):
        assert lay.carrier_channel(pair, feature, residue, lane) == pair * 2048 + 32 * feature + 2 * residue + lane
    for residue, lane, token in itertools.product(range(12), range(2), range(0, 128, 9)):
        assert lay.carrier_to_page_rotation(residue, lane, token) == 16 * residue + 7 * lane - 2 * token
    for pair_index, token, lane in itertools.product(range(16), range(0, 128, 13), range(2)):
        group, rem = divmod(pair_index, 8)
        pair, qo = divmod(rem, 4)
        assert lay.score_slot(group, pair, qo, token, lane) == 2 * (pair_index * 128 + token) + lane
    assert lay.token_rotations() == tuple(2 * (1 << b) for b in range(7))
    assert lay.feature_rotations() == tuple(256 * (1 << b) for b in range(6))


def test_8b_pin_rejects_other_shapes():
    with pytest.raises(ValueError):
        PairFold128Layout(head_dim=64)


@pytest.mark.parametrize("lay", [EIGHT_B, ONE_B_G2, ONE_B_G1], ids=["8b", "1b_g2", "1b_g1"])
def test_rotations_match_slot_differences(lay):
    """The rotation helpers are exactly the slot distances they claim to bridge."""
    for group in range(lay.key_value_groups):
        residues = [lay.key_residue(group), lay.value_residue(group)] +\
                   [lay.query_residue(group, qo) for qo in range(lay.gqa_ratio)]
        for pair, lane in itertools.product(range(lay.pair_blocks), range(lay.head_lanes)):
            for feature in range(0, lay.fold_dim, 3):
                for residue in residues:
                    for token in range(0, lay.seq_len, 17):
                        assert (lay.carrier_slot(pair, feature, residue, lane)
                                - lay.page_slot(pair, feature, token, lane)
                                == lay.carrier_to_page_rotation(residue, lane, token))
                for qo in range(lay.gqa_ratio):
                    assert (lay.page_slot(pair, feature, 0, lane)
                            - lay.output_slot(group, pair, qo, feature, lane)
                            == lay.page_to_output_rotation(group, qo, lane))
            for qo in range(lay.gqa_ratio):
                for token in range(0, lay.seq_len, 17):
                    assert (lay.page_slot(pair, 0, token, lane) - lay.score_slot(group, pair, qo, token, lane)
                            == lay.page_to_score_rotation(group, pair, qo))
                    assert lay.score_to_page_rotation(group, pair, qo) == -lay.page_to_score_rotation(group, pair, qo)


@pytest.mark.parametrize("lay", [EIGHT_B, ONE_B_G2, ONE_B_G1], ids=["8b", "1b_g2", "1b_g1"])
def test_slot_maps_are_injective_and_in_range(lay):
    pages = set()
    for group in range(lay.key_value_groups):
        for pair, feature, token, lane in itertools.product(
                range(lay.pair_blocks), range(lay.fold_dim), range(lay.seq_len), range(lay.head_lanes)):
            slot = lay.page_slot(pair, feature, token, lane)
            assert 0 <= slot < lay.slots
            pages.add((group, slot))
    assert len(pages) == lay.key_value_groups * lay.pair_blocks * lay.fold_dim * lay.seq_len * lay.head_lanes
    channels = lay.active_carrier_channels()
    assert len(channels) == len(set(channels)) == lay.active_channel_count
    assert max(channels) < lay.carrier_dimension
    carrier = {lay.carrier_slot(0, 0, r, l, token_lane=t)
               for r in range(lay.residue_slots) for l in range(lay.head_lanes) for t in range(lay.native_token_lanes)}
    assert len(carrier) == lay.residue_slots * lay.head_lanes * lay.native_token_lanes == lay.channel_row * lay.native_token_lanes
    scores = set()
    for group, pair, qo, token, lane in itertools.product(
            range(lay.key_value_groups), range(lay.pair_blocks), range(lay.gqa_ratio), range(lay.seq_len), range(lay.head_lanes)):
        slot = lay.score_slot(group, pair, qo, token, lane)
        assert 0 <= slot < lay.slots
        scores.add(slot)
    assert len(scores) == lay.query_heads * lay.seq_len
    heads = {lay.query_head(g, p, l, qo) for g in range(lay.key_value_groups) for p in range(lay.pair_blocks)
             for l in range(lay.head_lanes) for qo in range(lay.gqa_ratio)}
    assert heads == set(range(lay.query_heads))
    kv_heads = {lay.key_value_head(g, p, l) for g in range(lay.key_value_groups) for p in range(lay.pair_blocks)
                for l in range(lay.head_lanes)}
    assert kv_heads == set(range(lay.key_value_heads))


@pytest.mark.parametrize("lay", [EIGHT_B, ONE_B_G2, ONE_B_G1], ids=["8b", "1b_g2", "1b_g1"])
def test_pack_unpack_round_trips(lay):
    rng = np.random.default_rng(1)
    q = rng.standard_normal((lay.query_heads, lay.head_dim))
    k = rng.standard_normal((lay.key_value_heads, lay.head_dim))
    v = rng.standard_normal((lay.key_value_heads, lay.head_dim))
    qq, kk, vv = lay.unpack_qkv_carrier(lay.pack_qkv_carrier(q, k, v))
    np.testing.assert_allclose(qq, q, atol=1e-12)
    np.testing.assert_allclose(kk, k, atol=1e-12)
    np.testing.assert_allclose(vv, v, atol=1e-12)
    np.testing.assert_allclose(lay.unpack_query(lay.pack_query(q)), q, atol=1e-12)
    page_k = rng.standard_normal((lay.seq_len, lay.key_value_heads, lay.head_dim))
    page_v = rng.standard_normal((lay.seq_len, lay.key_value_heads, lay.head_dim))
    assert lay.pack_key(page_k).shape == (lay.cache_cipher_count // 2, lay.slots)
    np.testing.assert_allclose(lay.unpack_key(lay.pack_key(page_k)), page_k, atol=1e-12)
    np.testing.assert_allclose(lay.unpack_value(lay.pack_value(page_v)), page_v, atol=1e-12)
    scores = rng.standard_normal((lay.query_heads, lay.seq_len))
    np.testing.assert_allclose(lay.unpack_scores(lay.pack_scores(scores)), scores, atol=1e-12)
    out = rng.standard_normal((lay.query_heads, lay.head_dim))
    np.testing.assert_allclose(lay.unpack_output(lay.pack_output(out)), out, atol=1e-12)


def test_layout_for_geometry():
    assert isinstance(pairfold_layout_for(head_dim=128, query_heads=32, key_value_heads=8), PairFold128Layout)
    one_b = pairfold_layout_for(head_dim=64, query_heads=32, key_value_heads=8)
    assert (one_b.key_value_groups, one_b.pair_blocks, one_b.native_token_lanes, one_b.residue_slots) == (1, 4, 16, 8)
    assert one_b.cache_cipher_count == 2 and one_b.pair_blocks * one_b.page_block == one_b.slots
    assert one_b.active_channel_count == 4 * 32 * 6 * 2
    two = pairfold_layout_for(head_dim=64, query_heads=32, key_value_heads=8, key_value_groups=2)
    assert (two.key_value_groups, two.pair_blocks, two.native_token_lanes) == (2, 2, 8)


@pytest.mark.parametrize("layout", [ONE_B_G1, ONE_B_G2], ids=["g1", "g2"])
def test_1b_scores_match_compact_decode_payload(layout):
    scores = np.arange(layout.query_heads * layout.seq_len, dtype=np.float32).reshape(
        layout.query_heads, layout.seq_len
    )
    compact = AttentionPairLayout(head_dim=64)
    np.testing.assert_array_equal(
        layout.pack_scores(scores, dtype=np.float32),
        compact.pack_compact_decode_scores(scores, dtype=np.float32),
    )
