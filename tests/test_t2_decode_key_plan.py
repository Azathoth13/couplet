"""Decode / Prefill+Decode rotation-key plans of the T2 benches (CPU only)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bench.t2_decode import (
    decode_key_plan,
    decode_layouts,
    decode_schedule,
    resolve_attention_limb_cap,
    terminal_head_report,
)
from bench.t2_prefill_decode import (
    cached_prefill_only_rotations,
    combined_preset,
    decode_preset_and_schedule,
    emulated_phase_key_plans,
    guard_dropped_host_rotations,
    rank_key_plan,
    rank_phase_key_plans,
)
from llama3fhe.operators.attention.pairfold.pairfold_decode import (
    pv_pairfold_rotations,
    pv_probability_route_rotations,
)
from llama3fhe.parallel.decode.decode_keys import build_decode_key_plan
from llama3fhe.parallel.decode.decode_stations import decode_qkv_chunk_count
from llama3fhe.parallel.prefill.keys import KeyPlan, KeyTier, normalize_rotation, rotation_key_bytes
from llama3fhe.parallel.presets import preset_by_name

GIB = 2**30


def test_terminal_head_report_compares_fhe_and_poly_top1() -> None:
    class Head:
        def logits(self, hidden):
            return np.asarray(hidden, dtype=np.float32).reshape(-1)

    head = Head()
    poly = np.asarray([[0.0, 2.0, 1.0, 3.0, 4.0, 5.0]])
    matched = terminal_head_report(poly + 0.01, poly, head=head)
    assert matched["finite"] and matched["argmax_match"]
    assert matched["fhe_top_k"][0]["token_id"] == matched["poly_top_k"][0]["token_id"] == 5
    assert matched["top5_overlap"] == 5
    np.testing.assert_allclose(matched["logits_max_abs_diff"], 0.01, atol=1e-6)

    wrong = terminal_head_report(np.asarray([[7.0, 2.0, 1.0, 3.0, 4.0, 5.0]]), poly, head=head)
    assert not wrong["argmax_match"]
    assert wrong["fhe_top_k"][0]["token_id"] == 0
    invalid = terminal_head_report(np.asarray([[np.nan, 0.0]]), np.asarray([[1.0, 0.0]]), head=head)
    assert not invalid["finite"] and not invalid["argmax_match"]


def _resident_gib(plan, *, alpha: int = 9, special_limbs: int = 8) -> float:
    limits = plan.limb_limits()
    return sum(
        rotation_key_bytes(ring_dim=plan.ring_dim, limbs=limits.get(r, plan.full_limbs), alpha=alpha,
                           special_limbs=special_limbs)
        for r in plan.resident_rotations
    ) / GIB


def test_resolve_attention_limb_cap_defaults_to_the_preset_and_zero_means_uncapped() -> None:
    preset = preset_by_name("rtx6000pro-high")
    assert resolve_attention_limb_cap(preset, None) == 13
    assert resolve_attention_limb_cap(preset, 0) is None
    assert resolve_attention_limb_cap(preset, 11) == 11


def test_decode_plan_caps_attn_core_by_default() -> None:
    preset = preset_by_name("rtx6000pro-high")
    capped = decode_key_plan(preset, kv_limbs=12, attention_limb_cap=None, attn_resident=False, layers=range(32))
    uncapped = decode_key_plan(preset, kv_limbs=12, attention_limb_cap=0, attn_resident=False, layers=range(32))
    assert capped.tier("attn_core").limb_cap == 13
    assert uncapped.tier("attn_core").limb_cap is None
    # measured on the SL + sparse-BS layer (H100 W=8 emulation, L10 x 2 tokens, ): the attn_core
    # rotations run at <= 13 limbs (attention 10, kv_extract route 13), so 13 is the cap; far below the full chain
    assert 13 < capped.full_limbs
    assert len(capped.limb_limits()) > len(uncapped.limb_limits())


def test_llama32_1b_single_rank_decode_geometry_and_key_plan(monkeypatch) -> None:
    preset = replace(preset_by_name("rtx6000pro-high"), world_size=1, ciphers_per_rank=16,
                     mlp_feature_shard_k=1).for_model("llama3.2-1b")
    feature, attention, pairfold = decode_layouts(preset)
    assert (feature.hidden_dim, attention.head_dim, pairfold.carrier_dimension) == (2048, 64, 2048)
    assert pairfold.key_value_groups == 1 and pairfold.cache_cipher_count == 2
    schedule = decode_schedule(preset, kv_limbs=12)
    assert decode_qkv_chunk_count(schedule, pairfold) == 1
    plan = decode_key_plan(preset, kv_limbs=12, attention_limb_cap=None,
                           attn_resident=False, layers=range(16), silu_preset=preset)
    assert plan.tier("attn_core").limb_cap == 13
    assert len(plan.tier("linear")) > 0
    monkeypatch.setenv("LLAMA_FHE_DECODE_SL", "1")
    shifted = decode_schedule(preset, kv_limbs=12)
    assert decode_qkv_chunk_count(shifted, pairfold) == 4
    assert len(decode_key_plan(preset, kv_limbs=12, attention_limb_cap=None,
                               attn_resident=False, layers=(10,), silu_preset=preset).union) > 0


def test_decode_probability_route_keys_can_keep_more_limbs_than_other_attention_keys() -> None:
    preset = preset_by_name("rtx6000pro-high")
    feature, attention, pairfold = decode_layouts()
    schedule = decode_schedule(preset, kv_limbs=12)
    kwargs = dict(
        schedule=schedule,
        feature_layout=feature,
        attention_layout=attention,
        pairfold_layout=pairfold,
        attention_limb_cap=13,
    )
    baseline = build_decode_key_plan(preset, **kwargs)
    raised = build_decode_key_plan(preset, probability_route_limb_cap=19, **kwargs)
    route = {
        normalize_rotation(rotation, ring_dim=raised.ring_dim)
        for rotation in pv_probability_route_rotations(pairfold)
    }
    assert route <= {
        normalize_rotation(rotation, ring_dim=raised.ring_dim)
        for rotation in pv_pairfold_rotations(pairfold)
    }
    assert 17408 in route
    assert baseline.limb_limits()[17408] == 13
    assert raised.limb_limits()[17408] == 19
    assert set(raised.union) == set(baseline.union)
    assert raised.tier("attn_pv").rotation_set == route
    for rotation, limit in baseline.limb_limits().items():
        if rotation not in route:
            assert raised.limb_limits()[rotation] == limit


def test_combined_plan_cap_removes_the_full_chain_promotion_of_shared_attention_rotations() -> None:
    preset = combined_preset(preset_by_name("rtx6000pro-high"))
    decode_preset, schedule = decode_preset_and_schedule(preset, kv_limbs=12, shifted_lanes=True, sl_baby=16)
    layers = tuple(range(32))
    capped, sizes = rank_key_plan(preset, 0, decode_preset=decode_preset, decode_schedule=schedule, layers=layers)
    uncapped, _ = rank_key_plan(preset, 0, decode_preset=decode_preset, decode_schedule=schedule, layers=layers,
                                attention_limb_cap=0)
    assert set(capped.union) == set(uncapped.union)
    assert sizes["prefill_union"] > 0 and sizes["decode_union"] > 0
    # a rotation in any uncapped tier is stored at the full chain: the 8 x Pro 6000 PD_e2e run of
    # (uncapped) held 54.5 GiB of keys; the cap takes the estimate down by ~9 GiB
    saved = _resident_gib(uncapped) - _resident_gib(capped)
    assert saved > 6.0, saved
    assert _resident_gib(capped) < 48.0


def test_phase_prefill_plan_does_not_warm_decode_only_bootstrap_rotations() -> None:
    preset = combined_preset(preset_by_name("rtx6000pro-high"))
    decode_preset, schedule = decode_preset_and_schedule(preset, kv_limbs=12, shifted_lanes=True, sl_baby=16)
    merged, sizes, prefill_phase, decode = rank_phase_key_plans(
        preset, 0, decode_preset=decode_preset, decode_schedule=schedule, layers=tuple(range(32))
    )

    # The merged context still contains the complete union used by both
    # phases.  Only the warm-up target is stripped, so Decode can load its
    # bootstrap keys after the Prefill peak has passed.
    assert len(prefill_phase.tier("bootstrap")) < len(merged.tier("bootstrap"))
    assert len(prefill_phase.union) < sizes["prefill_union"]
    assert set(merged.union) == set(prefill_phase.union) | set(decode.union)


def test_emulated_phase_plans_cover_all_eight_ranks() -> None:
    preset = combined_preset(preset_by_name("rtx6000pro-high"))
    decode_preset, schedule = decode_preset_and_schedule(preset, kv_limbs=12, shifted_lanes=True, sl_baby=16)
    kwargs = dict(decode_preset=decode_preset, decode_schedule=schedule, layers=(0,))
    shared, shared_prefill, shared_decode = emulated_phase_key_plans(preset, **kwargs)

    assert int(preset.world_size) == 8
    for rank in range(int(preset.world_size)):
        merged, _, prefill, decode = rank_phase_key_plans(preset, rank, **kwargs)
        for rank_plan, shared_plan in (
            (merged, shared), (prefill, shared_prefill), (decode, shared_decode)
        ):
            assert set(rank_plan.union) <= set(shared_plan.union)
            rank_limits = rank_plan.limb_limits()
            shared_limits = shared_plan.limb_limits()
            assert all(shared_limits.get(rotation, shared_plan.full_limbs)
                       >= rank_limits.get(rotation, rank_plan.full_limbs)
                       for rotation in rank_plan.union)
    assert set(shared.union) == set(shared_prefill.union) | set(shared_decode.union)


def test_progressive_host_drop_requires_full_prefill_cache_and_excludes_decode_keys() -> None:
    prefill = KeyPlan(
        tiers=(KeyTier("bootstrap", 0, (5,), None), KeyTier("linear", 1, (1, 2, 3, 4), 5)),
        policy="resident", ring_dim=64, full_limbs=10,
    )
    decode = KeyPlan(tiers=(KeyTier("linear", 1, (4,), 5),),
                     policy="resident", ring_dim=64, full_limbs=10)

    class Native:
        alpha = 3
        auto_load_keys_resolved = True
        _rotation_key_cuda_cache = {
            (1, 5, 2): object(), (2, 4, 2): object(), (3, 5, 1): object(),
            (4, 5, 2): object(), (5, 10, 4): object(),
        }

        def _find_cached_rotation_key(self, rotation, limbs, beta):
            for (cached_rot, cached_limbs, cached_beta), value in self._rotation_key_cuda_cache.items():
                if cached_rot == rotation and cached_limbs >= limbs and cached_beta >= beta:
                    return value, cached_limbs
            return None

    native = Native()
    assert cached_prefill_only_rotations(native, prefill, decode) == (1, 5)
    native.auto_load_keys_resolved = False
    assert cached_prefill_only_rotations(native, prefill, decode) == ()


def test_dropped_host_rotation_guard_blocks_a_later_cache_miss() -> None:
    class Native:
        def _load_rotation_key_to_cuda(self, rotation, *_args):
            return int(rotation)

    native = Native()
    dropped = {1}
    guard_dropped_host_rotations(native, dropped)
    assert native._load_rotation_key_to_cuda(2, 5, 2, 10) == 2
    with pytest.raises(RuntimeError, match="rotation key 1 was dropped"):
        native._load_rotation_key_to_cuda(1, 6, 2, 10)
    dropped.add(2)
    guard_dropped_host_rotations(native, dropped)
    with pytest.raises(RuntimeError, match="rotation key 2 was dropped"):
        native._load_rotation_key_to_cuda(2, 6, 2, 10)
