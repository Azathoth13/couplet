"""``k >= 3`` merged-page Decode Softmax as block plans (``LLAMA_FHE_SOFTMAX_BLOCKPLAN``).

``s2,d1`` (k=3) and ``s2,d2`` (k=4) are algebraically Alg2 (every rsqrt sees
Alg2's round input), so the probabilities match the Alg2 merged kernel; the
normalisation multiplies and ``lambda`` bootstraps go 3 -> 2 and 4 -> 2.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_decode_softmax_merged_pages import (
    _context,
    _CountingRefresh,
    _decrypt,
    _layout,
)

from llama3fhe.backend import release_if_supported
from llama3fhe.operators.attention.decode.decode_softmax import (
    BOOTSTRAP_MIN_INPUT_LIMBS,
    SCHAIN_PROBABILITY_MIN_LIMBS,
    SOFTMAX_BLOCKPLAN_ENV,
    SOFTMAX_BLOCKPLAN_FORCE_ENV,
    SOFTMAX_MERGE_PAGES_ENV,
    SOFTMAX_PV_NEED_LIMBS_ENV,
    SOFTMAX_SCHAIN_ENV,
    DecodeAlg2Poly,
    blockplan_applicable,
    blockplan_limb_schedule,
    parse_softmax_blockplan,
    probability_min_limbs,
    pv_probability_need_limbs,
    softmax_alg2_row_pages_fhe,
    softmax_blockplan_for,
)


def test_parse_blockplan() -> None:
    assert parse_softmax_blockplan("3:s2,d1;4:s2,d2") == {
        3: (("s", 2), ("d", 1)),
        4: (("s", 2), ("d", 2)),
    }
    assert parse_softmax_blockplan(" 5 : s1 , d2 , d2 ") == {5: (("s", 1), ("d", 2), ("d", 2))}
    assert parse_softmax_blockplan("") == {}
    for bad in ("3:s2,d2", "3:d1,s2", "3:s3", "3:x2,d1", "s2,d1", "4:s2,d1"):
        with pytest.raises(ValueError):
            parse_softmax_blockplan(bad)


def test_blockplan_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SOFTMAX_BLOCKPLAN_ENV, raising=False)
    assert softmax_blockplan_for(3) is None
    monkeypatch.setenv(SOFTMAX_BLOCKPLAN_ENV, "3:s2,d1;4:s2,d2")
    assert softmax_blockplan_for(3) == (("s", 2), ("d", 1))
    assert softmax_blockplan_for(4) == (("s", 2), ("d", 2))
    assert softmax_blockplan_for(5) is None  # unlisted k keeps Alg2
    assert softmax_blockplan_for(2) is None


def test_blockplan_limb_schedule_and_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SOFTMAX_BLOCKPLAN_ENV, "3:s2,d1;4:s2,d2")
    monkeypatch.delenv(SOFTMAX_BLOCKPLAN_FORCE_ENV, raising=False)
    config = DecodeAlg2Poly.synthetic(k=3, degree=8)  # PS depth 4 (+1 unfused map)
    plan3 = parse_softmax_blockplan("3:s2,d1")[3]
    sched = blockplan_limb_schedule(config, 0, plan3, k=3, exit_limbs=18)
    # s2 pack + d1 pack: two aux bootstraps after post-exp, no pending scalar at the end
    assert sched["bootstraps"] == 2, sched
    assert sched["min_pack"] >= BOOTSTRAP_MIN_INPUT_LIMBS
    assert sched["probability"] >= SCHAIN_PROBABILITY_MIN_LIMBS
    assert blockplan_applicable(config, 0, 3, exit_limbs=18) == plan3
    plan4 = parse_softmax_blockplan("4:s2,d2")[4]
    sched4 = blockplan_limb_schedule(config, 0, plan4, k=4, exit_limbs=18)
    assert sched4["bootstraps"] in (2, 3), sched4  # + final lambda refresh when short
    assert sched4["probability"] >= SCHAIN_PROBABILITY_MIN_LIMBS
    assert blockplan_applicable(config, 0, 4, exit_limbs=18) == plan4
    assert blockplan_applicable(config, 0, 5, exit_limbs=18) is None  # unlisted
    # too shallow an exit level: pack would be too short or the probability too thin -> Alg2
    assert blockplan_applicable(config, 0, 3, exit_limbs=9) is None
    monkeypatch.setenv(SOFTMAX_BLOCKPLAN_FORCE_ENV, "1")
    shallow = blockplan_limb_schedule(config, 0, plan3, k=3, exit_limbs=9)
    if shallow["min_pack"] is not None and shallow["min_pack"] >= BOOTSTRAP_MIN_INPUT_LIMBS:
        assert blockplan_applicable(config, 0, 3, exit_limbs=9) == plan3
    monkeypatch.delenv(SOFTMAX_BLOCKPLAN_ENV, raising=False)
    assert blockplan_applicable(config, 0, 3, exit_limbs=18) is None


def test_pv_need_limbs_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SOFTMAX_PV_NEED_LIMBS_ENV, raising=False)
    assert pv_probability_need_limbs() == 6
    assert probability_min_limbs() == SCHAIN_PROBABILITY_MIN_LIMBS == 7
    monkeypatch.setenv(SOFTMAX_PV_NEED_LIMBS_ENV, "5")
    assert pv_probability_need_limbs() == 5
    assert probability_min_limbs() == 6
    monkeypatch.setenv(SOFTMAX_PV_NEED_LIMBS_ENV, "2")
    with pytest.raises(ValueError):
        pv_probability_need_limbs()
    # the gate follows the knob: a plan whose probability lands exactly one limb short
    # of the default bound becomes applicable at PV need 5
    monkeypatch.setenv(SOFTMAX_PV_NEED_LIMBS_ENV, "5")
    monkeypatch.setenv(SOFTMAX_BLOCKPLAN_ENV, "3:s2,d1")
    config = DecodeAlg2Poly.synthetic(k=3, degree=8)
    plan3 = parse_softmax_blockplan("3:s2,d1")[3]
    for exit_limbs in range(10, 20):
        sched = blockplan_limb_schedule(config, 0, plan3, k=3, exit_limbs=exit_limbs)
        if sched["probability"] == 6 and sched["min_pack"] >= BOOTSTRAP_MIN_INPUT_LIMBS:
            assert blockplan_applicable(config, 0, 3, exit_limbs=exit_limbs) == plan3
            monkeypatch.delenv(SOFTMAX_PV_NEED_LIMBS_ENV, raising=False)
            assert blockplan_applicable(config, 0, 3, exit_limbs=exit_limbs) is None
            break


def _run(context, layout, pages, config, *, occupied, head_rows, bootstrap):
    probs = softmax_alg2_row_pages_fhe(
        pages,
        layout=layout,
        crypto_context=context,
        config=config,
        occupied_pages=occupied,
        layer_idx=0,
        rounds=None,
        bootstrap_operator=bootstrap,
        bootstrap_iterations=1,
        bootstrap_precision_bits=0,
        head_rows=head_rows,
    )
    try:
        limbs = tuple(int(c.state.cur_limbs) for page in probs for c in page)
        return tuple(_decrypt(page, layout=layout, context=context) for page in probs), limbs
    finally:
        for page in probs:
            for cipher in page:
                release_if_supported(cipher)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "k, spec, alg2_calls",
    [
        (3, "3:s2,d1", 4),  # post_exp + r1.lam + r2.lam + r3.lam
        (4, "4:s2,d2", 5),
    ],
)
@pytest.mark.parametrize("head_rows", [None, (0, 2)])
def test_blockplan_matches_alg2_merged_with_fewer_bootstraps(
    monkeypatch: pytest.MonkeyPatch, k, spec, alg2_calls, head_rows
) -> None:
    rng = np.random.default_rng(26103001 + k)
    layout = _layout()
    occupied = (4, 2)
    scores = [rng.normal(scale=0.3, size=(4, 4)).astype(np.float32) for _ in occupied]
    if head_rows is not None:
        for page in scores:
            page[[1, 3], :] = 0.0
    config = DecodeAlg2Poly.synthetic(k=k, degree=8)
    context = _context(layout, depth=48)
    ciphers = tuple(
        tuple(context.encrypt(row, level=0, slots=layout.slots) for row in layout.pack_compact_decode_scores(s))
        for s in scores
    )
    monkeypatch.setenv(SOFTMAX_MERGE_PAGES_ENV, "1")
    monkeypatch.setenv(SOFTMAX_SCHAIN_ENV, "0")  # k != 2 anyway; keep the two switches independent
    try:
        monkeypatch.delenv(SOFTMAX_BLOCKPLAN_ENV, raising=False)
        counter = _CountingRefresh(context, limbs=18, slots=layout.slots)
        alg2, _ = _run(context, layout, ciphers, config, occupied=occupied, head_rows=head_rows, bootstrap=counter)
        got_alg2_calls = counter.calls
        monkeypatch.setenv(SOFTMAX_BLOCKPLAN_ENV, spec)
        counter = _CountingRefresh(context, limbs=18, slots=layout.slots)
        plan_out, limbs = _run(
            context, layout, ciphers, config, occupied=occupied, head_rows=head_rows, bootstrap=counter
        )
        plan_calls = counter.calls
    finally:
        for page in ciphers:
            for cipher in page:
                release_if_supported(cipher)
    plan = parse_softmax_blockplan(spec)[k]
    sched = blockplan_limb_schedule(config, 0, plan, k=k, exit_limbs=18)
    assert got_alg2_calls == alg2_calls
    assert plan_calls == 1 + sched["bootstraps"], (plan_calls, sched)  # post_exp + planned aux refreshes
    assert plan_calls < got_alg2_calls
    assert all(limb == sched["probability"] for limb in limbs), (limbs, sched)
    assert sched["probability"] >= SCHAIN_PROBABILITY_MIN_LIMBS
    assert len(plan_out) == len(alg2) == 2
    for got, want, occ in zip(plan_out, alg2, occupied, strict=True):
        np.testing.assert_allclose(got, want, atol=3e-3, rtol=3e-3)
        np.testing.assert_allclose(got[:, occ:], 0.0, atol=1e-3)
        if head_rows is not None:
            np.testing.assert_allclose(got[[1, 3], :], 0.0, atol=1e-3)
    rows = np.concatenate([page[:, :occ] for page, occ in zip(plan_out, occupied, strict=True)], axis=1)
    owned = list(head_rows) if head_rows is not None else list(range(4))
    np.testing.assert_allclose(rows[owned].sum(axis=1), 1.0, atol=0.25)  # synthetic degree-8 polys
