from __future__ import annotations

"""Decode Alg2 Softmax on compact Decode scores.  Not Delta ``softmax_alg2_fhe_delta``.

Public map, Chebyshev exp / ``1/√s``, and ``y←(y λ)^2`` reuse Prefill
coefficients.  Row reduction is the C token-axis fold (stride 2); ``λ`` is
isolated on column 0 and broadcast back to the occupied prefix.
"""

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from llama3fhe.approx.chebyshev import (
    chebyshev_ps_mul_depth,
    eval_chebyshev_series_direct,
)
from llama3fhe.backend import release_if_supported
from llama3fhe.layouts.attention import AttentionPairLayout

from ...nonlinear.bootstrap import bootstrap_tag
from ...nonlinear.elementwise import eval_exp_cipher, eval_invsqrt_cipher
from ...primitives import mul_cipher_rescale, square_rescale
from .decode import (
    _add_packed_rows,
    _add_scalar_ciphers,
    _homo_rotate,
    _mul_packed_rows,
    _rotate_add_tree,
    _token_broadcast_cipher,
    _token_reduce_cipher,
    decode_broadcast_tokens_fhe,
    decode_reduce_tokens_fhe,
    softmax_alg2_row_numpy,
)


def _scores_rows(
    scores: np.ndarray, *, layout: AttentionPairLayout
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    packed_shape = (int(layout.query_pair_count), int(layout.slots))
    compact_shape = (1, int(layout.slots))
    if scores.ndim == 2 and scores.shape == compact_shape:
        scores = layout.unpack_compact_decode_scores(scores)
    elif scores.ndim == 2 and scores.shape == packed_shape:
        scores = layout.unpack_decode_scores(scores)
    if scores.ndim == 3:
        if scores.shape[1] != 1:
            raise ValueError("decode Alg2 expects one query row or [H,S].")
        scores = scores[:, 0, :]
    expected = (int(layout.query_heads), int(layout.seq_len))
    if scores.shape != expected:
        raise ValueError(f"decode Alg2 scores {scores.shape} vs {expected}.")
    return scores


def _prefix_valid(layout: AttentionPairLayout, occupied: int | None) -> np.ndarray:
    seq = int(layout.seq_len)
    occupied = seq if occupied is None else int(occupied)
    if occupied < 0 or occupied > seq:
        raise ValueError(f"occupied={occupied} outside [0, {seq}].")
    valid = np.zeros((seq,), dtype=bool)
    valid[:occupied] = True
    return valid


def _head_rows_tuple(
    head_rows, *, layout: AttentionPairLayout
) -> tuple[int, ...] | None:
    """Validated, sorted, de-duplicated query-head rows (``None`` = all heads)."""

    if head_rows is None:
        return None
    heads = int(layout.query_heads)
    rows = tuple(sorted({int(row) for row in head_rows}))
    if not rows:
        raise ValueError("head_rows must name at least one query head.")
    for row in rows:
        if not 0 <= row < heads:
            raise IndexError(f"head_rows entry {row} outside [0, {heads}).")
    return rows


def _head_rows_valid(
    layout: AttentionPairLayout, head_rows
) -> np.ndarray | None:
    """``[query_heads]`` bool: which head rows carry real scores."""

    rows = _head_rows_tuple(head_rows, layout=layout)
    if rows is None:
        return None
    owned = np.zeros((int(layout.query_heads),), dtype=bool)
    owned[list(rows)] = True
    return owned


def _head_rows_mask(
    layout: AttentionPairLayout, head_rows
) -> np.ndarray | None:
    """Packed compact-Decode mask: 1 on ``head_rows``; ``None`` = no masking.

    Multi-card decode gives a rank only some ``(group, query_offset)``
    streams, so the other head rows of its compact score pages are exactly
    zero.  A zero row is *not* a valid softmax row: its uniform ``exp`` makes
    ``sum(y^2)`` leave the per-layer ``1/sqrt`` fit interval (layer 2 / 23
    blow up to 1e100+), and the resulting garbage slots poison every slot of
    the ciphertext through bootstrap and the slot-mixing linear transforms.
    With ``head_rows`` those rows are treated like invalid prefix columns
    (public map -> ``stats_min``, masked to zero after ``exp``) and their
    ``1/sqrt`` input is replaced by the safe fill constant.
    """

    rows = _head_rows_tuple(head_rows, layout=layout)
    if rows is None:
        return None
    mask = np.zeros(
        (int(layout.query_heads), int(layout.seq_len)), dtype=np.float32
    )
    mask[list(rows), :] = 1.0
    return layout.pack_compact_decode_scores(mask, dtype=np.float64)


def _with_head_rows(mask: np.ndarray, head_mask: np.ndarray | None) -> np.ndarray:
    """Restrict a packed compact-Decode mask to the owned head rows."""

    if head_mask is None:
        return mask
    return np.asarray(mask, dtype=np.float64) * np.asarray(head_mask, dtype=np.float64)


def apply_decode_alg2_public_map_numpy(
    scores: np.ndarray,
    *,
    layout: AttentionPairLayout,
    config,
    layer_idx: int,
    occupied: int | None = None,
    score_scale: float | None = None,
    head_rows=None,
) -> np.ndarray:
    """Valid slots: ``(s/√d - U_hat)/2^k``; invalid: ``(stats_min - U_hat)/2^k``.

    ``head_rows`` (multi-card decode) marks every slot of the other head rows
    invalid as well.
    """

    scores = _scores_rows(scores, layout=layout)
    valid = _prefix_valid(layout, occupied)
    owned = _head_rows_valid(layout, head_rows)
    k = int(config.k_for(layer_idx))
    score_scale = (
        1.0 / math.sqrt(float(layout.head_dim))
        if score_scale is None
        else float(score_scale)
    )
    shift = -float(config.U_hat_for(layer_idx)) / float(2**k)
    invalid = float(config.stats_min_for(layer_idx)) / float(2**k)
    mapped = np.full(scores.shape, shift + invalid, dtype=np.float64)
    mapped[:, valid] = scores[:, valid] * (score_scale / float(2**k)) + shift
    if owned is not None:
        mapped[~owned, :] = shift + invalid
    return mapped.astype(np.float32)


def apply_decode_alg2_public_map_fhe(
    score_ciphers: tuple[object, ...] | list[object],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    layer_idx: int,
    occupied: int | None = None,
    score_scale: float | None = None,
    head_rows=None,
) -> tuple[object, ...]:
    """Prefill Alg2 public map on compact Decode score packing.  Does not consume inputs.

    ``head_rows`` (multi-card decode): the other head rows are mapped like
    invalid columns (see :func:`_head_rows_mask`).
    """

    score_ciphers = tuple(score_ciphers)
    if len(score_ciphers) != 1:
        raise ValueError(
            f"Decode Alg2 expects 1 compact score cipher, got {len(score_ciphers)}."
        )
    layout._require_compact_q()
    k = int(config.k_for(layer_idx))
    score_scale = (
        1.0 / math.sqrt(float(layout.head_dim))
        if score_scale is None
        else float(score_scale)
    )
    valid = _with_head_rows(
        layout.compact_decode_prefix_mask(occupied),
        _head_rows_mask(layout, head_rows),
    )
    multiplier = np.asarray(valid, dtype=np.float64) * (
        score_scale / float(2**k)
    )
    shift = -float(config.U_hat_for(layer_idx)) / float(2**k)
    invalid = (1.0 - np.asarray(valid, dtype=np.float64)) * (
        float(config.stats_min_for(layer_idx)) / float(2**k)
    )
    scaled = shifted = None
    try:
        scaled = _mul_packed_rows(
            score_ciphers,
            multiplier,
            name="decode.alg2.valid_scale",
            crypto_context=crypto_context,
        )
        shifted = _add_scalar_ciphers(
            scaled, shift, crypto_context=crypto_context
        )
        return _add_packed_rows(
            shifted,
            invalid,
            name="decode.alg2.invalid",
            crypto_context=crypto_context,
        )
    finally:
        for cipher in tuple(scaled or ()) + tuple(shifted or ()):
            release_if_supported(cipher)


def _invsqrt_cheb(poly_entry: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    cheb = poly_entry.get("chebyshev_fhe") or poly_entry
    lo, hi = (float(value) for value in cheb["fit_interval"])
    coeffs = np.asarray(cheb["coefficients"], dtype=np.float64)
    return coeffs, lo, hi


def softmax_alg2_poly_numpy(
    scores: np.ndarray,
    *,
    layout: AttentionPairLayout,
    config,
    layer_idx: int,
    occupied: int | None = None,
    score_scale: float | None = None,
    rounds: int | None = None,
    head_rows=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Chebyshev Alg2 on a Decode score row. Returns ``(P, interval report)``.

    ``head_rows`` mirrors the FHE kernel's multi-card handling: the other head
    rows are invalid everywhere (probabilities exactly zero) and their
    ``1/sqrt`` input is the safe fill constant instead of the zero row sum.
    """

    scores = _scores_rows(scores, layout=layout)
    valid = _prefix_valid(layout, occupied)
    owned = _head_rows_valid(layout, head_rows)
    mapped = apply_decode_alg2_public_map_numpy(
        scores,
        layout=layout,
        config=config,
        layer_idx=layer_idx,
        occupied=occupied,
        score_scale=score_scale,
        head_rows=head_rows,
    ).astype(np.float64)
    exp_entry = config.layer_exp[int(layer_idx)]
    exp_lo, exp_hi = (float(value) for value in exp_entry["fit_interval"])
    mapped_valid = mapped[:, valid]
    if owned is not None:
        mapped_valid = mapped_valid[owned, :]
    report: dict[str, Any] = {
        "k": int(config.k_for(layer_idx)),
        "occupied": int(np.count_nonzero(valid)),
        "exp_interval": (exp_lo, exp_hi),
        "exp_min": float(np.min(mapped_valid)) if mapped_valid.size else None,
        "exp_max": float(np.max(mapped_valid)) if mapped_valid.size else None,
        "exp_in_range": bool(
            mapped_valid.size
            and float(np.min(mapped_valid)) >= exp_lo
            and float(np.max(mapped_valid)) <= exp_hi
        ),
        "invsqrt_rounds": [],
    }
    y = eval_chebyshev_series_direct(
        mapped,
        np.asarray(exp_entry["coefficients"], dtype=np.float64),
        exp_lo,
        exp_hi,
    ).astype(np.float64)
    y = np.where(valid[None, :], y, 0.0)
    if owned is not None:
        y = np.where(owned[:, None], y, 0.0)
    k = int(config.k_for(layer_idx) if rounds is None else rounds)
    rough = config.invsqrt_alg2["j1_rough"]
    precise = config.invsqrt_alg2["j2_precise"]
    for round_index in range(1, k + 1):
        poly = config.poly_for_layer(
            rough if round_index == 1 else precise, int(layer_idx)
        )
        coeffs, lo, hi = _invsqrt_cheb(poly)
        row_sum = np.sum(y * y, axis=-1, keepdims=True)
        if owned is not None:
            row_sum = np.where(
                owned[:, None], row_sum, _safe_invsqrt_fill(poly)
            )
        row_vals = row_sum.reshape(-1)
        if owned is not None:
            row_vals = row_vals[owned]
        round_report = {
            "round": round_index,
            "interval": (lo, hi),
            "min": float(np.min(row_vals)),
            "max": float(np.max(row_vals)),
            "in_range": bool(
                float(np.min(row_vals)) >= lo and float(np.max(row_vals)) <= hi
            ),
        }
        report["invsqrt_rounds"].append(round_report)
        lam = eval_chebyshev_series_direct(row_sum, coeffs, lo, hi).astype(
            np.float64
        )
        y = np.square(y * lam)
        y = np.where(valid[None, :], y, 0.0)
        if owned is not None:
            y = np.where(owned[:, None], y, 0.0)
    probabilities = np.asarray(y, dtype=np.float32)
    exact = softmax_alg2_row_numpy(
        scores,
        layout=layout,
        occupied=int(np.count_nonzero(valid)),
        k=k,
        scale=True,
    )
    difference = np.abs(probabilities - exact)
    if owned is not None:
        difference = difference[owned, :]
    report["vs_exact_max_abs"] = float(np.max(difference))
    return probabilities, report


def _align_cipher_pair(left, right, *, crypto_context):
    left_limbs = int(left.state.cur_limbs)
    right_limbs = int(right.state.cur_limbs)
    left_degree = int(left.state.scale_degree)
    right_degree = int(right.state.scale_degree)
    left_scale = left.state.scaling_factor
    right_scale = right.state.scaling_factor
    same_scale = (
        left_scale is None
        and right_scale is None
    ) or (
        left_scale is not None
        and right_scale is not None
        and math.isclose(
            float(left_scale), float(right_scale), rel_tol=1e-12, abs_tol=0.0
        )
    )
    if (
        left_limbs == right_limbs
        and left_degree == right_degree
        and same_scale
    ):
        return left, right, False, False
    if left_limbs == right_limbs:
        # With equal modulus depth choose the larger scale degree as the
        # common target: aligning a degree-1 ciphertext *up* to degree 2 is
        # depth-safe, while the reverse direction would ask align_to to
        # create a state with more consumed depth than its source has.
        if left_degree > right_degree:
            return (
                left,
                crypto_context.fhe.align_to(
                    right, left.state, crypto_context.context
                ),
                False,
                True,
            )
        return (
            crypto_context.fhe.align_to(
                left, right.state, crypto_context.context
            ),
            right,
            True,
            False,
        )
    if left_limbs > right_limbs:
        return (
            crypto_context.fhe.align_to(left, right.state, crypto_context.context),
            right,
            True,
            False,
        )
    return (
        left,
        crypto_context.fhe.align_to(right, left.state, crypto_context.context),
        False,
        True,
    )


def _safe_invsqrt_fill(poly_entry: dict[str, Any]) -> float:
    _, lo, hi = _invsqrt_cheb(poly_entry)
    return float(min(max(0.25, lo), hi))


def _release_tuple(ciphers) -> None:
    for cipher in ciphers or ():
        release_if_supported(cipher)


class _BroadcastLayerExp:
    """One synthetic exp poly, indexed like the 32-layer production pack."""

    def __init__(self, entry: dict[str, Any]):
        self._entry = dict(entry)

    def __getitem__(self, layer_idx: int) -> dict[str, Any]:
        del layer_idx
        return dict(self._entry)


@dataclass(frozen=True)
class DecodeAlg2Poly:
    """Minimal Alg2 artifact for Decode FHE smokes (not the 32-layer pack)."""

    k: int
    U_hat: float
    stats_min: float
    exp_entry: dict[str, Any]
    invsqrt_entry: dict[str, Any]

    def k_for(self, layer_idx: int) -> int:
        del layer_idx
        return int(self.k)

    def U_hat_for(self, layer_idx: int) -> float:
        del layer_idx
        return float(self.U_hat)

    def stats_min_for(self, layer_idx: int) -> float:
        del layer_idx
        return float(self.stats_min)

    @property
    def layer_exp(self) -> _BroadcastLayerExp:
        return _BroadcastLayerExp(self.exp_entry)

    @property
    def invsqrt_alg2(self) -> dict[str, dict[str, Any]]:
        entry = dict(self.invsqrt_entry)
        return {"j1_rough": entry, "j2_precise": entry}

    def poly_for_layer(self, poly_entry: dict[str, Any], layer_idx: int):
        del layer_idx
        return poly_entry

    @classmethod
    def synthetic(
        cls,
        *,
        k: int = 1,
        degree: int = 8,
        U_hat: float = 1.0,
        stats_min: float = -4.0,
        exp_interval: tuple[float, float] = (-2.0, 1.0),
        invsqrt_interval: tuple[float, float] = (0.05, 8.0),
    ) -> "DecodeAlg2Poly":
        """Low-degree exp / invsqrt for Decode FHE smokes (not production).

        Default degree 8 uses a well-tested Paterson–Stockmeyer split.
        Degree 2 fails FHE eval with a shallower-to-deeper align error.
        """

        from llama3fhe.approx.chebyshev import eval_chebyshev_coefficients

        exp_lo, exp_hi = (float(value) for value in exp_interval)
        inv_lo, inv_hi = (float(value) for value in invsqrt_interval)
        exp_coeffs = eval_chebyshev_coefficients(math.exp, exp_lo, exp_hi, int(degree))
        inv_coeffs = eval_chebyshev_coefficients(
            lambda value: 1.0 / math.sqrt(max(float(value), inv_lo)),
            inv_lo,
            inv_hi,
            int(degree),
        )
        return cls(
            k=int(k),
            U_hat=float(U_hat),
            stats_min=float(stats_min),
            exp_entry={
                "fit_interval": [exp_lo, exp_hi],
                "coefficients": np.asarray(exp_coeffs, dtype=np.float64),
            },
            invsqrt_entry={
                "fit_interval": [inv_lo, inv_hi],
                "coefficients": np.asarray(inv_coeffs, dtype=np.float64),
                "chebyshev_fhe": {
                    "fit_interval": [inv_lo, inv_hi],
                    "coefficients": np.asarray(inv_coeffs, dtype=np.float64),
                },
            },
        )


def _decode_exp_need_limbs(exp_entry: dict[str, Any]) -> int:
    """Limbs the public-mapped scores must exceed before Chebyshev exp."""

    return int(chebyshev_ps_mul_depth(exp_entry["coefficients"])) + 3


def _invsqrt_need_limbs(poly_entry: dict[str, Any]) -> int:
    """Limbs the Alg2 λ input must exceed before the Prefill invsqrt poly."""

    coeffs, _, _ = _invsqrt_cheb(poly_entry)
    return int(chebyshev_ps_mul_depth(coeffs)) + 3


#: Limbs one Alg2 round consumes on a ``y`` that sits at or below the
#: broadcast-λ level: the ``y·λ`` rescale, the ``square`` rescale and the
#: prefix-mask rescale (:func:`_alg2_round_fhe`).  A higher ``y`` is first
#: aligned down to λ, which only makes the round cheaper for this bound.
ALG2_ROUND_Y_LIMBS = 3
#: Limbs a ciphertext must still carry to be bootstrapped (the mod-raise
#: rescales it once first: ``rescale: cur_limbs must be > 1``).
BOOTSTRAP_MIN_INPUT_LIMBS = 2
#: ``y`` is refreshed *before* a round when it has at most this many limbs:
#: the round itself plus the minimum the post-Softmax probability refresh
#: (or a later round's refresh) needs.  With the production Meta-BTS exit
#: the ``k <= 4`` layers never reach this bound and keep their schedule
#: unchanged; ``k = 5`` (layer 31) refreshes ``y`` once before its last round
#: instead of ending the chain at 1 limb, where the probability refresh
#: failed (``rescale: cur_limbs must be > 1, got 1``).
ALG2_ROUND_NEED_LIMBS = ALG2_ROUND_Y_LIMBS + BOOTSTRAP_MIN_INPUT_LIMBS


def _refresh_row_ciphers(
    ciphers,
    *,
    bootstrap_operator,
    iterations: int | None,
    precision_bits: int | None,
    tag: str | None = None,
    periodic_slots: int | None = None,
):
    """Refresh one score-row stream.  Consumes ``ciphers`` when a refresh runs.

    ``tag`` labels the bootstraps for the bench timers (:func:`bootstrap_tag`).
    ``periodic_slots`` promises the message is periodic with that slot period
    and routes the refresh through the operator's matching sparse-slot
    program (:meth:`BootstrapOperator.refresh`).
    """

    ciphers = tuple(ciphers)
    if bootstrap_operator is None:
        return ciphers
    if tag is not None:
        with bootstrap_tag(tag):
            return _refresh_row_ciphers(
                ciphers,
                bootstrap_operator=bootstrap_operator,
                iterations=iterations,
                precision_bits=precision_bits,
                periodic_slots=periodic_slots,
            )
    selected_iterations = (
        int(bootstrap_operator.config.iterations)
        if iterations is None
        else int(iterations)
    )
    selected_precision = (
        int(bootstrap_operator.config.precision_bits)
        if precision_bits is None
        else int(precision_bits)
    )
    sparse_kwargs = {} if periodic_slots is None else {"periodic_slots": int(periodic_slots)}
    refreshed: list[object] = []
    try:
        for cipher in ciphers:
            refreshed.append(
                bootstrap_operator.refresh(
                    cipher,
                    iterations=selected_iterations,
                    precision_bits=selected_precision,
                    **sparse_kwargs,
                )
            )
        result = tuple(refreshed)
        refreshed = []
        _release_tuple(ciphers)
        return result
    finally:
        _release_tuple(refreshed)


def _refresh_row_ciphers_if_short(
    ciphers,
    *,
    bootstrap_operator,
    need_limbs: int,
    iterations: int | None,
    precision_bits: int | None,
    tag: str | None = None,
    periodic_slots: int | None = None,
):
    """Refresh when any ciphertext has ``cur_limbs <= need_limbs``."""

    ciphers = tuple(ciphers)
    if bootstrap_operator is None or not ciphers:
        return ciphers
    if min(int(cipher.state.cur_limbs) for cipher in ciphers) > int(need_limbs):
        return ciphers
    return _refresh_row_ciphers(
        ciphers,
        bootstrap_operator=bootstrap_operator,
        iterations=iterations,
        precision_bits=precision_bits,
        tag=tag,
        periodic_slots=periodic_slots,
    )


def softmax_alg2_row_fhe(
    score_ciphers: tuple[object, ...] | list[object],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    layer_idx: int = 0,
    occupied: int | None = None,
    score_scale: float | None = None,
    rounds: int | None = None,
    bootstrap_operator=None,
    bootstrap_iterations: int | None = None,
    bootstrap_precision_bits: int | None = None,
    head_rows=None,
) -> tuple[object, ...]:
    """Chebyshev Alg2 on Decode scores.

    Passing ``bootstrap_operator`` refreshes ``y`` after exp and refreshes
    each round's λ the way Prefill does, so Decode can consume the same
    Alg2 coefficients (including production ``k`` and invsqrt degree).
    ``y`` is refreshed again before a round that would run the chain dry
    (:data:`ALG2_ROUND_NEED_LIMBS`; only the ``k = 5`` layer needs it).

    ``head_rows`` (multi-card decode): only these query-head rows carry real
    scores; the others are forced to zero probability (:func:`_head_rows_mask`).
    """

    score_ciphers = tuple(score_ciphers)
    head_mask = _head_rows_mask(layout, head_rows)
    mapped = y_ciphers = None
    try:
        mapped = apply_decode_alg2_public_map_fhe(
            score_ciphers,
            layout=layout,
            crypto_context=crypto_context,
            config=config,
            layer_idx=layer_idx,
            occupied=occupied,
            score_scale=score_scale,
            head_rows=head_rows,
        )
        exp_entry = config.layer_exp[int(layer_idx)]
        mapped = _refresh_row_ciphers_if_short(
            mapped,
            bootstrap_operator=bootstrap_operator,
            need_limbs=_decode_exp_need_limbs(exp_entry),
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag="softmax.pre_exp",
        )
        exp_outputs = []
        for cipher in mapped:
            exp_outputs.append(
                eval_exp_cipher(cipher, exp_entry, crypto_context=crypto_context)
            )
        y_ciphers = tuple(exp_outputs)
        exp_outputs = []
        mapped = None
        masked = _mul_packed_rows(
            y_ciphers,
            _with_head_rows(layout.compact_decode_prefix_mask(occupied), head_mask),
            name="decode.alg2.post_exp",
            crypto_context=crypto_context,
        )
        _release_tuple(y_ciphers)
        y_ciphers = _refresh_row_ciphers(
            masked,
            bootstrap_operator=bootstrap_operator,
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag="softmax.post_exp",
        )
        k = int(config.k_for(layer_idx) if rounds is None else rounds)
        rough = config.invsqrt_alg2["j1_rough"]
        precise = config.invsqrt_alg2["j2_precise"]
        for round_index in range(1, k + 1):
            poly = config.poly_for_layer(
                rough if round_index == 1 else precise, int(layer_idx)
            )
            y_ciphers = _refresh_row_ciphers_if_short(
                y_ciphers,
                bootstrap_operator=bootstrap_operator,
                need_limbs=ALG2_ROUND_NEED_LIMBS,
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
                tag=f"softmax.r{round_index}.y",
            )
            nxt = _alg2_round_fhe(
                y_ciphers,
                poly_entry=poly,
                layout=layout,
                crypto_context=crypto_context,
                occupied=occupied,
                round_index=round_index,
                bootstrap_operator=bootstrap_operator,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_precision_bits=bootstrap_precision_bits,
                head_mask=head_mask,
            )
            _release_tuple(y_ciphers)
            y_ciphers = nxt
        result = y_ciphers
        y_ciphers = None
        return result
    finally:
        _release_tuple(mapped)
        _release_tuple(y_ciphers)


def _alg2_round_fhe(
    y_ciphers: tuple[object, ...],
    *,
    poly_entry: dict[str, Any],
    layout: AttentionPairLayout,
    crypto_context,
    occupied: int | None,
    round_index: int,
    bootstrap_operator=None,
    bootstrap_iterations: int | None = None,
    bootstrap_precision_bits: int | None = None,
    head_mask: np.ndarray | None = None,
) -> tuple[object, ...]:
    squared = reduced = isolated = filled = lam = lam_bcast = scaled = None
    try:
        squared = tuple(
            square_rescale(cipher, crypto_context=crypto_context)
            for cipher in y_ciphers
        )
        reduced = decode_reduce_tokens_fhe(
            squared, layout=layout, crypto_context=crypto_context
        )
        isolated = _mul_packed_rows(
            reduced,
            layout.compact_decode_token0_mask(),
            name=f"decode.alg2.r{round_index}.sum0",
            crypto_context=crypto_context,
        )
        fill = _invsqrt_fill_mask(layout, poly_entry, head_mask=head_mask)
        filled = _add_packed_rows(
            isolated,
            fill,
            name=f"decode.alg2.r{round_index}.invsqrt_fill",
            crypto_context=crypto_context,
        )
        _release_tuple(isolated)
        isolated = None
        filled = _refresh_row_ciphers_if_short(
            filled,
            bootstrap_operator=bootstrap_operator,
            need_limbs=_invsqrt_need_limbs(poly_entry),
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag=f"softmax.r{round_index}.fill",
        )
        lam_raw = []
        for cipher in filled:
            lam_raw.append(
                eval_invsqrt_cipher(
                    cipher, poly_entry, crypto_context=crypto_context
                )
            )
        lam = tuple(lam_raw)
        lam_raw = []
        lam = _refresh_row_ciphers(
            lam,
            bootstrap_operator=bootstrap_operator,
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag=f"softmax.r{round_index}.lam",
        )
        lam_bcast = decode_broadcast_tokens_fhe(
            lam, layout=layout, crypto_context=crypto_context, occupied=occupied
        )
        products = []
        for y_cipher, lam_cipher in zip(y_ciphers, lam_bcast, strict=True):
            aligned_y, aligned_lam, owned_y, owned_lam = _align_cipher_pair(
                y_cipher, lam_cipher, crypto_context=crypto_context
            )
            try:
                products.append(
                    mul_cipher_rescale(
                        aligned_y, aligned_lam, crypto_context=crypto_context
                    )
                )
            finally:
                if owned_y:
                    release_if_supported(aligned_y)
                if owned_lam:
                    release_if_supported(aligned_lam)
        scaled = tuple(products)
        products = []
        squared_y = tuple(
            square_rescale(cipher, crypto_context=crypto_context)
            for cipher in scaled
        )
        remasked = _mul_packed_rows(
            squared_y,
            _with_head_rows(layout.compact_decode_prefix_mask(occupied), head_mask),
            name=f"decode.alg2.r{round_index}.prefix",
            crypto_context=crypto_context,
        )
        _release_tuple(squared_y)
        return remasked
    finally:
        _release_tuple(squared)
        _release_tuple(reduced)
        _release_tuple(isolated)
        _release_tuple(filled)
        _release_tuple(lam)
        _release_tuple(lam_bcast)
        _release_tuple(scaled)


def _invsqrt_fill_mask(
    layout: AttentionPairLayout,
    poly_entry: dict[str, Any],
    *,
    head_mask: np.ndarray | None,
) -> np.ndarray:
    """Safe ``1/sqrt`` input everywhere except the owned rows' column-0 sums.

    Column 0 of an owned row holds ``sum(y^2)``; every other slot (and, with
    ``head_mask``, column 0 of the rows this rank does not own, whose sum is
    zero) receives the in-interval fill constant so the Chebyshev ``1/sqrt``
    never extrapolates.
    """

    token0 = _with_head_rows(layout.compact_decode_token0_mask(), head_mask)
    return (1.0 - np.asarray(token0, dtype=np.float64)) * (
        _safe_invsqrt_fill(poly_entry)
    )


def _release_pages(pages) -> None:
    for page in pages or ():
        _release_tuple(page)


def _sum_aligned_ciphers(ciphers, *, crypto_context):
    ciphers = tuple(ciphers)
    if not ciphers:
        raise ValueError("cannot sum an empty cipher list.")
    if len(ciphers) == 1:
        return ciphers[0].deep_copy()
    acc = ciphers[0]
    owned = False
    try:
        for cipher in ciphers[1:]:
            left, right, owned_left, owned_right = _align_cipher_pair(
                acc, cipher, crypto_context=crypto_context
            )
            try:
                summed = crypto_context.fhe.homo_add(
                    left, right, crypto_context.context
                )
            finally:
                if owned_left:
                    release_if_supported(left)
                if owned_right:
                    release_if_supported(right)
            if owned:
                release_if_supported(acc)
            acc = summed
            owned = True
        result = acc
        acc = None
        return result
    finally:
        if owned:
            release_if_supported(acc)


def softmax_alg2_row_pages_fhe(
    score_pages: tuple[tuple[object, ...], ...] | list[tuple[object, ...]],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    occupied_pages: tuple[int, ...] | list[int],
    layer_idx: int = 0,
    rounds: int | None = None,
    score_scale: float | None = None,
    bootstrap_operator=None,
    bootstrap_iterations: int | None = None,
    bootstrap_precision_bits: int | None = None,
    head_rows=None,
) -> tuple[tuple[object, ...], ...]:
    """Alg2 across C-form score pages that share one 1-query softmax.

    ``head_rows`` (multi-card decode): only these query-head rows carry real
    scores on every page; the others are forced to zero probability
    (:func:`_head_rows_mask`).  ``None`` keeps the single-card kernel.
    """

    score_pages = tuple(tuple(page) for page in score_pages)
    occupied_pages = tuple(int(value) for value in occupied_pages)
    if len(score_pages) != len(occupied_pages):
        raise ValueError("score pages and occupied_pages must have the same length.")
    if not score_pages:
        raise ValueError("softmax pages require at least one score page.")
    if len(score_pages) >= 2 and softmax_merge_pages_enabled() and (
        len(score_pages) <= int(layout.decode_page_merge_groups)
    ):
        return _softmax_alg2_merged_pages_fhe(
            score_pages,
            layout=layout,
            crypto_context=crypto_context,
            config=config,
            occupied_pages=occupied_pages,
            layer_idx=layer_idx,
            rounds=rounds,
            score_scale=score_scale,
            bootstrap_operator=bootstrap_operator,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_precision_bits=bootstrap_precision_bits,
            head_rows=head_rows,
        )
    head_mask = _head_rows_mask(layout, head_rows)
    mapped_pages: list[tuple[object, ...]] = []
    y_pages: list[tuple[object, ...]] = []
    try:
        for page, occupied in zip(score_pages, occupied_pages, strict=True):
            mapped = apply_decode_alg2_public_map_fhe(
                page,
                layout=layout,
                crypto_context=crypto_context,
                config=config,
                layer_idx=layer_idx,
                occupied=occupied,
                score_scale=score_scale,
                head_rows=head_rows,
            )
            exp_entry = config.layer_exp[int(layer_idx)]
            with bootstrap_tag("softmax.pre_exp"):
                mapped = _refresh_row_ciphers_if_short(
                    mapped,
                    bootstrap_operator=bootstrap_operator,
                    need_limbs=_decode_exp_need_limbs(exp_entry),
                    iterations=bootstrap_iterations,
                    precision_bits=bootstrap_precision_bits,
                )
            mapped_pages.append(mapped)
            exp_outputs = tuple(
                eval_exp_cipher(cipher, exp_entry, crypto_context=crypto_context)
                for cipher in mapped
            )
            masked = _mul_packed_rows(
                exp_outputs,
                _with_head_rows(
                    layout.compact_decode_prefix_mask(occupied), head_mask
                ),
                name="decode.alg2.post_exp",
                crypto_context=crypto_context,
            )
            _release_tuple(exp_outputs)
            y_pages.append(masked)
        _release_pages(mapped_pages)
        mapped_pages = []
        if bootstrap_operator is not None:
            with bootstrap_tag("softmax.post_exp"):
                y_pages = [
                    _refresh_row_ciphers(
                        page,
                        bootstrap_operator=bootstrap_operator,
                        iterations=bootstrap_iterations,
                        precision_bits=bootstrap_precision_bits,
                    )
                    for page in y_pages
                ]
        k = int(config.k_for(layer_idx) if rounds is None else rounds)
        rough = config.invsqrt_alg2["j1_rough"]
        precise = config.invsqrt_alg2["j2_precise"]
        for round_index in range(1, k + 1):
            poly = config.poly_for_layer(
                rough if round_index == 1 else precise, int(layer_idx)
            )
            with bootstrap_tag(f"softmax.r{round_index}.y"):
                y_pages = [
                    _refresh_row_ciphers_if_short(
                        page,
                        bootstrap_operator=bootstrap_operator,
                        need_limbs=ALG2_ROUND_NEED_LIMBS,
                        iterations=bootstrap_iterations,
                        precision_bits=bootstrap_precision_bits,
                    )
                    for page in y_pages
                ]
            nxt = _alg2_round_pages_fhe(
                y_pages,
                poly_entry=poly,
                layout=layout,
                crypto_context=crypto_context,
                occupied_pages=occupied_pages,
                round_index=round_index,
                bootstrap_operator=bootstrap_operator,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_precision_bits=bootstrap_precision_bits,
                head_mask=head_mask,
            )
            _release_pages(y_pages)
            y_pages = list(nxt)
        result = tuple(y_pages)
        y_pages = []
        return result
    finally:
        _release_pages(mapped_pages)
        _release_pages(y_pages)


def _alg2_round_pages_fhe(
    y_pages: list[tuple[object, ...]] | tuple[tuple[object, ...], ...],
    *,
    poly_entry: dict[str, Any],
    layout: AttentionPairLayout,
    crypto_context,
    occupied_pages: tuple[int, ...],
    round_index: int,
    bootstrap_operator=None,
    bootstrap_iterations: int | None = None,
    bootstrap_precision_bits: int | None = None,
    head_mask: np.ndarray | None = None,
) -> tuple[tuple[object, ...], ...]:
    y_pages = [tuple(page) for page in y_pages]
    squared_pages: list[tuple[object, ...]] = []
    reduced_pages: list[tuple[object, ...]] = []
    isolated_pages: list[tuple[object, ...]] = []
    combined = filled = lam = None
    next_pages: list[tuple[object, ...]] = []
    try:
        for y_ciphers in y_pages:
            squared = tuple(
                square_rescale(cipher, crypto_context=crypto_context)
                for cipher in y_ciphers
            )
            squared_pages.append(squared)
            reduced = decode_reduce_tokens_fhe(
                squared, layout=layout, crypto_context=crypto_context
            )
            reduced_pages.append(reduced)
            isolated_pages.append(
                _mul_packed_rows(
                    reduced,
                    layout.compact_decode_token0_mask(),
                    name=f"decode.alg2.r{round_index}.sum0",
                    crypto_context=crypto_context,
                )
            )
        pair_count = len(y_pages[0])
        if len(isolated_pages) == 1:
            # ``isolated_pages`` owns these sums.  The general borrowed
            # helper deep-copies a one-item sequence, which is unnecessary
            # for the common single-page Decode steady state.
            combined = tuple(isolated_pages[0])
            isolated_pages[0] = ()
        else:
            combined = tuple(
                _sum_aligned_ciphers(
                    tuple(page[pair] for page in isolated_pages),
                    crypto_context=crypto_context,
                )
                for pair in range(pair_count)
            )
        fill = _invsqrt_fill_mask(layout, poly_entry, head_mask=head_mask)
        filled = _add_packed_rows(
            combined,
            fill,
            name=f"decode.alg2.r{round_index}.invsqrt_fill",
            crypto_context=crypto_context,
        )
        # ``_add_packed_rows`` borrows its input rows.  Release the combined
        # sums explicitly; in the single-page case these are the transferred
        # isolated rows rather than a disposable deep copy.
        _release_tuple(combined)
        combined = None
        with bootstrap_tag(f"softmax.r{round_index}.fill"):
            filled = _refresh_row_ciphers_if_short(
                filled,
                bootstrap_operator=bootstrap_operator,
                need_limbs=_invsqrt_need_limbs(poly_entry),
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
            )
        lam = tuple(
            eval_invsqrt_cipher(cipher, poly_entry, crypto_context=crypto_context)
            for cipher in filled
        )
        with bootstrap_tag(f"softmax.r{round_index}.lam"):
            lam = _refresh_row_ciphers(
                lam,
                bootstrap_operator=bootstrap_operator,
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
            )
        for y_ciphers, occupied in zip(y_pages, occupied_pages, strict=True):
            lam_bcast = decode_broadcast_tokens_fhe(
                lam, layout=layout, crypto_context=crypto_context, occupied=occupied
            )
            products = []
            try:
                for y_cipher, lam_cipher in zip(y_ciphers, lam_bcast, strict=True):
                    aligned_y, aligned_lam, owned_y, owned_lam = _align_cipher_pair(
                        y_cipher, lam_cipher, crypto_context=crypto_context
                    )
                    try:
                        products.append(
                            mul_cipher_rescale(
                                aligned_y, aligned_lam, crypto_context=crypto_context
                            )
                        )
                    finally:
                        if owned_y:
                            release_if_supported(aligned_y)
                        if owned_lam:
                            release_if_supported(aligned_lam)
                scaled = tuple(products)
                products = []
                squared_y = tuple(
                    square_rescale(cipher, crypto_context=crypto_context)
                    for cipher in scaled
                )
                _release_tuple(scaled)
                remasked = _mul_packed_rows(
                    squared_y,
                    _with_head_rows(
                        layout.compact_decode_prefix_mask(occupied), head_mask
                    ),
                    name=f"decode.alg2.r{round_index}.prefix",
                    crypto_context=crypto_context,
                )
                _release_tuple(squared_y)
                next_pages.append(remasked)
            finally:
                _release_tuple(products)
                _release_tuple(lam_bcast)
        result = tuple(next_pages)
        next_pages = []
        return result
    finally:
        _release_pages(squared_pages)
        _release_pages(reduced_pages)
        _release_pages(isolated_pages)
        _release_tuple(combined)
        _release_tuple(filled)
        _release_tuple(lam)
        _release_pages(next_pages)


# --------------------------------------------------------------------------
# Merged-page Softmax: every CT cache page of one query shares one ciphertext
# --------------------------------------------------------------------------

#: ``LLAMA_FHE_SOFTMAX_MERGE_PAGES=0`` falls back to the per-page kernel.
SOFTMAX_MERGE_PAGES_ENV = "LLAMA_FHE_SOFTMAX_MERGE_PAGES"


SOFTMAX_SPARSE_BS_ENV = "LLAMA_FHE_SOFTMAX_SPARSE_BS"


#: ``LLAMA_FHE_SOFTMAX_SCHAIN=1`` runs the ``k = 2`` merged-page Softmax as the
#: S-chain (one aux bootstrap instead of two) -- see :func:`_schain_merged_tail`.
SOFTMAX_SCHAIN_ENV = "LLAMA_FHE_SOFTMAX_SCHAIN"


def softmax_schain_enabled() -> bool:
    """Whether ``k = 2`` merged-page Softmax layers use the S-chain tail (off by default)."""

    return os.environ.get(SOFTMAX_SCHAIN_ENV, "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


#: PV refreshes the probability when it carries ``<= PV_PROBABILITY_NEED_LIMBS``
#: limbs (``decode.py``: isolate + mul-V + conversion rescale = 3, and the
#: attention output must still be bootstrappable).  Default 6 keeps two limbs of
#: margin; ``LLAMA_FHE_SOFTMAX_PV_NEED_LIMBS=5`` is the experiment knob that
#: lets a 6-limb probability (k=3 ``s2,d1`` block plan) skip that refresh.
SOFTMAX_PV_NEED_LIMBS_ENV = "LLAMA_FHE_SOFTMAX_PV_NEED_LIMBS"
PV_PROBABILITY_NEED_LIMBS = 6


def pv_probability_need_limbs() -> int:
    """``need_limbs`` of the PV probability refresh (env override, default 6)."""

    raw = os.environ.get(SOFTMAX_PV_NEED_LIMBS_ENV, "").strip()
    if not raw:
        return PV_PROBABILITY_NEED_LIMBS
    value = int(raw)
    if value < 3:
        raise ValueError(f"{SOFTMAX_PV_NEED_LIMBS_ENV} must be >= 3 (PV consumes 3 limbs), got {value}")
    return value


def probability_min_limbs() -> int:
    """Limbs a Softmax probability must carry so PV does not refresh it (``pv need + 1``)."""

    return pv_probability_need_limbs() + 1


#: Limbs the S-chain probability must still carry so the PV station does not
#: have to refresh it (``_refresh_row_ciphers_if_short(need_limbs=6)``); the
#: default of :func:`probability_min_limbs`.
SCHAIN_PROBABILITY_MIN_LIMBS = PV_PROBABILITY_NEED_LIMBS + 1


def schain_limb_schedule(config, layer_idx: int, *, exit_limbs: int) -> dict[str, int]:
    """Static limb plan of :func:`_schain_merged_tail` for one layer.

    ``exit_limbs`` is the bootstrap exit level (18 on the production L3
    program).  Returns the limbs of ``lam1`` (after the j1 polynomial),
    ``pack`` (the co-packed bootstrap input) and ``probability`` (what the
    kernel hands to PV).  :func:`schain_applicable` requires ``pack >=
    BOOTSTRAP_MIN_INPUT_LIMBS`` and ``probability >=
    SCHAIN_PROBABILITY_MIN_LIMBS``; otherwise the S-chain would cost an extra
    refresh and the Alg2 rounds are kept.
    """

    layer_idx = int(layer_idx)
    rough = config.poly_for_layer(config.invsqrt_alg2["j1_rough"], layer_idx)
    precise = config.poly_for_layer(config.invsqrt_alg2["j2_precise"], layer_idx)
    d1 = int(chebyshev_ps_mul_depth(_invsqrt_cheb(rough)[0])) + 1  # unfused map
    d2 = int(chebyshev_ps_mul_depth(_invsqrt_cheb(precise)[0]))  # map fused into the mask
    exit_limbs = int(exit_limbs)
    # y0 (exit) -> y1 = y0^2 -> S1 = isolate(reduce(y1)) -> rsqrt_j1
    lam1 = exit_limbs - 2 - d1
    # lam1 -> lam1^2 -> lam1^4 -> T = lam1^4 * S2 (co-packed with c*lam1 on token 1)
    pack = lam1 - 3
    # exit -> alpha*token0 mask -> rsqrt_j2 -> mu = lam2 * lam1'^2 -> Lambda = mu^2 -> y2' * Lambda
    probability = exit_limbs - 1 - d2 - 1 - 1 - 1
    return {"lam1": int(lam1), "pack": int(pack), "probability": int(probability)}


def schain_applicable(config, layer_idx: int, k: int, *, exit_limbs: int | None) -> bool:
    """``k == 2``, the env switch is on and the static limb plan needs no extra refresh."""

    if int(k) != 2 or not softmax_schain_enabled():
        return False
    if exit_limbs is None:
        return True  # no bootstrap operator (sim smoke): depth is the caller's business
    plan = schain_limb_schedule(config, layer_idx, exit_limbs=exit_limbs)
    return (
        plan["pack"] >= BOOTSTRAP_MIN_INPUT_LIMBS
        and plan["probability"] >= probability_min_limbs()
    )


def _schain_copack_scale(rough: dict[str, Any], precise: dict[str, Any]) -> float:
    """``c`` so that ``c * lam1`` sits around the j2 fit interval (geometric centres).

    The co-packed bootstrap input holds ``T`` (inside the j2 interval) on
    token 0 and ``c * lam1`` on token 1; the refresh noise is relative to the
    ciphertext's largest slot, so both must have the same order of magnitude.
    """

    _, lo1, hi1 = _invsqrt_cheb(rough)
    _, lo2, hi2 = _invsqrt_cheb(precise)
    lam1_centre = (max(lo1, 1e-12) * max(hi1, 1e-12)) ** -0.25
    t_centre = math.sqrt(max(lo2, 1e-12) * max(hi2, 1e-12))
    return float(t_centre / lam1_centre)


def _add_aligned(left, right, *, crypto_context):
    """``left + right`` at the lower of the two levels (neither input consumed)."""

    aligned_left, aligned_right, owned_left, owned_right = _align_cipher_pair(
        left, right, crypto_context=crypto_context
    )
    try:
        return _homo_add(aligned_left, aligned_right, crypto_context=crypto_context)
    finally:
        if owned_left:
            release_if_supported(aligned_left)
        if owned_right:
            release_if_supported(aligned_right)


def _mul_aligned(left, right, *, crypto_context):
    """``left * right`` (relin + rescale) at the lower level (neither input consumed)."""

    aligned_left, aligned_right, owned_left, owned_right = _align_cipher_pair(
        left, right, crypto_context=crypto_context
    )
    try:
        return mul_cipher_rescale(aligned_left, aligned_right, crypto_context=crypto_context)
    finally:
        if owned_left:
            release_if_supported(aligned_left)
        if owned_right:
            release_if_supported(aligned_right)


def _schain_merged_tail(
    y,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    layer_idx: int,
    masks: "_MergedPageMasks",
    page_count: int,
    stride: int,
    groups: int,
    bootstrap_operator,
    bootstrap_iterations: int | None,
    bootstrap_precision_bits: int | None,
    sparse_period: int | None,
    own,
    drop,
) -> list[tuple[object, ...]]:
    """``k = 2`` Alg2 tail as the S-chain (``s2`` block + finish); see :func:`_schain_s2_block`."""

    y2, big = _schain_s2_block(
        y,
        layout=layout,
        crypto_context=crypto_context,
        config=config,
        layer_idx=layer_idx,
        masks=masks,
        stride=stride,
        groups=groups,
        bootstrap_operator=bootstrap_operator,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_precision_bits=bootstrap_precision_bits,
        sparse_period=sparse_period,
        own=own,
        drop=drop,
        round2_poly=None,
    )
    return _finish_merged_pages(
        y2,
        big,
        layout=layout,
        crypto_context=crypto_context,
        masks=masks,
        page_count=page_count,
        stride=stride,
        own=own,
        drop=drop,
    )


def _schain_s2_block(
    y,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    layer_idx: int,
    masks: "_MergedPageMasks",
    stride: int,
    groups: int,
    bootstrap_operator,
    bootstrap_iterations: int | None,
    bootstrap_precision_bits: int | None,
    sparse_period: int | None,
    own,
    drop,
    round2_poly: dict[str, Any] | None = None,
):
    """``k = 2`` Alg2 tail as the S-chain: same probabilities, one aux bootstrap.

    Returns ``(y2, Lambda)``: the un-normalised main track ``y0^4`` and the
    token-0 row scalar ``Lambda = lam1^4 lam2^2`` (stride-periodic, not yet
    broadcast).  ``round2_poly`` overrides the round-2 rsqrt table (the
    block plan passes ``j_mid`` when round 2 is not the layer's last round).

    Alg2 with ``k = 2`` computes ``P = (y0 lam1)^4 lam2^2`` with ``lam1 =
    rsqrt(sum y0^2)`` and ``lam2 = rsqrt(sum (y0 lam1)^4) = rsqrt(lam1^4 sum
    y0^4)``.  The S-chain keeps the main track un-normalised (``y2' = y0^4``,
    two squarings, no ``lam`` multiply in between) and folds every ``lam``
    into one row scalar ``Lambda = lam1^4 lam2^2 = (lam1^2 lam2)^2`` applied
    once at the end.  Both row sums (``S1 = sum y1'``, ``S2 = sum y2'``) come
    straight off the main track, so the aux track is one dependency chain
    ``S1 -> lam1 -> T = lam1^4 S2 -> lam2 -> Lambda``; the single refresh sits
    on ``T`` and carries a scaled copy of ``lam1`` on token column 1 of the
    same (stride-periodic) ciphertext, so no second ``lam`` bootstrap is
    needed.  The Chebyshev map of the j2 polynomial is fused into the token-0
    mask after the refresh (``alpha`` in the mask, ``beta`` and the mapped
    safe fill in the add).  Limb plan: :func:`schain_limb_schedule`.

    The un-normalised main track makes the result sensitive to the dynamic
    range between rows of one ciphertext (``exp(U_hat - rowmax)`` spread):
    the plaintext study in ``profile measurements`` found
    the ``k = 2`` layers acceptable at >= 24-bit refresh precision and the
    high-``k`` layers not, which is why the kernel is gated on ``k == 2``.
    """

    layer_idx = int(layer_idx)
    rough = config.poly_for_layer(config.invsqrt_alg2["j1_rough"], layer_idx)
    precise = (
        config.poly_for_layer(config.invsqrt_alg2["j2_precise"], layer_idx)
        if round2_poly is None
        else round2_poly
    )
    fill_mask = 1.0 - masks.merged_token0
    copack = _schain_copack_scale(rough, precise)

    # main track: y1' = y0^2, y2' = y0^4 (causal zeros stay zero under squaring)
    y1 = own(square_rescale(y, crypto_context=crypto_context))
    y2 = own(square_rescale(y1, crypto_context=crypto_context))
    drop(y)

    # S1 = sum y0^2 = row sum of y1', rough rsqrt
    reduced = own(_token_reduce_cipher(y1, layout=layout, crypto_context=crypto_context))
    drop(y1)
    isolated = own(
        _mul_packed_rows(
            (reduced,), masks.merged_token0, name="decode.schain.s1.sum0", crypto_context=crypto_context
        )[0]
    )
    drop(reduced)
    total1 = own(_rotate_add_tree(isolated, stride=stride, count=groups, crypto_context=crypto_context))
    drop(isolated)
    filled1 = own(
        _add_packed_rows(
            (total1,),
            fill_mask * _safe_invsqrt_fill(rough),
            name="decode.schain.s1.invsqrt_fill",
            crypto_context=crypto_context,
        )[0]
    )
    drop(total1)
    refreshed = _refresh_row_ciphers_if_short(
        (filled1,),
        bootstrap_operator=bootstrap_operator,
        need_limbs=_invsqrt_need_limbs(rough),
        iterations=bootstrap_iterations,
        precision_bits=bootstrap_precision_bits,
        tag="softmax.schain.s1.fill",
        periodic_slots=sparse_period,
    )
    if refreshed[0] is not filled1:
        drop(filled1)
        filled1 = own(refreshed[0])
    lam1 = own(eval_invsqrt_cipher(filled1, rough, crypto_context=crypto_context))
    drop(filled1)

    # S2 = sum y0^4 = row sum of y2' (token-0 only, no fill: it is multiplied, not fed to a poly)
    reduced = own(_token_reduce_cipher(y2, layout=layout, crypto_context=crypto_context))
    isolated = own(
        _mul_packed_rows(
            (reduced,), masks.merged_token0, name="decode.schain.s2.sum0", crypto_context=crypto_context
        )[0]
    )
    drop(reduced)
    total2 = own(_rotate_add_tree(isolated, stride=stride, count=groups, crypto_context=crypto_context))
    drop(isolated)

    # T = lam1^4 * S2 on token 0; c * lam1 on token 1; one sparse refresh for both
    lam1_sq = own(square_rescale(lam1, crypto_context=crypto_context))
    lam1_q = own(square_rescale(lam1_sq, crypto_context=crypto_context))
    drop(lam1_sq)
    t_cipher = own(_mul_aligned(lam1_q, total2, crypto_context=crypto_context))
    drop(lam1_q)
    drop(total2)
    lam1_copy = own(
        _mul_packed_rows(
            (lam1,), masks.merged_token0 * copack, name="decode.schain.lam1.copack", crypto_context=crypto_context
        )[0]
    )
    drop(lam1)
    lam1_shift = own(_homo_rotate(lam1_copy, -2, crypto_context=crypto_context))  # token 0 -> token 1
    drop(lam1_copy)
    pack = own(_add_aligned(t_cipher, lam1_shift, crypto_context=crypto_context))
    drop(t_cipher)
    drop(lam1_shift)
    refreshed = _refresh_row_ciphers(
        (pack,),
        bootstrap_operator=bootstrap_operator,
        iterations=bootstrap_iterations,
        precision_bits=bootstrap_precision_bits,
        tag="softmax.schain.pack",
        periodic_slots=sparse_period,
    )
    if refreshed[0] is not pack:
        drop(pack)
        pack = own(refreshed[0])

    # lam2 = rsqrt_j2(T) with the Chebyshev map fused into the token-0 mask
    _, lo2, hi2 = _invsqrt_cheb(precise)
    alpha = 2.0 / (hi2 - lo2)
    beta = -1.0 - 2.0 * lo2 / (hi2 - lo2)
    mapped_safe = alpha * _safe_invsqrt_fill(precise) + beta
    t_scaled = own(
        _mul_packed_rows(
            (pack,), masks.merged_token0 * alpha, name="decode.schain.t.alpha", crypto_context=crypto_context
        )[0]
    )
    t_mapped = own(
        _add_packed_rows(
            (t_scaled,),
            masks.merged_token0 * beta + fill_mask * mapped_safe,
            name="decode.schain.t.mapped_fill",
            crypto_context=crypto_context,
        )[0]
    )
    drop(t_scaled)
    lam2 = own(
        eval_invsqrt_cipher(
            t_mapped, precise, crypto_context=crypto_context, input_is_chebyshev_mapped=True
        )
    )
    drop(t_mapped)
    # lam1 back from token 1 (token-0 only after the mask, which also undoes c)
    lam1_back = own(_homo_rotate(pack, 2, crypto_context=crypto_context))
    drop(pack)
    lam1_b = own(
        _mul_packed_rows(
            (lam1_back,),
            masks.merged_token0 * (1.0 / copack),
            name="decode.schain.lam1.uncopack",
            crypto_context=crypto_context,
        )[0]
    )
    drop(lam1_back)
    lam1_b_sq = own(square_rescale(lam1_b, crypto_context=crypto_context))
    drop(lam1_b)
    # Lambda = (lam1^2 lam2)^2, token-0 only (lam1_b_sq zeroes lam2's fill slots)
    mu = own(_mul_aligned(lam2, lam1_b_sq, crypto_context=crypto_context))
    drop(lam2)
    drop(lam1_b_sq)
    big = own(square_rescale(mu, crypto_context=crypto_context))
    drop(mu)
    return y2, big


def _finish_merged_pages(
    y,
    big,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    masks: "_MergedPageMasks",
    page_count: int,
    stride: int,
    own,
    drop,
    name: str = "decode.schain",
) -> list[tuple[object, ...]]:
    """Un-merge the main track page by page and apply the pending row scalar.

    ``big`` (token-0 row scalar, stride-periodic after the group tree, or
    ``None`` for an already normalised main track) is broadcast along the
    token axis once; the per-page prefix mask goes on whichever side has more
    limbs left, so the probability keeps ``min(limbs(y), limbs(big)) - 1``.
    """

    outputs: list[tuple[object, ...]] = []
    if big is None:
        for group in range(int(page_count)):
            back = own(_homo_rotate(y, stride * group, crypto_context=crypto_context))
            page_out = _mul_packed_rows(
                (back,),
                masks.page_valid[group],
                name=f"{name}.prefix.g{group}",
                crypto_context=crypto_context,
            )[0]
            drop(back)
            outputs.append((page_out,))
        drop(y)
        return outputs
    big_b = own(_token_broadcast_cipher(big, layout=layout, crypto_context=crypto_context))
    drop(big)
    mask_big = int(big_b.state.cur_limbs) >= int(y.state.cur_limbs)
    for group in range(int(page_count)):
        back = own(_homo_rotate(y, stride * group, crypto_context=crypto_context))
        if mask_big:
            # Lambda has the limbs to spare: fold the page mask into it.
            big_g = own(
                _mul_packed_rows(
                    (big_b,),
                    masks.page_valid[group],
                    name=f"{name}.lambda.g{group}",
                    crypto_context=crypto_context,
                )[0]
            )
            page_out = _mul_aligned(back, big_g, crypto_context=crypto_context)
            drop(big_g)
        else:
            page_y = own(
                _mul_packed_rows(
                    (back,),
                    masks.page_valid[group],
                    name=f"{name}.prefix.g{group}",
                    crypto_context=crypto_context,
                )[0]
            )
            page_out = _mul_aligned(page_y, big_b, crypto_context=crypto_context)
            drop(page_y)
        drop(back)
        outputs.append((page_out,))
    drop(y)
    drop(big_b)
    return outputs


# --------------------------------------------------------------------------
# Block-plan Softmax for k >= 3 (``s``/``d`` segments; profile measurements §F)
# --------------------------------------------------------------------------

#: ``LLAMA_FHE_SOFTMAX_BLOCKPLAN="3:s2,d1;4:s2,d2"`` runs the merged-page
#: Softmax of a ``k``-round layer as the listed blocks instead of ``k`` Alg2
#: rounds.  ``sB``: S-chain block straight off exp (square ``B`` times, no
#: normalisation; only allowed first, ``B <= 2``).  ``dB``: multiply the main
#: track once by ``pending * rsqrt(sum y^2)`` (L2-normalises every row, max
#: in ``[1/sqrt(n), 1]``), square ``B`` times (``B <= 2`` keeps the
#: in-ciphertext range <= n^2), and -- for ``B = 2`` -- leave ``1/sum y`` as
#: the next pending scalar.  Algebraically identical to Alg2 (every rsqrt
#: sees Alg2's round input); the plaintext study found ``s2,d1`` (k=3) at
#: Alg2 precision and ``s2,d2`` (k=4) within 2x, while ``s2,s2`` fails
#: (normalised flat rows squared twice fall under the CKKS noise floor).
SOFTMAX_BLOCKPLAN_ENV = "LLAMA_FHE_SOFTMAX_BLOCKPLAN"
#: ``LLAMA_FHE_SOFTMAX_BLOCKPLAN_FORCE=1`` runs the plan even when the static
#: limb schedule predicts a probability below ``SCHAIN_PROBABILITY_MIN_LIMBS``
#: (the PV station then refreshes it; useful to measure).
SOFTMAX_BLOCKPLAN_FORCE_ENV = "LLAMA_FHE_SOFTMAX_BLOCKPLAN_FORCE"
#: Final-round ``lambda`` is refreshed when it carries fewer limbs than this:
#: ``Lambda = lambda^2`` (token-0 mask + square) must still leave the
#: probability ``>= probability_min_limbs()`` (default
#: ``SCHAIN_PROBABILITY_MIN_LIMBS + 3``).
BLOCKPLAN_FINAL_LAMBDA_NEED_LIMBS = SCHAIN_PROBABILITY_MIN_LIMBS + 3


def blockplan_final_lambda_need_limbs() -> int:
    return probability_min_limbs() + 3


def parse_softmax_blockplan(spec: str) -> dict[int, tuple[tuple[str, int], ...]]:
    """``"3:s2,d1;4:s2,d2"`` -> ``{3: (("s", 2), ("d", 1)), 4: (("s", 2), ("d", 2))}``."""

    plans: dict[int, tuple[tuple[str, int], ...]] = {}
    for item in str(spec).split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{SOFTMAX_BLOCKPLAN_ENV}: expected 'k:blocks', got {item!r}")
        k_text, blocks_text = item.split(":", 1)
        k = int(k_text)
        blocks: list[tuple[str, int]] = []
        for token in blocks_text.split(","):
            token = token.strip().lower()
            if len(token) != 2 or token[0] not in "sd" or not token[1].isdigit():
                raise ValueError(f"{SOFTMAX_BLOCKPLAN_ENV}: bad block {token!r} in {item!r}")
            kind, width = token[0], int(token[1])
            if width not in (1, 2):
                raise ValueError(f"{SOFTMAX_BLOCKPLAN_ENV}: block width must be 1 or 2 ({token!r})")
            if kind == "s" and blocks:
                raise ValueError(f"{SOFTMAX_BLOCKPLAN_ENV}: an 's' block is only allowed first ({item!r})")
            blocks.append((kind, width))
        if not blocks or sum(width for _, width in blocks) != k:
            raise ValueError(f"{SOFTMAX_BLOCKPLAN_ENV}: block widths of {item!r} must sum to k={k}")
        plans[k] = tuple(blocks)
    return plans


def softmax_blockplan_for(k: int) -> tuple[tuple[str, int], ...] | None:
    """The env block plan for a ``k``-round layer, or ``None`` (Alg2 rounds)."""

    spec = os.environ.get(SOFTMAX_BLOCKPLAN_ENV, "").strip()
    if not spec:
        return None
    return parse_softmax_blockplan(spec).get(int(k))


def _blockplan_round_poly(config, layer_idx: int, round_index: int, k: int) -> dict[str, Any]:
    """rsqrt table of Alg2 round ``round_index``: ``j1_rough`` / ``j_mid`` (if the
    table carries one) / ``j2_precise`` for the last round."""

    tables = config.invsqrt_alg2
    if int(round_index) == 1:
        entry = tables["j1_rough"]
    elif int(round_index) < int(k) and tables.get("j_mid") is not None:
        entry = tables["j_mid"]
    else:
        entry = tables["j2_precise"]
    return config.poly_for_layer(entry, int(layer_idx))


def _interval_centre(poly_entry: dict[str, Any]) -> float:
    _, lo, hi = _invsqrt_cheb(poly_entry)
    return math.sqrt(max(lo, 1e-12) * max(hi, 1e-12))


def _copack_scale(scalar_centre: float, target_poly: dict[str, Any]) -> float:
    """``c`` so that ``c * scalar`` sits around the target rsqrt interval (refresh noise
    is relative to the largest slot of the co-packed ciphertext)."""

    return float(_interval_centre(target_poly) / max(float(scalar_centre), 1e-300))


def _rsqrt_fused(
    pack,
    poly: dict[str, Any],
    *,
    masks: "_MergedPageMasks",
    crypto_context,
    own,
    drop,
    name: str,
):
    """``rsqrt(token-0 value)`` with the Chebyshev map fused into the token-0 mask
    (``alpha`` in the mask, ``beta`` and the mapped safe fill in the add); the
    other slots (fill and the co-packed scalar) are overwritten by the safe fill.
    ``pack`` is not consumed."""

    fill_mask = 1.0 - masks.merged_token0
    _, lo, hi = _invsqrt_cheb(poly)
    alpha = 2.0 / (hi - lo)
    beta = -1.0 - 2.0 * lo / (hi - lo)
    mapped_safe = alpha * _safe_invsqrt_fill(poly) + beta
    scaled = own(
        _mul_packed_rows(
            (pack,), masks.merged_token0 * alpha, name=f"{name}.alpha", crypto_context=crypto_context
        )[0]
    )
    mapped = own(
        _add_packed_rows(
            (scaled,),
            masks.merged_token0 * beta + fill_mask * mapped_safe,
            name=f"{name}.mapped_fill",
            crypto_context=crypto_context,
        )[0]
    )
    drop(scaled)
    lam = own(
        eval_invsqrt_cipher(mapped, poly, crypto_context=crypto_context, input_is_chebyshev_mapped=True)
    )
    drop(mapped)
    return lam


def _row_total(
    cipher,
    *,
    layout: AttentionPairLayout,
    masks: "_MergedPageMasks",
    stride: int,
    groups: int,
    crypto_context,
    own,
    drop,
    name: str,
):
    """Row sum of ``cipher`` on token column 0 of every group (stride-periodic).  Not consumed."""

    reduced = own(_token_reduce_cipher(cipher, layout=layout, crypto_context=crypto_context))
    isolated = own(
        _mul_packed_rows((reduced,), masks.merged_token0, name=f"{name}.sum0", crypto_context=crypto_context)[0]
    )
    drop(reduced)
    total = own(_rotate_add_tree(isolated, stride=stride, count=groups, crypto_context=crypto_context))
    drop(isolated)
    return total


def _s1_block(
    y,
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    layer_idx: int,
    masks: "_MergedPageMasks",
    stride: int,
    groups: int,
    bootstrap_operator,
    bootstrap_iterations: int | None,
    bootstrap_precision_bits: int | None,
    sparse_period: int | None,
    own,
    drop,
    poly: dict[str, Any],
):
    """Alg2 round 1 with the normalisation deferred: main ``y1' = y0^2``, pending
    ``Lambda = lam1^2`` (token 0).  Returns ``(y1', Lambda, Lambda_centre)``."""

    fill_mask = 1.0 - masks.merged_token0
    y1 = own(square_rescale(y, crypto_context=crypto_context))
    drop(y)
    total = _row_total(
        y1, layout=layout, masks=masks, stride=stride, groups=groups, crypto_context=crypto_context,
        own=own, drop=drop, name="decode.plan.s1",
    )
    filled = own(
        _add_packed_rows(
            (total,), fill_mask * _safe_invsqrt_fill(poly), name="decode.plan.s1.invsqrt_fill",
            crypto_context=crypto_context,
        )[0]
    )
    drop(total)
    refreshed = _refresh_row_ciphers_if_short(
        (filled,),
        bootstrap_operator=bootstrap_operator,
        need_limbs=_invsqrt_need_limbs(poly),
        iterations=bootstrap_iterations,
        precision_bits=bootstrap_precision_bits,
        tag="softmax.plan.s1.fill",
        periodic_slots=sparse_period,
    )
    if refreshed[0] is not filled:
        drop(filled)
        filled = own(refreshed[0])
    lam = own(eval_invsqrt_cipher(filled, poly, crypto_context=crypto_context))
    drop(filled)
    lam0 = own(
        _mul_packed_rows((lam,), masks.merged_token0, name="decode.plan.s1.lambda.token0", crypto_context=crypto_context)[0]
    )
    drop(lam)
    big = own(square_rescale(lam0, crypto_context=crypto_context))
    drop(lam0)
    return y1, big, 1.0 / _interval_centre(poly)


def _d_block(
    y,
    big,
    big_centre: float,
    *,
    squarings: int,
    round_index: int,
    k: int,
    last_block: bool,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    layer_idx: int,
    masks: "_MergedPageMasks",
    stride: int,
    groups: int,
    bootstrap_operator,
    bootstrap_iterations: int | None,
    bootstrap_precision_bits: int | None,
    sparse_period: int | None,
    own,
    drop,
):
    """``dB`` block: ``z = y * (Lambda * rsqrt(Lambda^2 sum y^2))`` (L2-normalised rows),
    ``z`` squared ``B`` times; for ``B = 2`` the next pending scalar is
    ``1/sum z^4 = rsqrt(sum z^4)^2``.  Returns ``(y, Lambda, Lambda_centre)``.

    With a pending ``Lambda`` the rsqrt input ``T = Lambda^2 * S`` is co-packed
    with ``c * Lambda`` on token 1 and refreshed once (the S-chain trick), so
    the block costs one aux bootstrap for ``lambda_r`` (plus one for the
    ``B = 2`` tail when its row sum is short).  Without a pending scalar it is
    an Alg2 round (fill refresh when short, unconditional ``lambda`` refresh).
    """

    fill_mask = 1.0 - masks.merged_token0
    poly = _blockplan_round_poly(config, layer_idx, round_index, k)
    tag = f"softmax.plan.r{int(round_index)}"
    name = f"decode.plan.r{int(round_index)}"

    refreshed = _refresh_row_ciphers_if_short(
        (y,),
        bootstrap_operator=bootstrap_operator,
        need_limbs=ALG2_ROUND_NEED_LIMBS,
        iterations=bootstrap_iterations,
        precision_bits=bootstrap_precision_bits,
        tag=f"{tag}.y",
    )
    if refreshed[0] is not y:
        drop(y)
        y = own(refreshed[0])

    squared = own(square_rescale(y, crypto_context=crypto_context))
    total = _row_total(
        squared, layout=layout, masks=masks, stride=stride, groups=groups, crypto_context=crypto_context,
        own=own, drop=drop, name=name,
    )
    drop(squared)

    if big is not None:
        # T = Lambda^2 * S (token 0), c * Lambda on token 1, one sparse refresh for both
        copack = _copack_scale(big_centre, poly)
        big_sq = own(square_rescale(big, crypto_context=crypto_context))
        t_cipher = own(_mul_aligned(big_sq, total, crypto_context=crypto_context))
        drop(big_sq)
        drop(total)
        big_copy = own(
            _mul_packed_rows(
                (big,), masks.merged_token0 * copack, name=f"{name}.lambda.copack", crypto_context=crypto_context
            )[0]
        )
        drop(big)
        big_shift = own(_homo_rotate(big_copy, -2, crypto_context=crypto_context))  # token 0 -> token 1
        drop(big_copy)
        pack = own(_add_aligned(t_cipher, big_shift, crypto_context=crypto_context))
        drop(t_cipher)
        drop(big_shift)
        refreshed = _refresh_row_ciphers(
            (pack,),
            bootstrap_operator=bootstrap_operator,
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag=f"{tag}.pack",
            periodic_slots=sparse_period,
        )
        if refreshed[0] is not pack:
            drop(pack)
            pack = own(refreshed[0])
        lam = _rsqrt_fused(
            pack, poly, masks=masks, crypto_context=crypto_context, own=own, drop=drop, name=f"{name}.t"
        )
        big_back = own(_homo_rotate(pack, 2, crypto_context=crypto_context))
        drop(pack)
        big_b = own(
            _mul_packed_rows(
                (big_back,), masks.merged_token0 * (1.0 / copack), name=f"{name}.lambda.uncopack",
                crypto_context=crypto_context,
            )[0]
        )
        drop(big_back)
        # m = lam * Lambda is token-0 only (Lambda zeroes lam's fill slots)
        m = own(_mul_aligned(lam, big_b, crypto_context=crypto_context))
        drop(lam)
        drop(big_b)
    else:
        filled = own(
            _add_packed_rows(
                (total,), fill_mask * _safe_invsqrt_fill(poly), name=f"{name}.invsqrt_fill",
                crypto_context=crypto_context,
            )[0]
        )
        drop(total)
        refreshed = _refresh_row_ciphers_if_short(
            (filled,),
            bootstrap_operator=bootstrap_operator,
            need_limbs=_invsqrt_need_limbs(poly),
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag=f"{tag}.fill",
            periodic_slots=sparse_period,
        )
        if refreshed[0] is not filled:
            drop(filled)
            filled = own(refreshed[0])
        lam = own(eval_invsqrt_cipher(filled, poly, crypto_context=crypto_context))
        drop(filled)
        refreshed = _refresh_row_ciphers(
            (lam,),
            bootstrap_operator=bootstrap_operator,
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag=f"{tag}.lam",
            periodic_slots=sparse_period,
        )
        if refreshed[0] is not lam:
            drop(lam)
            lam = own(refreshed[0])
        m = own(
            _mul_packed_rows((lam,), masks.merged_token0, name=f"{name}.lambda.token0", crypto_context=crypto_context)[0]
        )
        drop(lam)

    # main track: z = y * m (broadcast along the token axis; y is already zero
    # off the occupied prefix, so no prefix mask is spent here), then B squarings
    m_b = own(_token_broadcast_cipher(m, layout=layout, crypto_context=crypto_context))
    drop(m)
    z = own(_mul_aligned(y, m_b, crypto_context=crypto_context))
    drop(y)
    drop(m_b)
    for _ in range(int(squarings)):
        nxt = own(square_rescale(z, crypto_context=crypto_context))
        drop(z)
        z = nxt
    if int(squarings) == 1:
        return z, None, 1.0

    # B = 2: pending 1/sum z^4 = rsqrt_{r+1}(sum z^4)^2 (sum of z^4 itself, not squared)
    poly_next = _blockplan_round_poly(config, layer_idx, round_index + 1, k)
    tag_next = f"softmax.plan.r{int(round_index) + 1}"
    name_next = f"decode.plan.r{int(round_index) + 1}"
    total = _row_total(
        z, layout=layout, masks=masks, stride=stride, groups=groups, crypto_context=crypto_context,
        own=own, drop=drop, name=name_next,
    )
    filled = own(
        _add_packed_rows(
            (total,), fill_mask * _safe_invsqrt_fill(poly_next), name=f"{name_next}.invsqrt_fill",
            crypto_context=crypto_context,
        )[0]
    )
    drop(total)
    refreshed = _refresh_row_ciphers_if_short(
        (filled,),
        bootstrap_operator=bootstrap_operator,
        need_limbs=_invsqrt_need_limbs(poly_next),
        iterations=bootstrap_iterations,
        precision_bits=bootstrap_precision_bits,
        tag=f"{tag_next}.fill",
        periodic_slots=sparse_period,
    )
    if refreshed[0] is not filled:
        drop(filled)
        filled = own(refreshed[0])
    lam = own(eval_invsqrt_cipher(filled, poly_next, crypto_context=crypto_context))
    drop(filled)
    # the final normaliser must leave the probability >= probability_min_limbs()
    # (a non-last block's pending scalar feeds the next co-pack, same bound)
    refreshed = _refresh_row_ciphers_if_short(
        (lam,),
        bootstrap_operator=bootstrap_operator,
        need_limbs=blockplan_final_lambda_need_limbs(),
        iterations=bootstrap_iterations,
        precision_bits=bootstrap_precision_bits,
        tag=f"{tag_next}.lam",
        periodic_slots=sparse_period,
    )
    if refreshed[0] is not lam:
        drop(lam)
        lam = own(refreshed[0])
    lam0 = own(
        _mul_packed_rows((lam,), masks.merged_token0, name=f"{name_next}.lambda.token0", crypto_context=crypto_context)[0]
    )
    drop(lam)
    big = own(square_rescale(lam0, crypto_context=crypto_context))
    drop(lam0)
    return z, big, 1.0 / _interval_centre(poly_next)


def _blockplan_merged_tail(
    y,
    *,
    plan: tuple[tuple[str, int], ...],
    k: int,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    layer_idx: int,
    masks: "_MergedPageMasks",
    page_count: int,
    stride: int,
    groups: int,
    bootstrap_operator,
    bootstrap_iterations: int | None,
    bootstrap_precision_bits: int | None,
    sparse_period: int | None,
    own,
    drop,
) -> list[tuple[object, ...]]:
    """Alg2 tail of a ``k``-round layer as the ``plan`` blocks (see :data:`SOFTMAX_BLOCKPLAN_ENV`)."""

    common = dict(
        layout=layout,
        crypto_context=crypto_context,
        config=config,
        layer_idx=layer_idx,
        masks=masks,
        stride=stride,
        groups=groups,
        bootstrap_operator=bootstrap_operator,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_precision_bits=bootstrap_precision_bits,
        sparse_period=sparse_period,
        own=own,
        drop=drop,
    )
    big = None
    big_centre = 1.0
    round_index = 0
    for index, (kind, width) in enumerate(plan):
        last_block = index + 1 == len(plan)
        if kind == "s":
            if index != 0:
                raise ValueError("block plan: an 's' block is only allowed first.")
            if width == 2:
                round2 = _blockplan_round_poly(config, layer_idx, 2, k)
                y, big = _schain_s2_block(y, round2_poly=round2, **common)
                rough = _blockplan_round_poly(config, layer_idx, 1, k)
                big_centre = 1.0 / (_interval_centre(rough) ** 2 * _interval_centre(round2))
            else:
                y, big, big_centre = _s1_block(y, poly=_blockplan_round_poly(config, layer_idx, 1, k), **common)
            round_index += width
            continue
        y, big, big_centre = _d_block(
            y,
            big,
            big_centre,
            squarings=width,
            round_index=round_index + 1,
            k=k,
            last_block=last_block,
            **common,
        )
        round_index += width
    if round_index != int(k):
        raise ValueError(f"block plan {plan!r} covers {round_index} rounds, layer has k={k}.")
    if big is not None and big.state.cur_limbs < 2:
        # pending scalar too shallow for the final multiply: refresh it (counted)
        refreshed = _refresh_row_ciphers(
            (big,),
            bootstrap_operator=bootstrap_operator,
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag="softmax.plan.final.lambda",
            periodic_slots=sparse_period,
        )
        if refreshed[0] is not big:
            drop(big)
            big = own(refreshed[0])
    return _finish_merged_pages(
        y,
        big,
        layout=layout,
        crypto_context=crypto_context,
        masks=masks,
        page_count=page_count,
        stride=stride,
        own=own,
        drop=drop,
        name="decode.plan",
    )


def blockplan_limb_schedule(
    config, layer_idx: int, plan: tuple[tuple[str, int], ...], *, k: int, exit_limbs: int
) -> dict[str, Any]:
    """Static limb / bootstrap plan of :func:`_blockplan_merged_tail` (mirrors its guards).

    Returns ``probability`` (limbs handed to PV), ``bootstraps`` (aux refreshes
    after the post-exp one), ``min_pack`` (smallest co-packed bootstrap input;
    must be ``>= BOOTSTRAP_MIN_INPUT_LIMBS``) and the per-step trace.
    """

    exit_limbs = int(exit_limbs)
    layer_idx = int(layer_idx)

    def depth(poly):
        return int(chebyshev_ps_mul_depth(_invsqrt_cheb(poly)[0]))

    y = exit_limbs
    big = None
    bs = 0
    packs: list[int] = []
    trace: list[str] = []
    r = 0
    for index, (kind, width) in enumerate(plan):
        if kind == "s":
            r1 = _blockplan_round_poly(config, layer_idx, 1, k)
            if width == 2:
                r2 = _blockplan_round_poly(config, layer_idx, 2, k)
                y1, y2 = y - 1, y - 2
                s1 = y1 - 1
                if s1 < _invsqrt_need_limbs(r1):
                    bs += 1
                    s1 = exit_limbs
                lam1 = s1 - depth(r1) - 1
                t = min(lam1 - 2, y2 - 1) - 1
                pack = min(t, lam1 - 1)
                packs.append(pack)
                bs += 1
                lam2 = exit_limbs - 1 - depth(r2)
                mu = min(lam2, exit_limbs - 2) - 1
                y, big = y2, mu - 1
                trace.append(f"s2: y={y} Lambda={big} pack={pack}")
            else:
                y1 = y - 1
                s1 = y1 - 1
                if s1 < _invsqrt_need_limbs(r1):
                    bs += 1
                    s1 = exit_limbs
                lam1 = s1 - depth(r1) - 1
                y, big = y1, lam1 - 2
                trace.append(f"s1: y={y} Lambda={big}")
            r += width
            continue
        r += 1
        poly = _blockplan_round_poly(config, layer_idx, r, k)
        if y < ALG2_ROUND_NEED_LIMBS:
            bs += 1
            y = exit_limbs
        s = (y - 1) - 1
        if big is not None:
            t = min(big - 1, s) - 1
            pack = min(t, big - 1)
            packs.append(pack)
            bs += 1
            lam = exit_limbs - 1 - depth(poly)
            m = min(lam, exit_limbs - 1) - 1
        else:
            if s < _invsqrt_need_limbs(poly):
                bs += 1
                s = exit_limbs
            bs += 1  # unconditional lambda refresh (Alg2 round)
            m = exit_limbs - 1
        y = min(y, m) - 1 - width
        if width == 1:
            big = None
            trace.append(f"d1(r{r}): y={y}")
        else:
            r += 1
            poly_next = _blockplan_round_poly(config, layer_idx, r, k)
            s2 = y - 1
            if s2 < _invsqrt_need_limbs(poly_next):
                bs += 1
                s2 = exit_limbs
            lam = s2 - depth(poly_next) - 1
            if lam < blockplan_final_lambda_need_limbs():
                bs += 1
                lam = exit_limbs
            big = lam - 2
            trace.append(f"d2(r{r - 1},r{r}): y={y} Lambda={big}")
    if big is None:
        probability = y - 1
    else:
        if big < 2:
            bs += 1
            big = exit_limbs
        probability = min(y, big) - 1
    return {
        "probability": int(probability),
        "bootstraps": int(bs),
        "min_pack": int(min(packs)) if packs else None,
        "trace": trace,
    }


def blockplan_applicable(config, layer_idx: int, k: int, *, exit_limbs: int | None):
    """The env block plan for ``k`` when it is limb-feasible, else ``None`` (Alg2 rounds)."""

    plan = softmax_blockplan_for(k)
    if plan is None or int(k) < 2:
        return None
    if exit_limbs is None:
        return plan  # no bootstrap operator (sim smoke): depth is the caller's business
    sched = blockplan_limb_schedule(config, layer_idx, plan, k=int(k), exit_limbs=int(exit_limbs))
    if sched["min_pack"] is not None and sched["min_pack"] < BOOTSTRAP_MIN_INPUT_LIMBS:
        return None
    force = os.environ.get(SOFTMAX_BLOCKPLAN_FORCE_ENV, "0").strip().lower() in ("1", "true", "on", "yes")
    if sched["probability"] < probability_min_limbs() and not force:
        return None
    return plan


def softmax_sparse_bootstrap_enabled() -> bool:
    """Whether the merged-page Softmax may route its periodic rows through a sparse bootstrap program.

    On by default; the operator still has to carry a program of period
    ``decode_page_merge_stride`` slots (``preset.bootstrap_extra_sparse_log_slots``),
    otherwise the dense program is used.  ``LLAMA_FHE_SOFTMAX_SPARSE_BS=0``
    forces dense (A/B baseline).
    """

    return os.environ.get(SOFTMAX_SPARSE_BS_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def merged_page_sparse_period(layout, bootstrap_operator) -> int | None:
    """Slot period of the merged-page ``total`` / ``filled`` / ``lambda`` rows if it can go sparse.

    The packed payload (``groups * stride`` slots) tiles the slot vector, so
    after the cyclic group rotate-add every ``stride``-slot block of the
    merged ciphertext carries the same ``sum(y^2)`` row, ``fill_mask = 1 -
    merged_token0`` is itself a ``stride``-periodic pattern, and ``1/sqrt`` is
    slot-wise: those rows are exactly periodic with period ``stride`` and may
    be bootstrapped as a ``log2(stride)``-slot sparse view (production: 4096
    slots, log_slots 12).  ``None`` when the operator has no such program,
    the payload does not tile the slot vector, or the env switch is off.
    """

    if bootstrap_operator is None or not softmax_sparse_bootstrap_enabled():
        return None
    stride = int(layout.decode_page_merge_stride)
    groups = int(layout.decode_page_merge_groups)
    if int(layout.slots) % (stride * groups) or stride & (stride - 1):
        return None
    probe = getattr(bootstrap_operator, "supports_periodic_slots", None)
    if not callable(probe) or not probe(stride):
        return None
    return stride


def softmax_merge_pages_enabled() -> bool:
    return os.environ.get(SOFTMAX_MERGE_PAGES_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def softmax_page_merge_rotations(layout) -> tuple[int, ...]:
    """Rotation keys of :func:`_softmax_alg2_merged_pages_fhe` for any layout
    exposing ``seq_len`` / ``head_dim`` / ``query_heads`` (compact
    :class:`AttentionPairLayout` or ``PairFold128Layout``)."""

    pairs = int(layout.query_heads) // 2
    stride = 2 * int(layout.seq_len) * pairs
    groups = int(layout.head_dim) // pairs
    rotations: list[int] = []
    for group in range(1, groups):
        rotations.append(-stride * group)
        rotations.append(stride * group)
    bit = 1
    while bit < groups:
        rotations.append(stride * bit)
        bit <<= 1
    return tuple(dict.fromkeys(rotations))


@dataclass(frozen=True)
class _MergedPageMasks:
    """Packed ``[1, slots]`` masks of a merged-page Softmax.

    ``page_valid[g]`` is page ``g``'s own (prefix ∧ owned head rows) mask on
    rows ``0 .. pairs-1`` (used before the merge and after the un-merge);
    ``merged_valid`` is their union after shifting page ``g`` by ``g *
    stride`` slots; ``merged_token0`` selects column 0 of the owned rows in
    *every* group (also the empty ones), so after the cyclic group
    rotate-add every group's column 0 carries the full ``sum(y^2)`` and the
    ``1/sqrt`` fill never has to cover a total-plus-fill slot.
    """

    page_valid: tuple[np.ndarray, ...]
    merged_valid: np.ndarray
    merged_token0: np.ndarray


def _merged_page_masks(
    layout: AttentionPairLayout,
    occupied_pages: tuple[int, ...],
    head_mask: np.ndarray | None,
) -> _MergedPageMasks:
    stride = int(layout.decode_page_merge_stride)
    groups = int(layout.decode_page_merge_groups)
    slots = int(layout.slots)
    if stride * groups != int(layout.payload_slots):
        raise ValueError(
            "merged pages need stride*groups == payload_slots "
            f"({stride}*{groups} != {layout.payload_slots})."
        )
    page_valid = tuple(
        np.asarray(
            _with_head_rows(layout.compact_decode_prefix_mask(occupied), head_mask),
            dtype=np.float64,
        ).reshape(1, slots)
        for occupied in occupied_pages
    )
    merged_valid = np.zeros((1, slots), dtype=np.float64)
    for group, valid in enumerate(page_valid):
        merged_valid += np.roll(valid, stride * group, axis=1)
    token0 = np.asarray(
        _with_head_rows(layout.compact_decode_token0_mask(), head_mask), dtype=np.float64
    ).reshape(1, slots)
    merged_token0 = np.zeros((1, slots), dtype=np.float64)
    for group in range(groups):
        merged_token0 += np.roll(token0, stride * group, axis=1)
    return _MergedPageMasks(
        page_valid=page_valid,
        merged_valid=np.minimum(merged_valid, 1.0),
        merged_token0=np.minimum(merged_token0, 1.0),
    )


def _homo_add(left, right, *, crypto_context):
    return crypto_context.fhe.homo_add(left, right, crypto_context.context)


def _softmax_alg2_merged_pages_fhe(
    score_pages: tuple[tuple[object, ...], ...],
    *,
    layout: AttentionPairLayout,
    crypto_context,
    config,
    occupied_pages: tuple[int, ...],
    layer_idx: int,
    rounds: int | None,
    score_scale: float | None,
    bootstrap_operator,
    bootstrap_iterations: int | None,
    bootstrap_precision_bits: int | None,
    head_rows,
) -> tuple[tuple[object, ...], ...]:
    """Alg2 over ``G`` compact score pages packed into one ciphertext.

    Slot-for-slot the same arithmetic as :func:`softmax_alg2_row_pages_fhe`
    (public map, exp, prefix mask, ``k`` rounds of ``y <- (y*λ)^2`` with
    ``λ = 1/sqrt(sum y^2)``), but page ``g`` lives on rows ``g*pairs ..`` of
    one ciphertext: one exp, one post-exp bootstrap and one set of round
    multiplications serve every page, and the cross-page ``sum(y^2)`` is a
    3-rotation group tree instead of per-page adds.  Returns one
    probability ciphertext per page, back on rows ``0 .. pairs-1``.
    """

    layout._require_compact_q()
    pages = tuple(tuple(page) for page in score_pages)
    for page in pages:
        if len(page) != 1:
            raise ValueError("merged-page Softmax expects 1 compact cipher per page.")
    stride = int(layout.decode_page_merge_stride)
    groups = int(layout.decode_page_merge_groups)
    head_mask = _head_rows_mask(layout, head_rows)
    masks = _merged_page_masks(layout, occupied_pages, head_mask)
    k = int(config.k_for(layer_idx) if rounds is None else rounds)
    exp_entry = config.layer_exp[int(layer_idx)]
    scale = (
        1.0 / math.sqrt(float(layout.head_dim)) if score_scale is None else float(score_scale)
    )
    shift = -float(config.U_hat_for(layer_idx)) / float(2**k)
    invalid_value = float(config.stats_min_for(layer_idx)) / float(2**k)
    fill_mask = 1.0 - masks.merged_token0
    # ``total`` / ``filled`` / ``lam`` are ``stride``-periodic (see
    # merged_page_sparse_period); ``y`` is not (every page differs).
    sparse_period = merged_page_sparse_period(layout, bootstrap_operator)
    live: list[object] = []  # everything still to release

    def own(cipher):
        live.append(cipher)
        return cipher

    def drop(cipher):
        if cipher is not None and cipher in live:
            live.remove(cipher)
            release_if_supported(cipher)

    try:
        # 1. public map, page by page, merged by rotation into one ciphertext
        merged = None
        for group, page in enumerate(pages):
            scaled = own(
                _mul_packed_rows(
                    page,
                    masks.page_valid[group] * (scale / float(2**k)),
                    name=f"decode.alg2m.valid_scale.g{group}",
                    crypto_context=crypto_context,
                )[0]
            )
            rotated = own(_homo_rotate(scaled, -stride * group, crypto_context=crypto_context))
            drop(scaled)
            if merged is None:
                merged = rotated
            else:
                summed = own(_homo_add(merged, rotated, crypto_context=crypto_context))
                drop(merged)
                drop(rotated)
                merged = summed
        shifted = own(_add_scalar_ciphers((merged,), shift, crypto_context=crypto_context)[0])
        drop(merged)
        mapped = own(
            _add_packed_rows(
                (shifted,),
                (1.0 - masks.merged_valid) * invalid_value,
                name="decode.alg2m.invalid",
                crypto_context=crypto_context,
            )[0]
        )
        drop(shifted)
        # 2. exp, prefix mask, post-exp refresh (one bootstrap for all pages)
        refreshed = _refresh_row_ciphers_if_short(
            (mapped,),
            bootstrap_operator=bootstrap_operator,
            need_limbs=_decode_exp_need_limbs(exp_entry),
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag="softmax.pre_exp",
        )
        if refreshed[0] is not mapped:
            live.remove(mapped)  # consumed by the refresh
            mapped = own(refreshed[0])
        exp_out = own(eval_exp_cipher(mapped, exp_entry, crypto_context=crypto_context))
        drop(mapped)
        y = own(
            _mul_packed_rows(
                (exp_out,), masks.merged_valid, name="decode.alg2m.post_exp", crypto_context=crypto_context
            )[0]
        )
        drop(exp_out)
        refreshed = _refresh_row_ciphers(
            (y,),
            bootstrap_operator=bootstrap_operator,
            iterations=bootstrap_iterations,
            precision_bits=bootstrap_precision_bits,
            tag="softmax.post_exp",
        )
        if refreshed[0] is not y:
            live.remove(y)
            y = own(refreshed[0])
        # 3a. k = 2 S-chain tail (one aux bootstrap) when enabled and limb-feasible
        # (``y`` just left the unconditional post-exp refresh, so its limbs
        # are the operator's exit level.)
        if schain_applicable(
            config,
            layer_idx,
            k,
            exit_limbs=(None if bootstrap_operator is None else int(y.state.cur_limbs)),
        ):
            outputs = _schain_merged_tail(
                y,
                layout=layout,
                crypto_context=crypto_context,
                config=config,
                layer_idx=layer_idx,
                masks=masks,
                page_count=len(pages),
                stride=stride,
                groups=groups,
                bootstrap_operator=bootstrap_operator,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_precision_bits=bootstrap_precision_bits,
                sparse_period=sparse_period,
                own=own,
                drop=drop,
            )
            result = tuple(outputs)
            outputs = []
            return result
        # 3b. k >= 3 block plan (LLAMA_FHE_SOFTMAX_BLOCKPLAN) when limb-feasible
        plan = blockplan_applicable(
            config,
            layer_idx,
            k,
            exit_limbs=(None if bootstrap_operator is None else int(y.state.cur_limbs)),
        )
        if plan is not None:
            outputs = _blockplan_merged_tail(
                y,
                plan=plan,
                k=k,
                layout=layout,
                crypto_context=crypto_context,
                config=config,
                layer_idx=layer_idx,
                masks=masks,
                page_count=len(pages),
                stride=stride,
                groups=groups,
                bootstrap_operator=bootstrap_operator,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_precision_bits=bootstrap_precision_bits,
                sparse_period=sparse_period,
                own=own,
                drop=drop,
            )
            result = tuple(outputs)
            outputs = []
            return result
        # 3. Alg2 rounds on the merged ciphertext (round 1 j1_rough, middle
        # rounds j_mid when the table carries one, last round j2_precise)
        outputs: list[tuple[object, ...]] = []
        for round_index in range(1, k + 1):
            poly = _blockplan_round_poly(config, int(layer_idx), round_index, k)
            refreshed = _refresh_row_ciphers_if_short(
                (y,),
                bootstrap_operator=bootstrap_operator,
                need_limbs=ALG2_ROUND_NEED_LIMBS,
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
                tag=f"softmax.r{round_index}.y",
            )
            if refreshed[0] is not y:
                live.remove(y)
                y = own(refreshed[0])
            squared = own(square_rescale(y, crypto_context=crypto_context))
            reduced = own(_token_reduce_cipher(squared, layout=layout, crypto_context=crypto_context))
            drop(squared)
            isolated = own(
                _mul_packed_rows(
                    (reduced,),
                    masks.merged_token0,
                    name=f"decode.alg2m.r{round_index}.sum0",
                    crypto_context=crypto_context,
                )[0]
            )
            drop(reduced)
            total = own(
                _rotate_add_tree(isolated, stride=stride, count=groups, crypto_context=crypto_context)
            )
            drop(isolated)
            filled = own(
                _add_packed_rows(
                    (total,),
                    fill_mask * _safe_invsqrt_fill(poly),
                    name=f"decode.alg2m.r{round_index}.invsqrt_fill",
                    crypto_context=crypto_context,
                )[0]
            )
            drop(total)
            refreshed = _refresh_row_ciphers_if_short(
                (filled,),
                bootstrap_operator=bootstrap_operator,
                need_limbs=_invsqrt_need_limbs(poly),
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
                tag=f"softmax.r{round_index}.fill",
                periodic_slots=sparse_period,
            )
            if refreshed[0] is not filled:
                live.remove(filled)
                filled = own(refreshed[0])
            lam = own(eval_invsqrt_cipher(filled, poly, crypto_context=crypto_context))
            drop(filled)
            refreshed = _refresh_row_ciphers(
                (lam,),
                bootstrap_operator=bootstrap_operator,
                iterations=bootstrap_iterations,
                precision_bits=bootstrap_precision_bits,
                tag=f"softmax.r{round_index}.lam",
                periodic_slots=sparse_period,
            )
            if refreshed[0] is not lam:
                live.remove(lam)
                lam = own(refreshed[0])
            # broadcast λ along the token axis of every merged row (same three
            # steps as decode_broadcast_tokens_fhe: isolate, tree, remask)
            lam0 = own(
                _mul_packed_rows(
                    (lam,), masks.merged_token0, name="decode.alg2m.lambda.token0", crypto_context=crypto_context
                )[0]
            )
            drop(lam)
            lam_b = own(_token_broadcast_cipher(lam0, layout=layout, crypto_context=crypto_context))
            drop(lam0)
            lam_bcast = own(
                _mul_packed_rows(
                    (lam_b,), masks.merged_valid, name="decode.alg2m.lambda.prefix", crypto_context=crypto_context
                )[0]
            )
            drop(lam_b)
            aligned_y, aligned_lam, owned_y, owned_lam = _align_cipher_pair(
                y, lam_bcast, crypto_context=crypto_context
            )
            try:
                product = own(mul_cipher_rescale(aligned_y, aligned_lam, crypto_context=crypto_context))
            finally:
                if owned_y:
                    release_if_supported(aligned_y)
                if owned_lam:
                    release_if_supported(aligned_lam)
            drop(lam_bcast)
            drop(y)
            squared_y = own(square_rescale(product, crypto_context=crypto_context))
            drop(product)
            if round_index < k:
                y = own(
                    _mul_packed_rows(
                        (squared_y,),
                        masks.merged_valid,
                        name=f"decode.alg2m.r{round_index}.prefix",
                        crypto_context=crypto_context,
                    )[0]
                )
                drop(squared_y)
                continue
            # last round: un-merge straight into the per-page prefix mask
            for group in range(len(pages)):
                back = own(_homo_rotate(squared_y, stride * group, crypto_context=crypto_context))
                page_out = _mul_packed_rows(
                    (back,),
                    masks.page_valid[group],
                    name=f"decode.alg2m.r{round_index}.prefix.g{group}",
                    crypto_context=crypto_context,
                )
                drop(back)
                outputs.append(tuple(page_out))
            drop(squared_y)
        if k < 1:
            raise ValueError("Alg2 needs at least one round.")
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for cipher in live:
            release_if_supported(cipher)
        live.clear()


__all__ = [
    "ALG2_ROUND_NEED_LIMBS",
    "SOFTMAX_SCHAIN_ENV",
    "SOFTMAX_BLOCKPLAN_ENV",
    "SOFTMAX_BLOCKPLAN_FORCE_ENV",
    "SCHAIN_PROBABILITY_MIN_LIMBS",
    "SOFTMAX_PV_NEED_LIMBS_ENV",
    "PV_PROBABILITY_NEED_LIMBS",
    "pv_probability_need_limbs",
    "probability_min_limbs",
    "blockplan_applicable",
    "blockplan_limb_schedule",
    "parse_softmax_blockplan",
    "softmax_blockplan_for",
    "schain_applicable",
    "schain_limb_schedule",
    "softmax_schain_enabled",
    "SOFTMAX_MERGE_PAGES_ENV",
    "SOFTMAX_SPARSE_BS_ENV",
    "merged_page_sparse_period",
    "softmax_merge_pages_enabled",
    "softmax_sparse_bootstrap_enabled",
    "softmax_page_merge_rotations",
    "ALG2_ROUND_Y_LIMBS",
    "BOOTSTRAP_MIN_INPUT_LIMBS",
    "DecodeAlg2Poly",
    "apply_decode_alg2_public_map_fhe",
    "apply_decode_alg2_public_map_numpy",
    "softmax_alg2_poly_numpy",
    "softmax_alg2_row_fhe",
    "softmax_alg2_row_pages_fhe",
]
