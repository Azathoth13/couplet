from __future__ import annotations

"""PairFM-factored ``O_C -> feature-major`` conversion.

This module keeps the existing attention and public feature-major contracts
unchanged.  ``PairFM`` exists only as an internal, block-friendly layout::

    O_C[c, d, q, lane]
      -> PairFM[c, i=2*d+lane, q]
      -> blocked[c, token_shard, i, local_token]
      -> FM[token_shard, global_feature=(c, i), local_token]

Adjacent query-pair ciphertexts are carried in the real/imaginary components
through the two common slot permutations.  They are split only before the
final ciphertext-owner transpose.

Two factorizations of the first two permutations are available and the one
with fewer plaintext diagonals is selected per geometry:

``pairfm`` (Llama-3-8B: 16 pairs == 16 token shards, 255 + 511 rows)
    ``[d][q][lane] -> [i=(d,lane)][q]`` then ``[i][s][u] -> [s][i][u]``.
``lane_first`` (Llama-3.2-1B: 16 pairs, 8 token shards, 31 + 127 rows)
    ``[d][s][u][lane] -> [d][s][lane][u]`` then ``[d][s][lane][u] -> [s][d][lane][u]``.

Both end in the same ``blocked[c, s, i, u]`` layout.  When ``head_dim <
seq_len`` the O_C grid repeats each feature row ``seq_len / head_dim`` times;
only the first copy is routed, so ``blocked`` fills ``slots / feature_repeats``
slots and ``pairs / shards`` O_C ciphertexts feed each feature-major shard.
"""

import os
import time
from dataclasses import asdict, dataclass
from math import ceil

import numpy as np

from llama3fhe.backend import release_if_supported
from llama3fhe.backend import profile_sync_device as sync_device  # stage-timer sync, no-op under T2_STAGE_SYNC=0
from llama3fhe.config import Llama3CKKSConfig
from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout

from ...linear import pair_real_ciphers, split_complex_ciphers_twice
from ...primitives import (
    accumulate_owned as _add_owned,
)
from ...primitives import (
    drop_to_limbs,
)
from ...primitives import (
    encode_mask as _encode_mask,
)
from ...transform import apply_transform_by_level, transform_rotation_set
from ..config import AttentionOperatorConfig


@dataclass(frozen=True)
class PairFMOutputComplexity:
    query_pair_count: int
    token_shards: int
    token_lanes: int
    pair_feature_width: int
    complex_source_count: int
    pairfm_useful_rows: int
    pairfm_padded_rows: int
    pairfm_pt_ct: int
    pairfm_rotations: int
    blocked_useful_rows: int
    blocked_padded_rows: int
    blocked_pt_ct: int
    blocked_rotations: int
    owner_encoded_rows: int
    owner_pt_ct: int
    owner_rotations: int
    conversion_encoded_rows: int
    conversion_pt_ct: int
    conversion_rotations: int
    conversion_conjugations: int
    conversion_imults: int
    conversion_depth: int

    def to_dict(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in asdict(self).items()}


@dataclass
class PairFMConversionResult:
    real_ciphers: tuple[object, ...]
    carrier_ciphers: tuple[object, ...]
    crypto_context: object
    feature_layout: FeatureMajorPrefillLayout
    attention_layout: AttentionPairLayout
    complexity: PairFMOutputComplexity
    wall_seconds: float
    stage_seconds: dict[str, float]
    values: np.ndarray | None = None
    max_abs_diff: float | None = None

    def release(self) -> None:
        for cipher in self.real_ciphers + self.carrier_ciphers:
            release_if_supported(cipher)


def _geometry(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> tuple[int, int, int, int]:
    """Validate the selected production geometry.

    PairFM deliberately relies on the equality between query-pair ownership
    and feature-major token-shard ownership.  Keep that requirement next to
    the transform rather than importing it from the unrelated W_O operator.
    """

    pairs = int(attention_layout.query_pair_count)
    shards = int(feature_layout.cipher_count)
    lanes = int(feature_layout.tokens_per_cipher)
    pair_feature = 2 * int(attention_layout.head_dim)
    dimension = int(feature_layout.hidden_dim)
    if int(feature_layout.slots) != int(attention_layout.slots):
        raise ValueError("feature-major and attention slot counts must match.")
    if dimension != int(attention_layout.query_heads) * int(attention_layout.head_dim):
        raise ValueError("feature-major dimension must equal query_heads*head_dim.")
    if int(feature_layout.seq_len) != int(attention_layout.seq_len):
        raise ValueError("feature-major and attention sequence lengths must match.")
    if pairs % shards:
        raise ValueError("PairFM requires the query-pair count to be a multiple of the token shards.")
    if pairs // shards != int(attention_layout.feature_repeats):
        raise ValueError(
            "PairFM requires pairs / token shards == seq_len / head_dim "
            f"(got {pairs}/{shards} vs {attention_layout.feature_repeats})."
        )
    if int(attention_layout.seq_len) != shards * lanes:
        raise ValueError("PairFM requires a full token lane in every shard.")
    if dimension != pairs * pair_feature:
        raise ValueError("query-pair feature blocks must cover the hidden width.")
    if pair_feature % lanes:
        raise ValueError("pair feature width must be divisible by token lanes.")
    return pairs, lanes, pair_feature, pair_feature // lanes


def _shards(feature_layout: FeatureMajorPrefillLayout) -> int:
    """Feature-major token shards (== output ciphertext count)."""

    return int(feature_layout.cipher_count)


def _factorization(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> str:
    """Pick the two-stage permutation with fewer plaintext diagonals."""

    totals = {}
    for name in ("pairfm", "lane_first"):
        first = _stage_geometry(feature_layout, attention_layout, stage=1, mode=name)
        second = _stage_geometry(feature_layout, attention_layout, stage=2, mode=name)
        totals[name] = first[2] + second[2]
    # Deterministic tie-break keeps the historical factorization.
    return "pairfm" if totals["pairfm"] <= totals["lane_first"] else "lane_first"


def _stage_index_map(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    stage: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """``(source_slots, destination_slots)`` of one permutation stage.

    Only the first C copy (``d < head_dim``) of the O_C grid is routed.
    """

    pairs, lanes, pair_feature, _ = _geometry(feature_layout, attention_layout)
    shards = _shards(feature_layout)
    seq_len = int(attention_layout.seq_len)
    head_dim = int(attention_layout.head_dim)
    d = np.arange(head_dim, dtype=np.int64)
    s = np.arange(shards, dtype=np.int64)
    u = np.arange(lanes, dtype=np.int64)
    lane = np.arange(2, dtype=np.int64)
    grid = np.ix_(d, s, u, lane)
    if mode == "pairfm":
        if int(stage) == 1:
            # [d][q][lane] -> [i=2d+lane][q]
            src = (grid[0] * seq_len + grid[1] * lanes + grid[2]) * 2 + grid[3]
            dst = (2 * grid[0] + grid[3]) * seq_len + grid[1] * lanes + grid[2]
        elif int(stage) == 2:
            # [i][s][u] -> [s][i][u]
            i = 2 * grid[0] + grid[3]
            src = i * seq_len + grid[1] * lanes + grid[2]
            dst = (grid[1] * pair_feature + i) * lanes + grid[2]
        else:
            raise ValueError(f"stage must be 1 or 2, got {stage}.")
    elif mode == "lane_first":
        if int(stage) == 1:
            # [d][s][u][lane] -> [d][s][lane][u]
            src = (grid[0] * seq_len + grid[1] * lanes + grid[2]) * 2 + grid[3]
            dst = (grid[0] * seq_len + grid[1] * lanes) * 2 + grid[3] * lanes + grid[2]
        elif int(stage) == 2:
            # [d][s][lane][u] -> [s][d][lane][u] == [s][i][u]
            src = ((grid[0] * shards + grid[1]) * 2 + grid[3]) * lanes + grid[2]
            dst = (grid[1] * pair_feature + 2 * grid[0] + grid[3]) * lanes + grid[2]
        else:
            raise ValueError(f"stage must be 1 or 2, got {stage}.")
    else:
        raise ValueError(f"unknown PairFM factorization {mode!r}.")
    src = np.broadcast_to(src, (head_dim, shards, lanes, 2)).reshape(-1)
    dst = np.broadcast_to(dst, (head_dim, shards, lanes, 2)).reshape(-1)
    del pairs
    return src, dst


def _stage_geometry(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    stage: int,
    mode: str,
) -> tuple[int, int, int]:
    """Return ``(half_rows, stride, rows)`` of one centered diagonal stage."""

    src, dst = _stage_index_map(feature_layout, attention_layout, stage=stage, mode=mode)
    offsets = src - dst
    nonzero = np.abs(offsets[offsets != 0])
    stride = int(np.gcd.reduce(nonzero)) if nonzero.size else 1
    normalized = offsets // stride
    half_rows = int(max(abs(int(normalized.min())), abs(int(normalized.max()))))
    return half_rows, stride, 2 * half_rows + 1


def _build_stage_masks(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    stage: int,
    mode: str,
    dtype: np.dtype,
) -> tuple[np.ndarray, int]:
    src, dst = _stage_index_map(feature_layout, attention_layout, stage=stage, mode=mode)
    half_rows, stride, rows = _stage_geometry(
        feature_layout, attention_layout, stage=stage, mode=mode
    )
    masks = np.zeros((rows, int(feature_layout.slots)), dtype=dtype)
    masks[(src - dst) // stride + half_rows, dst] = 1.0
    return masks, stride


def _round_rows(rows: int, modulus: int) -> int:
    rows = int(rows)
    modulus = int(modulus)
    if rows <= 0 or modulus <= 0:
        raise ValueError("row counts and baby steps must be positive.")
    return int(ceil(rows / modulus)) * modulus


def _rotation_count(*, rows: int, baby_steps: int, ciphers: int) -> int:
    giant_count = int(ceil(int(rows) / int(baby_steps)))
    # One centered base rotation, B-1 baby rotations, and G-1 giant
    # rotations per ciphertext.  Both selected permutations have a nonzero
    # base offset at every supported nontrivial geometry.
    per_cipher = 1 + max(0, int(baby_steps) - 1) + max(0, giant_count - 1)
    return int(ciphers) * per_cipher


def _first_diagonal_geometry(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> tuple[int, int, int]:
    """``(half_rows, stride, rows)`` of the first permutation stage."""

    mode = _factorization(feature_layout, attention_layout)
    return _stage_geometry(feature_layout, attention_layout, stage=1, mode=mode)


def _blocked_diagonal_geometry(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> tuple[int, int, int]:
    """Return ``(half_rows, stride, rows)`` for the second stage (-> blocked).

    For the ``pairfm`` factorization (``i`` inside one query-pair feature
    block, token shard ``s``) this is ``source-destination = U*((C-1)*i -
    (F-1)*s)`` divided by its gcd: a centered, possibly sparse integer row
    range.  Empty rows are deliberately retained: this makes the existing
    BSGS helper applicable without changing its scale or ownership semantics.
    """

    mode = _factorization(feature_layout, attention_layout)
    return _stage_geometry(feature_layout, attention_layout, stage=2, mode=mode)


#: Process-level memo of the two public PairFM slot-permutation masks.  They
#: are pure functions of the (frozen, hashable) layouts and dtype, cost
#: ~0.1 s of NumPy per call (a 255 x 32768 and a 512 x 32768 dense mask
#: built row by row) and were rebuilt on every layer / every pairfm call
#: (plan §12.21: ~0.16 s untimed CPU per layer on the 4090).  The arrays are
#: returned read-only; callers only pass them on to the BSGS row packer.
_MASK_CACHE: dict[tuple, tuple[np.ndarray, int]] = {}


def _cached_masks(key: tuple, build):
    entry = _MASK_CACHE.get(key)
    if entry is None:
        masks, stride = build()
        masks.setflags(write=False)
        entry = (masks, int(stride))
        _MASK_CACHE[key] = entry
    return entry


def _pairfm_masks(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, int]:
    """Masks for ``[d,q,lane] -> [d,lane,q]`` (memoised per process)."""

    return _cached_masks(
        ("pairfm", feature_layout, attention_layout, str(np.dtype(dtype))),
        lambda: _build_pairfm_masks(feature_layout, attention_layout, dtype=dtype),
    )


def _build_pairfm_masks(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, int]:
    mode = _factorization(feature_layout, attention_layout)
    return _build_stage_masks(
        feature_layout, attention_layout, stage=1, mode=mode, dtype=dtype
    )


def _blocked_masks(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, int]:
    """Masks for ``[i, shard, u] -> [shard, i, u]`` (memoised per process)."""

    return _cached_masks(
        ("blocked", feature_layout, attention_layout, str(np.dtype(dtype))),
        lambda: _build_blocked_masks(feature_layout, attention_layout, dtype=dtype),
    )


def _build_blocked_masks(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, int]:
    mode = _factorization(feature_layout, attention_layout)
    return _build_stage_masks(
        feature_layout, attention_layout, stage=2, mode=mode, dtype=dtype
    )


def pairfm_output_complexity(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    first_baby_steps: int = 16,
    block_baby_steps: int = 32,
) -> PairFMOutputComplexity:
    pairs, lanes, pair_feature, _ = _geometry(feature_layout, attention_layout)
    shards = _shards(feature_layout)
    if pairs % 2:
        raise ValueError("PairFM complex source pairing requires an even pair count.")
    complex_sources = pairs // 2
    _, _, first_useful = _first_diagonal_geometry(feature_layout, attention_layout)
    first_padded = _round_rows(first_useful, int(first_baby_steps))
    first_rotations = _rotation_count(
        rows=first_useful,
        baby_steps=int(first_baby_steps),
        ciphers=complex_sources,
    )
    _, _, blocked_useful = _blocked_diagonal_geometry(
        feature_layout, attention_layout
    )
    blocked_padded = _round_rows(blocked_useful, int(block_baby_steps))
    owner_rows = pairs
    owner_pt_ct = pairs * shards
    # Source pair ``c`` reaches target shard ``s`` with rotation ``(s-c)*block``;
    # the identity rotation is free.
    owner_rotations = sum(
        1 for source in range(pairs) for target in range(shards) if target != source
    )
    return PairFMOutputComplexity(
        query_pair_count=pairs,
        token_shards=shards,
        token_lanes=lanes,
        pair_feature_width=pair_feature,
        complex_source_count=complex_sources,
        pairfm_useful_rows=first_useful,
        pairfm_padded_rows=first_padded,
        pairfm_pt_ct=first_padded * complex_sources,
        pairfm_rotations=first_rotations,
        blocked_useful_rows=blocked_useful,
        blocked_padded_rows=blocked_padded,
        blocked_pt_ct=blocked_padded * complex_sources,
        blocked_rotations=_rotation_count(
            rows=blocked_useful,
            baby_steps=int(block_baby_steps),
            ciphers=complex_sources,
        ),
        owner_encoded_rows=owner_rows,
        owner_pt_ct=owner_pt_ct,
        owner_rotations=owner_rotations,
        conversion_encoded_rows=first_padded + blocked_padded + owner_rows,
        conversion_pt_ct=(first_padded + blocked_padded) * complex_sources
        + owner_pt_ct,
        conversion_rotations=first_rotations
        + _rotation_count(
            rows=blocked_useful,
            baby_steps=int(block_baby_steps),
            ciphers=complex_sources,
        )
        + owner_rotations,
        conversion_conjugations=complex_sources,
        conversion_imults=3 * complex_sources,
        conversion_depth=3,
    )


def pairfm_output_rotations(
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    *,
    first_baby_steps: int = 16,
    block_baby_steps: int = 32,
    log_n: int = 16,
) -> tuple[int, ...]:
    pairs, _, _, _ = _geometry(feature_layout, attention_layout)
    shards = _shards(feature_layout)
    first_half_rows, first_stride, _ = _first_diagonal_geometry(
        feature_layout, attention_layout
    )
    half_rows, blocked_stride, _ = _blocked_diagonal_geometry(
        feature_layout, attention_layout
    )
    rotations: set[int] = set()
    rotations.update(
        transform_rotation_set(
            seq_len=first_half_rows + 1,
            stride=first_stride,
            baby_steps=int(first_baby_steps),
        )
    )
    rotations.update(
        transform_rotation_set(
            seq_len=half_rows + 1,
            stride=blocked_stride,
            baby_steps=int(block_baby_steps),
        )
    )
    block = int(feature_layout.slots) // pairs
    rotations.update(
        (target - source) * block
        for source in range(pairs)
        for target in range(shards)
        if target != source
    )
    rotations.add((1 << (int(log_n) + 1)) - 1)
    return tuple(sorted(int(value) for value in rotations if int(value)))


def output_c_to_pairfm_numpy(
    output_c: np.ndarray,
    *,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> np.ndarray:
    """Return real PairFM ciphertext payloads, ``[pair, i, q]``."""

    pairs, _, pair_feature, _ = _geometry(feature_layout, attention_layout)
    output_c = np.asarray(output_c)
    expected = (pairs, int(attention_layout.slots))
    if output_c.shape != expected:
        raise ValueError(f"O_C must have shape {expected}, got {output_c.shape}.")
    seq_len = int(attention_layout.seq_len)
    head_dim = int(attention_layout.head_dim)
    # Only the first C copy of the (possibly repeated) O_C grid is routed.
    first_copy = output_c.reshape(pairs, int(attention_layout.grid_rows), seq_len, 2)[
        :, :head_dim
    ]
    pairfm = first_copy.transpose(0, 1, 3, 2).reshape(pairs, pair_feature * seq_len)
    padded = np.zeros((pairs, int(attention_layout.slots)), dtype=output_c.dtype)
    padded[:, : pairfm.shape[1]] = pairfm
    return padded


def output_c_to_feature_major_pairfm_numpy(
    output_c: np.ndarray,
    *,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
) -> np.ndarray:
    """Reference PairFM factorization ending in the public real FM layout."""

    pairs, lanes, pair_feature, _ = _geometry(feature_layout, attention_layout)
    shards = _shards(feature_layout)
    seq_len = int(attention_layout.seq_len)
    pairfm = output_c_to_pairfm_numpy(
        output_c,
        feature_layout=feature_layout,
        attention_layout=attention_layout,
    )[:, : pair_feature * seq_len]
    pairfm = pairfm.reshape(pairs, pair_feature, shards, lanes)
    blocked = pairfm.transpose(0, 2, 1, 3)
    return blocked.transpose(1, 0, 2, 3).reshape(
        shards, int(feature_layout.slots)
    )


def _owner_transpose_fhe(
    logical_sources: tuple[object, ...],
    *,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    dtype: np.dtype,
    name: str,
    profile: dict[str, float],
) -> tuple[object, ...]:
    """Transpose external ``query_pair <-> token_shard`` ownership."""

    pairs, _, _, _ = _geometry(feature_layout, attention_layout)
    shards = _shards(feature_layout)
    if len(logical_sources) != pairs:
        raise ValueError(f"owner transpose expects {pairs} real sources.")
    levels = {
        int(crypto_context.level_for_cipher(cipher)) for cipher in logical_sources
    }
    if len(levels) != 1:
        raise ValueError("owner transpose inputs must enter at one common level.")
    level = levels.pop()
    # ``blocked[c, s, i, u]`` block ``s`` of source pair ``c`` lands at block
    # ``c`` of target shard ``s``; both blocks are ``pair_feature * lanes`` wide.
    block = int(feature_layout.slots) // pairs
    accumulators: list[object | None] = [None] * shards
    outputs: list[object] = []
    try:
        for source, cipher in enumerate(logical_sources):
            start = time.perf_counter()
            mask = np.zeros((int(feature_layout.slots),), dtype=dtype)
            # The complex split produced twice the wanted real value.
            mask[source * block : (source + 1) * block] = 0.5
            plain = None
            try:
                plain = _encode_mask(
                    mask,
                    name=f"{name}.source{source}",
                    level=level,
                    crypto_context=crypto_context,
                )
                sync_device(crypto_context.device)
                profile["pack_encode"] += time.perf_counter() - start
                start = time.perf_counter()
                for target in range(shards):
                    offset = (target - source) * block
                    rotated = (
                        cipher
                        if offset == 0
                        else crypto_context.fhe.homo_rotate(
                            cipher, offset, crypto_context.context
                        )
                    )
                    try:
                        product = crypto_context.fhe.homo_mul_pt(
                            rotated, plain, crypto_context.context
                        )
                        accumulators[target] = _add_owned(
                            accumulators[target],
                            product,
                            crypto_context=crypto_context,
                        )
                    finally:
                        if rotated is not cipher:
                            release_if_supported(rotated)
                sync_device(crypto_context.device)
                profile["online"] += time.perf_counter() - start
            finally:
                release_if_supported(plain)

        start = time.perf_counter()
        for index, accumulator in enumerate(accumulators):
            if accumulator is None:
                raise RuntimeError(f"empty owner-transpose accumulator {index}.")
            output = crypto_context.fhe.rescale(
                accumulator, crypto_context.context
            )
            release_if_supported(accumulator)
            accumulators[index] = None
            outputs.append(output)
        sync_device(crypto_context.device)
        profile["rescale"] += time.perf_counter() - start
        result = tuple(outputs)
        outputs = []
        return result
    finally:
        for accumulator in accumulators:
            release_if_supported(accumulator)
        for output in outputs:
            release_if_supported(output)


def prepare_w_o_inputs_fhe(
    output_ciphers: tuple[object, ...] | list[object],
    *,
    feature_layout: FeatureMajorPrefillLayout,
    attention_layout: AttentionPairLayout,
    crypto_context,
    model_config: Llama3CKKSConfig | None = None,
    operator_config: AttentionOperatorConfig | None = None,
    bootstrap_operator=None,
    verify: bool = False,
    expected_output_c: np.ndarray | None = None,
    function_prefix: str = "model.layers.0.self_attn.output_c_to_feature_major_pairfm",
) -> PairFMConversionResult:
    """Convert O_C inputs to feature-major W_O carriers (depth three)."""

    operator_config = (
        AttentionOperatorConfig() if operator_config is None else operator_config
    )
    first_baby_steps = int(operator_config.pairfm_first_baby_steps)
    block_baby_steps = int(operator_config.pairfm_block_baby_steps)
    refresh_paired_inputs = bool(operator_config.refresh_output_before_pairfm)
    refresh_paired_inputs_if_below = operator_config.pairfm_refresh_min_input_limbs
    post_refresh_input_limbs = operator_config.pairfm_post_refresh_input_limbs
    checkpoint_precision_bits = int(
        operator_config.checkpoint_bootstrap_precision_bits
    )

    output_ciphers = tuple(output_ciphers)
    pairs, _, _, _ = _geometry(feature_layout, attention_layout)
    if len(output_ciphers) != pairs:
        raise ValueError(f"PairFM conversion expects {pairs} O_C ciphertexts.")
    input_levels = {
        int(crypto_context.level_for_cipher(cipher)) for cipher in output_ciphers
    }
    if len(input_levels) != 1:
        raise ValueError("all O_C ciphertexts must enter at one common level.")
    if int(crypto_context.max_slots) != int(feature_layout.slots):
        raise ValueError("crypto context and feature layout slot counts must match.")
    if post_refresh_input_limbs is not None and int(
        post_refresh_input_limbs
    ) <= 1:
        raise ValueError("post_refresh_input_limbs must exceed one.")
    model_config = Llama3CKKSConfig() if model_config is None else model_config
    config_slots = 1 << (int(model_config.simulator.logN) - 1)
    if config_slots != int(feature_layout.slots):
        raise ValueError("config logN and feature layout slot counts must match.")
    dtype = np.dtype(model_config.layout_transform.dtype)
    complexity = pairfm_output_complexity(
        feature_layout,
        attention_layout,
        first_baby_steps=int(first_baby_steps),
        block_baby_steps=int(block_baby_steps),
    )
    pairfm_masks, pairfm_stride = _pairfm_masks(
        feature_layout, attention_layout, dtype=dtype
    )
    blocked_masks, blocked_stride = _blocked_masks(
        feature_layout, attention_layout, dtype=dtype
    )
    wall_start = time.perf_counter()
    paired: tuple[object, ...] = ()
    pairfm: tuple[object, ...] = ()
    blocked: tuple[object, ...] = ()
    logical_sources: tuple[object, ...] = ()
    real_outputs: tuple[object, ...] = ()
    carrier_outputs: tuple[object, ...] = ()
    success = False
    stages = {
        "input_pairing": 0.0,
        "input_pair_bootstrap": 0.0,
        "input_pair_post_refresh_modulus_drop": 0.0,
        "pairfm_pack_encode": 0.0,
        "pairfm_online_rescale": 0.0,
        "block_pack_encode": 0.0,
        "block_online_rescale": 0.0,
        "source_split": 0.0,
        "scatter_pack_encode": 0.0,
        "scatter_online": 0.0,
        "scatter_rescale": 0.0,
        "output_pairing": 0.0,
    }
    try:
        start = time.perf_counter()
        paired = pair_real_ciphers(output_ciphers, crypto_context=crypto_context)
        sync_device(crypto_context.device)
        stages["input_pairing"] = time.perf_counter() - start

        should_refresh = bool(refresh_paired_inputs)
        if refresh_paired_inputs_if_below is not None:
            minimum = int(refresh_paired_inputs_if_below)
            if minimum <= 1:
                raise ValueError(
                    "refresh_paired_inputs_if_below must exceed one."
                )
            should_refresh = should_refresh or min(
                int(cipher.state.cur_limbs) for cipher in paired
            ) < minimum
        if os.environ.get("LLAMA_FHE_PAIRFM_LIMB_TRACE", "") == "1":
            print(
                f"[pairfm-limb-trace] {function_prefix}: paired limbs="
                f"{[int(c.state.cur_limbs) for c in paired]} "
                f"threshold={refresh_paired_inputs_if_below} "
                f"refresh={should_refresh}",
                flush=True,
            )
        if should_refresh:
            if bootstrap_operator is None:
                raise ValueError(
                    "refresh_paired_inputs requires a prepared bootstrap operator."
                )
            start = time.perf_counter()
            refreshed: list[object] = []
            try:
                for index, cipher in enumerate(paired):
                    refreshed.append(
                        bootstrap_operator.refresh(
                            cipher,
                            iterations=int(
                                operator_config
                                .pairfm_refresh_bootstrap_iterations
                            ),
                            precision_bits=int(checkpoint_precision_bits),
                        )
                    )
            except Exception:
                for cipher in refreshed:
                    release_if_supported(cipher)
                raise
            for cipher in paired:
                release_if_supported(cipher)
            paired = tuple(refreshed)
            sync_device(crypto_context.device)
            stages["input_pair_bootstrap"] = time.perf_counter() - start

            if post_refresh_input_limbs is not None:
                current_limbs = int(paired[0].state.cur_limbs)
                if current_limbs != int(post_refresh_input_limbs):
                    start = time.perf_counter()
                    dropped = drop_to_limbs(
                        paired,
                        target_limbs=int(post_refresh_input_limbs),
                        crypto_context=crypto_context,
                    )
                    sync_device(crypto_context.device)
                    for cipher in paired:
                        release_if_supported(cipher)
                    paired = dropped
                    stages["input_pair_post_refresh_modulus_drop"] = (
                        time.perf_counter() - start
                    )

        profile: dict[str, float] = {}
        pairfm = apply_transform_by_level(
            paired,
            masks=pairfm_masks,
            stride=pairfm_stride,
            baby_steps=int(first_baby_steps),
            crypto_context=crypto_context,
            name=f"{function_prefix}.pairfm",
            profile=profile,
        )
        for cipher in paired:
            release_if_supported(cipher)
        paired = ()
        stages["pairfm_pack_encode"] = float(
            profile.get("pack_encode", 0.0)
        )
        stages["pairfm_online_rescale"] = float(
            profile.get("online_rescale", 0.0)
        )

        profile = {}
        blocked = apply_transform_by_level(
            pairfm,
            masks=blocked_masks,
            stride=blocked_stride,
            baby_steps=int(block_baby_steps),
            crypto_context=crypto_context,
            name=f"{function_prefix}.blocked",
            profile=profile,
        )
        for cipher in pairfm:
            release_if_supported(cipher)
        pairfm = ()
        stages["block_pack_encode"] = float(profile.get("pack_encode", 0.0))
        stages["block_online_rescale"] = float(
            profile.get("online_rescale", 0.0)
        )

        start = time.perf_counter()
        logical_sources = split_complex_ciphers_twice(
            blocked, crypto_context=crypto_context
        )
        sync_device(crypto_context.device)
        stages["source_split"] = time.perf_counter() - start
        for cipher in blocked:
            release_if_supported(cipher)
        blocked = ()

        profile = {"pack_encode": 0.0, "online": 0.0, "rescale": 0.0}
        real_outputs = _owner_transpose_fhe(
            logical_sources,
            feature_layout=feature_layout,
            attention_layout=attention_layout,
            crypto_context=crypto_context,
            dtype=dtype,
            name=f"{function_prefix}.owner",
            profile=profile,
        )
        for cipher in logical_sources:
            release_if_supported(cipher)
        logical_sources = ()
        stages["scatter_pack_encode"] = profile["pack_encode"]
        stages["scatter_online"] = profile["online"]
        stages["scatter_rescale"] = profile["rescale"]

        start = time.perf_counter()
        carrier_outputs = pair_real_ciphers(
            real_outputs, crypto_context=crypto_context
        )
        sync_device(crypto_context.device)
        stages["output_pairing"] = time.perf_counter() - start

        values = None
        max_abs_diff = None
        if verify:
            start = time.perf_counter()
            packed = np.stack(
                [np.asarray(crypto_context.decrypt(cipher)) for cipher in real_outputs]
            )
            values = feature_layout.unpack(packed)
            if expected_output_c is not None:
                wanted_packed = output_c_to_feature_major_pairfm_numpy(
                    expected_output_c,
                    feature_layout=feature_layout,
                    attention_layout=attention_layout,
                )
                wanted = feature_layout.unpack(wanted_packed)
                max_abs_diff = float(np.max(np.abs(values - wanted)))
            stages["verification"] = time.perf_counter() - start

        success = True
        return PairFMConversionResult(
            real_ciphers=real_outputs,
            carrier_ciphers=carrier_outputs,
            crypto_context=crypto_context,
            feature_layout=feature_layout,
            attention_layout=attention_layout,
            complexity=complexity,
            wall_seconds=float(time.perf_counter() - wall_start),
            stage_seconds={str(key): float(value) for key, value in stages.items()},
            values=values,
            max_abs_diff=max_abs_diff,
        )
    finally:
        for cipher in paired + pairfm + blocked + logical_sources:
            release_if_supported(cipher)
        if not success:
            for cipher in real_outputs + carrier_outputs:
                release_if_supported(cipher)
