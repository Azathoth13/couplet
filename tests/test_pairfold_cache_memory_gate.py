from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np

from llama3fhe.layouts.pairfold128 import PairFold128Layout
from llama3fhe.operators.pairfold_kv_cache import (
    PAIRFOLD_PAGE_SIZE,
    PairFoldLayerCache,
    PairFoldTokenWrite,
)

GATE_TOKENS = 257
BOUNDARY_TOKENS = (128, 129, 256, 257)
_MIB = 1024 ** 2
_RETURN_METRICS = ("device_used_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes")


def expected_cache_inventory(tokens: int) -> dict[str, object]:
    """Return the no-eviction PairFold page shape for ``tokens``."""

    tokens = int(tokens)
    if tokens < 0:
        raise ValueError("tokens must be non-negative.")
    if tokens == 0:
        occupancies: list[int] = []
    else:
        full_pages, tail = divmod(tokens, PAIRFOLD_PAGE_SIZE)
        occupancies = [PAIRFOLD_PAGE_SIZE] * full_pages
        if tail:
            occupancies.append(tail)
    pages = len(occupancies)
    return {
        "tokens": tokens,
        "pages": pages,
        "ciphertexts": 4 * pages,
        "page_occupancies": occupancies,
    }


def cache_inventory(cache: PairFoldLayerCache) -> dict[str, object]:
    """Return a JSON-safe inventory while also exercising ownership checks."""

    handles = cache.handles
    inventory = {
        "tokens": int(cache.content_tokens),
        "pages": len(cache.pages),
        "ciphertexts": len(handles),
        "page_occupancies": [int(page.occupied) for page in cache.pages],
    }
    expected = expected_cache_inventory(cache.content_tokens)
    if inventory != expected:
        raise AssertionError(
            f"PairFold cache inventory mismatch: {inventory} != {expected}."
        )
    return inventory


def append_token_sequence(
    cache: PairFoldLayerCache,
    *,
    crypto_context,
    token_factory: Callable[[int, int], PairFoldTokenWrite],
    total_tokens: int = GATE_TOKENS,
    checkpoints: Iterable[int] = BOUNDARY_TOKENS,
) -> list[dict[str, object]]:
    """Append token owners and capture selected page-boundary inventories.

    ``token_factory`` receives the zero-based token ordinal and required page
    column.  If append fails before consuming a token, this helper releases
    that still-live owner; the caller remains responsible for ``cache``.
    """

    total_tokens = int(total_tokens)
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive.")
    wanted = {int(value) for value in checkpoints}
    if any(value <= 0 or value > total_tokens for value in wanted):
        raise ValueError("checkpoints must lie inside the append sequence.")

    observations: list[dict[str, object]] = []
    for ordinal in range(total_tokens):
        column = ordinal % PAIRFOLD_PAGE_SIZE
        token = token_factory(ordinal, column)
        try:
            page_index, committed_column = cache.append(
                token,
                crypto_context=crypto_context,
            )
        except BaseException:
            if not token.is_consumed:
                token.release()
            raise
        token_count = ordinal + 1
        if token_count in wanted:
            inventory = cache_inventory(cache)
            expected_page, expected_column = divmod(
                token_count - 1, PAIRFOLD_PAGE_SIZE
            )
            if (page_index, committed_column) != (
                expected_page,
                expected_column,
            ):
                raise AssertionError(
                    "append location mismatch at token "
                    f"{token_count}: {(page_index, committed_column)} != "
                    f"{(expected_page, expected_column)}."
                )
            observations.append(
                {
                    **inventory,
                    "append_page_index": int(page_index),
                    "append_column": int(committed_column),
                }
            )
    return observations


def evaluate_return_gate(
    start: Mapping[str, object],
    after_trim: Mapping[str, object],
    *,
    tolerance_bytes: int,
) -> dict[str, object]:
    """Check that live and reserved CUDA bytes return near the warm baseline."""

    tolerance_bytes = int(tolerance_bytes)
    if tolerance_bytes < 0:
        raise ValueError("tolerance_bytes must be non-negative.")
    checks: dict[str, object] = {}
    for metric in _RETURN_METRICS:
        if metric not in start or metric not in after_trim:
            raise ValueError(f"memory snapshots are missing {metric!r}.")
        baseline = int(start[metric])
        observed = int(after_trim[metric])
        limit = baseline + tolerance_bytes
        checks[metric] = {
            "start_bytes": baseline,
            "after_trim_bytes": observed,
            "delta_bytes": observed - baseline,
            "limit_bytes": limit,
            "passed": observed <= limit,
        }
    return {
        "tolerance_bytes": tolerance_bytes,
        "tolerance_mib": tolerance_bytes / _MIB,
        "checks": checks,
        "passed": all(bool(row["passed"]) for row in checks.values()),
    }


def _token_slots(layout: PairFold128Layout, column: int) -> np.ndarray:
    """Vector of all valid folded slots for one page column and one KV group."""

    column = int(column)
    if not 0 <= column < PAIRFOLD_PAGE_SIZE:
        raise ValueError(f"column must lie in [0, 127], got {column}.")
    return np.fromiter(
        (
            layout.page_slot(pair, feature, column, lane)
            for pair in range(layout.pair_blocks)
            for feature in range(layout.fold_dim)
            for lane in range(layout.head_lanes)
        ),
        dtype=np.int64,
        count=layout.pair_blocks * layout.fold_dim * layout.head_lanes,
    )


def _folded_token_payloads(
    layout: PairFold128Layout,
    *,
    ordinal: int,
    column: int,
) -> np.ndarray:
    """Create four small nonzero K/K/V/V payloads at one page column."""

    slots = _token_slots(layout, column)
    payloads = np.zeros((4, int(layout.slots)), dtype=np.complex128)
    token_scale = 1.0 + (int(ordinal) % PAIRFOLD_PAGE_SIZE) / PAIRFOLD_PAGE_SIZE
    for cipher_index in range(4):
        real = 1e-5 * token_scale * (cipher_index + 1)
        imaginary = 0.5e-5 * token_scale * (cipher_index + 1)
        if cipher_index < int(layout.key_value_groups):
            imaginary = -imaginary
        payloads[cipher_index, slots] = complex(real, imaginary)
    return payloads



class _FakeCipher:
    def __init__(self, registry: list[_FakeCipher]) -> None:
        self.release_count = 0
        registry.append(self)

    def release(self) -> None:
        if self.release_count:
            raise AssertionError("cipher released more than once")
        self.release_count += 1


class _FakeFHE:
    def __init__(self, registry: list[_FakeCipher]) -> None:
        self.registry = registry
        self.add_count = 0

    def homo_add(self, left, right, context):  # noqa: ANN001
        assert left.release_count == 0
        assert right.release_count == 0
        assert context is not None
        self.add_count += 1
        return _FakeCipher(self.registry)


class _FakeContext:
    def __init__(self, registry: list[_FakeCipher]) -> None:
        self.context = object()
        self.fhe = _FakeFHE(registry)


def test_expected_inventory_pins_both_page_boundaries() -> None:
    assert expected_cache_inventory(0) == {
        "tokens": 0,
        "pages": 0,
        "ciphertexts": 0,
        "page_occupancies": [],
    }
    assert expected_cache_inventory(128)["page_occupancies"] == [128]
    assert expected_cache_inventory(129)["page_occupancies"] == [128, 1]
    assert expected_cache_inventory(256)["page_occupancies"] == [128, 128]
    assert expected_cache_inventory(257) == {
        "tokens": 257,
        "pages": 3,
        "ciphertexts": 12,
        "page_occupancies": [128, 128, 1],
    }


def test_append_sequence_exercises_transactional_tail_and_releases_all() -> None:
    registry: list[_FakeCipher] = []
    context = _FakeContext(registry)
    cache = PairFoldLayerCache()

    def token_factory(ordinal: int, column: int) -> PairFoldTokenWrite:
        assert column == ordinal % 128
        handles = [_FakeCipher(registry) for _ in range(4)]
        return PairFoldTokenWrite(
            key_groups=tuple(handles[:2]),
            value_groups=tuple(handles[2:]),
            column=column,
        )

    observations = append_token_sequence(
        cache,
        crypto_context=context,
        token_factory=token_factory,
    )

    assert observations == [
        {
            **expected_cache_inventory(128),
            "append_page_index": 0,
            "append_column": 127,
        },
        {
            **expected_cache_inventory(129),
            "append_page_index": 1,
            "append_column": 0,
        },
        {
            **expected_cache_inventory(256),
            "append_page_index": 1,
            "append_column": 127,
        },
        {
            **expected_cache_inventory(257),
            "append_page_index": 2,
            "append_column": 0,
        },
    ]
    # 254 partial-page appends, four independent K/K/V/V replacements each.
    assert context.fhe.add_count == 254 * 4
    assert sum(cipher.release_count == 0 for cipher in registry) == 12

    cache.release()
    assert all(cipher.release_count == 1 for cipher in registry)


def test_return_gate_checks_device_live_and_reserved_bytes() -> None:
    start = {
        "device_used_bytes": 1_000,
        "allocator_allocated_bytes": 200,
        "allocator_reserved_bytes": 400,
    }
    passing = {
        "device_used_bytes": 1_064,
        "allocator_allocated_bytes": 264,
        "allocator_reserved_bytes": 464,
    }
    failing = {**passing, "allocator_reserved_bytes": 465}

    passed = evaluate_return_gate(start, passing, tolerance_bytes=64)
    failed = evaluate_return_gate(start, failing, tolerance_bytes=64)

    assert passed["passed"] is True
    assert all(row["passed"] for row in passed["checks"].values())
    assert failed["passed"] is False
    assert failed["checks"]["allocator_reserved_bytes"]["delta_bytes"] == 65


def test_real_payload_helper_uses_only_requested_pairfold_column() -> None:
    layout = PairFold128Layout()
    payloads = _folded_token_payloads(layout, ordinal=129, column=1)
    expected_slots = _token_slots(layout, 1)

    assert payloads.shape == (4, layout.slots)
    assert np.all(np.count_nonzero(payloads, axis=1) == expected_slots.size)
    assert np.all(payloads[:2, expected_slots].imag < 0.0)
    assert np.all(payloads[2:, expected_slots].imag > 0.0)
    inactive = np.ones((layout.slots,), dtype=bool)
    inactive[expected_slots] = False
    assert not np.any(payloads[:, inactive])
