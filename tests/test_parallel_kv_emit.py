"""CPU checks for the multi-card Prefill -> Decode KV-cache emission (``parallel.prefill.kv_emit``).

Contribution / refresh ownership rules, the key-plan additions of
``emit_kv_cache`` and a faked run of :func:`prefill_kv_cache_station` over a
thread world (no GPU: the converter, bootstrap and modulus drop are stubbed).
"""

from __future__ import annotations

import threading
import sys
from types import SimpleNamespace

import pytest

from llama3fhe.parallel.prefill.kv_emit import (
    kv_cache_contributions,
    kv_cache_raw_output_index,
    kv_cache_raw_owner,
    kv_cache_refresh_indices,
)
from llama3fhe.parallel.prefill.ownership import build_world_ownership
from test_parallel_split_phase import _Cipher, _Packet, _ThreadWorld


def _world(world_size: int):
    """The production assignment at ``world_size`` (16 FM ciphers, 16 query pairs)."""

    pytest.importorskip("numpy")
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
    from llama3fhe.parallel.prefill.ownership import ownership_from_layouts

    return ownership_from_layouts(FeatureMajorPrefillLayout(), AttentionPairLayout(), world_size=world_size)


@pytest.mark.parametrize("world_size", [8, 4, 2, 1])
def test_every_raw_cipher_has_exactly_one_contributor(world_size: int) -> None:
    world = _world(world_size)
    key_owner: dict[int, int] = {}
    value_owner: dict[int, int] = {}
    for rank in world:
        key_pairs, value_pairs = kv_cache_contributions(world, rank)
        for pair in key_pairs:
            assert pair not in key_owner
            key_owner[pair] = rank.rank
        for pair in value_pairs:
            assert pair not in value_owner
            value_owner[pair] = rank.rank
    assert sorted(key_owner) == [0, 1, 2, 3]
    assert sorted(value_owner) == [0, 1, 2, 3]
    if world_size == 8:
        assert key_owner == {0: 0, 1: 2, 2: 4, 3: 6}
        assert value_owner == {0: 0, 1: 0, 2: 4, 3: 4}
    if world_size == 4:
        assert key_owner == {0: 0, 1: 1, 2: 2, 3: 3}
        assert value_owner == {0: 0, 1: 0, 2: 2, 3: 2}


@pytest.mark.parametrize("world_size", [8, 4, 2, 1])
def test_refresh_indices_partition_the_four_outputs(world_size: int) -> None:
    seen: list[int] = []
    for rank in range(world_size):
        seen.extend(kv_cache_refresh_indices(rank, world_size))
    assert sorted(seen) == [0, 1, 2, 3]
    if world_size == 8:
        assert kv_cache_refresh_indices(5, 8) == ()
        assert kv_cache_refresh_indices(2, 8) == (2,)


def test_owner_routing_maps_raw_pairs_to_one_output_owner() -> None:
    # Raw order is K0..K3,V0..V3; each PairFold group consumes two adjacent
    # rows and the output order is K0,K1,V0,V1.  The owner is the same rank
    # that refreshes that output, so every raw row has exactly one destination.
    assert [
        kv_cache_raw_output_index(index)
        for index in range(8)
    ] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [
        kv_cache_raw_owner(index, world_size=8)
        for index in range(8)
    ] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [
        kv_cache_raw_owner(index, world_size=2)
        for index in range(8)
    ] == [0, 0, 1, 1, 0, 0, 1, 1]
    with pytest.raises(IndexError):
        kv_cache_raw_output_index(8)


def test_owner_directed_raw_route_only_delivers_local_output_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in raw exchange sends each pair to one conversion owner."""

    from llama3fhe.parallel import wire
    from llama3fhe.parallel.decode import decode_collectives
    from llama3fhe.parallel.prefill import kv_emit

    world_size = 8
    world = _world(world_size)
    tw = _ThreadWorld(world_size, split=True)
    fake_pack = lambda cipher, *, tag="", host=True: _Packet(cipher, tag)  # noqa: E731
    fake_unpack = lambda packet, *, device=None: packet.cipher  # noqa: E731
    monkeypatch.setattr(wire, "pack_cipher", fake_pack)
    monkeypatch.setattr(wire, "unpack_cipher", fake_unpack)
    monkeypatch.setattr(decode_collectives, "pack_cipher", fake_pack)
    monkeypatch.setattr(decode_collectives, "unpack_cipher", fake_unpack)

    results: dict[int, dict[int, _Cipher]] = {}
    errors: dict[int, BaseException] = {}

    def run(rank_ownership) -> None:
        rank = int(rank_ownership.rank)
        key_pairs, value_pairs = kv_cache_contributions(world, rank_ownership)
        owned = {
            pair: _Cipher(f"k{pair}") for pair in key_pairs
        }
        owned.update({4 + pair: _Cipher(f"v{pair}") for pair in value_pairs})
        try:
            results[rank] = kv_emit._owner_directed_raw_route(
                owned,
                pair_count=4,
                pair_blocks=2,
                group_count=2,
                rank=rank,
                world_size=world_size,
                transport=tw.transport(rank),
                context=SimpleNamespace(device="cpu"),
                tag="test/kv_emit",
                comm={},
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors[rank] = exc
            tw.barrier.abort()

    threads = [threading.Thread(target=run, args=(rank,)) for rank in world]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors, errors

    for rank in range(world_size):
        expected = {
            index
            for index in range(8)
            if kv_cache_raw_owner(index, world_size=world_size) == rank
        }
        assert set(results[rank]) == expected
        assert all(
            result.name
            in {f"k{index}" if index < 4 else f"v{index - 4}" for index in expected}
            for result in results[rank].values()
        )


class _StrictLimbs(_Cipher):
    """Fake ciphertext that catches a second ownership release."""

    def __init__(self, name: str, limbs: int) -> None:
        super().__init__(name)
        self.state = SimpleNamespace(cur_limbs=int(limbs), scale_degree=1)
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1
        if self.release_calls != 1:
            raise AssertionError(f"double release: {self.name}")


def _station_raw_inputs(rank_ownership):
    from llama3fhe.parallel.prefill.ownership import key_pairs_for_rank

    key_c = tuple(
        (pair, _StrictLimbs(f"k{pair}", 13))
        for pair in key_pairs_for_rank(rank_ownership)
    )
    value_c = tuple(
        (pair, _StrictLimbs(f"v{pair}", 13)) for pair in range(4)
    )
    return key_c, value_c


def test_dense_direct_converter_failure_releases_untransferred_value_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed K conversion must not strand V inputs after ``raw`` is split."""

    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.operators.attention.pairfold import pairfold_prefill
    from llama3fhe.parallel.prefill import kv_emit

    world = _world(1)
    transport_world = _ThreadWorld(1, split=True)
    key_c, value_c = _station_raw_inputs(world[0])

    def fail_key(keys, **_kwargs):
        for cipher in keys:
            cipher.release()
        raise RuntimeError("injected K conversion failure")

    monkeypatch.setenv("LLAMA_FHE_PREFILL_DIRECT_PAIRFOLD_K", "1")
    monkeypatch.delenv("LLAMA_FHE_PREFILL_OWNER_ROUTING", raising=False)
    monkeypatch.setattr(pairfold_prefill, "prefill_emitted_key_c_direct_to_pairfold_fhe", fail_key)

    with pytest.raises(RuntimeError, match="injected K conversion failure"):
        kv_emit.prefill_kv_cache_station(
            key_c, value_c, ownership=world[0], world=world,
            transport=transport_world.transport(0),
            attention_layout=AttentionPairLayout(),
            context=SimpleNamespace(device="cpu", rank=0),
            bootstrap_operator=SimpleNamespace(), kv_limbs=12,
        )

    assert all(cipher.release_calls == 1 for _, cipher in key_c + value_c)


def test_dense_direct_value_failure_releases_first_converter_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed V conversion must release K outputs already produced."""

    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.operators.attention.pairfold import pairfold_prefill
    from llama3fhe.parallel.prefill import kv_emit

    world = _world(1)
    transport_world = _ThreadWorld(1, split=True)
    key_c, value_c = _station_raw_inputs(world[0])
    produced: list[_StrictLimbs] = []

    def fake_key(keys, **_kwargs):
        for cipher in keys:
            cipher.release()
        output = _StrictLimbs("Kout", 11)
        produced.append(output)
        return (output,)

    def fail_value(values, **_kwargs):
        for cipher in values:
            cipher.release()
        raise RuntimeError("injected V conversion failure")

    monkeypatch.setenv("LLAMA_FHE_PREFILL_DIRECT_PAIRFOLD_K", "1")
    monkeypatch.delenv("LLAMA_FHE_PREFILL_OWNER_ROUTING", raising=False)
    monkeypatch.setattr(pairfold_prefill, "prefill_emitted_key_c_direct_to_pairfold_fhe", fake_key)
    monkeypatch.setattr(pairfold_prefill, "prefill_real_value_c_to_pairfold_fhe", fail_value)

    with pytest.raises(RuntimeError, match="injected V conversion failure"):
        kv_emit.prefill_kv_cache_station(
            key_c, value_c, ownership=world[0], world=world,
            transport=transport_world.transport(0),
            attention_layout=AttentionPairLayout(),
            context=SimpleNamespace(device="cpu", rank=0),
            bootstrap_operator=SimpleNamespace(), kv_limbs=12,
        )

    assert all(cipher.release_calls == 1 for _, cipher in key_c + value_c)
    assert all(cipher.release_calls == 1 for cipher in produced)


def test_owner_group_failure_does_not_rerelease_consumed_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consumed raw groups are removed before conversion failure cleanup."""

    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.operators.attention.pairfold import pairfold_prefill
    from llama3fhe.parallel import wire
    from llama3fhe.parallel.prefill import kv_emit

    world = _world(1)
    transport_world = _ThreadWorld(1, split=True)
    key_c, value_c = _station_raw_inputs(world[0])
    produced: list[_StrictLimbs] = []

    def fake_pack(cipher, *, tag="", host=True):
        return _Packet(cipher, tag)

    def fake_key_group(keys, **_kwargs):
        for cipher in keys:
            cipher.release()
        if any(cipher.name == "k2" for cipher in keys):
            raise RuntimeError("injected grouped K failure")
        output = _StrictLimbs("K0", 11)
        produced.append(output)
        return output

    monkeypatch.setattr(wire, "pack_cipher", fake_pack)
    monkeypatch.setenv("LLAMA_FHE_PREFILL_OWNER_ROUTING", "1")
    monkeypatch.setenv("LLAMA_FHE_PREFILL_DIRECT_PAIRFOLD_K", "1")
    monkeypatch.setattr(
        pairfold_prefill,
        "prefill_emitted_key_c_direct_group_to_pairfold_fhe",
        fake_key_group,
    )

    with pytest.raises(RuntimeError, match="injected grouped K failure"):
        kv_emit.prefill_kv_cache_station(
            key_c, value_c, ownership=world[0], world=world,
            transport=transport_world.transport(0),
            attention_layout=AttentionPairLayout(),
            context=SimpleNamespace(device="cpu", rank=0),
            bootstrap_operator=SimpleNamespace(), kv_limbs=12,
        )

    assert all(cipher.release_calls == 1 for _, cipher in key_c + value_c)
    assert all(cipher.release_calls == 1 for cipher in produced)


def test_value_c_routes_restrict_to_sources_like_delta() -> None:
    pytest.importorskip("numpy")
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.layouts.linear import LinearCarrierLayout
    from llama3fhe.operators.attention.prefill import input as core

    carrier = LinearCarrierLayout(seq_len=128, dimension=4096, slots=32768)
    attention = AttentionPairLayout(seq_len=128, head_dim=128, query_heads=32, key_value_heads=8, slots=32768)
    params = core._layout_parameters(carrier, attention)
    full = core._value_c_routes(params)
    union: dict[int, list[int]] = {}
    for rank in build_world_ownership():
        part = core._value_c_routes(params, sources=tuple(rank.cipher_indices))
        for offset, destinations in part.items():
            union.setdefault(int(offset), []).extend(int(v) for v in destinations)
    assert set(union) == {int(v) for v in full}
    for offset, destinations in full.items():
        assert sorted(union[int(offset)]) == sorted(int(v) for v in destinations)
    with pytest.raises(IndexError):
        core._value_c_routes(params, sources=(16,))


def test_shard_rotations_emit_cache_c_is_a_superset() -> None:
    pytest.importorskip("numpy")
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.layouts.linear import LinearCarrierLayout
    from llama3fhe.operators.attention.distributed.distributed_input import attention_input_shard_rotations
    from llama3fhe.parallel.prefill.ownership import key_pairs_for_rank

    carrier = LinearCarrierLayout(seq_len=128, dimension=4096, slots=32768)
    attention = AttentionPairLayout(seq_len=128, head_dim=128, query_heads=32, key_value_heads=8, slots=32768)
    rank = build_world_ownership()[3]
    kwargs = dict(query_pair_indices=rank.query_pair_indices, key_pair_indices=key_pairs_for_rank(rank),
                  value_carrier_indices=(0, 1), value_source_indices=rank.cipher_indices)
    plain = attention_input_shard_rotations(carrier, attention, **kwargs)
    emit = attention_input_shard_rotations(carrier, attention, emit_cache_c=True, **kwargs)
    assert plain["lane"] == emit["lane"] and plain["query"] == emit["query"]
    assert set(plain["key"]) <= set(emit["key"]) and set(plain["value"]) <= set(emit["value"])
    assert -int(attention.payload_slots) // 2 in emit["key"]
    assert len(emit["value"]) > len(plain["value"])


def test_key_plan_emit_kv_cache_adds_the_converter_rotations() -> None:
    pytest.importorskip("numpy")
    from llama3fhe.parallel.prefill.keys import build_key_plan, normalize_rotation
    from llama3fhe.parallel.presets import RTX4090_LOW
    from llama3fhe.parallel.worker import production_layouts

    feature, attention = production_layouts()
    rank0 = build_world_ownership()[0]
    tiers = ("bootstrap", "attn_input_k", "attn_input_v")
    plain = build_key_plan(RTX4090_LOW, feature, attention, tiers=tiers, ownership=rank0)
    emit = build_key_plan(RTX4090_LOW, feature, attention, tiers=tiers, ownership=rank0, emit_kv_cache=True)
    slots = int(feature.slots)
    ring_dim = 1 << int(RTX4090_LOW.log_n)
    convert = {normalize_rotation(slots // 2, ring_dim=ring_dim), normalize_rotation(4 * slots - 1, ring_dim=ring_dim)}
    for name in ("attn_input_k", "attn_input_v"):
        assert set(plain.tier(name).rotations) <= set(emit.tier(name).rotations)
        assert convert <= set(emit.tier(name).rotations)
    assert plain.tier("bootstrap").rotations == emit.tier("bootstrap").rotations


def test_1b_key_plan_includes_four_block_pairfold_rotations() -> None:
    from llama3fhe.parallel.prefill.keys import build_key_plan, normalize_rotation
    from llama3fhe.parallel.prefill.ownership import ownership_from_layouts
    from llama3fhe.parallel.presets import RTX4090_LOW
    from llama3fhe.parallel.worker import production_layouts

    preset = RTX4090_LOW.with_world_size(4).for_model("llama3.2-1b")
    feature, attention = production_layouts(preset)
    rank = ownership_from_layouts(feature, attention, world_size=4)[0]
    plan = build_key_plan(
        preset, feature, attention,
        tiers=("attn_input_k", "attn_input_v"),
        ownership=rank, emit_kv_cache=True,
    )
    ring_dim = 1 << int(preset.log_n)
    required = {
        normalize_rotation(offset, ring_dim=ring_dim)
        for offset in (8192, 16384, 24576, 131071)
    }
    for tier in ("attn_input_k", "attn_input_v"):
        assert required <= set(plan.tier(tier).rotations)


# ----------------------------------------------------------------- faked station run
class _Limbs(_Cipher):
    def __init__(self, name: str, limbs: int) -> None:
        super().__init__(name)
        self.state = SimpleNamespace(cur_limbs=int(limbs), scale_degree=1)


def _fake_station(
    monkeypatch: pytest.MonkeyPatch, log: dict[int, list[str]],
    *, key_limbs: int = 11, value_limbs: int = 11,
):
    import llama3fhe.operators.attention.pairfold.pairfold_prefill as converter
    import llama3fhe.operators.primitives as primitives
    from llama3fhe.parallel import wire
    from llama3fhe.parallel.decode import decode_collectives

    def fake_convert(keys, values, *, attention_layout, pairfold_layout, crypto_context, emitted_key_c):
        assert emitted_key_c is True
        names = [c.name for c in keys] + [c.name for c in values]
        assert names == [f"kc{i}" for i in range(4)] + [f"vc{i}" for i in range(4)], names
        for cipher in list(keys) + list(values):
            cipher.release()
        log[int(crypto_context.rank)].append("convert")
        return tuple(_Limbs(f"K{g}", key_limbs) for g in range(2)), tuple(
            _Limbs(f"V{g}", value_limbs) for g in range(2)
        )

    def fake_drop(ciphers, *, target_limbs, crypto_context):
        return tuple(_Limbs(c.name, target_limbs) for c in ciphers)

    monkeypatch.setattr(converter, "prefill_c_to_pairfold_fhe", fake_convert)
    monkeypatch.setattr(primitives, "drop_to_limbs", fake_drop)
    # decode_collectives binds pack/unpack at import time -> patch both modules (test-order independent)
    fake_pack = lambda cipher, *, tag="", host=True: _Packet(cipher, tag)  # noqa: E731
    fake_unpack = lambda packet, *, device=None: packet.cipher  # noqa: E731
    for module in (wire, decode_collectives):
        monkeypatch.setattr(module, "pack_cipher", fake_pack)
        monkeypatch.setattr(module, "unpack_cipher", fake_unpack)


class _FakeBootstrap:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def refresh(self, cipher, *, iterations):
        self.calls.append(cipher.name)
        return _Limbs(cipher.name, 18)


def test_split_decode_kv_drop_keeps_value_at_a_separate_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K11 needs a refresh while V12 only drops; the two ranks agree on K12/V8."""

    from llama3fhe.operators import pairfold_kv_cache, primitives
    from llama3fhe.parallel.decode import decode_stations

    class Cipher:
        def __init__(self, name: str, limbs: int) -> None:
            self.name = name
            self.state = SimpleNamespace(cur_limbs=limbs, scale_degree=1)
            self.is_ext = False
            self.released = False

        def release(self) -> None:
            self.released = True

    def drop(ciphers, *, target_limbs, crypto_context):
        return tuple(Cipher(c.name, target_limbs) for c in ciphers)

    monkeypatch.setattr(pairfold_kv_cache, "drop_to_limbs", drop)
    monkeypatch.setattr(primitives, "drop_to_limbs", drop)

    gathered: dict[int, dict[int, Cipher]] = {}
    barrier = threading.Barrier(2)
    exchange_calls: list[int] = []

    def all_gather(transport, _tag, owned, *, count, crypto_context, comm):
        gathered[transport.rank] = dict(owned)
        exchange_calls.append(transport.rank)
        barrier.wait(timeout=5)
        merged = {index: cipher for part in gathered.values() for index, cipher in part.items()}
        assert set(merged) == set(range(count))
        return tuple(Cipher(merged[index].name, merged[index].state.cur_limbs) for index in range(count))

    monkeypatch.setattr(decode_stations, "all_gather_ciphers", all_gather)
    bootstraps = [_FakeBootstrap(), _FakeBootstrap()]
    outputs: dict[int, object] = {}
    errors: dict[int, BaseException] = {}

    def run(rank: int) -> None:
        token = pairfold_kv_cache.PairFoldTokenWrite(
            key_groups=(Cipher("K0", 11), Cipher("K1", 11)),
            value_groups=(Cipher("V0", 12), Cipher("V1", 12)),
            column=0,
        )
        ownership = SimpleNamespace(world_size=2, kv_refresh_indices=lambda _count: (rank, rank + 2))
        try:
            outputs[rank] = decode_stations._split_kv_limb_drop(
                token,
                target_limbs=12,
                value_target_limbs=8,
                crypto_context=SimpleNamespace(),
                bootstrap_operator=bootstraps[rank],
                bootstrap_iterations=1,
                ownership=ownership,
                transport=SimpleNamespace(rank=rank),
                comm={},
            )
        except BaseException as exc:
            errors[rank] = exc
            barrier.abort()

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors, errors
    assert sorted(exchange_calls) == [0, 1]
    assert [bootstrap.calls for bootstrap in bootstraps] == [["K0"], ["K1"]]
    for output in outputs.values():
        assert [cipher.state.cur_limbs for cipher in output.handles] == [12, 12, 8, 8]
        output.release()


def test_split_decode_kv_drop_uses_local_path_when_both_groups_have_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping V12 to V8 does not require a bootstrap or collective."""

    from llama3fhe.operators import pairfold_kv_cache
    from llama3fhe.parallel.decode import decode_stations

    class Cipher:
        def __init__(self, limbs: int) -> None:
            self.state = SimpleNamespace(cur_limbs=limbs, scale_degree=1)
            self.is_ext = False

        def release(self) -> None:
            pass

    def drop(ciphers, *, target_limbs, crypto_context):
        return tuple(Cipher(target_limbs) for _ in ciphers)

    monkeypatch.setattr(pairfold_kv_cache, "drop_to_limbs", drop)
    token = pairfold_kv_cache.PairFoldTokenWrite(
        key_groups=(Cipher(12), Cipher(12)), value_groups=(Cipher(12), Cipher(12)), column=0,
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("no bootstrap or collective should run")

    monkeypatch.setattr(decode_stations, "all_gather_ciphers", unexpected)
    output = decode_stations._split_kv_limb_drop(
        token, target_limbs=12, value_target_limbs=8,
        crypto_context=SimpleNamespace(),
        bootstrap_operator=SimpleNamespace(refresh=unexpected), bootstrap_iterations=1,
        ownership=SimpleNamespace(world_size=2, kv_refresh_indices=unexpected),
        transport=SimpleNamespace(), comm={},
    )
    assert token.is_released
    assert [cipher.state.cur_limbs for cipher in output.handles] == [12, 12, 8, 8]
    output.release()


class _NoSyncBootstrap(_FakeBootstrap):
    def __init__(self) -> None:
        super().__init__()
        self.synchronize: list[bool] = []

    def refresh(self, cipher, *, iterations, synchronize=True):
        self.synchronize.append(bool(synchronize))
        return super().refresh(cipher, iterations=iterations)


@pytest.mark.parametrize("world_size", [8, 4, 2, 1])
@pytest.mark.parametrize("kv_value_limbs", [None, 8])
def test_station_builds_the_same_page_on_every_rank(
    monkeypatch: pytest.MonkeyPatch, world_size: int, kv_value_limbs: int | None,
) -> None:
    import llama3fhe.operators.pairfold_kv_cache as kv
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.parallel.prefill.kv_emit import prefill_kv_cache_station
    from llama3fhe.parallel.prefill.ownership import key_pairs_for_rank, value_carriers_for_rank

    # the page / cache wrappers validate handles via the backend; keep them shape-only here
    monkeypatch.setattr(kv, "_validate_unique_handles", lambda handles, owner="": None)
    world = _world(world_size)
    attention = AttentionPairLayout()
    tw = _ThreadWorld(world_size, split=True)
    _fake_station(monkeypatch, tw.log)
    bootstraps = {r: _FakeBootstrap() for r in range(world_size)}
    results: dict[int, object] = {}
    errors: dict[int, BaseException] = {}

    def run(rank_ownership):
        rank = int(rank_ownership.rank)
        key_c = tuple((p, _Limbs(f"kc{p}", 13)) for p in key_pairs_for_rank(rank_ownership))
        # W=8/4: one value carrier per rank; W=2/1: several (the single-card chain holds all four)
        value_c = tuple(
            (p, _Limbs(f"vc{p}", 13))
            for carrier in value_carriers_for_rank(rank_ownership)
            for p in (2 * carrier, 2 * carrier + 1)
        )
        try:
            results[rank] = prefill_kv_cache_station(
                key_c, value_c, ownership=rank_ownership, world=world, transport=tw.transport(rank),
                attention_layout=attention, context=SimpleNamespace(device="cpu", rank=rank),
                bootstrap_operator=bootstraps[rank], kv_limbs=12, kv_value_limbs=kv_value_limbs,
            )
        except BaseException as exc:  # pragma: no cover
            errors[rank] = exc
            tw.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in world]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    for rank in range(world_size):
        out = results[rank]
        # W=1 holds every raw cipher locally: no all-gather on either side of the conversion
        assert tw.log[rank] == (["convert"] if world_size == 1 else ["ag", "convert", "ag"])
        cache = out.cache
        assert cache.content_tokens == 128 and len(cache.pages) == 1
        page = cache.pages[0]
        assert [c.name for c in page.key_groups] == ["K0", "K1"]
        assert [c.name for c in page.value_groups] == ["V0", "V1"]
        value_target = 12 if kv_value_limbs is None else kv_value_limbs
        assert [c.state.cur_limbs for c in page.handles] == [12, 12, value_target, value_target]
        assert out.limbs == {
            "raw_in": [13] * len(out.limbs["raw_in"]),
            "converted": [11] * 4,
            "cache": [12, 12, value_target, value_target],
        }
        assigned = kv_cache_refresh_indices(rank, world_size)
        expected_bootstrap = tuple(
            index for index in assigned if kv_value_limbs is None or index < 2
        )
        assert out.refreshed == assigned
        assert out.bootstrap_calls == len(expected_bootstrap)
        assert bootstraps[rank].calls == [["K0", "K1", "V0", "V1"][i] for i in expected_bootstrap]
        assert set(out.stage_seconds) == {"gather_raw", "convert", "refresh", "gather_pairfold", "page"}
        assert out.comm_bytes.get("kv_emit/raw_sent", 0) == 8 * (world_size - 1) * len(out.limbs["raw_in"])
    # the eight raw ciphers were contributed once in total
    assert sum(len(results[r].limbs["raw_in"]) for r in range(world_size)) == 8


def test_station_k4_v8_drops_without_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Already sufficient converted levels can populate unequal K/V targets directly."""

    import llama3fhe.operators.pairfold_kv_cache as kv
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.parallel.prefill.kv_emit import prefill_kv_cache_station

    monkeypatch.setattr(kv, "_validate_unique_handles", lambda handles, owner="": None)
    world = _world(1)
    tw = _ThreadWorld(1, split=True)
    _fake_station(monkeypatch, tw.log, key_limbs=4, value_limbs=8)
    bootstrap = _FakeBootstrap()
    key_c = tuple((pair, _Limbs(f"kc{pair}", 13)) for pair in range(4))
    value_c = tuple((pair, _Limbs(f"vc{pair}", 13)) for pair in range(4))
    result = prefill_kv_cache_station(
        key_c, value_c, ownership=world[0], world=world,
        transport=tw.transport(0), attention_layout=AttentionPairLayout(),
        context=SimpleNamespace(device="cpu", rank=0), bootstrap_operator=bootstrap,
        kv_limbs=4, kv_value_limbs=8,
    )
    assert result.bootstrap_calls == 0
    assert bootstrap.calls == []
    assert result.limbs["converted"] == [4, 4, 8, 8]
    assert result.limbs["cache"] == [4, 4, 8, 8]
    assert result.refreshed == (0, 1, 2, 3)  # assigned outputs, not actual BS calls
    result.cache.release()


def test_station_can_leave_bootstrap_on_current_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.parallel.prefill import kv_emit

    world = _world(1)
    tw = _ThreadWorld(1, split=True)
    log: dict[int, list[str]] = {0: []}
    _fake_station(monkeypatch, log)
    bootstrap = _NoSyncBootstrap()
    key_c = tuple((pair, _StrictLimbs(f"kc{pair}", 13)) for pair in range(4))
    value_c = tuple((pair, _StrictLimbs(f"vc{pair}", 13)) for pair in range(4))
    result = kv_emit.prefill_kv_cache_station(
        key_c, value_c, ownership=world[0], world=world,
        transport=tw.transport(0), attention_layout=AttentionPairLayout(),
        context=SimpleNamespace(device="cpu", rank=0), bootstrap_operator=bootstrap,
        kv_limbs=12, synchronize=False,
    )
    assert bootstrap.synchronize == [False] * 4
    result.release()


def test_w1_batches_cache_refreshes_by_limb(monkeypatch: pytest.MonkeyPatch) -> None:
    """One rank owns K0 K1 V0 V1. Same-limb ciphertexts share one refresh_many."""

    import llama3fhe.operators.attention.pairfold.pairfold_prefill as converter
    import llama3fhe.operators.pairfold_kv_cache as kv
    import llama3fhe.operators.primitives as primitives
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.parallel import wire
    from llama3fhe.parallel.decode import decode_collectives
    from llama3fhe.parallel.prefill.kv_emit import prefill_kv_cache_station

    monkeypatch.setattr(kv, "_validate_unique_handles", lambda handles, owner="": None)

    def fake_convert(keys, values, *, attention_layout, pairfold_layout, crypto_context, emitted_key_c):
        for cipher in list(keys) + list(values):
            cipher.release()
        return tuple(_Limbs(f"K{g}", 7) for g in range(2)), tuple(_Limbs(f"V{g}", 8) for g in range(2))

    monkeypatch.setattr(converter, "prefill_c_to_pairfold_fhe", fake_convert)
    monkeypatch.setattr(
        primitives, "drop_to_limbs",
        lambda ciphers, *, target_limbs, crypto_context: tuple(_Limbs(c.name, target_limbs) for c in ciphers),
    )
    fake_pack = lambda cipher, *, tag="", host=True: _Packet(cipher, tag)  # noqa: E731
    fake_unpack = lambda packet, *, device=None: packet.cipher  # noqa: E731
    for module in (wire, decode_collectives):
        monkeypatch.setattr(module, "pack_cipher", fake_pack)
        monkeypatch.setattr(module, "unpack_cipher", fake_unpack)

    class _BatchBootstrap:
        def __init__(self) -> None:
            self.batches: list[tuple[bool, list[str]]] = []
            self.refresh_calls = 0

        def refresh(self, cipher, *, iterations, synchronize=True):
            self.refresh_calls += 1
            return _Limbs(cipher.name, 18)

        def refresh_many(self, ciphers, *, iterations, synchronize=True):
            names = [cipher.name for cipher in ciphers]
            self.batches.append((bool(synchronize), names))
            return [_Limbs(name, 18) for name in names]

    world = _world(1)
    tw = _ThreadWorld(1, split=True)
    bootstrap = _BatchBootstrap()
    result = prefill_kv_cache_station(
        tuple((pair, _Limbs(f"kc{pair}", 9)) for pair in range(4)),
        tuple((pair, _Limbs(f"vc{pair}", 9)) for pair in range(4)),
        ownership=world[0], world=world, transport=tw.transport(0),
        attention_layout=AttentionPairLayout(),
        context=SimpleNamespace(device="cpu", rank=0),
        bootstrap_operator=bootstrap, kv_limbs=12, synchronize=False,
    )
    assert bootstrap.refresh_calls == 0
    assert bootstrap.batches == [(False, ["K0", "K1"]), (False, ["V0", "V1"])]
    assert result.bootstrap_calls == 4
    assert [cipher.state.cur_limbs for cipher in result.cache.pages[0].handles] == [12, 12, 12, 12]
    result.release()


def test_1b_cache_emission_refreshes_key_without_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 1B cache has one K/V group; V at eight limbs needs no refresh."""

    import llama3fhe.operators.attention.pairfold.pairfold_prefill as converter
    import llama3fhe.operators.pairfold_kv_cache as kv
    import llama3fhe.operators.primitives as primitives
    from llama3fhe.layouts.pairfold128 import pairfold_layout_for
    from llama3fhe.model_spec import LLAMA32_1B
    from llama3fhe.parallel.prefill.kv_emit import prefill_kv_cache_station
    from llama3fhe.parallel.prefill.ownership import ownership_from_layouts

    feature, attention = LLAMA32_1B.prefill_layouts()
    pairfold = pairfold_layout_for(
        head_dim=LLAMA32_1B.head_dim,
        query_heads=LLAMA32_1B.query_heads,
        key_value_heads=LLAMA32_1B.key_value_heads,
    )
    assert pairfold.key_value_groups == 1
    world = ownership_from_layouts(feature, attention, world_size=1)
    monkeypatch.setattr(kv, "_validate_unique_handles", lambda handles, owner="": None)

    def fake_convert(keys, values, *, attention_layout, pairfold_layout,
                     crypto_context, emitted_key_c):
        assert pairfold_layout.key_value_groups == 1
        assert len(keys) == len(values) == 4
        for cipher in tuple(keys) + tuple(values):
            cipher.release()
        return (_Limbs("K0", 7),), (_Limbs("V0", 8),)

    monkeypatch.setattr(converter, "prefill_c_to_pairfold_fhe", fake_convert)
    monkeypatch.setattr(
        primitives, "drop_to_limbs",
        lambda ciphers, *, target_limbs, crypto_context:
            tuple(_Limbs(cipher.name, target_limbs) for cipher in ciphers),
    )
    bootstrap = _FakeBootstrap()
    result = prefill_kv_cache_station(
        tuple((pair, _Limbs(f"kc{pair}", 13)) for pair in range(4)),
        tuple((pair, _Limbs(f"vc{pair}", 13)) for pair in range(4)),
        ownership=world[0], world=world, transport=_ThreadWorld(1, split=True).transport(0),
        attention_layout=attention, pairfold_layout=pairfold,
        context=SimpleNamespace(device="cpu", rank=0),
        bootstrap_operator=bootstrap, kv_limbs=12, kv_value_limbs=8,
    )
    page = result.cache.pages[0]
    assert [cipher.name for cipher in page.key_groups] == ["K0"]
    assert [cipher.name for cipher in page.value_groups] == ["V0"]
    assert [cipher.state.cur_limbs for cipher in page.handles] == [12, 8]
    assert result.bootstrap_calls == 1
    assert bootstrap.calls == ["K0"]
    result.release()


def test_stream_pending_waits_on_side_event_before_returning(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-in worker passes the station's no-sync flag and publishes an event dependency."""

    import types

    from llama3fhe.parallel.prefill import kv_emit

    assert "prefill_kv_cache_stream_begin" in kv_emit.__all__

    class FakeStream:
        def __init__(self, device=None):
            self.device = device
            self.entered = 0

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, *_exc):
            return False

        def synchronize(self):
            return None

    class FakeEvent:
        def __init__(self):
            self.recorded = None
            self.synchronized = 0

        def record(self, stream):
            self.recorded = stream

        def synchronize(self):
            self.synchronized += 1

    class FakeCurrent:
        def __init__(self):
            self.waited = []

        def wait_event(self, event):
            self.waited.append(event)

    current = FakeCurrent()
    fake_cuda = types.SimpleNamespace(
        Stream=FakeStream,
        Event=FakeEvent,
        current_stream=lambda device=None: current,
        stream=lambda stream: stream,
    )
    fake_easyfhe = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setitem(sys.modules, "easyfhe", fake_easyfhe)

    calls: dict[str, object] = {}
    result = SimpleNamespace(release=lambda: calls.setdefault("released", True))

    def fake_station(**kwargs):
        calls.update(kwargs)
        raw_ready = kwargs.get("raw_ready_event")
        if raw_ready is not None:
            raw_ready.set()
        conversion_ready = kwargs.get("gather_ready_event")
        if conversion_ready is not None:
            conversion_ready.record(None)
        return result

    monkeypatch.setattr(kv_emit, "prefill_kv_cache_station", fake_station)
    raw = SimpleNamespace()
    keys = SimpleNamespace(plan=SimpleNamespace(policy="resident"))
    pending = kv_emit.prefill_kv_cache_stream_begin(
        raw,
        keys=keys,
        context=SimpleNamespace(device="cuda:0"),
        ownership=SimpleNamespace(),
        world=SimpleNamespace(),
        transport=SimpleNamespace(),
        attention_layout=SimpleNamespace(),
        bootstrap_operator=SimpleNamespace(),
        kv_limbs=12,
    )
    assert pending is not None
    pending.wait_conversion()
    assert pending.finish() is result
    assert calls["raw_pending"] is raw
    assert calls["key_c_items"] == ()
    assert calls["value_c_items"] == ()
    assert calls["synchronize"] is False
    assert current.waited == [pending.conversion_ready, pending.ready]
    assert pending.ready.recorded is pending.stream


def test_stream_conversion_wait_requires_recorded_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """A delayed side worker must publish the CUDA event before the caller waits on it."""

    import types

    from llama3fhe.parallel.prefill import kv_emit

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def synchronize(self):
            return None

    class FakeEvent:
        def __init__(self):
            self.recorded = False

        def record(self, _stream=None):
            self.recorded = True

        def synchronize(self):
            assert self.recorded

    class FakeCurrent:
        def __init__(self):
            self.waited = []

        def wait_event(self, event):
            assert event.recorded, "wait_event was called before event.record"
            self.waited.append(event)

    current = FakeCurrent()
    fake_cuda = types.SimpleNamespace(
        Stream=lambda device=None: FakeStream(),
        Event=FakeEvent,
        current_stream=lambda device=None: current,
        stream=lambda stream: stream,
    )
    monkeypatch.setitem(sys.modules, "easyfhe", types.SimpleNamespace(cuda=fake_cuda))

    conversion_started = threading.Event()
    release_conversion = threading.Event()
    caller_started = threading.Event()
    caller_done = threading.Event()
    caller_errors: list[BaseException] = []
    result = SimpleNamespace(release=lambda: None)

    def fake_station(**kwargs):
        kwargs["raw_ready_event"].set()
        conversion_started.set()
        if not release_conversion.wait(timeout=5):
            raise AssertionError("conversion was not released")
        kwargs["gather_ready_event"].record()
        if not kwargs["gather_gate"].wait(timeout=5):
            raise AssertionError("final gather was not released")
        return result

    monkeypatch.setattr(kv_emit, "prefill_kv_cache_station", fake_station)
    pending = kv_emit.prefill_kv_cache_stream_begin(
        SimpleNamespace(),
        keys=SimpleNamespace(plan=SimpleNamespace(policy="resident")),
        context=SimpleNamespace(device="cuda:0"),
    )
    assert pending is not None
    assert conversion_started.wait(timeout=2)

    def wait_in_caller():
        caller_started.set()
        try:
            pending.wait_conversion()
        except BaseException as exc:
            caller_errors.append(exc)
        finally:
            caller_done.set()

    caller = threading.Thread(target=wait_in_caller)
    caller.start()
    try:
        assert caller_started.wait(timeout=2)
        assert not caller_done.wait(timeout=0.05)
        assert current.waited == []
        release_conversion.set()
        assert caller_done.wait(timeout=2)
        assert caller_errors == []
        assert current.waited == [pending.conversion_ready]
    finally:
        release_conversion.set()
        pending.allow_final_gather()
        caller.join(timeout=2)
        pending.finish()
    assert not caller.is_alive()


def test_stream_conversion_wait_surfaces_worker_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed conversion wakes wait_conversion without waiting on an unrecorded event."""

    import types

    from llama3fhe.parallel.prefill import kv_emit

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def synchronize(self):
            return None

    class FakeEvent:
        def record(self, _stream=None):
            return None

    current = SimpleNamespace(wait_event=lambda _event: pytest.fail("failed conversion waited on event"))
    fake_cuda = types.SimpleNamespace(
        Stream=lambda device=None: FakeStream(), Event=FakeEvent,
        current_stream=lambda device=None: current, stream=lambda stream: stream,
    )
    monkeypatch.setitem(sys.modules, "easyfhe", types.SimpleNamespace(cuda=fake_cuda))

    def fake_station(**kwargs):
        kwargs["raw_ready_event"].set()
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(kv_emit, "prefill_kv_cache_station", fake_station)
    pending = kv_emit.prefill_kv_cache_stream_begin(
        SimpleNamespace(),
        keys=SimpleNamespace(plan=SimpleNamespace(policy="resident")),
        context=SimpleNamespace(device="cuda:0"),
    )
    assert pending is not None
    with pytest.raises(RuntimeError, match="conversion failed"):
        pending.wait_conversion()
    assert not pending.thread.is_alive()


def test_stream_pending_holds_final_gather_until_pairfm_round(monkeypatch: pytest.MonkeyPatch) -> None:
    """The side worker may finish refresh but must wait for the PairFM gate."""

    import types

    from llama3fhe.parallel.prefill import kv_emit

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def synchronize(self):
            return None

    class FakeEvent:
        def __init__(self):
            self.recorded = None

        def record(self, stream):
            self.recorded = stream

        def synchronize(self):
            return None

    current = SimpleNamespace(waited=[])
    current.wait_event = lambda event: current.waited.append(event)
    fake_cuda = types.SimpleNamespace(
        Stream=lambda device=None: FakeStream(),
        Event=FakeEvent,
        current_stream=lambda device=None: current,
        stream=lambda stream: stream,
    )
    monkeypatch.setitem(sys.modules, "easyfhe", types.SimpleNamespace(cuda=fake_cuda))

    entered = threading.Event()
    result = SimpleNamespace(release=lambda: None)

    def fake_station(**kwargs):
        gate = kwargs["gather_gate"]
        entered.set()
        if not gate.wait(timeout=2):
            raise AssertionError("final-gather gate was not released")
        return result

    monkeypatch.setattr(kv_emit, "prefill_kv_cache_station", fake_station)
    pending = kv_emit.prefill_kv_cache_stream_begin(
        SimpleNamespace(),
        keys=SimpleNamespace(plan=SimpleNamespace(policy="resident")),
        context=SimpleNamespace(device="cuda:0"), ownership=SimpleNamespace(),
        world=SimpleNamespace(), transport=SimpleNamespace(), attention_layout=SimpleNamespace(),
        bootstrap_operator=SimpleNamespace(), kv_limbs=12,
    )
    assert pending is not None
    assert entered.wait(timeout=2)
    assert pending.thread.is_alive()
    pending.allow_final_gather()
    assert pending.finish() is result
    assert current.waited == [pending.ready]


def test_stream_pending_cancel_releases_completed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """An aborted layer drains the worker and releases an unconsumed cache."""

    import types

    from llama3fhe.parallel.prefill import kv_emit

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def synchronize(self):
            return None

    class FakeEvent:
        def record(self, _stream):
            return None

        def synchronize(self):
            return None

    fake_cuda = types.SimpleNamespace(
        Stream=lambda device=None: FakeStream(),
        Event=FakeEvent,
        current_stream=lambda device=None: FakeStream(),
        stream=lambda stream: stream,
    )
    monkeypatch.setitem(sys.modules, "easyfhe", types.SimpleNamespace(cuda=fake_cuda))

    released: list[bool] = []
    result = SimpleNamespace(release=lambda: released.append(True))
    monkeypatch.setattr(kv_emit, "prefill_kv_cache_station", lambda **_kwargs: result)
    pending = kv_emit.prefill_kv_cache_stream_begin(
        SimpleNamespace(),
        keys=SimpleNamespace(plan=SimpleNamespace(policy="resident")),
        context=SimpleNamespace(device="cuda:0"),
        ownership=SimpleNamespace(), world=SimpleNamespace(), transport=SimpleNamespace(),
        attention_layout=SimpleNamespace(), bootstrap_operator=SimpleNamespace(), kv_limbs=12,
    )
    assert pending is not None
    pending.cancel()
    assert released == [True]
    assert pending._cancelled


def test_stream_pending_surfaces_worker_error_once() -> None:
    """A conversion failure is raised at finish and the worker is not leaked."""

    from llama3fhe.parallel.prefill.kv_emit import PrefillKVCacheStreamPending

    class Event:
        def synchronize(self):
            raise AssertionError("failed workers must not wait on a success event")

    released: list[bool] = []
    box: dict[str, object] = {
        # Simulate a result constructed before an event-record failure.  The
        # pending object must reclaim it when surfacing the worker exception.
        "result": SimpleNamespace(release=lambda: released.append(True)),
    }

    def worker() -> None:
        box["error"] = RuntimeError("injected stream conversion failure")

    thread = threading.Thread(target=worker)
    thread.start()
    pending = PrefillKVCacheStreamPending(
        raw_pending=SimpleNamespace(), station_kwargs={}, stream=SimpleNamespace(),
        ready=Event(), conversion_ready=Event(), gather_gate=threading.Event(),
        thread=thread, box=box,
    )
    with pytest.raises(RuntimeError, match="injected stream conversion failure"):
        pending.finish()
    assert pending._finished
    assert released == [True]
    with pytest.raises(RuntimeError, match="injected stream conversion failure"):
        pending.finish()


def test_stream_eligibility_rejects_cpu_and_leased_keys() -> None:
    from llama3fhe.parallel.prefill.kv_emit import _stream_key_plan_is_resident

    resident = SimpleNamespace(plan=SimpleNamespace(policy="resident"))
    leased = SimpleNamespace(plan=SimpleNamespace(policy="lease"))
    assert not _stream_key_plan_is_resident(SimpleNamespace(device="cpu"), resident)
    assert not _stream_key_plan_is_resident(SimpleNamespace(device="cuda:0"), leased)
    assert _stream_key_plan_is_resident(SimpleNamespace(device="cuda:0"), resident)
