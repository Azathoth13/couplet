"""Regression coverage for the 1B eight-rank layout used by the release."""

from types import SimpleNamespace

import pytest

from bench.t2_prefill_decode import check_replay_cache_isolation, replay_token_expectation
from llama3fhe.model_spec import LLAMA32_1B
from llama3fhe.operators.linear.shard.mlp_token_shard import mlp_token_shard_plan
from llama3fhe.operators.linear.shard.qkv_source_shard import qkv_token_shard_plan
from llama3fhe.parallel.decode.decode_ownership import DecodeRankOwnership
from llama3fhe.parallel.prefill.ownership import ownership_from_layouts
from llama3fhe.parallel.presets import preset_by_name


@pytest.mark.parametrize("world_size", (1, 2, 4, 8))
def test_one_b_shards_cover_tokens_and_queries_without_omissions(world_size):
    layout, attention = LLAMA32_1B.prefill_layouts()
    preset = preset_by_name("rtx6000pro-high").with_world_size(world_size).for_model(LLAMA32_1B.name)
    world = ownership_from_layouts(layout, attention_layout=attention, world_size=world_size)
    qkv = qkv_token_shard_plan(layout, tuple(range(world_size)))
    mlp = mlp_token_shard_plan(layout, tuple(range(world_size)))
    assert preset.cipher_count == 8
    assert [i for rank in world for i in rank.cipher_indices] == list(range(8))
    assert [i for rank in world for i in rank.query_pair_indices] == list(range(16))
    for plan in (qkv, mlp):
        assert [(s.token_start, s.token_stop) for s in plan.assignments] == [
            (128 * rank // world_size, 128 * (rank + 1) // world_size)
            for rank in range(world_size)
        ]
    if world_size == 8:
        assert [r.cipher_indices for r in world] == [(i,) for i in range(8)]
        assert [r.target_carrier_indices for r in world] == [(i // 2,) for i in range(8)]


def test_one_b_decode_eight_ranks_owns_each_of_four_query_streams_once():
    ownership = [DecodeRankOwnership(rank, 8, key_value_groups=1, gqa_ratio=4, mlp_pages=4)
                 for rank in range(8)]
    assert [stream for rank in ownership for stream in rank.query_streams] == [
        (0, 0), (0, 1), (0, 2), (0, 3)
    ]
    assert [rank.rank for rank in ownership if not rank.query_streams] == [1, 3, 5, 7]


def test_replay_validates_real_prompt_length_and_preserves_original_cache():
    original = SimpleNamespace(content_tokens=73)
    replay = SimpleNamespace(content_tokens=74)
    check_replay_cache_isolation(original, replay, 73)
    original.content_tokens = 74
    with pytest.raises(RuntimeError, match="original cache changed"):
        check_replay_cache_isolation(original, replay, 73)


def test_replay_expectation_has_no_model_specific_token_assumption():
    tokens = [101, 202, 202]
    assert replay_token_expectation(tokens) == tokens
    assert replay_token_expectation(tokens, 303) == [101, 303, 303]
    assert replay_token_expectation([101, 202, 303]) != [101, 202, 303]
