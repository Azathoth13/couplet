"""Selected-layer reporting never changes arithmetic or unselected records."""
from copy import deepcopy

import pytest

from bench.breakdown.records import nested_stage_records, parse_profile_layers


def test_profile_selection_respects_model_geometry():
    assert parse_profile_layers("", count=16) == frozenset()
    assert parse_profile_layers("0,8-9,15", count=16) == frozenset({0, 8, 9, 15})
    with pytest.raises(ValueError):
        parse_profile_layers("16", count=16)


def test_nested_timer_copy_only_reports_selected_layer():
    nested = {"decode_attention": {"qk_s": 0.123456789, "bootstrap": [1, 2]},
              "mlp": {"gate_up_s": 0.75}, "other": {"x": 1}}
    original = deepcopy(nested)
    assert nested_stage_records(nested, 15, frozenset({16})) == {}
    report = nested_stage_records(nested, 16, frozenset({16}))
    assert report == {"decode_attention_stages": {"qk_s": 0.123457},
                      "decode_mlp_stages": {"gate_up_s": 0.75}}
    assert nested == original
