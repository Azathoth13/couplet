"""CPU checks for the physical layout of a semantics-preserving 3B export."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest

# The top-level package eagerly imports the EasyFHE runtime.  These tests only
# exercise the real CPU model/layout/preset modules, so on a workstation
# without EasyFHE load them through a namespace package instead.  A runtime
# installation still uses the ordinary package import path.
if importlib.util.find_spec("easyfhe") is None and "llama3fhe" not in sys.modules:
    package = types.ModuleType("llama3fhe")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "llama3fhe")]
    sys.modules["llama3fhe"] = package

from llama3fhe.application.args import model_poly_paths
from llama3fhe.model_spec import (
    LLAMA3_8B,
    LLAMA32_1B,
    LLAMA32_3B_PADDED,
    model_spec_by_name,
    model_spec_from_hf_config,
)
from llama3fhe.parallel.presets import RTX6000PRO_HIGH


def test_padded_3b_has_the_required_runtime_geometry() -> None:
    spec = model_spec_by_name("llama3.2-3b-padded")
    assert spec is LLAMA32_3B_PADDED
    assert (spec.source_hidden_dim, spec.source_query_heads) == (3072, 24)
    assert (spec.hidden_dim, spec.query_heads, spec.key_value_heads) == (4096, 32, 8)
    assert (spec.head_dim, spec.intermediate_dim, spec.num_layers) == (128, 8192, 28)
    assert math.isclose(spec.rms_eps, 1e-5 * 0.75)
    assert spec.fm_cipher_count() == spec.query_pair_count == 16

    feature, attention = spec.prefill_layouts()
    assert (feature.hidden_dim, feature.tokens_per_cipher, feature.cipher_count) == (4096, 8, 16)
    assert (attention.query_pair_count, attention.gqa_ratio, attention.key_value_pair_count) == (16, 4, 4)
    for world in (1, 2, 4, 8):
        preset = RTX6000PRO_HIGH.with_world_size(world).for_model(spec.name)
        assert preset.cipher_count == 16
        assert preset.ciphers_per_rank == 16 // world


def test_original_q_heads_map_within_their_kv_groups() -> None:
    spec = LLAMA32_3B_PADDED
    mapped = tuple(spec.physical_query_head(head) for head in range(24))
    assert mapped == tuple(group * 4 + offset for group in range(8) for offset in range(3))
    assert set(range(32)) - set(mapped) == {4 * group + 3 for group in range(8)}
    with pytest.raises(IndexError):
        spec.physical_query_head(24)
    assert LLAMA32_1B.physical_query_head(7) == 7
    assert LLAMA3_8B.physical_query_head(7) == 7
    assert LLAMA32_1B.fm_cipher_count() == 8
    assert LLAMA3_8B.fm_cipher_count() == 16


def test_only_padded_config_can_be_used_for_3b_runtime() -> None:
    # A bare "3b" CLI alias would falsely advertise support for the official
    # 3072-wide checkpoint, which the FM layout cannot represent directly.
    with pytest.raises(KeyError):
        model_spec_by_name("llama3.2-3b")
    padded = {
        "hidden_size": 4096,
        "intermediate_size": 8192,
        "num_hidden_layers": 28,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "rope_theta": 500000.0,
        "rms_norm_eps": 7.5e-6,
        "tie_word_embeddings": False,
        "odin_padding": {"source_hidden_dim": 3072, "source_query_heads": 24},
    }
    inferred = model_spec_from_hf_config(padded, name="llama3.2-3b-padded")
    assert (inferred.hidden_dim, inferred.query_heads, inferred.source_hidden_dim, inferred.source_query_heads) == (
        4096, 32, 3072, 24
    )
    assert math.isclose(inferred.rms_eps, LLAMA32_3B_PADDED.rms_eps)
    raw = {**padded, "hidden_size": 3072, "num_attention_heads": 24, "rms_norm_eps": 1e-5}
    raw.pop("odin_padding")
    with pytest.raises(ValueError):
        model_spec_from_hf_config(raw).prefill_layouts()


def test_padded_3b_owns_a_separate_polynomial_pack() -> None:
    paths = model_poly_paths("llama3.2-3b-padded")
    assert {path.name for path in paths} == {
        "rmsnorm_poly_coeffs.json", "silu_poly_coeffs.json", "softmax_poly_coeffs.json"
    }
    root = Path(__file__).resolve().parents[1] / "assets/polynomials"
    assert {path.relative_to(root).parts[0] for path in paths} == {"3b"}
