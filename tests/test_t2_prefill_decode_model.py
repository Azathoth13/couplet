"""CPU-only checks for the model-specific Prefill -> Decode runner setup."""

from __future__ import annotations

import json
import struct
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bench import t2_prefill_decode as runner
from bench.t2_decode import decode_layouts
from llama3fhe.graph import LayerApproximations
from llama3fhe.operators.attention.config import AttentionOperatorConfig
from llama3fhe.parallel.presets import preset_by_name
from llama3fhe.parallel.worker import production_layouts


def test_decode_rmsnorm_prescale_uses_absolute_position() -> None:
    table = json.loads((Path(__file__).resolve().parents[1] / "assets/polynomials" /
                        "1b/runtime/rmsnorm_poly_coeffs.json").read_text())
    attn = table["branches"]["attn"]["2"]
    ffn = table["branches"]["ffn"]["2"]
    assert "0" in attn["token_prescale"] and "0" in ffn["token_prescale"]
    layer = LayerApproximations(input_norm=attn, post_attention_norm=ffn, silu={}, softmax=None)

    decoded = runner.decode_approximations_at_position(layer, 128)
    assert "token_prescale" not in decoded.input_norm
    assert "token_prescale" not in decoded.post_attention_norm
    assert decoded.input_norm["mean_square_prescale"] == attn["mean_square_prescale"]
    assert "0" in layer.input_norm["token_prescale"]  # Prefill keeps its BOS scale.

    at_prompt_start = runner.decode_approximations_at_position(layer, 0)
    assert at_prompt_start.input_norm["token_prescale"] == {"0": attn["token_prescale"]["0"]}


def test_linked_prefill_uses_pure_prefill_post_exp_and_slim_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("T2_SOFTMAX_POST_EXP_LAYERS", "0-1")
    monkeypatch.setenv("T2_SOFTMAX_POST_EXP_QKV_LIMBS", "15")
    monkeypatch.setenv("T2_RMS_SLIM_MIN_AUTO", "1")
    monkeypatch.delenv("T2_RMS_SLIM_MIN_AUTO_BRANCHES", raising=False)
    monkeypatch.delenv("LLAMA_FHE_SOFTMAX_POST_EXP_BOOTSTRAP", raising=False)
    monkeypatch.delenv("LLAMA_FHE_SOFTMAX_POST_ALG2_BOOTSTRAP", raising=False)
    base = AttentionOperatorConfig(qk_score_level_drop=3, refresh_softmax_exp_inputs=True,
                                   pair_softmax_pre_exp_bootstrap=True)
    exits = {"exit_refresh": False, "slim_refresh_min_limbs": 16}

    layer_op, core, qkv, attn_exit, ffn_exit = runner.prefill_layer_runtime_settings(
        0, operator_config=base, core_config=base, qkv_limbs=13, mlp_limbs=10,
        rms_exit=exits, rms_exit_post=exits)
    assert qkv == 15
    assert layer_op.qk_score_level_drop == core.qk_score_level_drop == 0
    assert not core.refresh_softmax_exp_inputs and not core.pair_softmax_pre_exp_bootstrap
    assert attn_exit["slim_refresh_min_limbs"] == 15
    assert ffn_exit["slim_refresh_min_limbs"] == 10
    assert exits["slim_refresh_min_limbs"] == 16
    assert runner.softmax_post_exp_description()["post_exp_env"] == "0,1"

    layer_op, core, qkv, attn_exit, _ = runner.prefill_layer_runtime_settings(
        2, operator_config=base, core_config=base, qkv_limbs=13, mlp_limbs=10,
        rms_exit=exits, rms_exit_post=exits)
    assert qkv == 13 and layer_op.qk_score_level_drop == core.qk_score_level_drop == 3
    assert core.refresh_softmax_exp_inputs and core.pair_softmax_pre_exp_bootstrap
    assert attn_exit["slim_refresh_min_limbs"] == 13


def test_one_b_k12_v8_layout_and_merged_key_plan() -> None:
    preset = preset_by_name("rtx6000pro-high")
    preset = replace(preset, world_size=1, ciphers_per_rank=16, mlp_feature_shard_k=1,
                     linear_max_plaintext_rows=512).for_model("llama3.2-1b")
    preset = runner.combined_preset(preset)

    feature, attention = production_layouts(preset)
    decode_feature, decode_attention, pairfold = decode_layouts(preset)
    assert preset.ciphers_per_rank == 8
    assert (feature.hidden_dim, attention.head_dim) == (2048, 64)
    assert (decode_feature.hidden_dim, decode_attention.head_dim, pairfold.head_dim) == (2048, 64, 64)
    assert runner.decode_rows_for_world(2048, 4, preset.model_spec.hidden_dim) == 512
    assert runner.decode_rows_for_world(2048, 8) == 512  # unchanged 8B default

    decode_preset, schedule = runner.decode_preset_and_schedule(
        preset, kv_limbs=12, kv_value_limbs=8, shifted_lanes=True, sl_baby=16, rows=2048)
    assert decode_preset.linear_max_plaintext_rows == 2048
    assert (schedule.kv_limbs, schedule.kv_value_limbs) == (12, 8)
    merged, sizes, prefill, decode = runner.rank_phase_key_plans(
        preset, 0, decode_preset=decode_preset, decode_schedule=schedule,
        layers=tuple(range(16)), prefix_k=0)
    assert set(prefill.union) <= set(merged.union)
    assert set(decode.union) <= set(merged.union)
    assert sizes["prefill_union"] == len(prefill.union)
    assert sizes["decode_union"] == len(decode.union)


def _preflight(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> list[tuple[int, int]]:
    expanded: list[tuple[int, int]] = []
    monkeypatch.setattr(sys, "argv", ["t2_prefill_decode.py", *arguments])
    monkeypatch.setattr(runner, "expand_devices", lambda world, _devices, **_kwargs:
                        expanded.append((int(world), int(_kwargs["ranks_per_device"]))))
    return expanded


def _write_header_only_checkpoint(path: Path, names: tuple[str, ...]) -> None:
    """Write tiny valid safetensors entries; the runner must inspect headers only."""

    header = {name: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * (i + 1)]}
              for i, name in enumerate(names)}
    encoded = json.dumps(header).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(4 * len(names)))


def test_one_b_default_is_one_rank_and_rejects_8b_fixtures(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"model": "llama3-8b"}))
    expanded = _preflight(monkeypatch, ["--model", "llama3.2-1b", "--fixtures", str(tmp_path),
                                      "--layers", "0", "--decode-tokens", "0"])
    with pytest.raises(SystemExit, match="built for llama3-8b, not llama3.2-1b"):
        runner.main()
    assert expanded == [(1, 1)]


def test_one_b_decode_requires_checkpoint_before_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"model": "llama3.2-1b"}))
    (tmp_path / "layer0_weights.npz").touch()
    _preflight(monkeypatch, ["--model", "llama3.2-1b", "--fixtures", str(tmp_path), "--layers", "0"])
    with pytest.raises(SystemExit, match="Decode requires --model-dir"):
        runner.main()


def test_one_b_rejects_wrong_checkpoint_geometry_before_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"model": "llama3.2-1b"}))
    (tmp_path / "layer0_weights.npz").touch()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 8, "rope_theta": 500000.0,
    }))
    _preflight(monkeypatch, ["--model", "llama3.2-1b", "--fixtures", str(tmp_path),
                             "--model-dir", str(checkpoint), "--layers", "0"])
    with pytest.raises(SystemExit, match="expected llama3.2-1b"):
        runner.main()


def test_one_b_cli_wires_full_chain_and_v8_without_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "manifest.json").write_text(json.dumps({
        "model": "llama3.2-1b", "poly_top_k": [{"token_id": 12366}],
    }))
    for layer in range(16):
        (fixtures / f"layer{layer}_weights.npz").touch()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 16,
        "num_attention_heads": 32, "num_key_value_heads": 8, "rope_theta": 500000.0,
    }))
    _write_header_only_checkpoint(checkpoint, (
        "model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"))
    _preflight(monkeypatch, ["--model", "llama3.2-1b", "--fixtures", str(fixtures),
                             "--model-dir", str(checkpoint), "--kv-limbs", "12",
                             "--kv-value-limbs", "8", "--output", str(tmp_path / "out")])

    class StopBeforeGpu(Exception):
        pass

    seen = {}

    def capture_spawn(preset, _entry, *, kwargs, **_other):
        seen["preset"] = preset
        seen["kwargs"] = kwargs
        raise StopBeforeGpu

    monkeypatch.setattr(runner, "spawn_workers", capture_spawn)
    with pytest.raises(StopBeforeGpu):
        runner.main()
    assert seen["preset"].model_spec.name == "llama3.2-1b"
    assert seen["preset"].ciphers_per_rank == 8
    assert seen["kwargs"]["layers"] == tuple(range(16))
    assert seen["kwargs"]["decode_tokens"] == 3
    assert seen["kwargs"]["reference_tokens"] == (12366,)
    assert (seen["kwargs"]["kv_limbs"], seen["kwargs"]["kv_value_limbs"]) == (12, 8)


def test_one_b_head_only_checkpoint_rejected_before_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "manifest.json").write_text(json.dumps({"model": "llama3.2-1b"}))
    (fixtures / "layer0_weights.npz").touch()
    checkpoint = tmp_path / "head_only"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 16,
        "num_attention_heads": 32, "num_key_value_heads": 8, "rope_theta": 500000.0,
    }))
    _write_header_only_checkpoint(checkpoint, ("lm_head.weight", "model.norm.weight"))
    _preflight(monkeypatch, ["--model", "llama3.2-1b", "--fixtures", str(fixtures),
                             "--model-dir", str(checkpoint), "--layers", "0"])
    with pytest.raises(SystemExit, match="missing client tensors: model.embed_tokens.weight"):
        runner.main()


def test_one_b_head_only_checkpoint_accepts_exact_sparse_embedding_before_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "manifest.json").write_text(json.dumps({
        "model": "llama3.2-1b", "poly_top_k": [{"token_id": 12366}],
    }))
    (fixtures / "layer0_weights.npz").touch()
    checkpoint = tmp_path / "head_only"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 16,
        "num_attention_heads": 32, "num_key_value_heads": 8, "rope_theta": 500000.0,
    }))
    _write_header_only_checkpoint(checkpoint, ("lm_head.weight", "model.norm.weight"))
    rows = tmp_path / "embedding_rows.npz"
    vector = np.arange(2048, dtype=np.float32).reshape(1, 2048)
    np.savez(rows, token_ids=np.array([12366], dtype=np.int64), embeddings=vector)
    _preflight(monkeypatch, ["--model", "llama3.2-1b", "--fixtures", str(fixtures),
                             "--model-dir", str(checkpoint), "--client-embedding-rows", str(rows),
                             "--layers", "0", "--decode-tokens", "1"])
    assert np.array_equal(runner._load_client_embedding_rows(rows, hidden_dim=2048,
                            vocab_size=128256)[12366], vector[0])

    class StopBeforeGpu(Exception):
        pass

    seen = {}

    def capture_spawn(_preset, _entry, *, kwargs, **_other):
        seen.update(kwargs)
        raise StopBeforeGpu

    monkeypatch.setattr(runner, "spawn_workers", capture_spawn)
    with pytest.raises(StopBeforeGpu):
        runner.main()
    assert seen["client_embedding_rows"] == str(rows)
    assert seen["model_dir"] == str(checkpoint)


@pytest.mark.parametrize("token_ids,embeddings,error", [
    (np.array([12366, 12366]), np.zeros((2, 2048), dtype=np.float32), "unique"),
    (np.array([128256]), np.zeros((1, 2048), dtype=np.float32), "unique and in"),
    (np.array([12366]), np.zeros((1, 1024), dtype=np.float32), "shape"),
    (np.array([12366]), np.full((1, 2048), np.nan, dtype=np.float32), "finite"),
])
def test_sparse_embedding_rows_reject_invalid_data(tmp_path, token_ids, embeddings, error) -> None:
    path = tmp_path / "invalid.npz"
    np.savez(path, token_ids=token_ids, embeddings=embeddings)
    with pytest.raises(ValueError, match=error):
        runner._load_client_embedding_rows(path, hidden_dim=2048, vocab_size=128256)
