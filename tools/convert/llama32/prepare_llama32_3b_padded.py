#!/usr/bin/env python3
"""Export Llama-3.2-3B as an equivalent, QuaRot-fused 4096-wide checkpoint.

The CKKS FM layout requires a power-of-two physical hidden width and one query
head pair per FM ciphertext.  Official Llama-3.2-3B has 3072 hidden features,
24 Q heads, 8 KV heads and 128 features/head.  This export pads the residual
width to 4096 and each three-Q-head GQA group to four heads.  The new fourth
Q head contributes nothing because its O-projection columns are zero.

Every RMSNorm epsilon and gamma is multiplied by 3072/4096 and
sqrt(3072/4096), respectively.  Therefore the non-padded coordinates of a
pre-QuaRot padded model are mathematically identical to the official model.
The script then fuses norm gammas and performs the same full QuaRot transforms
as ``quarot_rotate_numpy.py`` using an O(d log d) Walsh-Hadamard transform.
No additional dense 4096 x 4096 rotation matrix is constructed.

The output is sharded by layer, so peak memory does not scale with all 28
layers.  Run ``--self-test`` to verify padding, GQA remapping and QuaRot on a
small complete random model before converting official weights.
For an exact export, the default two-token plaintext verification must have
the same top token and a maximum logit difference of at most 0.05 (float32)
or 0.5 (float16); ``--max-abs-logit-diff`` can tighten the gate.  A failed
verification leaves its report but removes the checkpoint index.

Examples::

    python tools/convert/llama32/prepare_llama32_3b_padded.py --self-test
    python tools/convert/llama32/prepare_llama32_3b_padded.py \
        assets/weights/Llama-3.2-3B \
        assets/weights/Llama-3.2-3B-padded-quarot --verify-tokens 2

The default leaves embedding means intact for exact equivalence.  ``--embed-mean``
matches the legacy 1B QuaRot preparation but changes the original model.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quarot_rotate_numpy import (
    RawSafetensors,
    fwht_last_dim,
    head_hadamard_input,
    head_hadamard_output,
)


@dataclass(frozen=True)
class PaddingGeometry:
    hidden: int
    physical_hidden: int
    q_heads: int
    physical_q_heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    layers: int
    vocab: int

    @property
    def rms_scale(self) -> float:
        return math.sqrt(self.hidden / self.physical_hidden)

    @property
    def source_gqa(self) -> int:
        return self.q_heads // self.kv_heads

    @property
    def physical_gqa(self) -> int:
        return self.physical_q_heads // self.kv_heads

    @property
    def q_head_map(self) -> tuple[int, ...]:
        return tuple(
            (head // self.source_gqa) * self.physical_gqa + head % self.source_gqa
            for head in range(self.q_heads)
        )


def geometry_from_config(config: dict, *, strict_3b: bool = True) -> PaddingGeometry:
    hidden = int(config["hidden_size"])
    q_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim", hidden // q_heads))
    physical_hidden = 1 << (hidden - 1).bit_length()
    if hidden != q_heads * head_dim:
        raise ValueError("source hidden_size must equal num_attention_heads * head_dim")
    if not 0 < kv_heads <= q_heads or q_heads % kv_heads:
        raise ValueError("source Q heads must be a positive multiple of KV heads")
    if physical_hidden % head_dim:
        raise ValueError("physical hidden width must be divisible by head_dim")
    physical_q_heads = physical_hidden // head_dim
    if physical_q_heads % kv_heads:
        raise ValueError("padded Q heads must be a multiple of KV heads")
    geometry = PaddingGeometry(
        hidden=hidden,
        physical_hidden=physical_hidden,
        q_heads=q_heads,
        physical_q_heads=physical_q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate=int(config["intermediate_size"]),
        layers=int(config["num_hidden_layers"]),
        vocab=int(config["vocab_size"]),
    )
    if strict_3b:
        expected = (3072, 4096, 24, 32, 8, 128, 8192, 28, 128256)
        actual = (
            geometry.hidden, geometry.physical_hidden, geometry.q_heads,
            geometry.physical_q_heads, geometry.kv_heads, geometry.head_dim,
            geometry.intermediate, geometry.layers, geometry.vocab,
        )
        if actual != expected:
            raise ValueError(f"expected official Llama-3.2-3B geometry {expected}, got {actual}")
        if config.get("model_type") != "llama":
            raise ValueError("expected a Llama model config")
        if not math.isclose(float(config.get("rope_theta", 0.0)), 500000.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Llama-3.2-3B rope_theta must be 500000")
        if not math.isclose(float(config.get("rms_norm_eps", 0.0)), 1e-5, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Llama-3.2-3B rms_norm_eps must be 1e-5")
        scaling = config.get("rope_scaling")
        expected_scaling = {
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192.0,
        }
        if not isinstance(scaling, dict) or str(scaling.get("rope_type", scaling.get("type", ""))).lower() != "llama3":
            raise ValueError("Llama-3.2-3B requires llama3 rope_scaling")
        for field, expected_value in expected_scaling.items():
            if field not in scaling or not math.isclose(float(scaling[field]), expected_value,
                                                        rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"Llama-3.2-3B rope_scaling.{field} must be {expected_value}")
    if len(set(geometry.q_head_map)) != geometry.q_heads:
        raise ValueError("Q head mapping is not one-to-one")
    return geometry


def validate_raw_tensors(reader: RawSafetensors, geometry: PaddingGeometry,
                         config: dict) -> None:
    """Validate every tensor header before creating a large output directory."""
    h = geometry.hidden
    kv = geometry.kv_heads * geometry.head_dim
    expected: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (geometry.vocab, h),
        "model.norm.weight": (h,),
    }
    for index in range(geometry.layers):
        p = f"model.layers.{index}."
        expected.update({
            p + "input_layernorm.weight": (h,),
            p + "post_attention_layernorm.weight": (h,),
            p + "self_attn.q_proj.weight": (h, h),
            p + "self_attn.k_proj.weight": (kv, h),
            p + "self_attn.v_proj.weight": (kv, h),
            p + "self_attn.o_proj.weight": (h, h),
            p + "mlp.gate_proj.weight": (geometry.intermediate, h),
            p + "mlp.up_proj.weight": (geometry.intermediate, h),
            p + "mlp.down_proj.weight": (h, geometry.intermediate),
        })
    available = set(reader.keys())
    if "lm_head.weight" in available:
        expected["lm_head.weight"] = (geometry.vocab, h)
    elif not bool(config.get("tie_word_embeddings", False)):
        raise ValueError("untied source config requires lm_head.weight")
    missing = sorted(set(expected) - available)
    if missing:
        raise ValueError(f"source checkpoint is missing {len(missing)} required tensors: {missing[:8]}")
    for name, shape in expected.items():
        observed = tuple(int(value) for value in reader.info(name)["shape"])
        if observed != shape:
            raise ValueError(f"source tensor {name} shape {observed} != {shape}")


def _signs(size: int, seed: int) -> np.ndarray:
    # Identical PRNG sequence and diagonal-H orientation to
    # quarot_rotate_numpy.random_hadamard_matrix().
    rng = np.random.default_rng(int(seed))
    return (rng.integers(0, 2, size=size) * 2 - 1).astype(np.float64)


def _right_rotate(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """W @ Q, where Q = diag(signs) @ H/sqrt(d)."""
    return fwht_last_dim(np.asarray(values, dtype=np.float64) * signs).astype(np.float32)


def _left_rotate(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Q.T @ W, without constructing Q."""
    return fwht_last_dim((np.asarray(values, dtype=np.float64) * signs[:, None]).T).T.astype(np.float32)


def _pad_input(weight: np.ndarray, geometry: PaddingGeometry, gamma: np.ndarray) -> np.ndarray:
    source = np.asarray(weight, dtype=np.float64)
    if source.shape[1] != geometry.hidden:
        raise ValueError(f"input width {source.shape[1]} != {geometry.hidden}")
    result = np.zeros((source.shape[0], geometry.physical_hidden), dtype=np.float64)
    result[:, :geometry.hidden] = source * gamma[None, :]
    return result


def _pad_q_projection(weight: np.ndarray, geometry: PaddingGeometry, gamma: np.ndarray) -> np.ndarray:
    source = np.asarray(weight, dtype=np.float64)
    if source.shape != (geometry.hidden, geometry.hidden):
        raise ValueError(f"Q weight shape {source.shape} is not {(geometry.hidden, geometry.hidden)}")
    result = np.zeros((geometry.physical_hidden, geometry.physical_hidden), dtype=np.float64)
    for original_head, padded_head in enumerate(geometry.q_head_map):
        old = slice(original_head * geometry.head_dim, (original_head + 1) * geometry.head_dim)
        new = slice(padded_head * geometry.head_dim, (padded_head + 1) * geometry.head_dim)
        result[new, :geometry.hidden] = source[old] * gamma[None, :]
    return result


def _pad_o_projection(weight: np.ndarray, geometry: PaddingGeometry) -> np.ndarray:
    source = np.asarray(weight, dtype=np.float64)
    if source.shape != (geometry.hidden, geometry.hidden):
        raise ValueError(f"O weight shape {source.shape} is not {(geometry.hidden, geometry.hidden)}")
    result = np.zeros((geometry.physical_hidden, geometry.physical_hidden), dtype=np.float64)
    for original_head, padded_head in enumerate(geometry.q_head_map):
        old = slice(original_head * geometry.head_dim, (original_head + 1) * geometry.head_dim)
        new = slice(padded_head * geometry.head_dim, (padded_head + 1) * geometry.head_dim)
        result[:geometry.hidden, new] = source[:, old]
    return result


def _pad_down_projection(weight: np.ndarray, geometry: PaddingGeometry) -> np.ndarray:
    source = np.asarray(weight, dtype=np.float64)
    if source.shape != (geometry.hidden, geometry.intermediate):
        raise ValueError(f"down weight shape {source.shape} is not {(geometry.hidden, geometry.intermediate)}")
    result = np.zeros((geometry.physical_hidden, geometry.intermediate), dtype=np.float64)
    result[:geometry.hidden] = source
    return result


def _embed_or_head(
    weight: np.ndarray,
    geometry: PaddingGeometry,
    signs: np.ndarray,
    *,
    gamma: np.ndarray | None = None,
    centre: bool = False,
    chunk_rows: int = 512,
) -> np.ndarray:
    source = np.asarray(weight, dtype=np.float32)
    if source.shape != (geometry.vocab, geometry.hidden):
        raise ValueError(f"embedding/head shape {source.shape} is not {(geometry.vocab, geometry.hidden)}")
    result = np.empty((geometry.vocab, geometry.physical_hidden), dtype=np.float32)
    for start in range(0, geometry.vocab, chunk_rows):
        stop = min(start + chunk_rows, geometry.vocab)
        block = source[start:stop].astype(np.float64)
        if centre:
            # Mean-centre the *logical* features before adding zero channels.
            block -= block.mean(axis=-1, keepdims=True)
        if gamma is not None:
            block *= gamma[None, :]
        padded = np.zeros((stop - start, geometry.physical_hidden), dtype=np.float64)
        padded[:, :geometry.hidden] = block
        result[start:stop] = _right_rotate(padded, signs)
    return result


def _layer_tensors(reader: RawSafetensors, index: int, geometry: PaddingGeometry,
                   signs: np.ndarray) -> dict[str, np.ndarray]:
    p = f"model.layers.{index}."
    hidden = geometry.hidden
    kv_width = geometry.kv_heads * geometry.head_dim
    gamma_in = reader.get(p + "input_layernorm.weight").astype(np.float64)
    gamma_post = reader.get(p + "post_attention_layernorm.weight").astype(np.float64)
    if gamma_in.shape != (hidden,) or gamma_post.shape != (hidden,):
        raise ValueError(f"layer {index} RMSNorm gamma shape mismatch")
    gamma_in *= geometry.rms_scale
    gamma_post *= geometry.rms_scale
    result: dict[str, np.ndarray] = {
        p + "input_layernorm.weight": np.ones(geometry.physical_hidden, dtype=np.float32),
        p + "post_attention_layernorm.weight": np.ones(geometry.physical_hidden, dtype=np.float32),
    }

    q = _pad_q_projection(reader.get(p + "self_attn.q_proj.weight"), geometry, gamma_in)
    result[p + "self_attn.q_proj.weight"] = _right_rotate(q, signs)
    del q
    for suffix in ("k_proj", "v_proj"):
        weight = reader.get(p + f"self_attn.{suffix}.weight")
        if weight.shape != (kv_width, hidden):
            raise ValueError(f"layer {index} {suffix} shape mismatch: {weight.shape}")
        padded = _pad_input(weight, geometry, gamma_in)
        if suffix == "v_proj":
            padded = head_hadamard_output(padded, head_dim=geometry.head_dim)
        result[p + f"self_attn.{suffix}.weight"] = _right_rotate(padded, signs)
        del padded

    o = _pad_o_projection(reader.get(p + "self_attn.o_proj.weight"), geometry)
    o = head_hadamard_input(o, head_dim=geometry.head_dim)
    result[p + "self_attn.o_proj.weight"] = _left_rotate(o, signs)
    del o

    for suffix in ("gate_proj", "up_proj"):
        weight = reader.get(p + f"mlp.{suffix}.weight")
        if weight.shape != (geometry.intermediate, hidden):
            raise ValueError(f"layer {index} {suffix} shape mismatch: {weight.shape}")
        padded = _pad_input(weight, geometry, gamma_post)
        result[p + f"mlp.{suffix}.weight"] = _right_rotate(padded, signs)
        del padded

    down = _pad_down_projection(reader.get(p + "mlp.down_proj.weight"), geometry)
    result[p + "mlp.down_proj.weight"] = _left_rotate(down, signs)
    return result


def _save_shard(tensors: dict[str, np.ndarray], path: Path, dtype: str) -> int:
    from safetensors.numpy import save_file

    target_dtype = {"float32": np.float32, "float16": np.float16}[dtype]
    casted = {name: np.ascontiguousarray(value.astype(target_dtype, copy=False))
              for name, value in tensors.items()}
    save_file(casted, str(path), metadata={"format": "pt"})
    return sum(value.nbytes for value in casted.values())


def _apply_rope(values: np.ndarray, *, heads: int, head_dim: int,
                theta: float, scaling: dict | None) -> np.ndarray:
    """Standalone HF Llama split-half RoPE, avoiding an EasyFHE import."""
    values = np.asarray(values, dtype=np.float64).reshape(-1, heads, head_dim)
    inv_freq = 1.0 / (float(theta) **
                      (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    if scaling and str(scaling.get("rope_type", scaling.get("type", ""))).lower() == "llama3":
        factor = float(scaling["factor"])
        low = float(scaling["low_freq_factor"])
        high = float(scaling["high_freq_factor"])
        old_context = float(scaling["original_max_position_embeddings"])
        wavelengths = 2.0 * np.pi / inv_freq
        low_limit = old_context / low
        high_limit = old_context / high
        smooth = (old_context / wavelengths - low) / (high - low)
        blended = (1.0 - smooth) * inv_freq / factor + smooth * inv_freq
        medium = (wavelengths >= high_limit) & (wavelengths <= low_limit)
        inv_freq = np.where(medium, blended,
                            np.where(wavelengths > low_limit, inv_freq / factor, inv_freq))
    angles = np.outer(np.arange(values.shape[0], dtype=np.float64), inv_freq)
    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]
    half = head_dim // 2
    out = np.empty_like(values)
    out[:, :, :half] = values[:, :, :half] * cos - values[:, :, half:] * sin
    out[:, :, half:] = values[:, :, half:] * cos + values[:, :, :half] * sin
    return out.reshape(values.shape[0], heads * head_dim)


def _forward_plain(get, config: dict, token_ids: np.ndarray, *, rotated: bool) -> np.ndarray:
    """Small-token plaintext reference using the checkpoint's own GQA shape."""
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim", hidden // heads))
    eps = float(config.get("rms_norm_eps", 1e-5))
    theta = float(config.get("rope_theta", 500000.0))
    scaling = config.get("rope_scaling")
    x = np.asarray(get("model.embed_tokens.weight")[token_ids], dtype=np.float64)
    count = len(token_ids)
    causal = np.tril(np.ones((count, count), dtype=bool))

    def norm(values: np.ndarray, gamma: np.ndarray | None) -> np.ndarray:
        result = values / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + eps)
        return result if gamma is None else result * gamma

    for index in range(int(config["num_hidden_layers"])):
        p = f"model.layers.{index}."
        gamma = None if rotated else get(p + "input_layernorm.weight")
        normalized = norm(x, gamma)
        q = normalized @ get(p + "self_attn.q_proj.weight").T
        k = normalized @ get(p + "self_attn.k_proj.weight").T
        v = normalized @ get(p + "self_attn.v_proj.weight").T
        q = _apply_rope(q, heads=heads, head_dim=head_dim, theta=theta, scaling=scaling)
        k = _apply_rope(k, heads=kv_heads, head_dim=head_dim, theta=theta, scaling=scaling)
        q = q.reshape(count, heads, head_dim)
        k = k.reshape(count, kv_heads, head_dim)
        v = v.reshape(count, kv_heads, head_dim)
        attention = np.empty_like(q)
        ratio = heads // kv_heads
        for head in range(heads):
            scores = q[:, head] @ k[:, head // ratio].T / math.sqrt(head_dim)
            scores = np.where(causal, scores, -np.inf)
            exponents = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities = exponents / exponents.sum(axis=-1, keepdims=True)
            attention[:, head] = probabilities @ v[:, head // ratio]
        x = x + attention.reshape(count, hidden) @ get(p + "self_attn.o_proj.weight").T
        gamma = None if rotated else get(p + "post_attention_layernorm.weight")
        normalized = norm(x, gamma)
        gate = normalized @ get(p + "mlp.gate_proj.weight").T
        up = normalized @ get(p + "mlp.up_proj.weight").T
        silu = gate / (1.0 + np.exp(-gate))
        x = x + (silu * up) @ get(p + "mlp.down_proj.weight").T
    gamma = None if rotated else get("model.norm.weight")
    return norm(x, gamma)[-1] @ get("lm_head.weight").T


def _verify(source: Path, target: Path, source_config: dict, target_config: dict,
            *, count: int, centred: bool) -> dict:
    rng = np.random.default_rng(1)
    token_ids = rng.integers(0, int(source_config["vocab_size"]), size=count)
    with_source = RawSafetensors(source)
    with_target = RawSafetensors(target)
    has_head = "lm_head.weight" in with_source.keys()

    def original(name: str) -> np.ndarray:
        if name == "lm_head.weight" and not has_head:
            name = "model.embed_tokens.weight"
        return with_source.get(name).astype(np.float64)

    def prepared(name: str) -> np.ndarray:
        return with_target.get(name).astype(np.float64)

    ref = _forward_plain(original, source_config, token_ids, rotated=False)
    got = _forward_plain(prepared, target_config, token_ids, rotated=True)
    difference = np.abs(ref - got)
    return {
        "token_ids": token_ids.tolist(),
        "max_abs_logit_diff": float(difference.max()),
        "rms_logit_diff": float(np.sqrt(np.mean(difference ** 2))),
        "max_abs_reference_logit": float(np.abs(ref).max()),
        "argmax_equal": bool(np.argmax(ref) == np.argmax(got)),
        "exact_model_target": not centred,
        "note": "Embedding mean centring changes the source model." if centred
                else "Difference comes from weight/output floating-point rounding only.",
    }


def convert(source: Path, target: Path, *, seed: int = 0, dtype: str = "float32",
            embed_mean: bool = False, verify_tokens: int = 0,
            strict_3b: bool = True,
            max_abs_logit_diff: float | None = None) -> dict:
    source = Path(source).resolve()
    target = Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("source and output directories must be independent")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"output directory is not empty: {target}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    geometry = geometry_from_config(config, strict_3b=strict_3b)
    if int(verify_tokens) < 0:
        raise ValueError("verify_tokens must be nonnegative")
    if dtype not in ("float32", "float16"):
        raise ValueError(f"unsupported output dtype {dtype!r}")
    tolerance = (0.05 if dtype == "float32" else 0.5) if max_abs_logit_diff is None else float(max_abs_logit_diff)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("max_abs_logit_diff must be finite and positive")
    reader = RawSafetensors(source)
    validate_raw_tensors(reader, geometry, config)
    signs = _signs(geometry.physical_hidden, seed)
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    final_gamma = reader.get("model.norm.weight").astype(np.float64)
    if final_gamma.shape != (geometry.hidden,):
        raise ValueError("final RMSNorm gamma shape mismatch")
    final_gamma *= geometry.rms_scale
    original_embed = reader.get("model.embed_tokens.weight")
    head_key = "lm_head.weight" if "lm_head.weight" in reader.keys() else "model.embed_tokens.weight"
    original_head = original_embed if head_key == "model.embed_tokens.weight" else reader.get(head_key)
    core = {
        "model.embed_tokens.weight": _embed_or_head(original_embed, geometry, signs, centre=embed_mean),
        "lm_head.weight": _embed_or_head(original_head, geometry, signs, gamma=final_gamma),
        "model.norm.weight": np.ones(geometry.physical_hidden, dtype=np.float32),
    }
    del original_embed, original_head
    weight_map = {name: "model-core.safetensors" for name in core}
    total_size = _save_shard(core, target / "model-core.safetensors", dtype)
    del core
    print(f"core: prepared ({time.perf_counter() - started:.1f}s)", flush=True)

    for index in range(geometry.layers):
        tensors = _layer_tensors(reader, index, geometry, signs)
        filename = f"model-layer-{index:02d}.safetensors"
        total_size += _save_shard(tensors, target / filename, dtype)
        weight_map.update({name: filename for name in tensors})
        del tensors
        print(f"layer {index}: prepared ({time.perf_counter() - started:.1f}s)", flush=True)

    prepared_config = dict(config)
    prepared_config.update({
        "hidden_size": geometry.physical_hidden,
        "num_attention_heads": geometry.physical_q_heads,
        "rms_norm_eps": float(config.get("rms_norm_eps", 1e-5)) *
                        geometry.hidden / geometry.physical_hidden,
        "tie_word_embeddings": False,
        "torch_dtype": dtype,
    })
    prepared_config["odin_padding"] = {
        "format": "llama32_3b_4096_q32_exact_v1",
        "source_hidden_dim": geometry.hidden,
        "source_query_heads": geometry.q_heads,
        "source_gqa_ratio": geometry.source_gqa,
        "physical_hidden_dim": geometry.physical_hidden,
        "physical_query_heads": geometry.physical_q_heads,
        "physical_gqa_ratio": geometry.physical_gqa,
        "query_head_map": list(geometry.q_head_map),
        "inactive_query_heads": sorted(set(range(geometry.physical_q_heads)) - set(geometry.q_head_map)),
        "rms_mean_square_factor": geometry.hidden / geometry.physical_hidden,
        "source_model": "meta-llama/Llama-3.2-3B" if strict_3b else str(source),
    }
    prepared_config["quarot"] = {
        "format": "numpy_quarot_full_fuse_v1",
        "seed": int(seed),
        "rotate_mode": "hadamard",
        "fuse_layer_norms": True,
        "embed_mean_centred": bool(embed_mean),
        "ov_head_hadamard": True,
        "down_had_fused": "identity",
        "source": str(source),
    }
    (target / "config.json").write_text(json.dumps(prepared_config, indent=2) + "\n", encoding="utf-8")
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "LICENSE", "USE_POLICY.md"):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    # Write the index last.  Until it exists, a partly written output is not a
    # valid checkpoint, and the converter refuses a second run into it.
    index_path = target / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2) + "\n",
        encoding="utf-8",
    )
    report: dict = {
        "source": str(source), "output": str(target),
        "physical_hidden": geometry.physical_hidden,
        "physical_query_heads": geometry.physical_q_heads,
        "layers": geometry.layers, "tensors": len(weight_map), "weight_bytes": total_size,
        "seconds": round(time.perf_counter() - started, 2),
    }
    if verify_tokens > 0:
        try:
            verification = _verify(source, target, config, prepared_config,
                                   count=verify_tokens, centred=embed_mean)
            finite = all(math.isfinite(float(verification[key])) for key in (
                "max_abs_logit_diff", "rms_logit_diff", "max_abs_reference_logit"
            ))
            passed = (finite and bool(verification["argmax_equal"])
                      and float(verification["max_abs_logit_diff"]) <= tolerance)
            verification["max_abs_logit_diff_allowed"] = tolerance
            verification["passed"] = passed
            report["verify"] = verification
        except Exception as exc:
            report["verify"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
            (target / "conversion_report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")
            index_path.unlink(missing_ok=True)
            raise RuntimeError(f"plaintext verification failed; see {target / 'conversion_report.json'}") from exc
    (target / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if verify_tokens > 0 and not report["verify"]["passed"]:
        index_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"prepared checkpoint failed plaintext verification; see {target / 'conversion_report.json'}: "
            f"{report['verify']}"
        )
    return report


def self_test() -> dict:
    """Exercise full GQA, RMSNorm, RoPE, FFN and output equivalence on a toy model."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(7)
    config = {
        "model_type": "llama", "hidden_size": 12, "intermediate_size": 32,
        "num_hidden_layers": 2, "num_attention_heads": 6, "num_key_value_heads": 2,
        "head_dim": 2, "vocab_size": 32, "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
    }
    with tempfile.TemporaryDirectory(prefix="llama32_3b_padding_test_") as tmp:
        source = Path(tmp) / "source"
        target = Path(tmp) / "target"
        source.mkdir()
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        tensors = {
            "model.embed_tokens.weight": rng.normal(0, 0.1, (32, 12)).astype(np.float32),
            "model.norm.weight": rng.uniform(0.8, 1.2, 12).astype(np.float32),
        }
        for index in range(2):
            p = f"model.layers.{index}."
            tensors.update({
                p + "input_layernorm.weight": rng.uniform(0.8, 1.2, 12).astype(np.float32),
                p + "post_attention_layernorm.weight": rng.uniform(0.8, 1.2, 12).astype(np.float32),
                p + "self_attn.q_proj.weight": rng.normal(0, 0.05, (12, 12)).astype(np.float32),
                p + "self_attn.k_proj.weight": rng.normal(0, 0.05, (4, 12)).astype(np.float32),
                p + "self_attn.v_proj.weight": rng.normal(0, 0.05, (4, 12)).astype(np.float32),
                p + "self_attn.o_proj.weight": rng.normal(0, 0.05, (12, 12)).astype(np.float32),
                p + "mlp.gate_proj.weight": rng.normal(0, 0.05, (32, 12)).astype(np.float32),
                p + "mlp.up_proj.weight": rng.normal(0, 0.05, (32, 12)).astype(np.float32),
                p + "mlp.down_proj.weight": rng.normal(0, 0.05, (12, 32)).astype(np.float32),
            })
        save_file(tensors, str(source / "model.safetensors"))
        report = convert(source, target, strict_3b=False, verify_tokens=5,
                         max_abs_logit_diff=1e-5)
        error = report["verify"]["max_abs_logit_diff"]
        if error > 1e-5 or not report["verify"]["argmax_equal"]:
            raise AssertionError(f"padding + QuaRot toy equivalence failed: {report['verify']}")
        # A deliberately over-strict threshold must fail closed, but retain
        # diagnostics and remove the checkpoint index.
        failed_target = Path(tmp) / "failed_target"
        try:
            convert(source, failed_target, strict_3b=False, verify_tokens=5,
                    max_abs_logit_diff=1e-12)
        except RuntimeError:
            assert (failed_target / "conversion_report.json").is_file()
            assert not (failed_target / "model.safetensors.index.json").exists()
        else:
            raise AssertionError("an over-strict verification gate unexpectedly passed")
        return {**report["verify"], "negative_gate_test": "passed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--embed-mean", action="store_true",
                        help="mean-centre original embedding features; changes the official model")
    parser.add_argument("--verify-tokens", type=int, default=2,
                        help="compare exact source vs prepared model on N fixed token ids (0 disables)")
    parser.add_argument("--max-abs-logit-diff", type=float, default=None,
                        help="maximum allowed absolute logit error during verification; default 0.05 for float32 "
                             "or 0.5 for float16; argmax must also match")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps({"self_test": self_test()}, indent=2))
        return 0
    if args.source is None or args.output is None:
        parser.error("source and output paths are required unless --self-test is used")
    report = convert(args.source, args.output, seed=args.seed, dtype=args.dtype,
                     embed_mean=args.embed_mean, verify_tokens=args.verify_tokens,
                     max_abs_logit_diff=args.max_abs_logit_diff)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
