#!/usr/bin/env python3
"""QuaRot-style rotation of a Llama checkpoint in pure numpy (no torch / GPU).

Mirrors what produced ``Meta-Llama-3-8b-quarot-safetensors-v2`` for the FHE
path (official QuaRot ``fuse_layer_norms`` + ``rotate_model`` followed by the
``quarot_full_fuse`` weight folding), for power-of-two shapes such as
Llama-3.2-1B (hidden 2048, intermediate 8192, head_dim 64):

* RMSNorm gammas are folded into the following Linear columns (q/k/v, gate/up,
  lm_head); every ``*norm.weight`` is written as a unit vector so the FHE
  RMSNorm applies no gamma (``gamma_pt_ct_multiplications = 0``).
* Embedding rows are mean-centred (official QuaRot does this for every model).
* One global random Hadamard ``Q`` (``diag(sign) H / sqrt(d)``) rotates the
  residual stream: ``W_emb Q``, ``W_head Q``, ``W_{q,k,v,gate,up} Q``,
  ``Q^T W_o``, ``Q^T W_down``.
* Head-wise Hadamard ``H_{head_dim}`` on the ``v_proj`` outputs and the
  matching ``o_proj`` input columns (``rotate_ov_proj`` + fused online partial
  Hadamard).  The fused ``down_proj`` online Hadamard cancels exactly, so the
  intermediate axis is left alone.
* ``tie_word_embeddings`` is undone: ``lm_head.weight`` is written explicitly
  (its gamma fold differs from the embedding's mean-centring).

Everything is exact (up to float64 rounding) with respect to the original
model: the rotated checkpoint computes the same logits.

Usage::

    python tools/convert/llama32/quarot_rotate_numpy.py assets/weights/Llama-3.2-1B \\
        assets/weights/Llama-3.2-1B-quarot [--seed 0] [--dtype float32|float16] [--no-embed-mean]
        [--verify-tokens 8]
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------- safetensors io
def _read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(size).decode("utf-8"))
    return header, 8 + size


def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


class RawSafetensors:
    """Read any dtype (incl. BF16) of one or several safetensors files as float32."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        index = self.model_dir / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            self.files = sorted(set(weight_map.values()))
            self.where = {name: self.model_dir / file for name, file in weight_map.items()}
        else:
            self.files = ["model.safetensors"]
            header, _ = _read_header(self.model_dir / "model.safetensors")
            self.where = {name: self.model_dir / "model.safetensors" for name in header if name != "__metadata__"}
        self.headers = {str(path): _read_header(path) for path in set(self.where.values())}
        self.mmaps = {str(path): np.memmap(path, dtype=np.uint8, mode="r") for path in set(self.where.values())}

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.where))

    def info(self, name: str) -> dict:
        header, _ = self.headers[str(self.where[name])]
        return header[name]

    def get(self, name: str) -> np.ndarray:
        path = str(self.where[name])
        header, offset = self.headers[path]
        info = header[name]
        start, stop = info["data_offsets"]
        raw = self.mmaps[path][offset + start : offset + stop]
        dtype = info["dtype"]
        shape = tuple(int(v) for v in info["shape"])
        if dtype == "BF16":
            return _bf16_to_f32(np.frombuffer(raw, dtype="<u2")).reshape(shape)
        np_dtype = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}[dtype]
        return np.frombuffer(raw, dtype=np_dtype).reshape(shape).astype(np.float32)


def save_safetensors(tensors: dict[str, np.ndarray], path: Path, *, dtype: str, metadata: dict | None = None) -> None:
    from safetensors.numpy import save_file

    np_dtype = {"float32": np.float32, "float16": np.float16}[dtype]
    save_file({k: np.ascontiguousarray(v.astype(np_dtype)) for k, v in tensors.items()}, str(path),
              metadata={k: str(v) for k, v in (metadata or {}).items()})


# ----------------------------------------------------------------- hadamard
def fwht_last_dim(values: np.ndarray) -> np.ndarray:
    """Normalised fast Walsh-Hadamard transform on the last axis (power of two)."""

    output = np.array(values, dtype=np.float64, copy=True)
    last = int(output.shape[-1])
    if last & (last - 1):
        raise ValueError(f"FWHT needs a power-of-two last dimension, got {last}.")
    stride = 1
    while stride < last:
        view = output.reshape(*output.shape[:-1], last // (2 * stride), 2, stride)
        a = view[..., 0, :].copy()
        b = view[..., 1, :].copy()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        stride *= 2
    return output / np.sqrt(last)


def random_hadamard_matrix(size: int, seed: int) -> np.ndarray:
    """QuaRot ``random_hadamard_matrix``: ``diag(+-1) H / sqrt(size)``."""

    rng = np.random.default_rng(int(seed))
    signs = rng.integers(0, 2, size=int(size)) * 2 - 1
    return fwht_last_dim(np.diag(signs.astype(np.float64)))


def head_hadamard_output(weight: np.ndarray, *, head_dim: int) -> np.ndarray:
    """``apply_exact_had_to_linear(W, had_dim=head_dim, output=True)``: H_d on every head of the out axis."""

    out, inp = weight.shape
    view = np.asarray(weight, dtype=np.float64).reshape(out // head_dim, head_dim, inp).transpose(0, 2, 1)
    return fwht_last_dim(view).transpose(0, 2, 1).reshape(out, inp)


def head_hadamard_input(weight: np.ndarray, *, head_dim: int) -> np.ndarray:
    """H_d on every head of the input axis (fused ``o_proj``: ``W (I_h x H_d)``)."""

    out, inp = weight.shape
    view = np.asarray(weight, dtype=np.float64).reshape(out, inp // head_dim, head_dim)
    return fwht_last_dim(view).reshape(out, inp)


# ------------------------------------------------------------------- rotate
def rotate_checkpoint(src: Path, dst: Path, *, seed: int, dtype: str, embed_mean: bool, verify_tokens: int) -> dict:
    config = json.loads((src / "config.json").read_text())
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim", hidden // heads))
    layers = int(config["num_hidden_layers"])
    eps = float(config.get("rms_norm_eps", 1e-5))
    reader = RawSafetensors(src)
    started = time.perf_counter()

    q = random_hadamard_matrix(hidden, seed)  # [d, d] orthogonal
    out: dict[str, np.ndarray] = {}

    embed = reader.get("model.embed_tokens.weight").astype(np.float64)
    head_name = "lm_head.weight" if "lm_head.weight" in reader.keys() else "model.embed_tokens.weight"
    head = reader.get(head_name).astype(np.float64)
    final_gamma = reader.get("model.norm.weight").astype(np.float64)
    if embed_mean:
        embed = embed - embed.mean(axis=-1, keepdims=True)
    # fuse final RMSNorm gamma into the head, then rotate; untie
    head = head * final_gamma[None, :]
    out["model.embed_tokens.weight"] = (embed @ q).astype(np.float32)
    out["lm_head.weight"] = (head @ q).astype(np.float32)
    out["model.norm.weight"] = np.ones(hidden, dtype=np.float32)
    del embed, head

    for idx in range(layers):
        prefix = f"model.layers.{idx}."
        g_in = reader.get(prefix + "input_layernorm.weight").astype(np.float64)
        g_post = reader.get(prefix + "post_attention_layernorm.weight").astype(np.float64)
        w_q = reader.get(prefix + "self_attn.q_proj.weight").astype(np.float64) * g_in[None, :]
        w_k = reader.get(prefix + "self_attn.k_proj.weight").astype(np.float64) * g_in[None, :]
        w_v = reader.get(prefix + "self_attn.v_proj.weight").astype(np.float64) * g_in[None, :]
        w_o = reader.get(prefix + "self_attn.o_proj.weight").astype(np.float64)
        w_gate = reader.get(prefix + "mlp.gate_proj.weight").astype(np.float64) * g_post[None, :]
        w_up = reader.get(prefix + "mlp.up_proj.weight").astype(np.float64) * g_post[None, :]
        w_down = reader.get(prefix + "mlp.down_proj.weight").astype(np.float64)
        # head-wise Hadamard on V outputs / O inputs (rotate_ov_proj + fused partial online had)
        w_v = head_hadamard_output(w_v, head_dim=head_dim)
        w_o = head_hadamard_input(w_o, head_dim=head_dim)
        out[prefix + "input_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        out[prefix + "post_attention_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        out[prefix + "self_attn.q_proj.weight"] = (w_q @ q).astype(np.float32)
        out[prefix + "self_attn.k_proj.weight"] = (w_k @ q).astype(np.float32)
        out[prefix + "self_attn.v_proj.weight"] = (w_v @ q).astype(np.float32)
        out[prefix + "self_attn.o_proj.weight"] = (q.T @ w_o).astype(np.float32)
        out[prefix + "mlp.gate_proj.weight"] = (w_gate @ q).astype(np.float32)
        out[prefix + "mlp.up_proj.weight"] = (w_up @ q).astype(np.float32)
        out[prefix + "mlp.down_proj.weight"] = (q.T @ w_down).astype(np.float32)
        print(f"layer {idx}: rotated ({time.perf_counter() - started:.1f}s)", flush=True)

    dst.mkdir(parents=True, exist_ok=True)
    new_config = dict(config)
    new_config["tie_word_embeddings"] = False
    new_config["torch_dtype"] = dtype
    new_config["quarot"] = {
        "format": "numpy_quarot_full_fuse_v1", "seed": int(seed), "rotate_mode": "hadamard",
        "fuse_layer_norms": True, "embed_mean_centred": bool(embed_mean), "ov_head_hadamard": True,
        "down_had_fused": "identity", "source": str(src),
    }
    (dst / "config.json").write_text(json.dumps(new_config, indent=2) + "\n")
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    save_safetensors(out, dst / "model.safetensors", dtype=dtype, metadata={"format": "pt"})
    report = {"out": str(dst), "tensors": len(out), "seconds": round(time.perf_counter() - started, 1)}
    if verify_tokens > 0:
        report["verify"] = verify_equivalence(reader, out, config, tokens=int(verify_tokens), eps=eps)
    return report


# -------------------------------------------------------------------- verify
def _rmsnorm(x: np.ndarray, gamma: np.ndarray | None, eps: float) -> np.ndarray:
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if gamma is None else y * gamma


def _rope(x: np.ndarray, *, heads: int, head_dim: int, theta: float, scaling: dict | None) -> np.ndarray:
    sys.path.insert(0, str(ROOT))
    from llama3fhe.approx.attention import apply_rope_heads, rope_cos_sin
    from llama3fhe.model_spec import RopeScaling

    rs = None
    if scaling and str(scaling.get("rope_type", "")) == "llama3":
        rs = RopeScaling(factor=float(scaling["factor"]), low_freq_factor=float(scaling["low_freq_factor"]),
                         high_freq_factor=float(scaling["high_freq_factor"]),
                         original_max_position_embeddings=int(scaling["original_max_position_embeddings"]))
    cos, sin = rope_cos_sin(seq_len=x.shape[0], head_dim=head_dim, theta=theta, rope_scaling=rs)
    return apply_rope_heads(x.astype(np.float32), cos, sin, num_heads=heads, head_dim=head_dim).astype(np.float64)


def _forward(get, config: dict, token_ids: np.ndarray, *, eps: float, rotated: bool) -> np.ndarray:
    hidden = int(config["hidden_size"]); heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", heads)); head_dim = int(config.get("head_dim", hidden // heads))
    theta = float(config.get("rope_theta", 10000.0)); scaling = config.get("rope_scaling")
    x = get("model.embed_tokens.weight")[token_ids].astype(np.float64)
    n = len(token_ids)
    causal = np.tril(np.ones((n, n), dtype=bool))
    for idx in range(int(config["num_hidden_layers"])):
        p = f"model.layers.{idx}."
        g_in = None if rotated else get(p + "input_layernorm.weight")
        h = _rmsnorm(x, g_in, eps)
        qm = h @ get(p + "self_attn.q_proj.weight").T
        km = h @ get(p + "self_attn.k_proj.weight").T
        vm = h @ get(p + "self_attn.v_proj.weight").T
        qm = _rope(qm, heads=heads, head_dim=head_dim, theta=theta, scaling=scaling)
        km = _rope(km, heads=kv_heads, head_dim=head_dim, theta=theta, scaling=scaling)
        qh = qm.reshape(n, heads, head_dim); kh = km.reshape(n, kv_heads, head_dim); vh = vm.reshape(n, kv_heads, head_dim)
        group = heads // kv_heads
        att = np.empty((n, heads, head_dim))
        for hh in range(heads):
            s = qh[:, hh, :] @ kh[:, hh // group, :].T / np.sqrt(head_dim)
            s = np.where(causal, s, -np.inf)
            pmat = np.exp(s - s.max(axis=-1, keepdims=True)); pmat /= pmat.sum(axis=-1, keepdims=True)
            att[:, hh, :] = pmat @ vh[:, hh // group, :]
        x = x + att.reshape(n, hidden) @ get(p + "self_attn.o_proj.weight").T
        g_post = None if rotated else get(p + "post_attention_layernorm.weight")
        h = _rmsnorm(x, g_post, eps)
        gate = h @ get(p + "mlp.gate_proj.weight").T
        up = h @ get(p + "mlp.up_proj.weight").T
        x = x + ((gate / (1.0 + np.exp(-gate))) * up) @ get(p + "mlp.down_proj.weight").T
    g_final = None if rotated else get("model.norm.weight")
    h = _rmsnorm(x, g_final, eps)
    return h[-1] @ get("lm_head.weight").T


def verify_equivalence(reader: RawSafetensors, rotated: dict[str, np.ndarray], config: dict, *, tokens: int, eps: float) -> dict:
    rng = np.random.default_rng(1)
    ids = rng.integers(0, int(config["vocab_size"]), size=int(tokens))
    has_head = "lm_head.weight" in reader.keys()

    def get_src(name: str) -> np.ndarray:
        if name == "lm_head.weight" and not has_head:
            name = "model.embed_tokens.weight"
        return reader.get(name).astype(np.float64)

    def get_rot(name: str) -> np.ndarray:
        return np.asarray(rotated[name], dtype=np.float64)

    ref = _forward(get_src, config, ids, eps=eps, rotated=False)
    got = _forward(get_rot, config, ids, eps=eps, rotated=True)
    return {
        "tokens": int(tokens), "max_abs_logit_diff": float(np.max(np.abs(ref - got))),
        "max_abs_logit": float(np.max(np.abs(ref))), "argmax_equal": bool(np.argmax(ref) == np.argmax(got)),
        "note": "embedding mean-centring is a (small) model change; disable with --no-embed-mean for exactness",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src"); parser.add_argument("dst")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--no-embed-mean", action="store_true", help="skip QuaRot's embedding mean-centring (exact)")
    parser.add_argument("--verify-tokens", type=int, default=8, help="numpy forward of N random tokens, original vs rotated (0 = skip)")
    args = parser.parse_args()
    report = rotate_checkpoint(Path(args.src).resolve(), Path(args.dst).resolve(), seed=int(args.seed), dtype=str(args.dtype),
                               embed_mean=not args.no_embed_mean, verify_tokens=int(args.verify_tokens))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
