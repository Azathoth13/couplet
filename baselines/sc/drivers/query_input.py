"""CPU-only input checks for the Sylph same-token Decode comparison.

The upstream Sylph chain adds a public signed Hadamard/permutation to the
QuaRot residual basis.  The checkpoint embedding must receive that same map
before it can enter the encrypted Decode layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def forward_sylph_residual(hidden, residual_signs, residual_permutation):
    """Inverse of terminal_logits.inverse_sylph_residual."""

    from sylph_easy.tools import terminal_logits as tl

    x = np.asarray(hidden, dtype=np.float64)
    signs = np.asarray(residual_signs)
    perm = np.asarray(residual_permutation)
    if (x.ndim != 2 or signs.shape != (x.shape[1],)
            or perm.shape != signs.shape or not np.isin(signs, (-1, 1)).all()
            or perm.dtype.kind not in "iu"
            or not np.array_equal(np.sort(perm), np.arange(len(perm)))):
        raise ValueError("invalid Sylph residual rotation or hidden dimensions")
    return tl._fwht(x * signs)[..., perm]


def embedding_rows(checkpoint: str | Path, token_ids):
    """Read only selected embedding rows from a sharded or single safetensors file."""

    from sylph_easy.tools import terminal_logits as tl

    checkpoint = Path(checkpoint).resolve()
    name = "model.embed_tokens.weight"
    index = checkpoint / "model.safetensors.index.json"
    shard_name = json.loads(index.read_text())["weight_map"][name] if index.exists() else "model.safetensors"
    shard = Path(shard_name)
    if shard.is_absolute() or ".." in shard.parts or shard.suffix != ".safetensors":
        raise ValueError("invalid embedding shard path")
    stored = tl.read_safetensors(checkpoint / shard).get(name)
    if stored is None:
        raise ValueError(f"missing {name} in {checkpoint / shard}")
    array, dtype = stored
    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    if np.any(ids < 0) or np.any(ids >= array.shape[0]):
        raise ValueError("token ID outside embedding vocabulary")
    rows = np.asarray(array[ids])
    if dtype == "BF16":
        rows = (rows.astype(np.uint32) << 16).view(np.float32)
    return np.asarray(rows, dtype=np.float64), str((checkpoint / shard).resolve())


def prepare_query(manifest, layer0_weights, layer0_plain_inputs, *, token_id: int, body_ids_path,
                  expected_prefix_ids=(128000, 13)):
    """Return the new token in Sylph's basis after checking all 128 context rows."""

    if "source_checkpoint" not in manifest:
        raise ValueError("chain manifest has no source_checkpoint")
    body_ids = np.asarray(json.loads(Path(body_ids_path).read_text())["input_ids"], dtype=np.int64)
    context = np.asarray(layer0_plain_inputs, dtype=np.float64)
    if body_ids.shape != (128,) or context.shape[0] != 128:
        raise ValueError("expected the same 128-token body on both sides")
    signs = layer0_weights["residual_signs"]
    perm = layer0_weights["residual_permutation"]
    rows, shard = embedding_rows(manifest["source_checkpoint"], np.r_[body_ids, int(token_id)])
    rotated = forward_sylph_residual(rows, signs, perm)
    context_error = float(np.max(np.abs(rotated[:-1] - context)))
    last_error = float(np.max(np.abs(rotated[-2] - context[-1])))
    if not np.isfinite(context_error) or context_error > 1e-3:
        raise ValueError(f"SC chain body differs from checkpoint/Paris128: max_abs={context_error:.6g}")
    prefix_ids = manifest.get("prefix_token_ids")
    if prefix_ids is None:
        raise ValueError("chain manifest has no prefix_token_ids; cannot verify the public Prefix")
    if list(prefix_ids) != list(expected_prefix_ids):
        raise ValueError(f"SC public prefix IDs {prefix_ids} differ from {list(expected_prefix_ids)}")
    active_prefix_tokens = int(np.asarray(layer0_weights["public_keys"]).shape[0])
    if active_prefix_tokens != len(expected_prefix_ids):
        raise ValueError(f"active prefix has {active_prefix_tokens} tokens; expected {len(expected_prefix_ids)}")
    return rotated[-1], {
        "token_id": int(token_id),
        "body_tokens": len(body_ids),
        "body_match_max_abs": context_error,
        "last_body_token_id": int(body_ids[-1]),
        "last_body_match_max_abs": last_error,
        "checkpoint": str(Path(manifest["source_checkpoint"]).resolve()),
        "embedding_shard": shard,
        "public_prefix_ids": prefix_ids,
        "public_prefix_ids_verified": True,
        "active_prefix_tokens": active_prefix_tokens,
    }
