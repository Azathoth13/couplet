"""Portable, byte-exact 8B client checkpoint subset for linked SC validation.

The 8B linked runner reads only the token embedding, final RMSNorm weight,
LM head, and ``config.json``.  This module copies those three tensor payloads
without numeric conversion into a single safetensors shard.  It deliberately
does not rewrite values or skip the terminal validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_TENSORS = (
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
)
DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}


@dataclass(frozen=True)
class TensorSlice:
    name: str
    source_file: Path
    source_offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as source:
        length_bytes = source.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"truncated safetensors file: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if not 1 <= header_length <= min(path.stat().st_size - 8, 100_000_000):
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(source.read(header_length))
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header object: {path}")
    return header, 8 + header_length


def inspect_terminal_subset(checkpoint: str | Path) -> dict[str, Any]:
    """Validate and describe exactly the files/tensor bytes staging will need."""

    root = Path(checkpoint).expanduser().resolve()
    config_file = root / "config.json"
    config = json.loads(config_file.read_text())
    hidden, vocab = int(config["hidden_size"]), int(config["vocab_size"])
    index_file = root / "model.safetensors.index.json"
    if index_file.is_file():
        weight_map = json.loads(index_file.read_text())["weight_map"]
    else:
        weight_map = {name: "model.safetensors" for name in REQUIRED_TENSORS}
    tensor_slices: list[TensorSlice] = []
    headers: dict[Path, tuple[dict[str, Any], int]] = {}
    for name in REQUIRED_TENSORS:
        shard_name = Path(weight_map[name])
        if shard_name.is_absolute() or ".." in shard_name.parts or shard_name.suffix != ".safetensors":
            raise ValueError(f"invalid shard path for {name}: {shard_name}")
        shard = root / shard_name
        if shard not in headers:
            headers[shard] = _read_header(shard)
        header, data_start = headers[shard]
        item = header.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"{shard} lacks {name}")
        dtype = item.get("dtype")
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported {name} dtype {dtype!r}")
        shape = tuple(int(size) for size in item["shape"])
        if any(size < 0 for size in shape):
            raise ValueError(f"negative {name} tensor shape")
        start, end = (int(value) for value in item["data_offsets"])
        nbytes = math.prod(shape) * DTYPE_BYTES[dtype]
        if start < 0 or end - start != nbytes or data_start + end > shard.stat().st_size:
            raise ValueError(f"invalid {name} tensor extent in {shard}")
        if name in ("model.embed_tokens.weight", "lm_head.weight") and shape != (vocab, hidden):
            raise ValueError(f"{name} shape {shape} != {(vocab, hidden)}")
        tensor_slices.append(TensorSlice(name, shard, data_start + start, nbytes, dtype, shape))
    return {
        "source_checkpoint": str(root),
        "config": str(config_file),
        "source_shards": sorted(str(path) for path in headers),
        "tensor_bytes": sum(item.nbytes for item in tensor_slices),
        "tensors": tensor_slices,
    }


def export_terminal_subset(checkpoint: str | Path, destination: str | Path) -> dict[str, Any]:
    """Write the three exact payloads plus config to a self-contained directory."""

    info = inspect_terminal_subset(checkpoint)
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(info["config"], target / "config.json")
    source_export = Path(info["source_checkpoint"]) / "export_metadata.json"
    if source_export.is_file():
        shutil.copy2(source_export, target / source_export.name)
    entries = {}
    offset = 0
    for item in info["tensors"]:
        entries[item.name] = {
            "dtype": item.dtype,
            "shape": list(item.shape),
            "data_offsets": [offset, offset + item.nbytes],
        }
        offset += item.nbytes
    header = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    output = target / "model.safetensors"
    tensor_hashes = {}
    with output.open("wb") as sink:
        sink.write(struct.pack("<Q", len(header)))
        sink.write(header)
        for item in info["tensors"]:
            digest = hashlib.sha256()
            remaining = item.nbytes
            with item.source_file.open("rb") as source:
                source.seek(item.source_offset)
                while remaining:
                    chunk = source.read(min(8 << 20, remaining))
                    if not chunk:
                        raise EOFError(f"short read of {item.name} from {item.source_file}")
                    sink.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            tensor_hashes[item.name] = digest.hexdigest()
    with output.open("rb") as source:
        subset_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    provenance = {
        "source_checkpoint": info["source_checkpoint"],
        "source_shards": info["source_shards"],
        "source_config_sha256": hashlib.sha256(Path(info["config"]).read_bytes()).hexdigest(),
        "payload_sha256": tensor_hashes,
        "subset_safetensors_sha256": subset_sha256,
    }
    (target / "subset_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


__all__ = ["REQUIRED_TENSORS", "inspect_terminal_subset", "export_terminal_subset"]
