"""Internal execution bridge for the public SC launcher in ``baselines/sc/run.py``.

The existing single-card, model-specific linked runners remain the source of
truth for input checks and terminal validation.  Each rank executes that same
chain; the two distributed adapters shard Prefill output blocks and Decode MV
plaintext groups.  All inter-rank payloads are CKKS ciphertext packets.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import struct
import zipfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
DRIVER_DIR = HERE.parent / "drivers"
DRIVERS = {
    "llama32-1b": DRIVER_DIR / "run_1b.py",
    "llama32-3b": DRIVER_DIR / "run_3b.py",
    "llama3-8b": DRIVER_DIR / "run_prefill_decode_mv_p0_cipher_kv_gpu.py",
}
LLAMA_BASE = DRIVER_DIR



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _npz_array_shape(path: Path, member: str) -> tuple[int, ...]:
    """Read one .npy header without requiring NumPy in the dry-run launcher."""

    with zipfile.ZipFile(path) as archive, archive.open(f"{member}.npy") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"{path} has an invalid {member}.npy header")
        major, _minor = stream.read(2)
        if major == 1:
            length = struct.unpack("<H", stream.read(2))[0]
        elif major in (2, 3):
            length = struct.unpack("<I", stream.read(4))[0]
        else:
            raise ValueError(f"unsupported NPY version {major} in {path}")
        header = ast.literal_eval(stream.read(length).decode("latin1"))
    return tuple(int(value) for value in header["shape"])


def _load_driver(model: str):
    path = DRIVERS[model].resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(LLAMA_BASE))
    spec = importlib.util.spec_from_file_location(f"sc_multigpu_driver_{model.replace('-', '_')}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load SC linked driver {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_preflight(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    root = Path(config["sylph_root"]).expanduser().resolve()
    chain = Path(config["chain"]).expanduser().resolve()
    manifest_file = chain / "chain-manifest.json"
    source_file = root / "sylph_easy" / "layer" / "decoder.py"
    if not source_file.is_file() or not (root / "sylph_easy" / "layer" / "decode_mv.py").is_file():
        raise FileNotFoundError(f"SYLPH_ROOT lacks the SC FHE layer source: {root}")
    if not manifest_file.is_file():
        raise FileNotFoundError(manifest_file)
    if model == "llama32-3b" and "SYLPH_HEAD24_COMPACT" not in (
        root / "sylph_easy" / "chain" / "manifest.py"
    ).read_text():
        raise ValueError("3B compact24 requires the head24 adapted Sylph source")
    body = Path(config["body_ids"]).expanduser().resolve()
    if not body.is_file():
        raise FileNotFoundError(body)
    if model in {"llama32-1b", "llama32-3b"}:
        row = Path(config["query_row"]).expanduser().resolve()
        if not row.is_file():
            raise FileNotFoundError(row)
        hidden = 2048 if model == "llama32-1b" else 4096
        shape = _npz_array_shape(row, "embeddings")
        if shape != (1, hidden):
            raise ValueError(f"{model} query row shape {shape} != {(1, hidden)}")
    manifest = json.loads(manifest_file.read_text())
    terminal = manifest.get("terminal", {})
    if terminal.get("kind") == "quarot_folded_norm":
        head_name = terminal.get("lm_head")
        if not head_name:
            raise ValueError(f"{model} folded terminal has no lm_head path")
        head_path = Path(head_name)
        head_path = head_path if head_path.is_absolute() else chain / head_path
        if not head_path.is_file():
            raise FileNotFoundError(head_path)
    checkpoint = None
    checkpoint_bytes = None
    if model == "llama3-8b":
        from .checkpoint_subset import inspect_terminal_subset

        manifest_source = manifest.get("source_checkpoint")
        terminal_source = terminal.get("checkpoint")
        if "checkpoint" in terminal:
            if not terminal_source:
                raise ValueError("8B terminal.checkpoint is empty")
            if manifest_source:
                def _chain_path(name: str) -> Path:
                    path = Path(name).expanduser()
                    return (path if path.is_absolute() else chain / path).resolve()

                if _chain_path(terminal_source) != _chain_path(manifest_source):
                    raise ValueError("8B terminal.checkpoint differs from manifest.source_checkpoint")
        source = config.get("source_checkpoint") or manifest.get("source_checkpoint")
        if not source:
            raise ValueError("8B linked SC requires manifest.source_checkpoint or a staging override")
        checkpoint = Path(source).expanduser()
        checkpoint = checkpoint if checkpoint.is_absolute() else chain / checkpoint
        checkpoint_info = inspect_terminal_subset(checkpoint)
        checkpoint = Path(checkpoint_info["source_checkpoint"])
        checkpoint_bytes = int(checkpoint_info["tensor_bytes"])
        expected_config_hash = manifest.get("source_config_sha256")
        if expected_config_hash and _sha256(checkpoint / "config.json") != expected_config_hash:
            raise ValueError("8B source checkpoint config hash differs from the SC chain manifest")
    plain_name = config.get("plain_inputs") or manifest.get("plain_inputs")
    if not plain_name:
        matches = [chain / name for name in ("plain_inputs.npz", "plain-inputs.npz")
                   if (chain / name).is_file()]
        if len(matches) != 1:
            raise ValueError(f"SC chain needs exactly one explicit plaintext trajectory: {chain}")
        plain_path = matches[0]
    else:
        plain_path = Path(plain_name)
        plain_path = plain_path if plain_path.is_absolute() else chain / plain_path
    if not plain_path.is_file():
        raise FileNotFoundError(plain_path)
    expected_layers = {"llama32-1b": 16, "llama32-3b": 28,
                       "llama3-8b": 32}[model]
    entries = manifest.get("layers", [])
    if len(entries) != expected_layers or [int(e["layer"]) for e in entries] != list(range(expected_layers)):
        raise ValueError(f"{model} requires a complete {expected_layers}-layer SC chain")
    prefix = int(manifest.get("public_prefix_tokens", len(manifest.get("prefix_token_ids") or [])))
    if prefix != 0:
        raise ValueError(f"{model} SC prefix is {prefix}; required 0")
    for entry in entries:
        cal = Path(entry["calibration"])
        cal = cal if cal.is_absolute() else chain / cal
        weights = cal.parent / "weights.npz"
        if not cal.is_file() or not weights.is_file():
            raise FileNotFoundError(f"SC layer {entry['layer']} missing {cal} or {weights}")
    return {
        "model": model,
        "layers": expected_layers,
        "prefix": prefix,
        "sylph_root": str(root),
        "chain": str(chain),
        "manifest_sha256": _sha256(manifest_file),
        "plain_inputs": str(plain_path.resolve()),
        "plain_inputs_sha256": _sha256(plain_path),
        "source_checkpoint": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_terminal_tensor_bytes": checkpoint_bytes,
        "driver_sha256": _sha256(DRIVERS[model]),
        "source_decoder_sha256": _sha256(source_file),
        "source_attention_sha256": _sha256(root / "sylph_easy" / "layer" / "attention.py"),
        "source_decode_mv_sha256": _sha256(root / "sylph_easy" / "layer" / "decode_mv.py"),
        "prefill_adapter_sha256": _sha256(HERE / "prefill_shard.py"),
        "decode_adapter_sha256": _sha256(HERE / "decode_input_sync.py"),
        "cache_capture_sha256": _sha256(LLAMA_BASE / "run_prefill_decode_mv_p0_cipher_kv.py"),
        "query_row_sha256": _sha256(Path(config["query_row"])) if config["query_row"] else None,
        "body_ids_sha256": _sha256(Path(config["body_ids"])) if config["body_ids"] else None,
    }


def run_rank(runtime, *, config: dict[str, Any]) -> dict[str, Any]:
    """Spawned rank entry.  Import CUDA and Sylph only after GPU pinning."""

    model = config["model"]
    os.environ["SYLPH_ROOT"] = config["sylph_root"]
    os.environ["SYLPH_CHAIN"] = config["chain"]
    os.environ["SYLPH_PLAIN_INPUTS"] = config["plain_inputs"]
    os.environ["SYLPH_PUBLIC_PREFIX"] = "0"
    if model == "llama32-3b":
        os.environ["SYLPH_HEAD24_COMPACT"] = "1"
    if config.get("query_row"):
        os.environ["SC_QUERY_EMBEDDING_ROW"] = config["query_row"]
    os.environ["SC_LINKED_BASE_DIR"] = str(LLAMA_BASE)
    output = Path(config["out"]).resolve() / f"rank{runtime.rank}"
    output.mkdir(parents=True, exist_ok=False)
    stage = Path(config["out"]).resolve() / "_staging" / f"rank{runtime.rank}"
    stage.mkdir(parents=True, exist_ok=False)
    os.environ["SYLPH_SHM"] = str(stage)
    os.environ["SYLPH_LOGS"] = str(output)
    os.environ["SYLPH_LOOKAHEAD"] = str(config["lookahead"])
    os.environ["SYLPH_WARMUP"] = "1"
    sys.path.insert(0, config["sylph_root"])
    sys.path.insert(0, str(Path(config["sylph_root"]) / "sylph_easy" / "tools"))
    sys.path.insert(0, str(HERE))

    import numpy as np
    from sylph_easy.layer import decoder as decoder_module, decode_mv as decode_mv_module, setup
    from sylph_easy.layer.setup import SLOTS

    driver = _load_driver(model)
    base = driver.base if hasattr(driver, "base") else driver
    gpu_driver = getattr(driver, "gpu", driver)
    original_gpu_memory = getattr(gpu_driver, "gpu_memory", None)
    if original_gpu_memory is not None:
        # The W1 GPU-resident linked runner probes physical GPU 0.  NVIDIA-SMI
        # does not remap -i 0 when CUDA_VISIBLE_DEVICES pins a child to GPU N.
        def rank_gpu_memory():
            line = subprocess.check_output(
                ["nvidia-smi", "-i", str(runtime.audit.gpu_uuid),
                 "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                text=True,
            ).strip().splitlines()[0]
            used, total = (int(field.strip()) for field in line.split(","))
            return {"used_mib": used, "total_mib": total, "free_mib": total - used}

        gpu_driver.gpu_memory = rank_gpu_memory
    from .prefill_shard import make_distributed_decoder_layer
    from .sc_decode_mv_pc import make_distributed_projector
    from .decode_input_sync import make_distributed_decode_layer

    original_layer = decoder_module.DecoderLayer
    original_decode_layer = decode_mv_module.DecodeMVLayer
    original_projector = decode_mv_module.DeviceProjector
    original_create = setup.create
    distributed_layer = make_distributed_decoder_layer(original_layer, runtime)
    distributed_decode_layer = make_distributed_decode_layer(original_decode_layer, runtime)
    distributed_projector = make_distributed_projector(original_projector, runtime)
    decoder_module.DecoderLayer = distributed_layer
    base.DecoderLayer = distributed_layer
    decode_mv_module.DeviceProjector = distributed_projector
    decode_mv_module.DecodeMVLayer = distributed_decode_layer
    base.DeviceProjector = distributed_projector
    base.DecodeMVLayer = distributed_decode_layer
    key_check: dict[str, Any] = {}

    def checked_create(*args, **kwargs):
        # Each SC context generates a large bootstrap/rotation-key set.  Let
        # only one rank keygen at a time so eight simultaneous scratch heaps
        # do not exhaust the host before the timed inference even starts.
        client = engine = None
        for owner in range(runtime.world_size):
            if runtime.rank == owner:
                client, engine = original_create(*args, **kwargs)
            runtime.transport.barrier(f"sc_context_create_{owner}", timeout=900.0)
        assert client is not None and engine is not None
        vector = np.zeros(SLOTS, dtype=np.float64)
        vector[:2] = [0.125, -0.25]
        packet = runtime.pack(client.encrypt(vector, slots=SLOTS), tag="keycheck") if runtime.rank == 0 else None
        shared = runtime.broadcast_packet("keycheck", packet)
        received = runtime.unpack(shared)
        values = client.decrypt(received).cpu().numpy()[:2].real
        error = float(np.max(np.abs(values - vector[:2])))
        if not np.isfinite(error) or error > 0.01:
            raise RuntimeError(f"SC rank {runtime.rank} cannot decrypt rank0 ciphertext: {error}")
        key_check.update(passed=True, max_abs_error=error)
        return client, engine

    setup.create = checked_create
    argv = [str(DRIVERS[model]), "--out", str(output),
            "--warmup-passes", str(config["warmup_passes"]),
            "--measure-passes", str(config["measure_passes"])]
    argv.extend(["--body-ids", config["body_ids"]])
    if config["preflight_only"]:
        argv.append("--preflight-only")
    transport_before = runtime.transport.stats.to_dict()
    old_argv = sys.argv
    try:
        sys.argv = argv
        # The 8B GPU-cache adapter patches its ``base`` module at import time
        # and only exposes ``base.main`` (its own main exists only under
        # ``if __name__ == '__main__'``).  The other linked drivers export main.
        (driver.main if hasattr(driver, "main") else base.main)()
    finally:
        sys.argv = old_argv
        setup.create = original_create
        decoder_module.DecoderLayer = original_layer
        decode_mv_module.DeviceProjector = original_projector
        decode_mv_module.DecodeMVLayer = original_decode_layer
        base.DecoderLayer = original_layer
        base.DeviceProjector = original_projector
        base.DecodeMVLayer = original_decode_layer
        if original_gpu_memory is not None:
            gpu_driver.gpu_memory = original_gpu_memory
    summary_file = output / "prefill_decode_cipher_kv_summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.is_file() else None
    if not key_check.get("passed"):
        raise RuntimeError("SC cross-rank key check did not run")
    if not config["preflight_only"]:
        if summary is None or summary.get("kv_from_fhe_prefill") is not True:
            raise RuntimeError("SC linked runner did not produce a ciphertext-KV summary")
        if len(summary.get("measured_next_token_ids", [])) != config["measure_passes"]:
            raise RuntimeError("SC linked runner omitted measured Decode passes")
    transport_total = runtime.transport.stats.to_dict()
    transport_inference = {
        key: transport_total[key] - transport_before[key] for key in transport_total
    }
    return {"rank": runtime.rank, "device": runtime.physical_device,
            "key_check": key_check, "summary": summary,
            "transport": transport_total,
            "transport_inference": transport_inference}
