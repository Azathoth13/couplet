#!/usr/bin/env python3
"""T2 full-model prefill (all 32 layers) on an N-card host: the per-rank station chain streamed over the layers.

``bench/t2_layer.py`` profiles one layer (fresh encryption per pass, per-station
oracle).  This script is the end-to-end prefill: every rank encrypts its FM
ciphertexts of the embedding output once, then runs the same station chain

    rmsnorm_in -> qkv -> attention_input_exchange -> attention_core_station
    -> pairfm_exchange -> w_o_station -> residual -> rmsnorm_post -> mlp -> residual

for layers ``--layers`` (default 0-31) with that layer's real weights and
per-layer approximation parameters (RMSNorm / SiLU coefficients, attention
core schedule, RoPE / steady chain via ``layer_idx``); the ``residual_mlp``
output (2 limbs) of layer ``i`` is the ``rmsnorm_in`` input of layer ``i+1`` --
no decryption in between.  After the last layer every rank decrypts its token
rows and rank 0 compares the assembled hidden state with the poly-model
trajectory, then (with ``--model-dir``) applies the client-side final RMSNorm +
LM head and compares the next-token distribution with the poly model's.

Weights come from ``bench/make_model_fixtures.py`` (``layer{i}_weights.npz``
+ ``hidden_trajectory.npz``); the next layer's file is read by a background
thread while the current layer computes (8 ranks share the page cache).

``--verify-every N`` additionally decrypts the layer output every N layers and
compares it with ``hidden_{i+1}`` (adds decrypt time to those layers' walls --
the summary marks them).

Usage (from the repository root, idle cards)::

    python bench/make_model_fixtures.py . fixtures/model            # once, needs the checkpoint
    python bench/t2_model.py --preset rtx4090-low --fixtures fixtures/model \\
        --model-dir weights/Meta-Llama-3-8b-quarot-safetensors-v2 --output artifacts/t2_model/run1
    python bench/t2_model.py --preset rtx4090-low --fixtures fixtures/model --layers 0-3 --verify-every 1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Optional kernel-level op counting (OPCOUNT=1) / CUDA-event timing (OPCOUNT_TIME=1) per station,
# shared with the Slot-Coeff benchmark; see bench/profiling/opcount.py.  Adds a sync per station boundary.
#: ``ODIN_STATION_TS=1``: print wall-clock timestamps at station start / end (aligns an external
#: ``nvidia-smi --query-gpu=timestamp,utilization.gpu -lms`` sampler with the stations)
_STATION_TS = os.environ.get("ODIN_STATION_TS", "0") == "1"
_PREFILL_RESIDENT = os.environ.get("LLAMA_FHE_PREFILL_PLAIN_RESIDENT", "0") not in ("", "0", "false", "no")

_OPCOUNT = None
if os.environ.get("OPCOUNT", "0") == "1" or os.environ.get("OPCOUNT_TIME", "0") == "1":
    from bench.profiling import opcount as _OPCOUNT  # noqa: E402
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.t2_layer import (  # noqa: E402
    PROCESS_TRANSPORTS,
    STATIONS,
    _BootstrapTimer,
    _attribution,
    _compare,
    _decrypt_rows,
    _median,
    _release_ciphers,
    add_runtime_knob_args,
    apply_runtime_knobs,
    apply_softmax_post_exp,
    softmax_post_exp_description,
    attn_in_overlap_requested,
    rmsnorm_coeffs_table,
    rmsnorm_exit_description,
    silu_coeffs_table,
    rmsnorm_exit_kwargs,
    rmsnorm_slim_min_auto,
    softmax_coeffs_table,
    softmax_table_degrees,
    runtime_knob_summary,
    worker_tiers,
)
from llama3fhe.parallel.launch import expand_devices, spawn_workers  # noqa: E402
from llama3fhe.operators.linear.resident_plaintexts import RESIDENT  # noqa: E402
from llama3fhe.parallel.presets import preset_by_name  # noqa: E402
from llama3fhe.backend import stage_events_mark, stage_events_report  # noqa: E402

WEIGHT_KEYS = ("input_layernorm", "q_proj", "k_proj", "v_proj", "o_proj", "post_attention_layernorm",
               "gate_proj", "up_proj", "down_proj")


def parse_layers(spec: str, *, count: int = 32) -> tuple[int, ...]:
    """``"0-31"`` / ``"0-3"`` / ``"5"`` / ``"0,1,2"`` -> tuple of layer indices (must be contiguous, ascending)."""

    values: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            values.extend(range(int(lo), int(hi) + 1))
        elif part:
            values.append(int(part))
    if not values or any(v < 0 or v >= count for v in values):
        raise ValueError(f"--layers {spec!r}: indices must lie in [0, {count})")
    if values != list(range(values[0], values[0] + len(values))):
        raise ValueError(f"--layers {spec!r}: must be a contiguous ascending range (the hidden state is chained)")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="rtx4090-low")
    add_runtime_knob_args(parser)
    parser.add_argument("--fixtures", default="fixtures/model",
                        help="directory written by bench/make_model_fixtures.py")
    parser.add_argument("--model-dir", default=None,
                        help="checkpoint directory for the client-side final RMSNorm + LM head (rank 0, after the last "
                             "layer); default: skip the logits")
    parser.add_argument("--layers", default="all", help="contiguous layer range, e.g. 0-31, 0-3, 10 (default: every layer of --model)")
    parser.add_argument("--model", default=None,
                        help="model spec (llama3-8b | llama3.2-1b): re-cuts the preset's ciphertext geometry and selects the "
                             "polynomial pack; the fixtures directory must come from make_model_fixtures.py --model of "
                             "the same model")
    parser.add_argument("--input-level", type=int, default=None,
                        help="levels consumed by the FM inputs entering the first rmsnorm_in (default depth - 2: "
                             "35 -> 3 limbs on the depth-37 presets)")
    parser.add_argument("--prefix-kv-cache", default=None,
                        help="plaintext per-layer attention-sink KV page (.npz, tools/build_prefix_kv_cache.py): the "
                             "attention core adds its k public columns to every Softmax row, content RoPE starts at k; "
                             "--fixtures must be a Prefix trajectory (make_model_fixtures.py --prefix-kv-cache)")
    parser.add_argument("--verify-every", type=int, default=0,
                        help="decrypt + compare the layer output with the poly trajectory every N layers (0 = only the "
                             "last layer)")
    parser.add_argument("--barrier-each", action="store_true",
                        help="barrier before every station (per-station walls without rank skew; slower)")
    parser.add_argument("--no-prefetch-weights", action="store_true",
                        help="load each layer's weights synchronously instead of on a background thread")
    parser.add_argument("--no-stage-weights", action="store_true",
                        help="do not pin + upload the next layer's weights on the loader's side stream; the linear "
                             "stations then copy every weight page host-to-device themselves as before")
    parser.add_argument("--warm-keys", action="store_true",
                        help="upload every resident rotation-key tier (compacted to its cap) during setup, before the "
                             "model timer starts.  Under key_policy=resident the keys otherwise stream in lazily on "
                             "first use inside layer 0 (Pro 6000 4-card: 497 misses / 13.7 GiB / 7.9 s of H2D in L0).  "
                             "A server holds its keys before the client's ciphertexts arrive, so this is setup, not "
                             "prefill.")
    parser.add_argument("--warmup-all-layers", action="store_true",
                        help="like --warmup-layer0 but run every layer of the chain once on dummy data before the "
                             "timer starts; with LLAMA_FHE_PREFILL_PLAIN_RESIDENT=1 (and LLAMA_FHE_LINEAR_PLAIN_RESIDENT=1) "
                             "this pass encodes every linear weight plaintext into the resident store, so the timed "
                             "pass pays no weight pack / encode (the served-model 'weights pre-encoded' setting)")
    parser.add_argument("--warmup-layer0", action="store_true",
                        help="after --warm-keys and before the model timer starts, run the whole layer-0 station "
                             "chain once on ciphertexts of random rows (same RMS as the real hidden state) with the "
                             "real layer-0 weights, then release everything.  First-use costs -- CUDA module "
                             "loading, allocator growth, plan caches, transport arena mapping -- land here instead "
                             "of in layer 0 of the timed prefill (8-card Pro 6000: L0 6.07 s vs 2.58 s steady).  A "
                             "served model has run before the client's request arrives, so this is setup, not "
                             "prefill; the summary reports the warm-up wall separately.")
    parser.add_argument("--lite", action="store_true",
                        help="production-style layer loop (also T2_LITE=1): no per-layer start barrier and no barrier "
                             "before attn_core (the two exchange stations keep theirs), stations synchronise the "
                             "current stream only, and the per-station bookkeeping shrinks to wall / BS count / peak "
                             "allocation (no host RSS, transport deltas or pre-station device sync)")
    parser.add_argument("--encode-depth", type=int, default=0,
                        help="plaintext chunks encoded ahead of the MAC on a side CUDA stream in every linear "
                             "station (QKV / W_O / gate-up / down): chunk i+D is packed + encoded while chunk i is "
                             "MAC'd, joined by events instead of the two host waits per chunk of the sequential "
                             "schedule.  0 (default) = the sequential schedule; D > 0 costs D extra plaintext "
                             "chunks of device memory (3.3-3.8 GiB each at 512 rows) and measured slower on "
                             "8 x Pro 6000 (L10: qkv 0.26 -> 0.33 s, mlp 0.77 -> 0.75 s -- the encode and MAC "
                             "kernels each saturate the device, so they only time-slice).  Sets "
                             "LLAMA_FHE_LINEAR_ENCODE_DEPTH for the workers.")
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="*", default=None)
    parser.add_argument("--ranks-per-device", type=int, default=1)
    parser.add_argument("--transport", choices=PROCESS_TRANSPORTS, default=None)
    parser.add_argument("--max-plaintext-rows", type=int, default=None)
    parser.add_argument("--prefetch-mode", choices=("lazy", "sync", "async"), default=None)
    parser.add_argument("--mlp-shard-k", type=int, default=None)
    parser.add_argument("--mlp-limbs-from-silu", default=None, metavar="PATH",
                        help="derive each layer's MLP input limb budget from this SiLU coefficient table while "
                             "evaluating SiLU with the active table; for a fixed-limb coefficient A/B")
    parser.add_argument("--probe-layer", type=int, default=None, metavar="LAYER",
                        help="diagnostic only: decrypt each station output of this layer and record magnitudes "
                             "in summary.json; adds untimed probe work to the layer wall")
    parser.add_argument("--output", default=None,
                        help="directory for summary.json (default artifacts/t2_model_<timestamp>)")
    return parser.parse_args()


def load_weights(path) -> dict:
    import numpy as np

    with np.load(path) as data:
        return {name: np.ascontiguousarray(data[name], dtype=np.float32) for name in WEIGHT_KEYS}


def npz_stored_members(path) -> dict[str, tuple[int, tuple[int, ...]]] | None:
    """``{name: (byte offset of the float32 C-order array data, shape)}`` for the seven weight members of an
    uncompressed ``.npz``, or ``None`` when any member is compressed / not float32 / Fortran-ordered.  Lets the
    loader thread ``preadv`` each matrix straight into its
    pinned buffer instead of going through ``np.load`` (zipfile CRC pass + a second copy into pinned memory,
    ~1.3 s of loader-thread CPU per layer that was contending for the GIL with the kernel-launching thread)."""

    import zipfile

    import numpy as np
    from numpy.lib import format as npy_format

    members: dict[str, tuple[int, tuple[int, ...]]] = {}
    with zipfile.ZipFile(path) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        raw = archive.fp
        for name in WEIGHT_KEYS:
            info = infos.get(f"{name}.npy")
            if info is None or info.compress_type != zipfile.ZIP_STORED:
                return None
            raw.seek(info.header_offset)
            local_header = raw.read(30)
            if len(local_header) != 30 or local_header[:4] != b"PK\x03\x04":
                return None
            name_len = int.from_bytes(local_header[26:28], "little")
            extra_len = int.from_bytes(local_header[28:30], "little")
            data_offset = info.header_offset + 30 + name_len + extra_len
            with archive.open(info) as member:
                version = npy_format.read_magic(member)
                read_header = (npy_format.read_array_header_1_0 if version == (1, 0)
                               else npy_format.read_array_header_2_0)
                shape, fortran_order, dtype = read_header(member)
                header_len = member.tell()
            if fortran_order or np.dtype(dtype) != np.dtype(np.float32):
                return None
            members[name] = (int(data_offset + header_len), tuple(int(d) for d in shape))
    return members


def read_into(fd: int, buffer: memoryview, offset: int) -> None:
    """Fill ``buffer`` from ``fd`` at ``offset`` with ``os.preadv`` (releases the GIL; loops over short reads)."""

    view = buffer
    while len(view):
        count = os.preadv(fd, [view], offset)
        if count <= 0:
            raise OSError(f"short read at offset {offset} ({len(view)} bytes left)")
        view = view[count:]
        offset += count


WEIGHT_UPLOAD_CHUNK_BYTES = 32 << 20  # split the side-stream H2D so the transport's pinned exchanges interleave


class LayerWeights:
    """One layer's weights: the host arrays the stations slice, plus (optionally) resident device copies.

    ``host[name]`` are the NumPy arrays handed to the stations (views of pinned buffers when staged).
    ``device[name]`` are their device copies, uploaded on the loader's side stream; ``ready`` is the
    event after the last copy.  :meth:`register` publishes them in the linear operators'
    :data:`~llama3fhe.operators.linear.weight.DEVICE_WEIGHTS` registry so ``LinearWeight.prepare`` finds
    every page (``gate[a:b]``, ``down[:, a:b]``, the full q/k/v/o) already on the device.
    """

    def __init__(self, host: dict, *, device: dict | None = None, ready=None, pinned: dict | None = None,
                 stage_seconds: float = 0.0, read_seconds: float = 0.0, retained: bool = False):
        self.host = dict(host)
        self.device = dict(device or {})
        self.ready = ready
        self._pinned = dict(pinned or {})
        self.stage_seconds = float(stage_seconds)
        self.read_seconds = float(read_seconds)
        self._registered = False
        #: ``WeightPrefetcher(retain="device")``: the bundle is reused by the next token, ``release`` only unregisters.
        self.retained = bool(retained)

    def __getitem__(self, name: str):
        return self.host[name]

    @property
    def staged(self) -> bool:
        return bool(self.device)

    def register(self) -> None:
        if not self.device or self._registered:
            return
        from llama3fhe.operators.linear.weight import DEVICE_WEIGHTS

        for name, tensor in self.device.items():
            DEVICE_WEIGHTS.register(self.host[name], tensor, ready_event=self.ready)
        self._registered = True

    def unregister(self) -> None:
        if not self._registered:
            return
        from llama3fhe.operators.linear.weight import DEVICE_WEIGHTS

        for name in self.device:
            DEVICE_WEIGHTS.unregister(self.host[name])
        self._registered = False

    def release(self) -> None:
        """Drop the device copies (call after the device work that read them has completed).

        A ``retained`` bundle (``WeightPrefetcher(retain="device")``) only unregisters: the device copies
        serve the next token.
        """

        self.unregister()
        if self.retained:
            return
        self.device = {}
        self.ready = None
        self._pinned = {}


class WeightPrefetcher:
    """Background-thread reader of ``layer{i}_weights.npz`` one layer ahead.

    With ``device`` set the loader thread also pins the seven arrays and uploads them on its own CUDA
    stream (``--stage-weights``, default), so the layer starts with its weights resident and the linear
    stations skip their per-page host-to-device copies (0.1 s/layer on 8 x Pro 6000, the down-projection
    column pages being non-contiguous).  The event recorded after the copies gates their first use.

    ``retain`` (Decode-style repeated passes over the same layers, ``bench/t2_prefill_decode.py``):
    ``"none"`` re-reads every layer from disk on each pass (the one-layer-ahead production shape);
    ``"host"`` keeps the pinned host copies after :meth:`LayerWeights.release` (32 x 872 MB of host RAM),
    so the next pass only re-uploads them on the side stream (~35 ms/layer, hidden behind compute) --
    the single-card Decode disk read (1.0-1.2 s/layer) would otherwise exceed the 0.8 s layer compute;
    ``"device"`` keeps the device copies too (28 GB of device memory for 32 layers).
    """

    RETAIN_MODES = ("none", "host", "device")

    def __init__(self, fixtures: Path, layers, *, enabled: bool = True, device: str | None = None,
                 retain: str = "none"):
        self.fixtures = Path(fixtures)
        self.layers = tuple(layers)
        self.enabled = bool(enabled)
        self.device = None if device is None else str(device)
        if str(retain) not in self.RETAIN_MODES:
            raise ValueError(f"retain must be one of {self.RETAIN_MODES}, got {retain!r}")
        self.retain = str(retain)
        self._stream = None
        self._pending: dict[int, threading.Thread] = {}
        self._results: dict[int, LayerWeights] = {}
        self._errors: dict[int, BaseException] = {}
        self._retained_pinned: dict[int, dict] = {}        # retain="host": layer -> pinned host tensors
        self._retained_device: dict[int, LayerWeights] = {}  # retain="device": layer -> resident bundle
        self.load_seconds: dict[int, float] = {}
        self.wait_seconds: dict[int, float] = {}
        self.reuploads = 0

    def path(self, idx: int) -> Path:
        return self.fixtures / f"layer{int(idx)}_weights.npz"

    def _read_pinned(self, idx: int) -> dict[str, object] | None:
        """The layer's seven matrices read straight into pinned tensors (``preadv``, no CRC pass, no second copy);
        ``None`` when the archive is not a plain stored float32 npz (``T2_WEIGHT_READER=npload`` forces that)."""

        import easyfhe

        if os.environ.get("T2_WEIGHT_READER", "pread") == "npload":
            return None
        members = npz_stored_members(self.path(idx))
        if members is None:
            return None
        pinned: dict[str, object] = {}
        fd = os.open(self.path(idx), os.O_RDONLY)
        try:
            for name, (offset, shape) in members.items():
                tensor = easyfhe.empty(shape, dtype=easyfhe.float32, pin_memory=True)
                read_into(fd, memoryview(tensor.numpy()).cast("B"), offset)
                pinned[name] = tensor
            if os.environ.get("T2_WEIGHT_EVICT", "1") == "1":
                # Drop the file's pages from the page cache: on the rented single-card boxes the container
                # memory cap counts page cache but the kernel does not reclaim it against that cap, so a 32-layer
                # pass (28 GB of weights) on top of the ~75 GB key/host state gets SIGKILLed (seen ).
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return pinned

    def _read_mmap(self, idx: int) -> dict:
        """Map stored float32 weights with private, file-backed pages shared across ranks."""

        import numpy as np

        path = self.path(idx)
        members = npz_stored_members(path)
        if members is None:
            raise ValueError(f"T2_WEIGHT_READER=mmap requires ZIP_STORED C-order float32 weights: {path}")
        # mode='c' permits consumers to write a page without modifying the fixture on disk.
        # Each array owns its mapping, which closes when LayerWeights and any retained views are dropped.
        return {name: np.memmap(path, dtype=np.float32, mode="c", offset=offset, shape=shape)
                for name, (offset, shape) in members.items()}

    def _upload(self, pinned: dict[str, object]) -> tuple[dict[str, object], object]:
        """Chunked non-blocking H2D of every pinned tensor on the loader's stream; returns ``(device, ready)``."""

        import easyfhe

        if self._stream is None:
            self._stream = easyfhe.cuda.Stream()
        device: dict[str, object] = {}
        with easyfhe.cuda.stream(self._stream):
            for name, tensor in pinned.items():
                target = easyfhe.empty(tensor.shape, dtype=tensor.dtype, device=self.device)
                source_flat, target_flat = tensor.view(-1), target.view(-1)
                step = max(1, WEIGHT_UPLOAD_CHUNK_BYTES // int(tensor.element_size()))
                for start in range(0, int(source_flat.numel()), step):
                    target_flat[start:start + step].copy_(source_flat[start:start + step], non_blocking=True)
                device[name] = target
            ready = easyfhe.cuda.Event()
            ready.record(self._stream)
        return device, ready

    def _stage(self, idx: int) -> LayerWeights:
        """Read + pin + upload layer ``idx`` on the loader thread; returns the staged bundle."""

        import easyfhe

        started = time.perf_counter()
        pinned = self._read_pinned(idx)
        if pinned is None:
            arrays = load_weights(self.path(idx))
            pinned = {name: easyfhe.from_numpy(array).pin_memory() for name, array in arrays.items()}
        read_seconds = time.perf_counter() - started
        device, ready = self._upload(pinned)
        host = {name: tensor.numpy() for name, tensor in pinned.items()}
        return LayerWeights(host, device=device, ready=ready, pinned=pinned,
                            stage_seconds=time.perf_counter() - started, read_seconds=read_seconds)

    def _reupload(self, idx: int, pinned: dict) -> LayerWeights:
        """retain="host" second pass: the pinned copies are still here, only the H2D upload is redone."""

        started = time.perf_counter()
        host = {name: (tensor.numpy() if hasattr(tensor, "numpy") else tensor) for name, tensor in pinned.items()}
        self.reuploads += 1
        if self.device is None:
            return LayerWeights(host, pinned=pinned)
        device, ready = self._upload(pinned)
        return LayerWeights(host, device=device, ready=ready, pinned=pinned,
                            stage_seconds=time.perf_counter() - started, read_seconds=0.0)

    def _load(self, idx: int) -> None:
        started = time.perf_counter()
        try:
            retained = self._retained_pinned.get(idx)
            if retained is not None:
                self._results[idx] = self._reupload(idx, retained)
            elif self.device is not None:
                self._results[idx] = self._stage(idx)
            else:
                if os.environ.get("T2_WEIGHT_READER", "pread") == "mmap":
                    pinned = None
                    arrays = self._read_mmap(idx)
                else:
                    pinned = self._read_pinned(idx)
                    arrays = ({name: tensor.numpy() for name, tensor in pinned.items()} if pinned is not None
                              else load_weights(self.path(idx)))
                self._results[idx] = LayerWeights(arrays, pinned=pinned)
        except BaseException as exc:  # noqa: BLE001 - surfaced in get()
            self._errors[idx] = exc
        self.load_seconds[idx] = time.perf_counter() - started

    def start(self, idx: int) -> None:
        if idx in self._results or idx in self._pending or idx in self._retained_device:
            return
        if not self.enabled:
            return
        thread = threading.Thread(target=self._load, args=(int(idx),), name=f"weights-{idx}", daemon=True)
        self._pending[idx] = thread
        thread.start()

    def get(self, idx: int) -> LayerWeights:
        started = time.perf_counter()
        resident = self._retained_device.get(idx)
        if resident is not None:
            self.wait_seconds[idx] = time.perf_counter() - started
            return resident
        thread = self._pending.pop(idx, None)
        if thread is not None:
            thread.join()
        elif idx not in self._results:
            self._load(idx)
        self.wait_seconds[idx] = time.perf_counter() - started
        if idx in self._errors:
            raise self._errors.pop(idx)
        result = self._results.pop(idx)
        if self.retain == "host":
            # pinned tensors when staged / pread; plain host arrays otherwise (T2_WEIGHT_READER=npload)
            self._retained_pinned[idx] = result._pinned if result._pinned else dict(result.host)
        elif self.retain == "device" and result.device:
            result.retained = True
            self._retained_device[idx] = result
        return result

    def close(self) -> None:
        """Drop every retained copy (end of the run)."""

        for bundle in self._retained_device.values():
            bundle.retained = False
            bundle.release()
        self._retained_device.clear()
        self._retained_pinned.clear()


# ----------------------------------------------------------------- worker
def worker_entry(rank: int, preset, endpoints, *, fixtures: str, layers: tuple[int, ...], input_level: int,
                 verify_every: int, barrier_each: bool, prefetch_weights: bool, model_dir: str | None,
                 warm_keys: bool = False, warmup_layer0: bool = False, stage_weights: bool = True,
                 lite: bool = False, encode_depth: int | None = None, warmup_all_layers: bool = False,
                 prefix_kv_cache: str | None = None, mlp_limbs_from_silu: str | None = None,
                 probe_layer: int | None = None):
    """One rank: encrypt once, stream the station chain over ``layers``, decrypt at the end."""

    import numpy as np

    if encode_depth is not None:
        # read by LinearOperatorConfig's default (llama3fhe.operators.linear.config) each time a
        # kernel builds its operator config; set before any linear station of this rank runs
        os.environ["LLAMA_FHE_LINEAR_ENCODE_DEPTH"] = str(int(encode_depth))

    import easyfhe

    # E3-E8 engineering patches (ODIN_SYLPH_KERNELS / ODIN_BATCH_BS / ODIN_FAST_ENCODE / ODIN_FAST_STAGE2 /
    # ODIN_MULTI_MAC), shared with the Decode benches: llama3fhe/acceleration/patches.py.  Must precede context creation.
    from llama3fhe.acceleration import patches as acceleration_patches
    _patches = acceleration_patches.install_from_env(rank)
    extra_kernel_modules = _patches["extra_kernel_modules"]
    fast_encode = _patches["fast_encode"]
    multi_mac = _patches["multi_mac"]

    if _OPCOUNT is not None:
        _OPCOUNT.install(extra_modules=extra_kernel_modules)
        _OPCOUNT.install_api_odin()

    from llama3fhe.application.args import model_poly_paths
    from llama3fhe.approx.attention import rope_cos_sin, set_default_rope_scaling
    from llama3fhe.approx.prefix_cache_io import load_prefix_kv_cache
    from llama3fhe.approx.rmsnorm import get_layer_coeffs
    from llama3fhe.approx.silu import load_silu_poly_coeffs
    from llama3fhe.approx.softmax import SoftmaxPolyConfig
    from llama3fhe.backend import plaintext_cache_snapshot, release_all
    from llama3fhe.graph import _linear_config
    from llama3fhe.layouts.linear import LinearCarrierLayout
    from llama3fhe.operators.attention.config import AttentionOperatorConfig
    from llama3fhe.operators.linear.weight import DEVICE_WEIGHTS
    from llama3fhe.parallel.memory import native_cache_inventory
    from llama3fhe.parallel.prefill.stations import (
        QKV_INPUT_LIMBS,
        QK_SCORE_LEVEL_DROP,
        attention_core_operator_config,
        attention_core_station,
        attention_input_exchange,
        mlp_input_limbs,
        mlp_station,
        pairfm_exchange,
        qkv_station,
        residual_station,
        rmsnorm_station,
        w_o_station,
    )
    from llama3fhe.parallel.prefill.transport import build_transport
    from llama3fhe.parallel.worker import build_worker_state, host_rss_gib, load_layer_config, production_layouts

    fixtures_dir = Path(fixtures)
    layers = tuple(int(v) for v in layers)
    set_default_rope_scaling(preset.model_spec.rope_scaling)
    # Prefix (attention sink): plaintext K/V page per layer; the key plan needs prefix_k before the context is built
    prefix_bundles = None
    if prefix_kv_cache is not None:
        prefix_bundles = load_prefix_kv_cache(
            prefix_kv_cache, layout=production_layouts(preset)[1], expected_layers=layers,
            expected_rope_theta=float(preset.model_spec.rope_theta))
    prefix_k = 0 if prefix_bundles is None else int(prefix_bundles[0].prefix_k)
    prefix_by_layer = {} if prefix_bundles is None else dict(zip(layers, prefix_bundles, strict=True))
    tiers = worker_tiers(STATIONS)
    transport = build_transport(rank, endpoints)
    extra_rotations = None
    _hoist_keys = acceleration_patches.hoist_rotations(preset, multi_mac)
    if _hoist_keys is not None:
        # E6: the double-hoisted giant step rotates every partial sum of a chunk by its own g * giant_offset
        # (Horner only needs giant_offset); add those keys to the linear tier before the key plan is built
        extra_rotations = {"linear": _hoist_keys}
        print(f"[rank {rank}] hoisted giant keys: {_hoist_keys}", flush=True)
    state = build_worker_state(rank, preset, transport=transport, tiers=tiers, trace_keys=True,
                               extra_rotations=extra_rotations, prefix_k=prefix_k)
    probe, context, layout, attention_layout = state.memory, state.context, state.feature_layout, state.attention_layout
    ownership, world, native, life, tracer = state.ownership, state.world, state.native, state.keys, state.tracer
    bootstrap = _BootstrapTimer(state.bootstrap_operator)
    config = load_layer_config(preset)
    schedule = preset.layer_schedule()
    qkv_config = _linear_config(schedule, schedule.qkv_max_plaintext_rows, schedule.qkv_hoist_strategy)
    mlp_config = _linear_config(schedule, schedule.mlp_max_plaintext_rows, schedule.mlp_hoist_strategy)
    w_o_config = _linear_config(schedule, schedule.output_max_plaintext_rows, schedule.output_hoist_strategy)
    resident = state.resident_report()
    key_plan = state.key_plan.summary(alpha=int(native.alpha), special_limbs=int(native.K))

    DEFAULT_RMS, DEFAULT_SILU, DEFAULT_SOFTMAX = model_poly_paths(getattr(preset, "model", None))
    rms_all, rms_table_info = rmsnorm_coeffs_table(DEFAULT_RMS, preset)
    rms_exit_policy = rmsnorm_exit_description(preset)
    silu_all, silu_table_info = silu_coeffs_table(DEFAULT_SILU, preset)
    limb_silu_all = (silu_all if mlp_limbs_from_silu is None
                     else load_silu_poly_coeffs(mlp_limbs_from_silu)["per_layer"])
    if set(limb_silu_all) != set(silu_all):
        raise ValueError("--mlp-limbs-from-silu must contain the same model layers as the active SiLU table")
    softmax_table, softmax_table_info = softmax_coeffs_table(DEFAULT_SOFTMAX, preset)
    softmax_config = SoftmaxPolyConfig.from_artifact(softmax_table)
    hidden_dim = int(layout.hidden_dim)
    rows = slice(int(ownership.token_start), int(ownership.token_stop))

    # -- the poly-model trajectory: hidden_{first} is the encrypted input, hidden_{i+1} the oracle of layer i
    with np.load(fixtures_dir / "hidden_trajectory.npz") as data:
        hidden_in = np.asarray(data[f"hidden_{layers[0]}"], dtype=np.float32)
        oracle_hidden = {idx: np.asarray(data[f"hidden_{idx + 1}"], dtype=np.float32) for idx in layers}
    if hidden_in.shape != (int(layout.seq_len), hidden_dim):
        raise ValueError(f"hidden_{layers[0]} has shape {hidden_in.shape}, layout wants {(layout.seq_len, hidden_dim)}")
    packed = layout.pack(hidden_in)

    fuse_silu = bool(schedule.fuse_silu_domain_map)
    mlp_limbs_by_layer = {}
    for key, active in silu_all.items():
        required = mlp_input_limbs(active["tiers"]["standard"], fuse_silu_domain_map=fuse_silu)
        selected = mlp_input_limbs(limb_silu_all[key]["tiers"]["standard"],
                                   fuse_silu_domain_map=fuse_silu)
        if selected < required:
            raise ValueError(f"--mlp-limbs-from-silu: layer {key} has {selected} limbs, "
                             f"but the active SiLU requires {required}")
        mlp_limbs_by_layer[key] = selected
    carrier_layout = LinearCarrierLayout(seq_len=int(layout.seq_len), dimension=hidden_dim, slots=int(layout.slots))
    cos, sin = rope_cos_sin(seq_len=int(attention_layout.seq_len), head_dim=int(attention_layout.head_dim),
                            theta=float(AttentionOperatorConfig().rope_theta), start_pos=int(prefix_k))
    operator_config = AttentionOperatorConfig(
        sparse_max_baby_offsets=int(schedule.sparse_max_baby_offsets),
        shear_baby_steps=int(schedule.shear_baby_steps),
        pv_complex_delta_folding=bool(schedule.pv_complex_delta_folding),
        pv_prefold_after_softmax=bool(schedule.pv_prefold_after_softmax),
        batched_sparse_input_bsgs=bool(schedule.batched_sparse_attention_input),
        qk_complex_feature_folding=True,
        qk_key_features_pre_folded=True,
        qk_query_features_pre_scaled=True,
        qk_query_features_pre_folded=True,
        # E10: Q/K lanes shed the value-stream surplus after qk_gather_axis1 (13 -> 10-limb score schedule);
        # LLAMA_FHE_QK_LEVEL_DROP=0 in the environment restores the old chain for A/B.
        qk_score_level_drop=int(QK_SCORE_LEVEL_DROP),
        # preset decides (rtx6000pro-high / a100-high / h100-emu: 1 pass, 4090: 2);
        # T2_PAIRFM_BS_ITERATIONS in the environment overrides for A/B.
        pairfm_refresh_bootstrap_iterations=int(os.environ.get(
            "T2_PAIRFM_BS_ITERATIONS", str(int(preset.pairfm_refresh_bootstrap_iterations)))),
        softmax_slim_periodic=bool(getattr(preset, "softmax_slim_periodic", False)),
    )

    stage_device = str(context.device) if (stage_weights and str(context.device).startswith("cuda")) else None
    prefetcher = WeightPrefetcher(fixtures_dir, layers, enabled=bool(prefetch_weights), device=stage_device)
    prefetcher.start(layers[0])

    def _stage_dict(values) -> dict[str, float]:
        return {str(k): round(float(v), 4) for k, v in dict(values or {}).items()}

    _resident_counters = [0, 0, 0]  # RESIDENT hits / misses / skipped at the end of the previous layer

    def run_layer(idx: int, inputs, weights: LayerWeights, *, verify: bool, tag_prefix: str | None = None):
        """One layer of the chain on ``inputs`` (this rank's FM ciphertexts, consumed); returns (outputs, record).

        ``tag_prefix`` (default ``model/{idx}``) prefixes the transport tags, tracer stage names and probe labels
        so a warm-up pass of the same layer is booked apart from the timed one.
        """

        stations: dict[str, dict] = {}
        order = [20]
        batch_bs_before = int(getattr(state.bootstrap_operator, "batch_calls", 0) or 0)
        tag_prefix = f"model/{idx}" if tag_prefix is None else str(tag_prefix)
        stage_suffix = str(idx) if tag_prefix == f"model/{idx}" else tag_prefix.replace("/", "_")
        prefix = f"t2m.r{rank}.l{idx}" if tag_prefix == f"model/{idx}" else f"t2m.r{rank}.{stage_suffix}"
        coeffs_attn = get_layer_coeffs(rms_all, "attn", idx)
        coeffs_ffn = get_layer_coeffs(rms_all, "ffn", idx)
        rms_exit = rmsnorm_exit_kwargs(idx, coeffs_attn, preset)
        rms_exit_post = rmsnorm_exit_kwargs(idx, coeffs_ffn, preset)
        silu_entry = silu_all[str(idx)]["tiers"]["standard"]
        mlp_entry = mlp_limbs_by_layer[str(idx)]
        core_config = attention_core_operator_config(schedule, layer_idx=idx, base=operator_config)
        layer_operator_config, core_config, qkv_entry = apply_softmax_post_exp(
            idx, operator_config=operator_config, core_config=core_config, qkv_limbs=QKV_INPUT_LIMBS)
        rms_exit = rmsnorm_slim_min_auto(rms_exit, qkv_entry, branch="attn")
        rms_exit_post = rmsnorm_slim_min_auto(rms_exit_post, mlp_entry, branch="ffn")
        gamma_in, gamma_post = weights["input_layernorm"], weights["post_attention_layernorm"]
        q_w, k_w, v_w, o_w = weights["q_proj"], weights["k_proj"], weights["v_proj"], weights["o_proj"]
        gate_w, up_w, down_w = weights["gate_proj"], weights["up_proj"], weights["down_proj"]
        weights.register()  # resident device copies (if staged) -> LinearWeight.prepare skips its H2D
        registry_hits_before, registry_misses_before = DEVICE_WEIGHTS.hits, DEVICE_WEIGHTS.misses
        # the stations only take the plaintext hidden state for its shape (shard bookkeeping)
        hidden_shape_only = np.zeros((int(layout.seq_len), hidden_dim), dtype=np.float32)

        _alloc_retry_seen = [0]

        def _resident_scope(name: str):
            """Resident-plaintext scope of a Prefill station (LLAMA_FHE_PREFILL_PLAIN_RESIDENT=1): the same linear of
            the same layer meets the same weights in the warm-up pass and the timed pass, so the encoded plaintexts of
            (layer, station) are stored once and handed back later.  None (= scope untouched, store inactive) otherwise;
            the warm-up and the timed pass share the key so the warm-up populates the store."""

            if not _PREFILL_RESIDENT:
                return None
            return f"prefill/L{int(idx)}/{name}"

        def _resident_ctx(name: str):
            scope = _resident_scope(name)
            return RESIDENT.scope(scope) if scope is not None else contextlib.nullcontext()

        def _alloc_retries_delta() -> int:
            try:
                total = int(easyfhe.cuda.memory_stats().get("num_alloc_retries", 0))
            except Exception:  # pragma: no cover - diagnostics only
                return 0
            delta = total - _alloc_retry_seen[0]
            _alloc_retry_seen[0] = total
            return delta

        def timed_lite(name: str, tier: str | None, fn, *, barrier: bool = False):
            """``timed`` without the profiling extras: the previous station already synchronised, so no leading
            device sync; the trailing one waits for the current stream only (side streams -- the weight uploads --
            keep running); peak allocation is the only memory figure."""

            if barrier or barrier_each:
                transport.barrier(f"{tag_prefix}/before/{name}")
            probe.reset_peak()
            calls_before = bootstrap.calls
            bootstrap.take()
            if _OPCOUNT is not None:
                _OPCOUNT.take()
                _OPCOUNT.take_trace()
            started = time.perf_counter()
            if _STATION_TS:
                print(f"[station-ts] start {name} {time.time():.3f}", flush=True)
            stage_events_mark(f"{name}.start")
            with tracer.stage(f"{name}_{stage_suffix}"), _resident_ctx(name):
                if tier is None:
                    value = fn()
                else:
                    with life.lease(tier, prefetch=False) as record:
                        value = fn()
                    value = (value, record)
            stage_events_mark(f"{name}.end")
            easyfhe.cuda.current_stream().synchronize()
            wall = time.perf_counter() - started
            if _STATION_TS:
                print(f"[station-ts] end {name} {time.time():.3f} {wall:.3f}", flush=True)
            stage_events_report(name)
            lease_record = None
            if tier is not None:
                value, lease_record = value
            stats = tracer.stages[f"{name}_{stage_suffix}"].as_row()
            bs_records = bootstrap.take()
            stations[name] = {
                "wall_s": wall,
                "tier": tier,
                "bootstrap_calls": bootstrap.calls - calls_before,
                "bootstrap_seconds": round(sum(r[0] for r in bs_records), 4),
                "peak_alloc_gib": float(easyfhe.cuda.max_memory_allocated()) / float(2**30),
                # allocator pressure: cudaMalloc retries (each one frees cached blocks and synchronises the
                # device) and the reserved pool size, per station
                "alloc_retries": _alloc_retries_delta(),
                "reserved_gib": float(easyfhe.cuda.memory_reserved()) / float(2**30),
                "keys": {k: stats.get(k) for k in ("requests", "misses", "gib_loaded", "load_seconds")},
                "leases": ([{"tier": getattr(lease_record, "tier", None),
                             "evict_s": getattr(lease_record, "evict_seconds", None),
                             "prefetch_s": getattr(lease_record, "prefetch_seconds", None)}]
                           if lease_record is not None else []),
            }
            if _OPCOUNT is not None:
                stations[name]["ops"] = _OPCOUNT.take()
                stations[name]["refresh_trace"] = _OPCOUNT.take_trace()
            return value

        def timed_full(name: str, tier: str | None, fn, *, barrier: bool = False):
            if barrier or barrier_each:
                transport.barrier(f"{tag_prefix}/before/{name}")
            probe.reset_peak()
            easyfhe.cuda.synchronize()
            calls_before = bootstrap.calls
            bootstrap.take()
            transport_before = transport.stats.to_dict()
            plain_before = plaintext_cache_snapshot(context)
            if _OPCOUNT is not None:
                _OPCOUNT.take()
                _OPCOUNT.take_trace()
            started = time.perf_counter()
            with tracer.stage(f"{name}_{stage_suffix}"), _resident_ctx(name):
                if tier is None:
                    value = fn()
                else:
                    with life.lease(tier, prefetch=False) as record:
                        value = fn()
                    value = (value, record)
            easyfhe.cuda.synchronize()
            wall = time.perf_counter() - started
            lease_record = None
            if tier is not None:
                value, lease_record = value
            row = probe.snap(f"{order[0]} {name}_{stage_suffix}", {"wall_s": round(wall, 4)})
            order[0] += 1
            stats = tracer.stages[f"{name}_{stage_suffix}"].as_row()
            transport_after = transport.stats.to_dict()
            plain_after = plaintext_cache_snapshot(context)
            bs_records = bootstrap.take()
            stations[name] = {
                "wall_s": wall,
                "tier": tier,
                "bootstrap_calls": bootstrap.calls - calls_before,
                "bootstrap_seconds": round(sum(r[0] for r in bs_records), 4),
                "peak_alloc_gib": row.get("peak_alloc_gib"),
                "alloc_after_gib": row.get("alloc_gib"),
                "host_rss_gib": round(host_rss_gib(), 2),
                "keys": {k: stats.get(k) for k in ("requests", "misses", "gib_loaded", "load_seconds")},
                "transport_delta": {k: round(transport_after[k] - transport_before[k], 4)
                                    for k in ("recv_wait_seconds", "send_seconds", "bytes_sent", "bytes_received")},
                "leases": ([{"tier": getattr(lease_record, "tier", None),
                             "evict_s": getattr(lease_record, "evict_seconds", None),
                             "prefetch_s": getattr(lease_record, "prefetch_seconds", None)}]
                           if lease_record is not None else []),
                "plain_cache": {k: int(plain_after[k] - plain_before[k]) for k in ("hits", "misses", "skips")},
            }
            if _OPCOUNT is not None:
                stations[name]["ops"] = _OPCOUNT.take()
                stations[name]["refresh_trace"] = _OPCOUNT.take_trace()
            return value

        timed = timed_lite if lite else timed_full

        probe_this_layer = probe_layer is not None and idx == probe_layer and tag_prefix == f"model/{idx}"

        def probe_ciphers(ciphers, *, feature_rows: bool = False) -> dict:
            """Read live station outputs without modifying the ciphertexts or their level."""

            ciphers = tuple(ciphers)
            decoded = ((_decrypt_rows(context, layout, ownership, ciphers),) if feature_rows
                       else (np.asarray(context.decrypt(c), dtype=np.complex128) for c in ciphers))
            max_abs = 0.0
            sample_count = nonfinite_count = 0
            for values in decoded:
                magnitudes = np.abs(values)
                finite = np.isfinite(magnitudes)
                sample_count += int(magnitudes.size)
                nonfinite_count += int(magnitudes.size - np.count_nonzero(finite))
                if np.any(finite):
                    max_abs = max(max_abs, float(np.max(magnitudes[finite])))
            return {"max_abs": max_abs if sample_count > nonfinite_count else None,
                    "finite": nonfinite_count == 0, "nonfinite_count": nonfinite_count,
                    "cipher_count": len(ciphers), "layout": "feature_rows" if feature_rows else "raw_complex_slots"}

        def record_exit(name: str, ciphers):
            ciphers = tuple(ciphers)
            stations[name]["output_limbs"] = [int(c.state.cur_limbs) for c in ciphers]
            if probe_this_layer:
                stations[name]["probe"] = probe_ciphers(
                    ciphers, feature_rows=name in {"rmsnorm_in", "w_o", "residual_attn",
                                                   "rmsnorm_post", "mlp", "residual_mlp"})

        if not lite:
            transport.barrier(f"{tag_prefix}/start")
        layer_started = time.perf_counter()
        keep: dict[str, object] = {}
        outputs: tuple = ()
        try:
            rms = timed("rmsnorm_in", None, lambda: rmsnorm_station(
                inputs, gamma_in, coeffs=coeffs_attn, ownership=ownership, layout=layout, context=context,
                bootstrap_operator=bootstrap, keep_residual=True, **rms_exit,
                function_prefix=f"{prefix}.rms_in"))
            keep["rms"] = rms
            record_exit("rmsnorm_in", rms.ciphers)
            stations["rmsnorm_in"]["input_limbs"] = [int(c.state.cur_limbs) for c in inputs]
            stations["rmsnorm_in"]["stage_seconds"] = _stage_dict(rms.stage_seconds)
            stations["rmsnorm_in"]["inner_stage_seconds"] = _stage_dict(rms.level_trace.get("rmsnorm_stage_seconds"))
            release_all(inputs)
            inputs = ()

            qkv = timed("qkv", "linear", lambda: qkv_station(
                rms.ciphers, hidden_shape_only, q_w, k_w, v_w, ownership=ownership, layout=layout,
                attention_layout=attention_layout, context=context, config=config, operator_config=qkv_config,
                input_column_scale=rms.linear_input_column_scale(), input_limbs=qkv_entry,
                function_prefix=f"{prefix}.qkv"))
            keep["qkv"] = qkv
            record_exit("qkv", qkv.carrier_ciphers)
            stations["qkv"]["stage_seconds"] = _stage_dict(qkv.stage_seconds)
            _release_ciphers(rms.ciphers)
            rms.ciphers = ()

            attn_in = timed("attn_input", None, lambda: attention_input_exchange(
                qkv.carrier_ciphers, ownership=ownership, world=world, transport=transport,
                carrier_layout=carrier_layout, attention_layout=attention_layout, context=context, cos=cos, sin=sin,
                model_config=config, operator_config=layer_operator_config, keys=life, prefetch_keys=False,
                overlap=attn_in_overlap_requested(preset),
                function_prefix=f"{prefix}.attn_in", tag=f"{tag_prefix}/attn_in"), barrier=True)
            keep["attn_in"] = attn_in
            if probe_this_layer:
                parts = {name: probe_ciphers(c for _, c in items) for name, items in (
                    ("query", attn_in.query_items), ("key", attn_in.key_items), ("value", attn_in.value_items))}
                maxima = [p["max_abs"] for p in parts.values() if p["max_abs"] is not None]
                stations["attn_input"]["probe"] = {
                    "max_abs": max(maxima) if maxima else None,
                    "finite": all(p["finite"] for p in parts.values()), "parts": parts}
            stations["attn_input"]["stage_seconds"] = _stage_dict(attn_in.stage_seconds)
            stations["attn_input"]["comm_gib"] = {k: round(int(v) / float(2**30), 4)
                                                  for k, v in attn_in.comm_bytes.items()}
            stations["attn_input"]["scheduler_stats"] = {k: float(v) for k, v in attn_in.scheduler_stats.items()}
            qkv.release()

            core = timed("attn_core", None, lambda: attention_core_station(
                attn_in.query_items, attn_in.key_items, attn_in.value_items, ownership=ownership,
                attention_layout=attention_layout, context=context, softmax_config=softmax_config,
                operator_config=core_config, bootstrap_operator=bootstrap, layer_idx=idx,
                keys=life, prefetch_keys=False, prefix_bundle=prefix_by_layer.get(idx),
                function_prefix=f"{prefix}.core"), barrier=not lite)
            keep["core"] = core
            record_exit("attn_core", [c for _, c in core.output_items])
            stations["attn_core"]["stage_seconds"] = _stage_dict(core.stage_seconds)
            attn_in.release()

            pf = timed("pairfm", None, lambda: pairfm_exchange(
                core.output_items, ownership=ownership, world=world, transport=transport, feature_layout=layout,
                attention_layout=attention_layout, context=context, model_config=config, operator_config=core_config,
                bootstrap_operator=bootstrap, keys=life, prefetch_keys=False,
                function_prefix=f"{prefix}.pairfm", tag=f"{tag_prefix}/pairfm"), barrier=True)
            keep["pf"] = pf
            record_exit("pairfm", pf.ciphers)
            stations["pairfm"]["stage_seconds"] = _stage_dict(pf.stage_seconds)
            stations["pairfm"]["comm_gib"] = {k: round(int(v) / float(2**30), 4) for k, v in pf.comm_bytes.items()}
            core.release()

            w_o_profile: dict[str, float] = {}
            wo = timed("w_o", "linear", lambda: w_o_station(
                pf.carriers, o_w, ownership=ownership, layout=layout, attention_layout=attention_layout,
                context=context, operator_config=w_o_config, profile=w_o_profile))
            keep["wo"] = tuple(wo)
            record_exit("w_o", wo)
            stations["w_o"]["stage_seconds"] = _stage_dict({k: v for k, v in w_o_profile.items() if "." not in k})
            pf.release()

            res1 = timed("residual_attn", None, lambda: residual_station(
                rms.residual_inputs, wo, ownership=ownership, layout=layout, context=context))
            keep["res1"] = res1
            record_exit("residual_attn", res1.ciphers)
            rms.release()
            _release_ciphers(keep.pop("wo"))

            rms2 = timed("rmsnorm_post", None, lambda: rmsnorm_station(
                res1.ciphers, gamma_post, coeffs=coeffs_ffn, ownership=ownership, layout=layout, context=context,
                bootstrap_operator=bootstrap, keep_residual=True, **rms_exit_post,
                function_prefix=f"{prefix}.rms_post"))
            keep["rms2"] = rms2
            record_exit("rmsnorm_post", rms2.ciphers)
            stations["rmsnorm_post"]["stage_seconds"] = _stage_dict(rms2.stage_seconds)
            stations["rmsnorm_post"]["inner_stage_seconds"] = _stage_dict(rms2.level_trace.get("rmsnorm_stage_seconds"))
            res1.release()

            mlp = timed("mlp", "linear", lambda: mlp_station(
                rms2.ciphers, hidden_shape_only, gate_w, up_w, down_w, silu_entry=silu_entry, ownership=ownership,
                layout=layout, context=context, config=config, operator_config=mlp_config,
                input_column_scale=rms2.linear_input_column_scale(),
                fuse_silu_domain_map=fuse_silu, input_limbs=mlp_entry,
                feature_shard_k=int(preset.mlp_feature_shard_k), transport=transport,
                function_prefix=f"{prefix}.mlp"))
            keep["mlp"] = mlp
            record_exit("mlp", mlp.ciphers)
            stations["mlp"]["stage_seconds"] = _stage_dict(mlp.stage_seconds)
            _release_ciphers(rms2.ciphers)
            rms2.ciphers = ()

            res2 = timed("residual_mlp", None, lambda: residual_station(
                rms2.residual_inputs, mlp.ciphers, ownership=ownership, layout=layout, context=context))
            record_exit("residual_mlp", res2.ciphers)
            # hand the layer output on: detach the ciphers from res2 so its release() does not free them
            outputs = tuple(res2.ciphers)
            res2.ciphers = ()
            keep["res2"] = res2
            rms2.release()
            mlp.release()
        finally:
            for value in keep.values():
                release = getattr(value, "release", None)
                if callable(release):
                    release()
                elif isinstance(value, (tuple, list)):
                    _release_ciphers(value)
            if inputs:
                release_all(inputs)
            weights.unregister()
        if lite:
            easyfhe.cuda.current_stream().synchronize()
        else:
            easyfhe.cuda.synchronize()
        layer_wall = time.perf_counter() - layer_started

        record = {
            "layer": int(idx),
            "layer_wall_s": layer_wall,
            "station_wall_sum_s": sum(v["wall_s"] for v in stations.values()),
            "bootstrap_calls": sum(v["bootstrap_calls"] for v in stations.values()),
            "bootstrap_seconds": round(sum(v["bootstrap_seconds"] for v in stations.values()), 4),
            "output_limbs": [int(c.state.cur_limbs) for c in outputs],
            "stations": stations,
            "weights_staged": bool(weights.staged),
            "weights_device_hits": int(DEVICE_WEIGHTS.hits - registry_hits_before),
            "weights_device_misses": int(DEVICE_WEIGHTS.misses - registry_misses_before),
            "verified": False,
        }
        if _PREFILL_RESIDENT:
            info = RESIDENT.info()
            record["resident_plaintexts"] = {
                "hits": int(info["hits"] - _resident_counters[0]), "misses": int(info["misses"] - _resident_counters[1]),
                "skipped": int(info["skipped"] - _resident_counters[2]), "entries": int(info["entries"]),
                "gib": float(info["gib"]), "budget_gib": float(info["budget_gib"]),
            }
            _resident_counters[0], _resident_counters[1], _resident_counters[2] = (
                int(info["hits"]), int(info["misses"]), int(info["skipped"]))
        if fast_encode is not None:
            # the deferred encoder range check (one device read per layer instead of one per plaintext)
            record["fast_encode"] = fast_encode.check()
        if multi_mac is not None:
            record["multi_mac"] = dict(multi_mac.STATS)
            if multi_mac.PROFILE:
                record["multi_mac_profile"] = {k: [v[0], round(v[1], 1), round(v[2], 1)] for k, v in multi_mac.PROFILE.items()}
                multi_mac.PROFILE.clear()
        batch_bs = int(getattr(state.bootstrap_operator, "batch_calls", 0) or 0) - batch_bs_before
        if batch_bs:
            record["bootstrap_batches"] = batch_bs
        if verify:
            started = time.perf_counter()
            got = _decrypt_rows(context, layout, ownership, outputs)
            record["oracle"] = _compare(got, oracle_hidden[idx][rows])
            record["verify_seconds"] = time.perf_counter() - started
            record["verified"] = True
        return outputs, record

    # ------------------------------------------------------------- the model
    inputs = tuple(
        context.encrypt(packed[index], level=int(input_level), slots=int(layout.slots))
        for index in ownership.cipher_indices
    )
    easyfhe.cuda.synchronize()
    # the first layer's weights are part of the setup (the server holds them before the client's ciphertexts arrive)
    first_weights = prefetcher.get(layers[0])
    warm_keys_report: dict = {"enabled": bool(warm_keys)}
    if warm_keys:
        # Same as the server holding its evaluation keys on the device before the request: upload every tier of
        # the key plan at its cap.  Under key_policy=resident nothing is evicted afterwards, so L0 no longer pays
        # the lazy first-use H2D (qkv / attn_input / attn_core misses).
        started = time.perf_counter()
        per_tier: dict[str, dict[str, float]] = {}
        for tier_name in state.key_plan.names:
            count, seconds = life.prefetch(tier_name)
            per_tier[str(tier_name)] = {"keys": int(count), "seconds": round(float(seconds), 3)}
        easyfhe.cuda.synchronize()
        warm_keys_report.update({"seconds": round(time.perf_counter() - started, 3), "tiers": per_tier,
                                 "resident_keys_gib": native_cache_inventory(native)["total_gib"]})
        probe.snap("04 warm keys", warm_keys_report)
    warmup_report: dict = {"enabled": bool(warmup_layer0 or warmup_all_layers), "all_layers": bool(warmup_all_layers)}
    if warmup_layer0 or warmup_all_layers:
        # Server-side warm-up on dummy data: the whole layer-0 chain once, on ciphertexts of random rows with the
        # real hidden state's RMS and the real layer-0 weights, all ranks in step (the exchange stations are
        # collectives).  Nothing of it survives: the outputs are released, the BS records dropped and the peak
        # counters reset; only the one-time costs (CUDA module loading, allocator growth, plan caches, transport
        # arena mapping) stay behind -- the same ones a served model paid long before the client's request.
        started = time.perf_counter()
        rng = np.random.default_rng(20260930)
        hidden_rms = float(np.sqrt(np.mean(np.square(hidden_in.astype(np.float64)))))
        dummy_rows = (rng.standard_normal(hidden_in.shape) * hidden_rms).astype(np.float32)
        dummy_packed = layout.pack(dummy_rows)
        dummy_inputs = tuple(
            context.encrypt(dummy_packed[index], level=int(input_level), slots=int(layout.slots))
            for index in ownership.cipher_indices
        )
        easyfhe.cuda.synchronize()
        encrypt_seconds = time.perf_counter() - started
        warm_outputs, warm_record = run_layer(layers[0], dummy_inputs, first_weights, verify=False,
                                              tag_prefix=f"warmup/{layers[0]}")
        warm_layers: list[dict] = [{"layer": int(layers[0]), "wall_s": round(float(warm_record["layer_wall_s"]), 3),
                                    "resident_plaintexts": warm_record.get("resident_plaintexts")}]
        if warmup_all_layers:
            # the rest of the chain on the dummy hidden state (limb chain is input-independent): every layer's
            # one-time costs -- and, under LLAMA_FHE_PREFILL_PLAIN_RESIDENT=1, every layer's encoded weight
            # plaintexts -- are paid here; the weights of layers > layers[0] are re-read by the prefetcher later
            for position in range(1, len(layers)):
                idx = layers[position]
                if position + 1 < len(layers):
                    prefetcher.start(layers[position + 1])
                warm_weights = prefetcher.get(idx)
                warm_outputs, warm_record = run_layer(idx, warm_outputs, warm_weights, verify=False,
                                                      tag_prefix=f"warmup/{idx}")
                warm_weights.release()
                del warm_weights
                warm_layers.append({"layer": int(idx), "wall_s": round(float(warm_record["layer_wall_s"]), 3),
                                    "resident_plaintexts": warm_record.get("resident_plaintexts")})
                if rank == 0:
                    print(json.dumps({"warmup_layer": idx, "wall_s": warm_layers[-1]["wall_s"],
                                      **({"resident": warm_record["resident_plaintexts"]}
                                         if warm_record.get("resident_plaintexts") else {})}), flush=True)
        _release_ciphers(warm_outputs)
        del warm_outputs, dummy_inputs, dummy_packed, dummy_rows
        easyfhe.cuda.synchronize()
        bootstrap.take()
        probe.reset_peak()
        warmup_report["layers"] = warm_layers
        if _PREFILL_RESIDENT:
            warmup_report["resident_plaintexts"] = RESIDENT.info()
        warmup_report.update({
            "layer": int(layers[0]),
            "seconds": round(time.perf_counter() - started, 3),
            "encrypt_seconds": round(encrypt_seconds, 3),
            "layer_wall_s": round(float(warm_record["layer_wall_s"]), 3),
            "bootstrap_calls": int(warm_record["bootstrap_calls"]),
            "bootstrap_seconds": float(warm_record["bootstrap_seconds"]),
            "stations": {name: round(float(v["wall_s"]), 4) for name, v in warm_record["stations"].items()},
            "keys_gib_loaded": round(sum(float(v["keys"].get("gib_loaded") or 0.0)
                                         for v in warm_record["stations"].values()), 3),
            "peak_alloc_gib": max((float(v["peak_alloc_gib"] or 0.0) for v in warm_record["stations"].values()),
                                  default=0.0),
        })
        probe.snap("05 warmup layer0", {k: v for k, v in warmup_report.items() if k != "stations"})
        if rank == 0:
            print(json.dumps({"warmup_layer0": {k: warmup_report[k] for k in ("layer", "seconds", "layer_wall_s",
                                                                              "bootstrap_calls", "keys_gib_loaded")}}),
                  flush=True)
    bootstrap_calls_before_model = bootstrap.calls
    transport.barrier("model/start")
    model_started = time.perf_counter()
    layer_records = []
    weight_seconds = []
    last = layers[-1]
    for position, idx in enumerate(layers):
        weights = first_weights if position == 0 else prefetcher.get(idx)
        first_weights = None
        if position + 1 < len(layers):
            prefetcher.start(layers[position + 1])
        verify = (idx == last) or (int(verify_every) > 0 and (position % int(verify_every) == 0))
        inputs, record = run_layer(idx, inputs, weights, verify=verify)
        record["weights_load_s"] = round(prefetcher.load_seconds.get(idx, 0.0), 3)
        record["weights_wait_s"] = round(prefetcher.wait_seconds.get(idx, 0.0), 3)
        record["weights_stage_s"] = round(weights.stage_seconds, 3)
        record["weights_read_s"] = round(weights.read_seconds, 3)
        weight_seconds.append(record["weights_wait_s"])
        weights.release()  # run_layer synchronised the device: the copies are no longer read
        del weights
        layer_records.append(record)
        if rank == 0:
            # chunks that went through the side-stream encode pipeline (--encode-depth > 0), all linear stations
            pipelined_chunks = sum(float(v) for st in record["stations"].values()
                                   for k, v in (st.get("stage_seconds") or {}).items() if k.endswith("pipelined_chunks"))
            print(json.dumps({"layer": idx, "wall_s": round(record["layer_wall_s"], 3),
                              "bs": record["bootstrap_calls"], "bs_s": record["bootstrap_seconds"],
                              "weights_wait_s": record["weights_wait_s"],
                              "w_dev": f"{record['weights_device_hits']}/{record['weights_device_misses']}",
                              **({"pipe_chunks": int(pipelined_chunks)} if pipelined_chunks else {}),
                              **({"rel_max_err": round(record["oracle"]["rel_max_err"], 6),
                                  "max_abs_err": round(record["oracle"]["max_abs_err"], 5)} if verify else {}),
                              **({"rss_gib": record["stations"]["residual_mlp"]["host_rss_gib"]} if not lite else {})}),
                  flush=True)
    easyfhe.cuda.synchronize()
    transport.barrier("model/end")
    model_wall = time.perf_counter() - model_started

    # -- final decrypt: this rank's token rows of hidden_{last+1}
    started = time.perf_counter()
    own_rows = _decrypt_rows(context, layout, ownership, inputs)
    decrypt_seconds = time.perf_counter() - started
    _release_ciphers(inputs)

    summary = {
        "rank": int(rank),
        "device_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ownership": ownership.to_dict(),
        "tiers": list(tiers),
        "layers": list(layers),
        "barrier_each": bool(barrier_each),
        "setup_seconds": state.setup_seconds,
        "warm_keys": warm_keys_report,
        "warmup_layer0": warmup_report,
        "stage_weights": stage_device is not None,
        "lite": bool(lite),
        "prefix_k": int(prefix_k),
        "prefix_kv_cache": None if prefix_kv_cache is None else str(prefix_kv_cache),
        "probe_layer": probe_layer,
        "encode_depth": int(os.environ.get("LLAMA_FHE_LINEAR_ENCODE_DEPTH", "0") or 0),
        "resident": {**resident, "pinned_keys": state.resident_report()["pinned_keys"]},
        "key_plan": key_plan,
        "rmsnorm_knobs": {"coeffs": rms_table_info, "exit_policy": rms_exit_policy},
        "softmax_post_exp": softmax_post_exp_description(),
        "softmax_knobs": {"coeffs": softmax_table_info, "degrees": softmax_table_degrees(softmax_table)},
        "silu_knobs": {"coeffs": silu_table_info,
                       "degrees": {k: len(v["tiers"]["standard"]["coefficients"]) - 1 for k, v in silu_all.items()},
                       "mlp_input_limbs": mlp_limbs_by_layer,
                       "mlp_input_limbs_from_silu": mlp_limbs_from_silu},
        "model_wall_s": model_wall,
        "decrypt_seconds": decrypt_seconds,
        "layer_records": layer_records,
        "bootstrap_calls_total": bootstrap.calls - bootstrap_calls_before_model,
        "transport": {
            **transport.stats.to_dict(),
            "backend": type(transport).__name__,
            **({"arena": dict(transport.arena.stats)} if hasattr(transport, "arena") else {}),
        },
        "own_rows": own_rows.astype(np.float32),
        "own_rows_oracle": _compare(own_rows, oracle_hidden[last][rows]),
        "timeline": probe.timeline,
    }
    gathered = transport.gather_to(0, "model/summary", summary)
    transport.barrier("model/final")
    transport.close()
    state.release()
    return gathered if rank == 0 else None


# ------------------------------------------------------------------- host
def _layer_table(per_rank, layers) -> list[dict]:
    """Per layer: rank-max wall, median station split, BS count, oracle error (worst over ranks)."""

    table = []
    for position, idx in enumerate(layers):
        records = [s["layer_records"][position] for s in per_rank]
        assert all(r["layer"] == idx for r in records)
        stations = {}
        for name in STATIONS:
            splits = [_attribution(r["stations"][name]) for r in records if name in r["stations"]]
            if splits:
                stations[name] = {"wall_med_s": round(_median(sp["wall_s"] for sp in splits), 3),
                                  "wall_max_s": round(max(sp["wall_s"] for sp in splits), 3),
                                  "bs_s": round(_median(sp["bs_s"] for sp in splits), 3),
                                  "wait_s": round(_median(sp["wait_s"] for sp in splits), 3),
                                  "bs": records[0]["stations"][name]["bootstrap_calls"]}
        row = {
            "layer": int(idx),
            "wall_max_s": round(max(r["layer_wall_s"] for r in records), 3),
            "wall_med_s": round(_median(r["layer_wall_s"] for r in records), 3),
            "station_sum_med_s": round(_median(r["station_wall_sum_s"] for r in records), 3),
            "bs": records[0]["bootstrap_calls"],
            "bs_s_med": round(_median(r["bootstrap_seconds"] for r in records), 3),
            "weights_wait_max_s": round(max(r["weights_wait_s"] for r in records), 3),
            "verified": bool(records[0]["verified"]),
            "stations": stations,
        }
        if records[0]["verified"]:
            row["verify_max_s"] = round(max(r.get("verify_seconds", 0.0) for r in records), 3)
            row["oracle_worst"] = {
                "rel_max_err": round(max(r["oracle"]["rel_max_err"] for r in records), 6),
                "max_abs_err": round(max(r["oracle"]["max_abs_err"] for r in records), 6),
                "oracle_max_abs": round(max(r["oracle"]["oracle_max_abs"] for r in records), 4),
                "finite": all(r["oracle"]["finite"] for r in records),
            }
        table.append(row)
    return table


def main() -> int:
    import numpy as np

    args = parse_args()
    preset = preset_by_name(args.preset)
    from dataclasses import replace

    if args.prefetch_mode is not None:
        preset = replace(preset, key_prefetch_mode=str(args.prefetch_mode))
    if args.mlp_shard_k is not None:
        preset = replace(preset, mlp_feature_shard_k=int(args.mlp_shard_k))
    preset = apply_runtime_knobs(args, preset)
    if args.input_level is None:
        args.input_level = int(preset.depth) - 2
    if args.world_size is not None:
        preset = preset.with_world_size(int(args.world_size),
                                        mlp_feature_shard_k=None if args.mlp_shard_k is None else int(args.mlp_shard_k))
    if args.model is not None:
        try:
            preset = preset.for_model(str(args.model))
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"--model {args.model}: {exc}") from exc
    num_layers = int(preset.model_spec.num_layers)
    if args.transport is not None:
        os.environ["T2_TRANSPORT"] = str(args.transport)
    expand_devices(preset.world_size, args.devices, ranks_per_device=int(args.ranks_per_device))
    if int(args.ranks_per_device) > 1:
        preset = preset.for_colocation(int(args.ranks_per_device))
    if args.max_plaintext_rows is not None:
        # After with_world_size / for_colocation: an explicit value also overrides their multi-slot 512 fallback.
        preset = replace(preset, linear_max_plaintext_rows=int(args.max_plaintext_rows))
    layers = parse_layers(f"0-{num_layers - 1}" if str(args.layers).lower() == "all" else args.layers, count=num_layers)
    if args.probe_layer is not None and args.probe_layer not in layers:
        raise SystemExit(f"--probe-layer {args.probe_layer} must be included in --layers")
    fixtures = Path(args.fixtures).resolve()
    manifest = json.loads((fixtures / "manifest.json").read_text())
    if str(manifest.get("model", "llama3-8b")) != str(preset.model_spec.name):
        raise SystemExit(f"fixtures {fixtures} were built for {manifest.get('model', 'llama3-8b')}, "
                         f"but the preset model is {preset.model_spec.name} (pass --model)")
    for idx in layers:
        if not (fixtures / f"layer{idx}_weights.npz").exists():
            raise SystemExit(f"missing {fixtures / f'layer{idx}_weights.npz'} (run bench/make_model_fixtures.py)")
    prefix_path = None if args.prefix_kv_cache is None else str(Path(args.prefix_kv_cache).resolve())
    if (prefix_path is not None) != bool(int(manifest.get("prefix_k", 0) or 0)):
        raise SystemExit(f"fixtures {fixtures} have prefix_k={manifest.get('prefix_k', 0)}: pass --prefix-kv-cache "
                         "together with a Prefix trajectory (make_model_fixtures.py --prefix-kv-cache), or neither")
    if prefix_path is not None and layers[0] != 0 and os.environ.get("T2_PREFIX_ANY_START", "0") != "1":
        # T2_PREFIX_ANY_START=1: single-layer diagnostics fed hidden_{first} of the Prefix trajectory
        raise SystemExit("Prefix Prefill must start at layer 0 (intermediate hidden states already carry the Prefix)")
    limb_silu_path = None if args.mlp_limbs_from_silu is None else str(Path(args.mlp_limbs_from_silu).expanduser().resolve())
    if limb_silu_path is not None and not Path(limb_silu_path).is_file():
        raise SystemExit(f"--mlp-limbs-from-silu: missing file {limb_silu_path}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else ROOT / "artifacts" / f"t2_model_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "fixtures": str(fixtures), "layers": layers, "input_level": int(args.input_level),
        "verify_every": int(args.verify_every), "barrier_each": bool(args.barrier_each),
        "prefetch_weights": not bool(args.no_prefetch_weights), "model_dir": args.model_dir,
        "warm_keys": bool(args.warm_keys), "warmup_layer0": bool(args.warmup_layer0),
        "warmup_all_layers": bool(args.warmup_all_layers),
        "stage_weights": not bool(args.no_stage_weights),
        "lite": bool(args.lite) or os.environ.get("T2_LITE", "0") == "1",
        "encode_depth": int(args.encode_depth),
        "prefix_kv_cache": prefix_path,
        "mlp_limbs_from_silu": limb_silu_path,
        "probe_layer": args.probe_layer,
    }
    if int(args.encode_depth) < 0:
        raise SystemExit("--encode-depth must be zero or positive")
    started = time.perf_counter()
    outcomes = spawn_workers(preset, worker_entry, kwargs=kwargs, devices=args.devices,
                             ranks_per_device=int(args.ranks_per_device), transport=args.transport)
    total = time.perf_counter() - started
    per_rank = outcomes[0].result
    assert per_rank is not None and len(per_rank) == preset.world_size
    per_rank.sort(key=lambda s: int(s["rank"]))

    # -- assemble hidden_{last+1} from the ranks' token rows and compare with the poly trajectory
    last = layers[-1]
    with np.load(fixtures / "hidden_trajectory.npz") as data:
        expected = np.asarray(data[f"hidden_{last + 1}"], dtype=np.float32)
        poly_logits = np.asarray(data["poly_logits"], dtype=np.float32) if "poly_logits" in data.files else None
    assembled = np.zeros_like(expected)
    for s in per_rank:
        own = s["ownership"]
        assembled[int(own["token_start"]):int(own["token_start"]) + int(own["token_count"])] = np.asarray(s.pop("own_rows"))
    final_error = _compare(assembled, expected)
    np.save(out_dir / f"hidden_{last + 1}_fhe.npy", assembled)

    logits_report = None
    if args.model_dir and last == num_layers - 1:
        from llama3fhe.application.checkpoint import CheckpointMetadata, WeightLoader
        from llama3fhe.application.client import final_logits, top_k_logits

        model_dir = Path(args.model_dir).resolve()
        metadata = CheckpointMetadata.from_json(model_dir / "config.json")
        with WeightLoader(model_dir, tensor_dtype=np.float32) as loader:
            t0 = time.perf_counter()
            fhe_logits = final_logits(assembled, loader=loader, metadata=metadata)
            logits_seconds = time.perf_counter() - t0
            ref_logits = poly_logits if poly_logits is not None else final_logits(expected, loader=loader, metadata=metadata)
        fhe_top = top_k_logits(fhe_logits, 10)
        ref_top = top_k_logits(ref_logits, 10)
        logits_report = {
            "seconds": round(logits_seconds, 3),
            "fhe_top_k": fhe_top, "poly_top_k": ref_top,
            "argmax_match": int(fhe_top[0]["token_id"]) == int(ref_top[0]["token_id"]),
            "top5_overlap": len({t["token_id"] for t in fhe_top[:5]} & {t["token_id"] for t in ref_top[:5]}),
            "logits_error": _compare(fhe_logits, ref_logits),
        }
        np.save(out_dir / "logits_fhe.npy", fhe_logits)

    table = _layer_table(per_rank, layers)
    walls = [row["wall_max_s"] for row in table]
    result = {
        "preset": preset.name,
        "model": preset.model_spec.name,
        "world_size": int(preset.world_size),
        "transport": per_rank[0]["transport"]["backend"],
        "layers": list(layers),
        "fixtures": str(fixtures),
        "manifest_source": manifest.get("source"),
        "timestamp": stamp,
        "total_wall_s": round(total, 2),
        "worker_wall_s": [round(o.wall_seconds, 2) for o in outcomes],
        "setup_seconds": per_rank[0]["setup_seconds"],
        "stage_weights": bool(per_rank[0].get("stage_weights")),
        "lite": bool(per_rank[0].get("lite")),
        "encode_depth": int(per_rank[0].get("encode_depth") or 0),
        "prefix_k": int(per_rank[0].get("prefix_k") or 0),
        "prefix_kv_cache": prefix_path,
        "mlp_input_limbs_from_silu": limb_silu_path,
        "mlp_input_limbs": per_rank[0]["silu_knobs"]["mlp_input_limbs"],
        "probe_layer": args.probe_layer,
        **runtime_knob_summary(preset),
        "warm_keys_seconds_max": (round(max(float(s["warm_keys"].get("seconds", 0.0)) for s in per_rank), 3)
                                  if per_rank[0]["warm_keys"].get("enabled") else None),
        "warmup_layer0": ({"seconds_max": round(max(float(s["warmup_layer0"]["seconds"]) for s in per_rank), 3),
                           "layer_wall_max_s": round(max(float(s["warmup_layer0"]["layer_wall_s"]) for s in per_rank), 3),
                           "stations_rank0": per_rank[0]["warmup_layer0"]["stations"]}
                          if per_rank[0]["warmup_layer0"].get("enabled") else None),
        "model_wall_s": {"max": round(max(s["model_wall_s"] for s in per_rank), 3),
                         "min": round(min(s["model_wall_s"] for s in per_rank), 3)},
        "layer_wall_sum_s": round(sum(walls), 3),
        "layer_wall_first_s": walls[0],
        "layer_wall_steady_med_s": round(_median(walls[1:]), 3) if len(walls) > 1 else None,
        "layer_wall_steady_max_s": round(max(walls[1:]), 3) if len(walls) > 1 else None,
        "bootstrap_calls_total": per_rank[0]["bootstrap_calls_total"],
        "decrypt_seconds_max": round(max(s["decrypt_seconds"] for s in per_rank), 3),
        "final_hidden_error": final_error,
        "final_hidden_error_per_rank": [s["own_rows_oracle"] for s in per_rank],
        "logits": logits_report,
        "args": vars(args),
        "table": table,
        "ranks": per_rank,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ("preset", "world_size", "transport", "total_wall_s", "model_wall_s",
                                              "layer_wall_sum_s", "layer_wall_first_s", "layer_wall_steady_med_s",
                                              "layer_wall_steady_max_s", "bootstrap_calls_total", "decrypt_seconds_max",
                                              "warm_keys_seconds_max", "stage_weights", "lite", "encode_depth")}))
    if result["warmup_layer0"] is not None:
        print(json.dumps({"warmup_layer0": {k: result["warmup_layer0"][k] for k in ("seconds_max", "layer_wall_max_s")}}))
    print(json.dumps({"final_hidden_error": {k: final_error[k] for k in ("max_abs_err", "mean_abs_err", "rms_err",
                                                                          "rel_max_err", "oracle_max_abs", "finite")}}))
    if logits_report is not None:
        print(json.dumps({"logits": {k: logits_report[k] for k in ("argmax_match", "top5_overlap", "seconds")},
                          "fhe_top5": logits_report["fhe_top_k"][:5], "poly_top5": logits_report["poly_top_k"][:5]}))
    for row in table:
        print(json.dumps({k: row[k] for k in row if k != "stations"}))
    print(json.dumps({"output": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
