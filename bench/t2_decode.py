#!/usr/bin/env python3
"""T2-Decode: one decode token through N transformer layers on a W-card host.

Every rank holds the whole DecodeFM hidden state (one ciphertext) and the
whole PairFold128 KV cache of every layer; the cheap stations run redundantly,
the three heavy ones are split (QKV diagonal chunks, attention ``(group,
query)`` streams, MLP pages / chunks) and glued with a ciphertext all-reduce
(:mod:`llama3fhe.parallel.decode.decode_stations`).  ``--world-size 1`` is the
single-card decode graph on the same code path.

    input_rmsnorm -> qkv_projection -> [qkv all-reduce] -> kv_extract -> kv_limb_drop -> kv_append
    -> decode_attention -> [folded-O all-reduce] -> folded_o_refresh -> folded_w_o (this rank's chunks)
    -> [W_O all-reduce] -> w_o_finalize -> attention_residual_add -> post_attention_bootstrap_rmsnorm -> mlp entry
    -> mlp ([down all-reduce], [gate/up partner all-reduce with W > 4]) -> mlp_residual

Inputs (plaintext, encrypted by every rank identically):

* ``--fixture fixtures/layer10_fixture.npz`` (default, ``--layers 10``): the
  layer-10 checkpoint weights, ``hidden_in[-1]`` as the decode hidden state
  and ``key_rope / value`` (128 prefill tokens) as the KV page;
* ``--fixtures-dir fixtures/model`` (``--layers all`` or ``0-31``): per-layer
  ``layer{i}_weights.npz`` plus ``hidden_trajectory.npz`` (``hidden_{i}[-1]``
  enters the first requested layer) and the KV of ``--poly-kv`` (``keys[i] /
  values[i]`` of the poly prefill) or of the layer fixture.

``--verify`` decrypts every layer output on every rank and compares with the
plaintext poly decode chain (``poly_decode_layer_numpy`` on its own outputs),
and rank 0 checks that all ranks decrypted the same values (the ranks must
stay bit-identical after every collective).  Rank 0 writes ``summary.json``.

Usage::

    # one H100 pretending to be 8 ranks (correctness; emulated per-rank time)
    python bench/t2_decode.py --preset h100-emu --emulate-single-device --world-size 8 --verify \
        --fixture fixtures/layer10_fixture.npz --output artifacts/t2_decode/emu_w8
    python bench/t2_decode.py --preset h100-emu --emulate-single-device --world-size 1 --verify ...
    # 8 x 4090
    python bench/t2_decode.py --preset rtx4090-low --world-size 8 --attention-limb-cap 16 --verify \
        --fixtures-dir fixtures/model --layers all --poly-kv /path/to/data/fixtures/8b/prefill_kv_poly.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llama3fhe.parallel.launch import expand_devices, spawn_workers  # noqa: E402
from llama3fhe.parallel.presets import preset_by_name  # noqa: E402
from llama3fhe.parallel.prefill.transport import TRANSPORT_BACKENDS  # noqa: E402

PROCESS_TRANSPORTS = tuple(kind for kind in TRANSPORT_BACKENDS if kind != "inprocess")

STATIONS = (
    "input_rmsnorm", "qkv_projection", "qkv_all_reduce", "kv_extract", "kv_limb_drop", "kv_append",
    "decode_attention", "folded_o_all_reduce", "folded_o_refresh", "folded_w_o", "w_o_all_reduce", "w_o_finalize",
    "attention_residual_add",
    "post_attention_bootstrap_rmsnorm", "mlp_input_modulus_drop", "pre_mlp_bootstrap", "mlp", "mlp_residual",
)
#: key tier a station leases (lease policy); the others only use resident T0 keys
STATION_TIERS = {
    "qkv_projection": "linear", "folded_w_o": "linear", "w_o_finalize": "linear", "mlp": "linear",
    "kv_extract": "attn_core", "decode_attention": "attn_core",
}
WEIGHT_KEYS = ("input_layernorm", "q_proj", "k_proj", "v_proj", "o_proj", "post_attention_layernorm",
               "gate_proj", "up_proj", "down_proj")
ROPE_THETA = 500000.0


def parse_layers(spec: str, *, num_layers: int = 32) -> tuple[int, ...]:
    spec = str(spec).strip().lower()
    if spec == "all":
        return tuple(range(num_layers))
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(part))
    if not layers or any(l < 0 or l >= num_layers for l in layers):
        raise ValueError(f"bad --layers {spec!r}")
    return tuple(layers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="rtx4090-low")
    parser.add_argument("--world-size", type=int, default=None,
                        help="ranks (1, 2, 4 or 8; default preset.world_size = 8).  1 = single-card decode graph")
    parser.add_argument("--devices", type=int, nargs="*", default=None)
    parser.add_argument("--ranks-per-device", type=int, default=1)
    parser.add_argument("--transport", choices=PROCESS_TRANSPORTS, default=None)
    parser.add_argument("--emulate-single-device", action="store_true",
                        help="all ranks as lock-stepped threads of one process on one card (key_policy=resident)")
    parser.add_argument("--layers", default="10", help="'10', '0-3', '0,5,31' or 'all'")
    parser.add_argument("--model", default=None,
                        help="model spec (llama3-8b | llama3.2-1b); selects Decode geometry and polynomial pack")
    parser.add_argument("--tokens", type=int, default=1,
                        help="decode tokens (each re-encrypts the same hidden input, appends to the caches)")
    parser.add_argument("--fixture", default=None, help="single-layer fixture .npz (weights + hidden_in + key_rope/value)")
    parser.add_argument("--fixtures-dir", default=None,
                        help="directory with layer{i}_weights.npz + hidden_trajectory.npz (bench/make_model_fixtures)")
    parser.add_argument("--model-dir", default=None,
                        help="optional checkpoint for final-head FHE/poly top-1 verification; requires --verify and the final layer")
    parser.add_argument("--poly-kv", default=None, help="paris128_prefill_kv_poly.npz (keys/values [32,128,1024])")
    parser.add_argument("--kv-limbs", type=int, default=12)
    parser.add_argument("--input-level", type=int, default=35,
                        help="levels consumed by the hidden input of the first layer (35 -> 3 limbs; RMSN1 refreshes)")
    parser.add_argument("--no-split-gate-up", action="store_true",
                        help="W=8: partners evaluate whole gate/up pages redundantly (skip the gu all-reduce)")
    parser.add_argument("--no-split-kv-refresh", action="store_true",
                        help="every rank bootstraps all four K/V ciphertexts of the new token itself "
                             "(default: one per rank + all-gather, tag kv_refresh)")
    parser.add_argument("--no-split-w-o", action="store_true",
                        help="every rank evaluates the whole W_O projection redundantly "
                             "(default: this rank's BSGS chunks + all-reduce, tag w_o)")
    parser.add_argument("--attention-limb-cap", type=int, default=None,
                        help="limb cap of the attn_core key tier (default: the preset's limb_caps['attn_core'], 13; "
                             "0 = full chain -- the measurement mode, see tables.required_limb_cap_by_tier; "
                             "the SL + sparse-BS layer traced 13 (kv_extract route, 2nd token) on the H100 W=8 emulation)")
    parser.add_argument("--attn-resident", action="store_true",
                        help="keep the attn_core tier resident under the lease policy (needs --attention-limb-cap)")
    parser.add_argument("--max-plaintext-rows", type=int, default=None)
    parser.add_argument("--prefetch-keys", action="store_true")
    parser.add_argument("--retain-weights", choices=LayerWeightSource.RETAIN_MODES, default="none",
                        help="what the weight loader keeps across tokens: none = one layer ahead from disk (the "
                             "8-card production shape; on one card the disk loader, ~1.1 s/layer, now trails the "
                             "0.8 s/layer compute: D1 token wall = layer sum + 5-11 s weight_wait); host = pinned "
                             "host copies stay (28 GB RAM, per-token side-stream re-upload); device = device "
                             "copies stay (28 GB of device memory, zero per-token weight work)")
    parser.add_argument("--lite", action="store_true",
                        help="skip the per-station memory / cache / host-RSS bookkeeping (17 stations x ~1.5 ms = "
                             "~0.027 s/layer of bench overhead in the D1 L10 rows; the station walls, BS records and "
                             "key-tracer counters stay)")
    parser.add_argument("--warm-keys", action="store_true",
                        help="upload every tier of the key plan at its cap before the token loop (same as "
                             "bench/t2_model.py --warm-keys).  Under key_policy=resident keys load lazily on first "
                             "use, so without it every *new* token column pays 1-2 kv_extract route-key H2D "
                             "(~0.045 s each at 13 limbs) in the first layer that touches it -- the whole "
                             "kv_extract wall of the single-layer benches (D1: 0.04-0.15 s/token on L10).")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="run the token chain this many times on fresh caches")
    parser.add_argument("--barrier-each", action="store_true")
    parser.add_argument("--no-stage-weights", action="store_true",
                        help="disable the loader-thread pin + side-stream upload of the next layer's weights "
                             "(DEVICE_WEIGHTS registry; every linear then does its own per-token weight_upload, "
                             "~0.09 s/layer on 8 x PRO 6000).  Staging is on by default on CUDA.")
    parser.add_argument("--softmax-sparse-bs", dest="softmax_sparse_bs", action="store_true", default=True,
                        help="(default) add the log_slots=12 sparse bootstrap program (preset.bootstrap_extra_sparse_log_slots) "
                             "so the merged-page Softmax runs its stride-periodic ``filled`` / ``lambda`` refreshes "
                             "through it (2 of the 3 steady-state Softmax BS per layer, 4-5 on layers 0/2/31; "
                             "sparse refreshes reduce bootstrap work).  "
                             "LLAMA_FHE_SOFTMAX_SPARSE_BS=0 keeps the program but forces the dense path (A/B).")
    parser.add_argument("--no-softmax-sparse-bs", dest="softmax_sparse_bs", action="store_false",
                        help="use dense Softmax refreshes only.")
    parser.add_argument("--softmax-schain", action="store_true", default=False,
                        help="LLAMA_FHE_SOFTMAX_SCHAIN=1: k=2 merged-page Softmax layers run the S-chain tail "
                             "(y0^4 * (lam1^2 lam2)^2, one co-packed sparse bootstrap instead of r1.lam + r2.lam; "
                             "decode_softmax.schain_limb_schedule).  Other layers keep Alg2.")
    parser.add_argument("--softmax-blockplan", default=None, metavar="SPEC",
                        help="LLAMA_FHE_SOFTMAX_BLOCKPLAN: k>=3 merged-page Softmax layers run the block plan "
                             "(e.g. '3:s2,d1;4:s2,d2': S-chain s2 block, then dB blocks = L2-normalise, "
                             "square B times; decode_softmax.blockplan_limb_schedule).  Layers whose k is not "
                             "listed keep Alg2.  '--softmax-blockplan-force' runs it even when the static "
                             "schedule predicts a probability refresh.")
    parser.add_argument("--softmax-blockplan-force", action="store_true", default=False,
                        help="LLAMA_FHE_SOFTMAX_BLOCKPLAN_FORCE=1 (see --softmax-blockplan)")
    parser.add_argument("--softmax-pv-need-limbs", type=int, default=None, metavar="N",
                        help="LLAMA_FHE_SOFTMAX_PV_NEED_LIMBS: PV refreshes the probability when it has <= N limbs "
                             "(default 6; 5 lets the 6-limb k=3 's2,d1' probability through without a refresh).")
    parser.add_argument("--no-shared-query-route", action="store_true",
                        help="disable serial Decode's per-GQA-group shared sparse route (A/B knob)")
    parser.add_argument("--no-rotate-add-fused", action="store_true",
                        help="disable EasyFHE fused rotate-add in Decode butterfly reductions (A/B knob)")
    parser.add_argument("--sl", dest="shifted_lanes", action="store_true", default=False,
                        help="shifted-lane (SL) decode packing for QKV and the MLP (LayerSchedule.decode_shifted_lanes): "
                             "512 instead of 4096 plaintext diagonals per 4096-block, +2 MLP levels "
                             "(docs/DESIGN.md).  W_O stays lane-replicated.")
    parser.add_argument("--sl-baby", type=int, default=None,
                        help="SL baby steps (LayerSchedule.decode_sl_baby_steps, default 32; 16 measured best per rank on W=8)")
    parser.add_argument("--output", default=None)
    _bench_dir = str(ROOT / "bench")
    if _bench_dir not in sys.path:
        sys.path.insert(0, _bench_dir)
    from t2_layer import add_runtime_knob_args

    # --rms-slim-period ROWS (periodic slim RMSNorm pack -> sparse rsqrt BS; the DecodeFM stream is one cipher,
    # so 1 is the natural period and the Decode default -- the preset's Prefill value (16) is not used here;
    # 0 = dense), --bs-cache-mode, --no-stage-sync, --attn-in-overlap (Prefill-only, inert here).
    add_runtime_knob_args(parser)
    args = parser.parse_args()
    if args.rms_slim_period is None:
        args.rms_slim_period = 1
    return args


# ------------------------------------------------------------------ inputs
def _load_layer_weights_npz(path) -> dict:
    import numpy as np

    with np.load(path) as payload:
        return {k: np.asarray(payload[k], dtype=np.float32) for k in WEIGHT_KEYS}


WEIGHT_UPLOAD_CHUNK_BYTES = 32 << 20  # split the side-stream H2D so the transport's pinned exchanges interleave


class LayerWeightSource:
    """Per-layer weights, loaded lazily one layer ahead of the station chain.

    A 32-layer model is 32 x 872 MB of float32 per rank; eight ranks holding
    all of it would need ~220 GB of host RAM.  Instead each rank keeps the
    current layer plus the next one: ``prefetch(layer)`` starts a loader
    thread (page cache is shared by the ranks, so the file is read from disk
    once), ``get(layer)`` joins it, ``drop(layer)`` frees the arrays.  Layers
    given up front (``preloaded``: the single ``--fixture`` case) are kept.

    With ``stage_device`` the loader thread also pins the seven arrays and
    uploads them on its own CUDA stream (same scheme as ``bench/t2_model.py``
    ``--stage-weights``); ``get`` publishes the device copies in
    :data:`~llama3fhe.operators.linear.weight.DEVICE_WEIGHTS`, so every
    linear station of the layer finds its page (full q/k/v/o, ``gate[a:b]``,
    ``down[:, a:b]``) already on the device and skips its own
    ``weight_upload`` (0.09 s/layer, 8 x PRO 6000 W=8 Decode: 4 linears x
    872 MB per token).  ``drop`` unregisters and frees the copies; preloaded
    layers stay registered for the whole run (resident weights).
    """

    RETAIN_MODES = ("none", "host", "device")

    def __init__(self, loader, *, preloaded: dict[int, dict] | None = None, stage_device: str | None = None,
                 retain: str = "none"):
        import threading

        self._loader = loader
        self._threading = threading
        self._resident: dict[int, dict] = dict(preloaded or {})
        self._pinned = frozenset(self._resident)
        self._pending: dict[int, tuple] = {}
        self._staged: dict[int, tuple[dict, object, dict]] = {}  # layer -> (device tensors, ready event, pinned)
        self._registered: set[int] = set()
        self._stage_device = None if stage_device is None else str(stage_device)
        self._stream = None
        if str(retain) not in self.RETAIN_MODES:
            raise ValueError(f"retain must be one of {self.RETAIN_MODES}, got {retain!r}")
        # ``retain``: what ``drop`` keeps for the next token.  "none" = the documented one-layer-ahead
        # disk loader (8-card production shape).  "host" = the pinned host copies stay (32 x 872 MB of host
        # RAM); the per-token loader thread only re-uploads them (~35 ms/layer on a side stream, hidden).
        # "device" = the device copies stay too (28 GB of device memory for 32 layers; the weights are
        # resident like a served model's).  D1 (single PRO 6000, 32 layers): the disk loader needs
        # ~1.0-1.2 s/layer while the layer compute dropped to 0.8 s, so the token wall grew 5-11 s of
        # ``weight_wait`` over the layer sum.
        self._retain = str(retain)
        self._retained_pinned: dict[int, dict] = {}  # layer -> pinned host tensors (retain="host")
        self.load_seconds: dict[int, float] = {}
        self.wait_seconds: dict[int, float] = {}
        self.stage_seconds: dict[int, float] = {}

    @property
    def staging(self) -> bool:
        return self._stage_device is not None

    def _stage(self, layer: int, arrays: dict, *, pinned: dict | None = None) -> dict:
        """Pin + upload ``arrays`` on the loader's stream; returns host views of the pinned buffers.

        ``pinned`` (retain="host" re-upload) skips the pinning and uploads the retained tensors.
        """

        import easyfhe
        import numpy as np

        started = time.perf_counter()
        if self._stream is None:
            self._stream = easyfhe.cuda.Stream()
        if pinned is None:
            pinned = {name: easyfhe.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).pin_memory()
                      for name, array in arrays.items()}
        device: dict[str, object] = {}
        with easyfhe.cuda.stream(self._stream):
            for name, tensor in pinned.items():
                target = easyfhe.empty(tensor.shape, dtype=tensor.dtype, device=self._stage_device)
                source_flat, target_flat = tensor.view(-1), target.view(-1)
                step = max(1, WEIGHT_UPLOAD_CHUNK_BYTES // int(tensor.element_size()))
                for start in range(0, int(source_flat.numel()), step):
                    target_flat[start:start + step].copy_(source_flat[start:start + step], non_blocking=True)
                device[name] = target
            ready = easyfhe.cuda.Event()
            ready.record(self._stream)
        host = {name: tensor.numpy() for name, tensor in pinned.items()}
        self._staged[int(layer)] = (device, ready, pinned)
        self.stage_seconds[int(layer)] = self.stage_seconds.get(int(layer), 0.0) + time.perf_counter() - started
        return host

    def prefetch(self, layer: int) -> None:
        layer = int(layer)
        if layer in self._resident or layer in self._pending:
            return
        box: dict = {}
        retained = self._retained_pinned.get(layer)

        def run():
            started = time.perf_counter()
            try:
                if retained is not None:
                    # retain="host": the pinned copies are still here, only the H2D upload is redone
                    box["value"] = self._stage(layer, {}, pinned=retained)
                else:
                    arrays = self._loader(layer)
                    box["value"] = self._stage(layer, arrays) if self.staging else arrays
            except BaseException as exc:  # noqa: BLE001 - re-raised by get()
                box["error"] = exc
            box["seconds"] = time.perf_counter() - started

        thread = self._threading.Thread(target=run, name=f"weights-l{layer}", daemon=True)
        thread.start()
        self._pending[layer] = (thread, box)

    def _register(self, layer: int) -> None:
        if layer in self._registered or layer not in self._staged:
            return
        from llama3fhe.operators.linear.weight import DEVICE_WEIGHTS

        device, ready, _pinned = self._staged[layer]
        host = self._resident[layer]
        for name, tensor in device.items():
            DEVICE_WEIGHTS.register(host[name], tensor, ready_event=ready)
        self._registered.add(layer)

    def _unregister(self, layer: int) -> None:
        if layer not in self._registered:
            return
        from llama3fhe.operators.linear.weight import DEVICE_WEIGHTS

        for name in self._staged[layer][0]:
            DEVICE_WEIGHTS.unregister(self._resident[layer][name])
        self._registered.discard(layer)

    def get(self, layer: int) -> dict:
        layer = int(layer)
        if layer in self._resident:
            if self.staging and layer not in self._staged:
                # preloaded (single-fixture) layer: stage once, keep resident for the whole run
                self._resident[layer] = self._stage(layer, self._resident[layer])
            self._register(layer)
            return self._resident[layer]
        if layer not in self._pending:
            self.prefetch(layer)
        thread, box = self._pending.pop(layer)
        waited = time.perf_counter()
        thread.join()
        self.wait_seconds[layer] = self.wait_seconds.get(layer, 0.0) + time.perf_counter() - waited
        if "error" in box:
            raise box["error"]
        self.load_seconds[layer] = self.load_seconds.get(layer, 0.0) + float(box["seconds"])
        self._resident[layer] = box["value"]
        self._register(layer)
        return self._resident[layer]

    def drop(self, layer: int) -> None:
        """Forget ``layer`` (device copies included); the caller has synchronised the device work that read them."""

        layer = int(layer)
        if layer in self._pinned or self._retain == "device":
            return
        self._unregister(layer)
        staged = self._staged.pop(layer, None)
        if self._retain == "host" and staged is not None:
            self._retained_pinned[layer] = staged[2]
        self._resident.pop(layer, None)

    @property
    def retain(self) -> str:
        return self._retain

    def close(self) -> None:
        for layer in list(self._registered):
            self._unregister(layer)
        self._staged.clear()


def load_inputs(*, fixture: str | None, fixtures_dir: str | None, poly_kv: str | None, layers: tuple[int, ...],
                stage_device: str | None = None, retain_weights: str = "none", model_spec=None):
    """Decode hidden state, lazy per-layer weights, and plaintext Prefill KV pages."""

    import numpy as np

    preloaded: dict[int, dict] = {}
    loader = None
    kv: dict[int, tuple] = {}
    hidden = None
    if fixtures_dir:
        base = Path(fixtures_dir)
        for layer in layers:
            if not (base / f"layer{layer}_weights.npz").is_file():
                raise SystemExit(f"missing {base / f'layer{layer}_weights.npz'}")
        loader = lambda layer: _load_layer_weights_npz(base / f"layer{layer}_weights.npz")  # noqa: E731
        with np.load(base / "hidden_trajectory.npz") as payload:
            hidden = np.asarray(payload[f"hidden_{layers[0]}"], dtype=np.float32)[-1:]
    if fixture:
        with np.load(fixture) as payload:
            fixture_weights = {k: np.asarray(payload[k], dtype=np.float32) for k in WEIGHT_KEYS}
            fixture_hidden = np.asarray(payload["hidden_in"], dtype=np.float32)[-1:]
            fixture_kv = (np.asarray(payload["key_rope"], dtype=np.float32), np.asarray(payload["value"], dtype=np.float32))
        if loader is None:
            for layer in layers:
                preloaded.setdefault(layer, fixture_weights)
        for layer in layers:
            kv.setdefault(layer, fixture_kv)
        if hidden is None:
            hidden = fixture_hidden
    weights = LayerWeightSource(loader, preloaded=preloaded, stage_device=stage_device,
                                retain=retain_weights if stage_device is not None else "none")
    if poly_kv:
        with np.load(poly_kv) as payload:
            keys = np.asarray(payload["keys"], dtype=np.float32)
            values = np.asarray(payload["values"], dtype=np.float32)
        for layer in layers:
            kv[layer] = (keys[layer], values[layer])
    missing = [l for l in layers if (loader is None and l not in preloaded) or l not in kv]
    if missing or hidden is None:
        raise SystemExit(f"no weights / KV for layers {missing} (give --fixture, or --fixtures-dir with --poly-kv)")
    if model_spec is not None:
        if hidden.shape != (1, int(model_spec.hidden_dim)):
            raise ValueError(f"decode hidden shape {hidden.shape} does not match {model_spec.name}")
        kv_width = int(model_spec.key_value_heads) * int(model_spec.head_dim)
        for layer in layers:
            key, value = kv[layer]
            if key.shape != value.shape or key.ndim != 2 or key.shape[1] != kv_width or not 1 <= key.shape[0] <= 128:
                raise ValueError(f"layer {layer} KV shape {key.shape}/{value.shape} does not match {model_spec.name}")
    return hidden, weights, kv


# ------------------------------------------------------------------ worker
class _BootstrapTimer:
    """Times every ``refresh`` of the rank's ``BootstrapOperator`` (see bench/t2_layer.py)."""

    def __init__(self, inner, *, shared: bool = False):
        self._inner = inner
        self.records: list[tuple[float, int, int, int, str | None]] = []
        self._shared = bool(shared)
        self._calls = 0
        self._native_calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def calls(self) -> int:
        return self._calls if self._shared else int(self._inner.calls)

    @property
    def native_calls(self) -> int:
        return self._native_calls if self._shared else int(self._inner.native_calls)

    def refresh(self, cipher, **kwargs):
        import easyfhe

        synchronize = bool(kwargs.pop("synchronize", True))
        if synchronize:
            easyfhe.cuda.synchronize()
        started = time.perf_counter()
        native_before = int(self._inner.native_calls)
        result = self._inner.refresh(cipher, synchronize=synchronize, **kwargs)
        if synchronize:
            easyfhe.cuda.synchronize()
        self._calls += 1
        self._native_calls += int(self._inner.native_calls) - native_before
        from llama3fhe.operators.nonlinear.bootstrap import current_bootstrap_tag

        self.records.append((time.perf_counter() - started, int(cipher.state.cur_limbs),
                             int(kwargs.get("iterations") or self._inner.config.iterations),
                             int(result.state.cur_limbs), current_bootstrap_tag()))
        return result

    def refresh_many(self, ciphers, **kwargs):
        import easyfhe

        from llama3fhe.operators.nonlinear.bootstrap import current_bootstrap_tag

        ciphers = list(ciphers)
        synchronize = bool(kwargs.pop("synchronize", True))
        if synchronize:
            easyfhe.cuda.synchronize()
        started = time.perf_counter()
        native_before = int(self._inner.native_calls)
        results = list(self._inner.refresh_many(ciphers, synchronize=synchronize, **kwargs))
        if synchronize:
            easyfhe.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self._calls += len(results)
        self._native_calls += int(self._inner.native_calls) - native_before
        iterations = int(kwargs.get("iterations") or self._inner.config.iterations)
        tag = current_bootstrap_tag()
        share = elapsed / max(1, len(results))
        for cipher, result in zip(ciphers, results, strict=True):
            self.records.append((share, int(cipher.state.cur_limbs), iterations,
                                 int(result.state.cur_limbs), tag))
        return results

    def take(self) -> list[tuple[float, int, int, int, str | None]]:
        taken, self.records = self.records, []
        return taken


def decode_layouts(preset=None):
    from llama3fhe.layouts.attention import AttentionPairLayout
    from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
    from llama3fhe.layouts.pairfold128 import pairfold_layout_for
    from llama3fhe.model_spec import LLAMA3_8B

    spec = LLAMA3_8B if preset is None else preset.model_spec
    slots = 32768 if preset is None else int(preset.slots)
    return (
        FeatureMajorPrefillLayout(seq_len=1, hidden_dim=int(spec.hidden_dim), slots=slots),
        AttentionPairLayout(seq_len=128, head_dim=int(spec.head_dim), query_heads=int(spec.query_heads),
                            key_value_heads=int(spec.key_value_heads), slots=slots),
        pairfold_layout_for(head_dim=int(spec.head_dim), query_heads=int(spec.query_heads),
                            key_value_heads=int(spec.key_value_heads), slots=slots),
    )


def decode_table_sources(preset):
    from dataclasses import replace
    from llama3fhe.application.args import model_poly_paths

    if preset is None:
        return model_poly_paths(None), None
    paths = model_poly_paths(preset.model)
    if preset.model_spec.name == "llama3-8b":
        return paths, preset
    # Card presets may select 8B-only polynomial overlays; other models start
    # from their own calibrated pack. Explicit T2_* environment overrides still apply.
    return paths, replace(preset, rmsnorm_coeffs=None, silu_coeffs=None, softmax_coeffs=None)


def decode_schedule(preset, *, kv_limbs: int):
    """The decode LayerSchedule of ``preset``; the SL switches travel through the
    environment so the spawned rank processes see the same policy
    (``LLAMA_FHE_DECODE_SL=1``, ``LLAMA_FHE_DECODE_SL_BABY=<n>``)."""

    from dataclasses import replace

    schedule = replace(preset.layer_schedule(), kv_limbs=int(kv_limbs))
    if os.environ.get("LLAMA_FHE_DECODE_SL", "0") == "1":
        baby = int(os.environ.get("LLAMA_FHE_DECODE_SL_BABY", str(schedule.decode_sl_baby_steps)))
        schedule = replace(schedule, decode_shifted_lanes=True, decode_sl_baby_steps=baby)
    return schedule


def decode_linear_floor(schedule, layers, *, silu_preset=None) -> int:
    """``linear`` key-tier limb floor for the layers that run (per-layer SiLU depth, see decode_stations).

    ``silu_preset`` selects the SiLU table the way the worker does (``preset.silu_coeffs`` /
    ``T2_SILU_COEFFS``); ``None`` keeps the production table."""

    from llama3fhe.application.approx_config import ApproximationConfig
    from llama3fhe.parallel.decode.decode_stations import decode_linear_limb_floor

    paths, table_preset = decode_table_sources(silu_preset)
    approximations = ApproximationConfig(rmsnorm_path=paths[0], silu_path=paths[1], softmax_path=paths[2])
    if silu_preset is not None:
        if str(ROOT / "bench") not in sys.path:
            sys.path.insert(0, str(ROOT / "bench"))
        from t2_layer import silu_coeffs_table

        table, _info = silu_coeffs_table(paths[1], table_preset)
        approximations._silu = {"per_layer": table}
    return decode_linear_limb_floor(schedule, approximations, layers)


def resolve_attention_limb_cap(preset, attention_limb_cap: int | None) -> int | None:
    """``None`` -> the preset's ``attn_core`` cap (13); ``0`` -> uncapped (measurement mode); else as given."""

    if attention_limb_cap is None:
        cap = preset.limb_caps.get("attn_core")
        return None if cap is None else int(cap)
    return None if int(attention_limb_cap) <= 0 else int(attention_limb_cap)


def decode_key_plan(preset, *, kv_limbs: int, attention_limb_cap: int | None, attn_resident: bool, layers=(10,),
                    silu_preset=None):
    from llama3fhe.parallel.decode.decode_keys import build_decode_key_plan

    attention_limb_cap = resolve_attention_limb_cap(preset, attention_limb_cap)
    feature_layout, attention_layout, pairfold_layout = decode_layouts(preset)
    resident_extra = tuple(preset.key_resident_extra)
    if attn_resident and "attn_core" not in resident_extra:
        resident_extra = resident_extra + ("attn_core",)
    schedule = decode_schedule(preset, kv_limbs=kv_limbs)
    return build_decode_key_plan(
        preset, schedule=schedule, feature_layout=feature_layout,
        attention_layout=attention_layout, pairfold_layout=pairfold_layout,
        attention_limb_cap=attention_limb_cap, resident_extra=resident_extra,
        linear_limb_floor=decode_linear_floor(schedule, layers, silu_preset=silu_preset),
    )


def worker_entry(rank: int, preset, endpoints, *, layers: tuple[int, ...], tokens: int, fixture: str | None,
                 fixtures_dir: str | None, poly_kv: str | None, kv_limbs: int, input_level: int,
                 split_gate_up: bool, split_kv_refresh: bool, split_w_o: bool, attention_limb_cap: int | None,
                 attn_resident: bool, prefetch_keys: bool,
                 verify: bool, repeat: int, barrier_each: bool, stage_weights: bool = True, emulation=None,
                 warm_keys: bool = False, lite: bool = False, retain_weights: str = "none",
                 model_dir: str | None = None):
    import numpy as np

    import easyfhe

    # Prefill-side engineering (Sylph kernels / fast encode / fused encode stage 2 / multi MAC), env-gated,
    # installed before the context exists -- see llama3fhe/acceleration/patches.py.  The Decode key plan has no hoisted
    # giant keys, so ODIN_MULTI_GIANT_HOIST is forced off for this process.
    from llama3fhe.acceleration import patches as acceleration_patches
    os.environ["ODIN_MULTI_GIANT_HOIST"] = "0"
    patches = acceleration_patches.install_from_env(rank)

    from llama3fhe.application.approx_config import ApproximationConfig
    from llama3fhe.approx.attention import set_default_rope_scaling
    from llama3fhe.approx.layer import poly_decode_layer_numpy
    from llama3fhe.backend import plaintext_cache_snapshot, release_all
    from llama3fhe.graph import LayerWeights
    from llama3fhe.operators.attention.pairfold.pairfold_prefill import encrypt_pairfold_plaintext_page
    from llama3fhe.operators.pairfold_kv_cache import PairFoldLayerCache
    from llama3fhe.parallel.decode.decode_collectives import broadcast_ciphers
    from llama3fhe.parallel.decode.decode_keys import DECODE_TIERS
    from llama3fhe.parallel.decode.decode_ownership import DecodeRankOwnership
    from llama3fhe.parallel.decode.decode_stations import decode_layer_parallel
    from llama3fhe.parallel.memory import native_cache_inventory
    from llama3fhe.parallel.prefill.transport import build_transport
    from llama3fhe.parallel.worker import build_worker_state, host_rss_gib, load_layer_config

    layers = tuple(int(l) for l in layers)
    set_default_rope_scaling(preset.model_spec.rope_scaling)
    emulated = emulation is not None
    plan = decode_key_plan(preset, kv_limbs=kv_limbs, attention_limb_cap=attention_limb_cap, attn_resident=attn_resident,
                           layers=layers, silu_preset=preset)
    if emulated:
        from dataclasses import replace as _replace

        from llama3fhe.parallel.memory import MemoryProbe

        transport = emulation.transport
        state = _replace(emulation.shared_state, rank=int(rank), transport=transport, memory=MemoryProbe(rank=int(rank)))
        state.memory.reset_peak()
    else:
        transport = build_transport(rank, endpoints)
        state = build_worker_state(rank, preset, transport=transport, tiers=DECODE_TIERS, trace_keys=True, key_plan=plan)
    probe, context, native, life, tracer = state.memory, state.context, state.native, state.keys, state.tracer
    bootstrap = _BootstrapTimer(state.bootstrap_operator, shared=emulated)
    stage_prefix = f"r{rank}." if emulated else ""
    world_size = int(transport.world_size)
    config = load_layer_config(preset)
    schedule = decode_schedule(preset, kv_limbs=kv_limbs)
    feature_layout, attention_layout, pairfold_layout = decode_layouts(preset)
    ownership = DecodeRankOwnership(
        int(rank), world_size, key_value_groups=int(pairfold_layout.key_value_groups),
        gqa_ratio=int(pairfold_layout.gqa_ratio),
        mlp_pages=(int(preset.model_spec.intermediate_dim) + int(feature_layout.hidden_dim) - 1)
                  // int(feature_layout.hidden_dim),
    )
    resident = state.resident_report()
    key_plan_summary = state.key_plan.summary(alpha=int(native.alpha), special_limbs=int(native.K))
    # rotations of the level-0 tiers (bootstrap / rmsnorm) always carry the full chain; the limb cap a
    # station really needs is the largest limb count it requested on any *other* rotation.  ``attn_core``
    # is measured even while it runs uncapped (``--attention-limb-cap`` None): that is how its cap is found.
    uncapped_rotations: set[int] = set()
    rotation_tiers: dict[int, str] = {}
    for tier_record in state.key_plan.tiers:
        if int(tier_record.level) == 0:
            uncapped_rotations.update(int(r) for r in tier_record.rotations)
        else:
            for r in tier_record.rotations:
                rotation_tiers.setdefault(int(r), str(tier_record.name))

    from llama3fhe.operators.linear.resident_plaintexts import RESIDENT
    from llama3fhe.operators.linear.weight import DEVICE_WEIGHTS

    stage_device = str(context.device) if (stage_weights and str(context.device).startswith("cuda")) else None
    hidden0, weight_source, kv_arrays = load_inputs(fixture=fixture, fixtures_dir=fixtures_dir, poly_kv=poly_kv, layers=layers,
                                                    stage_device=stage_device, retain_weights=str(retain_weights),
                                                    model_spec=preset.model_spec)
    terminal_head = None
    if model_dir is not None and int(rank) == 0:
        from llama3fhe.application.checkpoint import CheckpointMetadata, WeightLoader
        from llama3fhe.application.client import ClientHead

        checkpoint = Path(model_dir)
        metadata = CheckpointMetadata.from_json(checkpoint / "config.json")
        with WeightLoader(checkpoint, tensor_dtype=np.float32) as loader:
            terminal_head = ClientHead(loader, metadata)
    paths, table_preset = decode_table_sources(preset)
    approximations = ApproximationConfig(rmsnorm_path=paths[0], silu_path=paths[1], softmax_path=paths[2])
    # Same table knobs as bench/t2_layer.py: preset.rmsnorm_coeffs / softmax_coeffs
    # (rtx6000pro-high: reltol5e-3 rsqrt + joint_tol_v1 exp) or T2_RMS_COEFFS /
    # T2_SOFTMAX_COEFFS (+ *_LAYERS / T2_SOFTMAX_PARTS).  The poly reference used by
    # --verify is built from the same patched ``approximations`` object.
    if str(ROOT / "bench") not in sys.path:
        sys.path.insert(0, str(ROOT / "bench"))
    from t2_layer import rmsnorm_coeffs_table, silu_coeffs_table, softmax_coeffs_table
    from llama3fhe.approx.softmax import SoftmaxPolyConfig

    rms_table, rms_table_info = rmsnorm_coeffs_table(paths[0], table_preset)
    softmax_table, softmax_table_info = softmax_coeffs_table(paths[2], table_preset)
    # SiLU table follows the preset / T2_SILU_COEFFS like the Prefill benches (E8: joint_tol_v1 is the
    # rtx6000pro-high default, depth 6 -> 4/5 so the MLP enters 1-2 limbs lower; ``T2_SILU_COEFFS=default``
    # reproduces the old table).  ``decode_key_plan`` derives its linear limb floor from the same table.
    silu_table, silu_table_info = silu_coeffs_table(paths[1], table_preset)
    approximations._rmsnorm = rms_table
    approximations._softmax = SoftmaxPolyConfig.from_artifact(softmax_table)
    approximations._silu = {"per_layer": silu_table}
    approximations._layers.clear()
    warm_keys_report: dict = {"enabled": bool(warm_keys)}
    if warm_keys:
        started = time.perf_counter()
        per_tier: dict[str, dict[str, float]] = {}
        for tier_name in state.key_plan.names:
            count, seconds = life.prefetch(tier_name)
            per_tier[str(tier_name)] = {"keys": int(count), "seconds": round(float(seconds), 3)}
        easyfhe.cuda.synchronize()
        warm_keys_report.update({"seconds": round(time.perf_counter() - started, 3), "tiers": per_tier,
                                 "resident_keys_gib": native_cache_inventory(native)["total_gib"]})
        probe.snap("04 warm keys", warm_keys_report)
        if tracer is not None:
            for row in tracer.stages.values():
                row.requests = 0; row.misses = 0; row.loads = 0; row.bytes_loaded = 0; row.load_seconds = 0.0
                row.miss_log.clear()
    weight_source.prefetch(layers[0])
    position0 = int(kv_arrays[layers[0]][0].shape[0])
    kv_level = int(native.L) - int(kv_limbs)
    hidden_stub = np.zeros((1, int(feature_layout.hidden_dim)), dtype=np.float32)

    def _stage_dict(values) -> dict[str, float]:
        return {str(k): round(float(v), 4) for k, v in dict(values or {}).items()}

    input_comm: dict[str, float] = {}

    def encrypt_hidden(hidden):
        """Rank 0 encrypts the client's hidden state; everybody gets the *same* ciphertexts."""
        own = None
        if int(rank) == 0:
            packed = feature_layout.pack(np.asarray(hidden, dtype=np.float32), dtype=np.float64)
            own = [context.encrypt(np.asarray(row).reshape(-1), level=int(input_level), slots=int(feature_layout.slots))
                   for row in packed]
        return tuple(broadcast_ciphers(transport, "input_hidden", own, root=0, crypto_context=context, comm=input_comm))

    def decrypt_hidden(ciphers):
        packed = np.stack([np.asarray(context.decrypt(c), dtype=np.float64) for c in ciphers])
        return np.asarray(feature_layout.unpack(packed), dtype=np.float32)

    def build_caches():
        """Rank 0 encrypts the prefill KV page of every layer and broadcasts it (K, K, V, V ciphertexts)."""
        from llama3fhe.operators.pairfold_kv_cache import PairFoldCipherPage

        caches = {}
        for layer in layers:
            key, value = kv_arrays[layer]
            occupied = int(key.shape[0])
            own = None
            if int(rank) == 0:
                page = encrypt_pairfold_plaintext_page(key, value, pairfold_layout=pairfold_layout, crypto_context=context, level=kv_level)
                occupied = int(page.occupied)
                own = list(page._detach())  # hand the ciphertexts over; the page wrapper is rebuilt below
            handles = broadcast_ciphers(transport, f"input_kv_{layer}", own, root=0, crypto_context=context, comm=input_comm)
            groups = int(pairfold_layout.key_value_groups)
            page = PairFoldCipherPage(key_groups=tuple(handles[:groups]), value_groups=tuple(handles[groups:]), occupied=occupied)
            caches[layer] = PairFoldLayerCache(pages=(page,), content_tokens=int(page.occupied))
        easyfhe.cuda.synchronize()
        return caches

    iterations: list[dict] = []
    for iteration in range(int(repeat)):
        caches = build_caches()
        poly_kv = {l: (np.array(kv_arrays[l][0]), np.array(kv_arrays[l][1])) for l in layers}
        token_rows: list[dict] = []
        try:
            for token in range(int(tokens)):
                position = position0 + token
                hidden_ciphers = encrypt_hidden(hidden0)
                poly_hidden = np.asarray(hidden0, dtype=np.float32)
                layer_rows: list[dict] = []
                token_start = time.perf_counter()
                for layer_index, layer in enumerate(layers):
                    registry_hits_before, registry_misses_before = DEVICE_WEIGHTS.hits, DEVICE_WEIGHTS.misses
                    layer_weights = LayerWeights(**weight_source.get(layer))
                    if layer_index + 1 < len(layers):
                        weight_source.prefetch(layers[layer_index + 1])
                    elif token + 1 < int(tokens) or iteration + 1 < int(repeat):
                        weight_source.prefetch(layers[0])  # next token starts at the first layer
                    stations: dict[str, dict] = {}
                    order = [20]
                    lease_tier = [None]

                    def timed(name: str, fn):
                        tier = STATION_TIERS.get(name)
                        if barrier_each:
                            transport.barrier(f"decode/{iteration}/{token}/{layer}/{name}")
                        if not lite:
                            probe.reset_peak()
                        easyfhe.cuda.synchronize()
                        calls_before = bootstrap.calls
                        bootstrap.take()
                        stage_name = f"{stage_prefix}{name}_{iteration}_{token}_{layer}"
                        if lite:
                            started = time.perf_counter()
                            lease_record = None
                            with tracer.stage(stage_name):
                                if tier is None:
                                    value = fn()
                                else:
                                    with life.lease(tier, prefetch=bool(prefetch_keys)) as lease_record:
                                        value = fn()
                            easyfhe.cuda.synchronize()
                            wall = time.perf_counter() - started
                            order[0] += 1
                            stats = tracer.stages[stage_name].as_row()
                            bs_records = bootstrap.take()
                            stations[name] = {
                                "wall_s": round(wall, 4),
                                "tier": tier,
                                "bootstrap_calls": bootstrap.calls - calls_before,
                                "bootstrap_seconds": round(sum(r[0] for r in bs_records), 4),
                                "bootstrap_records": [{"s": round(r[0], 4), "in_limbs": r[1], "iterations": r[2], "out_limbs": r[3], "tag": r[4]} for r in bs_records],
                                "keys": {k: stats.get(k) for k in ("requests", "misses", "gib_loaded", "load_seconds",
                                                                     "max_limbs", "distinct_rotations")},
                                "key_misses": [list(map(int, m)) for m in tracer.stages[stage_name].miss_log[:16]],
                                "lite": True,
                            }
                            return value
                        transport_before = transport.stats.to_dict()
                        plain_before = plaintext_cache_snapshot(context)
                        started = time.perf_counter()
                        lease_record = None
                        with tracer.stage(stage_name):
                            if tier is None:
                                value = fn()
                            else:
                                with life.lease(tier, prefetch=bool(prefetch_keys)) as lease_record:
                                    value = fn()
                        easyfhe.cuda.synchronize()
                        wall = time.perf_counter() - started
                        row = probe.snap(f"{order[0]} {stage_name}", {"wall_s": round(wall, 4)})
                        order[0] += 1
                        stats = tracer.stages[stage_name].as_row()
                        transport_after = transport.stats.to_dict()
                        plain_after = plaintext_cache_snapshot(context)
                        bs_records = bootstrap.take()
                        limb_hist: dict[int, int] = {}
                        required_cap = 0
                        tier_caps: dict[str, int] = {}
                        for rotation, limbs in tracer.stages[stage_name].max_limbs.items():
                            limb_hist[int(limbs)] = limb_hist.get(int(limbs), 0) + 1
                            if int(rotation) not in uncapped_rotations:
                                required_cap = max(required_cap, int(limbs))
                                owner = rotation_tiers.get(int(rotation), "?")
                                tier_caps[owner] = max(tier_caps.get(owner, 0), int(limbs))
                        stations[name] = {
                            "wall_s": round(wall, 4),
                            "tier": tier,
                            "bootstrap_calls": bootstrap.calls - calls_before,
                            "bootstrap_seconds": round(sum(r[0] for r in bs_records), 4),
                            "bootstrap_records": [{"s": round(r[0], 4), "in_limbs": r[1], "iterations": r[2], "out_limbs": r[3], "tag": r[4]} for r in bs_records],
                            "peak_alloc_gib": row.get("peak_alloc_gib"),
                            "alloc_after_gib": row.get("alloc_gib"),
                            "reserved_after_gib": row.get("reserved_gib"),
                            "cache_after_gib": native_cache_inventory(native)["total_gib"],
                            "host_rss_gib": round(host_rss_gib(), 2),
                            "keys": {k: stats.get(k) for k in ("requests", "misses", "gib_loaded", "load_seconds",
                                                                 "max_limbs", "distinct_rotations")},
                            # (rotation, limbs, beta) of the lazy loads: a steady-state miss here is a key the
                            # resident plan should have held (8 x Pro 6000 D_l10: kv_extract paid 2 x 0.06 s/layer).
                            "key_misses": [list(map(int, m)) for m in tracer.stages[stage_name].miss_log[:16]],
                            "key_limb_histogram": {str(k): v for k, v in sorted(limb_hist.items())},
                            "required_limb_cap": required_cap,
                            "required_limb_cap_by_tier": tier_caps,
                            "transport_delta": {k: round(transport_after[k] - transport_before[k], 4)
                                                for k in ("recv_wait_seconds", "send_seconds", "bytes_sent", "bytes_received")},
                            "lease": None if lease_record is None else {
                                "prefetch_s": round(lease_record.prefetch_seconds, 4), "prefetched_keys": lease_record.prefetched_keys,
                                "evict_s": round(lease_record.evict_seconds, 4), "evicted": lease_record.evicted_entries},
                            "plain_cache": {**{k: int(plain_after[k] - plain_before[k]) for k in ("hits", "misses", "skips")},
                                            "entries": int(plain_after["entries"]), "gib": round(plain_after["bytes"] / 2**30, 3)},
                        }
                        return value

                    layer_start = time.perf_counter()
                    result = decode_layer_parallel(
                        crypto_context=context, transport=transport, ownership=ownership,
                        weights=layer_weights, approximations=approximations.for_layer(layer),
                        schedule=schedule, config=config, input_ciphers=hidden_ciphers, kv_cache=caches[layer],
                        position=position, layer_idx=layer, bootstrap_operator=bootstrap,
                        feature_layout=feature_layout, attention_layout=attention_layout, pairfold_layout=pairfold_layout,
                        rope_theta=float(preset.model_spec.rope_theta), prefix_bundle=None, split_gate_up=bool(split_gate_up),
                        split_kv_refresh=bool(split_kv_refresh), split_w_o=bool(split_w_o),
                        hidden_stub=hidden_stub, timed=timed,
                        rms_slim_period_rows=getattr(preset, "rmsnorm_slim_period_rows", None),
                    )
                    layer_wall = time.perf_counter() - layer_start
                    release_all(hidden_ciphers)
                    hidden_ciphers = result.ciphers
                    result.ciphers = ()
                    layer_row = {
                        "layer": int(layer), "position": int(position), "wall_s": round(layer_wall, 4),
                        "stations": stations,
                        "station_wall_sum_s": round(sum(s["wall_s"] for s in stations.values()), 4),
                        "bootstrap_calls": dict(result.bootstrap_calls),
                        "bootstrap_calls_total": int(sum(result.bootstrap_calls.values())),
                        "bootstrap_seconds": round(sum(s["bootstrap_seconds"] for s in stations.values()), 4),
                        "comm": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.comm.items()},
                        "comm_bytes_sent": int(sum(v for k, v in result.comm.items() if k.endswith("_sent"))),
                        "comm_seconds": round(sum(v for k, v in result.comm.items() if k.endswith("_seconds")), 4),
                        "output_levels": {k: list(v) if isinstance(v, tuple) else v for k, v in result.output_levels.items()},
                        "nested": {k: v for k, v in result.nested.items()
                                   if k in ("mlp_input_modulus_drop", "folded_o_refresh", "pairfold_kv_append", "ownership")},
                        "nested_stages": {k: _stage_dict(v) for k, v in result.nested.items()
                                          if k in ("input_rmsnorm", "qkv_projection", "decode_attention",
                                                   "post_attention_bootstrap_rmsnorm", "mlp", "folded_w_o")},
                        "output_limbs": [int(c.state.cur_limbs) for c in hidden_ciphers],
                    }
                    if patches["fast_encode"] is not None:
                        # deferred encoder range check (one device read per layer; raises on overflow)
                        layer_row["fast_encode"] = patches["fast_encode"].check()
                    if verify:
                        started = time.perf_counter()
                        fhe_hidden = decrypt_hidden(hidden_ciphers)
                        poly = poly_decode_layer_numpy(
                            poly_hidden, cache_key=poly_kv[layer][0], cache_value=poly_kv[layer][1],
                            weights=layer_weights, approximations=approximations.for_layer(layer),
                            attention_layout=attention_layout, layer_idx=int(layer), position=int(position),
                            rope_theta=float(preset.model_spec.rope_theta), use_alg2_softmax=True,
                        )
                        poly_kv[layer] = (np.concatenate([poly_kv[layer][0], poly.key], axis=0),
                                          np.concatenate([poly_kv[layer][1], poly.value], axis=0))
                        poly_hidden = np.asarray(poly.hidden, dtype=np.float32)
                        diff = fhe_hidden.astype(np.float64) - poly_hidden.astype(np.float64)
                        layer_row["verify"] = {
                            "seconds": round(time.perf_counter() - started, 3),
                            "max_abs_diff_vs_poly_chain": float(np.max(np.abs(diff))),
                            "mean_abs_diff_vs_poly_chain": float(np.mean(np.abs(diff))),
                            "fhe_max_abs": float(np.max(np.abs(fhe_hidden))), "poly_max_abs": float(np.max(np.abs(poly_hidden))),
                            "fhe_rms": float(np.sqrt(np.mean(fhe_hidden.astype(np.float64) ** 2))),
                            "finite": bool(np.all(np.isfinite(fhe_hidden))),
                            "output_head": [float(v) for v in fhe_hidden.reshape(-1)[:8]],
                            # rank consistency: every rank must hold the same ciphertext
                            "output_digest": int(np.asarray(fhe_hidden, dtype=np.float32).view(np.uint32).astype(np.uint64).sum()),
                        }
                    layer_row["weight_load_s"] = round(weight_source.load_seconds.get(layer, 0.0), 4)
                    layer_row["weight_wait_s"] = round(weight_source.wait_seconds.get(layer, 0.0), 4)
                    layer_row["weight_stage_s"] = round(weight_source.stage_seconds.get(layer, 0.0), 4)
                    layer_row["weights_device_hits"] = int(DEVICE_WEIGHTS.hits - registry_hits_before)
                    layer_row["weights_device_misses"] = int(DEVICE_WEIGHTS.misses - registry_misses_before)
                    if RESIDENT.enabled():
                        # cross-token weight plaintext store: per-layer hits/misses, cumulative bytes
                        layer_row["resident_plaintexts"] = RESIDENT.info()
                        RESIDENT.reset_counters()
                    weight_source.wait_seconds.pop(layer, None)
                    weight_source.load_seconds.pop(layer, None)
                    weight_source.stage_seconds.pop(layer, None)
                    layer_weights = None
                    weight_source.drop(layer)  # the station wrapper synchronised the device after the last station
                    layer_rows.append(layer_row)
                    print(json.dumps({"rank": int(rank), "iteration": iteration, "token": token, "layer": int(layer),
                                      "wall_s": layer_row["wall_s"], "bs_calls": layer_row["bootstrap_calls_total"],
                                      "bs_s": layer_row["bootstrap_seconds"], "comm_s": layer_row["comm_seconds"],
                                      "comm_mib_sent": round(layer_row["comm_bytes_sent"] / 2**20, 1),
                                      "peak_gib": max((s.get("peak_alloc_gib") or 0.0) for s in stations.values()),
                                      "peak_station": max(stations, key=lambda n: stations[n].get("peak_alloc_gib") or 0.0),
                                      "alloc_after_gib": round(float(stations[list(stations)[-1]].get("alloc_after_gib") or 0.0), 2),
                                      "fhe_max_abs": (round(layer_row["verify"]["fhe_max_abs"], 3) if verify else None),
                                      "poly_max_abs": (round(layer_row["verify"]["poly_max_abs"], 3) if verify else None),
                                      **({"vs_poly": round(layer_row["verify"]["max_abs_diff_vs_poly_chain"], 5)} if verify else {})}),
                          flush=True)
                token_wall = time.perf_counter() - token_start
                release_all(hidden_ciphers)
                hidden_ciphers = ()
                terminal_logits = (
                    terminal_head_report(fhe_hidden, poly_hidden, head=terminal_head)
                    if terminal_head is not None and verify else None
                )
                token_row = {
                    "token": int(token), "position": int(position), "wall_s": round(token_wall, 4),
                    "layers": layer_rows,
                    "layer_wall_sum_s": round(sum(r["wall_s"] for r in layer_rows), 4),
                    "bootstrap_calls": int(sum(r["bootstrap_calls_total"] for r in layer_rows)),
                    "bootstrap_seconds": round(sum(r["bootstrap_seconds"] for r in layer_rows), 4),
                    "comm_seconds": round(sum(r["comm_seconds"] for r in layer_rows), 4),
                    "comm_bytes_sent": int(sum(r["comm_bytes_sent"] for r in layer_rows)),
                    "cache_tokens": {str(l): int(caches[l].content_tokens) for l in layers},
                }
                if terminal_logits is not None:
                    token_row["terminal_logits"] = terminal_logits
                token_rows.append(token_row)
        finally:
            for cache in caches.values():
                cache.release()
        iterations.append({"iteration": iteration, "tokens": token_rows})
    probe.snap("90 end", {"resident_keys_gib": native_cache_inventory(native)["total_gib"]})

    summary = {
        "rank": int(rank),
        "world_size": world_size,
        "emulated": bool(emulated),
        "device_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ownership": {
            "query_streams": list(ownership.query_streams), "mlp_pages": list(ownership.mlp_page_indices),
            "mlp_partners": list(ownership.mlp_partners), "mlp_chunk_split": int(ownership.mlp_chunk_split),
        },
        "layers": list(layers), "tokens": int(tokens), "position0": position0, "kv_limbs": int(kv_limbs),
        "split_gate_up": bool(split_gate_up), "split_kv_refresh": bool(split_kv_refresh), "split_w_o": bool(split_w_o),
        "attention_limb_cap": resolve_attention_limb_cap(preset, attention_limb_cap),
        "tables": {"rmsnorm": rms_table_info, "softmax": softmax_table_info, "silu": silu_table_info,
                   "silu_degrees": {k: len(v["tiers"]["standard"]["coefficients"]) - 1
                                    for k, v in silu_table.items() if int(k) in layers}},
        "odin_patches": patches["info"],
        "multi_mac": dict(patches["multi_mac"].STATS) if patches["multi_mac"] is not None else None,
        "resident_plaintexts": RESIDENT.info() if RESIDENT.enabled() else None,
        "setup_seconds": state.setup_seconds,
        "warm_keys": warm_keys_report,
        "retain_weights": weight_source.retain,
        "resident": resident,
        "key_plan": key_plan_summary,
        "iterations": iterations,
        # last iteration = steady state
        "tokens_rows": iterations[-1]["tokens"],
        "bootstrap_calls_total": bootstrap.calls,
        "transport": {**transport.stats.to_dict(), "backend": type(transport).__name__},
        "timeline": probe.timeline,
    }
    gathered = transport.gather_to(0, "decode/summary", summary)
    transport.barrier("decode/final")
    transport.close()
    weight_source.close()
    if not emulated:
        state.release()
    return gathered if rank == 0 else None


# ------------------------------------------------------------------ driver
def _run_emulated(preset, kwargs: dict):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("LLAMA_FHE_PLAIN_CACHE_GB", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("LLAMA_FHE_TRANSFORM_HOIST", str(preset.transform_hoist_strategy))
    os.environ["T2_WORLD_SIZE"] = str(int(preset.world_size))
    if str(preset.key_policy) != "resident":
        raise SystemExit("--emulate-single-device needs key_policy='resident' (h100-emu / rtx6000pro-high)")

    import easyfhe

    from llama3fhe.parallel.decode.decode_keys import DECODE_TIERS
    from llama3fhe.parallel.emulation import run_emulated_world
    from llama3fhe.parallel.memory import native_cache_inventory
    from llama3fhe.parallel.worker import build_worker_state

    plan = decode_key_plan(preset, kv_limbs=int(kwargs["kv_limbs"]), attention_limb_cap=kwargs["attention_limb_cap"],
                           attn_resident=bool(kwargs["attn_resident"]), layers=tuple(kwargs["layers"]), silu_preset=preset)
    started = time.perf_counter()
    shared = build_worker_state(0, preset, tiers=DECODE_TIERS, trace_keys=True, key_plan=plan)
    easyfhe.cuda.synchronize()
    setup = {
        "seconds": round(time.perf_counter() - started, 2),
        "worker_setup_seconds": dict(shared.setup_seconds),
        "key_plan": shared.key_plan.summary(alpha=int(shared.native.alpha), special_limbs=int(shared.native.K)),
        "resident_keys_gib": native_cache_inventory(shared.native)["total_gib"],
    }
    print(json.dumps({"emulation_setup": setup}, default=str), flush=True)
    try:
        outcomes = run_emulated_world(preset, worker_entry, shared_state=shared, kwargs=kwargs)
    finally:
        shared.release()
    setup["baton_wait_seconds"] = [round(o.baton_wait_seconds, 3) for o in outcomes]
    setup["thread_wall_seconds"] = [round(o.wall_seconds, 2) for o in outcomes]
    return outcomes, setup


def _consistency(per_rank: list[dict]) -> dict:
    """Every rank must decrypt the same layer outputs (digests) and hold the same cache occupancy."""

    mismatches = []
    base = per_rank[0]["tokens_rows"]
    for summary in per_rank[1:]:
        rows = summary["tokens_rows"]
        for token_base, token_row in zip(base, rows, strict=True):
            if token_base["cache_tokens"] != token_row["cache_tokens"]:
                mismatches.append({"rank": summary["rank"], "token": token_row["token"], "cache_tokens": token_row["cache_tokens"]})
            for layer_base, layer_row in zip(token_base["layers"], token_row["layers"], strict=True):
                a, b = layer_base.get("verify"), layer_row.get("verify")
                if a and b and a["output_digest"] != b["output_digest"]:
                    mismatches.append({"rank": summary["rank"], "token": token_row["token"], "layer": layer_row["layer"],
                                       "digest": b["output_digest"], "rank0_digest": a["output_digest"]})
                if layer_base["output_levels"].get("layer_output") != layer_row["output_levels"].get("layer_output"):
                    mismatches.append({"rank": summary["rank"], "layer": layer_row["layer"], "levels": layer_row["output_levels"]})
    return {"ok": not mismatches, "mismatches": mismatches[:32], "verified": bool(base and base[0]["layers"][0].get("verify"))}


def _median(values):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])


def terminal_head_report(fhe_hidden, poly_hidden, *, head) -> dict:
    """Compare terminal top-1 after the measured Decode token wall."""

    import numpy as np

    from llama3fhe.application.client import top_k_logits

    started = time.perf_counter()
    fhe_logits = head.logits(fhe_hidden)
    poly_logits = head.logits(poly_hidden)
    finite = bool(np.all(np.isfinite(fhe_logits)) and np.all(np.isfinite(poly_logits)))
    fhe_top = top_k_logits(fhe_logits, 5) if finite else []
    poly_top = top_k_logits(poly_logits, 5) if finite else []
    return {
        "seconds": round(time.perf_counter() - started, 3),
        "finite": finite,
        "fhe_top_k": fhe_top,
        "poly_top_k": poly_top,
        "argmax_match": bool(finite and fhe_top[0]["token_id"] == poly_top[0]["token_id"]),
        "top5_overlap": len({row["token_id"] for row in fhe_top} & {row["token_id"] for row in poly_top}),
        "logits_max_abs_diff": (float(np.max(np.abs(fhe_logits - poly_logits))) if finite else None),
    }


def _tables(per_rank: list[dict], *, emulated: bool) -> dict:
    """Per-station median wall over ranks / layers (last token of the last iteration) + per-layer walls."""

    station_walls: dict[str, list[float]] = {}
    station_bs: dict[str, list[float]] = {}
    station_peak: dict[str, list[float]] = {}
    station_limbs: dict[str, int] = {}
    station_caps: dict[str, int] = {}
    tier_caps: dict[str, int] = {}
    layer_walls: dict[int, list[float]] = {}
    for summary in per_rank:
        last_token = summary["tokens_rows"][-1]
        for layer_row in last_token["layers"]:
            layer_walls.setdefault(int(layer_row["layer"]), []).append(float(layer_row["wall_s"]))
            for name, station in layer_row["stations"].items():
                station_walls.setdefault(name, []).append(float(station["wall_s"]))
                station_bs.setdefault(name, []).append(float(station["bootstrap_seconds"]))
                if station.get("peak_alloc_gib") is not None:
                    station_peak.setdefault(name, []).append(float(station["peak_alloc_gib"]))
                station_limbs[name] = max(station_limbs.get(name, 0), int(station["keys"].get("max_limbs") or 0))
                station_caps[name] = max(station_caps.get(name, 0), int(station.get("required_limb_cap") or 0))
                for tier_name, limbs in dict(station.get("required_limb_cap_by_tier") or {}).items():
                    tier_caps[tier_name] = max(tier_caps.get(tier_name, 0), int(limbs))
    stations = {
        name: {
            "median_wall_s": round(_median(walls), 4), "max_wall_s": round(max(walls), 4),
            "median_bs_s": round(_median(station_bs[name]), 4),
            "max_peak_gib": round(max(station_peak.get(name, [0.0])), 2),
            "max_limbs": station_limbs.get(name),
            "required_limb_cap": station_caps.get(name),
        }
        for name, walls in station_walls.items()
    }
    per_layer_median = {str(l): round(_median(w), 4) for l, w in sorted(layer_walls.items())}
    # emulated: the rank threads share one card, so a collective mostly waits for the *other* ranks' compute.
    # Per station take the max over ranks of (wall - recv wait) and sum: the compute-only layer wall of a
    # real W-card host, without the (real) transfer cost -- see token_comm_mib_sent_max for the volume.
    emulated_layer = None
    emulated_layer_raw = None
    if emulated:
        per_layer_station_max: dict[int, float] = {}
        per_layer_station_max_raw: dict[int, float] = {}
        for layer in layer_walls:
            total = 0.0
            total_raw = 0.0
            for name in station_walls:
                rows = [s["tokens_rows"][-1]["layers"][i]["stations"].get(name, {})
                        for s in per_rank for i, r in enumerate(s["tokens_rows"][-1]["layers"]) if int(r["layer"]) == layer]
                compute = [float(st.get("wall_s", 0.0)) - float((st.get("transport_delta") or {}).get("recv_wait_seconds", 0.0))
                           for st in rows]
                total += max(compute, default=0.0)
                total_raw += max((float(st.get("wall_s", 0.0)) for st in rows), default=0.0)
            per_layer_station_max[layer] = total
            per_layer_station_max_raw[layer] = total_raw
        emulated_layer = {str(l): round(v, 4) for l, v in sorted(per_layer_station_max.items())}
        emulated_layer_raw = {str(l): round(v, 4) for l, v in sorted(per_layer_station_max_raw.items())}
    token_walls = [float(s["tokens_rows"][-1]["wall_s"]) for s in per_rank]
    return {
        "stations": stations,
        "required_limb_cap_by_tier": dict(sorted(tier_caps.items())),
        "layer_wall_median_s": per_layer_median,
        "layer_wall_sum_of_medians_s": round(sum(_median(w) for w in layer_walls.values()), 3),
        "emulated_layer_wall_s": emulated_layer,
        "emulated_token_wall_s": None if emulated_layer is None else round(sum(per_layer_station_max.values()), 3),
        "emulated_layer_wall_with_wait_s": emulated_layer_raw,
        "token_wall_max_over_ranks_s": round(max(token_walls), 3),
        "token_bootstrap_calls": int(per_rank[0]["tokens_rows"][-1]["bootstrap_calls"]),
        "token_comm_seconds_max": round(max(float(s["tokens_rows"][-1]["comm_seconds"]) for s in per_rank), 3),
        "token_comm_mib_sent_max": round(max(float(s["tokens_rows"][-1]["comm_bytes_sent"]) for s in per_rank) / 2**20, 1),
        "vs_poly_last_layer_max_abs": (
            [r["verify"]["max_abs_diff_vs_poly_chain"] for r in per_rank[0]["tokens_rows"][-1]["layers"]][-1]
            if per_rank[0]["tokens_rows"][-1]["layers"][-1].get("verify") else None),
    }


def main() -> int:
    args = parse_args()
    from dataclasses import replace

    preset = preset_by_name(args.preset)
    if str(ROOT / "bench") not in sys.path:
        sys.path.insert(0, str(ROOT / "bench"))
    from t2_layer import apply_runtime_knobs, runtime_knob_summary

    preset = apply_runtime_knobs(args, preset)
    if args.softmax_sparse_bs:
        from llama3fhe.layouts.attention import AttentionPairLayout

        stride = int(AttentionPairLayout().decode_page_merge_stride)
        extra = tuple(sorted(set(preset.bootstrap_extra_sparse_log_slots) | {stride.bit_length() - 1}))
        preset = replace(preset, bootstrap_extra_sparse_log_slots=extra)
    if args.max_plaintext_rows is not None:
        preset = replace(preset, linear_max_plaintext_rows=int(args.max_plaintext_rows))
    if args.softmax_schain:
        os.environ["LLAMA_FHE_SOFTMAX_SCHAIN"] = "1"
    if args.softmax_blockplan:
        from llama3fhe.operators.attention.decode.decode_softmax import parse_softmax_blockplan
        parse_softmax_blockplan(args.softmax_blockplan)  # validate early
        os.environ["LLAMA_FHE_SOFTMAX_BLOCKPLAN"] = str(args.softmax_blockplan)
    if args.softmax_blockplan_force:
        os.environ["LLAMA_FHE_SOFTMAX_BLOCKPLAN_FORCE"] = "1"
    if args.softmax_pv_need_limbs is not None:
        os.environ["LLAMA_FHE_SOFTMAX_PV_NEED_LIMBS"] = str(int(args.softmax_pv_need_limbs))
    if args.no_shared_query_route:
        os.environ["LLAMA_FHE_DECODE_QUERY_SHARED"] = "0"
    if args.no_rotate_add_fused:
        os.environ["LLAMA_FHE_ROTATE_ADD"] = "0"
    if args.shifted_lanes:
        os.environ["LLAMA_FHE_DECODE_SL"] = "1"
        if args.sl_baby is not None:
            os.environ["LLAMA_FHE_DECODE_SL_BABY"] = str(int(args.sl_baby))
    world_size = int(preset.world_size if args.world_size is None else args.world_size)
    if world_size == 1:
        preset = replace(preset, world_size=1, ciphers_per_rank=16, mlp_feature_shard_k=1)
    elif world_size != int(preset.world_size):
        preset = preset.with_world_size(world_size)
    if args.model is not None:
        try:
            preset = preset.for_model(str(args.model))
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"--model {args.model}: {exc}") from exc
    if world_size not in (1, 2, 4, 8):
        raise SystemExit("--world-size must be 1, 2, 4 or 8 (the 8 (group, query) streams split evenly)")
    if world_size > 1 and args.max_plaintext_rows is None:
        # The QKV / W_O diagonal chunks are the decode split axis: 4096 / rows chunks must divide
        # over the ranks.  The prefill-tuned high tier carries 2048 rows (2 chunks) -> clamp to the
        # largest row count that still gives world_size | chunks (512 rows = 8 chunks on W=8).
        rows = int(preset.linear_max_plaintext_rows)
        while rows > 1 and (int(preset.model_spec.hidden_dim) // rows) % world_size:
            rows //= 2
        if rows != int(preset.linear_max_plaintext_rows):
            print(json.dumps({"note": "linear_max_plaintext_rows clamped for the decode chunk split",
                              "preset_rows": int(preset.linear_max_plaintext_rows), "rows": rows,
                              "world_size": world_size}), flush=True)
            preset = replace(preset, linear_max_plaintext_rows=rows)
    if args.attn_resident and resolve_attention_limb_cap(preset, args.attention_limb_cap) is None:
        raise SystemExit("--attn-resident needs a capped attn_core tier (the full-chain tier does not fit a lease card)")
    if args.transport is not None:
        os.environ["T2_TRANSPORT"] = str(args.transport)
    if not args.emulate_single_device:
        expand_devices(preset.world_size, args.devices, ranks_per_device=int(args.ranks_per_device))
        if int(args.ranks_per_device) > 1:
            preset = preset.for_colocation(int(args.ranks_per_device))
    layers = parse_layers(args.layers, num_layers=int(preset.model_spec.num_layers))
    model_dir = None if args.model_dir is None else str(Path(args.model_dir).resolve())
    if model_dir is not None:
        if not args.verify or layers[-1] != int(preset.model_spec.num_layers) - 1:
            raise SystemExit("--model-dir requires --verify and a layer range ending at the final model layer")
        if not (Path(model_dir) / "config.json").is_file():
            raise SystemExit(f"missing {Path(model_dir) / 'config.json'}")
    fixture = None if args.fixture is None else str(Path(args.fixture).resolve())
    fixtures_dir = None if args.fixtures_dir is None else str(Path(args.fixtures_dir).resolve())
    poly_kv = None if args.poly_kv is None else str(Path(args.poly_kv).resolve())
    if fixture is None and fixtures_dir is None:
        if preset.model_spec.name == "llama3.2-1b":
            fixtures_dir = str((ROOT / "fixtures" / "llama32_1b_model").resolve())
        else:
            default = ROOT / "fixtures" / "layer10_fixture.npz"
            if not default.exists():
                raise SystemExit("give --fixture or --fixtures-dir")
            fixture = str(default)
    if fixtures_dir is not None:
        manifest_path = Path(fixtures_dir) / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            built_for = str(manifest.get("model", "llama3-8b"))
            if built_for != preset.model_spec.name:
                raise SystemExit(f"fixtures {fixtures_dir} were built for {built_for}, not {preset.model_spec.name}")
        elif preset.model_spec.name != "llama3-8b":
            raise SystemExit(f"missing {manifest_path} for model {preset.model_spec.name}")
        if poly_kv is None and preset.model_spec.name == "llama3.2-1b":
            poly_kv = str(Path(fixtures_dir) / "prefill128_kv_poly.npz")
    if poly_kv is not None and not Path(poly_kv).is_file():
        raise SystemExit(f"missing --poly-kv {poly_kv}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else ROOT / "artifacts" / f"t2_decode_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "layers": layers, "tokens": int(args.tokens), "fixture": fixture, "fixtures_dir": fixtures_dir, "poly_kv": poly_kv,
        "kv_limbs": int(args.kv_limbs), "input_level": int(args.input_level),
        "split_gate_up": not bool(args.no_split_gate_up), "split_kv_refresh": not bool(args.no_split_kv_refresh),
        "split_w_o": not bool(args.no_split_w_o), "attention_limb_cap": args.attention_limb_cap,
        "attn_resident": bool(args.attn_resident), "prefetch_keys": bool(args.prefetch_keys),
        "verify": bool(args.verify), "repeat": int(args.repeat), "barrier_each": bool(args.barrier_each),
        "stage_weights": not bool(args.no_stage_weights), "warm_keys": bool(args.warm_keys), "lite": bool(args.lite),
        "retain_weights": str(args.retain_weights), "model_dir": model_dir,
    }
    started = time.perf_counter()
    setup: dict = {}
    if args.emulate_single_device:
        outcomes, setup = _run_emulated(preset, kwargs)
    else:
        outcomes = spawn_workers(preset, worker_entry, kwargs=kwargs, devices=args.devices,
                                 ranks_per_device=int(args.ranks_per_device), transport=args.transport)
    total = time.perf_counter() - started
    failed = [o for o in outcomes if not getattr(o, "ok", True)]
    if failed:
        for o in failed:
            print(json.dumps({"rank": o.rank, "error": o.error}), flush=True)
        return 2
    per_rank = outcomes[0].result
    assert per_rank is not None and len(per_rank) == preset.world_size
    consistency = _consistency(per_rank)
    tables = _tables(per_rank, emulated=bool(args.emulate_single_device))
    terminal_logits = ([row["terminal_logits"] for row in per_rank[0]["tokens_rows"]]
                       if model_dir is not None else None)
    terminal_argmax_match = (all(row["argmax_match"] for row in terminal_logits)
                             if terminal_logits is not None else None)
    result = {
        "preset": preset.name, "model": preset.model_spec.name, "world_size": int(preset.world_size),
        "emulated_single_device": bool(args.emulate_single_device), "emulation_setup": setup or None,
        "linear_max_plaintext_rows": int(preset.linear_max_plaintext_rows),
        "runtime_knobs": {**runtime_knob_summary(preset),
                          "bootstrap_extra_sparse_log_slots": list(preset.bootstrap_extra_sparse_log_slots),
                          "stage_weights": not bool(args.no_stage_weights),
                          "softmax_sparse_bs_env": os.environ.get("LLAMA_FHE_SOFTMAX_SPARSE_BS", "1"),
                          "softmax_schain_env": os.environ.get("LLAMA_FHE_SOFTMAX_SCHAIN", "0"),
                          "softmax_blockplan_env": os.environ.get("LLAMA_FHE_SOFTMAX_BLOCKPLAN", ""),
                          "softmax_blockplan_force_env": os.environ.get("LLAMA_FHE_SOFTMAX_BLOCKPLAN_FORCE", "0"),
                          "softmax_pv_need_limbs_env": os.environ.get("LLAMA_FHE_SOFTMAX_PV_NEED_LIMBS", ""),
                          "shared_query_route_env": os.environ.get("LLAMA_FHE_DECODE_QUERY_SHARED", "1"),
                          "rotate_add_fused_env": os.environ.get("LLAMA_FHE_ROTATE_ADD", "1"),
                          "shifted_lanes": bool(args.shifted_lanes),
                          "sl_baby_steps": (None if not args.shifted_lanes
                                            else int(os.environ.get("LLAMA_FHE_DECODE_SL_BABY", "32")))},
        "timestamp": stamp, "total_wall_s": round(total, 2),
        "worker_wall_s": [round(o.wall_seconds, 2) for o in outcomes],
        "args": vars(args), "consistency": consistency, "tables": tables, "ranks": per_rank,
        "terminal_logits": terminal_logits, "terminal_argmax_match": terminal_argmax_match,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"preset": preset.name, "world_size": int(preset.world_size), "total_wall_s": result["total_wall_s"],
                      "consistency_ok": consistency["ok"], "output": str(out_dir)}))
    print(json.dumps({"tables": {k: v for k, v in tables.items() if k != "stations"}}))
    if terminal_logits is not None:
        print(json.dumps({"terminal_argmax_match": terminal_argmax_match,
                          "terminal_top1": [{"token": token, "fhe": row["fhe_top_k"][0]["token_id"] if row["finite"] else None,
                                             "poly": row["poly_top_k"][0]["token_id"] if row["finite"] else None}
                                            for token, row in enumerate(terminal_logits)]}))
    for name, row in tables["stations"].items():
        print(json.dumps({"station": name, **row}))
    if not consistency["ok"]:
        print(json.dumps({"mismatches": consistency["mismatches"]}))
    return 0 if consistency["ok"] and terminal_argmax_match is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
