#!/usr/bin/env python3
"""T2 end-to-end Prefill -> Decode on an N-card host (one process per rank).

The multi-card Prefill station chain (``bench/t2_model.py``) emits the Decode
KV cache of every layer while it runs (``attention_input_exchange(...,
emit_kv_cache=True)`` + :func:`~llama3fhe.parallel.prefill.kv_emit.prefill_kv_cache_station`),
rank 0 applies the client-side LM head to the decrypted hidden state and feeds
the greedy token back as the DecodeFM input of the multi-card Decode chain
(``decode_layer_parallel``, ``bench/t2_decode.py``) for ``--decode-tokens``
steps.  One worker state serves both phases: the key plan is the per-tier
union of the Prefill plan (``emit_kv_cache=True``) and the Decode plan, and
the bootstrap operator carries every sparse program the two phases need
(Prefill period-16 slim RMSNorm + periodic Softmax slim, Decode period-1 slim
RMSNorm + merged-page Softmax).

8B gold greedy continuation of Paris128 (``hidden_0`` of ``fixtures/model``):
``12366, 13, 578, 6864`` (Prefill token + 3 Decode tokens).

Usage (8 x Pro 6000, from the repository root)::

    python bench/t2_prefill_decode.py --preset rtx6000pro-high --world-size 8 --transport pinned \\
        --fixtures fixtures/model --model-dir $MODEL_DIR --layers 0-31 --decode-tokens 3 \\
        --sl --sl-baby 16 --warm-keys --output artifacts/t2_prefill_decode/run1

Plaintext Prefix (the cache must contain the selected layers)::

    python bench/t2_prefill_decode.py --preset rtx6000pro-high --world-size 8 --transport pinned \\
        --fixtures fixtures/model --model-dir $MODEL_DIR --layers 0-31 --decode-tokens 3 \\
        --prefix-kv-cache fixtures/model/prefix2_kv.npz --output artifacts/t2_prefix2/run1

Single card, real serial Prefill -> Decode (``--world-size 1``: one rank owns
all 16 ciphertexts, no transport; Prefill chunk rows are clamped to 512 like
``t2_model.py --world-size 1`` unless ``--max-plaintext-rows`` is given, the
Decode side keeps the preset's 2048 rows).  ``--retain-weights host`` keeps the
pinned layer weights across tokens (the one-layer-ahead disk read, 1.0-1.2 s
per layer, is slower than the 0.8 s single-card Decode layer);
``--evict-prefill-keys`` drops the Prefill-only rotation keys after the
Prefill phase before the Decode keys are warmed (Prefill peaks at ~89 GiB on a
96 GB card with its 45 GiB of keys resident; the Decode plan adds ~25 GiB)::

    python bench/t2_prefill_decode.py --preset rtx6000pro-high --world-size 1 --devices 0 \\
        --fixtures fixtures/model --model-dir $MODEL_DIR --layers 0-31 --decode-tokens 3 \\
        --sl --sl-baby 16 --attention-limb-cap 17 --warm-keys --retain-weights host --evict-prefill-keys \\
        --output artifacts/t2_prefill_decode/single1

Llama-3.2-1B on one card (8 FM ciphertexts and 16 layers); its Decode tokens
are selected by the supplied checkpoint, and no 8B gold continuation is used::

    python bench/t2_prefill_decode.py --preset rtx6000pro-high --model llama3.2-1b --world-size 1 --devices 0 \\
        --fixtures fixtures/llama32_1b_model --model-dir assets/weights/Llama-3.2-1B-quarot \\
        --layers 0-15 --decode-tokens 1 --kv-limbs 12 --kv-value-limbs 8 --warm-keys --warmup-layer0 \\
        --output artifacts/t2_prefill_decode/llama32_1b

Smoke test on one card (fixtures hold layers 0-5 only; the LM head of a
truncated model is meaningless, ``--verify`` checks every layer against the
poly chain instead)::

    python bench/t2_prefill_decode.py --emulate-single-device --preset h100-emu --layers 0-1 \\
        --decode-tokens 1 --fixtures fixtures/model --model-dir $MODEL_DIR \\
        --poly-kv /path/to/data/fixtures/8b/prefill_kv_poly.npz --verify
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(os.environ.get("ODIN_SOURCE_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "bench") not in sys.path:
    sys.path.insert(0, str(ROOT / "bench"))

from bench.t2_decode import STATION_TIERS as DECODE_STATION_TIERS  # noqa: E402
from bench.t2_decode import _BootstrapTimer as _DecodeBootstrapTimer  # noqa: E402
from bench.t2_decode import decode_table_sources  # noqa: E402
from bench.t2_layer import (  # noqa: E402
    PROCESS_TRANSPORTS,
    STATIONS,
    _compare,
    _decrypt_rows,
    _median,
    _release_ciphers,
    add_runtime_knob_args,
    apply_softmax_post_exp,
    apply_runtime_knobs,
    attn_in_overlap_requested,
    rmsnorm_coeffs_table,
    rmsnorm_exit_kwargs,
    rmsnorm_slim_min_auto,
    silu_coeffs_table,
    softmax_coeffs_table,
    softmax_post_exp_description,
    worker_tiers,
)
from bench.t2_model import WeightPrefetcher, parse_layers  # noqa: E402
from bench.breakdown.records import nested_stage_records, parse_profile_layers  # noqa: E402

from llama3fhe.acceleration import patches as acceleration_patches  # noqa: E402
from llama3fhe.parallel.launch import expand_devices, spawn_workers  # noqa: E402
from llama3fhe.parallel.presets import preset_by_name  # noqa: E402

GOLD_TOKENS = (12366, 13, 578, 6864)
PREFILL_STATIONS = STATIONS[:4] + ("kv_emit",) + STATIONS[4:]


class _BootstrapTimer(_DecodeBootstrapTimer):
    """Match pure Prefill's profile sync, then restore Decode's timer policy.

    The Decode timer defaults to a full-device sync around every bootstrap,
    even when ``T2_STAGE_SYNC=0`` disables profile syncs elsewhere.  The
    per-refresh barriers serialize the Prefill main path unnecessarily. Decode
    retains its original timer behavior so this change isolates Prefill.
    """

    def __init__(self, inner, *, shared: bool = False):
        super().__init__(inner, shared=shared)
        self._prefill_phase = True

    def enter_decode(self) -> None:
        self._prefill_phase = False

    def refresh(self, cipher, **kwargs):
        from llama3fhe.backend import stage_sync_enabled

        if self._prefill_phase:
            kwargs.setdefault("synchronize", stage_sync_enabled())
        return super().refresh(cipher, **kwargs)

    def refresh_many(self, ciphers, **kwargs):
        from llama3fhe.backend import stage_sync_enabled

        if self._prefill_phase:
            kwargs.setdefault("synchronize", stage_sync_enabled())
        return super().refresh_many(ciphers, **kwargs)


# ------------------------------------------------------------------ key plans
def decode_rows_for_world(rows: int, world_size: int, hidden_dim: int = 4096) -> int:
    """Largest chunk row count <= ``rows`` with ``world_size | hidden_dim / rows``."""

    rows = int(rows)
    while rows > 1 and (int(hidden_dim) // rows) % int(world_size):
        rows //= 2
    return rows


def decode_sparse_log_slots(preset) -> tuple[int, ...]:
    """Sparse bootstrap programs the Decode chain needs: period-1 slim RMSNorm + merged-page Softmax."""

    from llama3fhe.parallel.worker import production_layouts, slim_sparse_log_slots

    feature_layout, attention_layout = production_layouts(preset)
    ks: set[int] = set()
    slim = slim_sparse_log_slots(replace(preset, rmsnorm_slim_period_rows=1), feature_layout)
    if slim is not None:
        ks.add(int(slim))
    stride = int(attention_layout.decode_page_merge_stride)
    ks.add(stride.bit_length() - 1)
    return tuple(sorted(ks))


def combined_preset(preset):
    """``preset`` with the Decode sparse programs added to ``bootstrap_extra_sparse_log_slots``."""

    extra = set(int(k) for k in preset.bootstrap_extra_sparse_log_slots) | set(decode_sparse_log_slots(preset))
    return replace(preset, bootstrap_extra_sparse_log_slots=tuple(sorted(extra)))


def decode_preset_and_schedule(preset, *, kv_limbs: int, shifted_lanes: bool, sl_baby: int | None,
                               rows: int | None = None, kv_value_limbs: int | None = None):
    """The Decode-side preset copy (clamped chunk rows) and its LayerSchedule.

    ``rows`` overrides the Prefill preset's chunk rows before the ``world_size`` clamp (``--world-size 1``
    runs the Prefill at 512 rows but the Decode linears at the preset's 2048, like ``t2_decode.py``).
    """

    base_rows = int(preset.linear_max_plaintext_rows if rows is None else rows)
    rows = decode_rows_for_world(base_rows, int(preset.world_size), int(preset.model_spec.hidden_dim))
    decode_preset = replace(preset, linear_max_plaintext_rows=rows)
    schedule = replace(decode_preset.layer_schedule(), kv_limbs=int(kv_limbs),
                       kv_value_limbs=None if kv_value_limbs is None else int(kv_value_limbs))
    if shifted_lanes:
        baby = int(schedule.decode_sl_baby_steps if sl_baby is None else sl_baby)
        schedule = replace(schedule, decode_shifted_lanes=True, decode_sl_baby_steps=baby)
    return decode_preset, schedule


def decode_approximations_at_position(layer_approximations, position: int):
    """Map prompt-position RMSNorm scales onto a one-row Decode input.

    ``token_prescale`` keys are absolute token positions.  Decode always packs
    its current token as row zero, which must not inherit the prompt BOS scale
    stored under position zero when the current position is 128 or later.
    """

    def norm_at_position(entry):
        factors = entry.get("token_prescale") or {}
        if not factors:
            return entry
        factor = factors.get(str(position), factors.get(int(position)))
        selected = dict(entry)
        if factor is None:
            selected.pop("token_prescale", None)
        else:
            selected["token_prescale"] = {"0": factor}
        return selected

    return replace(layer_approximations,
                   input_norm=norm_at_position(layer_approximations.input_norm),
                   post_attention_norm=norm_at_position(layer_approximations.post_attention_norm))


def prefill_layer_runtime_settings(layer_idx: int, *, operator_config, core_config,
                                   qkv_limbs: int | None, mlp_limbs: int | None,
                                   rms_exit: dict, rms_exit_post: dict):
    """Apply the same post-exp and RMSNorm exit schedule as pure Prefill."""

    layer_operator_config, core_config, qkv_entry = apply_softmax_post_exp(
        layer_idx, operator_config=operator_config, core_config=core_config, qkv_limbs=qkv_limbs)
    rms_exit = rmsnorm_slim_min_auto(rms_exit, qkv_entry, branch="attn")
    rms_exit_post = rmsnorm_slim_min_auto(rms_exit_post, mlp_limbs, branch="ffn")
    return layer_operator_config, core_config, qkv_entry, rms_exit, rms_exit_post


def merge_key_plans(*plans):
    """Per-tier union of several :class:`~llama3fhe.parallel.prefill.keys.KeyPlan` objects.

    A tier present in several plans gets the union of their rotations, the
    lowest residency level and the largest limb cap (``None`` -- full chain --
    wins, a full-chain key serves every lower limb count).
    """

    from llama3fhe.parallel.prefill.keys import KeyPlan, KeyTier

    rotations: dict[str, set[int]] = collections.defaultdict(set)
    levels: dict[str, int] = {}
    caps: dict[str, int | None] = {}
    resident_extra: set[str] = set()
    first = plans[0]
    for plan in plans:
        resident_extra.update(plan.resident_extra)
        for tier in plan.tiers:
            rotations[tier.name].update(int(v) for v in tier.rotations)
            levels[tier.name] = min(levels.get(tier.name, tier.level), tier.level)
            if tier.name in caps:
                old = caps[tier.name]
                caps[tier.name] = None if (old is None or tier.limb_cap is None) else max(int(old), int(tier.limb_cap))
            else:
                caps[tier.name] = tier.limb_cap
    tiers = tuple(KeyTier(name, levels[name], tuple(sorted(rotations[name])), caps[name]) for name in rotations)
    return KeyPlan(tiers=tiers, policy=str(first.policy), ring_dim=int(first.ring_dim),
                   full_limbs=int(first.full_limbs), resident_extra=tuple(sorted(resident_extra)))


def _prefill_key_preset(preset):
    """Remove Decode-only sparse bootstrap programs from the Prefill plan."""

    decode_slots = set(int(v) for v in decode_sparse_log_slots(preset))
    extras = tuple(sorted(int(v) for v in preset.bootstrap_extra_sparse_log_slots
                          if int(v) not in decode_slots))
    return replace(preset, bootstrap_extra_sparse_log_slots=extras)


def rank_phase_key_plans(
    preset,
    rank: int,
    *,
    decode_preset,
    decode_schedule,
    layers,
    attention_limb_cap: int | None = None,
    prefix_k: int = 0,
    multi_mac=None,
    emit_kv_cache: bool = True,
):
    """Return ``(merged, sizes, prefill, decode)`` plans for one rank.

    The merged plan remains the context/lifecycle compatibility plan.  The
    phase plans are used for selective warm-up, so :func:`rank_key_plan` keeps
    its original two-item return shape for existing callers.
    """

    from llama3fhe.parallel.decode.decode_keys import build_decode_key_plan
    from llama3fhe.parallel.prefill.keys import build_key_plan
    from llama3fhe.parallel.prefill.ownership import ownership_from_layouts
    from llama3fhe.parallel.worker import production_layouts

    feature_layout, attention_layout = production_layouts(preset)
    world = ownership_from_layouts(feature_layout, attention_layout, world_size=int(preset.world_size))
    # Keep the legacy Prefill plan for the merged context and summary sizes;
    # the stripped copy is used only for phase-specific warm-up.
    # ODIN_MULTI_MAC=1 + ODIN_MULTI_GIANT_HOIST=1: the Prefill MLP double-hoisted giant step needs the
    # ``g * giant_offset`` keys in the linear tier (llama3fhe/acceleration/patches.py); the Decode SL plan has none,
    # so the Decode phase runs under ``acceleration_patches.hoist_disabled()``.
    extra_rotations = None
    if multi_mac is not None:
        _hoist_keys = acceleration_patches.hoist_rotations(preset, multi_mac)
        if _hoist_keys:
            extra_rotations = {"linear": _hoist_keys}
    prefill = build_key_plan(
        preset, feature_layout, attention_layout, tiers=worker_tiers(STATIONS),
        schedule=preset.layer_schedule(),
        limb_caps=dict(preset.limb_caps), resident_extra=tuple(preset.key_resident_extra),
        policy=str(preset.key_policy), ownership=world[int(rank)], emit_kv_cache=bool(emit_kv_cache),
        prefix_k=int(prefix_k), extra_rotations=extra_rotations,
    )
    prefill_phase = build_key_plan(
        _prefill_key_preset(preset), feature_layout, attention_layout, tiers=worker_tiers(STATIONS),
        schedule=preset.layer_schedule(),
        limb_caps=dict(preset.limb_caps), resident_extra=tuple(preset.key_resident_extra),
        policy=str(preset.key_policy), ownership=world[int(rank)], emit_kv_cache=bool(emit_kv_cache),
        prefix_k=int(prefix_k), extra_rotations=extra_rotations,
    )
    if os.environ.get("T2_PD_PREFILL_ONLY_KEYS", "0").lower() in {"1", "true", "yes", "on"}:
        import resource
        rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        print(json.dumps({"phase": "prefill_plan_only", "union": len(prefill.union),
                          "phase_union": len(prefill_phase.union), "rss_gib": round(rss_gib, 3)}), flush=True)
        sizes = {"prefill_union": len(prefill.union), "decode_union": 0}
        return prefill, sizes, prefill_phase, prefill_phase
    from bench.t2_decode import decode_layouts, decode_linear_floor, resolve_attention_limb_cap

    decode_feature, decode_attention, pairfold_layout = decode_layouts(decode_preset)
    decode = build_decode_key_plan(
        decode_preset, schedule=decode_schedule, feature_layout=decode_feature, attention_layout=decode_attention,
        pairfold_layout=pairfold_layout, attention_limb_cap=resolve_attention_limb_cap(decode_preset, attention_limb_cap),
        resident_extra=tuple(preset.key_resident_extra),
        linear_limb_floor=decode_linear_floor(decode_schedule, layers, silu_preset=preset),
    )
    merged = merge_key_plans(prefill, decode)
    sizes = {"prefill_union": len(prefill.union), "decode_union": len(decode.union)}
    return merged, sizes, prefill_phase, decode


def rank_key_plan(preset, rank: int, *, decode_preset, decode_schedule, layers,
                  attention_limb_cap: int | None = None, prefix_k: int = 0, multi_mac=None,
                  emit_kv_cache: bool = True):
    """Prefill + Decode key plan of ``rank`` with the legacy two-item result."""

    merged, sizes, _prefill, _decode = rank_phase_key_plans(
        preset, rank, decode_preset=decode_preset, decode_schedule=decode_schedule, layers=layers,
        attention_limb_cap=attention_limb_cap, prefix_k=prefix_k, multi_mac=multi_mac,
        emit_kv_cache=bool(emit_kv_cache),
    )
    return merged, sizes


def emulated_phase_key_plans(preset, *, decode_preset, decode_schedule, layers,
                             attention_limb_cap: int | None = None, prefix_k: int = 0, multi_mac=None,
                             emit_kv_cache: bool = True):
    """Shared-context plans covering every emulated rank in each phase."""

    rank_plans = [rank_phase_key_plans(
        preset, rank, decode_preset=decode_preset, decode_schedule=decode_schedule, layers=layers,
        attention_limb_cap=attention_limb_cap, prefix_k=prefix_k, multi_mac=multi_mac,
        emit_kv_cache=emit_kv_cache,
    ) for rank in range(int(preset.world_size))]
    return tuple(merge_key_plans(*(plans[index] for plans in rank_plans)) for index in (0, 2, 3))


def cached_prefill_only_rotations(native, prefill_plan, decode_plan) -> tuple[int, ...]:
    """Host keys safe to drop while Prefill still uses the shared CUDA cache."""

    finder = getattr(native, "_find_cached_rotation_key", None)
    if not getattr(native, "auto_load_keys_resolved", False) or not callable(finder):
        return ()
    limits = prefill_plan.limb_limits()
    decode_rotations = set(decode_plan.union)
    alpha = int(native.alpha)
    return tuple(
        rot for rot in prefill_plan.union if rot not in decode_rotations
        and finder(rot, limits.get(rot, prefill_plan.full_limbs),
                   (limits.get(rot, prefill_plan.full_limbs) + alpha - 1) // alpha) is not None
    )


def guard_dropped_host_rotations(native, dropped: set[int]) -> None:
    """Fail explicitly if a later request misses a key whose host copy was freed."""

    if getattr(native, "_odin_drop_guard", False):
        return
    original_load = native._load_rotation_key_to_cuda

    def guarded_load(rot_index, *args, **kwargs):
        if int(rot_index) in dropped:
            raise RuntimeError(
                f"rotation key {int(rot_index)} was dropped from host memory (T2_PD_DROP_HOST_KEYS) "
                "but is requested again at a limb count the device cache does not hold"
            )
        return original_load(rot_index, *args, **kwargs)

    native._load_rotation_key_to_cuda = guarded_load
    native._odin_drop_guard = True


# ------------------------------------------------------------------ worker
def worker_entry(rank: int, preset, endpoints, *, fixtures: str, layers: tuple[int, ...], decode_tokens: int,
                 input_level: int, decode_input_level: int, kv_limbs: int, model_dir: str | None,
                 client_embedding_rows: str | None = None,
                 poly_kv: str | None, prefix_kv_cache: str | None, prefix_k: int,
                 prefix_model_metadata: dict[str, object] | None,
                 verify: bool, verify_every: int, warm_keys: bool, shifted_lanes: bool,
                 sl_baby: int | None, split_gate_up: bool, split_kv_refresh: bool, split_w_o: bool,
                 kv_bootstrap_iterations: int, prefetch_weights: bool = True, stage_weights: bool = True,
                 attention_limb_cap: int | None = None, decode_rows: int | None = None,
                 retain_weights: str = "none", evict_prefill_keys: bool = False, emulation=None,
                 emit_kv_cache: bool = True, qk_score_level_drop: int | None = None,
                 warmup_layer0: bool = False, encode_depth: int = 0, emulated_phase_plans=None,
                 timing_only_gold: bool = False, kv_value_limbs: int | None = None,
                 reference_tokens: tuple[int, ...] | None = None,
                 replay_decode_same_token: bool = False):
    """One rank: Prefill chain with KV emission, then ``decode_tokens`` Decode steps on the emitted caches."""

    import numpy as np

    import easyfhe

    os.environ["LLAMA_FHE_LINEAR_ENCODE_DEPTH"] = str(int(encode_depth))
    emit_kv_cache = bool(emit_kv_cache)
    model_name = str(preset.model_spec.name)
    profile_layers = parse_profile_layers(os.environ.get("COUPLET_PROFILE_LAYERS", ""),
                                         count=int(preset.model_spec.num_layers))
    if model_name != "llama3-8b" and timing_only_gold:
        raise ValueError("timing_only_gold has only the verified Llama-3-8B continuation")
    if model_name != "llama3-8b" and model_dir is None and int(decode_tokens) > 0:
        raise ValueError(f"{model_name} Decode requires --model-dir to select each token")
    reference_tokens = tuple(int(v) for v in (reference_tokens or (GOLD_TOKENS if model_name == "llama3-8b" else ())))

    # Prefill-side engineering (Sylph kernels / batched BS / fast encode / fused encode stage 2 / multi MAC),
    # env-gated, installed before the context exists -- see llama3fhe/acceleration/patches.py.
    patches = acceleration_patches.install_from_env(rank)
    multi_mac, fast_encode = patches["multi_mac"], patches["fast_encode"]

    from llama3fhe.application.approx_config import ApproximationConfig
    from llama3fhe.approx.attention import rope_cos_sin, set_default_rope_scaling
    from llama3fhe.approx.prefix_cache_io import load_prefix_kv_cache
    from llama3fhe.approx.rmsnorm import get_layer_coeffs
    from llama3fhe.approx.softmax import SoftmaxPolyConfig
    from llama3fhe.backend import release_all, stage_sync_enabled
    from llama3fhe.graph import LayerWeights as GraphLayerWeights
    from llama3fhe.graph import _linear_config
    from llama3fhe.layouts.linear import LinearCarrierLayout
    from llama3fhe.operators.attention.config import AttentionOperatorConfig
    from llama3fhe.operators.linear.resident_plaintexts import RESIDENT
    from llama3fhe.operators.linear.weight import DEVICE_WEIGHTS
    from llama3fhe.parallel.decode.decode_collectives import broadcast_ciphers
    from llama3fhe.parallel.decode.decode_ownership import DecodeRankOwnership
    from llama3fhe.parallel.decode.decode_stations import decode_layer_parallel
    from llama3fhe.parallel.memory import native_cache_inventory
    from llama3fhe.parallel.prefill.kv_emit import (
        prefill_kv_cache_begin,
        prefill_kv_cache_station,
        prefill_kv_cache_stream_begin,
    )
    from llama3fhe.parallel.prefill.stations import (
        QKV_INPUT_LIMBS,
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
    from llama3fhe.parallel.worker import build_worker_state, load_layer_config, production_layouts
    from bench.t2_decode import decode_layouts

    layers = tuple(int(v) for v in layers)
    set_default_rope_scaling(preset.model_spec.rope_scaling)
    emulated = emulation is not None
    prefix_bundles = (
        load_prefix_kv_cache(
            prefix_kv_cache,
            layout=production_layouts(preset)[1],
            expected_layers=layers,
            expected_prefix_k=int(prefix_k),
            expected_rope_theta=float(preset.model_spec.rope_theta),
            expected_model_dir=model_dir,
            expected_model_metadata=prefix_model_metadata,
        )
        if prefix_kv_cache is not None
        else None
    )
    if (prefix_bundles is None) != (int(prefix_k) == 0):
        raise ValueError("prefix_k must match the optional Prefix KV cache")
    prefix_by_layer = (
        {} if prefix_bundles is None else dict(zip(layers, prefix_bundles, strict=True))
    )
    decode_preset, decode_sched = decode_preset_and_schedule(preset, kv_limbs=kv_limbs, shifted_lanes=shifted_lanes,
                                                             sl_baby=sl_baby, rows=decode_rows,
                                                             kv_value_limbs=kv_value_limbs)
    plan, plan_sizes, prefill_plan, decode_plan = rank_phase_key_plans(
        preset, rank, decode_preset=decode_preset, decode_schedule=decode_sched, layers=layers,
        attention_limb_cap=attention_limb_cap, prefix_k=int(prefix_k), multi_mac=multi_mac,
        emit_kv_cache=emit_kv_cache,
    )
    if emulated:
        if emulated_phase_plans is None:
            raise ValueError("emulation requires the all-rank Prefill and Decode key plans")
        prefill_plan, decode_plan = emulated_phase_plans
        from llama3fhe.parallel.memory import MemoryProbe

        transport = emulation.transport
        state = replace(emulation.shared_state, rank=int(rank), transport=transport, memory=MemoryProbe(rank=int(rank)))
        state.memory.reset_peak()
    else:
        transport = build_transport(rank, endpoints)
        # Host cgroup is 110 GB. The merged Prefill+Decode key plan is killed
        # (-9) before the first layer. T2_PD_PREFILL_ONLY_KEYS keeps the
        # emit-enabled Prefill plan so the KV-emit split can be timed.
        active_plan = prefill_plan if os.environ.get("T2_PD_PREFILL_ONLY_KEYS", "0").lower() in {"1", "true", "yes", "on"} else plan
        state = build_worker_state(rank, preset, transport=transport, tiers=active_plan.names, trace_keys=True, key_plan=active_plan)
    probe, context, layout, attention_layout = state.memory, state.context, state.feature_layout, state.attention_layout
    ownership, world, native, life, tracer = state.ownership, state.world, state.native, state.keys, state.tracer
    bootstrap = _BootstrapTimer(state.bootstrap_operator, shared=emulated)
    world_size = int(transport.world_size)
    config = load_layer_config(preset)
    schedule = preset.layer_schedule()
    qkv_config = _linear_config(schedule, schedule.qkv_max_plaintext_rows, schedule.qkv_hoist_strategy)
    mlp_config = _linear_config(schedule, schedule.mlp_max_plaintext_rows, schedule.mlp_hoist_strategy)
    w_o_config = _linear_config(schedule, schedule.output_max_plaintext_rows, schedule.output_hoist_strategy)
    decode_feature, decode_attention, pairfold_layout = decode_layouts(decode_preset)
    decode_ownership = DecodeRankOwnership(
        int(rank), world_size, key_value_groups=int(pairfold_layout.key_value_groups),
        gqa_ratio=int(pairfold_layout.gqa_ratio),
        mlp_pages=(int(preset.model_spec.intermediate_dim) + int(decode_feature.hidden_dim) - 1)
                  // int(decode_feature.hidden_dim),
    )
    stage_prefix = f"r{rank}." if emulated else ""

    (default_rms, default_silu, default_softmax), table_preset = decode_table_sources(preset)
    rms_all, rms_table_info = rmsnorm_coeffs_table(default_rms, table_preset)
    # SiLU table follows the preset / T2_SILU_COEFFS like t2_model / t2_decode (E8 joint_tol_v1): the Prefill
    # MLP entry limbs *and* the Decode SiLU polynomial / linear limb floor come from this one table.
    silu_all, silu_table_info = silu_coeffs_table(default_silu, table_preset)
    softmax_table, softmax_table_info = softmax_coeffs_table(default_softmax, table_preset)
    softmax_config = SoftmaxPolyConfig.from_artifact(softmax_table)
    approximations = ApproximationConfig(rmsnorm_path=default_rms, silu_path=default_silu,
                                         softmax_path=default_softmax)
    approximations._rmsnorm = rms_all
    approximations._softmax = softmax_config
    approximations._silu = {"per_layer": silu_all}
    approximations._layers.clear()
    hidden_dim = int(layout.hidden_dim)
    rows = slice(int(ownership.token_start), int(ownership.token_stop))
    fixtures_dir = Path(fixtures)

    with np.load(fixtures_dir / "hidden_trajectory.npz") as data:
        hidden_in = np.asarray(data[f"hidden_{layers[0]}"], dtype=np.float32)
        oracle_hidden = (
            {idx: np.asarray(data[f"hidden_{idx + 1}"], dtype=np.float32) for idx in layers}
            if not prefix_k else {}
        )
    if hidden_in.shape != (int(layout.seq_len), hidden_dim):
        raise ValueError(f"hidden_{layers[0]} has shape {hidden_in.shape}, layout wants {(layout.seq_len, hidden_dim)}")
    packed = layout.pack(hidden_in)
    poly_kv_arrays = None
    if poly_kv:
        with np.load(poly_kv) as payload:
            poly_keys = np.asarray(payload["keys"], dtype=np.float32)
            poly_values = np.asarray(payload["values"], dtype=np.float32)
        poly_kv_arrays = {idx: (np.array(poly_keys[idx]), np.array(poly_values[idx])) for idx in layers}

    fuse_silu = bool(schedule.fuse_silu_domain_map)
    carrier_layout = LinearCarrierLayout(seq_len=int(layout.seq_len), dimension=hidden_dim, slots=int(layout.slots))
    cos, sin = rope_cos_sin(seq_len=int(attention_layout.seq_len), head_dim=int(attention_layout.head_dim),
                            theta=float(preset.model_spec.rope_theta), start_pos=int(prefix_k))
    operator_kwargs = dict(
        sparse_max_baby_offsets=int(schedule.sparse_max_baby_offsets),
        shear_baby_steps=int(schedule.shear_baby_steps),
        pv_complex_delta_folding=bool(schedule.pv_complex_delta_folding),
        pv_prefold_after_softmax=bool(schedule.pv_prefold_after_softmax),
        batched_sparse_input_bsgs=bool(schedule.batched_sparse_attention_input),
        qk_complex_feature_folding=True, qk_key_features_pre_folded=True,
        qk_query_features_pre_scaled=True, qk_query_features_pre_folded=True,
        rope_theta=float(preset.model_spec.rope_theta),
        pairfm_refresh_bootstrap_iterations=int(preset.pairfm_refresh_bootstrap_iterations),
        softmax_slim_periodic=bool(getattr(preset, "softmax_slim_periodic", False)),
    )
    # None keeps the operator default (0). t2_model passes QK_SCORE_LEVEL_DROP (3);
    # the aligned A/B sets that explicitly so this bench matches BEST10.
    if qk_score_level_drop is not None:
        operator_kwargs["qk_score_level_drop"] = int(qk_score_level_drop)
    operator_config = AttentionOperatorConfig(**operator_kwargs)
    stage_device = str(context.device) if (stage_weights and str(context.device).startswith("cuda")) else None
    # retain="host"/"device": the Decode tokens re-use the layer weights the Prefill pass read (see WeightPrefetcher)
    prefetcher = WeightPrefetcher(fixtures_dir, layers, enabled=bool(prefetch_weights), device=stage_device,
                                  retain=str(retain_weights) if stage_device is not None else "none")
    hidden_shape_only = np.zeros((int(layout.seq_len), hidden_dim), dtype=np.float32)

    def _stage_dict(values) -> dict[str, float]:
        return {str(k): round(float(v), 4) for k, v in dict(values or {}).items()}

    memlog_enabled = os.environ.get("T2_PD_MEMLOG", "0").lower() in {"1", "true", "yes"}

    memlog_deep = os.environ.get("T2_PD_MEMLOG", "0") == "2"

    def memlog_tensors(top: int = 12) -> dict:
        """T2_PD_MEMLOG=2: ``gc.collect()`` then bucket every live CUDA tensor by shape (GiB, count) -- the owners
        of the allocator footprint that are neither rotation keys nor BS constants."""

        import gc

        gc.collect()
        buckets: dict[str, list] = {}
        seen: set[int] = set()
        total = 0.0
        for obj in gc.get_objects():
            try:
                if not easyfhe.is_tensor(obj) or not obj.is_cuda:
                    continue
                storage = obj.untyped_storage()
                ptr = storage.data_ptr()
                if ptr in seen:
                    continue
                seen.add(ptr)
                gib = float(storage.nbytes()) / float(2**30)
            except Exception:
                continue
            total += gib
            key = f"{tuple(obj.shape)}:{str(obj.dtype).replace('torch.', '')}"
            entry = buckets.setdefault(key, [0.0, 0])
            entry[0] += gib
            entry[1] += 1
        ranked = sorted(buckets.items(), key=lambda kv: -kv[1][0])[:top]
        return {"reachable_gib": round(total, 2), "top": {k: [round(v[0], 2), v[1]] for k, v in ranked}}

    def memlog(tag: str, deep: bool = False, **extra) -> None:
        """T2_PD_MEMLOG=1: rank-0 line with the allocator / resident key footprint (GiB) at ``tag``."""

        last_emulated_rank_sample = (
            emulated and int(rank) == world_size - 1
            and (tag == "prefill/end" or tag.startswith("prefill/layer/"))
        )
        if not memlog_enabled or (int(rank) != 0 and not last_emulated_rank_sample):
            return
        if deep and memlog_deep:
            extra = {**extra, **memlog_tensors()}
        try:
            alloc = float(easyfhe.cuda.memory_allocated()) / float(2**30)
            reserved = float(easyfhe.cuda.memory_reserved()) / float(2**30)
            keys_gib = native_cache_inventory(native)["total_gib"]
        except Exception:
            return
        print(json.dumps({"phase": "mem", "rank": int(rank), "tag": tag,
                          "alloc_gib": round(alloc, 2), "reserved_gib": round(reserved, 2),
                          "keys_gib": round(float(keys_gib), 2), "bs_consts_gib": round(bootstrap_constants_gib(), 2),
                          **host_memory_gib(), **extra}), flush=True)

    def host_memory_gib() -> dict:
        """Process RSS and the container's cgroup usage (what the -9 host OOM kill is measured against)."""

        out: dict = {}
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        out["host_rss_gib"] = round(int(line.split()[1]) / float(2**20), 2)
                        break
        except OSError:
            pass
        for path, key in (("/sys/fs/cgroup/memory.current", "cgroup_gib"),
                          ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "cgroup_gib")):
            try:
                with open(path) as fh:
                    out[key] = round(int(fh.read().strip()) / float(2**30), 2)
                break
            except (OSError, ValueError):
                continue
        return out

    def bootstrap_programs():
        op = state.bootstrap_operator
        programs = [getattr(op, "program", None), getattr(op, "deep_program", None)]
        programs += list(getattr(op, "sparse_programs", {}).values())
        return [p for p in programs if p is not None]

    def bootstrap_constants_gib() -> float:
        """Device bytes held by the BS programs' encoded-constant caches (CtS / StC / EvalMod plaintexts per level)."""

        total = 0
        for program in bootstrap_programs():
            consts = getattr(program, "constants", None)
            info = getattr(consts, "cache_info", None)
            if callable(info):
                try:
                    total += int(info().get("total_bytes", 0))
                except Exception:
                    pass
        return float(total) / float(2**30)

    def timed_station(stations: dict, stage_name: str, name: str, tier: str | None, fn, *, barrier_tag: str | None = None):
        """wall / BS count / BS seconds / peak allocation of one station (the ``--lite`` profile of t2_model)."""

        if barrier_tag is not None:
            transport.barrier(barrier_tag)
        probe.reset_peak()
        calls_before = bootstrap.calls
        bootstrap.take()
        transport_before = transport.stats.to_dict()
        started = time.perf_counter()
        with tracer.stage(stage_name):
            if tier is None:
                value = fn()
            else:
                with life.lease(tier, prefetch=False):
                    value = fn()
        easyfhe.cuda.current_stream().synchronize()
        wall = time.perf_counter() - started
        bs_records = bootstrap.take()
        stats = tracer.stages[stage_name].as_row()
        transport_after = transport.stats.to_dict()
        memlog(stage_name, peak_gib=round(float(easyfhe.cuda.max_memory_allocated()) / float(2**30), 2))
        stations[name] = {
            "wall_s": round(wall, 4), "tier": tier,
            "bootstrap_calls": bootstrap.calls - calls_before,
            "bootstrap_seconds": round(sum(r[0] for r in bs_records), 4),
            "peak_alloc_gib": round(float(easyfhe.cuda.max_memory_allocated()) / float(2**30), 3),
            "keys": {k: stats.get(k) for k in ("requests", "misses", "gib_loaded", "load_seconds")},
            "transport_delta": {k: round(transport_after[k] - transport_before[k], 4)
                                for k in ("recv_wait_seconds", "send_seconds", "bytes_sent", "bytes_received")},
        }
        if timing_only_gold:
            ghost_construct = transport_after["ghost_construct_seconds"] - transport_before["ghost_construct_seconds"]
            ghost_unpack = transport_after["ghost_unpack_seconds"] - transport_before["ghost_unpack_seconds"]
            stations[name]["ghost_overhead"] = {
                "construct_s": round(ghost_construct, 4), "unpack_s": round(ghost_unpack, 4),
            }
            stations[name]["wall_without_ghost_s"] = round(max(0.0, wall - ghost_construct - ghost_unpack), 4)
        return value

    # ------------------------------------------------------------ Prefill
    caches: dict[int, object] = {}

    def run_prefill_layer(idx: int, inputs, weights, *, verify_layer: bool, tag_prefix: str | None = None):
        stations: dict[str, dict] = {}
        tag_prefix = f"model/{idx}" if tag_prefix is None else str(tag_prefix)
        prefix = f"t2pd.r{rank}.l{idx}"
        coeffs_attn = get_layer_coeffs(rms_all, "attn", idx)
        coeffs_ffn = get_layer_coeffs(rms_all, "ffn", idx)
        rms_exit = rmsnorm_exit_kwargs(idx, coeffs_attn, preset)
        rms_exit_post = rmsnorm_exit_kwargs(idx, coeffs_ffn, preset)
        silu_entry = silu_all[str(idx)]["tiers"]["standard"]
        mlp_entry = mlp_input_limbs(silu_entry, fuse_silu_domain_map=fuse_silu)
        core_config = attention_core_operator_config(schedule, layer_idx=idx, base=operator_config)
        layer_operator_config, core_config, qkv_entry, rms_exit, rms_exit_post = prefill_layer_runtime_settings(
            idx, operator_config=operator_config, core_config=core_config, qkv_limbs=QKV_INPUT_LIMBS,
            mlp_limbs=mlp_entry, rms_exit=rms_exit, rms_exit_post=rms_exit_post)
        gamma_in, gamma_post = weights["input_layernorm"], weights["post_attention_layernorm"]
        q_w, k_w, v_w, o_w = weights["q_proj"], weights["k_proj"], weights["v_proj"], weights["o_proj"]
        gate_w, up_w, down_w = weights["gate_proj"], weights["up_proj"], weights["down_proj"]
        weights.register()

        def timed(name, tier, fn, *, barrier=False):
            return timed_station(stations, f"{stage_prefix}p_{name}_{idx}", name, tier, fn,
                                 barrier_tag=f"{tag_prefix}/before/{name}" if barrier else None)

        keep: dict[str, object] = {}
        outputs: tuple = ()
        kv_result = None
        cache_owned = False
        kv_emit_pending = None
        kv_emit_stream_pending = None
        kv_emit_overlap_requested = emit_kv_cache and os.environ.get("T2_PREFILL_KV_EMIT_OVERLAP", "0").lower() in {
            "1", "true", "yes", "on"
        }
        kv_emit_stream_requested = emit_kv_cache and os.environ.get("T2_PREFILL_KV_EMIT_STREAM", "0").lower() in {
            "1", "true", "yes", "on"
        }
        # A side stream may share rotation-key state only when the merged plan
        # keeps every tier resident.  Leased plans would evict keys while the
        # worker is still using them, so they stay on the synchronous path.
        kv_emit_stream_allowed = bool(
            kv_emit_stream_requested
            and str(getattr(getattr(life, "plan", None), "policy", "")) == "resident"
            and str(getattr(context, "device", "")).startswith("cuda")
        )
        kv_emit_overlap = bool(kv_emit_overlap_requested or kv_emit_stream_allowed)
        layer_started = time.perf_counter()
        try:
            rms = timed("rmsnorm_in", None, lambda: rmsnorm_station(
                inputs, gamma_in, coeffs=coeffs_attn, ownership=ownership, layout=layout, context=context,
                bootstrap_operator=bootstrap, keep_residual=True, **rms_exit, function_prefix=f"{prefix}.rms_in"))
            keep["rms"] = rms
            stations["rmsnorm_in"]["stage_seconds"] = _stage_dict(rms.stage_seconds)
            release_all(inputs)
            inputs = ()
            qkv = timed("qkv", "linear", lambda: qkv_station(
                rms.ciphers, hidden_shape_only, q_w, k_w, v_w, ownership=ownership, layout=layout,
                attention_layout=attention_layout, context=context, config=config, operator_config=qkv_config,
                input_column_scale=rms.linear_input_column_scale(), input_limbs=qkv_entry,
                function_prefix=f"{prefix}.qkv"))
            keep["qkv"] = qkv
            stations["qkv"]["stage_seconds"] = _stage_dict(qkv.stage_seconds)
            _release_ciphers(rms.ciphers)
            rms.ciphers = ()
            attn_in = timed("attn_input", None, lambda: attention_input_exchange(
                qkv.carrier_ciphers, ownership=ownership, world=world, transport=transport,
                carrier_layout=carrier_layout, attention_layout=attention_layout, context=context, cos=cos, sin=sin,
                model_config=config, operator_config=layer_operator_config, keys=life, prefetch_keys=False,
                overlap=attn_in_overlap_requested(preset), emit_kv_cache=emit_kv_cache,
                function_prefix=f"{prefix}.attn_in", tag=f"{tag_prefix}/attn_in"), barrier=True)
            keep["attn_in"] = attn_in
            stations["attn_input"]["stage_seconds"] = _stage_dict(attn_in.stage_seconds)
            stations["attn_input"]["comm_gib"] = {k: round(int(v) / float(2**30), 4) for k, v in attn_in.comm_bytes.items()}
            qkv.release()
            if kv_emit_overlap:
                key_c_items, value_c_items = attn_in.take_kv_cache_items()
                kv_emit_pending = prefill_kv_cache_begin(
                    key_c_items, value_c_items,
                    ownership=ownership, world=world, transport=transport,
                    attention_layout=attention_layout, context=context,
                    pairfold_layout=pairfold_layout,
                    tag=f"{tag_prefix}/kv_emit",
                )
                if kv_emit_stream_allowed:
                    kv_emit_stream_pending = prefill_kv_cache_stream_begin(
                        kv_emit_pending,
                        keys=life,
                        ownership=ownership,
                        world=world,
                        transport=transport,
                        attention_layout=attention_layout,
                        context=context,
                        bootstrap_operator=bootstrap,
                        kv_limbs=int(kv_limbs),
                        kv_value_limbs=kv_value_limbs,
                        bootstrap_iterations=int(kv_bootstrap_iterations),
                        pairfold_layout=pairfold_layout,
                        tag=f"{tag_prefix}/kv_emit",
                    )
                    if kv_emit_stream_pending is not None:
                        # The stream worker now owns the raw pending handle.
                        kv_emit_pending = None
            core = timed("attn_core", None, lambda: attention_core_station(
                attn_in.query_items, attn_in.key_items, attn_in.value_items, ownership=ownership,
                attention_layout=attention_layout, context=context, softmax_config=softmax_config,
                operator_config=core_config, bootstrap_operator=bootstrap, layer_idx=idx,
                keys=life, prefetch_keys=False, prefix_bundle=prefix_by_layer.get(idx),
                function_prefix=f"{prefix}.core"))
            keep["core"] = core
            stations["attn_core"]["stage_seconds"] = _stage_dict(core.stage_seconds)
            # The synchronous path detaches snapshots only after attention core.
            if emit_kv_cache and not kv_emit_overlap:
                key_c_items, value_c_items = attn_in.take_kv_cache_items()
            attn_in.release()
            if kv_emit_stream_pending is not None:
                # Conversion/refresh runs on the side stream while Attention
                # Core is active.  Join that event before PairFM so its key
                # work never races the shared resident bootstrap program.
                kv_emit_stream_pending.wait_conversion()
            if kv_emit_stream_pending is None and kv_emit_overlap:
                # The raw route was issued before Attention Core.  The
                # transport allows only one collective in flight, so wait via
                # the station before PairFM starts its next all-to-all.
                kv_result = timed("kv_emit", None, lambda: prefill_kv_cache_station(
                    (), (), ownership=ownership, world=world, transport=transport,
                    attention_layout=attention_layout, context=context,
                    bootstrap_operator=bootstrap,
                    kv_limbs=int(kv_limbs), kv_value_limbs=kv_value_limbs,
                    bootstrap_iterations=int(kv_bootstrap_iterations),
                    pairfold_layout=pairfold_layout, tag=f"{tag_prefix}/kv_emit",
                    raw_pending=kv_emit_pending), barrier=False)
                stations["kv_emit"]["overlapped"] = True
            elif emit_kv_cache and kv_emit_stream_pending is None:
                kv_result = timed("kv_emit", None, lambda: prefill_kv_cache_station(
                    key_c_items, value_c_items, ownership=ownership, world=world, transport=transport,
                    attention_layout=attention_layout, context=context, bootstrap_operator=bootstrap,
                    kv_limbs=int(kv_limbs), kv_value_limbs=kv_value_limbs,
                    bootstrap_iterations=int(kv_bootstrap_iterations),
                    pairfold_layout=pairfold_layout, tag=f"{tag_prefix}/kv_emit"), barrier=True)
            pairfm_after_exchange = (
                kv_emit_stream_pending.allow_final_gather
                if kv_emit_stream_pending is not None else None
            )
            pf = timed("pairfm", None, lambda: pairfm_exchange(
                core.output_items, ownership=ownership, world=world, transport=transport, feature_layout=layout,
                attention_layout=attention_layout, context=context, model_config=config, operator_config=core_config,
                bootstrap_operator=bootstrap, keys=life, prefetch_keys=False,
                function_prefix=f"{prefix}.pairfm", tag=f"{tag_prefix}/pairfm",
                after_exchange=pairfm_after_exchange), barrier=True)
            keep["pf"] = pf
            stations["pairfm"]["stage_seconds"] = _stage_dict(pf.stage_seconds)
            stations["pairfm"]["comm_bytes"] = {str(k): int(v) for k, v in pf.comm_bytes.items()}
            stations["pairfm"]["adapter"] = (
                "replicated_1b" if world_size > 1 and
                int(attention_layout.query_pair_count) != int(layout.cipher_count)
                else "local_or_distributed"
            )
            # The side worker was gated before its final all-gather.  Its
            # transport round now runs while PairFM reduces received partials
            # locally and while the rest of this layer performs W_O, residual,
            # RMSNorm, and MLP.  The cache is not consumed until the next
            # Decode phase, so joining here would serialize independent local
            # work behind the gather.
            core.release()
            wo = timed("w_o", "linear", lambda: w_o_station(
                pf.carriers, o_w, ownership=ownership, layout=layout, attention_layout=attention_layout,
                context=context, operator_config=w_o_config))
            keep["wo"] = tuple(wo)
            pf.release()
            res1 = timed("residual_attn", None, lambda: residual_station(
                rms.residual_inputs, wo, ownership=ownership, layout=layout, context=context))
            keep["res1"] = res1
            rms.release()
            _release_ciphers(keep.pop("wo"))
            rms2 = timed("rmsnorm_post", None, lambda: rmsnorm_station(
                res1.ciphers, gamma_post, coeffs=coeffs_ffn, ownership=ownership, layout=layout, context=context,
                bootstrap_operator=bootstrap, keep_residual=True, **rms_exit_post, function_prefix=f"{prefix}.rms_post"))
            keep["rms2"] = rms2
            stations["rmsnorm_post"]["stage_seconds"] = _stage_dict(rms2.stage_seconds)
            res1.release()
            mlp = timed("mlp", "linear", lambda: mlp_station(
                rms2.ciphers, hidden_shape_only, gate_w, up_w, down_w, silu_entry=silu_entry, ownership=ownership,
                layout=layout, context=context, config=config, operator_config=mlp_config,
                input_column_scale=rms2.linear_input_column_scale(), fuse_silu_domain_map=fuse_silu,
                input_limbs=mlp_entry, feature_shard_k=int(preset.mlp_feature_shard_k), transport=transport,
                function_prefix=f"{prefix}.mlp"))
            keep["mlp"] = mlp
            stations["mlp"]["stage_seconds"] = _stage_dict(mlp.stage_seconds)
            _release_ciphers(rms2.ciphers)
            rms2.ciphers = ()
            res2 = timed("residual_mlp", None, lambda: residual_station(
                rms2.residual_inputs, mlp.ciphers, ownership=ownership, layout=layout, context=context))
            outputs = tuple(res2.ciphers)
            res2.ciphers = ()
            keep["res2"] = res2
            rms2.release()
            mlp.release()

            # Join the resident side stream only after all local post-attention
            # work has been queued.  For the synchronous/fallback paths this
            # is already a completed result; keeping the bookkeeping here
            # gives both paths the same cache ownership boundary.
            if kv_emit_stream_pending is not None:
                kv_result = timed("kv_emit", None, lambda: kv_emit_stream_pending.finish(), barrier=False)
                stations["kv_emit"]["overlapped"] = True
                stations["kv_emit"]["stream"] = "resident-side"
                stations["kv_emit"]["gather_after_pairfm"] = True
                stations["kv_emit"]["background_wall_s"] = round(
                    float(kv_emit_stream_pending.box.get("result_wall_s", 0.0)), 4
                )
            if emit_kv_cache:
                stations["kv_emit"].update({
                    "stream_requested": bool(kv_emit_stream_requested),
                    "stream_fallback": bool(kv_emit_stream_requested and kv_emit_stream_pending is None),
                    "stage_seconds": _stage_dict(kv_result.stage_seconds),
                    "comm_gib": {k: round(int(v) / float(2**30), 4) for k, v in kv_result.comm_bytes.items()},
                    "comm_seconds": round(kv_result.comm_seconds, 4),
                    "contributed": {k: list(v) for k, v in kv_result.contributed.items()},
                    "refreshed": list(kv_result.refreshed), "limbs": kv_result.limbs,
                })
                caches[idx] = kv_result.take_cache()
                cache_owned = True
            else:
                stations["kv_emit"] = {
                    "wall_s": 0.0, "tier": None, "bootstrap_calls": 0, "bootstrap_seconds": 0.0,
                    "peak_alloc_gib": 0.0, "stage_seconds": {}, "limbs": {},
                }
        finally:
            # If Attention Core or a later local stage fails after the
            # split-phase raw route was issued, drain that one in-flight
            # collective before the worker can enter another layer.  The
            # received raw handles are never needed on this error path.
            if kv_emit_pending is not None and not kv_emit_pending._waited:
                try:
                    pending_raw, pending_map, _ = kv_emit_pending.wait_raw()
                    _release_ciphers(pending_raw)
                    _release_ciphers(tuple(pending_map.values()))
                except BaseException:
                    pass
            if kv_emit_stream_pending is not None and not kv_emit_stream_pending._finished:
                try:
                    kv_emit_stream_pending.cancel()
                except BaseException:
                    pass
            if kv_result is not None and not cache_owned:
                release = getattr(kv_result, "release", None)
                if callable(release):
                    try:
                        release()
                    except BaseException:
                        pass
            for value in keep.values():
                release = getattr(value, "release", None)
                if callable(release):
                    release()
                elif isinstance(value, (tuple, list)):
                    _release_ciphers(value)
            if inputs:
                release_all(inputs)
            weights.unregister()
        easyfhe.cuda.current_stream().synchronize()
        record = {
            "layer": int(idx),
            "layer_wall_s": round(time.perf_counter() - layer_started, 4),
            "station_wall_sum_s": round(sum(v["wall_s"] for v in stations.values()), 4),
            "bootstrap_calls": sum(v["bootstrap_calls"] for v in stations.values()),
            "bootstrap_seconds": round(sum(v["bootstrap_seconds"] for v in stations.values()), 4),
            "kv_emit_wall_s": stations["kv_emit"]["wall_s"],
            "output_limbs": [int(c.state.cur_limbs) for c in outputs],
            "cache_limbs": (
                sorted({int(c.state.cur_limbs) for page in caches[idx].pages
                        for c in (*page.key_groups, *page.value_groups)})
                if idx in caches else []
            ),
            "cache_tokens": int(caches[idx].content_tokens) if idx in caches else 0,
            "stations": stations,
            "verified": False,
        }
        if verify_layer:
            started = time.perf_counter()
            got = _decrypt_rows(context, layout, ownership, outputs)
            record["oracle"] = _compare(got, oracle_hidden[idx][rows])
            record["verify_seconds"] = round(time.perf_counter() - started, 3)
            record["verified"] = True
        return outputs, record

    inputs = tuple(context.encrypt(packed[index], level=int(input_level), slots=int(layout.slots))
                   for index in ownership.cipher_indices)
    easyfhe.cuda.synchronize()
    prefetcher.start(layers[0])
    first_weights = prefetcher.get(layers[0])

    def warm_phase(target_plan) -> dict:
        """Warm only the rotations needed by one phase of the combined run."""

        started = time.perf_counter()
        per_tier = {}
        for tier_name in target_plan.names:
            tier = target_plan.tier(tier_name)
            count, seconds = life.prefetch(
                tier_name, limbs=tier.limb_cap, rotations=tier.rotations,
            )
            per_tier[str(tier_name)] = {"keys": int(count), "seconds": round(float(seconds), 3)}
        easyfhe.cuda.synchronize()
        return {
            "seconds": round(time.perf_counter() - started, 3),
            "tiers": per_tier,
            "resident_keys_gib": native_cache_inventory(native)["total_gib"],
        }

    # T2_PD_DROP_HOST_KEYS=1 (default on a real single card with --warm-keys): once a phase's rotations are resident
    # at their caps the host master copies are dead weight (the D1 smoke runs report 0 key misses in both phases),
    # yet they are ~46 GB of the ~83 GB host RSS the worker holds before Prefill layer 0 -- with --retain-weights
    # host (32 x 0.87 GB pinned) that crossed the container's 110 GiB cgroup limit at Prefill layer ~26 (-9).
    # After the Prefill warm the Prefill-only masters are replaced by zero-storage stubs (the shape is still read by
    # Context._rotation_key_special_mod_start); after the Decode warm every master goes.  A guard turns any later
    # device load of a dropped rotation into an explicit error instead of garbage.
    drop_host_keys = (
        os.environ.get("T2_PD_DROP_HOST_KEYS", "1").lower() in {"1", "true", "yes"}
        and (bool(warm_keys) or emulated) and (not emulated or rank == 0 or timing_only_gold)
        and str(getattr(life.plan, "policy", "")) == "resident"
    )
    dropped_host_keys: set[int] = set()

    def drop_host_rotation_keys(keep, *, candidates=None) -> dict:
        import gc

        keep_set = {int(v) for v in keep}
        candidate_set = None if candidates is None else {int(v) for v in candidates}
        key_map = getattr(native, "left_rot_key_map", None)
        if not isinstance(key_map, dict):
            return {"dropped": 0}
        source = getattr(native, "_source_material", None)
        source_map = getattr(source, "left_rot_key_map", None)
        rss_before = host_memory_gib().get("host_rss_gib")
        count = 0
        for rot in list(key_map):
            if (int(rot) in keep_set or int(rot) in dropped_host_keys
                    or (candidate_set is not None and int(rot) not in candidate_set)):
                continue
            bx, ax = key_map[rot]
            stub = [bx.new_empty((0, int(bx.shape[1]), 0)), ax.new_empty((0, int(ax.shape[1]), 0))]
            key_map[rot] = stub
            if isinstance(source_map, dict) and rot in source_map:
                source_map[rot] = stub
            dropped_host_keys.add(int(rot))
            count += 1
        if count:
            guard_dropped_host_rotations(native, dropped_host_keys)
        gc.collect()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        return {"dropped": count, "kept": len(key_map) - len(dropped_host_keys),
                "host_rss_gib_before": rss_before, "host_rss_gib_after": host_memory_gib().get("host_rss_gib")}

    warm_keys_report: dict = {"enabled": bool(warm_keys), "phases": {}, "drop_host_keys": bool(drop_host_keys)}
    if emulated and warm_keys:
        transport.barrier("prefill/warm/start")
    if warm_keys and (not emulated or rank == 0):
        prefill_warm = warm_phase(prefill_plan)
        if drop_host_keys:
            prefill_warm["host_keys_dropped"] = drop_host_rotation_keys(keep=decode_plan.union)
            if rank == 0:
                print(json.dumps({"phase": "prefill/drop_host_keys", **prefill_warm["host_keys_dropped"]}), flush=True)
        warm_keys_report["phases"]["prefill"] = prefill_warm
        warm_keys_report.update({
            "prefill_seconds": prefill_warm["seconds"],
            "tiers": prefill_warm["tiers"],
            "seconds": prefill_warm["seconds"],
            "resident_keys_gib": prefill_warm["resident_keys_gib"],
            "resident_keys_gib_after_prefill": prefill_warm["resident_keys_gib"],
        })
    if emulated and warm_keys:
        transport.barrier("prefill/warm/done")
    warmup_report: dict = {"enabled": bool(warmup_layer0)}
    if warmup_layer0:
        # Same window as t2_model --warmup-layer0: one discarded pass of the first
        # layer on dummy rows, then drop its bootstrap records. The timed loop
        # below is the comparison window.
        transport.barrier("warmup/start")
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
        warm_outputs, warm_record = run_prefill_layer(
            layers[0], dummy_inputs, first_weights, verify_layer=False, tag_prefix=f"warmup/{layers[0]}",
        )
        _release_ciphers(warm_outputs)
        warm_cache = caches.pop(layers[0], None)
        if warm_cache is not None:
            warm_cache.release()
        bootstrap.take()
        probe.reset_peak()
        easyfhe.cuda.synchronize()
        warmup_report.update({
            "layer": int(layers[0]),
            "seconds": round(time.perf_counter() - started, 3),
            "layer_wall_s": round(float(warm_record["layer_wall_s"]), 4),
            "bootstrap_calls": int(warm_record["bootstrap_calls"]),
            "stations": {name: round(float(v["wall_s"]), 4) for name, v in warm_record["stations"].items()},
        })
        if rank == 0:
            print(json.dumps({"warmup_layer0": {k: warmup_report[k] for k in
                                                ("layer", "seconds", "layer_wall_s", "bootstrap_calls")}}), flush=True)
    transport.barrier("prefill/start")
    prefill_started = time.perf_counter()
    prefill_records = []
    last = layers[-1]
    for position, idx in enumerate(layers):
        weights = first_weights if position == 0 else prefetcher.get(idx)
        first_weights = None
        if position + 1 < len(layers):
            prefetcher.start(layers[position + 1])
        verify_layer = not prefix_k and (
            (idx == last) or (int(verify_every) > 0 and position % int(verify_every) == 0)
        )
        inputs, record = run_prefill_layer(idx, inputs, weights, verify_layer=verify_layer)
        record["weights_wait_s"] = round(prefetcher.wait_seconds.get(idx, 0.0), 3)
        if fast_encode is not None:
            record["fast_encode"] = fast_encode.check()
        weights.release()
        prefill_records.append(record)
        if rank == 0:
            print(json.dumps({"phase": "prefill", "layer": idx, "wall_s": record["layer_wall_s"],
                              "kv_emit_s": record["kv_emit_wall_s"], "bs": record["bootstrap_calls"],
                              "bs_s": record["bootstrap_seconds"],
                              **({"rel_max_err": round(record["oracle"]["rel_max_err"], 6)} if verify_layer else {})}),
                              flush=True)
            if emulated and drop_host_keys and not warm_keys and (rank == 0 or timing_only_gold):
                ready = cached_prefill_only_rotations(native, prefill_plan, decode_plan)
                if ready:
                    dropped = drop_host_rotation_keys(keep=decode_plan.union, candidates=ready)
                    if dropped["dropped"]:
                        print(json.dumps({"phase": "prefill/drop_host_keys", "layer": idx, **dropped}), flush=True)
        if emulated and rank == world_size - 1:
            memlog(f"prefill/layer/{idx}")
    easyfhe.cuda.synchronize()
    memlog("prefill/end", deep=True)
    transport.barrier("prefill/end")
    if emulated:
        memlog("prefill/all-ranks-done", deep=True)
    # The Decode plan is warmed only after this barrier.  Record the resident
    # key footprint here so reports can distinguish completed Prefill from the
    # merged Prefill+Decode run.
    if warm_keys and (not emulated or rank == 0):
        resident_after_prefill = native_cache_inventory(native)["total_gib"]
        warm_keys_report["resident_keys_gib_after_prefill"] = resident_after_prefill
        warm_keys_report["phases"]["prefill"]["resident_keys_gib_after_prefill"] = resident_after_prefill
    prefill_wall = time.perf_counter() - prefill_started
    if emulated and (rank == 0 or timing_only_gold) and drop_host_keys and not warm_keys:
        host_drop = drop_host_rotation_keys(keep=decode_plan.union)
        warm_keys_report["prefill_host_keys_dropped"] = host_drop
        print(json.dumps({"phase": "prefill/drop_host_keys", **host_drop}), flush=True)
    evict_report = None
    if evict_prefill_keys and (not emulated or rank == 0 or timing_only_gold):
        # Single-card budget: drop the Prefill-only rotations (keep the Decode plan + resident extras) before the
        # Decode plan is warmed.  Rotation indices shared by both phases stay cached at the Prefill cap.
        easyfhe.cuda.synchronize()
        evict_stats = life.evict(keep_extra=decode_plan.union, keep_resident=False)
        easyfhe.cuda.synchronize()
        evict_report = {**{k: v for k, v in dict(evict_stats).items() if not isinstance(v, (list, tuple, dict))},
                        "prefill_union": len(prefill_plan.union), "decode_union": len(decode_plan.union),
                        "shared": len(set(prefill_plan.union) & set(decode_plan.union)),
                        "resident_keys_gib_after_evict": native_cache_inventory(native)["total_gib"]}
        warm_keys_report["prefill_evict"] = evict_report
        if rank == 0:
            print(json.dumps({"phase": "prefill/evict", **evict_report}, default=str), flush=True)
        memlog("prefill/evict", deep=True)

    # -- client side: decrypt my rows, assemble on rank 0, LM head -> first token
    started = time.perf_counter()
    own_rows = _decrypt_rows(context, layout, ownership, inputs).astype(np.float32)
    _release_ciphers(inputs)
    inputs = ()
    gathered_rows = transport.gather_to(0, "prefill/rows", {"start": int(ownership.token_start), "rows": own_rows})
    loader = metadata = embedding = client_head = None
    generated: list[dict] = []
    prefill_hidden = None
    def reference_token(step: int) -> int:
        if not reference_tokens:
            raise ValueError(f"{model_name} has no reference token for step {step}; provide --model-dir")
        return int(reference_tokens[min(int(step), len(reference_tokens) - 1)])

    if rank == 0:
        prefill_hidden = np.zeros((int(layout.seq_len), hidden_dim), dtype=np.float32)
        for item in gathered_rows:
            prefill_hidden[int(item["start"]):int(item["start"]) + len(item["rows"])] = item["rows"]
        if model_dir:
            from llama3fhe.application.checkpoint import CheckpointMetadata, WeightLoader
            from llama3fhe.application.client import ClientHead, top_k_logits

            loader = WeightLoader(Path(model_dir), tensor_dtype=np.float32)
            loader.__enter__()
            metadata = CheckpointMetadata.from_json(Path(model_dir) / "config.json")
            embedding = (_load_client_embedding_rows(
                client_embedding_rows, hidden_dim=hidden_dim, vocab_size=int(preset.model_spec.vocab_size))
                if client_embedding_rows is not None else
                loader.get("model.embed_tokens.weight", out_dtype=np.float32))
            if timing_only_gold:
                token_id = reference_token(0)
                top = None
            else:
                client_head = ClientHead(loader, metadata)  # resident: the per-token head read was 1.4-1.65 s
                logits = client_head.logits(prefill_hidden, token_index=-1)
                top = top_k_logits(logits, 5)
                token_id = int(top[0]["token_id"])
        else:
            token_id = reference_token(0)
            top = None
        generated.append({"step": 0, "source": "prefill", "token_id": token_id, "top_k": top})
    client_prefill_seconds = time.perf_counter() - started
    token_ids = transport.all_gather("prefill/token", int(generated[0]["token_id"]) if rank == 0 else -1)
    token_id = int(token_ids[0])
    prefill_oracle = None if prefix_k else _compare(own_rows, oracle_hidden[last][rows])

    # ------------------------------------------------------------ Decode
    hidden_stub = np.zeros((1, int(decode_feature.hidden_dim)), dtype=np.float32)
    input_comm: dict[str, float] = {}

    def encrypt_decode_input(hidden_row):
        own = None
        if int(rank) == 0:
            packed_row = decode_feature.pack(np.asarray(hidden_row, dtype=np.float32).reshape(1, -1), dtype=np.float64)
            own = [context.encrypt(np.asarray(r).reshape(-1), level=int(decode_input_level), slots=int(decode_feature.slots))
                   for r in packed_row]
        return tuple(broadcast_ciphers(transport, "decode/input", own, root=0, crypto_context=context, comm=input_comm))

    def decrypt_decode(ciphers):
        packed_out = np.stack([np.asarray(context.decrypt(c), dtype=np.float64) for c in ciphers])
        return np.asarray(decode_feature.unpack(packed_out), dtype=np.float32)

    decode_rows: list[dict] = []
    poly_hidden_chain = None
    if warm_keys and (not emulated or rank == 0) and int(decode_tokens) > 0:
        decode_warm = warm_phase(decode_plan)
        if drop_host_keys:
            decode_warm["host_keys_dropped"] = drop_host_rotation_keys(keep=())
            if rank == 0:
                print(json.dumps({"phase": "decode/drop_host_keys", **decode_warm["host_keys_dropped"]}), flush=True)
        warm_keys_report["phases"]["decode"] = decode_warm
        warm_keys_report["decode_seconds"] = decode_warm["seconds"]
        warm_keys_report["resident_keys_gib"] = decode_warm["resident_keys_gib"]
        warm_keys_report["seconds"] = round(
            float(warm_keys_report.get("prefill_seconds", 0.0)) + float(decode_warm["seconds"]), 3
        )
        memlog("decode/warm", deep=True)
    elif warm_keys:
        warm_keys_report["seconds"] = warm_keys_report.get("prefill_seconds", 0.0)
    bootstrap.enter_decode()
    if int(decode_tokens) > 0:
        prefetcher.start(layers[0])
    transport.barrier("decode/start")
    decode_started = time.perf_counter()
    replay_query_token = int(token_id) if replay_decode_same_token else None
    for step in range(int(decode_tokens)):
        if replay_decode_same_token:
            token_id = int(replay_query_token)
        token_started = time.perf_counter()
        if rank == 0:
            if isinstance(embedding, dict) and token_id not in embedding:
                raise ValueError(
                    f"client embedding rows {client_embedding_rows} do not contain generated token {token_id} "
                    f"needed for Decode step {step + 1}"
                )
            hidden_row = (embedding[token_id] if embedding is not None else np.zeros(hidden_dim, dtype=np.float32))
            poly_hidden_chain = np.asarray(hidden_row, dtype=np.float32).reshape(1, -1)
        else:
            hidden_row = None
        hidden_ciphers = encrypt_decode_input(hidden_row)
        layer_rows = []
        for layer_index, idx in enumerate(layers):
            weights = prefetcher.get(idx)
            if layer_index + 1 < len(layers):
                prefetcher.start(layers[layer_index + 1])
            elif step + 1 < int(decode_tokens):
                prefetcher.start(layers[0])
            weights.register()
            graph_weights = GraphLayerWeights(**weights.host)
            stations: dict[str, dict] = {}

            def timed(name, fn, _idx=idx, _step=step, _stations=stations):
                return timed_station(_stations, f"{stage_prefix}d_{name}_{_step}_{_idx}", name,
                                     DECODE_STATION_TIERS.get(name), fn)

            original_content_tokens = int(caches[idx].content_tokens)
            if replay_decode_same_token:
                clone_started = time.perf_counter()
                layer_cache = caches[idx].clone()
                easyfhe.cuda.synchronize()
                cache_clone_s = time.perf_counter() - clone_started
            else:
                layer_cache = caches[idx]
                cache_clone_s = 0.0
            position = int(prefix_k) + int(layer_cache.content_tokens)
            decode_approximations = decode_approximations_at_position(approximations.for_layer(idx), position)
            layer_start = time.perf_counter()
            cache_release_s = 0.0
            try:
                with acceleration_patches.hoist_disabled():
                    result = decode_layer_parallel(
                        crypto_context=context, transport=transport, ownership=decode_ownership, weights=graph_weights,
                        approximations=decode_approximations, schedule=decode_sched, config=config,
                        input_ciphers=hidden_ciphers, kv_cache=layer_cache, position=position, layer_idx=idx,
                        bootstrap_operator=bootstrap, feature_layout=decode_feature, attention_layout=decode_attention,
                        pairfold_layout=pairfold_layout, rope_theta=float(preset.model_spec.rope_theta),
                        prefix_bundle=prefix_by_layer.get(idx),
                        split_gate_up=bool(split_gate_up), split_kv_refresh=bool(split_kv_refresh),
                        split_w_o=bool(split_w_o), hidden_stub=hidden_stub, timed=timed, rms_slim_period_rows=1,
                    )
                layer_wall = time.perf_counter() - layer_start
                if replay_decode_same_token:
                    check_replay_cache_isolation(caches[idx], layer_cache, original_content_tokens)
            finally:
                if replay_decode_same_token:
                    release_started = time.perf_counter()
                    layer_cache.release()
                    easyfhe.cuda.synchronize()
                    cache_release_s = time.perf_counter() - release_started
            release_all(hidden_ciphers)
            hidden_ciphers = result.ciphers
            result.ciphers = ()
            row = {
                "layer": int(idx), "position": position, "wall_s": round(layer_wall, 4),
                "cache_clone_s": round(cache_clone_s, 4),
                "cache_release_s": round(cache_release_s, 4),
                "resident_plaintexts": RESIDENT.info() if RESIDENT.enabled() else None,
                "station_wall_sum_s": round(sum(s["wall_s"] for s in stations.values()), 4),
                "bootstrap_calls_total": int(sum(result.bootstrap_calls.values())),
                "bootstrap_seconds": round(sum(s["bootstrap_seconds"] for s in stations.values()), 4),
                "comm_seconds": round(sum(v for k, v in result.comm.items() if k.endswith("_seconds")), 4),
                "comm_bytes_sent": int(sum(v for k, v in result.comm.items() if k.endswith("_sent"))),
                "stations": stations,
                "weights_wait_s": round(prefetcher.wait_seconds.get(idx, 0.0), 3),
            }
            if int(idx) in profile_layers:
                row.update(nested_stage_records(result.nested, int(idx), profile_layers))
            if timing_only_gold:
                row["output_levels"] = result.output_levels
            if verify and poly_kv_arrays is not None and rank == 0:
                from llama3fhe.approx.layer import poly_decode_layer_numpy

                started = time.perf_counter()
                fhe_hidden = decrypt_decode(hidden_ciphers)
                poly = poly_decode_layer_numpy(
                    poly_hidden_chain, cache_key=poly_kv_arrays[idx][0], cache_value=poly_kv_arrays[idx][1],
                    weights=graph_weights, approximations=decode_approximations,
                    attention_layout=decode_attention, layer_idx=int(idx), position=position,
                    rope_theta=float(preset.model_spec.rope_theta), use_alg2_softmax=True,
                    prefix_bundle=prefix_by_layer.get(idx))
                poly_kv_arrays[idx] = (np.concatenate([poly_kv_arrays[idx][0], poly.key], axis=0),
                                       np.concatenate([poly_kv_arrays[idx][1], poly.value], axis=0))
                poly_hidden_chain = np.asarray(poly.hidden, dtype=np.float32)
                diff = np.abs(fhe_hidden.astype(np.float64) - poly_hidden_chain.astype(np.float64))
                row["verify"] = {"seconds": round(time.perf_counter() - started, 3),
                                 "max_abs_diff_vs_poly_chain": float(diff.max()),
                                 "mean_abs_diff_vs_poly_chain": float(diff.mean()),
                                 "fhe_max_abs": float(np.abs(fhe_hidden).max()),
                                 "poly_max_abs": float(np.abs(poly_hidden_chain).max()),
                                 "finite": bool(np.all(np.isfinite(fhe_hidden)))}
            weights.release()
            layer_rows.append(row)
            if rank == 0:
                print(json.dumps({"phase": "decode", "step": step + 1, "layer": idx, "wall_s": row["wall_s"],
                                  "bs": row["bootstrap_calls_total"], "bs_s": row["bootstrap_seconds"],
                                  "comm_s": row["comm_seconds"],
                                  **({"vs_poly": round(row["verify"]["max_abs_diff_vs_poly_chain"], 5)} if "verify" in row else {})}),
                      flush=True)
        easyfhe.cuda.synchronize()
        layers_wall = time.perf_counter() - token_started
        # -- client: decrypt, LM head, next token
        started = time.perf_counter()
        if rank == 0:
            step_hidden = decrypt_decode(hidden_ciphers)
            if timing_only_gold:
                next_id = reference_token(step + 1)
                top = None
            elif client_head is not None:
                from llama3fhe.application.client import top_k_logits

                logits = client_head.logits(step_hidden, token_index=0)
                top = top_k_logits(logits, 5)
                next_id = int(top[0]["token_id"])
            else:
                next_id = reference_token(step + 1)
                top = None
            generated.append({"step": step + 1, "source": "decode", "token_id": next_id, "top_k": top,
                              "finite": bool(np.all(np.isfinite(step_hidden)))})
        release_all(hidden_ciphers)
        hidden_ciphers = ()
        next_ids = transport.all_gather(f"decode/token/{step}", int(generated[-1]["token_id"]) if rank == 0 else -1)
        token_id = int(next_ids[0])
        client_seconds = time.perf_counter() - started
        decode_rows.append({
            "step": step + 1, "token_id": token_id, "wall_s": round(time.perf_counter() - token_started, 4),
            "layers_wall_s": round(layers_wall, 4), "client_seconds": round(client_seconds, 3),
            "layer_wall_sum_s": round(sum(r["wall_s"] for r in layer_rows), 4),
            "cache_clone_sum_s": round(sum(r["cache_clone_s"] for r in layer_rows), 4),
            "cache_release_sum_s": round(sum(r["cache_release_s"] for r in layer_rows), 4),
            "bootstrap_calls": int(sum(r["bootstrap_calls_total"] for r in layer_rows)),
            "bootstrap_seconds": round(sum(r["bootstrap_seconds"] for r in layer_rows), 4),
            "comm_seconds": round(sum(r["comm_seconds"] for r in layer_rows), 4),
            "layers": layer_rows,
        })
        if rank == 0:
            print(json.dumps({"phase": "decode", "step": step + 1, "token_id": token_id,
                              "wall_s": decode_rows[-1]["wall_s"], "layers_wall_s": decode_rows[-1]["layers_wall_s"],
                              "client_s": decode_rows[-1]["client_seconds"]}), flush=True)
    transport.barrier("decode/end")
    decode_wall = time.perf_counter() - decode_started
    for cache in caches.values():
        cache.release()
    caches.clear()
    prefetcher.close()
    if loader is not None:
        loader.__exit__(None, None, None)

    summary = {
        "rank": int(rank), "model": model_name,
        "world_size": world_size, "emulated": bool(emulated),
        "layers": list(layers), "decode_tokens": int(decode_tokens),
        "replay_decode_same_token": bool(replay_decode_same_token),
        "prefix_k": int(prefix_k), "prefix_kv_cache": prefix_kv_cache,
        "setup_seconds": state.setup_seconds, "key_plan": state.key_plan.summary(alpha=int(native.alpha), special_limbs=int(native.K)),
        "key_phase_plans": {
            "prefill": prefill_plan.summary(alpha=int(native.alpha), special_limbs=int(native.K)),
            "decode": decode_plan.summary(alpha=int(native.alpha), special_limbs=int(native.K)),
        },
        "key_plan_sizes": plan_sizes, "resident_keys_gib": native_cache_inventory(native)["total_gib"],
        "sparse_log_slots": list(preset.bootstrap_extra_sparse_log_slots),
        "decode_rows": int(decode_preset.linear_max_plaintext_rows),
        "prefill_rows": int(preset.linear_max_plaintext_rows),
        "mlp_feature_shard_k": int(preset.mlp_feature_shard_k),
        "emit_kv_cache": bool(emit_kv_cache),
        "kv_limbs": int(kv_limbs),
        "kv_value_limbs": int(kv_limbs if kv_value_limbs is None else kv_value_limbs),
        "prefill_bootstrap_timer_sync": bool(stage_sync_enabled()),
        "qk_score_level_drop": None if qk_score_level_drop is None else int(qk_score_level_drop),
        "encode_depth": int(os.environ.get("LLAMA_FHE_LINEAR_ENCODE_DEPTH", "0") or 0),
        "warmup_layer0": warmup_report,
        "shifted_lanes": bool(shifted_lanes), "sl_baby": int(decode_sched.decode_sl_baby_steps),
        "retain_weights": str(prefetcher.retain), "weight_reuploads": int(prefetcher.reuploads),
        "softmax_schain_env": os.environ.get("LLAMA_FHE_SOFTMAX_SCHAIN"),
        "softmax_blockplan_env": os.environ.get("LLAMA_FHE_SOFTMAX_BLOCKPLAN"),
        "softmax_post_exp": softmax_post_exp_description(),
        "evict_prefill_keys": evict_report,
        "warm_keys": warm_keys_report,
        "tables": {"rmsnorm": rms_table_info, "softmax": softmax_table_info, "silu": silu_table_info,
                   "silu_degrees": {k: len(v["tiers"]["standard"]["coefficients"]) - 1 for k, v in silu_all.items()}},
        "odin_patches": patches["info"],
        "profile_layers": sorted(profile_layers),
        "multi_mac": (dict(multi_mac.STATS) if multi_mac is not None and hasattr(multi_mac, "STATS") else None),
        "prefill": {"wall_s": round(prefill_wall, 3), "layers": prefill_records,
                    "client_seconds": round(client_prefill_seconds, 3), "own_rows_oracle": prefill_oracle,
                    "kv_emit_sum_s": round(sum(r["kv_emit_wall_s"] for r in prefill_records), 3),
                    "kv_emit_stream_layers": sum(
                        r["stations"]["kv_emit"].get("stream") == "resident-side"
                        for r in prefill_records
                    ),
                    "kv_emit_stream_fallback_layers": sum(
                        bool(r["stations"]["kv_emit"].get("stream_fallback", False))
                        for r in prefill_records
                    ),
                    "kv_emit_background_sum_s": round(sum(
                        float(r["stations"]["kv_emit"].get("background_wall_s", 0.0))
                        for r in prefill_records
                    ), 3),
                    "kv_emit_bs": int(sum(r["stations"]["kv_emit"]["bootstrap_calls"] for r in prefill_records))},
        "decode": {"wall_s": round(decode_wall, 3), "tokens": decode_rows, "input_comm": dict(input_comm)},
        "generated": generated if rank == 0 else None,
        "prefill_hidden": prefill_hidden,
        "transport": {**transport.stats.to_dict(), "backend": type(transport).__name__},
        "timeline": probe.timeline,
    }
    gathered = summary if timing_only_gold else transport.gather_to(0, "summary", summary)
    transport.barrier("final")
    transport.close()
    if not emulated:
        state.release()
    return gathered if rank == 0 or timing_only_gold else None


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

    # Emulation installs the patches once in this process and shares the (hoist-aware) key plan.
    emu_patches = acceleration_patches.install_from_env(0)

    from llama3fhe.parallel.emulation import run_emulated_world
    from llama3fhe.parallel.memory import native_cache_inventory
    from llama3fhe.parallel.worker import build_worker_state

    decode_preset, decode_sched = decode_preset_and_schedule(preset, kv_limbs=kwargs["kv_limbs"],
                                                             shifted_lanes=kwargs["shifted_lanes"], sl_baby=kwargs["sl_baby"],
                                                             rows=kwargs.get("decode_rows"),
                                                             kv_value_limbs=kwargs.get("kv_value_limbs"))
    started = time.perf_counter()
    union, prefill_phase, decode_phase = emulated_phase_key_plans(
        preset, decode_preset=decode_preset, decode_schedule=decode_sched,
        layers=tuple(kwargs["layers"]), attention_limb_cap=kwargs.get("attention_limb_cap"),
        prefix_k=int(kwargs.get("prefix_k", 0)), multi_mac=emu_patches["multi_mac"],
        emit_kv_cache=bool(kwargs.get("emit_kv_cache", True)),
    )
    shared = build_worker_state(0, preset, tiers=union.names, trace_keys=True, key_plan=union)
    easyfhe.cuda.synchronize()
    setup = {"seconds": round(time.perf_counter() - started, 2),
             "key_plan": shared.key_plan.summary(alpha=int(shared.native.alpha), special_limbs=int(shared.native.K)),
             "resident_keys_gib": native_cache_inventory(shared.native)["total_gib"]}
    print(json.dumps({"emulation_setup": setup}, default=str), flush=True)
    try:
        outcomes = run_emulated_world(
            preset, worker_entry, shared_state=shared,
            kwargs={**kwargs, "emulated_phase_plans": (prefill_phase, decode_phase)},
        )
    finally:
        shared.release()
    return outcomes, setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="rtx6000pro-high")
    parser.add_argument("--model", default=None, help="model geometry (default: preset model, Llama-3-8B)")
    add_runtime_knob_args(parser)
    parser.add_argument("--fixtures", default=None, help="model fixtures (default: fixtures/model for 8B, "
                        "fixtures/llama32_1b_model for 1B)")
    parser.add_argument("--model-dir", default=None, help="checkpoint dir (LM head and norm, plus embedding "
                        "unless --client-embedding-rows is supplied); "
                        "required for 1B Decode, optional for 8B's known gold continuation")
    parser.add_argument("--client-embedding-rows", default=None,
                        help="optional NPZ with token_ids and embeddings [N, hidden_dim]; use selected exact "
                             "embedding rows with a head-only checkpoint for known Decode input tokens")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--decode-tokens", type=int, default=3,
                        help="Decode steps after the Prefill token (8B has 3 known gold steps)")
    parser.add_argument("--replay-decode-same-token", action="store_true",
                        help="warm one full Decode token, then replay its query at the same position "
                             "against cloned, Prefill-emitted ciphertext KV; this is not continuous generation")
    parser.add_argument("--expected-replay-token", type=int, default=None,
                        help="optional independent gold token for every fixed-query replay; "
                             "otherwise only agreement with the first Decode is checked")
    parser.add_argument("--input-level", type=int, default=35)
    parser.add_argument("--decode-input-level", type=int, default=35)
    parser.add_argument("--kv-limbs", type=int, default=12)
    parser.add_argument("--kv-value-limbs", type=int, default=None,
                        help="V cache limb target; defaults to --kv-limbs")
    parser.add_argument("--kv-bootstrap-iterations", type=int, default=1, help="Meta-BTS passes of the kv_emit refresh")
    parser.add_argument("--poly-kv", default=None, help="paris128_prefill_kv_poly.npz for --verify of the Decode layers")
    parser.add_argument("--prefix-kv-cache", default=None, help="plaintext per-layer Prefix KV cache (.npz)")
    parser.add_argument("--poly-kv-matches-prefix", action="store_true",
                        help="declare that --poly-kv was generated with the selected Prefix")
    parser.add_argument("--verify", action="store_true", help="rank 0 decrypts every Decode layer and compares with the poly chain")
    parser.add_argument("--verify-every", type=int, default=0, help="Prefill: decrypt + compare every N layers")
    parser.add_argument("--warm-keys", action="store_true")
    parser.add_argument("--warmup-layer0", action="store_true",
                        help="discard one untimed pass of the first layer before the measured Prefill loop")
    parser.add_argument("--encode-depth", type=int, default=0,
                        help="LLAMA_FHE_LINEAR_ENCODE_DEPTH (BEST10 uses 0)")
    parser.add_argument("--qk-score-level-drop", type=int, default=None,
                        help="Q/K limb drop after qk_gather_axis1; omit to keep the operator default 0. "
                             "BEST10 / t2_model passes 3")
    parser.add_argument("--emit-kv-cache", dest="emit_kv_cache", action="store_true",
                        help="materialise the Decode KV cache during Prefill (default)")
    parser.add_argument("--no-emit-kv-cache", dest="emit_kv_cache", action="store_false",
                        help="Prefill only; requires --decode-tokens 0")
    parser.set_defaults(emit_kv_cache=True)
    parser.add_argument("--mlp-shard-k", type=int, default=None,
                        help="override mlp_feature_shard_k after the world_size=1 clamp (that clamp sets 1)")
    parser.add_argument("--sl", dest="shifted_lanes", action="store_true", default=True, help="(default) SL decode packing")
    parser.add_argument("--no-sl", dest="shifted_lanes", action="store_false")
    parser.add_argument("--sl-baby", type=int, default=16)
    parser.add_argument("--no-split-gate-up", action="store_true")
    parser.add_argument("--no-split-kv-refresh", action="store_true")
    parser.add_argument("--no-split-w-o", action="store_true")
    parser.add_argument("--no-prefetch-weights", action="store_true")
    parser.add_argument("--no-stage-weights", action="store_true")
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="*", default=None)
    parser.add_argument("--ranks-per-device", type=int, default=1)
    parser.add_argument("--transport", choices=PROCESS_TRANSPORTS, default=None)
    parser.add_argument("--max-plaintext-rows", type=int, default=None, help="Prefill chunk rows (Decode clamps its own)")
    parser.add_argument("--decode-max-plaintext-rows", type=int, default=None,
                        help="Decode chunk rows before the world_size clamp (default: the preset's rows, i.e. 2048 "
                             "on rtx6000pro-high even when --world-size 1 clamps the Prefill to 512)")
    parser.add_argument("--retain-weights", choices=WeightPrefetcher.RETAIN_MODES, default="none",
                        help="keep layer weights across Decode tokens: host = pinned copies and re-upload, "
                             "device = resident device copies, none = disk read per token")
    parser.add_argument("--evict-prefill-keys", action="store_true",
                        help="drop the Prefill-only rotation keys after the Prefill phase (single-card memory budget)")
    parser.add_argument("--attention-limb-cap", type=int, default=None,
                        help="Decode attn_core key cap (default: the preset's 13; 0 = full chain, +15 GiB resident)")
    parser.add_argument("--emulate-single-device", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def check_replay_cache_isolation(original_cache, replay_cache, original_content_tokens: int) -> None:
    """Check one appended token without assuming a 128-token input prompt."""

    if (int(original_cache.content_tokens) != int(original_content_tokens)
            or int(replay_cache.content_tokens) != int(original_content_tokens) + 1):
        raise RuntimeError("replay KV isolation failed: original cache changed or append count differs")


def replay_token_expectation(tokens, expected_decode_token: int | None = None) -> list[int]:
    """Separate an explicit gold from repeatability against the first Decode.

    No model-specific continuation is assumed. Without an explicit reference,
    this checks repeatability only and must not be advertised as model accuracy.
    """

    tokens = [int(token) for token in tokens]
    if len(tokens) < 2:
        raise ValueError("a replay expectation needs a Prefill token and a Decode token")
    expected = int(tokens[1] if expected_decode_token is None else expected_decode_token)
    return [tokens[0], *([expected] * (len(tokens) - 1))]


def _station_table(records: list[list[dict]], names) -> dict[str, dict]:
    """``{station: {wall_med_s, wall_max_s, bs}}`` over the records of all ranks x layers."""

    table = {}
    for name in names:
        walls = [r["stations"][name]["wall_s"] for per_rank in records for r in per_rank if name in r["stations"]]
        if walls:
            bs = [r["stations"][name]["bootstrap_calls"] for per_rank in records for r in per_rank if name in r["stations"]]
            table[name] = {"wall_med_s": round(_median(walls), 4), "wall_max_s": round(max(walls), 4),
                           "wall_sum_med_per_rank_s": round(_median(sum(r["stations"][name]["wall_s"] for r in per_rank)
                                                                    for per_rank in records), 3),
                           "bs_med": round(_median(bs), 2)}
    return table


def _prefix_cache_token_count(path: str | Path) -> int:
    """Read Prefix occupancy before creating the worker key plans."""

    import numpy as np

    with np.load(path, allow_pickle=False) as cache:
        prefix_k = int(cache["prefix_k"].item())
    if not 1 <= prefix_k <= 128:
        raise ValueError(f"Prefix cache has unsupported prefix_k={prefix_k}")
    return prefix_k


def _load_client_embedding_rows(path: str | Path, *, hidden_dim: int, vocab_size: int) -> dict[int, object]:
    """Load a small exact checkpoint embedding subset for the client Decode input."""

    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as payload:
            token_ids = np.asarray(payload["token_ids"])
            embeddings = np.asarray(payload["embeddings"])
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"invalid client embedding rows {path}: {exc}") from exc
    if token_ids.ndim != 1 or token_ids.size == 0 or token_ids.dtype.kind not in "iu":
        raise ValueError("client embedding token_ids must be a nonempty integer vector")
    if embeddings.shape != (token_ids.size, int(hidden_dim)) or embeddings.dtype != np.float32:
        raise ValueError(
            f"client embeddings must be float32 with shape {(token_ids.size, int(hidden_dim))}, "
            f"got {embeddings.shape} {embeddings.dtype}"
        )
    ids = [int(token_id) for token_id in token_ids]
    if len(set(ids)) != len(ids) or any(token_id < 0 or token_id >= int(vocab_size) for token_id in ids):
        raise ValueError(f"client embedding token_ids must be unique and in [0, {int(vocab_size)})")
    if not bool(np.isfinite(embeddings).all()):
        raise ValueError("client embeddings must all be finite")
    return {token_id: embeddings[index] for index, token_id in enumerate(ids)}


def _check_client_checkpoint_tensors(model_dir: str | Path, *, require_embedding: bool = True) -> None:
    """Check client tensors from safetensors headers before starting GPU workers."""

    from llama3fhe.application.checkpoint import WeightLoader

    required = (("model.embed_tokens.weight",) if require_embedding else ()) + (
        "lm_head.weight", "model.norm.weight")
    try:
        with WeightLoader(model_dir) as loader:
            missing = []
            for name in required:
                if not loader.has_tensor(name):
                    missing.append(name)
                elif loader.weight_map is not None:
                    # The index alone may be stale: inspect the named shard's
                    # header without decoding its tensor payload.
                    if name not in loader._reader(loader.weight_map[name]):
                        missing.append(name)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot inspect checkpoint tensors in {model_dir}: {exc}") from exc
    if missing:
        raise SystemExit(f"checkpoint {model_dir} is missing client tensors: {', '.join(missing)}")


def main() -> int:
    import numpy as np

    args = parse_args()
    preset = preset_by_name(args.preset)
    preset = apply_runtime_knobs(args, preset)
    requested_model = None
    if args.model is not None:
        from llama3fhe.model_spec import model_spec_by_name

        try:
            requested_model = model_spec_by_name(str(args.model))
        except KeyError as exc:
            raise SystemExit(f"--model {args.model}: {exc}") from exc
    # The 1B FM prompt has 8 ciphertexts; the preset's 8-rank default would
    # leave one unpaired ciphertext per rank. Select W1 unless requested.
    default_world_size = (
        1 if requested_model is not None and requested_model.name == "llama3.2-1b"
        else int(preset.world_size)
    )
    world_size = int(default_world_size if args.world_size is None else args.world_size)
    preset_rows = int(preset.linear_max_plaintext_rows)
    if world_size == 1:
        # one rank owns the 16 ciphertexts (t2_decode.py --world-size 1); Prefill rows clamped like
        # ParallelPreset.with_world_size (16 ciphertexts beside a 2048-row chunk is unmeasured)
        from llama3fhe.parallel.presets import SHARED_CARD_MAX_PLAINTEXT_ROWS

        preset = replace(preset, world_size=1, ciphers_per_rank=16, mlp_feature_shard_k=1,
                         linear_max_plaintext_rows=min(preset_rows, int(SHARED_CARD_MAX_PLAINTEXT_ROWS)))
    elif world_size != int(preset.world_size):
        preset = preset.with_world_size(world_size)
    if world_size not in (1, 2, 4, 8):
        raise SystemExit("--world-size must be 1, 2, 4 or 8")
    if requested_model is not None:
        try:
            preset = preset.for_model(requested_model.name)
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"--model {args.model}: {exc}") from exc
    if args.emulate_single_device and world_size == 1:
        raise SystemExit("--emulate-single-device needs --world-size >= 2 (use the real --world-size 1 path instead)")
    if args.transport is not None:
        os.environ["T2_TRANSPORT"] = str(args.transport)
    if not args.emulate_single_device:
        expand_devices(preset.world_size, args.devices, ranks_per_device=int(args.ranks_per_device))
        if int(args.ranks_per_device) > 1:
            preset = preset.for_colocation(int(args.ranks_per_device))
    if args.max_plaintext_rows is not None:
        preset = replace(preset, linear_max_plaintext_rows=int(args.max_plaintext_rows))
    if args.mlp_shard_k is not None:
        preset = replace(preset, mlp_feature_shard_k=int(args.mlp_shard_k))
    if int(args.encode_depth) < 0:
        raise SystemExit("--encode-depth must be zero or positive")
    if int(args.kv_limbs) <= 1 or (args.kv_value_limbs is not None and int(args.kv_value_limbs) <= 1):
        raise SystemExit("KV cache targets must be at least 2 limbs")
    if not args.emit_kv_cache and int(args.decode_tokens) > 0:
        raise SystemExit("--no-emit-kv-cache requires --decode-tokens 0")
    if args.replay_decode_same_token and (int(args.decode_tokens) < 2 or not args.emit_kv_cache):
        raise SystemExit("--replay-decode-same-token requires --decode-tokens >= 2 and KV emission")
    if args.expected_replay_token is not None and not args.replay_decode_same_token:
        raise SystemExit("--expected-replay-token requires --replay-decode-same-token")
    if os.environ.get("T2_PD_PREFILL_ONLY_KEYS", "0").lower() not in {"1", "true", "yes", "on"}:
        preset = combined_preset(preset)
    decode_rows_base = int(args.decode_max_plaintext_rows) if args.decode_max_plaintext_rows is not None else (
        preset_rows if world_size == 1 else int(preset.linear_max_plaintext_rows))
    num_layers = int(preset.model_spec.num_layers)
    layers = parse_layers(f"0-{num_layers - 1}" if str(args.layers).lower() == "all" else args.layers,
                          count=num_layers)
    default_fixtures = "fixtures/llama32_1b_model" if preset.model_spec.name == "llama3.2-1b" else "fixtures/model"
    fixtures = Path(default_fixtures if args.fixtures is None else args.fixtures).resolve()
    manifest_path = fixtures / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        built_for = str(manifest.get("model", "llama3-8b"))
        if built_for != preset.model_spec.name:
            raise SystemExit(f"fixtures {fixtures} were built for {built_for}, not {preset.model_spec.name}")
    elif preset.model_spec.name != "llama3-8b":
        raise SystemExit(f"missing {manifest_path} for model {preset.model_spec.name}")
    else:
        manifest = {}
    for idx in layers:
        if not (fixtures / f"layer{idx}_weights.npz").exists():
            raise SystemExit(f"missing {fixtures / f'layer{idx}_weights.npz'}")
    if args.verify and args.poly_kv is None:
        raise SystemExit("--verify needs --poly-kv (paris128_prefill_kv_poly.npz)")
    if preset.model_spec.name != "llama3-8b" and args.model_dir is None and int(args.decode_tokens) > 0:
        raise SystemExit(f"{preset.model_spec.name} Decode requires --model-dir for real token selection")
    if args.client_embedding_rows is not None and args.model_dir is None:
        raise SystemExit("--client-embedding-rows requires --model-dir")
    embedding_rows_path = None
    embedding_rows_lookup = None
    if args.client_embedding_rows is not None:
        embedding_rows_path = Path(args.client_embedding_rows).expanduser().resolve()
        try:
            embedding_rows_lookup = _load_client_embedding_rows(
                embedding_rows_path, hidden_dim=int(preset.model_spec.hidden_dim),
                vocab_size=int(preset.model_spec.vocab_size))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if args.model_dir is not None:
        from llama3fhe.application.checkpoint import CheckpointMetadata

        checkpoint_config = Path(args.model_dir).resolve() / "config.json"
        if not checkpoint_config.is_file():
            raise SystemExit(f"missing {checkpoint_config}")
        metadata = CheckpointMetadata.from_json(checkpoint_config)
        spec = preset.model_spec
        actual = (metadata.hidden_size, metadata.intermediate_size, metadata.num_hidden_layers,
                  metadata.num_attention_heads, metadata.num_key_value_heads)
        expected = (spec.hidden_dim, spec.intermediate_dim, spec.num_layers,
                    spec.query_heads, spec.key_value_heads)
        if actual != expected or float(metadata.rope_theta) != float(spec.rope_theta):
            raise SystemExit(f"checkpoint {checkpoint_config} has geometry {actual}, rope_theta={metadata.rope_theta}; "
                             f"expected {spec.name} {expected}, rope_theta={spec.rope_theta}")
        _check_client_checkpoint_tensors(checkpoint_config.parent,
                                         require_embedding=embedding_rows_path is None)
    prefix_path = None if args.prefix_kv_cache is None else Path(args.prefix_kv_cache).resolve()
    prefix_k = 0 if prefix_path is None else _prefix_cache_token_count(prefix_path)
    if manifest and int(manifest.get("prefix_k", 0) or 0) != int(prefix_k):
        raise SystemExit(f"fixtures {fixtures} have prefix_k={manifest.get('prefix_k', 0)}, "
                         f"but the selected Prefix KV cache has prefix_k={prefix_k}")
    if prefix_k and layers[0] != 0:
        raise SystemExit("Prefix Prefill must start at layer 0; intermediate fixture hidden states have no Prefix")
    if prefix_k and args.model_dir is None:
        raise SystemExit("Prefix cache verification requires --model-dir")
    if prefix_k and args.verify and not args.poly_kv_matches_prefix:
        raise SystemExit("Prefix --verify needs --poly-kv-matches-prefix and a Prefix-matched --poly-kv")
    if prefix_k and int(args.verify_every) > 0:
        raise SystemExit("--verify-every uses the no-Prefix fixture trajectory and is unavailable with Prefix")
    if prefix_k:
        from llama3fhe.approx.prefix_cache_io import model_cache_metadata

        prefix_model_metadata = model_cache_metadata(args.model_dir)
    else:
        prefix_model_metadata = None
    reference_tokens = GOLD_TOKENS if preset.model_spec.name == "llama3-8b" else ()
    if preset.model_spec.name in ("llama3.2-1b", "llama3.2-3b-padded"):
        poly_top_k = manifest.get("poly_top_k") or []
        if poly_top_k:
            reference_tokens = (int(poly_top_k[0]["token_id"]),)
        elif args.model_dir is None:
            raise SystemExit(f"fixtures {fixtures} have no poly_top_k token and --model-dir was not supplied")
    if (int(args.decode_tokens) > 0 and embedding_rows_lookup is not None and reference_tokens
            and reference_tokens[0] not in embedding_rows_lookup):
        raise SystemExit(f"client embedding rows {embedding_rows_path} do not contain "
                         f"the fixture's first token {reference_tokens[0]}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else ROOT / "artifacts" / f"t2_prefill_decode_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "fixtures": str(fixtures), "layers": layers, "decode_tokens": int(args.decode_tokens),
        "input_level": int(args.input_level), "decode_input_level": int(args.decode_input_level),
        "kv_limbs": int(args.kv_limbs),
        "kv_value_limbs": None if args.kv_value_limbs is None else int(args.kv_value_limbs),
        "model_dir": None if args.model_dir is None else str(Path(args.model_dir).resolve()),
        "client_embedding_rows": None if embedding_rows_path is None else str(embedding_rows_path),
        "poly_kv": None if args.poly_kv is None else str(Path(args.poly_kv).resolve()),
        "prefix_kv_cache": None if prefix_path is None else str(prefix_path), "prefix_k": int(prefix_k),
        "prefix_model_metadata": prefix_model_metadata,
        "verify": bool(args.verify), "verify_every": int(args.verify_every), "warm_keys": bool(args.warm_keys),
        "shifted_lanes": bool(args.shifted_lanes), "sl_baby": int(args.sl_baby),
        "split_gate_up": not args.no_split_gate_up, "split_kv_refresh": not args.no_split_kv_refresh,
        "split_w_o": not args.no_split_w_o, "kv_bootstrap_iterations": int(args.kv_bootstrap_iterations),
        "prefetch_weights": not args.no_prefetch_weights, "stage_weights": not args.no_stage_weights,
        "attention_limb_cap": args.attention_limb_cap, "decode_rows": int(decode_rows_base),
        "retain_weights": str(args.retain_weights), "evict_prefill_keys": bool(args.evict_prefill_keys),
        "emit_kv_cache": bool(args.emit_kv_cache),
        "qk_score_level_drop": args.qk_score_level_drop,
        "warmup_layer0": bool(args.warmup_layer0),
        "encode_depth": int(args.encode_depth),
        "reference_tokens": tuple(reference_tokens),
        "replay_decode_same_token": bool(args.replay_decode_same_token),
    }
    decode_rows = decode_rows_for_world(int(decode_rows_base), world_size, int(preset.model_spec.hidden_dim))
    print(json.dumps({"preset": preset.name, "model": preset.model_spec.name,
                      "world_size": world_size, "layers": [layers[0], layers[-1]],
                      "prefix_k": int(prefix_k),
                      "sparse_log_slots": list(preset.bootstrap_extra_sparse_log_slots),
                      "prefill_rows": int(preset.linear_max_plaintext_rows),
                      "mlp_feature_shard_k": int(preset.mlp_feature_shard_k),
                      "decode_rows": decode_rows,
                      "emit_kv_cache": bool(args.emit_kv_cache),
                      "qk_score_level_drop": args.qk_score_level_drop,
                      "warmup_layer0": bool(args.warmup_layer0),
                      "encode_depth": int(args.encode_depth),
                      "kv_limbs": int(args.kv_limbs),
                      "kv_value_limbs": int(args.kv_limbs if args.kv_value_limbs is None else args.kv_value_limbs),
                      "sl": bool(args.shifted_lanes), "sl_baby": int(args.sl_baby)}), flush=True)
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
    assert per_rank is not None and len(per_rank) == world_size
    per_rank.sort(key=lambda s: int(s["rank"]))
    rank0 = per_rank[0]
    prefill_hidden = rank0.pop("prefill_hidden")
    for s in per_rank[1:]:
        s.pop("prefill_hidden", None)
    last = layers[-1]
    if prefix_k:
        final_error = None
    else:
        with np.load(fixtures / "hidden_trajectory.npz") as data:
            expected = np.asarray(data[f"hidden_{last + 1}"], dtype=np.float32)
        final_error = _compare(np.asarray(prefill_hidden), expected)
    np.save(out_dir / f"hidden_{last + 1}_fhe.npy", np.asarray(prefill_hidden))

    generated = rank0["generated"]
    tokens = [int(g["token_id"]) for g in generated]
    gold = list(GOLD_TOKENS[:len(tokens)]) if (
        not args.replay_decode_same_token and not prefix_k and preset.model_spec.name == "llama3-8b"
    ) else None
    gold_match = (tokens == gold) if (gold is not None and args.model_dir and last == num_layers - 1) else None
    replay_expected = replay_token_expectation(tokens, args.expected_replay_token) if (
        args.replay_decode_same_token) else None
    replay_reference_match = (tokens == replay_expected) if replay_expected is not None else None
    prefill_layers = [s["prefill"]["layers"] for s in per_rank]
    decode_layer_records = [[row for tok in s["decode"]["tokens"] for row in tok["layers"]] for s in per_rank]
    prefill_walls = [max(s["prefill"]["layers"][i]["layer_wall_s"] for s in per_rank) for i in range(len(layers))]
    decode_token_walls = [max(s["decode"]["tokens"][i]["wall_s"] for s in per_rank) for i in range(int(args.decode_tokens))]

    def warm_phase_max(field: str):
        values = [s["warm_keys"].get(field) for s in per_rank
                  if s["warm_keys"].get(field) is not None]
        return round(max(float(value) for value in values), 3) if values else None

    result = {
        "preset": preset.name, "model": preset.model_spec.name,
        "client_embedding_rows": None if embedding_rows_path is None else str(embedding_rows_path),
        "world_size": world_size, "emulated_single_device": bool(args.emulate_single_device),
        "prefix_k": int(prefix_k), "prefix_kv_cache": None if prefix_path is None else str(prefix_path),
        "retain_weights": str(args.retain_weights), "evict_prefill_keys": bool(args.evict_prefill_keys),
        "weight_reuploads_rank0": rank0.get("weight_reuploads"), "prefill_evict": rank0.get("evict_prefill_keys"),
        "softmax_schain_env": rank0.get("softmax_schain_env"),
        "softmax_blockplan_env": rank0.get("softmax_blockplan_env"),
        "softmax_post_exp": rank0.get("softmax_post_exp"),
        "emulation_setup": setup or None, "timestamp": stamp, "layers": list(layers),
        "sparse_log_slots": list(preset.bootstrap_extra_sparse_log_slots),
        "prefill_rows": int(preset.linear_max_plaintext_rows), "decode_rows": decode_rows,
        "mlp_feature_shard_k": int(preset.mlp_feature_shard_k),
        "emit_kv_cache": bool(args.emit_kv_cache),
        "kv_limbs": int(args.kv_limbs),
        "kv_value_limbs": int(args.kv_limbs if args.kv_value_limbs is None else args.kv_value_limbs),
        "prefill_bootstrap_timer_sync": rank0.get("prefill_bootstrap_timer_sync"),
        "qk_score_level_drop": args.qk_score_level_drop,
        "encode_depth": int(args.encode_depth),
        "warmup_layer0": rank0.get("warmup_layer0"),
        "shifted_lanes": bool(args.shifted_lanes), "sl_baby": int(args.sl_baby),
        "total_wall_s": round(total, 2), "worker_wall_s": [round(o.wall_seconds, 2) for o in outcomes],
        "setup_seconds": rank0["setup_seconds"], "key_plan": rank0["key_plan"], "key_plan_sizes": rank0["key_plan_sizes"],
        "key_phase_plans": rank0["key_phase_plans"],
        "resident_keys_gib_max": round(max(float(s["resident_keys_gib"]) for s in per_rank), 3),
        "warm_keys_seconds_max": warm_phase_max("seconds"),
        "warm_keys_prefill_seconds_max": warm_phase_max("prefill_seconds"),
        "warm_keys_decode_seconds_max": warm_phase_max("decode_seconds"),
        "resident_keys_gib_after_prefill_max": warm_phase_max("resident_keys_gib_after_prefill"),
        "prefill": {
            "wall_max_s": round(max(s["prefill"]["wall_s"] for s in per_rank), 3),
            "layer_wall_sum_s": round(sum(prefill_walls), 3),
            "layer_wall_first_s": prefill_walls[0],
            "layer_wall_steady_med_s": round(_median(prefill_walls[1:]), 3) if len(prefill_walls) > 1 else None,
            "kv_emit_sum_max_s": round(max(s["prefill"]["kv_emit_sum_s"] for s in per_rank), 3),
            "kv_emit_stream_layers_min": min(s["prefill"].get("kv_emit_stream_layers", 0) for s in per_rank),
            "kv_emit_stream_layers_max": max(s["prefill"].get("kv_emit_stream_layers", 0) for s in per_rank),
            "kv_emit_stream_fallback_layers_max": max(
                s["prefill"].get("kv_emit_stream_fallback_layers", 0) for s in per_rank
            ),
            "kv_emit_background_sum_max_s": round(max(
                s["prefill"].get("kv_emit_background_sum_s", 0.0) for s in per_rank
            ), 3),
            "kv_emit_bs_per_rank": rank0["prefill"]["kv_emit_bs"],
            "client_seconds": rank0["prefill"]["client_seconds"],
            "bootstrap_calls_rank0": int(sum(r["bootstrap_calls"] for r in rank0["prefill"]["layers"])),
            "final_hidden_error": final_error,
            "stations": _station_table(prefill_layers, PREFILL_STATIONS),
            "peak_alloc_gib_max": round(max(st["peak_alloc_gib"] for s in per_rank for r in s["prefill"]["layers"]
                                            for st in r["stations"].values()), 2),
        },
        "decode": {
            "wall_max_s": round(max(s["decode"]["wall_s"] for s in per_rank), 3),
            "token_wall_max_s": decode_token_walls,
            "token_wall_med_s": round(_median(decode_token_walls), 3) if decode_token_walls else None,
            "replay_same_token": bool(args.replay_decode_same_token),
            "replay_warmup_passes": 1 if args.replay_decode_same_token else 0,
            "replay_position": (
                rank0["decode"]["tokens"][0]["layers"][0]["position"]
                if args.replay_decode_same_token else None
            ),
            "cache_clone_sum_s_rank0": [t["cache_clone_sum_s"] for t in rank0["decode"]["tokens"]],
            "cache_release_sum_s_rank0": [t["cache_release_sum_s"] for t in rank0["decode"]["tokens"]],
            "hot_layer_wall_sum_median_s": (
                round(_median(t["layer_wall_sum_s"] for t in rank0["decode"]["tokens"][1:]), 4)
                if args.replay_decode_same_token else None
            ),
            "layer_wall_med_s": round(_median(r["wall_s"] for per in decode_layer_records for r in per), 4)
            if decode_layer_records and decode_layer_records[0] else None,
            "client_seconds": [t["client_seconds"] for t in rank0["decode"]["tokens"]],
            "weights_wait_sum_s": [round(sum(r["weights_wait_s"] for r in t["layers"]), 3) for t in rank0["decode"]["tokens"]],
            "stations": _station_table(decode_layer_records, list(DECODE_STATION_TIERS) + [
                "input_rmsnorm", "qkv_all_reduce", "kv_append", "folded_o_all_reduce", "folded_o_refresh", "w_o_all_reduce",
                "attention_residual_add", "post_attention_bootstrap_rmsnorm", "mlp_input_modulus_drop", "pre_mlp_bootstrap",
                "mlp_residual"]),
            "verify_worst": (max((r["verify"]["max_abs_diff_vs_poly_chain"] for r in decode_layer_records[0] if "verify" in r),
                                 default=None)),
        },
        "generated": generated, "tokens": tokens,
        "replay_expected_tokens": replay_expected,
        "replay_reference_match": replay_reference_match,
        "replay_reference_source": (
            "explicit_gold" if args.expected_replay_token is not None else "first_decode_self_consistency"
        ) if args.replay_decode_same_token else None,
        "gold_tokens": None if prefix_k or preset.model_spec.name != "llama3-8b" else list(GOLD_TOKENS),
        "gold_match": gold_match,
        "args": vars(args), "ranks": per_rank,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ("preset", "world_size", "total_wall_s", "resident_keys_gib_max",
                                              "warm_keys_seconds_max", "warm_keys_prefill_seconds_max",
                                              "warm_keys_decode_seconds_max", "resident_keys_gib_after_prefill_max",
                                              "sparse_log_slots")}))
    print(json.dumps({"prefill": {k: v for k, v in result["prefill"].items() if k not in ("stations", "final_hidden_error")},
                      "final_hidden_error": (None if final_error is None else
                                             {k: final_error[k] for k in ("max_abs_err", "rel_max_err", "finite")})}))
    print(json.dumps({"prefill_stations": result["prefill"]["stations"]}))
    print(json.dumps({"decode": {k: v for k, v in result["decode"].items() if k != "stations"}}))
    print(json.dumps({"decode_stations": result["decode"]["stations"]}))
    print(json.dumps({"tokens": tokens, "gold": gold, "gold_match": gold_match, "output": str(out_dir)}))
    return 0 if ((gold_match is None or gold_match) and
                 (replay_reference_match is None or replay_reference_match)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
