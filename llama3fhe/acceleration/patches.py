"""Install the requested runtime accelerations before creating an FHE context.

The shared entry point installs fused EasyFHE kernels, optional host caches,
batched bootstrap, deferred Encode range checks, fused Encode stage 2 and
multi-input BSGS MAC. Each optimization is selected by its environment switch.
The returned handles support hoist-key planning, range checks and operation
counting; ``info`` records the effective configuration.
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

ENV_KEYS = (
    "ODIN_SYLPH_KERNELS", "ODIN_SYLPH_HOSTCACHE", "ODIN_BATCH_BS", "ODIN_BATCH_BS_MAX", "ODIN_FAST_ENCODE",
    "ODIN_FAST_STAGE2", "ODIN_MULTI_MAC", "ODIN_MULTI_MAC_TI", "ODIN_MULTI_GIANT_HOIST", "ODIN_SPARSE_NONEXT_BABY",
    "ODIN_SPARSE_GPU_MASKS", "ODIN_SPARSE_FUSED", "COUPLET_KERNEL_ROOT", "COUPLET_KERNEL_DIR",
    "COUPLET_CUDA_ARCH", "SYLPH_ROOT", "SYLPH_CUDA_ARCH", "PYTORCH_CUDA_ALLOC_CONF",
    # Decode-side knobs recorded alongside (not installed here): cross-token weight plaintext store and the
    # side-stream encode pipeline of the linear operator
    "LLAMA_FHE_LINEAR_PLAIN_RESIDENT", "LLAMA_FHE_LINEAR_PLAIN_RESIDENT_GIB", "LLAMA_FHE_LINEAR_ENCODE_DEPTH",
    "LLAMA_FHE_SOFTMAX_SCHAIN", "T2_SILU_COEFFS",
)


def env_summary() -> dict[str, str]:
    """Effective acceleration and source settings for the run record."""
    return {k: os.environ[k] for k in ENV_KEYS if k in os.environ}


def _on(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def any_requested() -> bool:
    return any(_on(k) for k in ("ODIN_SYLPH_KERNELS", "ODIN_BATCH_BS", "ODIN_FAST_ENCODE", "ODIN_FAST_STAGE2",
                                "ODIN_MULTI_MAC"))


_INSTALLED: dict | None = None
_INSTALL_LOCK = __import__("threading").Lock()


def install_from_env(rank: int = 0, *, log=print) -> dict:
    """Install every requested patch; idempotent per process.

    The single-card emulation runs every rank as a thread of one process (``llama3fhe.parallel.emulation``):
    the first caller installs, later callers get the same handle (``shared_kernels.install`` can not run
    twice -- the shared kernel library is loaded once per process).
    """
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED is not None:
            return _INSTALLED
        _INSTALLED = _install_from_env(rank, log=log)
        return _INSTALLED


def _install_from_env(rank: int = 0, *, log=print) -> dict:
    out: dict = {"extra_kernel_modules": [], "fast_encode": None, "multi_mac": None, "info": {}}
    tag = f"[rank {rank}]"
    if _on("ODIN_SYLPH_KERNELS"):
        from . import shared_kernels

        info = shared_kernels.install()
        out["extra_kernel_modules"] = info.pop("modules")
        out["info"]["sylph_kernels"] = {k: v for k, v in info.items() if k != "patched"}
        out["info"]["sylph_kernels"]["patched_count"] = len(info.get("patched", ()))
        log(f"{tag} shared kernels: {info}", flush=True)
        if _on("ODIN_BATCH_BS"):
            from . import batch_bootstrap

            out["info"]["batch_bs"] = batch_bootstrap.install()
            log(f"{tag} batch bs: {out['info']['batch_bs']}", flush=True)
    elif _on("ODIN_BATCH_BS"):
        raise SystemExit("ODIN_BATCH_BS=1 needs ODIN_SYLPH_KERNELS=1 (shared batched BSGS / EvalMod kernels)")
    if _on("ODIN_FAST_ENCODE"):
        from . import fast_encode

        out["fast_encode"] = fast_encode
        out["info"]["fast_encode"] = fast_encode.install()
        log(f"{tag} fast encode: {out['info']['fast_encode']}", flush=True)
    if _on("ODIN_FAST_STAGE2"):
        from . import fast_stage2

        out["info"]["fast_stage2"] = fast_stage2.install()
        log(f"{tag} fast stage2: {out['info']['fast_stage2']}", flush=True)
    if _on("ODIN_MULTI_MAC"):
        from . import multi_mac

        out["multi_mac"] = multi_mac
        out["info"]["multi_mac"] = multi_mac.install()
        log(f"{tag} multi mac: {out['info']['multi_mac']}", flush=True)
    if _on("ODIN_SYNC_CENSUS"):
        from . import sync_census as sync_census_module

        out["info"]["sync_census"] = sync_census_module.install()
        log(f"{tag} sync census: {out['info']['sync_census']}", flush=True)
    out["info"]["env"] = env_summary()
    return out


def hoist_rotations(preset, multi_mac, *, dimension: int = 4096) -> tuple[int, ...] | None:
    """Linear-tier rotations the double-hoisted giant step needs for the Prefill MLP plan (``None`` when off)."""
    if multi_mac is None or not multi_mac.hoist_enabled():
        return None
    from llama3fhe.graph import _linear_config

    sched = preset.layer_schedule()
    plan = _linear_config(sched, sched.mlp_max_plaintext_rows, sched.mlp_hoist_strategy).plan(
        dimension=int(dimension), token_lanes=int(preset.slots) // int(dimension))
    return tuple(multi_mac.hoist_key_offsets(int(plan.giant_offset), max(c.giant_count for c in plan.chunks)))


class hoist_disabled:
    """``with hoist_disabled():`` -- run a phase whose key plan lacks the hoisted giant keys (the Decode SL
    linear tiers) with ``ODIN_MULTI_GIANT_HOIST`` forced off; restores the previous value on exit."""

    def __enter__(self):
        self._prev = os.environ.get("ODIN_MULTI_GIANT_HOIST")
        if self._prev is not None:
            os.environ["ODIN_MULTI_GIANT_HOIST"] = "0"
        return self

    def __exit__(self, *exc):
        if self._prev is not None:
            os.environ["ODIN_MULTI_GIANT_HOIST"] = self._prev
        return False
