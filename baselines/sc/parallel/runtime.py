"""SC worker launcher using Couplet's ciphertext wire and host transport.

One spawned process is bound to one physical GPU.  The entrypoint is imported
*after* ``CUDA_VISIBLE_DEVICES`` is set in the child, so its EasyFHE imports
cannot initialize CUDA on the wrong card.  This module imports no CUDA code at
module scope.  ``cpu_only`` exists solely for protocol tests; it produces no
inference or performance data.

Example::

    outcomes = run_world(
        "my_sc_runner:run_rank", 4, odin_root="/path/to/repo",
        kwargs={"chain": "/path/to/assets/chain.json"},
    )

``run_rank(runtime, *, chain)`` receives ``runtime.transport`` (Couplet's
``HostStagingTransport``), and may move real ciphertexts using
``runtime.pack`` / ``runtime.unpack`` and the rank-ordered collectives.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
import queue
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class RankAudit:
    rank: int
    world_size: int
    physical_device: int
    visible_devices: str
    device: str
    cuda_device_count: int
    cuda_device_name: str | None
    gpu_uuid: str | None
    hostname: str
    pid: int
    cpu_protocol_only: bool
    emulated_colocation: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerOutcome:
    rank: int
    ok: bool
    audit: RankAudit | None = None
    result: Any = None
    error: str | None = None


@dataclass
class RankRuntime:
    rank: int
    world_size: int
    physical_device: int
    odin_root: str
    transport: Any
    audit: RankAudit

    @property
    def device(self) -> str:
        return self.audit.device

    def pack(self, cipher: Any, *, tag: str = "", host: bool = True) -> Any:
        """Create an Couplet ``CipherPacket`` preserving all CKKS metadata."""

        _, wire = _odin_modules(Path(self.odin_root))
        return wire.pack_cipher(cipher, tag=tag, host=host)

    def unpack(self, packet: Any, *, template: Any | None = None) -> Any:
        """Reconstruct a ciphertext on this rank's local ``cuda`` device."""

        _, wire = _odin_modules(Path(self.odin_root))
        return wire.unpack_cipher(packet, device=self.device, template=template)

    def all_gather(self, tag: str, payload: Any, *, timeout: float | None = None) -> list[Any]:
        """Rank-ordered symmetric gather; payload may contain CipherPackets."""

        return self.transport.all_gather(tag, payload, timeout=timeout)

    def broadcast_packet(
        self, tag: str, payload: Any | None = None, *, root: int = 0,
        timeout: float | None = None,
    ) -> Any:
        """Broadcast one CipherPacket through Couplet's shared-memory collective."""

        root = int(root)
        if not 0 <= root < self.world_size:
            raise ValueError(f"root {root} outside world of size {self.world_size}")
        if (self.rank == root) != (payload is not None):
            raise ValueError("only the broadcast root must supply a packet")
        gathered = self.transport.all_gather(f"broadcast/{tag}", payload, timeout=timeout)
        if len(gathered) != self.world_size or any(
            value is not None for rank, value in enumerate(gathered) if rank != root
        ) or gathered[root] is None:
            raise RuntimeError(f"invalid broadcast payloads at tag {tag!r}")
        return gathered[root]

    def send(self, dst: int, tag: str, payload: Any) -> None:
        self.transport.send(dst, tag, payload)

    def recv(self, src: int, tag: str, *, timeout: float | None = None) -> Any:
        return self.transport.recv(src, tag, timeout=timeout)


def _discover_odin_root(explicit: str | os.PathLike[str] | None) -> Path:
    if explicit is not None:
        candidates = [Path(explicit)]
    elif os.environ.get("SC_ODIN_ROOT"):
        candidates = [Path(os.environ["SC_ODIN_ROOT"])]
    else:
        candidates = []
        for parent in Path(__file__).resolve().parents:
            candidates.append(parent)
            worktrees = parent / "worktrees"
            if worktrees.is_dir():
                candidates.extend(sorted(worktrees.glob("odin-main-experiments-*"), reverse=True))
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "llama3fhe" / "parallel" / "wire.py").is_file() and (
            root / "llama3fhe" / "parallel" / "prefill" / "transport.py"
        ).is_file():
            return root
    raise FileNotFoundError(
        "Couplet wire/transport source not found; pass odin_root= or set SC_ODIN_ROOT"
    )


def _odin_modules(root: Path) -> tuple[Any, Any]:
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    transport = importlib.import_module("llama3fhe.parallel.prefill.transport")
    wire = importlib.import_module("llama3fhe.parallel.wire")
    for module in (transport, wire):
        source = Path(module.__file__).resolve()
        if not source.is_relative_to(root):
            raise RuntimeError(
                f"loaded {module.__name__} from {source}, expected Couplet root {root}"
            )
    return transport, wire


def _entry_path(entry: str | Callable[..., Any]) -> str:
    if isinstance(entry, str):
        if ":" not in entry or not all(entry.split(":", 1)):
            raise ValueError("entry must be 'module:function'")
        return entry
    module = getattr(entry, "__module__", "")
    name = getattr(entry, "__qualname__", "")
    if not module or module in {"__main__", "__mp_main__"} or not name or "<locals>" in name:
        raise ValueError("entry callable must be importable; use 'module:function'")
    return f"{module}:{name}"


def _resolve_entry(path: str) -> Callable[..., Any]:
    module_name, qualname = path.split(":", 1)
    target: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"entry {path!r} is not callable")
    return target


def _gpu_inventory() -> dict[int, tuple[str, str]]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("nvidia-smi GPU inventory failed; real SC workers require visible GPUs") from exc
    inventory: dict[int, tuple[str, str]] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            raise RuntimeError(f"unexpected nvidia-smi inventory row: {line!r}")
        inventory[int(fields[0])] = (fields[1], fields[2])
    if not inventory:
        raise RuntimeError("nvidia-smi found no GPUs")
    return inventory


def _audit(rank: int, world_size: int, physical_device: int, gpu_uuid: str | None,
           cpu_only: bool, allow_colocation: bool) -> RankAudit:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(physical_device):
        raise RuntimeError(
            f"rank {rank} expected CUDA_VISIBLE_DEVICES={physical_device}, got {visible!r}"
        )
    if cpu_only:
        count, name, device = 0, None, "cpu"
    else:
        import easyfhe as torch  # imported only after the device mask is set

        count = int(torch.cuda.device_count())
        if count != 1:
            raise RuntimeError(f"rank {rank} sees {count} CUDA devices after pinning; expected 1")
        name = str(torch.cuda.get_device_name(0))
        device = "cuda"
        actual = _gpu_inventory().get(physical_device)
        if actual is None or actual[0] != gpu_uuid:
            raise RuntimeError(
                f"rank {rank} GPU UUID changed: expected {gpu_uuid}, got {actual}"
            )
    return RankAudit(
        rank=rank, world_size=world_size, physical_device=physical_device,
        visible_devices=visible, device=device, cuda_device_count=count,
        cuda_device_name=name, gpu_uuid=gpu_uuid, hostname=socket.gethostname(),
        pid=os.getpid(), cpu_protocol_only=cpu_only,
        emulated_colocation=allow_colocation,
    )


def _child_main(
    rank: int, world_size: int, physical_device: int, gpu_uuid: str | None,
    odin_root: str, entry_path: str, kwargs: dict[str, Any], endpoints: Any,
    result_queue: Any, worker_env: dict[str, str], cpu_only: bool, transport_kind: str,
    allow_colocation: bool,
) -> None:
    os.environ.update(worker_env)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    os.environ["SC_RANK"] = str(rank)
    os.environ["SC_WORLD_SIZE"] = str(world_size)
    os.environ["SC_ODIN_ROOT"] = odin_root
    transport = None
    audit = None
    try:
        odin_transport, _ = _odin_modules(Path(odin_root))
        audit = _audit(rank, world_size, physical_device, gpu_uuid, cpu_only, allow_colocation)
        transport = odin_transport.build_transport(
            rank, endpoints, kind=transport_kind,
            default_timeout=900.0, shared_memory=not cpu_only,
        )
        runtime = RankRuntime(rank, world_size, physical_device, odin_root, transport, audit)
        result = _resolve_entry(entry_path)(runtime, **kwargs)
        transport.barrier("sc_worker_done", timeout=900.0)
        transport.close()
        result_queue.put(WorkerOutcome(rank=rank, ok=True, audit=audit, result=result))
    except BaseException as exc:
        result_queue.put(WorkerOutcome(
            rank=rank, ok=False, audit=audit,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        ))
        raise
    finally:
        if transport is not None:
            transport.close()


def run_world(
    entry: str | Callable[..., Any],
    world_size: int,
    *,
    odin_root: str | os.PathLike[str] | None = None,
    devices: Sequence[int] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    timeout: float = 3600.0,
    worker_env: Mapping[str, str] | None = None,
    cpu_only: bool = False,
    transport_kind: str = "host",
    allow_colocation: bool = False,
) -> list[WorkerOutcome]:
    """Run one SC rank per GPU and return rank-ordered results plus device audit.

    ``entry`` must resolve to an importable module-level callable.  Real mode
    requires distinct physical devices discovered by ``nvidia-smi``.  The
    ``cpu_only`` switch is for transport tests and must never be used to report
    SC inference timing or accuracy.
    """

    world_size = int(world_size)
    if world_size not in (1, 2, 4, 8):
        raise ValueError("world_size must be 1, 2, 4, or 8")
    if transport_kind not in ("host", "pinned"):
        raise ValueError("transport_kind must be 'host' or 'pinned'")
    mapping = tuple(range(world_size)) if devices is None else tuple(int(d) for d in devices)
    if len(mapping) != world_size or any(d < 0 for d in mapping):
        raise ValueError("devices must give one nonnegative physical GPU index per rank")
    if not allow_colocation and len(set(mapping)) != world_size:
        raise ValueError("devices must give one distinct nonnegative physical GPU index per rank")
    root = _discover_odin_root(odin_root)
    odin_transport, _ = _odin_modules(root)
    path = _entry_path(entry)
    inventory = {} if cpu_only else _gpu_inventory()
    missing = [device for device in mapping if not cpu_only and device not in inventory]
    if missing:
        raise ValueError(f"requested physical GPUs {missing} absent from nvidia-smi inventory")
    context = multiprocessing.get_context("spawn")
    endpoints = odin_transport.build_transport_endpoints(world_size, context=context)
    result_queue = context.Queue()
    env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    env.update({str(k): str(v) for k, v in dict(worker_env or {}).items()})
    processes = []
    outcomes: dict[int, WorkerOutcome] = {}
    deadline = time.monotonic() + float(timeout)
    try:
        for rank, physical_device in enumerate(mapping):
            process = context.Process(
                target=_child_main,
                args=(
                    rank, world_size, physical_device,
                    None if cpu_only else inventory[physical_device][0],
                    str(root), path, dict(kwargs or {}), endpoints, result_queue, env, cpu_only,
                    transport_kind, allow_colocation,
                ),
                name=f"sc-rank{rank}-gpu{physical_device}",
                daemon=False,
            )
            # ``spawn`` re-imports the parent's __main__ module before it calls
            # _child_main.  Set the mask in the environment inherited at exec,
            # too, so a top-level EasyFHE import cannot touch the wrong GPU.
            startup_env = {
                **env,
                "CUDA_VISIBLE_DEVICES": str(physical_device),
                "SC_RANK": str(rank),
                "SC_WORLD_SIZE": str(world_size),
                "SC_ODIN_ROOT": str(root),
            }
            previous = {key: os.environ.get(key) for key in startup_env}
            try:
                os.environ.update(startup_env)
                process.start()
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            processes.append(process)
        while len(outcomes) < world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"SC workers timed out; reported ranks {sorted(outcomes)}")
            try:
                outcome = result_queue.get(timeout=min(1.0, remaining))
            except queue.Empty:
                for rank, process in enumerate(processes):
                    if process.exitcode not in (None, 0) and rank not in outcomes:
                        raise RuntimeError(
                            f"SC rank {rank} exited with code {process.exitcode} without a report"
                        )
                continue
            if int(outcome.rank) in outcomes:
                raise RuntimeError(f"duplicate SC worker report from rank {outcome.rank}")
            outcomes[int(outcome.rank)] = outcome
            if not outcome.ok:
                raise RuntimeError(f"SC rank {outcome.rank} failed:\n{outcome.error}")
        for rank, process in enumerate(processes):
            process.join(timeout=10.0)
            if process.exitcode != 0:
                raise RuntimeError(f"SC rank {rank} exited with code {process.exitcode}")
        return [outcomes[rank] for rank in range(world_size)]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10.0)
        for inbox in endpoints.inboxes:
            inbox.close()
            inbox.join_thread()
        result_queue.close()
        result_queue.join_thread()


__all__ = ["RankAudit", "RankRuntime", "WorkerOutcome", "run_world"]
