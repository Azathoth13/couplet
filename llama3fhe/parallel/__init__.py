"""One-process-per-GPU (T2) multi-GPU runtime for the feature-major Llama layer.

Design (docs/DESIGN.md):

* every rank is one OS process bound to one physical GPU
  (``CUDA_VISIBLE_DEVICES`` is set by :mod:`.launch`, so inside a worker the
  device is always the exact string ``"cuda"``);
* rank ``r`` permanently owns the adjacent feature-major ciphertext pair
  ``(2r, 2r+1)`` (= tokens ``16r .. 16r+15``) and, inside attention, the two
  global query pairs ``(2r, 2r+1)`` plus W_O carrier ``r`` (:mod:`.ownership`);
* every rank keeps its own CKKS context (deterministic keygen: all ranks
  share one secret key without exchanging material), the L3 bootstrap
  program (dnum5, ``ConstantBundle`` cache mode ``middle``, level budget
  (5,5), 17 post-bootstrap levels on the 4090 preset) and its rotation keys
  in *tiers*: BS + RMSNorm resident for the whole layer, the shared linear
  set and the attention sub-stage sets leased per station under the preset's
  ``key_policy`` (:mod:`.keys`, :mod:`.worker`, :mod:`.presets`); no
  bootstrap ever crosses a card;
* the only cross-card traffic in a layer is the attention-input all-gather
  and the PairFM all-to-all, both pure data movement behind the
  :class:`~.transport.Transport` protocol; both real backends are host
  staged over :mod:`multiprocessing` queues (``host``: D2H -> NumPy -> shared
  memory -> H2D; ``pinned``: D2H straight into a pinned shared-memory arena
  -> H2D) because the EasyFHE torch shim ships neither NCCL nor CUDA IPC
  (:mod:`.wire`, :mod:`.transport`, ``T2_TRANSPORT``);
* device-local layer steps are :mod:`.stations`; RMSNorm has *no* slim
  refresh (two complex-pair bootstraps per rank instead).

The station code assumes one T2 slot (two ciphertexts) per rank, so the
16-ciphertext production layout means ``world_size == 8``; a host with fewer
cards runs several ranks per card (:func:`.launch.spawn_workers`
``ranks_per_device``), see docs/REPRODUCING.md.

``presets``, ``ownership``, ``wire``, ``transport`` and ``keys`` are importable
without CUDA and have CPU unit tests; ``worker``/``stations``/``launch``
import EasyFHE lazily inside functions; ``emulation`` runs the eight ranks as
threads of one process on one card (``bench/t2_layer.py --emulate-single-device``).
"""

from .presets import (
    A100_HIGH,
    A800_4CARD,
    H100_EMU,
    HIGH_PRESETS,
    PRESETS,
    RTX4090_LOW,
    RTX6000PRO_HIGH,
    BootstrapPreset,
    ParallelPreset,
    preset_by_name,
)
from .prefill.keys import (
    KEY_POLICIES,
    T0_TIERS,
    T1_TIERS,
    T2_TIERS,
    TIER_ORDER,
    KeyLifecycle,
    KeyPlan,
    KeyTier,
    KeyTracer,
    build_key_plan,
    normalize_rotation,
    rotation_key_bytes,
)
from .prefill.ownership import (
    RankOwnership,
    WorldOwnership,
    attention_query_pairs_for_rank,
    build_world_ownership,
    cipher_indices_for_rank,
    owner_of_cipher,
    owner_of_query_pair,
    ownership_from_layouts,
)
from .wire import (
    CipherMetadata,
    CipherPacket,
    cipher_metadata,
    pack_cipher,
    rebuild_cipher,
    unpack_cipher,
    validate_cipher,
)
from .prefill.transport import (
    DEFAULT_TRANSPORT,
    TRANSPORT_BACKENDS,
    HostStagingTransport,
    SharedArrayHandle,
    Message,
    Transport,
    TransportEndpoints,
    TransportStats,
    build_transport,
    build_transport_endpoints,
)

__all__ = [
    "A100_HIGH",
    "A800_4CARD",
    "H100_EMU",
    "HIGH_PRESETS",
    "PRESETS",
    "RTX4090_LOW",
    "RTX6000PRO_HIGH",
    "BootstrapPreset",
    "ParallelPreset",
    "preset_by_name",
    "KEY_POLICIES",
    "T0_TIERS",
    "T1_TIERS",
    "T2_TIERS",
    "TIER_ORDER",
    "KeyLifecycle",
    "KeyPlan",
    "KeyTier",
    "KeyTracer",
    "build_key_plan",
    "normalize_rotation",
    "rotation_key_bytes",
    "RankOwnership",
    "WorldOwnership",
    "attention_query_pairs_for_rank",
    "build_world_ownership",
    "cipher_indices_for_rank",
    "owner_of_cipher",
    "owner_of_query_pair",
    "ownership_from_layouts",
    "CipherMetadata",
    "CipherPacket",
    "cipher_metadata",
    "pack_cipher",
    "rebuild_cipher",
    "unpack_cipher",
    "validate_cipher",
    "DEFAULT_TRANSPORT",
    "TRANSPORT_BACKENDS",
    "HostStagingTransport",
    "SharedArrayHandle",
    "Message",
    "Transport",
    "TransportEndpoints",
    "TransportStats",
    "build_transport",
    "build_transport_endpoints",
]
