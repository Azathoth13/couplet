"""Hardware presets for the T2 runtime.

All presets share one code path and one bootstrap program -- the agreed L3
foundation (dnum 5, level budget (5,5), ``middle`` constant cache, 17
post-bootstrap levels at depth 37, measured in
profile measurements: BS + RMSNorm keys resident = 11.16 GiB
on one 4090, warm BS 0.124 s).  Only the memory knobs differ:

* ``RTX4090_LOW`` (24 GB): leased key tiers with async prefetch, 256-row
  linear chunks, MLP feature shard k = 2, plain (non-ext) PairFM hoist.
* ``A100_HIGH`` / ``RTX6000PRO_HIGH`` / ``H100_EMU`` / ``A800_4CARD`` (80 / 96 / 80 / 80 GB):
  the efficiency-first profile built by :func:`_high_preset` -- every key
  tier resident (~40 GiB at the measured caps), the ``plain`` BS constant
  cache, k = 4, ext hoist, the split-phase attention-input exchange.
  ``H100_EMU`` is the same profile used by the single-card 8-rank emulation
  (:mod:`.emulation`, ``bench/t2_layer.py --emulate-single-device``).  The
  high tier also carries the numeric knobs validated on the 4 x Pro 6000
  32-layer run (): one PairFM refresh pass, the slim RMSNorm exit
  on layers 3-31, the 5e-3 rsqrt table and the joint_tol_v1 exp series
  (``rmsnorm_*`` / ``softmax_*`` fields; the ``T2_*`` environment variables
  the benches read override them for A/B).  ``RTX6000PRO_HIGH`` alone runs
  2048-row linear chunks: the 8 x Pro 6000 32-layer A/B of
  (profile measurements) measured plain cache +
  overlap + 2048 rows at 74.2 s against 79.8 s for the previous defaults
  (middle / blocking exchange / 512 rows), peak 74.5 GiB per card at one
  slot per card.  A card that hosts more than one slot
  (:meth:`ParallelPreset.with_world_size` 4, :meth:`ParallelPreset.for_colocation`
  2) falls back to 512 rows, the only multi-slot value measured
  (``--max-plaintext-rows`` on the benches forces either).

The 16-ciphertext production layout is split into ``world_size`` ranks of
``ciphers_per_rank = 16 / world_size`` adjacent feature-major ciphertexts
(:mod:`.ownership`: one T2 *slot* = two ciphertexts = one PairFM source shard
= one W_O carrier).  The presets default to ``world_size = 8`` (one slot per
rank, one process per card on an 8-card host); :meth:`ParallelPreset.with_world_size`
derives the 4-card variant (``world_size = 4``, two slots = four ciphertexts
per rank, still one process per card: ``bench/t2_layer.py --world-size 4``).
The older way of using 4 cards -- eight one-slot ranks, two processes per card
(``--ranks-per-device 2``, :meth:`ParallelPreset.for_colocation`) -- keeps
working; see docs/REPRODUCING.md for the trade-off.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BootstrapPreset:
    """Bootstrap program shape and constant-cache policy for one card."""

    level_budget: tuple[int, ...] = (5, 5)
    cache_mode: str = "middle"
    log_slots: int = 15
    iterations: int = 1
    precision_bits: int = 14
    mode: str = "modraise_first"
    strategy: str = "double_hoist"

    def __post_init__(self) -> None:
        if len(self.level_budget) != 2 or any(int(v) <= 0 for v in self.level_budget):
            raise ValueError("level_budget must be two positive integers")
        if self.cache_mode not in {"none", "middle", "plain"}:
            raise ValueError(f"unsupported cache_mode {self.cache_mode!r}")
        object.__setattr__(
            self, "level_budget", tuple(int(v) for v in self.level_budget)
        )

    @property
    def program_depth(self) -> int:
        """Levels consumed by the Meta-BTS program: 10 + b1 + b2."""

        return 10 + int(self.level_budget[0]) + int(self.level_budget[1])

    def post_levels(self, depth: int) -> int:
        """Levels left after a bootstrap in a ``depth``-level context."""

        post = int(depth) - self.program_depth
        if post <= 0:
            raise ValueError(
                f"depth {depth} is too shallow for level budget {self.level_budget}"
            )
        return post


@dataclass(frozen=True)
class ParallelPreset:
    """Everything a worker needs to size itself, in one place."""

    name: str
    world_size: int = 8
    ciphers_per_rank: int = 2
    #: Model shape (:mod:`llama3fhe.model_spec`): ``"llama3-8b"`` (16 FM
    #: ciphertexts, head_dim 128) or ``"llama3.2-1b"`` (8 FM ciphertexts,
    #: head_dim 64).  Use :meth:`for_model` to re-cut a preset for another model.
    model: str = "llama3-8b"
    depth: int = 37
    dnum: int = 5
    log_n: int = 16
    bootstrap: BootstrapPreset = field(default_factory=BootstrapPreset)
    #: GiB of device memory the launcher refuses to exceed per rank (soft).
    device_budget_gib: float = 23.0
    #: Whether the rotation-key CUDA cache may be filled lazily (EasyFHE
    #: ``auto_load_keys``).  Always ``True`` in T2: the key lifecycle
    #: (:mod:`.keys`) decides what stays resident, EasyFHE just caches.
    auto_load_keys: bool = True
    #: Rotation-key residency (:mod:`.keys`): ``"lease"`` keeps only the T0
    #: tiers (bootstrap + RMSNorm) resident and brackets every other station
    #: with prefetch/evict; ``"resident"`` never evicts (A100, 80 GiB).
    key_policy: str = "lease"
    #: Non-T0 tiers promoted to layer residency under ``"lease"`` (the
    #: ``linear`` tier is the candidate; decided by the lifecycle probe).
    key_resident_extra: tuple[str, ...] = ()
    #: Q-limb caps per key tier as ``((tier, limbs), ...)`` (host key trimming
    #: + prefetch limbs).  Empty means full-chain keys everywhere; filled in
    #: from the lifecycle probe's ``max_limbs`` per stage.
    key_limb_caps: tuple[tuple[str, int], ...] = ()
    #: Keep the leased (non-resident, capped) tiers pre-compacted in pinned
    #: host memory (:class:`~.keys.PinnedCompactKeys`) so a per-layer reload
    #: is a plain pinned H2D copy.  Costs ~40% of those tiers' host key bytes
    #: extra per rank; irrelevant under ``"resident"`` (nothing is reloaded).
    pin_leased_keys: bool = True
    #: ``max_plaintext_rows`` for the three linear boundaries (QKV, W_O,
    #: gate/up + down).  The production optimum is 512 (``baby_steps=64`` x 8
    #: giant steps per plaintext chunk); 256 halves the resident diagonal
    #: chunk and the hoisted baby basis that goes with it (~-1.7 GiB at the
    #: QKV / MLP peaks on the 4090) at the cost of twice as many chunk
    #: alignments.  Both the resident ``linear`` key tier and every station
    #: must read it through :meth:`layer_schedule` so the giant-alignment
    #: rotation keys match the chunk shape actually executed.  2048 (32 giant
    #: steps per chunk) quarters the chunk alignments and the per-chunk
    #: baby-basis reuse overhead for +22.5 GiB of resident encoded weights per
    #: rank (8 x Pro 6000, : 52.0 -> 74.5 GiB peak, 32 layers
    #: 75.5 -> 74.6 s); only a 96 GB card with one slot per rank takes it, see
    #: :data:`SHARED_CARD_MAX_PLAINTEXT_ROWS`.
    linear_max_plaintext_rows: int = 512
    #: How the leased key tiers reach the device (:class:`~.keys.KeyLifecycle`):
    #: ``"lazy"`` loads each key on its first request inside the operators;
    #: ``"sync"`` uploads a whole tier at lease entry; ``"async"`` uploads the
    #: *next* tier of the station chain on a side CUDA stream while the current
    #: station computes (needs one lazy learning pass per tier).  Irrelevant
    #: under ``key_policy="resident"``.
    key_prefetch_mode: str = "lazy"
    #: Free device memory (GiB) an async prefetch must leave untouched at issue
    #: time; below that the tier falls back to lazy loading for this lease.
    key_prefetch_margin_gib: float = 1.5
    #: What a lease's exit evict waits for: ``"device"`` = EasyFHE's
    #: ``clear_cuda_rotation_cache`` (full ``torch.cuda.synchronize`` +
    #: ``empty_cache``, joins any prefetch copy in flight); ``"stream"`` =
    #: compute-stream sync only, blocks stay in the caching allocator.
    key_evict_sync: str = "device"
    #: MLP intermediate-page split (:mod:`llama3fhe.operators.linear.shard.mlp_feature_shard`,
    #: Megatron-style tensor parallelism of the SwiGLU hidden dimension): the
    #: ``k`` adjacent ranks of a shard group each own ``pages/k`` of the 4
    #: intermediate pages and run gate/up + SiLU + the matching down column
    #: block for the tokens of the whole group, so the plaintext encode (the one
    #: linear cost that does not shrink with the world size) drops by
    #: ``1 - 1/k`` at the price of ``k`` x the hoisted baby basis (~0.7 GiB per
    #: ciphertext at 11 limbs) and two small all-to-alls.  ``1`` = plain token
    #: sharding (:func:`~llama3fhe.parallel.prefill.stations.mlp_station` runs the
    #: unchanged kernel).  Must divide ``world_size``.
    mlp_feature_shard_k: int = 1
    #: BSGS hoisting of the PairFM mask transforms
    #: (:func:`llama3fhe.operators.transform.apply_transform_by_level`, env
    #: ``LLAMA_FHE_TRANSFORM_HOIST``): ``"ext_normal"`` keeps the mask
    #: plaintexts in the extended basis (``cur_limbs + K`` limbs, +73% bytes at
    #: L3 ``K=8``) and saves one moddown per baby step; ``"normal"`` encodes
    #: plain plaintexts and lets the hoisted MAC do the moddown.  The launcher
    #: exports it to every worker; a value already in the environment wins.
    transform_hoist_strategy: str = "ext_normal"
    #: Default :func:`~.transport.build_transport` backend for the workers
    #: (``T2_TRANSPORT``; ``--transport`` and a value already in the
    #: environment win).  ``"pinned"`` moves the ciphers through a pinned
    #: shared-memory arena (D2H/H2D ~10x faster than the ``"host"`` NumPy
    #: path); measured -0.37 s/layer on the 8x4090 host (5.96 -> 5.59 s).
    transport: str = "host"
    #: Meta-BTS passes of the PairFM paired-input refresh
    #: (:attr:`~llama3fhe.operators.attention.config.AttentionOperatorConfig.pairfm_refresh_bootstrap_iterations`).
    #: The single-card graph only fires that refresh at layer 31; under the L3
    #: distributed graph every layer pays it (one BS per owned slot).  2 = the
    #: inherited default; 1 measured on the 4 x Pro 6000 32-layer run
    #: (records/t2_model_pro6000_20260930): -0.13 s/layer with two slots per
    #: rank, argmax / top-5 unchanged, logits rel_max_err 1.32% -> 0.78%,
    #: pairfm station rel_err 4.0e-3 -> 5.1e-3.  ``T2_PAIRFM_BS_ITERATIONS``
    #: in the environment overrides (the benches read it).
    pairfm_refresh_bootstrap_iterations: int = 2
    #: How the RMSNorm station refreshes its output
    #: (:func:`~llama3fhe.parallel.prefill.stations.rmsnorm_station`).  ``"pair"`` =
    #: the production flow: an exit pair-BS on the normalised stream (one call
    #: per owned FM pair, so two per rank in world 4) plus the adaptive slim
    #: refresh of the mean-square row only when the rsqrt polynomial leaves
    #: < 4 limbs.  ``"slim"`` = no exit pair-BS; the slim mean-square row is
    #: always bootstrapped right after the rsqrt polynomial (one sparse BS per
    #: rank whatever the slot count) and the normalised stream inherits the
    #: fresh level through the extract / scale multiply (18 - 2 = 16 limbs >=
    #: the QKV 13 / MLP <= 15 entry).  Measured on the 4 x Pro 6000 32-layer run
    #: (records/t2_model_pro6000_20260930, ``T2_RMS_EXIT=slim``): rmsnorm_in +
    #: rmsnorm_post 28.1 -> 20.2 s over 32 layers, argmax / top-5 unchanged.
    #: ``T2_RMS_EXIT`` in the environment overrides (the benches read it).
    rmsnorm_exit_mode: str = "pair"
    #: Layers that take the ``"slim"`` exit when :attr:`rmsnorm_exit_mode` is
    #: ``"slim"``; the others keep ``"pair"`` (``"all"`` / ``"3-31"`` /
    #: ``"0,1,2"`` / ``"0-2,14,31"``).  Layers 0-2 must stay on ``pair``: their
    #: rsqrt row is 80-190x the normalised stream and the 14-bit slim BS loses
    #: it (prescale and a second Meta-BTS pass do not recover it).
    #: ``T2_RMS_SLIM_LAYERS`` in the environment overrides.
    rmsnorm_slim_exit_layers: str = "all"
    #: Periodic slim row pack of the RMSNorm mean squares (power of two ``P``
    #: rows, ``None`` = legacy row-``j`` pack).  Cipher ``j`` lands on every
    #: feature row ``d == j (mod P)``, so the slim message is periodic every
    #: ``P * tokens_per_cipher`` slots: the rank's bootstrap operator gets a
    #: second, sparse program of that slot count
    #: (:func:`~.worker.slim_sparse_log_slots`; +21 rotation keys at
    #: ``P=16``) and the slim refresh runs through it as a metadata-only view
    #: (H100 probe: 72 -> 54 ms per call), while the extract broadcast needs
    #: ``log2(P)`` instead of 12 rotations per owned ciphertext.  ``P`` must
    #: keep the rank's owned cipher indices distinct mod ``P`` (16 = the
    #: cipher count works for every world size).  ``--rms-slim-period`` on the
    #: benches overrides (0 = off).
    rmsnorm_slim_period_rows: int | None = None
    #: Periodic slim pack of the Prefill Softmax Alg2 row sums
    #: (``AttentionOperatorConfig.softmax_slim_periodic``): the rank's slim
    #: ciphertext repeats every ``pairs_per_rank * 2 * seq_len`` slots and its
    #: slim-input / lambda refreshes (1-2 per Alg2 round) take a sparse program
    #: of ``log2`` that period (:func:`~.worker.softmax_slim_sparse_log_slots`:
    #: 9 at W=8, 10 at W=4, 12 single-card).  ``--softmax-slim-periodic`` /
    #: ``--no-softmax-slim-periodic`` on the benches override.
    softmax_slim_periodic: bool = False
    #: Extra sparse-slot bootstrap programs (``log2`` slot counts) every rank
    #: prepares next to the dense one, on top of the slim RMSNorm program
    #: :func:`~.worker.sparse_bootstrap_log_slots` derives.  The Decode
    #: merged-page Softmax runs its ``filled`` / ``lambda`` refreshes (the
    #: cross-page ``sum(y^2)`` row, periodic every ``decode_page_merge_stride``
    #: slots) through the program of ``log2(decode_page_merge_stride) = 12``
    #: whenever it is present (``--softmax-sparse-bs`` on the Decode bench;
    #: ``LLAMA_FHE_SOFTMAX_SPARSE_BS=0`` forces dense).  Each extra program
    #: costs its CtoS/StoC constants and a few rotation keys.
    bootstrap_extra_sparse_log_slots: tuple[int, ...] = ()
    #: RMSNorm rsqrt Chebyshev table the benches load instead of the production
    #: ``assets/polynomials/8b/reference/rmsnorm_poly_coeffs.json`` (``None`` = production).
    #: A bare file name resolves under ``assets/polynomials/``.  The high tier
    #: runs ``8b/runtime/rmsnorm_poly_coeffs.json``
    #: (``tools/autodl/refit_rmsnorm_tol.py --tol 5e-3 --no-increase``: every
    #: layer's degree is lowered until the rel. tolerance would exceed 5e-3,
    #: never raised): hidden_32 vs exact fp32 0.0733 -> 0.0702 at -3 s / 32
    #: layers of rsqrt evaluation.  ``T2_RMS_COEFFS`` in the environment
    #: overrides (``T2_RMS_COEFFS=default`` forces the production table).
    rmsnorm_coeffs: str | None = None
    #: Softmax Alg2 table the benches patch over the production
    #: ``assets/polynomials/8b/reference/softmax_poly_coeffs.json`` (``None`` = production;
    #: a bare file name resolves under ``assets/polynomials/``) and which parts
    #: of it are taken: ``exp`` = the per-layer exp Chebyshev series (only the
    #: polynomial fields -- ``U_hat`` / ``stats_min`` / ``k`` stay production),
    #: ``j1`` / ``j2`` = the per-layer ``invsqrt_alg2`` rsqrt series.  The high
    #: tier takes only ``exp`` from ``8b/runtime/softmax_poly_coeffs.json``
    #: (exp degree 17 -> 5-12, depth 5 -> 4): the extra limb removes the hidden
    #: slim-input refresh of rounds 1/2 (attn_core BS 184 -> 155 calls) and
    #: leaves the attention output at 6 limbs, which also removes the PairFM
    #: paired-input BS of 26 layers; -0.36 s/layer at L10, top-5 5/5.  Taking
    #: ``j1`` / ``j2`` from the same table is not faster and loses top-5 (4/5).
    #: ``T2_SOFTMAX_COEFFS`` / ``T2_SOFTMAX_PARTS`` in the environment override.
    softmax_coeffs: str | None = None
    #: SiLU Chebyshev table ``bench/t2_model.py`` loads instead of the production
    #: ``assets/polynomials/8b/reference/silu_poly_coeffs.json`` (``None`` = production; a
    #: bare file name resolves under ``assets/polynomials/``).  The degree sets
    #: the SiLU PS depth and with it the MLP entry limbs
    #: (:func:`~.prefill.stations.mlp_input_limbs`).  The high tier runs
    #: ``8b/runtime/silu_poly_coeffs.json`` (degree 48/96/128/384 -> 12-30,
    #: L29-31 60/74/236; MLP entry 11 -> 9/10 limbs): single Pro 6000 32-layer
    #: run  (records/odin_vs_slot_coeff_pro6000_20261013_e8_e2e)
    #: 179.8 -> 169.5 s, steady 5.39 -> 5.05 s/layer, L31 8.5 -> 7.3 s,
    #: argmax / top-5 5/5 unchanged, logits rel_max_err 1.94% -> 1.80%.
    #: ``T2_SILU_COEFFS`` in the environment overrides (``default`` = production).
    silu_coeffs: str | None = None
    softmax_coeffs_parts: str = "exp,j1,j2"
    #: Split-phase order of the attention-input exchange
    #: (:func:`~.stations.attention_input_exchange` ``overlap``): the lane
    #: all-gather is issued before the V partials and the partial all-to-all
    #: before Q/K, so the transfer and the rank skew hide behind the rank's
    #: own GPU work.  Same ciphertexts in the same order on the wire; the
    #: result is bit-identical.  8 x Pro 6000 32-layer A/B ():
    #: attn_input exchange 19 -> 4 ms per layer, wall 79.8 -> 78.9 s.  Off on
    #: the 4090 only because it has not been measured there.
    #: ``T2_ATTN_IN_OVERLAP`` in the environment overrides (the benches read
    #: it; ``--attn-in-overlap`` / ``--no-attn-in-overlap`` set it).
    attention_input_overlap: bool = False

    def __post_init__(self) -> None:
        if int(self.world_size) <= 0:
            raise ValueError("world_size must be positive")
        if int(self.pairfm_refresh_bootstrap_iterations) not in (1, 2):
            raise ValueError("pairfm_refresh_bootstrap_iterations must be 1 or 2")
        if str(self.rmsnorm_exit_mode) not in {"pair", "slim"}:
            raise ValueError(f"unsupported rmsnorm_exit_mode {self.rmsnorm_exit_mode!r}; choose pair or slim")
        if not str(self.rmsnorm_slim_exit_layers).strip():
            raise ValueError("rmsnorm_slim_exit_layers must be 'all' or a layer spec such as '3-31'")
        if self.rmsnorm_slim_period_rows is not None:
            p = int(self.rmsnorm_slim_period_rows)
            if p <= 0 or p & (p - 1):
                raise ValueError(f"rmsnorm_slim_period_rows must be a power of two, got {p}")
        parts = {p.strip().lower() for p in str(self.softmax_coeffs_parts).split(",") if p.strip()}
        if not parts or parts - {"exp", "j1", "j2", "k", "jmid"}:
            raise ValueError(
                f"softmax_coeffs_parts {self.softmax_coeffs_parts!r} must be a comma list drawn from exp, j1, j2, k, jmid"
            )
        object.__setattr__(
            self, "softmax_coeffs_parts", ",".join(p for p in ("exp", "j1", "j2", "k", "jmid") if p in parts)
        )
        for attr in ("rmsnorm_coeffs", "softmax_coeffs"):
            value = getattr(self, attr)
            if value is not None and not str(value).strip():
                raise ValueError(f"{attr} must be None or a non-empty path")
        from .prefill.transport import TRANSPORT_BACKENDS

        if str(self.transport) not in TRANSPORT_BACKENDS:
            raise ValueError(f"unsupported transport {self.transport!r}; choose from {TRANSPORT_BACKENDS}")
        if int(self.mlp_feature_shard_k) <= 0 or int(self.world_size) % int(self.mlp_feature_shard_k):
            raise ValueError(
                f"mlp_feature_shard_k {self.mlp_feature_shard_k} must be a positive divisor of "
                f"world_size {self.world_size}"
            )
        singleton_1b = self.model == "llama3.2-1b" and int(self.world_size) == 8 and int(self.ciphers_per_rank) == 1
        if int(self.ciphers_per_rank) <= 0 or (int(self.ciphers_per_rank) % 2 and not singleton_1b):
            raise ValueError("ciphers_per_rank must be a positive even number, except the 1B W8 singleton layout")
        if str(self.key_policy) not in {"lease", "resident"}:
            raise ValueError(f"unsupported key_policy {self.key_policy!r}")
        if str(self.transform_hoist_strategy) not in {"ext_normal", "normal"}:
            raise ValueError(
                f"unsupported transform_hoist_strategy {self.transform_hoist_strategy!r}"
            )
        object.__setattr__(self, "key_resident_extra", tuple(str(v) for v in self.key_resident_extra))
        object.__setattr__(
            self, "key_limb_caps", tuple((str(k), int(v)) for k, v in dict(self.key_limb_caps).items())
        )
        if int(self.linear_max_plaintext_rows) <= 0:
            raise ValueError("linear_max_plaintext_rows must be positive")
        if str(self.key_prefetch_mode) not in {"lazy", "sync", "async"}:
            raise ValueError(f"unsupported key_prefetch_mode {self.key_prefetch_mode!r}")
        if float(self.key_prefetch_margin_gib) < 0:
            raise ValueError("key_prefetch_margin_gib must be non-negative")
        if str(self.key_evict_sync) not in {"device", "stream"}:
            raise ValueError(f"unsupported key_evict_sync {self.key_evict_sync!r}")

    def layer_schedule(self, base=None):
        """The :class:`~llama3fhe.schedule.LayerSchedule` this preset executes.

        ``base`` defaults to the production ``LayerSchedule()`` with
        ``steady_state_attention_refresh=True``: the T2 stations always run
        the steady-layer attention chain (pre-exp Softmax refresh), the way
        ``bench/single_gpu/model.py`` resolves it for every layer of a multi-layer run.
        The bare ``LayerSchedule()`` default (``False``) is the fused
        single-card *Layer0* graph, whose Softmax enters exp straight from
        the QK adapter level; under the T2 station chain that path runs out
        of limbs inside the layer-0 exp polynomial (k=4, ps_k=6), so it is
        not what a ``--coeff-layer 0`` T2 run wants.          The preset then
        overrides the linear ``max_plaintext_rows`` on all three boundaries.
        ``T2_LINEAR_BABY_STEPS`` optionally replaces ``linear_baby_steps`` and
        the anchor (24 GB hosts that hold every FM ciphertext of a small
        model).  The key plan (:func:`~.keys.build_key_plan`) and the stations must be
        built from the same object, otherwise the chunk giant-base rotations
        of one do not exist in the other.
        """

        from dataclasses import replace

        from llama3fhe.schedule import LayerSchedule

        schedule = LayerSchedule(steady_state_attention_refresh=True) if base is None else base
        rows = int(self.linear_max_plaintext_rows)
        # 1B is single-card (8 FM ciphertexts on one rank).  baby_steps=64 caches
        # a hoisted basis per ciphertext and does not fit a 24 GB 4090; set
        # T2_LINEAR_BABY_STEPS (and the matching anchor) to a divisor of the
        # linear diagonals.  Unset keeps the production 64.
        baby_raw = os.environ.get("T2_LINEAR_BABY_STEPS", "").strip()
        baby = int(baby_raw) if baby_raw else None
        if baby is not None and baby < 2:
            raise ValueError("T2_LINEAR_BABY_STEPS must be at least 2")
        same_rows = (
            int(schedule.qkv_max_plaintext_rows) == rows
            and int(schedule.output_max_plaintext_rows) == rows
            and int(schedule.mlp_max_plaintext_rows) == rows
        )
        same_baby = baby is None or (
            int(schedule.linear_baby_steps) == baby and int(schedule.linear_anchor_step) == baby
        )
        if same_rows and same_baby:
            return schedule
        updates: dict = {}
        if not same_rows:
            updates["qkv_max_plaintext_rows"] = rows
            updates["output_max_plaintext_rows"] = rows
            updates["mlp_max_plaintext_rows"] = rows
        if baby is not None:
            updates["linear_baby_steps"] = baby
            updates["linear_anchor_step"] = baby
        return replace(schedule, **updates)

    @property
    def slots(self) -> int:
        return 1 << (int(self.log_n) - 1)

    @property
    def limb_caps(self) -> dict[str, int]:
        caps = {name: int(limbs) for name, limbs in self.key_limb_caps}
        # T2_KEY_LIMB_CAPS="linear=16,attn_input_v=15" overrides single tiers (A/B of chains that carry
        # more limbs through a station than the measured caps, e.g. the Softmax post-exp checkpoint).
        raw = os.environ.get("T2_KEY_LIMB_CAPS", "").strip()
        for part in raw.split(","):
            if "=" in part:
                name, limbs = part.split("=", 1)
                caps[name.strip()] = int(limbs)
        return caps

    @property
    def bootstrap_post_levels(self) -> int:
        return self.bootstrap.post_levels(self.depth)

    @property
    def cipher_count(self) -> int:
        return int(self.world_size) * int(self.ciphers_per_rank)

    @property
    def model_spec(self):
        from llama3fhe.model_spec import model_spec_by_name

        return model_spec_by_name(self.model)

    def for_model(self, model: str) -> "ParallelPreset":
        """The same card profile for another model shape.

        ``ciphers_per_rank`` is re-derived from the model's FM ciphertext count
        (``slots / hidden_dim`` tokens per ciphertext over the 128-token
        prompt) and the current ``world_size``; the MLP feature shard is
        clamped to a divisor of the world size exactly like
        :meth:`with_world_size`.
        """

        from dataclasses import replace

        from llama3fhe.model_spec import model_spec_by_name

        spec = model_spec_by_name(model)
        if spec.name == self.model_spec.name:
            return self
        total = spec.fm_cipher_count(slots=int(self.slots))
        world_size = int(self.world_size)
        singleton_1b = spec.name == "llama3.2-1b" and world_size == 8 and total // world_size == 1
        if total % world_size or ((total // world_size) % 2 and not singleton_1b):
            raise ValueError(
                f"{spec.name}: {total} FM ciphertexts do not split evenly over world_size {world_size}"
            )
        return replace(self, model=spec.name, ciphers_per_rank=total // world_size)

    @property
    def slots_per_rank(self) -> int:
        return max(1, int(self.ciphers_per_rank) // 2)

    def with_world_size(self, world_size: int, *, mlp_feature_shard_k: int | None = None) -> "ParallelPreset":
        """The same preset re-cut for ``world_size`` ranks over the same 16 ciphertexts.

        ``ciphers_per_rank`` becomes ``cipher_count / world_size`` (must be a
        positive even number: 8 -> 2, 4 -> 4, 2 -> 8).  ``mlp_feature_shard_k``
        must divide the new world size; when it does not (``k = 4`` on
        ``world_size = 2``) it is clamped to the largest divisor of the new
        world size not above the old value unless an explicit ``k`` is given.
        Everything else (bootstrap program, key policy, hoist strategy,
        device budget) is unchanged -- those are per-card knobs and the card
        is the same.  The linear chunk rows stay too, except that a world
        with more than one slot per rank is capped at
        :data:`SHARED_CARD_MAX_PLAINTEXT_ROWS` (the 2048-row profile is
        measured at one slot per card only).
        """

        from dataclasses import replace

        world_size = int(world_size)
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        total = int(self.cipher_count)
        if total % world_size:
            raise ValueError(f"world_size {world_size} does not divide the {total}-ciphertext layout")
        per_rank = total // world_size
        singleton_1b = self.model == "llama3.2-1b" and world_size == 8 and per_rank == 1
        if per_rank % 2 and not singleton_1b:
            raise ValueError(
                f"world_size {world_size} gives {per_rank} ciphertexts per rank; a rank owns whole slots (even count)"
            )
        if mlp_feature_shard_k is None:
            k = int(self.mlp_feature_shard_k)
            while world_size % k:
                k -= 1
        else:
            k = int(mlp_feature_shard_k)
        rows = int(self.linear_max_plaintext_rows)
        if per_rank > 2:
            # More than one slot per card: the 2048-row chunk of the one-slot
            # 96 GB profile has not been measured with two slots' worth of
            # ciphertexts beside it (est. ~79 GiB, not verified); keep the
            # measured 512.  ``--max-plaintext-rows`` on the benches overrides.
            rows = min(rows, SHARED_CARD_MAX_PLAINTEXT_ROWS)
        if (
            world_size == int(self.world_size)
            and k == int(self.mlp_feature_shard_k)
            and rows == int(self.linear_max_plaintext_rows)
        ):
            return self
        return replace(
            self,
            world_size=world_size,
            ciphers_per_rank=per_rank,
            mlp_feature_shard_k=k,
            linear_max_plaintext_rows=rows,
        )

    def for_colocation(self, ranks_per_device: int) -> "ParallelPreset":
        """The same preset with ``device_budget_gib`` split between the
        ``ranks_per_device`` ranks that share one physical card (a 4-card
        96 GB host runs the 8-rank layout as two processes per card).

        Only the recorded budget changes (plus the linear chunk rows capped at
        :data:`SHARED_CARD_MAX_PLAINTEXT_ROWS`: two 2048-row ranks are 2 x
        74.5 GiB on one 96 GB card); nothing in the runtime enforces the
        budget.  Callers must check that ``ranks_per_device`` x the per-rank
        resident footprint (~40 GiB under ``key_policy="resident"`` at 512
        rows) fits the card.
        """

        from dataclasses import replace

        ranks_per_device = int(ranks_per_device)
        if ranks_per_device <= 0:
            raise ValueError("ranks_per_device must be positive")
        if ranks_per_device == 1:
            return self
        return replace(
            self,
            device_budget_gib=float(self.device_budget_gib) / ranks_per_device,
            linear_max_plaintext_rows=min(int(self.linear_max_plaintext_rows), SHARED_CARD_MAX_PLAINTEXT_ROWS),
        )


#: The largest ``linear_max_plaintext_rows`` a card that hosts more than one
#: T2 slot keeps (:meth:`ParallelPreset.with_world_size` below 8,
#: :meth:`ParallelPreset.for_colocation` above 1): 512 is the production
#: optimum of the single-slot-per-rank schedule and the only value measured
#: with several slots per card (4 x Pro 6000 ``--world-size 4``: peak
#: 56.5 GiB).  The 2048-row profile of :data:`RTX6000PRO_HIGH` is measured at
#: one slot per 96 GB card only (peak 74.5 GiB).
SHARED_CARD_MAX_PLAINTEXT_ROWS = 512


#: Q-limb caps measured by ``local diagnostic tooling`` (,
#: L3, layer-10 coefficients): the tracer's ``max_limbs`` per station.  QKV
#: enters at 13 limbs and asks for 14 (conjugation); the MLP enters at
#: ``SiLU depth + 6`` = 11.  ``linear`` must stay <= 16 to keep beta = 2
#: (17 limbs would need a third digit and +70% key bytes).  The attention
#: input / QK / PV tiers were traced at 11-13.  ``softmax`` rotates the
#: lambda row broadcast right after its bootstrap (post 17 -> square ->
#: mask = 15 limbs; pair rotations ``pair * stride`` are not powers of two
#: so the bootstrap tier does not cover them): 16 is the last beta = 2 cap.
MEASURED_LIMB_CAPS = (
    ("linear", 14),
    ("attn_input", 13),
    ("attn_input_lane", 13),
    ("attn_input_q", 13),
    ("attn_input_k", 13),
    ("attn_input_v", 13),
    ("attn_core", 13),
    ("attn_qk", 13),
    ("attn_pv", 13),
    ("softmax", 16),
    ("pairfm_output", 13),
)

RTX4090_LOW = ParallelPreset(
    name="rtx4090-low",
    depth=37,
    dnum=5,
    bootstrap=BootstrapPreset(level_budget=(5, 5), cache_mode="middle"),
    device_budget_gib=23.0,
    key_policy="lease",
    # 57 exclusive keys, 2.45 GiB at cap 14; leasing them costs 0.4-0.7 s
    # prefetch + 0.25-0.3 s evict per station x3 stations = 2-3 s/layer,
    # more than the whole 2.5 s/layer budget -> resident.
    key_resident_extra=("linear",),
    key_limb_caps=MEASURED_LIMB_CAPS,
    # QKV 22.34 / MLP 21.20 GiB peaks at 512 sit right under the 23 GiB
    # budget; 256 buys the headroom the async key prefetch needs.
    linear_max_plaintext_rows=256,
    # : pinned arena transport, attn_input exchange+unpack 0.32 ->
    # 0.04 s, layer 5.96 -> 5.59 s (profile measurements).
    transport="pinned",
    # run13 -> run15 (): lazy + device-sync evict spent 1.21 s key
    # H2D + 0.70 s evict + 0.24 s wait per layer; side-stream prefetch of the
    # next tier (11.3 GiB/layer, fully hidden: async_wait 0.0004 s) with a
    # compute-stream-only evict brings that to 0.02 s.  Peaks 20.6 GiB.
    key_prefetch_mode="async",
    key_prefetch_margin_gib=1.5,
    key_evict_sync="stream",
    # run16: gate_up/down pack_encode 1.1 s of the 2.0 s MLP at 1x speedup.
    # k=2 halves it for +2 resident baby bases (~1.4 GiB, peak ~20.2 GiB);
    # k=4 would need +6 (~4.2 GiB) and overflows the 23 GiB budget.
    # run18 (page split): MLP 2.06 -> 1.60 s.
    mlp_feature_shard_k=2,
    # run19: the ext ``blocked`` pairfm mask (512 rows x 19 limbs ~ 3.25 GiB)
    # is what forces the caching allocator into a retry (+0.35 s
    # block_pack_encode, plan §12.22); plain plaintexts drop pairfm 0.74 ->
    # 0.38 s and every station's alloc_retries to 0 at unchanged oracle error.
    transform_hoist_strategy="normal",
    # dq4 (, records/t2_decode_4090_20261001): on the W=8 decode
    # heavy layers {2,4,5,6,10} the 5e-3 rsqrt table takes the layer sum
    # 13.59 -> 11.50 s (L2 ffn rsqrt 4096 -> 2048 degree) and the joint_tol_v1
    # exp series is neutral (11.74 s); vs-poly error 1.7e-3 -> 1.5e-3 / 1.7e-3.
    # Same tables as the high tier so both presets run one configuration.
    rmsnorm_coeffs="8b/runtime/rmsnorm_poly_coeffs.json",
    softmax_coeffs="8b/runtime/softmax_poly_coeffs.json",
    softmax_coeffs_parts="exp",
)

def _high_preset(
    name: str, device_budget_gib: float, *, linear_max_plaintext_rows: int = 512
) -> ParallelPreset:
    """The efficiency-first profile shared by every >= 80 GB card.

    Same L3 bootstrap *shape* as :data:`RTX4090_LOW` (the only agreed BS
    program) with the ``plain`` constant cache (every CtoS / StoC / EvalMod
    constant resident as a ready plaintext, +2.7 GiB per rank: 8 x Pro 6000
    , 75 -> 61 ms per BS, 32 layers 79.8 -> 75.5 s; the 4090 keeps
    ``middle`` for memory); every key tier resident (the 651-key union at the
    measured caps is ~40 GiB, so a per-layer H2D / evict is pure loss on a
    card that fits it), 512-row linear chunks unless the caller passes the
    2048 of the 96 GB card, a 4-way MLP feature shard (the +6 hoisted baby
    bases fit; 8-way spends the encode saving on redundant babies again, plan
    §12.21), the moddown-free ext PairFM hoist (the 4090 only prefers
    ``normal`` because of caching-allocator retries; A/B on the target card
    with ``bench/t2_layer.py --transform-hoist``) and the split-phase
    attention-input exchange.
    """

    return ParallelPreset(
        name=str(name),
        depth=37,
        dnum=5,
        bootstrap=BootstrapPreset(level_budget=(5, 5), cache_mode="plain"),
        device_budget_gib=float(device_budget_gib),
        key_policy="resident",
        # Same caps as the 4090: they are a property of the operators' limb
        # schedule, not of the card.
        key_limb_caps=MEASURED_LIMB_CAPS,
        linear_max_plaintext_rows=int(linear_max_plaintext_rows),
        mlp_feature_shard_k=4,
        transform_hoist_strategy="ext_normal",
        # 8 x Pro 6000 32-layer A/B (): exchange 19 -> 4 ms per
        # layer at identical results; -0.4 s on 32 layers (records
        # t2_model_pro6000_8card_20261001, a4 vs a0).
        attention_input_overlap=True,
        # 4 x Pro 6000 A/B (): one Meta-BTS pass on the PairFM refresh
        # is accuracy-neutral (argmax / top-5 / logits) and -0.13 s/layer.
        pairfm_refresh_bootstrap_iterations=1,
        # 4 x Pro 6000 32-layer A/B (, records/t2_model_pro6000_20260930
        # combo_all): slim RMSNorm exit on layers 3-31 + the 5e-3 rsqrt table +
        # the joint_tol_v1 exp series together take the 32-layer wall
        # 131.5 -> 118.7 s (BS calls 496 -> 354, BS time 39.0 -> 26.5 s);
        # hidden_32 vs exact fp32 0.0733 -> 0.0708, argmax / top-5 5/5 unchanged.
        rmsnorm_exit_mode="slim",
        rmsnorm_slim_exit_layers="3-31",
        rmsnorm_coeffs="8b/runtime/rmsnorm_poly_coeffs.json",
        softmax_coeffs="8b/runtime/softmax_poly_coeffs.json",
        softmax_coeffs_parts="exp",
        silu_coeffs="8b/runtime/silu_poly_coeffs.json",
        # 8 x Pro 6000 A/B (, records/t2_prefill_floor_pro6000_8card_20261002):
        # period-16 slim pack -> sparse rsqrt BS, 32 layers 75.5 -> 70.8 s, L10
        # 2.27 -> 2.13 s/layer, argmax / top-5 5/5 unchanged.  Decode benches
        # override to 1 (one cipher per stream; bench/t2_decode.py).
        rmsnorm_slim_period_rows=16,
        # 8 x Pro 6000 A/B (, records/softmax_slim_pro6000_8card_20261002):
        # periodic slim pack of the Softmax Alg2 row sums -> sparse lambda BS,
        # 32 layers 70.8 -> 68.5 s (two back-to-back pairs: -2.0 / -2.3 s),
        # L10 attn_core 0.42 -> 0.37 s, argmax / top-5 5/5 unchanged.
        softmax_slim_periodic=True,
    )


#: 8 x A100 80 GB (the eventual full-model host).
A100_HIGH = _high_preset("a100-high", device_budget_gib=78.0)

#: 4 or 8 x RTX PRO 6000 Blackwell 96 GB (rented host).  8 cards: the
#: default (``world_size 8``, one slot per card).  4 cards: either
#: ``--world-size 4`` (one process per card owning two slots = four FM
#: ciphertexts; :meth:`ParallelPreset.with_world_size`) or the older
#: ``--ranks-per-device 2`` (two one-slot ~40 GiB processes per card).
#: Per card the FHE work is the same either way; ``--world-size 4`` encodes
#: the linear weights once per card instead of twice and halves the process
#: count / host RAM (measured 411 GB for 8 processes on the 440 GiB host).
#: 2048-row linear chunks: the 8-card 32-layer A/B of  (plain cache
#: + overlap + rows 2048 = 74.2 / 74.2 / 74.1 s against 75.4 s at 512 rows,
#: peak 52.0 -> 74.5 GiB, argmax / top-5 5/5, logits rel 1.5-1.8%).  Only at
#: one slot per card; ``--world-size 4`` / ``--ranks-per-device 2`` fall back
#: to 512 (:data:`SHARED_CARD_MAX_PLAINTEXT_ROWS`).
RTX6000PRO_HIGH = _high_preset("rtx6000pro-high", device_budget_gib=92.0, linear_max_plaintext_rows=2048)

#: One H100 80 GB pretending to be 8 ranks (:mod:`.emulation`); same knobs
#: as the real high-tier cards so the emulated numbers transfer.
H100_EMU = _high_preset("h100-emu", device_budget_gib=78.0)

#: 4 x A800 80 GB (NVLink-reduced A100, sm_80): the A100 profile re-cut to
#: ``world_size = 4`` -- one process per card owning two T2 slots (four FM
#: ciphertexts), the layout the 4 x Pro 6000 ``--world-size 4`` runs used
#: (peak 56.5 GiB per card at 512 rows, so the 80 GB card fits with ~20 GiB
#: of headroom; ``--mlp-shard-k 2`` is the fallback if the MLP station ever
#: OOMs).  The transport is the host-staged pinned arena, so the slower
#: A800 NVLink (400 vs 600 GB/s) is not on the data path.  Needs an EasyFHE
#: wheel with sm_80 cubins (tools/backend/build/run_build_sm80.sh; the
#:  wheel carries sm_90 / sm_120 only).  ``--world-size 8`` on an
#: 8-card A800 host recuts it back to one slot per card.
A800_4CARD = _high_preset("a800-4card", device_budget_gib=78.0).with_world_size(4)

#: ``<repo>/assets/polynomials``: base for model/role table paths in presets.
POLYNOMIALS_DIR = Path(__file__).resolve().parents[2] / "assets" / "polynomials"


def polynomial_table_path(spec: str | os.PathLike[str]) -> Path:
    """Resolve a preset / environment table spec to a file.

    An existing explicit path is accepted as given. A relative model/role
    path resolves under :data:`POLYNOMIALS_DIR`. The file must exist.
    """

    path = Path(spec).expanduser()
    if path.exists():
        return path
    if not path.is_absolute() and (POLYNOMIALS_DIR / path).exists():
        return POLYNOMIALS_DIR / path
    raise FileNotFoundError(f"polynomial table {str(spec)!r} not found (looked in {POLYNOMIALS_DIR})")


PRESETS: dict[str, ParallelPreset] = {
    RTX4090_LOW.name: RTX4090_LOW,
    A100_HIGH.name: A100_HIGH,
    RTX6000PRO_HIGH.name: RTX6000PRO_HIGH,
    H100_EMU.name: H100_EMU,
    A800_4CARD.name: A800_4CARD,
}

#: Presets whose memory knobs assume an >= 80 GB card (all key tiers resident).
HIGH_PRESETS: tuple[str, ...] = (A100_HIGH.name, RTX6000PRO_HIGH.name, H100_EMU.name, A800_4CARD.name)


def preset_by_name(name: str) -> ParallelPreset:
    try:
        return PRESETS[str(name)]
    except KeyError as exc:
        raise KeyError(
            f"unknown preset {name!r}; choose from {sorted(PRESETS)}"
        ) from exc


__all__ = [
    "A100_HIGH",
    "A800_4CARD",
    "H100_EMU",
    "HIGH_PRESETS",
    "PRESETS",
    "RTX4090_LOW",
    "RTX6000PRO_HIGH",
    "SHARED_CARD_MAX_PLAINTEXT_ROWS",
    "BootstrapPreset",
    "POLYNOMIALS_DIR",
    "ParallelPreset",
    "polynomial_table_path",
    "preset_by_name",
]
