"""Compose Couplet's Prefill and Decode layer execution.

``common`` defines shared records, audits and rotation planning. ``prefill``
implements ``evaluate`` and ``decode`` implements ``evaluate_decode``. This
package re-exports their public interfaces; patch helpers in the submodule
that calls them when instrumenting a layer.
"""

from __future__ import annotations

from .common import (
    LayerApproximations,
    LayerResult,
    LayerSchedule,
    LayerWeights,
    _attention_exclusive_cache_keep_rotations,
    _cipher_levels,
    _cipher_runtime_state,
    _cipher_stream_state_audit,
    _evict_attention_exclusive_cuda_cache,
    _decode_linear_config,
    _linear_config,
    _pairfold_prefill_cache_from_raw_c,
    _residual_state_audit,
    _select_residual_identity,
    operation_audit,
    required_rotations,
)
from .decode import (
    decode_layer_rotations,
    decode_sparse_bootstrap_log_slots,
    evaluate_decode,
    inspect_decode_layer_input,
)
from .prefill import evaluate

__all__ = [
    "LayerResult",
    "LayerSchedule",
    "decode_layer_rotations",
    "decode_sparse_bootstrap_log_slots",
    "evaluate",
    "evaluate_decode",
    "inspect_decode_layer_input",
    "operation_audit",
    "required_rotations",
]
