"""Compose the released SC sharding and cache policies in each GPU worker."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .parallel import run_linked as linked
from .parallel import prefill_shard


class PrefillComplete(Exception):
    def __init__(self, report):
        super().__init__('encrypted Prefill and ciphertext KV capture complete')
        self.report = report


def run_rank(runtime: Any, *, config: dict[str, Any]) -> dict[str, Any]:
    """The FHE arithmetic is the archived driver; only execution policy is composed."""
    original_factory = prefill_shard.make_distributed_decoder_layer
    original_loader = linked._load_driver
    restorations = []
    rms_audit = []
    os.environ['SC_PREFILL_ATTENTION_SHARD'] = '1'
    os.environ['SC_DECODE_INPUT_SYNC'] = 'every'
    os.environ['SYLPH_HEAD24_COMPACT'] = '1' if config['model'] == 'llama32-3b' else '0'

    def factory(base_class, rank_runtime, **kwargs):
        if rank_runtime.world_size == 1 or config['nonlinear_mode'] == 'legacy':
            return original_factory(base_class, rank_runtime, **kwargs)
        from .parallel.rms_two_roots import make_rmsnorm_factory
        from .parallel.silu_batch import make_silu_batch_decoder_layer
        rms_factory = make_rmsnorm_factory(original_factory, mode='two_roots', audit=rms_audit)
        return make_silu_batch_decoder_layer(base_class, rank_runtime,
                                              original_factory=rms_factory, **kwargs)

    def load_driver(model):
        driver = original_loader(model)
        base = driver.base if hasattr(driver, 'base') else driver
        if config['phase'] == 'prefill':
            old_capture = base.capture_prefill_caches

            def capture_and_stop(*args, **kwargs):
                result = old_capture(*args, **kwargs)
                report, _, _, saved, meta = result[:5]
                comparison = report.get('terminal', {}).get('comparison', {})
                passed = (len(report.get('layers', [])) == config['layers']
                          and len(saved) == config['layers']
                          and report.get('next_token_id') == config['expected_prefill_token']
                          and comparison.get('passed') is True
                          and comparison.get('top_k_match') is True)
                payload = dict(rank=runtime.rank, passed=passed,
                               kv_layers=len(saved), decode_executed=False,
                               prefill_report=report, kv_meta=meta,
                               rmsnorm_comm_compute=rms_audit)
                out = Path(config['out']) / f'rank{runtime.rank}'
                (out / 'prefill_only_report.json').write_text(json.dumps(payload, indent=2))
                if not passed:
                    raise RuntimeError('complete SC Prefill token/top-5/ciphertext KV validation failed')
                raise PrefillComplete(payload)

            restorations.append((base, 'capture_prefill_caches', old_capture))
            base.capture_prefill_caches = capture_and_stop
        else:
            old_pass = base.decode_pass
            from .parallel.no_resident import make_decode_pass as replay_unstaged
            if config['model'] == 'llama3-8b':
                from .parallel.oracle import make_decode_pass as replay
                inner = replay(base, runtime, old_pass, mode='replay')
                if runtime.world_size == 8:
                    from .parallel.resident8b import make_decode_pass as pool
                else:
                    from .parallel.host_tables import make_decode_pass as pool
                new_pass = pool(base, runtime, inner,
                                total_passes=config['warmup_passes'] + config['measure_passes'])
            elif config['model'] == 'llama32-3b' and runtime.world_size == 8:
                from .parallel.oracle import make_decode_pass as replay
                from .parallel.resident3b import make_decode_pass as pool
                inner = replay(base, runtime, old_pass, mode='replay')
                new_pass = pool(base, runtime, inner,
                                total_passes=config['warmup_passes'] + config['measure_passes'])
            else:
                new_pass = replay_unstaged(base, runtime, old_pass)
            restorations.append((base, 'decode_pass', old_pass))
            def recorded_pass(*args, **kwargs):
                report = new_pass(*args, **kwargs)
                # Original drivers only persist measured passes. Keep the first
                # call as well, after all original correctness checks have run.
                target = Path(kwargs['out']) / f"decode_cipher_kv_pass{kwargs['index']}.json"
                target.write_text(json.dumps(report, indent=2))
                return report

            base.decode_pass = recorded_pass
        return driver

    prefill_shard.make_distributed_decoder_layer = factory
    linked._load_driver = load_driver
    try:
        try:
            return linked.run_rank(runtime, config=config)
        except PrefillComplete as complete:
            return complete.report
    finally:
        linked._load_driver = original_loader
        prefill_shard.make_distributed_decoder_layer = original_factory
        for module, name, old in reversed(restorations):
            setattr(module, name, old)
