#!/usr/bin/env python3
"""SC Llama 1B/3B/8B: encrypted Prefill, ciphertext KV and first/hot Decode.

--dry-run prints a portable execution plan without importing EasyFHE or reading
large weights. --preflight checks the prepared assets without launching GPUs.
Decode always derives its encrypted KV from this invocation's FHE Prefill.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MODELS = {'1b': ('llama32-1b', 16), '3b': ('llama32-3b', 28), '8b': ('llama3-8b', 32)}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=tuple(MODELS), required=True)
    parser.add_argument('--phase', choices=('prefill', 'decode', 'linked'), default='linked')
    parser.add_argument('--world-size', type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument('--assets-root', type=Path, required=True,
                        help='generated assets root; SC assets are under sc/{1b,3b,8b}/chain')
    parser.add_argument('--output', '--out', type=Path, required=True)
    parser.add_argument('--devices', default='', help='comma-separated physical GPU indices')
    parser.add_argument('--chain', type=Path)
    parser.add_argument('--query-row', type=Path)
    parser.add_argument('--body-ids', type=Path, default=REPO / 'assets/prefill128_input_ids.json')
    parser.add_argument('--source-checkpoint', type=Path,
                        help='8B terminal checkpoint; must agree with chain-manifest.json')
    parser.add_argument('--nonlinear-mode', choices=('combined', 'legacy'), default='combined')
    parser.add_argument('--lookahead', type=int, default=2)
    parser.add_argument('--warmup-passes', type=int, default=1)
    parser.add_argument('--measure-passes', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=1200., help='worker deadline, seconds')
    parser.add_argument('--transport', choices=('pinned', 'host'), default='pinned')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    return parser


def make_config(args):
    model, layers = MODELS[args.model]
    assets = args.assets_root.expanduser().resolve() / 'sc' / args.model
    provider = Path(__file__).resolve().parent / 'providers' / f'llama{args.model}'
    return dict(model=model, layers=layers, phase=args.phase,
                sylph_root=str(provider), odin_root=str(REPO),
                chain=str((args.chain or assets / 'chain').resolve()),
                query_row=str((args.query_row or assets / 'input/query_embedding.npz').resolve())
                    if args.model != '8b' else '',
                body_ids=str(args.body_ids.resolve()),
                source_checkpoint=str(args.source_checkpoint.resolve()) if args.source_checkpoint else '',
                out=str(args.output.resolve()), lookahead=args.lookahead,
                warmup_passes=args.warmup_passes, measure_passes=args.measure_passes,
                nonlinear_mode=args.nonlinear_mode,
                preflight_only=False)


def _critical_sum(reports):
    layers = len(reports[0]['layers'])
    if any(len(r['layers']) != layers for r in reports):
        raise ValueError('rank reports cover different layer counts')
    return sum(max(float(r['layers'][i]['seconds']) for r in reports) for i in range(layers))


def summarize(output, rank_results, config, world):
    result = dict(model=config['model'], provider='sc', world_size=world,
                  phase=config['phase'], kv_source='same-run encrypted Prefill',
                  context_tokens=128, prefix_tokens=0,
                  timing='sum of per-layer maximum-rank forward; excludes independent setup and uploads',
                  decode_semantics='first call plus same-query/position/original-KV replay; not continuous generation',
                  nonlinear_mode=config['nonlinear_mode'])
    if config['phase'] == 'prefill':
        reports = [r['prefill_report'] for r in rank_results]
        result.update(valid_inference=all(r['passed'] for r in rank_results),
                      prefill_online_s=_critical_sum(reports),
                      prefill_chain_wall_s=[r.get('chain_seconds') for r in reports])
    else:
        result['kv_link_verified'] = all(r.get('summary', {}).get('kv_from_fhe_prefill') is True
                                       for r in rank_results)
        passes = []
        for index in range(config['warmup_passes'] + config['measure_passes']):
            reports = [json.loads((output / f'rank{rank}' / f'decode_cipher_kv_pass{index}.json').read_text())
                       for rank in range(world)]
            valid = all(r.get('kv_from_fhe_prefill') is True
                        and len(r.get('layers', [])) == config['layers']
                        and r.get('next_token_id') == r.get('reference_next_token_id')
                        and r.get('terminal', {}).get('comparison', {}).get('passed') is True
                        and r.get('terminal', {}).get('comparison', {}).get('top_k_match') is True
                        and math.isfinite(float(r.get('hidden_rel_error', float('nan'))))
                        for r in reports)
            passes.append(dict(pass_index=index, measured=index >= config['warmup_passes'],
                               valid_inference=valid,
                               first_decode=index == 0, decode_online_s=_critical_sum(reports),
                               chain_wall_s=[r.get('chain_seconds') for r in reports],
                               tokens=[r.get('next_token_id') for r in reports]))
        result['decode_passes'] = passes
        result['valid_inference'] = (result['kv_link_verified']
            and all(r.get('summary', {}).get('prefill_terminal_passed') is True for r in rank_results)
            and all(p['valid_inference'] for p in passes))
        prefills = []
        for rank in range(world):
            paths = sorted((output / f'rank{rank}').glob('chain-*.json'))
            reports = [json.loads(p.read_text()) for p in paths]
            report = next((r for r in reports if len(r.get('layers', [])) == config['layers']), None)
            if report is not None:
                prefills.append(report)
        if len(prefills) == world:
            result['prefill_online_s'] = _critical_sum(prefills)
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.lookahead, args.warmup_passes, args.measure_passes) < 1:
        parser.error('lookahead/warmup-passes/measure-passes must be positive')
    try:
        devices = tuple(int(v) for v in args.devices.split(',')) if args.devices else tuple(range(args.world_size))
    except ValueError:
        parser.error('--devices must be comma-separated integers')
    if len(devices) != args.world_size or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error('--devices must contain one distinct nonnegative index per rank')
    config = make_config(args)
    protocol = dict(config=config, devices=devices, backend='EasyFHE CKKS; host-staged ciphertext communication',
                    workload='128-token P0 frozen benchmark fixture',
                    decode_initialization='FHE Prefill emitted ciphertext KV; never encrypted plaintext KV',
                    weight_policy=('CPU projection tables; per-layer H2D' if args.model == '8b' and args.world_size < 8
                                   else 'GPU-resident Decode tables' if args.world_size == 8 and args.model in ('3b','8b')
                                   else 'layer streaming'),
                    validation='dry-run plan only; no runtime or asset validation yet')
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return 0
    from baselines.sc.parallel import run_linked, runtime
    source = run_linked._source_preflight(config)
    config['plain_inputs'] = source['plain_inputs']
    manifest = json.loads((Path(config['chain']) / 'chain-manifest.json').read_text())
    config['expected_prefill_token'] = manifest.get('expected_next_token_id', 12366)
    protocol.update(source=source, validation='asset preflight passed')
    if args.preflight:
        print(json.dumps(protocol, indent=2))
        return 0
    output = Path(config['out'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    outcomes = runtime.run_world('baselines.sc.worker:run_rank', args.world_size,
        odin_root=str(REPO), devices=devices, kwargs={'config': config},
        timeout=args.timeout, transport_kind=args.transport,
        worker_env={'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
                    'SC_DECODE_INPUT_SYNC': 'every', 'SC_RESIDENT_UNSTAGED_LOAD': '1',
                    'SC_RESIDENT_MIN_FREE_MIB': '4096'})
    result = summarize(output, [o.result for o in outcomes], config, args.world_size)
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
