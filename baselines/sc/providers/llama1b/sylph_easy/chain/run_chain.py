"""Encrypt the Paris128 input once, run the QuaRot decoder layers (32 for Llama-3-8B, 16 for
Llama-3.2-1B; ``SYLPH_CHAIN`` selects the chain) as one ciphertext chain on EasyFHE, decrypt once,
and decode the next token with the client terminal head.

usage: run_chain.py [first_layer] [last_layer] [--diagnose]
--diagnose decrypts after each layer to measure error against the plaintext chain (debug only;
the acceptance run never decrypts in between).
"""
import os, gc, json, sys, time
# Public plan fits (small lstsq) run in prefetch threads: one BLAS thread each is ~5x faster
# than OpenBLAS's 64 oversubscribed threads, with bit-identical coefficients.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, __import__('os').environ.get('SYLPH_ROOT', __import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '../..'))))
sys.path.insert(0, __import__('os').path.join(__import__('os').environ['SYLPH_ROOT'], 'sylph_easy/tools'))
import numpy as np
import easyfhe as torch
from sylph_easy.chain.manifest import CHAIN, PLAIN_INPUTS, geometry, layers, load, unstage
from sylph_easy.layer.setup import create, SLOTS
from sylph_easy.layer import layout
from sylph_easy.layer.decoder import DecoderLayer
import terminal_logits as tl

args = [a for a in sys.argv[1:] if not a.startswith('--')]
diagnose = '--diagnose' in sys.argv
manifest, entries = layers()
n_layers = len(entries)
first = int(args[0]) if args else 0
last = int(args[1]) if len(args) > 1 else n_layers - 1
plain_inputs = np.load(PLAIN_INPUTS)
x = plain_inputs[f'layer{first:02d}']
head_dim, heads, kv_heads = geometry(manifest)
LOGS = os.environ.get('SYLPH_LOGS', os.path.abspath('artifacts/sc'))
os.makedirs(LOGS, exist_ok=True)
tag = manifest.get('model_name', 'llama3-8b')

client, b = create(head_dim=head_dim, heads=heads, kv_heads=kv_heads)
dec = lambda ct: np.stack([client.decrypt(c).cpu().numpy()[:SLOTS].real for c in b.unstack(b.norm(ct))])
ct = b.stack([client.encrypt(v, slots=SLOTS) for v in layout.pack(x)])
report = dict(model=tag, chain=CHAIN, first=first, last=last, diagnose=diagnose, layers=[])
t_chain = time.time()
chain = entries[first:last + 1]
LOOKAHEAD = int(os.environ.get('SYLPH_LOOKAHEAD', '4'))   # layers read and planned ahead on the host


def prepare(entry):
    """Host-only: read the weights and build the layer's public plans (no GPU work)."""
    w, cal = load(entry, staged=True)
    return w, DecoderLayer(b, w, cal, progress=False, upload=False)


def to_gpu(host):
    """Upload the next layer's projection weights (side stream) once its host part is ready."""
    w, layer = host.result()
    layer.upload()
    return w, layer


host_pool, gpu_pool = ThreadPoolExecutor(max(1, LOOKAHEAD)), ThreadPoolExecutor(1)
host = {i: host_pool.submit(prepare, chain[i]) for i in range(min(max(1, LOOKAHEAD), len(chain)))}
pending = gpu_pool.submit(to_gpu, host.pop(0))
WARMUP = os.environ.get('SYLPH_WARMUP', '0') == '1'
if WARMUP:
    # Server-side warm-up mirroring Couplet's --warmup-layer0: run the first layer once on ciphertexts of
    # random rows with the real input's RMS and the real first-layer weights, then drop the outputs,
    # the per-layer plaintext cache, the stage timers and the op counters.  Only one-time costs
    # (CUDA module load / JIT, allocator growth, plan + rotation caches) stay behind.
    t_w = time.time()
    rng = np.random.default_rng(20260930)
    dummy = (rng.standard_normal(x.shape) * float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))).astype(np.float32)
    ct_dummy = b.stack([client.encrypt(v, slots=SLOTS) for v in layout.pack(dummy)])
    w_warm, layer_warm = pending.result()
    torch.cuda.synchronize(); t_w1 = time.time()
    out_warm = layer_warm.forward(ct_dummy)
    torch.cuda.synchronize()
    report['warmup'] = dict(layer=chain[0]['layer'], seconds=round(time.time() - t_w, 2),
                            forward_seconds=round(time.time() - t_w1, 2),
                            stages={k: round(v, 2) for k, v in layer_warm.times.items()},
                            counts=dict(b.counts))
    print(json.dumps({'warmup': {k: report['warmup'][k] for k in ('layer', 'seconds', 'forward_seconds')}}), flush=True)
    del out_warm, ct_dummy, dummy, w_warm, layer_warm
    b.counts = {}
    b.clear_cache(); gc.collect(); torch.cuda.synchronize()
    pending.result()[1].times = {}     # same uploaded layer-0 object, stage timers reset
    t_chain = time.time()
for idx, e in enumerate(chain):
    t0 = time.time()
    if LOOKAHEAD == 0 and idx > 0:           # no prefetch: read, plan and upload in line
        pending = gpu_pool.submit(to_gpu, host_pool.submit(prepare, e))
    w, layer = pending.result()
    t_load = time.time() - t0                # time spent waiting for the prefetch
    if LOOKAHEAD > 0:
        if idx + LOOKAHEAD < len(chain):
            host[idx + LOOKAHEAD] = host_pool.submit(prepare, chain[idx + LOOKAHEAD])
    torch.cuda.synchronize(); t1 = time.time()
    ct = layer.forward(ct)
    torch.cuda.synchronize(); t_run = time.time() - t1
    row = dict(layer=e['layer'], seconds=round(t_run, 2), load_seconds=round(t_load, 2),
               limbs=b.limbs(ct), stages={k: round(v, 2) for k, v in layer.times.items()})
    if diagnose:
        key = f"layer{e['layer'] + 1:02d}" if e['layer'] < n_layers - 1 else 'final'
        want = plain_inputs[key]
        got = layout.unpack(dec(ct))
        row['max_error'] = float(np.abs(got - want).max())
        row['rel_error'] = row['max_error'] / float(np.abs(want).max())
    report['layers'].append(row)
    print(json.dumps(row), flush=True)
    last_w = w
    del layer, w
    unstage(e)
    b.clear_cache(); gc.collect()
    if LOOKAHEAD > 0 and idx + 1 < len(chain):
        pending = gpu_pool.submit(to_gpu, host.pop(idx + 1))
report['chain_seconds'] = round(time.time() - t_chain, 1)
hidden = layout.unpack(dec(ct))
np.save(f'{LOGS}/chain-final-hidden-{tag}.npy', hidden)
terminal = manifest.get('terminal', {})
if last == n_layers - 1 and terminal.get('kind') == 'quarot_folded_norm':
    # QuaRot chain (1B): the final norm gamma is folded into the rotated lm_head; no extra Sylph R.
    want = plain_inputs['final']
    report['hidden_max_error'] = float(np.abs(hidden - want).max())
    lm_head = np.load(os.path.join(CHAIN, terminal['lm_head']))
    eps = float(terminal['epsilon'])
    logit = lambda h: ((h[-1:] / np.sqrt(np.mean(h[-1:] ** 2, axis=-1, keepdims=True) + eps)) @ lm_head.T.astype(np.float64)).reshape(-1)
    actual, reference = logit(hidden.astype(np.float64)), logit(want.astype(np.float64))
    report['terminal'] = dict(actual=tl.summarize_logits(actual), reference=tl.summarize_logits(reference),
                              comparison=tl.compare_last_token_logits(actual, reference))
    report['next_token_id'] = report['terminal']['actual']['argmax_token_id']
    report['expected_next_token_id'] = manifest.get('expected_next_token_id')
    report['paris'] = report['next_token_id'] == 12366
elif last == n_layers - 1:
    h = tl.inverse_sylph_residual(hidden, last_w['residual_signs'], last_w['residual_permutation'])
    ref = np.load(CHAIN + '/client-plain-final-hidden.npy')
    logits, rep = tl.evaluate_terminal_logits(manifest['source_checkpoint'], h, reference_hidden=ref, top_k=5)
    report['hidden_max_error'] = float(np.abs(h - ref).max())
    report['terminal'] = dict(actual=rep['actual'], reference=rep['reference'], comparison=rep['comparison'])
    report['next_token_id'] = rep['actual']['argmax_token_id']
    report['paris'] = report['next_token_id'] == 12366
else:
    want = plain_inputs[f'layer{last + 1:02d}']
    report['hidden_max_error'] = float(np.abs(hidden - want).max())
report['counts'] = b.counts
out = f'{LOGS}/chain-{tag}-{first:02d}-{last:02d}{"-diag" if diagnose else ""}.json'
json.dump(report, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in report.items() if k != 'layers'}, indent=1))
