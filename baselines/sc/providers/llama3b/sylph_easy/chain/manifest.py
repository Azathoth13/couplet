"""Resolve a QuaRot chain manifest (weights, calibration, input) per layer.

Defaults are the 32-layer Llama-3-8B upstream chain; ``SYLPH_CHAIN`` / ``SYLPH_PLAIN_INPUTS`` /
``SYLPH_L0_INPUT`` select another chain directory (e.g. the Llama-3.2-1B chain written by
``tools/make_1b_chain.py``, whose manifest carries ``model`` / ``plain_inputs`` / ``terminal``)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np

CHAIN = os.environ.get('SYLPH_CHAIN', os.path.abspath('assets/sc/chain'))
L0_INPUT = os.environ.get('SYLPH_L0_INPUT', os.path.join(CHAIN, 'layer-00/rotated-inputs.npy'))
_manifest_plain = None
try:
    _manifest_plain = json.load(open(os.path.join(CHAIN, 'chain-manifest.json'))).get('plain_inputs')
except (OSError, ValueError):
    pass
PLAIN_INPUTS = os.environ.get('SYLPH_PLAIN_INPUTS', os.path.join(CHAIN, _manifest_plain) if _manifest_plain
                              else os.path.join(CHAIN, 'plain_inputs.npz'))


def layers(chain=CHAIN):
    manifest = json.load(open(os.path.join(chain, 'chain-manifest.json')))
    out = []
    for entry in manifest['layers']:
        cal = entry['calibration']
        cal = cal if os.path.isabs(cal) else os.path.join(chain, cal)
        out.append(dict(layer=int(entry['layer']), calibration=cal,
                        weights=os.path.join(os.path.dirname(cal), 'weights.npz')))
    return manifest, out


def geometry(manifest):
    """(head_dim, heads, kv_heads) for ``setup.create``: the manifest's ``model`` or Llama-3-8B."""
    m = manifest.get('model', {})
    if os.environ.get('SYLPH_HEAD24_COMPACT') == '1':
        if (manifest.get('model_name') != 'llama3.2-3b-padded-quarot'
                or (int(m.get('head_dim', 0)), int(m.get('heads', 0)), int(m.get('kv_heads', 0)),
                    int(m.get('hidden', 0))) != (128, 32, 8, 4096)):
            raise ValueError('head24 compaction requires the padded Llama-3.2-3B chain')
        return 128, 24, 8
    return int(m.get('head_dim', 128)), int(m.get('heads', 32)), int(m.get('kv_heads', 8))


SHM = os.environ.get('SYLPH_SHM', '/dev/shm/sylph-easy')
# The network share serves ~45 MB/s per stream but scales with streams: pull the archive in
# parallel 64 MB ranges into shared memory, then unpack it from there.
_UNPACK = """
import os, sys, numpy as np
from concurrent.futures import ThreadPoolExecutor
src, dst = sys.argv[1:3]
size, tmp, chunk = os.path.getsize(src), dst + '/weights.npz', 64 << 20
with open(tmp, 'wb') as f:
    f.truncate(size)
mm = np.memmap(tmp, dtype=np.uint8, mode='r+', shape=(size,))
fd = os.open(src, os.O_RDONLY)
def pull(off):
    end, pos = min(off + chunk, size), off
    while pos < end:
        buf = os.pread(fd, end - pos, pos)
        mm[pos:pos + len(buf)] = np.frombuffer(buf, dtype=np.uint8)
        pos += len(buf)
with ThreadPoolExecutor(16) as pool:
    list(pool.map(pull, range(0, size, chunk)))
mm.flush(); del mm
z = np.load(tmp)
for k in z.files:
    np.save(f'{dst}/{k}.npy', z[k])
os.remove(tmp)
open(dst + '/DONE', 'w').close()
"""


def stage(entry):
    """Unpack one layer's weights.npz into raw .npy files in shared memory.

    Reading the 1.7 GB npz runs zipfile's Python loop (GIL held) for the whole network read; done
    in a subprocess it leaves the GPU thread alone, and the .npy reads are single GIL-free calls.
    """
    d = os.path.join(SHM, f"layer-{entry['layer']:02d}")
    if not os.path.exists(os.path.join(d, 'DONE')):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        subprocess.run([sys.executable, '-c', _UNPACK, entry['weights'], d], check=True)
    return d


def unstage(entry):
    shutil.rmtree(os.path.join(SHM, f"layer-{entry['layer']:02d}"), ignore_errors=True)


def _drop_public_prefix(w, cal):
    """SYLPH_PUBLIC_PREFIX=0: run a chain built with P public prefix tokens as P = 0 (timing A/B against
    the no-prefix Couplet runs).  The private 128 tokens, weights and calibration are unchanged; only the
    plaintext prefix K/V pages are removed, so the layer outputs no longer match the chain's plain
    inputs (the error columns of run_chain are not meaningful in this mode)."""
    if os.environ.get('SYLPH_PUBLIC_PREFIX', '') != '0':
        return w, cal
    pk = np.asarray(w['public_keys'])
    if pk.shape[0]:
        w = dict(w)
        w['public_keys'] = np.zeros((0,) + pk.shape[1:], dtype=pk.dtype)
        w['public_values'] = np.zeros((0,) + np.asarray(w['public_values']).shape[1:], dtype=pk.dtype)
        cal = dict(cal, public_prefix_tokens=0)
    return w, cal


def _compact_head24(w, cal):
    """Remove only the eight zero-padded Q heads; retain the original fit and oracle.

    This is an in-memory transform of a loaded layer.  It avoids writing 28
    nearly duplicate weight archives and keeps the archived chain immutable.
    """
    if os.environ.get('SYLPH_HEAD24_COMPACT') != '1':
        return w, cal
    if (np.shape(w['q']) != (4096, 4096) or np.shape(w['o']) != (4096, 4096)
            or int(cal.get('model', {}).get('heads', 0)) != 32
            or int(cal.get('model', {}).get('kv_heads', 0)) != 8):
        raise ValueError('head24 compaction received the wrong weights or calibration')
    active = (4 * np.arange(8)[:, None] + np.arange(3)[None, :]).reshape(-1)
    inactive = 4 * np.arange(8) + 3
    features = (active[:, None] * 128 + np.arange(128)[None, :]).reshape(-1)
    unused = (inactive[:, None] * 128 + np.arange(128)[None, :]).reshape(-1)
    if np.any(w['q'][unused]) or np.any(w['o'][:, unused]):
        raise ValueError('archived Q/O padding is not exactly zero')
    head_intervals = np.asarray(cal['softmax_head_intervals'])
    if head_intervals.shape != (32, 2):
        raise ValueError('expected one calibrated interval per padded Q head')
    w = dict(w)
    w['q'] = np.ascontiguousarray(w['q'][features])
    w['o'] = np.ascontiguousarray(w['o'][:, features])
    cal = dict(cal)
    cal['model'] = dict(cal['model'], heads=24)
    cal['softmax_head_intervals'] = head_intervals[active].tolist()
    return w, cal


def load(entry, staged=False):
    """(weights dict, calibration); ``staged`` reads through the shared-memory copy (same arrays)."""
    cal = json.load(open(entry['calibration']))
    if not staged:
        w, cal = _drop_public_prefix(dict(np.load(entry['weights'])), cal)
        return _compact_head24(w, cal)
    d = stage(entry)
    names = sorted(f[:-4] for f in os.listdir(d) if f.endswith('.npy'))
    w, cal = _drop_public_prefix({k: np.load(os.path.join(d, k + '.npy')) for k in names}, cal)
    return _compact_head24(w, cal)
