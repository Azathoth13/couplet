"""CKKS context, bootstrap program and rotation keys for the Sylph layer on EasyFHE."""
from __future__ import annotations

import time

import easyfhe.fhe as fhe
import easyfhe.bs.openfhe as bs

from ..easyfhe_ext import evalmod, hostcache, keyswitch
from ..easyfhe_ext.coeff import CoefficientKit, packing_rotations
from ..easyfhe_ext.engine import Engine
from .attention import attention_rotations
from .layout import HeadGeometry

LOGN = 16
N = 1 << LOGN
SLOTS = N // 2
TOKENS = 128
LANES = 64            # tokens per real slot ciphertext
FEATURES = 512        # features per slot ciphertext


def layer_rotations():
    rots = set()
    # feature reduction: strides 64 * 2^k inside one ciphertext
    rots |= {LANES << k for k in range(9)}
    # head-position (hdp) moves for GQA expansion
    rots |= {64, 128, 192, SLOTS - 64, SLOTS - 128}
    # lane diagonals j and j - 64 (masked wrap)
    rots |= {j for j in range(1, LANES)} | {(j - LANES) % SLOTS for j in range(1, LANES)}
    return rots


def create(*, extra_rotations=(), verbose=True, level_budget=(3, 3), output_levels=10, dnum=3,
           head_dim=128, heads=32, kv_heads=8):
    t0 = time.time()
    geometry = HeadGeometry(head_dim, heads, kv_heads)
    hostcache.install()
    keyswitch.install()
    evalmod.install()
    spec_bs = bs.BootstrapSpec(log_slots=15, level_budget=tuple(level_budget), output_levels=output_levels)
    req = bs.requirements(spec_bs, log_n=LOGN)
    rots = set(req.rotations) | {2 * N - 1} | set(packing_rotations(N, N // TOKENS)) | layer_rotations() | attention_rotations(geometry)
    rots |= set(int(r) % SLOTS for r in extra_rotations)
    rots.discard(0)
    client, ctx = fhe.generate_client_context(fhe.CKKSContextSpec(
        depth=req.context_depth, log_n=LOGN, dnum=dnum, dcrt_bits=59, first_mod=60,
        secret_key_dist="SPARSE_TERNARY", scale_mode="fixed", rescale_policy="manual",
        rotations=tuple(sorted(rots)), auto_load_keys=True), device="cuda")
    prog = bs.generate(ctx, spec_bs)
    kit = CoefficientKit(ctx, prog, tokens=TOKENS)
    eng = Engine(ctx, prog, kit)
    if verbose:
        print(f"[setup] {time.time() - t0:.1f}s L={ctx.L} dnum={dnum} rotations={len(rots)}", flush=True)
    return client, eng
