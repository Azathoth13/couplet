"""Batched same-offset rotation with cached device plans.

EasyFHE's ``homo_rotate`` on a batch rebuilds its key-product index and special-modulus
start tensors on the host and copies them to the GPU on every call. Sylph's packing and
attention call it tens of thousands of times per layer with a handful of distinct
(offset, batch, limbs) triples, so this module caches those device tensors. The kernel
sequence is the one of ``easyfhe.fhe.ops.rotation._batch_same_rotate``.
"""
from __future__ import annotations

import easyfhe as torch
from easyfhe.fhe.ops import arithmetic, kernels as F, rotation
from easyfhe.fhe.ops.metadata import active_limbs

from . import keyswitch

MAX_BATCH = 64  # limit of EasyFHE's fused key-switch kernels

_plans = {}


def clear_plans():
    """Drop cached rotation-key plans. The next rotation reloads the key it needs."""
    _plans.clear()


def _plan(ctx, offset, batch, limbs):
    key = (id(ctx), offset, batch, limbs)
    p = _plans.get(key)
    if p is None:
        offsets = (offset,) * batch
        _, indices = rotation._batch_rotation_product_plan(offsets, ctx)
        bxs, axs, starts = rotation._batch_rotation_keys_and_starts(offsets, ctx, limbs)
        maps = rotation._precompute_auto_maps(offsets, ctx)
        p = _plans[key] = (indices, bxs, axs, starts, maps)
    return p


def _digits(c1, limbs, ctx):
    if keyswitch.P1():
        return keyswitch.modup_p1(c1, limbs, ctx), c1
    return F.cv_modup(c1, limbs, ctx), None


def _one(ct, offset, ctx, addend):
    if ct.batch_size == 1:
        return rotation.homo_rotate_add(ct, offset, ctx, addend=addend)
    if keyswitch.enabled():
        digits, own = _digits(ct.cv[1], ct.state.cur_limbs, ctx)
        add = (None, None) if addend is None else (addend.cv[0], addend.cv[1])
        cv = keyswitch.rotate_pair(ct.cv[0], digits, offset, ct.state.cur_limbs, ctx, *add, own=own)
        return ct.cipher_like(list(cv))
    indices, bxs, axs, starts, maps = _plan(ctx, offset, int(ct.batch_size), ct.state.cur_limbs)
    digits = rotation._modup_to_ext(ct, ctx)
    kb, ka = F.cv_innerproduct_pairwise(digits.cv[0], digits.state.cur_limbs, starts, bxs, axs, ctx)
    cv = F.cv_finalize_fast_rotation_ext(kb, ka, indices, ct.cv[0], ct.cv[1], maps,
                                         ct.state.cur_limbs, active_limbs(digits, ctx), ctx)
    out = rotation.moddown_from_ext(ct.cipher_like(list(cv), is_ext=True, batch_size=ct.batch_size), ctx)
    return out if addend is None else arithmetic.homo_add(addend, out, ctx)


def rotate_hoisted(ct, offsets, ctx):
    """[rot(ct, o) for o in offsets] for a batch of any size <= MAX_BATCH: the key-switch
    decomposition (ModUp) of ``ct`` does not depend on the offset and runs once."""
    offsets = [int(o) for o in offsets]
    n = int(ct.batch_size)
    nz = [o for o in offsets if o]
    if n == 1 or n > MAX_BATCH or len(set(nz)) < 2:
        return [ct if o == 0 else rotate(ct, o, ctx) for o in offsets]
    if keyswitch.enabled():
        limbs = ct.state.cur_limbs
        digits, own = _digits(ct.cv[1], limbs, ctx)
        done = {o: ct.cipher_like(list(keyswitch.rotate_pair(ct.cv[0], digits, o, limbs, ctx, own=own)))
                for o in set(nz)}
        return [ct if o == 0 else done[o] for o in offsets]
    digits = rotation._modup_to_ext(ct, ctx)
    limbs, act = ct.state.cur_limbs, active_limbs(digits, ctx)
    done = {}
    for o in set(nz):
        indices, bxs, axs, starts, maps = _plan(ctx, o, n, limbs)
        kb, ka = F.cv_innerproduct_pairwise(digits.cv[0], digits.state.cur_limbs, starts, bxs, axs, ctx)
        cv = F.cv_finalize_fast_rotation_ext(kb, ka, indices, ct.cv[0], ct.cv[1], maps, limbs, act, ctx)
        done[o] = rotation.moddown_from_ext(ct.cipher_like(list(cv), is_ext=True, batch_size=n), ctx)
    return [ct if o == 0 else done[o] for o in offsets]


def _slice(ct, s, e):
    return ct.cipher_like([x[s:e] for x in ct.cv], batch_size=e - s)


def rotate(ct, offset, ctx, addend=None):
    """addend + rot(ct, offset) for a batch of any size (one automorphism for all items)."""
    offset = int(offset)
    if offset == 0:
        return ct.deep_copy() if addend is None else arithmetic.homo_add(addend, ct, ctx)
    n = int(ct.batch_size)
    if n <= MAX_BATCH:
        return _one(ct, offset, ctx, addend)
    parts = [_one(_slice(ct, s, min(n, s + MAX_BATCH)), offset, ctx,
                  None if addend is None else _slice(addend, s, min(n, s + MAX_BATCH)))
             for s in range(0, n, MAX_BATCH)]
    cv = [torch.cat([p.cv[i] for p in parts], 0) for i in range(len(parts[0].cv))]
    return parts[0].cipher_like(cv, batch_size=n)


def clear():
    _plans.clear()
