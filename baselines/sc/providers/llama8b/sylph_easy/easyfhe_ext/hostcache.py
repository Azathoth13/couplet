"""Device caches for the small constant tables EasyFHE uploads on every call.

EasyFHE builds key-switch start indices, rotation product indices and encoded scalars as
NumPy arrays and copies them to the GPU per operation. A pageable host->device copy waits
for the stream, so each one drains the GPU queue (about 5.6k per decoder layer). The tables
are pure functions of their arguments, so they are uploaded once and reused; the computed
ciphertexts are unchanged. ``SYLPH_HOSTCACHE=0`` disables the caches.
"""
from __future__ import annotations

import os
import weakref

import numpy as np
import easyfhe as torch
from easyfhe.fhe import constants as _constants
from easyfhe.fhe.ops import encoding as _encoding
from easyfhe.fhe.ops import kernels as _kernels
from easyfhe.fhe.ops import rotation as _rotation

_LIMIT = 1 << 16
_installed = False
_key_caches = []


def clear_device_keys():
    """Drop device copies of rotation keys.

    Masters stay on the host. Sylph also memoizes the loaded keys in its rotation
    plans; those copies are what fill a 24 GB card across a layer.
    """
    for cache in _key_caches:
        cache.clear()
    from . import keyswitch, rotate
    keyswitch.clear_plans()
    rotate.clear_plans()


def _bounded(cache, key, make):
    hit = cache.get(key)
    if hit is None:
        if len(cache) >= _LIMIT:
            cache.clear()
        hit = cache[key] = make()
    return hit


_STAGING = {}
_RING = 8


def upload(array, device="cuda"):
    """Host array -> device without draining the stream.

    A pageable copy waits for every queued kernel; pinning a fresh buffer per call costs about
    0.3 ms. The array is staged through a small ring of pinned buffers per shape, each guarded
    by the event of its last copy, and copied asynchronously."""
    src = torch.from_numpy(np.ascontiguousarray(array))
    ring = _STAGING.setdefault((tuple(src.shape), src.dtype), [])
    slot = next((r for r in ring if r[1] is None or r[1].query()), None)
    if slot is None:
        if len(ring) < _RING:
            slot = [torch.empty(src.shape, dtype=src.dtype).pin_memory(), None]
            ring.append(slot)
        else:
            slot = ring[0]
            slot[1].synchronize()
            ring.append(ring.pop(0))
    slot[0].copy_(src)
    out = slot[0].to(device, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    slot[1] = ev
    return out


def _starts_tensor(cache, starts, device):
    if torch.is_tensor(starts):
        return starts
    if isinstance(starts, int):
        starts = [starts]
    key = (tuple(int(s) for s in starts), str(device))
    return _bounded(cache, key, lambda: upload(np.array(key[0], dtype=np.int64), device))


def install():
    global _installed
    if _installed or os.environ.get("SYLPH_HOSTCACHE", "1") == "0":
        return
    _installed = True
    starts_cache, plan_cache, keys_cache, scalar_cache, table_cache = {}, {}, {}, {}, {}
    _key_caches.append(keys_cache)

    def wrap_inner(f):
        def g(x, curr_limbs, special_mod_starts, swk_bxs, swk_axs, context):
            return f(x, curr_limbs, _starts_tensor(starts_cache, special_mod_starts, context.device),
                     swk_bxs, swk_axs, context)
        return g

    _kernels.cv_innerproduct_broadcast = wrap_inner(_kernels.cv_innerproduct_broadcast)
    _kernels.cv_innerproduct_pairwise = wrap_inner(_kernels.cv_innerproduct_pairwise)

    product_plan = _rotation._batch_rotation_product_plan
    keys_and_starts = _rotation._batch_rotation_keys_and_starts

    def cached_plan(offsets, context):
        key = (tuple(int(o) for o in offsets), id(context))
        return _bounded(plan_cache, key, lambda: product_plan(offsets, context))

    def cached_keys(offsets, context, cur_limbs):
        # The key tensors are held through weak references: the entry is a memo of the
        # context's own rotation-key cache, not a second owner.  When the context evicts
        # or reloads a key (Prefill -> Decode key swap, dominated-version eviction) the
        # weak references die and the lookup falls through to ``keys_and_starts``;
        # with strong references every evicted key would stay resident on the device.
        key = (tuple(int(o) for o in offsets), id(context), int(cur_limbs))
        hit = keys_cache.get(key)
        if hit is not None:
            bxs = [r() for r in hit[0]]
            axs = [r() for r in hit[1]]
            if all(t is not None for t in bxs) and all(t is not None for t in axs):
                return bxs, axs, hit[2]
        swk_bxs, swk_axs, starts = keys_and_starts(offsets, context, cur_limbs)
        if len(keys_cache) >= _LIMIT:
            keys_cache.clear()
        try:
            keys_cache[key] = ([weakref.ref(t) for t in swk_bxs], [weakref.ref(t) for t in swk_axs], starts)
        except TypeError:
            # non-weakrefable key objects (CPU / list keys): memo the values themselves
            keys_cache[key] = ([(lambda t=t: t) for t in swk_bxs], [(lambda t=t: t) for t in swk_axs], starts)
        return list(swk_bxs), list(swk_axs), starts

    _rotation._batch_rotation_product_plan = cached_plan
    _rotation._batch_rotation_keys_and_starts = cached_keys

    encode_values = _constants.ConstantBundle._encode_scalar_values

    def encode_values_async(self, values, cur_limbs, scale_degree, context, mode, absolute, scaling_factor):
        # ConstantBundle._encode_scalar_values with the pageable upload replaced by ``upload``
        rows = []
        for value in values:
            value = abs(value) if absolute else value
            if mode == "scaled":
                rows.append(_constants._encode_scaled_scalar_value(value, cur_limbs, scale_degree, context,
                                                                   scaling_factor))
            else:
                rows.append([int(value) % int(context.moduliQ_scalar[level]) for level in range(cur_limbs)])
        return upload(np.asarray(rows, dtype=np.uint64), context.device)

    def cached_values(self, values, cur_limbs, scale_degree, context, mode, absolute, scaling_factor):
        try:
            vals = tuple(float(v) if mode == "scaled" else int(v) for v in values)
        except (TypeError, ValueError):
            return encode_values(self, values, cur_limbs, scale_degree, context, mode, absolute, scaling_factor)
        sf = None if scaling_factor is None else float(scaling_factor)
        key = (vals, int(cur_limbs), int(scale_degree), id(context), mode, bool(absolute), sf)
        return _bounded(scalar_cache, key, lambda: encode_values_async(self, values, cur_limbs, scale_degree,
                                                                       context, mode, absolute, scaling_factor))

    _constants.ConstantBundle._encode_scalar_values = cached_values

    table_numpy = _encoding._table_numpy

    def cached_table(value):
        # encode_stage1 copies the context's static encode tables to the host on every call
        if not torch.is_tensor(value):
            return table_numpy(value)
        key = (value.data_ptr(), tuple(value.shape), str(value.dtype))
        return _bounded(table_cache, key, lambda: table_numpy(value))

    _encoding._table_numpy = cached_table
