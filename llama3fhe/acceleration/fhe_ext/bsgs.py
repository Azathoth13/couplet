"""Batched CoeffsToSlots / SlotsToCoeffs (EasyFHE ``coeffs_slots_conversion``) for several
ciphertexts at one state.

Each BSGS level of EasyFHE's ``ext_double_hoist`` strategy rotates the ciphertext to every
baby offset in the extended basis (ModUp once), multiplies by the level's diagonal
plaintexts into ``giant_count`` partial sums and folds the partial sums with the
double-hoisted giant rotations. ``bsgs_mac_ext`` (kernels/keyswitch.cu) computes the
partial sums of a whole batch in one launch without writing the rotated ciphertexts, and
reads every rotation key and plaintext once per four ciphertexts. The giant step is
EasyFHE's ``giant_rotate_sum``. The residues equal EasyFHE's per-ciphertext transform.
``SYLPH_BSGS=0`` disables it.
"""
from __future__ import annotations

import ctypes
import os

import easyfhe as torch
from easyfhe.fhe.ciphertext import CipherState
from easyfhe.fhe.ops import alignment, rotation
from easyfhe.fhe.ops import kernels as F
from easyfhe.bs.openfhe.runtime import transforms as _transforms

from . import cudalib, keyswitch

_bound = False


def enabled():
    return os.environ.get("SYLPH_BSGS", "1") != "0"


def _lib():
    global _bound
    lib = keyswitch._load()
    if not _bound:
        p, i, ll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
        lib.sylph_bsgs_mac_ext.argtypes = [p, p, ll, p, i, p, p, ll, p, p, ll, ll, i, i, i, i, i, i, i, p, p, p, p, p]
        lib.sylph_bsgs_mac_ext.restype = i
        lib.sylph_pts_shoup.argtypes = [p, p, ll, ll, i, i, i, i, p, p]
        lib.sylph_pts_shoup.restype = i
        lib.sylph_giant_sum_ext.argtypes = [p, p, p, p, p, ll, p, ll, p, i, i, i, i, i, i, p, p, p, p, p]
        lib.sylph_giant_sum_ext.restype = i
        _bound = True
    return lib


_shoup_cache = {}


def pts_shoup(pv, limbs, ctx):
    """floor(pv * 2^64 / p) of the plaintext rows pv [rows][>= limbs + K][N] for the first limbs + K
    limbs, cached per device tensor (the constants bundle keeps the plaintexts on the device)."""
    key = (pv.data_ptr(), tuple(pv.shape), int(limbs))
    hit = _shoup_cache.get(key)
    if hit is not None and hit[0] is pv:
        return hit[1]
    active = int(limbs) + int(ctx.K)
    out = torch.empty((pv.size(0), active, int(ctx.N)), dtype=torch.uint64, device=pv.device)
    cudalib.check(_lib().sylph_pts_shoup(
        out.data_ptr(), pv.data_ptr(), int(pv.stride(0)), int(out.stride(0)), int(pv.size(0)), active, int(limbs), int(ctx.L),
        ctx.primes.data_ptr(), torch.cuda.current_stream().cuda_stream), "pts_shoup")
    if len(_shoup_cache) >= 64:
        _shoup_cache.clear()
    _shoup_cache[key] = (pv, out)
    return out


def baby_partial_sums(ct, offsets, pts, groups, ctx):
    """[B] batch -> device tensor [2, groups, B, limbs + K, N]: item b, group g holds
    sum_i pts[g * len(offsets) + i] * rot_ext(ct_b, offsets[i])."""
    offsets = tuple(int(o) for o in offsets)
    limbs, n, k = int(ct.state.cur_limbs), int(ctx.N), int(ctx.K)
    batch, nb = int(ct.batch_size), len(offsets)
    c0, c1 = ct.cv
    if c0.dim() == 2:
        c0, c1 = c0.unsqueeze(0), c1.unsqueeze(0)
    for c in (c0, c1):
        if c.dim() != 3 or c.size(0) != batch or c.size(1) < limbs or c.size(2) != n\
                or c.stride(2) != 1 or c.stride(1) != n:
            raise ValueError(f"unexpected component layout {tuple(c.shape)}")
    if c1.stride(0) != c0.stride(0):
        raise ValueError("components must share batch layout")
    beta = (limbs + int(ctx.alpha) - 1) // int(ctx.alpha)
    digits = F.cv_modup(c1, limbs, ctx)
    if not digits.is_contiguous() or digits.numel() != batch * beta * (limbs + k) * n:
        raise ValueError(f"unexpected digits layout {tuple(digits.shape)}")
    pv = pts.cv[0]
    if pv.dim() != 3 or pv.size(0) != groups * nb or pv.size(1) < limbs + k or pv.stride(1) != n\
            or pv.stride(2) != 1:
        raise ValueError(f"unexpected plaintext layout {tuple(pv.shape)}")
    table = keyswitch._fast_table(ctx, offsets, limbs)
    res = torch.empty((2, groups, batch, limbs + k, n), dtype=torch.uint64, device=c0.device)
    tb = 1 if batch == 1 else 2 if batch == 2 else 4
    tb = int(os.environ.get('SYLPH_TB', tb))  # Optional batch-width override
    c0p = keyswitch.scale_pmodq(c0, limbs, ctx)
    psh = pts_shoup(pv, limbs, ctx)
    cudalib.check(_lib().sylph_bsgs_mac_ext(
        res.data_ptr(), digits.data_ptr(), beta * (limbs + k) * n, table.data_ptr(), nb, c0p.data_ptr(),
        c1.data_ptr(), int(c0.stride(0)), pv.data_ptr(), psh.data_ptr(), int(pv.stride(0)), int(psh.stride(0)),
        int(groups), batch, tb, limbs, k,
        int(ctx.L), beta, ctx.primes.data_ptr(), ctx.barret_ratio.data_ptr(), ctx.barret_k.data_ptr(),
        ctx.PModq.data_ptr(), torch.cuda.current_stream().cuda_stream), "bsgs_mac_ext")
    return res


def giant_sum(part, giant, groups, limbs, ctx):
    """EasyFHE giant_rotate_sum (ext_double_hoist) of every batch item of the partial sums
    ``part`` [2, groups, B, limbs + K, N] -> (c0, c1) [B, limbs, N]."""
    n, k = int(ctx.N), int(ctx.K)
    batch, active = int(part.size(2)), limbs + k
    beta = (limbs + int(ctx.alpha) - 1) // int(ctx.alpha)
    tail = [part[c, 1:].reshape((groups - 1) * batch, active, n) for c in range(2)]
    inner0, inner1 = keyswitch.moddown_pair(tail[0], tail[1], limbs, ctx)
    digits = F.cv_modup(inner1, limbs, ctx)
    if not digits.is_contiguous() or digits.numel() != (groups - 1) * batch * beta * active * n:
        raise ValueError(f"unexpected digits layout {tuple(digits.shape)}")
    if not inner0.is_contiguous():
        raise ValueError("unexpected ModDown output layout")
    offsets = tuple(rotation._double_hoist_giant_key_offset(i * giant, ctx) for i in range(1, groups))
    if any(o == 0 for o in offsets):
        raise ValueError("giant key offset 0")
    table = keyswitch._fast_table(ctx, offsets, limbs)
    out = torch.empty((2, batch, active, n), dtype=torch.uint64, device=part.device)
    cudalib.check(_lib().sylph_giant_sum_ext(
        out[0].data_ptr(), out[1].data_ptr(), part[0, 0].data_ptr(), part[1, 0].data_ptr(), digits.data_ptr(),
        beta * active * n, inner0.data_ptr(), limbs * n, table.data_ptr(), groups - 1, batch, limbs, k,
        int(ctx.L), beta, ctx.primes.data_ptr(), ctx.barret_ratio.data_ptr(), ctx.barret_k.data_ptr(),
        ctx.PModq.data_ptr(), torch.cuda.current_stream().cuda_stream), "giant_sum_ext")
    return keyswitch.moddown_pair(out[0], out[1], limbs, ctx)


def _item(ct, i):
    return ct.cipher_like([x[i] for x in ct.cv], batch_size=1)


def _stack(items):
    if len(items) == 1:
        return items[0]
    cv = [torch.stack([it.cv[c] if it.cv[c].dim() == 2 else it.cv[c][0] for it in items], 0)
          for c in range(len(items[0].cv))]
    return items[0].cipher_like(cv, batch_size=len(items))


def conversion(ct, transform_plan, constants, bootstrap_plan, ctx):
    """EasyFHE coeffs_slots_conversion of every item of the batch ``ct``."""
    if bootstrap_plan.strategy in ("normal_bsgs", "normal_giant") or len(ct.cv) != 2 or ct.is_ext:
        items = [_transforms.coeffs_slots_conversion(_item(ct, i), transform_plan, constants, bootstrap_plan, ctx)
                 for i in range(int(ct.batch_size))]
        return _stack(items)
    result = ct
    for pos, step in enumerate(transform_plan.steps):
        if pos != 0:
            result = alignment.normalize_scale(result, ctx)
        pts = constants.plaintext(step.plaintext_name, state=result.state, slots=step.plaintext_slots,
                                  context=ctx, is_ext=True)
        groups, giant = int(step.baby_step), int(step.giant_offset)
        part = baby_partial_sums(result, step.input_offsets, pts, groups, ctx)
        state = CipherState(result.state.cur_limbs, result.state.scale_degree + pts.state.scale_degree,
                            result.state.scaling_factor * pts.state.scaling_factor)
        limbs, batch = int(result.state.cur_limbs), int(result.batch_size)
        if groups > 1 and giant != 0:
            c0, c1 = giant_sum(part, giant, groups, limbs, ctx)
            result = result.cipher_like([c0 if batch > 1 else c0[0], c1 if batch > 1 else c1[0]], state=state,
                                        slots=pts.slots, is_ext=False, batch_size=batch)
            continue
        outs = []
        for b in range(batch):
            sums = result.cipher_like([part[0, :, b], part[1, :, b]], state=state, slots=pts.slots, is_ext=True,
                                      batch_size=groups)
            if groups == 1:
                out = rotation.moddown_from_ext(_item(sums, 0), ctx)
            else:
                out = rotation._sum_batch_items_without_rotation(sums, ctx)
            outs.append(out)
        result = _stack(outs)
    return result


def coeffs_to_slots(ct, ctx, constants, plan):
    return conversion(ct, plan.c2s_plan, constants, plan, ctx)


def slots_to_coeffs(ct, ctx, constants, plan):
    return conversion(ct, plan.s2c_plan, constants, plan, ctx)
