"""Fused, batched ModDown for EasyFHE key switching (kernels/keyswitch.cu).

``install()`` routes EasyFHE's relinearization (``cv_keyswitch`` / ``homo_mul_relin``) and
every ``moddown_from_ext`` of a two-component ciphertext through ``moddown_pair``, which
brings both components of all batch items down in four launches and can fold in an addend.
The results are the same residues as EasyFHE's ``cv_moddown``. ``SYLPH_KS=0`` disables it.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

import easyfhe as torch
from easyfhe.fhe.ciphertext import CipherState
from easyfhe.fhe.ops import alignment as _alignment
from easyfhe.fhe.ops import arithmetic as _arith
from easyfhe.fhe.ops import kernels as _kernels
from easyfhe.fhe.ops import rotation as _rotation
from easyfhe.fhe.ops import validation as _validation

from . import cudalib

_lib = None
_installed = False


def _load():
    global _lib
    if _lib is None:
        lib = cudalib.load("keyswitch")
        p, i, ll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
        lib.sylph_moddown_pair.argtypes = [p, p, ll, p, p, ll, p, p, ll, i, i, i, i, p] + [p] * 12 + [p]
        lib.sylph_moddown_pair.restype = i
        lib.sylph_rotate_pair.argtypes = ([p, ll, i, p, p, i, i, p, p, ll, p, ll, i, i, p, p, ll, p, p, ll, i, i, i,
                                           i, p] + [p] * 12 + [p])
        lib.sylph_rotate_pair.restype = i
        lib.sylph_fast_rotate_ext.argtypes = [p, p, p, p, p, p, i, i, i, i, i, p, p, p, p, p]
        lib.sylph_fast_rotate_ext.restype = i
        lib.sylph_hmul_raw.argtypes = [p, p, p, p, p, i, i, ll, ll, p, p, p, p]
        lib.sylph_hmul_raw.restype = i
        u = ctypes.c_ulonglong
        lib.sylph_hmul_relin_rescale.argtypes = ([p, i, i, i, p, p, i, i, p, p, p, p, p, p, p, i, i, u, p, p, p, p,
                                                  p, i, i, i, i, ll, p] + [p] * 12 + [p, i, p, p, p, p, p, ll, ll])
        lib.sylph_hmul_relin_rescale.restype = i
        lib.sylph_modup.argtypes = [p, ll, p, ll, i, i, i, i, i, p, p, ll, p, p, ll] + [p] * 7 + [p]
        lib.sylph_modup.restype = i
        lib.sylph_modup_p1.argtypes = lib.sylph_modup.argtypes
        lib.sylph_modup_p1.restype = i
        lib.sylph_modup_prod.argtypes = [p, ll, p, ll, p, ll, i, i, i, i, i, p, p, ll, p, p, ll] + [p] * 7 + [p, i]
        lib.sylph_modup_prod.restype = i
        lib.sylph_rescale.argtypes = [p, p, ll, p, p, ll, i, i, ctypes.c_ulonglong] + [p] * 11 + [p]
        lib.sylph_rescale.restype = i
        lib.sylph_scale_pmodq.argtypes = [p, p, ll, i, i, p, p, p, p, p]
        lib.sylph_scale_pmodq.restype = i
        _lib = lib
    return _lib


def _tables(ctx):
    return (ctx.primes.data_ptr(), ctx.barret_ratio.data_ptr(), ctx.barret_k.data_ptr(),
            ctx.power_of_roots.data_ptr(), ctx.power_of_roots_shoup.data_ptr(),
            ctx.inverse_power_of_roots_div_two.data_ptr(), ctx.inverse_scaled_power_of_roots_div_two.data_ptr(),
            ctx.hat_inverse_vec_moddown.data_ptr(), ctx.hat_inverse_vec_shoup_moddown.data_ptr(),
            ctx.prod_q_i_mod_q_j_moddown.data_ptr(), ctx.prod_inv_moddown.data_ptr(),
            ctx.prod_inv_shoup_moddown.data_ptr())


def _layout(t, limbs, n):
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.dim() != 3 or t.size(1) < limbs or t.size(2) != n or t.stride(2) != 1 or t.stride(1) != n:
        raise ValueError(f"unexpected component layout {tuple(t.shape)} / {tuple(t.stride())}")
    return t


def moddown_pair(x0, x1, limbs, ctx, add0=None, add1=None):
    """(ModDown(x0), ModDown(x1)) [+ (add0, add1)] for extended components [B, limbs + K, N]."""
    n, k = int(ctx.N), int(ctx.K)
    x0, x1 = _layout(x0, limbs + k, n), _layout(x1, limbs + k, n)
    batch = int(x0.size(0))
    if x1.size(0) != batch or x1.stride(0) != x0.stride(0):
        raise ValueError("components must share batch layout")
    out = torch.empty((2, batch, limbs, n), dtype=torch.uint64, device=x0.device)
    if add0 is not None:
        add0, add1 = _layout(add0, limbs, n), _layout(add1, limbs, n)
        if add0.size(0) != batch or add1.size(0) != batch or add0.stride(0) != add1.stride(0):
            raise ValueError("addend must match the batch layout")
        a0, a1, astride = add0.data_ptr(), add1.data_ptr(), int(add0.stride(0))
    else:
        a0 = a1 = None
        astride = 0
    ws = torch.empty((2 * batch * (k + limbs) * n,), dtype=torch.uint64, device=x0.device)
    stream = torch.cuda.current_stream().cuda_stream
    cudalib.check(_load().sylph_moddown_pair(
        x0.data_ptr(), x1.data_ptr(), int(x0.stride(0)), out[0].data_ptr(), out[1].data_ptr(), limbs * n,
        a0, a1, astride, batch, int(limbs), k, int(ctx.L), ws.data_ptr(), *_tables(ctx), stream), "moddown_pair")
    return out[0], out[1]


# SYLPH_P1=1: ModUp stops after NTT phase 1 (3 launches instead of 4) and the key-switch
# consumers (rotate_pair, hmul_relin_rescale) finish the NTT in shared memory.
def P1():
    return os.environ.get("SYLPH_P1", "1") == "1"


def NORAW():
    """hmul_relin_rescale without the stored tensor product: the ModUp reads c1 * d1 and the
    finishing kernels compute raw[0], raw[1] from the factors (same reductions, same residues)."""
    return os.environ.get("SYLPH_NORAW", "1") == "1"


def modup(x, limbs, ctx):
    """EasyFHE ``cv_modup``: digits [B, beta * (limbs + K), N] of x [B, >= limbs, N] (NTT domain)."""
    return _modup(x, limbs, ctx, False)


def modup_p1(x, limbs, ctx):
    """Phase-1 digits for rotate_pair(..., own=x, p1=True): same layout as ``modup`` but the
    limbs a digit does not own hold NTT phase 1 outputs and the own limbs are left unwritten."""
    return _modup(x, limbs, ctx, True)


def modup_prod(x, sx, y, sy, batch, limbs, ctx, p1):
    """``modup`` (or ``modup_p1``) of x * y (mod q_j) given the two factors' item strides."""
    return _modup(x, limbs, ctx, p1, y=y, sx=sx, sy=sy, batch=batch)


def _modup(x, limbs, ctx, p1, y=None, sx=None, sy=0, batch=None):
    n, k, L, alpha = int(ctx.N), int(ctx.K), int(ctx.L), int(ctx.alpha)
    limbs = int(limbs)
    if y is None:
        x = _layout(x, limbs, n)
        batch, sx = int(x.size(0)), int(x.stride(0))
    beta = (limbs + alpha - 1) // alpha
    out = torch.empty((batch, beta * (limbs + k), n), dtype=torch.uint64, device=x.device)
    ws = torch.empty((batch * limbs * n,), dtype=torch.uint64, device=x.device)
    tbl = ctx.prod_q_i_mod_q_j_modup[limbs - 1]
    hat, hat_s = ctx.hat_inverse_vec_modup, ctx.hat_inverse_vec_shoup_modup
    lib = _load()
    tail = (batch, limbs, alpha, k, L, ws.data_ptr(), tbl.data_ptr(), int(tbl.stride(0)), hat.data_ptr(),
            hat_s.data_ptr(), int(hat.stride(0)), ctx.primes.data_ptr(), ctx.barret_ratio.data_ptr(),
            ctx.barret_k.data_ptr(), ctx.power_of_roots.data_ptr(), ctx.power_of_roots_shoup.data_ptr(),
            ctx.inverse_power_of_roots_div_two.data_ptr(), ctx.inverse_scaled_power_of_roots_div_two.data_ptr(),
            torch.cuda.current_stream().cuda_stream)
    if y is None:
        cudalib.check((lib.sylph_modup_p1 if p1 else lib.sylph_modup)(
            x.data_ptr(), sx, out.data_ptr(), int(out.stride(0)), *tail), "modup")
    else:
        cudalib.check(lib.sylph_modup_prod(x.data_ptr(), sx, y.data_ptr(), sy, out.data_ptr(), int(out.stride(0)),
                                           *tail, int(p1)), "modup_prod")
    return out


def rescale(comps, limbs, ctx):
    """EasyFHE ``cv_rescale_one_level`` of one or two components [B, >= limbs, N] -> [B, limbs - 1, N]."""
    n, L = int(ctx.N), int(ctx.L)
    limbs = int(limbs)
    comps = [_layout(c, limbs, n) for c in comps]
    batch = int(comps[0].size(0))
    if any(c.size(0) != batch or c.stride(0) != comps[0].stride(0) for c in comps):
        raise ValueError("rescale: components must share the batch layout")
    l = limbs - 1
    out = torch.empty((len(comps), batch, l, n), dtype=torch.uint64, device=comps[0].device)
    ws = torch.empty((len(comps) * batch * n,), dtype=torch.uint64, device=comps[0].device)
    nprimes = int(ctx.primes.numel())
    sw, fin = (L - limbs) * (L - 1), l * L
    in1 = comps[1].data_ptr() if len(comps) == 2 else None
    out1 = out[1].data_ptr() if len(comps) == 2 else None
    cudalib.check(_load().sylph_rescale(
        comps[0].data_ptr(), in1, int(comps[0].stride(0)), out[0].data_ptr(), out1, l * n, batch, limbs,
        int(ctx.primes_list[l]) >> 1, ws.data_ptr(), _ptr_at(ctx.switch_modulus_map, l * nprimes),
        _ptr_at(ctx.qlql_inv_mod_ql_div_ql_mod_q, sw), _ptr_at(ctx.qlql_inv_mod_ql_div_ql_mod_q_shoup, sw),
        _ptr_at(ctx.q_inv_mod_q, fin), _ptr_at(ctx.q_inv_mod_q_shoup, fin), ctx.primes.data_ptr(),
        ctx.power_of_roots.data_ptr(), ctx.power_of_roots_shoup.data_ptr(),
        ctx.inverse_power_of_roots_div_two.data_ptr(), ctx.inverse_scaled_power_of_roots_div_two.data_ptr(),
        torch.cuda.current_stream().cuda_stream), "rescale")
    return [out[i] for i in range(len(comps))]


def scale_pmodq(c0, limbs, ctx):
    """c0 [B, >= limbs, N] -> c0 * P (mod q) [B, limbs, N]: the c0 term of extended rotations."""
    n = int(ctx.N)
    c0 = _layout(c0, limbs, n)
    out = torch.empty((int(c0.size(0)), limbs, n), dtype=torch.uint64, device=c0.device)
    cudalib.check(_load().sylph_scale_pmodq(
        out.data_ptr(), c0.data_ptr(), int(c0.stride(0)), int(c0.size(0)), int(limbs), ctx.primes.data_ptr(),
        ctx.barret_ratio.data_ptr(), ctx.barret_k.data_ptr(), ctx.PModq.data_ptr(),
        torch.cuda.current_stream().cuda_stream), "scale_pmodq")
    return out


def _rescale_enabled():
    return os.environ.get("SYLPH_RESCALE", "1") != "0"


def _cv_rescale_one_level(input, cur_limbs, l, context):
    if not _rescale_enabled() or input.dim() != 3 or not input.is_cuda:
        return _orig_cv_rescale_one_level(input, cur_limbs, l, context)
    return rescale([input], cur_limbs, context)[0]


def _rescale_one_level(cipher, context):
    cv = cipher.cv
    if not _rescale_enabled() or cipher.is_ext or len(cv) != 2 or cv[0].dim() != 3 or cv[1].dim() != 3\
            or not cv[0].is_cuda or cv[0].size(0) != cv[1].size(0) or cv[0].stride(0) != cv[1].stride(0):
        return _orig_rescale_one_level(cipher, context)
    limbs = int(cipher.state.cur_limbs)
    if limbs <= 1 or cipher.state.scale_degree <= 1:
        return _orig_rescale_one_level(cipher, context)
    res_cv = rescale(cv, limbs, context)
    divisor = context.rescale_divisor_at(limbs - 1)
    return cipher.cipher_like(res_cv, state=CipherState(limbs - 1, cipher.state.scale_degree - 1,
                                                        cipher.state.scaling_factor / divisor))


def _cv_modup(x, curr_limbs, context):
    if os.environ.get("SYLPH_MODUP", "1") == "0" or x.dim() != 3 or int(context.alpha) > 16:
        return _orig_cv_modup(x, curr_limbs, context)
    return modup(x, curr_limbs, context)


_rot_plans = {}
_fast_tables = {}


def clear_plans():
    """Drop cached rotation-key plans. The next rotation reloads the key it needs."""
    _rot_plans.clear()
    _fast_tables.clear()


def _rot_plan(ctx, offset, limbs):
    key = (id(ctx), offset, limbs)
    p = _rot_plans.get(key)
    if p is None:
        bx, ax, sms = _rotation._rotation_key_and_start(offset, ctx, limbs)
        amap = _rotation._precompute_auto_maps((offset,), ctx)[0].contiguous()
        if amap.dtype != torch.int32 or bx.dim() != 3 or not (bx.is_contiguous() and ax.is_contiguous())\
                or bx.size(1) != int(sms) + int(ctx.K):
            raise ValueError("unexpected rotation key / map layout")
        p = _rot_plans[key] = (bx, ax, int(sms), amap)
    return p


def rotate_pair(c0, digits, offset, limbs, ctx, add0=None, add1=None, own=None):
    """rot(ct, offset) [+ (add0, add1)] for a batch [B, limbs, N] sharing one offset, from
    c0 and the ModUp digits of c1 [B, beta * (limbs + K), N] (EasyFHE's ``_batch_same_rotate``
    in four launches). ``own`` = c1 itself: ``digits`` are phase-1 digits (``modup_p1``)."""
    n, k = int(ctx.N), int(ctx.K)
    c0 = _layout(c0, limbs, n)
    batch = int(c0.size(0))
    if own is not None:
        own = _layout(own, limbs, n)
        if own.size(0) != batch:
            raise ValueError("own component must match the batch")
        own_ptr, own_stride = own.data_ptr(), int(own.stride(0))
    else:
        own_ptr, own_stride = None, 0
    beta = (limbs + int(ctx.alpha) - 1) // int(ctx.alpha)
    if digits.dim() != 3 or digits.size(0) != batch or not digits.is_contiguous()\
            or digits.size(1) != beta * (limbs + k) or digits.size(2) != n:
        raise ValueError(f"unexpected digits layout {tuple(digits.shape)}")
    bx, ax, sms, amap = _rot_plan(ctx, int(offset), int(limbs))
    if bx.size(0) < beta or bx.size(1) < sms + k:
        raise ValueError("rotation key too short")
    out = torch.empty((2, batch, limbs, n), dtype=torch.uint64, device=c0.device)
    if add0 is not None:
        add0, add1 = _layout(add0, limbs, n), _layout(add1, limbs, n)
        if add0.size(0) != batch or add1.size(0) != batch or add0.stride(0) != add1.stride(0):
            raise ValueError("addend must match the batch layout")
        a0, a1, astride = add0.data_ptr(), add1.data_ptr(), int(add0.stride(0))
    else:
        a0 = a1 = None
        astride = 0
    ws = torch.empty((2 * batch * (k + limbs) * n,), dtype=torch.uint64, device=c0.device)
    stream = torch.cuda.current_stream().cuda_stream
    cudalib.check(_load().sylph_rotate_pair(
        digits.data_ptr(), int(digits.stride(0)), beta, bx.data_ptr(), ax.data_ptr(), int(bx.size(1)), sms,
        amap.data_ptr(), c0.data_ptr(), int(c0.stride(0)), own_ptr, own_stride, int(ctx.alpha),
        int(own is not None), out[0].data_ptr(), out[1].data_ptr(), limbs * n,
        a0, a1, astride, batch, int(limbs), k, int(ctx.L), ws.data_ptr(), *_tables(ctx), stream), "rotate_pair")
    return out[0], out[1]


def _fast_table(ctx, offsets, limbs):
    key = (id(ctx), offsets, limbs)
    t = _fast_tables.get(key)
    if t is None:
        rows, keep = [], []
        for o in offsets:
            if o == 0:
                rows.append((0, 0, 0, 0))
                continue
            bx, ax, sms, amap = _rot_plan(ctx, o, limbs)
            keep.append((bx, ax, amap))
            rows.append((bx.data_ptr(), ax.data_ptr(), sms, amap.data_ptr()))
        host = np.array(rows, dtype=np.uint64).view(np.int64)
        t = _fast_tables[key] = (torch.from_numpy(host).to(ctx.device), keep)
    return t[0]


def fast_rotate_ext(cipher, offsets, context):
    """EasyFHE ``fast_rotate(cipher, offsets, output_ext=True)`` in one fused launch after ModUp."""
    offsets = tuple(int(o) for o in offsets)
    limbs, n, k = int(cipher.state.cur_limbs), int(context.N), int(context.K)
    c0, c1 = cipher.cv[0], cipher.cv[1]
    c0 = c0[0] if c0.dim() == 3 else c0
    c1 = c1[0] if c1.dim() == 3 else c1
    if c0.size(0) < limbs or c0.stride(0) != n or c1.stride(0) != n or c0.stride(1) != 1 or c1.stride(1) != 1:
        raise ValueError("fast_rotate_ext: unexpected component layout")
    digits = _kernels.cv_modup(cipher.cv[1], limbs, context)
    beta = (limbs + int(context.alpha) - 1) // int(context.alpha)
    if not digits.is_contiguous() or digits.numel() != beta * (limbs + k) * n:
        raise ValueError("fast_rotate_ext: unexpected digits layout")
    table = _fast_table(context, offsets, limbs)
    batch = len(offsets)
    out_bx = torch.empty((batch, limbs + k, n), dtype=torch.uint64, device=c0.device)
    out_ax = torch.empty((batch, limbs + k, n), dtype=torch.uint64, device=c0.device)
    cudalib.check(_load().sylph_fast_rotate_ext(
        out_bx.data_ptr(), out_ax.data_ptr(), digits.data_ptr(), table.data_ptr(), c0.data_ptr(), c1.data_ptr(),
        batch, limbs, k, int(context.L), beta, context.primes.data_ptr(), context.barret_ratio.data_ptr(),
        context.barret_k.data_ptr(), context.PModq.data_ptr(), torch.cuda.current_stream().cuda_stream),
        "fast_rotate_ext")
    return cipher.cipher_like([out_bx, out_ax], is_ext=True, batch_size=batch)


def _fast_rotate(cipher, offsets, context, *, output_ext=False):
    if not output_ext or len(cipher.cv) != 2 or int(cipher.batch_size) != 1 or cipher.is_ext:
        return _orig_fast_rotate(cipher, offsets, context, output_ext=output_ext)
    if isinstance(offsets, (int, np.integer)):
        offsets = (int(offsets),)
    offsets = tuple(int(o) for o in offsets)
    if not offsets or all(o == 0 for o in offsets):
        return _orig_fast_rotate(cipher, offsets, context, output_ext=output_ext)
    return fast_rotate_ext(cipher, offsets, context)


_idmaps = {}


def _ptr_at(t, offset):
    """Device address of element `offset` of the flattened table t."""
    if not t.is_contiguous():
        raise ValueError("table must be contiguous")
    if offset < 0 or offset >= t.numel():
        raise ValueError("table offset out of range")
    return t.data_ptr() + 8 * int(offset)


def _plane(t, limbs, n):
    t = t[0] if t.dim() == 3 else t
    if t.dim() != 2 or t.size(0) < limbs or t.size(1) != n or not t.is_contiguous():
        raise ValueError(f"unexpected plane layout {tuple(t.shape)}")
    return t


def _planes(t, limbs, n, batch):
    """[batch, >= limbs, n] contiguous (or [limbs, n] for batch 1) -> (tensor, item stride)."""
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.dim() != 3 or t.size(0) != batch or t.size(1) < limbs or t.size(2) != n or not t.is_contiguous():
        raise ValueError(f"unexpected batch plane layout {tuple(t.shape)} for batch {batch}")
    return t, int(t.size(1)) * n


def hmul_relin_rescale(c0, c1, d0, d1, curr_limbs, special_mod_start, swk_bx, swk_ax, context, *,
                       apply_double=False, post_op=0, post_c0=None, post_c1=None, post_scalar=None,
                       profile_reps=0, times=None):
    """EasyFHE ``cv_hmul_relin_rescale`` (tensor product, relinearization, rescale and one
    post op) in nine launches with the ModDown, the rescale switch and the post op fused."""
    limbs, n, k, L = int(curr_limbs), int(context.N), int(context.K), int(context.L)
    sms = int(special_mod_start)
    if limbs < 2 or swk_bx.dim() != 3 or swk_bx.size(1) != sms + k or swk_ax.shape != swk_bx.shape\
            or not (swk_bx.is_contiguous() and swk_ax.is_contiguous()):
        return _orig_hmul_relin_rescale(c0, c1, d0, d1, curr_limbs, special_mod_start, swk_bx, swk_ax, context,
                                        apply_double=apply_double, post_op=post_op, post_c0=post_c0,
                                        post_c1=post_c1, post_scalar=post_scalar)
    batch = int(c0.size(0)) if c0.dim() == 3 else 1
    (c0, sc), (c1, sc1), (d0, sd), (d1, sd1) = (_planes(t, limbs, n, batch) for t in (c0, c1, d0, d1))
    if sc != sc1 or sd != sd1:
        raise ValueError("hmul_relin_rescale: component strides differ")
    lib = _load()
    stream = torch.cuda.current_stream().cuda_stream
    dev = c0.device
    if NORAW():
        raw = None
        digits = modup_prod(c1, sc, d1, sd, batch, limbs, context, P1())
    else:
        raw = torch.empty((3, batch, limbs, n), dtype=torch.uint64, device=dev)
        cudalib.check(lib.sylph_hmul_raw(raw.data_ptr(), c0.data_ptr(), c1.data_ptr(), d0.data_ptr(), d1.data_ptr(),
                                         limbs, batch, sc, sd, context.primes.data_ptr(),
                                         context.barret_ratio.data_ptr(), context.barret_k.data_ptr(), stream),
                      "hmul_raw")
        digits = modup_p1(raw[2], limbs, context) if P1() else _kernels.cv_modup(raw[2], limbs, context)
    beta = (limbs + int(context.alpha) - 1) // int(context.alpha)
    if not digits.is_contiguous() or digits.numel() != batch * beta * (limbs + k) * n or swk_bx.size(0) < beta:
        raise ValueError("hmul_relin_rescale: unexpected digits / key layout")
    l = limbs - 1
    p0 = p1 = sc_ptr = None
    pstride = 0
    post_op = int(post_op)
    keep = []
    if post_op in (1, 2, 4):
        a0, pstride = _planes(post_c0, l, n, batch)
        keep.append(a0)
        p0 = a0.data_ptr()
        if post_op != 4:
            a1, ps1 = _planes(post_c1, l, n, batch)
            if ps1 != pstride:
                raise ValueError("hmul_relin_rescale: post strides differ")
            keep.append(a1)
            p1 = a1.data_ptr()
    elif post_op == 3:
        if post_scalar.numel() != l or not post_scalar.is_contiguous():
            raise ValueError("hmul_relin_rescale: post scalar shape mismatch")
        sc_ptr = post_scalar.data_ptr()
    elif post_op != 0:
        raise ValueError(f"hmul_relin_rescale: unknown post op {post_op}")
    imap = _idmaps.get(id(context))
    if imap is None:
        imap = _idmaps[id(context)] = torch.arange(n, dtype=torch.int32, device=dev)
    nprimes = int(context.primes.numel())
    sw = (L - limbs) * (L - 1)
    fin = l * L
    out = torch.empty((2, batch, l, n), dtype=torch.uint64, device=dev)
    ws = torch.empty((2 * batch * (k + limbs + 1) * n,), dtype=torch.uint64, device=dev)
    t = _tables(context)
    cudalib.check(lib.sylph_hmul_relin_rescale(
        digits.data_ptr(), beta, int(context.alpha), int(P1()), swk_bx.data_ptr(), swk_ax.data_ptr(), sms + k, sms,
        imap.data_ptr(),
        None if raw is None else raw.data_ptr(), out[0].data_ptr(), out[1].data_ptr(), p0, p1, sc_ptr, post_op,
        int(bool(apply_double)),
        int(context.primes_list[limbs - 1]) >> 1, _ptr_at(context.switch_modulus_map, l * nprimes),
        _ptr_at(context.qlql_inv_mod_ql_div_ql_mod_q, sw), _ptr_at(context.qlql_inv_mod_ql_div_ql_mod_q_shoup, sw),
        _ptr_at(context.q_inv_mod_q, fin), _ptr_at(context.q_inv_mod_q_shoup, fin),
        limbs, k, L, batch, pstride, ws.data_ptr(), *t, stream, int(profile_reps),
        None if times is None else ctypes.cast(times, ctypes.c_void_p),
        c0.data_ptr(), c1.data_ptr(), d0.data_ptr(), d1.data_ptr(), sc, sd), "hmul_relin_rescale")
    return out[0], out[1]


def enabled():
    return _installed


def _inner(modup_res, cur_limbs, special_mod_start, swk_bx, swk_ax, context):
    batch = int(modup_res.shape[0])
    if batch == 1:
        return _kernels.cv_innerproduct_broadcast(modup_res, cur_limbs, [special_mod_start], [swk_bx], [swk_ax],
                                                  context)
    return _kernels.cv_innerproduct_pairwise(modup_res, cur_limbs, [special_mod_start] * batch, [swk_bx] * batch,
                                             [swk_ax] * batch, context)


def keyswitch(input, cur_limbs, special_mod_start, swk_bx, swk_ax, context, add=None):
    """EasyFHE ``cv_keyswitch`` with the fused ModDown (optionally + add)."""
    modup_res = _kernels.cv_modup(input, curr_limbs=cur_limbs, context=context)
    kb, ka = _inner(modup_res, cur_limbs, special_mod_start, swk_bx, swk_ax, context)
    if add is None:
        return moddown_pair(kb, ka, cur_limbs, context)
    return moddown_pair(kb, ka, cur_limbs, context, add[0], add[1])


def _mul_relin(in0, in1, context):
    _validation.validate_binary_cipher_op("homo_mul_relin", in0, in1, require_ext=False,
                                          require_same_metadata=("slots",))
    in0, in1 = _arith._align_for_mul(in0, in1, context)
    product = _arith._cipher_mul(in0, in1, context)
    limbs = product.state.cur_limbs
    d0, d1 = product.cv[0], product.cv[1]
    if d0.dim() == 3 and (d0.size(1) != limbs or d1.size(1) != limbs):
        d0, d1 = d0[:, :limbs], d1[:, :limbs]
    c0, c1 = keyswitch(product.cv[2], limbs, context.L, context.mult_swk_bx, context.mult_swk_ax, context,
                       add=(d0, d1))
    if product.cv[0].dim() == 2:
        c0, c1 = c0[0], c1[0]
    return product.cipher_like([c0, c1])


def _moddown_from_ext(cipher, context):
    _validation.validate_cipher_op("moddown_from_ext", cipher, require_ext=True)
    if len(cipher.cv) != 2:
        return _orig_moddown_from_ext(cipher, context)
    c0, c1 = moddown_pair(cipher.cv[0], cipher.cv[1], cipher.state.cur_limbs, context)
    if cipher.cv[0].dim() == 2:
        c0, c1 = c0[0], c1[0]
    return cipher.cipher_like([c0, c1], is_ext=False)


_orig_moddown_from_ext = _rotation.moddown_from_ext
_orig_fast_rotate = _rotation.fast_rotate
_orig_hmul_relin_rescale = _kernels.cv_hmul_relin_rescale
_orig_cv_modup = _kernels.cv_modup
_orig_cv_rescale_one_level = _kernels.cv_rescale_one_level
_orig_rescale_one_level = _alignment._rescale_one_level


def install():
    global _installed
    if _installed or os.environ.get("SYLPH_KS", "1") == "0":
        return
    _installed = True
    _kernels.cv_keyswitch = keyswitch
    _arith.homo_mul_relin = _mul_relin
    _rotation.moddown_from_ext = _moddown_from_ext
    _kernels.cv_modup = _cv_modup
    _kernels.cv_rescale_one_level = _cv_rescale_one_level
    _alignment._rescale_one_level = _rescale_one_level
    if os.environ.get("SYLPH_HMUL", "1") != "0":
        _kernels.cv_hmul_relin_rescale = hmul_relin_rescale
    if os.environ.get("SYLPH_FASTROT", "1") != "0":
        _rotation.fast_rotate = _fast_rotate
