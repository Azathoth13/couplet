"""Batched EvalMod (EasyFHE ``eval_bootstrap_approx_mod`` on a cipher batch).

EasyFHE's ``homo_mul_relin_rescale_postop`` takes one ciphertext, so its bootstrap EvalMod runs
item by item. ``install()`` lets the postop accept a batch (``keyswitch.hmul_relin_rescale``
with a batch dimension; batch 1 keeps EasyFHE's own path) and ``eval_mod`` runs EasyFHE's
Chebyshev / Paterson-Stockmeyer / double-angle schedule once on the whole batch. The tail
table (EasyFHE's ``grouped_scalar_weighted_acc`` over the Chebyshev basis, one ciphertext at
a time) is one launch for the batch (kernels/evalmod.cu). Every item gets the same residues
as the one-at-a-time EvalMod. ``SYLPH_EVALMOD=0`` disables it.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np
import easyfhe as torch
from easyfhe.fhe.ciphertext import CipherState
from easyfhe.fhe.ops import alignment, arithmetic, layout
from easyfhe.bs.openfhe.runtime import approx as A

from . import cudalib, keyswitch
from .hostcache import upload

_lib = None


def _load():
    global _lib
    if _lib is None:
        lib = cudalib.load("evalmod")
        p, i = ctypes.c_void_p, ctypes.c_int
        lib.sylph_tail_table.argtypes = [p, p, p, i, i, i, i, p, p, p, p]
        lib.sylph_tail_table.restype = i
        _lib = lib
    return _lib

_orig_postop = arithmetic.homo_mul_relin_rescale_postop
_installed = False


def enabled():
    return _installed and os.environ.get("SYLPH_EVALMOD", "1") != "0"


def _postop(in0, in1, context, *, apply_double=False, add=None, sub=None, scalar=None, plaintext=None):
    """EasyFHE ``homo_mul_relin_rescale_postop`` for a batch of B > 1 ciphertexts."""
    if in0.batch_size == 1 and in1.batch_size == 1 or plaintext is not None or not enabled():
        return _orig_postop(in0, in1, context, apply_double=apply_double, add=add, sub=sub, scalar=scalar,
                            plaintext=plaintext)
    if sum(v is not None for v in (add, sub, scalar)) > 1:
        raise ValueError("homo_mul_relin_rescale_postop accepts at most one post op")
    in0, in1 = arithmetic._align_for_mul(in0, in1, context)
    if in0.batch_size != in1.batch_size or in0.state.cur_limbs != in1.state.cur_limbs:
        raise ValueError("batched postop needs operands of the same batch and level")
    out_limbs = in0.state.cur_limbs - 1
    out_state = CipherState(out_limbs, in0.state.scale_degree + in1.state.scale_degree - 1,
                            in0.state.scaling_factor * in1.state.scaling_factor
                            / context.rescale_divisor_at(out_limbs))
    kw = {}
    other = add if add is not None else sub
    if other is not None:
        if other.batch_size != in0.batch_size:
            raise ValueError("batched postop operand batch mismatch")
        other = alignment.align_to(other, out_state, context)
        kw = dict(post_op=1 if add is not None else 2, post_c0=other.cv[0], post_c1=other.cv[1])
    elif scalar is not None:
        scalar = arithmetic._require_encoded_scalar(scalar, "homo_mul_relin_rescale_postop scalar post op")
        if int(scalar.cur_limbs) != out_limbs or int(scalar.scale_degree) != out_state.scale_degree:
            raise ValueError("batched postop scalar state mismatch")
        kw = dict(post_op=3, post_scalar=arithmetic._scalar_residues_for_cipher(scalar, in0))
    c0, c1 = keyswitch.hmul_relin_rescale(in0.cv[0], in0.cv[1], in1.cv[0], in1.cv[1], in0.state.cur_limbs,
                                          context.L, context.mult_swk_bx, context.mult_swk_ax, context,
                                          apply_double=apply_double, **kw)
    return in0.cipher_like([c0, c1], state=out_state)


def _chebyshev_basis(unit_x, k, ctx, constants, plan):
    """approx._chebyshev_basis without the packed T batch (it is built per item)."""
    T = [unit_x]
    for order in range(2, k + 1):
        lhs, rhs = A._align_mul_operands(T[order // 2 - 1], T[(order + 1) // 2 - 1], ctx)
        if order & 1:
            value = arithmetic.homo_mul_relin_rescale_postop(lhs, rhs, ctx, apply_double=True, sub=T[0])
        else:
            value = arithmetic.homo_mul_relin_rescale_postop(
                lhs, rhs, ctx, apply_double=True,
                scalar=A._mul_rescale_constant(plan.chebyshev_neg_one_scalar_name, lhs, rhs, constants, ctx))
        T.append(value)
    final = T[-1].state
    target = CipherState(final.cur_limbs, final.scale_degree, final.scaling_factor)
    return tuple(alignment.align_to(item, target, ctx) for item in T)


def _tail_values(flat, T_items, ctx, constants, plan):
    """approx._eval_tail_table for a batch: one batch cipher per tail spec."""
    if not flat.tail_specs:
        return ()
    batch = T_items[0].batch_size
    S, D = len(flat.tail_specs), int(flat.tail_max_deg)
    Ts = T_items[:D]
    st = Ts[0].state
    limbs, n = int(st.cur_limbs), int(ctx.N)
    planes = [c for t in Ts for c in t.cv]
    if S <= 16 and D <= 16 and len(Ts) == D and all(t.state == st and len(t.cv) == 2 for t in Ts) and all(
            c.dim() == 3 and c.size(0) == batch and c.size(1) >= limbs and c.size(2) == n and c.is_contiguous()
            for c in planes):
        scalars = A._tail_scalar_table(flat, Ts[0], constants, ctx, plan)
        res = arithmetic._scalar_residues_for_cipher(scalars, Ts[0]).contiguous()
        if tuple(res.shape) != (S, D, limbs):
            raise ValueError(f"unexpected tail scalar table {tuple(res.shape)}")
        table = upload(np.array([[t.cv[0].data_ptr(), t.cv[1].data_ptr(), int(t.cv[0].size(1)) * n] for t in Ts],
                                dtype=np.int64), Ts[0].cv[0].device)
        if any(int(t.cv[1].size(1)) != int(t.cv[0].size(1)) for t in Ts):
            raise ValueError("tail basis components differ in layout")
        out = torch.empty((2, S * batch, limbs, n), dtype=torch.uint64, device=Ts[0].cv[0].device)
        cudalib.check(_load().sylph_tail_table(out.data_ptr(), table.data_ptr(), res.data_ptr(), S, D, limbs, batch,
                                               ctx.primes.data_ptr(), ctx.barret_ratio.data_ptr(),
                                               ctx.barret_k.data_ptr(), torch.cuda.current_stream().cuda_stream),
                      "tail_table")
        state = CipherState(limbs, st.scale_degree + scalars.scale_degree, st.scaling_factor * scalars.scaling_factor)
        tails = alignment.rescale(Ts[0].cipher_like([out[0], out[1]], state=state, batch_size=S * batch), ctx)
        return tuple(tails.cipher_like([c[s * batch:(s + 1) * batch] for c in tails.cv], batch_size=batch)
                     for s in range(S))
    per_item = []
    for b in range(batch):
        T_b = layout.pack_cipher_batch([layout.cipher_batch_item(t, b) for t in T_items[:flat.tail_max_deg]])
        per_item.append(A._eval_tail_table(flat, T_b, ctx, constants, plan))
    return tuple(layout.pack_cipher_batch([per_item[b][i] for b in range(batch)])
                 for i in range(len(flat.tail_specs)))


def eval_mod(ct, ctx, constants, plan):
    """EasyFHE eval_bootstrap_approx_mod on a cipher batch (batch 1 runs EasyFHE's own)."""
    if not enabled() or ct.batch_size == 1:
        return A.eval_bootstrap_approx_mod(ct, ctx, constants, plan)
    flat = plan.approx_eval_plan
    A._require_unit_interval(-1, 1)
    T_items = _chebyshev_basis(ct, flat.k, ctx, constants, plan)
    T2 = A._chebyshev_doubling_basis(T_items[-1], flat.m, ctx, constants, plan)
    tails = _tail_values(flat, T_items, ctx, constants, plan)
    small = [None] * len(flat.small_specs)
    for spec in flat.small_specs:
        small[spec.out_idx] = A._finish_small_spec(spec, A._small_tail(spec, tails, T_items), T_items, T2,
                                                   constants, plan, ctx)
    nodes = A._eval_combine_specs(flat, small, T2, constants, plan, ctx)
    result = A._read_ref(flat.root_ref, small, nodes)
    x = A._sub_aligned(result, A._chebyshev_odd_multiple(T2, ctx), ctx)
    return A.apply_double_angle_iterations(x, ctx, constants, plan)


def install():
    global _installed
    if _installed or os.environ.get("SYLPH_EVALMOD", "1") == "0" or not keyswitch.enabled():
        return
    _installed = True
    arithmetic.homo_mul_relin_rescale_postop = _postop
