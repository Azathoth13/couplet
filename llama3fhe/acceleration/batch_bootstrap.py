"""Batched pair bootstrap using the packaged EasyFHE extensions.

``bootstrap_batch`` follows EasyFHE's ``bootstrap_modraise_first`` stages:
raise each ciphertext, stack, batched CoeffsToSlots, real/imaginary EvalMod,
recombine, SlotsToCoeffs, scale and align, then unstack. Constants and plans
match the per-ciphertext path. ``ODIN_BATCH_BS_MAX`` caps each batch; sparse-slot
programs keep the per-ciphertext implementation.

``install`` registers the implementation on ``BootstrapOperator.batch_backend``
after the shared key-switch kernels have been installed.
"""
from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("ODIN_BATCH_BS", "0") == "1"


def max_batch() -> int:
    return max(1, int(os.environ.get("ODIN_BATCH_BS_MAX", "16")))


def _stack(cts):
    import easyfhe as torch

    if len(cts) == 1:
        return cts[0]
    cv = [torch.stack([c.cv[i] if c.cv[i].dim() == 2 else c.cv[i][0] for c in cts], 0) for i in range(len(cts[0].cv))]
    return cts[0].cipher_like(cv, batch_size=len(cts))


def _unstack(ct):
    if int(ct.batch_size) == 1:
        return [ct]
    return [ct.cipher_like([x[i] for x in ct.cv], batch_size=1) for i in range(int(ct.batch_size))]


def _monomial_inplace(ct, degree, ctx):
    from easyfhe.fhe.ops import kernels as F

    for comp in ct.cv:
        F.cv_mul_by_monomial(comp, ct.state.cur_limbs, int(degree), ctx)
    return ct


def _eval_mod(part, ctx, consts, plan, evalmod):
    if evalmod.enabled():
        return evalmod.eval_mod(part, ctx, consts, plan)
    from easyfhe.bs.openfhe.runtime import approx as bs_approx

    # SYLPH_EVALMOD=0 ablation: EasyFHE's EvalMod takes one ciphertext
    return _stack([bs_approx.eval_bootstrap_approx_mod(item, ctx, consts, plan) for item in _unstack(part)])


def bootstrap_batch(ciphers, ctx, program):
    """``[easyfhe.bs.openfhe.bootstrap(c, ctx, program) for c in ciphers]`` as one batched pass."""

    from easyfhe.bs.openfhe.runtime.raising import raise_ciphertext
    from easyfhe.bs.openfhe.runtime.scaling import (
        scale_after_approx,
        scale_to_original_message,
    )
    from easyfhe.fhe.ops import alignment, arithmetic
    from llama3fhe.acceleration.fhe_ext import bsgs, evalmod
    from llama3fhe.acceleration.fhe_ext.rotate import rotate as fast_rotate

    ciphers = list(ciphers)
    cap = max_batch()
    if len(ciphers) > cap:
        out = []
        for start in range(0, len(ciphers), cap):
            out.extend(bootstrap_batch(ciphers[start:start + cap], ctx, program))
        return out

    consts, plan = program.constants, program._runtime_plan
    raise_to = int(program.raise_to_limbs)
    slots = int(ciphers[0].slots)
    if slots != int(ctx.M) // 4 or int(plan.slots) != slots:
        # sparse-slot programs keep EasyFHE's per-ciphertext path (replicate / restore steps)
        import easyfhe.bs.openfhe as bs
        return [bs.bootstrap(c, ctx, program) for c in ciphers]

    raised = [raise_ciphertext(c, ctx, consts, raise_to) for c in ciphers]
    if any(r.state != raised[0].state for r in raised):
        import easyfhe.bs.openfhe as bs
        return [bs.bootstrap(c, ctx, program) for c in ciphers]
    stack = _stack(raised)

    # CoeffsToSlots on the batch
    encoded = bsgs.coeffs_to_slots(stack, ctx, consts, plan)
    # EvalMod (full-slot variant of easyfhe ... runtime.evalmod.eval_mod_full) on the batch
    conjugate = fast_rotate(encoded, 2 * int(ctx.N) - 1, ctx)
    imag = arithmetic.homo_sub(encoded, conjugate, ctx)
    real = arithmetic.homo_add(encoded, conjugate, ctx)
    imag = _monomial_inplace(imag, 3 * int(ctx.M) // 4, ctx)
    real = alignment.normalize_scale(real, ctx)
    imag = alignment.normalize_scale(imag, ctx)
    real = _eval_mod(real, ctx, consts, plan, evalmod)
    imag = _eval_mod(imag, ctx, consts, plan, evalmod)
    imag = _monomial_inplace(imag, int(ctx.M) // 4, ctx)
    encoded = arithmetic.homo_add(real, imag, ctx)
    encoded = scale_after_approx(encoded, ctx, consts, raise_to_limbs=raise_to)
    encoded = alignment.normalize_scale(encoded, ctx)
    # SlotsToCoeffs on the batch, then EasyFHE's output alignment
    decoded = bsgs.slots_to_coeffs(encoded, ctx, consts, plan)
    decoded = scale_to_original_message(decoded, ctx, consts)
    decoded = decoded.cipher_like(decoded.cv, slots=slots)
    decoded = alignment.normalize_scale(decoded, ctx)
    if decoded.state != program.output_state:
        decoded = alignment.align_to(decoded, program.output_state, ctx)
    return _unstack(decoded)


def install() -> dict:
    """Register :func:`bootstrap_batch` on ``BootstrapOperator`` (needs the shared kernels installed)."""

    from llama3fhe.acceleration.fhe_ext import bsgs, evalmod, keyswitch

    if not keyswitch.enabled():
        raise RuntimeError("ODIN_BATCH_BS needs the shared key-switch kernels (ODIN_SYLPH_KERNELS=1)")
    from llama3fhe.operators.nonlinear.bootstrap import BootstrapOperator

    BootstrapOperator.batch_backend = staticmethod(bootstrap_batch)
    return {"batch_bs": True, "max_batch": max_batch(), "bsgs": bsgs.enabled(), "evalmod": evalmod.enabled()}


def uninstall() -> None:
    from llama3fhe.operators.nonlinear.bootstrap import BootstrapOperator

    BootstrapOperator.batch_backend = None
