#!/usr/bin/env python3
"""Pure NumPy client terminal head for full-fused QuaRot checkpoints.

The final decrypted hidden states must be in the original checkpoint residual
basis. Undo only the additional public Sylph residual rotation before calling
this module; QuaRot's checkpoint basis and fused lm_head remain together.
No ciphertext runtime, intermediate decryption, or transformer chain is used.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SOURCE_PROVENANCE = [{'source_path': 'examples/llama3-8b-fhe/easyfhe_llm_backbone/llama3_lm/plaintext/rmsnorm.py', 'source_file_sha256': '9d794af9d477e572014dd49725c662bd12d1529b1be2ed3819d63ef30381df03', 'functions': [{'name': 'is_rmsn_weight', 'source_line': 6, 'source_sha256': '317b091bfc97fcff523f386abd525f2b2cbc4c581dd4ce77f428a2f20614b551', 'adaptation': 'verbatim function source'}, {'name': 'rmsnorm_numpy', 'source_line': 20, 'source_sha256': 'e126b6c2247662a7d36a120ee8c66947a9a4607148c8ef026964aabab55a5a9c', 'adaptation': 'verbatim function source'}, {'name': 'final_rmsnorm_lm_head_logits', 'source_line': 42, 'source_sha256': 'ea14f6028fd8132b18cb624e6119475cbec52dcb9f52b252f9088807becd92f2', 'adaptation': 'verbatim function source'}]}, {'source_path': 'turbofhe/apps/sylph/checkpoint.py', 'source_file_sha256': '90a6d0571d0bd0f7805ca8db599e2a0a79a0d3b2b8005c0caec8610afd8bd5a8', 'functions': [{'name': 'read_safetensors', 'source_line': 21, 'source_sha256': 'a2c7dc348e68413e3c16e749044ea921cce69e48e6312cf72c01fc3c5b54ce4e', 'adaptation': 'verbatim function source'}]}, {'source_path': 'turbofhe/apps/sylph/calibration_artifact.py', 'source_file_sha256': 'b5c39c9cabb6219bba75abfb9641d2d3e010235000682250ce05b7eca0443130', 'functions': [{'name': '_fwht', 'source_line': 27, 'source_sha256': '98704809b89e077db834a106bf9467b2c41eb06ab5e24fb9f02724302f1e4df0', 'adaptation': 'verbatim function source'}]}]


def is_rmsn_weight(weight: np.ndarray, *, atol: float = 0.0) -> bool:
    """Return True when QuaRot fuse_layer_norms zeroed out the RMSNorm gamma vector.

    QuaRot checkpoints may store a scalar dummy ``weight`` of shape ``(1,)`` after fusion.
    """

    weight = np.asarray(weight, dtype=np.float32).reshape(-1)
    if weight.size == 0:
        return True
    if weight.size == 1:
        return float(np.abs(weight[0])) <= float(atol)
    return float(np.max(np.abs(weight))) <= float(atol)


def rmsnorm_numpy(values: np.ndarray, weight: np.ndarray, *, eps: float) -> np.ndarray:
    """Row RMSNorm reference used by baseline and QuaRot (RMSN) paths.

    Baseline HF: ``out = x / RMS(x) * gamma``.
    QuaRot RMSN: ``gamma == 0`` after ``fuse_layer_norms``; output is ``x / RMS(x)`` only.
    """

    values = np.asarray(values, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32).reshape(-1)
    if values.ndim != 2:
        raise ValueError(f"RMSNorm values must be [seq, width], got {values.shape}.")
    rmsn = is_rmsn_weight(weight)
    if not rmsn and weight.shape[0] != values.shape[1]:
        raise ValueError(f"RMSNorm weight width {weight.shape[0]} does not match values {values.shape[1]}.")
    variance = np.mean(np.square(values, dtype=np.float32), axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(variance + np.float32(eps))
    normalized = values * inv_rms
    if rmsn:
        return normalized.astype(np.float32, copy=False)
    return (normalized * weight).astype(np.float32, copy=False)


def final_rmsnorm_lm_head_logits(
    hidden: np.ndarray,
    norm_weight: np.ndarray,
    lm_head: np.ndarray,
    *,
    eps: float,
    last_token_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Plaintext ``final RMSNorm → lm_head`` used for next-token decode.

    QuaRot ``fuse_layer_norms`` folds final γ into ``lm_head`` and leaves
    ``model.norm.weight`` as a scalar ``0`` stub (``RMSN``). In that case this
    path is normalize-only then ``hidden @ lm_head.T`` — never multiply by the
    stub weight or by a broadcast ones vector of the wrong width.
    """

    hidden = np.asarray(hidden, dtype=np.float32)
    lm_head = np.asarray(lm_head, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError(f"hidden must be [seq, width], got {hidden.shape}.")
    if lm_head.ndim != 2 or lm_head.shape[1] != hidden.shape[1]:
        raise ValueError(
            f"lm_head must be [vocab, width={hidden.shape[1]}], got {lm_head.shape}."
        )
    normed = rmsnorm_numpy(hidden, norm_weight, eps=float(eps))
    tokens = normed[-1:, :] if last_token_only else normed
    logits = (tokens @ lm_head.T).astype(np.float32, copy=False)
    return normed.astype(np.float32, copy=False), logits.reshape(logits.shape[-1]) if last_token_only else logits


def read_safetensors(path):
    """Map validated F16/F32/F64/BF16 storage with only NumPy."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("truncated safetensors header")
        length = int.from_bytes(raw, "little")
        if not 1 <= length <= min(size - 8, 100_000_000):
            raise ValueError("invalid safetensors header length")
        header = json.loads(stream.read(length))
    dtypes = {"F16": "<f2", "F32": "<f4", "F64": "<f8", "BF16": "<u2"}
    result, extents = {}, []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        shape = tuple(entry["shape"])
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("invalid safetensors shape")
        start, end = entry["data_offsets"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= size - 8 - length:
            raise ValueError("invalid safetensors data extent")
        extents.append((start, end))
        if entry["dtype"] not in dtypes:
            # Unselected tensors may use another dtype. Selection rejects it.
            result[name] = None
            continue
        dtype = np.dtype(dtypes[entry["dtype"]])
        if math.prod(shape) * dtype.itemsize != end - start:
            raise ValueError("safetensors shape disagrees with data extent")
        array = np.memmap(path, mode="r", dtype=dtype, offset=8 + length + start, shape=shape)
        result[name] = (array, entry["dtype"])
    extents.sort()
    if any(a[1] > b[0] for a, b in zip(extents, extents[1:])):
        raise ValueError("overlapping safetensors data extents")
    return result


def _fwht(value):
    out = np.array(value, dtype=np.float64, copy=True, order='C')
    size = out.shape[-1]
    if size < 1 or size & (size-1):
        raise ValueError('orthogonal rotation needs a power-of-two axis')
    for bit in range(size.bit_length()-1):
        step = 1 << bit
        blocks = out.reshape(-1, size//(2*step), 2, step)
        left, right = blocks[:, :, 0].copy(), blocks[:, :, 1].copy()
        blocks[:, :, 0], blocks[:, :, 1] = left+right, left-right
    return out/math.sqrt(size)


@dataclass(frozen=True)
class TerminalWeights:
    norm_weight: np.ndarray
    lm_head: np.ndarray
    eps: float
    provenance: dict


def _sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _source_tensor(checkpoint, name):
    index = checkpoint/'model.safetensors.index.json'
    filename = (json.loads(index.read_text())['weight_map'][name]
                if index.exists() else 'model.safetensors')
    relative = Path(filename)
    if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.safetensors':
        raise ValueError('terminal tensor shard must be a relative safetensors path')
    shard = checkpoint/relative
    selected = read_safetensors(shard).get(name)
    if selected is None:
        raise ValueError(f'missing or unsupported terminal tensor {name}')
    stored, dtype = selected
    payload_hash = hashlib.sha256(memoryview(stored).cast('B')).hexdigest()
    array = ((np.asarray(stored, dtype=np.dtype('<u4')) << 16).view(np.float32)
             if dtype == 'BF16' else stored)
    if not np.isfinite(array).all():
        raise ValueError(f'non-finite terminal tensor {name}')
    array.setflags(write=False)
    return array, dict(name=name, shard=str(shard.resolve()), stored_dtype=dtype,
                      stored_shape=list(stored.shape), stored_payload_sha256=payload_hash)


def load_terminal_weights(checkpoint) -> TerminalWeights:
    """Read the actual checkpoint final norm and head without a model framework."""
    checkpoint = Path(checkpoint).resolve()
    config = json.loads((checkpoint/'config.json').read_text())
    norm, norm_source = _source_tensor(checkpoint, 'model.norm.weight')
    head, head_source = _source_tensor(checkpoint, 'lm_head.weight')
    width, vocab = int(config['hidden_size']), int(config['vocab_size'])
    if head.shape != (vocab, width):
        raise ValueError(f'lm_head shape must be {(vocab, width)}, got {head.shape}')
    if not is_rmsn_weight(norm) and norm.shape != (width,):
        raise ValueError('terminal RMSNorm weight must match the hidden width')
    eps = float(config['rms_norm_eps'])
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('terminal RMSNorm epsilon must be positive and finite')
    export_path = checkpoint/'export_metadata.json'
    export = json.loads(export_path.read_text()) if export_path.exists() else None
    provenance = dict(checkpoint=str(checkpoint), config_sha256=_sha256_file(checkpoint/'config.json'),
                      norm=norm_source, lm_head=head_source, eps=eps,
                      norm_semantics='normalize-only' if is_rmsn_weight(norm) else 'normalize-times-gamma',
                      export_metadata=export, copied_sources=SOURCE_PROVENANCE)
    return TerminalWeights(norm, head, eps, provenance)


def inverse_sylph_residual(hidden, residual_signs, residual_permutation):
    """Undo only Sylph's additional public R; preserve the QuaRot checkpoint basis.

    The signs/permutation must come from the last layer's frozen Sylph artifact.
    This is the inverse branch of the upstream OrthogonalRotation.apply.
    """
    hidden = np.asarray(hidden, dtype=np.float64)
    signs = np.asarray(residual_signs)
    permutation = np.asarray(residual_permutation)
    if (hidden.ndim != 2 or signs.ndim != 1 or hidden.shape[1] != len(signs)
            or not np.isin(signs, (-1, 1)).all() or permutation.dtype.kind not in 'iu'
            or permutation.shape != signs.shape
            or sorted(permutation.tolist()) != list(range(len(signs)))):
        raise ValueError('invalid Sylph residual rotation or hidden dimensions')
    return _fwht(hidden[..., np.argsort(permutation)])*signs


def summarize_logits(logits, *, top_k=5):
    """Return deterministic last-token argmax and top-k; no token is predetermined."""
    logits = np.asarray(logits)
    if logits.ndim != 1 or logits.size < 1 or not np.isfinite(logits).all():
        raise ValueError('last-token logits must be one finite vocabulary vector')
    if type(top_k) is not int or not 1 <= top_k <= logits.size:
        raise ValueError('top_k must fit the vocabulary')
    order = np.argsort(-logits, kind='stable')[:max(top_k, min(2, logits.size))]
    return dict(argmax_token_id=int(order[0]), argmax_logit=float(logits[order[0]]),
                top_1_margin=None if logits.size < 2 else float(logits[order[0]]-logits[order[1]]),
                top_k=[dict(token_id=int(i), logit=float(logits[i])) for i in order[:top_k]],
                vocabulary_size=int(logits.size), dtype=str(logits.dtype))


def compare_last_token_logits(actual, reference, *, top_k=5, logit_atol=None, logit_rtol=0.0):
    """Compare against an independently generated full-chain public oracle."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        raise ValueError('actual and reference vocabulary shapes differ')
    a, r = summarize_logits(actual, top_k=top_k), summarize_logits(reference, top_k=top_k)
    if (logit_atol is not None and (not math.isfinite(logit_atol) or logit_atol < 0)
            or not math.isfinite(logit_rtol) or logit_rtol < 0):
        raise ValueError('logit tolerances must be nonnegative and finite')
    delta = actual.astype(np.float64)-reference.astype(np.float64)
    within = (None if logit_atol is None else bool(np.all(
        np.abs(delta) <= float(logit_atol)+float(logit_rtol)*np.abs(reference))))
    match = a['argmax_token_id'] == r['argmax_token_id']
    return dict(argmax_match=match,
                top_k_match=[x['token_id'] for x in a['top_k']] == [x['token_id'] for x in r['top_k']],
                max_absolute_error=float(np.max(np.abs(delta))), rmse=float(np.sqrt(np.mean(delta*delta))),
                logit_atol=logit_atol, logit_rtol=float(logit_rtol), within_tolerance=within,
                passed=bool(match and (within is None or within)))


def load_input_ids(path):
    """Read the canonical Paris object or a plain token-ID list without tokenization."""
    path = Path(path)
    data = json.loads(path.read_text())
    values = data.get('input_ids') if isinstance(data, dict) else data
    ids = np.asarray(values)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) != 128 or np.any(ids < 0):
        raise ValueError('Paris128 input must contain exactly 128 nonnegative integer token IDs')
    return ids, dict(path=str(path.resolve()), file_sha256=_sha256_file(path), tokens=len(ids),
                    historical_expected_token_hint=data.get('expected_next_token_id') if isinstance(data, dict) else None)


def evaluate_terminal_logits(checkpoint, final_hidden, *, reference_hidden=None, top_k=5,
                             logit_atol=None, logit_rtol=0.0, expected_token_hint=None,
                             input_token_ids=None):
    """Client API: checkpoint-basis final hidden -> last-token logits and report.

    The caller decrypts only the final chain output and undoes the extra Sylph R.
    reference_hidden must be the independent full-model oracle with the same
    checkpoint, private input and public-prefix semantics. No QuaRot inverse is
    applied here. A historical token hint never determines the pass/fail gate.
    """
    weights = checkpoint if isinstance(checkpoint, TerminalWeights) else load_terminal_weights(checkpoint)
    final_hidden = np.asarray(final_hidden)
    if final_hidden.ndim != 2 or not len(final_hidden) or not np.isfinite(final_hidden).all():
        raise ValueError('final client hidden must be finite [tokens, hidden]')
    if input_token_ids is not None:
        ids = np.asarray(input_token_ids)
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) != len(final_hidden):
            raise ValueError('private token IDs must match the final hidden token rows')
    _, logits = final_rmsnorm_lm_head_logits(final_hidden, weights.norm_weight, weights.lm_head,
                                            eps=weights.eps, last_token_only=True)
    report = dict(mode='pure NumPy client final RMSNorm and lm_head',
                  hidden_basis='full-fused QuaRot checkpoint residual basis',
                  input_hidden_shape=list(final_hidden.shape), token_index=len(final_hidden)-1,
                  quarot_inverse_applied=False, sylph_inverse_performed_in_module=False,
                  caller_supplies_checkpoint_basis=True,
                  actual=summarize_logits(logits, top_k=top_k), reference=None, comparison=None,
                  expected_token_hint=expected_token_hint, hint_used_for_pass=False,
                  weight_provenance=weights.provenance, passed=None)
    if reference_hidden is not None:
        reference_hidden = np.asarray(reference_hidden)
        if (reference_hidden.shape != final_hidden.shape or not np.isfinite(reference_hidden).all()):
            raise ValueError('reference hidden must be finite and match the actual hidden shape')
        _, reference_logits = final_rmsnorm_lm_head_logits(reference_hidden, weights.norm_weight, weights.lm_head,
                                                          eps=weights.eps, last_token_only=True)
        report['reference'] = summarize_logits(reference_logits, top_k=top_k)
        report['comparison'] = compare_last_token_logits(logits, reference_logits, top_k=top_k,
                                                         logit_atol=logit_atol, logit_rtol=logit_rtol)
        report['passed'] = report['comparison']['passed']
        report['hint_matches_reference'] = (None if expected_token_hint is None else
                                            report['reference']['argmax_token_id'] == expected_token_hint)
    return logits, report


def terminal_hook(final_hidden, manifest, args):
    """Optional chain-runner hook; terminal paths must already be resolved."""
    terminal = manifest['terminal']
    reference_path = terminal.get('reference_hidden_path', terminal.get('reference_hidden'))
    reference = (None if reference_path is None else np.load(reference_path, allow_pickle=False))
    logits, report = evaluate_terminal_logits(terminal['checkpoint'], final_hidden,
        reference_hidden=reference, top_k=int(terminal.get('top_k', 5)),
        logit_atol=terminal.get('logit_atol'), logit_rtol=float(terminal.get('logit_rtol', 0.0)),
        expected_token_hint=terminal.get('expected_token_hint'))
    output = terminal.get('logits_path')
    if output:
        np.save(output, logits, allow_pickle=False)
        report['logits_path'] = str(Path(output).resolve())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--hidden', type=Path, required=True, help='final client hidden in checkpoint basis')
    parser.add_argument('--reference-hidden', type=Path, help='matching full-public-chain oracle hidden')
    parser.add_argument('--token-ids', type=Path, help='canonical Paris object or a 128-ID array')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--logit-atol', type=float)
    parser.add_argument('--logit-rtol', type=float, default=0.0)
    parser.add_argument('--expected-token-hint', type=int, help='metadata only, never a pass condition')
    parser.add_argument('--logits', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    hidden = np.load(args.hidden, allow_pickle=False)
    reference = None if args.reference_hidden is None else np.load(args.reference_hidden, allow_pickle=False)
    ids, inputs = (None, None) if args.token_ids is None else load_input_ids(args.token_ids)
    hint = args.expected_token_hint
    if hint is None and inputs is not None:
        hint = inputs['historical_expected_token_hint']
    logits, report = evaluate_terminal_logits(args.checkpoint, hidden, reference_hidden=reference,
        top_k=args.top_k, logit_atol=args.logit_atol, logit_rtol=args.logit_rtol,
        expected_token_hint=hint, input_token_ids=ids)
    report['inputs'] = inputs
    report['hidden_path'] = str(args.hidden.resolve())
    report['hidden_sha256'] = _sha256_file(args.hidden)
    report['reference_hidden_path'] = None if args.reference_hidden is None else str(args.reference_hidden.resolve())
    if args.logits is not None:
        args.logits.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.logits, logits, allow_pickle=False)
        report['logits_path'] = str(args.logits.resolve())
    text = json.dumps(report, indent=2)+'\n'
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end='')
    return 1 if report['passed'] is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
