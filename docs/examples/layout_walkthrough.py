"""A tiny, verified walk-through of every packing layout in this repository.

Runs the real layout classes on a doll-sized shape so every slot index below is
produced by the production code, not asserted by hand.
"""
from __future__ import annotations

import numpy as np

from llama3fhe.layouts.attention import AttentionPairLayout
from llama3fhe.layouts.feature_major import FeatureMajorPrefillLayout
from llama3fhe.layouts.linear import LinearCarrierLayout

BAR = "=" * 78


def show_feature_major() -> None:
    print(BAR)
    print("1. FeatureMajorPrefillLayout  —  the persistent boundary")
    print(BAR)
    # Doll size: 4 tokens, 8 features, 16 slots  ->  2 tokens/cipher, 2 ciphers
    fm = FeatureMajorPrefillLayout(seq_len=4, hidden_dim=8, slots=16)
    print(f"shape: S={fm.seq_len} D={fm.hidden_dim} slots={fm.slots}")
    print(f"  tokens_per_cipher = slots/D = {fm.tokens_per_cipher}")
    print(f"  cipher_count      = S/T     = {fm.cipher_count}")
    print("  slot(d,u) = d*T + u    (feature is the SLOW axis, token the FAST one)")

    # a recognisable hidden state: value = token*100 + feature
    hidden = np.array(
        [[t * 100 + f for f in range(fm.hidden_dim)] for t in range(fm.seq_len)],
        dtype=np.float32,
    )
    print("\nlogical hidden [token, feature]:")
    for t, row in enumerate(hidden):
        print(f"  token {t}: {row.astype(int).tolist()}")

    packed = fm.pack(hidden)
    print(f"\npacked -> {packed.shape[0]} ciphertexts x {packed.shape[1]} slots")
    for c, row in enumerate(packed):
        first, last = c * fm.tokens_per_cipher, (c + 1) * fm.tokens_per_cipher - 1
        print(f"  cipher {c} (tokens {first}..{last}): {row.astype(int).tolist()}")
    print("\n  read it as pairs: slot 2d holds token 0's feature d,")
    print("                    slot 2d+1 holds token 1's feature d.")
    print(f"  slot(d=3,u=1) = 3*{fm.tokens_per_cipher}+1 = {fm.slot(3, 1)}"
          f"  -> value {int(packed[0, fm.slot(3, 1)])} = token 1, feature 3")
    assert np.array_equal(fm.unpack(packed), hidden)
    print("  unpack(pack(x)) == x  ✓")

    print("\nwhy this shape: a linear map W·x touches every feature of ONE token,")
    print("and all features of a token live in one ciphertext at stride T, so a")
    print("diagonal (BSGS) matvec needs rotations by multiples of T only.")


def show_linear_carrier() -> None:
    print("\n" + BAR)
    print("2. LinearCarrierLayout  —  private QKV output carrier")
    print(BAR)
    car = LinearCarrierLayout(seq_len=4, dimension=8, slots=16)
    print("slot(d,lane) = d*token_lanes + lane   (same algebra as feature-major)")
    print(f"  token_lanes={car.token_lanes} cipher_count={car.cipher_count}")
    print("difference from feature-major: `dimension` is the PHYSICAL row width,")
    print("and only a prefix is active. In production QKV emits 3072 heterogeneous")
    print("channels (Q 2048 folded + K 512 + V 512) inside a 4096-wide carrier,")
    print("so the operator may read a prefix without repacking. It never crosses a")
    print("residual boundary, which is why it is a separate type.")


def show_attention_pair() -> None:
    print("\n" + BAR)
    print("3. AttentionPairLayout  —  the C and Delta working forms")
    print(BAR)
    # Doll size: S=head_dim=4, 2 Q heads, 2 KV heads, payload = 2*4*4 = 32 slots
    at = AttentionPairLayout(
        seq_len=4, head_dim=4, query_heads=2, key_value_heads=2, slots=32
    )
    print(f"shape: S={at.seq_len} d_h={at.head_dim} Hq={at.query_heads} "
          f"Hkv={at.key_value_heads} slots={at.slots}")
    print(f"  payload_slots      = 2*S*d_h = {at.payload_slots}")
    print(f"  payload_repetitions= {at.payload_repetitions}")
    print(f"  query pairs={at.query_pair_count}  kv pairs={at.key_value_pair_count}"
          f"   (two heads share one ciphertext, interleaved by `lane`)")

    print("\n-- C form (used by Q and by the attention output O) --")
    print("   slot = 2*(feature*S + token) + lane")
    q = np.zeros((at.seq_len, at.query_heads * at.head_dim), dtype=np.float32)
    for t in range(at.seq_len):
        for h in range(at.query_heads):
            for f in range(at.head_dim):
                q[t, h * at.head_dim + f] = h * 100 + t * 10 + f
    packed_q = at.pack_query_c(q)
    print(f"   Q_C -> {packed_q.shape[0]} cipher(s) x {packed_q.shape[1]} slots")
    print("   value encoding is head*100 + token*10 + feature")
    print(f"   {packed_q[0].real.astype(int).tolist()}")
    print("   lane 0 = head 0 (even slots), lane 1 = head 1 (odd slots)")
    for (f, t, lane) in ((0, 0, 0), (0, 0, 1), (2, 3, 0)):
        slot = 2 * (f * at.seq_len + t) + lane
        print(f"     slot {slot:2d} = 2*({f}*{at.seq_len}+{t})+{lane}"
              f"  -> {int(packed_q[0, slot].real):3d}  (head {lane}, token {t}, feature {f})")

    print("\n-- Delta form (used by K, V, the scores S and the probabilities P) --")
    print("   slot = 2*(delta*S + query) + lane,  holding the value whose")
    print("   source index is (query + delta) mod S.  A cyclic 'diagonal' view:")
    print("   row `delta` of the ciphertext is the diagonal at offset delta.")
    k = np.zeros((at.seq_len, at.key_value_heads * at.head_dim), dtype=np.float32)
    for t in range(at.seq_len):
        for h in range(at.key_value_heads):
            for f in range(at.head_dim):
                k[t, h * at.head_dim + f] = h * 100 + t * 10 + f
    packed_k = at.pack_key_delta(k)
    print(f"   K_Delta -> {packed_k.shape[0]} cipher(s) x {packed_k.shape[1]} slots")
    print(f"   {packed_k[0].real.astype(int).tolist()}")
    for (delta, query, lane) in ((0, 0, 0), (1, 0, 0), (1, 3, 0)):
        slot = 2 * (delta * at.seq_len + query) + lane
        src = (query + delta) % at.seq_len
        print(f"     slot {slot:2d} (delta={delta}, query={query}, lane={lane})"
              f" -> {int(packed_k[0, slot].real):3d}"
              f"   feature={delta}, source token=({query}+{delta})%{at.seq_len}={src}")
    print("   note the feature index IS the delta row: feature d of the source")
    print("   token (q+d) sits at delta row d, query column q.")

    print("\n-- why QK and PV need no repacking --")
    print("   S_Delta[d, q] = sum_f Q_C[f, q] * K_Delta[f, q]  after a rotation by d")
    print("   so the SCORE lands in the same (delta, query, lane) grid the")
    print("   probabilities and the PV input use. One slot contract carries")
    print("   Q_C -> S_Delta -> P_Delta -> O_C with no layout change in between.")
    scores = np.zeros((at.query_heads, at.seq_len, at.seq_len), dtype=np.float32)
    for h in range(at.query_heads):
        for qi in range(at.seq_len):
            for ki in range(at.seq_len):
                scores[h, qi, ki] = h * 100 + qi * 10 + ki
    packed_s = at.pack_score_delta(scores)
    print(f"\n   S_Delta -> {packed_s.shape[0]} cipher(s), value = head*100+query*10+key")
    print(f"   {packed_s[0].real.astype(int).tolist()}")
    for (delta, query) in ((0, 2), (1, 2), (3, 2)):
        slot = 2 * (delta * at.seq_len + query)
        key = (query + delta) % at.seq_len
        print(f"     slot {slot:2d} (delta={delta}, query={query}, lane=0) ->"
              f" {int(packed_s[0, slot].real):3d}   key=({query}+{delta})%{at.seq_len}={key}")
    print("   causality: key <= query is exactly `delta == 0 or delta >= S-query`,")
    print("   a PUBLIC mask over delta rows -- no data-dependent branching.")
    assert np.allclose(at.unpack_score_delta(packed_s), scores)
    print("   unpack_score_delta(pack_score_delta(x)) == x  ✓")


def show_qkv_boundary() -> None:
    print("\n" + BAR)
    print("3b. The QKV boundary: FM pairs in, heterogeneous carrier out")
    print(BAR)
    from llama3fhe.operators.linear.prefill.qkv_kernel import _projection_weights

    # Doll size chosen so the folded channels fill the carrier exactly:
    # D=32, S=d_h=4, Hq=8, Hkv=4  ->  Q fold 16 + K fold 8 + V pair 8 = 32 = D
    at = AttentionPairLayout(
        seq_len=4, head_dim=4, query_heads=8, key_value_heads=4, slots=64
    )
    dim = 32
    rng = np.random.default_rng(1)
    hidden = rng.standard_normal((4, dim)).astype(np.float32)
    q_w = rng.standard_normal((32, dim)).astype(np.float32)
    k_w = rng.standard_normal((16, dim)).astype(np.float32)
    v_w = rng.standard_normal((16, dim)).astype(np.float32)
    _, carrier_w, _ = _projection_weights(
        hidden, q_w, k_w, v_w, attention_layout=at
    )

    print("input: FM ciphertexts, adjacent token shards paired as z = x + i*y,")
    print("so 2 real ciphers become 1 complex projection input. The kernel")
    print("projects z AND conj(z): u+v = W*Re(z) recovers shard x's tokens and")
    print("i*(v-u) = W*Im(z) recovers shard y's -- one pass serves both shards.")

    def decode(vec) -> str:
        for name, w, heads in (
            ("Q", q_w, at.query_heads),
            ("K", k_w, at.key_value_heads),
            ("V", v_w, at.key_value_heads),
        ):
            rows = w.reshape(heads, at.head_dim, dim)
            for h in range(heads):
                for f in range(at.head_dim):
                    base = rows[h, f]
                    s = float(vec @ base) / float(base @ base)
                    if abs(s) > 1e-6 and np.allclose(vec, s * base, atol=1e-5):
                        return f"{s:+.2f}*{name}[h{h},f{f}]"
        return "0" if np.allclose(vec, 0) else "???"

    print(f"\noutput: ONE carrier matrix ({carrier_w.shape[0]} rows), decoded row")
    print("by exact vector match against the Q/K/V weight rows:")
    for r in (0, 1, 2, 7, 16, 18, 23, 24, 26, 31):
        print(f"  row {r:2d}: real = {decode(carrier_w[r].real):18s}"
              f" imag = {decode(carrier_w[r].imag)}")
    print("\nthe three blocks (rows 0..15 / 16..23 / 24..31 here):")
    print("  Q  folded:  real=Q[h,f], imag=Q[h,f+d_h/2]  -- one complex slot")
    print("              carries the feature pair the C form interleaves")
    print("  K  folded, conjugated: real=K[h,f], imag=-K[h,f+d_h/2] -- the -i")
    print("              factor pre-applies the conjugation the Delta shear needs")
    print("  V  paired:  real=V[h_even,f], imag=V[h_odd,f] -- the two heads of")
    print("              one KV pair ride real/imag, features stay sequential")
    print("\nproduction: rows 0..2047 folded Q, 2048..2559 K, 2560..3071 V,")
    print("rows 3072..4095 zero -- 3072 active channels in a 4096-wide carrier,")
    print("one diagonal projection producing Q, K and V together.")


def show_production_numbers() -> None:
    print("\n" + BAR)
    print("4. Production geometry, cipher by cipher")
    print(BAR)
    fm = FeatureMajorPrefillLayout()
    at = AttentionPairLayout()
    car = LinearCarrierLayout(seq_len=128, dimension=4096, slots=32768)
    print(f"feature-major : S={fm.seq_len} D={fm.hidden_dim} slots={fm.slots}"
          f"  -> {fm.tokens_per_cipher} tokens/cipher, {fm.cipher_count} real ciphers")
    print(f"                 {fm.cipher_count} x {fm.slots} = "
          f"{fm.cipher_count*fm.slots} slots for S*D = {fm.seq_len*fm.hidden_dim}"
          f" values -> 100% dense")
    print(f"linear carrier: dimension={car.dimension} lanes={car.token_lanes}"
          f" -> {car.cipher_count} ciphers")
    print(f"attention     : payload={at.payload_slots} = slots -> "
          f"{at.payload_repetitions} copy per ciphertext")
    print(f"                Q_C {at.query_pair_count} ciphers "
          f"({at.query_heads} heads / 2 lanes)")
    print(f"                K_Delta / V_Delta {at.key_value_pair_count} ciphers each "
          f"({at.key_value_heads} heads / 2 lanes)")
    print(f"                S_Delta / P_Delta {at.query_pair_count} ciphers")
    print(f"                GQA ratio = {at.gqa_ratio} "
          f"(each KV pair serves {at.gqa_ratio} query pairs)")
    print("\nQ head pairing (GQA-aligned so one K/V pair serves R query pairs):")
    print(f"  {at.query_head_pairs()[:6]} ...")
    print(f"KV head pairing: {at.key_value_head_pairs()}")


if __name__ == "__main__":
    show_feature_major()
    show_linear_carrier()
    show_attention_pair()
    show_qkv_boundary()
    show_production_numbers()
