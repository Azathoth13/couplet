# Packing and per-operator layouts

This guide gives worked 128-token / 4096-feature examples. Runtime geometry
comes from `configs/models/`: 1B uses 2048 features, generic 3B is padded to
4096, and 8B uses 4096. The three interfaces illustrated below connect linear
and attention operators; persistent PairFold KV has a separate page contract
described in [Design](DESIGN.md).

A runnable version of every example below:

```bash
python docs/examples/layout_walkthrough.py
```

Every slot index in this document is produced by that script from the
production layout classes, at a doll-sized shape so the arrays fit on a page.

## The three contracts at a glance

| | crosses a layer boundary? | slot formula | production geometry |
|---|---|---|---|
| `FeatureMajorPrefillLayout` | **yes** — the only one | `d*T + u` | 16 real ciphers, T=8 tokens each |
| `LinearCarrierLayout` | no (QKV → attention) | `d*lanes + lane` | 16 ciphers, 3072 of 4096 rows active |
| `AttentionPairLayout` | no (inside attention) | C: `2*(f*S+t)+lane`  Δ: `2*(δ*S+q)+lane` | Q 16, K/V 4 each, 2 heads per cipher |

CKKS gives us 32768 complex slots at logN=16. The persistent state is
128×4096 = 524288 real values = 16 ciphertexts exactly, so the boundary is
100% dense with no padding to carry.

---

## 1. FeatureMajorPrefillLayout — the persistent boundary

```text
slot(d, u) = d * tokens_per_cipher + u
cipher c holds tokens [c*T, (c+1)*T)
T = slots / hidden_dim = 32768 / 4096 = 8
cipher_count = seq_len / T = 128 / 8 = 16
```

Feature is the **slow** axis, token the **fast** one. At S=4, D=8, slots=16
(so T=2, two ciphers), a hidden state whose value is `token*100 + feature`
packs to:

```text
logical [token, feature]          packed slots
  token 0: 0 1 2 3 4 5 6 7        cipher 0: [  0 100   1 101   2 102   3 103   4 104   5 105   6 106   7 107]
  token 1: 100 ... 107                       └──d=0──┘└──d=1──┘└──d=2──┘ ...
  token 2: 200 ... 207            cipher 1: [200 300 201 301 202 302 203 303 204 304 205 305 206 306 207 307]
  token 3: 300 ... 307
```

`slot(d=3, u=1) = 3*2+1 = 7`, and slot 7 holds 103 — token 1, feature 3.

**Why this axis order.** A linear map `W·x` needs every feature of one token.
All features of a token live in one ciphertext at stride `T`, so a diagonal
(BSGS) matvec only ever rotates by multiples of `T`: the baby steps are
`{0, T, 2T, …}` and one giant step is `baby_steps*T`. Nothing in the layer
needs a cross-token rotation at this boundary, which is what keeps the
residual path rotation-free.

**What crosses a layer.** The residual add, inter-layer bootstrap and layer
graph input/output preserve these 16 real feature-major ciphertexts. The
execution profile selects the modulus levels.

---

## 2. LinearCarrierLayout — the private QKV carrier

```text
slot(d, lane) = d * token_lanes + lane
```

Same algebra as feature-major, two differences that earn it a separate type:

- `dimension` is the **physical** row width (4096), and operators may read an
  active prefix. Production QKV emits 3072 heterogeneous channels — folded Q
  2048, K 512, V 512 — inside a 4096-wide carrier, so one projection produces
  Q, K and V together and the adapter slices them without repacking.
- It never crosses a residual boundary, so its geometry is free to change with
  the kernel; the persistent contract above is not.

### The QKV boundary in detail

QKV is where all three contracts meet, so it deserves its own example
(section 3b of the walkthrough script). At D=32, S=d_h=4, Hq=8, Hkv=4 the
folded channels fill the carrier exactly, and decoding each carrier row by
exact vector match against the weight rows gives:

```text
row  0: real = +0.50*Q[h0,f0]   imag = +0.50*Q[h0,f2]   ┐
row  2: real = +0.50*Q[h0,f1]   imag = +0.50*Q[h0,f3]   │ Q folded
row  7: real = +0.50*Q[h3,f1]   imag = +0.50*Q[h3,f3]   ┘
row 16: real = +1.00*K[h0,f0]   imag = -1.00*K[h0,f2]   ┐ K folded,
row 23: real = +1.00*K[h3,f1]   imag = -1.00*K[h3,f3]   ┘ conjugated
row 24: real = +0.50*V[h0,f0]   imag = +0.50*V[h2,f0]   ┐ V head-paired,
row 31: real = +0.50*V[h1,f3]   imag = +0.50*V[h3,f3]   ┘ features sequential
```

Three different complex packings inside **one** projection:

- **Q folded**: one complex slot carries the feature pair `(f, f+d_h/2)` —
  exactly the pair the C form's RoPE rotation consumes together.
- **K folded and conjugated**: the `-i` factor pre-applies the conjugation the
  Δ shear needs, so the adapter's K path saves that step.
- **V head-paired**: the two heads of one KV pair ride real/imaginary with
  features sequential, matching the V-Δ gather downstream.

The input side is complex too: adjacent FM token shards pair as `z = x + i·y`,
and the kernel projects both `z` and `conj(z)` — `u+v = W·Re(z)` recovers
shard x and `i·(v−u) = W·Im(z)` recovers shard y, so one diagonal pass serves
two token shards and emits Q, K and V together. In production the carrier is
rows 0–2047 folded Q, 2048–2559 K, 2560–3071 V, 3072–4095 zero.

---

## 3. AttentionPairLayout — the C and Δ working forms

Two heads share one ciphertext, interleaved by `lane ∈ {0,1}`:

```text
payload_slots = 2 * S * head_dim = 2*128*128 = 32768 = slots   (one copy)
Q:  32 heads / 2 = 16 pair-ciphers
K,V: 8 heads / 2 =  4 pair-ciphers each
GQA ratio 4: each K/V pair serves 4 query pairs
```

Head pairing is GQA-aligned — `((0,4),(1,5),(2,6),(3,7),(8,12),…)` for Q and
`((0,1),(2,3),(4,5),(6,7))` for K/V — so one K/V ciphertext feeds exactly R
consecutive query ciphertexts with no gather.

### C form — used by Q and by the attention output O

```text
slot = 2 * (feature * S + token) + lane
```

At S=d_h=4 with two heads, values encoded as `head*100 + token*10 + feature`:

```text
Q_C: [  0 100  10 110  20 120  30 130 |   1 101  11 111  21 121  31 131 | ...]
       └──────── feature 0 ─────────┘   └──────── feature 1 ─────────┘
        t0  t0  t1  t1  t2  t2  t3  t3
        h0  h1  h0  h1  h0  h1  h0  h1

slot  0 = 2*(0*4+0)+0 →   0   head 0, token 0, feature 0
slot  1 = 2*(0*4+0)+1 → 100   head 1, token 0, feature 0
slot 22 = 2*(2*4+3)+0 →  32   head 0, token 3, feature 2
```

### Δ form — used by K, V, the scores S and the probabilities P

```text
slot = 2 * (delta * S + query) + lane
the value stored is the one whose source index is (query + delta) mod S
```

Row `delta` is the cyclic diagonal at offset `delta`. For K, the **feature
index is the delta row**: feature `d` of source token `(q+d) mod S` sits at
delta row `d`, query column `q`.

```text
K_Δ: [  0 100  10 110  20 120  30 130 |  11 111  21 121  31 131   1 101 | ...]
       └───── delta 0 (feature 0) ────┘  └───── delta 1 (feature 1) ────┘

slot  0 (δ=0, q=0) →  0   feature 0, source token (0+0)%4 = 0
slot  8 (δ=1, q=0) → 11   feature 1, source token (0+1)%4 = 1
slot 14 (δ=1, q=3) →  1   feature 1, source token (3+1)%4 = 0   ← wraps
```

For the scores, `delta` indexes the key offset instead:

```text
S_Δ: value = head*100 + query*10 + key
slot  4 (δ=0, q=2) → 22   key = (2+0)%4 = 2
slot 12 (δ=1, q=2) → 23   key = (2+1)%4 = 3
slot 28 (δ=3, q=2) → 21   key = (2+3)%4 = 1
```

**Causality is a public mask.** `key ≤ query` is exactly
`delta == 0 or delta ≥ S - query`, a fixed pattern over delta rows. The causal
mask is therefore a plaintext multiply with no data-dependent branch, and it
is the same mask for every layer and every head.

**Why this form removes the repack.** Write the QK product in Δ coordinates:
rotating `Q_C` by `d` and multiplying by `K_Δ` accumulates
`S_Δ[d, q] = Σ_f Q[q, f] · K[(q+d), f]` — the score for query `q` against key
`q+d`, landing in the `(delta, query, lane)` grid. Softmax operates row-wise
over that same grid, and PV consumes it directly. So

```text
Q_C ──QK──▶ S_Δ ──softmax──▶ P_Δ ──PV──▶ O_C
```

runs under **one slot contract**, with no intermediate layout conversion. A
diagonal-only layout (Halevi–Shoup style) has the scores land in a form PV
cannot consume, which forces an explicit repack of the probabilities between
softmax and PV — the single most expensive stage such designs pay for.

---

## Per-operator layout table

Levels are from a steady-state layer at depth 41 (`tools/verify.sh steady`).

| stage | in | out | ciphers in → out | out level |
|---|---|---|---|---|
| input RMSNorm (+ fused bootstrap) | FM | FM | 16 → 16 | L27 |
| modulus drop to the attention budget | FM | FM | 16 → 16 | L28 |
| QKV diagonal linear | FM (paired to 8 complex) | LinearCarrier | 16 → 16 | L29 |
| attention input adapter | LinearCarrier | AttentionPair C/Δ | 16 → Q 16, K 4, V 4 | K L36, V L32 |
| QK | Q_C + K_Δ | S_Δ | 16+4 → 16 | L38 |
| softmax (Alg2) | S_Δ | P_Δ | 16 → 16 | L33 |
| PV | P_Δ + V_Δ | O_C | 16+4 → 16 | L36 |
| PairFM output | O_C | FM token-pair carriers | 16 → 8 complex | L39 |
| W_O linear | FM carriers | FM | 8 → 16 | L40 |
| residual add | FM + FM | FM | 16+16 → 16 | L40 |
| post-attention RMSNorm (+ fused bootstrap) | FM | FM | 16 → 16 | L28 |
| modulus drop to the MLP budget | FM | FM | 16 → 16 | L30 |
| MLP (gate/up fused, SiLU, down) | FM | FM | 16 → 16 | L39 |
| residual add | FM + FM | FM | 16+16 → 16 | **L40** |

The layer opens and closes at L40 in feature-major, which is what lets the
32-layer chain run with no decrypt boundary.

### Notes per operator

**RMSNorm** packs all 128 token mean-squares into one *slim* ciphertext, so the
inverse-square-root polynomial is evaluated **once** per layer instead of 16
times. The reduction is a rotate-and-add tree over the feature orbit (stride
`T`, log2(D) steps), and the broadcast back over the orbit is the same tree with
the rotation sign flipped (`_fold_feature_orbit`).

**QKV** pairs adjacent token shards as `x + i·y` before projecting, so 16 real
ciphertexts become 8 complex ones and one diagonal pass produces both. The
weight is a `StructuredLinearWeight` descriptor — the Triton packer emits BSGS
diagonals straight from the model matrices with permutation, scaling and
real/imaginary fusion applied on the fly, never materializing a dense 4096²
physical matrix.

**The attention input adapter** is the one genuinely intricate stage: it turns
carriers into Q_C/K_Δ/V_Δ with a lane/phase step (RoPE folded in), a gather
fused with the first source-feature axis swap, a second axis swap, and — for K —
the Δ shear fused with upper-half reconstruction. Relative output depths are
Q 3 / K 4 / V 3.

**QK and PV** keep each CT-CT product as a degree-2 triplet, accumulate all
feature terms, and relinearize **once** per output ciphertext. Both also fold
complex feature/Δ halves so one shift serves two, halving the rotations.

**Softmax** is polynomial Alg2 in the Δ layout: the causal mask and the row
reduction are public-mask operations on the same grid the scores arrived in.

**PairFM output** is the only layout conversion in the layer, and it is at the
*end* of attention, where it converts O_C into the complex token-pair carriers
W_O consumes: 6400 PT-CT, 864 rotations, depth 3, 16 → 8 ciphertexts.

**MLP** packs gate and up into **one** complex weight, `0.25·(gate + i·up)`, so
a single projection produces both halves; SiLU then evaluates on the real part
and the product with the up half is one CT-CT multiply. The intermediate 14336
width is paged as 4 × 4096, and the whole MLP fits in its limb budget with **no
bootstrap inside it**.

---

## What the packing buys, in one list

- **100% slot density** at the persistent boundary — 524288 values in 16
  ciphertexts, no padding carried across layers.
- **No repack in attention.** Q_C → S_Δ → P_Δ → O_C is one slot contract; the
  only conversion (PairFM) is at the attention/W_O seam and costs 864 rotations.
- **Causal masking is public.** A fixed pattern over delta rows, identical for
  every layer and head.
- **Two heads per ciphertext**, GQA-aligned, so a K/V ciphertext feeds R query
  ciphertexts with no gather.
- **Complex pairing twice over**: adjacent token shards pair for the linear
  passes and for the bootstrap (16 real → 8 complex refreshes), and gate/up pair
  into one projection.
- **Rotation strides are all multiples of `T`** at the residual boundary, so the
  BSGS baby set is small and the hoisted baby rotations are reused across every
  plaintext chunk and output page.

See [`REPRODUCING.md`](REPRODUCING.md) for execution profiles and timing
protocols. [`DESIGN.md`](DESIGN.md) describes runtime components, parallel
execution and the persistent PairFold KV cache consumed by Decode.
