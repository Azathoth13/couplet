// Fused, batched key-switch pieces for EasyFHE (uint64 RNS, N = 2^16).
//
// ModDown of both ciphertext components and every batch item runs in four launches:
//   1. iNTT phase 1 of the special-prime (P) limbs,
//   2. iNTT phase 2 of the P limbs, scaled by [P/p_j]^{-1}_{p_j},
//   3. base conversion P -> Q fused with NTT phase 1,
//   4. NTT phase 2 fused with (x_Q - conv) * P^{-1} and an optional addend.
// EasyFHE runs about five launches per component (iNTT x2, const_mult, base conversion,
// NTT x2, sub-mult). The modular arithmetic and the NTT schedule follow EasyFHE's own
// NttImpl.cu, HRot.cu and ModDown.cu, and every intermediate is fully reduced, so the
// outputs are the same residues as EasyFHE's ModDown.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

namespace sylph_ks {

constexpr int kLogN = 16;
constexpr int kN = 1 << kLogN;
constexpr int kWarp = 32;
constexpr int kLogRadix = kLogN - 6;          // 10
constexpr int kData = 1 << kLogRadix;         // 1024
constexpr int kXThreads = kData / 2;          // 512
constexpr int kXWarps = kXThreads / kWarp;    // 16
constexpr int kMaxP = 16;

struct u128 { uint64_t hi, lo; };

__device__ __forceinline__ u128 mul128(uint64_t a, uint64_t b) {
  u128 r; r.lo = a * b; r.hi = __umul64hi(a, b); return r;
}
__device__ __forceinline__ void add128(const u128 a, u128& r) {
  asm("add.cc.u64 %1, %3, %1;" "addc.cc.u64 %0, %2, %0;" : "+l"(r.hi), "+l"(r.lo) : "l"(a.hi), "l"(a.lo));
}
__device__ __forceinline__ uint64_t barrett128(const u128 in, uint64_t prime, uint64_t ratio, uint64_t k) {
  u128 t1 = mul128(in.lo, ratio);
  u128 t2 = mul128(in.hi, ratio);
  asm("add.cc.u64 %0, %0, %1;" : "+l"(t1.hi) : "l"(t2.lo));
  asm("{addc.cc.u64 %0, %0, %1;}" : "+l"(t2.hi) : "l"((unsigned long)0));
  const unsigned shift = static_cast<unsigned>(k - 64);
  t1.hi >>= shift;
  t2.hi <<= 64 - shift;
  t1.hi += t2.hi;
  t1.hi *= prime;
  uint64_t res = in.lo - t1.hi;
  if (res >= prime) res -= prime;
  return res;
}
__device__ __forceinline__ uint64_t add_mod(uint64_t a, uint64_t b, uint64_t mod) {
  const uint64_t r = a + b; return r >= mod ? r - mod : r;
}
__device__ __forceinline__ uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t mod) {
  return a >= b ? a - b : a + mod - b;
}
__device__ __forceinline__ uint64_t shoup(uint64_t a, uint64_t b, uint64_t b_shoup, uint64_t prime) {
  const uint64_t hi = __umul64hi(b_shoup, a);
  return a * b - hi * prime;
}
__device__ __forceinline__ uint64_t shoup_full(uint64_t a, uint64_t b, uint64_t b_shoup, uint64_t prime) {
  uint64_t r = shoup(a, b, b_shoup, prime);
  return r >= prime ? r - prime : r;
}

// ---- forward NTT (Cooley-Tukey, lazy in [0, 4p)) ----
__device__ __forceinline__ void butterfly(uint64_t& a, uint64_t& b, uint64_t root, uint64_t root_shoup,
                                          uint64_t prime, uint64_t two_p) {
  const uint64_t product = shoup(b, root, root_shoup, prime);
  if (a >= two_p) a -= two_p;
  b = a + (two_p - product);
  a += product;
}
__device__ __forceinline__ void radix8(uint64_t (&x)[8], uint32_t off, const uint64_t* __restrict__ w,
                                       const uint64_t* __restrict__ ws, uint64_t p, uint64_t tp) {
  {
    const uint64_t r = w[off], rs = ws[off];
    butterfly(x[0], x[4], r, rs, p, tp); butterfly(x[1], x[5], r, rs, p, tp);
    butterfly(x[2], x[6], r, rs, p, tp); butterfly(x[3], x[7], r, rs, p, tp);
  }
  {
    const uint32_t o = 2 * off;
    butterfly(x[0], x[2], w[o], ws[o], p, tp); butterfly(x[1], x[3], w[o], ws[o], p, tp);
    butterfly(x[4], x[6], w[o + 1], ws[o + 1], p, tp); butterfly(x[5], x[7], w[o + 1], ws[o + 1], p, tp);
  }
  {
    const uint32_t o = 4 * off;
    butterfly(x[0], x[1], w[o], ws[o], p, tp); butterfly(x[2], x[3], w[o + 1], ws[o + 1], p, tp);
    butterfly(x[4], x[5], w[o + 2], ws[o + 2], p, tp); butterfly(x[6], x[7], w[o + 3], ws[o + 3], p, tp);
  }
}
template <int Rounds>
__device__ __forceinline__ void warp_butterfly(uint64_t& a, uint64_t& b, uint32_t& stage, uint32_t lane,
                                               const uint64_t* __restrict__ w, const uint64_t* __restrict__ ws,
                                               uint64_t p, uint64_t tp) {
  butterfly(a, b, w[stage], ws[stage], p, tp);
#pragma unroll
  for (int shift = Rounds - 2; shift >= 0; --shift) {
    const uint32_t off = 1u << shift;
    const bool lower = (lane & off) == 0;
    uint64_t ex = lower ? b : a;
    ex = __shfl_xor_sync(0xFFFFFFFF, ex, off);
    if (lower) b = ex; else a = ex;
    stage <<= 1;
    const uint32_t idx = stage + (lane >> shift);
    butterfly(a, b, w[idx], ws[idx], p, tp);
  }
}
__device__ __forceinline__ uint64_t reduce4p(uint64_t v, uint64_t p, uint64_t tp) {
  if (v >= tp) v -= tp;
  if (v >= p) v -= p;
  return v;
}
// NTT phase 2 of one polynomial (EasyFHE NTTXPointPhase2 schedule); returns the fully
// reduced pair at positions block * 1024 + 2 * threadIdx.x + {0, 1}.
__device__ __forceinline__ ulonglong2 ntt_phase2_pair(const uint64_t* __restrict__ poly, uint32_t block,
                                                      const uint64_t* __restrict__ w, const uint64_t* __restrict__ ws,
                                                      uint64_t p, uint64_t tp, uint64_t (&tile)[2][kXWarps][kWarp + 1]) {
  uint32_t stage = 64 + block;
  uint64_t a = poly[block * kData + threadIdx.x];
  uint64_t b = poly[block * kData + threadIdx.x + kXThreads];
  tile[0][threadIdx.x % kXWarps][threadIdx.x / kXWarps] = a;
  tile[1][threadIdx.x % kXWarps][threadIdx.x / kXWarps] = b;
  __syncthreads();
  uint32_t lane = threadIdx.x % kWarp, group = threadIdx.x / kWarp;
  a = tile[0][group][lane];
  b = tile[1][group][lane];
  warp_butterfly<6>(a, b, stage, lane, w, ws, p, tp);
  tile[0][group][lane] = a;
  tile[1][group][lane] = b;
  __syncthreads();
  constexpr int kSecond = kData / kWarp / 4;  // 8
  lane = threadIdx.x % kSecond;
  group = threadIdx.x / kSecond;
  const uint32_t half = group >> 1;
  if ((group & 1) == 0) {
    a = tile[0][lane][half];
    b = tile[0][lane + kSecond][half];
  } else {
    a = tile[1][lane][half];
    b = tile[1][lane + kSecond][half];
  }
  stage = stage * 2 + group;
  warp_butterfly<kLogRadix - 6>(a, b, stage, lane, w, ws, p, tp);
  return {reduce4p(a, p, tp), reduce4p(b, p, tp)};
}

// Two polynomials through NTT phase 2 together (one tile each, so two barriers instead of
// four); each transform is exactly ntt_phase2_pair.
__device__ __forceinline__ void ntt_phase2_pair2(const uint64_t* __restrict__ poly0,
                                                 const uint64_t* __restrict__ poly1, uint32_t block,
                                                 const uint64_t* __restrict__ w, const uint64_t* __restrict__ ws,
                                                 uint64_t p, uint64_t tp, uint64_t (&t0)[2][kXWarps][kWarp + 1],
                                                 uint64_t (&t1)[2][kXWarps][kWarp + 1], ulonglong2& r0,
                                                 ulonglong2& r1) {
  uint32_t stage = 64 + block;
  uint64_t a0 = poly0[block * kData + threadIdx.x], b0 = poly0[block * kData + threadIdx.x + kXThreads];
  uint64_t a1 = poly1[block * kData + threadIdx.x], b1 = poly1[block * kData + threadIdx.x + kXThreads];
  t0[0][threadIdx.x % kXWarps][threadIdx.x / kXWarps] = a0;
  t0[1][threadIdx.x % kXWarps][threadIdx.x / kXWarps] = b0;
  t1[0][threadIdx.x % kXWarps][threadIdx.x / kXWarps] = a1;
  t1[1][threadIdx.x % kXWarps][threadIdx.x / kXWarps] = b1;
  __syncthreads();
  uint32_t lane = threadIdx.x % kWarp, group = threadIdx.x / kWarp;
  a0 = t0[0][group][lane]; b0 = t0[1][group][lane];
  a1 = t1[0][group][lane]; b1 = t1[1][group][lane];
  uint32_t stage1 = stage;
  warp_butterfly<6>(a0, b0, stage, lane, w, ws, p, tp);
  warp_butterfly<6>(a1, b1, stage1, lane, w, ws, p, tp);
  t0[0][group][lane] = a0; t0[1][group][lane] = b0;
  t1[0][group][lane] = a1; t1[1][group][lane] = b1;
  __syncthreads();
  constexpr int kSecond = kData / kWarp / 4;  // 8
  lane = threadIdx.x % kSecond;
  group = threadIdx.x / kSecond;
  const uint32_t half = group >> 1;
  if ((group & 1) == 0) {
    a0 = t0[0][lane][half]; b0 = t0[0][lane + kSecond][half];
    a1 = t1[0][lane][half]; b1 = t1[0][lane + kSecond][half];
  } else {
    a0 = t0[1][lane][half]; b0 = t0[1][lane + kSecond][half];
    a1 = t1[1][lane][half]; b1 = t1[1][lane + kSecond][half];
  }
  stage = stage * 2 + group;
  stage1 = stage;
  warp_butterfly<kLogRadix - 6>(a0, b0, stage, lane, w, ws, p, tp);
  warp_butterfly<kLogRadix - 6>(a1, b1, stage1, lane, w, ws, p, tp);
  r0 = {reduce4p(a0, p, tp), reduce4p(b0, p, tp)};
  r1 = {reduce4p(a1, p, tp), reduce4p(b1, p, tp)};
}

// ---- inverse NTT (Gentleman-Sande with halving, EasyFHE iNTT schedule) ----
__device__ __forceinline__ void ibutterfly(uint64_t& x, uint64_t& y, uint64_t w, uint64_t w_, uint64_t p,
                                           uint64_t tp) {
  const uint64_t t = tp - y + x;
  uint64_t nx = x + y;
  if (nx >= tp) nx -= tp;
  if (t & 1) nx += p;
  x = nx >> 1;
  y = shoup(t, w, w_, p);
}
template <int Rounds>
__device__ __forceinline__ void iwarp_butterfly(uint64_t& a, uint64_t& b, uint32_t& stage, uint32_t lane,
                                                const uint64_t* __restrict__ w, const uint64_t* __restrict__ ws,
                                                uint64_t p, uint64_t tp) {
  ibutterfly(a, b, w[stage + lane], ws[stage + lane], p, tp);
#pragma unroll
  for (int shift = 1; shift < Rounds; ++shift) {
    const uint32_t off = 1u << shift;
    const bool lower = (lane & (off - 1)) < (off >> 1);
    uint64_t ex = lower ? b : a;
    ex = __shfl_xor_sync(0xFFFFFFFF, ex, off >> 1);
    if (lower) b = ex; else a = ex;
    stage >>= 1;
    const uint32_t idx = stage + (lane >> shift);
    ibutterfly(a, b, w[idx], ws[idx], p, tp);
  }
}
__device__ __forceinline__ void iradix8(uint64_t (&x)[8], uint32_t off, const uint64_t* __restrict__ w,
                                        const uint64_t* __restrict__ ws, uint64_t p, uint64_t tp) {
  {
    const uint32_t o = 4 * off;
    ibutterfly(x[0], x[1], w[o], ws[o], p, tp); ibutterfly(x[2], x[3], w[o + 1], ws[o + 1], p, tp);
    ibutterfly(x[4], x[5], w[o + 2], ws[o + 2], p, tp); ibutterfly(x[6], x[7], w[o + 3], ws[o + 3], p, tp);
  }
  {
    const uint32_t o = 2 * off;
    ibutterfly(x[0], x[2], w[o], ws[o], p, tp); ibutterfly(x[1], x[3], w[o], ws[o], p, tp);
    ibutterfly(x[4], x[6], w[o + 1], ws[o + 1], p, tp); ibutterfly(x[5], x[7], w[o + 1], ws[o + 1], p, tp);
  }
  {
    const uint64_t r = w[off], rs = ws[off];
    ibutterfly(x[0], x[4], r, rs, p, tp); ibutterfly(x[1], x[5], r, rs, p, tp);
    ibutterfly(x[2], x[6], r, rs, p, tp); ibutterfly(x[3], x[7], r, rs, p, tp);
  }
}

// Pointers of the two components; blockIdx.z = comp * batch + item.
struct Pair {
  const uint64_t* in0; const uint64_t* in1; long long in_stride;
  uint64_t* out0; uint64_t* out1; long long out_stride;
  const uint64_t* add0; const uint64_t* add1; long long add_stride;
};

// Rotation key switch of the batch: the extended component of item b is, at NTT slot j,
//   sum_beta digits[b][beta][limb][src] * key[beta][key_limb][src],  src = map[j]
// (EasyFHE's innerproduct_pairwise followed by finalize_fast_rotation_ext), and component 0
// additionally carries c0[b][limb][src] * P on the Q limbs.
struct Rot {
  const uint64_t* digits; long long d_stride; int length;   // [batch][beta][length][N]
  const uint64_t* kb; const uint64_t* ka; int mult_len; int sms; int beta;
  const int* map;                                           // null: identity
  const uint64_t* c0; long long c0_stride;                  // null: no c0 term
  // p1 != 0: `digits` holds NTT phase 1 outputs for the limbs a digit does not own (the own
  // limbs [d alpha, d alpha + alpha) of digit d are the switched polynomial itself, read from
  // `own` [batch][>= limbs][N]); phase 2 runs in the consumer (digit_products).
  const uint64_t* own; long long own_stride; int limbs; int alpha; int p1;
  const uint64_t* own_b; long long own_b_stride;   // non-null: the own limbs are own * own_b (mod p)
};

// The pair (pos, pos + 1) of item's own polynomial (own, or own * own_b).
__device__ __forceinline__ ulonglong2 rot_own_pair(const Rot& r, int item, long long pos, uint64_t p, uint64_t ratio,
                                                   uint64_t bk) {
  ulonglong2 v = *reinterpret_cast<const ulonglong2*>(r.own + item * r.own_stride + pos);
  if (r.own_b) {
    const ulonglong2 b = *reinterpret_cast<const ulonglong2*>(r.own_b + item * r.own_b_stride + pos);
    v.x = barrett128(mul128(v.x, b.x), p, ratio, bk);
    v.y = barrett128(mul128(v.y, b.y), p, ratio, bk);
  }
  return v;
}

// Both components of the rotated key product of item at (limb, src).
__device__ __forceinline__ void rot_inner2(const Rot& r, int item, int limb, int key_limb, int src, uint64_t p,
                                           uint64_t ratio, uint64_t bk, uint64_t& o0, uint64_t& o1) {
  const uint64_t* d = r.digits + item * r.d_stride + (long long)limb * kN + src;
  const long long ko = (long long)key_limb * kN + src;
  const uint64_t* kb = r.kb + ko;
  const uint64_t* ka = r.ka + ko;
  const long long ds = (long long)r.length * kN, ks = (long long)r.mult_len * kN;
  u128 acc0{0, 0}, acc1{0, 0};
  for (int i = 0; i < r.beta; ++i) {
    const uint64_t x = d[i * ds];
    add128(mul128(x, kb[i * ks]), acc0);
    add128(mul128(x, ka[i * ks]), acc1);
  }
  o0 = barrett128(acc0, p, ratio, bk);
  o1 = barrett128(acc1, p, ratio, bk);
}

// Both components of the key product of item at every position of tile `t` (1024 slots) of
// (limb, key_limb), into prod0 / prod1 (shared, indexed by the position within the tile):
//   prod[s] = sum_beta digit[beta][limb][s] * key[beta][key_limb][s]
// for phase-1 digits (Rot::p1): the digit of a non-own limb is put through NTT phase 2 here,
// the own limb is read from Rot::own. Same products, same summation order and one Barrett
// reduction as rot_inner2, so the residues are identical.
__device__ __forceinline__ void digit_products(const Rot& r, int item, int limb, int key_limb, int t, uint64_t p,
                                               uint64_t ratio, uint64_t bk, const uint64_t* __restrict__ w,
                                               const uint64_t* __restrict__ w_,
                                               uint64_t (&tile)[2][kXWarps][kWarp + 1],
                                               uint64_t (&tile2)[2][kXWarps][kWarp + 1], uint64_t* prod0,
                                               uint64_t* prod1) {
  const int off = t * kData + 2 * threadIdx.x;
  const long long ks = (long long)r.mult_len * kN, ko = (long long)key_limb * kN + off;
  u128 b0{0, 0}, b1{0, 0}, a0{0, 0}, a1{0, 0};
  auto mac = [&](int i, const ulonglong2 v) {
    const ulonglong2 kb = *reinterpret_cast<const ulonglong2*>(r.kb + i * ks + ko);
    const ulonglong2 ka = *reinterpret_cast<const ulonglong2*>(r.ka + i * ks + ko);
    add128(mul128(v.x, kb.x), b0); add128(mul128(v.y, kb.y), b1);
    add128(mul128(v.x, ka.x), a0); add128(mul128(v.y, ka.y), a1);
  };
  auto own = [&](int i) { return limb < r.limbs && limb / r.alpha == i; };
  auto digit = [&](int i) { return r.digits + item * r.d_stride + ((long long)i * r.length + limb) * kN; };
  for (int i = 0; i < r.beta;) {
    if (own(i)) {
      mac(i, rot_own_pair(r, item, (long long)limb * kN + off, p, ratio, bk));
      ++i;
    } else if (i + 1 < r.beta && !own(i + 1)) {
      // two digits through phase 2 together (tile and tile2), accumulated in order i, i + 1
      ulonglong2 v0, v1;
      ntt_phase2_pair2(digit(i), digit(i + 1), t, w, w_, p, p << 1, tile, tile2, v0, v1);
      __syncthreads();
      mac(i, v0); mac(i + 1, v1);
      i += 2;
    } else {
      mac(i, ntt_phase2_pair(digit(i), t, w, w_, p, p << 1, tile));
      __syncthreads();
      ++i;
    }
  }
  prod0[2 * threadIdx.x] = barrett128(b0, p, ratio, bk);
  prod0[2 * threadIdx.x + 1] = barrett128(b1, p, ratio, bk);
  prod1[2 * threadIdx.x] = barrett128(a0, p, ratio, bk);
  prod1[2 * threadIdx.x + 1] = barrett128(a1, p, ratio, bk);
  __syncthreads();
}

// Source positions of slots (j, j + 1) of tile blk under the gather map (identity if null); with
// phase-1 digits every position of a tile must come from one source tile (checked by the caller).
__device__ __forceinline__ void rot_sources(const Rot& r, int blk, int j, int& s0, int& s1, int& src_tile) {
  if (r.map) { s0 = r.map[j]; s1 = r.map[j + 1]; src_tile = r.map[blk * kData] >> kLogRadix; }
  else { s0 = j; s1 = j + 1; src_tile = blk; }
}

// iNTT phase 1 (1024-point rows) of the pair (a, b) held by each thread into out.
__device__ __forceinline__ void intt1_rows(uint64_t a, uint64_t b, uint32_t blk, uint64_t* out,
                                           const uint64_t* __restrict__ w,
                                           const uint64_t* __restrict__ w_, uint64_t p, uint64_t tp,
                                           uint64_t (&tile)[2][kXWarps][kWarp + 1]) {
  uint32_t stage = (1u << (kLogN - 1)) + blk * kXThreads;
  iwarp_butterfly<6>(a, b, stage, threadIdx.x, w, w_, p, tp);
  uint32_t lane = threadIdx.x % kWarp, group = threadIdx.x / kWarp;
  tile[0][group][lane] = a;
  tile[1][group][lane] = b;
  __syncthreads();
  constexpr int kSecond = kData / kWarp / 4;
  const uint32_t lid = threadIdx.x / kSecond, gid = threadIdx.x % kSecond;
  const uint32_t plane = lid >> 5, col = lid & (kWarp - 1);
  a = tile[plane][2 * gid][col];
  b = tile[plane][2 * gid + 1][col];
  stage >>= 1;
  iwarp_butterfly<kLogRadix - 6>(a, b, stage, gid, w, w_, p, tp);
  tile[plane][gid][col] = a;
  tile[plane][gid + kSecond][col] = b;
  __syncthreads();
  const uint32_t oc = threadIdx.x % (2 * kWarp), op = oc >> 5, ol = oc & (kWarp - 1);
  out[blk * kData + threadIdx.x] = tile[op][threadIdx.x / (2 * kWarp)][ol];
  out[blk * kData + threadIdx.x + kXThreads] = tile[op][(threadIdx.x + kXThreads) / (2 * kWarp)][ol];
}

// Two (a, b) pairs through iNTT phase 1 together into out0 / out1 (one tile each): each
// transform is exactly intt1_rows, with two barriers instead of five.
__device__ __forceinline__ void intt1_rows2(uint64_t a0, uint64_t b0, uint64_t a1, uint64_t b1, uint32_t blk,
                                            uint64_t* out0, uint64_t* out1, const uint64_t* __restrict__ w,
                                            const uint64_t* __restrict__ w_, uint64_t p, uint64_t tp,
                                            uint64_t (&t0)[2][kXWarps][kWarp + 1],
                                            uint64_t (&t1)[2][kXWarps][kWarp + 1]) {
  uint32_t stage = (1u << (kLogN - 1)) + blk * kXThreads, stage1 = stage;
  iwarp_butterfly<6>(a0, b0, stage, threadIdx.x, w, w_, p, tp);
  iwarp_butterfly<6>(a1, b1, stage1, threadIdx.x, w, w_, p, tp);
  uint32_t lane = threadIdx.x % kWarp, group = threadIdx.x / kWarp;
  t0[0][group][lane] = a0; t0[1][group][lane] = b0;
  t1[0][group][lane] = a1; t1[1][group][lane] = b1;
  __syncthreads();
  constexpr int kSecond = kData / kWarp / 4;
  const uint32_t lid = threadIdx.x / kSecond, gid = threadIdx.x % kSecond;
  const uint32_t plane = lid >> 5, col = lid & (kWarp - 1);
  a0 = t0[plane][2 * gid][col]; b0 = t0[plane][2 * gid + 1][col];
  a1 = t1[plane][2 * gid][col]; b1 = t1[plane][2 * gid + 1][col];
  stage >>= 1;
  stage1 = stage;
  iwarp_butterfly<kLogRadix - 6>(a0, b0, stage, gid, w, w_, p, tp);
  iwarp_butterfly<kLogRadix - 6>(a1, b1, stage1, gid, w, w_, p, tp);
  t0[plane][gid][col] = a0; t0[plane][gid + kSecond][col] = b0;
  t1[plane][gid][col] = a1; t1[plane][gid + kSecond][col] = b1;
  __syncthreads();
  const uint32_t oc = threadIdx.x % (2 * kWarp), op = oc >> 5, ol = oc & (kWarp - 1);
  const uint32_t r0 = threadIdx.x / (2 * kWarp), r1 = (threadIdx.x + kXThreads) / (2 * kWarp);
  out0[blk * kData + threadIdx.x] = t0[op][r0][ol];
  out0[blk * kData + threadIdx.x + kXThreads] = t0[op][r1][ol];
  out1[blk * kData + threadIdx.x] = t1[op][r0][ol];
  out1[blk * kData + threadIdx.x + kXThreads] = t1[op][r1][ol];
}

// 1. iNTT phase 1 of the P limbs into ws[(z * K + k) * N].
__global__ void __launch_bounds__(kXThreads) md_intt1(Pair io, int batch, int limbs, int K, int L, uint64_t* ws,
                                                      const uint64_t* __restrict__ primes,
                                                      const uint64_t* __restrict__ iroots,
                                                      const uint64_t* __restrict__ iroots_s) {
  const int z = blockIdx.z, comp = z / batch, item = z - comp * batch, k = blockIdx.y;
  const uint64_t* in = (comp ? io.in1 : io.in0) + item * io.in_stride + (long long)(limbs + k) * kN;
  const uint64_t p = primes[L + k];
  const ulonglong2 v = reinterpret_cast<const ulonglong2*>(in)[blockIdx.x * kXThreads + threadIdx.x];
  __shared__ uint64_t tile[2][kXWarps][kWarp + 1];
  intt1_rows(v.x, v.y, blockIdx.x, ws + ((long long)z * K + k) * kN, iroots + (long long)(L + k) * kN,
             iroots_s + (long long)(L + k) * kN, p, p << 1, tile);
}

// 1'. Rotation: P limbs of the rotated key product of both components, then iNTT phase 1.
__global__ void __launch_bounds__(kXThreads, 2) rot_intt1(Rot rot, int batch, int limbs, int K, int L, uint64_t* ws,
                                                       const uint64_t* __restrict__ primes,
                                                       const uint64_t* __restrict__ ratios,
                                                       const uint64_t* __restrict__ ks,
                                                       const uint64_t* __restrict__ roots,
                                                       const uint64_t* __restrict__ roots_s,
                                                       const uint64_t* __restrict__ iroots,
                                                       const uint64_t* __restrict__ iroots_s) {
  // Items vary fastest over the grid so that consecutive blocks share the key reads in L2.
  const int item = blockIdx.x, blk = blockIdx.y, k = blockIdx.z;
  const uint64_t p = primes[L + k], ratio = ratios[L + k], bk = ks[L + k];
  const int pos = 2 * (blk * kXThreads + threadIdx.x);
  __shared__ uint64_t tile[2][kXWarps][kWarp + 1];
  __shared__ uint64_t tile2[2][kXWarps][kWarp + 1];
  uint64_t a0, a1, b0, b1;
  int s0, s1, st;
  rot_sources(rot, blk, pos, s0, s1, st);
  if (rot.p1) {
    __shared__ uint64_t prod0[kData], prod1[kData];
    digit_products(rot, item, limbs + k, rot.sms + k, st, p, ratio, bk, roots + (long long)(L + k) * kN,
                   roots_s + (long long)(L + k) * kN, tile, tile2, prod0, prod1);
    const int base = st * kData;
    a0 = prod0[s0 - base]; b0 = prod0[s1 - base]; a1 = prod1[s0 - base]; b1 = prod1[s1 - base];
  } else {
    rot_inner2(rot, item, limbs + k, rot.sms + k, s0, p, ratio, bk, a0, a1);
    rot_inner2(rot, item, limbs + k, rot.sms + k, s1, p, ratio, bk, b0, b1);
  }
  const uint64_t* w = iroots + (long long)(L + k) * kN;
  const uint64_t* w_ = iroots_s + (long long)(L + k) * kN;
  intt1_rows2(a0, b0, a1, b1, blk, ws + ((long long)item * K + k) * kN, ws + ((long long)(batch + item) * K + k) * kN,
              w, w_, p, p << 1, tile, tile2);
}

// 2. iNTT phase 2 (64-point columns) in place, then times hat_inv[k] (Shoup).
__global__ void __launch_bounds__(64) md_intt2(int K, int L, uint64_t* ws, const uint64_t* __restrict__ primes,
                                               const uint64_t* __restrict__ iroots,
                                               const uint64_t* __restrict__ iroots_s,
                                               const uint64_t* __restrict__ hat, const uint64_t* __restrict__ hat_s) {
  const int k = blockIdx.y;
  uint64_t* poly = ws + ((long long)blockIdx.z * K + k) * kN;
  const uint64_t* w = iroots + (long long)(L + k) * kN;
  const uint64_t* w_ = iroots_s + (long long)(L + k) * kN;
  const uint64_t p = primes[L + k], tp = p << 1;
  const int group = threadIdx.x / 8, lane = threadIdx.x % 8;
  const int n0 = 8 * blockIdx.x + lane;
  uint64_t x[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = poly[group * 8192 + j * 1024 + n0];
  iradix8(x, 8 + group, w, w_, p, tp);
  __shared__ uint64_t tr[8][9][9];
#pragma unroll
  for (int j = 0; j < 8; ++j) tr[lane][j][group] = x[j];
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = tr[lane][group][j];
  iradix8(x, 1, w, w_, p, tp);
  if (hat == nullptr) {
#pragma unroll
    for (int j = 0; j < 8; ++j) poly[j * 8192 + group * 1024 + n0] = x[j] >= p ? x[j] - p : x[j];
    return;
  }
  const uint64_t h = hat[k], hs = hat_s[k];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    uint64_t v = x[j] >= p ? x[j] - p : x[j];
    poly[j * 8192 + group * 1024 + n0] = shoup_full(v, h, hs, p);
  }
}

// Relinearize + rescale (EasyFHE hmul_relin_rescale) state shared by its kernels.
struct HMul {
  const uint64_t* raw;   // [3][batch][limbs][N]: c0*d0, c0*d1 + c1*d0, c1*d1; null: from the factors
  const uint64_t* t;     // [2][N]: last limb of the relinearized pair in coefficient form
  const uint64_t* diffs; uint64_t half;                   // rescale switch of the last limb
  const uint64_t* csw; const uint64_t* csw_s;             // switch constant per q_j
  const uint64_t* cfin; const uint64_t* cfin_s;           // q_last^{-1} mod q_j
  const uint64_t* inv; const uint64_t* inv_s;             // P^{-1} mod q_j
  const uint64_t* post0; const uint64_t* post1; const uint64_t* scalar;
  uint64_t* out0; uint64_t* out1;
  int dbl;
  int batch; long long raw_stride, post_stride, out_stride;   // per-item strides (words)
  const uint64_t* c0; const uint64_t* c1; const uint64_t* d0; const uint64_t* d1; long long sc, sd;   // raw == null
};

// raw[0] and raw[1] of item at (pos, pos + 1): from the stored tensor product or the factors.
__device__ __forceinline__ void hm_raw01(const HMul& hm, int item, long long pos, uint64_t p, uint64_t ratio,
                                         uint64_t bk, ulonglong2& r0, ulonglong2& r1) {
  if (hm.raw) {
    const uint64_t* raw = hm.raw + item * hm.raw_stride;
    const long long ln = (long long)hm.batch * hm.raw_stride;
    r0 = *reinterpret_cast<const ulonglong2*>(raw + pos);
    r1 = *reinterpret_cast<const ulonglong2*>(raw + ln + pos);
    return;
  }
  const ulonglong2 a0 = *reinterpret_cast<const ulonglong2*>(hm.c0 + item * hm.sc + pos);
  const ulonglong2 a1 = *reinterpret_cast<const ulonglong2*>(hm.c1 + item * hm.sc + pos);
  const ulonglong2 b0 = *reinterpret_cast<const ulonglong2*>(hm.d0 + item * hm.sd + pos);
  const ulonglong2 b1 = *reinterpret_cast<const ulonglong2*>(hm.d1 + item * hm.sd + pos);
  r0.x = barrett128(mul128(a0.x, b0.x), p, ratio, bk);
  r0.y = barrett128(mul128(a0.y, b0.y), p, ratio, bk);
  r1.x = add_mod(barrett128(mul128(a0.x, b1.x), p, ratio, bk), barrett128(mul128(a1.x, b0.x), p, ratio, bk), p);
  r1.y = add_mod(barrett128(mul128(a0.y, b1.y), p, ratio, bk), barrett128(mul128(a1.y, b0.y), p, ratio, bk), p);
}

// 3. base conversion P -> q_limb fused with NTT phase 1, into ys[(z * limbs + limb) * N]
// for the G limbs limb0 + G * blockIdx.y + g (< limb0 + nout): the K input columns are read
// once per block and feed G accumulators (same products and order per limb as one at a time).
// With Sw (relinearize + rescale), the polynomial put through the NTT is
// switch(t) * csw - conv * P^{-1} * cfin (doubled conv term if dbl): the NTT is linear, so this
// one transform carries both the ModDown conversion and the rescale switch.
template <int G, bool Sw>
__global__ void __launch_bounds__(64) md_conv_ntt1(int limbs, int K, const uint64_t* ws, uint64_t* ys,
                                                   const uint64_t* __restrict__ prod_q,
                                                   const uint64_t* __restrict__ primes,
                                                   const uint64_t* __restrict__ ratios,
                                                   const uint64_t* __restrict__ ks,
                                                   const uint64_t* __restrict__ roots,
                                                   const uint64_t* __restrict__ roots_s, HMul hm, int limb0,
                                                   int nout) {
  const int lb = limb0 + G * blockIdx.y, z = blockIdx.z, ng = min(G, limb0 + nout - lb);
  __shared__ uint64_t hat[G][kMaxP];
  if (threadIdx.x < G * kMaxP) {
    const int g = threadIdx.x / kMaxP, i = threadIdx.x % kMaxP;
    if (g < ng && i < K) hat[g][i] = prod_q[(lb + g) * K + i];
  }
  __syncthreads();
  const uint64_t* pl = ws + (long long)z * K * kN;
  const int group = threadIdx.x / 8, lane = threadIdx.x % 8;
  const int n0 = 8 * blockIdx.x + lane;
  // The 8 rows of a thread are converted kJ at a time: the 128-bit accumulators of one slice
  // are reduced to 64 bits before the next slice starts, which keeps the register count (and
  // so the occupancy) in check. Every accumulator sees the same products in the same order.
  constexpr int kJ = 4;
  uint64_t xs[G][8];
#pragma unroll
  for (int j0 = 0; j0 < 8; j0 += kJ) {
    u128 acc[G][kJ];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int j = 0; j < kJ; ++j) acc[g][j] = u128{0, 0};
    for (int i = 0; i < K; ++i) {
      const uint64_t* row = pl + (long long)i * kN + group * 1024 + n0;
      uint64_t v[kJ];
#pragma unroll
      for (int j = 0; j < kJ; ++j) v[j] = row[(j0 + j) * 8192];
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const uint64_t h = hat[g][i];
#pragma unroll
        for (int j = 0; j < kJ; ++j) add128(mul128(v[j], h), acc[g][j]);
      }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
      if (g >= ng) break;
      const int limb = lb + g;
      const uint64_t p = primes[limb], ratio = ratios[limb], bk = ks[limb];
#pragma unroll
      for (int j = 0; j < kJ; ++j) xs[g][j0 + j] = barrett128(acc[g][j], p, ratio, bk);
    }
  }
  __shared__ uint64_t tr[8][9][9];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (g >= ng) break;
    const int limb = lb + g;
    uint64_t* out = ys + ((long long)z * limbs + limb) * kN;
    const uint64_t* w = roots + (long long)limb * kN;
    const uint64_t* w_ = roots_s + (long long)limb * kN;
    const uint64_t p = primes[limb], tp = p << 1;
    uint64_t x[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      x[j] = xs[g][j];
      if constexpr (Sw) {
        const int c = j * 8192 + group * 1024 + n0;
        uint64_t m = shoup_full(x[j], hm.inv[limb], hm.inv_s[limb], p);
        if (hm.dbl) m = add_mod(m, m, p);
        m = shoup_full(m, hm.cfin[limb], hm.cfin_s[limb], p);
        const uint64_t tv = hm.t[(long long)z * kN + c];
        uint64_t sv = tv + (tv > hm.half ? hm.diffs[limb] : uint64_t{0});
        if (sv >= p) sv -= p;
        sv = shoup_full(sv, hm.csw[limb], hm.csw_s[limb], p);
        x[j] = sub_mod(sv, m, p);
      }
    }
    radix8(x, 1, w, w_, p, tp);
    if (g) __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) tr[lane][j][group] = x[j];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = tr[lane][group][j];
    radix8(x, 8 + group, w, w_, p, tp);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[group * 8192 + j * 1024 + n0] = x[j];
  }
}

struct Tables {
  const uint64_t *primes, *ratios, *ks, *roots, *roots_s, *iroots, *iroots_s, *hat, *hat_s, *prod_q, *inv, *inv_s;
};

// md_conv_ntt1 over the nout limbs from limb0 of z items, up to kConvG limbs per block.
constexpr int kConvG = 3;
template <bool Sw>
static void launch_md_conv_ntt1(int limbs, int K, const uint64_t* ws, uint64_t* ys, const Tables& t, const HMul& hm,
                                int limb0, int nout, int z, cudaStream_t stream) {
  const int g = std::min(kConvG, nout);
  const dim3 grid(kN / 512, (nout + g - 1) / g, z);
#define MD_CONV(GG) md_conv_ntt1<GG, Sw><<<grid, 64, 0, stream>>>(limbs, K, ws, ys, t.prod_q, t.primes, t.ratios, t.ks, \
                                                                  t.roots, t.roots_s, hm, limb0, nout)
  switch (g) {
    case 1: MD_CONV(1); break;
    case 2: MD_CONV(2); break;
    default: MD_CONV(3); break;
  }
#undef MD_CONV
}

// 4. NTT phase 2 fused with out = (x_Q - conv) * P^{-1} (+ addend).
template <bool Add>
__global__ void __launch_bounds__(kXThreads) md_ntt2_finish(Pair io, int batch, int limbs, const uint64_t* ys,
                                                            const uint64_t* __restrict__ inv,
                                                            const uint64_t* __restrict__ inv_s,
                                                            const uint64_t* __restrict__ primes,
                                                            const uint64_t* __restrict__ roots,
                                                            const uint64_t* __restrict__ roots_s) {
  const int limb = blockIdx.y, z = blockIdx.z, comp = z / batch, item = z - comp * batch;
  const uint64_t p = primes[limb], tp = p << 1;
  __shared__ uint64_t tile[2][kXWarps][kWarp + 1];
  const ulonglong2 c = ntt_phase2_pair(ys + ((long long)z * limbs + limb) * kN, blockIdx.x,
                                       roots + (long long)limb * kN, roots_s + (long long)limb * kN, p, tp, tile);
  const long long pos = (long long)limb * kN + blockIdx.x * kData + 2 * threadIdx.x;
  const uint64_t s = inv[limb], ss = inv_s[limb];
  ulonglong2 r;
  const ulonglong2 x = *reinterpret_cast<const ulonglong2*>((comp ? io.in1 : io.in0) + item * io.in_stride + pos);
  r.x = shoup_full(sub_mod(x.x, c.x, p), s, ss, p);
  r.y = shoup_full(sub_mod(x.y, c.y, p), s, ss, p);
  if constexpr (Add) {
    const ulonglong2 a = *reinterpret_cast<const ulonglong2*>((comp ? io.add1 : io.add0) + item * io.add_stride + pos);
    r.x = add_mod(r.x, a.x, p);
    r.y = add_mod(r.y, a.y, p);
  }
  *reinterpret_cast<ulonglong2*>((comp ? io.out1 : io.out0) + item * io.out_stride + pos) = r;
}

// 4'. Rotation: NTT phase 2 of both conversions, then out = (keyQ - conv) * P^{-1} (+ c0 on
// component 0, which is c0 * P brought down) (+ addend).
template <bool Add>
__global__ void __launch_bounds__(kXThreads, 2) rot_finish(Pair io, Rot rot, int batch, int limbs, const uint64_t* ys,
                                                        const uint64_t* __restrict__ inv,
                                                        const uint64_t* __restrict__ inv_s,
                                                        const uint64_t* __restrict__ primes,
                                                        const uint64_t* __restrict__ ratios,
                                                        const uint64_t* __restrict__ ks,
                                                        const uint64_t* __restrict__ roots,
                                                        const uint64_t* __restrict__ roots_s) {
  const int item = blockIdx.x, blk = blockIdx.y, limb = blockIdx.z;
  const uint64_t p = primes[limb], tp = p << 1;
  const uint64_t* w = roots + (long long)limb * kN;
  const uint64_t* w_ = roots_s + (long long)limb * kN;
  __shared__ uint64_t tile[2][kXWarps][kWarp + 1];
  __shared__ uint64_t tile2[2][kXWarps][kWarp + 1];
  __shared__ uint64_t prod0[kData], prod1[kData];
  const int j = blk * kData + 2 * threadIdx.x;
  int s0, s1, st;
  rot_sources(rot, blk, j, s0, s1, st);
  const uint64_t ratio = ratios[limb], bk = ks[limb], s = inv[limb], ss = inv_s[limb];
  // Key products first (shared), then the two conversion transforms: fewer live registers.
  int i0 = 2 * threadIdx.x, i1 = 2 * threadIdx.x + 1;
  if (rot.p1) {
    digit_products(rot, item, limb, limb, st, p, ratio, bk, w, w_, tile, tile2, prod0, prod1);
    i0 = s0 - st * kData; i1 = s1 - st * kData;
  } else {
    rot_inner2(rot, item, limb, limb, s0, p, ratio, bk, prod0[i0], prod1[i0]);
    rot_inner2(rot, item, limb, limb, s1, p, ratio, bk, prod0[i1], prod1[i1]);
  }
  ulonglong2 cb, ca;
  ntt_phase2_pair2(ys + ((long long)item * limbs + limb) * kN, ys + ((long long)(batch + item) * limbs + limb) * kN,
                   blk, w, w_, p, tp, tile, tile2, cb, ca);
  const uint64_t kb0 = prod0[i0], kb1 = prod0[i1], ka0 = prod1[i0], ka1 = prod1[i1];
  ulonglong2 rb, ra;
  rb.x = shoup_full(sub_mod(kb0, cb.x, p), s, ss, p);
  rb.y = shoup_full(sub_mod(kb1, cb.y, p), s, ss, p);
  if (rot.c0) {
    const uint64_t* c0 = rot.c0 + item * rot.c0_stride + (long long)limb * kN;
    rb.x = add_mod(rb.x, c0[s0], p);
    rb.y = add_mod(rb.y, c0[s1], p);
  }
  ra.x = shoup_full(sub_mod(ka0, ca.x, p), s, ss, p);
  ra.y = shoup_full(sub_mod(ka1, ca.y, p), s, ss, p);
  const long long pos = (long long)limb * kN + j;
  if constexpr (Add) {
    const ulonglong2 a0 = *reinterpret_cast<const ulonglong2*>(io.add0 + item * io.add_stride + pos);
    const ulonglong2 a1 = *reinterpret_cast<const ulonglong2*>(io.add1 + item * io.add_stride + pos);
    rb.x = add_mod(rb.x, a0.x, p); rb.y = add_mod(rb.y, a0.y, p);
    ra.x = add_mod(ra.x, a1.x, p); ra.y = add_mod(ra.y, a1.y, p);
  }
  *reinterpret_cast<ulonglong2*>(io.out0 + item * io.out_stride + pos) = rb;
  *reinterpret_cast<ulonglong2*>(io.out1 + item * io.out_stride + pos) = ra;
}

// ---- relinearize + rescale ----
// Tensor product of (c0, c1) x (d0, d1) on the first `limbs` limbs.
__global__ void __launch_bounds__(256) hm_raw(uint64_t* __restrict__ raw, const uint64_t* __restrict__ c0,
                                              const uint64_t* __restrict__ c1, const uint64_t* __restrict__ d0,
                                              const uint64_t* __restrict__ d1, int limbs, long long sc,
                                              long long sd, const uint64_t* __restrict__ primes,
                                              const uint64_t* __restrict__ ratios,
                                              const uint64_t* __restrict__ ks) {
  const int limb = blockIdx.y, item = blockIdx.z;
  const long long i = (long long)limb * kN + blockIdx.x * 256 + threadIdx.x, ln = (long long)limbs * kN;
  const uint64_t p = primes[limb], r = ratios[limb], k = ks[limb];
  const long long cs = gridDim.z * ln;   // raw is [3][batch][limbs][N]
  raw += item * ln;
  const uint64_t a0 = c0[item * sc + i], a1 = c1[item * sc + i], b0 = d0[item * sd + i], b1 = d1[item * sd + i];
  raw[i] = barrett128(mul128(a0, b0), p, r, k);
  raw[cs + i] = add_mod(barrett128(mul128(a0, b1), p, r, k), barrett128(mul128(a1, b0), p, r, k), p);
  raw[2 * cs + i] = barrett128(mul128(a1, b1), p, r, k);
}

// Last limb l: NTT phase 2 of both conversions, value = raw + (keyQ - conv) * P^{-1}
// (doubled if dbl), then iNTT phase 1 of the value into tws[comp][N].
__global__ void __launch_bounds__(kXThreads, 2) hm_last(Rot rot, HMul hm, int limbs, const uint64_t* ys, uint64_t* tws,
                                                     const uint64_t* __restrict__ primes,
                                                     const uint64_t* __restrict__ ratios,
                                                     const uint64_t* __restrict__ ks,
                                                     const uint64_t* __restrict__ roots,
                                                     const uint64_t* __restrict__ roots_s,
                                                     const uint64_t* __restrict__ iroots,
                                                     const uint64_t* __restrict__ iroots_s) {
  // Items vary fastest over the grid so that consecutive blocks share the key reads in L2.
  const int item = blockIdx.x, blk = blockIdx.y, l = limbs - 1, nb = hm.batch;
  const uint64_t p = primes[l], tp = p << 1;
  const uint64_t* w = roots + (long long)l * kN;
  const uint64_t* w_ = roots_s + (long long)l * kN;
  __shared__ uint64_t tile[2][kXWarps][kWarp + 1];
  __shared__ uint64_t tile2[2][kXWarps][kWarp + 1];
  __shared__ uint64_t prod0[kData], prod1[kData];
  const int j = blk * kData + 2 * threadIdx.x;
  const uint64_t ratio = ratios[l], bk = ks[l], s = hm.inv[l], ss = hm.inv_s[l];
  if (rot.p1) {
    digit_products(rot, item, l, l, blk, p, ratio, bk, w, w_, tile, tile2, prod0, prod1);
  } else {
    rot_inner2(rot, item, l, l, j, p, ratio, bk, prod0[2 * threadIdx.x], prod1[2 * threadIdx.x]);
    rot_inner2(rot, item, l, l, j + 1, p, ratio, bk, prod0[2 * threadIdx.x + 1], prod1[2 * threadIdx.x + 1]);
  }
  ulonglong2 cb, ca;
  ntt_phase2_pair2(ys + ((long long)item * limbs + l) * kN, ys + ((long long)(nb + item) * limbs + l) * kN, blk, w, w_,
                   p, tp, tile, tile2, cb, ca);
  const uint64_t kb0 = prod0[2 * threadIdx.x], kb1 = prod0[2 * threadIdx.x + 1];
  const uint64_t ka0 = prod1[2 * threadIdx.x], ka1 = prod1[2 * threadIdx.x + 1];
  const long long pos = (long long)l * kN + j;
  ulonglong2 r0, r1;
  hm_raw01(hm, item, pos, p, ratio, bk, r0, r1);
  uint64_t vb0 = add_mod(r0.x, shoup_full(sub_mod(kb0, cb.x, p), s, ss, p), p);
  uint64_t vb1 = add_mod(r0.y, shoup_full(sub_mod(kb1, cb.y, p), s, ss, p), p);
  uint64_t va0 = add_mod(r1.x, shoup_full(sub_mod(ka0, ca.x, p), s, ss, p), p);
  uint64_t va1 = add_mod(r1.y, shoup_full(sub_mod(ka1, ca.y, p), s, ss, p), p);
  if (hm.dbl) {
    vb0 = add_mod(vb0, vb0, p); vb1 = add_mod(vb1, vb1, p);
    va0 = add_mod(va0, va0, p); va1 = add_mod(va1, va1, p);
  }
  const uint64_t* iw = iroots + (long long)l * kN;
  const uint64_t* iw_ = iroots_s + (long long)l * kN;
  __syncthreads();
  intt1_rows2(vb0, vb1, va0, va1, blk, tws + (long long)item * kN, tws + (long long)(nb + item) * kN, iw, iw_, p, tp,
              tile, tile2);
}

// Limbs j < l: NTT phase 2 of the combined conversion/switch polynomial, plus
// (raw + keyQ * P^{-1}) * cfin (doubled if dbl), plus the post op.
template <int Post>
__global__ void __launch_bounds__(kXThreads, 2) hm_finish(Rot rot, HMul hm, int limbs, const uint64_t* ys,
                                                       const uint64_t* __restrict__ primes,
                                                       const uint64_t* __restrict__ ratios,
                                                       const uint64_t* __restrict__ ks,
                                                       const uint64_t* __restrict__ roots,
                                                       const uint64_t* __restrict__ roots_s) {
  const int item = blockIdx.x, blk = blockIdx.y, limb = blockIdx.z, nb = hm.batch;
  const uint64_t p = primes[limb], tp = p << 1;
  const uint64_t* w = roots + (long long)limb * kN;
  const uint64_t* w_ = roots_s + (long long)limb * kN;
  __shared__ uint64_t tile[2][kXWarps][kWarp + 1];
  __shared__ uint64_t tile2[2][kXWarps][kWarp + 1];
  __shared__ uint64_t prod0[kData], prod1[kData];
  const int j = blk * kData + 2 * threadIdx.x;
  const uint64_t ratio = ratios[limb], bk = ks[limb], s = hm.inv[limb], ss = hm.inv_s[limb];
  const uint64_t cf = hm.cfin[limb], cfs = hm.cfin_s[limb];
  // Key products first, parked in shared memory, so that nothing but the accumulators is live
  // through the digit loop (the two conversion transforms follow; same arithmetic as before).
  if (rot.p1) {
    digit_products(rot, item, limb, limb, blk, p, ratio, bk, w, w_, tile, tile2, prod0, prod1);
  } else {
    rot_inner2(rot, item, limb, limb, j, p, ratio, bk, prod0[2 * threadIdx.x], prod1[2 * threadIdx.x]);
    rot_inner2(rot, item, limb, limb, j + 1, p, ratio, bk, prod0[2 * threadIdx.x + 1], prod1[2 * threadIdx.x + 1]);
  }
  ulonglong2 sb, sa;
  ntt_phase2_pair2(ys + ((long long)item * limbs + limb) * kN, ys + ((long long)(nb + item) * limbs + limb) * kN, blk,
                   w, w_, p, tp, tile, tile2, sb, sa);
  const uint64_t kb0 = prod0[2 * threadIdx.x], kb1 = prod0[2 * threadIdx.x + 1];
  const uint64_t ka0 = prod1[2 * threadIdx.x], ka1 = prod1[2 * threadIdx.x + 1];
  const long long pos = (long long)limb * kN + j;
  const long long ppos = item * hm.post_stride + pos, opos = item * hm.out_stride + pos;
  ulonglong2 r0, r1;
  hm_raw01(hm, item, pos, p, ratio, bk, r0, r1);
  auto fin = [&](uint64_t sw, uint64_t ip, uint64_t r) {
    uint64_t v = add_mod(r, shoup_full(ip, s, ss, p), p);
    if (hm.dbl) v = add_mod(v, v, p);
    return add_mod(sw, shoup_full(v, cf, cfs, p), p);
  };
  ulonglong2 rb, ra;
  rb.x = fin(sb.x, kb0, r0.x);
  rb.y = fin(sb.y, kb1, r0.y);
  ra.x = fin(sa.x, ka0, r1.x);
  ra.y = fin(sa.y, ka1, r1.y);
  if constexpr (Post == 1 || Post == 2) {
    const ulonglong2 a0 = *reinterpret_cast<const ulonglong2*>(hm.post0 + ppos);
    const ulonglong2 a1 = *reinterpret_cast<const ulonglong2*>(hm.post1 + ppos);
    if constexpr (Post == 1) {
      rb.x = add_mod(rb.x, a0.x, p); rb.y = add_mod(rb.y, a0.y, p);
      ra.x = add_mod(ra.x, a1.x, p); ra.y = add_mod(ra.y, a1.y, p);
    } else {
      rb.x = sub_mod(rb.x, a0.x, p); rb.y = sub_mod(rb.y, a0.y, p);
      ra.x = sub_mod(ra.x, a1.x, p); ra.y = sub_mod(ra.y, a1.y, p);
    }
  } else if constexpr (Post == 3) {
    const uint64_t c = hm.scalar[limb];
    rb.x = add_mod(rb.x, c, p); rb.y = add_mod(rb.y, c, p);
  } else if constexpr (Post == 4) {
    const ulonglong2 a0 = *reinterpret_cast<const ulonglong2*>(hm.post0 + ppos);
    rb.x = add_mod(rb.x, a0.x, p); rb.y = add_mod(rb.y, a0.y, p);
  }
  *reinterpret_cast<ulonglong2*>(hm.out0 + opos) = rb;
  *reinterpret_cast<ulonglong2*>(hm.out1 + opos) = ra;
}

// Hoisted rotations of one ciphertext to several offsets, kept in the extended basis
// (EasyFHE's innerproduct_broadcast followed by finalize_fast_rotation_ext). Entry b of
// `table` is {key bx, key ax, special-limb start, gather map}; a null map is offset 0.
struct FastEntry { const uint64_t* kb; const uint64_t* ka; long long sms; const int* map; };

__global__ void __launch_bounds__(256) fast_rotate_ext(uint64_t* __restrict__ out_bx, uint64_t* __restrict__ out_ax,
                                                       const uint64_t* __restrict__ digits,
                                                       const FastEntry* __restrict__ table,
                                                       const uint64_t* __restrict__ c0,
                                                       const uint64_t* __restrict__ c1, int limbs, int K, int L,
                                                       int beta, const uint64_t* __restrict__ primes,
                                                       const uint64_t* __restrict__ ratios,
                                                       const uint64_t* __restrict__ ks,
                                                       const uint64_t* __restrict__ pmodq) {
  const int j = blockIdx.x * 256 + threadIdx.x, limb = blockIdx.y, b = blockIdx.z;
  const int active = limbs + K;
  const long long o = ((long long)b * active + limb) * kN + j;
  const FastEntry e = table[b];
  const bool q = limb < limbs;
  const int pi = q ? limb : L + (limb - limbs);
  const uint64_t p = primes[pi], ratio = ratios[pi], bk = ks[pi];
  if (e.map == nullptr) {
    if (q) {
      out_bx[o] = barrett128(mul128(c0[(long long)limb * kN + j], pmodq[limb]), p, ratio, bk);
      out_ax[o] = barrett128(mul128(c1[(long long)limb * kN + j], pmodq[limb]), p, ratio, bk);
    } else {
      out_bx[o] = 0;
      out_ax[o] = 0;
    }
    return;
  }
  const int src = e.map[j];
  const int key_limb = q ? limb : (int)e.sms + (limb - limbs);
  const long long ds = (long long)active * kN, kstride = (e.sms + K) * kN;
  const uint64_t* d = digits + (long long)limb * kN + src;
  const uint64_t* kb = e.kb + (long long)key_limb * kN + src;
  const uint64_t* ka = e.ka + (long long)key_limb * kN + src;
  u128 acc0{0, 0}, acc1{0, 0};
  for (int i = 0; i < beta; ++i) {
    const uint64_t x = d[i * ds];
    add128(mul128(x, kb[i * kstride]), acc0);
    add128(mul128(x, ka[i * kstride]), acc1);
  }
  uint64_t vb = barrett128(acc0, p, ratio, bk);
  if (q) vb = add_mod(vb, barrett128(mul128(c0[(long long)limb * kN + src], pmodq[limb]), p, ratio, bk), p);
  out_bx[o] = vb;
  out_ax[o] = barrett128(acc1, p, ratio, bk);
}

// Baby-step MAC of a BSGS linear transform for a batch of ciphertexts, fused with the hoisted
// extended rotations: res[b][g] = sum_i pt[g * nb + i] * rot_ext(ct_b, offset_i). The rotated
// value is the one fast_rotate_ext writes (128-bit sum of the beta digit products plus the c0 * P
// term, one Barrett reduction). Each product with a plaintext is reduced on its own with the
// plaintext's Shoup constant (pts_shoup = floor(pt * 2^64 / p)) and the nb < 2^4 reduced products
// are summed in 64 bits (nb * p < 2^64) and reduced once more, so every output is the exact
// residue of EasyFHE's batched_pairwise_mac. The 64-bit accumulators halve the register
// footprint of the 128-bit ones and let two blocks share an SM. Each thread keeps TB batch
// items, so keys and plaintexts are read once per TB items, and the batch groups are adjacent
// in blockIdx.x so they share them through L2. The rotation table is staged in shared memory and
// the gather index of the next entry is loaded one entry ahead.
// c0p = c0 * P (mod q) [batch][limbs][N] (sylph_scale_pmodq).
constexpr int kMaxNb = 16;

template <int TB, int G>
__global__ void __launch_bounds__(256, 2) bsgs_mac_ext(uint64_t* __restrict__ res, const uint64_t* __restrict__ digits,
                                                       long long dstride, const FastEntry* __restrict__ table, int nb,
                                                       const uint64_t* __restrict__ c0p, const uint64_t* __restrict__ c1,
                                                       long long cstride, const uint64_t* __restrict__ pts,
                                                       const uint64_t* __restrict__ pts_shoup, long long pstride,
                                                       long long sstride, int batch, int limbs, int K, int L, int beta,
                                                       const uint64_t* __restrict__ primes,
                                                       const uint64_t* __restrict__ ratios,
                                                       const uint64_t* __restrict__ ks,
                                                       const uint64_t* __restrict__ pmodq) {
  __shared__ FastEntry stab[kMaxNb];
  if (threadIdx.x < nb) stab[threadIdx.x] = table[threadIdx.x];
  __syncthreads();
  const int nbg = (batch + TB - 1) / TB;
  const int j = (blockIdx.x / nbg) * 256 + threadIdx.x, limb = blockIdx.y, b0 = (blockIdx.x % nbg) * TB;
  const int active = limbs + K;
  const bool q = limb < limbs;
  const int pi = q ? limb : L + (limb - limbs);
  const uint64_t p = primes[pi], ratio = ratios[pi], bk = ks[pi];
  const uint64_t pq = q ? pmodq[limb] : 0;
  const long long lj = (long long)limb * kN;
  const long long ds = (long long)active * kN;
  uint64_t acc0[TB][G], acc1[TB][G];
#pragma unroll
  for (int t = 0; t < TB; ++t)
#pragma unroll
    for (int g = 0; g < G; ++g) acc0[t][g] = acc1[t][g] = 0;
  int src_next = stab[0].map == nullptr ? j : stab[0].map[j];
  for (int i = 0; i < nb; ++i) {
    const FastEntry e = stab[i];
    const int src = src_next;
    if (i + 1 < nb) src_next = stab[i + 1].map == nullptr ? j : stab[i + 1].map[j];
    uint64_t w[G], wsh[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      w[g] = pts[(long long)(g * nb + i) * pstride + lj + j];
      wsh[g] = pts_shoup[(long long)(g * nb + i) * sstride + lj + j];
    }
    if (e.map == nullptr) {
#pragma unroll
      for (int t = 0; t < TB; ++t) {
        const int b = b0 + t;
        if (b >= batch) break;
        uint64_t vb = 0, va = 0;
        if (q) {
          vb = c0p[(long long)b * limbs * kN + lj + j];
          va = barrett128(mul128(c1[b * cstride + lj + j], pq), p, ratio, bk);
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
          acc0[t][g] += shoup_full(vb, w[g], wsh[g], p);
          acc1[t][g] += shoup_full(va, w[g], wsh[g], p);
        }
      }
      continue;
    }
    const int key_limb = q ? limb : (int)e.sms + (limb - limbs);
    const long long kstride = (e.sms + K) * kN;
    uint64_t kb[3], ka[3];
#pragma unroll
    for (int d = 0; d < 3; ++d) {
      if (d < beta) {
        kb[d] = e.kb[d * kstride + (long long)key_limb * kN + src];
        ka[d] = e.ka[d * kstride + (long long)key_limb * kN + src];
      }
    }
#pragma unroll
    for (int t = 0; t < TB; ++t) {
      const int b = b0 + t;
      if (b >= batch) break;
      const uint64_t* dg = digits + b * dstride + lj + src;
      u128 s0{0, 0}, s1{0, 0};
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        if (d < beta) {
          const uint64_t x = dg[d * ds];
          add128(mul128(x, kb[d]), s0);
          add128(mul128(x, ka[d]), s1);
        }
      }
      if (q) add128(u128{0, c0p[(long long)b * limbs * kN + lj + src]}, s0);
      const uint64_t vb = barrett128(s0, p, ratio, bk);
      const uint64_t va = barrett128(s1, p, ratio, bk);
#pragma unroll
      for (int g = 0; g < G; ++g) {
        acc0[t][g] += shoup_full(vb, w[g], wsh[g], p);
        acc1[t][g] += shoup_full(va, w[g], wsh[g], p);
      }
    }
  }
#pragma unroll
  for (int t = 0; t < TB; ++t) {
    const int b = b0 + t;
    if (b >= batch) break;
#pragma unroll
    for (int g = 0; g < G; ++g) {
      res[((long long)g * batch + b) * ds + lj + j] = barrett128(u128{0, acc0[t][g]}, p, ratio, bk);
      res[((long long)(G + g) * batch + b) * ds + lj + j] = barrett128(u128{0, acc1[t][g]}, p, ratio, bk);
    }
  }
}

// out[r][limb][j] = floor(pt[r][limb][j] * 2^64 / p_limb): the Shoup constants of the plaintext
// rows (pt < p). A double-precision estimate of the quotient is corrected exactly in 128 bits.
__global__ void __launch_bounds__(256) pts_shoup(uint64_t* __restrict__ out, const uint64_t* __restrict__ pts,
                                                 long long pstride, long long sstride, int limbs, int L,
                                                 const uint64_t* __restrict__ primes) {
  const int j = blockIdx.x * 256 + threadIdx.x, limb = blockIdx.y, r = blockIdx.z;
  const int pi = limb < limbs ? limb : L + (limb - limbs);
  const uint64_t p = primes[pi];
  const uint64_t w = pts[(long long)r * pstride + (long long)limb * kN + j];
  double est = (double)w / (double)p * 18446744073709551616.0;
  if (est > 18446744073709549568.0) est = 18446744073709549568.0;
  uint64_t qv = (uint64_t)est;
  // rem = w * 2^64 - qv * p as a signed 128-bit value {hi, lo}
  u128 qp = mul128(qv, p);
  uint64_t lo = 0 - qp.lo;
  uint64_t hi = w - qp.hi - (qp.lo != 0 ? 1 : 0);
  // |rem| < 2^12 * p: a second double estimate of rem / p leaves an error of at most one
  const double remd = (double)(long long)hi * 18446744073709551616.0 + (double)lo;
  const long long dq = (long long)floor(remd / (double)p);
  qv += (uint64_t)dq;
  const u128 dp = mul128((uint64_t)(dq < 0 ? -dq : dq), p);
  if (dq >= 0) {
    uint64_t nlo = lo - dp.lo;
    hi = hi - dp.hi - (lo < dp.lo ? 1 : 0);
    lo = nlo;
  } else {
    uint64_t nlo = lo + dp.lo;
    hi = hi + dp.hi + (nlo < lo ? 1 : 0);
    lo = nlo;
  }
  while ((long long)hi < 0) {  // rem < 0
    qv -= 1;
    uint64_t nlo = lo + p;
    hi = hi + (nlo < lo ? 1 : 0);
    lo = nlo;
  }
  while (hi != 0 || lo >= p) {  // rem >= p
    qv += 1;
    uint64_t nlo = lo - p;
    hi = hi - (lo < p ? 1 : 0);
    lo = nlo;
  }
  out[(long long)r * sstride + (long long)limb * kN + j] = qv;
}

// Giant step of the double-hoisted BSGS transform for a batch (EasyFHE innerproduct_pairwise +
// double_hoist_giant_sum_ext): out[b] = base[b] + sum_k rot_ext(inner[k * batch + b], offset_k),
// from the ModUp digits and c0 of the ModDown'ed partial sums inner (nk = groups - 1 offsets).
// The c0 * P product joins the 128-bit digit sum before its Barrett reduction (sum < 2^122: exact).
template <int TB>
__global__ void __launch_bounds__(256) giant_sum_ext(uint64_t* __restrict__ out0, uint64_t* __restrict__ out1,
                                                     const uint64_t* __restrict__ base0,
                                                     const uint64_t* __restrict__ base1,
                                                     const uint64_t* __restrict__ digits, long long dstride,
                                                     const uint64_t* __restrict__ c0, long long cstride,
                                                     const FastEntry* __restrict__ table, int nk, int batch,
                                                     int limbs, int K, int L, int beta,
                                                     const uint64_t* __restrict__ primes,
                                                     const uint64_t* __restrict__ ratios,
                                                     const uint64_t* __restrict__ ks,
                                                     const uint64_t* __restrict__ pmodq) {
  const int nbg = (batch + TB - 1) / TB;
  const int j = (blockIdx.x / nbg) * 256 + threadIdx.x, limb = blockIdx.y, b0 = (blockIdx.x % nbg) * TB;
  const int active = limbs + K;
  const bool q = limb < limbs;
  const int pi = q ? limb : L + (limb - limbs);
  const uint64_t p = primes[pi], ratio = ratios[pi], bk = ks[pi];
  const uint64_t pq = q ? pmodq[limb] : 0;
  const long long lj = (long long)limb * kN, ds = (long long)active * kN;
  uint64_t bx[TB], ax[TB];
#pragma unroll
  for (int t = 0; t < TB; ++t) {
    const int b = b0 + t;
    if (b < batch) {
      bx[t] = base0[b * ds + lj + j];
      ax[t] = base1[b * ds + lj + j];
    }
  }
  for (int k = 0; k < nk; ++k) {
    const FastEntry e = table[k];
    const int src = e.map[j];
    const int key_limb = q ? limb : (int)e.sms + (limb - limbs);
    const long long kstride = (e.sms + K) * kN;
    uint64_t kb[3], ka[3];
#pragma unroll
    for (int d = 0; d < 3; ++d) {
      if (d < beta) {
        kb[d] = e.kb[d * kstride + (long long)key_limb * kN + src];
        ka[d] = e.ka[d * kstride + (long long)key_limb * kN + src];
      }
    }
#pragma unroll
    for (int t = 0; t < TB; ++t) {
      const int b = b0 + t;
      if (b >= batch) break;
      const long long item = (long long)k * batch + b;
      const uint64_t* dg = digits + item * dstride + lj + src;
      u128 s0{0, 0}, s1{0, 0};
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        if (d < beta) {
          const uint64_t x = dg[d * ds];
          add128(mul128(x, kb[d]), s0);
          add128(mul128(x, ka[d]), s1);
        }
      }
      if (q) add128(mul128(c0[item * cstride + lj + src], pq), s0);
      bx[t] = add_mod(bx[t], barrett128(s0, p, ratio, bk), p);
      ax[t] = add_mod(ax[t], barrett128(s1, p, ratio, bk), p);
    }
  }
#pragma unroll
  for (int t = 0; t < TB; ++t) {
    const int b = b0 + t;
    if (b < batch) {
      out0[b * ds + lj + j] = bx[t];
      out1[b * ds + lj + j] = ax[t];
    }
  }
}

// out[b][limb][j] = c0[b][limb][j] * pmodq[limb] (mod q_limb) for the Q limbs: the c0 term of
// an extended rotation (EasyFHE finalize_fast_rotation_ext), computed once per ciphertext.
__global__ void __launch_bounds__(256) scale_pmodq(uint64_t* __restrict__ out, const uint64_t* __restrict__ c0,
                                                   long long cstride, int limbs,
                                                   const uint64_t* __restrict__ primes,
                                                   const uint64_t* __restrict__ ratios,
                                                   const uint64_t* __restrict__ ks,
                                                   const uint64_t* __restrict__ pmodq) {
  const int j = blockIdx.x * 256 + threadIdx.x, limb = blockIdx.y, b = blockIdx.z;
  const long long o = (long long)limb * kN + j;
  out[(long long)b * limbs * kN + o] =
      barrett128(mul128(c0[b * cstride + o], pmodq[limb]), primes[limb], ratios[limb], ks[limb]);
}

}  // namespace sylph_ks

using namespace sylph_ks;

extern "C" int sylph_scale_pmodq(uint64_t* out, const uint64_t* c0, long long cstride, int batch, int limbs,
                                 const uint64_t* primes, const uint64_t* ratios, const uint64_t* ks,
                                 const uint64_t* pmodq, void* stream_ptr) {
  auto st = static_cast<cudaStream_t>(stream_ptr);
  scale_pmodq<<<dim3(kN / 256, limbs, batch), 256, 0, st>>>(out, c0, cstride, limbs, primes, ratios, ks, pmodq);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int sylph_giant_sum_ext(uint64_t* out0, uint64_t* out1, const uint64_t* base0, const uint64_t* base1,
                                   const uint64_t* digits, long long dstride, const uint64_t* c0, long long cstride,
                                   const void* table, int nk, int batch, int limbs, int K, int L, int beta,
                                   const uint64_t* primes, const uint64_t* ratios, const uint64_t* ks,
                                   const uint64_t* pmodq, void* stream_ptr) {
  if (beta > 3) return -1;
  auto st = static_cast<cudaStream_t>(stream_ptr);
  const auto* tab = static_cast<const FastEntry*>(table);
  constexpr int TB = 4;
  giant_sum_ext<TB><<<dim3(kN / 256 * ((batch + TB - 1) / TB), limbs + K, 1), 256, 0, st>>>(
      out0, out1, base0, base1, digits, dstride, c0, cstride, tab, nk, batch, limbs, K, L, beta, primes, ratios, ks,
      pmodq);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int sylph_pts_shoup(uint64_t* out, const uint64_t* pts, long long pstride, long long sstride, int rows,
                               int active, int limbs, int L, const uint64_t* primes, void* stream_ptr) {
  auto st = static_cast<cudaStream_t>(stream_ptr);
  pts_shoup<<<dim3(kN / 256, active, rows), 256, 0, st>>>(out, pts, pstride, sstride, limbs, L, primes);
  return static_cast<int>(cudaGetLastError());
}

// res [2][G][batch][limbs + K][N]: baby-step partial sums of the hoisted extended rotations of
// (c0, c1) [batch][limbs][N] (c1 batch stride cstride; c0p = c0 * P contiguous) for the nb table entries, digits
// [batch][beta * (limbs + K)][N] (batch stride dstride), plaintexts and their Shoup constants
// [G * nb][>= limbs + K][N] (row strides pstride and sstride).
extern "C" int sylph_bsgs_mac_ext(uint64_t* res, const uint64_t* digits, long long dstride, const void* table, int nb,
                                  const uint64_t* c0p, const uint64_t* c1, long long cstride, const uint64_t* pts,
                                  const uint64_t* pts_shoup, long long pstride, long long sstride, int groups, int batch, int tb, int limbs,
                                  int K, int L, int beta, const uint64_t* primes, const uint64_t* ratios,
                                  const uint64_t* ks, const uint64_t* pmodq, void* stream_ptr) {
  if (beta > 3) return -1;
  if (nb < 1 || nb > kMaxNb) return -4;
  auto st = static_cast<cudaStream_t>(stream_ptr);
  const auto* tab = static_cast<const FastEntry*>(table);
#define BSGS(TB, G)                                                                                            \
  bsgs_mac_ext<TB, G><<<dim3(kN / 256 * ((batch + TB - 1) / TB), limbs + K, 1), 256, 0, st>>>(                \
      res, digits, dstride, tab, nb, c0p, c1, cstride, pts, pts_shoup, pstride, sstride, batch, limbs, K, L, beta, \
      primes, ratios, ks, pmodq)
  if (groups == 4) {
    if (tb == 1) BSGS(1, 4);
    else if (tb == 2) BSGS(2, 4);
    else if (tb == 4) BSGS(4, 4);
    else return -2;
  } else if (groups == 2) {
    if (tb == 1) BSGS(1, 2);
    else if (tb == 2) BSGS(2, 2);
    else if (tb == 4) BSGS(4, 2);
    else return -2;
  } else if (groups == 1) {
    if (tb == 1) BSGS(1, 1);
    else if (tb == 2) BSGS(2, 1);
    else if (tb == 4) BSGS(4, 1);
    else return -2;
  } else {
    return -3;
  }
#undef BSGS
  return static_cast<int>(cudaGetLastError());
}

template <bool R>
static void launch_moddown(const Pair& io, const Rot& rot, int batch, int limbs, int K, int L, uint64_t* workspace,
                           const Tables& t, cudaStream_t stream) {
  const int z = 2 * batch;
  uint64_t* ws = workspace;
  uint64_t* ys = workspace + (long long)z * K * kN;
  if constexpr (R)
    rot_intt1<<<dim3(batch, 64, K), kXThreads, 0, stream>>>(rot, batch, limbs, K, L, ws, t.primes, t.ratios, t.ks,
                                                            t.roots, t.roots_s, t.iroots, t.iroots_s);
  else
    md_intt1<<<dim3(64, K, z), kXThreads, 0, stream>>>(io, batch, limbs, K, L, ws, t.primes, t.iroots, t.iroots_s);
  md_intt2<<<dim3(kN / 512, K, z), 64, 0, stream>>>(K, L, ws, t.primes, t.iroots, t.iroots_s, t.hat, t.hat_s);
  launch_md_conv_ntt1<false>(limbs, K, ws, ys, t, HMul{}, 0, limbs, z, stream);
  const dim3 fg = R ? dim3(batch, 64, limbs) : dim3(64, limbs, z);
  if constexpr (R) {
    if (io.add0)
      rot_finish<true><<<fg, kXThreads, 0, stream>>>(io, rot, batch, limbs, ys, t.inv, t.inv_s, t.primes, t.ratios,
                                                     t.ks, t.roots, t.roots_s);
    else
      rot_finish<false><<<fg, kXThreads, 0, stream>>>(io, rot, batch, limbs, ys, t.inv, t.inv_s, t.primes, t.ratios,
                                                      t.ks, t.roots, t.roots_s);
  } else if (io.add0) {
    md_ntt2_finish<true><<<fg, kXThreads, 0, stream>>>(io, batch, limbs, ys, t.inv, t.inv_s, t.primes, t.roots,
                                                       t.roots_s);
  } else {
    md_ntt2_finish<false><<<fg, kXThreads, 0, stream>>>(io, batch, limbs, ys, t.inv, t.inv_s, t.primes, t.roots,
                                                        t.roots_s);
  }
}

// ---- ModUp (EasyFHE cv_modup) ----
// Digits of in [batch][limbs][N]: digit d holds limbs [d alpha, d alpha + gs) of in, copied as
// they are, and their base conversion to every other limb of Q_l P, into
// out [batch][beta (limbs + K)][N]. Four launches for all digits and items: iNTT phase 1 into
// ws[(item * limbs + limb) N]; iNTT phase 2 times [Q_d / q_i]^{-1}_{q_i} in place; base
// conversion fused with NTT phase 1 written to out; NTT phase 2 in place plus the copies.
// EasyFHE runs the same arithmetic (iNTT, scaling, 128-bit sums with one Barrett reduction,
// NTT) in seven launches per digit when beta = 1; the residues are identical.
// The pair at (pos, pos + 1) of the polynomial put through ModUp: in, or in * in_b (mod p).
__device__ __forceinline__ ulonglong2 mu_in_pair(const uint64_t* __restrict__ in, const uint64_t* __restrict__ in_b,
                                                 long long pos, uint64_t p, uint64_t ratio, uint64_t bk) {
  ulonglong2 v = *reinterpret_cast<const ulonglong2*>(in + pos);
  if (in_b) {
    const ulonglong2 b = *reinterpret_cast<const ulonglong2*>(in_b + pos);
    v.x = barrett128(mul128(v.x, b.x), p, ratio, bk);
    v.y = barrett128(mul128(v.y, b.y), p, ratio, bk);
  }
  return v;
}

__global__ void __launch_bounds__(kXThreads) mu_intt1(const uint64_t* __restrict__ in, long long in_stride,
                                                      const uint64_t* __restrict__ in_b, long long in_b_stride,
                                                      int limbs, uint64_t* ws, const uint64_t* __restrict__ primes,
                                                      const uint64_t* __restrict__ ratios,
                                                      const uint64_t* __restrict__ ks,
                                                      const uint64_t* __restrict__ iroots,
                                                      const uint64_t* __restrict__ iroots_s) {
  const int item = blockIdx.z, limb = blockIdx.y;
  const uint64_t p = primes[limb];
  const long long pos = (long long)limb * kN + 2 * (blockIdx.x * kXThreads + threadIdx.x);
  const ulonglong2 v = mu_in_pair(in + item * in_stride, in_b ? in_b + item * in_b_stride : nullptr, pos, p,
                                  ratios[limb], ks[limb]);
  __shared__ uint64_t tile[2][kXWarps][kWarp + 1];
  intt1_rows(v.x, v.y, blockIdx.x, ws + ((long long)item * limbs + limb) * kN, iroots + (long long)limb * kN,
             iroots_s + (long long)limb * kN, p, p << 1, tile);
}

__global__ void __launch_bounds__(64) mu_intt2(int limbs, int alpha, uint64_t* ws, const uint64_t* __restrict__ primes,
                                               const uint64_t* __restrict__ iroots,
                                               const uint64_t* __restrict__ iroots_s,
                                               const uint64_t* __restrict__ hat, const uint64_t* __restrict__ hat_s,
                                               long long hat_stride) {
  const int limb = blockIdx.y;
  uint64_t* poly = ws + ((long long)blockIdx.z * limbs + limb) * kN;
  const uint64_t* w = iroots + (long long)limb * kN;
  const uint64_t* w_ = iroots_s + (long long)limb * kN;
  const uint64_t p = primes[limb], tp = p << 1;
  const int begin = (limb / alpha) * alpha, gs = min(alpha, limbs - begin);
  const long long row = (long long)(begin + gs - 1) * hat_stride + (limb - begin);
  const uint64_t h = hat[row], hs = hat_s[row];
  const int group = threadIdx.x / 8, lane = threadIdx.x % 8;
  const int n0 = 8 * blockIdx.x + lane;
  uint64_t x[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = poly[group * 8192 + j * 1024 + n0];
  iradix8(x, 8 + group, w, w_, p, tp);
  __shared__ uint64_t tr[8][9][9];
#pragma unroll
  for (int j = 0; j < 8; ++j) tr[lane][j][group] = x[j];
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = tr[lane][group][j];
  iradix8(x, 1, w, w_, p, tp);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint64_t v = x[j] >= p ? x[j] - p : x[j];
    poly[j * 8192 + group * 1024 + n0] = shoup_full(v, h, hs, p);
  }
}

// Base conversion of digit d (limbs [begin, begin + gs)) to the G limbs it does not own with
// out_iter = G * blockIdx.y + g (< ext - gs; phys j = out_iter (+ gs past the own range)), fused
// with NTT phase 1: the gs input columns are read once per block and feed G accumulators.
template <int G>
__global__ void __launch_bounds__(64) mu_conv_ntt1(int limbs, int alpha, int K, int L, int slices, const uint64_t* ws,
                                                   uint64_t* out, long long out_stride,
                                                   const uint64_t* __restrict__ tbl, long long tbl_stride,
                                                   const uint64_t* __restrict__ primes,
                                                   const uint64_t* __restrict__ ratios,
                                                   const uint64_t* __restrict__ ks,
                                                   const uint64_t* __restrict__ roots,
                                                   const uint64_t* __restrict__ roots_s) {
  const int d = blockIdx.y / slices, item = blockIdx.z, ext = limbs + K;
  const int begin = d * alpha, gs = min(alpha, limbs - begin), nout = ext - gs;
  const int ob = G * (blockIdx.y - d * slices), ng = min(G, nout - ob);
  if (ng <= 0) return;
  __shared__ uint64_t hat[G][kMaxP];
  if (threadIdx.x < G * kMaxP) {
    const int g = threadIdx.x / kMaxP, i = threadIdx.x % kMaxP;
    if (g < ng && i < gs) hat[g][i] = tbl[(long long)d * tbl_stride + (long long)(ob + g) * alpha + i];
  }
  __syncthreads();
  const uint64_t* pl = ws + ((long long)item * limbs + begin) * kN;
  const int group = threadIdx.x / 8, lane = threadIdx.x % 8;
  const int n0 = 8 * blockIdx.x + lane;
  // Rows converted kJ at a time (see md_conv_ntt1): same products, same order, fewer registers.
  constexpr int kJ = 4;
  uint64_t xs[G][8];
#pragma unroll
  for (int j0 = 0; j0 < 8; j0 += kJ) {
    u128 acc[G][kJ];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int j = 0; j < kJ; ++j) acc[g][j] = u128{0, 0};
    for (int i = 0; i < gs; ++i) {
      const uint64_t* row = pl + (long long)i * kN + group * 1024 + n0;
      uint64_t v[kJ];
#pragma unroll
      for (int j = 0; j < kJ; ++j) v[j] = row[(j0 + j) * 8192];
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const uint64_t h = hat[g][i];
#pragma unroll
        for (int j = 0; j < kJ; ++j) add128(mul128(v[j], h), acc[g][j]);
      }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
      if (g >= ng) break;
      const int out_iter = ob + g, j = out_iter < begin ? out_iter : out_iter + gs;
      const int prime_idx = j < limbs ? j : L + (j - limbs);
      const uint64_t p = primes[prime_idx], ratio = ratios[prime_idx], bk = ks[prime_idx];
#pragma unroll
      for (int jj = 0; jj < kJ; ++jj) xs[g][j0 + jj] = barrett128(acc[g][jj], p, ratio, bk);
    }
  }
  __shared__ uint64_t tr[8][9][9];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (g >= ng) break;
    const int out_iter = ob + g, j = out_iter < begin ? out_iter : out_iter + gs;
    const int prime_idx = j < limbs ? j : L + (j - limbs);
    uint64_t* o = out + item * out_stride + ((long long)d * ext + j) * kN;
    const uint64_t* w = roots + (long long)prime_idx * kN;
    const uint64_t* w_ = roots_s + (long long)prime_idx * kN;
    const uint64_t p = primes[prime_idx], tp = p << 1;
    uint64_t x[8];
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) x[jj] = xs[g][jj];
    radix8(x, 1, w, w_, p, tp);
    if (g) __syncthreads();
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) tr[lane][jj][group] = x[jj];
    __syncthreads();
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) x[jj] = tr[lane][group][jj];
    radix8(x, 8 + group, w, w_, p, tp);
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) o[group * 8192 + jj * 1024 + n0] = x[jj];
  }
}

// mu_conv_ntt1 of all digits of nb items; every digit has ext - gs outputs (gs = alpha except
// for the last digit), sliced kConvG per block over the largest count.
static void launch_mu_conv_ntt1(int limbs, int alpha, int K, int L, const uint64_t* ws, uint64_t* out,
                                long long out_stride, const uint64_t* tbl, long long tbl_stride,
                                const uint64_t* primes, const uint64_t* ratios, const uint64_t* ks,
                                const uint64_t* roots, const uint64_t* roots_s, int nb, cudaStream_t stream) {
  const int beta = (limbs + alpha - 1) / alpha, ext = limbs + K;
  const int nout = ext - std::min(alpha, limbs - (beta - 1) * alpha);   // the last digit has the most
  const int g = std::min(kConvG, nout), slices = (nout + g - 1) / g;
  const dim3 grid(kN / 512, beta * slices, nb);
#define MU_CONV(GG) mu_conv_ntt1<GG><<<grid, 64, 0, stream>>>(limbs, alpha, K, L, slices, ws, out, out_stride, tbl, \
                                                              tbl_stride, primes, ratios, ks, roots, roots_s)
  switch (g) {
    case 1: MU_CONV(1); break;
    case 2: MU_CONV(2); break;
    default: MU_CONV(3); break;
  }
#undef MU_CONV
}

__global__ void __launch_bounds__(kXThreads) mu_ntt2(int limbs, int alpha, int K, int L,
                                                     const uint64_t* __restrict__ in, long long in_stride,
                                                     const uint64_t* __restrict__ in_b, long long in_b_stride,
                                                     uint64_t* out, long long out_stride,
                                                     const uint64_t* __restrict__ primes,
                                                     const uint64_t* __restrict__ ratios,
                                                     const uint64_t* __restrict__ ks,
                                                     const uint64_t* __restrict__ roots,
                                                     const uint64_t* __restrict__ roots_s) {
  const int phys = blockIdx.y, item = blockIdx.z, ext = limbs + K;
  const int d = phys / ext, j = phys - d * ext, begin = d * alpha, gs = min(alpha, limbs - begin);
  uint64_t* o = out + item * out_stride + (long long)phys * kN;
  const long long pos = blockIdx.x * kData + 2 * threadIdx.x;
  if (j >= begin && j < begin + gs) {
    *reinterpret_cast<ulonglong2*>(o + pos) = mu_in_pair(in + item * in_stride,
                                                        in_b ? in_b + item * in_b_stride : nullptr,
                                                        (long long)j * kN + pos, primes[j], ratios[j], ks[j]);
    return;
  }
  const int prime_idx = j < limbs ? j : L + (j - limbs);
  const uint64_t p = primes[prime_idx];
  __shared__ uint64_t tile[2][kXWarps][kWarp + 1];
  const ulonglong2 c = ntt_phase2_pair(o, blockIdx.x, roots + (long long)prime_idx * kN,
                                       roots_s + (long long)prime_idx * kN, p, p << 1, tile);
  *reinterpret_cast<ulonglong2*>(o + pos) = c;
}

// Workspace: batch * limbs * N words. tbl = prod_q_i_mod_q_j_modup[limbs - 1] ([dnum, *], row
// stride tbl_stride), hat = hat_inverse_vec_modup ([dnum alpha, alpha], row stride hat_stride).
static int modup_impl(const uint64_t* in, long long in_stride, const uint64_t* in_b, long long in_b_stride,
                      uint64_t* out, long long out_stride, int batch, int limbs, int alpha, int K, int L,
                      uint64_t* ws, const uint64_t* tbl, long long tbl_stride, const uint64_t* hat,
                      const uint64_t* hat_s, long long hat_stride, const uint64_t* primes, const uint64_t* ratios,
                      const uint64_t* ks, const uint64_t* roots, const uint64_t* roots_s, const uint64_t* iroots,
                      const uint64_t* iroots_s, cudaStream_t stream, bool p1) {
  if (alpha > kMaxP || limbs < 1) return -1;
  const int beta = (limbs + alpha - 1) / alpha, phys = beta * (limbs + K);
  // Items per round: the digits of a round (phys polynomials of 512 KB each per item) stay in
  // L2 between the conversion/NTT phase 1 launch and the in-place NTT phase 2.
  const int chunk = std::max(1, std::min(batch, 96 / phys));
  for (int s = 0; s < batch; s += chunk) {
    const int nb = std::min(chunk, batch - s);
    const uint64_t* cin = in + s * in_stride;
    const uint64_t* cin_b = in_b ? in_b + s * in_b_stride : nullptr;
    uint64_t* cout = out + s * out_stride;
    mu_intt1<<<dim3(64, limbs, nb), kXThreads, 0, stream>>>(cin, in_stride, cin_b, in_b_stride, limbs, ws, primes,
                                                            ratios, ks, iroots, iroots_s);
    mu_intt2<<<dim3(kN / 512, limbs, nb), 64, 0, stream>>>(limbs, alpha, ws, primes, iroots, iroots_s, hat, hat_s,
                                                           hat_stride);
    launch_mu_conv_ntt1(limbs, alpha, K, L, ws, cout, out_stride, tbl, tbl_stride, primes, ratios, ks, roots, roots_s,
                        nb, stream);
    if (!p1)
      mu_ntt2<<<dim3(64, phys, nb), kXThreads, 0, stream>>>(limbs, alpha, K, L, cin, in_stride, cin_b, in_b_stride,
                                                            cout, out_stride, primes, ratios, ks, roots, roots_s);
  }
  return static_cast<int>(cudaGetLastError());
}

// Benchmark only: per-launch times (ms) of sylph_modup for one chunk of nb items.
extern "C" int sylph_modup_profile(const uint64_t* in, long long in_stride, uint64_t* out, long long out_stride,
                                   int nb, int limbs, int alpha, int K, int L, uint64_t* ws, const uint64_t* tbl,
                                   long long tbl_stride, const uint64_t* hat, const uint64_t* hat_s,
                                   long long hat_stride, const uint64_t* primes, const uint64_t* ratios,
                                   const uint64_t* ks, const uint64_t* roots, const uint64_t* roots_s,
                                   const uint64_t* iroots, const uint64_t* iroots_s, int reps, float* times) {
  const int beta = (limbs + alpha - 1) / alpha, phys = beta * (limbs + K);
  cudaEvent_t ev[5];
  for (auto& e : ev) cudaEventCreate(&e);
  for (int s = 0; s < 4; ++s) {
    cudaEventRecord(ev[s]);
    for (int r = 0; r < reps; ++r) {
      if (s == 0)
        mu_intt1<<<dim3(64, limbs, nb), kXThreads>>>(in, in_stride, nullptr, 0, limbs, ws, primes, ratios, ks, iroots,
                                                     iroots_s);
      if (s == 1)
        mu_intt2<<<dim3(kN / 512, limbs, nb), 64>>>(limbs, alpha, ws, primes, iroots, iroots_s, hat, hat_s, hat_stride);
      if (s == 2)
        launch_mu_conv_ntt1(limbs, alpha, K, L, ws, out, out_stride, tbl, tbl_stride, primes, ratios, ks, roots,
                            roots_s, nb, nullptr);
      if (s == 3)
        mu_ntt2<<<dim3(64, phys, nb), kXThreads>>>(limbs, alpha, K, L, in, in_stride, nullptr, 0, out, out_stride,
                                                   primes, ratios, ks, roots, roots_s);
    }
    cudaEventRecord(ev[s + 1]);
  }
  cudaEventSynchronize(ev[4]);
  for (int s = 0; s < 4; ++s) {
    cudaEventElapsedTime(&times[s], ev[s], ev[s + 1]);
    times[s] /= reps;
  }
  for (auto& e : ev) cudaEventDestroy(e);
  return static_cast<int>(cudaGetLastError());
}

// Workspace: batch * limbs * N words. tbl = prod_q_i_mod_q_j_modup[limbs - 1] ([dnum, *], row
// stride tbl_stride), hat = hat_inverse_vec_modup ([dnum alpha, alpha], row stride hat_stride).
extern "C" int sylph_modup(const uint64_t* in, long long in_stride, uint64_t* out, long long out_stride, int batch,
                           int limbs, int alpha, int K, int L, uint64_t* ws, const uint64_t* tbl,
                           long long tbl_stride, const uint64_t* hat, const uint64_t* hat_s, long long hat_stride,
                           const uint64_t* primes, const uint64_t* ratios, const uint64_t* ks,
                           const uint64_t* roots, const uint64_t* roots_s, const uint64_t* iroots,
                           const uint64_t* iroots_s, void* stream_ptr) {
  return modup_impl(in, in_stride, nullptr, 0, out, out_stride, batch, limbs, alpha, K, L, ws, tbl, tbl_stride, hat,
                    hat_s, hat_stride, primes, ratios, ks, roots, roots_s, iroots, iroots_s,
                    static_cast<cudaStream_t>(stream_ptr), false);
}

// Phase-1 digits for sylph_rotate_pair / sylph_hmul_relin_rescale with p1 != 0: the same as
// sylph_modup without NTT phase 2 and without the copies of the own limbs (three launches); the
// consumers finish the transform in registers, so the digits cross DRAM once in a shorter form.
extern "C" int sylph_modup_p1(const uint64_t* in, long long in_stride, uint64_t* out, long long out_stride,
                              int batch, int limbs, int alpha, int K, int L, uint64_t* ws, const uint64_t* tbl,
                              long long tbl_stride, const uint64_t* hat, const uint64_t* hat_s,
                              long long hat_stride, const uint64_t* primes, const uint64_t* ratios,
                              const uint64_t* ks, const uint64_t* roots, const uint64_t* roots_s,
                              const uint64_t* iroots, const uint64_t* iroots_s, void* stream_ptr) {
  return modup_impl(in, in_stride, nullptr, 0, out, out_stride, batch, limbs, alpha, K, L, ws, tbl, tbl_stride, hat,
                    hat_s, hat_stride, primes, ratios, ks, roots, roots_s, iroots, iroots_s,
                    static_cast<cudaStream_t>(stream_ptr), true);
}

// ModUp of the product in * in_b (mod q_j, same Barrett reduction as hm_raw's c1 * d1), so the
// tensor product's third component never materializes; p1 as in sylph_modup_p1.
extern "C" int sylph_modup_prod(const uint64_t* in, long long in_stride, const uint64_t* in_b, long long in_b_stride,
                                uint64_t* out, long long out_stride, int batch, int limbs, int alpha, int K, int L,
                                uint64_t* ws, const uint64_t* tbl, long long tbl_stride, const uint64_t* hat,
                                const uint64_t* hat_s, long long hat_stride, const uint64_t* primes,
                                const uint64_t* ratios, const uint64_t* ks, const uint64_t* roots,
                                const uint64_t* roots_s, const uint64_t* iroots, const uint64_t* iroots_s,
                                void* stream_ptr, int p1) {
  return modup_impl(in, in_stride, in_b, in_b_stride, out, out_stride, batch, limbs, alpha, K, L, ws, tbl,
                    tbl_stride, hat, hat_s, hat_stride, primes, ratios, ks, roots, roots_s, iroots, iroots_s,
                    static_cast<cudaStream_t>(stream_ptr), p1 != 0);
}

// ---- Rescale by the last limb (EasyFHE cv_rescale_one_level) ----
// out[j] = NTT(csw[j] * switch_j(iNTT(in[l]))) + in[j] * cfin[j] for j < l = limbs - 1, both
// components of all items; four launches (EasyFHE: iNTT x2, switch, const mult, NTT x2, mult-add).
// Pointers of two components (in1 may be null); blockIdx.z = comp * batch + item.
struct RsIO {
  const uint64_t* in0; const uint64_t* in1; long long in_stride;
  uint64_t* out0; uint64_t* out1; long long out_stride;
};
__global__ void __launch_bounds__(kXThreads) rs_intt1(RsIO io, int batch, int l, uint64_t* ws,
                                                      const uint64_t* __restrict__ primes,
                                                      const uint64_t* __restrict__ iroots,
                                                      const uint64_t* __restrict__ iroots_s) {
  const int z = blockIdx.z, comp = z / batch, item = z - comp * batch;
  const uint64_t* in = (comp ? io.in1 : io.in0) + item * io.in_stride + (long long)l * kN;
  const uint64_t p = primes[l];
  const ulonglong2 v = reinterpret_cast<const ulonglong2*>(in)[blockIdx.x * kXThreads + threadIdx.x];
  __shared__ uint64_t tile[2][kXWarps][kWarp + 1];
  intt1_rows(v.x, v.y, blockIdx.x, ws + (long long)z * kN, iroots + (long long)l * kN, iroots_s + (long long)l * kN,
             p, p << 1, tile);
}
__global__ void __launch_bounds__(64) rs_intt2(int l, uint64_t* ws, const uint64_t* __restrict__ primes,
                                               const uint64_t* __restrict__ iroots,
                                               const uint64_t* __restrict__ iroots_s) {
  uint64_t* poly = ws + (long long)blockIdx.z * kN;
  const uint64_t* w = iroots + (long long)l * kN;
  const uint64_t* w_ = iroots_s + (long long)l * kN;
  const uint64_t p = primes[l], tp = p << 1;
  const int group = threadIdx.x / 8, lane = threadIdx.x % 8;
  const int n0 = 8 * blockIdx.x + lane;
  uint64_t x[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = poly[group * 8192 + j * 1024 + n0];
  iradix8(x, 8 + group, w, w_, p, tp);
  __shared__ uint64_t tr[8][9][9];
#pragma unroll
  for (int j = 0; j < 8; ++j) tr[lane][j][group] = x[j];
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = tr[lane][group][j];
  iradix8(x, 1, w, w_, p, tp);
#pragma unroll
  for (int j = 0; j < 8; ++j) poly[j * 8192 + group * 1024 + n0] = x[j] >= p ? x[j] - p : x[j];
}
__global__ void __launch_bounds__(64) rs_ntt1(RsIO io, int batch, int l, const uint64_t* ws, uint64_t half,
                                              const uint64_t* __restrict__ diffs, const uint64_t* __restrict__ csw,
                                              const uint64_t* __restrict__ csw_s,
                                              const uint64_t* __restrict__ primes,
                                              const uint64_t* __restrict__ roots,
                                              const uint64_t* __restrict__ roots_s) {
  const int limb = blockIdx.y, z = blockIdx.z, comp = z / batch, item = z - comp * batch;
  const uint64_t* t = ws + (long long)z * kN;
  uint64_t* o = (comp ? io.out1 : io.out0) + item * io.out_stride + (long long)limb * kN;
  const uint64_t* w = roots + (long long)limb * kN;
  const uint64_t* w_ = roots_s + (long long)limb * kN;
  const uint64_t p = primes[limb], tp = p << 1, df = diffs[limb], c = csw[limb], cs = csw_s[limb];
  const int group = threadIdx.x / 8, lane = threadIdx.x % 8;
  const int n0 = 8 * blockIdx.x + lane;
  uint64_t x[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint64_t tv = t[j * 8192 + group * 1024 + n0];
    uint64_t sv = tv + (tv > half ? df : uint64_t{0});
    if (sv >= p) sv -= p;
    x[j] = shoup_full(sv, c, cs, p);
  }
  radix8(x, 1, w, w_, p, tp);
  __shared__ uint64_t tr[8][9][9];
#pragma unroll
  for (int j = 0; j < 8; ++j) tr[lane][j][group] = x[j];
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = tr[lane][group][j];
  radix8(x, 8 + group, w, w_, p, tp);
#pragma unroll
  for (int j = 0; j < 8; ++j) o[group * 8192 + j * 1024 + n0] = x[j];
}
__global__ void __launch_bounds__(kXThreads) rs_ntt2(RsIO io, int batch, const uint64_t* __restrict__ cfin,
                                                     const uint64_t* __restrict__ cfin_s,
                                                     const uint64_t* __restrict__ primes,
                                                     const uint64_t* __restrict__ roots,
                                                     const uint64_t* __restrict__ roots_s) {
  const int limb = blockIdx.y, z = blockIdx.z, comp = z / batch, item = z - comp * batch;
  const uint64_t p = primes[limb];
  uint64_t* o = (comp ? io.out1 : io.out0) + item * io.out_stride + (long long)limb * kN;
  __shared__ uint64_t tile[2][kXWarps][kWarp + 1];
  const ulonglong2 c = ntt_phase2_pair(o, blockIdx.x, roots + (long long)limb * kN, roots_s + (long long)limb * kN,
                                       p, p << 1, tile);
  const long long pos = blockIdx.x * kData + 2 * threadIdx.x;
  const ulonglong2 x = *reinterpret_cast<const ulonglong2*>((comp ? io.in1 : io.in0) + item * io.in_stride
                                                             + (long long)limb * kN + pos);
  const uint64_t f = cfin[limb], fs = cfin_s[limb];
  *reinterpret_cast<ulonglong2*>(o + pos) = {add_mod(c.x, shoup_full(x.x, f, fs, p), p),
                                             add_mod(c.y, shoup_full(x.y, f, fs, p), p)};
}

// in [batch][>= limbs][N] (one or two components) -> out [batch][limbs - 1][N].
// Workspace: comps * batch * N words. diffs = switch_modulus_map row of limb limbs - 1,
// csw = qlql_inv_mod_ql_div_ql_mod_q (+ (L - limbs)(L - 1)), cfin = q_inv_mod_q (+ (limbs - 1) L).
extern "C" int sylph_rescale(const uint64_t* in0, const uint64_t* in1, long long in_stride, uint64_t* out0,
                             uint64_t* out1, long long out_stride, int batch, int limbs, uint64_t half,
                             uint64_t* ws, const uint64_t* diffs, const uint64_t* csw, const uint64_t* csw_s,
                             const uint64_t* cfin, const uint64_t* cfin_s, const uint64_t* primes,
                             const uint64_t* roots, const uint64_t* roots_s, const uint64_t* iroots,
                             const uint64_t* iroots_s, void* stream_ptr) {
  if (limbs < 2) return -1;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  const int l = limbs - 1, comps = in1 ? 2 : 1;
  // Items per round so that the round's output (comps * l polynomials per item) stays in L2
  // between NTT phase 1 and the in-place phase 2.
  const int chunk = std::max(1, std::min(batch, 96 / (comps * l)));
  for (int s = 0; s < batch; s += chunk) {
    const int nb = std::min(chunk, batch - s), z = comps * nb;
    RsIO io{in0 + s * in_stride, in1 ? in1 + s * in_stride : nullptr, in_stride, out0 + s * out_stride,
            out1 ? out1 + s * out_stride : nullptr, out_stride};
    rs_intt1<<<dim3(64, 1, z), kXThreads, 0, stream>>>(io, nb, l, ws, primes, iroots, iroots_s);
    rs_intt2<<<dim3(kN / 512, 1, z), 64, 0, stream>>>(l, ws, primes, iroots, iroots_s);
    rs_ntt1<<<dim3(kN / 512, l, z), 64, 0, stream>>>(io, nb, l, ws, half, diffs, csw, csw_s, primes, roots, roots_s);
    rs_ntt2<<<dim3(64, l, z), kXThreads, 0, stream>>>(io, nb, cfin, cfin_s, primes, roots, roots_s);
  }
  return static_cast<int>(cudaGetLastError());
}

// Items are processed in chunks whose intermediates (about 2 * (K + limbs) polynomials
// per item) stay resident in L2 between the launches.
static int chunk_of(int batch, int limbs) { return std::max(1, std::min(batch, 24 / std::max(1, limbs))); }

// ModDown of the extended pair (in0, in1) [batch, limbs + K, N] into (out0, out1)
// [batch, limbs, N]; optionally adds (add0, add1) [batch, limbs, N].
// Workspace: batch * 2 * (K + limbs) * N words.
extern "C" int sylph_moddown_pair(const uint64_t* in0, const uint64_t* in1, long long in_stride,
                                  uint64_t* out0, uint64_t* out1, long long out_stride,
                                  const uint64_t* add0, const uint64_t* add1, long long add_stride,
                                  int batch, int limbs, int K, int L, uint64_t* workspace,
                                  const uint64_t* primes, const uint64_t* ratios, const uint64_t* ks,
                                  const uint64_t* roots, const uint64_t* roots_s,
                                  const uint64_t* iroots, const uint64_t* iroots_s,
                                  const uint64_t* hat, const uint64_t* hat_s, const uint64_t* prod_q,
                                  const uint64_t* inv, const uint64_t* inv_s, void* stream_ptr) {
  if (K > kMaxP) return -1;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  const Tables t{primes, ratios, ks, roots, roots_s, iroots, iroots_s, hat, hat_s, prod_q, inv, inv_s};
  const int chunk = chunk_of(batch, limbs);
  for (int s = 0; s < batch; s += chunk) {
    const int nb = std::min(chunk, batch - s);
    Pair io{in0 + s * in_stride, in1 + s * in_stride, in_stride, out0 + s * out_stride, out1 + s * out_stride,
            out_stride, add0 ? add0 + s * add_stride : nullptr, add1 ? add1 + s * add_stride : nullptr, add_stride};
    launch_moddown<false>(io, Rot{}, nb, limbs, K, L, workspace, t, stream);
  }
  return static_cast<int>(cudaGetLastError());
}

// Key switch of a batch: (out0, out1) [batch, limbs, N] = KS(digits) brought down (+ c0 on
// component 0, gathered through `map`) (+ add), from the ModUp digits of the switched
// polynomial [batch, beta * (limbs + K), N] and the key (kb, ka) [>= beta, mult_len, N] starting
// its special limbs at sms. map = gather map of the automorphism in the NTT domain (same-offset
// rotation of the batch) or null (identity: relinearization with add = (d0, d1), c0 = null).
// p1 != 0: digits are NTT phase 1 outputs from sylph_modup_p1 and `own` [batch, >= limbs, N] is
// the switched polynomial (its limbs stand in for the digits' own limbs). Same residues as
// EasyFHE's batched rotation / key switch.
extern "C" int sylph_rotate_pair(const uint64_t* digits, long long d_stride, int beta, const uint64_t* kb,
                                 const uint64_t* ka, int mult_len, int sms, const int* map, const uint64_t* c0,
                                 long long c0_stride, const uint64_t* own, long long own_stride, int alpha, int p1,
                                 uint64_t* out0, uint64_t* out1, long long out_stride,
                                 const uint64_t* add0, const uint64_t* add1, long long add_stride, int batch,
                                 int limbs, int K, int L, uint64_t* workspace, const uint64_t* primes,
                                 const uint64_t* ratios, const uint64_t* ks, const uint64_t* roots,
                                 const uint64_t* roots_s, const uint64_t* iroots, const uint64_t* iroots_s,
                                 const uint64_t* hat, const uint64_t* hat_s, const uint64_t* prod_q,
                                 const uint64_t* inv, const uint64_t* inv_s, void* stream_ptr) {
  if (K > kMaxP || (p1 && (own == nullptr || alpha < 1))) return -1;
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  const Tables t{primes, ratios, ks, roots, roots_s, iroots, iroots_s, hat, hat_s, prod_q, inv, inv_s};
  // Larger chunks than for ModDown: consecutive items then share the rotation-key reads.
  const int chunk = std::min(batch, 8);
  for (int s = 0; s < batch; s += chunk) {
    const int nb = std::min(chunk, batch - s);
    Pair io{nullptr, nullptr, 0, out0 + s * out_stride, out1 + s * out_stride, out_stride,
            add0 ? add0 + s * add_stride : nullptr, add1 ? add1 + s * add_stride : nullptr, add_stride};
    Rot rot{digits + s * d_stride, d_stride, limbs + K, kb, ka, mult_len, sms, beta, map,
            c0 ? c0 + s * c0_stride : nullptr, c0_stride, own ? own + s * own_stride : nullptr, own_stride, limbs,
            alpha, p1};
    launch_moddown<true>(io, rot, nb, limbs, K, L, workspace, t, stream);
  }
  return static_cast<int>(cudaGetLastError());
}

// out_bx/out_ax [batch, limbs + K, N] = extended rotations of (c0, c1) [limbs, N] from the
// ModUp digits of c1 [beta * (limbs + K), N] for the `batch` table entries (device memory).
extern "C" int sylph_fast_rotate_ext(uint64_t* out_bx, uint64_t* out_ax, const uint64_t* digits, const void* table,
                                     const uint64_t* c0, const uint64_t* c1, int batch, int limbs, int K, int L,
                                     int beta, const uint64_t* primes, const uint64_t* ratios, const uint64_t* ks,
                                     const uint64_t* pmodq, void* stream_ptr) {
  fast_rotate_ext<<<dim3(kN / 256, limbs + K, batch), 256, 0, static_cast<cudaStream_t>(stream_ptr)>>>(
      out_bx, out_ax, digits, static_cast<const FastEntry*>(table), c0, c1, limbs, K, L, beta, primes, ratios, ks,
      pmodq);
  return static_cast<int>(cudaGetLastError());
}

// Tensor product raw [3][limbs][N] of (c0, c1) x (d0, d1), limb stride N.
extern "C" int sylph_hmul_raw(uint64_t* raw, const uint64_t* c0, const uint64_t* c1, const uint64_t* d0,
                              const uint64_t* d1, int limbs, int batch, long long sc, long long sd,
                              const uint64_t* primes, const uint64_t* ratios, const uint64_t* ks, void* stream_ptr) {
  hm_raw<<<dim3(kN / 256, limbs, batch), 256, 0, static_cast<cudaStream_t>(stream_ptr)>>>(raw, c0, c1, d0, d1, limbs, sc,
                                                                                          sd, primes, ratios, ks);
  return static_cast<int>(cudaGetLastError());
}

struct HmArgs {
  Rot rot; HMul hm; int limbs, K, L, post_op; uint64_t* ws; uint64_t* ys; Tables t;
};

// Stage s (0..6) of the relinearize + rescale pipeline.
static void hmul_stage(int s, const HmArgs& a, cudaStream_t st) {
  const Tables& t = a.t;
  const int limbs = a.limbs, K = a.K, l = limbs - 1, nb = a.hm.batch, z = 2 * nb;
  switch (s) {
    case 0:
      rot_intt1<<<dim3(nb, 64, K), kXThreads, 0, st>>>(a.rot, nb, limbs, K, a.L, a.ws, t.primes, t.ratios, t.ks,
                                                       t.roots, t.roots_s, t.iroots, t.iroots_s);
      break;
    case 1: md_intt2<<<dim3(kN / 512, K, z), 64, 0, st>>>(K, a.L, a.ws, t.primes, t.iroots, t.iroots_s, t.hat, t.hat_s);
      break;
    case 2:
      launch_md_conv_ntt1<false>(limbs, K, a.ws, a.ys, t, a.hm, l, 1, z, st);
      break;
    case 3:
      hm_last<<<dim3(nb, 64), kXThreads, 0, st>>>(a.rot, a.hm, limbs, a.ys, const_cast<uint64_t*>(a.hm.t), t.primes,
                                                  t.ratios, t.ks, t.roots, t.roots_s, t.iroots, t.iroots_s);
      break;
    case 4:
      md_intt2<<<dim3(kN / 512, 1, z), 64, 0, st>>>(1, l, const_cast<uint64_t*>(a.hm.t), t.primes, t.iroots,
                                                    t.iroots_s, nullptr, nullptr);
      break;
    case 5:
      launch_md_conv_ntt1<true>(limbs, K, a.ws, a.ys, t, a.hm, 0, l, z, st);
      break;
    default: {
      const dim3 fg(nb, 64, l);
#define HM_FIN(P) hm_finish<P><<<fg, kXThreads, 0, st>>>(a.rot, a.hm, limbs, a.ys, t.primes, t.ratios, t.ks, \
                                                         t.roots, t.roots_s)
      switch (a.post_op) {
        case 0: HM_FIN(0); break;
        case 1: HM_FIN(1); break;
        case 2: HM_FIN(2); break;
        case 3: HM_FIN(3); break;
        default: HM_FIN(4); break;
      }
#undef HM_FIN
    }
  }
}
constexpr int kHmStages = 7;

// Relinearize + rescale from raw and the ModUp digits of raw[2] [beta * (limbs + K), N]
// into (out0, out1) [batch][limbs - 1, N] (+ post op 1 add / 2 sub / 3 scalar / 4 plaintext; the
// scalar is shared, the add / sub / plaintext operands have item stride post_stride).
// Same residues as EasyFHE's hmul_relin_rescale per item. Workspace: 2 * batch * (K + limbs + 1) * N words.
// p1 != 0: the digits are NTT phase 1 outputs (sylph_modup_p1 of raw[2]).
// With reps > 0 nothing else runs: each stage is timed over reps launches into times[7] (ms).
extern "C" int sylph_hmul_relin_rescale(const uint64_t* digits, int beta, int alpha, int p1, const uint64_t* kb,
                                        const uint64_t* ka, int mult_len, int sms, const int* imap,
                                        const uint64_t* raw,
                                        uint64_t* out0, uint64_t* out1, const uint64_t* post0,
                                        const uint64_t* post1, const uint64_t* scalar, int post_op, int dbl,
                                        unsigned long long half, const uint64_t* diffs, const uint64_t* csw,
                                        const uint64_t* csw_s, const uint64_t* cfin, const uint64_t* cfin_s,
                                        int limbs, int K, int L, int batch, long long post_stride,
                                        uint64_t* workspace, const uint64_t* primes,
                                        const uint64_t* ratios, const uint64_t* ks, const uint64_t* roots,
                                        const uint64_t* roots_s, const uint64_t* iroots, const uint64_t* iroots_s,
                                        const uint64_t* hat, const uint64_t* hat_s, const uint64_t* prod_q,
                                        const uint64_t* inv, const uint64_t* inv_s, void* stream_ptr, int reps,
                                        float* times, const uint64_t* c0, const uint64_t* c1, const uint64_t* d0,
                                        const uint64_t* d1, long long sc, long long sd) {
  if (K > kMaxP || limbs < 2 || post_op < 0 || post_op > 4 || batch < 1 || (p1 && alpha < 1)) return -1;
  if (!raw && !(c0 && c1 && d0 && d1)) return -1;
  cudaStream_t st = static_cast<cudaStream_t>(stream_ptr);
  uint64_t* ys = workspace + 2LL * batch * K * kN;
  uint64_t* tws = ys + 2LL * batch * limbs * kN;
  const long long dstride = (long long)beta * (limbs + K) * kN;
  // The ModUp input (raw[2] = c1 * d1): stored, or the product of the own limbs' factors.
  const Rot rot = raw ? Rot{digits, dstride, limbs + K, kb, ka, mult_len, sms, beta, imap, nullptr, 0,
                            raw + 2LL * batch * limbs * kN, (long long)limbs * kN, limbs, alpha, p1}
                      : Rot{digits, dstride, limbs + K, kb, ka, mult_len, sms, beta, imap, nullptr, 0,
                            c1, sc, limbs, alpha, p1, d1, sd};
  HmArgs a{rot,
           HMul{raw, tws, diffs, (uint64_t)half, csw, csw_s, cfin, cfin_s, inv, inv_s, post0, post1, scalar, out0,
                out1, dbl, batch, (long long)limbs * kN, post_stride, (long long)(limbs - 1) * kN, c0, c1, d0, d1,
                sc, sd},
           limbs, K, L, post_op, workspace, ys,
           Tables{primes, ratios, ks, roots, roots_s, iroots, iroots_s, hat, hat_s, prod_q, inv, inv_s}};
  if (reps <= 0) {
    for (int s = 0; s < kHmStages; ++s) hmul_stage(s, a, st);
    return static_cast<int>(cudaGetLastError());
  }
  cudaEvent_t ev[kHmStages + 1];
  for (auto& e : ev) cudaEventCreate(&e);
  for (int s = 0; s < kHmStages; ++s) {
    cudaEventRecord(ev[s], st);
    for (int r = 0; r < reps; ++r) hmul_stage(s, a, st);
    cudaEventRecord(ev[s + 1], st);
  }
  cudaEventSynchronize(ev[kHmStages]);
  for (int s = 0; s < kHmStages; ++s) {
    cudaEventElapsedTime(&times[s], ev[s], ev[s + 1]);
    times[s] /= reps;
  }
  for (auto& e : ev) cudaEventDestroy(e);
  return static_cast<int>(cudaGetLastError());
}

// Debug: per-kernel times (ms, averaged over reps) of sylph_rotate_pair (first chunk of up to
// 8 items, no addend) into times[4].
extern "C" int sylph_rotate_pair_profile(const uint64_t* digits, long long d_stride, int beta, const uint64_t* kb,
                                         const uint64_t* ka, int mult_len, int sms, const int* map,
                                         const uint64_t* c0, long long c0_stride, const uint64_t* own,
                                         long long own_stride, int alpha, int p1, uint64_t* out0, uint64_t* out1,
                                         long long out_stride, int batch, int limbs, int K, int L,
                                         uint64_t* workspace, const uint64_t* primes, const uint64_t* ratios,
                                         const uint64_t* ks, const uint64_t* roots, const uint64_t* roots_s,
                                         const uint64_t* iroots, const uint64_t* iroots_s, const uint64_t* hat,
                                         const uint64_t* hat_s, const uint64_t* prod_q, const uint64_t* inv,
                                         const uint64_t* inv_s, int reps, float* times) {
  const Tables t{primes, ratios, ks, roots, roots_s, iroots, iroots_s, hat, hat_s, prod_q, inv, inv_s};
  const int nb = std::min(batch, 8), z = 2 * nb;
  Pair io{nullptr, nullptr, 0, out0, out1, out_stride, nullptr, nullptr, 0};
  Rot rot{digits, d_stride, limbs + K, kb, ka, mult_len, sms, beta, map, c0, c0_stride, own, own_stride, limbs,
          alpha, p1};
  uint64_t* ws = workspace;
  uint64_t* ys = workspace + (long long)z * K * kN;
  cudaEvent_t ev[5];
  for (auto& e : ev) cudaEventCreate(&e);
  for (int s = 0; s < 4; ++s) {
    cudaEventRecord(ev[s]);
    for (int r = 0; r < reps; ++r) {
      if (s == 0)
        rot_intt1<<<dim3(nb, 64, K), kXThreads>>>(rot, nb, limbs, K, L, ws, primes, ratios, ks, roots, roots_s,
                                                  iroots, iroots_s);
      if (s == 1) md_intt2<<<dim3(kN / 512, K, z), 64>>>(K, L, ws, primes, iroots, iroots_s, hat, hat_s);
      if (s == 2) launch_md_conv_ntt1<false>(limbs, K, ws, ys, t, HMul{}, 0, limbs, z, nullptr);
      if (s == 3)
        rot_finish<false><<<dim3(nb, 64, limbs), kXThreads>>>(io, rot, nb, limbs, ys, inv, inv_s, primes, ratios, ks,
                                                              roots, roots_s);
    }
    cudaEventRecord(ev[s + 1]);
  }
  cudaEventSynchronize(ev[4]);
  for (int s = 0; s < 4; ++s) {
    cudaEventElapsedTime(&times[s], ev[s], ev[s + 1]);
    times[s] /= reps;
  }
  for (auto& e : ev) cudaEventDestroy(e);
  return static_cast<int>(cudaGetLastError());
}

// Debug: per-kernel times (ms, averaged over reps) of sylph_moddown_pair into times[4].
extern "C" int sylph_moddown_pair_profile(const uint64_t* in0, const uint64_t* in1, long long in_stride,
                                          uint64_t* out0, uint64_t* out1, long long out_stride, int batch,
                                          int limbs, int K, int L, uint64_t* workspace, const uint64_t* primes,
                                          const uint64_t* ratios, const uint64_t* ks, const uint64_t* roots,
                                          const uint64_t* roots_s, const uint64_t* iroots,
                                          const uint64_t* iroots_s, const uint64_t* hat, const uint64_t* hat_s,
                                          const uint64_t* prod_q, const uint64_t* inv, const uint64_t* inv_s,
                                          int reps, float* times) {
  Pair io{in0, in1, in_stride, out0, out1, out_stride, nullptr, nullptr, 0};
  const int z = 2 * batch;
  uint64_t* ws = workspace;
  uint64_t* ys = workspace + (long long)z * K * kN;
  cudaEvent_t ev[5];
  for (auto& e : ev) cudaEventCreate(&e);
  for (int s = 0; s < 4; ++s) {
    cudaEventRecord(ev[s]);
    for (int r = 0; r < reps; ++r) {
      if (s == 0)
        md_intt1<<<dim3(64, K, z), kXThreads>>>(io, batch, limbs, K, L, ws, primes, iroots, iroots_s);
      if (s == 1) md_intt2<<<dim3(kN / 512, K, z), 64>>>(K, L, ws, primes, iroots, iroots_s, hat, hat_s);
      if (s == 2) {
        const Tables t{primes, ratios, ks, roots, roots_s, iroots, iroots_s, hat, hat_s, prod_q, inv, inv_s};
        launch_md_conv_ntt1<false>(limbs, K, ws, ys, t, HMul{}, 0, limbs, z, nullptr);
      }
      if (s == 3)
        md_ntt2_finish<false><<<dim3(64, limbs, z), kXThreads>>>(io, batch, limbs, ys, inv, inv_s, primes, roots,
                                                                 roots_s);
    }
    cudaEventRecord(ev[s + 1]);
  }
  cudaEventSynchronize(ev[4]);
  for (int s = 0; s < 4; ++s) {
    cudaEventElapsedTime(&times[s], ev[s], ev[s + 1]);
    times[s] /= reps;
  }
  for (auto& e : ev) cudaEventDestroy(e);
  return static_cast<int>(cudaGetLastError());
}
