// Fused Rhombus PCMM for EasyFHE (uint64 RNS, N = 2^16).
//
// The unfused path materializes every weight plaintext: a float64 scatter buffer, the
// encoded integer polynomial, two NTT passes in place and a read by the MAC kernel.
// Here kernel A computes each plaintext coefficient from the device weights on the fly
// and runs the first ten NTT rounds on it; kernel B runs the last six rounds inside a warp and
// multiplies-accumulates the result with the ciphertexts in registers. The plaintexts are streamed in L2-sized chunks
// (sylph_pcmm), so they never round-trip through DRAM. The modular arithmetic (encode rounding and
// reduction, NTT butterflies and roots, 128-bit MAC with one Barrett reduction) follows
// EasyFHE's Encode.cu, NttImpl.cu (same root order and lazy butterflies) and
// FusedPMulSum.cu, so the products are the same residues as encode + grouped MAC.
#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

namespace sylph {

constexpr int kLogN = 16;
constexpr int kN = 1 << kLogN;
constexpr int kWarp = 32;
constexpr long long kMax64 = 9223372036854775295LL;

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
__device__ __forceinline__ uint64_t barrett64(uint64_t in, uint64_t prime, uint64_t ratio, uint64_t k) {
  uint64_t hi = __umul64hi(in, ratio);
  hi >>= (k - 64);
  uint64_t res = in - hi * prime;
  if (res >= prime) res -= prime;
  return res;
}
// IEEE negation is a sign flip and llround (round half away from zero) a function of the bits:
// both are done on the integer pipes (the FP64 pipe of this GPU issues one warp per 16 cycles).
__device__ __forceinline__ double neg_bits(double x) {
  return __longlong_as_double(__double_as_longlong(x) ^ (1LL << 63));
}
// llround(x) for |x| < 2^63 (zeros and subnormals round to 0)
__device__ __forceinline__ long long llround_bits(double x) {
  const long long b = __double_as_longlong(x);
  const int e = (int)((b >> 52) & 0x7ff);
  const long long m = (b & ((1LL << 52) - 1)) | (1LL << 52);   // |x| = m 2^(e - 1075)
  const int sh = 1075 - e;
  long long q;
  if (e == 0 || sh > 53) q = 0;
  else if (sh <= 0) q = m << (-sh);
  else q = (m + (1LL << (sh - 1))) >> sh;                    // half up on the magnitude
  return (b < 0) ? -q : q;
}
__device__ __forceinline__ uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t mod) {
  return a >= b ? a - b : a + mod - b;
}
__device__ __forceinline__ uint64_t add_mod(uint64_t a, uint64_t b, uint64_t mod) {
  const uint64_t s = a + b;
  return s >= mod ? s - mod : s;
}
__device__ __forceinline__ uint64_t shoup(uint64_t a, uint64_t b, uint64_t b_shoup, uint64_t prime) {
  const uint64_t hi = __umul64hi(b_shoup, a);
  return a * b - hi * prime;
}
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
template <int NumRounds>
__device__ __forceinline__ void warp_butterfly(uint64_t& first, uint64_t& second, uint32_t& stage, uint32_t lane,
                                               const uint64_t* __restrict__ w, const uint64_t* __restrict__ ws,
                                               uint64_t p, uint64_t tp) {
  butterfly(first, second, w[stage], ws[stage], p, tp);
#pragma unroll
  for (int s = NumRounds - 2; s >= 0; --s) {
    const uint32_t offset = 1u << s;
    const bool lower = (lane & offset) == 0;
    auto ex = lower ? second : first;
    ex = __shfl_xor_sync(0xFFFFFFFF, ex, offset);
    if (lower) second = ex; else first = ex;
    stage <<= 1;
    const uint32_t ri = stage + (lane >> s);
    butterfly(first, second, w[ri], ws[ri], p, tp);
  }
}

// ---- NTT split ------------------------------------------------------------------------------------
// Round l (0 <= l < 16) of the N = 2^16 NTT pairs elements n and n + 2^(15 - l) with root
// roots[2^l + (n >> (16 - l))] (EasyFHE's NttImpl.cu root order). Kernel A runs rounds 0..9: they
// stay inside the column n mod 64 (1024 elements, stride 64). Kernel B runs rounds 10..15: they
// stay inside the aligned 64-element block n / 64, one warp per block, lane l holding elements
// l and l + 32, so kernel B needs no shared memory and no block barrier.

// ---- kernel A: plaintext coefficients + NTT rounds 0..9 ----------------------------------------
// Plaintext pt = (p * chunk + c) * A + i holds z(X) = sum_{i0, r} sgn * W[pc i0 + p, c K + r] * scale
// at exponent e = shift i0 - r g_i (mod 2N), coefficient e mod N, sign -1 when e >= N.
// shift and K are multiples of 8 and every g_i is 1 mod 256, so for r = 8 q + k the eight
// coefficients C - k g_i (k = 0..7, C = shift i0 - 8 q g_i, C = 0 mod 8) form a chain: they
// receive their terms from the same (i0, q) pairs, with the eight contiguous weights
// W[pc i0 + p, c K + 8 q .. 8 q + 7]. The chains partition the coefficients (the k-th element
// of a chain is k mod 8 below C) and a chain lies in one block of eight residues
// {8 v + 1, ..., 8 v + 8} mod 64 (block v = C / 8 - 1 mod 8, column cg = 7 - k).
// Per automorphism i the chain terms are listed in ents[i][offs[i][slot] .. offs[i][slot + 1])
// in increasing i0, packed as (i0 << 14) | (q << 8) | sign bits of k = 0..7. The slots of
// group v are v * 1024 .. v * 1024 + 1023 and hold the chains (v, m' = perm[i][slot]) in order
// of decreasing term count, so that the lanes of a warp run equally long summation chains
// and warps of empty chains (most of them: a group has 512 terms for 1024 chains) skip the
// FP64 work. (This GPU's FP64 pipe issues one warp instruction per 16 cycles.)
// Kernel A sums the chain terms, reduces the rounded coefficients for every limb and runs
// NTT rounds 0..9 on each column.
#ifndef SYLPH_A_COLS
#define SYLPH_A_COLS 4
#endif
#ifndef SYLPH_A_BLOCKS_PER_SM
#define SYLPH_A_BLOCKS_PER_SM (8 / SYLPH_A_COLS)
#endif
constexpr int kCols = SYLPH_A_COLS;      // columns (residues mod 64) per block of kernel A
constexpr int kBlocks = 64 / kCols;      // blocks per plaintext
constexpr int kPerV = 8 / kCols;         // blocks per chain group v
constexpr int kColThreads = 128;         // threads per column (1024 elements)
constexpr int kColPad = 1024 + 128 + 4;  // padded shared column: pad(m) = m + 2 (m >> 4); column
                                         // stride = 8 mod 32 words, so store_cols is 2-way (optimal)
constexpr int kShared = kCols * kColPad * 8;
constexpr int kThreads = kCols * kColThreads;
constexpr int kChains = 1024 / kThreads; // chains per thread
__device__ __forceinline__ int pad(int m) { return m + 2 * (m >> 4); }

// Element m of column cg (residue kCols blockIdx.x + 1 + cg) is stored at
// row[64 m + kCols blockIdx.x + cg]: the row keeps residue res at index (res - 1) mod 64, so
// the block's columns of one m are one aligned run of 8 kCols bytes; kCols / 2 lanes per m each
// store 16 bytes.
// Expects a barrier after the last write to sh; the caller must put a barrier before sh is
// overwritten again.
__device__ __forceinline__ void store_cols(const uint64_t* __restrict__ sh, uint64_t* __restrict__ o) {
  __syncthreads();
  constexpr int kLanes = kCols / 2, kPerRound = kThreads / kLanes;
  const int q = threadIdx.x % kLanes;
#pragma unroll
  for (int r = 0; r < 1024 / kPerRound; ++r) {
    const int m = threadIdx.x / kLanes + kPerRound * r;
    const int pm = pad(m);
    *reinterpret_cast<ulonglong2*>(o + 64 * m + kCols * blockIdx.x + 2 * q) =
        make_ulonglong2(sh[(2 * q) * kColPad + pm], sh[(2 * q + 1) * kColPad + pm]);
  }
}

// Kernel A: block kPerV v + h of plaintext pt holds the chain coefficients k in [k0, k0 + kCols),
// k0 = kCols (kPerV - 1 - h), of the 1024 chains of v (residues 8 v + 1 + k0' .. 8 v + kCols + k0'
// with k0' = kCols h, local column kCols - 1 - (k - k0)). The block stages the terms of its
// chains through shared memory (one entry load and kCols / 2 16-byte weight loads per term:
// the weights of its k's); thread tau sums those of the chains in slots tau + kThreads u
// (chain m' = perm[slot]), rounds their coefficients and scatters them to column
// kCols - 1 - kk, element m_k = m' + [v = 7, k = 0] - 4 k t_g (mod 1024) with g_i = 1 + 256 t_g; thread
// (cg, t) then transforms column cg from the elements m = t + 128 j, for every limb.
// All blocks of a chain group sum the same terms in the same order: a coefficient's double
// is the same whichever block computes it.
template <int DBG>
__global__ void __launch_bounds__(kThreads, SYLPH_A_BLOCKS_PER_SM) build_phase1(
    uint64_t* __restrict__ out, const double* __restrict__ w, long long w_stride,
    int pc, int p0, int j0, int jc, int a, int k, const int* __restrict__ offs, const int* __restrict__ ents,
    const short* __restrict__ perm, const int* __restrict__ autos, double scale, int use_scale, double sf, int limbs,
    const uint64_t* __restrict__ primes, const uint64_t* __restrict__ diffs,
    const uint64_t* __restrict__ ratios, const uint64_t* __restrict__ ks,
    const uint64_t* __restrict__ roots, const uint64_t* __restrict__ roots_shoup) {
  extern __shared__ uint64_t sh[];
  const int cg = threadIdx.x / kColThreads, t = threadIdx.x % kColThreads;
  const int v = blockIdx.x / kPerV, h = blockIdx.x % kPerV, k0 = kCols * (kPerV - 1 - h), mp = threadIdx.x;
  const int pt = blockIdx.z;            // local plaintext pl * jc + jl of this chunk
  const int p = p0 + pt / jc;
  const int j = j0 + pt % jc;           // ciphertext c * A + i
  const int c = j / a, i = j % a;
  const double* wbase = w + (long long)p * w_stride + (long long)c * k;
  const long long wstep = (long long)pc * w_stride;     // next i0 row
  const int* off = offs + (size_t)i * (kN / 8 + 1) + v * 1024;      // the 1024 slots of v
  const short* prm = perm + (size_t)i * (kN / 8) + v * 1024;
  const int* ent = ents + (size_t)i * a * (k / 8);
  int e0[kChains], e1[kChains], mpr[kChains];                        // terms and chain m' of slots mp + kThreads u
#pragma unroll
  for (int u = 0; u < kChains; ++u) {
    e0[u] = off[mp + kThreads * u]; e1[u] = off[mp + kThreads * u + 1];
    mpr[u] = prm[mp + kThreads * u];
  }
  const int b0 = off[0], b1 = off[1024];                             // terms of the chains of v
  double sum[kChains][kCols];
#pragma unroll
  for (int u = 0; u < kChains; ++u)
#pragma unroll
    for (int kk = 0; kk < kCols; ++kk) sum[u][kk] = 0.0;
  // The terms are staged kStage at a time: thread tau loads term b + tau (entry and its kCols
  // weights) and writes x_kk = sgn (w * scale) to shared; the owner of each chain then adds
  // its staged terms in increasing i0. Every coefficient gets the same three IEEE operations
  // per term in the same order (multiply, negate, add; the intrinsics keep nvcc from
  // contracting them), only the latency of the loads is paid once per stage.
  constexpr int kStage = kThreads;
  double* xs = reinterpret_cast<double*>(sh);                        // [kCols][kStage]
  if (DBG != 1) for (int b = b0; b < b1; b += kStage) {
    const int e = b + mp;
    if (e < b1) {
      const int code = ent[e];
      const double2* wp = reinterpret_cast<const double2*>(wbase + (long long)(code >> 14) * wstep + 8 * ((code >> 8) & 63)) + (k0 >> 1);
      double2 v2[kCols / 2];
#pragma unroll
      for (int u = 0; u < kCols / 2; ++u) v2[u] = wp[u];
#pragma unroll
      for (int kk = 0; kk < kCols; ++kk) {
        const double vv = (kk & 1) ? v2[kk >> 1].y : v2[kk >> 1].x;
        double x = use_scale ? __dmul_rn(vv, scale) : vv;
        if ((code >> (k0 + kk)) & 1) x = neg_bits(x);
        xs[kk * kStage + mp] = x;
      }
    }
    __syncthreads();
#pragma unroll
    for (int u = 0; u < kChains; ++u) {
      const int lo = max(e0[u], b), hi = min(e1[u], b + kStage);
      for (int ee = lo; ee < hi; ++ee) {
#pragma unroll
        for (int kk = 0; kk < kCols; ++kk) {
          const double x = xs[kk * kStage + (ee - b)];
          sum[u][kk] = (ee == e0[u]) ? x : __dadd_rn(sum[u][kk], x);
        }
      }
    }
    __syncthreads();
  }
  long long re[kChains][kCols];
#pragma unroll
  for (int u = 0; u < kChains; ++u) {
    const bool any = __ballot_sync(0xffffffffu, e1[u] > e0[u]) != 0;   // empty chains: sum = +0 -> 0
#pragma unroll
    for (int kk = 0; kk < kCols; ++kk) {
      const long long q = any ? llround_bits(__dmul_rn(sum[u][kk], sf)) : 0;
      re[u][kk] = (q < 0) ? (kMax64 + q) : q;
    }
  }
  const int tg = (autos[i] - 1) >> 8;
  int mk[kChains][kCols];
#pragma unroll
  for (int u = 0; u < kChains; ++u)
#pragma unroll
    for (int kk = 0; kk < kCols; ++kk)
      mk[u][kk] = pad((mpr[u] + ((v == 7 && k0 + kk == 0) ? 1 : 0) - 4 * (k0 + kk) * tg) & 1023);
  uint64_t* mine = sh + cg * kColPad;
  if (DBG == 2) {
#pragma unroll
    for (int u = 0; u < kChains; ++u)
#pragma unroll
      for (int kk = 0; kk < kCols; ++kk) sh[(kCols - 1 - kk) * kColPad + mk[u][kk]] = (uint64_t)re[u][kk];
    for (int limb = 0; limb < limbs; ++limb) store_cols(sh, out + ((size_t)pt * limbs + limb) * kN);
    return;
  }
  for (int limb = 0; limb < limbs; ++limb) {
    const uint64_t prime = primes[limb], tp = prime << 1;
    const uint64_t ratio = ratios[limb], bk = ks[limb], diff = diffs[limb];
    const uint64_t* W = roots + (size_t)limb * kN;
    const uint64_t* Ws = roots_shoup + (size_t)limb * kN;
    uint64_t x[8];
    if (limb) __syncthreads();          // sh free (store_cols of the previous limb done)
#pragma unroll
    for (int u = 0; u < kChains; ++u)
#pragma unroll
      for (int kk = 0; kk < kCols; ++kk) {
        uint64_t r = barrett64((uint64_t)re[u][kk], prime, ratio, bk);
        if (re[u][kk] > (kMax64 >> 1)) r = sub_mod(r, diff, prime);
        sh[(kCols - 1 - kk) * kColPad + mk[u][kk]] = r;
      }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = mine[pad(t + kColThreads * j)];
    // rounds 0..2: m = t + 128 j, roots 1, 2 + (m >> 8), 4 + (m >> 7)
    radix8(x, 1, W, Ws, prime, tp);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) mine[pad(t + kColThreads * j)] = x[j];
    __syncthreads();
    // rounds 3..5: m = (t & 15) + 16 j + 128 (t >> 4), roots 8 + (m >> 7), ...
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = mine[pad((t & 15) + 16 * j + 128 * (t >> 4))];
    radix8(x, 8 + (t >> 4), W, Ws, prime, tp);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) mine[pad((t & 15) + 16 * j + 128 * (t >> 4))] = x[j];
    __syncthreads();
    // rounds 6..8: m = (t & 1) + 2 j + 16 (t >> 1), roots 64 + (m >> 4), ...
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = mine[pad((t & 1) + 2 * j + 16 * (t >> 1))];
    radix8(x, 64 + (t >> 1), W, Ws, prime, tp);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) mine[pad((t & 1) + 2 * j + 16 * (t >> 1))] = x[j];
    __syncthreads();
    // round 9: m = 8 t + j, pairs (2 q, 2 q + 1) with root 512 + (m >> 1) = 512 + 4 t + q
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = mine[pad(8 * t + j)];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int ri = 512 + 4 * t + q;
      butterfly(x[2 * q], x[2 * q + 1], W[ri], Ws[ri], prime, tp);
    }
    // back into the slots this thread just read (no other thread touches them)
#pragma unroll
    for (int j = 0; j < 8; ++j) mine[pad(8 * t + j)] = x[j];
    store_cols(sh, out + ((size_t)pt * limbs + limb) * kN);
  }
}

// The PT rows of one step (same limb, same row index) share every root: one root load per
// butterfly stage for PT independent butterfly chains.
template <int NumRounds, int PT>
__device__ __forceinline__ void warp_butterfly_rows(uint64_t (&first)[PT], uint64_t (&second)[PT], uint32_t& stage,
                                                    uint32_t lane, const uint64_t* __restrict__ w,
                                                    const uint64_t* __restrict__ ws, uint64_t p, uint64_t tp) {
  {
    const uint64_t r = w[stage], rs = ws[stage];
#pragma unroll
    for (int t = 0; t < PT; ++t) butterfly(first[t], second[t], r, rs, p, tp);
  }
#pragma unroll
  for (int s = NumRounds - 2; s >= 0; --s) {
    const uint32_t offset = 1u << s;
    const bool lower = (lane & offset) == 0;
#pragma unroll
    for (int t = 0; t < PT; ++t) {
      auto ex = lower ? second[t] : first[t];
      ex = __shfl_xor_sync(0xFFFFFFFF, ex, offset);
      if (lower) second[t] = ex; else first[t] = ex;
    }
    stage <<= 1;
    const uint32_t ri = stage + (lane >> s);
    const uint64_t r = w[ri], rs = ws[ri];
#pragma unroll
    for (int t = 0; t < PT; ++t) butterfly(first[t], second[t], r, rs, p, tp);
  }
}

// ---- kernel B: NTT rounds 10..15 + grouped MAC ---------------------------------------------------
// acc_p = sum_j ct_j * pt_{p * NC + j} for p in [p0, p0 + PT); this block owns 1024 positions of one limb,
// each warp the 64-element block row = 16 blk + warp. Lane l reads the row elements l and l + 32
// (the pair of round 10); after the five exchanges of rounds 11..15 it holds the elements 2 l and
// 2 l + 1, which it multiplies with the ciphertexts and stores.
// The chunk holds the plaintexts of output groups p0 .. p0 + pcc and ciphertexts j0 .. j0 + jc
// (p-major); the partial sums of the chunk are reduced and added to res (first chunk: stored).
// The result is the canonical residue of the whole sum, as with one 128-bit accumulation.
template <int PT>
__global__ void __launch_bounds__(512) phase2_mac(
    uint64_t* __restrict__ res, const uint64_t* __restrict__ inter, const uint64_t* __restrict__ bx,
    const uint64_t* __restrict__ ax, long long ct_stride, int p0, int pcc, int j0, int jc, int first,
    int groups, int limbs, const uint64_t* __restrict__ primes, const uint64_t* __restrict__ ratios,
    const uint64_t* __restrict__ ks, const uint64_t* __restrict__ roots,
    const uint64_t* __restrict__ roots_shoup) {
  constexpr int kData = 1024;                 // positions per block
  const int pl0 = blockIdx.x * PT, blk = blockIdx.y, limb = blockIdx.z;   // output groups vary fastest: ct reuse in L2
  const uint64_t prime = primes[limb], tp = prime << 1;
  const uint64_t* W = roots + (size_t)limb * kN;
  const uint64_t* Ws = roots_shoup + (size_t)limb * kN;
  const uint32_t lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
  // input row elements of residues lane, lane + 32 of this warp's m (the row keeps residue res
  // at index (res - 1) mod 64, see store_cols)
  const int loc0 = warp * 64 + ((lane + 63) & 63), loc1 = warp * 64 + lane + 31;
  const size_t pos = (size_t)blk * kData + warp * 64 + 2 * lane;   // elements pos, pos + 1 (output)
  u128 sb[PT][2], sa[PT][2];
#pragma unroll
  for (int t = 0; t < PT; ++t) sb[t][0] = sb[t][1] = sa[t][0] = sa[t][1] = {0, 0};
  const int npt = (pcc - pl0 < PT) ? pcc - pl0 : PT;    // output groups of this block
  // One step = one ciphertext j: the PT plaintext rows (one per output group) go through the
  // NTT together; the rows of step j + 1 are fetched before the work of step j and the
  // ciphertext of step j at its start (used after the NTT).
  auto row_at = [&](int j, int t) {
    return inter + ((size_t)((pl0 + t) * jc + j) * limbs + limb) * kN + (size_t)blk * kData;
  };
  auto ct_at = [&](int j) { return (size_t)(j0 + j) * ct_stride + (size_t)limb * kN + pos; };
  uint64_t nf[PT], ns[PT];
#pragma unroll
  for (int t = 0; t < PT; ++t) {
    nf[t] = (t < npt) ? row_at(0, t)[loc0] : 0;
    ns[t] = (t < npt) ? row_at(0, t)[loc1] : 0;
  }
  for (int j = 0; j < jc; ++j) {
    const ulonglong2 cb = *reinterpret_cast<const ulonglong2*>(bx + ct_at(j));
    const ulonglong2 ca = *reinterpret_cast<const ulonglong2*>(ax + ct_at(j));
    uint64_t f[PT], s[PT];
#pragma unroll
    for (int t = 0; t < PT; ++t) { f[t] = nf[t]; s[t] = ns[t]; }
    if (j + 1 < jc) {
#pragma unroll
      for (int t = 0; t < PT; ++t) {
        if (t < npt) {
          const uint64_t* rows = row_at(j + 1, t);
          nf[t] = rows[loc0];
          ns[t] = rows[loc1];
        }
      }
    }
    uint32_t stage = 1024 + blk * 16 + warp;             // round 10: root 2^10 + (n >> 6)
    warp_butterfly_rows<6, PT>(f, s, stage, lane, W, Ws, prime, tp);
#pragma unroll
    for (int t = 0; t < PT; ++t) {
#pragma unroll
      for (int r = 0; r < 3; ++r) {
        if (f[t] >= prime) f[t] -= prime;
        if (s[t] >= prime) s[t] -= prime;
      }
      add128(mul128(cb.x, f[t]), sb[t][0]);
      add128(mul128(cb.y, s[t]), sb[t][1]);
      add128(mul128(ca.x, f[t]), sa[t][0]);
      add128(mul128(ca.y, s[t]), sa[t][1]);
    }
  }
  const uint64_t ratio = ratios[limb], bk = ks[limb];
#pragma unroll
  for (int t = 0; t < PT; ++t) {
    const int pl = pl0 + t;
    if (pl >= pcc) break;
    const size_t o = ((size_t)(p0 + pl) * limbs + limb) * kN + pos;
    uint64_t vbf = barrett128(sb[t][0], prime, ratio, bk), vbs = barrett128(sb[t][1], prime, ratio, bk);
    uint64_t vaf = barrett128(sa[t][0], prime, ratio, bk), vas = barrett128(sa[t][1], prime, ratio, bk);
    uint64_t* ob = res + o;
    uint64_t* oa = res + (size_t)groups * limbs * kN + o;
    if (!first) {
      const ulonglong2 pb = *reinterpret_cast<const ulonglong2*>(ob), pa = *reinterpret_cast<const ulonglong2*>(oa);
      vbf = add_mod(vbf, pb.x, prime); vbs = add_mod(vbs, pb.y, prime);
      vaf = add_mod(vaf, pa.x, prime); vas = add_mod(vas, pa.y, prime);
    }
    *reinterpret_cast<ulonglong2*>(ob) = make_ulonglong2(vbf, vbs);
    *reinterpret_cast<ulonglong2*>(oa) = make_ulonglong2(vaf, vas);
  }
}

}  // namespace sylph

// acc = sum_j ct_j * pt_{p, j} for all output groups p < pc and ciphertexts j < nc, streamed in
// chunks of pcc groups x jc ciphertexts: kernel A writes the chunk's phase-1 plaintexts into
// `inter` (pcc * jc * limbs * N words per buffer) and kernel B consumes them. With nbuf >= 2 the
// chunks are pipelined over two streams: kernel A of chunk c + 1 (side stream, buffer (c + 1) % nbuf)
// overlaps kernel B of chunk c (caller's stream), so the DRAM writes of one chunk and the reads of
// the previous one proceed together. The chunks are still accumulated in order (first chunk stores,
// later chunks add): the same canonical residues as a single accumulation.
static cudaStream_t side_stream() {
  static cudaStream_t s = nullptr;
  if (!s) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  return s;
}

extern "C" int sylph_pcmm(
    void* res, void* inter, const void* w, long long w_stride, int pc, int pcc, int nc, int jc, int a, int k,
    const void* offs, const void* ents, const void* perm, const void* autos, double scale, int use_scale, double sf, int limbs, const void* bx,
    const void* ax, long long ct_stride, const void* primes, const void* diffs, const void* ratios, const void* ks,
    const void* roots, const void* roots_shoup, void* stream, int parts, int pt, int nbuf) {
  cudaStream_t st = (cudaStream_t)stream;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(sylph::build_phase1<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, sylph::kShared);
    cudaFuncSetAttribute(sylph::build_phase1<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, sylph::kShared);
    cudaFuncSetAttribute(sylph::build_phase1<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sylph::kShared);
    attr_set = true;
  }
  const int PT = pt > 0 ? pt : ((pcc >= pc) ? 2 : 1);     // all groups in one chunk: two output groups per block
  const bool pipe = nbuf >= 2 && (parts & 3) == 3;
  cudaStream_t sb = pipe ? side_stream() : st;
  const size_t buf_words = (size_t)pcc * jc * limbs * sylph::kN;
  constexpr int kMaxBuf = 4;
  if (nbuf > kMaxBuf) return -1;
  cudaEvent_t built[kMaxBuf], consumed[kMaxBuf], start;
  if (pipe) {
    cudaEventCreateWithFlags(&start, cudaEventDisableTiming);
    cudaEventRecord(start, st);
    cudaStreamWaitEvent(sb, start, 0);     // inputs (ciphertexts, weights) are ready on the caller's stream
    for (int i = 0; i < nbuf; ++i) {
      cudaEventCreateWithFlags(&built[i], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&consumed[i], cudaEventDisableTiming);
    }
  }
  int chunk = 0;
  for (int p0 = 0; p0 < pc; p0 += pcc) {
    const int np = std::min(pcc, pc - p0);
    for (int j0 = 0; j0 < nc; j0 += jc, ++chunk) {
      const int nj = std::min(jc, nc - j0);
      const int b = pipe ? chunk % nbuf : 0;
      uint64_t* buf = (uint64_t*)inter + (size_t)b * buf_words;
      if (pipe && chunk >= nbuf) cudaStreamWaitEvent(sb, consumed[b], 0);   // buffer b is free again
#define BUILD(D)                                                                                                 \
  sylph::build_phase1<D><<<dim3(sylph::kBlocks, 1, np * nj), sylph::kThreads, sylph::kShared, sb>>>( \
      buf, (const double*)w, w_stride, pc, p0, j0, nj, a, k, (const int*)offs, (const int*)ents,                  \
      (const short*)perm, (const int*)autos, scale, use_scale, sf, limbs, (const uint64_t*)primes, (const uint64_t*)diffs,            \
      (const uint64_t*)ratios, (const uint64_t*)ks, (const uint64_t*)roots, (const uint64_t*)roots_shoup)
      if (parts & 1) { if (parts & 4) BUILD(1); else if (parts & 8) BUILD(2); else BUILD(0); }
#undef BUILD
      if (pipe) {
        cudaEventRecord(built[b], sb);
        cudaStreamWaitEvent(st, built[b], 0);
      }
#define MAC(P)                                                                                                   \
  sylph::phase2_mac<P><<<dim3((np + P - 1) / P, 64, limbs), 512, 0, st>>>(                                        \
      (uint64_t*)res, buf, (const uint64_t*)bx, (const uint64_t*)ax, ct_stride, p0, np, j0, nj,                   \
      j0 == 0, pc, limbs, (const uint64_t*)primes, (const uint64_t*)ratios, (const uint64_t*)ks,                  \
      (const uint64_t*)roots, (const uint64_t*)roots_shoup)
      if (parts & 2) { if (PT >= 4) MAC(4); else if (PT == 2) MAC(2); else MAC(1); }
#undef MAC
      if (pipe) cudaEventRecord(consumed[b], st);
    }
  }
  if (pipe) {
    cudaEventDestroy(start);
    for (int i = 0; i < nbuf; ++i) { cudaEventDestroy(built[i]); cudaEventDestroy(consumed[i]); }
  }
  return (int)cudaGetLastError();
}
