// Batched EvalMod tail table for EasyFHE (uint64 RNS, N = 2^16).
//
// EasyFHE's grouped_scalar_weighted_acc forms, for one ciphertext, tail_s = sum_d T_d * c[s][d]
// with a 128-bit accumulator and one Barrett reduction (FusedPMulSum.cu). Here every item of
// the batch is done in one launch, reading the Chebyshev basis T_d in place: the integer sum
// and the reduction are the same, so every residue equals the per-item kernel.
#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr int kN = 1 << 16;
constexpr int kMaxD = 16;
constexpr int kMaxS = 16;

struct u128 { uint64_t hi, lo; };

__device__ __forceinline__ u128 mul128(uint64_t a, uint64_t b) {
  u128 r; r.lo = a * b; r.hi = __umul64hi(a, b); return r;
}
__device__ __forceinline__ void add128(const u128 a, u128& r) {
  asm("add.cc.u64 %1, %3, %1;" "addc.cc.u64 %0, %2, %0;" : "+l"(r.hi), "+l"(r.lo) : "l"(a.hi), "l"(a.lo));
}
// EasyFHE barrett_reduction_128_64.
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

// One basis polynomial T_d: both components, item stride in words.
struct Src { const uint64_t* bx; const uint64_t* ax; long long stride; };

// res[2][S][B][limbs][N]; scalars [S][D][limbs].
__global__ void __launch_bounds__(256) tail_table(uint64_t* __restrict__ res, const Src* __restrict__ src,
                                                  const uint64_t* __restrict__ scalars, int S, int D, int limbs,
                                                  int B, const uint64_t* __restrict__ primes,
                                                  const uint64_t* __restrict__ ratios,
                                                  const uint64_t* __restrict__ ks) {
  const int limb = blockIdx.y, item = blockIdx.z;
  const long long pos = (long long)limb * kN + blockIdx.x * 256 + threadIdx.x;
  __shared__ uint64_t sc[kMaxS * kMaxD];
  for (int i = threadIdx.x; i < S * D; i += blockDim.x) sc[i] = scalars[(long long)i * limbs + limb];
  __syncthreads();
  uint64_t xb[kMaxD], xa[kMaxD];
#pragma unroll
  for (int d = 0; d < kMaxD; ++d) {
    if (d < D) {
      const long long o = item * src[d].stride + pos;
      xb[d] = src[d].bx[o];
      xa[d] = src[d].ax[o];
    }
  }
  const uint64_t p = primes[limb], r = ratios[limb], k = ks[limb];
  const long long plane = (long long)limbs * kN, comp = (long long)S * B * plane;
  for (int s = 0; s < S; ++s) {
    u128 sb{0, 0}, sa{0, 0};
#pragma unroll
    for (int d = 0; d < kMaxD; ++d) {
      if (d < D) {
        const uint64_t c = sc[s * D + d];
        add128(mul128(xb[d], c), sb);
        add128(mul128(xa[d], c), sa);
      }
    }
    const long long o = ((long long)s * B + item) * plane + pos;
    res[o] = barrett128(sb, p, r, k);
    res[comp + o] = barrett128(sa, p, r, k);
  }
}

}  // namespace

extern "C" int sylph_tail_table(uint64_t* res, const void* src, const uint64_t* scalars, int S, int D, int limbs,
                                int B, const uint64_t* primes, const uint64_t* ratios, const uint64_t* ks,
                                void* stream) {
  if (S < 1 || S > kMaxS || D < 1 || D > kMaxD) return -1;
  tail_table<<<dim3(kN / 256, limbs, B), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      res, static_cast<const Src*>(src), scalars, S, D, limbs, B, primes, ratios, ks);
  return static_cast<int>(cudaGetLastError());
}
