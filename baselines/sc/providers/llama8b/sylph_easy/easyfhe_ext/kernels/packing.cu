// Subring-packing butterfly for EasyFHE (uint64 RNS, NTT domain, N = 2^16).
//
// One packing level splits a batch of m ciphertexts into halves p = [0, h) and q = [h, m) and
// forms a = p + X^h q and b = p - X^h q. EasyFHE's mul_by_monomial does iNTT, a negacyclic
// shift and an NTT; in the NTT domain multiplying by X^h is the pointwise product with
// NTT(X^h), so one pass over p and q gives both a and b. Every value is a fully reduced
// residue, so the outputs equal EasyFHE's mul_by_monomial + homo_add / homo_sub.
#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr int kN = 1 << 16;

__device__ __forceinline__ uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t prime, uint64_t ratio,
                                                uint64_t k) {
  // EasyFHE barrett_reduction_128_64 of the 128-bit product a * b.
  const uint64_t lo = a * b, hi = __umul64hi(a, b);
  uint64_t t1hi = __umul64hi(lo, ratio);
  const uint64_t t2lo = hi * ratio;
  uint64_t t2hi = __umul64hi(hi, ratio);
  asm("add.cc.u64 %0, %0, %1;" : "+l"(t1hi) : "l"(t2lo));
  asm("{addc.cc.u64 %0, %0, %1;}" : "+l"(t2hi) : "l"((unsigned long)0));
  const unsigned shift = static_cast<unsigned>(k - 64);
  t1hi >>= shift;
  t2hi <<= 64 - shift;
  t1hi += t2hi;
  t1hi *= prime;
  uint64_t r = lo - t1hi;
  if (r >= prime) r -= prime;
  return r;
}

// grid (N / 256, limbs, 2 * h): z = comp * h + item.
__global__ void __launch_bounds__(256) pack_butterfly(const uint64_t* __restrict__ x0,
                                                      const uint64_t* __restrict__ x1, long long x_stride,
                                                      uint64_t* __restrict__ a0, uint64_t* __restrict__ a1,
                                                      uint64_t* __restrict__ b0, uint64_t* __restrict__ b1,
                                                      const uint64_t* __restrict__ mono, int h, int limbs,
                                                      const uint64_t* __restrict__ primes,
                                                      const uint64_t* __restrict__ ratios,
                                                      const uint64_t* __restrict__ ks) {
  const int j = blockIdx.x * 256 + threadIdx.x, limb = blockIdx.y;
  const int comp = blockIdx.z / h, item = blockIdx.z - comp * h;
  const uint64_t* x = comp ? x1 : x0;
  const long long off = (long long)limb * kN + j;
  const uint64_t p = primes[limb];
  const uint64_t pv = x[item * x_stride + off];
  const uint64_t qv = x[(item + h) * x_stride + off];
  const uint64_t t = barrett_mul(qv, mono[off], p, ratios[limb], ks[limb]);
  const long long o = (long long)item * limbs * kN + off;
  const uint64_t s = pv + t;
  (comp ? a1 : a0)[o] = s >= p ? s - p : s;
  (comp ? b1 : b0)[o] = pv >= t ? pv - t : pv + p - t;
}

}  // namespace

// x0, x1: components [2h, >= limbs, N] with batch stride x_stride (limb stride N);
// a, b: [h, limbs, N]; mono: NTT(X^h) [limbs, N].
extern "C" int sylph_pack_butterfly(const uint64_t* x0, const uint64_t* x1, long long x_stride, uint64_t* a0,
                                    uint64_t* a1, uint64_t* b0, uint64_t* b1, const uint64_t* mono, int h,
                                    int limbs, const uint64_t* primes, const uint64_t* ratios, const uint64_t* ks,
                                    void* stream) {
  pack_butterfly<<<dim3(kN / 256, limbs, 2 * h), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      x0, x1, x_stride, a0, a1, b0, b1, mono, h, limbs, primes, ratios, ks);
  return static_cast<int>(cudaGetLastError());
}
