// Plaintext multiply-accumulate over several ciphertexts for EasyFHE (uint64 RNS, NTT domain).
//
// out[b] = sum_t ct_t[b] * pt_t[b] for a batch of B items; every ciphertext or plaintext may be
// shared by all items (batch stride 0). The slot engine otherwise runs one homo_mul_pt per term
// and homo_add between them, each a full pass over the batch; here the 128-bit products are
// summed in registers and reduced once (EasyFHE's FusedPMulSum does the same for its grouped
// MAC), so the residues equal the sequential products and additions mod q.
#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr int kN = 1 << 16;
constexpr int kMaxTerms = 32;

struct u128 { uint64_t hi, lo; };

__device__ __forceinline__ void mac128(uint64_t a, uint64_t b, u128& r) {
  const uint64_t lo = a * b, hi = __umul64hi(a, b);
  asm("add.cc.u64 %1, %3, %1;" "addc.cc.u64 %0, %2, %0;" : "+l"(r.hi), "+l"(r.lo) : "l"(hi), "l"(lo));
}
__device__ __forceinline__ uint64_t barrett128(const u128 in, uint64_t prime, uint64_t ratio, uint64_t k) {
  uint64_t t1hi = __umul64hi(in.lo, ratio);
  const uint64_t t2lo = in.hi * ratio;
  uint64_t t2hi = __umul64hi(in.hi, ratio);
  asm("add.cc.u64 %0, %0, %1;" : "+l"(t1hi) : "l"(t2lo));
  asm("{addc.cc.u64 %0, %0, %1;}" : "+l"(t2hi) : "l"((unsigned long)0));
  const unsigned shift = static_cast<unsigned>(k - 64);
  t1hi >>= shift;
  t2hi <<= 64 - shift;
  t1hi += t2hi;
  t1hi *= prime;
  uint64_t r = in.lo - t1hi;
  if (r >= prime) r -= prime;
  return r;
}

}  // namespace

// Passed by value: no per-call device table.
struct SylphMacArgs {
  long long c0[kMaxTerms], c1[kMaxTerms], pt[kMaxTerms];   // device addresses
  long long cbs[kMaxTerms], pbs[kMaxTerms];                 // batch strides in words (0 = shared)
  int n;
};

namespace {

// grid (N / 512, limbs, batch); each thread owns two consecutive slots.
__global__ void __launch_bounds__(256) pt_mac(uint64_t* __restrict__ out0, uint64_t* __restrict__ out1,
                                              long long out_stride, SylphMacArgs a, int limbs,
                                              const uint64_t* __restrict__ primes,
                                              const uint64_t* __restrict__ ratios,
                                              const uint64_t* __restrict__ ks) {
  const int limb = blockIdx.y, item = blockIdx.z;
  const long long off = (long long)limb * kN + 2 * (blockIdx.x * 256 + threadIdx.x);
  u128 s0x{0, 0}, s0y{0, 0}, s1x{0, 0}, s1y{0, 0};
  const uint64_t prime = primes[limb], ratio = ratios[limb], k = ks[limb];
  for (int t = 0; t < a.n; ++t) {
    const ulonglong2 c0 = *reinterpret_cast<const ulonglong2*>((const uint64_t*)a.c0[t] + item * a.cbs[t] + off);
    const ulonglong2 c1 = *reinterpret_cast<const ulonglong2*>((const uint64_t*)a.c1[t] + item * a.cbs[t] + off);
    const ulonglong2 p = *reinterpret_cast<const ulonglong2*>((const uint64_t*)a.pt[t] + item * a.pbs[t] + off);
    mac128(c0.x, p.x, s0x); mac128(c0.y, p.y, s0y);
    mac128(c1.x, p.x, s1x); mac128(c1.y, p.y, s1y);
    // Barrett with k = bits(p) + 62 needs the input below 2^k: fold after every four
    // products (4 * 2^(2 bits) + residue < 2^(2 bits + 2) <= 2^k for bits <= 60).
    if ((t & 3) == 3 && t + 1 < a.n) {
      s0x = {0, barrett128(s0x, prime, ratio, k)}; s0y = {0, barrett128(s0y, prime, ratio, k)};
      s1x = {0, barrett128(s1x, prime, ratio, k)}; s1y = {0, barrett128(s1y, prime, ratio, k)};
    }
  }
  const long long o = item * out_stride + off;
  *reinterpret_cast<ulonglong2*>(out0 + o) = {barrett128(s0x, prime, ratio, k), barrett128(s0y, prime, ratio, k)};
  *reinterpret_cast<ulonglong2*>(out1 + o) = {barrett128(s1x, prime, ratio, k), barrett128(s1y, prime, ratio, k)};
}

}  // namespace

extern "C" int sylph_pt_mac(uint64_t* out0, uint64_t* out1, long long out_stride, SylphMacArgs args, int batch,
                            int limbs, const uint64_t* primes, const uint64_t* ratios, const uint64_t* ks,
                            void* stream) {
  if (args.n < 1 || args.n > kMaxTerms) return -1;
  pt_mac<<<dim3(kN / 512, limbs, batch), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      out0, out1, out_stride, args, limbs, primes, ratios, ks);
  return static_cast<int>(cudaGetLastError());
}
