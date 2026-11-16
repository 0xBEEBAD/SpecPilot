"""SPECPILOT_D2A_KPREFILL: the target's prefill (EXTEND) attention at batch 1 as a hand-written sm_120 CUDA kernel.

Replaces the triton backend's ``extend_attention_fwd`` for the target's full-attention layers (head dim 256, 6 q heads
per kv head, bf16 KV, causal, no custom mask / window / sinks / logit cap, page size 1). The stock kernel streams the
whole prefix per (q block, head) and runs at 32-46 TFLOP/s for 20-80k-token prefixes on sm_120 (agent kattn microbench).

``kattn_pf_kernel`` (mma.sync m16n8k16 bf16, fp32 accumulation), grid (q block of 128 packed rows, 1, kv head), 8 warps
x 16 rows, row = new token * 6 + head: prefix keys [0, P) gathered through kv_indices (P = kv_indptr[1] - kv_indptr[0]
on device), then the chunk's own keys causal (only up to the block's last token); cp.async 2-stage ring of 32-key
tiles, online softmax in the exp2 domain, normalised bf16 output. Same math as the stock kernel; summation order differs.

Knob (read once at import; unset = strict no-op): SPECPILOT_D2A_KPREFILL=1.
"""
from __future__ import annotations

import hashlib
import os

import torch

ENABLED = os.environ.get("SPECPILOT_D2A_KPREFILL", "").strip() not in ("", "0")
_WM, _BN, _STAGES = 8, 32, 2

_CPP = r"""
#include <torch/extension.h>
void kattn_pf(torch::Tensor q, torch::Tensor ke, torch::Tensor ve, torch::Tensor kb, torch::Tensor vb,
              torch::Tensor indptr, torch::Tensor indices, torch::Tensor out, double scale, int64_t wm, int64_t bn,
              int64_t stages);
"""

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define D 256
#define G 6
#define CHUNKS (D / 8)   // 16-byte chunks per row

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ int swz(int row, int ch) { return row * CHUNKS + (ch ^ (row & 7)); }  // in 16B units

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int n> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float fast_exp2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }


// Prefill (extend) attention, batch 1: rows = new token * G + head (packed per kv head); CTA = BM packed rows x one kv
// head; prefix keys [0, P) gathered through indices (all visible), then the new tokens' own keys [0, kend) causal.
// Output normalised bf16 (no splits).
template <int WM, int BN, int ST, typename TI, typename TP>
__global__ void __launch_bounds__(32 * WM, (WM == 4 ? 2 : 1)) kattn_pf_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ ke, const __nv_bfloat16* __restrict__ ve,
    const __nv_bfloat16* __restrict__ kb, const __nv_bfloat16* __restrict__ vb, const TP* __restrict__ indptr,
    const TI* __restrict__ indices_base, __nv_bfloat16* __restrict__ out, float qk_scale, int N, int64_t sq_t, int64_t sq_h, int64_t ske_t,
    int64_t ske_h, int64_t sve_t, int64_t sve_h, int64_t skb_t, int64_t skb_h, int64_t svb_t, int64_t svb_h,
    int64_t so_t, int64_t so_h) {
  constexpr int BM = 16 * WM, NTHR = 32 * WM;
  extern __shared__ __align__(128) uint8_t smem[];
  const int nqb = gridDim.x;
  const int i_qb = nqb - 1 - blockIdx.x;  // longest causal tails first
  const int i_kv = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int NG = N * G;
  const TI* indices = indices_base + (int64_t)indptr[0];
  const int P = (int)((int64_t)indptr[1] - (int64_t)indptr[0]);
  uint4* sq = reinterpret_cast<uint4*>(smem);
  for (int c = tid; c < BM * CHUNKS; c += NTHR) {
    int r = c / CHUNKS, ch = c % CHUNKS;
    int row = i_qb * BM + r;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row < NG) {
      int tok = row / G, hq = i_kv * G + row % G;
      v = *reinterpret_cast<const uint4*>(q + (int64_t)tok * sq_t + (int64_t)hq * sq_h + ch * 8);
    }
    sq[swz(r, ch)] = v;
  }
  __syncthreads();
  uint32_t qa[D / 16][4];
  {
    int r = warp * 16 + (lane & 15);
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      int ch = kk * 2 + (lane >> 4);
      ldsm_x4(qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], smem_u32(sq + swz(r, ch)));
    }
  }
  __syncthreads();
  float acc[D / 8][4];
#pragma unroll
  for (int i = 0; i < D / 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int rlo = i_qb * BM + warp * 16 + (lane >> 2);
  const int tok0 = rlo / G, tok1 = (rlo + 8) / G;
  const int kend = min(N, (min(i_qb * BM + BM, NG) - 1) / G + 1);
  uint4* sk = reinterpret_cast<uint4*>(smem);
  uint4* sv = sk + ST * BN * CHUNKS;
  constexpr int PER = BN * CHUNKS / NTHR;
  constexpr int RSTEP = NTHR / CHUNKS;
  const int ch_t = tid % CHUNKS, r_t = tid / CHUNKS;
  // virtual key index t in [0, P + kend): t < P = prefix slot indices[t], else new key t - P (causal)
  const int total = P + kend;
  const int nblk_p = (P + BN - 1) / BN;           // prefix tiles (last one may be partial)
  const int nblk = nblk_p + (kend + BN - 1) / BN;  // + new-key tiles, aligned at P's tile boundary
  auto key_of = [&](int blk, int r) -> int {  // virtual key of row r of tile blk, or -1
    if (blk < nblk_p) { int j = blk * BN + r; return j < P ? j : -1; }
    int j = (blk - nblk_p) * BN + r; return j < kend ? P + j : -1;
  };
  auto load_slots = [&](int blk, int* sl) {
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      int t = blk < nblk ? key_of(blk, r_t + i * RSTEP) : -1;
      sl[i] = t < 0 ? -1 : (t < P ? (int)indices[t] : (t - P) | 0x40000000);
    }
  };
  auto issue = [&](int blk, const int* sl) {
    int stg = blk % ST;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      int r = r_t + i * RSTEP;
      bool ok = sl[i] >= 0;
      bool nw = ok && (sl[i] & 0x40000000);
      int64_t slot = ok ? (int64_t)(sl[i] & 0x3fffffff) : 0;
      const __nv_bfloat16* ksrc = nw ? ke + slot * ske_t + (int64_t)i_kv * ske_h : kb + slot * skb_t + (int64_t)i_kv * skb_h;
      const __nv_bfloat16* vsrc = nw ? ve + slot * sve_t + (int64_t)i_kv * sve_h : vb + slot * svb_t + (int64_t)i_kv * svb_h;
      cp_async16(smem_u32(sk + stg * BN * CHUNKS + swz(r, ch_t)), ksrc + ch_t * 8, ok);
      cp_async16(smem_u32(sv + stg * BN * CHUNKS + swz(r, ch_t)), vsrc + ch_t * 8, ok);
    }
  };
  (void)total;
  int nx[PER];
#pragma unroll
  for (int p = 0; p < ST - 1; ++p) {
    load_slots(p, nx);
    if (p < nblk) issue(p, nx);
    cp_commit();
  }
  load_slots(ST - 1, nx);
  for (int blk = 0; blk < nblk; ++blk) {
    if (blk + ST - 1 < nblk) issue(blk + ST - 1, nx);
    cp_commit();
    load_slots(blk + ST, nx);
    cp_wait<ST - 1>();
    __syncthreads();
    const int stg = blk % ST;
    const uint4* skt = sk + stg * BN * CHUNKS;
    const uint4* svt = sv + stg * BN * CHUNKS;
    float s[BN / 8][4];
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < D / 16; kk += 2) {
#pragma unroll
      for (int n = 0; n < BN / 8; ++n) {
        int r = n * 8 + (lane & 7);
        int ch = kk * 2 + (lane >> 3);
        uint32_t b0, b1, b2, b3;
        ldsm_x4(b0, b1, b2, b3, smem_u32(skt + swz(r, ch)));
        mma16816(s[n], qa[kk], b0, b1);
        mma16816(s[n], qa[kk + 1], b2, b3);
      }
    }
    const bool newtile = blk >= nblk_p;
    const int jb = newtile ? (blk - nblk_p) * BN : blk * BN;
    const int jlim = newtile ? kend : P;
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int j = jb + n * 8 + (lane & 3) * 2 + (e & 1);
        bool vis = j < jlim;
        if (newtile) vis = vis && j <= (e < 2 ? tok0 : tok1);
        s[n][e] = vis ? s[n][e] * qk_scale : -INFINITY;
      }
    }
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
      mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
      mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const float ms0 = mx0 == -INFINITY ? 0.f : mx0, ms1 = mx1 == -INFINITY ? 0.f : mx1;
    const float al0 = fast_exp2(m0 - ms0), al1 = fast_exp2(m1 - ms1);
    float rs0 = 0.f, rs1 = 0.f;
    uint32_t pa[BN / 16][4];
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
      float p0 = fast_exp2(s[n][0] - ms0), p1 = fast_exp2(s[n][1] - ms0);
      float p2 = fast_exp2(s[n][2] - ms1), p3 = fast_exp2(s[n][3] - ms1);
      rs0 += p0 + p1;
      rs1 += p2 + p3;
      pa[n / 2][(n & 1) * 2 + 0] = pack_bf16(p0, p1);
      pa[n / 2][(n & 1) * 2 + 1] = pack_bf16(p2, p3);
    }
    rs0 += __shfl_xor_sync(0xffffffff, rs0, 1);
    rs0 += __shfl_xor_sync(0xffffffff, rs0, 2);
    rs1 += __shfl_xor_sync(0xffffffff, rs1, 1);
    rs1 += __shfl_xor_sync(0xffffffff, rs1, 2);
    l0 = l0 * al0 + rs0;
    l1 = l1 * al1 + rs1;
    m0 = mx0;
    m1 = mx1;
#pragma unroll
    for (int i = 0; i < D / 8; ++i) {
      acc[i][0] *= al0; acc[i][1] *= al0; acc[i][2] *= al1; acc[i][3] *= al1;
    }
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) {
#pragma unroll
      for (int n = 0; n < D / 8; n += 2) {
        int r = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        int ch = n + (lane >> 4);
        uint32_t b0, b1, b2, b3;
        ldsm_x4_t(b0, b1, b2, b3, smem_u32(svt + swz(r, ch)));
        mma16816(acc[n], pa[kk], b0, b1);
        mma16816(acc[n + 1], pa[kk], b2, b3);
      }
    }
    __syncthreads();
  }
  cp_wait<0>();
  const float il0 = l0 > 0.f ? 1.f / l0 : 0.f, il1 = l1 > 0.f ? 1.f / l1 : 0.f;
  if (rlo < NG) {
    __nv_bfloat16* o0 = out + (int64_t)tok0 * so_t + (int64_t)(i_kv * G + rlo % G) * so_h;
#pragma unroll
    for (int n = 0; n < D / 8; ++n)
      *reinterpret_cast<__nv_bfloat162*>(o0 + n * 8 + (lane & 3) * 2) = __floats2bfloat162_rn(acc[n][0] * il0, acc[n][1] * il0);
  }
  if (rlo + 8 < NG) {
    __nv_bfloat16* o1 = out + (int64_t)tok1 * so_t + (int64_t)(i_kv * G + (rlo + 8) % G) * so_h;
#pragma unroll
    for (int n = 0; n < D / 8; ++n)
      *reinterpret_cast<__nv_bfloat162*>(o1 + n * 8 + (lane & 3) * 2) = __floats2bfloat162_rn(acc[n][2] * il1, acc[n][3] * il1);
  }
}

template <int WM, int BN, int ST, typename TI, typename TP>
static void launch_pf(torch::Tensor& q, torch::Tensor& ke, torch::Tensor& ve, torch::Tensor& kb, torch::Tensor& vb,
                      torch::Tensor& indptr, torch::Tensor& indices, torch::Tensor& out, double scale, int N) {
  constexpr int BM = 16 * WM;
  int HKV = kb.size(1);
  dim3 grid((N * G + BM - 1) / BM, 1, HKV);
  size_t smem = (size_t)ST * BN * D * 2 * 2;
  size_t qsm = (size_t)BM * D * 2;
  if (qsm > smem) smem = qsm;
  auto kern = kattn_pf_kernel<WM, BN, ST, TI, TP>;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  kern<<<grid, 32 * WM, smem, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)ke.data_ptr(), (const __nv_bfloat16*)ve.data_ptr(),
      (const __nv_bfloat16*)kb.data_ptr(), (const __nv_bfloat16*)vb.data_ptr(), (const TP*)indptr.data_ptr(),
      (const TI*)indices.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (float)(scale * 1.4426950408889634), N, q.stride(0), q.stride(1),
      ke.stride(0), ke.stride(1), ve.stride(0), ve.stride(1), kb.stride(0), kb.stride(1), vb.stride(0), vb.stride(1),
      out.stride(0), out.stride(1));
}

void kattn_pf(torch::Tensor q, torch::Tensor ke, torch::Tensor ve, torch::Tensor kb, torch::Tensor vb,
              torch::Tensor indptr, torch::Tensor indices, torch::Tensor out, double scale, int64_t wm, int64_t bn,
              int64_t stages) {
  TORCH_CHECK(q.size(2) == D && kb.size(2) == D && q.size(1) == kb.size(1) * G, "kattn_pf: D=256, G=6 only");
  int N = (int)q.size(0);
  if (N == 0) return;
  bool i64 = indices.scalar_type() == at::kLong, p64 = indptr.scalar_type() == at::kLong;
#define PF_CASE(WM_, BN_, ST_) if (wm == WM_ && bn == BN_ && stages == ST_) { \
    if (i64 && p64) launch_pf<WM_, BN_, ST_, int64_t, int64_t>(q, ke, ve, kb, vb, indptr, indices, out, scale, N); \
    else if (i64) launch_pf<WM_, BN_, ST_, int64_t, int32_t>(q, ke, ve, kb, vb, indptr, indices, out, scale, N); \
    else if (p64) launch_pf<WM_, BN_, ST_, int32_t, int64_t>(q, ke, ve, kb, vb, indptr, indices, out, scale, N); \
    else launch_pf<WM_, BN_, ST_, int32_t, int32_t>(q, ke, ve, kb, vb, indptr, indices, out, scale, N); return; }
  PF_CASE(8, 16, 2) PF_CASE(8, 32, 2)
  TORCH_CHECK(false, "kattn_pf: unsupported wm/bn/stages");
}
"""

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define D 256
#define G 6
#define CHUNKS (D / 8)   // 16-byte chunks per row

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ int swz(int row, int ch) { return row * CHUNKS + (ch ^ (row & 7)); }  // in 16B units

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int n> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float fast_exp2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }


// Prefill (extend) attention, batch 1: rows = new token * G + head (packed per kv head); CTA = BM packed rows x one kv
// head; prefix keys [0, P) gathered through indices (all visible), then the new tokens' own keys [0, kend) causal.
// Output normalised bf16 (no splits).
template <int WM, int BN, int ST, typename TI, typename TP>
__global__ void __launch_bounds__(32 * WM, (WM == 4 ? 2 : 1)) kattn_pf_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ ke, const __nv_bfloat16* __restrict__ ve,
    const __nv_bfloat16* __restrict__ kb, const __nv_bfloat16* __restrict__ vb, const TP* __restrict__ indptr,
    const TI* __restrict__ indices_base, __nv_bfloat16* __restrict__ out, float qk_scale, int N, int64_t sq_t, int64_t sq_h, int64_t ske_t,
    int64_t ske_h, int64_t sve_t, int64_t sve_h, int64_t skb_t, int64_t skb_h, int64_t svb_t, int64_t svb_h,
    int64_t so_t, int64_t so_h) {
  constexpr int BM = 16 * WM, NTHR = 32 * WM;
  extern __shared__ __align__(128) uint8_t smem[];
  const int nqb = gridDim.x;
  const int i_qb = nqb - 1 - blockIdx.x;  // longest causal tails first
  const int i_kv = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int NG = N * G;
  const TI* indices = indices_base + (int64_t)indptr[0];
  const int P = (int)((int64_t)indptr[1] - (int64_t)indptr[0]);
  uint4* sq = reinterpret_cast<uint4*>(smem);
  for (int c = tid; c < BM * CHUNKS; c += NTHR) {
    int r = c / CHUNKS, ch = c % CHUNKS;
    int row = i_qb * BM + r;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (row < NG) {
      int tok = row / G, hq = i_kv * G + row % G;
      v = *reinterpret_cast<const uint4*>(q + (int64_t)tok * sq_t + (int64_t)hq * sq_h + ch * 8);
    }
    sq[swz(r, ch)] = v;
  }
  __syncthreads();
  uint32_t qa[D / 16][4];
  {
    int r = warp * 16 + (lane & 15);
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      int ch = kk * 2 + (lane >> 4);
      ldsm_x4(qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], smem_u32(sq + swz(r, ch)));
    }
  }
  __syncthreads();
  float acc[D / 8][4];
#pragma unroll
  for (int i = 0; i < D / 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int rlo = i_qb * BM + warp * 16 + (lane >> 2);
  const int tok0 = rlo / G, tok1 = (rlo + 8) / G;
  const int kend = min(N, (min(i_qb * BM + BM, NG) - 1) / G + 1);
  uint4* sk = reinterpret_cast<uint4*>(smem);
  uint4* sv = sk + ST * BN * CHUNKS;
  constexpr int PER = BN * CHUNKS / NTHR;
  constexpr int RSTEP = NTHR / CHUNKS;
  const int ch_t = tid % CHUNKS, r_t = tid / CHUNKS;
  // virtual key index t in [0, P + kend): t < P = prefix slot indices[t], else new key t - P (causal)
  const int total = P + kend;
  const int nblk_p = (P + BN - 1) / BN;           // prefix tiles (last one may be partial)
  const int nblk = nblk_p + (kend + BN - 1) / BN;  // + new-key tiles, aligned at P's tile boundary
  auto key_of = [&](int blk, int r) -> int {  // virtual key of row r of tile blk, or -1
    if (blk < nblk_p) { int j = blk * BN + r; return j < P ? j : -1; }
    int j = (blk - nblk_p) * BN + r; return j < kend ? P + j : -1;
  };
  auto load_slots = [&](int blk, int* sl) {
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      int t = blk < nblk ? key_of(blk, r_t + i * RSTEP) : -1;
      sl[i] = t < 0 ? -1 : (t < P ? (int)indices[t] : (t - P) | 0x40000000);
    }
  };
  auto issue = [&](int blk, const int* sl) {
    int stg = blk % ST;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      int r = r_t + i * RSTEP;
      bool ok = sl[i] >= 0;
      bool nw = ok && (sl[i] & 0x40000000);
      int64_t slot = ok ? (int64_t)(sl[i] & 0x3fffffff) : 0;
      const __nv_bfloat16* ksrc = nw ? ke + slot * ske_t + (int64_t)i_kv * ske_h : kb + slot * skb_t + (int64_t)i_kv * skb_h;
      const __nv_bfloat16* vsrc = nw ? ve + slot * sve_t + (int64_t)i_kv * sve_h : vb + slot * svb_t + (int64_t)i_kv * svb_h;
      cp_async16(smem_u32(sk + stg * BN * CHUNKS + swz(r, ch_t)), ksrc + ch_t * 8, ok);
      cp_async16(smem_u32(sv + stg * BN * CHUNKS + swz(r, ch_t)), vsrc + ch_t * 8, ok);
    }
  };
  (void)total;
  int nx[PER];
#pragma unroll
  for (int p = 0; p < ST - 1; ++p) {
    load_slots(p, nx);
    if (p < nblk) issue(p, nx);
    cp_commit();
  }
  load_slots(ST - 1, nx);
  for (int blk = 0; blk < nblk; ++blk) {
    if (blk + ST - 1 < nblk) issue(blk + ST - 1, nx);
    cp_commit();
    load_slots(blk + ST, nx);
    cp_wait<ST - 1>();
    __syncthreads();
    const int stg = blk % ST;
    const uint4* skt = sk + stg * BN * CHUNKS;
    const uint4* svt = sv + stg * BN * CHUNKS;
    float s[BN / 8][4];
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < D / 16; kk += 2) {
#pragma unroll
      for (int n = 0; n < BN / 8; ++n) {
        int r = n * 8 + (lane & 7);
        int ch = kk * 2 + (lane >> 3);
        uint32_t b0, b1, b2, b3;
        ldsm_x4(b0, b1, b2, b3, smem_u32(skt + swz(r, ch)));
        mma16816(s[n], qa[kk], b0, b1);
        mma16816(s[n], qa[kk + 1], b2, b3);
      }
    }
    const bool newtile = blk >= nblk_p;
    const int jb = newtile ? (blk - nblk_p) * BN : blk * BN;
    const int jlim = newtile ? kend : P;
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int j = jb + n * 8 + (lane & 3) * 2 + (e & 1);
        bool vis = j < jlim;
        if (newtile) vis = vis && j <= (e < 2 ? tok0 : tok1);
        s[n][e] = vis ? s[n][e] * qk_scale : -INFINITY;
      }
    }
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
      mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
      mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const float ms0 = mx0 == -INFINITY ? 0.f : mx0, ms1 = mx1 == -INFINITY ? 0.f : mx1;
    const float al0 = fast_exp2(m0 - ms0), al1 = fast_exp2(m1 - ms1);
    float rs0 = 0.f, rs1 = 0.f;
    uint32_t pa[BN / 16][4];
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
      float p0 = fast_exp2(s[n][0] - ms0), p1 = fast_exp2(s[n][1] - ms0);
      float p2 = fast_exp2(s[n][2] - ms1), p3 = fast_exp2(s[n][3] - ms1);
      rs0 += p0 + p1;
      rs1 += p2 + p3;
      pa[n / 2][(n & 1) * 2 + 0] = pack_bf16(p0, p1);
      pa[n / 2][(n & 1) * 2 + 1] = pack_bf16(p2, p3);
    }
    rs0 += __shfl_xor_sync(0xffffffff, rs0, 1);
    rs0 += __shfl_xor_sync(0xffffffff, rs0, 2);
    rs1 += __shfl_xor_sync(0xffffffff, rs1, 1);
    rs1 += __shfl_xor_sync(0xffffffff, rs1, 2);
    l0 = l0 * al0 + rs0;
    l1 = l1 * al1 + rs1;
    m0 = mx0;
    m1 = mx1;
#pragma unroll
    for (int i = 0; i < D / 8; ++i) {
      acc[i][0] *= al0; acc[i][1] *= al0; acc[i][2] *= al1; acc[i][3] *= al1;
    }
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) {
#pragma unroll
      for (int n = 0; n < D / 8; n += 2) {
        int r = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        int ch = n + (lane >> 4);
        uint32_t b0, b1, b2, b3;
        ldsm_x4_t(b0, b1, b2, b3, smem_u32(svt + swz(r, ch)));
        mma16816(acc[n], pa[kk], b0, b1);
        mma16816(acc[n + 1], pa[kk], b2, b3);
      }
    }
    __syncthreads();
  }
  cp_wait<0>();
  const float il0 = l0 > 0.f ? 1.f / l0 : 0.f, il1 = l1 > 0.f ? 1.f / l1 : 0.f;
  if (rlo < NG) {
    __nv_bfloat16* o0 = out + (int64_t)tok0 * so_t + (int64_t)(i_kv * G + rlo % G) * so_h;
#pragma unroll
    for (int n = 0; n < D / 8; ++n)
      *reinterpret_cast<__nv_bfloat162*>(o0 + n * 8 + (lane & 3) * 2) = __floats2bfloat162_rn(acc[n][0] * il0, acc[n][1] * il0);
  }
  if (rlo + 8 < NG) {
    __nv_bfloat16* o1 = out + (int64_t)tok1 * so_t + (int64_t)(i_kv * G + (rlo + 8) % G) * so_h;
#pragma unroll
    for (int n = 0; n < D / 8; ++n)
      *reinterpret_cast<__nv_bfloat162*>(o1 + n * 8 + (lane & 3) * 2) = __floats2bfloat162_rn(acc[n][2] * il1, acc[n][3] * il1);
  }
}

template <int WM, int BN, int ST, typename TI, typename TP>
static void launch_pf(torch::Tensor& q, torch::Tensor& ke, torch::Tensor& ve, torch::Tensor& kb, torch::Tensor& vb,
                      torch::Tensor& indptr, torch::Tensor& indices, torch::Tensor& out, double scale, int N) {
  constexpr int BM = 16 * WM;
  int HKV = kb.size(1);
  dim3 grid((N * G + BM - 1) / BM, 1, HKV);
  size_t smem = (size_t)ST * BN * D * 2 * 2;
  size_t qsm = (size_t)BM * D * 2;
  if (qsm > smem) smem = qsm;
  auto kern = kattn_pf_kernel<WM, BN, ST, TI, TP>;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  kern<<<grid, 32 * WM, smem, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)ke.data_ptr(), (const __nv_bfloat16*)ve.data_ptr(),
      (const __nv_bfloat16*)kb.data_ptr(), (const __nv_bfloat16*)vb.data_ptr(), (const TP*)indptr.data_ptr(),
      (const TI*)indices.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (float)(scale * 1.4426950408889634), N, q.stride(0), q.stride(1),
      ke.stride(0), ke.stride(1), ve.stride(0), ve.stride(1), kb.stride(0), kb.stride(1), vb.stride(0), vb.stride(1),
      out.stride(0), out.stride(1));
}

void kattn_pf(torch::Tensor q, torch::Tensor ke, torch::Tensor ve, torch::Tensor kb, torch::Tensor vb,
              torch::Tensor indptr, torch::Tensor indices, torch::Tensor out, double scale, int64_t wm, int64_t bn,
              int64_t stages) {
  TORCH_CHECK(q.size(2) == D && kb.size(2) == D && q.size(1) == kb.size(1) * G, "kattn_pf: D=256, G=6 only");
  int N = (int)q.size(0);
  if (N == 0) return;
  bool i64 = indices.scalar_type() == at::kLong, p64 = indptr.scalar_type() == at::kLong;
#define PF_CASE(WM_, BN_, ST_) if (wm == WM_ && bn == BN_ && stages == ST_) { \
    if (i64 && p64) launch_pf<WM_, BN_, ST_, int64_t, int64_t>(q, ke, ve, kb, vb, indptr, indices, out, scale, N); \
    else if (i64) launch_pf<WM_, BN_, ST_, int64_t, int32_t>(q, ke, ve, kb, vb, indptr, indices, out, scale, N); \
    else if (p64) launch_pf<WM_, BN_, ST_, int32_t, int64_t>(q, ke, ve, kb, vb, indptr, indices, out, scale, N); \
    else launch_pf<WM_, BN_, ST_, int32_t, int32_t>(q, ke, ve, kb, vb, indptr, indices, out, scale, N); return; }
  PF_CASE(8, 32, 2)
  TORCH_CHECK(false, "kattn_pf: unsupported wm/bn/stages");
}
"""

_mod = None


def _ext():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
        cflags = ["-O3", "-gencode=arch=compute_120a,code=sm_120a", "--use_fast_math"]
        key = hashlib.sha1("\0".join([_CPP, _SRC, *cflags]).encode()).hexdigest()[:16]
        base = os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
        bdir = os.path.join(base, "specpilot_d2a_kprefill", key)
        os.makedirs(bdir, exist_ok=True)
        _mod = load_inline(name=f"specpilot_d2a_kprefill_{key}", cpp_sources=_CPP, cuda_sources=_SRC,
                           functions=["kattn_pf"], extra_cuda_cflags=cflags, build_directory=bdir, verbose=False)
    return _mod


def eligible(q, k_buffer, kv_indptr) -> bool:
    """q [T, HQ, D]; k_buffer [slots, HKV, D]; batch 1 (kv_indptr [2])."""
    return (
        ENABLED
        and kv_indptr.shape[0] == 2
        and q.shape[-1] == 256
        and k_buffer.shape[-1] == 256
        and q.shape[1] == 6 * k_buffer.shape[1]
        and q.dtype == torch.bfloat16
        and k_buffer.dtype == torch.bfloat16
    )


def prefill(q, k_extend, v_extend, o, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    """q / o [N, HQ, D]; k/v_extend [N, HKV, D]; k/v_buffer [slots, HKV, D]; prefix = kv_indices[kv_indptr[0]:kv_indptr[1]]."""
    _ext().kattn_pf(q, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices, o, float(sm_scale), _WM, _BN,
                    _STAGES)
    return o
