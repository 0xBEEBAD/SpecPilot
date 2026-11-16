"""SPECPILOT_D2A_KATTN: tree-verify attention of the target's full-attention layers as a hand-written sm_120 CUDA kernel.

Replaces the d5 split-KV Triton kernel (``specpilot_d5_splitkv``) for the target's verify forwards (exact pass and the
two-level verify's Q passes) at batch 1, head dim 256, 6 q heads per kv head, bf16 KV, no window. Same math: fp32
online softmax in the exp2 domain, bf16 tensor-core dots with fp32 accumulation, fp32 partials merged with the same
weights; only the summation order differs (split boundaries, tile size, merge order).

  * ``kattn_tv_kernel`` (CUDA, mma.sync m16n8k16), grid (q block of 64 packed rows, split, kv head), 4 warps x 16 rows,
    row = token * 6 + head: each K/V tile is read by 3 CTAs per kv head (d5 BM32: 6). Q fragments live in registers;
    K/V tiles of 16 keys are gathered through ``kv_indices`` by cp.async into an XOR-swizzled 2-3 stage ring (slots of
    the next tile prefetched into registers); S = Q K^T and O += P V on tensor cores, P taken from the S accumulators.
    Splits s < S_eff = clamp(cdiv(L, MINCH), 1, S) (computed on device from ``kv_indptr``: graph safe) run the prefix
    in equal 16-aligned chunks; split S runs the N tree keys (k/v_extend) masked by the custom mask's suffix block.
    Every program writes its unnormalised partial (acc, m, l).
  * ``_d2a_combine`` (Triton), grid (8-row block, kv head, 32-column block): all live partials loaded in one shot,
    merged, stored bf16.

Knobs (read once at import; unset = strict no-op, the d5 kernel runs as in the parent):
  SPECPILOT_D2A_KATTN=1            enable for the target's verify forwards
  SPECPILOT_D2A_KATTN_CFG=S,MINCH,STAGES   default 31,64,2 (31 prefix splits = one wave at 2 CTAs/SM on 188 SMs)
Microbench (agent kattn, GPU 5, 16 layers, N=32, CUDA graph): 100k prefix 9.63 -> 5.4 ms/pass, 50k 5.2 -> 3.0,
16k 1.79 -> 1.25, 4k 0.74 -> 0.51, 1k 0.45 -> 0.33 (vs d5 S48 BM32 BN32 4 warps).
"""
from __future__ import annotations

import hashlib
import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("SPECPILOT_D2A_KATTN", "").strip() not in ("", "0")


def _cfg():
    raw = os.environ.get("SPECPILOT_D2A_KATTN_CFG", "").strip()
    vals = tuple(int(x) for x in raw.split(",")) if raw else (31, 64, 2)
    if len(vals) != 3:
        raise ValueError(f"SPECPILOT_D2A_KATTN_CFG={raw!r}: expected splits,minch,stages")
    return vals


SPLITS, MINCH, STAGES = _cfg()
BN = 16
RB, DC = 8, 32

_CPP = r"""
#include <torch/extension.h>
void kattn_tv(torch::Tensor q, torch::Tensor ke, torch::Tensor ve, torch::Tensor kb, torch::Tensor vb,
              torch::Tensor indptr, torch::Tensor indices, torch::Tensor mask, torch::Tensor mindptr,
              torch::Tensor pacc, torch::Tensor pml, double scale, int64_t N, int64_t S, int64_t minch,
              int64_t has_mask, int64_t bn, int64_t stages);
"""

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define D 256
#define G 6
#define WM 4
#define BM (16 * WM)
#define NTHR (32 * WM)
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

template <int BN, int ST, typename TI, typename TP>
__global__ void __launch_bounds__(NTHR, 2) kattn_tv_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ ke, const __nv_bfloat16* __restrict__ ve,
    const __nv_bfloat16* __restrict__ kb, const __nv_bfloat16* __restrict__ vb, const TP* __restrict__ indptr,
    const TI* __restrict__ indices, const uint8_t* __restrict__ mask, const int64_t* __restrict__ mindptr,
    float* __restrict__ pacc, float* __restrict__ pml, float qk_scale, int N, int S, int QB, int minch, int has_mask,
    int64_t sq_t, int64_t sq_h, int64_t ske_t, int64_t ske_h, int64_t sve_t, int64_t sve_h,
    int64_t skb_t, int64_t skb_h, int64_t svb_t, int64_t svb_h) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int i_qb = blockIdx.x, i_s = blockIdx.y, i_kv = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int64_t kv_start = (int64_t)indptr[0];
  const int L = (int)((int64_t)indptr[1] - kv_start);
  int s_eff = (L + minch - 1) / minch;
  s_eff = s_eff < 1 ? 1 : (s_eff > S ? S : s_eff);
  if (i_s < S && i_s >= s_eff) return;
  const int NG = N * G;

  // ---- Q tile -> smem (swizzled) -> A fragments in registers
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
    // A fragment for k-step kk: rows warp*16 + (lane%16), chunk kk*2 + lane/16
    int r = warp * 16 + (lane & 15);
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      int ch = kk * 2 + (lane >> 4);
      ldsm_x4(qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], smem_u32(sq + swz(r, ch)));
    }
  }
  __syncthreads();  // Q smem region is reused by the K/V ring

  float acc[D / 8][4];
#pragma unroll
  for (int i = 0; i < D / 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;  // rows lane/4 and lane/4 + 8 of this warp

  const bool suffix = (i_s == S);
  int start, end;
  if (!suffix) {
    int chunk = ((L + s_eff - 1) / s_eff + BN - 1) / BN * BN;
    start = i_s * chunk;
    end = min(L, start + chunk);
  } else {
    start = 0;
    end = N;
  }
  const int nblk = end > start ? (end - start + BN - 1) / BN : 0;
  uint4* sk = reinterpret_cast<uint4*>(smem);
  uint4* sv = sk + ST * BN * CHUNKS;
  const __nv_bfloat16* kbase = suffix ? ke : kb;
  const __nv_bfloat16* vbase = suffix ? ve : vb;
  const int64_t sk_t = suffix ? ske_t : skb_t, sk_h = suffix ? ske_h : skb_h;
  const int64_t sv_t = suffix ? sve_t : svb_t, sv_h = suffix ? sve_h : svb_h;

  constexpr int PER = BN * CHUNKS / NTHR;  // 16-byte chunks per thread per tile (rows tid/32 + i*NTHR/32)
  const int ch_t = tid % CHUNKS, r_t = tid / CHUNKS;
  auto load_slots = [&](int blk, int* sl) {
    int j0 = start + blk * BN;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      int j = j0 + r_t + i * (NTHR / CHUNKS);
      sl[i] = (blk < nblk && j < end) ? (suffix ? j : (int)indices[kv_start + j]) : -1;
    }
  };
  auto issue = [&](int blk, const int* sl) {
    int stg = blk % ST;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      int r = r_t + i * (NTHR / CHUNKS);
      bool ok = sl[i] >= 0;
      int64_t slot = ok ? (int64_t)sl[i] : 0;
      const __nv_bfloat16* ksrc = kbase + slot * sk_t + (int64_t)i_kv * sk_h + ch_t * 8;
      const __nv_bfloat16* vsrc = vbase + slot * sv_t + (int64_t)i_kv * sv_h + ch_t * 8;
      cp_async16(smem_u32(sk + stg * BN * CHUNKS + swz(r, ch_t)), ksrc, ok);
      cp_async16(smem_u32(sv + stg * BN * CHUNKS + swz(r, ch_t)), vsrc, ok);
    }
  };

  const int rlo = i_qb * BM + warp * 16 + (lane >> 2);  // packed row of c0/c1; c2/c3 = rlo + 8
  const int tok0 = rlo / G, tok1 = (rlo + 8) / G;
  int64_t mrow0 = 0, mrow1 = 0;
  if (suffix && has_mask) {
    int64_t ms = mindptr[0];
    mrow0 = ms + (int64_t)min(tok0, N - 1) * (L + N) + L;
    mrow1 = ms + (int64_t)min(tok1, N - 1) * (L + N) + L;
  }

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
    load_slots(blk + ST, nx);  // next tile's slots: the loads complete under this tile's math
    cp_wait<ST - 1>();
    __syncthreads();
    const int stg = blk % ST;
    const uint4* skt = sk + stg * BN * CHUNKS;
    const uint4* svt = sv + stg * BN * CHUNKS;
    // S = Q K^T : [16 x BN] per warp
    float s[BN / 8][4];
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < D / 16; kk += 2) {
#pragma unroll
      for (int n = 0; n < BN / 8; ++n) {
        // x4: matrices = (key rows n*8..+7) x chunks (2kk, 2kk+1, 2kk+2, 2kk+3)
        int r = n * 8 + (lane & 7);
        int ch = kk * 2 + (lane >> 3);
        uint32_t b0, b1, b2, b3;
        ldsm_x4(b0, b1, b2, b3, smem_u32(skt + swz(r, ch)));
        mma16816(s[n], qa[kk], b0, b1);
        mma16816(s[n], qa[kk + 1], b2, b3);
      }
    }
    // mask + online softmax
    const int j0 = start + blk * BN;
#pragma unroll
    for (int n = 0; n < BN / 8; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int j = j0 + n * 8 + (lane & 3) * 2 + (e & 1);
        bool vis = j < end;
        if (suffix && vis) {
          if (has_mask) vis = mask[(e < 2 ? mrow0 : mrow1) + j] != 0;
          else vis = j <= (e < 2 ? tok0 : tok1);
        }
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
    // O += P V : A = P [16 x BN] (k = key), B = V [BN x D] via ldmatrix.trans
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) {
#pragma unroll
      for (int n = 0; n < D / 8; n += 2) {
        // x4.trans: matrices = keys (kk*16 + 0..7, +8..15) x d-chunks (n, n+1)
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

  // ---- partial store (d5 layout, BM rows): acc [pid][BM][D], pml [pid][2][BM]
  const int64_t pid = ((int64_t)i_kv * QB + i_qb) * (S + 1) + i_s;
  const int rr0 = warp * 16 + (lane >> 2), rr1 = rr0 + 8;
  float* pa0 = pacc + (pid * BM + rr0) * D;
  float* pa1 = pacc + (pid * BM + rr1) * D;
#pragma unroll
  for (int n = 0; n < D / 8; ++n) {
    int col = n * 8 + (lane & 3) * 2;
    *reinterpret_cast<float2*>(pa0 + col) = make_float2(acc[n][0], acc[n][1]);
    *reinterpret_cast<float2*>(pa1 + col) = make_float2(acc[n][2], acc[n][3]);
  }
  if ((lane & 3) == 0) {
    pml[(pid * 2) * BM + rr0] = m0;
    pml[(pid * 2) * BM + rr1] = m1;
    pml[(pid * 2 + 1) * BM + rr0] = l0;
    pml[(pid * 2 + 1) * BM + rr1] = l1;
  }
}

template <int BN, int ST, typename TI, typename TP>
static void launch(torch::Tensor& q, torch::Tensor& ke, torch::Tensor& ve, torch::Tensor& kb, torch::Tensor& vb,
                   torch::Tensor& indptr, torch::Tensor& indices, torch::Tensor& mask, torch::Tensor& mindptr,
                   torch::Tensor& pacc, torch::Tensor& pml, double scale, int N, int S, int minch, int has_mask) {
  int HKV = kb.size(1);
  int QB = (N * G + BM - 1) / BM;
  dim3 grid(QB, S + 1, HKV);
  size_t smem = (size_t)ST * BN * D * 2 * 2;
  size_t qsm = (size_t)BM * D * 2;
  if (qsm > smem) smem = qsm;
  auto kern = kattn_tv_kernel<BN, ST, TI, TP>;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
  auto stream = at::cuda::getCurrentCUDAStream();
  kern<<<grid, NTHR, smem, stream>>>(
      (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)ke.data_ptr(), (const __nv_bfloat16*)ve.data_ptr(),
      (const __nv_bfloat16*)kb.data_ptr(), (const __nv_bfloat16*)vb.data_ptr(), (const TP*)indptr.data_ptr(),
      (const TI*)indices.data_ptr(), (const uint8_t*)mask.data_ptr(), (const int64_t*)mindptr.data_ptr(),
      pacc.data_ptr<float>(), pml.data_ptr<float>(), (float)(scale * 1.4426950408889634), N, S, QB, minch, has_mask,
      q.stride(0), q.stride(1), ke.stride(0), ke.stride(1), ve.stride(0), ve.stride(1), kb.stride(0), kb.stride(1),
      vb.stride(0), vb.stride(1));
}

template <int BN, int ST>
static void dispatch_idx(torch::Tensor& q, torch::Tensor& ke, torch::Tensor& ve, torch::Tensor& kb, torch::Tensor& vb,
                         torch::Tensor& indptr, torch::Tensor& indices, torch::Tensor& mask, torch::Tensor& mindptr,
                         torch::Tensor& pacc, torch::Tensor& pml, double scale, int N, int S, int minch, int has_mask) {
  bool i64 = indices.scalar_type() == at::kLong, p64 = indptr.scalar_type() == at::kLong;
  if (i64 && p64) launch<BN, ST, int64_t, int64_t>(q, ke, ve, kb, vb, indptr, indices, mask, mindptr, pacc, pml, scale, N, S, minch, has_mask);
  else if (i64) launch<BN, ST, int64_t, int32_t>(q, ke, ve, kb, vb, indptr, indices, mask, mindptr, pacc, pml, scale, N, S, minch, has_mask);
  else if (p64) launch<BN, ST, int32_t, int64_t>(q, ke, ve, kb, vb, indptr, indices, mask, mindptr, pacc, pml, scale, N, S, minch, has_mask);
  else launch<BN, ST, int32_t, int32_t>(q, ke, ve, kb, vb, indptr, indices, mask, mindptr, pacc, pml, scale, N, S, minch, has_mask);
}

void kattn_tv(torch::Tensor q, torch::Tensor ke, torch::Tensor ve, torch::Tensor kb, torch::Tensor vb,
              torch::Tensor indptr, torch::Tensor indices, torch::Tensor mask, torch::Tensor mindptr,
              torch::Tensor pacc, torch::Tensor pml, double scale, int64_t N, int64_t S, int64_t minch,
              int64_t has_mask, int64_t bn, int64_t stages) {
  TORCH_CHECK(q.size(2) == D && kb.size(2) == D && q.size(1) == kb.size(1) * G, "kattn: D=256, G=6 only");
  TORCH_CHECK(N * G <= 4 * 1024, "kattn: N too large");
#define KT_CASE(BN_, ST_) if (bn == BN_ && stages == ST_) { dispatch_idx<BN_, ST_>(q, ke, ve, kb, vb, indptr, indices, mask, mindptr, pacc, pml, scale, (int)N, (int)S, (int)minch, (int)has_mask); return; }
  KT_CASE(16, 2) KT_CASE(16, 3)
  TORCH_CHECK(false, "kattn: unsupported bn/stages");
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
        bdir = os.path.join(base, "specpilot_d2a_kattn", key)
        os.makedirs(bdir, exist_ok=True)
        _mod = load_inline(name=f"specpilot_d2a_kattn_{key}", cpp_sources=_CPP, cuda_sources=_SRC,
                           functions=["kattn_tv"], extra_cuda_cflags=cflags, build_directory=bdir, verbose=False)
    return _mod


def eligible(q, k_buffer, mask_indptr, bs: int, window: int, q_window: int) -> bool:
    """q [T, HQ, D]; k_buffer [slots, HKV, D]."""
    return (
        ENABLED
        and bs == 1
        and (mask_indptr is None or mask_indptr.dtype == torch.int64)
        and window <= 0
        and q_window == 0
        and q.shape[-1] == 256
        and k_buffer.shape[-1] == 256
        and q.shape[1] == 6 * k_buffer.shape[1]
        and q.dtype == torch.bfloat16
        and k_buffer.dtype == torch.bfloat16
    )


@triton.jit
def _d2a_combine(
    out_ptr, pacc_ptr, pml_ptr, kv_indptr_ptr, so_t, so_h,
    N: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, S: tl.constexpr, QB: tl.constexpr,
    RB: tl.constexpr, RBLK: tl.constexpr, DC: tl.constexpr, MINCH: tl.constexpr, SP: tl.constexpr,
):
    """One shot: all S + 1 partials of RB rows x DC columns loaded together ([SP, RB, DC], SP = pow2 >= S + 1)."""
    i_br, i_kv, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    L = tl.load(kv_indptr_ptr + 1) - tl.load(kv_indptr_ptr)
    s_eff = tl.minimum(tl.maximum(tl.cdiv(L, MINCH), 1), S)
    rows = i_br * RB + tl.arange(0, RB)
    rvalid = rows < N * G
    tok = rows // G
    hq = i_kv * G + rows % G
    dd = i_d * DC + tl.arange(0, DC)
    qb = rows // BM
    rr = rows % BM
    base = (i_kv * QB + qb) * (S + 1)
    t = tl.arange(0, SP)
    tval = (t < s_eff) | (t == S)
    pidx = base[None, :] + t[:, None]  # [SP, RB]
    ok = tval[:, None] & rvalid[None, :]
    mt = tl.load(pml_ptr + (pidx * 2) * BM + rr[None, :], mask=ok, other=float("-inf"))
    lt = tl.load(pml_ptr + (pidx * 2 + 1) * BM + rr[None, :], mask=ok, other=0.0)
    mx = tl.max(mt, 0)
    w = tl.where(mt == float("-inf"), 0.0, tl.exp2(mt - mx[None, :]))
    den = tl.sum(w * lt, 0)
    at = tl.load(pacc_ptr + ((pidx * BM + rr[None, :]).to(tl.int64) * D)[:, :, None] + dd[None, None, :],
                 mask=ok[:, :, None], other=0.0)
    num = tl.sum(w[:, :, None] * at, 0)
    o = num / tl.where(rvalid & (den > 0), den, 1.0)[:, None]
    tl.store(out_ptr + tok[:, None].to(tl.int64) * so_t + hq[:, None] * so_h + dd[None, :],
             o.to(out_ptr.dtype.element_ty), mask=rvalid[:, None])


# Partial buffers per (shape, device); a tensor first allocated during graph capture stays referenced here.
_SCRATCH: dict = {}


def _buf(name, shape, dtype, device):
    key = (name, tuple(int(s) for s in shape), dtype, str(device))
    t = _SCRATCH.get(key)
    if t is None:
        t = _SCRATCH[key] = torch.empty(key[1], dtype=dtype, device=device)
    return t


def tree_verify(q, k_extend, v_extend, o, k_buffer, v_buffer, kv_indptr, kv_indices, custom_mask, mask_indptr,
                sm_scale, *, n_tokens):
    """Batch 1. q / o [N, HQ, D]; k/v_extend [N, HKV, D] (unit last stride); k/v_buffer [slots, HKV, D]; kv_indptr
    [2] prefix slots in kv_indices; custom_mask uint8/bool N x (L + N) at mask_indptr[0], or None (causal suffix)."""
    N = int(n_tokens)
    HQ, D = q.shape[1], q.shape[2]
    HKV = k_buffer.shape[1]
    G = HQ // HKV
    S, BM = SPLITS, 64
    QB = triton.cdiv(N * G, BM)
    dev = q.device
    pacc = _buf("pacc", (HKV * QB * (S + 1), BM, D), torch.float32, dev)
    pml = _buf("pml", (HKV * QB * (S + 1), 2, BM), torch.float32, dev)
    has_mask = custom_mask is not None
    _ext().kattn_tv(q, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices,
                    custom_mask if has_mask else kv_indptr, mask_indptr if has_mask else kv_indptr, pacc, pml,
                    float(sm_scale), N, S, MINCH, int(has_mask), BN, STAGES)
    RBLK = triton.cdiv(N * G, RB)
    _d2a_combine[(RBLK, HKV, D // DC)](
        o, pacc, pml, kv_indptr, o.stride(0), o.stride(1),
        N=N, G=G, D=D, BM=BM, S=S, QB=QB, RB=RB, RBLK=RBLK, DC=DC, MINCH=MINCH,
        SP=triton.next_power_of_2(S + 1), num_warps=4,
    )
    return o
