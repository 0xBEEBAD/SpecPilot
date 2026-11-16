"""SPECPILOT_D3K_QMAP: INT4 / INT5 formats of the D3Q Q copy and a per-matrix bit map (TLV Q pass, sm_120).

Formats (symmetric, one bf16 scale per (row, 128 input channels), round-to-nearest at the per-group MSE-best clip):
  INT4  u = q + 8 in [0, 16): the nibble plane only (record 8192 + 256 B per 128 x 128 tile stage)
  INT5  u = q + 16 in [0, 32): nibble plane + a 1-bit plane, 4 B per lane and quarter: bit 4 of u of the weight in
        register r (0..3) / half h of k16 step j sits at word bit 16 h + 4 r + j (record 8192 + 2048 + 256 B)
  INT6 / INT8 as specpilot_d3_qcopy (unchanged records and arithmetic).
The kernel is specpilot_d3_qcopy's D3Q source with two more decode paths (HB = 0: INT4, HB = 1: INT5, both exact
bf16 (128 + u) - (128 + 2^(b-1)) by one bf16x2 fma) and their instantiations; 6 / 8-bit instantiations compile from the
same template code. Clip grid: _GRID for 6 / 8 bits (as the parent), _GRID_LOW (1.0 .. 0.5) for 4 / 5 bits.

Knob (read once at import): SPECPILOT_D3K_QMAP = ";"-separated rules "<class>[@<lo>-<hi>]=<bits>", later rules win;
classes gate_up, down, in_proj (GDN qkvz + ba, folded), qkv (attention), out (GDN out_proj), o (attention o_proj),
mlp (= gate_up + down), all; layer range inclusive; bits in {0 (nested view), 4, 5, 6, 8}. Matrices no rule covers keep
SPECPILOT_D3_QBITS_MLP / _IN / _REST. Unset = parent behaviour (parent kernel source, formats 6 / 8 only).
"""
from __future__ import annotations

import hashlib
import os
import re

import torch

from sglang.kernels.ops.gemm import specpilot_d3_qcopy as _d3q

QMAP_RAW = (os.environ.get("SPECPILOT_D3K_QMAP", "") or "").strip()
_GRID_LOW = tuple(round(1.0 - 0.05 * i, 2) for i in range(11))
_NS_LOW = {4: 5, 5: 4}  # ring depths that fit ~99 KB with the smaller records (NT = 32, SiLU exchange included)
CLASSES = ("gate_up", "down", "in_proj", "qkv", "out", "o")
_ALIAS = {"mlp": ("gate_up", "down"), "all": CLASSES, "rest": ("out", "qkv", "o"),
          "gu": ("gate_up",), "dn": ("down",), "in": ("in_proj",)}  # gu / dn / in / rest: kattn's SPECPILOT_D2A_QMAP names


def parse_qmap(raw: str):
    """-> list of (classes tuple, lo, hi, bits)."""
    rules = []
    for part in [p.strip() for p in raw.split(";") if p.strip()]:
        m = re.fullmatch(r"([a-z_]+)(?:@(\d+)-(\d+))?=(\d+)", part)
        m2 = re.fullmatch(r"(\d+)-(\d+):([a-z_]+)=(\d+)", part)  # kattn's form <lo>-<hi>:<class>=<bits>
        if m:
            cls, lo, hi, bits = m.group(1), m.group(2), m.group(3), int(m.group(4))
        elif m2:
            lo, hi, cls, bits = m2.group(1), m2.group(2), m2.group(3), int(m2.group(4))
        else:
            raise ValueError(f"SPECPILOT_D3K_QMAP: bad rule {part!r} (want <class>[@<lo>-<hi>]=<bits> or <lo>-<hi>:<class>=<bits>)")
        classes = _ALIAS.get(cls, (cls,))
        if any(c not in CLASSES for c in classes):
            raise ValueError(f"SPECPILOT_D3K_QMAP: unknown class {cls!r} (one of {CLASSES + tuple(_ALIAS)})")
        if bits not in (0, 4, 5, 6, 8):
            raise ValueError(f"SPECPILOT_D3K_QMAP: bits {bits} not in 0/4/5/6/8")
        rules.append((classes, int(lo) if lo else 0, int(hi) if hi else 1 << 30, bits))
    return rules


RULES = parse_qmap(QMAP_RAW) if QMAP_RAW else []
# SPECPILOT_D3K_CAPTURE (debug): "<dir>:<module name>,<module name>,..": eager (prefill, M > 64) input rows of these
# target linears are appended to <dir>/<module name>.<n>.pt (bf16, every SPECPILOT_D3K_CAPTURE_STRIDE-th row)
_CAP_RAW = (os.environ.get("SPECPILOT_D3K_CAPTURE", "") or "").strip()
CAPTURE_DIR, _, _cap_names = _CAP_RAW.partition(":")
CAPTURE = frozenset(n for n in _cap_names.split(",") if n)
_CAP_STRIDE = int(os.environ.get("SPECPILOT_D3K_CAPTURE_STRIDE", "2") or 2)
_CAP_COUNT: dict = {}


def capture(name: str, x2: torch.Tensor) -> None:
    if x2.shape[0] <= 64 or torch.cuda.is_current_stream_capturing():
        return
    i = _CAP_COUNT.get(name, 0)
    _CAP_COUNT[name] = i + 1
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    torch.save(x2[::_CAP_STRIDE].to(torch.bfloat16).cpu(), os.path.join(CAPTURE_DIR, f"{name}.{i}.pt"))


def enabled() -> bool:
    return bool(RULES)


def matrix_class(name: str) -> str | None:
    """Module name of an encoded target linear -> its QMAP class (None: not a Q-copied linear)."""
    if name.endswith("gate_up_proj"):
        return "gate_up"
    if name.endswith("down_proj"):
        return "down"
    if name.endswith("in_proj_qkvz") or name.endswith("in_proj_ba"):
        return "in_proj"
    if name.endswith("qkv_proj"):
        return "qkv"
    if name.endswith("out_proj"):
        return "out"
    if name.endswith("o_proj"):
        return "o"
    return None


def layer_index(name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def bits_for(name: str, default: int) -> int:
    cls, li = matrix_class(name), layer_index(name)
    bits = default
    for classes, lo, hi, b in RULES:
        if cls in classes and lo <= li <= hi:
            bits = b
    return bits


# ------------------------------------------------------------------------------------------------------------ kernel


def _kernel_source() -> str:
    kern = _d3q._KERN
    old = '''    if (HB == 2) {
      const uint32_t hw = H[j >> 1] >> (2 * (j & 1));'''
    new = '''    if (HB == 0) {  // INT4, q = u - 8: bf16x2 (128 + u) - 136, exact
      A[j][0] = bfma(a0, 0x3F803F80u, 0xC308C308u); A[j][1] = bfma(a1, 0x3F803F80u, 0xC308C308u);
      A[j][2] = bfma(a2, 0x3F803F80u, 0xC308C308u); A[j][3] = bfma(a3, 0x3F803F80u, 0xC308C308u);
    } else if (HB == 1) {  // INT5: bit 4 of u at word bit 16 h + 4 r + j; q = u - 16: (128 + u) - 144, exact
      const uint32_t hw = H[0];
      a0 = or5(a0, hw >> (0 + j)); a1 = or5(a1, hw >> (4 + j)); a2 = or5(a2, hw >> (8 + j)); a3 = or5(a3, hw >> (12 + j));
      A[j][0] = bfma(a0, 0x3F803F80u, 0xC310C310u); A[j][1] = bfma(a1, 0x3F803F80u, 0xC310C310u);
      A[j][2] = bfma(a2, 0x3F803F80u, 0xC310C310u); A[j][3] = bfma(a3, 0x3F803F80u, 0xC310C310u);
    } else if (HB == 2) {
      const uint32_t hw = H[j >> 1] >> (2 * (j & 1));'''
    reps = [
        (old, new),
        ('''__device__ __forceinline__ uint32_t bfma(''',
         '''__device__ __forceinline__ uint32_t or5(uint32_t a, uint32_t h) {  // a | ((h << 4) & 0x00100010)
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, 0x00100010, 0xF8;" : "=r"(r) : "r"(a), "r"(h << 4));
  return r;
}
__device__ __forceinline__ uint32_t bfma('''),
        ('''  constexpr int HQ = HB == 2 ? 8 : 16;       // high-plane bytes per lane and quarter''',
         '''  constexpr int HQ = HB * 4;                 // high-plane bytes per lane and quarter (INT4 0 .. INT8 16)'''),
        ('''        uint4 hv;
        if (HB == 2) {''', '''        uint4 hv = make_uint4(0u, 0u, 0u, 0u);
        if (HB == 1) {
          hv.x = reinterpret_cast<const uint32_t*>(rb + HOFF)[li];
        } else if (HB == 2) {'''),
        ('''        } else {
          hv = reinterpret_cast<const uint4*>(rb + HOFF)[li];
        }''', '''        } else if (HB == 4) {
          hv = reinterpret_cast<const uint4*>(rb + HOFF)[li];
        }'''),
    ]
    for a, b in reps:
        assert kern.count(a) == 1, a[:60]
        kern = kern.replace(a, b)
    host = _d3q._HOST.replace("NS6", str(_d3q._NS[6])).replace("NS8", str(_d3q._NS[8]))
    old = '''  const int key = (int)(nt == 32) * 4 + (int)(bits == 8) * 2 + (int)(silu != 0);
  switch (key) {'''
    assert host.count(old) == 1
    host = host.replace(old, '''  if (bits == 4 || bits == 5) {
    const int k2 = (int)(nt == 32) * 4 + (int)(bits == 5) * 2 + (int)(silu != 0);
    switch (k2) {
      case 0: D3Q_L(16, 0, NS4, 0); break;
      case 1: D3Q_L(16, 0, NS4, 1); break;
      case 2: D3Q_L(16, 1, NS5, 0); break;
      case 3: D3Q_L(16, 1, NS5, 1); break;
      case 4: D3Q_L(32, 0, NS4, 0); break;
      case 5: D3Q_L(32, 0, NS4, 1); break;
      case 6: D3Q_L(32, 1, NS5, 0); break;
      case 7: D3Q_L(32, 1, NS5, 1); break;
    }
    return;
  }
''' + old).replace("NS4", str(_NS_LOW[4])).replace("NS5", str(_NS_LOW[5]))
    # own symbol names: two torch extensions exporting the same symbols interpose each other's host code
    src = kern + host
    for a, b in (("d3q_kernel", "d3kf_kernel"), ("void launch(", "void launchf("), ("launch<", "launchf<"),
                 ("d3q_gemm", "d3kf_gemm"), ("get_encode", "get_encode_kf")):
        src = src.replace(a, b)
    return src


_CPP = _d3q._CPP.replace("d3q_gemm", "d3kf_gemm")
_mod = None


def ext():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        src = _kernel_source()
        base = os.environ.get("SGLANG_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "sglang")
        key = hashlib.sha1("\0".join([_CPP, src, *_d3q._CFLAGS]).encode()).hexdigest()[:16]
        bdir = os.path.join(base, "specpilot_d3k_qfmt", key)
        os.makedirs(bdir, exist_ok=True)
        _mod = load_inline(name="specpilot_d3k_qfmt_" + key, cpp_sources=_CPP, cuda_sources=src,
                           functions=["d3kf_gemm"], extra_cuda_cflags=_d3q._CFLAGS, build_directory=bdir, verbose=False)
    return _mod


# ------------------------------------------------------------------------------------------------- quantize / records


def quantize_low(w: torch.Tensor, bits: int, chunk: int = 2048):
    """bf16 [N, K] -> (u uint8 [N, K], scale bf16 [N, K/128]) for 4 / 5 bits: _d3q.quantize_q on the finer grid."""
    keep = _d3q._GRID
    _d3q._GRID = _GRID_LOW
    try:
        return _d3q.quantize_q(w, bits, chunk)
    finally:
        _d3q._GRID = keep


def build_records_low(u: torch.Tensor, sc: torch.Tensor, bits: int, chunk_tiles: int = 16):
    """-> (recs uint8 [T, S * rec_bytes], rec_bytes, xoff) for INT4 / INT5. Padding rows: q = 0, s = 0."""
    assert bits in (4, 5)
    n, k = u.shape
    S = k // _d3q._STK
    T = (n + _d3q._RB - 1) // _d3q._RB
    hq = 0 if bits == 4 else 4
    rec_bytes = 8192 + hq * 512 + 256
    xoff = (rec_bytes + 1023) // 1024 * 1024
    recs = torch.empty((T, S * rec_bytes), dtype=torch.uint8, device=u.device)
    mid = 2 ** (bits - 1)
    # word bit of (j, byte, nib) for INT5: 16 h + 4 r + j with h = byte & 1, r = (byte >> 1) + 2 nib
    jj = torch.arange(4, device=u.device).view(4, 1, 1)
    bb = torch.arange(4, device=u.device).view(1, 4, 1)
    nn = torch.arange(2, device=u.device).view(1, 1, 2)
    shift = (16 * (bb & 1) + 4 * ((bb >> 1) + 2 * nn) + jj).to(torch.int64)  # [4 j, 4 byte, 2 nib]
    for t0 in range(0, T, chunk_tiles):
        t1 = min(T, t0 + chunk_tiles)
        r0, r1 = t0 * _d3q._RB, min(n, t1 * _d3q._RB)
        uu = torch.full(((t1 - t0) * _d3q._RB, k), mid, dtype=torch.uint8, device=u.device)
        uu[: r1 - r0] = u[r0:r1]
        L = _d3q._lane_order(uu, t1 - t0, S)  # [T, S, w, hh, lane, j, byte, nib]
        lo = L & 0xF
        nibw = (lo[..., 0] | (lo[..., 1] << 4)).reshape(t1 - t0, S, 8192).to(torch.uint8)
        parts = [nibw]
        if bits == 5:
            hi = (L >> 4).to(torch.int64) & 1
            word = (hi << shift).sum(dim=(-3, -2, -1))  # [T, S, w, hh, lane] uint32 values
            wb = torch.stack([(word >> (8 * i)) & 0xFF for i in range(4)], -1).to(torch.uint8)  # little endian
            parts.append(wb.reshape(t1 - t0, S, 2048))
        scs = torch.zeros(((t1 - t0) * _d3q._RB, S), dtype=torch.bfloat16, device=u.device)
        scs[: r1 - r0] = sc[r0:r1]
        parts.append(scs.view(t1 - t0, _d3q._RB, S).permute(0, 2, 1).contiguous().view(torch.uint8).view(t1 - t0, S, 256))
        recs[t0:t1] = torch.cat(parts, dim=2).view(t1 - t0, S * rec_bytes)
    return recs, rec_bytes, xoff


class D3QLowLinear(_d3q.D3QLinear):
    """D3QLinear for bits in (4, 5) (6 / 8: use D3QLinear); same call contract (M <= 32)."""

    def __init__(self, w: torch.Tensor, bits: int, silu: bool = False, pdl: int = 1, nsm: int | None = None,
                 trig: int | None = None):
        assert bits in (4, 5) and w.dim() == 2 and w.shape[1] % _d3q._STK == 0
        self.bits, self.n, self.k, self.silu = bits, w.shape[0], w.shape[1], int(silu)
        if silu:
            assert self.n % 128 == 0
            w = w.view(2, self.n // 128, 64, self.k).transpose(0, 1).reshape(self.n, self.k)
        self.nout = self.n // 2 if silu else self.n
        u, sc = quantize_low(w, bits)
        self.recs, self.rec_bytes, self.xoff = build_records_low(u, sc, bits)
        del u, sc
        self.T, self.S = self.recs.shape[0], self.k // _d3q._STK
        nsm = nsm or torch.cuda.get_device_properties(w.device).multi_processor_count
        self.P = min(nsm, self.T * self.S)
        self.maxseg = _d3q._maxseg(self.T, self.S, self.P)
        trig = _d3q._K_TRIG if trig is None else trig
        self.pdl = (1 if pdl else 0) | (2 if pdl and trig else 0)
        _d3q._ensure_shared(w.device, self.T * self.maxseg * _d3q._RB * 32, self.T)

    def __call__(self, x2: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        m = x2.shape[0]
        assert 0 < m <= 32 and x2.dtype == torch.bfloat16 and x2.stride(-1) == 1 and x2.stride(0) % 8 == 0
        nt = 16 if m <= 16 else 32
        if out is None:
            out = torch.empty((m, self.nout), device=x2.device, dtype=torch.bfloat16)
        sh = _d3q._SHARED[x2.device]
        ext().d3kf_gemm(x2, m, self.recs, self.S, self.rec_bytes, self.xoff, out, sh["ws"], sh["cnt"], self.P,
                       self.maxseg, self.nout, self.bits, nt, self.pdl, self.silu)
        return out


def make_linear(w: torch.Tensor, bits: int, silu: bool = False):
    return D3QLowLinear(w, bits, silu=silu) if bits in (4, 5) else _d3q.D3QLinear(w, bits, silu=silu)


def relative_mse(w: torch.Tensor, bits: int) -> float:
    """Weight-space ||deq(q(w)) - w||^2 / ||w||^2 of this format (data-free sensitivity prior)."""
    u, sc = quantize_low(w, bits) if bits in (4, 5) else _d3q.quantize_q(w, bits)
    d = _d3q.dequantize_q(u, sc, bits) - w.float()
    return float(d.pow(2).sum() / w.float().pow(2).sum())


# --------------------------------------------------------------------------------------- data-aware (GPTQ) Q copy
# SPECPILOT_D3K_GPTQ=<dir>: matrices with a Hessian file <dir>/<module name>.H.pt ({"H": fp32 [K, K] = X^T X / n of
# captured input rows, "n": rows}) get GPTQ-rounded codes in the D3Q format at their QMAP / class bits (6 / 8 bits
# through the parent kernel, 4 / 5 through the extended one); codes are cached in <dir>/codes/. Proposal-side derived
# data: the exact pass is unchanged. Unset = parent RTN copies.
GPTQ_DIR = (os.environ.get("SPECPILOT_D3K_GPTQ", "") or "").strip()
GPTQ_DAMP = float(os.environ.get("SPECPILOT_D3K_GPTQ_DAMP", "0.01") or 0.01)


def _scales_group(W: torch.Tensor, bits: int, grid) -> torch.Tensor:
    """[N, 128] fp32 -> [N] fp32 (bf16-representable) MSE-best clip scale of one 128-column group."""
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(-1, keepdim=True)
    best_e = best_s = None
    for a in grid:
        s = (amax * (a / qmax)).to(torch.bfloat16).float().clamp_min(1e-12)
        e = (torch.clamp(torch.round(W / s), -qmax - 1, qmax) * s - W).pow(2).sum(-1, keepdim=True)
        if best_e is None:
            best_e, best_s = e, s
        else:
            t = e < best_e
            best_e, best_s = torch.where(t, e, best_e), torch.where(t, s, best_s)
    return best_s.view(-1)


@torch.no_grad()
def gptq_codes(w: torch.Tensor, H: torch.Tensor, bits: int, damp: float = GPTQ_DAMP):
    """GPTQ (group 128 = the D3Q scale group, scale fit on the error-updated group) -> (u uint8 [N, K], sc bf16 [N, K/128])."""
    grid = _GRID_LOW if bits in (4, 5) else _d3q._GRID
    W = w.float().clone()
    n, k = W.shape
    qmax = 2 ** (bits - 1) - 1
    H = H.float().to(W.device).clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(k, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    u = torch.empty((n, k), dtype=torch.uint8, device=W.device)
    sc = torch.empty((n, k // 128), dtype=torch.bfloat16, device=W.device)
    for g, i1 in enumerate(range(0, k, 128)):
        i2 = i1 + 128
        W1 = W[:, i1:i2].clone()
        Hi = Hinv[i1:i2, i1:i2]
        s = _scales_group(W1, bits, grid)
        sc[:, g] = s.to(torch.bfloat16)
        Err = torch.zeros_like(W1)
        for i in range(128):
            wcol = W1[:, i]
            q = torch.clamp(torch.round(wcol / s), -qmax - 1, qmax)
            u[:, i1 + i] = (q + qmax + 1).to(torch.uint8)
            e = (wcol - q * s) / Hi[i, i]
            W1[:, i:] -= e.unsqueeze(1) * Hi[i, i:].unsqueeze(0)
            Err[:, i] = e
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return u, sc


def _interleave_rows(t: torch.Tensor) -> torch.Tensor:
    """[gate | up] rows -> per 128-row tile [64 gate | 64 up] (D3QLinear's silu layout)."""
    n = t.shape[0]
    return t.view(2, n // 128, 64, *t.shape[1:]).transpose(0, 1).reshape(n, *t.shape[1:])


def linear_from_codes(u: torch.Tensor, sc: torch.Tensor, bits: int, silu: bool, pdl: int = 1):
    n, k = u.shape
    if silu:
        u, sc = _interleave_rows(u), _interleave_rows(sc)
    cls = D3QLowLinear if bits in (4, 5) else _d3q.D3QLinear
    lin = cls.__new__(cls)
    lin.bits, lin.n, lin.k, lin.silu = bits, n, k, int(silu)
    lin.nout = n // 2 if silu else n
    build = build_records_low if bits in (4, 5) else _d3q.build_records
    lin.recs, lin.rec_bytes, lin.xoff = build(u, sc, bits)
    lin.T, lin.S = lin.recs.shape[0], k // _d3q._STK
    lin.P = min(torch.cuda.get_device_properties(u.device).multi_processor_count, lin.T * lin.S)
    lin.maxseg = _d3q._maxseg(lin.T, lin.S, lin.P)
    lin.pdl = (1 if pdl else 0) | (2 if pdl and _d3q._K_TRIG else 0)
    _d3q._ensure_shared(u.device, lin.T * lin.maxseg * _d3q._RB * 32, lin.T)
    return lin


def gptq_linear(name: str, w: torch.Tensor, bits: int, silu: bool):
    """None when no Hessian file exists for this module; else the GPTQ-rounded Q copy (codes cached on disk)."""
    hp = os.path.join(GPTQ_DIR, f"{name}.H.pt")
    if not GPTQ_DIR or bits not in (4, 5, 6, 8) or not os.path.exists(hp):
        return None
    cdir = os.path.join(GPTQ_DIR, "codes")
    cp = os.path.join(cdir, f"{name}.b{bits}.d{GPTQ_DAMP:g}.pt")
    if os.path.exists(cp):
        d = torch.load(cp, map_location=w.device)
        u, sc = d["u"], d["sc"]
    else:
        H = torch.load(hp, map_location=w.device)["H"]
        u, sc = gptq_codes(w, H, bits)
        os.makedirs(cdir, exist_ok=True)
        torch.save({"u": u.cpu(), "sc": sc.cpu()}, cp)
        del H
    return linear_from_codes(u, sc, bits, silu)
