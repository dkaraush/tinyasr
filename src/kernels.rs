//! Matrix kernels. Float: the micro-kernel is explicit NEON on aarch64; elsewhere it is written so LLVM
//! auto-vectorizes it (SIMD128 on wasm, SSE on x86-64). Integer weights (`Quant`): see below.

use alloc::vec;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use crate::compat::F32Ext;

// Micro-kernel tile: MR rows x NR columns of accumulators. aarch64 has 32 vector registers
// (4x16 = 16 accumulators); wasm SIMD / x86-64 SSE have 16 (2x16 = 8).
const NR: usize = 16; // output columns per panel
#[cfg(target_arch = "aarch64")]
const MR: usize = 6; // output rows per micro-kernel call
#[cfg(not(target_arch = "aarch64"))]
const MR: usize = 6;
const MB: usize = 64; // rows per cache block
/// Rows up to which x86-64 multiplies integer weights widened in registers (more: widened once per cache block).
#[cfg(target_arch = "x86_64")]
const SMALL: usize = 16;

/// Right-hand matrix packed into panels of NR output columns, k-major inside a panel, so the
/// micro-kernel reads NR contiguous weights per k step (outer-product form, no reductions).
pub struct Packed {
    pub n: usize,
    pub k: usize,
    data: Vec<f32>, // ceil(n / NR) panels x k x NR, zero-padded
}

impl Packed {
    /// From b (n x k): row j holds the weights of output column j (PyTorch Linear layout).
    pub fn from_rows(b: &[f32], n: usize, k: usize) -> Packed {
        let mut data = vec![0.0; n.div_ceil(NR) * k * NR];
        for j in 0..n {
            let (panel, l) = (j / NR, j % NR);
            for (p, v) in b[j * k..(j + 1) * k].iter().enumerate() {
                data[(panel * k + p) * NR + l] = *v;
            }
        }
        Packed { n, k, data }
    }

    fn panel(&self, i: usize) -> &[f32] {
        &self.data[i * self.k * NR..(i + 1) * self.k * NR]
    }
}

/// out (m x n) = a (m x k) * B (+ bias per column).
pub fn gemm(a: &[f32], m: usize, b: &Packed, bias: Option<&[f32]>) -> Vec<f32> {
    let (n, k) = (b.n, b.k);
    assert!(a.len() >= m * k);
    let mut out = crate::pool::any(m * n); // every value is written
    gemm_rows(&a[..m * k], b, bias, &mut out);
    out
}

/// `gemm` into out (m x n, written whole).
pub fn gemm_into(a: &[f32], m: usize, b: &Packed, bias: Option<&[f32]>, out: &mut [f32]) {
    assert!(a.len() >= m * b.k && out.len() == m * b.n);
    gemm_rows(&a[..m * b.k], b, bias, out);
}

fn gemm_rows(a: &[f32], b: &Packed, bias: Option<&[f32]>, out: &mut [f32]) {
    blocked(a, b, bias, out);
}

/// Weights as panels of NR output columns, k-major inside a panel (see `Packed`).
trait Panels {
    fn n(&self) -> usize;
    fn k(&self) -> usize;
    /// Panel `jp`: stored, or widened into `scratch` (k x NR).
    fn panel<'s>(&'s self, jp: usize, scratch: &'s mut [f32]) -> &'s [f32];
    /// Whether `panel` needs the scratch (false: it may be empty).
    fn widens(&self) -> bool {
        true
    }
}

impl Panels for Packed {
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn panel<'s>(&'s self, jp: usize, _: &'s mut [f32]) -> &'s [f32] {
        Packed::panel(self, jp)
    }
    fn widens(&self) -> bool {
        false
    }
}

/// The cache-blocked loop around the micro-kernel: blocks of MB rows of `a`, one weight panel at a time.
fn blocked(a: &[f32], b: &(impl Panels + Sync), bias: Option<&[f32]>, out: &mut [f32]) {
    let (n, k) = (b.n(), b.k());
    let m = a.len() / k;
    // blocks of rows are independent: one task each
    let mb = crate::par::block_rows(m, MB);
    crate::par::chunks_mut(out, mb * n, m * n * k, |blk, out| {
        let i0 = blk * mb;
        let i1 = (i0 + mb).min(m);
        let mut scratch = vec![0f32; if b.widens() { k * NR } else { 0 }];
        for jp in 0..n.div_ceil(NR) {
            let panel = b.panel(jp, &mut scratch);
            let j0 = jp * NR;
            let cols = NR.min(n - j0);
            let mut init = [0f32; NR];
            if let Some(bias) = bias {
                init[..cols].copy_from_slice(&bias[j0..j0 + cols]);
            }
            let o = &mut out[j0..];
            #[cfg(target_arch = "x86_64")]
            #[allow(unsafe_code)]
            // SAFETY: the level was detected on this CPU (`x86`)
            match x86() {
                X86::Avx512 => unsafe { x86::rows_avx512(&a[i0 * k..i1 * k], k, panel, &init, o, n, cols) },
                X86::Avx2 => unsafe { x86::rows_avx2(&a[i0 * k..i1 * k], k, panel, &init, o, n, cols) },
                X86::Sse2 => rows(&a[i0 * k..i1 * k], k, panel, &init, o, n, cols),
            }
            #[cfg(not(target_arch = "x86_64"))]
            rows(&a[i0 * k..i1 * k], k, panel, &init, o, n, cols);
        }
    });
}

/// Rows of `a` (k each) x one panel into the first `cols` columns of `out` (row r at r * stride), MR rows per
/// micro-kernel call.
#[inline(always)]
fn rows(a: &[f32], k: usize, panel: &[f32], init: &[f32; NR], out: &mut [f32], stride: usize, cols: usize) {
    let m = a.len() / k;
    let mut tile = [[0f32; NR]; MR];
    let mut i = 0;
    while i < m {
        let rows = MR.min(m - i);
        if rows == MR {
            kernel::<MR>(&a[i * k..(i + MR) * k], k, panel, init, &mut tile);
        } else {
            for r in 0..rows {
                let mut one = [[0f32; NR]; 1];
                kernel::<1>(&a[(i + r) * k..(i + r + 1) * k], k, panel, init, &mut one);
                tile[r] = one[0];
            }
        }
        for r in 0..rows {
            out[(i + r) * stride..(i + r) * stride + cols].copy_from_slice(&tile[r][..cols]);
        }
        i += rows;
    }
}

/// out = init + rows(a) x panel, for R rows of `a` (each k long) and one panel of NR columns.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn kernel<const R: usize>(a: &[f32], k: usize, panel: &[f32], init: &[f32; NR], out: &mut [[f32; NR]; R]) {
    let rows: [&[f32]; R] = core::array::from_fn(|r| &a[r * k..(r + 1) * k]);
    let mut acc = [*init; R];
    for (p, b) in panel[..k * NR].chunks_exact(NR).enumerate() {
        for r in 0..R {
            let x = rows[r][p];
            for l in 0..NR {
                acc[r][l] += x * b[l];
            }
        }
    }
    *out = acc;
}

/// NEON version: R x 16 accumulators in q registers, fused multiply-add (fmla by element).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_code)]
fn kernel<const R: usize>(a: &[f32], k: usize, panel: &[f32], init: &[f32; NR], out: &mut [[f32; NR]; R]) {
    use core::arch::aarch64::{float32x4_t, vdupq_n_f32, vfmaq_f32, vfmaq_laneq_f32, vld1q_f32, vst1q_f32};
    const { assert!(NR == 16) };
    assert!(a.len() >= R * k && panel.len() >= k * NR);
    // SAFETY: NEON is mandatory on aarch64; every load/store stays inside the slices checked above
    // (row r reads a[r*k + p], p < k; panel reads panel[p*16 .. p*16+16], p < k).
    unsafe {
        let ip = init.as_ptr();
        let start: [float32x4_t; 4] = [vld1q_f32(ip), vld1q_f32(ip.add(4)), vld1q_f32(ip.add(8)), vld1q_f32(ip.add(12))];
        let mut acc = [start; R];
        let (ap, bp) = (a.as_ptr(), panel.as_ptr());
        // 4 k at a time: one load of each row's 4 values, used by lane (the same multiply-adds in the same order)
        macro_rules! step {
            ($l:literal, $p:expr, $av:expr) => {
                let b = bp.add($p * NR);
                let (b0, b1, b2, b3) = (vld1q_f32(b), vld1q_f32(b.add(4)), vld1q_f32(b.add(8)), vld1q_f32(b.add(12)));
                for (acc, x) in acc.iter_mut().zip($av.iter()) {
                    acc[0] = vfmaq_laneq_f32::<$l>(acc[0], b0, *x);
                    acc[1] = vfmaq_laneq_f32::<$l>(acc[1], b1, *x);
                    acc[2] = vfmaq_laneq_f32::<$l>(acc[2], b2, *x);
                    acc[3] = vfmaq_laneq_f32::<$l>(acc[3], b3, *x);
                }
            };
        }
        let k4 = k / 4 * 4;
        let mut p = 0;
        while p < k4 {
            let av: [float32x4_t; R] = core::array::from_fn(|r| vld1q_f32(ap.add(r * k + p)));
            step!(0, p, av);
            step!(1, p + 1, av);
            step!(2, p + 2, av);
            step!(3, p + 3, av);
            p += 4;
        }
        for p in k4..k {
            let b = bp.add(p * NR);
            let (b0, b1, b2, b3) = (vld1q_f32(b), vld1q_f32(b.add(4)), vld1q_f32(b.add(8)), vld1q_f32(b.add(12)));
            for (r, acc) in acc.iter_mut().enumerate() {
                let x = vdupq_n_f32(*ap.add(r * k + p));
                acc[0] = vfmaq_f32(acc[0], b0, x);
                acc[1] = vfmaq_f32(acc[1], b1, x);
                acc[2] = vfmaq_f32(acc[2], b2, x);
                acc[3] = vfmaq_f32(acc[3], b3, x);
            }
        }
        for (o, acc) in out.iter_mut().zip(&acc) {
            for (j, v) in acc.iter().enumerate() {
                vst1q_f32(o.as_mut_ptr().add(4 * j), *v);
            }
        }
    }
}

/// exp for x <= ~88 via 2^x = 2^i * 2^f with a degree-5 polynomial (rel. err ~2e-7).
/// Branch-free so loops over it vectorize, unlike the libm call.
#[inline(always)]
#[allow(clippy::approx_constant)] // polynomial coefficients, not ln 2
pub fn exp(x: f32) -> f32 {
    let t = (x * core::f32::consts::LOG2_E).clamp(-126.0, 126.0);
    #[cfg(not(target_arch = "x86_64"))]
    let i = t.floor();
    // floor without SSE4.1 (a libm call): t rounded to an integer by adding and taking away 1.5 * 2^23 (exact for
    // |t| < 2^22), one less if that rounded up; the same bits (for t = -0, f below is the same 0)
    #[cfg(target_arch = "x86_64")]
    let i = {
        let r = (t + 12_582_912.0) - 12_582_912.0;
        if r > t { r - 1.0 } else { r }
    };
    let f = t - i;
    let p = 1.0 + f * (0.693_147_2 + f * (0.240_226_5 + f * (0.055_504_11 + f * (0.009_618_129 + f * 0.001_333_355))));
    // 2^i: the biased exponent i + 127 (an integer in 1..=253) read off the mantissa of i + 127 + 2^23, which is exact;
    // the same bits as `(i as i32 + 127) << 23`, but a saturating float -> int cast does not vectorize
    f32::from_bits(((i + 127.0 + 8_388_608.0).to_bits() - 0x4b00_0000) << 23) * p
}

/// ln for x > 0 (normal): x = m 2^e with m in [sqrt(1/2), sqrt 2), ln m = 2 atanh(s), s = (m - 1) / (m + 1), as an odd
/// series to s^11 (|s| < 0.172: rel. err < 1e-8). Branch-free so loops over it vectorize, unlike the libm call.
#[inline(always)]
pub fn ln(x: f32) -> f32 {
    let b = x.to_bits();
    // exponent relative to sqrt(1/2), so the mantissa lands in [sqrt(1/2), sqrt 2)
    let e = ((b.wrapping_sub(0x3f35_04f3)) as i32) >> 23;
    let m = f32::from_bits(b.wrapping_sub((e as u32) << 23));
    let s = (m - 1.0) / (m + 1.0);
    let z = s * s;
    let p = 2.0 + z * (0.666_666_7 + z * (0.4 + z * (0.285_714_3 + z * (0.222_222_2 + z * 0.181_818_2))));
    e as f32 * core::f32::consts::LN_2 + s * p
}

/// `swoosh` (model.rs) on NEON, 4 values at a time: softplus(v - shift) - 0.08 v - c with softplus(y) =
/// max(y, 0) + ln(1 + e^-|y|): e^-|y| with the polynomial of `exp`, ln(1 + e) with one polynomial in e (e <= 1), within
/// ~2e-7 of the scalar form. x.len() a multiple of 4.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, clippy::approx_constant)]
pub fn swoosh4(x: &mut [f32], shift: f32, c: f32) {
    use core::arch::aarch64::*;
    assert!(x.len().is_multiple_of(4));
    // SAFETY: NEON is mandatory on aarch64; every access is a whole 4-value chunk of x.
    unsafe {
        let f = |v: f32| vdupq_n_f32(v);
        for ch in x.chunks_exact_mut(4) {
            let v = vld1q_f32(ch.as_ptr());
            let y = vsubq_f32(v, f(shift));
            // e^-|y| = 2^i 2^t (as `exp`)
            let t = vmaxq_f32(vminq_f32(vmulq_f32(vnegq_f32(vabsq_f32(y)), f(core::f32::consts::LOG2_E)), f(126.0)), f(-126.0));
            let i = vrndmq_f32(t);
            let fr = vsubq_f32(t, i);
            let mut p = f(0.001_333_355);
            for k in [0.009_618_129, 0.055_504_11, 0.240_226_5, 0.693_147_2, 1.0] {
                p = vfmaq_f32(f(k), p, fr);
            }
            let e = vmulq_f32(vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(i), vdupq_n_s32(127)))), p);
            // ln(1 + e) for e in (0, 1]: e q(e), q of degree 7 (near-minimax fit, abs. err ~1e-7 with the rounding)
            let mut q = f(-0.006_452_612_6);
            for k in [0.036_084_52, -0.095_322_49, 0.167_647_85, -0.240_730_76, 0.331_798_26, -0.499_874_06, 0.999_996_4] {
                q = vfmaq_f32(f(k), q, e);
            }
            let ln = vmulq_f32(e, q);
            let sp = vaddq_f32(vmaxq_f32(y, f(0.0)), ln);
            vst1q_f32(ch.as_mut_ptr(), vsubq_f32(vsubq_f32(sp, vmulq_f32(f(0.08), v)), f(c)));
        }
    }
}

/// Runs `f` (inlined) with this CPU's vector extensions on, so the loops in it that LLVM vectorizes use them (x86-64:
/// AVX2 / AVX-512, detected at run time). The same operations per value: the same bits.
#[inline(always)]
pub fn vector<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2,fma")]
        fn avx2<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
        fn avx512<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        #[allow(unsafe_code)]
        // SAFETY: the level was detected on this CPU (`x86`).
        match x86() {
            X86::Avx512 => return unsafe { avx512(f) },
            X86::Avx2 => return unsafe { avx2(f) },
            X86::Sse2 => {}
        }
    }
    f()
}

/// x = f(x) for every value, vectorized (`vector`).
#[inline(always)]
pub fn map(x: &mut [f32], f: impl Fn(f32) -> f32) {
    vector(#[inline(always)] || x.iter_mut().for_each(|v| *v = f(*v)));
}

/// log(sum(e^v)) of v with its maximum m (the exponentials vectorized, summed in order).
#[inline(always)]
pub fn log_sum_exp(v: &[f32], m: f32) -> f32 {
    let mut e = vec![0f32; v.len()];
    vector(#[inline(always)] || e.iter_mut().zip(v).for_each(|(e, x)| *e = exp(x - m)));
    m + e.iter().sum::<f32>().ln()
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    1.0 - 2.0 / (exp(2.0 * x) + 1.0)
}

// ---------------------------------------------------------------------------------------------- integer weights
//
// `Quant`: weights as int4 (two per byte, as in the file) or int8 codes with one f32 scale per `group` weights of a row,
// repacked at load into panels of 8 output rows that every integer kernel reads directly:
//
// - a panel is k-major; for every 8 k (int8) it holds 4 vectors of 16 bytes, one per pair of outputs (2p, 2p + 1):
//   [output 2p: k..k+8 | output 2p+1: k..k+8] - the operand layout of `smmla` (i8mm), also used by the `sdot`,
//   plain NEON and wasm kernels;
// - int4: one 16-byte vector per pair for every 16 k: its low nibbles are the 8 k-step at k, its high nibbles the one at
//   k + 8 (two's complement nibbles). The kernels unpack `v << 4` and `v & 0xf0`: the weights times 16 (exact int8), so
//   int4 scales are stored / 16;
// - scales: per panel, per group, 8 f32 (one per output).
//
// Activations are quantized per call to int8 per row and per group (`pack_act`), interleaved the same way in pairs of
// rows, so one 16-byte load is [row 2r: k..k+8 | row 2r+1: k..k+8]. A kernel call computes 2 x 2 output blocks for 1 to
// 4 row pairs x the 4 output pairs of a panel; int32 sums are scaled into f32 at every group end.
//
// The `sdot` path (dotprod without i8mm) has its own panel layout instead (`lay4`, chosen at load): for every 4 k, two
// 16-byte vectors of 4 outputs x those 4 k ([o0: k..k+4 | o1 | o2 | o3], then o4..o7) - the operand of `sdot` by
// element, which multiplies them by 4 k of one activation row (a lane) into the 4 outputs' sums. int4: one vector per
// 4 outputs and 8 k, low nibbles the 4 k at k, high nibbles those at k + 4. Activations: plain int8 rows (`pack_rows`).

/// How `Quant` multiplies, picked once from the CPU.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)] // each target builds only its own paths
pub enum Path {
    /// Weights widened to f32, float micro-kernel (exact; no activation quantization).
    F32,
    /// Portable integer arithmetic (to check accuracy off-device).
    Scalar,
    /// aarch64 without the dot-product extension: smull + sadalp.
    Neon,
    /// aarch64 `sdot` (ARMv8.2 dotprod).
    Dot,
    /// aarch64 `smmla` (ARMv8.6 i8mm).
    Mmla,
    /// wasm SIMD128: widening multiplies (`i16x8.extmul_*_i8x16_s`).
    Wasm,
    /// x86-64 AVX2 (`vpmaddubsw`), AVX-VNNI or AVX-512 VNNI (`vpdpbusd`): `IntX86`.
    X86,
}

/// Row pairs per kernel call at most (i8mm: 4, the others 2).
const RPMAX: usize = 4;

/// Lanes of a 2 x 2 result block (row offset, output offset), the `smmla` order: every pair-layout kernel hands it back.
const MMLA_LANES: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];

/// The integer kernel of this CPU. std builds: TINYASR_KERNEL=f32|scalar|neon|dot|i8mm forces one (for tests;
/// one the CPU lacks is ignored); INT8_DOT=0 is TINYASR_KERNEL=f32.
pub fn path() -> Path {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return Path::Wasm;
    #[cfg(all(target_arch = "aarch64", feature = "std"))]
    {
        static P: std::sync::OnceLock<Path> = std::sync::OnceLock::new();
        *P.get_or_init(|| {
            let mmla = std::arch::is_aarch64_feature_detected!("i8mm");
            let dot = std::arch::is_aarch64_feature_detected!("dotprod");
            if std::env::var("INT8_DOT").is_ok_and(|v| v == "0") {
                return Path::F32;
            }
            match std::env::var("TINYASR_KERNEL").as_deref() {
                Ok("f32") => return Path::F32,
                Ok("scalar") => return Path::Scalar,
                Ok("neon") => return Path::Neon,
                Ok("dot") if dot => return Path::Dot,
                _ => {}
            }
            if mmla && dot {
                Path::Mmla
            } else if dot {
                Path::Dot
            } else {
                Path::Neon
            }
        })
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "std")))]
    {
        // 0 not known yet, else Path as u8 + 1
        use core::sync::atomic::{AtomicU8, Ordering};
        static P: AtomicU8 = AtomicU8::new(0);
        let v = P.load(Ordering::Relaxed);
        if v != 0 {
            return [Path::F32, Path::Scalar, Path::Neon, Path::Dot, Path::Mmla, Path::Wasm, Path::X86][v as usize - 1];
        }
        let (mmla, dot) = cpu_features();
        let p = if mmla && dot {
            Path::Mmla
        } else if dot {
            Path::Dot
        } else {
            Path::Neon
        };
        P.store(p as u8 + 1, Ordering::Relaxed);
        p
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        // the integer arithmetic of the ARM kernels: x86-64 with AVX2 by default (its own kernels, the same results),
        // TINYASR_INT8=scalar (portable); TINYASR_INT8=0 or no AVX2: f32
        #[cfg(feature = "std")]
        {
            static P: std::sync::OnceLock<Path> = std::sync::OnceLock::new();
            *P.get_or_init(|| match std::env::var("TINYASR_INT8").as_deref() {
                Ok("0") => Path::F32,
                Ok("scalar") => Path::Scalar,
                #[cfg(target_arch = "x86_64")]
                _ if x86() >= X86::Avx2 => Path::X86,
                Ok("1") => Path::Scalar,
                _ => Path::F32,
            })
        }
        #[cfg(not(feature = "std"))]
        {
            #[cfg(target_arch = "x86_64")]
            if x86() >= X86::Avx2 {
                return Path::X86;
            }
            Path::F32
        }
    }
}

/// Whether the CPU has i8mm (with dotprod).
#[cfg(target_arch = "aarch64")]
fn has_i8mm() -> bool {
    #[cfg(feature = "std")]
    return std::arch::is_aarch64_feature_detected!("i8mm") && std::arch::is_aarch64_feature_detected!("dotprod");
    #[cfg(not(feature = "std"))]
    {
        let (mmla, dot) = cpu_features();
        mmla && dot
    }
}

/// (i8mm, dotprod) without std: what the kernel reports (Linux, Android: auxiliary vector), sysctl (Apple), else what
/// the build targets.
#[cfg(all(target_arch = "aarch64", not(feature = "std")))]
#[allow(unsafe_code)]
fn cpu_features() -> (bool, bool) {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        unsafe extern "C" {
            fn getauxval(t: core::ffi::c_ulong) -> core::ffi::c_ulong;
        }
        const AT_HWCAP: core::ffi::c_ulong = 16;
        const AT_HWCAP2: core::ffi::c_ulong = 26;
        const HWCAP_ASIMDDP: core::ffi::c_ulong = 1 << 20;
        const HWCAP2_I8MM: core::ffi::c_ulong = 1 << 13;
        // SAFETY: getauxval only reads the auxiliary vector.
        let (h1, h2) = unsafe { (getauxval(AT_HWCAP), getauxval(AT_HWCAP2)) };
        (cfg!(target_feature = "i8mm") || h2 & HWCAP2_I8MM != 0, cfg!(target_feature = "dotprod") || h1 & HWCAP_ASIMDDP != 0)
    }
    #[cfg(target_vendor = "apple")]
    {
        unsafe extern "C" {
            fn sysctlbyname(name: *const core::ffi::c_char, old: *mut core::ffi::c_void, oldlen: *mut usize, new: *const core::ffi::c_void, newlen: usize) -> i32;
        }
        let has = |name: &core::ffi::CStr| {
            let mut v: u32 = 0;
            let mut n = core::mem::size_of::<u32>();
            // SAFETY: v holds n bytes; sysctlbyname writes at most n.
            let r = unsafe { sysctlbyname(name.as_ptr(), &mut v as *mut u32 as *mut core::ffi::c_void, &mut n, core::ptr::null(), 0) };
            r == 0 && v != 0
        };
        (cfg!(target_feature = "i8mm") || has(c"hw.optional.arm.FEAT_I8MM"), cfg!(target_feature = "dotprod") || has(c"hw.optional.arm.FEAT_DotProd"))
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    {
        (cfg!(target_feature = "i8mm"), cfg!(target_feature = "dotprod"))
    }
}

/// Integer weights (n x k, PyTorch Linear layout), packed as described above.
pub struct Quant {
    pub n: usize,
    pub k: usize,
    group: usize,
    int4: bool,
    lay: Lay,
    w: Vec<u8>,   // ceil(n / 8) panels of k * 8 (int8) or k * 4 (int4) bytes
    s: Vec<f32>, // ceil(n / 8) panels x k / group groups x 8 scales
}

/// The weight layout, chosen at load for the path that multiplies it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Lay {
    /// Panels of 8 outputs in pairs (`smmla`, plain NEON, wasm, portable).
    Pair,
    /// Panels of 8 outputs, 4 outputs x 4 k per vector (`sdot` by element).
    Dot4,
}

impl Lay {
    /// Outputs per panel.
    fn width(self) -> usize {
        8
    }
}

/// Where weight (output o of a panel, k index kk) is in the panel: (byte, in the high nibble) - int4: nibbles.
#[inline(always)]
fn place(lay: Lay, int4: bool, o: usize, kk: usize) -> (usize, bool) {
    match (lay, int4) {
        (Lay::Pair, false) => ((o / 2) * 16 + (o % 2) * 8 + (kk / 8) * 64 + kk % 8, false),
        (Lay::Pair, true) => ((o / 2) * 16 + (o % 2) * 8 + (kk / 16) * 64 + kk % 8, (kk / 8) % 2 == 1),
        (Lay::Dot4, false) => ((kk / 4) * 32 + (o / 4) * 16 + (o % 4) * 4 + kk % 4, false),
        (Lay::Dot4, true) => ((kk / 8) * 32 + (o / 4) * 16 + (o % 4) * 4 + kk % 4, (kk / 4) % 2 == 1),
    }
}

/// Activations quantized to int8 per row and per group, rows interleaved in pairs (see above).
struct Act {
    q: Vec<i8>,   // ceil(m / 2) pairs x 2k
    s: Vec<f32>, // pairs x k / group x 4: the scale of each lane of a 2 x 2 block (the row's part of it)
}

impl Quant {
    /// From codes (int8 values, or int4 in -8..=7) and the scale of each (row, group); `group` divides k and is a
    /// multiple of 16.
    pub fn new(n: usize, k: usize, group: usize, int4: bool, code: impl Fn(usize, usize) -> i8, scale: impl Fn(usize, usize) -> f32) -> Quant {
        let lay = if path() == Path::Dot { Lay::Dot4 } else { Lay::Pair };
        Quant::with_layout(n, k, group, int4, lay, code, scale)
    }

    /// `lay`: the layout of the path that will multiply it.
    fn with_layout(n: usize, k: usize, group: usize, int4: bool, lay: Lay, code: impl Fn(usize, usize) -> i8, scale: impl Fn(usize, usize) -> f32) -> Quant {
        assert!(n > 0 && group > 0 && group.is_multiple_of(16) && k.is_multiple_of(group));
        let wd = lay.width();
        let pb = k * wd / if int4 { 2 } else { 1 };
        let ng = k / group;
        let np = n.div_ceil(wd);
        let mut w = vec![0u8; np * pb];
        let mut s = vec![0f32; np * ng * wd];
        for j in 0..n {
            let (jp, o) = (j / wd, j % wd);
            for kk in 0..k {
                let v = code(j, kk);
                let (b, hi) = place(lay, int4, o, kk);
                if int4 {
                    debug_assert!((-8..=7).contains(&v));
                    let nib = v as u8 & 15;
                    w[jp * pb + b] |= if hi { nib << 4 } else { nib };
                } else {
                    w[jp * pb + b] = v as u8;
                }
            }
            for c in 0..ng {
                s[(jp * ng + c) * wd + o] = scale(j, c) / if int4 { 16.0 } else { 1.0 };
            }
        }
        Quant { n, k, group, int4, lay, w, s }
    }

    fn panel_bytes(&self) -> usize {
        self.k * self.lay.width() / if self.int4 { 2 } else { 1 }
    }

    /// Weight (j, kk) as the kernels see it (int4: times 16) and its scale.
    fn at(&self, j: usize, kk: usize) -> (i8, f32) {
        let wd = self.lay.width();
        let (jp, o) = (j / wd, j % wd);
        let (b, hi) = place(self.lay, self.int4, o, kk);
        let b = self.w[jp * self.panel_bytes() + b];
        let v = if self.int4 { (if hi { b & 0xf0 } else { b << 4 }) as i8 } else { b as i8 };
        (v, self.s[(jp * (self.k / self.group) + kk / self.group) * wd + o])
    }

    /// out (m x n) = a (m x k) * W^T (+ bias).
    fn gemm(&self, a: &[f32], m: usize, bias: Option<&[f32]>) -> Vec<f32> {
        self.gemm_as(a, m, bias, false)
    }

    /// `gemm` with the activations at 16-bit precision, for layers where int8 activations would cost accuracy: every
    /// row x becomes two int8 rows, x ~ (255 hi + lo) * max / 32512 (`quant16`), multiplied by the same integer kernels
    /// (twice the work of `gemm`), and the two products are added. Integer paths only.
    fn gemm_wide(&self, a: &[f32], m: usize, bias: Option<&[f32]>) -> Vec<f32> {
        let n = self.n;
        let two = self.gemm_as(a, 2 * m, None, true);
        let mut out = crate::pool::any(m * n); // every value is written
        for (r, o) in out.chunks_exact_mut(n).enumerate() {
            let (hi, lo) = (&two[2 * r * n..(2 * r + 1) * n], &two[(2 * r + 1) * n..(2 * r + 2) * n]);
            match bias {
                Some(b) => o.iter_mut().zip(hi).zip(lo).zip(b).for_each(|(((o, h), l), b)| *o = h + l + b),
                None => o.iter_mut().zip(hi).zip(lo).for_each(|((o, h), l)| *o = h + l),
            }
        }
        crate::pool::give(two);
        out
    }

    /// `gemm` of m rows; `wide`: they are the m / 2 rows of `a` as (hi, lo) pairs of int8 rows (`gemm_wide`).
    fn gemm_as(&self, a: &[f32], m: usize, bias: Option<&[f32]>, wide: bool) -> Vec<f32> {
        let per = if wide { 2 } else { 1 }; // rows of the product per row of `a`
        assert!(m.is_multiple_of(per) && a.len() >= m / per * self.k);
        let a = &a[..m / per * self.k];
        let path = path();
        assert!(!wide || !matches!(path, Path::F32 | Path::X86), "16-bit activations: integer kernels only");
        if !matches!(path, Path::F32 | Path::X86) && self.lay == Lay::Dot4 {
            return self.gemm4(a, m, bias, wide);
        }
        let mut out = crate::pool::any(m * self.n); // every value is written (blocks of rows, or groups of columns)
        if path == Path::F32 {
            #[cfg(target_arch = "x86_64")]
            if m <= SMALL && x86() >= X86::Avx2 {
                self.gemm_small(a, m, bias, &mut out);
                return out;
            }
            blocked(a, self, bias, &mut out);
            return out;
        }
        #[cfg(target_arch = "x86_64")]
        if path == Path::X86 {
            self.gemm_x86(a, m, bias, &mut out);
            return out;
        }
        // the pair layout's own kernels (its layout follows the path; the others meet it only in tests)
        #[cfg(target_arch = "aarch64")]
        let path = match path {
            Path::Dot if has_i8mm() => Path::Mmla,
            Path::Dot => Path::Neon,
            p => p,
        };
        let (n, k, g) = (self.n, self.k, self.group);
        let pairs = m.div_ceil(2);
        let np = n.div_ceil(8);
        let zero = vec![0f32; if bias.is_some() { 0 } else { np * 8 }];
        let bias = bias.unwrap_or(&zero);
        let work = m * n * k;
        const MBP: usize = 16; // row pairs per cache block (32 rows: 16 KiB of activations at k = 512)
        let mbp = crate::par::block_rows(2 * pairs, 2 * MBP) / 2;
        let threads = crate::par::threads();
        if threads > 1 && pairs.div_ceil(mbp) < 2 * threads && np >= 2 * threads && work >= crate::par::MIN_WORK {
            // few rows (streaming chunks, decoding): groups of output panels instead, one task each, into their own
            // columns
            let act = if wide { pack_act16(a, m / 2, k, g) } else { pack_act(a, m, k, g) };
            let pg = np.div_ceil(4 * threads);
            let parts = crate::par::map(np.div_ceil(pg), work, |grp| {
                let (jp0, jp1) = (grp * pg, (grp * pg + pg).min(np));
                let w = (jp1 - jp0) * 8;
                let mut o = vec![0f32; m * w];
                self.block(path, &act, 0, pairs, m, jp0, jp1, bias, &mut o, 0, jp0 * 8, w);
                o
            });
            for (grp, o) in parts.iter().enumerate() {
                let j0 = grp * pg * 8;
                let w = o.len() / m;
                let c = w.min(n - j0);
                for r in 0..m {
                    out[r * n + j0..r * n + j0 + c].copy_from_slice(&o[r * w..r * w + c]);
                }
            }
            return out;
        }
        // blocks of row pairs are independent: one task each, which quantizes its own rows
        crate::par::chunks_mut(&mut out, 2 * mbp * n, work, |blk, out| {
            let (p0, r0) = (blk * mbp, 2 * blk * mbp); // first pair, first row of the block
            let rows = out.len() / n;
            let act = if wide { pack_act16(&a[r0 / 2 * k..(r0 + rows) / 2 * k], rows / 2, k, g) } else { pack_act(&a[r0 * k..(r0 + rows) * k], rows, k, g) };
            self.block(path, &act, p0, (p0 + mbp).min(pairs), m, 0, np, bias, out, r0, 0, n);
        });
        out
    }

    /// The f32 product for few rows, the weights widened in registers (x86-64 AVX2 / AVX-512): groups of output panels,
    /// one task each, into their own columns. The same bits as `blocked`.
    #[cfg(target_arch = "x86_64")]
    fn gemm_small(&self, a: &[f32], m: usize, bias: Option<&[f32]>, out: &mut [f32]) {
        let (n, k) = (self.n, self.k);
        let np = n.div_ceil(8);
        // bias per output, padded to whole pairs of panels
        let mut init = vec![0f32; np.div_ceil(2) * 16];
        if let Some(b) = bias {
            init[..n].copy_from_slice(&b[..n]);
        }
        const PG: usize = 8; // panels per task (even)
        let lvl = x86();
        let cols = crate::par::map(np.div_ceil(PG), m * n * k, |grp| {
            let (jp0, jp1) = (grp * PG, (grp * PG + PG).min(np));
            let w = (jp1 - jp0).div_ceil(2) * 16;
            let mut o = vec![0f32; m * w];
            #[allow(unsafe_code)]
            // SAFETY: the level was detected on this CPU (`x86`).
            unsafe {
                if lvl == X86::Avx512 {
                    x86::quant_avx512(self, a, m, &init, jp0, jp1, &mut o, w)
                } else {
                    x86::quant_avx2(self, a, m, &init, jp0, jp1, &mut o, w)
                }
            }
            o
        });
        for (grp, o) in cols.iter().enumerate() {
            let j0 = grp * PG * 8;
            let w = o.len() / m;
            let c = w.min(n - j0);
            for r in 0..m {
                out[r * n + j0..r * n + j0 + c].copy_from_slice(&o[r * w..r * w + c]);
            }
        }
    }

    /// The integer product on x86-64 (`IntX86`): blocks of rows (each quantizes its own), or for few rows groups of
    /// output panels, one task each. The results of `tile_scalar` (exact integer sums, then the same float steps).
    #[cfg(target_arch = "x86_64")]
    fn gemm_x86(&self, a: &[f32], m: usize, bias: Option<&[f32]>, out: &mut [f32]) {
        let (n, k, g) = (self.n, self.k, self.group);
        let np = n.div_ceil(8);
        let mut init = vec![0f32; np * 8];
        if let Some(b) = bias {
            init[..n].copy_from_slice(&b[..n]);
        }
        let work = m * n * k;
        let kern = int_x86();
        const MBX: usize = 32; // rows per cache block
        let mb = crate::par::block_rows(m, MBX);
        let threads = crate::par::threads();
        if threads > 1 && m.div_ceil(mb) < 2 * threads && np >= 2 * threads && work >= crate::par::MIN_WORK {
            let act = ActX::new(a, m, k, g);
            let pg = np.div_ceil(4 * threads);
            assert!(out.len() >= m * n);
            // every task writes its own columns of out
            struct Out(*mut f32);
            // SAFETY: the tasks write disjoint columns, and out outlives them (`map` returns when all are done).
            #[allow(unsafe_code)]
            unsafe impl Sync for Out {}
            let o = Out(out.as_mut_ptr());
            crate::par::map(np.div_ceil(pg), work, |grp| {
                let (jp0, jp1) = (grp * pg, (grp * pg + pg).min(np));
                let o = &o;
                // SAFETY: columns jp0 * 8.. (up to n) of the m rows of out (asserted), this task's alone.
                #[allow(unsafe_code)]
                unsafe { x86::int_raw(kern, self, &act, &init, jp0, jp1, ((jp1 - jp0) * 8).min(n - jp0 * 8), o.0.add(jp0 * 8), n) }
            });
            return;
        }
        crate::par::chunks_mut(out, mb * n, work, |blk, out| {
            let (r0, rows) = (blk * mb, out.len() / n);
            let act = ActX::new(&a[r0 * k..(r0 + rows) * k], rows, k, g);
            x86::int(kern, self, &act, &init, 0, np, n, out, n);
        });
    }

    /// Row pairs p0..p1 x output panels jp0..jp1 into `out` (row r, column j at (r - r0) * stride + j - c0). `act`
    /// holds the rows from pair p0 on; `m`: all rows of the product.
    #[allow(clippy::too_many_arguments)]
    fn block(&self, path: Path, act: &Act, p0: usize, p1: usize, m: usize, jp0: usize, jp1: usize, bias: &[f32], out: &mut [f32], r0: usize, c0: usize, stride: usize) {
        let (n, k, g) = (self.n, self.k, self.group);
        let ng = k / g;
        let pb = self.panel_bytes();
        for jp in jp0..jp1 {
            let w = &self.w[jp * pb..(jp + 1) * pb];
            let sw = &self.s[jp * ng * 8..(jp + 1) * ng * 8];
            let j0 = jp * 8;
            #[cfg(target_arch = "aarch64")]
            #[allow(unsafe_code)]
            if path == Path::Mmla {
                // a whole panel of whole row pairs: sums + bias straight into `out`; an odd last row or the last,
                // narrower panel through `part`
                let pairs = p1 - p0;
                let (rows, cols) = ((2 * p1).min(m) - 2 * p0, 8.min(n - j0));
                let direct = rows == 2 * pairs && cols == 8;
                let mut b8 = [0f32; 8];
                b8[..cols].copy_from_slice(&bias[j0..j0 + cols]);
                let mut part = if direct { Vec::new() } else { vec![0f32; pairs * 16] };
                let o0 = (2 * p0 - r0) * stride + j0 - c0;
                let (op, st) = if direct { (out[o0..].as_mut_ptr(), stride) } else { (part.as_mut_ptr(), 8) };
                assert!(!direct || pairs == 0 || o0 + (rows - 1) * stride + 8 <= out.len());
                assert!(act.q.len() >= pairs * 2 * k && act.s.len() >= pairs * ng * 4 && w.len() >= k * if self.int4 { 4 } else { 8 } && sw.len() >= ng * 8);
                assert!(g.is_multiple_of(16) && k.is_multiple_of(g));
                // SAFETY: the path was picked for this CPU's features (`path`); `pairs` row pairs of 2k codes and of
                // ng x 4 scales in act, the panel's weights and scales, 8 bias values and 2 * pairs rows of 8 outputs
                // `st` apart from op (asserted).
                unsafe {
                    if self.int4 {
                        neon::panel_mm::<true>(act.q.as_ptr(), act.s.as_ptr(), pairs, w.as_ptr(), sw.as_ptr(), k, ng, g, b8.as_ptr(), op, st)
                    } else {
                        neon::panel_mm::<false>(act.q.as_ptr(), act.s.as_ptr(), pairs, w.as_ptr(), sw.as_ptr(), k, ng, g, b8.as_ptr(), op, st)
                    }
                }
                for (r, p) in part.chunks_exact(8).take(rows).enumerate() {
                    out[o0 + r * stride..o0 + r * stride + cols].copy_from_slice(&p[..cols]);
                }
                continue;
            }
            let mut p = p0;
            while p < p1 {
                let rp = if path == Path::Mmla && p1 - p >= 4 { 4 } else if p1 - p >= 2 { 2 } else { 1 };
                let (a, sa) = (&act.q[(p - p0) * 2 * k..(p - p0 + rp) * 2 * k], &act.s[(p - p0) * ng * 4..(p - p0 + rp) * ng * 4]);
                let mut t = [[[0f32; 4]; 4]; RPMAX];
                tile(path, rp, self.int4, a, sa, w, sw, k, g, &mut t);
                for (r, t) in t.iter().enumerate().take(rp) {
                    let rr = 2 * (p + r);
                    if rr + 2 <= m && j0 + 8 <= n {
                        // whole block: row rr is lanes 0, 1 of every pair, row rr + 1 lanes 2, 3
                        let b = &bias[j0..j0 + 8];
                        #[cfg(target_arch = "aarch64")]
                        #[allow(unsafe_code)]
                        {
                            use core::arch::aarch64::*;
                            let o0 = (rr - r0) * stride + j0 - c0;
                            assert!(o0 + stride + 8 <= out.len());
                            // SAFETY: NEON is mandatory on aarch64; t[pp] holds 4 floats, b 8, and rows rr, rr + 1 of
                            // out 8 each from o0 (asserted). The same sums as below.
                            unsafe {
                                let tv: [float32x4_t; 4] = core::array::from_fn(|pp| vld1q_f32(t[pp].as_ptr()));
                                let (b0, b1) = (vld1q_f32(b.as_ptr()), vld1q_f32(b.as_ptr().add(4)));
                                let op = out.as_mut_ptr().add(o0);
                                vst1q_f32(op, vaddq_f32(vcombine_f32(vget_low_f32(tv[0]), vget_low_f32(tv[1])), b0));
                                vst1q_f32(op.add(4), vaddq_f32(vcombine_f32(vget_low_f32(tv[2]), vget_low_f32(tv[3])), b1));
                                vst1q_f32(op.add(stride), vaddq_f32(vcombine_f32(vget_high_f32(tv[0]), vget_high_f32(tv[1])), b0));
                                vst1q_f32(op.add(stride + 4), vaddq_f32(vcombine_f32(vget_high_f32(tv[2]), vget_high_f32(tv[3])), b1));
                            }
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        for h in 0..2 {
                            let o0 = (rr + h - r0) * stride + j0 - c0;
                            let o = &mut out[o0..o0 + 8];
                            for pp in 0..4 {
                                o[2 * pp] = t[pp][2 * h] + b[2 * pp];
                                o[2 * pp + 1] = t[pp][2 * h + 1] + b[2 * pp + 1];
                            }
                        }
                        continue;
                    }
                    for (pp, t) in t.iter().enumerate() {
                        for (lane, &(ri, oi)) in MMLA_LANES.iter().enumerate() {
                            let (row, o) = (rr + ri, j0 + 2 * pp + oi);
                            if row < m && o < n {
                                out[(row - r0) * stride + o - c0] = t[lane] + bias[o];
                            }
                        }
                    }
                }
                p += rp;
            }
        }
    }
}

impl Quant {
    /// `gemm` for the `sdot` layout: blocks of rows (or, few rows, groups of output panels), one task each.
    fn gemm4(&self, a: &[f32], m: usize, bias: Option<&[f32]>, wide: bool) -> Vec<f32> {
        let (n, k, g) = (self.n, self.k, self.group);
        let np = n.div_ceil(8);
        let zero = vec![0f32; if bias.is_some() { 0 } else { np * 8 }];
        let bias = bias.unwrap_or(&zero);
        let mut out = crate::pool::any(m * n); // every value is written
        let work = m * n * k;
        let mb = crate::par::block_rows(m, 32);
        let threads = crate::par::threads();
        if threads > 1 && m.div_ceil(mb) < 2 * threads && np >= 2 * threads && work >= crate::par::MIN_WORK {
            let (q, sa) = if wide { pack_rows16(a, m / 2, k, g) } else { pack_rows(a, m, k, g) };
            let pg = np.div_ceil(4 * threads);
            let parts = crate::par::map(np.div_ceil(pg), work, |grp| {
                let (jp0, jp1) = (grp * pg, (grp * pg + pg).min(np));
                let w = (jp1 - jp0) * 8;
                let mut o = vec![0f32; m * w];
                self.block4(&q, &sa, m, jp0, jp1, bias, &mut o, jp0 * 8, w);
                o
            });
            for (grp, o) in parts.iter().enumerate() {
                let j0 = grp * pg * 8;
                let w = o.len() / m;
                let c = w.min(n - j0);
                for r in 0..m {
                    out[r * n + j0..r * n + j0 + c].copy_from_slice(&o[r * w..r * w + c]);
                }
            }
            return out;
        }
        crate::par::chunks_mut(&mut out, mb * n, work, |blk, out| {
            let (r0, rows) = (blk * mb, out.len() / n);
            let (q, sa) = if wide { pack_rows16(&a[r0 / 2 * k..(r0 + rows) / 2 * k], rows / 2, k, g) } else { pack_rows(&a[r0 * k..(r0 + rows) * k], rows, k, g) };
            self.block4(&q, &sa, rows, 0, np, bias, out, 0, n);
        });
        out
    }

    /// `rows` activation rows (`q`, `sa`) x output panels jp0..jp1 into `out` (row r, column j at r * stride + j - c0).
    #[allow(clippy::too_many_arguments)]
    fn block4(&self, q: &[i8], sa: &[f32], rows: usize, jp0: usize, jp1: usize, bias: &[f32], out: &mut [f32], c0: usize, stride: usize) {
        let (n, k, g) = (self.n, self.k, self.group);
        let ng = k / g;
        let pb = self.panel_bytes();
        assert!(q.len() >= rows * k && sa.len() >= rows * ng && g.is_multiple_of(16));
        for jp in jp0..jp1 {
            let w = &self.w[jp * pb..(jp + 1) * pb];
            let sw = &self.s[jp * ng * 8..(jp + 1) * ng * 8];
            let j0 = jp * 8;
            let cols = 8.min(n - j0);
            #[cfg(target_arch = "aarch64")]
            #[allow(unsafe_code)]
            {
                // a whole panel: every row's 8 sums + bias straight into `out`; the last, narrower one through `part`
                let mut b8 = [0f32; 8];
                b8[..cols].copy_from_slice(&bias[j0..j0 + cols]);
                let mut part = if cols < 8 { vec![0f32; rows * 8] } else { Vec::new() };
                let (op, st) = if cols < 8 { (part.as_mut_ptr(), 8) } else { (out[j0 - c0..].as_mut_ptr(), stride) };
                assert!(cols < 8 || rows == 0 || (rows - 1) * stride + j0 - c0 + 8 <= out.len());
                assert!(w.len() >= k * if self.int4 { 4 } else { 8 } && sw.len() >= ng * 8 && k.is_multiple_of(g));
                // SAFETY: the `sdot` layout is built only on the dotprod path (`Quant::new`); `rows` rows of k codes in
                // q and of ng scales in sa (asserted above the loop), the panel's weights and scales, 8 bias values and
                // rows of 8 outputs `st` apart from op (asserted here).
                unsafe {
                    if self.int4 {
                        neon::panel4::<true>(q.as_ptr(), sa.as_ptr(), rows, w.as_ptr(), sw.as_ptr(), k, ng, g, b8.as_ptr(), op, st)
                    } else {
                        neon::panel4::<false>(q.as_ptr(), sa.as_ptr(), rows, w.as_ptr(), sw.as_ptr(), k, ng, g, b8.as_ptr(), op, st)
                    }
                }
                for (r, p) in part.chunks_exact(8).enumerate() {
                    out[r * stride + j0 - c0..r * stride + j0 - c0 + cols].copy_from_slice(&p[..cols]);
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut r = 0;
                while r < rows {
                    let rr = 4.min(rows - r);
                    let mut t = [[0f32; 8]; 4];
                    tile4(rr, self.int4, &q[r * k..(r + rr) * k], &sa[r * ng..(r + rr) * ng], w, sw, k, g, &mut t);
                    for (i, t) in t.iter().enumerate().take(rr) {
                        let o = (r + i) * stride + j0 - c0;
                        for l in 0..cols {
                            out[o + l] = t[l] + bias[j0 + l];
                        }
                    }
                    r += rr;
                }
            }
        }
    }
}

/// `rr` (1..=4) activation rows x one panel (8 outputs) in the `sdot` layout -> t[row][output], portable (the layout
/// is built only for the aarch64 `sdot` path, which has its own loop: `neon::panel4`; this keeps other targets compiling).
#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn tile4(rr: usize, int4: bool, a: &[i8], sa: &[f32], w: &[u8], sw: &[f32], k: usize, g: usize, t: &mut [[f32; 8]; 4]) {
    let ng = k / g;
    assert!((1..=4).contains(&rr) && a.len() >= rr * k && sa.len() >= rr * ng);
    assert!(w.len() >= k * if int4 { 4 } else { 8 } && sw.len() >= ng * 8 && g.is_multiple_of(16) && k.is_multiple_of(g));
    for r in 0..rr {
        for o in 0..8 {
            let mut f = 0f32;
            for c in 0..ng {
                let mut d = 0i32;
                for kk in c * g..(c + 1) * g {
                    let (b, hi) = place(Lay::Dot4, int4, o, kk);
                    let v = if int4 { (if hi { w[b] & 0xf0 } else { w[b] << 4 }) as i8 } else { w[b] as i8 };
                    d += a[r * k + kk] as i32 * v as i32;
                }
                f += d as f32 * (sw[c * 8 + o] * sa[r * ng + c]);
            }
            t[r][o] = f;
        }
    }
}

/// Rows of `a` (k each) -> int8 per `g` values, interleaved in row pairs; lane scales in the order of the kernel's
/// 2 x 2 blocks (MMLA_LANES).
fn pack_act(a: &[f32], m: usize, k: usize, g: usize) -> Act {
    let pairs = m.div_ceil(2);
    let ng = k / g;
    let mut q = vec![0i8; pairs * 2 * k];
    let mut s = vec![0f32; pairs * ng * 4];
    for r in 0..m {
        let (p, h) = (r / 2, r % 2);
        let dst = &mut q[p * 2 * k..(p + 1) * 2 * k];
        for c in 0..ng {
            let x = &a[r * k + c * g..r * k + (c + 1) * g];
            let max = absmax(x);
            let inv = if max > 0.0 { 127.0 / max } else { 0.0 };
            for st in 0..g / 8 {
                let kk = c * g + st * 8;
                let o = (kk / 8) * 16 + h * 8;
                quant8(&x[st * 8..st * 8 + 8], inv, &mut dst[o..o + 8]);
            }
            let sv = &mut s[(p * ng + c) * 4..(p * ng + c + 1) * 4];
            for (l, &(ri, _)) in MMLA_LANES.iter().enumerate() {
                if ri == h {
                    sv[l] = max / 127.0;
                }
            }
        }
    }
    Act { q, s }
}

/// Rows of `a` (k each) -> plain int8 rows (m x k) and one scale per row and group (m x k / g): the `sdot` path's
/// activations, quantized exactly as `pack_act` does.
fn pack_rows(a: &[f32], m: usize, k: usize, g: usize) -> (Vec<i8>, Vec<f32>) {
    let ng = k / g;
    let mut q = vec![0i8; m * k];
    let mut s = vec![0f32; m * ng];
    for r in 0..m {
        for c in 0..ng {
            let x = &a[r * k + c * g..r * k + (c + 1) * g];
            let max = absmax(x);
            let inv = if max > 0.0 { 127.0 / max } else { 0.0 };
            for st in 0..g / 8 {
                let o = r * k + c * g + st * 8;
                quant8(&x[st * 8..st * 8 + 8], inv, &mut q[o..o + 8]);
            }
            s[r * ng + c] = max / 127.0;
        }
    }
    (q, s)
}

/// `pack_act` at 16-bit precision: row r of `a` becomes the pair (hi, lo) of int8 rows (`quant16`), their lane scales
/// 255 s and s with s = max / 32512.
fn pack_act16(a: &[f32], m: usize, k: usize, g: usize) -> Act {
    let ng = k / g;
    let mut q = vec![0i8; m * 2 * k];
    let mut s = vec![0f32; m * ng * 4];
    for r in 0..m {
        let dst = &mut q[r * 2 * k..(r + 1) * 2 * k];
        for c in 0..ng {
            let x = &a[r * k + c * g..r * k + (c + 1) * g];
            let max = absmax(x);
            let inv = if max > 0.0 { W16 / max } else { 0.0 };
            for st in 0..g / 8 {
                let o = (c * g / 8 + st) * 16;
                let (hi, lo) = dst[o..o + 16].split_at_mut(8);
                quant16(&x[st * 8..st * 8 + 8], inv, hi, lo);
            }
            let sv = &mut s[(r * ng + c) * 4..(r * ng + c + 1) * 4];
            for (l, &(ri, _)) in MMLA_LANES.iter().enumerate() {
                sv[l] = max / W16 * if ri == 0 { 255.0 } else { 1.0 };
            }
        }
    }
    Act { q, s }
}

/// `pack_rows` at 16-bit precision: row r of `a` becomes rows 2r (hi) and 2r + 1 (lo), as `pack_act16`.
fn pack_rows16(a: &[f32], m: usize, k: usize, g: usize) -> (Vec<i8>, Vec<f32>) {
    let ng = k / g;
    let mut q = vec![0i8; 2 * m * k];
    let mut s = vec![0f32; 2 * m * ng];
    for r in 0..m {
        let (hi, lo) = q[2 * r * k..(2 * r + 2) * k].split_at_mut(k);
        for c in 0..ng {
            let x = &a[r * k + c * g..r * k + (c + 1) * g];
            let max = absmax(x);
            let inv = if max > 0.0 { W16 / max } else { 0.0 };
            for st in 0..g / 8 {
                let o = c * g + st * 8;
                quant16(&x[st * 8..st * 8 + 8], inv, &mut hi[o..o + 8], &mut lo[o..o + 8]);
            }
            s[2 * r * ng + c] = 255.0 * max / W16;
            s[(2 * r + 1) * ng + c] = max / W16;
        }
    }
    (q, s)
}

/// The largest 16-bit activation code: 127 * 255 + 127 (two digits in -127..=127, base 255).
const W16: f32 = 32512.0;

/// 8 values times `inv` (at most 32512 in size) as v ~ 255 hi + lo, both in -127..=127 (so every integer kernel takes
/// them as it takes `quant8`'s): hi = round(v / 255), lo = round(v - 255 hi).
#[inline(always)]
fn quant16(x: &[f32], inv: f32, hi: &mut [i8], lo: &mut [i8]) {
    assert!(x.len() >= 8 && hi.len() >= 8 && lo.len() >= 8);
    #[cfg(target_arch = "aarch64")]
    #[allow(unsafe_code)]
    // SAFETY: NEON is mandatory on aarch64; x, hi and lo hold 8 values (asserted).
    unsafe {
        use core::arch::aarch64::*;
        let (iv, b, ib, lim) = (vdupq_n_f32(inv), vdupq_n_f32(255.0), vdupq_n_f32(1.0 / 255.0), vdupq_n_s32(127));
        let mut h = [vdupq_n_s32(0); 2];
        let mut l = [vdupq_n_s32(0); 2];
        for i in 0..2 {
            let v = vmulq_f32(vld1q_f32(x.as_ptr().add(4 * i)), iv);
            let hv = vcvtaq_s32_f32(vmulq_f32(v, ib));
            let lv = vcvtaq_s32_f32(vfmsq_f32(v, vcvtq_f32_s32(hv), b));
            h[i] = hv;
            l[i] = vmaxq_s32(vminq_s32(lv, lim), vnegq_s32(lim));
        }
        vst1_s8(hi.as_mut_ptr(), vqmovn_s16(vcombine_s16(vqmovn_s32(h[0]), vqmovn_s32(h[1]))));
        vst1_s8(lo.as_mut_ptr(), vqmovn_s16(vcombine_s16(vqmovn_s32(l[0]), vqmovn_s32(l[1]))));
    }
    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..8 {
        let v = x[i] * inv;
        let h = (v * (1.0 / 255.0)).round();
        hi[i] = h as i8;
        lo[i] = (v - 255.0 * h).round().clamp(-127.0, 127.0) as i8;
    }
}

#[inline(always)]
fn absmax(x: &[f32]) -> f32 {
    // 16 independent lanes so the reduction vectorizes (max is exact in any order); `v > m` skips NaN as `max` does,
    // and is one instruction
    let mut m = [0f32; 16];
    let mut it = x.chunks_exact(16);
    for c in &mut it {
        for l in 0..16 {
            let v = c[l].abs();
            m[l] = if v > m[l] { v } else { m[l] };
        }
    }
    for c in it.remainder().chunks_exact(8) {
        for l in 0..8 {
            let v = c[l].abs();
            m[l] = if v > m[l] { v } else { m[l] };
        }
    }
    m.iter().fold(0f32, |a, &b| a.max(b))
}

/// 8 values times `inv`, rounded half away from zero, to int8.
#[inline(always)]
fn quant8(x: &[f32], inv: f32, out: &mut [i8]) {
    #[cfg(target_arch = "aarch64")]
    #[allow(unsafe_code)]
    // SAFETY: NEON is mandatory on aarch64; x and out hold 8 values (asserted).
    unsafe {
        use core::arch::aarch64::*;
        assert!(x.len() >= 8 && out.len() >= 8);
        let v = vdupq_n_f32(inv);
        let a = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x.as_ptr()), v));
        let b = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(x.as_ptr().add(4)), v));
        vst1_s8(out.as_mut_ptr(), vqmovn_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b))));
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (v, o) in x[..8].iter().zip(out[..8].iter_mut()) {
        let r = v * inv;
        *o = (r + if r >= 0.0 { 0.5 } else { -0.5 }) as i8; // round half away from zero
    }
}

/// One kernel call: rp (1, 2) row pairs x the 4 output pairs of one panel, 2 x 2 blocks of f32 into t[row pair][pair].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn tile(path: Path, rp: usize, int4: bool, a: &[i8], sa: &[f32], w: &[u8], sw: &[f32], k: usize, g: usize, t: &mut [[[f32; 4]; 4]; RPMAX]) {
    let ng = k / g;
    assert!((1..=RPMAX).contains(&rp) && (rp <= 2 || path == Path::Mmla) && a.len() >= rp * 2 * k && sa.len() >= rp * ng * 4);
    assert!(w.len() >= k * if int4 { 4 } else { 8 } && sw.len() >= ng * 8 && g.is_multiple_of(16) && k.is_multiple_of(g));
    match path {
        #[cfg(target_arch = "aarch64")]
        Path::Mmla | Path::Neon => {
            #[allow(unsafe_code)]
            // SAFETY: the path was picked for this CPU's features (`path`); every read stays inside the slices asserted
            // above (row pair r: a[r*2k + 0..2k], sa[r*ng*4 + 0..ng*4]; the panel: w[0..k*8 or k*4], sw[0..ng*8]).
            unsafe {
                macro_rules! go {
                    ($f:ident) => {
                        match (rp, int4) {
                            (2, true) => neon::$f::<2, true>(a.as_ptr(), sa.as_ptr(), w.as_ptr(), sw.as_ptr(), k, g, t),
                            (2, false) => neon::$f::<2, false>(a.as_ptr(), sa.as_ptr(), w.as_ptr(), sw.as_ptr(), k, g, t),
                            (_, true) => neon::$f::<1, true>(a.as_ptr(), sa.as_ptr(), w.as_ptr(), sw.as_ptr(), k, g, t),
                            (_, false) => neon::$f::<1, false>(a.as_ptr(), sa.as_ptr(), w.as_ptr(), sw.as_ptr(), k, g, t),
                        }
                    };
                }
                match path {
                    Path::Mmla if rp == 4 => {
                        if int4 {
                            neon::mmla::<4, true>(a.as_ptr(), sa.as_ptr(), w.as_ptr(), sw.as_ptr(), k, g, t)
                        } else {
                            neon::mmla::<4, false>(a.as_ptr(), sa.as_ptr(), w.as_ptr(), sw.as_ptr(), k, g, t)
                        }
                    }
                    Path::Mmla => go!(mmla),
                    _ => go!(plain),
                }
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Path::Wasm => {
            #[allow(unsafe_code)]
            // SAFETY: as above.
            unsafe {
                match (rp, int4) {
                    (2, true) => wasm::tile::<2, true>(a.as_ptr(), sa.as_ptr(), w.as_ptr(), sw.as_ptr(), k, g, t),
                    (2, false) => wasm::tile::<2, false>(a.as_ptr(), sa.as_ptr(), w.as_ptr(), sw.as_ptr(), k, g, t),
                    (_, true) => wasm::tile::<1, true>(a.as_ptr(), sa.as_ptr(), w.as_ptr(), sw.as_ptr(), k, g, t),
                    (_, false) => wasm::tile::<1, false>(a.as_ptr(), sa.as_ptr(), w.as_ptr(), sw.as_ptr(), k, g, t),
                }
            }
        }
        _ => tile_scalar(rp, int4, a, sa, w, sw, k, g, t),
    }
}

/// The integer product in plain Rust, same layout and lane order (MMLA_LANES).
#[allow(clippy::too_many_arguments)]
fn tile_scalar(rp: usize, int4: bool, a: &[i8], sa: &[f32], w: &[u8], sw: &[f32], k: usize, g: usize, t: &mut [[[f32; 4]; 4]; RPMAX]) {
    let ng = k / g;
    let wv = |o: usize, kk: usize| -> i32 {
        let base = (o / 2) * 16 + (o % 2) * 8 + kk % 8;
        if int4 {
            let b = w[base + (kk / 16) * 64];
            (if (kk / 8).is_multiple_of(2) { b << 4 } else { b & 0xf0 }) as i8 as i32
        } else {
            w[base + (kk / 8) * 64] as i8 as i32
        }
    };
    for r in 0..rp {
        for pp in 0..4 {
            for (lane, &(ri, oi)) in MMLA_LANES.iter().enumerate() {
                let o = 2 * pp + oi;
                let mut f = 0f32;
                for c in 0..ng {
                    let mut d = 0i32;
                    for kk in c * g..(c + 1) * g {
                        d += a[r * 2 * k + (kk / 8) * 16 + ri * 8 + kk % 8] as i32 * wv(o, kk);
                    }
                    f += d as f32 * (sa[(r * ng + c) * 4 + lane] * sw[c * 8 + o]);
                }
                t[r][pp][lane] = f;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, clippy::needless_range_loop)]
mod neon {
    use core::arch::aarch64::*;

    /// [scale of output 2p, 2p + 1] twice: the weight half of a 2 x 2 block's lane scales (both lane orders).
    #[inline(always)]
    unsafe fn wscale(sw: *const f32) -> float32x4_t {
        // SAFETY: the caller's sw points to 2 floats.
        unsafe {
            let v = vld1_f32(sw);
            vcombine_f32(v, v)
        }
    }

    /// int4 vector -> (low nibbles, high nibbles), times 16.
    #[inline(always)]
    unsafe fn unpack(v: int8x16_t) -> (int8x16_t, int8x16_t) {
        // SAFETY: register-only NEON (mandatory on aarch64).
        unsafe { (vshlq_n_s8::<4>(v), vandq_s8(v, vdupq_n_s8(0xf0u8 as i8))) }
    }

    #[inline(always)]
    unsafe fn store<const RP: usize>(accf: &[[float32x4_t; 4]; RP], t: &mut [[[f32; 4]; 4]; super::RPMAX]) {
        for r in 0..RP {
            for p in 0..4 {
                // SAFETY: t[r][p] holds 4 floats.
                unsafe { vst1q_f32(t[r][p].as_mut_ptr(), accf[r][p]) };
            }
        }
    }

    macro_rules! smmla {
        ($acc:expr, $a:expr, $w:expr) => {
            // smmla (the i8mm intrinsics are still unstable in core::arch)
            core::arch::asm!("smmla {acc:v}.4s, {a:v}.16b, {w:v}.16b", acc = inout(vreg) $acc, a = in(vreg) $a, w = in(vreg) $w, options(pure, nomem, nostack))
        };
    }
    /// i8mm: one smmla per (row pair, output pair) and 8 k gives the whole 2 x 2 block: RP row pairs x the 4 output
    /// pairs of a panel, each block's lanes in MMLA_LANES order.
    #[inline]
    #[target_feature(enable = "i8mm")]
    unsafe fn mmla_acc<const RP: usize, const INT4: bool>(a: *const i8, sa: *const f32, w: *const u8, sw: *const f32, k: usize, ng: usize, g: usize) -> [[float32x4_t; 4]; RP] {
        // SAFETY: the caller guarantees RP row pairs of 2k activations and ng x 4 scales each, and a panel.
        unsafe {
            let mut accf = [[vdupq_n_f32(0.0); 4]; RP];
            for c in 0..ng {
                let mut acc = [[vdupq_n_s32(0); 4]; RP];
                let mut kk = c * g;
                while kk < (c + 1) * g {
                    if INT4 {
                        let wp = w.add(kk * 4) as *const i8;
                        let wv: [(int8x16_t, int8x16_t); 4] = core::array::from_fn(|p| unpack(vld1q_s8(wp.add(16 * p))));
                        for r in 0..RP {
                            let ap = a.add(r * 2 * k + kk * 2);
                            let (a0, a1) = (vld1q_s8(ap), vld1q_s8(ap.add(16)));
                            for p in 0..4 {
                                smmla!(acc[r][p], a0, wv[p].0);
                                smmla!(acc[r][p], a1, wv[p].1);
                            }
                        }
                        kk += 16;
                    } else {
                        let wp = w.add(kk * 8) as *const i8;
                        let wv: [int8x16_t; 4] = core::array::from_fn(|p| vld1q_s8(wp.add(16 * p)));
                        for r in 0..RP {
                            let a0 = vld1q_s8(a.add(r * 2 * k + kk * 2));
                            for p in 0..4 {
                                smmla!(acc[r][p], a0, wv[p]);
                            }
                        }
                        kk += 8;
                    }
                }
                for p in 0..4 {
                    let s = wscale(sw.add(c * 8 + 2 * p));
                    for r in 0..RP {
                        let sc = vmulq_f32(vld1q_f32(sa.add((r * ng + c) * 4)), s);
                        accf[r][p] = vfmaq_f32(accf[r][p], vcvtq_f32_s32(acc[r][p]), sc);
                    }
                }
            }
            accf
        }
    }

    /// `mmla_acc` into t[row pair][output pair].
    #[target_feature(enable = "i8mm")]
    pub unsafe fn mmla<const RP: usize, const INT4: bool>(a: *const i8, sa: *const f32, w: *const u8, sw: *const f32, k: usize, g: usize, t: &mut [[[f32; 4]; 4]; super::RPMAX]) {
        // SAFETY: as `mmla_acc`.
        unsafe { store(&mmla_acc::<RP, INT4>(a, sa, w, sw, k, k / g, g), t) }
    }

    /// i8mm: `pairs` row pairs x one panel of 8 outputs, 4 pairs per `mmla_acc` while they last (then 2, 1): rows 2p and
    /// 2p + 1 of the sums + bias at out + 2p * stride and one row further. The same sums as `mmla`.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "i8mm")]
    pub unsafe fn panel_mm<const INT4: bool>(a: *const i8, sa: *const f32, pairs: usize, w: *const u8, sw: *const f32, k: usize, ng: usize, g: usize, bias: *const f32, out: *mut f32, stride: usize) {
        // SAFETY: the caller's: pairs of 2k codes in a and of ng x 4 scales in sa, a panel, 8 bias values, 2 * pairs
        // rows of 8 outputs.
        unsafe {
            let (b0, b1) = (vld1q_f32(bias), vld1q_f32(bias.add(4)));
            let mut p = 0;
            macro_rules! go {
                ($n:literal) => {{
                    let acc = mmla_acc::<$n, INT4>(a.add(p * 2 * k), sa.add(p * ng * 4), w, sw, k, ng, g);
                    for (r, t) in acc.iter().enumerate() {
                        // row 2p: lanes 0, 1 of every output pair; row 2p + 1: lanes 2, 3
                        let o = out.add(2 * (p + r) * stride);
                        vst1q_f32(o, vaddq_f32(vcombine_f32(vget_low_f32(t[0]), vget_low_f32(t[1])), b0));
                        vst1q_f32(o.add(4), vaddq_f32(vcombine_f32(vget_low_f32(t[2]), vget_low_f32(t[3])), b1));
                        vst1q_f32(o.add(stride), vaddq_f32(vcombine_f32(vget_high_f32(t[0]), vget_high_f32(t[1])), b0));
                        vst1q_f32(o.add(stride + 4), vaddq_f32(vcombine_f32(vget_high_f32(t[2]), vget_high_f32(t[3])), b1));
                    }
                    p += $n;
                }};
            }
            while p < pairs {
                match pairs - p {
                    4.. => go!(4),
                    2..4 => go!(2),
                    _ => go!(1),
                }
            }
        }
    }

    /// dotprod: per (row pair, output pair) and 8 k, sdot of [r0|r1] . [o0|o1] (r0 o0, r1 o1) and of [r1|r0] . [o0|o1]
    /// (r1 o0, r0 o1), two lanes each; added pairwise at the group end: DOT_LANES. Two output pairs at a time (registers).
    macro_rules! sdot_lane {
        ($acc:expr, $w:expr, $a:expr, $l:literal) => {
            // sdot by element (vdotq_laneq_s32 is still unstable in core::arch)
            core::arch::asm!(concat!("sdot {acc:v}.4s, {w:v}.16b, {a:v}.4b[", $l, "]"), acc = inout(vreg) $acc, w = in(vreg) $w, a = in(vreg) $a, options(pure, nomem, nostack))
        };
    }

    /// dotprod, `sdot` layout: `rows` rows x one panel of 8 outputs, 8 rows per `dot4` while they last (then 4, 2, 1):
    /// row r's sums + bias at out + r * stride.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "dotprod")]
    pub unsafe fn panel4<const INT4: bool>(q: *const i8, sa: *const f32, rows: usize, w: *const u8, sw: *const f32, k: usize, ng: usize, g: usize, bias: *const f32, out: *mut f32, stride: usize) {
        let mut r = 0;
        // SAFETY: the caller's: rows of k codes in q and of ng scales in sa, a panel, 8 bias values, rows of 8 outputs.
        unsafe {
            macro_rules! go {
                ($n:literal) => {{
                    dot4::<$n, INT4>(q.add(r * k), sa.add(r * ng), w, sw, k, ng, g, bias, out.add(r * stride), stride);
                    r += $n;
                }};
            }
            while r < rows {
                match rows - r {
                    8.. => go!(8),
                    4..8 => go!(4),
                    2..4 => go!(2),
                    _ => go!(1),
                }
            }
        }
    }

    /// dotprod, `sdot` layout: R (1, 2, 4, 8) rows x 8 outputs. Per 16 k: each row's 16 activations in one register,
    /// used by lane (4 k each) against 2 vectors of 4 outputs x 4 k, two k-steps' weights loaded at a time (R = 8: 16
    /// accumulators, 8 activation rows and 4 weight vectors live); one accumulator per (row, 4 outputs), lane = output.
    /// Each row's sums are the same whatever R is.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[target_feature(enable = "dotprod")]
    unsafe fn dot4<const R: usize, const INT4: bool>(a: *const i8, sa: *const f32, w: *const u8, sw: *const f32, k: usize, ng: usize, g: usize, bias: *const f32, out: *mut f32, stride: usize) {
        // SAFETY: the caller (`panel4`) guarantees R rows of k activations and ng scales each, a panel, 8 bias values
        // and R rows of 8 outputs.
        unsafe {
            let mut accf = [[vdupq_n_f32(0.0); 2]; R];
            for c in 0..ng {
                let mut acc = [[vdupq_n_s32(0); 2]; R];
                let mut kk = c * g;
                while kk < (c + 1) * g {
                    let av: [int8x16_t; R] = core::array::from_fn(|r| vld1q_s8(a.add(r * k + kk)));
                    // two k-steps (4 k each): their weights for outputs 0..4 and 4..8, against lanes $l0, $l1
                    macro_rules! half {
                        ($w0:expr, $w1:expr, $l0:literal, $l1:literal) => {
                            let (w0, w1) = ($w0, $w1);
                            for r in 0..R {
                                sdot_lane!(acc[r][0], w0.0, av[r], $l0);
                                sdot_lane!(acc[r][1], w0.1, av[r], $l0);
                                sdot_lane!(acc[r][0], w1.0, av[r], $l1);
                                sdot_lane!(acc[r][1], w1.1, av[r], $l1);
                            }
                        };
                    }
                    if INT4 {
                        // per 8 k: low nibbles the 4 k at k, high nibbles those at k + 4
                        let wp = w.add(kk * 4) as *const i8;
                        let ((l0, h0), (l1, h1)) = (unpack(vld1q_s8(wp)), unpack(vld1q_s8(wp.add(16))));
                        half!((l0, l1), (h0, h1), "0", "1");
                        let ((l2, h2), (l3, h3)) = (unpack(vld1q_s8(wp.add(32))), unpack(vld1q_s8(wp.add(48))));
                        half!((l2, l3), (h2, h3), "2", "3");
                    } else {
                        let wp = w.add(kk * 8) as *const i8;
                        half!((vld1q_s8(wp), vld1q_s8(wp.add(16))), (vld1q_s8(wp.add(32)), vld1q_s8(wp.add(48))), "0", "1");
                        half!((vld1q_s8(wp.add(64)), vld1q_s8(wp.add(80))), (vld1q_s8(wp.add(96)), vld1q_s8(wp.add(112))), "2", "3");
                    }
                    kk += 16;
                }
                let (s0, s1) = (vld1q_f32(sw.add(c * 8)), vld1q_f32(sw.add(c * 8 + 4)));
                for r in 0..R {
                    let x = *sa.add(r * ng + c);
                    accf[r][0] = vfmaq_f32(accf[r][0], vcvtq_f32_s32(acc[r][0]), vmulq_n_f32(s0, x));
                    accf[r][1] = vfmaq_f32(accf[r][1], vcvtq_f32_s32(acc[r][1]), vmulq_n_f32(s1, x));
                }
            }
            let (b0, b1) = (vld1q_f32(bias), vld1q_f32(bias.add(4)));
            for r in 0..R {
                vst1q_f32(out.add(r * stride), vaddq_f32(accf[r][0], b0));
                vst1q_f32(out.add(r * stride + 4), vaddq_f32(accf[r][1], b1));
            }
        }
    }

    /// Plain ARMv8.0 NEON: per lane of the 2 x 2 block, smull of 8-byte halves into int16 and smlal of the next k-step
    /// into the same int16 (|a| <= 127, |w| <= 128: two products fit), then sadalp into int32; the accumulators are
    /// reduced pairwise at the group end (MMLA_LANES). Two output pairs at a time.
    #[inline(never)]
    pub unsafe fn plain<const RP: usize, const INT4: bool>(a: *const i8, sa: *const f32, w: *const u8, sw: *const f32, k: usize, g: usize, t: &mut [[[f32; 4]; 4]; super::RPMAX]) {
        let ng = k / g;
        /// Two 8-k steps (a0, w0), (a1, w1) of one 2 x 2 block.
        #[inline(always)]
        unsafe fn mac2(acc: &mut [int32x4_t; 4], a0: int8x16_t, w0: int8x16_t, a1: int8x16_t, w1: int8x16_t) {
            // SAFETY: register-only NEON (mandatory on aarch64).
            unsafe {
                let (al, ah, wl, wh) = (vget_low_s8(a0), vget_high_s8(a0), vget_low_s8(w0), vget_high_s8(w0));
                let (bl, bh, vl, vh) = (vget_low_s8(a1), vget_high_s8(a1), vget_low_s8(w1), vget_high_s8(w1));
                acc[0] = vpadalq_s16(acc[0], vmlal_s8(vmull_s8(al, wl), bl, vl));
                acc[1] = vpadalq_s16(acc[1], vmlal_s8(vmull_s8(al, wh), bl, vh));
                acc[2] = vpadalq_s16(acc[2], vmlal_s8(vmull_s8(ah, wl), bh, vl));
                acc[3] = vpadalq_s16(acc[3], vmlal_s8(vmull_s8(ah, wh), bh, vh));
            }
        }
        // SAFETY: as `mmla`.
        unsafe {
            let mut accf = [[vdupq_n_f32(0.0); 4]; RP];
            for half in 0..2 {
                for c in 0..ng {
                    let mut acc = [[[vdupq_n_s32(0); 4]; 2]; RP];
                    let mut kk = c * g;
                    while kk < (c + 1) * g {
                        for r in 0..RP {
                            let ap = a.add(r * 2 * k + kk * 2);
                            let (a0, a1) = (vld1q_s8(ap), vld1q_s8(ap.add(16)));
                            for p in 0..2 {
                                if INT4 {
                                    let (lo, hi) = unpack(vld1q_s8(w.add(kk * 4 + 32 * half + 16 * p) as *const i8));
                                    mac2(&mut acc[r][p], a0, lo, a1, hi);
                                } else {
                                    let wp = w.add(kk * 8 + 32 * half + 16 * p) as *const i8;
                                    mac2(&mut acc[r][p], a0, vld1q_s8(wp), a1, vld1q_s8(wp.add(64)));
                                }
                            }
                        }
                        kk += 16;
                    }
                    for p in 0..2 {
                        let pp = 2 * half + p;
                        let s = wscale(sw.add(c * 8 + 2 * pp));
                        for r in 0..RP {
                            let q = &acc[r][p];
                            let sum = vpaddq_s32(vpaddq_s32(q[0], q[1]), vpaddq_s32(q[2], q[3]));
                            let sc = vmulq_f32(vld1q_f32(sa.add((r * ng + c) * 4)), s);
                            accf[r][pp] = vfmaq_f32(accf[r][pp], vcvtq_f32_s32(sum), sc);
                        }
                    }
                }
            }
            store(&accf, t);
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code, clippy::needless_range_loop)]
mod wasm {
    use core::arch::wasm32::*;

    /// 2 x 2 block accumulators [r0 o0, r0 o1, r1 o0, r1 o1], one i32x4 each, summed across their lanes at the end.
    #[derive(Clone, Copy)]
    struct Blk(v128, v128, v128, v128);

    impl Blk {
        #[inline(always)]
        fn zero() -> Blk {
            Blk(i32x4_splat(0), i32x4_splat(0), i32x4_splat(0), i32x4_splat(0))
        }
        /// Two 8-k steps: a = [row 0 | row 1], x = the same halves swapped, w = [output 0 | output 1], per step.
        /// Widening multiplies (smull / pmullw), two steps' int16 products added (|a| <= 127, |w| <= 128: no
        /// overflow), then widened pairwise into the int32 sums.
        #[inline(always)]
        fn mac2(&mut self, a0: v128, x0: v128, w0: v128, a1: v128, x1: v128, w1: v128) {
            let d00 = i16x8_add(i16x8_extmul_low_i8x16(a0, w0), i16x8_extmul_low_i8x16(a1, w1));
            let d11 = i16x8_add(i16x8_extmul_high_i8x16(a0, w0), i16x8_extmul_high_i8x16(a1, w1));
            let d10 = i16x8_add(i16x8_extmul_low_i8x16(x0, w0), i16x8_extmul_low_i8x16(x1, w1));
            let d01 = i16x8_add(i16x8_extmul_high_i8x16(x0, w0), i16x8_extmul_high_i8x16(x1, w1));
            self.0 = i32x4_add(self.0, i32x4_extadd_pairwise_i16x8(d00));
            self.1 = i32x4_add(self.1, i32x4_extadd_pairwise_i16x8(d01));
            self.2 = i32x4_add(self.2, i32x4_extadd_pairwise_i16x8(d10));
            self.3 = i32x4_add(self.3, i32x4_extadd_pairwise_i16x8(d11));
        }
        /// [r0 o0, r0 o1, r1 o0, r1 o1] (MMLA_LANES)
        #[inline(always)]
        fn sum(&self) -> v128 {
            let u = i32x4_add(i32x4_shuffle::<0, 4, 2, 6>(self.0, self.1), i32x4_shuffle::<1, 5, 3, 7>(self.0, self.1));
            let v = i32x4_add(i32x4_shuffle::<0, 4, 2, 6>(self.2, self.3), i32x4_shuffle::<1, 5, 3, 7>(self.2, self.3));
            i32x4_add(i32x4_shuffle::<0, 1, 4, 5>(u, v), i32x4_shuffle::<2, 3, 6, 7>(u, v))
        }
    }

    #[inline(always)]
    fn swap(v: v128) -> v128 {
        i64x2_shuffle::<1, 0>(v, v)
    }

    /// int4 on relaxed SIMD: `i32x4.relaxed_dot_i8x16_i7x16_add_s` (sdot / pmaddubsw-like). Its second operand is
    /// exact on every engine only with the top bit clear (0..=127), so the weights go in + 8 (0..=15) and 8 x the sum
    /// of the activations comes off (`sum`). d = [r0 o0, r0 o0, r1 o1, r1 o1] (a: [row 0 | row 1] x w: [output 0 |
    /// output 1], 4-k halves), x = the same with the rows swapped: [r1 o0, r1 o0, r0 o1, r0 o1].
    #[cfg(target_feature = "relaxed-simd")]
    #[derive(Clone, Copy)]
    struct Dot(v128, v128);

    #[cfg(target_feature = "relaxed-simd")]
    impl Dot {
        #[inline(always)]
        fn zero() -> Dot {
            Dot(i32x4_splat(0), i32x4_splat(0))
        }
        #[inline(always)]
        fn mac(&mut self, a: v128, x: v128, w: v128) {
            self.0 = i32x4_relaxed_dot_i8x16_i7x16_add(a, w, self.0);
            self.1 = i32x4_relaxed_dot_i8x16_i7x16_add(x, w, self.1);
        }
        /// [r0 o0, r0 o1, r1 o0, r1 o1] (MMLA_LANES), x 16: the sums with the weights x 16 as `Blk` makes them. sa:
        /// the activations' sums [r0, r0, r1, r1] (4-k halves), the same dot with ones.
        #[inline(always)]
        fn sum(&self, sa: v128) -> v128 {
            let (d, x) = (i32x4_sub(self.0, i32x4_shl(sa, 3)), i32x4_sub(self.1, i32x4_shl(swap(sa), 3)));
            let e = i32x4_shuffle::<0, 2, 4, 6>(d, x); // r0o0 r1o1 r1o0 r0o1 (first halves)
            let o = i32x4_shuffle::<1, 3, 5, 7>(d, x);
            let s = i32x4_add(e, o);
            i32x4_shl(i32x4_shuffle::<0, 3, 2, 1>(s, s), 4)
        }
    }

    /// wasm SIMD128 kernel: one row pair x two output pairs (8 accumulators: x86, which runs most wasm, has 16 vector
    /// registers), 16 k per iteration.
    pub unsafe fn tile<const RP: usize, const INT4: bool>(a: *const i8, sa: *const f32, w: *const u8, sw: *const f32, k: usize, g: usize, t: &mut [[[f32; 4]; 4]; super::RPMAX]) {
        let ng = k / g;
        let mask = i8x16_splat(0xf0u8 as i8);
        // SAFETY: the caller (`tile`) guarantees RP row pairs of 2k activations and ng x 4 scales each, and a panel.
        unsafe {
            for r in 0..RP {
                let (a, sa) = (a.add(r * 2 * k), sa.add(r * ng * 4));
                for half in 0..2 {
                    let w = w.add(32 * half);
                    let (mut f0, mut f1) = (f32x4_splat(0.0), f32x4_splat(0.0));
                    for c in 0..ng {
                        #[cfg(target_feature = "relaxed-simd")]
                        if INT4 {
                            let (mut d0, mut d1, mut asum) = (Dot::zero(), Dot::zero(), i32x4_splat(0));
                            let (m0f, m08, one) = (i8x16_splat(0x0f), i8x16_splat(8), i8x16_splat(1));
                            let mut kk = c * g;
                            while kk < (c + 1) * g {
                                let (a0, a1) = (v128_load(a.add(kk * 2) as *const v128), v128_load(a.add(kk * 2 + 16) as *const v128));
                                let (x0, x1) = (swap(a0), swap(a1));
                                asum = i32x4_relaxed_dot_i8x16_i7x16_add(a1, one, i32x4_relaxed_dot_i8x16_i7x16_add(a0, one, asum));
                                let (v0, v1) = (v128_load(w.add(kk * 4) as *const v128), v128_load(w.add(kk * 4 + 16) as *const v128));
                                // low nibbles (k .. k + 8), high nibbles (k + 8 .. k + 16): two's complement + 8
                                d0.mac(a0, x0, v128_xor(v128_and(v0, m0f), m08));
                                d0.mac(a1, x1, v128_xor(u8x16_shr(v0, 4), m08));
                                d1.mac(a0, x0, v128_xor(v128_and(v1, m0f), m08));
                                d1.mac(a1, x1, v128_xor(u8x16_shr(v1, 4), m08));
                                kk += 16;
                            }
                            let sav = v128_load(sa.add(c * 4) as *const v128);
                            let s0 = f32x4_mul(sav, v128_load64_splat(sw.add(c * 8 + 4 * half) as *const u64));
                            let s1 = f32x4_mul(sav, v128_load64_splat(sw.add(c * 8 + 4 * half + 2) as *const u64));
                            f0 = f32x4_add(f0, f32x4_mul(f32x4_convert_i32x4(d0.sum(asum)), s0));
                            f1 = f32x4_add(f1, f32x4_mul(f32x4_convert_i32x4(d1.sum(asum)), s1));
                            continue;
                        }
                        let (mut b0, mut b1) = (Blk::zero(), Blk::zero());
                        let mut kk = c * g;
                        while kk < (c + 1) * g {
                            let (a0, a1) = (v128_load(a.add(kk * 2) as *const v128), v128_load(a.add(kk * 2 + 16) as *const v128));
                            let (x0, x1) = (swap(a0), swap(a1));
                            if INT4 {
                                let (v0, v1) = (v128_load(w.add(kk * 4) as *const v128), v128_load(w.add(kk * 4 + 16) as *const v128));
                                b0.mac2(a0, x0, i8x16_shl(v0, 4), a1, x1, v128_and(v0, mask));
                                b1.mac2(a0, x0, i8x16_shl(v1, 4), a1, x1, v128_and(v1, mask));
                            } else {
                                let wp = w.add(kk * 8);
                                b0.mac2(a0, x0, v128_load(wp as *const v128), a1, x1, v128_load(wp.add(64) as *const v128));
                                b1.mac2(a0, x0, v128_load(wp.add(16) as *const v128), a1, x1, v128_load(wp.add(80) as *const v128));
                            }
                            kk += 16;
                        }
                        let sav = v128_load(sa.add(c * 4) as *const v128);
                        let s0 = f32x4_mul(sav, v128_load64_splat(sw.add(c * 8 + 4 * half) as *const u64));
                        let s1 = f32x4_mul(sav, v128_load64_splat(sw.add(c * 8 + 4 * half + 2) as *const u64));
                        f0 = f32x4_add(f0, f32x4_mul(f32x4_convert_i32x4(b0.sum()), s0));
                        f1 = f32x4_add(f1, f32x4_mul(f32x4_convert_i32x4(b1.sum()), s1));
                    }
                    v128_store(t[r][2 * half].as_mut_ptr() as *mut v128, f0);
                    v128_store(t[r][2 * half + 1].as_mut_ptr() as *mut v128, f1);
                }
            }
        }
    }
}

impl Panels for Quant {
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn panel<'s>(&'s self, jp: usize, scratch: &'s mut [f32]) -> &'s [f32] {
        const { assert!(NR == 16) };
        let (k, g) = (self.k, self.group);
        if self.lay != Lay::Pair {
            // another layout (met here only when the path is forced to f32, and in tests): weight by weight
            let j0 = jp * NR;
            let cols = NR.min(self.n - j0);
            if cols < NR {
                scratch.fill(0.0);
            }
            for l in 0..cols {
                for p in 0..k {
                    let (q, s) = self.at(j0 + l, p);
                    scratch[p * NR + l] = q as f32 * s;
                }
            }
            return scratch;
        }
        let (ng, pb) = (k / g, self.panel_bytes());
        // f32 panel jp (16 columns) = integer panels 2jp, 2jp + 1 (8 outputs each); a missing second one is zeros
        for h in 0..2 {
            let jq = 2 * jp + h;
            if jq >= self.n.div_ceil(8) {
                scratch[..k * NR].chunks_exact_mut(NR).for_each(|r| r[8..].fill(0.0));
                continue;
            }
            widen8(self.int4, &self.w[jq * pb..(jq + 1) * pb], &self.s[jq * ng * 8..(jq + 1) * ng * 8], k, g, &mut scratch[..k * NR], 8 * h);
        }
        scratch
    }
}

/// One integer panel (8 outputs) widened to f32 (q * scale, as the integer kernels see them) into columns c0..c0 + 8
/// of a k x NR panel.
#[inline(always)]
fn widen8(int4: bool, w: &[u8], s: &[f32], k: usize, g: usize, out: &mut [f32], c0: usize) {
    assert!(w.len() >= k * if int4 { 4 } else { 8 } && s.len() >= k / g * 8 && out.len() >= k * NR && c0 + 8 <= NR);
    #[cfg(target_arch = "x86_64")]
    if x86() >= X86::Avx2 {
        #[allow(unsafe_code)]
        // SAFETY: AVX2 detected (`x86`); the slices hold what was asserted above.
        unsafe { x86::widen8(int4, w, s, k, g, out, c0) };
        return;
    }
    widen8_plain(int4, w, s, k, g, out, c0);
}

#[inline(always)]
fn widen8_plain(int4: bool, w: &[u8], s: &[f32], k: usize, g: usize, out: &mut [f32], c0: usize) {
    // per block of 64 bytes: 8 (int8) or 16 (int4) k-steps of the 8 outputs
    let ks = if int4 { 16 } else { 8 };
    for (b, blk) in w[..k / ks * 64].chunks_exact(64).enumerate() {
        let k0 = b * ks;
        let sc: &[f32; 8] = s[k0 / g * 8..k0 / g * 8 + 8].try_into().unwrap();
        for i in 0..8 {
            let byte = |o: usize| blk[(o / 2) * 16 + (o % 2) * 8 + i];
            if int4 {
                let lo = &mut out[(k0 + i) * NR + c0..(k0 + i) * NR + c0 + 8];
                for o in 0..8 {
                    lo[o] = ((byte(o) << 4) as i8) as f32 * sc[o];
                }
                let hi = &mut out[(k0 + 8 + i) * NR + c0..(k0 + 8 + i) * NR + c0 + 8];
                for o in 0..8 {
                    hi[o] = ((byte(o) & 0xf0) as i8) as f32 * sc[o];
                }
            } else {
                let r = &mut out[(k0 + i) * NR + c0..(k0 + i) * NR + c0 + 8];
                for o in 0..8 {
                    r[o] = (byte(o) as i8) as f32 * sc[o];
                }
            }
        }
    }
}

/// int8 weights with one scale per output row, multiplied in f32: a quarter of the memory of `Packed`. Many rows: each
/// weight widened to q * scale (the dequantized f32 matrix, same results as `Packed`); few rows (decoding): the sums of
/// a * q scaled once per output (equal up to float rounding). For layers where activation
/// quantization would cost accuracy (the joiner's output, the front end's projection).
pub struct QuantF {
    pub n: usize,
    pub k: usize,
    q: Vec<i8>,  // ceil(n / NR) panels x k x NR (k-major, as `Packed`)
    s: Vec<f32>, // ceil(n / NR) * NR
    /// The weights for the integer kernels instead (then `q`, `s` are empty): see `QuantF::new`.
    wide: Option<Quant>,
}

impl QuantF {
    /// On aarch64 with an integer kernel (`path`), the product runs on it with the activations at 16-bit precision
    /// (`Quant::gemm_wide`: within ~1e-4 of the f32 product relative to the row's largest term, 3-7x faster) and the
    /// weights are stored for it alone.
    pub fn new(n: usize, k: usize, code: impl Fn(usize, usize) -> i8, scale: impl Fn(usize) -> f32) -> QuantF {
        let int = cfg!(target_arch = "aarch64") && matches!(path(), Path::Neon | Path::Dot | Path::Mmla) && k.is_multiple_of(16);
        QuantF::new_as(n, k, int, code, scale)
    }

    /// `wide`: for the integer kernels (see `new`); else multiplied in f32.
    fn new_as(n: usize, k: usize, wide: bool, code: impl Fn(usize, usize) -> i8, scale: impl Fn(usize) -> f32) -> QuantF {
        if wide {
            return QuantF { n, k, q: Vec::new(), s: Vec::new(), wide: Some(Quant::new(n, k, k, false, code, |j, _| scale(j))) };
        }
        let mut q = vec![0i8; n.div_ceil(NR) * k * NR];
        let mut s = vec![0f32; n.div_ceil(NR) * NR];
        for j in 0..n {
            let (panel, l) = (j / NR, j % NR);
            for p in 0..k {
                q[(panel * k + p) * NR + l] = code(j, p);
            }
            s[j] = scale(j);
        }
        QuantF { n, k, q, s, wide: None }
    }

    fn gemm(&self, a: &[f32], m: usize, bias: Option<&[f32]>) -> Vec<f32> {
        assert!(a.len() >= m * self.k);
        if let Some(w) = &self.wide {
            return w.gemm_wide(a, m, bias);
        }
        let mut out = vec![0.0; m * self.n];
        if m <= 8 {
            // few rows (decoding): widen in registers instead of a scratch panel per call; groups of panels are
            // independent (one task each when worth it), each into its own columns
            let (n, k) = (self.n, self.k);
            let np = n.div_ceil(NR);
            const PG: usize = 8; // panels per task
            let cols = crate::par::map(np.div_ceil(PG), m * n * k, |grp| {
                let (jp0, jp1) = (grp * PG, (grp * PG + PG).min(np));
                let w = (jp1 - jp0) * NR;
                let mut o = vec![0f32; m * w];
                #[cfg(target_arch = "x86_64")]
                {
                    #[target_feature(enable = "avx2,fma")]
                    fn avx2(q: &QuantF, a: &[f32], m: usize, bias: Option<&[f32]>, jp0: usize, jp1: usize, o: &mut [f32]) {
                        q.panels_small(a, m, bias, jp0, jp1, o)
                    }
                    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
                    fn avx512(q: &QuantF, a: &[f32], m: usize, bias: Option<&[f32]>, jp0: usize, jp1: usize, o: &mut [f32]) {
                        q.panels_small(a, m, bias, jp0, jp1, o)
                    }
                    #[allow(unsafe_code)]
                    // SAFETY: the level was detected on this CPU (`x86`).
                    match x86() {
                        X86::Avx512 => unsafe { avx512(self, a, m, bias, jp0, jp1, &mut o) },
                        X86::Avx2 => unsafe { avx2(self, a, m, bias, jp0, jp1, &mut o) },
                        X86::Sse2 => self.panels_small(a, m, bias, jp0, jp1, &mut o),
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                self.panels_small(a, m, bias, jp0, jp1, &mut o);
                o
            });
            for (grp, o) in cols.iter().enumerate() {
                let j0 = grp * PG * NR;
                let w = o.len() / m;
                let c = w.min(n - j0);
                for r in 0..m {
                    out[r * n + j0..r * n + j0 + c].copy_from_slice(&o[r * w..r * w + c]);
                }
            }
            return out;
        }
        blocked(&a[..m * self.k], self, bias, &mut out);
        out
    }
}

/// `kernel_q8` written for auto-vectorization (wasm SIMD128, SSE): b = q * scale per weight, then the same sums.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn kernel_q8<const R: usize>(a: &[f32], k: usize, panel: &[i8], sc: &[f32; NR], init: &[f32; NR], out: &mut [[f32; NR]; R]) {
    let rows: [&[f32]; R] = core::array::from_fn(|r| &a[r * k..(r + 1) * k]);
    let mut acc = [[0f32; NR]; R];
    for (p, q) in panel[..k * NR].chunks_exact(NR).enumerate() {
        let mut b = [0f32; NR];
        for l in 0..NR {
            b[l] = q[l] as f32;
        }
        for r in 0..R {
            let x = rows[r][p];
            for l in 0..NR {
                acc[r][l] += x * b[l];
            }
        }
    }
    // the column scales once, at the end
    for r in 0..R {
        for l in 0..NR {
            out[r][l] = init[l] + acc[r][l] * sc[l];
        }
    }
}

/// `kernel` with the panel widened in registers: b = (int8 -> f32) * scale, then the same fused multiply-adds.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_code)]
fn kernel_q8<const R: usize>(a: &[f32], k: usize, panel: &[i8], sc: &[f32; NR], init: &[f32; NR], out: &mut [[f32; NR]; R]) {
    use core::arch::aarch64::*;
    assert!(a.len() >= R * k && panel.len() >= k * NR);
    // SAFETY: NEON is mandatory on aarch64; reads stay inside the slices checked above.
    unsafe {
        let ip = init.as_ptr();
        let start: [float32x4_t; 4] = [vld1q_f32(ip), vld1q_f32(ip.add(4)), vld1q_f32(ip.add(8)), vld1q_f32(ip.add(12))];
        let s: [float32x4_t; 4] = core::array::from_fn(|j| vld1q_f32(sc.as_ptr().add(4 * j)));
        let mut acc = [[vdupq_n_f32(0.0); 4]; R];
        let (ap, bp) = (a.as_ptr(), panel.as_ptr());
        for p in 0..k {
            let q = vld1q_s8(bp.add(p * NR));
            let (lo, hi) = (vmovl_s8(vget_low_s8(q)), vmovl_high_s8(q));
            let b = [
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
                vcvtq_f32_s32(vmovl_high_s16(lo)),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
                vcvtq_f32_s32(vmovl_high_s16(hi)),
            ];
            for (r, acc) in acc.iter_mut().enumerate() {
                let x = vdupq_n_f32(*ap.add(r * k + p));
                for j in 0..4 {
                    acc[j] = vfmaq_f32(acc[j], b[j], x);
                }
            }
        }
        // the column scales once, at the end
        for (o, acc) in out.iter_mut().zip(&acc) {
            for (j, v) in acc.iter().enumerate() {
                vst1q_f32(o.as_mut_ptr().add(4 * j), vfmaq_f32(start[j], *v, s[j]));
            }
        }
    }
}

impl QuantF {
    /// Panels jp0..jp1 of m (<= 8) rows into o (m x (jp1 - jp0) * NR), weights widened in registers.
    #[inline(always)]
    fn panels_small(&self, a: &[f32], m: usize, bias: Option<&[f32]>, jp0: usize, jp1: usize, o: &mut [f32]) {
        let (n, k) = (self.n, self.k);
        let w = (jp1 - jp0) * NR;
        // few rows on x86-64: several panels at a time (four for one or two rows, two for three or four on AVX-512)
        #[cfg(target_arch = "x86_64")]
        let start = {
            let mut jp = jp0;
            let lvl = x86();
            let pp = if m <= 2 && lvl >= X86::Avx2 { 4 } else if m <= 4 && lvl >= X86::Avx512 { 2 } else { 0 };
            if pp > 0 {
                let zero = [0f32; 4 * NR];
                while jp + pp <= jp1 && (jp + pp) * NR <= n {
                    let init = bias.map_or(&zero[..], |b| &b[jp * NR..(jp + pp) * NR]);
                    let c0 = (jp - jp0) * NR;
                    assert!(a.len() >= m * k && c0 + pp * NR + (m - 1) * w <= o.len() && self.q.len() >= (jp + pp) * k * NR && self.s.len() >= (jp + pp) * NR);
                    #[allow(unsafe_code)]
                    // SAFETY: panels jp..jp + pp of q (k x NR each), their pp x NR scales and init values, m rows of k
                    // in a, pp x NR columns of m rows of o (row stride w) from c0 (asserted); the level was detected on
                    // this CPU.
                    unsafe {
                        let (q, sc, op) = (self.q.as_ptr().add(jp * k * NR), self.s.as_ptr().add(jp * NR), o.as_mut_ptr().add(c0));
                        match (lvl, m) {
                            (X86::Avx512, 4) => x86::q8p_avx512::<4, 2>(a.as_ptr(), k, q, sc, init.as_ptr(), op, w),
                            (X86::Avx512, 3) => x86::q8p_avx512::<3, 2>(a.as_ptr(), k, q, sc, init.as_ptr(), op, w),
                            (X86::Avx512, 2) => x86::q8p_avx512::<2, 4>(a.as_ptr(), k, q, sc, init.as_ptr(), op, w),
                            (X86::Avx512, _) => x86::q8p_avx512::<1, 4>(a.as_ptr(), k, q, sc, init.as_ptr(), op, w),
                            (_, 2) => x86::q8p_avx2::<2>(a.as_ptr(), k, q, sc, init.as_ptr(), op, w),
                            _ => x86::q8p_avx2::<1>(a.as_ptr(), k, q, sc, init.as_ptr(), op, w),
                        }
                    }
                    jp += pp;
                }
            }
            jp
        };
        #[cfg(not(target_arch = "x86_64"))]
        let start = jp0;
        for jp in start..jp1 {
            let j0 = jp * NR;
            let cols = NR.min(n - j0);
            let mut init = [0f32; NR];
            if let Some(b) = bias {
                init[..cols].copy_from_slice(&b[j0..j0 + cols]);
            }
            let panel = &self.q[jp * k * NR..(jp + 1) * k * NR];
            let sc: &[f32; NR] = self.s[j0..j0 + NR].try_into().unwrap();
            let mut i = 0;
            while i < m {
                let rows = 4.min(m - i);
                let mut tile = [[0f32; NR]; 4];
                let a = &a[i * k..(i + rows) * k];
                match rows {
                    4 => kernel_q8::<4>(a, k, panel, sc, &init, &mut tile),
                    3 => kernel_q8::<3>(a, k, panel, sc, &init, (&mut tile[..3]).try_into().unwrap()),
                    2 => kernel_q8::<2>(a, k, panel, sc, &init, (&mut tile[..2]).try_into().unwrap()),
                    _ => kernel_q8::<1>(a, k, panel, sc, &init, (&mut tile[..1]).try_into().unwrap()),
                }
                for r in 0..rows {
                    let o0 = (i + r) * w + (jp - jp0) * NR;
                    o[o0..o0 + NR].copy_from_slice(&tile[r]);
                }
                i += rows;
            }
        }
    }
}

impl Panels for QuantF {
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn panel<'s>(&'s self, jp: usize, scratch: &'s mut [f32]) -> &'s [f32] {
        let k = self.k;
        let q = &self.q[jp * k * NR..(jp + 1) * k * NR];
        let s = &self.s[jp * NR..(jp + 1) * NR];
        for (o, q) in scratch[..k * NR].chunks_exact_mut(NR).zip(q.chunks_exact(NR)) {
            for l in 0..NR {
                o[l] = q[l] as f32 * s[l];
            }
        }
        scratch
    }
}

pub enum Weight {
    F32(Packed),
    Int(Quant),
    I8(QuantF),
}

/// y = x W^T + b, with W in PyTorch Linear layout (n x k).
pub struct Linear {
    pub w: Weight,
    pub b: Option<Vec<f32>>,
}

impl Linear {
    pub fn n(&self) -> usize {
        match &self.w {
            Weight::F32(p) => p.n,
            Weight::Int(q) => q.n,
            Weight::I8(q) => q.n,
        }
    }

    /// x: m rows of k.
    pub fn apply(&self, x: &[f32], m: usize) -> Vec<f32> {
        match &self.w {
            Weight::F32(p) => gemm(x, m, p, self.b.as_deref()),
            Weight::Int(q) => q.gemm(x, m, self.b.as_deref()),
            Weight::I8(q) => q.gemm(x, m, self.b.as_deref()),
        }
    }
}

// ---------------------------------------------------------------------------------------------- x86-64

/// SIMD level of an x86-64 CPU, detected at run time: SSE2 (every x86-64; what the plain Rust auto-vectorizes to),
/// AVX2 with FMA (Haswell, Zen 1 on), AVX-512 F/BW/DQ/VL (Skylake-X, Ice Lake, Zen 4 on). Every level computes the
/// same products and sums in the same order: the same bits. std builds: TINYASR_X86=sse2|avx2 caps it (for tests).
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum X86 {
    Sse2,
    Avx2,
    Avx512,
}

#[cfg(target_arch = "x86_64")]
pub fn x86() -> X86 {
    use core::sync::atomic::{AtomicU8, Ordering};
    // 0 not known yet, else X86 as u8 + 1
    static L: AtomicU8 = AtomicU8::new(0);
    let v = L.load(Ordering::Relaxed);
    if v != 0 {
        return [X86::Sse2, X86::Avx2, X86::Avx512][v as usize - 1];
    }
    #[cfg(feature = "std")]
    let l = {
        use std::arch::is_x86_feature_detected as has;
        let avx2 = has!("avx2") && has!("fma");
        let avx512 = avx2 && has!("avx512f") && has!("avx512bw") && has!("avx512dq") && has!("avx512vl");
        let l = if avx512 { X86::Avx512 } else if avx2 { X86::Avx2 } else { X86::Sse2 };
        match std::env::var("TINYASR_X86").as_deref() {
            Ok("sse2") => X86::Sse2,
            Ok("avx2") => l.min(X86::Avx2),
            _ => l,
        }
    };
    #[cfg(not(feature = "std"))]
    let l = if cfg!(all(target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl")) {
        X86::Avx512
    } else if cfg!(all(target_feature = "avx2", target_feature = "fma")) {
        X86::Avx2
    } else {
        X86::Sse2
    };
    L.store(l as u8 + 1, Ordering::Relaxed);
    l
}

/// Activations for the x86-64 integer kernels: rows of int8 codes (quantized as `pack_act`), per row and group the
/// scale (max / 127) and the sum of the codes.
#[cfg(target_arch = "x86_64")]
struct ActX {
    q: Vec<i8>,
    s: Vec<f32>,
    sum: Vec<i32>,
    k: usize,
    ng: usize,
}

#[cfg(target_arch = "x86_64")]
impl ActX {
    fn new(a: &[f32], m: usize, k: usize, g: usize) -> ActX {
        let ng = k / g;
        let (mut q, mut s, mut sum) = (vec![0i8; m * k], vec![0f32; m * ng], vec![0i32; m * ng]);
        if g.is_multiple_of(16) && x86() >= X86::Avx2 {
            assert!(a.len() >= m * k);
            #[allow(unsafe_code)]
            // SAFETY: a holds m rows of k, q m x k, s and sum m x ng (just made); the level was detected on this CPU.
            unsafe {
                if x86() >= X86::Avx512 { x86::act_avx512(a, m, k, g, &mut q, &mut s, &mut sum) } else { x86::act_avx2(a, m, k, g, &mut q, &mut s, &mut sum) }
            }
            return ActX { q, s, sum, k, ng };
        }
        vector(#[inline(always)] || {
            for r in 0..m {
                for c in 0..ng {
                    let x = &a[r * k + c * g..r * k + (c + 1) * g];
                    let max = absmax(x);
                    let inv = if max > 0.0 { 127.0 / max } else { 0.0 };
                    let qr = &mut q[r * k + c * g..r * k + (c + 1) * g];
                    // `quant8`, vectorizable: half away from zero, truncated, to int through the float's bits (exact
                    // for |v| < 2^22; finite activations give |v| <= 127.5)
                    let mut sm = 0i32;
                    for (o, v) in qr.iter_mut().zip(x) {
                        let r = v * inv;
                        let y = r + 0.5f32.copysign(r);
                        // y truncated: rounded to nearest by adding and taking away 1.5 * 2^23, then one step back
                        // toward zero if that went past y
                        let n = (y + 12_582_912.0) - 12_582_912.0;
                        let t = if y >= 0.0 { if n > y { n - 1.0 } else { n } } else if n < y { n + 1.0 } else { n };
                        let i = (t + 12_582_912.0).to_bits() as i32 - 0x4b40_0000;
                        *o = i as i8;
                        sm += i;
                    }
                    s[r * ng + c] = max / 127.0;
                    sum[r * ng + c] = sm;
                }
            }
        });
        ActX { q, s, sum, k, ng }
    }

    fn rows(&self) -> usize {
        self.q.len() / self.k
    }
}

/// The x86-64 integer kernel of this CPU (`Path::X86`).
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntX86 {
    /// AVX2: `vpmaddubsw` (unsigned weight x signed activation pairs into int16) + `vpmaddwd`.
    Avx2,
    /// AVX-VNNI (Alder Lake, Zen 5 on): `vpdpbusd` on ymm.
    AvxVnni,
    /// AVX-512 VNNI (Ice Lake, Zen 4 on): `vpdpbusd` on zmm.
    Avx512Vnni,
}

#[cfg(target_arch = "x86_64")]
pub fn int_x86() -> IntX86 {
    #[cfg(feature = "std")]
    {
        static K: std::sync::OnceLock<IntX86> = std::sync::OnceLock::new();
        *K.get_or_init(|| {
            use std::arch::is_x86_feature_detected as has;
            let lvl = x86();
            let forced = std::env::var("TINYASR_INT_X86");
            if lvl >= X86::Avx512 && has!("avx512vnni") && forced.as_deref().is_err() {
                IntX86::Avx512Vnni
            } else if has!("avxvnni") && forced.as_deref() != Ok("avx2") {
                IntX86::AvxVnni
            } else {
                IntX86::Avx2
            }
        })
    }
    #[cfg(not(feature = "std"))]
    if cfg!(all(target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl", target_feature = "avx512vnni")) {
        IntX86::Avx512Vnni
    } else if cfg!(target_feature = "avxvnni") {
        IntX86::AvxVnni
    } else {
        IntX86::Avx2
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, clippy::needless_range_loop)]
mod x86 {
    use super::NR;
    use core::arch::x86_64::*;

    /// `kernel` on AVX2: R rows x 16 columns in 2R ymm accumulators; product, then sum (not fused): the same bits.
    #[inline(always)]
    unsafe fn kernel_avx2<const R: usize>(a: *const f32, k: usize, panel: *const f32, init: &[f32; NR], out: &mut [[f32; NR]]) {
        // SAFETY: the caller's a holds R rows of k, panel k x 16, out R rows.
        unsafe {
            let start = [_mm256_loadu_ps(init.as_ptr()), _mm256_loadu_ps(init.as_ptr().add(8))];
            let mut acc = [start; R];
            for p in 0..k {
                let b = panel.add(p * NR);
                let (b0, b1) = (_mm256_loadu_ps(b), _mm256_loadu_ps(b.add(8)));
                for r in 0..R {
                    let x = _mm256_broadcast_ss(&*a.add(r * k + p));
                    acc[r][0] = _mm256_add_ps(acc[r][0], _mm256_mul_ps(x, b0));
                    acc[r][1] = _mm256_add_ps(acc[r][1], _mm256_mul_ps(x, b1));
                }
            }
            for r in 0..R {
                _mm256_storeu_ps(out[r].as_mut_ptr(), acc[r][0]);
                _mm256_storeu_ps(out[r].as_mut_ptr().add(8), acc[r][1]);
            }
        }
    }

    /// `kernel` on AVX-512: R rows x 16 columns, one zmm accumulator per row.
    #[inline(always)]
    unsafe fn kernel_avx512<const R: usize>(a: *const f32, k: usize, panel: *const f32, init: &[f32; NR], out: &mut [[f32; NR]]) {
        // SAFETY: as `kernel_avx2`.
        unsafe {
            let mut acc = [_mm512_loadu_ps(init.as_ptr()); R];
            for p in 0..k {
                let b = _mm512_loadu_ps(panel.add(p * NR));
                for r in 0..R {
                    acc[r] = _mm512_add_ps(acc[r], _mm512_mul_ps(_mm512_set1_ps(*a.add(r * k + p)), b));
                }
            }
            for r in 0..R {
                _mm512_storeu_ps(out[r].as_mut_ptr(), acc[r]);
            }
        }
    }

    /// `super::rows` with MR rows per call (fewer for the last ones).
    macro_rules! rows {
        ($name:ident, $feat:literal, $kernel:ident, $mr:literal, $($r:literal)*) => {
            #[target_feature(enable = $feat)]
            pub unsafe fn $name(a: &[f32], k: usize, panel: &[f32], init: &[f32; NR], out: &mut [f32], stride: usize, cols: usize) {
                let m = a.len() / k;
                assert!(panel.len() >= k * NR && cols <= NR && (m == 0 || out.len() >= (m - 1) * stride + cols));
                let mut tile = [[0f32; NR]; $mr];
                let mut i = 0;
                while i < m {
                    let rows = (m - i).min($mr);
                    // SAFETY: rows i..i + rows of a, the whole panel (asserted).
                    unsafe {
                        match rows {
                            $($r => $kernel::<$r>(a.as_ptr().add(i * k), k, panel.as_ptr(), init, &mut tile),)*
                            _ => unreachable!(),
                        }
                    }
                    for r in 0..rows {
                        out[(i + r) * stride..(i + r) * stride + cols].copy_from_slice(&tile[r][..cols]);
                    }
                    i += rows;
                }
            }
        };
    }
    rows!(rows_avx2, "avx2,fma", kernel_avx2, 6, 1 2 3 4 5 6);
    rows!(rows_avx512, "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma", kernel_avx512, 12, 1 2 3 4 5 6 7 8 9 10 11 12);

    /// One 64-byte block of an integer panel (8 outputs x 8 bytes, in pairs [output 2p | output 2p + 1]) transposed:
    /// vector j holds byte 2j of outputs 0..8 (low half), byte 2j + 1 (high half).
    #[inline(always)]
    unsafe fn transpose(wp: *const u8) -> [__m128i; 4] {
        // SAFETY: the caller's wp points to 64 bytes; SSSE3 is part of every level that calls this.
        unsafe {
            let wp = wp as *const __m128i;
            // 16-bit lanes (output 2p, 2p + 1) of byte i
            let il = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
            let q: [__m128i; 4] = core::array::from_fn(|p| _mm_shuffle_epi8(_mm_loadu_si128(wp.add(p)), il));
            // 32-bit lanes: outputs 0..4 (resp. 4..8) of bytes 0..4 and 4..8
            let (l01, h01) = (_mm_unpacklo_epi16(q[0], q[1]), _mm_unpackhi_epi16(q[0], q[1]));
            let (l23, h23) = (_mm_unpacklo_epi16(q[2], q[3]), _mm_unpackhi_epi16(q[2], q[3]));
            [_mm_unpacklo_epi32(l01, l23), _mm_unpackhi_epi32(l01, l23), _mm_unpacklo_epi32(h01, h23), _mm_unpackhi_epi32(h01, h23)]
        }
    }

    /// `Quant` times R rows with the weights widened in registers (no f32 panel), AVX2: 8 outputs (one integer panel,
    /// ymm). Per output the sum of the panel path (init, then + x * w in k order, product then sum): the same bits.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn quant8_avx2<const R: usize, const INT4: bool>(a: *const f32, k: usize, w: *const u8, s: *const f32, g: usize, init: *const f32, out: *mut f32, stride: usize) {
        let ks = if INT4 { 16 } else { 8 };
        // SAFETY: the caller's a holds R rows of k, w one panel, s its scales, out R rows of 8 at `stride`.
        unsafe {
            let mask = _mm_set1_epi8(0xf0u8 as i8);
            let mut acc = [_mm256_loadu_ps(init); R];
            for b in 0..k / ks {
                let k0 = b * ks;
                let sc = _mm256_loadu_ps(s.add(k0 / g * 8));
                let t = transpose(w.add(b * 64));
                let byte = |i: usize| if i.is_multiple_of(2) { t[i / 2] } else { _mm_unpackhi_epi64(t[i / 2], t[i / 2]) };
                let wv = |v: __m128i| _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)), sc);
                for h in 0..if INT4 { 2 } else { 1 } {
                    for i in 0..8 {
                        let v = byte(i);
                        let wf = wv(if !INT4 { v } else if h == 0 { _mm_and_si128(_mm_slli_epi16::<4>(v), mask) } else { _mm_and_si128(v, mask) });
                        let kk = k0 + 8 * h + i;
                        for r in 0..R {
                            acc[r] = _mm256_add_ps(acc[r], _mm256_mul_ps(_mm256_broadcast_ss(&*a.add(r * k + kk)), wf));
                        }
                    }
                }
            }
            for r in 0..R {
                _mm256_storeu_ps(out.add(r * stride), acc[r]);
            }
        }
    }

    /// As `quant8_avx2` on AVX-512: 16 outputs (two integer panels w0, w1 with scales s0, s1; zmm).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn quant16_avx512<const R: usize, const INT4: bool>(a: *const f32, k: usize, w0: *const u8, w1: *const u8, s0: *const f32, s1: *const f32, g: usize, init: *const f32, out: *mut f32, stride: usize) {
        let ks = if INT4 { 16 } else { 8 };
        // SAFETY: as `quant8_avx2`, for two panels and 16 outputs.
        unsafe {
            let mask = _mm_set1_epi8(0xf0u8 as i8);
            let mut acc = [_mm512_loadu_ps(init); R];
            for b in 0..k / ks {
                let k0 = b * ks;
                let sc = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(_mm256_loadu_ps(s0.add(k0 / g * 8))), _mm256_loadu_ps(s1.add(k0 / g * 8)));
                let (t0, t1) = (transpose(w0.add(b * 64)), transpose(w1.add(b * 64)));
                // byte i of outputs 0..8 (low half), 8..16 (high half)
                let byte = |i: usize| if i.is_multiple_of(2) { _mm_unpacklo_epi64(t0[i / 2], t1[i / 2]) } else { _mm_unpackhi_epi64(t0[i / 2], t1[i / 2]) };
                let wv = |v: __m128i| _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v)), sc);
                for h in 0..if INT4 { 2 } else { 1 } {
                    for i in 0..8 {
                        let v = byte(i);
                        let wf = wv(if !INT4 { v } else if h == 0 { _mm_and_si128(_mm_slli_epi16::<4>(v), mask) } else { _mm_and_si128(v, mask) });
                        let kk = k0 + 8 * h + i;
                        for r in 0..R {
                            acc[r] = _mm512_add_ps(acc[r], _mm512_mul_ps(_mm512_set1_ps(*a.add(r * k + kk)), wf));
                        }
                    }
                }
            }
            for r in 0..R {
                _mm512_storeu_ps(out.add(r * stride), acc[r]);
            }
        }
    }

    /// `Quant` panels jp0..jp1 (of 8 outputs) times m rows into o (row r at r * stride, panel jp at (jp - jp0) * 8;
    /// whole panels: o is wide enough), bias in `init` (8 per panel), weights widened in registers.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn quant_avx2(q: &super::Quant, a: &[f32], m: usize, init: &[f32], jp0: usize, jp1: usize, o: &mut [f32], stride: usize) {
        let (k, g, ng, pb) = (q.k, q.group, q.k / q.group, q.panel_bytes());
        assert!(a.len() >= m * k && init.len() >= jp1 * 8 && q.w.len() >= jp1 * pb && q.s.len() >= jp1 * ng * 8 && (m == 0 || o.len() >= (m - 1) * stride + (jp1 - jp0) * 8));
        for jp in jp0..jp1 {
            let mut i = 0;
            while i < m {
                let rows = (m - i).min(8);
                // SAFETY: rows i..i + rows, panel jp, its 8 outputs of o (asserted).
                unsafe {
                    let (ap, w, s, ip, op) = (a.as_ptr().add(i * k), q.w.as_ptr().add(jp * pb), q.s.as_ptr().add(jp * ng * 8), init.as_ptr().add(jp * 8), o.as_mut_ptr().add(i * stride + (jp - jp0) * 8));
                    macro_rules! go {
                        ($($r:literal)*) => {
                            match (rows, q.int4) {
                                $(($r, true) => quant8_avx2::<$r, true>(ap, k, w, s, g, ip, op, stride),
                                ($r, false) => quant8_avx2::<$r, false>(ap, k, w, s, g, ip, op, stride),)*
                                _ => unreachable!(),
                            }
                        };
                    }
                    go!(1 2 3 4 5 6 7 8);
                }
                i += rows;
            }
        }
    }

    /// `quant_avx2` on AVX-512, two panels at a time (jp0 even; a last odd panel pairs with zero scales, so its
    /// missing outputs are zeros: o must be wide enough for whole pairs).
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn quant_avx512(q: &super::Quant, a: &[f32], m: usize, init: &[f32], jp0: usize, jp1: usize, o: &mut [f32], stride: usize) {
        let (k, g, ng, pb) = (q.k, q.group, q.k / q.group, q.panel_bytes());
        let np = q.n.div_ceil(8);
        let pairs = (jp1 - jp0).div_ceil(2);
        assert!(jp0.is_multiple_of(2) && jp1 <= np && a.len() >= m * k && init.len() >= jp0 * 8 + pairs * 16 && q.w.len() >= jp1 * pb && q.s.len() >= jp1 * ng * 8);
        assert!(m == 0 || o.len() >= (m - 1) * stride + pairs * 16);
        let zero = vec![0f32; ng * 8];
        for pp in 0..pairs {
            let (j0, j1) = (jp0 + 2 * pp, jp0 + 2 * pp + 1);
            let mut i = 0;
            while i < m {
                let rows = (m - i).min(8);
                // SAFETY: rows i..i + rows, panels j0 and j1 (or j0 again, with zero scales), their 16 outputs of o.
                unsafe {
                    let ap = a.as_ptr().add(i * k);
                    let (w0, s0) = (q.w.as_ptr().add(j0 * pb), q.s.as_ptr().add(j0 * ng * 8));
                    let (w1, s1) = if j1 < jp1 { (q.w.as_ptr().add(j1 * pb), q.s.as_ptr().add(j1 * ng * 8)) } else { (w0, zero.as_ptr()) };
                    let (ip, op) = (init.as_ptr().add(j0 * 8), o.as_mut_ptr().add(i * stride + 2 * pp * 8));
                    macro_rules! go {
                        ($($r:literal)*) => {
                            match (rows, q.int4) {
                                $(($r, true) => quant16_avx512::<$r, true>(ap, k, w0, w1, s0, s1, g, ip, op, stride),
                                ($r, false) => quant16_avx512::<$r, false>(ap, k, w0, w1, s0, s1, g, ip, op, stride),)*
                                _ => unreachable!(),
                            }
                        };
                    }
                    go!(1 2 3 4 5 6 7 8);
                }
                i += rows;
            }
        }
    }

    // ------------------------------------------------------------------ integer weights x int8 activations
    //
    // A 64-byte block of an integer panel is 8 outputs x 8 bytes (output o at byte 8o): its 32-bit lanes 2o, 2o + 1
    // hold k-steps 0..4, 4..8 of output o, so a row's 8 activation bytes broadcast to every 64-bit lane line up with
    // them. Weights become the unsigned operand: int4 nibble + 8 (0..=15), int8 code + 128; offset x (sum of the
    // row's codes in the group) comes off at the group end. Integer sums are exact in any order: the same int32 as
    // `tile_scalar`, then the same float steps (int4: x 16, times sa * sw, added group by group).

    /// R rows x one panel (8 outputs) on AVX-512 VNNI: per row one zmm of lane pairs.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn int512<const R: usize, const INT4: bool, const _V: bool>(act: &super::ActX, r0: usize, w: *const u8, sw: *const f32, g: usize, o: *mut f32, stride: usize, init: *const f32, c: usize) {
        let (k, ng) = (act.k, act.ng);
        let ks = if INT4 { 16 } else { 8 };
        // SAFETY: rows r0..r0 + R of act (the caller checks), one panel of w and its ng x 8 scales.
        unsafe {
            let aq = act.q.as_ptr().add(r0 * k);
            let (m0f, m88, m80) = (_mm512_set1_epi8(0x0f), _mm512_set1_epi8(0x88u8 as i8), _mm512_set1_epi8(0x80u8 as i8));
            let mut f = [_mm256_setzero_ps(); R];
            for c in 0..ng {
                let mut acc = [_mm512_setzero_si512(); R];
                for b in c * g / ks..(c + 1) * g / ks {
                    let v = _mm512_loadu_si512(w.add(b * 64) as *const _);
                    let k0 = b * ks;
                    let a8 = |r: usize, kk: usize| _mm512_set1_epi64((aq.add(r * k + kk) as *const i64).read_unaligned());
                    if INT4 {
                        let lo = _mm512_and_si512(_mm512_xor_si512(v, m88), m0f);
                        let hi = _mm512_and_si512(_mm512_srli_epi16::<4>(_mm512_xor_si512(v, m88)), m0f);
                        for r in 0..R {
                            acc[r] = _mm512_dpbusd_epi32(acc[r], lo, a8(r, k0));
                            acc[r] = _mm512_dpbusd_epi32(acc[r], hi, a8(r, k0 + 8));
                        }
                    } else {
                        let u = _mm512_xor_si512(v, m80);
                        for r in 0..R {
                            acc[r] = _mm512_dpbusd_epi32(acc[r], u, a8(r, k0));
                        }
                    }
                }
                let sw8 = _mm256_loadu_ps(sw.add(c * 8));
                for r in 0..R {
                    // lane pairs added: output o in the low half of 64-bit lane o
                    let d = _mm512_cvtepi64_epi32(_mm512_add_epi32(acc[r], _mm512_srli_epi64::<32>(acc[r])));
                    let i = (r0 + r) * ng + c;
                    f[r] = step::<INT4>(f[r], d, *act.sum.as_ptr().add(i), *act.s.as_ptr().add(i), sw8);
                }
            }
            // + bias, the first c columns of rows r0.. (o at row r0)
            let mask = ((1u32 << c) - 1) as __mmask8;
            let b = _mm256_maskz_loadu_ps(mask, init);
            for r in 0..R {
                _mm256_mask_storeu_ps(o.add(r * stride), mask, _mm256_add_ps(f[r], b));
            }
        }
    }

    /// `ActX::new` on AVX-512 (g a multiple of 16): per group absmax (NaN skipped, as `max`), codes rounded half away
    /// from zero and truncated (`vcvttps2dq`: the same integers as `quant8`, |v| <= 127.5), their sum.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
    pub unsafe fn act_avx512(a: &[f32], m: usize, k: usize, g: usize, q: &mut [i8], s: &mut [f32], sum: &mut [i32]) {
        let ng = k / g;
        assert!(g.is_multiple_of(16) && a.len() >= m * k && q.len() >= m * k && s.len() >= m * ng && sum.len() >= m * ng);
        // SAFETY: every access is inside the m x k (q, a) and m x ng (s, sum) asserted above.
        unsafe {
            let (ap, qp) = (a.as_ptr(), q.as_mut_ptr());
            let (half, sign) = (_mm512_set1_ps(0.5), _mm512_set1_ps(-0.0));
            for i in 0..m * ng {
                let x = ap.add(i * g);
                let mut mx = _mm512_setzero_ps();
                for j in (0..g).step_by(16) {
                    // max(v, mx) is mx when v is NaN
                    mx = _mm512_max_ps(_mm512_andnot_ps(sign, _mm512_loadu_ps(x.add(j))), mx);
                }
                let max = _mm512_reduce_max_ps(mx);
                let inv = if max > 0.0 { 127.0 / max } else { 0.0 };
                let vi = _mm512_set1_ps(inv);
                let mut sm = _mm512_setzero_si512();
                for j in (0..g).step_by(16) {
                    let r = _mm512_mul_ps(_mm512_loadu_ps(x.add(j)), vi);
                    let y = _mm512_add_ps(r, _mm512_or_ps(half, _mm512_and_ps(r, sign)));
                    let t = _mm512_cvttps_epi32(y);
                    sm = _mm512_add_epi32(sm, t);
                    _mm_storeu_si128(qp.add(i * g + j) as *mut _, _mm512_cvtepi32_epi8(t));
                }
                *s.get_unchecked_mut(i) = max / 127.0;
                *sum.get_unchecked_mut(i) = _mm512_reduce_add_epi32(sm);
            }
        }
    }

    /// `act_avx512` on AVX2.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn act_avx2(a: &[f32], m: usize, k: usize, g: usize, q: &mut [i8], s: &mut [f32], sum: &mut [i32]) {
        let ng = k / g;
        assert!(g.is_multiple_of(16) && a.len() >= m * k && q.len() >= m * k && s.len() >= m * ng && sum.len() >= m * ng);
        // SAFETY: as `act_avx512`.
        unsafe {
            let (ap, qp) = (a.as_ptr(), q.as_mut_ptr());
            let (half, sign) = (_mm256_set1_ps(0.5), _mm256_set1_ps(-0.0));
            for i in 0..m * ng {
                let x = ap.add(i * g);
                let mut mx = [_mm256_setzero_ps(); 2];
                for j in (0..g).step_by(16) {
                    for h in 0..2 {
                        mx[h] = _mm256_max_ps(_mm256_andnot_ps(sign, _mm256_loadu_ps(x.add(j + 8 * h))), mx[h]);
                    }
                }
                // every lane is a non-NaN value >= 0: the maximum in any order
                let m8 = _mm256_max_ps(mx[0], mx[1]);
                let m4 = _mm_max_ps(_mm256_castps256_ps128(m8), _mm256_extractf128_ps::<1>(m8));
                let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
                let max = _mm_cvtss_f32(_mm_max_ss(m2, _mm_shuffle_ps::<1>(m2, m2)));
                let inv = if max > 0.0 { 127.0 / max } else { 0.0 };
                let vi = _mm256_set1_ps(inv);
                let mut sm = _mm256_setzero_si256();
                for j in (0..g).step_by(16) {
                    let t: [__m256i; 2] = core::array::from_fn(|h| {
                        let r = _mm256_mul_ps(_mm256_loadu_ps(x.add(j + 8 * h)), vi);
                        _mm256_cvttps_epi32(_mm256_add_ps(r, _mm256_or_ps(half, _mm256_and_ps(r, sign))))
                    });
                    sm = _mm256_add_epi32(sm, _mm256_add_epi32(t[0], t[1]));
                    // int32 -> int16 (lanes interleaved per 128 bits, put back in order) -> int8; |t| <= 127
                    let p16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packs_epi32(t[0], t[1]));
                    let p8 = _mm_packs_epi16(_mm256_castsi256_si128(p16), _mm256_extracti128_si256::<1>(p16));
                    _mm_storeu_si128(qp.add(i * g + j) as *mut _, p8);
                }
                *s.get_unchecked_mut(i) = max / 127.0;
                let s4 = _mm_add_epi32(_mm256_castsi256_si128(sm), _mm256_extracti128_si256::<1>(sm));
                let s2 = _mm_add_epi32(s4, _mm_unpackhi_epi64(s4, s4));
                *sum.get_unchecked_mut(i) = _mm_cvtsi128_si32(_mm_add_epi32(s2, _mm_shuffle_epi32::<1>(s2)));
            }
        }
    }

    /// `kernel_q8` (non-aarch64) for R rows x P consecutive panels (q: P x k x NR codes, sc, init: P x NR) on
    /// AVX-512, the PR sums independent; per output the same operations in the same order: the same bits.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
    pub unsafe fn q8p_avx512<const R: usize, const P: usize>(a: *const f32, k: usize, q: *const i8, sc: *const f32, init: *const f32, o: *mut f32, stride: usize) {
        // SAFETY: the caller's bounds (see the call).
        unsafe {
            let mut acc = [[_mm512_setzero_ps(); P]; R];
            for p in 0..k {
                let b: [__m512; P] = core::array::from_fn(|j| _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(q.add((j * k + p) * NR) as *const _))));
                for r in 0..R {
                    let x = _mm512_set1_ps(*a.add(r * k + p));
                    for j in 0..P {
                        acc[r][j] = _mm512_add_ps(acc[r][j], _mm512_mul_ps(x, b[j]));
                    }
                }
            }
            for r in 0..R {
                for j in 0..P {
                    let v = _mm512_add_ps(_mm512_loadu_ps(init.add(j * NR)), _mm512_mul_ps(acc[r][j], _mm512_loadu_ps(sc.add(j * NR))));
                    _mm512_storeu_ps(o.add(r * stride + j * NR), v);
                }
            }
        }
    }

    /// `q8p_avx512` (P = 4) on AVX2: every NR = 16 columns as two halves of 8.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn q8p_avx2<const R: usize>(a: *const f32, k: usize, q: *const i8, sc: *const f32, init: *const f32, o: *mut f32, stride: usize) {
        // SAFETY: the caller's bounds (see the call).
        unsafe {
            let mut acc = [[_mm256_setzero_ps(); 8]; R];
            for p in 0..k {
                let b: [__m256; 8] = core::array::from_fn(|h| _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(q.add(((h / 2) * k + p) * NR + 8 * (h % 2)) as *const _))));
                for r in 0..R {
                    let x = _mm256_set1_ps(*a.add(r * k + p));
                    for h in 0..8 {
                        acc[r][h] = _mm256_add_ps(acc[r][h], _mm256_mul_ps(x, b[h]));
                    }
                }
            }
            for r in 0..R {
                for h in 0..8 {
                    let v = _mm256_add_ps(_mm256_loadu_ps(init.add(8 * h)), _mm256_mul_ps(acc[r][h], _mm256_loadu_ps(sc.add(8 * h))));
                    _mm256_storeu_ps(o.add(r * stride + 8 * h), v);
                }
            }
        }
    }

    /// `int512` for two panels (w, w + pb) at once: 16 outputs per zmm, the activation broadcasts and the group ends
    /// shared. The same integer sums and float steps per output: the same bits.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn int512x2<const R: usize, const INT4: bool, const _V: bool>(act: &super::ActX, r0: usize, w: *const u8, pb: usize, sw: *const f32, swb: *const f32, g: usize, o: *mut f32, stride: usize, init: *const f32, c: usize) {
        let (k, ng) = (act.k, act.ng);
        let ks = if INT4 { 16 } else { 8 };
        // SAFETY: rows r0..r0 + R of act (the caller checks), two panels of w and their ng x 8 scales each.
        unsafe {
            let aq = act.q.as_ptr().add(r0 * k);
            let (m0f, m88, m80) = (_mm512_set1_epi8(0x0f), _mm512_set1_epi8(0x88u8 as i8), _mm512_set1_epi8(0x80u8 as i8));
            // the low and high 32-bit lane of every 64-bit lane, panel a then panel b
            let ev = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
            let od = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
            let mut f = [_mm512_setzero_ps(); R];
            for c in 0..ng {
                let mut acc = [[_mm512_setzero_si512(); 2]; R];
                // one or two rows: the blocks alternate between two sets of sums, so the vpdpbusd are not one chain
                // (integer sums: exact in any order)
                let mut acc2 = [[_mm512_setzero_si512(); 2]; R];
                macro_rules! blk {
                    ($b:expr, $acc:ident) => {{
                        let (b, acc) = ($b, &mut $acc);
                        let k0 = b * ks;
                        let a8 = |r: usize, kk: usize| _mm512_set1_epi64((aq.add(r * k + kk) as *const i64).read_unaligned());
                        let v = [_mm512_loadu_si512(w.add(b * 64) as *const _), _mm512_loadu_si512(w.add(pb + b * 64) as *const _)];
                        if INT4 {
                            let v = v.map(|v| _mm512_xor_si512(v, m88));
                            let lo = v.map(|v| _mm512_and_si512(v, m0f));
                            let hi = v.map(|v| _mm512_and_si512(_mm512_srli_epi16::<4>(v), m0f));
                            for r in 0..R {
                                let (x, y) = (a8(r, k0), a8(r, k0 + 8));
                                for p in 0..2 {
                                    acc[r][p] = _mm512_dpbusd_epi32(acc[r][p], lo[p], x);
                                    acc[r][p] = _mm512_dpbusd_epi32(acc[r][p], hi[p], y);
                                }
                            }
                        } else {
                            let u = v.map(|v| _mm512_xor_si512(v, m80));
                            for r in 0..R {
                                let x = a8(r, k0);
                                for p in 0..2 {
                                    acc[r][p] = _mm512_dpbusd_epi32(acc[r][p], u[p], x);
                                }
                            }
                        }
                    }};
                }
                let (b0, b1) = (c * g / ks, (c + 1) * g / ks);
                if R <= 2 {
                    let mut b = b0;
                    while b + 2 <= b1 {
                        blk!(b, acc);
                        blk!(b + 1, acc2);
                        b += 2;
                    }
                    if b < b1 {
                        blk!(b, acc);
                    }
                    for r in 0..R {
                        for p in 0..2 {
                            acc[r][p] = _mm512_add_epi32(acc[r][p], acc2[r][p]);
                        }
                    }
                } else {
                    for b in b0..b1 {
                        blk!(b, acc);
                    }
                }
                let sw16 = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(_mm256_loadu_ps(sw.add(c * 8))), _mm256_loadu_ps(swb.add(c * 8)));
                for r in 0..R {
                    let (x, y) = (acc[r][0], acc[r][1]);
                    let d = _mm512_add_epi32(_mm512_permutex2var_epi32(x, ev, y), _mm512_permutex2var_epi32(x, od, y));
                    let i = (r0 + r) * ng + c;
                    let (sum, sa) = (*act.sum.as_ptr().add(i), *act.s.as_ptr().add(i));
                    let d = _mm512_sub_epi32(d, _mm512_set1_epi32(if INT4 { 8 } else { 128 } * sum));
                    let d = if INT4 { _mm512_slli_epi32::<4>(d) } else { d };
                    f[r] = _mm512_add_ps(f[r], _mm512_mul_ps(_mm512_cvtepi32_ps(d), _mm512_mul_ps(_mm512_set1_ps(sa), sw16)));
                }
            }
            let mask = ((1u32 << c) - 1) as __mmask16;
            let b = _mm512_maskz_loadu_ps(mask, init);
            for r in 0..R {
                _mm512_mask_storeu_ps(o.add(r * stride), mask, _mm512_add_ps(f[r], b));
            }
        }
    }

    /// Group end: f + (d - offset x sum) (x 16 for int4) as f32 x (sa x sw).
    #[inline(always)]
    unsafe fn step<const INT4: bool>(f: __m256, d: __m256i, sum: i32, sa: f32, sw8: __m256) -> __m256 {
        // SAFETY: register-only AVX2.
        unsafe {
            let d = _mm256_sub_epi32(d, _mm256_set1_epi32(if INT4 { 8 } else { 128 } * sum));
            let d = if INT4 { _mm256_slli_epi32::<4>(d) } else { d };
            _mm256_add_ps(f, _mm256_mul_ps(_mm256_cvtepi32_ps(d), _mm256_mul_ps(_mm256_set1_ps(sa), sw8)))
        }
    }

    /// acc + the 4-byte dot products of unsigned w and signed a per 32-bit lane: AVX-VNNI `vpdpbusd`, or AVX2
    /// `vpmaddubsw` + `vpmaddwd` (exact while w x a pairs fit int16: w <= 15, |a| <= 127 for int4).
    #[inline(always)]
    unsafe fn dp<const VNNI: bool>(acc: __m256i, w: __m256i, a: __m256i) -> __m256i {
        // SAFETY: register-only; the caller enables avxvnni when VNNI.
        unsafe {
            if VNNI {
                _mm256_dpbusd_avx_epi32(acc, w, a)
            } else {
                _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(w, a), _mm256_set1_epi16(1)))
            }
        }
    }

    /// int8 weights on plain AVX2: w + 128 x a overflows `vpmaddubsw`'s int16, so |w| x (a with w's sign) instead
    /// (pairs <= 2 x 128 x 127 < 2^15: exact); `wa` = |w|.
    #[inline(always)]
    unsafe fn dp8(acc: __m256i, wa: __m256i, w: __m256i, a: __m256i) -> __m256i {
        // SAFETY: register-only AVX2.
        unsafe { _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(wa, _mm256_sign_epi8(a, w)), _mm256_set1_epi16(1))) }
    }

    /// R rows x one panel on AVX2 / AVX-VNNI: per row two ymm (outputs 0..4, 4..8) of lane pairs.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn int256<const R: usize, const INT4: bool, const VNNI: bool>(act: &super::ActX, r0: usize, w: *const u8, sw: *const f32, g: usize, o: *mut f32, stride: usize, init: *const f32, c: usize) {
        let (k, ng) = (act.k, act.ng);
        let ks = if INT4 { 16 } else { 8 };
        // SAFETY: as `int512`.
        unsafe {
            let aq = act.q.as_ptr().add(r0 * k);
            let (m0f, m88, m80) = (_mm256_set1_epi8(0x0f), _mm256_set1_epi8(0x88u8 as i8), _mm256_set1_epi8(0x80u8 as i8));
            let mut f = [_mm256_setzero_ps(); R];
            let ones = _mm256_set1_epi16(1);
            let nb = g / ks;
            for c in 0..ng {
                let mut acc = [[_mm256_setzero_si256(); 2]; R];
                if INT4 && !VNNI && nb <= 4 {
                    // the whole group in int16 sums: no int32 accumulator alive across blocks (16 ymm registers)
                    let mut s16 = [[_mm256_setzero_si256(); 2]; R];
                    for b in c * nb..(c + 1) * nb {
                        let k0 = b * ks;
                        for h in 0..2 {
                            let v = _mm256_loadu_si256(w.add(b * 64 + 32 * h) as *const _);
                            let lo = _mm256_and_si256(_mm256_xor_si256(v, m88), m0f);
                            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(_mm256_xor_si256(v, m88)), m0f);
                            for r in 0..R {
                                let x = _mm256_set1_epi64x((aq.add(r * k + k0) as *const i64).read_unaligned());
                                let y = _mm256_set1_epi64x((aq.add(r * k + k0 + 8) as *const i64).read_unaligned());
                                s16[r][h] = _mm256_add_epi16(s16[r][h], _mm256_add_epi16(_mm256_maddubs_epi16(lo, x), _mm256_maddubs_epi16(hi, y)));
                            }
                        }
                    }
                    for r in 0..R {
                        for h in 0..2 {
                            acc[r][h] = _mm256_madd_epi16(s16[r][h], ones);
                        }
                    }
                }
                let mut b = if INT4 && !VNNI && nb <= 4 { (c + 1) * nb } else { c * nb };
                while b < (c + 1) * nb {
                    // plain AVX2, int4: `vpmaddubsw` sums kept in int16 over up to 4 blocks (8 x 2 x 15 x 127 < 2^15)
                    let be = if INT4 && !VNNI { (b + 4).min((c + 1) * nb) } else { (c + 1) * nb };
                    let mut s16 = [[_mm256_setzero_si256(); 2]; R];
                    for b in b..be {
                        let k0 = b * ks;
                        let a8 = |r: usize, kk: usize| _mm256_set1_epi64x((aq.add(r * k + kk) as *const i64).read_unaligned());
                        for h in 0..2 {
                            let v = _mm256_loadu_si256(w.add(b * 64 + 32 * h) as *const _);
                            if INT4 {
                                let lo = _mm256_and_si256(_mm256_xor_si256(v, m88), m0f);
                                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(_mm256_xor_si256(v, m88)), m0f);
                                for r in 0..R {
                                    if VNNI {
                                        acc[r][h] = dp::<true>(acc[r][h], lo, a8(r, k0));
                                        acc[r][h] = dp::<true>(acc[r][h], hi, a8(r, k0 + 8));
                                    } else {
                                        let p = _mm256_add_epi16(_mm256_maddubs_epi16(lo, a8(r, k0)), _mm256_maddubs_epi16(hi, a8(r, k0 + 8)));
                                        s16[r][h] = _mm256_add_epi16(s16[r][h], p);
                                    }
                                }
                            } else if VNNI {
                                let u = _mm256_xor_si256(v, m80);
                                for r in 0..R {
                                    acc[r][h] = dp::<true>(acc[r][h], u, a8(r, k0));
                                }
                            } else {
                                let va = _mm256_abs_epi8(v);
                                for r in 0..R {
                                    acc[r][h] = dp8(acc[r][h], va, v, a8(r, k0));
                                }
                            }
                        }
                    }
                    if INT4 && !VNNI {
                        for r in 0..R {
                            for h in 0..2 {
                                acc[r][h] = _mm256_add_epi32(acc[r][h], _mm256_madd_epi16(s16[r][h], ones));
                            }
                        }
                    }
                    b = be;
                }
                let sw8 = _mm256_loadu_ps(sw.add(c * 8));
                for r in 0..R {
                    // lane pairs added: [o0 o1 o4 o5 | o2 o3 o6 o7], then the 64-bit lanes in order
                    let d = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_hadd_epi32(acc[r][0], acc[r][1]));
                    let i = (r0 + r) * ng + c;
                    // plain AVX2 int8 (`dp8`) works on the codes themselves: no offset
                    let sum = if !INT4 && !VNNI { 0 } else { *act.sum.as_ptr().add(i) };
                    f[r] = step::<INT4>(f[r], d, sum, *act.s.as_ptr().add(i), sw8);
                }
            }
            // + bias, the first c columns of rows r0.. (o at row r0)
            if c == 8 {
                let b = _mm256_loadu_ps(init);
                for r in 0..R {
                    _mm256_storeu_ps(o.add(r * stride), _mm256_add_ps(f[r], b));
                }
            } else {
                let mut t = [0f32; 8];
                for r in 0..R {
                    _mm256_storeu_ps(t.as_mut_ptr(), f[r]);
                    for j in 0..c {
                        *o.add(r * stride + j) = t[j] + *init.add(j);
                    }
                }
            }
        }
    }

    /// `kern::<rows, int4, extra...>(args)` for rows 1..=8.
    macro_rules! int_rows {
        ($rows:expr, $int4:expr, $kern:ident, $v:tt, $args:tt) => {
            int_rows!(@ $rows, $int4, $kern, $v, $args, 1 2 3 4 5 6 7 8)
        };
        (@ $rows:expr, $int4:expr, $kern:ident, $v:tt, $args:tt, $($r:literal)*) => {
            match ($rows, $int4) {
                $(($r, true) => $kern::<$r, true, $v> $args,
                ($r, false) => $kern::<$r, false, $v> $args,)*
                _ => unreachable!(),
            }
        };
    }

    /// Panels jp0..jp1 of q times all rows of `act` into o (row r at r * stride, panel jp at (jp - jp0) * 8, the first
    /// `cols` columns), + init (bias per output).
    #[allow(clippy::too_many_arguments)]
    pub fn int(kern: super::IntX86, q: &super::Quant, act: &super::ActX, init: &[f32], jp0: usize, jp1: usize, cols: usize, o: &mut [f32], stride: usize) {
        let m = act.rows();
        assert!(m == 0 || o.len() >= (m - 1) * stride + cols);
        // SAFETY: o holds the m rows of cols values (asserted).
        unsafe { int_raw(kern, q, act, init, jp0, jp1, cols, o.as_mut_ptr(), stride) }
    }

    /// `int` into a raw o: tasks writing disjoint columns of the same rows.
    ///
    /// # Safety
    /// o must be valid for writes of the m rows of `cols` values (row r at r * stride), not read or written by anyone
    /// else meanwhile.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn int_raw(kern: super::IntX86, q: &super::Quant, act: &super::ActX, init: &[f32], jp0: usize, jp1: usize, cols: usize, o: *mut f32, stride: usize) {
        let m = act.rows();
        let (ng, pb) = (q.k / q.group, q.panel_bytes());
        assert!(act.k == q.k && act.ng == ng && jp1 <= q.n.div_ceil(8) && init.len() >= jp1 * 8 && cols <= (jp1 - jp0) * 8);
        macro_rules! drive {
            ($name:ident, $feat:literal, $kern:ident, $v:tt, $mr:literal) => {{
                #[target_feature(enable = $feat)]
                #[allow(clippy::too_many_arguments)]
                fn $name(q: &super::Quant, act: &super::ActX, init: &[f32], jp0: usize, jp1: usize, cols: usize, o: *mut f32, stride: usize, m: usize, ng: usize, pb: usize) {
                    for jp in jp0..jp1 {
                        let c0 = (jp - jp0) * 8;
                        if c0 >= cols {
                            break;
                        }
                        let c = (cols - c0).min(8);
                        let mut i = 0;
                        while i < m {
                            let rows = (m - i).min($mr);
                            // SAFETY: rows i..i + rows of act, panel jp, c columns of `rows` rows of o from (i, c0)
                            // (asserted by `int`), init's 8 values from jp * 8.
                            unsafe {
                                let (w, sw) = (q.w.as_ptr().add(jp * pb), q.s.as_ptr().add(jp * ng * 8));
                                let (op, ip) = (o.add(i * stride + c0), init.as_ptr().add(jp * 8));
                                int_rows!(rows, q.int4, $kern, $v, (act, i, w, sw, q.group, op, stride, ip, c));
                            }
                            i += rows;
                        }
                    }
                }
                // SAFETY: `kern` was picked for this CPU (`int_x86`).
                unsafe { $name(q, act, init, jp0, jp1, cols, o, stride, m, ng, pb) }
            }};
        }
        match kern {
            super::IntX86::Avx512Vnni => {
                #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx2,fma")]
                #[allow(clippy::too_many_arguments)]
                fn d512(q: &super::Quant, act: &super::ActX, init: &[f32], jp0: usize, jp1: usize, cols: usize, o: *mut f32, stride: usize, m: usize, ng: usize, pb: usize) {
                    const MR: usize = 6;
                    let mut jp = jp0;
                    while jp < jp1 {
                        let c0 = (jp - jp0) * 8;
                        if c0 >= cols {
                            break;
                        }
                        // two panels when both are in range
                        let two = jp + 1 < jp1 && c0 + 8 < cols;
                        let c = (cols - c0).min(if two { 16 } else { 8 });
                        let mut i = 0;
                        while i < m {
                            let rows = (m - i).min(if two { MR } else { 8 });
                            // SAFETY: rows i..i + rows of act, panels jp (and jp + 1), c columns of `rows` rows of o
                            // from (i, c0) (asserted by `int`), the c values of init from jp * 8.
                            unsafe {
                                let (w, sw) = (q.w.as_ptr().add(jp * pb), q.s.as_ptr().add(jp * ng * 8));
                                let (op, ip) = (o.add(i * stride + c0), init.as_ptr().add(jp * 8));
                                if two {
                                    int_rows!(rows, q.int4, int512x2, true, (act, i, w, pb, sw, sw.add(ng * 8), q.group, op, stride, ip, c));
                                } else {
                                    int_rows!(rows, q.int4, int512, true, (act, i, w, sw, q.group, op, stride, ip, c));
                                }
                            }
                            i += rows;
                        }
                        jp += if two { 2 } else { 1 };
                    }
                }
                // SAFETY: `kern` was picked for this CPU (`int_x86`).
                unsafe { d512(q, act, init, jp0, jp1, cols, o, stride, m, ng, pb) }
            }
            super::IntX86::AvxVnni => drive!(dvnni, "avxvnni,avx2,fma", int256, true, 4),
            super::IntX86::Avx2 => drive!(d256, "avx2,fma", int256, false, 4),
        }
    }

    /// `super::widen8` on AVX2: every 64-byte block (8 outputs x 8 bytes, in pairs) transposed in registers to 8
    /// vectors of one byte of each output, widened to f32 and scaled (the same product: the same bits).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn widen8(int4: bool, w: &[u8], s: &[f32], k: usize, g: usize, out: &mut [f32], c0: usize) {
        let ks = if int4 { 16 } else { 8 };
        assert!(w.len() >= k / ks * 64 && s.len() >= k / g * 8 && out.len() >= k * NR && c0 + 8 <= NR);
        // SAFETY: block b reads w[64b..64b + 64] and the 8 scales of its group, writes rows k0..k0 + ks of out (asserted).
        unsafe {
            let mask = _mm_set1_epi8(0xf0u8 as i8);
            let op = out.as_mut_ptr().add(c0);
            for b in 0..k / ks {
                let k0 = b * ks;
                let sc = _mm256_loadu_ps(s.as_ptr().add(k0 / g * 8));
                let t = transpose(w.as_ptr().add(b * 64));
                for i in 0..8 {
                    let v = if i % 2 == 0 { t[i / 2] } else { _mm_unpackhi_epi64(t[i / 2], t[i / 2]) };
                    let f = |v: __m128i| _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)), sc);
                    if int4 {
                        _mm256_storeu_ps(op.add((k0 + i) * NR), f(_mm_and_si128(_mm_slli_epi16::<4>(v), mask)));
                        _mm256_storeu_ps(op.add((k0 + 8 + i) * NR), f(_mm_and_si128(v, mask)));
                    } else {
                        _mm256_storeu_ps(op.add((k0 + i) * NR), f(v));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    /// Every integer kernel this CPU has agrees with the f32 product of the dequantized weights (within int8
    /// activation rounding) and with the portable one (to float rounding).
    #[test]
    fn quant_kernels() {
        let mut seed = 1;
        for &(m, n, k, g, int4) in &[(1, 8, 64, 64, true), (5, 19, 128, 64, true), (33, 40, 512, 64, true), (3, 13, 512, 512, false), (36, 24, 96, 32, false), (2, 7, 32, 16, true)] {
            let codes: Vec<i8> = (0..n * k).map(|_| (rnd(&mut seed) * if int4 { 8.0 } else { 127.0 }).floor().clamp(if int4 { -8.0 } else { -127.0 }, if int4 { 7.0 } else { 127.0 }) as i8).collect();
            let scales: Vec<f32> = (0..n * k / g).map(|_| 0.01 + rnd(&mut seed).abs() * 0.05).collect();
            let q = Quant::with_layout(n, k, g, int4, Lay::Pair, |j, p| codes[j * k + p], |j, c| scales[j * (k / g) + c]);
            let a: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed)).collect();
            let bias: Vec<f32> = (0..n).map(|_| rnd(&mut seed)).collect();
            let dense: Vec<f32> = (0..n * k).map(|i| codes[i] as f32 * scales[(i / k) * (k / g) + (i % k) / g]).collect();
            let want_f32 = gemm(&a, m, &Packed::from_rows(&dense, n, k), Some(&bias));
            // the exact product with the activations quantized as the kernels do
            let mut want = vec![0f32; m * n];
            for r in 0..m {
                for j in 0..n {
                    let mut f = bias[j] as f64;
                    for c in 0..k / g {
                        let x = &a[r * k + c * g..r * k + (c + 1) * g];
                        let max = x.iter().fold(0f32, |m, v| m.max(v.abs()));
                        let inv = if max > 0.0 { 127.0 / max } else { 0.0 };
                        let d: i64 = (0..g).map(|p| (x[p] * inv).round() as i64 * codes[j * k + c * g + p] as i64).sum();
                        f += d as f64 * (max / 127.0) as f64 * scales[j * (k / g) + c] as f64;
                    }
                    want[r * n + j] = f as f32;
                    // and that is close to the f32 product
                    assert!((want[r * n + j] - want_f32[r * n + j]).abs() < 0.05 * (k as f32).sqrt() * if int4 { 0.1 } else { 1.0 });
                }
            }
            let mut paths = vec![Path::Scalar];
            #[cfg(target_arch = "aarch64")]
            {
                paths.push(Path::Neon);
                if std::arch::is_aarch64_feature_detected!("i8mm") {
                    paths.push(Path::Mmla);
                }
            }
            let mut first: Option<Vec<f32>> = None;
            for &path in &paths {
                let act = pack_act(&a, m, k, g);
                let lanes = &MMLA_LANES;
                let mut got = vec![0f32; m * n];
                let pb = k * if int4 { 4 } else { 8 };
                let ng = k / g;
                for p in 0..m.div_ceil(2) {
                    for jp in 0..n.div_ceil(8) {
                        let mut t = [[[0f32; 4]; 4]; RPMAX];
                        tile(path, 1, int4, &act.q[p * 2 * k..], &act.s[p * ng * 4..], &q.w[jp * pb..], &q.s[jp * ng * 8..], k, g, &mut t);
                        for pp in 0..4 {
                            for (l, &(ri, oi)) in lanes.iter().enumerate() {
                                let (r, o) = (2 * p + ri, jp * 8 + 2 * pp + oi);
                                if r < m && o < n {
                                    got[r * n + o] = t[0][pp][l] + bias[o];
                                }
                            }
                        }
                    }
                }
                // the 2-row-pair kernels, through the driver
                let full = if path == super::path() { Some(q.gemm(&a, m, Some(&bias))) } else { None };
                for i in 0..m * n {
                    let tol = 1e-3 * (1.0 + want[i].abs());
                    assert!((got[i] - want[i]).abs() < tol, "{path:?} m{m} n{n} k{k} g{g} int4 {int4} [{i}]: {} vs {}", got[i], want[i]);
                    if let Some(f) = &full {
                        assert!((f[i] - got[i]).abs() < 1e-4 * (1.0 + got[i].abs()), "{path:?} driver [{i}]");
                    }
                    if let Some(f) = &first {
                        assert!((f[i] - got[i]).abs() < 1e-4 * (1.0 + got[i].abs()), "{path:?} vs scalar [{i}]: {} vs {}", got[i], f[i]);
                    }
                }
                first.get_or_insert(got);
            }
            // the `sdot` layout, through its driver (dotprod CPUs), and its f32 view
            let q4 = Quant::with_layout(n, k, g, int4, Lay::Dot4, |j, p| codes[j * k + p], |j, c| scales[j * (k / g) + c]);
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                let got = q4.gemm4(&a, m, Some(&bias), false);
                let f = first.as_ref().unwrap();
                for i in 0..m * n {
                    assert!((got[i] - want[i]).abs() < 1e-3 * (1.0 + want[i].abs()), "sdot m{m} n{n} k{k} [{i}]: {} vs {}", got[i], want[i]);
                    assert!((got[i] - f[i]).abs() < 1e-4 * (1.0 + f[i].abs()), "sdot vs scalar [{i}]");
                }
            }
            let mut scratch4 = vec![0f32; k * NR];
            for jp in 0..n.div_ceil(NR) {
                let pnl = Panels::panel(&q4, jp, &mut scratch4).to_vec();
                for l in 0..NR.min(n - jp * NR) {
                    for p in 0..k {
                        assert_eq!(pnl[p * NR + l], dense[(jp * NR + l) * k + p]);
                    }
                }
            }
            // the f32 view of the packed weights is the dense matrix exactly
            let mut scratch = vec![0f32; k * NR];
            for jp in 0..n.div_ceil(NR) {
                let pnl = Panels::panel(&q, jp, &mut scratch).to_vec();
                for l in 0..NR.min(n - jp * NR) {
                    for p in 0..k {
                        assert_eq!(pnl[p * NR + l], dense[(jp * NR + l) * k + p]);
                    }
                }
            }
        }
    }

    /// The x86-64 integer kernels give the bits of the portable integer product (`tile_scalar`).
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn x86_int_same_bits() {
        if x86() < X86::Avx2 {
            return;
        }
        let mut kerns = vec![IntX86::Avx2];
        if std::arch::is_x86_feature_detected!("avxvnni") {
            kerns.push(IntX86::AvxVnni);
        }
        if x86() >= X86::Avx512 && std::arch::is_x86_feature_detected!("avx512vnni") {
            kerns.push(IntX86::Avx512Vnni);
        }
        let mut seed = 13;
        for &(m, n, k, g, int4) in &[(1, 8, 64, 64, true), (5, 19, 128, 64, true), (33, 40, 512, 64, true), (3, 13, 512, 512, false), (36, 24, 96, 32, false), (2, 7, 32, 16, true), (17, 70, 1152, 1152, true), (9, 16, 64, 32, false), (1, 32, 512, 512, false), (2, 24, 1024, 64, true), (1, 40, 96, 32, false), (2, 16, 48, 48, true)] {
            let codes: Vec<i8> = (0..n * k).map(|_| (rnd(&mut seed) * if int4 { 8.0 } else { 128.0 }).floor().clamp(if int4 { -8.0 } else { -128.0 }, if int4 { 7.0 } else { 127.0 }) as i8).collect();
            let scales: Vec<f32> = (0..n * k / g).map(|_| 0.01 + rnd(&mut seed).abs() * 0.05).collect();
            let q = Quant::new(n, k, g, int4, |j, p| codes[j * k + p], |j, c| scales[j * (k / g) + c]);
            let a: Vec<f32> = (0..m * k).map(|i| if i % 97 == 3 { 0.0 } else { rnd(&mut seed) * 3.0 }).collect();
            let bias: Vec<f32> = (0..n).map(|_| rnd(&mut seed)).collect();
            // the portable product, kernel call by kernel call
            let act = pack_act(&a, m, k, g, false);
            let (pb, ng) = (q.panel_bytes(), k / g);
            let mut want = vec![0f32; m * n];
            for p in 0..m.div_ceil(2) {
                for jp in 0..n.div_ceil(8) {
                    let mut t = [[[0f32; 4]; 4]; RPMAX];
                    tile_scalar(1, int4, &act.q[p * 2 * k..], &act.s[p * ng * 4..], &q.w[jp * pb..], &q.s[jp * ng * 8..], k, g, &mut t);
                    for pp in 0..4 {
                        for (l, &(ri, oi)) in MMLA_LANES.iter().enumerate() {
                            let (r, o) = (2 * p + ri, jp * 8 + 2 * pp + oi);
                            if r < m && o < n {
                                want[r * n + o] = t[0][pp][l] + bias[o];
                            }
                        }
                    }
                }
            }
            let ax = ActX::new(&a, m, k, g);
            let np = n.div_ceil(8);
            let mut init = vec![0f32; np * 8];
            init[..n].copy_from_slice(&bias);
            for &kern in &kerns {
                let mut got = vec![0f32; m * n];
                x86::int(kern, &q, &ax, &init, 0, np, n, &mut got, n);
                for i in 0..m * n {
                    assert_eq!(got[i].to_bits(), want[i].to_bits(), "{kern:?} m{m} n{n} k{k} g{g} int4 {int4} [{i}]: {} vs {}", got[i], want[i]);
                }
            }
        }
    }

    /// x86-64 activation codes are `quant8`'s, ties (half away from zero) and signed zeros included.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[allow(unsafe_code)]
    fn x86_act_codes() {
        let mut seed = 21;
        // max 127: inv 1, so v * inv is v exactly
        let mut a: Vec<f32> = vec![127.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 126.5, -126.5, 0.49999997, -0.49999997, 3.0, -127.0, 0.0, 64.5];
        a.extend((0..48).map(|_| rnd(&mut seed) * 127.0));
        let ax = ActX::new(&a, 1, 64, 16);
        for c in 0..4 {
            let x = &a[c * 16..(c + 1) * 16];
            let max = absmax(x);
            let inv = if max > 0.0 { 127.0 / max } else { 0.0 };
            let mut want = [0i8; 16];
            quant8(&x[..8], inv, &mut want[..8]);
            quant8(&x[8..], inv, &mut want[8..]);
            assert_eq!(&ax.q[c * 16..(c + 1) * 16], &want, "group {c}");
            assert_eq!(ax.sum[c], want.iter().map(|&v| v as i32).sum::<i32>());
        }
        // both x86 routines against `quant8` per group: codes, sums, scales
        for &(m, k, g) in &[(1, 64, 16), (3, 384, 64), (5, 512, 512), (2, 96, 32), (4, 1152, 1152)] {
            let a: Vec<f32> = (0..m * k).map(|i| if i % 37 == 5 { 0.0 } else if i % 41 == 7 { -0.0 } else { rnd(&mut seed) * if i % 3 == 0 { 0.01 } else { 4.0 } }).collect();
            let ng = k / g;
            for lvl in [X86::Avx2, X86::Avx512] {
                if x86() < lvl {
                    continue;
                }
                let (mut q, mut s, mut sum) = (vec![0i8; m * k], vec![0f32; m * ng], vec![0i32; m * ng]);
                unsafe {
                    if lvl == X86::Avx512 { x86::act_avx512(&a, m, k, g, &mut q, &mut s, &mut sum) } else { x86::act_avx2(&a, m, k, g, &mut q, &mut s, &mut sum) }
                }
                for i in 0..m * ng {
                    let x = &a[i * g..(i + 1) * g];
                    let max = absmax(x);
                    let inv = if max > 0.0 { 127.0 / max } else { 0.0 };
                    let mut want = vec![0i8; g];
                    for (o, x) in want.chunks_exact_mut(8).zip(x.chunks_exact(8)) {
                        quant8(x, inv, o);
                    }
                    assert_eq!(&q[i * g..(i + 1) * g], &want[..], "{lvl:?} m{m} k{k} g{g} group {i}");
                    assert_eq!(sum[i], want.iter().map(|&v| v as i32).sum::<i32>());
                    assert_eq!(s[i].to_bits(), (max / 127.0).to_bits());
                }
            }
        }
    }

    /// `exp` gives the bits of its plain form (floor, then a float -> int cast) over the whole range.
    #[test]
    #[allow(clippy::approx_constant)] // the polynomial's coefficients, as in `exp`
    fn exp_same_bits() {
        let plain = |x: f32| {
            let t = (x * core::f32::consts::LOG2_E).clamp(-126.0, 126.0);
            let i = t.floor();
            let f = t - i;
            let p = 1.0 + f * (0.693_147_2 + f * (0.240_226_5 + f * (0.055_504_11 + f * (0.009_618_129 + f * 0.001_333_355))));
            f32::from_bits(((i as i32 + 127) as u32) << 23) * p
        };
        let mut seed = 9;
        let special = [0.0, -0.0, 1.0, -1.0, 87.0, -87.0, 88.7, -88.0, 1e30, -1e30, f32::MAX, f32::MIN, f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 1e-40, f32::INFINITY, f32::NEG_INFINITY];
        let xs = special.into_iter().chain((0..2_000_000).map(|i| if i % 2 == 0 { rnd(&mut seed) * 100.0 } else { f32::from_bits((i as u32).wrapping_mul(2_654_435_761)) }));
        for x in xs {
            if x.is_nan() {
                continue;
            }
            assert_eq!(exp(x).to_bits(), plain(x).to_bits(), "{x}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn swoosh_neon_matches_scalar() {
        let mut seed = 3;
        let x: Vec<f32> = (0..4096).map(|i| if i < 64 { (i as f32 - 32.0) * 3.0 } else { rnd(&mut seed) * 20.0 }).collect();
        for (shift, c) in [(4.0, 0.035), (1.0, 0.313_261_7)] {
            let mut y = x.clone();
            swoosh4(&mut y, shift, c);
            for (a, &v) in y.iter().zip(&x) {
                let sp = (v - shift).max(0.0) + ln(1.0 + exp(-(v - shift).abs()));
                let want = sp - 0.08 * v - c;
                assert!((a - want).abs() <= 4e-7 * (1.0 + want.abs()), "{v}: {a} vs {want}");
            }
        }
    }

    /// Any number of threads gives the same bits (row blocks, column groups, the few-rows paths).
    #[cfg(feature = "std")]
    #[test]
    fn threads_same_bits() {
        use crate::par::{Exec, Pool, enter};
        let ex = std::sync::Arc::new(Exec::Pool(Pool::new(3)));
        let mut seed = 11;
        for &(m, n, k, g, int4) in &[(1, 2080, 512, 512, false), (4, 2048, 512, 512, false), (16, 1536, 512, 64, true), (300, 768, 256, 64, true), (77, 384, 128, 128, false)] {
            let codes: Vec<i8> = (0..n * k).map(|_| (rnd(&mut seed) * 7.9).floor() as i8).collect();
            let sc: Vec<f32> = (0..n * k / g).map(|_| 0.01 + rnd(&mut seed).abs() * 0.05).collect();
            let q = Quant::with_layout(n, k, g, int4, Lay::Pair, |j, p| codes[j * k + p], |j, c| sc[j * (k / g) + c]);
            let qf = QuantF::new_as(n, k, false, |j, p| codes[j * k + p], |j| sc[j * (k / g)]);
            if cfg!(target_arch = "aarch64") && path() != Path::F32 {
                let qw = QuantF::new_as(n, k, true, |j, p| codes[j * k + p], |j| sc[j * (k / g)]);
                assert!(qw.gemm(&a_w(m, k), m, None) == enter(Some(&ex), || qw.gemm(&a_w(m, k), m, None)), "wide m{m} n{n} k{k}");
            }
            let a: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed)).collect();
            let bias: Vec<f32> = (0..n).map(|_| rnd(&mut seed)).collect();
            let one = (q.gemm(&a, m, Some(&bias)), qf.gemm(&a, m, Some(&bias)), gemm(&a, m, &Packed::from_rows(&qf_dense(&qf), n, k), None));
            let three = enter(Some(&ex), || (q.gemm(&a, m, Some(&bias)), qf.gemm(&a, m, Some(&bias)), gemm(&a, m, &Packed::from_rows(&qf_dense(&qf), n, k), None)));
            assert!(one == three, "m{m} n{n} k{k}");
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                let q4 = Quant::with_layout(n, k, g, int4, Lay::Dot4, |j, p| codes[j * k + p], |j, c| sc[j * (k / g) + c]);
                let (one, three) = (q4.gemm4(&a, m, Some(&bias), false), enter(Some(&ex), || q4.gemm4(&a, m, Some(&bias), false)));
                assert!(one == three, "sdot m{m} n{n} k{k}");
                // and the same bits as the pair layout's integer kernels
                assert!(one == q.gemm(&a, m, Some(&bias)), "sdot vs pair layout m{m} n{n} k{k}");
            }
        }
    }

    #[cfg(feature = "std")]
    fn a_w(m: usize, k: usize) -> Vec<f32> {
        let mut seed = 5;
        (0..m * k).map(|_| rnd(&mut seed)).collect()
    }

    /// 16-bit activations on the integer kernels against the f32 product of the same int8 weights.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn quantf_wide_matches_f32() {
        if path() == Path::F32 {
            return;
        }
        let mut seed = 3;
        for &(m, n, k) in &[(1, 37, 64), (4, 2080, 512), (3, 5, 16), (9, 33, 96), (70, 192, 2432)] {
            let codes: Vec<i8> = (0..n * k).map(|_| (rnd(&mut seed) * 127.0) as i8).collect();
            let sc: Vec<f32> = (0..n).map(|_| 0.01 + rnd(&mut seed).abs()).collect();
            let dense: Vec<f32> = (0..n * k).map(|i| codes[i] as f32 * sc[i / k]).collect();
            let a: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed) * 3.0).collect();
            let bias: Vec<f32> = (0..n).map(|_| rnd(&mut seed)).collect();
            let q = QuantF::new_as(n, k, true, |j, p| codes[j * k + p], |j| sc[j]);
            let (got, want) = (q.gemm(&a, m, Some(&bias)), gemm(&a, m, &Packed::from_rows(&dense, n, k), Some(&bias)));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                // per activation: half a code of 3 / 32512; times |w| <= 127 sc, sqrt(k) of them (random signs), x8 margin
                let tol = 8.0 * 0.5 * 3.0 / 32512.0 * 127.0 * sc[i % n] * (k as f32).sqrt() + 1e-4 * w.abs();
                assert!((g - w).abs() < tol, "m{m} n{n} k{k}: {g} vs {w} (tol {tol})");
            }
        }
    }

    #[cfg(feature = "std")]
    fn qf_dense(q: &QuantF) -> Vec<f32> {
        let mut s = vec![0f32; q.k * NR];
        let mut d = vec![0f32; q.n * q.k];
        for jp in 0..q.n.div_ceil(NR) {
            let p = Panels::panel(q, jp, &mut s);
            for l in 0..NR.min(q.n - jp * NR) {
                for kk in 0..q.k {
                    d[(jp * NR + l) * q.k + kk] = p[kk * NR + l];
                }
            }
        }
        d
    }

    /// Every x86-64 level gives the bits of the plain Rust (float micro-kernels, integer panels widened).
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    #[allow(unsafe_code)]
    fn x86_levels_same_bits() {
        let mut seed = 5;
        let lvl = x86();
        for &(m, k) in &[(1, 16), (5, 64), (13, 512), (64, 96), (7, 1536), (30, 48)] {
            let a: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed)).collect();
            let panel: Vec<f32> = (0..k * NR).map(|_| rnd(&mut seed)).collect();
            let init: [f32; NR] = core::array::from_fn(|_| rnd(&mut seed));
            for cols in [16, 5] {
                let stride = 21;
                let mut want = vec![0f32; m * stride];
                rows(&a, k, &panel, &init, &mut want, stride, cols);
                let mut got = vec![0f32; m * stride];
                if lvl >= X86::Avx2 {
                    unsafe { x86::rows_avx2(&a, k, &panel, &init, &mut got, stride, cols) };
                    assert!(got == want, "avx2 m{m} k{k}");
                }
                if lvl >= X86::Avx512 {
                    got.fill(0.0);
                    unsafe { x86::rows_avx512(&a, k, &panel, &init, &mut got, stride, cols) };
                    assert!(got == want, "avx512 m{m} k{k}");
                }
            }
        }
        // integer weights widened in registers: the bits of the f32 panels
        for &(m, n, k, g, int4) in &[(1, 8, 64, 64, true), (5, 19, 128, 64, true), (16, 40, 512, 64, true), (3, 13, 512, 512, false), (8, 24, 96, 32, false), (2, 7, 32, 16, true), (11, 1536, 256, 128, true)] {
            let codes: Vec<i8> = (0..n * k).map(|_| (rnd(&mut seed) * if int4 { 8.0 } else { 127.0 }).floor().clamp(if int4 { -8.0 } else { -127.0 }, if int4 { 7.0 } else { 127.0 }) as i8).collect();
            let scales: Vec<f32> = (0..n * k / g).map(|_| 0.01 + rnd(&mut seed).abs() * 0.05).collect();
            let q = Quant::new(n, k, g, int4, |j, p| codes[j * k + p], |j, c| scales[j * (k / g) + c]);
            let a: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed)).collect();
            let bias: Vec<f32> = (0..n).map(|_| rnd(&mut seed)).collect();
            let mut want = vec![0f32; m * n];
            blocked(&a, &q, Some(&bias), &mut want);
            let np = n.div_ceil(8);
            let mut init = vec![0f32; np.div_ceil(2) * 16];
            init[..n].copy_from_slice(&bias);
            let w = np.div_ceil(2) * 16;
            for l in [X86::Avx2, X86::Avx512] {
                if lvl < l {
                    continue;
                }
                let mut o = vec![0f32; m * w];
                unsafe {
                    if l == X86::Avx2 { x86::quant_avx2(&q, &a, m, &init, 0, np, &mut o, w) } else { x86::quant_avx512(&q, &a, m, &init, 0, np, &mut o, w) }
                }
                for r in 0..m {
                    assert!(o[r * w..r * w + n] == want[r * n..(r + 1) * n], "{l:?} m{m} n{n} k{k} g{g} int4 {int4}");
                }
            }
            if lvl >= X86::Avx2 && m <= SMALL && path() == Path::F32 {
                assert!(q.gemm(&a, m, Some(&bias)) == want, "gemm_small m{m} n{n}");
            }
        }
        for &(k, g, int4) in &[(64, 64, true), (512, 32, true), (96, 32, false), (512, 512, false), (16, 16, true)] {
            let w: Vec<u8> = (0..k * 8).map(|_| (rnd(&mut seed) * 1000.0) as i32 as u8).collect();
            let s: Vec<f32> = (0..k / g * 8).map(|_| rnd(&mut seed)).collect();
            for c0 in [0, 8] {
                let mut want = vec![0f32; k * NR];
                widen8_plain(int4, &w, &s, k, g, &mut want, c0);
                if lvl >= X86::Avx2 {
                    let mut got = vec![0f32; k * NR];
                    unsafe { x86::widen8(int4, &w, &s, k, g, &mut got, c0) };
                    assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "widen8 k{k} g{g} int4 {int4}");
                }
            }
        }
    }

    #[test]
    fn quantf_matches_f32() {
        let mut seed = 7;
        for &(m, n, k) in &[(1, 37, 64), (4, 16, 512), (3, 5, 8), (9, 33, 100), (70, 20, 48)] {
            let codes: Vec<i8> = (0..n * k).map(|_| (rnd(&mut seed) * 127.0) as i8).collect();
            let sc: Vec<f32> = (0..n).map(|_| 0.01 + rnd(&mut seed).abs()).collect();
            let q = QuantF::new_as(n, k, false, |j, p| codes[j * k + p], |j| sc[j]);
            let dense: Vec<f32> = (0..n * k).map(|i| codes[i] as f32 * sc[i / k]).collect();
            let a: Vec<f32> = (0..m * k).map(|_| rnd(&mut seed)).collect();
            let bias: Vec<f32> = (0..n).map(|_| rnd(&mut seed)).collect();
            let (got, want) = (q.gemm(&a, m, Some(&bias)), gemm(&a, m, &Packed::from_rows(&dense, n, k), Some(&bias)));
            for (g, w) in got.iter().zip(&want) {
                if m > 8 {
                    assert_eq!(g, w);
                } else {
                    assert!((g - w).abs() < 1e-5 * (1.0 + w.abs()) * k as f32, "{g} vs {w}");
                }
            }
        }
    }
}
