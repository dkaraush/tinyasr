//! Float kernels. The matrix-multiply micro-kernel is explicit NEON on aarch64; elsewhere it is
//! written so LLVM auto-vectorizes it (SIMD128 on wasm, SSE on x86-64).

use alloc::vec;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use crate::compat::F32Ext;

// Micro-kernel tile: MR rows x NR columns of accumulators. aarch64 has 32 vector registers
// (4x16 = 16 accumulators); wasm SIMD / x86-64 SSE have 16 (2x16 = 8). Measured, see README.
const NR: usize = 16; // output columns per panel
#[cfg(target_arch = "aarch64")]
const MR: usize = 4; // output rows per micro-kernel call
#[cfg(not(target_arch = "aarch64"))]
const MR: usize = 2;
const MB: usize = 64; // rows per cache block

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
    let mut out = vec![0.0; m * n];
    gemm_rows(&a[..m * k], b, bias, &mut out);
    out
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
}

/// The cache-blocked loop around the micro-kernel: blocks of MB rows of `a`, one weight panel at a time.
fn blocked(a: &[f32], b: &impl Panels, bias: Option<&[f32]>, out: &mut [f32]) {
    let (n, k) = (b.n(), b.k());
    let m = a.len() / k;
    let mut tile = [[0f32; NR]; MR];
    let mut scratch = vec![0f32; k * NR];
    for i0 in (0..m).step_by(MB) {
        let i1 = (i0 + MB).min(m);
        for jp in 0..n.div_ceil(NR) {
            let panel = b.panel(jp, &mut scratch);
            let j0 = jp * NR;
            let cols = NR.min(n - j0);
            let mut init = [0f32; NR];
            if let Some(bias) = bias {
                init[..cols].copy_from_slice(&bias[j0..j0 + cols]);
            }
            let mut i = i0;
            while i < i1 {
                let rows = MR.min(i1 - i);
                if rows == MR {
                    kernel::<MR>(&a[i * k..(i + MR) * k], k, panel, &init, &mut tile);
                } else {
                    for r in 0..rows {
                        let mut one = [[0f32; NR]; 1];
                        kernel::<1>(&a[(i + r) * k..(i + r + 1) * k], k, panel, &init, &mut one);
                        tile[r] = one[0];
                    }
                }
                for r in 0..rows {
                    out[(i + r) * n + j0..(i + r) * n + j0 + cols].copy_from_slice(&tile[r][..cols]);
                }
                i += rows;
            }
        }
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
    use core::arch::aarch64::{float32x4_t, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};
    const { assert!(NR == 16) };
    assert!(a.len() >= R * k && panel.len() >= k * NR);
    // SAFETY: NEON is mandatory on aarch64; every load/store stays inside the slices checked above
    // (row r reads a[r*k + p], p < k; panel reads panel[p*16 .. p*16+16], p < k).
    unsafe {
        let ip = init.as_ptr();
        let start: [float32x4_t; 4] = [vld1q_f32(ip), vld1q_f32(ip.add(4)), vld1q_f32(ip.add(8)), vld1q_f32(ip.add(12))];
        let mut acc = [start; R];
        let (ap, bp) = (a.as_ptr(), panel.as_ptr());
        for p in 0..k {
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

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let k = a.len().min(b.len());
    let k8 = k / 8 * 8;
    let (x, y) = (&a[..k8], &b[..k8]);
    let mut acc = [0f32; 8];
    for c in 0..k8 / 8 {
        for l in 0..8 {
            acc[l] += x[c * 8 + l] * y[c * 8 + l];
        }
    }
    acc.iter().sum::<f32>() + (k8..k).map(|p| a[p] * b[p]).sum::<f32>()
}

/// exp for x <= ~88 via 2^x = 2^i * 2^f with a degree-5 polynomial (rel. err ~2e-7).
/// Branch-free so loops over it vectorize, unlike the libm call.
#[inline(always)]
#[allow(clippy::approx_constant)] // polynomial coefficients, not ln 2
pub fn exp(x: f32) -> f32 {
    let t = (x * core::f32::consts::LOG2_E).clamp(-126.0, 126.0);
    let i = t.floor();
    let f = t - i;
    let p = 1.0 + f * (0.693_147_2 + f * (0.240_226_5 + f * (0.055_504_11 + f * (0.009_618_129 + f * 0.001_333_355))));
    f32::from_bits(((i as i32 + 127) as u32) << 23) * p
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

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    1.0 - 2.0 / (exp(2.0 * x) + 1.0)
}

/// Integer weights with one f32 scale per `group` consecutive weights of a row (group = k: per row).
pub struct Quant {
    pub n: usize,
    pub k: usize,
    group: usize,
    scale: Vec<f32>, // n * k / group
    q: Vec<i8>,      // n x k
}

impl Quant {
    pub fn new(n: usize, k: usize, group: usize, scale: Vec<f32>, q: Vec<i8>) -> Quant {
        assert!(group > 0 && k % group == 0 && scale.len() == n * k / group && q.len() == n * k);
        Quant { n, k, group, scale, q }
    }

    /// out (m x n) = a (m x k) * W^T (+ bias).
    ///
    /// With the dot-product extension (aarch64 `sdot`), the activations are quantized to int8 per row and per
    /// weight group and multiplied by the integer weights directly (`INT8_DOT=0` turns this off). Otherwise the
    /// weights are widened to f32 one panel of NR rows at a time and reused for a block of MB rows of `a` by the
    /// float micro-kernel.
    fn gemm(&self, a: &[f32], m: usize, bias: Option<&[f32]>) -> Vec<f32> {
        assert!(a.len() >= m * self.k);
        if self.group % 16 == 0 && int8_dot() {
            return self.gemm_i8(&a[..m * self.k], m, bias);
        }
        let mut out = vec![0.0; m * self.n];
        blocked(&a[..m * self.k], self, bias, &mut out);
        out
    }

    fn gemm_i8(&self, a: &[f32], m: usize, bias: Option<&[f32]>) -> Vec<f32> {
        let (n, k, g) = (self.n, self.k, self.group);
        let (aq, sa) = quantize_rows(a, k, g);
        let mut out = vec![0.0; m * n];
        self.cols_i8(&aq, &sa, m, &mut out);
        if let Some(b) = bias {
            for row in out.chunks_exact_mut(n) {
                for (o, b) in row.iter_mut().zip(b) {
                    *o += b;
                }
            }
        }
        out
    }

    /// The int8 product (m x n) into `out`.
    fn cols_i8(&self, aq: &[i8], sa: &[f32], m: usize, out: &mut [f32]) {
        let (k, g, n) = (self.k, self.group, self.n);
        let ng = k / g;
        let mut j = 0;
        while j < n {
            let c = if n - j >= TC { TC } else { 1 };
            let (wq, sw) = (&self.q[j * k..(j + c) * k], &self.scale[j * ng..(j + c) * ng]);
            let mut i = 0;
            while i < m {
                let r = if m - i >= TR { TR } else { 1 };
                let (aq, sa) = (&aq[i * k..(i + r) * k], &sa[i * ng..(i + r) * ng]);
                let o = &mut out[i * n + j..];
                match (r == TR, c == TC) {
                    (true, true) => tile_i8::<TR, TC>(aq, sa, wq, sw, k, g, o, n),
                    (false, true) => tile_i8::<1, TC>(aq, sa, wq, sw, k, g, o, n),
                    (true, false) => tile_i8::<TR, 1>(aq, sa, wq, sw, k, g, o, n),
                    (false, false) => tile_i8::<1, 1>(aq, sa, wq, sw, k, g, o, n),
                }
                i += r;
            }
            j += c;
        }
    }
}

// int8 tile: TR activation rows x TC weight rows per micro-kernel call (2x4: shapes up to 4x4 and 2x8 measured the
// same on an A76)
const TR: usize = 2;
const TC: usize = 4;

/// Whether `Quant` multiplies with int8 dot products (aarch64 with `sdot`, not disabled by INT8_DOT=0).
fn int8_dot() -> bool {
    #[cfg(all(target_arch = "aarch64", feature = "std"))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod") && std::env::var("INT8_DOT").map_or(true, |v| v != "0"))
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "std"), any(target_os = "linux", target_os = "android")))]
    #[allow(unsafe_code)]
    {
        // what the kernel reports (HWCAP_ASIMDDP), read once: 0 not known yet, 1 no, 2 yes
        use core::sync::atomic::{AtomicU8, Ordering};
        unsafe extern "C" {
            fn getauxval(t: core::ffi::c_ulong) -> core::ffi::c_ulong;
        }
        static ON: AtomicU8 = AtomicU8::new(0);
        let mut on = ON.load(Ordering::Relaxed);
        if on == 0 {
            const AT_HWCAP: core::ffi::c_ulong = 16;
            const HWCAP_ASIMDDP: core::ffi::c_ulong = 1 << 20;
            // SAFETY: getauxval only reads the auxiliary vector.
            let dp = cfg!(target_feature = "dotprod") || unsafe { getauxval(AT_HWCAP) } & HWCAP_ASIMDDP != 0;
            on = if dp { 2 } else { 1 };
            ON.store(on, Ordering::Relaxed);
        }
        on == 2
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "std"), not(any(target_os = "linux", target_os = "android"))))]
    {
        cfg!(target_feature = "dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // the same integer arithmetic, portable (to check accuracy off-device): TINYASR_INT8=1
        #[cfg(feature = "std")]
        {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("TINYASR_INT8").is_ok_and(|v| v == "1"))
        }
        #[cfg(not(feature = "std"))]
        {
            false
        }
    }
}

/// Rows of `a` (k each) -> int8 with one scale per `g` values: (values, scales).
fn quantize_rows(a: &[f32], k: usize, g: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; a.len()];
    let mut s = vec![0f32; a.len() / g];
    for ((x, y), s) in a.chunks_exact(g).zip(q.chunks_exact_mut(g)).zip(s.iter_mut()) {
        let max = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        *s = max / 127.0;
        let inv = if max > 0.0 { 127.0 / max } else { 0.0 };
        for (v, o) in x.iter().zip(y.iter_mut()) {
            let r = v * inv;
            *o = (r + if r >= 0.0 { 0.5 } else { -0.5 }) as i8; // round half away from zero
        }
    }
    let _ = k;
    (q, s)
}

/// out[r * stride + c] = sum over groups of (int8 dot of activation row r and weight row c) x both scales,
/// for R activation rows and C weight rows.
#[inline(always)]
fn tile_i8<const R: usize, const C: usize>(aq: &[i8], sa: &[f32], wq: &[i8], sw: &[f32], k: usize, g: usize, out: &mut [f32], stride: usize) {
    assert!(aq.len() >= R * k && wq.len() >= C * k && sa.len() >= R * (k / g) && sw.len() >= C * (k / g));
    assert!(out.len() > (R - 1) * stride + C - 1);
    let mut res = [[0f32; C]; R];
    #[cfg(target_arch = "aarch64")]
    #[allow(unsafe_code)]
    // SAFETY: `int8_dot()` checked the dot-product extension; all reads are inside the slices asserted above.
    unsafe {
        tile_sdot::<R, C>(aq.as_ptr(), sa.as_ptr(), wq.as_ptr(), sw.as_ptr(), k, g, &mut res)
    };
    #[cfg(not(target_arch = "aarch64"))]
    for (r, res) in res.iter_mut().enumerate() {
        for (c, res) in res.iter_mut().enumerate() {
            let (a, w) = (&aq[r * k..(r + 1) * k], &wq[c * k..(c + 1) * k]);
            for (p, (a, w)) in a.chunks_exact(g).zip(w.chunks_exact(g)).enumerate() {
                let d: i32 = a.iter().zip(w).map(|(a, w)| *a as i32 * *w as i32).sum();
                *res += d as f32 * (sa[r * (k / g) + p] * sw[c * (k / g) + p]);
            }
        }
    }
    for (r, res) in res.iter().enumerate() {
        out[r * stride..r * stride + C].copy_from_slice(res);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unsafe_code)]
unsafe fn tile_sdot<const R: usize, const C: usize>(aq: *const i8, sa: *const f32, wq: *const i8, sw: *const f32, k: usize, g: usize, res: &mut [[f32; C]; R]) {
    use core::arch::aarch64::*;
    let ng = k / g;
    // SAFETY: the caller guarantees R rows of k activations / C rows of k weights and their ng scales each.
    unsafe {
        let mut accf = [[vdupq_n_f32(0.0); C]; R];
        for c in 0..ng {
            let mut acci = [[vdupq_n_s32(0); C]; R];
            let mut p = c * g;
            while p < (c + 1) * g {
                let a: [int8x16_t; R] = core::array::from_fn(|r| vld1q_s8(aq.add(r * k + p)));
                let w: [int8x16_t; C] = core::array::from_fn(|j| vld1q_s8(wq.add(j * k + p)));
                for r in 0..R {
                    for j in 0..C {
                        // sdot (vdotq_s32 is still unstable in core::arch)
                        core::arch::asm!("sdot {acc:v}.4s, {a:v}.16b, {w:v}.16b", acc = inout(vreg) acci[r][j], a = in(vreg) a[r], w = in(vreg) w[j], options(pure, nomem, nostack));
                    }
                }
                p += 16;
            }
            for r in 0..R {
                for j in 0..C {
                    accf[r][j] = vfmaq_n_f32(accf[r][j], vcvtq_f32_s32(acci[r][j]), *sa.add(r * ng + c) * *sw.add(j * ng + c));
                }
            }
        }
        for r in 0..R {
            for j in 0..C {
                res[r][j] = vaddvq_f32(accf[r][j]);
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
        let (n, k, g) = (self.n, self.k, self.group);
        let j0 = jp * NR;
        let cols = NR.min(n - j0);
        if cols < NR {
            scratch.fill(0.0);
        }
        for l in 0..cols {
            let j = j0 + l;
            let (row, sc) = (&self.q[j * k..(j + 1) * k], &self.scale[j * k / g..(j + 1) * k / g]);
            for (c, (q, s)) in row.chunks_exact(g).zip(sc).enumerate() {
                for (p, q) in q.iter().enumerate() {
                    scratch[(c * g + p) * NR + l] = *q as f32 * s;
                }
            }
        }
        scratch
    }
}

pub enum Weight {
    F32(Packed),
    Int(Quant),
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
        }
    }

    /// x: m rows of k.
    pub fn apply(&self, x: &[f32], m: usize) -> Vec<f32> {
        match &self.w {
            Weight::F32(p) => gemm(x, m, p, self.b.as_deref()),
            Weight::Int(q) => q.gemm(x, m, self.b.as_deref()),
        }
    }
}
