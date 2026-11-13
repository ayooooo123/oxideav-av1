//! §7.13.3 two-dimensional inverse transform in NEON lanes: four rows,
//! then four columns, of a block per 1-D transform, one 32-bit lane each.
//!
//! A lane holds what the scalar kernels hold in an `i64`. That is exact
//! because [`inverse_2d`] takes only blocks whose coefficients fit the
//! `8 + BitDepth` bits (at most 20) §7.12.3 clamps them to: every `H()`
//! output is clamped to those bits again, a `B()` rotation grows a pair by
//! at most `sqrt( 2 )` and the ADST4 and identity kernels scale by less
//! than 4, so every lane stays far inside 32 bits, while the rotations and
//! scalings form their products in 64 bits and narrow only after the
//! `Round2`. The DCT replays the schedule recorded for the portable lane
//! kernels ([`dct_program`]); the differential tests bound it step by step.
//!
//! The entry point is safe and reports whether it ran: it checks for NEON
//! at run time (a constant on the AArch64 targets, where NEON is part of
//! the baseline) and leaves `residual` untouched when it is missing or the
//! block is not one it takes, so the caller's portable passes are the
//! fallback.
use super::{
    BREV_TABLE, FlowStep, SINPI_1_9, SINPI_2_9, SINPI_3_9, SINPI_4_9, TRANSFORM_ROW_SHIFT, cos128,
    dct_program, round2, sin128,
};
use crate::cdf::{
    ADST_ADST, ADST_DCT, ADST_FLIPADST, DCT_ADST, DCT_DCT, DCT_FLIPADST, FLIPADST_ADST,
    FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT, H_FLIPADST, IDTX, TX_HEIGHT, TX_WIDTH,
    TX_WIDTH_LOG2, V_ADST, V_DCT, V_FLIPADST,
};
use core::arch::aarch64::*;

#[inline(always)]
fn available() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}

/// Element `i` of four 1-D arrays: four rows in the row pass, four
/// columns in the column pass.
type Lanes = int32x4_t;

/// The arrays of one pass, `T[ i ]` of all four in slot `i`.
type Arrays = [Lanes; 64];

#[inline]
#[target_feature(enable = "neon")]
fn load(values: [i32; 4]) -> Lanes {
    // SAFETY: the array holds exactly the four lanes read.
    unsafe { vld1q_s32(values.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store(lanes: Lanes) -> [i32; 4] {
    let mut values = [0; 4];
    // SAFETY: the array holds exactly the four lanes written.
    unsafe { vst1q_s32(values.as_mut_ptr(), lanes) };
    values
}

/// `Round2( sum_i values[ i ] * coefficients[ i ], shift )`, the sum in
/// 64 bits.
#[inline]
#[target_feature(enable = "neon")]
fn linear<const N: usize>(values: &[Lanes; N], coefficients: &[i32; N], shift: u32) -> Lanes {
    let bias = vdupq_n_s64(if shift == 0 { 0 } else { 1 << (shift - 1) });
    let (mut lo, mut hi) = (bias, bias);
    for (&v, &c) in values.iter().zip(coefficients) {
        lo = vmlal_n_s32(lo, vget_low_s32(v), c);
        hi = vmlal_high_n_s32(hi, v, c);
    }
    let shifts = vdupq_n_s64(-i64::from(shift));
    vcombine_s32(
        vmovn_s64(vshlq_s64(lo, shifts)),
        vmovn_s64(vshlq_s64(hi, shifts)),
    )
}

/// §7.13.2.1 `B()` of one pair: `( Round2( a * cos - b * sin, 12 ),
/// Round2( a * sin + b * cos, 12 ) )`, the products in 64 bits.
#[inline]
#[target_feature(enable = "neon")]
fn rotate(a: Lanes, b: Lanes, cos: i32, sin: i32) -> (Lanes, Lanes) {
    let (a_lo, b_lo) = (vget_low_s32(a), vget_low_s32(b));
    let x_lo = vmlsl_n_s32(vmull_n_s32(a_lo, cos), b_lo, sin);
    let x_hi = vmlsl_high_n_s32(vmull_high_n_s32(a, cos), b, sin);
    let y_lo = vmlal_n_s32(vmull_n_s32(a_lo, sin), b_lo, cos);
    let y_hi = vmlal_high_n_s32(vmull_high_n_s32(a, sin), b, cos);
    (
        vcombine_s32(vrshrn_n_s64::<12>(x_lo), vrshrn_n_s64::<12>(x_hi)),
        vcombine_s32(vrshrn_n_s64::<12>(y_lo), vrshrn_n_s64::<12>(y_hi)),
    )
}

/// §7.13.2.1 `H()` of one pair: the sum and the difference, each clamped
/// to `r` bits.
#[inline]
#[target_feature(enable = "neon")]
fn add_sub(a: Lanes, b: Lanes, r: u32) -> (Lanes, Lanes) {
    let lo = vdupq_n_s32(-(1 << (r - 1)));
    let hi = vdupq_n_s32((1 << (r - 1)) - 1);
    (
        vminq_s32(hi, vmaxq_s32(lo, vaddq_s32(a, b))),
        vminq_s32(hi, vmaxq_s32(lo, vsubq_s32(a, b))),
    )
}

/// §7.13.2.3 inverse DCT of arrays of length `1 << n`: the §7.13.2.2
/// permutation, then the recorded butterfly schedule.
#[target_feature(enable = "neon")]
fn dct(t: &mut Arrays, n: u32, r: u32) {
    let natural = *t;
    for (slot, &from) in t.iter_mut().zip(&BREV_TABLE[n as usize][..1 << n]) {
        *slot = natural[usize::from(from) & 63];
    }
    for step in &dct_program(n).steps {
        match *step {
            FlowStep::Rotate {
                a,
                b,
                cos,
                sin,
                x_to,
                y_to,
            } => {
                let (x, y) = rotate(t[usize::from(a) & 63], t[usize::from(b) & 63], cos, sin);
                t[usize::from(x_to) & 63] = x;
                t[usize::from(y_to) & 63] = y;
            }
            FlowStep::AddSub { sum, diff } => {
                let (s, d) = add_sub(t[usize::from(sum) & 63], t[usize::from(diff) & 63], r);
                t[usize::from(sum) & 63] = s;
                t[usize::from(diff) & 63] = d;
            }
        }
    }
}

/// `B( a, b, angle, 1 )` in place.
#[inline]
#[target_feature(enable = "neon")]
fn rotate_flipped(t: &mut Arrays, a: usize, b: usize, angle: i32) {
    let (x, y) = rotate(t[a & 63], t[b & 63], cos128(angle), sin128(angle));
    t[a & 63] = y;
    t[b & 63] = x;
}

/// `H( a, b, 0, r )` in place.
#[inline]
#[target_feature(enable = "neon")]
fn add_sub_at(t: &mut Arrays, a: usize, b: usize, r: u32) {
    let (sum, diff) = add_sub(t[a & 63], t[b & 63], r);
    t[a & 63] = sum;
    t[b & 63] = diff;
}

/// §7.13.2.6 – §7.13.2.9 inverse ADST of arrays of length `1 << n`
/// (`2 <= n <= 4`).
#[target_feature(enable = "neon")]
fn adst(t: &mut Arrays, n: u32, r: u32) {
    if n == 2 {
        // §7.13.2.6 with its seven products gathered per output.
        let input = [t[0], t[1], t[2], t[3]];
        let (s1, s2, s3, s4) = (SINPI_1_9, SINPI_2_9, SINPI_3_9, SINPI_4_9);
        let outputs = [
            [s1, s3, s4, s2],
            [s2, s3, -s1, -s4],
            [s3, 0, -s3, s3],
            [s1 + s2, -s3, s4 - s1, s2 - s4],
        ];
        for (slot, coefficients) in t.iter_mut().zip(&outputs) {
            *slot = linear(&input, coefficients, 12);
        }
        return;
    }
    let len = 1usize << n;
    // §7.13.2.4 input permutation.
    let input = *t;
    for (i, slot) in t[..len].iter_mut().enumerate() {
        *slot = input[if i & 1 == 1 { i - 1 } else { len - i - 1 }];
    }
    // §7.13.2.7 (`n == 3`) / §7.13.2.8 (`n == 4`): rotations of adjacent
    // pairs alternate with butterflies whose partners sit half as far
    // apart at each stage.
    for i in 0..len / 2 {
        let angle = 64 - (32 / len) as i32 - (128 / len * i) as i32;
        rotate_flipped(t, 2 * i, 2 * i + 1, angle);
    }
    for i in 0..len / 2 {
        add_sub_at(t, i, len / 2 + i, r);
    }
    if n == 4 {
        for i in 0..2 {
            rotate_flipped(t, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i as i32);
            rotate_flipped(t, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i as i32);
        }
        for j in 0..2 {
            for i in 0..4 {
                add_sub_at(t, 8 * j + i, 4 + 8 * j + i, r);
            }
        }
    }
    for j in 0..len / 8 {
        for i in 0..2 {
            rotate_flipped(t, 4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i as i32);
        }
    }
    for j in 0..len / 4 {
        for i in 0..2 {
            add_sub_at(t, 4 * j + i, 2 + 4 * j + i, r);
        }
    }
    for i in 0..len / 4 {
        rotate_flipped(t, 2 + 4 * i, 3 + 4 * i, 32);
    }
    // §7.13.2.5 output permutation, odd outputs negated.
    let output = *t;
    for (i, slot) in t[..len].iter_mut().enumerate() {
        let a = (i >> 3) & 1;
        let b = ((i >> 2) ^ (i >> 3)) & 1;
        let c = ((i >> 1) ^ (i >> 2)) & 1;
        let d = (i ^ (i >> 1)) & 1;
        let from = output[(((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n)) & 63];
        *slot = if i & 1 == 1 { vnegq_s32(from) } else { from };
    }
}

/// The 1-D kernel §7.13.3 selects for one pass of a transform type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kernel {
    Dct,
    Adst,
    Identity,
}

impl Kernel {
    /// The row-pass kernel of `tx_type` (the sets of `apply_row_kernel`).
    fn of_rows(tx_type: usize) -> Option<Self> {
        match tx_type {
            DCT_DCT | ADST_DCT | FLIPADST_DCT | H_DCT => Some(Self::Dct),
            DCT_ADST | ADST_ADST | DCT_FLIPADST | FLIPADST_FLIPADST | ADST_FLIPADST
            | FLIPADST_ADST | H_ADST | H_FLIPADST => Some(Self::Adst),
            IDTX | V_DCT | V_ADST | V_FLIPADST => Some(Self::Identity),
            _ => None,
        }
    }

    /// The column-pass kernel of `tx_type` (the sets of `apply_col_kernel`).
    fn of_columns(tx_type: usize) -> Option<Self> {
        match tx_type {
            DCT_DCT | DCT_ADST | DCT_FLIPADST | V_DCT => Some(Self::Dct),
            ADST_DCT | ADST_ADST | FLIPADST_DCT | FLIPADST_FLIPADST | ADST_FLIPADST
            | FLIPADST_ADST | V_ADST | V_FLIPADST => Some(Self::Adst),
            IDTX | H_DCT | H_ADST | H_FLIPADST => Some(Self::Identity),
            _ => None,
        }
    }

    /// Whether §7.13.2 defines the kernel for arrays of length `1 << n`.
    fn defined(self, n: u32) -> bool {
        match self {
            Self::Dct => (2..=6).contains(&n),
            Self::Adst => (2..=4).contains(&n),
            Self::Identity => (2..=5).contains(&n),
        }
    }

    /// The kernel over the first `1 << n` slots of `t`, `H()` clamping to
    /// `r` bits.
    #[inline]
    #[target_feature(enable = "neon")]
    fn run(self, t: &mut Arrays, n: u32, r: u32) {
        match self {
            Self::Dct => dct(t, n, r),
            Self::Adst => adst(t, n, r),
            Self::Identity => {
                // §7.13.2.15: sqrt( 2 ), 2, 2 * sqrt( 2 ) and 4.
                let (factor, shift) =
                    [(5793, 12), (2, 0), (11586, 12), (4, 0)][(n as usize - 2) & 3];
                for slot in &mut t[..1 << n] {
                    *slot = linear(&[*slot], &[factor], shift);
                }
            }
        }
    }
}

/// The §7.13.3 inverse transform of one block into `residual`, for the
/// lossy transform types whose kernels §7.13.2 defines at the block's
/// size and coefficients of at most `8 + bit_depth` bits (`dequant` holds
/// the block's `TX_WIDTH × TX_HEIGHT` coefficients).
pub(super) fn inverse_2d(
    dequant: &[i64],
    tx_sz: usize,
    tx_type: usize,
    bit_depth: u32,
    residual: &mut Vec<i64>,
) -> bool {
    let (w, h) = (TX_WIDTH[tx_sz], TX_HEIGHT[tx_sz]);
    let log2_w = TX_WIDTH_LOG2[tx_sz] as u32;
    let log2_h = h.trailing_zeros();
    let (Some(rows), Some(columns)) = (Kernel::of_rows(tx_type), Kernel::of_columns(tx_type))
    else {
        return false;
    };
    if !available()
        || !matches!(bit_depth, 8 | 10 | 12)
        || !rows.defined(log2_w)
        || !columns.defined(log2_h)
        || dequant.len() != w * h
    {
        return false;
    }
    let r = bit_depth + 8;
    let limit = 1i64 << (r - 1);
    if dequant.iter().any(|&v| v < -limit || v >= limit) {
        return false;
    }
    residual.clear();
    residual.resize(w * h, 0);
    // SAFETY: NEON was detected above.
    unsafe {
        inverse_2d_neon(
            dequant, residual, tx_sz, rows, columns, log2_w, log2_h, bit_depth,
        )
    };
    true
}

/// # Safety
///
/// Needs NEON only. `dequant` and the zeroed `residual` hold the block's
/// `TX_WIDTH × TX_HEIGHT` values and the coefficients fit `8 + bit_depth
/// <= 20` bits ([`inverse_2d`] checks all of it).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn inverse_2d_neon(
    dequant: &[i64],
    residual: &mut [i64],
    tx_sz: usize,
    rows: Kernel,
    columns: Kernel,
    log2_w: u32,
    log2_h: u32,
    bit_depth: u32,
) {
    let (w, h) = (1usize << log2_w, 1usize << log2_h);
    let rectangular = log2_w.abs_diff(log2_h) == 1;
    let row_shift = TRANSFORM_ROW_SHIFT[tx_sz];
    let column_range = (bit_depth + 6).max(16);
    let high = (1i64 << (column_range - 1)) - 1;
    let mut t: Arrays = [vdupq_n_s32(0); 64];
    // Rows and input columns from 32 on read as zero; zero rows transform
    // to the zero rows `residual` already holds.
    for i0 in (0..h.min(32)).step_by(4) {
        let group = &dequant[i0 * w..(i0 + 4) * w];
        if group.iter().all(|&v| v == 0) {
            continue;
        }
        for (j, slot) in t[..w].iter_mut().enumerate() {
            *slot = if j < 32 {
                load(core::array::from_fn(|lane| group[lane * w + j] as i32))
            } else {
                vdupq_n_s32(0)
            };
            if rectangular {
                *slot = linear(&[*slot], &[2896], 12);
            }
        }
        rows.run(&mut t, log2_w, bit_depth + 8);
        for (j, &lanes) in t[..w].iter().enumerate() {
            for (lane, v) in store(lanes).into_iter().enumerate() {
                residual[(i0 + lane) * w + j] =
                    round2(i64::from(v), row_shift).clamp(-high - 1, high);
            }
        }
    }
    for j0 in (0..w).step_by(4) {
        for (i, slot) in t[..h].iter_mut().enumerate() {
            let at = i * w + j0;
            *slot = load(core::array::from_fn(|lane| residual[at + lane] as i32));
        }
        columns.run(&mut t, log2_h, column_range);
        for (i, &lanes) in t[..h].iter().enumerate() {
            for (lane, v) in store(lanes).into_iter().enumerate() {
                residual[i * w + j0 + lane] = round2(i64::from(v), 4);
            }
        }
    }
}

/// One kernel over four arrays of length `1 << n` (`values[ i ][ lane ]`
/// is element `i` of array `lane`), for the differential tests.
#[cfg(test)]
pub(super) fn transform_1d(values: &mut [[i32; 4]], kernel: Kernel, n: u32, r: u32) -> bool {
    if !available() || !kernel.defined(n) || values.len() != 1 << n {
        return false;
    }
    // SAFETY: NEON was detected above.
    unsafe { transform_1d_neon(values, kernel, n, r) };
    true
}

/// # Safety
///
/// Needs NEON only.
#[cfg(test)]
#[target_feature(enable = "neon")]
unsafe fn transform_1d_neon(values: &mut [[i32; 4]], kernel: Kernel, n: u32, r: u32) {
    let mut t: Arrays = [vdupq_n_s32(0); 64];
    for (slot, &v) in t.iter_mut().zip(values.iter()) {
        *slot = load(v);
    }
    kernel.run(&mut t, n, r);
    for (v, &slot) in values.iter_mut().zip(&t) {
        *v = store(slot);
    }
}
