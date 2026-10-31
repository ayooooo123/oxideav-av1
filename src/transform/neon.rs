//! Clean-room §7.13 transforms across four independent 1-D inputs.
//! All rotations retain signed 64-bit products and the prescribed rounding.
use core::arch::aarch64::*;
use super::{DctValue, dct_b, dct_h};

#[derive(Clone, Copy)]
pub(super) struct Lanes(int32x4_t);
impl Lanes {
    pub(super) fn from_array(a: [i32; 4]) -> Self {
        // SAFETY: the array contains exactly the four readable lanes.
        unsafe { Self(vld1q_s32(a.as_ptr())) }
    }
    pub(super) fn array(self) -> [i32; 4] {
        let mut a = [0; 4];
        // SAFETY: the array contains exactly the four writable lanes.
        unsafe { vst1q_s32(a.as_mut_ptr(), self.0); }
        a
    }
    fn neg(self) -> Self {
        // SAFETY: register-only NEON operation; bounded transform values
        // never reach i32::MIN, so negation agrees with scalar i64.
        unsafe { Self(vnegq_s32(self.0)) }
    }
    pub(super) fn scale(self, factor: i32, shift: u32) -> Self {
        Self::linear(&[self], &[factor], shift)
    }
    fn linear(values: &[Self], coefficients: &[i32], shift: u32) -> Self {
        assert_eq!(values.len(), coefficients.len());
        // SAFETY: all operations are register-only, and bounded AV1 20-bit
        // inputs keep the i64 sums and the rounded i32 result exact.
        unsafe {
            let mut lo = vdupq_n_s64(if shift == 0 { 0 } else { 1 << (shift - 1) });
            let mut hi = lo;
            for (&Self(v), &c) in values.iter().zip(coefficients) {
                lo = vmlal_n_s32(lo, vget_low_s32(v), c);
                hi = vmlal_high_n_s32(hi, v, c);
            }
            let shifts = vdupq_n_s64(-(shift as i64));
            Self(vcombine_s32(vmovn_s64(vshlq_s64(lo, shifts)), vmovn_s64(vshlq_s64(hi, shifts))))
        }
    }
}
impl DctValue for Lanes {
    #[inline(always)]
    fn rotate(a: Self, b: Self, cos: i32, sin: i32) -> (Self, Self) {
        // SAFETY: NEON is mandatory on aarch64. Products widen before
        // accumulation; narrowing happens only after the exact 12-bit round.
        unsafe {
            let al = vget_low_s32(a.0); let bl = vget_low_s32(b.0);
            let xlo = vmlsl_n_s32(vmull_n_s32(al, cos), bl, sin);
            let xhi = vmlsl_high_n_s32(vmull_high_n_s32(a.0, cos), b.0, sin);
            let ylo = vmlal_n_s32(vmull_n_s32(al, sin), bl, cos);
            let yhi = vmlal_high_n_s32(vmull_high_n_s32(a.0, sin), b.0, cos);
            (Self(vcombine_s32(vrshrn_n_s64::<12>(xlo), vrshrn_n_s64::<12>(xhi))),
             Self(vcombine_s32(vrshrn_n_s64::<12>(ylo), vrshrn_n_s64::<12>(yhi))))
        }
    }
    #[inline(always)]
    fn hadamard(a: Self, b: Self, r: u32) -> (Self, Self) {
        // SAFETY: bounded transform stages fit i32 before the specification
        // clamp. The same signed precision limits are used by the scalar twin.
        unsafe {
            let lo = vdupq_n_s32(-(1 << (r - 1)));
            let hi = vdupq_n_s32((1 << (r - 1)) - 1);
            (Self(vminq_s32(hi, vmaxq_s32(lo, vaddq_s32(a.0, b.0)))),
             Self(vminq_s32(hi, vmaxq_s32(lo, vsubq_s32(a.0, b.0)))))
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Kernel { Dct, Adst, Identity }

pub(super) fn transform(t: &mut [Lanes], n: u32, r: u32, kernel: Kernel) {
    match kernel {
        Kernel::Dct => match n {
            2 => super::inverse_dct_4_impl((&mut t[..4]).try_into().unwrap(), r),
            3 => super::inverse_dct_8_impl((&mut t[..8]).try_into().unwrap(), r),
            4 => super::inverse_dct_16_impl((&mut t[..16]).try_into().unwrap(), r),
            5 => super::inverse_dct_32_impl((&mut t[..32]).try_into().unwrap(), r),
            6 => super::inverse_dct_64_impl((&mut t[..64]).try_into().unwrap(), r),
            _ => unreachable!(),
        },
        Kernel::Identity => {
            let (factor, shift) = match n { 2 => (5793, 12), 3 => (2, 0),
                4 => (11586, 12), 5 => (4, 0), _ => unreachable!() };
            for v in &mut t[..1 << n] { *v = v.scale(factor, shift); }
        }
        Kernel::Adst => adst(t, n, r),
    }
}

fn adst(t: &mut [Lanes], n: u32, r: u32) {
    if n == 2 {
        let a = [t[0], t[1], t[2], t[3]];
        let (s1, s2, s3, s4) = (super::SINPI_1_9, super::SINPI_2_9, super::SINPI_3_9, super::SINPI_4_9);
        for (v, c) in t[..4].iter_mut().zip([[s1,s3,s4,s2], [s2,s3,-s1,-s4],
            [s3,0,-s3,s3], [s1+s2,-s3,s4-s1,s2-s4]]) {
            *v = Lanes::linear(&a, &c, 12);
        }
        return;
    }
    let len = 1 << n;
    let mut original = [t[0]; 16];
    original[..len].copy_from_slice(&t[..len]);
    for i in 0..len { t[i] = original[if i & 1 == 1 { i - 1 } else { len - i - 1 }]; }
    for i in 0..len / 2 { dct_b(t, 2*i, 2*i+1, 64 - (32 / len) as i32 - (128 / len) as i32 * i as i32, 1, r); }
    // §7.13.2.7/.8: each stage halves the distance between Hadamard
    // partners, alternating rotations on the upper half of each group.
    for i in 0..len / 2 { dct_h(t, i, len/2+i, 0, r); }
    if n == 4 {
        for i in 0..2 {
            dct_b(t, 8+2*i, 9+2*i, 56-32*i as i32, 1, r);
            dct_b(t, 13+2*i, 12+2*i, 8+32*i as i32, 1, r);
        }
        for j in 0..2 { for i in 0..4 { dct_h(t, 8*j+i, 4+8*j+i, 0, r); } }
    }
    for j in 0..len/8 { for i in 0..2 { dct_b(t, 4+8*j+3*i, 5+8*j+i, 48-32*i as i32, 1, r); } }
    for j in 0..len/4 { for i in 0..2 { dct_h(t, 4*j+i, 2+4*j+i, 0, r); } }
    for i in 0..len/4 { dct_b(t, 2+4*i, 3+4*i, 32, 1, r); }
    original[..len].copy_from_slice(&t[..len]);
    for i in 0..len {
        let a = ((i >> 3) & 1) as u32;
        let b = (((i >> 2) & 1) ^ ((i >> 3) & 1)) as u32;
        let c = (((i >> 1) & 1) ^ ((i >> 2) & 1)) as u32;
        let d = ((i & 1) ^ ((i >> 1) & 1)) as u32;
        let index = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
        let v = original[index as usize]; t[i] = if i & 1 == 1 { v.neg() } else { v };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_lanes_match_scalar_at_depth_precision_edges() {
        let mut state = 0x832ba071u32;
        for kernel in [Kernel::Dct, Kernel::Adst, Kernel::Identity] {
            let last = match kernel { Kernel::Dct => 6, Kernel::Adst => 4, Kernel::Identity => 5 };
            for n in 2..=last { for r in [16,18,20] { for case in 0..260 {
                let len = 1 << n;
                let input: Vec<[i32;4]> = (0..len).map(|i| core::array::from_fn(|lane| {
                    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                    let lo = -(1 << (r - 1)); let hi = -lo - 1;
                    match case { 0 => lo, 1 => hi, 2 => if (i + lane) & 1 == 0 { lo } else { hi },
                        3 => 0, _ => (state & ((1 << r) - 1)) as i32 + lo }
                })).collect();
                let mut lanes: Vec<Lanes> = input.iter().copied().map(Lanes::from_array).collect();
                transform(&mut lanes, n, r, kernel);
                for lane in 0..4 {
                    let mut scalar: Vec<i64> = input.iter().map(|v| i64::from(v[lane])).collect();
                    match kernel { Kernel::Dct => super::super::inverse_dct(&mut scalar, n, r),
                        Kernel::Adst => super::super::inverse_adst(&mut scalar, n, r),
                        Kernel::Identity => super::super::inverse_identity(&mut scalar, n) }
                    for i in 0..len { assert_eq!(i64::from(lanes[i].array()[lane]), scalar[i],
                        "kernel={kernel:?} n={n} r={r} case={case} lane={lane} index={i}"); }
                }
            } } }
        }
    }
}

fn row_kernel(tx: usize) -> Kernel {
    use crate::cdf::*;
    if matches!(tx, DCT_DCT | ADST_DCT | FLIPADST_DCT | H_DCT) { Kernel::Dct }
    else if matches!(tx, DCT_ADST | ADST_ADST | DCT_FLIPADST | FLIPADST_FLIPADST |
        ADST_FLIPADST | FLIPADST_ADST | H_ADST | H_FLIPADST) { Kernel::Adst }
    else { Kernel::Identity }
}
fn col_kernel(tx: usize) -> Kernel {
    use crate::cdf::*;
    if matches!(tx, DCT_DCT | DCT_ADST | DCT_FLIPADST | V_DCT) { Kernel::Dct }
    else if matches!(tx, ADST_DCT | ADST_ADST | FLIPADST_DCT | FLIPADST_FLIPADST |
        ADST_FLIPADST | FLIPADST_ADST | V_ADST | V_FLIPADST) { Kernel::Adst }
    else { Kernel::Identity }
}

pub(super) fn inverse_2d(dequant: &[i64], size: usize, tx: usize, depth: u32) -> Option<Vec<i64>> {
    if !matches!(depth, 8 | 10 | 12) || tx >= 16 { return None; }
    let r = depth + 8;
    if dequant.iter().any(|&v| v < -(1 << (r - 1)) || v >= 1 << (r - 1)) { return None; }
    let w = super::TX_WIDTH[size];
    let h = super::TX_HEIGHT[size];
    let nw = super::TX_WIDTH_LOG2[size] as u32;
    let nh = (h as u32).trailing_zeros();
    let rectangular = nw.abs_diff(nh) == 1;
    let row_shift = super::TRANSFORM_ROW_SHIFT[size];
    let col_range = (depth + 6).max(16);
    let low = -(1i64 << (col_range - 1));
    let high = -low - 1;
    let mut residual = vec![0; w * h];
    let mut lanes = [Lanes::from_array([0; 4]); 64];
    for first in (0..h.min(32)).step_by(4) {
        if dequant[first * w..(first + 4) * w].iter().all(|&v| v == 0) { continue; }
        for j in 0..w {
            lanes[j] = Lanes::from_array(core::array::from_fn(|lane|
                if j < 32 { dequant[(first + lane) * w + j] as i32 } else { 0 }));
            if rectangular { lanes[j] = lanes[j].scale(2896, 12); }
        }
        transform(&mut lanes[..w], nw, r, row_kernel(tx));
        for j in 0..w {
            for (lane, v) in lanes[j].array().into_iter().enumerate() {
                residual[(first + lane) * w + j] = super::round2(i64::from(v), row_shift).clamp(low, high);
            }
        }
    }
    for first in (0..w).step_by(4) {
        for i in 0..h {
            lanes[i] = Lanes::from_array(core::array::from_fn(|lane| residual[i * w + first + lane] as i32));
        }
        transform(&mut lanes[..h], nh, col_range, col_kernel(tx));
        for i in 0..h {
            for (lane, v) in lanes[i].array().into_iter().enumerate() {
                residual[i * w + first + lane] = super::round2(i64::from(v), 4);
            }
        }
    }
    Some(residual)
}

#[cfg(test)]
mod plumbing_tests {
    use super::*;
    #[test]
    fn two_dimensional_passes_match_scalar_every_supported_type_and_size() {
        let mut state = 0x4275d8a1u32;
        for size in 0..super::super::TX_SIZES_ALL {
            let w = super::super::TX_WIDTH[size];
            let h = super::super::TX_HEIGHT[size];
            for tx in 0..16 {
                let supports = |kernel, len| match kernel {
                    Kernel::Dct => len <= 64, Kernel::Adst => len <= 16, Kernel::Identity => len <= 32 };
                if !supports(row_kernel(tx), w) || !supports(col_kernel(tx), h) { continue; }
                for depth in [8,10,12] { for case in 0..20 {
                    let bits = depth + 8;
                    let lo = -(1i64 << (bits - 1)); let hi = -lo - 1;
                    let input: Vec<i64> = (0..w*h).map(|i| {
                        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                        if i / w >= 32 || i % w >= 32 { return 0; }
                        match case { 0 => lo, 1 => hi, 2 => if i & 1 == 0 { lo } else { hi },
                            3 => if i == 0 { hi } else { 0 },
                            _ => i64::from(state & ((1 << bits) - 1)) + lo }
                    }).collect();
                    assert_eq!(inverse_2d(&input, size, tx, depth).unwrap(),
                        super::super::inverse_transform_2d_scalar(&input, size, tx, depth, false),
                        "size={size} tx={tx} depth={depth} case={case}");
                } }
            }
        }
    }
}
