//! Clean-room §7.11 convolution, combination, and sample representation lanes.
use core::arch::aarch64::*;

pub(super) fn horizontal(src: &[u16], coeff: &[i32; 8], shift: u32, dst: &mut [i32], w: usize) {
    assert!(src.len() >= w + 7 && dst.len() >= w);
    assert!((1..32).contains(&shift));
    let end = w / 8 * 8;
    // SAFETY: the entire eight-tap row and output were checked above;
    // every vector covers eight outputs. AV1 samples fit signed 16 bits,
    // and all products/sums fit signed 32 bits at 8/10/12-bit depth.
    unsafe {
        let offset = vdupq_n_s32(1 << (shift - 1));
        let shifts = vdupq_n_s32(-(shift as i32));
        for x in (0..end).step_by(8) {
            let p = src.as_ptr().add(x);
            let mut lo = offset;
            let mut hi = offset;
            for (k, &c) in coeff.iter().enumerate() {
                let s = vreinterpretq_s16_u16(vld1q_u16(p.add(k)));
                lo = vmlal_n_s16(lo, vget_low_s16(s), c as i16);
                hi = vmlal_high_n_s16(hi, s, c as i16);
            }
            vst1q_s32(dst.as_mut_ptr().add(x), vshlq_s32(lo, shifts));
            vst1q_s32(dst.as_mut_ptr().add(x + 4), vshlq_s32(hi, shifts));
        }
    }
    for x in end..w {
        let sum: i32 = (0..8).map(|k| coeff[k] * i32::from(src[x + k])).sum();
        dst[x] = (sum + (1 << (shift - 1))) >> shift;
    }
}

pub(super) fn vertical(src: &[i32], w: usize, r: usize, coeff: &[i32; 8], shift: u32, dst: &mut [i32]) {
    assert!(src.len() >= (r + 8) * w && dst.len() >= w);
    assert!((1..32).contains(&shift));
    let end = w / 8 * 8;
    // SAFETY: all eight source rows and the output row were checked above.
    // Horizontal intermediate bounds and the specification's coefficient
    // range keep products/sums in i32, including compound rounding paths.
    unsafe {
        let offset = vdupq_n_s32(1 << (shift - 1));
        let shifts = vdupq_n_s32(-(shift as i32));
        for x in (0..end).step_by(8) {
            let mut lo = offset;
            let mut hi = offset;
            for (k, &c) in coeff.iter().enumerate() {
                let p = src.as_ptr().add((r + k) * w + x);
                lo = vmlaq_n_s32(lo, vld1q_s32(p), c);
                hi = vmlaq_n_s32(hi, vld1q_s32(p.add(4)), c);
            }
            vst1q_s32(dst.as_mut_ptr().add(x), vshlq_s32(lo, shifts));
            vst1q_s32(dst.as_mut_ptr().add(x + 4), vshlq_s32(hi, shifts));
        }
    }
    for x in end..w {
        let sum: i32 = (0..8).map(|k| coeff[k] * src[(r + k) * w + x]).sum();
        dst[x] = (sum + (1 << (shift - 1))) >> shift;
    }
}

pub(super) fn clip(depth: u8, src: &[i32], dst: &mut [u16]) {
    assert!(dst.len() >= src.len());
    let end = src.len() / 8 * 8;
    let max = (1 << depth) - 1;
    // SAFETY: the destination covers the complete source; each iteration
    // loads/stores exactly eight elements. Saturation is exactly Clip1.
    unsafe {
        let high = vdupq_n_s32(max);
        for x in (0..end).step_by(8) {
            let lo = vqmovun_s32(vminq_s32(high, vld1q_s32(src.as_ptr().add(x))));
            let hi = vqmovun_s32(vminq_s32(high, vld1q_s32(src.as_ptr().add(x + 4))));
            vst1q_u16(dst.as_mut_ptr().add(x), vcombine_u16(lo, hi));
        }
    }
    for x in end..src.len() { dst[x] = src[x].clamp(0, max) as u16; }
}

pub(super) fn average(depth: u8, shift: u32, a: &[i32], b: &[i32], dst: &mut [u16]) {
    assert!(a.len() <= b.len() && a.len() <= dst.len());
    assert!((1..32).contains(&shift));
    let end = a.len() / 8 * 8;
    let max = (1 << depth) - 1;
    // SAFETY: slices cover all eight lanes. Halving add avoids overflow
    // even for full-range i32 inputs; for shift > 1, dropping the sum's
    // low bit before the remaining rounded shift is algebraically exact.
    unsafe {
        let high = vdupq_n_s32(max);
        let shifts = vdupq_n_s32(1 - shift as i32);
        for x in (0..end).step_by(8) {
            let mut v = [vdup_n_u16(0); 2];
            for (half, value) in v.iter_mut().enumerate() {
                let index = x + half * 4;
                let av = vld1q_s32(a.as_ptr().add(index));
                let bv = vld1q_s32(b.as_ptr().add(index));
                let rounded = if shift == 1 { vrhaddq_s32(av, bv) }
                    else { vrshlq_s32(vhaddq_s32(av, bv), shifts) };
                *value = vqmovun_s32(vminq_s32(rounded, high));
            }
            vst1q_u16(dst.as_mut_ptr().add(x), vcombine_u16(v[0], v[1]));
        }
    }
    for x in end..a.len() {
        dst[x] = (((i64::from(a[x]) + i64::from(b[x]) + (1i64 << (shift - 1))) >> shift) as i32).clamp(0, max) as u16;
    }
}

pub(super) fn pack(src: &[i32], dst: &mut [u16]) {
    assert!(dst.len() >= src.len());
    let end = src.len() / 8 * 8;
    // SAFETY: both slices cover eight elements per iteration. Non-saturating
    // narrowing preserves Rust's wrapping `as u16`, including arbitrary i32.
    unsafe {
        for x in (0..end).step_by(8) {
            let lo = vmovn_s32(vld1q_s32(src.as_ptr().add(x)));
            let hi = vmovn_s32(vld1q_s32(src.as_ptr().add(x + 4)));
            vst1q_u16(dst.as_mut_ptr().add(x), vreinterpretq_u16_s16(vcombine_s16(lo, hi)));
        }
    }
    for x in end..src.len() { dst[x] = src[x] as u16; }
}

pub(super) fn unpack(src: &[u16], dst: &mut [i32]) {
    assert!(dst.len() >= src.len());
    let end = src.len() / 8 * 8;
    // SAFETY: both slices cover eight elements per iteration. Zero extension
    // exactly matches `u16 as i32`, including the complete u16 range.
    unsafe {
        for x in (0..end).step_by(8) {
            let v = vld1q_u16(src.as_ptr().add(x));
            vst1q_s32(dst.as_mut_ptr().add(x), vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v))));
            vst1q_s32(dst.as_mut_ptr().add(x + 4), vreinterpretq_s32_u32(vmovl_high_u16(v)));
        }
    }
    for x in end..src.len() { dst[x] = i32::from(src[x]); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inter_pred::{filter_8tap_h_scalar, filter_8tap_v_scalar, SUBPEL_FILTERS};
    fn next(s: &mut u32) -> u32 { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; *s }
    #[test]
    fn filters_match_scalar_all_phases_depths_and_widths() {
        let mut state = 0x9e3ac5b1;
        for depth in [8, 10, 12] {
            let max = (1 << depth) - 1;
            let round0 = if depth == 12 { 5 } else { 3 };
            for w in 1..=129 {
                for coeff in SUBPEL_FILTERS.iter().flatten() {
                    for case in 0..5 {
                        let src: Vec<u16> = (0..(w + 7) * 8).map(|i| match case {
                            0 => 0, 1 => max, 2 => if i & 1 == 0 { 0 } else { max },
                            _ => (next(&mut state) & u32::from(max)) as u16 }).collect();
                        let mut expected = vec![0; w * 8];
                        let mut actual = expected.clone();
                        for r in 0..8 {
                            filter_8tap_h_scalar(&src[r * (w + 7)..], coeff, round0, &mut expected[r * w..], w);
                            horizontal(&src[r * (w + 7)..], coeff, round0, &mut actual[r * w..], w);
                        }
                        assert_eq!(actual, expected, "horizontal depth={depth} w={w}");
                        for shift in [7, 9, 11] {
                            let mut a = vec![0; w]; let mut b = a.clone();
                            filter_8tap_v_scalar(&expected, w, 0, coeff, shift, &mut a);
                            vertical(&actual, w, 0, coeff, shift, &mut b);
                            assert_eq!(a, b, "vertical depth={depth} w={w} shift={shift}");
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn combinations_and_bridge_match_full_range_scalar() {
        let mut state = 0xa197c489;
        for n in 0..=513 {
            let a: Vec<i32> = (0..n).map(|i| match i % 8 { 0 => i32::MIN, 1 => i32::MAX,
                2 => -1, 3 => 4095, _ => next(&mut state) as i32 }).collect();
            let b: Vec<i32> = (0..n).map(|_| next(&mut state) as i32).collect();
            let mut packed = vec![0; n]; pack(&a, &mut packed);
            assert_eq!(packed, a.iter().map(|&v| v as u16).collect::<Vec<_>>());
            let mut unpacked = vec![0; n]; unpack(&packed, &mut unpacked);
            assert_eq!(unpacked, packed.iter().map(|&v| i32::from(v)).collect::<Vec<_>>());
            for depth in [8, 10, 12] {
                let max = (1 << depth) - 1;
                let mut dst = vec![0; n]; clip(depth, &a, &mut dst);
                assert_eq!(dst, a.iter().map(|&v| v.clamp(0, max) as u16).collect::<Vec<_>>());
                for shift in 1..=8 {
                    average(depth, shift, &a, &b, &mut dst);
                    let expected: Vec<u16> = a.iter().zip(&b).map(|(&a, &b)|
                        (((i64::from(a) + i64::from(b) + (1 << (shift - 1))) >> shift) as i32).clamp(0, max) as u16).collect();
                    assert_eq!(dst, expected);
                }
            }
        }
    }
}
