//! Clean-room §7.11 convolution, combination, and sample representation lanes.
use core::arch::aarch64::*;
use crate::loop_filter::Pixel;

pub(super) fn horizontal<R: Pixel>(src: &[R], coeff: &[i32; 8], shift: u32, dst: &mut [i32], w: usize) {
    assert!(src.len() >= w + 7 && dst.len() >= w);
    assert!((1..32).contains(&shift));
    let end = w / 8 * 8;
    let wide_end = if core::mem::size_of::<R>() == 1 { w / 16 * 16 } else { 0 };
    // SAFETY: Pixel is sealed to u8/u16/i32. The complete source tap
    // rectangle and destination were checked above. Native u8 consumes
    // sixteen samples per input register; u16 consumes eight. Samples fit
    // signed 16 bits and convolution products/sums retain exact i32 widths.
    unsafe {
        let offset = vdupq_n_s32(1 << (shift - 1));
        let shifts = vdupq_n_s32(-(shift as i32));
        for x in (0..wide_end).step_by(16) {
            let p = src.as_ptr().add(x).cast::<u8>();
            let (mut a0, mut a1, mut b0, mut b1) = (offset, offset, offset, offset);
            for (k, &c) in coeff.iter().enumerate() {
                let bytes = vld1q_u8(p.add(k));
                let a = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bytes)));
                let b = vreinterpretq_s16_u16(vmovl_high_u8(bytes));
                a0 = vmlal_n_s16(a0, vget_low_s16(a), c as i16);
                a1 = vmlal_high_n_s16(a1, a, c as i16);
                b0 = vmlal_n_s16(b0, vget_low_s16(b), c as i16);
                b1 = vmlal_high_n_s16(b1, b, c as i16);
            }
            vst1q_s32(dst.as_mut_ptr().add(x), vshlq_s32(a0, shifts));
            vst1q_s32(dst.as_mut_ptr().add(x + 4), vshlq_s32(a1, shifts));
            vst1q_s32(dst.as_mut_ptr().add(x + 8), vshlq_s32(b0, shifts));
            vst1q_s32(dst.as_mut_ptr().add(x + 12), vshlq_s32(b1, shifts));
        }
        for x in (wide_end..end).step_by(8) {
            let p = src.as_ptr().add(x);
            let mut lo = offset;
            let mut hi = offset;
            for (k, &c) in coeff.iter().enumerate() {
                let p = p.add(k);
                let s = if core::mem::size_of::<R>() == 1 {
                    vreinterpretq_s16_u16(vmovl_u8(vld1_u8(p.cast())))
                } else if core::mem::size_of::<R>() == 2 {
                    vreinterpretq_s16_u16(vld1q_u16(p.cast()))
                } else {
                    let p = p.cast::<i32>();
                    vcombine_s16(vmovn_s32(vld1q_s32(p)), vmovn_s32(vld1q_s32(p.add(4))))
                };
                lo = vmlal_n_s16(lo, vget_low_s16(s), c as i16);
                hi = vmlal_high_n_s16(hi, s, c as i16);
            }
            vst1q_s32(dst.as_mut_ptr().add(x), vshlq_s32(lo, shifts));
            vst1q_s32(dst.as_mut_ptr().add(x + 4), vshlq_s32(hi, shifts));
        }
    }
    for x in end..w {
        let sum: i32 = (0..8).map(|k| coeff[k] * src[x + k].to_i32()).sum();
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

#[inline(always)]
unsafe fn store_samples<P: Pixel>(dst: *mut P, values: uint16x8_t) {
    // SAFETY: callers prove eight writable samples. Pixel is sealed to
    // u8/u16/i32; narrowing matches from_i32 and widening is exact.
    unsafe {
        if core::mem::size_of::<P>() == 1 {
            vst1_u8(dst.cast(), vmovn_u16(values));
        } else if core::mem::size_of::<P>() == 2 {
            vst1q_u16(dst.cast(), values);
        } else {
            let dst = dst.cast::<i32>();
            vst1q_s32(dst, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(values))));
            vst1q_s32(dst.add(4), vreinterpretq_s32_u32(vmovl_high_u16(values)));
        }
    }
}

pub(super) fn clip<P: Pixel>(depth: u8, src: &[i32], dst: &mut [P]) {
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
            store_samples(dst.as_mut_ptr().add(x), vcombine_u16(lo, hi));
        }
    }
    for x in end..src.len() { dst[x] = P::from_i32(src[x].clamp(0, max)); }
}

pub(super) fn average<P: Pixel>(depth: u8, shift: u32, a: &[i32], b: &[i32], dst: &mut [P]) {
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
            store_samples(dst.as_mut_ptr().add(x), vcombine_u16(v[0], v[1]));
        }
    }
    for x in end..a.len() {
        dst[x] = P::from_i32((((i64::from(a[x]) + i64::from(b[x]) + (1i64 << (shift - 1))) >> shift) as i32).clamp(0, max));
    }
}

#[inline(always)]
pub(super) fn project_destinations(mv: [i32; 2], multipliers: &[i32; 8]) -> [[i16; 2]; 8] {
    let mut out = [[0; 2]; 8];
    // SAFETY: each multiplier load covers four of the eight entries; each
    // interleaved store covers four two-component outputs. Wrapping products,
    // signed rounding and the 15-bit clamp match the scalar projection.
    unsafe {
        for first in [0, 4] {
            let multiplier = vld1q_s32(multipliers.as_ptr().add(first));
            let components = mv.map(|value| {
                let product = vmulq_n_s32(multiplier, value);
                let magnitude = vshrq_n_s32::<14>(vaddq_s32(vabsq_s32(product), vdupq_n_s32(8192)));
                let signed = vbslq_s32(vcltq_s32(product, vdupq_n_s32(0)), vnegq_s32(magnitude), magnitude);
                vmovn_s32(vminq_s32(vdupq_n_s32(16383), vmaxq_s32(vdupq_n_s32(-16383), signed)))
            });
            let pairs = vzip_s16(components[0], components[1]);
            vst1q_s16(out.as_mut_ptr().add(first).cast(), vcombine_s16(pairs.0, pairs.1));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inter_pred::{filter_8tap_h_scalar, filter_8tap_v_scalar, SUBPEL_FILTERS};
    fn next(s: &mut u32) -> u32 { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; *s }
    #[test]
    fn destination_projection_matches_scalar_wrapping_and_rounding() {
        let mut state = 0x71a49a31;
        for case in 0..4096 {
            let mv = if case < 7 {
                let value = [-32768, -4095, -1, 0, 1, 4095, 32767][case];
                [value, -value]
            } else { [next(&mut state) as i16 as i32, next(&mut state) as i16 as i32] };
            let multipliers = core::array::from_fn(|_| next(&mut state) as i32);
            let actual = project_destinations(mv, &multipliers);
            for (dst, &multiplier) in multipliers.iter().enumerate() {
                let expected = mv.map(|value| {
                    let product = value.wrapping_mul(multiplier);
                    let magnitude = product.wrapping_abs().wrapping_add(8192) >> 14;
                    let signed = if product < 0 { magnitude.wrapping_neg() } else { magnitude };
                    signed.clamp(-16383, 16383) as i16
                });
                assert_eq!(actual[dst], expected, "case={case} dst={dst}");
            }
        }
    }
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
                        if depth == 8 {
                            let native: Vec<u8> = src.iter().map(|&v| v as u8).collect();
                            let mut native_actual = vec![0; w * 8];
                            for r in 0..8 {
                                horizontal(&native[r * (w + 7)..], coeff, round0, &mut native_actual[r * w..], w);
                            }
                            assert_eq!(native_actual, expected, "native u8 horizontal w={w}");
                        }
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
    fn combinations_match_full_range_scalar() {
        let mut state = 0xa197c489;
        for n in 0..=513 {
            let a: Vec<i32> = (0..n).map(|i| match i % 8 { 0 => i32::MIN, 1 => i32::MAX,
                2 => -1, 3 => 4095, _ => next(&mut state) as i32 }).collect();
            let b: Vec<i32> = (0..n).map(|_| next(&mut state) as i32).collect();
            for depth in [8, 10, 12] {
                let max = (1 << depth) - 1;
                let mut dst = vec![0; n]; clip(depth, &a, &mut dst);
                assert_eq!(dst, a.iter().map(|&v| v.clamp(0, max) as u16).collect::<Vec<_>>());
                let mut wide = vec![0i32; n];
                clip(depth, &a, &mut wide);
                assert_eq!(wide, dst.iter().map(|&v| i32::from(v)).collect::<Vec<_>>());
                let mut native8 = vec![0u8; n];
                if depth == 8 {
                    clip(depth, &a, &mut native8);
                    assert_eq!(native8, dst.iter().map(|&v| v as u8).collect::<Vec<_>>());
                }
                for shift in 1..=8 {
                    average(depth, shift, &a, &b, &mut dst);
                    let expected: Vec<u16> = a.iter().zip(&b).map(|(&a, &b)|
                        (((i64::from(a) + i64::from(b) + (1 << (shift - 1))) >> shift) as i32).clamp(0, max) as u16).collect();
                    assert_eq!(dst, expected);
                    average(depth, shift, &a, &b, &mut wide);
                    assert_eq!(wide, expected.iter().map(|&v| i32::from(v)).collect::<Vec<_>>());
                    if depth == 8 {
                        average(depth, shift, &a, &b, &mut native8);
                        assert_eq!(native8, expected.iter().map(|&v| v as u8).collect::<Vec<_>>());
                    }
                }
            }
        }
    }
}
