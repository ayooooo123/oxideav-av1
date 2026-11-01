//! Exact AV1 convolution and combination over eight SIMD128 sample lanes.
use crate::loop_filter::Pixel;
use core::arch::wasm32::*;

#[inline(always)]
unsafe fn samples<P: Pixel>(p: *const P) -> v128 {
    // SAFETY: callers prove eight readable samples. Pixel is sealed to u8/u16/i32;
    // clipped AV1 samples fit signed 16 bits, so narrowing is exact.
    unsafe {
        if core::mem::size_of::<P>() == 1 {
            i16x8_extend_low_u8x16(v128_load64_zero(p.cast()))
        } else if core::mem::size_of::<P>() == 2 {
            v128_load(p.cast())
        } else {
            i16x8_narrow_i32x4(v128_load(p.cast()), v128_load(p.add(4).cast()))
        }
    }
}

pub(super) fn horizontal<R: Pixel>(
    src: &[R],
    coeff: &[i32; 8],
    shift: u32,
    dst: &mut [i32],
    w: usize,
) {
    assert!(src.len() >= w + 7 && dst.len() >= w);
    assert!((1..32).contains(&shift));
    let end = w / 8 * 8;
    // SAFETY: the checked source includes every tap of each eight-sample group;
    // the destination covers both four-lane stores. AV1 products fit i32.
    unsafe {
        for x in (0..end).step_by(8) {
            let mut lo = i32x4_splat(1 << (shift - 1));
            let mut hi = lo;
            for (k, &c) in coeff.iter().enumerate() {
                let v = samples(src.as_ptr().add(x + k));
                let c = i16x8_splat(c as i16);
                lo = i32x4_add(lo, i32x4_extmul_low_i16x8(v, c));
                hi = i32x4_add(hi, i32x4_extmul_high_i16x8(v, c));
            }
            v128_store(dst.as_mut_ptr().add(x).cast(), i32x4_shr(lo, shift));
            v128_store(dst.as_mut_ptr().add(x + 4).cast(), i32x4_shr(hi, shift));
        }
    }
    for x in end..w {
        let sum: i32 = (0..8).map(|k| coeff[k] * src[x + k].to_i32()).sum();
        dst[x] = (sum + (1 << (shift - 1))) >> shift;
    }
}

pub(super) fn vertical(
    src: &[i32],
    w: usize,
    r: usize,
    coeff: &[i32; 8],
    shift: u32,
    dst: &mut [i32],
) {
    assert!(src.len() >= (r + 8) * w && dst.len() >= w);
    assert!((1..32).contains(&shift));
    let end = w / 4 * 4;
    // SAFETY: all eight source rows and four-element destinations were checked.
    // AV1 horizontal intermediate bounds keep every product and sum in i32.
    unsafe {
        for x in (0..end).step_by(4) {
            let mut sum = i32x4_splat(1 << (shift - 1));
            for (k, &c) in coeff.iter().enumerate() {
                sum = i32x4_add(
                    sum,
                    i32x4_mul(
                        v128_load(src.as_ptr().add((r + k) * w + x).cast()),
                        i32x4_splat(c),
                    ),
                );
            }
            v128_store(dst.as_mut_ptr().add(x).cast(), i32x4_shr(sum, shift));
        }
    }
    for x in end..w {
        let sum: i32 = (0..8).map(|k| coeff[k] * src[(r + k) * w + x]).sum();
        dst[x] = (sum + (1 << (shift - 1))) >> shift;
    }
}

#[inline(always)]
unsafe fn store<P: Pixel>(dst: *mut P, lo: v128, hi: v128) {
    // SAFETY: callers prove eight writable samples; values are clipped to 0..4095.
    unsafe {
        let narrow = i16x8_narrow_i32x4(lo, hi);
        if core::mem::size_of::<P>() == 1 {
            v128_store64_lane::<0>(u8x16_narrow_i16x8(narrow, narrow), dst.cast());
        } else if core::mem::size_of::<P>() == 2 {
            v128_store(dst.cast(), narrow);
        } else {
            v128_store(dst.cast(), lo);
            v128_store(dst.add(4).cast(), hi);
        }
    }
}

pub(super) fn clip<P: Pixel>(depth: u8, src: &[i32], dst: &mut [P]) {
    assert!(dst.len() >= src.len());
    let end = src.len() / 8 * 8;
    let max = (1 << depth) - 1;
    // SAFETY: both slices cover every eight-lane load and store.
    unsafe {
        for x in (0..end).step_by(8) {
            let clip = |v| i32x4_min(i32x4_splat(max), i32x4_max(i32x4_splat(0), v));
            store(
                dst.as_mut_ptr().add(x),
                clip(v128_load(src.as_ptr().add(x).cast())),
                clip(v128_load(src.as_ptr().add(x + 4).cast())),
            );
        }
    }
    for x in end..src.len() {
        dst[x] = P::from_i32(src[x].clamp(0, max));
    }
}

pub(super) fn average<P: Pixel>(depth: u8, shift: u32, a: &[i32], b: &[i32], dst: &mut [P]) {
    assert!(a.len() <= b.len() && a.len() <= dst.len());
    assert!((1..32).contains(&shift));
    let end = a.len() / 8 * 8;
    let max = (1 << depth) - 1;
    // SAFETY: slices cover every vector. The bitwise halving sum cannot overflow
    // even at i32 extrema. Rounding the low remainder separately preserves i64
    // scalar semantics without overflowing the halved sum near i32::MAX.
    unsafe {
        for x in (0..end).step_by(8) {
            let mut result = [i32x4_splat(0); 2];
            for (half, out) in result.iter_mut().enumerate() {
                let av = v128_load(a.as_ptr().add(x + 4 * half).cast());
                let bv = v128_load(b.as_ptr().add(x + 4 * half).cast());
                let xor = v128_xor(av, bv);
                let sum = i32x4_add(v128_and(av, bv), i32x4_shr(xor, 1));
                let rounded = if shift == 1 {
                    i32x4_add(sum, v128_and(xor, i32x4_splat(1)))
                } else {
                    let s = shift - 1;
                    let remainder = v128_and(sum, i32x4_splat((1 << s) - 1));
                    i32x4_add(
                        i32x4_shr(sum, s),
                        i32x4_shr(i32x4_add(remainder, i32x4_splat(1 << (s - 1))), s),
                    )
                };
                *out = i32x4_min(i32x4_splat(max), i32x4_max(i32x4_splat(0), rounded));
            }
            store(dst.as_mut_ptr().add(x), result[0], result[1]);
        }
    }
    for x in end..a.len() {
        dst[x] = P::from_i32(
            (((i64::from(a[x]) + i64::from(b[x]) + (1i64 << (shift - 1))) >> shift) as i32)
                .clamp(0, max),
        );
    }
}

/// Project one saved motion vector onto all eight destination reference slots.
#[inline(always)]
pub(super) fn project_destinations(mv: [i32; 2], multipliers: &[i32; 8]) -> [[i16; 2]; 8] {
    let mut out = [[0; 2]; 8];
    // SAFETY: two four-lane multiplier loads and output stores exactly cover
    // their eight-element arrays. Packed components are clipped to signed 15
    // bits before narrowing, and SIMD wrapping/rounding matches the scalar twin.
    unsafe {
        for first in [0, 4] {
            let multiplier = v128_load(multipliers.as_ptr().add(first).cast());
            let components = mv.map(|value| {
                let product = i32x4_mul(i32x4_splat(value), multiplier);
                let magnitude = i32x4_shr(i32x4_add(i32x4_abs(product), i32x4_splat(8192)), 14);
                let signed = v128_bitselect(
                    i32x4_neg(magnitude),
                    magnitude,
                    i32x4_lt(product, i32x4_splat(0)),
                );
                i32x4_min(i32x4_splat(16383), i32x4_max(i32x4_splat(-16383), signed))
            });
            let pairs = i16x8_narrow_i32x4(components[0], components[1]);
            let interleaved = i16x8_shuffle::<0, 4, 1, 5, 2, 6, 3, 7>(pairs, pairs);
            v128_store(out.as_mut_ptr().add(first).cast(), interleaved);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inter_pred::{filter_8tap_h_scalar, filter_8tap_v_scalar, SUBPEL_FILTERS};
    #[test]
    fn motion_field_destinations_match_scalar() {
        let mut state = 0xa19294bdu32;
        for numerator in -31..=31 {
            for denominator in 1..=31 {
                let multiplier = numerator * (16384 / denominator);
                for value in [-4095, -63, -1, 0, 1, 63, 4095] {
                    let mv = [value, -value];
                    let multipliers = core::array::from_fn(|i| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        if i & 1 == 0 {
                            multiplier
                        } else {
                            (state % 63) as i32 * (16384 / denominator) - 31 * (16384 / denominator)
                        }
                    });
                    let actual = project_destinations(mv, &multipliers);
                    for i in 0..8 {
                        assert_eq!(
                            actual[i],
                            crate::inter_pred::project_motion_vector(mv, multipliers[i])
                                .map(|v| v as i16)
                        );
                    }
                }
            }
        }
    }
    fn next(s: &mut u32) -> u32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        *s
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
                        let src: Vec<u16> = (0..(w + 7) * 8)
                            .map(|i| match case {
                                0 => 0,
                                1 => max,
                                2 => {
                                    if i & 1 == 0 {
                                        0
                                    } else {
                                        max
                                    }
                                }
                                _ => (next(&mut state) & u32::from(max)) as u16,
                            })
                            .collect();
                        let mut expected = vec![0; w * 8];
                        let mut actual = expected.clone();
                        for r in 0..8 {
                            filter_8tap_h_scalar(
                                &src[r * (w + 7)..],
                                coeff,
                                round0,
                                &mut expected[r * w..],
                                w,
                            );
                            horizontal(&src[r * (w + 7)..], coeff, round0, &mut actual[r * w..], w);
                        }
                        assert_eq!(actual, expected, "horizontal depth={depth} w={w}");
                        if depth == 8 {
                            let native: Vec<u8> = src.iter().map(|&v| v as u8).collect();
                            let mut native_actual = vec![0; w * 8];
                            for r in 0..8 {
                                horizontal(
                                    &native[r * (w + 7)..],
                                    coeff,
                                    round0,
                                    &mut native_actual[r * w..],
                                    w,
                                );
                            }
                            assert_eq!(native_actual, expected, "native u8 horizontal w={w}");
                        }
                        for shift in [7, 9, 11] {
                            let mut a = vec![0; w];
                            let mut b = a.clone();
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
            let a: Vec<i32> = (0..n)
                .map(|i| match i % 8 {
                    0 => i32::MIN,
                    1 => i32::MAX,
                    2 => -1,
                    3 => 4095,
                    _ => next(&mut state) as i32,
                })
                .collect();
            let b: Vec<i32> = (0..n).map(|_| next(&mut state) as i32).collect();
            for depth in [8, 10, 12] {
                let max = (1 << depth) - 1;
                let mut dst = vec![0; n];
                clip(depth, &a, &mut dst);
                assert_eq!(
                    dst,
                    a.iter()
                        .map(|&v| v.clamp(0, max) as u16)
                        .collect::<Vec<_>>()
                );
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
                    let expected: Vec<u16> = a
                        .iter()
                        .zip(&b)
                        .map(|(&a, &b)| {
                            (((i64::from(a) + i64::from(b) + (1 << (shift - 1))) >> shift) as i32)
                                .clamp(0, max) as u16
                        })
                        .collect();
                    assert_eq!(dst, expected);
                    average(depth, shift, &a, &b, &mut wide);
                    assert_eq!(
                        wide,
                        expected.iter().map(|&v| i32::from(v)).collect::<Vec<_>>()
                    );
                    if depth == 8 {
                        average(depth, shift, &a, &b, &mut native8);
                        assert_eq!(
                            native8,
                            expected.iter().map(|&v| v as u8).collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
}
