//! Four-lane, symmetric seven-tap Wiener restoration with exact i32 rounding.
use crate::loop_filter::Pixel;
use core::arch::wasm32::*;

#[inline(always)]
unsafe fn convolution(
    rows: [*const i32; 7],
    filter: &[i32; 7],
    shift: u32,
    low: i32,
    high: i32,
) -> v128 {
    // SAFETY: the caller proves four readable i32 samples at each pointer.
    // AV1 coefficient and intermediate bounds keep these sums within i32,
    // exactly as in the scalar restoration pass.
    unsafe {
        let mut sum = i32x4_splat(1 << (shift - 1));
        for k in 0..3 {
            let pair = i32x4_add(v128_load(rows[k].cast()), v128_load(rows[6 - k].cast()));
            sum = i32x4_add(sum, i32x4_mul(pair, i32x4_splat(filter[k])));
        }
        sum = i32x4_add(
            sum,
            i32x4_mul(v128_load(rows[3].cast()), i32x4_splat(filter[3])),
        );
        i32x4_min(
            i32x4_splat(high),
            i32x4_max(i32x4_splat(low), i32x4_shr(sum, shift)),
        )
    }
}

pub(super) fn horizontal(
    src: &[i32],
    filter: &[i32; 7],
    shift: u32,
    low: i32,
    high: i32,
    dst: &mut [i32],
) {
    assert!(src.len() >= dst.len() + 6 && (1..32).contains(&shift));
    let end = dst.len() / 4 * 4;
    // SAFETY: every complete group has four outputs and all seven input taps.
    unsafe {
        for x in (0..end).step_by(4) {
            let p = src.as_ptr().add(x);
            let values = convolution(core::array::from_fn(|k| p.add(k)), filter, shift, low, high);
            v128_store(dst.as_mut_ptr().add(x).cast(), values);
        }
    }
    super::wiener_horizontal_scalar(&src[end..], filter, shift, low, high, &mut dst[end..]);
}

pub(super) fn vertical<P: Pixel>(
    rows: [&[i32]; 7],
    filter: &[i32; 7],
    shift: u32,
    max: i32,
    dst: &mut [P],
) {
    assert!(rows.iter().all(|r| r.len() >= dst.len()) && (1..32).contains(&shift));
    let end = dst.len() / 4 * 4;
    // SAFETY: all seven rows and the destination cover every four-sample group.
    // Values are clipped before native-depth narrowing, so saturation is exact.
    unsafe {
        for x in (0..end).step_by(4) {
            let values = convolution(
                core::array::from_fn(|k| rows[k].as_ptr().add(x)),
                filter,
                shift,
                0,
                max,
            );
            let p = dst.as_mut_ptr().add(x);
            if core::mem::size_of::<P>() == 1 {
                let half = i16x8_narrow_i32x4(values, values);
                v128_store32_lane::<0>(u8x16_narrow_i16x8(half, half), p.cast());
            } else if core::mem::size_of::<P>() == 2 {
                v128_store64_lane::<0>(i16x8_narrow_i32x4(values, values), p.cast());
            } else {
                v128_store(p.cast(), values);
            }
        }
    }
    super::wiener_vertical_scalar(rows.map(|r| &r[end..]), filter, shift, max, &mut dst[end..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wiener_matches_scalar_depths_coefficients_and_tails() {
        let mut seed = 0x1a5394bdu32;
        for depth in [8, 10, 12] {
            let max = (1 << depth) - 1;
            let shift0 = if depth == 12 { 5 } else { 3 };
            let shift1 = 14 - shift0;
            let offset = 1 << (depth + 7 - shift0 - 1);
            let limit = (1 << (depth + 8 - shift0)) - 1;
            for width in 1..=129 {
                for case in 0..32 {
                    let coeff = core::array::from_fn(|k| {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        let low = super::super::WIENER_TAPS_MIN[k];
                        let high = super::super::WIENER_TAPS_MAX[k];
                        match case {
                            0 => low,
                            1 => high,
                            2 => 0,
                            _ => low + (seed % (high - low + 1) as u32) as i32,
                        }
                    });
                    let filter = super::super::wiener_coefficients(&coeff);
                    let input: Vec<i32> = (0..(width + 6) * 7)
                        .map(|i| {
                            seed ^= seed << 13;
                            seed ^= seed >> 17;
                            seed ^= seed << 5;
                            match case {
                                0 => 0,
                                1 => max,
                                2 => {
                                    if i & 1 == 0 {
                                        0
                                    } else {
                                        max
                                    }
                                }
                                _ => seed as i32 & max,
                            }
                        })
                        .collect();
                    let mut expected = vec![0; width * 7];
                    let mut actual = expected.clone();
                    for y in 0..7 {
                        let src = &input[y * (width + 6)..];
                        super::super::wiener_horizontal_scalar(
                            src,
                            &filter,
                            shift0,
                            -offset,
                            limit - offset,
                            &mut expected[y * width..(y + 1) * width],
                        );
                        horizontal(
                            src,
                            &filter,
                            shift0,
                            -offset,
                            limit - offset,
                            &mut actual[y * width..(y + 1) * width],
                        );
                    }
                    assert_eq!(actual, expected);
                    let rows = core::array::from_fn(|y| &actual[y * width..(y + 1) * width]);
                    let mut out = vec![0i32; width];
                    let mut reference = out.clone();
                    super::super::wiener_vertical_scalar(
                        rows,
                        &filter,
                        shift1,
                        max,
                        &mut reference,
                    );
                    vertical(rows, &filter, shift1, max, &mut out);
                    assert_eq!(out, reference);
                    let mut native16 = vec![0u16; width];
                    vertical(rows, &filter, shift1, max, &mut native16);
                    assert_eq!(
                        native16,
                        reference.iter().map(|&v| v as u16).collect::<Vec<_>>()
                    );
                    if depth == 8 {
                        let mut native8 = vec![0u8; width];
                        vertical(rows, &filter, shift1, max, &mut native8);
                        assert_eq!(
                            native8,
                            reference.iter().map(|&v| v as u8).collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
}
