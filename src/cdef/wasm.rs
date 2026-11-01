//! Exact CDEF filtering and directional projections in eight signed sample lanes.
use crate::loop_filter::Pixel;
use core::arch::wasm32::*;

#[inline(always)]
unsafe fn load<const W: usize, P: Pixel>(p: *const P) -> v128 {
    // SAFETY: callers prove W readable samples; W is four or eight. Pixel is
    // sealed to u8/u16/i32 and clipped AV1 values fit signed 16 bits.
    unsafe {
        if core::mem::size_of::<P>() == 1 {
            i16x8_extend_low_u8x16(if W == 8 {
                v128_load64_zero(p.cast())
            } else {
                v128_load32_zero(p.cast())
            })
        } else if core::mem::size_of::<P>() == 2 {
            if W == 8 {
                v128_load(p.cast())
            } else {
                v128_load64_zero(p.cast())
            }
        } else {
            let lo = v128_load(p.cast());
            i16x8_narrow_i32x4(
                lo,
                if W == 8 {
                    v128_load(p.add(4).cast())
                } else {
                    i32x4_splat(0)
                },
            )
        }
    }
}

#[inline(always)]
fn constrained(d: v128, strength: v128, shift: u32) -> v128 {
    let a = i16x8_abs(d);
    // SIMD shifts mask their count: explicitly cap at 15, where all AV1
    // differences already shift to zero, including damping counts >=16.
    let limit = i16x8_max(
        i16x8_sub(strength, u16x8_shr(a, shift.min(15))),
        i16x8_splat(0),
    );
    let magnitude = i16x8_min(a, limit);
    v128_bitselect(i16x8_neg(magnitude), magnitude, i16x8_lt(d, i16x8_splat(0)))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn filter<const W: usize, P: Pixel>(
    src: &[P],
    base: usize,
    stride: usize,
    dst: &mut [P],
    dst_stride: usize,
    height: usize,
    pri: &[(isize, i32); 4],
    sec: &[(isize, i32); 8],
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
) {
    assert!(W == 4 || W == 8);
    assert!(height > 0);
    let mut lowest = 0;
    let mut highest = 0;
    for &(off, _) in pri.iter().chain(sec.iter()) {
        lowest = lowest.min(off);
        highest = highest.max(off);
    }
    assert!(base as isize + lowest >= 0);
    assert!((base as isize + highest) as usize + (height - 1) * stride + W <= src.len());
    assert!((height - 1) * dst_stride + W <= dst.len());
    // SAFETY: the entire source tap rectangle and output rectangle are checked
    // above. Samples/differences and weighted constrained sums fit signed i16.
    unsafe {
        for y in 0..height {
            let row = src.as_ptr().add(base + y * stride);
            let center = load::<W, P>(row);
            let mut min = center;
            let mut max = center;
            let mut sum = i16x8_splat(0);
            for (taps, strength, shift) in [
                (&pri[..], pri_strength, pri_shift),
                (&sec[..], sec_strength, sec_shift),
            ] {
                if strength != 0 {
                    for &(off, tap) in taps {
                        let value = load::<W, P>(row.offset(off));
                        min = i16x8_min(min, value);
                        max = i16x8_max(max, value);
                        sum = i16x8_add(
                            sum,
                            i16x8_mul(
                                constrained(
                                    i16x8_sub(value, center),
                                    i16x8_splat(strength as i16),
                                    shift as u32,
                                ),
                                i16x8_splat(tap as i16),
                            ),
                        );
                    }
                }
            }
            let delta = i16x8_shr(
                i16x8_add(
                    i16x8_add(sum, i16x8_splat(8)),
                    i16x8_lt(sum, i16x8_splat(0)),
                ),
                4,
            );
            let out = i16x8_min(max, i16x8_max(min, i16x8_add(center, delta)));
            let target = dst.as_mut_ptr().add(y * dst_stride);
            if core::mem::size_of::<P>() == 1 {
                let bytes = u8x16_narrow_i16x8(out, out);
                if W == 8 {
                    v128_store64_lane::<0>(bytes, target.cast());
                } else {
                    v128_store32_lane::<0>(bytes, target.cast());
                }
            } else if core::mem::size_of::<P>() == 2 {
                if W == 8 {
                    v128_store(target.cast(), out);
                } else {
                    v128_store64_lane::<0>(out, target.cast());
                }
            } else {
                v128_store(target.cast(), i32x4_extend_low_i16x8(out));
                if W == 8 {
                    v128_store(target.add(4).cast(), i32x4_extend_high_i16x8(out));
                }
            }
        }
    }
}

pub(super) fn partial<P: Pixel>(
    src: &[P],
    base: usize,
    stride: usize,
    shift: i32,
) -> [[i32; 15]; 8] {
    assert!(base + 7 * stride + 8 <= src.len());
    let mut partial = [[0i16; 16]; 8];
    // SAFETY: source rows are checked above. Every projection offset leaves
    // eight lanes in its 16-element row; sums of normalized pixels fit i16.
    unsafe {
        for i in 0..8 {
            let values = i16x8_sub(
                i16x8_shr(
                    load::<8, P>(src.as_ptr().add(base + i * stride)),
                    shift as u32,
                ),
                i16x8_splat(128),
            );
            let reverse = i16x8_shuffle::<7, 6, 5, 4, 3, 2, 1, 0>(values, values);
            for (dir, offset, v) in [
                (0, i, values),
                (4, i, reverse),
                (5, 3 - i / 2, values),
                (6, 0, values),
                (7, i / 2, values),
            ] {
                let p = partial[dir].as_mut_ptr().add(offset);
                v128_store(p.cast(), i16x8_add(v128_load(p.cast()), v));
            }
            let mut lanes = [0i16; 8];
            v128_store(lanes.as_mut_ptr().cast(), values);
            partial[2][i] = lanes.iter().sum();
            for j in 0..4 {
                let pair = lanes[2 * j] + lanes[2 * j + 1];
                partial[1][i + j] += pair;
                partial[3][3 + i - j] += pair;
            }
        }
    }
    core::array::from_fn(|d| core::array::from_fn(|i| partial[d][i] as i32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdef::{cdef_interior_row, CDEF_DIRECTIONS, CDEF_PRI_TAPS, CDEF_SEC_TAPS};

    fn next(state: &mut u32) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        *state
    }

    #[test]
    fn projections_match_spec_full_depth_and_edges() {
        let mut state = 0x91ab0347;
        for depth in [8, 10, 12] {
            let max = (1 << depth) - 1;
            let mut src = [0i32; 17 * 11];
            for case in 0..max + 4097 {
                for (index, v) in src.iter_mut().enumerate() {
                    *v = if case <= max {
                        case
                    } else if case <= max + 128 {
                        if index % 64 == (case - max - 1) as usize % 64 {
                            if case - max <= 64 {
                                max
                            } else {
                                0
                            }
                        } else if case - max <= 64 {
                            0
                        } else {
                            max
                        }
                    } else {
                        next(&mut state) as i32 & max
                    };
                }
                let base = 19;
                let stride = 17;
                let mut expected = [[0; 15]; 8];
                for i in 0..8 {
                    for j in 0..8 {
                        let v = (src[base + i * stride + j] >> (depth - 8)) - 128;
                        for (d, k) in [
                            (0, i + j),
                            (1, i + j / 2),
                            (2, i),
                            (3, 3 + i - j / 2),
                            (4, 7 + i - j),
                            (5, 3 - i / 2 + j),
                            (6, j),
                            (7, i / 2 + j),
                        ] {
                            expected[d][k] += v;
                        }
                    }
                }
                assert_eq!(
                    partial(&src, base, stride, depth - 8),
                    expected,
                    "depth={depth} case={case}"
                );
                let native16 = src.map(|v| v as u16);
                assert_eq!(partial(&native16, base, stride, depth - 8), expected);
                if depth == 8 {
                    let native8 = src.map(|v| v as u8);
                    assert_eq!(partial(&native8, base, stride, 0), expected);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn native_filter_matches<const W: usize, P: Pixel>(
        source: &[i32],
        expected: &[i32],
        base: usize,
        stride: usize,
        height: usize,
        pri: &[(isize, i32); 4],
        sec: &[(isize, i32); 8],
        strength: i32,
        secondary: i32,
        adj: i32,
    ) {
        let source: Vec<P> = source.iter().map(|&v| P::from_i32(v)).collect();
        let mut actual = vec![P::from_i32(0x123456); expected.len()];
        filter::<W, P>(
            &source,
            base,
            stride,
            &mut actual,
            stride,
            height,
            pri,
            sec,
            strength,
            secondary,
            adj,
            adj,
        );
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(&a, &b)| a == P::from_i32(b)));
    }

    fn filter_cases<const W: usize>() {
        let mut state = 0x8f43c521;
        const STRIDE: usize = 19;
        const BASE: usize = 2 * STRIDE + 3;
        let mut src = [0i32; STRIDE * 13];
        for depth in [8, 10, 12] {
            let scale = 1 << (depth - 8);
            let max = (1 << depth) - 1;
            for dir in 0..8 {
                for strength in 0..=15 {
                    for secondary in [0, 1, 2, 4] {
                        let mut pri = [(0, 0); 4];
                        let mut sec = [(0, 0); 8];
                        for k in 0..2 {
                            let [dy, dx] = CDEF_DIRECTIONS[dir][k];
                            let off = dy as isize * STRIDE as isize + dx as isize;
                            pri[2 * k] = (-off, CDEF_PRI_TAPS[strength & 1][k]);
                            pri[2 * k + 1] = (off, CDEF_PRI_TAPS[strength & 1][k]);
                            for (s, d) in [((dir + 6) & 7), ((dir + 2) & 7)].into_iter().enumerate()
                            {
                                let [dy, dx] = CDEF_DIRECTIONS[d][k];
                                let off = dy as isize * STRIDE as isize + dx as isize;
                                sec[4 * k + 2 * s] = (-off, CDEF_SEC_TAPS[strength & 1][k]);
                                sec[4 * k + 2 * s + 1] = (off, CDEF_SEC_TAPS[strength & 1][k]);
                            }
                        }
                        for adj in [0, 1, 2, 3, 4, 5, 6, 8, 16, 31, 32] {
                            for case in 0..12 {
                                for (i, v) in src.iter_mut().enumerate() {
                                    *v = match case {
                                        0 => 0,
                                        1 => max,
                                        2 => {
                                            if i & 1 == 0 {
                                                0
                                            } else {
                                                max
                                            }
                                        }
                                        3 => {
                                            if i % STRIDE & 1 == 0 {
                                                max / 2
                                            } else {
                                                max / 2 + 1
                                            }
                                        }
                                        4..=7 => (max / 2
                                            + (i as i32 % 7 - 3) * strength as i32 * scale)
                                            .clamp(0, max),
                                        _ => next(&mut state) as i32 & max,
                                    };
                                }
                                let height = 1 + case % 8;
                                let mut expected = [0x123456; STRIDE * 8];
                                let mut actual = expected;
                                for y in 0..height {
                                    cdef_interior_row::<W, i32>(
                                        &src,
                                        BASE + y * STRIDE,
                                        (&mut expected[y * STRIDE..y * STRIDE + W])
                                            .try_into()
                                            .unwrap(),
                                        &pri,
                                        &sec,
                                        strength as i32 * scale,
                                        secondary * scale,
                                        adj,
                                        adj,
                                    );
                                }
                                filter::<W, i32>(
                                    &src,
                                    BASE,
                                    STRIDE,
                                    &mut actual,
                                    STRIDE,
                                    height,
                                    &pri,
                                    &sec,
                                    strength as i32 * scale,
                                    secondary * scale,
                                    adj,
                                    adj,
                                );
                                assert_eq!(actual, expected,
                                    "W={W} depth={depth} direction={dir} strength={strength} secondary={secondary} adj={adj} case={case}");
                                native_filter_matches::<W, u16>(
                                    &src,
                                    &expected,
                                    BASE,
                                    STRIDE,
                                    height,
                                    &pri,
                                    &sec,
                                    strength as i32 * scale,
                                    secondary * scale,
                                    adj,
                                );
                                if depth == 8 {
                                    native_filter_matches::<W, u8>(
                                        &src,
                                        &expected,
                                        BASE,
                                        STRIDE,
                                        height,
                                        &pri,
                                        &sec,
                                        strength as i32 * scale,
                                        secondary * scale,
                                        adj,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn filter_matches_scalar_all_strengths_directions_and_shapes() {
        filter_cases::<4>();
        filter_cases::<8>();
    }
}
