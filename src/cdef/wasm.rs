//! §7.15 arithmetic in eight signed 16-bit SIMD128 lanes: the same sums,
//! envelopes and rounding as the scalar process (see `cdef/neon.rs` for the
//! bounds that make 16 bits exact).
//!
//! WebAssembly has no run-time feature detection: this module is compiled
//! only when the target enables `simd128`, so the entry points always run
//! and report `true` / `Some`, with the same signatures as the NEON ones.
use crate::loop_filter::Sample;
use core::arch::wasm32::*;

#[inline(always)]
unsafe fn load<const W: usize, P: Sample>(p: *const P) -> v128 {
    // SAFETY: callers prove W readable samples; W is four or eight. `Sample`
    // storage is u8, u16 or i32 and clipped AV1 values fit signed 16 bits.
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
pub(super) fn filter<const W: usize, P: Sample>(
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
) -> bool {
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
    true
}

pub(super) fn partial<P: Sample>(
    src: &[P],
    base: usize,
    stride: usize,
    shift: i32,
) -> Option<[[i32; 15]; 8]> {
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
    Some(core::array::from_fn(|d| {
        core::array::from_fn(|i| partial[d][i] as i32)
    }))
}
