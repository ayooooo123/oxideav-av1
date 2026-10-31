//! Clean-room §7.15 arithmetic: eight signed 16-bit sample lanes.
//! Samples and differences fit 13 bits; the weighted constrained sum is
//! bounded by 24 * 240 = 5760 at 12-bit depth. No saturating arithmetic
//! changes the specification's result.
use core::arch::aarch64::*;

#[inline(always)]
unsafe fn load<const W: usize>(p: *const i32) -> int16x8_t {
    // SAFETY: the block wrapper proves W readable samples at every tap.
    unsafe {
        let lo = vmovn_s32(vld1q_s32(p));
        let hi = if W == 8 { vmovn_s32(vld1q_s32(p.add(4))) } else { vdup_n_s16(0) };
        vcombine_s16(lo, hi)
    }
}

#[inline(always)]
unsafe fn constrained(d: int16x8_t, strength: int16x8_t, shift: int16x8_t) -> int16x8_t {
    // SAFETY: NEON is mandatory on aarch64; all operands are register values.
    unsafe {
        let a = vabsq_s16(d);
        let limit = vmaxq_s16(vsubq_s16(strength, vshlq_s16(a, shift)), vdupq_n_s16(0));
        let magnitude = vminq_s16(a, limit);
        vbslq_s16(vcltq_s16(d, vdupq_n_s16(0)), vnegq_s16(magnitude), magnitude)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn filter<const W: usize>(
    src: &[i32], base: usize, src_stride: usize, dst: &mut [i32], dst_stride: usize,
    height: usize, pri: &[(isize, i32); 4], sec: &[(isize, i32); 8],
    pri_strength: i32, sec_strength: i32, pri_shift: i32, sec_shift: i32,
) {
    assert!(W == 4 || W == 8);
    assert!(height > 0);
    let mut lowest = 0isize;
    let mut highest = 0isize;
    for &(off, _) in pri.iter().chain(sec.iter()) {
        lowest = lowest.min(off);
        highest = highest.max(off);
    }
    assert!(base as isize + lowest >= 0);
    assert!((base as isize + highest) as usize + (height - 1) * src_stride + W <= src.len());
    assert!((height - 1) * dst_stride + W <= dst.len());
    // SAFETY: the complete source tap rectangle and destination rectangle
    // were checked above, once for this block. Narrowing is exact for AV1
    // 8/10/12-bit samples; shifts >=16 correctly yield zero for |diff|.
    unsafe {
        let ps = vdupq_n_s16(pri_strength as i16);
        let ss = vdupq_n_s16(sec_strength as i16);
        let pshift = vdupq_n_s16(-(pri_shift.min(16) as i16));
        let sshift = vdupq_n_s16(-(sec_shift.min(16) as i16));
        for y in 0..height {
            let row = src.as_ptr().add(base + y * src_stride);
            let center = load::<W>(row);
            let mut min = center;
            let mut max = center;
            let mut sum = vdupq_n_s16(0);
            if pri_strength != 0 {
                for &(off, tap) in pri {
                    let value = load::<W>(row.offset(off));
                    min = vminq_s16(min, value);
                    max = vmaxq_s16(max, value);
                    sum = vmlaq_n_s16(sum, constrained(vsubq_s16(value, center), ps, pshift), tap as i16);
                }
            }
            if sec_strength != 0 {
                for &(off, tap) in sec {
                    let value = load::<W>(row.offset(off));
                    min = vminq_s16(min, value);
                    max = vmaxq_s16(max, value);
                    sum = vmlaq_n_s16(sum, constrained(vsubq_s16(value, center), ss, sshift), tap as i16);
                }
            }
            let negative = vreinterpretq_s16_u16(vcltq_s16(sum, vdupq_n_s16(0)));
            let delta = vshrq_n_s16::<4>(vaddq_s16(vaddq_s16(sum, vdupq_n_s16(8)), negative));
            let out = vminq_s16(max, vmaxq_s16(min, vaddq_s16(center, delta)));
            let target = dst.as_mut_ptr().add(y * dst_stride);
            vst1q_s32(target, vmovl_s16(vget_low_s16(out)));
            if W == 8 { vst1q_s32(target.add(4), vmovl_s16(vget_high_s16(out))); }
        }
    }
}

pub(super) fn partial(src: &[i32], base: usize, stride: usize, shift: i32) -> [[i32; 15]; 8] {
    assert!(base + 7 * stride + 8 <= src.len());
    let mut partial = [[0i16; 16]; 8];
    // SAFETY: the eight complete source rows were checked above. Every
    // projection offset lies in 0..15 (16 allocated lanes), so vector
    // loads/stores stay inside one projection row. Each sum is at most
    // eight normalized samples (-1024..1016), exactly representable in i16.
    unsafe {
        for i in 0..8 {
            let values = vsubq_s16(vshlq_s16(load::<8>(src.as_ptr().add(base + i * stride)),
                vdupq_n_s16(-(shift as i16))), vdupq_n_s16(128));
            let reverse = vextq_s16::<4>(vrev64q_s16(values), vrev64q_s16(values));
            for (dir, offset, v) in [(0, i, values), (4, i, reverse),
                (5, 3 - i / 2, values), (6, 0, values), (7, i / 2, values)] {
                let p = partial[dir].as_mut_ptr().add(offset);
                vst1q_s16(p, vaddq_s16(vld1q_s16(p), v));
            }
            partial[2][i] = vaddvq_s16(values);
            let pairs = vpadd_s16(vget_low_s16(values), vget_high_s16(values));
            let p = partial[1].as_mut_ptr().add(i);
            vst1_s16(p, vadd_s16(vld1_s16(p), pairs));
            let p = partial[3].as_mut_ptr().add(i);
            vst1_s16(p, vadd_s16(vld1_s16(p), vrev64_s16(pairs)));
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
                    *v = if case <= max { case } else if case <= max + 128 {
                        if index % 64 == (case - max - 1) as usize % 64 {
                            if case - max <= 64 { max } else { 0 }
                        } else if case - max <= 64 { 0 } else { max }
                    } else { next(&mut state) as i32 & max };
                }
                let base = 19;
                let stride = 17;
                let mut expected = [[0; 15]; 8];
                for i in 0..8 {
                    for j in 0..8 {
                        let v = (src[base + i * stride + j] >> (depth - 8)) - 128;
                        for (d, k) in [(0, i + j), (1, i + j / 2), (2, i),
                            (3, 3 + i - j / 2), (4, 7 + i - j),
                            (5, 3 - i / 2 + j), (6, j), (7, i / 2 + j)] {
                            expected[d][k] += v;
                        }
                    }
                }
                assert_eq!(partial(&src, base, stride, depth - 8), expected,
                    "depth={depth} case={case}");
            }
        }
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
                            for (s, d) in [((dir + 6) & 7), ((dir + 2) & 7)].into_iter().enumerate() {
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
                                        2 => if i & 1 == 0 { 0 } else { max },
                                        3 => if i % STRIDE & 1 == 0 { max / 2 } else { max / 2 + 1 },
                                        4..=7 => (max / 2 + (i as i32 % 7 - 3) * strength as i32 * scale).clamp(0, max),
                                        _ => next(&mut state) as i32 & max,
                                    };
                                }
                                let height = 1 + case % 8;
                                let mut expected = [0x123456; STRIDE * 8];
                                let mut actual = expected;
                                for y in 0..height {
                                    cdef_interior_row::<W>(&src, BASE + y * STRIDE,
                                        (&mut expected[y * STRIDE..y * STRIDE + W]).try_into().unwrap(),
                                        &pri, &sec, strength as i32 * scale, secondary * scale, adj, adj);
                                }
                                filter::<W>(&src, BASE, STRIDE, &mut actual, STRIDE, height,
                                    &pri, &sec, strength as i32 * scale, secondary * scale, adj, adj);
                                assert_eq!(actual, expected,
                                    "W={W} depth={depth} direction={dir} strength={strength} secondary={secondary} adj={adj} case={case}");
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
