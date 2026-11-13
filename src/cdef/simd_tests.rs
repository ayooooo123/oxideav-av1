//! The explicit-lane CDEF kernels (`cdef/neon.rs`, `cdef/wasm.rs`) against
//! the §7.15.2 / §7.15.3 sample processes written out literally.
use super::simd::{filter, partial};
use crate::cdef::{CDEF_DIRECTIONS, CDEF_PRI_TAPS, CDEF_SEC_TAPS, constrain_adj};
use crate::loop_filter::Sample;

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
                        if case - max <= 64 { max } else { 0 }
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
                Some(expected),
                "depth={depth} case={case}"
            );
            let native16 = src.map(|v| v as u16);
            assert_eq!(partial(&native16, base, stride, depth - 8), Some(expected));
            if depth == 8 {
                let native8 = src.map(|v| v as u8);
                assert_eq!(partial(&native8, base, stride, 0), Some(expected));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn native_filter_matches<const W: usize, P: Sample>(
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
    // `expected` marks the samples outside the block with 0x123456, which no
    // sample storage holds: those must keep the in-range sentinel below.
    const UNTOUCHED: i32 = 0x55;
    let source: Vec<P> = source.iter().map(|&v| P::narrow(v)).collect();
    let mut actual = vec![P::narrow(UNTOUCHED); expected.len()];
    assert!(filter::<W, P>(
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
        adj
    ));
    for (&a, &b) in actual.iter().zip(expected) {
        assert_eq!(a.widen(), if b == 0x123456 { UNTOUCHED } else { b });
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
        for (dir, directions) in CDEF_DIRECTIONS.iter().enumerate() {
            for strength in 0..=15 {
                for secondary in [0, 1, 2, 4] {
                    let mut pri = [(0, 0); 4];
                    let mut sec = [(0, 0); 8];
                    for k in 0..2 {
                        let [dy, dx] = directions[k];
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
                                    2 => {
                                        if i & 1 == 0 {
                                            0
                                        } else {
                                            max
                                        }
                                    }
                                    3 => {
                                        if (i % STRIDE) & 1 == 0 {
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
                            // The §7.15.3 sample process, every tap available:
                            // each tap widens the min / max envelope whatever
                            // its strength.
                            for y in 0..height {
                                for x in 0..W {
                                    let at = BASE + y * STRIDE + x;
                                    let centre = src[at];
                                    let (mut sum, mut lo, mut hi) = (0, centre, centre);
                                    let taps = pri
                                        .iter()
                                        .map(|&t| (t, strength as i32 * scale))
                                        .chain(sec.iter().map(|&t| (t, secondary * scale)));
                                    for ((off, weight), threshold) in taps {
                                        let p = src[(at as isize + off) as usize];
                                        sum += weight
                                            * constrain_adj(p - centre, threshold, adj.min(31));
                                        lo = lo.min(p);
                                        hi = hi.max(p);
                                    }
                                    let delta = (8 + sum - i32::from(sum < 0)) >> 4;
                                    expected[y * STRIDE + x] = (centre + delta).clamp(lo, hi);
                                }
                            }
                            assert!(filter::<W, i32>(
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
                                adj
                            ));
                            assert_eq!(
                                actual, expected,
                                "W={W} depth={depth} direction={dir} strength={strength} secondary={secondary} adj={adj} case={case}"
                            );
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
