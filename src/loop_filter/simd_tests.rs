//! The explicit-lane horizontal-edge kernel (`loop_filter/neon.rs`,
//! `loop_filter/wasm.rs`) against [`sample_filtering`] column by column.
use super::simd::edge;
use super::{PlaneBuffer, sample_filtering};
#[test]
fn contiguous_edges_match_scalar_masks_and_depths() {
    let mut seed = 0x73a149bdu32;
    for depth in [8, 10, 12] {
        let max = (1 << depth) - 1;
        for case in 0..96 {
            let samples: Vec<u16> = (0..20 * 20)
                .map(|i| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    if case % 3 == 0 {
                        ((max / 2) + (i % 3) - 1) as u16
                    } else if case % 3 == 1 {
                        (max / 2 + (i / 20 % 2) * (1 << (depth - 8))) as u16
                    } else {
                        (seed as i32 & max) as u16
                    }
                })
                .collect();
            for plane in [0, 1] {
                for size in [4, 8, 16] {
                    for y in [1, 8, 19] {
                        let mut a = samples.clone();
                        let mut b = samples.clone();
                        let mut expected = PlaneBuffer {
                            rows: 20,
                            cols: 20,
                            samples: &mut a,
                        };
                        let mut actual = PlaneBuffer {
                            rows: 20,
                            cols: 20,
                            samples: &mut b,
                        };
                        let limit = 1 + (case % 63) as u8;
                        let blimit = 128 + (case % 64) as u8;
                        let thresh = (case % 4) as u8;
                        for x in 4..8 {
                            sample_filtering(
                                &mut expected,
                                x,
                                y,
                                plane,
                                limit,
                                blimit,
                                thresh,
                                0,
                                1,
                                size,
                                depth,
                            );
                        }
                        assert!(edge(
                            &mut actual,
                            4,
                            y,
                            plane,
                            limit,
                            blimit,
                            thresh,
                            size,
                            depth
                        ));
                        assert_eq!(
                            a, b,
                            "depth={depth} case={case} plane={plane} size={size} y={y}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn vertical_edges_match_scalar_masks_and_depths() {
    use super::simd::edge_vertical;
    let mut seed = 0x51c7a3e9u32;
    for depth in [8, 10, 12] {
        let max = (1 << depth) - 1;
        for case in 0..96 {
            let samples: Vec<u16> = (0..20 * 20)
                .map(|i| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    if case % 3 == 0 {
                        ((max / 2) + (i % 3) - 1) as u16
                    } else if case % 3 == 1 {
                        (max / 2 + (i % 20 % 2) * (1 << (depth - 8))) as u16
                    } else {
                        (seed as i32 & max) as u16
                    }
                })
                .collect();
            for plane in [0, 1] {
                for size in [4, 8, 16] {
                    for x in [7, 8, 12] {
                        let mut a = samples.clone();
                        let mut b = samples.clone();
                        let mut expected = PlaneBuffer {
                            rows: 20,
                            cols: 20,
                            samples: &mut a,
                        };
                        let mut actual = PlaneBuffer {
                            rows: 20,
                            cols: 20,
                            samples: &mut b,
                        };
                        let limit = 1 + (case % 63) as u8;
                        let blimit = 128 + (case % 64) as u8;
                        let thresh = (case % 4) as u8;
                        for y in 4..8 {
                            sample_filtering(
                                &mut expected,
                                x,
                                y,
                                plane,
                                limit,
                                blimit,
                                thresh,
                                1,
                                0,
                                size,
                                depth,
                            );
                        }
                        assert!(edge_vertical(
                            &mut actual,
                            x,
                            4,
                            plane,
                            limit,
                            blimit,
                            thresh,
                            size,
                            depth
                        ));
                        assert_eq!(
                            a, b,
                            "vertical depth={depth} case={case} plane={plane} size={size} x={x}"
                        );
                    }
                    // Border edges with `x < reach` or `x + reach > cols` fall back cleanly.
                    if size == 16 {
                        let mut b = samples.clone();
                        let mut actual = PlaneBuffer {
                            rows: 20,
                            cols: 20,
                            samples: &mut b,
                        };
                        assert!(!edge_vertical(&mut actual, 1, 4, plane, 10, 140, 1, size, depth));
                        assert!(!edge_vertical(&mut actual, 19, 4, plane, 10, 140, 1, size, depth));
                        assert_eq!(samples, b);
                    }
                }
            }
        }
    }
}
