//! The explicit-lane prediction rows (`inter_pred/neon.rs`) against the
//! §7.11.3.4 / §7.11.3.1 sample processes written out literally.
use super::simd::{average, clip, convolve_row};

fn next(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn convolve_rows_match_spec_every_filter_width_and_sample_range() {
    const UNTOUCHED: i32 = 0x1234_5678;
    let mut state = 0x5bd1_e995;
    // Every `Subpel_Filters` row, then taps at the limits the kernel takes.
    let mut kernels: Vec<[i32; 8]> = crate::inter_pred::SUBPEL_FILTERS
        .iter()
        .flat_map(|set| set.iter().copied())
        .collect();
    kernels.push([-65535, 65535, -65535, 65535, -65535, 65535, -65535, 65535]);
    kernels.push([65535; 8]);
    kernels.push([0; 8]);
    for (index, k) in kernels.iter().enumerate() {
        for w in [8usize, 16, 24, 32, 64, 128] {
            for (case, shift) in [(0, 3u32), (1, 5), (2, 3), (3, 5), (4, 1), (5, 31)] {
                let src: Vec<u16> = (0..w + 7)
                    .map(|i| match case {
                        0 => 0xffff,
                        1 => [0, 0x0fff][i & 1],
                        2 => (next(&mut state) & 0xff) as u16,
                        3 => (next(&mut state) & 0x0fff) as u16,
                        _ => next(&mut state) as u16,
                    })
                    .collect();
                // One slot past the row: the kernel must not write it.
                let mut out = vec![UNTOUCHED; w + 1];
                assert!(convolve_row(k, &src, &mut out[..w], shift));
                for c in 0..w {
                    let sum = (0..8).fold(0i32, |sum, t| {
                        sum.wrapping_add(k[t].wrapping_mul(i32::from(src[c + t])))
                    });
                    let expected = sum.wrapping_add(1 << (shift - 1)) >> shift;
                    assert_eq!(out[c], expected, "kernel={index} w={w} case={case} c={c}");
                }
                assert_eq!(out[w], UNTOUCHED);
            }
        }
    }
}

#[test]
fn convolve_row_declines_what_it_does_not_handle() {
    let k = [0, 2, -10, 72, 72, -10, 2, 0];
    let src = [100u16; 71];
    for w in [0usize, 2, 4, 12, 20] {
        let mut out = vec![7i32; w];
        assert!(!convolve_row(&k, &src, &mut out, 3));
        assert!(out.iter().all(|&v| v == 7), "w={w}");
    }
    let mut out = [7i32; 8];
    assert!(!convolve_row(&k, &src, &mut out, 0));
    assert!(!convolve_row(&k, &src, &mut out, 32));
    assert!(!convolve_row(
        &[0, 0, 0, 65536, 0, 0, 0, 0],
        &src,
        &mut out,
        3
    ));
    assert_eq!(out, [7; 8]);
}

fn prediction(state: &mut u32, case: usize, i: usize) -> i32 {
    match case {
        0 => i32::MAX,
        1 => i32::MIN,
        2 => [i32::MIN, i32::MAX, -1, 0, 1, 4095, 4096, 65535, 65536][i % 9],
        // A compound prediction's range at 12 bits, and beyond it.
        3 => (next(state) % (1 << 17)) as i32 - (1 << 16),
        _ => next(state) as i32,
    }
}

#[test]
fn clip_matches_spec_every_depth_length_and_extreme() {
    let mut state = 0x2545_f491;
    for max in [255, 1023, 4095, 65535] {
        for len in [0usize, 1, 7, 8, 9, 16, 31, 64, 130] {
            for case in 0..6 {
                let pred: Vec<i32> = (0..len).map(|i| prediction(&mut state, case, i)).collect();
                // The output may be longer than the prediction: the rest stays.
                let mut out = vec![0xa5a5u16; len + 3];
                assert!(clip(max, &pred, &mut out));
                for (i, &p) in pred.iter().enumerate() {
                    assert_eq!(
                        i32::from(out[i]),
                        p.clamp(0, max),
                        "max={max} len={len} case={case} i={i}"
                    );
                }
                assert!(out[len..].iter().all(|&v| v == 0xa5a5));
            }
        }
    }
    let mut out = [9u16; 8];
    assert!(!clip(65536, &[1; 8], &mut out));
    assert!(!clip(-1, &[1; 8], &mut out));
    assert!(!clip(255, &[1; 9], &mut out));
    assert_eq!(out, [9; 8]);
}

#[test]
fn average_matches_spec_every_depth_shift_and_extreme() {
    let mut state = 0x9e37_79b9;
    for max in [255, 1023, 4095, 65535] {
        for shift in [1u32, 2, 3, 5, 7, 16, 31] {
            for len in [0usize, 1, 7, 8, 9, 16, 31, 64, 130] {
                for case in 0..6 {
                    let a: Vec<i32> = (0..len).map(|i| prediction(&mut state, case, i)).collect();
                    let b: Vec<i32> = (0..len)
                        .map(|i| prediction(&mut state, (case + i) % 6, i / 2))
                        .collect();
                    let mut out = vec![0xa5a5u16; len + 3];
                    assert!(average(max, shift, &a, &b, &mut out));
                    for i in 0..len {
                        let sum = i64::from(a[i]) + i64::from(b[i]);
                        let rounded = (sum + (1i64 << (shift - 1))) >> shift;
                        assert_eq!(
                            i64::from(out[i]),
                            rounded.clamp(0, i64::from(max)),
                            "max={max} shift={shift} len={len} case={case} i={i}"
                        );
                    }
                    assert!(out[len..].iter().all(|&v| v == 0xa5a5));
                }
            }
        }
    }
    let mut out = [9u16; 8];
    assert!(!average(65536, 1, &[1; 8], &[1; 8], &mut out));
    assert!(!average(255, 0, &[1; 8], &[1; 8], &mut out));
    assert!(!average(255, 32, &[1; 8], &[1; 8], &mut out));
    assert!(!average(255, 1, &[1; 8], &[1; 7], &mut out));
    assert!(!average(255, 1, &[1; 9], &[1; 9], &mut out));
    assert_eq!(out, [9; 8]);
}
