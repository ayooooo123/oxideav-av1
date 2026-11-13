//! The explicit-lane transform (`transform/neon.rs`) against the scalar
//! §7.13.2 kernels and the §7.13.3 passes built from them.
use super::simd::{Kernel, inverse_2d, transform_1d};
use super::tests::inverse_transform_2d_scalar;
use super::{FlowStep, dct_program, inverse_adst, inverse_dct, inverse_identity};
use crate::cdf::{
    ADST_DCT, DCT_ADST, DCT_DCT, DCT_FLIPADST, FLIPADST_DCT, H_ADST, H_DCT, H_FLIPADST, IDTX,
    TX_HEIGHT, TX_SIZES_ALL, TX_WIDTH, V_ADST, V_DCT, V_FLIPADST,
};

fn next(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn kernels_match_scalar_at_every_size_and_precision_edge() {
    let mut state = 0x832b_a071;
    for (kernel, largest) in [(Kernel::Dct, 6), (Kernel::Adst, 4), (Kernel::Identity, 5)] {
        for n in 2..=largest {
            let len = 1usize << n;
            for r in [16u32, 18, 20] {
                let (lo, hi) = (-(1i32 << (r - 1)), (1i32 << (r - 1)) - 1);
                for case in 0..260 {
                    let input: Vec<[i32; 4]> = (0..len)
                        .map(|i| {
                            core::array::from_fn(|lane| match case {
                                0 => lo,
                                1 => hi,
                                2 => [lo, hi][(i + lane) & 1],
                                3 => 0,
                                _ => (next(&mut state) & ((1 << r) - 1)) as i32 + lo,
                            })
                        })
                        .collect();
                    let mut lanes = input.clone();
                    assert!(transform_1d(&mut lanes, kernel, n, r));
                    for lane in 0..4 {
                        let mut scalar: Vec<i64> =
                            input.iter().map(|v| i64::from(v[lane])).collect();
                        match kernel {
                            Kernel::Dct => inverse_dct(&mut scalar, n, r),
                            Kernel::Adst => inverse_adst(&mut scalar, n, r),
                            Kernel::Identity => inverse_identity(&mut scalar, n),
                        }
                        for i in 0..len {
                            assert_eq!(
                                i64::from(lanes[i][lane]),
                                scalar[i],
                                "kernel={kernel:?} n={n} r={r} case={case} lane={lane} i={i}"
                            );
                        }
                    }
                }
            }
        }
    }
    let mut lanes = [[1i32; 4]; 32];
    assert!(!transform_1d(&mut lanes, Kernel::Adst, 5, 16));
    assert!(!transform_1d(&mut lanes[..2], Kernel::Dct, 1, 16));
    assert_eq!(lanes, [[1; 4]; 32]);
}

/// Every value the DCT lanes form from coefficients of the widest clamp
/// (20 bits) fits 32 bits: a magnitude bound per slot through the recorded
/// schedule, `H()` clamping its outputs as the lanes do.
#[test]
fn dct_lanes_stay_inside_32_bits_at_the_widest_clamp() {
    let clamp = 1i64 << 19;
    for n in 2..=6 {
        let mut bound = [clamp; 64];
        for step in &dct_program(n).steps {
            match *step {
                FlowStep::Rotate {
                    a,
                    b,
                    cos,
                    sin,
                    x_to,
                    y_to,
                } => {
                    let (a, b) = (bound[usize::from(a)], bound[usize::from(b)]);
                    let (cos, sin) = (i64::from(cos.abs()), i64::from(sin.abs()));
                    let x = (cos * a + sin * b + 2048) >> 12;
                    let y = (sin * a + cos * b + 2048) >> 12;
                    assert!(x.max(y) <= i64::from(i32::MAX), "n={n}");
                    bound[usize::from(x_to)] = x;
                    bound[usize::from(y_to)] = y;
                }
                FlowStep::AddSub { sum, diff } => {
                    let unclamped = bound[usize::from(sum)] + bound[usize::from(diff)];
                    assert!(unclamped <= i64::from(i32::MAX), "n={n}");
                    bound[usize::from(sum)] = unclamped.min(clamp);
                    bound[usize::from(diff)] = unclamped.min(clamp);
                }
            }
        }
    }
}

#[test]
fn blocks_match_scalar_every_type_size_and_depth() {
    const UNTOUCHED: i64 = 0x1234_5678;
    // The longest array §7.13.2 defines each pass's kernel for.
    let longest_row = |tx_type| match tx_type {
        DCT_DCT | ADST_DCT | FLIPADST_DCT | H_DCT => 64,
        IDTX | V_DCT | V_ADST | V_FLIPADST => 32,
        _ => 16,
    };
    let longest_column = |tx_type| match tx_type {
        DCT_DCT | DCT_ADST | DCT_FLIPADST | V_DCT => 64,
        IDTX | H_DCT | H_ADST | H_FLIPADST => 32,
        _ => 16,
    };
    let mut state = 0x4275_d8a1;
    for tx_sz in 0..TX_SIZES_ALL {
        let (w, h) = (TX_WIDTH[tx_sz], TX_HEIGHT[tx_sz]);
        for tx_type in 0..16 {
            let defined = w <= longest_row(tx_type) && h <= longest_column(tx_type);
            for bit_depth in [8u32, 10, 12] {
                let bits = bit_depth + 8;
                let (lo, hi) = (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1);
                for case in 0..20 {
                    let dequant: Vec<i64> = (0..w * h)
                        .map(|i| {
                            let random = i64::from(next(&mut state) & ((1 << bits) - 1)) + lo;
                            // Rows and columns from 32 on are not coded: zero
                            // as a decoder leaves them, or noise to be ignored.
                            if i / w >= 32 || i % w >= 32 {
                                return if case % 2 == 0 { 0 } else { random };
                            }
                            match case {
                                0 => lo,
                                1 => hi,
                                2 => [lo, hi][i & 1],
                                3 => [0, hi][usize::from(i == 0)],
                                4 => [0, random][usize::from(i / w == 5 % h)],
                                5 => random >> (bits - 4),
                                _ => random,
                            }
                        })
                        .collect();
                    let mut residual = vec![UNTOUCHED; 3];
                    let ran = inverse_2d(&dequant, tx_sz, tx_type, bit_depth, &mut residual);
                    assert_eq!(ran, defined, "tx_sz={tx_sz} tx_type={tx_type}");
                    if ran {
                        assert_eq!(
                            residual,
                            inverse_transform_2d_scalar(&dequant, tx_sz, tx_type, bit_depth),
                            "tx_sz={tx_sz} tx_type={tx_type} bd={bit_depth} case={case}"
                        );
                    } else {
                        assert_eq!(residual, [UNTOUCHED; 3]);
                    }
                }
            }
        }
    }
}

#[test]
fn blocks_outside_the_lanes_are_declined() {
    const UNTOUCHED: i64 = 0x1234_5678;
    let mut residual = vec![UNTOUCHED; 3];
    let mut dequant = vec![0i64; 64];
    for (bit_depth, coefficient) in [(8u32, 1i64 << 15), (8, -(1 << 15) - 1), (12, 1 << 19)] {
        for at in [0, 63] {
            dequant[at] = coefficient;
            assert!(!inverse_2d(&dequant, 1, DCT_DCT, bit_depth, &mut residual));
            dequant[at] = 0;
        }
    }
    assert!(!inverse_2d(&dequant, 1, DCT_DCT, 9, &mut residual));
    assert!(!inverse_2d(&dequant, 1, 16, 8, &mut residual));
    assert!(!inverse_2d(&dequant[..63], 1, DCT_DCT, 8, &mut residual));
    assert_eq!(residual, [UNTOUCHED; 3]);
    assert!(inverse_2d(&dequant, 1, DCT_DCT, 8, &mut residual));
    assert_eq!(residual, [0; 64]);
}
