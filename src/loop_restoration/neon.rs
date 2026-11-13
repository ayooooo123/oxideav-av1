//! Exact four-lane symmetric Wiener restoration.
use crate::loop_filter::Pixel;
use core::arch::aarch64::*;

#[inline(always)]
unsafe fn convolution(rows: [*const i32; 7], filter: &[i32; 7], shift: u32, low: i32, high: i32) -> int32x4_t {
    // SAFETY: callers prove four readable elements at all seven pointers.
    unsafe {
        let mut sum = vdupq_n_s32(1 << (shift - 1));
        for k in 0..3 {
            if filter[k] == 0 { continue; }
            sum = vmlaq_n_s32(sum, vaddq_s32(vld1q_s32(rows[k]), vld1q_s32(rows[6 - k])), filter[k]);
        }
        sum = vmlaq_n_s32(sum, vld1q_s32(rows[3]), filter[3]);
        vmaxq_s32(vdupq_n_s32(low), vminq_s32(vdupq_n_s32(high), vshlq_s32(sum, vdupq_n_s32(-(shift as i32)))))
    }
}

pub(super) fn horizontal(src: &[i32], filter: &[i32; 7], shift: u32, low: i32, high: i32, dst: &mut [i32]) {
    assert!(src.len() >= dst.len() + 6 && (1..32).contains(&shift));
    let end = dst.len() / 4 * 4;
    // SAFETY: every complete group has four outputs and all seven taps.
    unsafe {
        for x in (0..end).step_by(4) {
            let p = src.as_ptr().add(x);
            vst1q_s32(dst.as_mut_ptr().add(x), convolution(core::array::from_fn(|k| p.add(k)), filter, shift, low, high));
        }
    }
    super::wiener_horizontal_scalar(&src[end..], filter, shift, low, high, &mut dst[end..]);
}

pub(super) fn vertical<P: Pixel>(rows: [&[i32]; 7], filter: &[i32; 7], shift: u32, max: i32, dst: &mut [P]) {
    assert!(rows.iter().all(|r| r.len() >= dst.len()) && (1..32).contains(&shift));
    let end = dst.len() / 4 * 4;
    // SAFETY: all source rows and the destination cover four elements per group.
    unsafe {
        for x in (0..end).step_by(4) {
            let v = convolution(core::array::from_fn(|k| rows[k].as_ptr().add(x)), filter, shift, 0, max);
            let p = dst.as_mut_ptr().add(x);
            if core::mem::size_of::<P>() == 1 {
                let v = vqmovun_s32(v);
                let bytes = vmovn_u16(vcombine_u16(v, v));
                core::ptr::write_unaligned(p.cast::<u32>(), vget_lane_u32::<0>(vreinterpret_u32_u8(bytes)));
            } else if core::mem::size_of::<P>() == 2 {
                vst1_u16(p.cast(), vqmovun_s32(v));
            } else {
                vst1q_s32(p.cast(), v);
            }
        }
    }
    super::wiener_vertical_scalar(rows.map(|r| &r[end..]), filter, shift, max, &mut dst[end..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wiener_matches_scalar_depths_and_tails() {
        for depth in [8, 10, 12] {
            let max = (1 << depth) - 1;
            let shift = if depth == 12 { 5 } else { 3 };
            for width in 1..=129 {
                for coeff in [[0, 0, 0], [-5, -23, -17], [10, 8, 46]] {
                    let filter = super::super::wiener_coefficients(&coeff);
                    let src: Vec<i32> = (0..(width + 6) * 7).map(|i| (i as i32 * 197 + 71) & max).collect();
                    let mut expected = vec![0; width * 7];
                    let mut actual = expected.clone();
                    for y in 0..7 {
                        let input = &src[y * (width + 6)..];
                        super::super::wiener_horizontal_scalar(input, &filter, shift, -65536, 65535, &mut expected[y * width..(y + 1) * width]);
                        horizontal(input, &filter, shift, -65536, 65535, &mut actual[y * width..(y + 1) * width]);
                    }
                    assert_eq!(actual, expected);
                    let rows = core::array::from_fn(|k| &actual[k * width..(k + 1) * width]);
                    let mut a = vec![0u16; width];
                    let mut b = a.clone();
                    super::super::wiener_vertical_scalar(rows, &filter, 14 - shift, max, &mut a);
                    vertical(rows, &filter, 14 - shift, max, &mut b);
                    assert_eq!(a, b);
                }
            }
        }
    }
}
