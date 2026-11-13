//! §7.15 arithmetic in eight signed 16-bit NEON lanes.
//! Samples and differences fit 13 bits; the weighted constrained sum is
//! bounded by 24 * 240 = 5760 at 12-bit depth. No saturating arithmetic
//! changes the specification's result.
//!
//! The entry points are safe and report whether they ran: they check for
//! NEON at run time (a constant on the AArch64 targets, where NEON is part
//! of the baseline) and leave their outputs untouched when it is missing,
//! so the caller's scalar path is the fallback.
use crate::loop_filter::Sample;
use core::arch::aarch64::*;

#[inline(always)]
fn available() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn load<const W: usize, P: Sample>(p: *const P) -> int16x8_t {
    // SAFETY: `Sample` storage is u8, u16 or i32 and the caller proves W
    // readable samples. Four-byte loads avoid overreading chroma rows.
    unsafe {
        if core::mem::size_of::<P>() == 1 {
            let bytes = if W == 8 {
                vld1_u8(p.cast())
            } else {
                vreinterpret_u8_u32(vld1_lane_u32::<0>(p.cast(), vdup_n_u32(0)))
            };
            vreinterpretq_s16_u16(vmovl_u8(bytes))
        } else if core::mem::size_of::<P>() == 2 {
            if W == 8 {
                vreinterpretq_s16_u16(vld1q_u16(p.cast()))
            } else {
                vcombine_s16(vreinterpret_s16_u16(vld1_u16(p.cast())), vdup_n_s16(0))
            }
        } else {
            let p = p.cast::<i32>();
            let lo = vmovn_s32(vld1q_s32(p));
            let hi = if W == 8 {
                vmovn_s32(vld1q_s32(p.add(4)))
            } else {
                vdup_n_s16(0)
            };
            vcombine_s16(lo, hi)
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn constrained(d: int16x8_t, strength: int16x8_t, shift: int16x8_t) -> int16x8_t {
    let a = vabsq_s16(d);
    let limit = vmaxq_s16(vsubq_s16(strength, vshlq_s16(a, shift)), vdupq_n_s16(0));
    let magnitude = vminq_s16(a, limit);
    vbslq_s16(
        vcltq_s16(d, vdupq_n_s16(0)),
        vnegq_s16(magnitude),
        magnitude,
    )
}

/// The §7.15.3 filter over a `W × height` block whose every tap lies
/// inside `src`: `base` is the block's first sample in `src`, `dst` starts
/// at the block's first output sample, `pri` / `sec` are the `(offset,
/// weight)` taps and the shifts the hoisted `dampingAdj` terms.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn filter<const W: usize, P: Sample>(
    src: &[P],
    base: usize,
    src_stride: usize,
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
    if !available() {
        return false;
    }
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
    // SAFETY: NEON was detected above, and the complete source tap rectangle
    // and destination rectangle were just checked, once for this block.
    unsafe {
        filter_neon::<W, P>(
            src,
            base,
            src_stride,
            dst,
            dst_stride,
            height,
            pri,
            sec,
            pri_strength,
            sec_strength,
            pri_shift,
            sec_shift,
        );
    }
    true
}

/// # Safety
///
/// Every tap row of the `W × height` block at `base` must lie inside `src`
/// and the block's output rectangle inside `dst` ([`filter`] checks both).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn filter_neon<const W: usize, P: Sample>(
    src: &[P],
    base: usize,
    src_stride: usize,
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
    // SAFETY: the caller checked the source tap rectangle and the
    // destination rectangle. Narrowing is exact for AV1 8/10/12-bit
    // samples; shifts >= 16 correctly yield zero for |diff|.
    unsafe {
        let ps = vdupq_n_s16(pri_strength as i16);
        let ss = vdupq_n_s16(sec_strength as i16);
        let pshift = vdupq_n_s16(-(pri_shift.min(16) as i16));
        let sshift = vdupq_n_s16(-(sec_shift.min(16) as i16));
        for y in 0..height {
            let row = src.as_ptr().add(base + y * src_stride);
            let center = load::<W, P>(row);
            let mut min = center;
            let mut max = center;
            let mut sum = vdupq_n_s16(0);
            if pri_strength != 0 {
                for &(off, tap) in pri {
                    let value = load::<W, P>(row.offset(off));
                    min = vminq_s16(min, value);
                    max = vmaxq_s16(max, value);
                    sum = vmlaq_n_s16(
                        sum,
                        constrained(vsubq_s16(value, center), ps, pshift),
                        tap as i16,
                    );
                }
            }
            if sec_strength != 0 {
                for &(off, tap) in sec {
                    let value = load::<W, P>(row.offset(off));
                    min = vminq_s16(min, value);
                    max = vmaxq_s16(max, value);
                    sum = vmlaq_n_s16(
                        sum,
                        constrained(vsubq_s16(value, center), ss, sshift),
                        tap as i16,
                    );
                }
            }
            let negative = vreinterpretq_s16_u16(vcltq_s16(sum, vdupq_n_s16(0)));
            let delta = vshrq_n_s16::<4>(vaddq_s16(vaddq_s16(sum, vdupq_n_s16(8)), negative));
            let out = vminq_s16(max, vmaxq_s16(min, vaddq_s16(center, delta)));
            let target = dst.as_mut_ptr().add(y * dst_stride);
            if core::mem::size_of::<P>() == 1 {
                let bytes = vmovn_u16(vreinterpretq_u16_s16(out));
                if W == 8 {
                    vst1_u8(target.cast(), bytes);
                } else {
                    vst1_lane_u32::<0>(target.cast(), vreinterpret_u32_u8(bytes));
                }
            } else if core::mem::size_of::<P>() == 2 {
                if W == 8 {
                    vst1q_s16(target.cast(), out);
                } else {
                    vst1_s16(target.cast(), vget_low_s16(out));
                }
            } else {
                vst1q_s32(target.cast(), vmovl_s16(vget_low_s16(out)));
                if W == 8 {
                    vst1q_s32(target.cast::<i32>().add(4), vmovl_s16(vget_high_s16(out)));
                }
            }
        }
    }
}

/// The §7.15.2 `partial[ 8 ][ 15 ]` projection sums of the 8×8 block at
/// `base` in `src`.
#[inline]
pub(super) fn partial<P: Sample>(
    src: &[P],
    base: usize,
    stride: usize,
    shift: i32,
) -> Option<[[i32; 15]; 8]> {
    if !available() {
        return None;
    }
    assert!(base + 7 * stride + 8 <= src.len());
    // SAFETY: NEON was detected above and the eight source rows were just checked.
    Some(unsafe { partial_neon(src, base, stride, shift) })
}

/// # Safety
///
/// `src` must hold the eight 8-sample rows at `base + i * stride`
/// ([`partial`] checks it).
#[target_feature(enable = "neon")]
unsafe fn partial_neon<P: Sample>(
    src: &[P],
    base: usize,
    stride: usize,
    shift: i32,
) -> [[i32; 15]; 8] {
    let mut partial = [[0i16; 16]; 8];
    // SAFETY: the caller checked the eight complete source rows. Every
    // projection offset lies in 0..15 (16 allocated lanes), so vector
    // loads/stores stay inside one projection row. Each sum is at most
    // eight normalized samples (-1024..1016), exactly representable in i16.
    unsafe {
        for i in 0..8 {
            let values = vsubq_s16(
                vshlq_s16(
                    load::<8, P>(src.as_ptr().add(base + i * stride)),
                    vdupq_n_s16(-(shift as i16)),
                ),
                vdupq_n_s16(128),
            );
            let reverse = vextq_s16::<4>(vrev64q_s16(values), vrev64q_s16(values));
            for (dir, offset, v) in [
                (0, i, values),
                (4, i, reverse),
                (5, 3 - i / 2, values),
                (6, 0, values),
                (7, i / 2, values),
            ] {
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
