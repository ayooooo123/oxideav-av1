//! §7.11.3 prediction rows in explicit NEON lanes: the horizontal pass of
//! the §7.11.3.4 convolution, and the §7.11.3.1 `Clip1` of one prediction
//! and of the rounded average of two.
//!
//! Every kernel computes the integers the specification's process does:
//! the convolution accumulates modulo 2^32 from widening 16-bit products
//! (sums of at most eight products of a 16-bit sample and a 16-bit tap
//! magnitude wrap exactly as the portable `i32` sums do, and no
//! `Subpel_Filters` row can make either wrap), and the average halves
//! before it rounds, so no input reaches a lane limit.
//!
//! The entry points are safe and report whether they ran: they check for
//! NEON at run time (a constant on the AArch64 targets, where NEON is part
//! of the baseline) and leave their outputs untouched when it is missing
//! or the row is not one they handle, so the caller's portable code is
//! the fallback.
use core::arch::aarch64::*;

#[inline(always)]
fn available() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}

/// One §7.11.3.4 tap as the widening multiply takes it: magnitude, and
/// whether its products are subtracted.
type Tap = (u16, bool);

/// One output row of the §7.11.3.4 horizontal pass, `out[ c ] = Round2(
/// sum_t k[ t ] * src[ c + t ], shift )` for `c in 0..out.len()`.
///
/// Runs for rows of a whole number of eight-sample groups (every block
/// width from 8 up) whose taps fit 16 bits in magnitude (every
/// `Subpel_Filters` tap does).
#[inline]
pub(super) fn convolve_row(k: &[i32; 8], src: &[u16], out: &mut [i32], shift: u32) -> bool {
    let w = out.len();
    if !available() || w == 0 || !w.is_multiple_of(8) || !(1..32).contains(&shift) {
        return false;
    }
    let mut taps: [Tap; 8] = [(0, false); 8];
    for (tap, &c) in taps.iter_mut().zip(k) {
        let Ok(magnitude) = u16::try_from(c.unsigned_abs()) else {
            return false;
        };
        *tap = (magnitude, c < 0);
    }
    assert!(src.len() >= w + 7);
    // SAFETY: NEON was detected above, `out` is a whole number of
    // eight-sample groups and `src` holds the `w + 7` samples their taps read.
    unsafe { convolve_row_neon(&taps, src, out, shift) };
    true
}

/// `acc + tap * samples` in the low (`HIGH == false`) or high four lanes
/// of `samples`, modulo 2^32.
#[inline]
#[target_feature(enable = "neon")]
fn accumulate<const HIGH: bool>(acc: uint32x4_t, samples: uint16x8_t, tap: Tap) -> uint32x4_t {
    match (HIGH, tap.1) {
        (false, false) => vmlal_n_u16(acc, vget_low_u16(samples), tap.0),
        (false, true) => vmlsl_n_u16(acc, vget_low_u16(samples), tap.0),
        (true, false) => vmlal_high_n_u16(acc, samples, tap.0),
        (true, true) => vmlsl_high_n_u16(acc, samples, tap.0),
    }
}

/// # Safety
///
/// `out.len()` must be a multiple of eight and `src.len() >= out.len() + 7`
/// ([`convolve_row`] checks both).
#[target_feature(enable = "neon")]
unsafe fn convolve_row_neon(taps: &[Tap; 8], src: &[u16], out: &mut [i32], shift: u32) {
    let w = out.len();
    let (src, dst) = (src.as_ptr(), out.as_mut_ptr());
    let bias = vdupq_n_u32(1 << (shift - 1));
    let shifts = vdupq_n_s32(-(shift as i32));
    let finish = |acc: uint32x4_t| vshlq_s32(vreinterpretq_s32_u32(acc), shifts);
    let mut x = 0;
    while x + 16 <= w {
        let mut acc = [bias; 4];
        for (t, &tap) in taps.iter().enumerate() {
            // The six- and four-tap filter sets carry zero outer taps.
            if tap.0 == 0 {
                continue;
            }
            // SAFETY: the sixteen samples at `x + t` end at most at
            // `x + 7 + 16 <= w + 7 <= src.len()`.
            let (a, b) = unsafe { (vld1q_u16(src.add(x + t)), vld1q_u16(src.add(x + t + 8))) };
            acc[0] = accumulate::<false>(acc[0], a, tap);
            acc[1] = accumulate::<true>(acc[1], a, tap);
            acc[2] = accumulate::<false>(acc[2], b, tap);
            acc[3] = accumulate::<true>(acc[3], b, tap);
        }
        for (i, lanes) in acc.into_iter().enumerate() {
            // SAFETY: `x + 16 <= w` outputs are writable.
            unsafe { vst1q_s32(dst.add(x + 4 * i), finish(lanes)) };
        }
        x += 16;
    }
    if x + 8 <= w {
        let mut acc = [bias; 2];
        for (t, &tap) in taps.iter().enumerate() {
            if tap.0 == 0 {
                continue;
            }
            // SAFETY: the eight samples at `x + t` end at most at
            // `x + 7 + 8 <= w + 7 <= src.len()`.
            let a = unsafe { vld1q_u16(src.add(x + t)) };
            acc[0] = accumulate::<false>(acc[0], a, tap);
            acc[1] = accumulate::<true>(acc[1], a, tap);
        }
        for (i, lanes) in acc.into_iter().enumerate() {
            // SAFETY: `x + 8 <= w` outputs are writable.
            unsafe { vst1q_s32(dst.add(x + 4 * i), finish(lanes)) };
        }
    }
}

/// §7.11.3.1 `Clip1` of a single prediction: `out[ i ] = Clip3( 0, max,
/// pred[ i ] )` for every `i` of `pred`, for a `max` a sample can hold.
#[inline]
pub(super) fn clip(max: i32, pred: &[i32], out: &mut [u16]) -> bool {
    let Ok(max) = u16::try_from(max) else {
        return false;
    };
    if !available() || out.len() < pred.len() {
        return false;
    }
    // SAFETY: NEON was detected above.
    unsafe { clip_neon(max, pred, &mut out[..pred.len()]) };
    true
}

/// `Clip3( 0, max, v )` of eight 32-bit lanes: the narrowing saturates to
/// `0 ..= 65535`, the minimum brings it under `max`.
#[inline]
#[target_feature(enable = "neon")]
fn clip8(lo: int32x4_t, hi: int32x4_t, max: uint16x8_t) -> uint16x8_t {
    vminq_u16(vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi)), max)
}

/// # Safety
///
/// Needs NEON only; `out` is as long as `pred` ([`clip`] slices it).
#[target_feature(enable = "neon")]
unsafe fn clip_neon(max: u16, pred: &[i32], out: &mut [u16]) {
    let high = vdupq_n_u16(max);
    let mut groups = pred.chunks_exact(8);
    let mut outputs = out.chunks_exact_mut(8);
    for (p, o) in groups.by_ref().zip(outputs.by_ref()) {
        // SAFETY: each chunk holds exactly eight elements.
        unsafe {
            let p = p.as_ptr();
            vst1q_u16(
                o.as_mut_ptr(),
                clip8(vld1q_s32(p), vld1q_s32(p.add(4)), high),
            );
        }
    }
    for (o, &p) in outputs.into_remainder().iter_mut().zip(groups.remainder()) {
        *o = p.clamp(0, i32::from(max)) as u16;
    }
}

/// §7.11.3.1 average of two predictions: `out[ i ] = Clip3( 0, max,
/// Round2( a[ i ] + b[ i ], shift ) )` for every `i` of `a`, for a `max`
/// a sample can hold.
#[inline]
pub(super) fn average(max: i32, shift: u32, a: &[i32], b: &[i32], out: &mut [u16]) -> bool {
    let n = a.len();
    let Ok(max) = u16::try_from(max) else {
        return false;
    };
    if !available() || b.len() < n || out.len() < n || !(1..32).contains(&shift) {
        return false;
    }
    // SAFETY: NEON was detected above.
    unsafe { average_neon(max, shift, a, &b[..n], &mut out[..n]) };
    true
}

/// # Safety
///
/// Needs NEON only; the three slices are equally long and `1 <= shift <
/// 32` ([`average`] checks both).
#[target_feature(enable = "neon")]
unsafe fn average_neon(max: u16, shift: u32, a: &[i32], b: &[i32], out: &mut [u16]) {
    let high = vdupq_n_u16(max);
    // `Round2( a + b, shift )` without forming the 33-bit sum: the halving
    // add is `floor( ( a + b ) / 2 )` (rounded up for `shift == 1`, which
    // is the whole `Round2`), and rounding that by the remaining
    // `shift - 1` bits equals rounding the sum by `shift`.
    let rest = vdupq_n_s32(1 - shift as i32);
    let round = |a: int32x4_t, b: int32x4_t| {
        if shift == 1 {
            vrhaddq_s32(a, b)
        } else {
            vrshlq_s32(vhaddq_s32(a, b), rest)
        }
    };
    let mut lefts = a.chunks_exact(8);
    let mut rights = b.chunks_exact(8);
    let mut outputs = out.chunks_exact_mut(8);
    for ((a, b), o) in lefts.by_ref().zip(rights.by_ref()).zip(outputs.by_ref()) {
        // SAFETY: each chunk holds exactly eight elements.
        unsafe {
            let (a, b) = (a.as_ptr(), b.as_ptr());
            let lo = round(vld1q_s32(a), vld1q_s32(b));
            let hi = round(vld1q_s32(a.add(4)), vld1q_s32(b.add(4)));
            vst1q_u16(o.as_mut_ptr(), clip8(lo, hi, high));
        }
    }
    let tail = lefts.remainder().iter().zip(rights.remainder());
    for (o, (&a, &b)) in outputs.into_remainder().iter_mut().zip(tail) {
        let sum = i64::from(a) + i64::from(b);
        *o = ((sum + (1 << (shift - 1))) >> shift).clamp(0, i64::from(max)) as u16;
    }
}
