//! §7.14.6 sample filtering of the four adjacent columns of one horizontal
//! edge in NEON lanes, with the masks, clamps and rounding of the scalar
//! process.
//!
//! [`edge`] is safe and reports whether it ran: it checks for NEON at run
//! time (a constant on the AArch64 targets, where NEON is part of the
//! baseline) and returns `false`, samples untouched, when the lanes or the
//! four columns are unavailable — the caller's scalar path is the fallback.
use super::{PlaneBuffer, Sample};
use core::arch::aarch64::*;

type V = int32x4_t;

#[inline]
#[target_feature(enable = "neon")]
fn splat(v: i32) -> V {
    vdupq_n_s32(v)
}
#[inline]
#[target_feature(enable = "neon")]
fn add(a: V, b: V) -> V {
    vaddq_s32(a, b)
}
#[inline]
#[target_feature(enable = "neon")]
fn sub(a: V, b: V) -> V {
    vsubq_s32(a, b)
}
#[inline]
#[target_feature(enable = "neon")]
fn mul(a: V, b: i32) -> V {
    vmulq_n_s32(a, b)
}
#[inline]
#[target_feature(enable = "neon")]
fn shr(a: V, b: u32) -> V {
    vshlq_s32(a, vdupq_n_s32(-(b as i32)))
}
#[inline]
#[target_feature(enable = "neon")]
fn abs(a: V) -> V {
    vabsq_s32(a)
}
#[inline]
#[target_feature(enable = "neon")]
fn min(a: V, b: V) -> V {
    vminq_s32(a, b)
}
#[inline]
#[target_feature(enable = "neon")]
fn max(a: V, b: V) -> V {
    vmaxq_s32(a, b)
}
#[inline]
#[target_feature(enable = "neon")]
fn gt(a: V, b: V) -> V {
    vreinterpretq_s32_u32(vcgtq_s32(a, b))
}
#[inline]
#[target_feature(enable = "neon")]
fn and(a: V, b: V) -> V {
    vandq_s32(a, b)
}
#[inline]
#[target_feature(enable = "neon")]
fn or(a: V, b: V) -> V {
    vorrq_s32(a, b)
}
#[inline]
#[target_feature(enable = "neon")]
fn not(a: V) -> V {
    vmvnq_s32(a)
}
#[inline]
#[target_feature(enable = "neon")]
fn select(m: V, a: V, b: V) -> V {
    vbslq_s32(vreinterpretq_u32_s32(m), a, b)
}
#[inline]
#[target_feature(enable = "neon")]
fn any(a: V) -> bool {
    vmaxvq_u32(vreinterpretq_u32_s32(a)) != 0
}

/// Four samples at `p`, widened to `i32` lanes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load<P: Sample>(p: *const P) -> V {
    // SAFETY: the caller proves four readable samples; `Sample` storage is
    // u8, u16 or i32.
    unsafe {
        if core::mem::size_of::<P>() == 1 {
            let bytes = vcreate_u8(u64::from(core::ptr::read_unaligned(p.cast::<u32>())));
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(bytes))))
        } else if core::mem::size_of::<P>() == 2 {
            vreinterpretq_s32_u32(vmovl_u16(vld1_u16(p.cast())))
        } else {
            vld1q_s32(p.cast())
        }
    }
}

/// Four in-range lanes narrowed into the samples at `p`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn store<P: Sample>(p: *mut P, v: V) {
    // SAFETY: the caller proves four writable samples; the lanes are
    // post-clamp sample values, so narrowing is exact.
    unsafe {
        if core::mem::size_of::<P>() == 1 {
            let h = vmovn_s32(v);
            let b = vmovn_u16(vreinterpretq_u16_s16(vcombine_s16(h, h)));
            core::ptr::write_unaligned(p.cast::<u32>(), vget_lane_u32::<0>(vreinterpret_u32_u8(b)));
        } else if core::mem::size_of::<P>() == 2 {
            vst1_u16(p.cast(), vreinterpret_u16_s16(vmovn_s32(v)));
        } else {
            vst1q_s32(p.cast(), v);
        }
    }
}

/// The four columns `x .. x + 4` of the horizontal edge above row `y` of
/// `buf` (rows clamped to the buffer, as [`PlaneBuffer::get`] does).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn edge<P: Sample>(
    buf: &mut PlaneBuffer<'_, P>,
    x: i32,
    y: i32,
    plane: u8,
    limit: u8,
    blimit: u8,
    thresh: u8,
    size: u32,
    depth: u8,
) -> bool {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        return false;
    }
    if x < 0
        || x as usize + 4 > buf.cols as usize
        || buf.rows == 0
        || buf.samples.len() < buf.rows as usize * buf.cols as usize
    {
        return false;
    }
    // SAFETY: NEON was detected above, and the four columns and the whole
    // `rows * cols` backing buffer were just checked.
    unsafe { edge_neon(buf, x, y, plane, limit, blimit, thresh, size, depth) };
    true
}

/// # Safety
///
/// Columns `x .. x + 4` must be inside `buf` and `buf.samples` must hold
/// `rows * cols` samples ([`edge`] checks both).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn edge_neon<P: Sample>(
    buf: &mut PlaneBuffer<'_, P>,
    x: i32,
    y: i32,
    plane: u8,
    limit: u8,
    blimit: u8,
    thresh: u8,
    size: u32,
    depth: u8,
) {
    let stride = buf.cols as usize;
    // SAFETY: each row is clamped exactly like `PlaneBuffer::get`; the caller
    // checked the four adjacent columns and the full backing buffer.
    unsafe {
        // s[ 7 + i ] = the row `i` below the edge (s[ 6 ] = p0, s[ 7 ] = q0);
        // only the rows the filter size reaches are loaded and stored.
        let reach: usize = match size {
            16 => 7,
            8 => 4,
            _ => 2,
        };
        let mut s = [splat(0); 14];
        for (k, v) in s.iter_mut().enumerate().take(7 + reach).skip(7 - reach) {
            let row = (y + k as i32 - 7).clamp(0, buf.rows as i32 - 1) as usize;
            *v = load(buf.samples.as_ptr().add(row * stride + x as usize));
        }
        let p0 = s[6];
        let p1 = s[5];
        let q0 = s[7];
        let q1 = s[8];
        let shift = depth - 8;
        let lim = splat((limit as i32) << shift);
        let flat_lim = splat(1 << shift);
        let hev = or(
            gt(abs(sub(p1, p0)), splat((thresh as i32) << shift)),
            gt(abs(sub(q1, q0)), splat((thresh as i32) << shift)),
        );
        let mut reject = or(gt(abs(sub(p1, p0)), lim), gt(abs(sub(q1, q0)), lim));
        reject = or(
            reject,
            gt(
                add(mul(abs(sub(p0, q0)), 2), shr(abs(sub(p1, q1)), 1)),
                splat((blimit as i32) << shift),
            ),
        );
        let len = if size == 4 {
            4
        } else if plane != 0 {
            6
        } else if size == 8 {
            8
        } else {
            16
        };
        for k in 2..if len >= 8 {
            4
        } else if len >= 6 {
            3
        } else {
            2
        } {
            reject = or(
                reject,
                or(
                    gt(abs(sub(s[6 - k], s[7 - k])), lim),
                    gt(abs(sub(s[7 + k], s[6 + k])), lim),
                ),
            );
        }
        let enabled = not(reject);
        if !any(enabled) {
            return;
        }
        let mut flat = splat(0);
        if size >= 8 {
            let mut bad = splat(0);
            for k in 1..if len >= 8 { 4 } else { 3 } {
                bad = or(
                    bad,
                    or(
                        gt(abs(sub(s[6 - k], p0)), flat_lim),
                        gt(abs(sub(s[7 + k], q0)), flat_lim),
                    ),
                );
            }
            flat = and(enabled, not(bad));
        }
        let mut flat2 = splat(0);
        if size >= 16 {
            let mut bad = splat(0);
            for k in 4..7 {
                bad = or(
                    bad,
                    or(
                        gt(abs(sub(s[6 - k], p0)), flat_lim),
                        gt(abs(sub(s[7 + k], q0)), flat_lim),
                    ),
                );
            }
            flat2 = and(flat, not(bad));
        }
        let mut out = s;
        let half = 128 << shift;
        let clamp = |v| min(splat(half - 1), max(splat(-half), v));
        let ps0 = sub(p0, splat(half));
        let ps1 = sub(p1, splat(half));
        let qs0 = sub(q0, splat(half));
        let qs1 = sub(q1, splat(half));
        let f = clamp(add(
            select(hev, clamp(sub(ps1, qs1)), splat(0)),
            mul(sub(qs0, ps0), 3),
        ));
        let f1 = shr(clamp(add(f, splat(4))), 3);
        let f2 = shr(clamp(add(f, splat(3))), 3);
        let narrow = and(enabled, not(flat));
        out[6] = select(narrow, add(clamp(add(ps0, f2)), splat(half)), out[6]);
        out[7] = select(narrow, add(clamp(sub(qs0, f1)), splat(half)), out[7]);
        let outer = and(narrow, not(hev));
        let f = shr(add(f1, splat(1)), 1);
        out[5] = select(outer, add(clamp(add(ps1, f)), splat(half)), out[5]);
        out[8] = select(outer, add(clamp(sub(qs1, f)), splat(half)), out[8]);
        for (n, log2, mask) in [
            (if plane == 0 { 3 } else { 2 }, 3, and(flat, not(flat2))),
            (6, 4, flat2),
        ] {
            if !any(mask) {
                continue;
            }
            let n2 = if log2 == 3 && plane == 0 { 0 } else { 1 };
            for i in -n..n {
                let mut sum = splat(1 << (log2 - 1));
                for j in i - n..=i + n {
                    sum = add(sum, s[(j.clamp(-n - 1, n) + 7) as usize]);
                }
                for j in i - n2..=i + n2 {
                    sum = add(sum, s[(j + 7) as usize]);
                }
                let index = (i + 7) as usize;
                out[index] = select(mask, shr(sum, log2), out[index]);
            }
        }
        for (k, value) in out
            .iter()
            .enumerate()
            .take((7 + reach).min(13))
            .skip((7 - reach).max(1))
        {
            let row = y + k as i32 - 7;
            if row >= 0 && row < buf.rows as i32 {
                store(
                    buf.samples
                        .as_mut_ptr()
                        .add(row as usize * stride + x as usize),
                    *value,
                );
            }
        }
    }
}
