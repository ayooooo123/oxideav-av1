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

/// Transpose 4×4 `u16` rows in NEON registers.
#[inline]
#[target_feature(enable = "neon")]
fn transpose4x4_u16(
    r0: uint16x4_t,
    r1: uint16x4_t,
    r2: uint16x4_t,
    r3: uint16x4_t,
) -> (uint16x4_t, uint16x4_t, uint16x4_t, uint16x4_t) {
    let t0 = vtrn1_u16(r0, r1);
    let t1 = vtrn2_u16(r0, r1);
    let t2 = vtrn1_u16(r2, r3);
    let t3 = vtrn2_u16(r2, r3);
    let c0 = vreinterpret_u16_u32(vtrn1_u32(vreinterpret_u32_u16(t0), vreinterpret_u32_u16(t2)));
    let c2 = vreinterpret_u16_u32(vtrn2_u32(vreinterpret_u32_u16(t0), vreinterpret_u32_u16(t2)));
    let c1 = vreinterpret_u16_u32(vtrn1_u32(vreinterpret_u32_u16(t1), vreinterpret_u32_u16(t3)));
    let c3 = vreinterpret_u16_u32(vtrn2_u32(vreinterpret_u32_u16(t1), vreinterpret_u32_u16(t3)));
    (c0, c1, c2, c3)
}

/// Direction-agnostic §7.14.6 filter core over four parallel lines in `s`.
/// Returns `true` if any lane was enabled (so `out` holds updated taps).
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn filter_core(
    s: &[V; 14],
    out: &mut [V; 14],
    plane: u8,
    limit: u8,
    blimit: u8,
    thresh: u8,
    size: u32,
    depth: u8,
) -> bool {
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
        return false;
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
    *out = *s;
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
    true
}

/// The four rows `y .. y + 4` of the vertical edge left of column `x` of `buf`
/// (every tap `x - reach .. x + reach` inside `buf`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn edge_vertical<P: Sample>(
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
    let reach: usize = match size {
        16 => 7,
        8 => 4,
        _ => 2,
    };
    if y < 0
        || y as usize + 4 > buf.rows as usize
        || x < reach as i32
        || x as usize + reach > buf.cols as usize
        || buf.samples.len() < buf.rows as usize * buf.cols as usize
    {
        return false;
    }
    // SAFETY: NEON was detected above, and the 4 rows × 2*reach columns sit
    // strictly inside `buf.samples`.
    unsafe {
        edge_vertical_neon(buf, x as usize, y as usize, reach, plane, limit, blimit, thresh, size, depth)
    };
    true
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn edge_vertical_neon<P: Sample>(
    buf: &mut PlaneBuffer<'_, P>,
    x: usize,
    y: usize,
    reach: usize,
    plane: u8,
    limit: u8,
    blimit: u8,
    thresh: u8,
    size: u32,
    depth: u8,
) {
    let stride = buf.cols as usize;
    let mut s = [splat(0); 14];
    // SAFETY: caller verified `y + 4 <= buf.rows`, `reach <= x`, `x + reach <= buf.cols`,
    // and `buf.samples.len() >= buf.rows * buf.cols`.
    unsafe {
        if core::mem::size_of::<P>() == 2 {
            let base = buf.samples.as_ptr().cast::<u16>();
            let load4x4 = |col0: usize, dst: &mut [V]| {
                let r0 = vld1_u16(base.add(y * stride + col0));
                let r1 = vld1_u16(base.add((y + 1) * stride + col0));
                let r2 = vld1_u16(base.add((y + 2) * stride + col0));
                let r3 = vld1_u16(base.add((y + 3) * stride + col0));
                let (c0, c1, c2, c3) = transpose4x4_u16(r0, r1, r2, r3);
                dst[0] = vreinterpretq_s32_u32(vmovl_u16(c0));
                dst[1] = vreinterpretq_s32_u32(vmovl_u16(c1));
                dst[2] = vreinterpretq_s32_u32(vmovl_u16(c2));
                dst[3] = vreinterpretq_s32_u32(vmovl_u16(c3));
            };
            match reach {
                2 => load4x4(x - 2, &mut s[5..9]),
                4 => {
                    load4x4(x - 4, &mut s[3..7]);
                    load4x4(x, &mut s[7..11]);
                }
                _ => {
                    load4x4(x - 7, &mut s[0..4]);
                    load4x4(x - 3, &mut s[4..8]);
                    load4x4(x + 1, &mut s[8..12]);
                    // Final 2 columns (x + 5, x + 6): gather directly into lanes.
                    for k in 0..2usize {
                        let c = x + 5 + k;
                        let vals = [
                            i32::from(*base.add(y * stride + c)),
                            i32::from(*base.add((y + 1) * stride + c)),
                            i32::from(*base.add((y + 2) * stride + c)),
                            i32::from(*base.add((y + 3) * stride + c)),
                        ];
                        s[12 + k] = vld1q_s32(vals.as_ptr());
                    }
                }
            }
        } else {
            let start = 7 - reach;
            let end = 7 + reach;
            for (k, slot) in s.iter_mut().enumerate().take(end).skip(start) {
                let c = x + k - 7;
                let vals = [
                    buf.samples[y * stride + c].widen(),
                    buf.samples[(y + 1) * stride + c].widen(),
                    buf.samples[(y + 2) * stride + c].widen(),
                    buf.samples[(y + 3) * stride + c].widen(),
                ];
                *slot = vld1q_s32(vals.as_ptr());
            }
        }
        let mut out = [splat(0); 14];
        if !filter_core(&s, &mut out, plane, limit, blimit, thresh, size, depth) {
            return;
        }
        if core::mem::size_of::<P>() == 2 {
            let base = buf.samples.as_mut_ptr().cast::<u16>();
            let store4x4 = |col0: usize, src: &[V]| {
                let c0 = vreinterpret_u16_s16(vmovn_s32(src[0]));
                let c1 = vreinterpret_u16_s16(vmovn_s32(src[1]));
                let c2 = vreinterpret_u16_s16(vmovn_s32(src[2]));
                let c3 = vreinterpret_u16_s16(vmovn_s32(src[3]));
                let (r0, r1, r2, r3) = transpose4x4_u16(c0, c1, c2, c3);
                vst1_u16(base.add(y * stride + col0), r0);
                vst1_u16(base.add((y + 1) * stride + col0), r1);
                vst1_u16(base.add((y + 2) * stride + col0), r2);
                vst1_u16(base.add((y + 3) * stride + col0), r3);
            };
            match reach {
                2 => store4x4(x - 2, &out[5..9]),
                4 => {
                    store4x4(x - 4, &out[3..7]);
                    store4x4(x, &out[7..11]);
                }
                _ => {
                    store4x4(x - 6, &out[1..5]);
                    store4x4(x - 2, &out[5..9]);
                    store4x4(x + 2, &out[9..13]);
                }
            }
        } else {
            let lo = (7 - reach).max(1);
            let hi = (7 + reach).min(13);
            for (k, value) in out.iter().enumerate().take(hi).skip(lo) {
                let mut lane = [0i32; 4];
                vst1q_s32(lane.as_mut_ptr(), *value);
                let c = x + k - 7;
                for (r, &v) in lane.iter().enumerate() {
                    buf.samples[(y + r) * stride + c] = P::narrow(v);
                }
            }
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
        let mut out = [splat(0); 14];
        if !filter_core(&s, &mut out, plane, limit, blimit, thresh, size, depth) {
            return;
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
