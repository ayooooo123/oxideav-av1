//! §7.14.6 sample filtering of the four adjacent columns of one horizontal
//! edge in SIMD128 lanes, with the masks, clamps and rounding of the scalar
//! process — the WebAssembly twin of `loop_filter/neon.rs`.
//!
//! WebAssembly has no run-time feature detection: this module is compiled
//! only when the target enables `simd128`. [`edge`] returns `false`,
//! samples untouched, only when the four columns are not inside the buffer.
use super::{PlaneBuffer, Sample};
use core::arch::wasm32::*;
type V = v128;
#[inline(always)]
fn splat(v: i32) -> V {
    i32x4_splat(v)
}
#[inline(always)]
fn add(a: V, b: V) -> V {
    i32x4_add(a, b)
}
#[inline(always)]
fn sub(a: V, b: V) -> V {
    i32x4_sub(a, b)
}
#[inline(always)]
fn mul(a: V, b: i32) -> V {
    i32x4_mul(a, i32x4_splat(b))
}
#[inline(always)]
fn shr(a: V, b: u32) -> V {
    i32x4_shr(a, b)
}
#[inline(always)]
fn abs(a: V) -> V {
    i32x4_abs(a)
}
#[inline(always)]
fn min(a: V, b: V) -> V {
    i32x4_min(a, b)
}
#[inline(always)]
fn max(a: V, b: V) -> V {
    i32x4_max(a, b)
}
#[inline(always)]
fn gt(a: V, b: V) -> V {
    i32x4_gt(a, b)
}
#[inline(always)]
fn and(a: V, b: V) -> V {
    v128_and(a, b)
}
#[inline(always)]
fn or(a: V, b: V) -> V {
    v128_or(a, b)
}
#[inline(always)]
fn not(a: V) -> V {
    v128_not(a)
}
#[inline(always)]
fn select(m: V, a: V, b: V) -> V {
    v128_bitselect(a, b, m)
}
#[inline(always)]
fn any(a: V) -> bool {
    v128_any_true(a)
}
#[inline(always)]
unsafe fn load<P: Sample>(p: *const P) -> V {
    // SAFETY: the caller proves four readable samples; `Sample` storage is
    // u8, u16 or i32.
    unsafe {
        if core::mem::size_of::<P>() == 1 {
            i32x4_extend_low_u16x8(i16x8_extend_low_u8x16(v128_load32_zero(p.cast())))
        } else if core::mem::size_of::<P>() == 2 {
            i32x4_extend_low_u16x8(v128_load64_zero(p.cast()))
        } else {
            v128_load(p.cast())
        }
    }
}
#[inline(always)]
unsafe fn store<P: Sample>(p: *mut P, v: V) {
    // SAFETY: the caller proves four writable samples; the lanes are
    // post-clamp sample values, so narrowing is exact.
    unsafe {
        let h = i16x8_narrow_i32x4(v, v);
        if core::mem::size_of::<P>() == 1 {
            v128_store32_lane::<0>(u8x16_narrow_i16x8(h, h), p.cast());
        } else if core::mem::size_of::<P>() == 2 {
            v128_store64_lane::<0>(h, p.cast());
        } else {
            v128_store(p.cast(), v);
        }
    }
}

#[allow(clippy::too_many_arguments)]
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
    if x < 0
        || x as usize + 4 > buf.cols as usize
        || buf.rows == 0
        || buf.samples.len() < buf.rows as usize * buf.cols as usize
    {
        return false;
    }
    let stride = buf.cols as usize;
    // SAFETY: each row is clamped exactly like PlaneBuffer::get; four adjacent
    // columns and the full backing buffer were checked before pointer access.
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
            return true;
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
    true
}
