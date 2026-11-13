//! Four adjacent AV1 edge lanes with exact masks, clamps, and rounding.
use super::{Pixel, PlaneBuffer};
use core::arch::wasm32::*;
type V = v128;
#[inline(always)]
fn splat(v:i32) -> V { i32x4_splat(v) }
#[inline(always)]
fn add(a:V,b:V) -> V { i32x4_add(a,b) }
#[inline(always)]
fn sub(a:V,b:V) -> V { i32x4_sub(a,b) }
#[inline(always)]
fn mul(a:V,b:i32) -> V { i32x4_mul(a,i32x4_splat(b)) }
#[inline(always)]
fn shr(a:V,b:u32) -> V { i32x4_shr(a,b) }
#[inline(always)]
fn abs(a:V) -> V { i32x4_abs(a) }
#[inline(always)]
fn min(a:V,b:V) -> V { i32x4_min(a,b) }
#[inline(always)]
fn max(a:V,b:V) -> V { i32x4_max(a,b) }
#[inline(always)]
fn gt(a:V,b:V) -> V { i32x4_gt(a,b) }
#[inline(always)]
fn and(a:V,b:V) -> V { v128_and(a,b) }
#[inline(always)]
fn or(a:V,b:V) -> V { v128_or(a,b) }
#[inline(always)]
fn not(a:V) -> V { v128_not(a) }
#[inline(always)]
fn select(m:V,a:V,b:V) -> V { v128_bitselect(a,b,m) }
#[inline(always)]
fn any(a:V) -> bool { v128_any_true(a) }
#[inline(always)]
unsafe fn load<P: Pixel>(p: *const P) -> V {
    unsafe {
        if core::mem::size_of::<P>() == 1 { i32x4_extend_low_u16x8(i16x8_extend_low_u8x16(v128_load32_zero(p.cast()))) }
        else if core::mem::size_of::<P>() == 2 { i32x4_extend_low_u16x8(v128_load64_zero(p.cast())) }
        else { v128_load(p.cast()) }
    }
}
#[inline(always)]
unsafe fn store<P: Pixel>(p: *mut P, v: V) {
    unsafe {
        let h = i16x8_narrow_i32x4(v,v);
        if core::mem::size_of::<P>() == 1 { v128_store32_lane::<0>(u8x16_narrow_i16x8(h,h),p.cast()); }
        else if core::mem::size_of::<P>() == 2 { v128_store64_lane::<0>(h,p.cast()); }
        else { v128_store(p.cast(),v); }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn edge<P: Pixel>(buf: &mut PlaneBuffer<'_, P>, x: i32, y: i32, plane: u8, limit: u8, blimit: u8, thresh: u8, size: u32, depth: u8) -> bool {
    if x < 0 || x as usize + 4 > buf.cols as usize || buf.rows == 0 || buf.samples.len() < buf.rows as usize * buf.cols as usize { return false; }
    let stride = buf.cols as usize;
    // SAFETY: each row is clamped exactly like PlaneBuffer::get; four adjacent
    // columns and the full backing buffer were checked before pointer access.
    unsafe {
        let mut s = [splat(0); 14];
        for (k, v) in s.iter_mut().enumerate() {
            let row = (y + k as i32 - 7).clamp(0, buf.rows as i32 - 1) as usize;
            *v = load(buf.samples.as_ptr().add(row * stride + x as usize));
        }
        let p0 = s[6]; let p1 = s[5]; let q0 = s[7]; let q1 = s[8];
        let shift = depth - 8;
        let lim = splat((limit as i32) << shift);
        let flat_lim = splat(1 << shift);
        let hev = or(gt(abs(sub(p1, p0)), splat((thresh as i32) << shift)), gt(abs(sub(q1, q0)), splat((thresh as i32) << shift)));
        let mut reject = or(gt(abs(sub(p1, p0)), lim), gt(abs(sub(q1, q0)), lim));
        reject = or(reject, gt(add(mul(abs(sub(p0, q0)), 2), shr(abs(sub(p1, q1)), 1)), splat((blimit as i32) << shift)));
        let len = if size == 4 { 4 } else if plane != 0 { 6 } else if size == 8 { 8 } else { 16 };
        for k in 2..if len >= 8 { 4 } else if len >= 6 { 3 } else { 2 } {
            reject = or(reject, or(gt(abs(sub(s[6-k], s[7-k])), lim), gt(abs(sub(s[7+k], s[6+k])), lim)));
        }
        let enabled = not(reject);
        if !any(enabled) { return true; }
        let mut flat = splat(0);
        if size >= 8 {
            let mut bad = splat(0);
            for k in 1..if len >= 8 { 4 } else { 3 } {
                bad = or(bad, or(gt(abs(sub(s[6-k], p0)), flat_lim), gt(abs(sub(s[7+k], q0)), flat_lim)));
            }
            flat = and(enabled, not(bad));
        }
        let mut flat2 = splat(0);
        if size >= 16 {
            let mut bad = splat(0);
            for k in 4..7 {
                bad = or(bad, or(gt(abs(sub(s[6-k], p0)), flat_lim), gt(abs(sub(s[7+k], q0)), flat_lim)));
            }
            flat2 = and(flat, not(bad));
        }
        let mut out = s;
        let half = 128 << shift;
        let clamp = |v| min(splat(half - 1), max(splat(-half), v));
        let ps0 = sub(p0, splat(half)); let ps1 = sub(p1, splat(half));
        let qs0 = sub(q0, splat(half)); let qs1 = sub(q1, splat(half));
        let f = clamp(add(select(hev, clamp(sub(ps1, qs1)), splat(0)), mul(sub(qs0, ps0), 3)));
        let f1 = shr(clamp(add(f, splat(4))), 3);
        let f2 = shr(clamp(add(f, splat(3))), 3);
        let narrow = and(enabled, not(flat));
        out[6] = select(narrow, add(clamp(add(ps0, f2)), splat(half)), out[6]);
        out[7] = select(narrow, add(clamp(sub(qs0, f1)), splat(half)), out[7]);
        let outer = and(narrow, not(hev));
        let f = shr(add(f1, splat(1)), 1);
        out[5] = select(outer, add(clamp(add(ps1, f)), splat(half)), out[5]);
        out[8] = select(outer, add(clamp(sub(qs1, f)), splat(half)), out[8]);
        for (n, log2, mask) in [(if plane == 0 { 3 } else { 2 }, 3, and(flat, not(flat2))), (6, 4, flat2)] {
            if !any(mask) { continue; }
            let n2 = if log2 == 3 && plane == 0 { 0 } else { 1 };
            for i in -n..n {
                let mut sum = splat(1 << (log2 - 1));
                for j in i-n..=i+n { sum = add(sum, s[(j.clamp(-n-1, n) + 7) as usize]); }
                for j in i-n2..=i+n2 { sum = add(sum, s[(j + 7) as usize]); }
                let index = (i + 7) as usize;
                out[index] = select(mask, shr(sum, log2), out[index]);
            }
        }
        for (k, value) in out.iter().enumerate().take(13).skip(1) {
            let row = y + k as i32 - 7;
            if row >= 0 && row < buf.rows as i32 {
                store(buf.samples.as_mut_ptr().add(row as usize * stride + x as usize), *value);
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contiguous_edges_match_scalar_masks_and_depths() {
        let mut seed = 0x73a149bdu32;
        for depth in [8, 10, 12] {
            let max = (1 << depth) - 1;
            for case in 0..96 {
                let samples: Vec<u16> = (0..20 * 20).map(|i| {
                    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                    if case % 3 == 0 { ((max / 2) + (i as i32 % 3) - 1) as u16 }
                    else if case % 3 == 1 { (max / 2 + (i as i32 / 20 % 2) * (1 << (depth - 8))) as u16 }
                    else { (seed as i32 & max) as u16 }
                }).collect();
                for plane in [0, 1] { for size in [4, 8, 16] { for y in [1, 8, 19] {
                    let mut a = samples.clone(); let mut b = samples.clone();
                    let mut expected = PlaneBuffer { rows: 20, cols: 20, samples: &mut a };
                    let mut actual = PlaneBuffer { rows: 20, cols: 20, samples: &mut b };
                    let limit = 1 + (case % 63) as u8;
                    let blimit = 128 + (case % 64) as u8;
                    let thresh = (case % 4) as u8;
                    for x in 4..8 { super::super::sample_filtering(&mut expected, x, y, plane, limit, blimit, thresh, 0, 1, size, depth); }
                    assert!(edge(&mut actual, 4, y, plane, limit, blimit, thresh, size, depth));
                    assert_eq!(a, b, "depth={depth} case={case} plane={plane} size={size} y={y}");
                } } }
            }
        }
    }
}
