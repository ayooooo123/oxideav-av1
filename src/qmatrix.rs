//! §9.5.3 Quantizer matrix tables (av1-spec §9.5.3, p.510-554).
//!
//! Stores the normative `Quantizer_Matrix[ 15 ][ 2 ][ QM_TOTAL_SIZE ]` packed
//! with lossless cross-level delta + LZSS (`39,438` B instead of `100,320` B)
//! and expands it once on first QM use via `LazyLock<Box<...>>` so `.rodata`
//! sheds 61 KiB without losing a single bit.

use std::sync::LazyLock;

use crate::cdf::{TX_HEIGHT, TX_SIZES_ALL, TX_WIDTH};

/// `QM_TOTAL_SIZE` per spec §3 p.12 — number of values per plane per
/// quantizer level inside `Quantizer_Matrix[ ][ ][ ]`.
pub const QM_TOTAL_SIZE: usize = 3344;

/// Number of quantizer-matrix levels carried in `Quantizer_Matrix`.
pub const QM_LEVELS: usize = 15;

/// Number of plane classes carried in `Quantizer_Matrix`.
pub const QM_PLANES: usize = 2;

/// `Qm_Offset[ TX_SIZES_ALL ]` per §9.5.3 p.510.
pub const QM_OFFSET: [usize; TX_SIZES_ALL] = [
    0,    // TX_4X4   (4*4   = 16)
    16,   // TX_8X8   (8*8   = 64)
    80,   // TX_16X16 (16*16 = 256)
    336,  // TX_32X32 (32*32 = 1024)
    336,  // TX_64X64 — shares TX_32X32 (tw=th=32)
    1360, // TX_4X8   (4*8   = 32)
    1392, // TX_8X4   (8*4   = 32)
    1424, // TX_8X16  (8*16  = 128)
    1552, // TX_16X8  (16*8  = 128)
    1680, // TX_16X32 (16*32 = 512)
    2192, // TX_32X16 (32*16 = 512)
    336,  // TX_32X64 — shares TX_32X32
    336,  // TX_64X32 — shares TX_32X32
    2704, // TX_4X16  (4*16  = 64)
    2768, // TX_16X4  (16*4  = 64)
    2832, // TX_8X32  (8*32  = 256)
    3088, // TX_32X8  (32*8  = 256)
    1680, // TX_16X64 — shares TX_16X32 (tw=16, th=32)
    2192, // TX_64X16 — shares TX_32X16 (tw=32, th=16)
];

const PACKED_QM: &[u8] = include_bytes!("qmatrix_packed.bin");
const TOTAL_QM_BYTES: usize = QM_LEVELS * QM_PLANES * QM_TOTAL_SIZE;

fn unpack_quantizer_matrix() -> Box<[[[u8; QM_TOTAL_SIZE]; QM_PLANES]; QM_LEVELS]> {
    let mut deltas = Vec::with_capacity(TOTAL_QM_BYTES);
    let mut cpos = 0usize;
    while deltas.len() < TOTAL_QM_BYTES {
        let flags = PACKED_QM[cpos];
        cpos += 1;
        for bit in 0..8 {
            if deltas.len() >= TOTAL_QM_BYTES {
                break;
            }
            if ((flags >> bit) & 1) != 0 {
                let b0 = PACKED_QM[cpos] as usize;
                let b1 = PACKED_QM[cpos + 1] as usize;
                cpos += 2;
                let off = (b0 | ((b1 >> 4) << 8)) + 1;
                let len = (b1 & 0x0f) + 3;
                let start = deltas.len() - off;
                for k in 0..len {
                    let v = deltas[start + k];
                    deltas.push(v);
                }
            } else {
                deltas.push(PACKED_QM[cpos]);
                cpos += 1;
            }
        }
    }
    let mut mat = Box::new([[[0u8; QM_TOTAL_SIZE]; QM_PLANES]; QM_LEVELS]);
    let mut pos = 0usize;
    for p in 0..QM_PLANES {
        let mut prev = [32u8; QM_TOTAL_SIZE];
        for lv in (0..QM_LEVELS).rev() {
            let slot = &mut mat[lv][p];
            for k in 0..QM_TOTAL_SIZE {
                slot[k] = prev[k].wrapping_add(deltas[pos + k]);
            }
            prev = *slot;
            pos += QM_TOTAL_SIZE;
        }
    }
    mat
}

/// Normative `Quantizer_Matrix[ QM_LEVELS ][ QM_PLANES ][ QM_TOTAL_SIZE ]`, unpacked on first touch.
pub static QUANTIZER_MATRIX: LazyLock<Box<[[[u8; QM_TOTAL_SIZE]; QM_PLANES]; QM_LEVELS]>> =
    LazyLock::new(unpack_quantizer_matrix);

/// §7.12.3 step-1b `Quantizer_Matrix[ seg_qm_level ][ plane > 0 ][ Qm_Offset[ txSz ] + i * tw + j ]` lookup.
#[inline]
pub fn qmatrix_value(seg_qm_level: u8, plane: u8, tx_size: usize, i: usize, j: usize) -> u8 {
    debug_assert!(seg_qm_level < QM_LEVELS as u8);
    debug_assert!(tx_size < TX_SIZES_ALL);
    let tw = core::cmp::min(32, TX_WIDTH[tx_size]);
    let th = core::cmp::min(32, TX_HEIGHT[tx_size]);
    debug_assert!(i < th, "qmatrix_value: i {} >= th {}", i, th);
    debug_assert!(j < tw, "qmatrix_value: j {} >= tw {}", j, tw);
    let plane_idx = if plane > 0 { 1 } else { 0 };
    let offset = QM_OFFSET[tx_size] + i * tw + j;
    QUANTIZER_MATRIX[seg_qm_level as usize][plane_idx][offset]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::{
        TX_4X4, TX_4X8, TX_4X16, TX_8X4, TX_8X8, TX_8X16, TX_8X32, TX_16X4, TX_16X8, TX_16X16,
        TX_16X32, TX_16X64, TX_32X8, TX_32X16, TX_32X32, TX_32X64, TX_64X16, TX_64X32, TX_64X64,
    };

    #[test]
    fn qmatrix_fnv1a64_matches_full_spec_table() {
        let mut h: u64 = 0xcbf29ce484222325;
        for level_planes in QUANTIZER_MATRIX.iter() {
            for plane in level_planes.iter() {
                for &v in plane.iter() {
                    h ^= u64::from(v);
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
        }
        assert_eq!(h, 0x49344e39b977df34);
    }

    #[test]
    fn qm_total_size_constant() {
        assert_eq!(QM_TOTAL_SIZE, 3344);
        assert_eq!(QM_LEVELS, 15);
        assert_eq!(QM_PLANES, 2);
    }

    #[test]
    fn qm_offset_matches_spec_table() {
        assert_eq!(
            QM_OFFSET,
            [
                0, 16, 80, 336, 336, 1360, 1392, 1424, 1552, 1680, 2192, 336, 336, 2704, 2768,
                2832, 3088, 1680, 2192
            ]
        );
    }

    #[test]
    fn qm_offset_unique_entries_sum_to_total_size() {
        let unique: [usize; 14] = [
            4 * 4,
            8 * 8,
            16 * 16,
            32 * 32,
            4 * 8,
            8 * 4,
            8 * 16,
            16 * 8,
            16 * 32,
            32 * 16,
            4 * 16,
            16 * 4,
            8 * 32,
            32 * 8,
        ];
        assert_eq!(unique.iter().sum::<usize>(), QM_TOTAL_SIZE);
    }

    #[test]
    fn qmatrix_level0_luma_4x4_first_row() {
        assert_eq!(qmatrix_value(0, 0, TX_4X4, 0, 0), 32);
        assert_eq!(qmatrix_value(0, 0, TX_4X4, 0, 1), 43);
        assert_eq!(qmatrix_value(0, 0, TX_4X4, 0, 2), 73);
        assert_eq!(qmatrix_value(0, 0, TX_4X4, 0, 3), 97);
        assert_eq!(qmatrix_value(0, 0, TX_4X4, 3, 3), 200);
    }

    #[test]
    fn qmatrix_level0_chroma_4x4_first_row() {
        assert_eq!(qmatrix_value(0, 1, TX_4X4, 0, 0), 35);
        assert_eq!(qmatrix_value(0, 2, TX_4X4, 0, 0), 35);
        assert_eq!(qmatrix_value(0, 1, TX_4X4, 0, 1), 46);
        assert_eq!(qmatrix_value(0, 1, TX_4X4, 0, 2), 57);
        assert_eq!(qmatrix_value(0, 1, TX_4X4, 0, 3), 66);
        assert_eq!(qmatrix_value(0, 1, TX_4X4, 3, 3), 109);
    }

    #[test]
    fn qmatrix_level14_luma_4x4_first_row_is_31() {
        for j in 0..4 {
            assert_eq!(qmatrix_value(14, 0, TX_4X4, 0, j), 31);
        }
    }

    #[test]
    fn qmatrix_all_values_in_observed_range() {
        for level_planes in QUANTIZER_MATRIX.iter() {
            for plane in level_planes.iter() {
                for &v in plane.iter() {
                    assert!((30..=242).contains(&v));
                }
            }
        }
    }

    #[test]
    fn qmatrix_value_covers_every_tx_size_within_loop_bounds() {
        let sizes = [
            TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64, TX_4X8, TX_8X4, TX_8X16, TX_16X8,
            TX_16X32, TX_32X16, TX_32X64, TX_64X32, TX_4X16, TX_16X4, TX_8X32, TX_32X8, TX_16X64,
            TX_64X16,
        ];
        for &sz in &sizes {
            let tw = core::cmp::min(32, crate::cdf::TX_WIDTH[sz]);
            let th = core::cmp::min(32, crate::cdf::TX_HEIGHT[sz]);
            for i in 0..th {
                for j in 0..tw {
                    let _ = qmatrix_value(0, 0, sz, i, j);
                    let _ = qmatrix_value(0, 1, sz, i, j);
                }
            }
        }
    }

    #[test]
    fn qmatrix_tx_64x64_shares_tx_32x32_storage() {
        for i in [0, 7, 31] {
            for j in [0, 5, 31] {
                let v32 = qmatrix_value(3, 0, TX_32X32, i, j);
                let v64 = qmatrix_value(3, 0, TX_64X64, i, j);
                assert_eq!(v32, v64, "TX_32X32 vs TX_64X64 at ({i}, {j})");
            }
        }
    }

    #[test]
    fn qmatrix_tx_share_pairs_match() {
        for i in [0, 15, 31] {
            for j in [0, 15] {
                assert_eq!(
                    qmatrix_value(5, 0, TX_16X32, i, j),
                    qmatrix_value(5, 0, TX_16X64, i, j),
                );
            }
        }
        for i in [0, 15] {
            for j in [0, 31] {
                assert_eq!(
                    qmatrix_value(5, 1, TX_32X16, i, j),
                    qmatrix_value(5, 1, TX_64X16, i, j),
                );
            }
        }
    }

    #[test]
    fn qm_offset_advance_matches_tw_times_th() {
        let pairs = [
            (TX_4X4, 16),
            (TX_8X8, 80),
            (TX_16X16, 336),
            (TX_32X32, 1360),
            (TX_4X8, 1392),
            (TX_8X4, 1424),
            (TX_8X16, 1552),
            (TX_16X8, 1680),
            (TX_16X32, 2192),
            (TX_32X16, 2704),
            (TX_4X16, 2768),
            (TX_16X4, 2832),
            (TX_8X32, 3088),
            (TX_32X8, 3344),
        ];
        for (sz, next) in pairs {
            let tw = core::cmp::min(32, crate::cdf::TX_WIDTH[sz]);
            let th = core::cmp::min(32, crate::cdf::TX_HEIGHT[sz]);
            assert_eq!(QM_OFFSET[sz] + tw * th, next, "tx_size = {sz}");
        }
    }
}
