/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use wide::u32x8;

use crate::consts::*;
use crate::helpers::*;

use super::jpeg_header::JPegHeader;

// The maximal quantization coefficient for which division-by-multiplication scheme
// of `calc_coefficient_context8_lak` is working. This should cover all standard JPEGs,
// as normal DCT coefficients are less than 2048 in magnitude for 8-bit JPEGs
// (and they should have 8-bit quantization tables, citing JPEG standard, sec. B.2.4.1
// "Quantization table-specification syntax": "An 8-bit DCT-based process
// shall not use a 16-bit precision quantization table.").
const MAX_NORMAL_Q: u16 = 4177;

pub struct QuantizationTables {
    quantization_table: [u16; 64],
    quantization_table_transposed: [u16; 64],

    quantization_table_transposed_recip_vert: u32x8,
    quantization_table_transposed_recip_horiz: u32x8,

    // Values for discrimination between "regular" and "noise" part of
    // edge AC coefficients, used in `read/write_edge_coefficient`.
    // Calculated using approximate maximal magnitudes
    // of these coefficients `FREQ_MAX`
    min_noise_threshold: [u8; 14],

    normal_table: bool,
}

impl QuantizationTables {
    pub fn new(jpeg_header: &JPegHeader, component: usize) -> Self {
        Self::new_from_table(
            &jpeg_header.q_tables[usize::from(jpeg_header.cmp_info[component].q_table_index)],
        )
    }

    pub fn new_from_table(quantization_table: &[u16; 64]) -> Self {
        let mut retval = QuantizationTables {
            quantization_table: [0; 64],
            quantization_table_transposed: [0; 64],
            min_noise_threshold: [0; 14],
            quantization_table_transposed_recip_vert: u32x8::default(),
            quantization_table_transposed_recip_horiz: u32x8::default(),
            normal_table: true,
        };

        for pixel_row in 0..8 {
            for pixel_column in 0..8 {
                let coord = (pixel_row * 8) + pixel_column;
                let coord_tr = (pixel_column * 8) + pixel_row;
                let q = quantization_table[RASTER_TO_ZIGZAG[coord] as usize];

                retval.quantization_table[coord] = q;
                retval.quantization_table_transposed[coord_tr] = q;

                // the division-by-reciprocal-multiplication method used is working
                // up to values of 8355, else we fallback to division
                if q > MAX_NORMAL_Q {
                    retval.normal_table = false;
                }
            }
        }

        for i in 0..8 {
            retval
                .quantization_table_transposed_recip_horiz
                .as_array_mut()[i] = Self::recip(retval.quantization_table[i]);
            retval
                .quantization_table_transposed_recip_vert
                .as_array_mut()[i] = Self::recip(retval.quantization_table_transposed[i]);
        }

        for i in 0..14 {
            let coord = if i < 7 { i + 1 } else { (i - 6) * 8 };
            if retval.quantization_table[coord] < 9 {
                let mut freq_max = FREQ_MAX[i] + retval.quantization_table[coord] - 1;
                if retval.quantization_table[coord] != 0 {
                    freq_max /= retval.quantization_table[coord];
                }

                let max_len = u16_bit_length(freq_max) as u8;
                if max_len > RESIDUAL_NOISE_FLOOR as u8 {
                    retval.min_noise_threshold[i] = max_len - RESIDUAL_NOISE_FLOOR as u8;
                }
            }
        }

        retval
    }

    fn recip(v: u16) -> u32 {
        // for divide by zero, just return zero since this was the behavior
        // in the original code (rather than rejecting the JPEG outright)
        if v == 0 {
            return 0;
        }
        let mut retval = (1u32 << 31) + v as u32 - 1;
        retval /= v as u32;

        return retval;
    }

    pub fn quantization_table_transposed_recip<const HORIZONTAL: bool>(&self) -> u32x8 {
        if HORIZONTAL {
            self.quantization_table_transposed_recip_horiz
        } else {
            self.quantization_table_transposed_recip_vert
        }
    }

    pub fn get_quantization_table(&self) -> &[u16; 64] {
        &self.quantization_table
    }

    pub fn get_quantization_table_transposed(&self) -> &[u16; 64] {
        &self.quantization_table_transposed
    }

    pub fn get_min_noise_threshold(&self, coef: usize) -> u8 {
        self.min_noise_threshold[coef]
    }

    pub fn is_normal_table(&self) -> bool {
        self.normal_table
    }
}

#[test]
fn recip_test() {
    use rayon::prelude::*;

    assert!((1..MAX_NORMAL_Q as u64 + 1).into_par_iter().all(|q| {
        let recip = QuantizationTables::recip(q as u16) as u64;
        for i in 0u64..(1 << 19) + 1 {
            if i / q != i * recip >> 31 {
                return false;
            };
        }
        return true;
    }));

    // for q in 1..MAX_NORMAL_Q as u64 + 1 {
    //     let recip = QuantizationTables::recip(q as u16) as u64;
    //     for i in 0u64..(1 << 19) + 1 {
    //         assert_eq!(i / q, i * recip >> 31, "{} {}", i, q);
    //     }
    // }
}
