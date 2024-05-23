/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use bytemuck::{cast, cast_ref};
use log::info;
use wide::i16x8;

use crate::consts::ZIGZAG_TO_TRANSPOSED;

use super::{block_context::BlockContext, jpeg_header::JPegHeader};

use unroll::unroll_for_loops;

/// holds the 8x8 blocks for a given component. Since we do multithreaded encoding,
/// the image may only hold a subset of the components (specified by dpos_offset),
/// but they can be merged
pub struct BlockBasedImage {
    block_width: i32,

    original_height: i32,

    dpos_offset: i32,

    image: Vec<AlignedBlock>,
}

static EMPTY: AlignedBlock = AlignedBlock { raw_data: [0; 64] };

impl BlockBasedImage {
    // constructs new block image for the given y-coordinate range
    pub fn new(
        jpeg_header: &JPegHeader,
        component: usize,
        luma_y_start: i32,
        luma_y_end: i32,
    ) -> Self {
        let block_width = jpeg_header.cmp_info[component].bch;
        let original_height = jpeg_header.cmp_info[component].bcv;
        let max_size = block_width * original_height;

        let image_capcity = usize::try_from(
            (i64::from(max_size) * i64::from(luma_y_end - luma_y_start)
                + i64::from(jpeg_header.cmp_info[0].bcv - 1 /* round up */))
                / i64::from(jpeg_header.cmp_info[0].bcv),
        )
        .unwrap();

        let dpos_offset = i32::try_from(
            i64::from(max_size) * i64::from(luma_y_start) / i64::from(jpeg_header.cmp_info[0].bcv),
        )
        .unwrap();

        return BlockBasedImage {
            block_width: block_width,
            original_height: original_height,
            image: Vec::with_capacity(image_capcity),
            dpos_offset: dpos_offset,
        };
    }

    /// merges a bunch of block images generated by different threads into a single one used by progressive decoding
    pub fn merge(images: &mut Vec<Vec<BlockBasedImage>>, index: usize) -> Self {
        // figure out the total size of all the blocks so we can set the capacity correctly
        let total_size = images.iter().map(|x| x[index].image.len()).sum();

        let mut contents = Vec::with_capacity(total_size);
        let mut block_width = None;
        let mut original_height = None;

        for v in images {
            assert!(
                v[index].dpos_offset == contents.len() as i32,
                "previous content should match new content"
            );

            if let Some(w) = block_width {
                assert_eq!(w, v[index].block_width, "all block_width must match")
            } else {
                block_width = Some(v[index].block_width);
            }

            if let Some(w) = original_height {
                assert_eq!(
                    w, v[index].original_height,
                    "all original_height must match"
                )
            } else {
                original_height = Some(v[index].original_height);
            }

            contents.append(&mut v[index].image);
        }

        return BlockBasedImage {
            block_width: block_width.unwrap(),
            original_height: original_height.unwrap(),
            image: contents,
            dpos_offset: 0,
        };
    }

    #[allow(dead_code)]
    pub fn dump(&self) {
        info!(
            "size = {0}, capacity = {1}, dpos_offset = {2}",
            self.image.len(),
            self.image.capacity(),
            self.dpos_offset
        );
    }

    pub fn off_y(&self, y: i32) -> BlockContext {
        return BlockContext::new(
            self.block_width * y,
            if y != 0 {
                self.block_width * (y - 1)
            } else {
                -1
            },
            if (y & 1) != 0 { self.block_width } else { 0 },
            if (y & 1) != 0 { 0 } else { self.block_width },
            self,
        );
    }

    pub fn get_block_width(&self) -> i32 {
        self.block_width
    }

    pub fn get_original_height(&self) -> i32 {
        self.original_height
    }

    fn fill_up_to_dpos(&mut self, dpos: i32) {
        // set our dpos the first time we get set, since we should be seeing our data in order
        if self.image.len() == 0 {
            assert!(self.dpos_offset == dpos);
        }

        assert!(dpos >= self.dpos_offset);

        while self.image.len() <= (dpos - self.dpos_offset) as usize {
            if self.image.len() >= self.image.capacity() {
                panic!("out of memory");
            }
            self.image.push(AlignedBlock { raw_data: [0; 64] });
        }
    }

    pub fn set_block_data(&mut self, dpos: i32, block_data: &AlignedBlock) {
        self.fill_up_to_dpos(dpos);
        *self.image[(dpos - self.dpos_offset) as usize].get_block_mut() = *block_data.get_block();
    }

    pub fn get_block(&self, dpos: i32) -> &AlignedBlock {
        if (dpos - self.dpos_offset) as usize >= self.image.len() {
            return &EMPTY;
        } else {
            return &self.image[(dpos - self.dpos_offset) as usize];
        }
    }

    #[inline(always)]
    pub fn append_block(&mut self, block: AlignedBlock) {
        assert!(
            self.image.len() < self.image.capacity(),
            "capacity should be set correctly"
        );
        self.image.push(block);
    }

    pub fn get_block_mut(&mut self, dpos: i32) -> &mut AlignedBlock {
        self.fill_up_to_dpos(dpos);
        return &mut self.image[(dpos - self.dpos_offset) as usize];
    }
}

/// block of 64 coefficients in the aligned order, which is similar to zigzag except that the 7x7 lower right square comes first,
/// followed by the DC, followed by the edges
#[repr(C, align(32))]
pub struct AlignedBlock {
    raw_data: [i16; 64],
}

pub static EMPTY_BLOCK: AlignedBlock = AlignedBlock { raw_data: [0; 64] };

impl Default for AlignedBlock {
    fn default() -> Self {
        AlignedBlock { raw_data: [0; 64] }
    }
}

impl AlignedBlock {
    pub fn new(block: [i16; 64]) -> Self {
        AlignedBlock { raw_data: block }
    }

    #[allow(dead_code)]
    pub fn as_i16x8(&self, index: usize) -> i16x8 {
        let v: &[i16x8; 8] = cast_ref(&self.raw_data);
        v[index]
    }

    #[allow(dead_code)]
    pub fn transpose(&self) -> AlignedBlock {
        return AlignedBlock::new(cast(i16x8::transpose(cast(*self.get_block()))));
    }

    pub fn get_dc(&self) -> i16 {
        return self.raw_data[0];
    }

    pub fn set_dc(&mut self, value: i16) {
        self.raw_data[0] = value
    }

    /// gets underlying array of 64 coefficients (guaranteed to be 32-byte aligned)
    #[unroll_for_loops]
    pub fn zigzag_from_transposed(&self) -> AlignedBlock {
        let mut block = AlignedBlock::default();
        for i in 0..64 {
            block.raw_data[i] = self.raw_data[usize::from(ZIGZAG_TO_TRANSPOSED[i])];
        }
        return block;
    }

    // used for debugging
    #[allow(dead_code)]
    pub fn get_block(&self) -> &[i16; 64] {
        return &self.raw_data;
    }

    // used for debugging
    #[allow(dead_code)]
    pub fn get_block_mut(&mut self) -> &mut [i16; 64] {
        return &mut self.raw_data;
    }

    // used for debugging
    #[allow(dead_code)]
    pub fn get_hash(&self) -> i32 {
        let mut sum = 0;
        for i in 0..64 {
            sum += self.raw_data[i] as i32
        }
        return sum;
    }

    pub fn get_count_of_non_zeros_7x7(&self) -> u8 {
        let mut num_non_zeros7x7: u8 = 0;
        for index in 9..64 {
            if index & 0x7 != 0 && self.raw_data[index] != 0 {
                num_non_zeros7x7 += 1;
            }
        }

        return num_non_zeros7x7;
    }

    pub fn get_coefficient(&self, index: usize) -> i16 {
        return self.raw_data[index];
    }

    pub fn set_coefficient(&mut self, index: usize, v: i16) {
        self.raw_data[index] = v;
    }

    pub fn set_transposed_from_zigzag(&mut self, index: usize, v: i16) {
        self.raw_data[usize::from(ZIGZAG_TO_TRANSPOSED[index])] = v;
    }

    pub fn get_transposed_from_zigzag(&self, index: usize) -> i16 {
        return self.raw_data[usize::from(ZIGZAG_TO_TRANSPOSED[index])];
    }

    pub fn from_stride(&self, offset: usize, stride: usize) -> i16x8 {
        return i16x8::new([
            self.raw_data[offset],
            self.raw_data[offset + (1 * stride)],
            self.raw_data[offset + (2 * stride)],
            self.raw_data[offset + (3 * stride)],
            self.raw_data[offset + (4 * stride)],
            self.raw_data[offset + (5 * stride)],
            self.raw_data[offset + (6 * stride)],
            self.raw_data[offset + (7 * stride)],
        ]);
    }
}
