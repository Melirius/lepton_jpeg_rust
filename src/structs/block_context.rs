/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use super::block_based_image::{AlignedBlock, BlockBasedImage};
use super::neighbor_summary::NeighborSummary;

pub struct BlockContext {
    block_width: i32,

    cur_block_index: i32,
    above_block_index: i32,

    cur_num_non_zeros_index: i32,
    above_num_non_zero_index: i32,
}

impl BlockContext {
    // for debugging
    #[allow(dead_code)]
    pub fn get_here_index(&self) -> i32 {
        self.cur_block_index
    }

    pub fn next(&mut self, has_more: bool) -> i32 {
        self.cur_block_index += 1;

        let retval = self.cur_block_index;

        if retval < self.block_width {
            self.above_block_index = self.cur_block_index + self.block_width;
        } else {
            self.above_block_index = self.cur_block_index - self.block_width;
        }

        self.cur_num_non_zeros_index += 1;
        self.above_num_non_zero_index += 1;

        if !has_more {
            let cur_row_first = self.cur_num_non_zeros_index < self.above_num_non_zero_index;
            if cur_row_first {
                self.above_num_non_zero_index -= self.block_width * 2;
            } else {
                self.cur_num_non_zeros_index -= self.block_width * 2;
            }
        }

        return retval;
    }

    pub fn new(
        cur_block_index: i32,
        above_block_index: i32,
        cur_num_non_zeros_index: i32,
        above_num_non_zero_index: i32,
        image_data: &BlockBasedImage,
    ) -> Self {
        return BlockContext {
            block_width: image_data.get_block_width(),
            cur_block_index,
            above_block_index,
            cur_num_non_zeros_index,
            above_num_non_zero_index,
        };
    }

    /// returns references to the surrounding blocks plus the current mutable one
    pub fn get_blocks_mut<'a>(
        &self,
        image_data: &'a mut BlockBasedImage,
        left_present: bool,
        above_present: bool,
    ) -> (
        &'a [i16; 64],
        &'a [i16; 64],
        &'a [i16; 64],
        &'a mut AlignedBlock,
    ) {
        image_data.get_blocks_mut(
            if above_present {
                self.above_block_index
            } else {
                -1
            },
            if left_present {
                self.cur_block_index - 1
            } else {
                -1
            },
            if above_present && left_present {
                self.above_block_index - 1
            } else {
                -1
            },
            self.cur_block_index,
        )
    }

    /// returns references to the surrounding blocks plus the current one 
    pub fn get_blocks<'a>(
        &self,
        image_data: &'a BlockBasedImage,
        left_present: bool,
        above_present: bool,
    ) -> (
        &'a [i16; 64],
        &'a [i16; 64],
        &'a [i16; 64],
        &'a AlignedBlock,
    ) {
        image_data.get_blocks(
            if above_present {
                self.above_block_index
            } else {
                -1
            },
            if left_present {
                self.cur_block_index - 1
            } else {
                -1
            },
            if above_present && left_present {
                self.above_block_index - 1
            } else {
                -1
            },
            self.cur_block_index,
        )
    }

    pub fn get_non_zeros_above(&self, num_non_zeros: &[NeighborSummary]) -> u8 {
        return num_non_zeros[self.above_num_non_zero_index as usize].get_num_non_zeros();
    }

    pub fn get_non_zeros_left(&self, num_non_zeros: &[NeighborSummary]) -> u8 {
        return num_non_zeros[(self.cur_num_non_zeros_index - 1) as usize].get_num_non_zeros();
    }

    pub fn neighbor_context_here<'a>(
        &mut self,
        num_non_zeros: &'a mut [NeighborSummary],
    ) -> &'a mut NeighborSummary {
        return &mut num_non_zeros[self.cur_num_non_zeros_index as usize];
    }

    pub fn neighbor_context_above<'a>(
        &self,
        num_non_zeros: &'a [NeighborSummary],
    ) -> &'a NeighborSummary {
        return &num_non_zeros[self.above_num_non_zero_index as usize];
    }

    pub fn neighbor_context_left<'a>(
        &self,
        num_non_zeros: &'a [NeighborSummary],
    ) -> &'a NeighborSummary {
        return &num_non_zeros[(self.cur_num_non_zeros_index - 1) as usize];
    }
}
