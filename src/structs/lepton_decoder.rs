/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use anyhow::{Context, Result};

use bytemuck::cast_mut;
use wide::i32x8;

use default_boxed::DefaultBoxed;

use std::io::Read;

use crate::consts::UNZIGZAG_49_TR;
use crate::enabled_features::EnabledFeatures;
use crate::helpers::{err_exit_code, here, u32_bit_length};
use crate::lepton_error::ExitCode;

use crate::metrics::Metrics;
use crate::structs::{
    block_based_image::AlignedBlock, block_based_image::BlockBasedImage, model::Model,
    model::ModelPerColor, neighbor_summary::NeighborSummary, probability_tables::ProbabilityTables,
    probability_tables_set::ProbabilityTablesSet, quantization_tables::QuantizationTables,
    row_spec::RowSpec, truncate_components::*, vpx_bool_reader::VPXBoolReader,
};

use super::block_context::{BlockContext, NeighborData};

// reads stream from reader and populates image_data with the decoded data

#[inline(never)] // don't inline so that the profiler can get proper data
pub fn lepton_decode_row_range<R: Read>(
    pts: &ProbabilityTablesSet,
    qt: &[QuantizationTables],
    trunc: &TruncateComponents,
    image_data: &mut [BlockBasedImage],
    reader: &mut R,
    min_y: i32,
    max_y: i32,
    is_last_thread: bool,
    full_file_compression: bool,
    features: &EnabledFeatures,
) -> Result<Metrics> {
    let component_size_in_blocks = trunc.get_component_sizes_in_blocks();
    let max_coded_heights = trunc.get_max_coded_heights();

    let mut is_top_row = Vec::new();
    let mut neighbor_summary_cache = Vec::new();

    // Init helper structures
    for i in 0..image_data.len() {
        is_top_row.push(true);

        let num_non_zeros_length = (image_data[i].get_block_width() << 1) as usize;

        let mut num_non_zero_list = Vec::new();
        num_non_zero_list.resize(num_non_zeros_length, NeighborSummary::default());

        neighbor_summary_cache.push(num_non_zero_list);
    }

    let mut model = Model::default_boxed();
    let mut bool_reader = VPXBoolReader::new(reader)?;

    let mut decode_index = 0;

    loop {
        let cur_row = RowSpec::get_row_spec_from_index(
            decode_index,
            &image_data[..],
            trunc.mcu_count_vertical,
            &max_coded_heights,
        );
        decode_index += 1;

        if cur_row.done {
            break;
        }

        if cur_row.luma_y >= max_y && !(is_last_thread && full_file_compression) {
            break;
        }

        if cur_row.skip {
            continue;
        }

        if cur_row.luma_y < min_y {
            continue;
        }

        decode_row_wrapper(
            &mut model,
            &mut bool_reader,
            pts,
            &mut image_data[cur_row.component],
            &qt[cur_row.component],
            &mut neighbor_summary_cache[cur_row.component],
            &mut is_top_row[..],
            &component_size_in_blocks[..],
            cur_row.component,
            cur_row.curr_y,
            features,
        )
        .context(here!())?;
    }
    Ok(bool_reader.drain_stats())
}

#[inline(never)] // don't inline so that the profiler can get proper data
fn decode_row_wrapper<R: Read>(
    model: &mut Model,
    bool_reader: &mut VPXBoolReader<R>,
    pts: &ProbabilityTablesSet,
    image_data: &mut BlockBasedImage,
    qt: &QuantizationTables,
    neighbor_summary_cache: &mut Vec<NeighborSummary>,
    is_top_row: &mut [bool],
    component_size_in_blocks: &[i32],
    component: usize,
    curr_y: i32,
    features: &EnabledFeatures,
) -> Result<()> {
    let mut context = image_data.off_y(curr_y);

    let block_width = image_data.get_block_width();
    if is_top_row[component] {
        is_top_row[component] = false;
        decode_row(
            model,
            bool_reader,
            &qt,
            &pts.corner[component],
            &pts.top[component],
            &pts.top[component],
            image_data,
            &mut context,
            neighbor_summary_cache,
            component_size_in_blocks[component],
            features,
        )
        .context(here!())?;
    } else if block_width > 1 {
        let _bt = component;
        decode_row(
            model,
            bool_reader,
            &qt,
            &pts.mid_left[component],
            &pts.middle[component],
            &pts.mid_right[component],
            image_data,
            &mut context,
            neighbor_summary_cache,
            component_size_in_blocks[component],
            features,
        )
        .context(here!())?;
    } else {
        assert!(block_width == 1, "block_width == 1");
        decode_row(
            model,
            bool_reader,
            &qt,
            &pts.width_one[component],
            &pts.width_one[component],
            &pts.width_one[component],
            image_data,
            &mut context,
            neighbor_summary_cache,
            component_size_in_blocks[component],
            features,
        )
        .context(here!())?;
    }

    Ok(())
}

fn decode_row<R: Read>(
    model: &mut Model,
    bool_reader: &mut VPXBoolReader<R>,
    qt: &QuantizationTables,
    left_model: &ProbabilityTables,
    middle_model: &ProbabilityTables,
    right_model: &ProbabilityTables,
    image_data: &mut BlockBasedImage,
    block_context: &mut BlockContext,
    neighbor_summary_cache: &mut [NeighborSummary],
    component_size_in_blocks: i32,
    features: &EnabledFeatures,
) -> Result<()> {
    let block_width = image_data.get_block_width();
    if block_width > 0 {
        parse_token::<R, false>(
            model,
            bool_reader,
            image_data,
            block_context,
            neighbor_summary_cache,
            qt,
            left_model,
            features,
        )
        .context(here!())?;
        let offset = block_context.next(true);

        if offset >= component_size_in_blocks {
            return Ok(()); // no sure if this is an error
        }
    }

    for _jpeg_x in 1..block_width - 1 {
        if middle_model.is_all_present() {
            parse_token::<R, true>(
                model,
                bool_reader,
                image_data,
                block_context,
                neighbor_summary_cache,
                qt,
                middle_model,
                features,
            )
            .context(here!())?;
        } else {
            parse_token::<R, false>(
                model,
                bool_reader,
                image_data,
                block_context,
                neighbor_summary_cache,
                qt,
                middle_model,
                features,
            )
            .context(here!())?;
        }

        let offset = block_context.next(true);

        if offset >= component_size_in_blocks {
            return Ok(()); // no sure if this is an error
        }
    }

    if block_width > 1 {
        if right_model.is_all_present() {
            parse_token::<R, true>(
                model,
                bool_reader,
                image_data,
                block_context,
                neighbor_summary_cache,
                qt,
                right_model,
                features,
            )
            .context(here!())?;
        } else {
            parse_token::<R, false>(
                model,
                bool_reader,
                image_data,
                block_context,
                neighbor_summary_cache,
                qt,
                right_model,
                features,
            )
            .context(here!())?;
        }

        block_context.next(false);
    }
    Ok(())
}

#[inline(never)] // don't inline so that the profiler can get proper data
fn parse_token<R: Read, const ALL_PRESENT: bool>(
    model: &mut Model,
    bool_reader: &mut VPXBoolReader<R>,
    image_data: &mut BlockBasedImage,
    context: &mut BlockContext,
    neighbor_summary_cache: &mut [NeighborSummary],
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    features: &EnabledFeatures,
) -> Result<()> {
    debug_assert!(pt.is_all_present() == ALL_PRESENT);

    let neighbors =
        context.get_neighbor_data::<ALL_PRESENT>(image_data, neighbor_summary_cache, pt);

    let (output, ns) =
        read_coefficient_block::<ALL_PRESENT, R>(pt, &neighbors, model, bool_reader, qt, features)?;

    context.set_neighbor_summary_here(neighbor_summary_cache, ns);

    image_data.append_block(output);

    Ok(())
}

/// Reads the 8x8 coefficient block from the bit reader, taking into account the neighboring
/// blocks, probability tables and model.
///
/// This function is designed to be independently callable without needing to know the context,
/// image data, etc so it can be extensively unit tested.
pub fn read_coefficient_block<const ALL_PRESENT: bool, R: Read>(
    pt: &ProbabilityTables,
    neighbor_data: &NeighborData,
    model: &mut Model,
    bool_reader: &mut VPXBoolReader<R>,
    qt: &QuantizationTables,
    features: &EnabledFeatures,
) -> Result<(AlignedBlock, NeighborSummary)> {
    let model_per_color = model.get_per_color(pt);

    // First we read the 49 inner coefficients

    // calculate the predictor context bin based on the neighbors
    let num_non_zeros_7x7_context_bin =
        pt.calc_num_non_zeros_7x7_context_bin::<ALL_PRESENT>(neighbor_data);

    // read how many of these are non-zero, which is used both
    // to terminate the loop early and as a predictor for the model
    let num_non_zeros_7x7 = model_per_color
        .read_non_zero_7x7_count(bool_reader, num_non_zeros_7x7_context_bin)
        .context(here!())?;

    if num_non_zeros_7x7 > 49 {
        // most likely a stream or model synchronization error
        return err_exit_code(ExitCode::StreamInconsistent, "numNonzeros7x7 > 49");
    }

    let mut output = AlignedBlock::default();
    let mut raster = [i32x8::ZERO; 8];
    let raster_col: &mut [i32; 64] = cast_mut(&mut raster);
    let mut nonzero_mask: u64 = 0;

    let mut num_non_zeros_7x7_remaining = num_non_zeros_7x7 as usize;

    if num_non_zeros_7x7_remaining > 0 {
        let best_priors = pt.calc_coefficient_context_7x7_aavg_block::<ALL_PRESENT>(
            neighbor_data.left,
            neighbor_data.above,
            neighbor_data.above_left,
        );

        // calculate the bin we are using for the number of non-zeros
        let mut num_non_zeros_bin =
            ProbabilityTables::num_non_zeros_to_bin_7x7(num_non_zeros_7x7_remaining);

        // now loop through the coefficients in zigzag, terminating once we hit the number of non-zeros
        for (zig49, &coord) in UNZIGZAG_49_TR.iter().enumerate() {
            let best_prior_bit_length = u32_bit_length(best_priors[coord as usize] as u32);

            let coef = model_per_color
                .read_coef(
                    bool_reader,
                    zig49,
                    num_non_zeros_bin,
                    best_prior_bit_length as usize,
                )
                .context(here!())?;

            if coef != 0 {
                output.set_coefficient(coord as usize, coef);
                raster_col[coord as usize] = i32::from(coef)
                    * i32::from(qt.get_quantization_table_transposed()[coord as usize]);

                nonzero_mask |= 1 << coord;

                num_non_zeros_7x7_remaining -= 1;
                if num_non_zeros_7x7_remaining == 0 {
                    break;
                }

                // update the bin since we've changed the number of non-zeros
                num_non_zeros_bin =
                    ProbabilityTables::num_non_zeros_to_bin_7x7(num_non_zeros_7x7_remaining);
            }
        }
    }

    if num_non_zeros_7x7_remaining > 0 {
        return err_exit_code(
            ExitCode::StreamInconsistent,
            "not enough nonzeros in 7x7 block",
        );
    }

    // step 2, read the edge coefficients
    // Here we produce the first part of edge DCT coefficients predictions for neighborhood blocks
    // and build transposed raster of dequantized DCT coefficients with 0 in DC
    let (horiz_pred, vert_pred) = decode_edge::<R, ALL_PRESENT>(
        neighbor_data,
        model_per_color,
        bool_reader,
        &mut output,
        qt,
        pt,
        num_non_zeros_7x7,
        &mut nonzero_mask,
        &mut raster,
    )?;

    // step 3, read the DC coefficient (0,0 of the block)
    let q0 = qt.get_quantization_table()[0] as i32;
    let predicted_dc = pt.adv_predict_dc_pix::<ALL_PRESENT>(&raster, q0, &neighbor_data, features);

    let coef = model
        .read_dc(
            bool_reader,
            pt.get_color_index(),
            predicted_dc.uncertainty,
            predicted_dc.uncertainty2,
        )
        .context(here!())?;
    output.set_dc(ProbabilityTables::adv_predict_or_unpredict_dc(
        coef,
        true,
        predicted_dc.predicted_dc,
    ) as i16);

    // neighbor summary is used as a predictor for the next block
    let neighbor_summary = NeighborSummary::new(
        &predicted_dc.advanced_predict_dc_pixels_sans_dc,
        output.get_dc() as i32 * q0,
        num_non_zeros_7x7,
        horiz_pred,
        vert_pred,
        features,
    );

    Ok((output, neighbor_summary))
}

//#[inline(never)] // don't inline so that the profiler can get proper data
fn decode_edge<R: Read, const ALL_PRESENT: bool>(
    neighbor_data: &NeighborData,
    model_per_color: &mut ModelPerColor,
    bool_reader: &mut VPXBoolReader<R>,
    here_mut: &mut AlignedBlock,
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    num_non_zeros_7x7: u8,
    nonzero_mask: &mut u64,
    raster: &mut [i32x8; 8],
) -> Result<(i32x8, i32x8)> {
    // here we calculate the furthest x and y coordinates that have non-zero coefficients
    // which are used as predictors for the number of edge coefficients
    let mask_7x7 = *nonzero_mask | 1;
    let mut mask_y = mask_7x7 | (mask_7x7 << 32);
    mask_y |= mask_y << 16;
    mask_y |= mask_y << 8;

    // effectively (7 - eob) of DB Lepton
    let eob_y = mask_y.leading_zeros() as u8;
    let eob_x = (mask_7x7.leading_zeros() >> 3) as u8;

    let num_non_zeros_bin = (num_non_zeros_7x7 + 3) / 7;

    let (h_pred, v_pred) =
        ProbabilityTables::predict_current_edges(neighbor_data, *nonzero_mask, raster);

    decode_one_edge::<R, ALL_PRESENT, true>(
        model_per_color,
        bool_reader,
        &h_pred.to_array(),
        here_mut,
        qt,
        pt,
        num_non_zeros_bin,
        nonzero_mask,
        eob_x,
        cast_mut(raster),
    )?;
    decode_one_edge::<R, ALL_PRESENT, false>(
        model_per_color,
        bool_reader,
        &v_pred.to_array(),
        here_mut,
        qt,
        pt,
        num_non_zeros_bin,
        nonzero_mask,
        eob_y,
        cast_mut(raster),
    )?;

    let (horiz_pred, vert_pred) = ProbabilityTables::predict_next_edges(raster, *nonzero_mask);

    Ok((horiz_pred, vert_pred))
}

fn decode_one_edge<R: Read, const ALL_PRESENT: bool, const HORIZONTAL: bool>(
    model_per_color: &mut ModelPerColor,
    bool_reader: &mut VPXBoolReader<R>,
    pred: &[i32; 8],
    here_mut: &mut AlignedBlock,
    qt: &QuantizationTables,
    pt: &ProbabilityTables,
    num_non_zeros_bin: u8,
    nonzero_mask: &mut u64,
    est_eob: u8,
    raster: &mut [i32; 64],
) -> Result<()> {
    let num_non_zeros_edge = model_per_color
        .read_non_zero_edge_count::<R, HORIZONTAL>(bool_reader, est_eob, num_non_zeros_bin)
        .context(here!())?;

    let delta;
    let mut zig15offset;

    if HORIZONTAL {
        delta = 8;
        zig15offset = 0;
    } else {
        delta = 1;
        zig15offset = 7;
    }

    let mut coord_tr = delta;
    let mut num_non_zeros_remaining = num_non_zeros_edge;

    for _lane in 0..7 {
        if num_non_zeros_remaining == 0 {
            break;
        }

        let best_prior = pt.calc_coefficient_context8_lak::<ALL_PRESENT, HORIZONTAL>(
            qt, coord_tr, pred,
        );

        let coef = model_per_color.read_edge_coefficient(
            bool_reader,
            qt,
            zig15offset,
            num_non_zeros_remaining,
            best_prior,
        )?;

        if coef != 0 {
            num_non_zeros_remaining -= 1;
            here_mut.set_coefficient(coord_tr, coef);
            raster[coord_tr as usize] =
                i32::from(coef) * i32::from(qt.get_quantization_table_transposed()[coord_tr]);

            *nonzero_mask |= 1 << coord_tr;
        }

        coord_tr += delta;
        zig15offset += 1;
    }

    if num_non_zeros_remaining != 0 {
        return err_exit_code(ExitCode::StreamInconsistent, "StreamInconsistent");
    }

    Ok(())
}
