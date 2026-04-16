/*****************************************************************************
 * weBIGeo Clouds
 * Copyright (C) 2026 Wendelin Muth
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

@group(0) @binding(0) var input_tex : texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> output_buf : array<vec2<u32>>;

fn get_index(val: f32, min_v: f32, max_v: f32) -> u32 {
    if (max_v <= min_v) { return 0u; }
    let range_v = max_v - min_v;
    let offset = val - min_v;
    let n = clamp(offset / range_v, 0.0, 1.0);
    let steps = 7.0;
    let select = round(n * steps); 
    let raw_idx = 7u - u32(select);
    if (raw_idx == 0u) { return 0u; } 
    if (raw_idx == 7u) { return 1u; } 
    return raw_idx + 1u;
}

fn get_value(idx: u32, min_v: f32, max_v: f32) -> f32 {
    if (idx == 0u) { return max_v; }
    if (idx == 1u) { return min_v; }
    let w = f32(8u - idx) / 7.0; 
    return min_v + (max_v - min_v) * w;
}

fn calculate_block_error(pixels: array<f32, 16>, min_v: f32, max_v: f32) -> f32 {
    var total_err = 0.0;
    for (var i = 0u; i < 16u; i++) {
        let p = pixels[i];
        let idx = get_index(p, min_v, max_v);
        let recon = get_value(idx, min_v, max_v);
        let diff = p - recon;
        total_err += diff * diff;
    }
    return total_err;
}

// --- Main Kernel ---

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let dims = textureDimensions(input_tex);
    let block_x = global_id.x;
    let block_y = global_id.y;
    let slice_z = global_id.z;
    
    // Bounds check
    if (block_x * 4u >= dims.x || block_y * 4u >= dims.y || slice_z >= dims.z) {
        return;
    }

    // 1. Load 4x4 Block from the specific Z-slice
    var pixels: array<f32, 16>;
    var raw_min = 1.0;
    var raw_max = 0.0;

    for (var y = 0u; y < 4u; y++) {
        for (var x = 0u; x < 4u; x++) {
            let coords = vec3<i32>(i32(block_x * 4u + x), i32(block_y * 4u + y), i32(slice_z));
            // Read RED channel
            let p = textureLoad(input_tex, coords, 0).r;
            pixels[y * 4u + x] = p;
            raw_min = min(raw_min, p);
            raw_max = max(raw_max, p);
        }
    }

    // 2. Iterative Refinement
    var best_min = raw_min;
    var best_max = raw_max;
    var best_err = 100000.0;
    let step = 0.004; 

    for (var i = 0u; i < 5u; i++) {
        var t_min = raw_min;
        var t_max = raw_max;
        if (i == 1u) { t_min += step; t_max -= step; }
        if (i == 2u) { t_min += step; }
        if (i == 3u) { t_max -= step; }
        if (i == 4u) { t_min -= step; t_max += step; }

        t_min = clamp(t_min, 0.0, 1.0);
        t_max = clamp(t_max, 0.0, 1.0);
        if (t_min >= t_max) { t_max = t_min + 0.0001; }

        let err = calculate_block_error(pixels, t_min, t_max);
        if (err < best_err) {
            best_err = err;
            best_min = t_min;
            best_max = t_max;
        }
    }

    // 3. Quantize
    var r0 = u32(round(best_max * 255.0));
    var r1 = u32(round(best_min * 255.0));

    // Conservative quantization: Preserve non-zero values.
    if (raw_max > 0.0 && r0 == 0u) {
        // Force the endpoints to be the smallest possible non-zero range.
        // This creates the [0, 1/255] palette, which guarantees the block is not all
        // zeros and has the highest possible precision for values near zero.
        r0 = 1u;
        r1 = 0u;
    }

    let final_max = f32(r0) / 255.0;
    let final_min = f32(r1) / 255.0;

    var word0 = (r1 << 8u) | r0; 
    var word1 = 0u;

    // 4. Pack Indices
    var current_bit_pos = 16u;
    for (var i = 0u; i < 16u; i++) {
        var idx = 0u;
        if (r0 > r1) {
            idx = get_index(pixels[i], final_min, final_max);
            // Conservative quantization: Preserve non-zero values.
            if (pixels[i] > 0.0 && idx == 1u && r1 == 0u) {
                idx = 7u;
            }
        }
        // Note: 6-value mode (else branch) not implemented.

        if (current_bit_pos < 32u) {
            let bits_fit = 32u - current_bit_pos;
            if (bits_fit >= 3u) {
                word0 = word0 | (idx << current_bit_pos);
            } else {
                let mask0 = (1u << bits_fit) - 1u;
                word0 = word0 | ((idx & mask0) << current_bit_pos);
                word1 = word1 | (idx >> bits_fit);
            }
        } else {
            word1 = word1 | (idx << (current_bit_pos - 32u));
        }
        current_bit_pos += 3u;
    }

    // 5. Output calculation for 3D
    // BC4 blocks are linear in memory. Usually organized slice by slice.
    let blocks_x = (dims.x + 3u) / 4u;
    let blocks_y = (dims.y + 3u) / 4u;
    let blocks_per_slice = blocks_x * blocks_y;
    
    let block_idx = (slice_z * blocks_per_slice) + (block_y * blocks_x) + block_x;
    
    output_buf[block_idx] = vec2<u32>(word0, word1);
}