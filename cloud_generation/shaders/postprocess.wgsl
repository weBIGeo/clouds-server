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
 
struct Params {
    nx_src: f32, ny_src: f32, nz_src: f32, nz_hhl: f32,
    global_x_m: f32, global_y_m: f32, res_h_m: f32, res_v_m: f32,
    off_x: f32, off_y: f32, inv_scale_x: f32, inv_scale_y: f32,
    nx_f: f32, ny_f: f32, nz_f: f32, unused: f32,
};

@group(0) @binding(0) var<storage, read_write> b_in : array<f32>;
@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var<storage, read_write> b_out : array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x);
    let y = i32(id.y);
    let nx = i32(params.nx_f);
    let ny = i32(params.ny_f);
    let nz = i32(params.nz_f);

    if (x >= nx || y >= ny) { return; }

    for (var z = 0; z < nz; z++) {
        var sum: f32 = 0.0;
        var count: f32 = 0.0;

        for (var dz = -1; dz <= 1; dz++) {
            for (var dy = -1; dy <= 1; dy++) {
                for (var dx = -1; dx <= 1; dx++) {
                    let nx_ = clamp(x + dx, 0, nx - 1);
                    let ny_ = clamp(y + dy, 0, ny - 1);
                    let nz_ = clamp(z + dz, 0, nz - 1);

                    // Separable 1D kernel: center=2, adjacent=1 (sum=4 per axis).
                    // 3D product gives center=8, face=4, edge=2, corner=1, total=64.
                    let kx = select(1, 2, dx == 0);
                    let ky = select(1, 2, dy == 0);
                    let kz = select(1, 2, dz == 0);

                    sum += b_in[(nz_ * ny + ny_) * nx + nx_] * f32(kx * ky * kz);
                }
            }
        }

        b_out[(z * ny + y) * nx + x] = sum / 64.0;
    }
}
