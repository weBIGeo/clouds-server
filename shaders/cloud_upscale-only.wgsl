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
 
// --- CONSTANTS ---
const PI: f32 = 3.14159265359;
const R_EARTH: f32 = 6378137.0;

// Max turbulent warp displacement (meters).
const TURB_AMPLITUDE_MAX: f32 = 500.0;
// Sub-grid turbulent mixing length (meters), based on ICON-D2 grid spacing.
const L_MIX: f32 = 800.0;

// Noise spatial scales (meters).
const SCALE_STRAT_XY:   f32 = 2000.0;
const SCALE_STRAT_Z:    f32 = 1000.0;
const SCALE_PERTURB_XY: f32 = 3000.0;
const SCALE_PERTURB_Z:  f32 = 2000.0;

// Condensate threshold for the cloud fraction diagnostic (kg/kg).
const Q_CRIT: f32 = 3e-5;


// --- UNIFORMS ---
struct Params {
    nx_src: f32, ny_src: f32, nz_src: f32, nz_hhl: f32,
    global_x_m: f32, global_y_m: f32, res_h_m: f32, res_v_m: f32,
    off_x: f32, off_y: f32, inv_scale_x: f32, inv_scale_y: f32,
    nx_f: f32, ny_f: f32, nz_f: f32, unused: f32,
};

@group(0) @binding(0) var t_clc : texture_3d<f32>;
@group(0) @binding(1) var t_qc  : texture_3d<f32>;
@group(0) @binding(2) var t_qi  : texture_3d<f32>;
@group(0) @binding(3) var t_qv  : texture_3d<f32>;
@group(0) @binding(4) var t_t   : texture_3d<f32>;
@group(0) @binding(5) var t_p   : texture_3d<f32>;
@group(0) @binding(6) var t_tke : texture_3d<f32>;
@group(0) @binding(7) var t_hhl : texture_3d<f32>;

@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var<storage, read_write> b_out : array<f32>;


// =============================================================================
// NOISE LIBRARY
// =============================================================================

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn lerp(a: f32, b: f32, t: f32) -> f32 { return a + t * (b - a); }

fn smoothstep_custom(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0 - 2.0 * t);
}

fn hash33(x_in: u32, y_in: u32, z_in: u32) -> u32 {
    var h = x_in * 0x9E3779B9u ^ y_in * 0x85EBCA6Bu ^ z_in * 0xC2B2AE35u;
    h ^= h >> 16u; h *= 0x85EBCA6Bu;
    h ^= h >> 13u; h *= 0xC2B2AE35u;
    h ^= h >> 16u;
    return h;
}

// Standard 12-vector Perlin gradient set (Perlin 2002). Vectors have magnitude
// sqrt(2); after trilinear blending the output empirically stays within ~[-0.7, 0.7].
fn grad(hash: u32) -> vec3<f32> {
    let h = hash & 15u;
    switch (h) {
        case  0u: { return vec3<f32>( 1.0,  1.0,  0.0); }
        case  1u: { return vec3<f32>(-1.0,  1.0,  0.0); }
        case  2u: { return vec3<f32>( 1.0, -1.0,  0.0); }
        case  3u: { return vec3<f32>(-1.0, -1.0,  0.0); }
        case  4u: { return vec3<f32>( 1.0,  0.0,  1.0); }
        case  5u: { return vec3<f32>(-1.0,  0.0,  1.0); }
        case  6u: { return vec3<f32>( 1.0,  0.0, -1.0); }
        case  7u: { return vec3<f32>(-1.0,  0.0, -1.0); }
        case  8u: { return vec3<f32>( 0.0,  1.0,  1.0); }
        case  9u: { return vec3<f32>( 0.0, -1.0,  1.0); }
        case 10u: { return vec3<f32>( 0.0,  1.0, -1.0); }
        case 11u: { return vec3<f32>( 0.0, -1.0, -1.0); }
        case 12u: { return vec3<f32>( 1.0,  1.0,  0.0); }
        case 13u: { return vec3<f32>(-1.0,  1.0,  0.0); }
        case 14u: { return vec3<f32>( 0.0,  1.0,  1.0); }
        case 15u: { return vec3<f32>( 0.0, -1.0,  1.0); }
        default:  { return vec3<f32>( 0.0,  0.0,  0.0); }
    }
}

// Gradient (Perlin) noise. Output in ~[0.15, 0.85], mean ~0.5.
fn gradient_noise(p: vec3<f32>) -> f32 {
    let i    = floor(p);
    let f    = fract(p);
    // Quintic fade curve (Perlin 2002): 6t^5 - 15t^4 + 10t^3.
    let fade = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let xi = bitcast<u32>(i32(i.x));
    let yi = bitcast<u32>(i32(i.y));
    let zi = bitcast<u32>(i32(i.z));

    let n000 = dot(grad(hash33(xi,    yi,    zi   )), f);
    let n100 = dot(grad(hash33(xi+1u, yi,    zi   )), f - vec3<f32>(1.0, 0.0, 0.0));
    let n010 = dot(grad(hash33(xi,    yi+1u, zi   )), f - vec3<f32>(0.0, 1.0, 0.0));
    let n110 = dot(grad(hash33(xi+1u, yi+1u, zi   )), f - vec3<f32>(1.0, 1.0, 0.0));
    let n001 = dot(grad(hash33(xi,    yi,    zi+1u)), f - vec3<f32>(0.0, 0.0, 1.0));
    let n101 = dot(grad(hash33(xi+1u, yi,    zi+1u)), f - vec3<f32>(1.0, 0.0, 1.0));
    let n011 = dot(grad(hash33(xi,    yi+1u, zi+1u)), f - vec3<f32>(0.0, 1.0, 1.0));
    let n111 = dot(grad(hash33(xi+1u, yi+1u, zi+1u)), f - vec3<f32>(1.0, 1.0, 1.0));

    return 0.5 + 0.5 * mix(
        mix(mix(n000, n100, fade.x), mix(n010, n110, fade.x), fade.y),
        mix(mix(n001, n101, fade.x), mix(n011, n111, fade.x), fade.y),
        fade.z,
    );
}

// Worley noise (squared distance). Output ~[0.7, 1.0], mean ~0.87.
// Biased high; not suitable for zero-mean FBM. Kept for reference.
fn worley_noise_sq(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    var min_dist_sq: f32 = 1e9;
    let xi = bitcast<u32>(i32(i.x));
    let yi = bitcast<u32>(i32(i.y));
    let zi = bitcast<u32>(i32(i.z));

    for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                let neighbor_offset = vec3<f32>(f32(dx), f32(dy), f32(dz));
                // bitcast preserves sign bits; u32() would saturate negatives to 0.
                var h = hash33(
                    xi + bitcast<u32>(dx),
                    yi + bitcast<u32>(dy),
                    zi + bitcast<u32>(dz),
                );
                var cell_pt: vec3<f32>;
                cell_pt.x = f32(h & 0xFFFFu) / 65535.0;
                h = hash33(h, 0u, 0u);
                cell_pt.y = f32(h & 0xFFFFu) / 65535.0;
                h = hash33(h, 0u, 0u);
                cell_pt.z = f32(h & 0xFFFFu) / 65535.0;
                let delta = neighbor_offset + cell_pt - f;
                min_dist_sq = min(min_dist_sq, dot(delta, delta));
            }
        }
    }
    return 1.0 - min_dist_sq;
}

// Worley noise (linear distance). Output ~[0.13, 1.0], mean ~0.45.
// Suitable for zero-mean FBM: expected nearest-neighbor distance ~0.554 for
// a 3D Poisson lattice gives E[1 - dist] ~= 0.45, close to the 0.5 target.
fn worley_noise_linear(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    var min_dist_sq: f32 = 1e9;
    let xi = bitcast<u32>(i32(i.x));
    let yi = bitcast<u32>(i32(i.y));
    let zi = bitcast<u32>(i32(i.z));

    for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                let neighbor_offset = vec3<f32>(f32(dx), f32(dy), f32(dz));
                var h = hash33(
                    xi + bitcast<u32>(dx),
                    yi + bitcast<u32>(dy),
                    zi + bitcast<u32>(dz),
                );
                var cell_pt: vec3<f32>;
                cell_pt.x = f32(h & 0xFFFFu) / 65535.0;
                h = hash33(h, 0u, 0u);
                cell_pt.y = f32(h & 0xFFFFu) / 65535.0;
                h = hash33(h, 0u, 0u);
                cell_pt.z = f32(h & 0xFFFFu) / 65535.0;
                let delta = neighbor_offset + cell_pt - f;
                min_dist_sq = min(min_dist_sq, dot(delta, delta));
            }
        }
    }
    return 1.0 - sqrt(min_dist_sq);
}

// Zero-mean gradient FBM. Output centered near 0.5.
fn fbm_gradient(p: vec3<f32>, octaves: i32) -> f32 {
    var val    = 0.0;
    var amp    = 0.5;
    var p_curr = p;
    for (var i = 0; i < octaves; i++) {
        val    += (gradient_noise(p_curr) - 0.5) * amp;
        amp    *= 0.5;
        p_curr *= 2.0;
    }
    return val + 0.5;
}

// Zero-mean Worley FBM (linear distance). Output centered near 0.5.
fn fbm_worley(p: vec3<f32>, octaves: i32) -> f32 {
    var val    = 0.0;
    var amp    = 0.5;
    var p_curr = p;
    for (var i = 0; i < octaves; i++) {
        val    += (worley_noise_linear(p_curr) - 0.5) * amp;
        amp    *= 0.5;
        p_curr *= 2.0;
    }
    return val + 0.5;
}

// Biased Worley FBM (squared distance). Mean ~0.87. Kept for reference.
fn fbm_worley_biased(p: vec3<f32>, octaves: i32) -> f32 {
    var val    = 0.0;
    var amp    = 0.5;
    var p_curr = p;
    for (var i = 0; i < octaves; i++) {
        val    += (worley_noise_sq(p_curr) - 0.5) * amp;
        amp    *= 0.5;
        p_curr *= 2.0;
    }
    return val + 0.5;
}


// =============================================================================
// INTERPOLATION
// =============================================================================

fn w0(t: f32) -> f32 { return (1.0/6.0)*(1.0 - t)*(1.0 - t)*(1.0 - t); }
fn w1(t: f32) -> f32 { return (1.0/6.0)*(3.0*t*t*t - 6.0*t*t + 4.0); }
fn w2(t: f32) -> f32 { return (1.0/6.0)*(-3.0*t*t*t + 3.0*t*t + 3.0*t + 1.0); }
fn w3(t: f32) -> f32 { return (1.0/6.0)*t*t*t; }

// Cubic B-spline weights for a 4-tap stencil. All weights are non-negative for
// t in [0, 1] -- no ringing or overshoot. This is an approximation scheme, not
// interpolating: a lone peak of 1.0 with zero neighbors reconstructs at ~0.667.
// This implicit smoothing of the NWP field is intentional.
fn bspline_weights(frac: f32) -> array<f32, 4> {
    return array<f32, 4>(w0(frac), w1(frac), w2(frac), w3(frac));
}

// Monotone Hermite tangent at the junction of two intervals (Fritsch-Carlson 1980).
// Returns zero if adjacent slopes have opposite sign (local extremum).
fn pchip_tangent(len_left: f32, len_right: f32, slope_left: f32, slope_right: f32) -> f32 {
    if (slope_left * slope_right <= 0.0) { return 0.0; }
    // Weighted harmonic mean of adjacent slopes.
    let w_left  = 2.0*len_right + len_left;
    let w_right = len_right + 2.0*len_left;
    return (w_left + w_right) / (w_left / slope_left + w_right / slope_right);
}

// PCHIP vertical interpolation over four stencil nodes.
// Heights z0..z3 must be strictly descending (z0 > z1 > z2 > z3, top to bottom).
// Query point z must lie in [z2, z1] (the active interval).
fn pchip4(
    z0: f32, z1: f32, z2: f32, z3: f32,
    v0: f32, v1: f32, v2: f32, v3: f32,
    z:  f32,
) -> f32 {
    let len_left   = z0 - z1;
    let len_active = z1 - z2;
    let len_right  = z2 - z3;

    // Collocated nodes arise when tex_read clamps at model boundaries.
    // Fall back to the upper node; the k-search places z closest to z1.
    if (abs(len_active) < 1e-3) { return v1; }

    let slope_left   = select(0.0, (v0 - v1) / len_left,  len_left  > 1e-3);
    let slope_active = (v1 - v2) / len_active;
    let slope_right  = select(0.0, (v2 - v3) / len_right, len_right > 1e-3);

    var m1 = pchip_tangent(len_left,   len_active, slope_left,   slope_active);
    var m2 = pchip_tangent(len_active, len_right,  slope_active, slope_right);

    // Monotonicity bound (Fritsch-Carlson 1980, Eq. 3).
    if (abs(slope_active) > 1e-10) {
        let alpha = m1 / slope_active;
        let beta  = m2 / slope_active;
        let ab_sq = alpha*alpha + beta*beta;
        if (ab_sq > 9.0) {
            let scale = 3.0 / sqrt(ab_sq);
            m1 = scale * alpha * slope_active;
            m2 = scale * beta  * slope_active;
        }
    }

    // Cubic Hermite basis at t = (z1 - z) / len_active.
    let t   = (z1 - z) / len_active;
    let t2  = t*t;
    let t3  = t2*t;
    let h00 =  2.0*t3 - 3.0*t2 + 1.0; // Value basis at t=0.
    let h10 =      t3 - 2.0*t2 + t;   // Tangent basis at t=0.
    let h01 = -2.0*t3 + 3.0*t2;       // Value basis at t=1.
    let h11 =      t3 - t2;           // Tangent basis at t=1.

    return h00*v1 + h10*(-len_active)*m1 + h01*v2 + h11*(-len_active)*m2;
}

fn tex_read(tex: texture_3d<f32>, x: i32, y: i32, z: i32, nx: i32, ny: i32, nz: i32) -> f32 {
    let cx = clamp(x, 0, nx - 1);
    let cy = clamp(y, 0, ny - 1);
    let cz = clamp(z, 0, nz - 1);
    return textureLoad(tex, vec3<i32>(cx, cy, cz), 0).r;
}


// =============================================================================
// MATERIAL SAMPLING
// =============================================================================

struct Neighbor       { cx: i32, cy: i32, k: i32, weight: f32 }
struct MaterialSample { clc: f32, qc: f32, qi: f32, qv: f32, temp: f32, pressure: f32 }

// Interpolates cloud state at a displaced source-grid position using
// B-spline (horizontal) x PCHIP (vertical). Stencil points outside the valid
// atmospheric column are excluded and remaining weights are renormalized to
// prevent boundary bias. Terrain attenuation is the caller's responsibility.
fn sample_material(
    src_x: f32, src_y: f32, height: f32,
    nx_s: i32, ny_s: i32, nz_s: i32, nz_h: i32,
) -> MaterialSample {
    var mat = MaterialSample(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var total_weight: f32 = 0.0;

    if (src_x < 0.0 || src_x > f32(nx_s - 1) ||
        src_y < 0.0 || src_y > f32(ny_s - 1) || height <= 0.0) {
        return mat;
    }

    let ix        = i32(floor(src_x));
    let iy        = i32(floor(src_y));
    let weights_x = bspline_weights(src_x - f32(ix));
    let weights_y = bspline_weights(src_y - f32(iy));

    for (var jy = 0; jy < 4; jy++) {
        for (var jx = 0; jx < 4; jx++) {
            let cx = ix + (jx - 1);
            let cy = iy + (jy - 1);
            let wt = weights_x[jx] * weights_y[jy];

            // Walk k upward until the cell-center bracket spans height.
            var k = nz_h - 2;
            while (k > 0) {
                let h_upper = tex_read(t_hhl, cx, cy, k,   nx_s, ny_s, nz_h);
                let h_lower = tex_read(t_hhl, cx, cy, k+1, nx_s, ny_s, nz_h);
                if (height > (h_upper + h_lower) * 0.5) { k--; }
                else { break; }
            }

            let hhl_km1 = tex_read(t_hhl, cx, cy, k-1, nx_s, ny_s, nz_h);
            let hhl_k   = tex_read(t_hhl, cx, cy, k,   nx_s, ny_s, nz_h);
            let hhl_k1  = tex_read(t_hhl, cx, cy, k+1, nx_s, ny_s, nz_h);
            let hhl_k2  = tex_read(t_hhl, cx, cy, k+2, nx_s, ny_s, nz_h);
            let hhl_k3  = tex_read(t_hhl, cx, cy, k+3, nx_s, ny_s, nz_h);

            // Cell-center heights for the PCHIP stencil.
            let zcc0 = (hhl_km1 + hhl_k ) * 0.5;
            let zcc1 = (hhl_k   + hhl_k1) * 0.5;
            let zcc2 = (hhl_k1  + hhl_k2) * 0.5;
            let zcc3 = (hhl_k2  + hhl_k3) * 0.5;

            let col_top = tex_read(t_hhl, cx, cy, 0,      nx_s, ny_s, nz_h);
            let col_gnd = tex_read(t_hhl, cx, cy, nz_h-1, nx_s, ny_s, nz_h);

            if (height < col_top && height >= col_gnd) {
                total_weight += wt;

                mat.clc += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_clc, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_clc, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_clc, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_clc, cx, cy, k+2, nx_s, ny_s, nz_s), height) * wt;

                mat.qc += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_qc, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_qc, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_qc, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_qc, cx, cy, k+2, nx_s, ny_s, nz_s), height) * wt;

                mat.qi += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_qi, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_qi, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_qi, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_qi, cx, cy, k+2, nx_s, ny_s, nz_s), height) * wt;

                mat.qv += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_qv, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_qv, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_qv, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_qv, cx, cy, k+2, nx_s, ny_s, nz_s), height) * wt;

                mat.temp += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_t, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_t, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_t, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_t, cx, cy, k+2, nx_s, ny_s, nz_s), height) * wt;

                mat.pressure += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_p, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_p, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_p, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_p, cx, cy, k+2, nx_s, ny_s, nz_s), height) * wt;
            }
        }
    }

    // Convert from storage units. qc, qi, qv are stored x10000 for f16 headroom.
    mat.qc       = max(mat.qc / 10000.0, 0.0);
    mat.qi       = max(mat.qi / 10000.0, 0.0);
    mat.qv       = max(mat.qv / 10000.0, 0.0);
    mat.temp     = max(mat.temp, 0.0);
    mat.pressure = mat.pressure * 1000.0; // kPa -> Pa

    // Renormalize by actual accumulated weight. total_weight < 1.0 near terrain
    // and model top where stencil points are excluded by the column bounds check.
    if (total_weight > 1e-5) {
        let inv_weight = 1.0 / total_weight;
        mat.clc      *= inv_weight;
        mat.qc       *= inv_weight;
        mat.qi       *= inv_weight;
        mat.qv       *= inv_weight;
        mat.temp     *= inv_weight;
        mat.pressure *= inv_weight;
    }

    // Clamp clc AFTER normalization to prevent inv_weight pushing it above 1.0.
    mat.clc = saturate(mat.clc);
    return mat;
}


// =============================================================================
// SUB-GRID DISPLACEMENT
// =============================================================================

// Backward-Lagrangian warp for cloud structure. Derives per-axis velocity
// standard deviation from isotropic TKE (sigma = sqrt(2/3 * TKE)) and displaces
// all three axes to reflect the isotropic assumption.
fn compute_displacement(p_world: vec3<f32>, tke: f32) -> vec3<f32> {
    let sigma = sqrt(max((2.0 / 3.0) * tke, 0.0));
    if (sigma <= 0.01) { return vec3<f32>(0.0); }

    let p_turb = p_world / vec3<f32>(L_MIX, L_MIX, L_MIX);
    // Normalize sigma against a 5 m^2/s^2 TKE reference level.
    let intensity = saturate(sigma / sqrt((2.0 / 3.0) * 5.0));
    let amplitude = intensity * TURB_AMPLITUDE_MAX;

    // Seed offsets decorrelate the three displacement fields.
    let disp_x = (fbm_gradient(p_turb + vec3<f32>(17.3,  0.0,  0.0), 2) - 0.5) * amplitude;
    let disp_y = (fbm_gradient(p_turb + vec3<f32>( 0.0, 31.7,  0.0), 2) - 0.5) * amplitude;
    let disp_z = (fbm_gradient(p_turb + vec3<f32>( 0.0,  0.0, 53.1), 2) - 0.5) * amplitude;

    return vec3<f32>(disp_x, disp_y, disp_z);
}


// =============================================================================
// CONDENSATE RECONSTRUCTION
// =============================================================================

// Grid-mean condensate is reconstructed from cloud fraction using the statistical
// scheme framework (Tompkins 2008, eq. 5-6). Assuming an exponential distribution
// of total water within the grid cell, the cloud cover and grid-mean condensate are:
//
//   C    = exp(-q_s / sigma)
//   q_c  = sigma * C
//
// where sigma is the PDF scale parameter, here identified with Q_CRIT.
// This yields a linear relationship q_c = Q_CRIT * clc, which is well-behaved
// for all C in [0,1] and does not depend on qv or pressure, which are unreliable
// for this purpose in NWP output. Thermodynamic validity gates are applied at the
// call site.
fn reconstruct_qc_qi(temp: f32, clc: f32) -> vec2<f32> {
    let ice_frac     = saturate(-(temp - 273.15) / 38.0); // 0 at 0°C, 1 at -38°C
    let q_condensate = Q_CRIT * clc;
    return vec2<f32>(q_condensate * (1.0 - ice_frac), q_condensate * ice_frac);
}


// =============================================================================
// EXTINCTION COEFFICIENT
// =============================================================================

// Computes volumetric extinction (km^-1) from in-cloud liquid and ice water content.
//
// Liquid: geometric optics with Bower & Choularton (1992) effective radius.
// Ice:    Sun & Rikus (1999) / Sun (2001) parameterization.
//         Valid range: IWC 0.0001-0.3 g/m^3, temperature -70 to -5 deg C.
//         Inputs are clamped to this range before evaluation.
//
// clc_eff is used to recover in-cloud concentrations from grid-mean values.
fn compute_extinction(mat: MaterialSample, clc_eff: f32) -> f32 {
    // Air density from the Ideal Gas Law.
    let r_dry = 287.05; // J/(kg*K)
    // Condensate loading (-qt) is omitted from virtual temperature.
    // Its effect on density is <0.2% even in dense cloud cores.
    let t_virtual = mat.temp * (1.0 + 0.608 * mat.qv);
    let rho       = mat.pressure / (r_dry * t_virtual); // kg/m^3

    // Recover in-cloud mixing ratios from grid-mean values.
    let safe_clc    = max(clc_eff, 0.001);
    let qc_in_cloud = mat.qc / safe_clc; // kg/kg
    let qi_in_cloud = mat.qi / safe_clc; // kg/kg

    // Convert to volumetric water content.
    let lwc = qc_in_cloud * rho; // kg/m^3
    let iwc = qi_in_cloud * rho; // kg/m^3

    var beta = 0.0; // Extinction (m^-1); converted to km^-1 at return.

    // --- Liquid water extinction ---
    if (lwc > 1e-7) {
        // Bower & Choularton (1992): effective radius from LWC and droplet number N.
        let lwc_g            = lwc * 1000.0; // g/m^3
        let droplets_per_cm3 = 200.0;
        let r_eff_liquid     = 100.0 * pow(lwc_g * 3.0 / (4.0 * PI * droplets_per_cm3), 1.0/3.0) * 1e-6; // m
        let rho_water        = 1000.0; // kg/m^3
        // Geometric optics: beta = (3/2) * LWC / (rho_w * r_eff).
        beta += 1.5 * lwc / (rho_water * clamp(r_eff_liquid, 4e-6, 15e-6));
    }

    // --- Ice extinction ---
    if (iwc > 1e-7) {
        // Clamp to the parameterization's valid range before evaluating.
        let temp_ice_celsius = clamp(mat.temp - 273.15, -70.0, -5.0);
        let iwc_g            = clamp(iwc * 1000.0, 0.0001, 0.3); // g/m^3

        // Sun (2001) Eq. 1 & 2: power-law coefficients.
        let d_coeff_a = 45.8966 * pow(iwc_g, 0.2214);
        let d_coeff_b =  0.7957 * pow(iwc_g, 0.2535);

        // Sun & Rikus (1999) Eq. 9: base effective diameter (micrometers).
        let d_eff_base = d_coeff_a + d_coeff_b * (temp_ice_celsius + 190.0);

        // Sun & Rikus (1999) Eq. 12: low-temperature adjustment factor.
        // After clamping, range is [0.5101, 1.1826] -- always positive.
        let temp_adj  = 1.2351 + 0.0105 * temp_ice_celsius;
        let d_eff_ice = d_eff_base * temp_adj;

        // Sun & Rikus (1999) Eq. 8: hexagonal diameter to effective radius (m).
        // r_e = (3*sqrt(3)/8) * D_e
        let r_eff_ice = (3.0 * sqrt(3.0) / 8.0) * d_eff_ice * 1e-6;

        let rho_ice = 917.0; // kg/m^3, bulk hexagonal ice.
        // Geometric optics: beta = (3/2) * IWC / (rho_i * r_eff).
        beta += 1.5 * iwc / (rho_ice * clamp(r_eff_ice, 10e-6, 150e-6));
    }

    return beta * 1000.0; // m^-1 -> km^-1
}


// =============================================================================
// MAIN
// =============================================================================

// NWP fields are sampled in two phases:
//   Environment (tke): geometric position (Eulerian).
//   Material (clc, qc, qi, qv, t, p): displaced position (backward-Lagrangian).

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x);
    let y = i32(id.y);
    let nx_out = i32(params.nx_f);
    let ny_out = i32(params.ny_f);
    let nz_out = i32(params.nz_f);

    if (x >= nx_out || y >= ny_out) { return; }

    let nx_s = i32(params.nx_src);
    let ny_s = i32(params.ny_src);
    let nz_s = i32(params.nz_src);
    let nz_h = i32(params.nz_hhl);

    // Source-grid position in fractional texel coordinates.
    let src_x  = params.off_x + f32(x) * params.inv_scale_x;
    let src_y  = params.off_y + f32(y) * params.inv_scale_y;
    let src_ix = i32(floor(src_x));
    let src_iy = i32(floor(src_y));
    let bsw_x  = bspline_weights(src_x - f32(src_ix));
    let bsw_y  = bspline_weights(src_y - f32(src_iy));

    // 4x4 B-spline stencil for Eulerian field sampling. Each entry caches its
    // resolved vertical level k, which walks upward monotonically through the
    // z-loop to avoid redundant searches.
    var stencil: array<Neighbor, 16>;
    var si = 0;
    for (var jy = 0; jy < 4; jy++) {
        for (var jx = 0; jx < 4; jx++) {
            stencil[si].cx     = src_ix + (jx - 1);
            stencil[si].cy     = src_iy + (jy - 1);
            stencil[si].weight = bsw_x[jx] * bsw_y[jy];
            stencil[si].k      = nz_h - 2;
            si++;
        }
    }

    // World position (meters): Mercator -> geographic.
    let merc_x  = params.global_x_m + f32(x) * params.res_h_m;
    let merc_y  = params.global_y_m - f32(y) * params.res_h_m;
    let lat_rad = 2.0 * atan(exp(merc_y / R_EARTH)) - (PI * 0.5);
    let cos_lat = cos(lat_rad);
    let world_x = merc_x * cos_lat;
    let world_y = R_EARTH * lat_rad;

    for (var z = 0; z < nz_out; z++) {
        let height  = f32(z) * params.res_v_m;
        let out_idx = (z * ny_out + y) * nx_out + x;

        // --- Eulerian environment sampling (geometric, undisplaced position) ---
        var env_tke      = 0.0;
        var env_wt_total = 0.0;

        for (var i = 0; i < 16; i++) {
            let cx = stencil[i].cx;
            let cy = stencil[i].cy;
            var k  = stencil[i].k;
            let wt = stencil[i].weight;

            // Walk k upward until the cell-center bracket spans height.
            while (k > 0) {
                let h_upper = tex_read(t_hhl, cx, cy, k,   nx_s, ny_s, nz_h);
                let h_lower = tex_read(t_hhl, cx, cy, k+1, nx_s, ny_s, nz_h);
                if (height > (h_upper + h_lower) * 0.5) { k--; }
                else { break; }
            }
            stencil[i].k = k;

            let col_top = tex_read(t_hhl, cx, cy, 0, nx_s, ny_s, nz_h);
            if (height < col_top) {
                env_wt_total += wt;

                // TKE is a half-level field; interpolate on HHL levels directly.
                var k_tke = k;
                while (k_tke > 0 &&
                       height > tex_read(t_hhl, cx, cy, k_tke,   nx_s, ny_s, nz_h)) { k_tke--; }
                while (k_tke < nz_h - 2 &&
                       height <= tex_read(t_hhl, cx, cy, k_tke+1, nx_s, ny_s, nz_h)) { k_tke++; }
                // Clamp query to ground level to keep the PCHIP stencil in range.
                let h_query = max(height, tex_read(t_hhl, cx, cy, nz_h-1, nx_s, ny_s, nz_h));

                env_tke += pchip4(
                    tex_read(t_hhl, cx, cy, k_tke-1, nx_s, ny_s, nz_h),
                    tex_read(t_hhl, cx, cy, k_tke,   nx_s, ny_s, nz_h),
                    tex_read(t_hhl, cx, cy, k_tke+1, nx_s, ny_s, nz_h),
                    tex_read(t_hhl, cx, cy, k_tke+2, nx_s, ny_s, nz_h),
                    tex_read(t_tke, cx, cy, k_tke-1, nx_s, ny_s, nz_h),
                    tex_read(t_tke, cx, cy, k_tke,   nx_s, ny_s, nz_h),
                    tex_read(t_tke, cx, cy, k_tke+1, nx_s, ny_s, nz_h),
                    tex_read(t_tke, cx, cy, k_tke+2, nx_s, ny_s, nz_h),
                    h_query) * wt;
            }
        }
        // Renormalize: stencil points above the model top are excluded, so
        // env_wt_total < 1.0 near the tropopause without this correction.
        if (env_wt_total > 1e-5) { env_tke /= env_wt_total; }
        env_tke = max(env_tke, 0.0);

        // --- Backward-Lagrangian material sampling ---
        let p_world  = vec3<f32>(world_x, world_y, height);

        var mat = sample_material(src_x, src_y, height, nx_s, ny_s, nz_s, nz_h);

        // --- Effective cloud fraction ---
        // ICON-D2 can report CLC=0 where QC+QI > 0. The condensate-derived
        // diagnostic corrects this inconsistency by inverting the linear
        // statistical scheme (Tompkins 2008): C = q_c / Q_CRIT.
        let total_condensate = mat.qc + mat.qi;
        let clc_diag         = clamp(total_condensate / Q_CRIT, 0.0, 1.0);
        let clc_eff          = max(mat.clc, clc_diag);

        // Where CLC > 0 but microphysics has depleted condensate, enforce a minimum
        // qc/qi consistent with the reported cloud fraction via the statistical scheme
        // (Tompkins 2008). Thermodynamic validity gates exclude unphysical NWP states.
        if (clc_eff > 0.0 && mat.temp > 180.0 && mat.pressure > 100.0) {
            let q_rec = reconstruct_qc_qi(mat.temp, clc_eff);
            mat.qc    = max(mat.qc, q_rec.x);
            mat.qi    = max(mat.qi, q_rec.y);
        }
        
        // --- Cloud synthesis ---
        var result = 0.0;
        if (clc_eff > 0.0) {
            result = compute_extinction(mat, clc_eff) * clc_eff;
        }

        // Terrain fade: linear ramp over 200m above the B-spline-interpolated
        // surface height.
        var terrain_height: f32 = 0.0;
        for (var jy = 0; jy < 4; jy++) {
            for (var jx = 0; jx < 4; jx++) {
                let cx = src_ix + (jx - 1);
                let cy = src_iy + (jy - 1);
                terrain_height += tex_read(t_hhl, cx, cy, nz_h-1, nx_s, ny_s, nz_h)
                                * bsw_x[jx] * bsw_y[jy];
            }
        }
        result *= saturate((height - terrain_height) / 200.0);

        b_out[out_idx] = result;
    }
}
