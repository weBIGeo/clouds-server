// --- CONSTANTS ---
const PI: f32 = 3.14159265359;
const R_EARTH: f32 = 6378137.0; 

// Kelvin-Helmholtz Instability (Unit is kind of arbitrary)
const KH_SENSITIVITY: f32 = 100.0;
// meters. Max turbulent warp displacement.
const TURB_AMPLITUDE_MAX: f32 = 150.0;
// Turbulent sub-grid mixing length in meters (based on ICON-D2 grid scale)
const L_MIX: f32 = 600.0;

// Noise Scales
const SCALE_STRAT_XY: f32 = 2000.0;
const SCALE_STRAT_Z: f32 = 1000.0;
const SCALE_DETAIL_XY: f32 = 400.0;
const SCALE_DETAIL_Z: f32 = 300.0;
const SCALE_PERTURB_XY: f32 = 3000.0;
const SCALE_PERTURB_Z: f32 = 2000.0;
const CUM_SCALE_XY_1: f32 = 600.0;
const CUM_SCALE_XY_2: f32 = 1400.0;
const CUM_SCALE_XY_3: f32 = 3200.0;
const CUM_SCALE_Z_1: f32 = 400.0;
const CUM_SCALE_Z_2: f32 = 1400.0;
const CUM_SCALE_Z_3: f32 = 4000.0;

// --- UNIFORMS ---
struct Params {
    nx_src: f32, ny_src: f32, nz_src: f32, nz_hhl: f32,
    global_x_m: f32, global_y_m: f32, res_h_m: f32, res_v_m: f32,
    off_x: f32, off_y: f32, inv_scale_x: f32, inv_scale_y: f32,
    nx_f: f32, ny_f: f32, nz_f: f32, unused: f32,
};

@group(0) @binding( 0) var t_clc : texture_3d<f32>;
@group(0) @binding( 1) var t_qc  : texture_3d<f32>;
@group(0) @binding( 2) var t_qi  : texture_3d<f32>;
@group(0) @binding( 3) var t_qv  : texture_3d<f32>;
@group(0) @binding( 4) var t_t   : texture_3d<f32>;
@group(0) @binding( 5) var t_p   : texture_3d<f32>;
@group(0) @binding( 6) var t_u   : texture_3d<f32>;
@group(0) @binding( 7) var t_v   : texture_3d<f32>;
@group(0) @binding( 8) var t_w   : texture_3d<f32>;
@group(0) @binding( 9) var t_tke : texture_3d<f32>;
@group(0) @binding(10) var t_hhl : texture_3d<f32>;

@group(1) @binding(0) var<uniform> params : Params;
@group(2) @binding(0) var<storage, read_write> b_out : array<f32>;

// --- NOISE LIB (Ported) ---
fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn lerp(a: f32, b: f32, t: f32) -> f32 { return a + t * (b - a); }
fn smoothstep_custom(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0 - 2.0 * t);
}

fn hash33(x_in: u32, y_in: u32, z_in: u32) -> u32 {
    var x = x_in; var y = y_in; var z = z_in;
    let q: u32 = 1103515245u;
    x = x * q + 12345u;
    y = y * q + 12345u;
    z = z * q + 12345u;
    x ^= x >> 16u; x *= 0x85EBCA6Bu;
    x ^= x >> 13u; x *= 0xC2B2AE35u;
    x ^= x >> 16u;
    return x + y * 0x1B873593u + z;
}

fn grad(hash: u32) -> vec3<f32> {
    let h = hash & 15u;
    switch(h) {
        case 0u: { return vec3<f32>(1.0, 1.0, 0.0); }
        case 1u: { return vec3<f32>(-1.0, 1.0, 0.0); }
        case 2u: { return vec3<f32>(1.0, -1.0, 0.0); }
        case 3u: { return vec3<f32>(-1.0, -1.0, 0.0); }
        case 4u: { return vec3<f32>(1.0, 0.0, 1.0); }
        case 5u: { return vec3<f32>(-1.0, 0.0, 1.0); }
        case 6u: { return vec3<f32>(1.0, 0.0, -1.0); }
        case 7u: { return vec3<f32>(-1.0, 0.0, -1.0); }
        case 8u: { return vec3<f32>(0.0, 1.0, 1.0); }
        case 9u: { return vec3<f32>(0.0, -1.0, 1.0); }
        case 10u: { return vec3<f32>(0.0, 1.0, -1.0); }
        case 11u: { return vec3<f32>(0.0, -1.0, -1.0); }
        case 12u: { return vec3<f32>(1.0, 1.0, 0.0); }
        case 13u: { return vec3<f32>(-1.0, 1.0, 0.0); }
        case 14u: { return vec3<f32>(0.0, 1.0, 1.0); }
        case 15u: { return vec3<f32>(0.0, -1.0, 1.0); }
        default: { return vec3<f32>(0.0); }
    }
}

fn gradient_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    
    let x = u32(i.x); let y = u32(i.y); let z = u32(i.z);
    
    // Gradients
    let n000 = dot(grad(hash33(x, y, z)), f);
    let n100 = dot(grad(hash33(x+1u, y, z)), f - vec3<f32>(1.0,0.0,0.0));
    let n010 = dot(grad(hash33(x, y+1u, z)), f - vec3<f32>(0.0,1.0,0.0));
    let n110 = dot(grad(hash33(x+1u, y+1u, z)), f - vec3<f32>(1.0,1.0,0.0));
    let n001 = dot(grad(hash33(x, y, z+1u)), f - vec3<f32>(0.0,0.0,1.0));
    let n101 = dot(grad(hash33(x+1u, y, z+1u)), f - vec3<f32>(1.0,0.0,1.0));
    let n011 = dot(grad(hash33(x, y+1u, z+1u)), f - vec3<f32>(0.0,1.0,1.0));
    let n111 = dot(grad(hash33(x+1u, y+1u, z+1u)), f - vec3<f32>(1.0,1.0,1.0));
    
    return 0.5 + 0.5 * mix(
        mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
        u.z
    );
}

fn worley_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    var min_dist_sq: f32 = 1.0;
    let xi = u32(i.x); let yi = u32(i.y); let zi = u32(i.z);
    
    for(var z: i32 = -1; z <= 1; z++) {
        for(var y: i32 = -1; y <= 1; y++) {
            for(var x: i32 = -1; x <= 1; x++) {
                let neighbor = vec3<f32>(f32(x), f32(y), f32(z));
                var h = hash33(xi + u32(x), yi + u32(y), zi + u32(z));
                var p_rng: vec3<f32>;
                p_rng.x = f32(h & 0xFFFFu) / 65535.0;
                h = hash33(h, 0u, 0u);
                p_rng.y = f32(h & 0xFFFFu) / 65535.0;
                h = hash33(h, 0u, 0u);
                p_rng.z = f32(h & 0xFFFFu) / 65535.0;
                
                let diff = neighbor + p_rng - f;
                let d_sq = dot(diff, diff);
                min_dist_sq = min(min_dist_sq, d_sq);
            }
        }
    }
    return 1.0 - min_dist_sq;
}

fn fbm_gradient(p: vec3<f32>, octaves: i32) -> f32 {
    var val = 0.0; var amp = 0.5; var norm = 0.0;
    var p_curr = p;
    for(var i=0; i<octaves; i++) {
        val += gradient_noise(p_curr) * amp;
        norm += amp; amp *= 0.5; p_curr *= 2.0;
    }
    return val / norm;
}

fn fbm_worley(p: vec3<f32>, octaves: i32) -> f32 {
    var val = 0.0; var amp = 0.5; var norm = 0.0;
    var p_curr = p;
    for(var i=0; i<octaves; i++) {
        val += worley_noise(p_curr) * amp;
        norm += amp; amp *= 0.5; p_curr *= 2.0;
    }
    return val / norm;
}

// --- INTERPOLATION ---
// Horizontal B-spline weights (uniform spacing, unchanged).
fn w0(t: f32) -> f32 { return (1.0/6.0)*(1.0-t)*(1.0-t)*(1.0-t); }
fn w1(t: f32) -> f32 { return (1.0/6.0)*(3.0*t*t*t - 6.0*t*t + 4.0); }
fn w2(t: f32) -> f32 { return (1.0/6.0)*(-3.0*t*t*t + 3.0*t*t + 3.0*t + 1.0); }
fn w3(t: f32) -> f32 { return (1.0/6.0)*t*t*t; }

// --- PCHIP vertical interpolation (Fritsch-Carlson, 1980) ---
// Full two-step Fritsch-Carlson monotone cubic Hermite.
// The output is strictly bounded by the two bracketing sample values.
// No ringing, no overshoot, values never leave the input data range.
//
// pchip_tangent: weighted harmonic mean of adjacent slopes.
// h0 = interval width ABOVE the knot, h1 = interval width BELOW.
// s0, s1 = dv/dz (upward) in those two intervals.
// Returns zero whenever the slopes disagree in sign (monotone limiter).
fn pchip_tangent(h0: f32, h1: f32, s0: f32, s1: f32) -> f32 {
    if (s0 * s1 <= 0.0) { return 0.0; }
    let w0t = 2.0*h1 + h0;
    let w1t = h1 + 2.0*h0;
    return (w0t + w1t) / (w0t / s0 + w1t / s1);
}

// pchip4: interpolate at physical height z, given four knot heights
// (z0 > z1 > z2 > z3, descending as in HHL) and their values.
// z must lie in [z2, z1]; that is always true after the HHL search.
//
// At domain boundaries the outer stencil knots (z0 or z3) may be
// clamped duplicates of their inner neighbour, collapsing that interval
// to zero width. Guard against division by zero: when an outer interval
// is degenerate its slope is set to 0, which makes the corresponding
// tangent 0 via pchip_tangent, a safe natural-boundary fallback.
fn pchip4(z0: f32, z1: f32, z2: f32, z3: f32,
          v0: f32, v1: f32, v2: f32, v3: f32,
          z: f32) -> f32 {
    // Positive interval widths (heights are descending with index).
    let h01 = z0 - z1;
    let h12 = z1 - z2;
    let h23 = z2 - z3;
    // Slopes as dv/dz (upward). Guard outer intervals against zero width
    // (occurs when the stencil is clamped at the top or bottom boundary).
    let s01 = select(0.0, (v0 - v1) / h01, h01 > 1e-3);
    let s12 = (v1 - v2) / h12;   // inner interval, always valid after HHL search
    let s23 = select(0.0, (v2 - v3) / h23, h23 > 1e-3);
    // PCHIP tangents (dv/dz) at the two interior knots, step 1.
    var m1 = pchip_tangent(h01, h12, s01, s12);
    var m2 = pchip_tangent(h12, h23, s12, s23);

    // Fritsch-Carlson step 2: rescale tangents if α²+β² > 9, where
    // α = m1/s12, β = m2/s12. This is the condition that strictly
    // guarantees the cubic stays within [v1, v2] on the interval,
    // preventing any overshoot even on steep monotone sections.
    if (abs(s12) > 1e-10) {
        let alpha = m1 / s12;
        let beta  = m2 / s12;
        let ab_sq = alpha * alpha + beta * beta;
        if (ab_sq > 9.0) {
            let tau = 3.0 / sqrt(ab_sq);
            m1 = tau * alpha * s12;
            m2 = tau * beta  * s12;
        }
    }

    // Local parameter: t=0 at z1 (top knot), t=1 at z2 (bottom knot).
    let t  = (z1 - z) / h12;
    let t2 = t*t; let t3 = t2*t;
    let h00 =  2.0*t3 - 3.0*t2 + 1.0;
    let h10 =      t3 - 2.0*t2 + t;
    let h01h = -2.0*t3 + 3.0*t2;
    let h11 =      t3 -      t2;
    // Tangents are dv/dt = dv/dz * dz/dt = m * (-h12) because z decreases as t increases.
    return h00*v1 + h10*(-h12)*m1 + h01h*v2 + h11*(-h12)*m2;
}

// Helper to access 3D texture with clamping
fn tex_read(tex: texture_3d<f32>, x: i32, y: i32, z: i32, nx: i32, ny: i32, nz: i32) -> f32 {
    let cx = clamp(x, 0, nx - 1);
    let cy = clamp(y, 0, ny - 1);
    let cz = clamp(z, 0, nz - 1);
    return textureLoad(tex, vec3<i32>(cx, cy, cz), 0).r;
}

struct Neighbor { cx: i32, cy: i32, k: i32, w_horiz: f32, };

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x);
    let y = i32(id.y);
    let nx_t = i32(params.nx_f);
    let ny_t = i32(params.ny_f);
    let nz_t = i32(params.nz_f);

    if (x >= nx_t || y >= ny_t) { return; }

    // Source Dims
    let nx_s = i32(params.nx_src);
    let ny_s = i32(params.ny_src);
    let nz_s = i32(params.nz_src);
    let nz_h = i32(params.nz_hhl);

    // 1. Interpolation Setup
    let src_x_f = params.off_x + f32(x) * params.inv_scale_x;
    let src_y_f = params.off_y + f32(y) * params.inv_scale_y;
    let sx_int = i32(floor(src_x_f));
    let sy_int = i32(floor(src_y_f));
    let fx = src_x_f - f32(sx_int);
    let fy = src_y_f - f32(sy_int);

    var wx: array<f32, 4>; var wy: array<f32, 4>;
    wx[0] = w0(fx); wx[1] = w1(fx); wx[2] = w2(fx); wx[3] = w3(fx);
    wy[0] = w0(fy); wy[1] = w1(fy); wy[2] = w2(fy); wy[3] = w3(fy);

    // 2. Neighbor State
    var n: array<Neighbor, 16>;
    var n_idx = 0;
    for(var iy = 0; iy < 4; iy++) {
        for(var ix = 0; ix < 4; ix++) {
            n[n_idx].cx = sx_int + (ix - 1);
            n[n_idx].cy = sy_int + (iy - 1);
            n[n_idx].w_horiz = wx[ix] * wy[iy];
            n[n_idx].k = nz_h - 2; 
            n_idx++;
        }
    }

    // 1. Calculate raw Web Mercator coordinates (Meters)
    let raw_x_merc = params.global_x_m + f32(x) * params.res_h_m;
    let raw_y_merc = params.global_y_m - f32(y) * params.res_h_m;

    // 2. Inverse Web Mercator: Derive Latitude from Y (in Radians)
    // lat = 2 * atan(exp(y / R)) - pi/2
    let lat_rad = 2.0 * atan(exp(raw_y_merc / R_EARTH)) - (PI * 0.5);

    // 3. Calculate local scale factor (cos(lat))
    // This shrinks the Mercator coordinates back to Physical Earth Meters
    let local_scale = cos(lat_rad);

    // 4. Apply scale to get Physical World position for Noise Lookups
    let col_x_world = raw_x_merc * local_scale;
    let col_y_world = raw_y_merc * local_scale;

    var u_prev: f32 = 0.0;
    var v_prev: f32 = 0.0;
    var h_prev: f32 = 0.0;

    for (var z = 0; z < nz_t; z++) {
        let h_m = f32(z) * params.res_v_m;
        var val_clc=0.0; var val_qc=0.0; var val_qi=0.0; var val_qv=0.0;
        var val_t=0.0; var val_p=0.0;
        var val_u=0.0; var val_v=0.0; var val_w=0.0; var val_tke=0.0;

        for(var i=0; i<16; i++) {
            let cx = n[i].cx;
            let cy = n[i].cy;
            var k = n[i].k;

            // HHL Search
            while (k > 0) {
                let h_u = tex_read(t_hhl, cx, cy, k, nx_s, ny_s, nz_h);
                let h_l = tex_read(t_hhl, cx, cy, k+1, nx_s, ny_s, nz_h);
                if (h_m > (h_u + h_l) * 0.5) { k--; } 
                else { break; }
            }
            n[i].k = k;

            // Store individual boundary heights for reuse in both cell-center and W interpolation.
            let hhl_km1 = tex_read(t_hhl, cx, cy, k-1, nx_s, ny_s, nz_h);
            let hhl_k  = tex_read(t_hhl, cx, cy, k,   nx_s, ny_s, nz_h);
            let hhl_k1 = tex_read(t_hhl, cx, cy, k+1, nx_s, ny_s, nz_h);
            let hhl_k2 = tex_read(t_hhl, cx, cy, k+2, nx_s, ny_s, nz_h);
            let hhl_k3 = tex_read(t_hhl, cx, cy, k+3, nx_s, ny_s, nz_h);

            // Cell-centre heights (midpoints of adjacent HHL boundaries).
            let zcc0 = (hhl_km1 + hhl_k ) * 0.5;  // centre of cell k-1
            let zcc1 = (hhl_k   + hhl_k1) * 0.5;  // centre of cell k   (= h1)
            let zcc2 = (hhl_k1  + hhl_k2) * 0.5;  // centre of cell k+1 (= h2)
            let zcc3 = (hhl_k2  + hhl_k3) * 0.5;  // centre of cell k+2

            let w_tot = n[i].w_horiz;

            let z_top = tex_read(t_hhl, cx, cy, 0,      nx_s, ny_s, nz_h);
            let z_gnd = tex_read(t_hhl, cx, cy, nz_h-1, nx_s, ny_s, nz_h);

            if (h_m < z_top && h_m > z_gnd) {
                // Cell-centred variables: stencil k-1..k+2, heights zcc0..zcc3.
                val_clc += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_clc, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_clc, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_clc, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_clc, cx, cy, k+2, nx_s, ny_s, nz_s), h_m) * w_tot;

                val_qc  += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_qc, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_qc, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_qc, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_qc, cx, cy, k+2, nx_s, ny_s, nz_s), h_m) * w_tot;

                val_qi  += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_qi, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_qi, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_qi, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_qi, cx, cy, k+2, nx_s, ny_s, nz_s), h_m) * w_tot;

                val_qv  += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_qv, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_qv, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_qv, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_qv, cx, cy, k+2, nx_s, ny_s, nz_s), h_m) * w_tot;

                val_t   += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_t, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_t, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_t, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_t, cx, cy, k+2, nx_s, ny_s, nz_s), h_m) * w_tot;

                val_p   += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_p, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_p, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_p, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_p, cx, cy, k+2, nx_s, ny_s, nz_s), h_m) * w_tot;

                val_u   += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_u, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_u, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_u, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_u, cx, cy, k+2, nx_s, ny_s, nz_s), h_m) * w_tot;

                val_v   += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_v, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_v, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_v, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_v, cx, cy, k+2, nx_s, ny_s, nz_s), h_m) * w_tot;

                // W is defined at HHL boundary levels (nz_h levels), not cell centres.
                // The cell-centred HHL search guarantees h_m ∈ (zcc2, zcc1], but zcc2 < hhl_k1,
                // so h_m may fall in either [hhl_k1, hhl_k] or [hhl_k2, hhl_k1].
                // Select the correct boundary interval so t stays in [0,1].
                let w_km1 = tex_read(t_w, cx, cy, k-1, nx_s, ny_s, nz_h);
                let w_k   = tex_read(t_w, cx, cy, k,   nx_s, ny_s, nz_h);
                let w_k1  = tex_read(t_w, cx, cy, k+1, nx_s, ny_s, nz_h);
                let w_k2  = tex_read(t_w, cx, cy, k+2, nx_s, ny_s, nz_h);
                if (h_m >= hhl_k1) {
                    // when h_m in [hhl_k1, hhl_k] then stencil (k-1, k, k+1, k+2), interpolate on [k, k+1]
                    val_w += pchip4(hhl_km1, hhl_k, hhl_k1, hhl_k2,
                                    w_km1, w_k, w_k1, w_k2, h_m) * w_tot;
                } else {
                    // when h_m in [hhl_k2, hhl_k1] then stencil (k, k+1, k+2, k+3), interpolate on [k+1, k+2]
                    let w_k3 = tex_read(t_w, cx, cy, k+3, nx_s, ny_s, nz_h);
                    val_w += pchip4(hhl_k, hhl_k1, hhl_k2, hhl_k3,
                                    w_k, w_k1, w_k2, w_k3, h_m) * w_tot;
                }

                val_tke += pchip4(zcc0, zcc1, zcc2, zcc3,
                    tex_read(t_tke, cx, cy, k-1, nx_s, ny_s, nz_s),
                    tex_read(t_tke, cx, cy, k,   nx_s, ny_s, nz_s),
                    tex_read(t_tke, cx, cy, k+1, nx_s, ny_s, nz_s),
                    tex_read(t_tke, cx, cy, k+2, nx_s, ny_s, nz_s), h_m) * w_tot;
            }
        }

        // Physics
        let clc = saturate(val_clc);
        let qc  = max(val_qc, 0.0);
        let qi  = max(val_qi, 0.0);
        let qv  = max(val_qv, 0.0);
        let t   = max(val_t, 0.0);
        let p   = val_p * 1000.0; // Convert kPa to Pa
        let u   = val_u;
        let v   = val_v;
        let w   = val_w;
        let tke = max(val_tke, 0.0);
        
        var output_val: f32 = 0.0;

        if (clc > 0.0) {
            var p_tex = vec3<f32>(col_x_world, col_y_world, h_m);

            // --- Kelvin-Helmholtz Billowing ---
            // Backward-difference vertical wind shear (du/dz, dv/dz).
            // Use the actual height difference to the previous sampled level,
            // which may span multiple model levels if intermediate ones were cloud-free.

            // Undefined in first loop iteration
            var du_dz = 0.0;
            var dv_dz = 0.0;
            if (z > 0) {
                 let dz = max(h_m - h_prev, 1.0); // clamp to 1m to avoid division by zero 
                du_dz = (u - u_prev) / dz;
                dv_dz = (v - v_prev) / dz;
            }
            let shear_mag = sqrt(du_dz * du_dz + dv_dz * dv_dz);

            if (shear_mag > 1e-4) {
                // Richardson number proxy.
                // True Ri = N²/S² where N² requires d(theta)/dz which we lack.
                // TKE acts as a stability inverse: high TKE implies low static stability.
                // ri_proxy < 0.25 triggers instability, matching the classical Miles-Howard criterion.
                let ri_proxy = 1.0 / (1.0 + tke * shear_mag * KH_SENSITIVITY);
                let kh_strength = saturate((0.25 - ri_proxy) / 0.25);

                let inv_shear = 1.0 / shear_mag;
                let shear_nx = du_dz * inv_shear;  // Unit vector along shear direction
                let shear_ny = dv_dz * inv_shear;

                // l_turb = δu / S where δu = sqrt(2·TKE) is the turbulent velocity scale
                // and S is the shear magnitude. Dimensionally a length (m), but note this
                // is NOT the shear layer depth D, it is a turbulent length scale that
                // serves as a proxy for the dominant spatial scale of KH instability.
                //
                // Wavelength from Michalke (1964) linear stability of a tanh shear profile:
                // most unstable wavenumber k·h = 0.4446, h = D/2 → λ = 2πh/0.4446 = 7.07·D
                // Substituting l_turb for D gives λ = 7.07·l_turb.
                //
                // Clamp bounds keep λ within the ranges observed in radar/lidar literature:
                //   lower 50m  → λ_min ≈ 350m  (Boundary layer KHBs)
                //   upper 500m → λ_max ≈ 3535m (Upper tropospheric KHBs, e.g., Browning 1971)
                //
                // Amplitude from Thorpe (1973): wave steepness s = a/λ ranges 0.10-0.50
                // for atmospheric KH billows. We use s = 0.30 (mid-range), scaled by
                // instability strength → a = kh_strength * 0.30 * λ
                let delta_u       = sqrt(2.0 * tke);
                let l_turb        = clamp(delta_u / shear_mag, 50.0, 500.0);
                let kh_wavelength = 7.07 * l_turb;
                let kh_amplitude  = kh_strength * 0.30 * kh_wavelength;

                // Phase varies along shear direction (billows propagate with the flow)
                let wave_phase = (p_tex.x * shear_nx + p_tex.y * shear_ny) / kh_wavelength;

                // Horizontal compression/expansion (along shear)
                let kh_warp_h = sin(wave_phase * 2.0 * PI) * kh_amplitude;
                // Vertical overturning displacement (90 degrees out of phase)
                let kh_warp_v = cos(wave_phase * 2.0 * PI) * kh_amplitude; 

                p_tex.x += kh_warp_h * shear_nx;
                p_tex.y += kh_warp_h * shear_ny;
                p_tex.z += kh_warp_v;
            }


            // --- Sub-Grid Turbulent Displacement ---
            // TKE = 0.5*(σ_u² + σ_v² + σ_w²), so σ_h ≈ sqrt(2/3 * TKE) assuming isotropy.
            // The characteristic Lagrangian displacement scale is σ_h * T_L ≈ L_MIX
            // (the TKE and T_L cancel, leaving just the mixing length, see Rodean 1996).
            let sigma_h = sqrt(max((2.0 / 3.0) * tke, 0.0));
            if (sigma_h > 0.01) {
                let turb_p = p_tex / vec3<f32>(L_MIX, L_MIX, L_MIX);
                // sigma_h / L_MIX gives a 0..1 intensity based on how energetic the turbulence is
                // relative to what the mixing length predicts
                let turb_intensity = saturate(sigma_h / sqrt((2.0 / 3.0) * 5.0)); // normalize to ~max expected TKE
                let turb_amplitude = turb_intensity * TURB_AMPLITUDE_MAX;

                // Two decorrelated fbm samples give a pseudo-curl (divergence-free) warp
                let turb_wx = (fbm_gradient(turb_p + vec3<f32>(17.3, 0.0, 0.0), 2) - 0.5) * turb_amplitude;
                let turb_wy = (fbm_gradient(turb_p + vec3<f32>(0.0, 31.7, 0.0), 2) - 0.5) * turb_amplitude;
                p_tex.x += turb_wx;
                p_tex.y += turb_wy;
            }
            
            let mix_stratus = smoothstep_custom(0.0, 2.0, w);
            
            let p_strat = p_tex / vec3<f32>(SCALE_STRAT_XY, SCALE_STRAT_XY, SCALE_STRAT_Z);
            let n_strat = fbm_gradient(p_strat, 2) * 0.5 + 0.5;

            let scale_mod = smoothstep_custom(0.0, 0.8, clc);
            let target_xy = lerp(CUM_SCALE_XY_1, CUM_SCALE_XY_3, scale_mod);
            var n_cum: f32 = 0.0;
            if (target_xy < CUM_SCALE_XY_2) {
                let t = (target_xy - CUM_SCALE_XY_1) / (CUM_SCALE_XY_2 - CUM_SCALE_XY_1);
                let n1 = fbm_worley(p_tex / vec3<f32>(CUM_SCALE_XY_1, CUM_SCALE_XY_1, CUM_SCALE_Z_1), 2);
                let n2 = fbm_worley(p_tex / vec3<f32>(CUM_SCALE_XY_2, CUM_SCALE_XY_2, CUM_SCALE_Z_2), 2);
                n_cum = lerp(n1, n2, t);
            } else {
                let t = (target_xy - CUM_SCALE_XY_2) / (CUM_SCALE_XY_3 - CUM_SCALE_XY_2);
                let n2 = fbm_worley(p_tex / vec3<f32>(CUM_SCALE_XY_2, CUM_SCALE_XY_2, CUM_SCALE_Z_2), 2);
                let n3 = fbm_worley(p_tex / vec3<f32>(CUM_SCALE_XY_3, CUM_SCALE_XY_3, CUM_SCALE_Z_3), 2);
                n_cum = lerp(n2, n3, t);
            }
            
            var base_shape = lerp(n_strat, n_cum, mix_stratus);
            let erosion = saturate(tke * 0.2);
            var thresh_offset: f32 = 0.0;
            if (erosion > 0.01) {
                let n_det = fbm_worley(p_tex / vec3<f32>(SCALE_DETAIL_XY, SCALE_DETAIL_XY, SCALE_DETAIL_Z), 3);
                base_shape -= n_det * erosion;
                thresh_offset = erosion * 0.5;
            }
            let n_pert = fbm_worley(p_tex / vec3<f32>(SCALE_PERTURB_XY, SCALE_PERTURB_XY, SCALE_PERTURB_Z), 2);
            let pert_mask = smoothstep_custom(0.0, 0.2, clc);
            let thresh_shift = (0.35 - n_pert) * 0.2 * pert_mask;
            
            let base_thresh = 1.0 - clc;
            let final_thresh = base_thresh - thresh_offset + thresh_shift;
            let softness = max(clc, 0.15);
            let dens_factor = saturate((base_shape - final_thresh) / softness);
            
            if (dens_factor > 0.0) {
                // Air density from temperature and pressure (Ideal Gas Law)
                let r_dry = 287.05; // J/(kg*K)
                let t_virtual = t * (1.0 + 0.608 * qv); // Virtual temperature accounts for moisture
                let rho = p / (r_dry * t_virtual);

                let safe_clc = max(clc, 0.001);

                // Grid-mean values divided by cloud fraction
                let qc_in_cloud = qc / safe_clc; // Cloud liquid water mixing ratio (kg/kg)
                let qi_in_cloud = qi / safe_clc; // Cloud ice mixing ratio (kg/kg)

                // Convert to kg/m^3 (water content)
                let lwc = qc_in_cloud * rho;
                let iwc = qi_in_cloud * rho;

                var beta = 0.0;
                if (lwc > 1e-7) {
                    // Effective radius (Bower & Choularton 1992)
                    let lwc_g = lwc * 1000;
                    let n_cm3 = 200.0; // Continental cloud droplet number concentration
                    let r_eff_liquid = 100.0 * pow(lwc_g * 3.0 / (4.0 * PI * n_cm3), 1.0/3.0) * 1e-6;
                    // β = (3/2) * (LWC / r_eff)
                    beta += 1.5 * lwc / clamp(r_eff_liquid, 4e-6, 15e-6);
                }

                if (iwc > 1e-7) {
                    let t_celsius = t - 273.15;
    
                    // IWC must be in g/m^3 for the Sun (2001) equations
                    let iwc_g = iwc * 1000.0; 
                    
                    // Sun (2001) Equations 1 & 2: Power-law revision (Page 269)
                    let a_term = 45.8966 * pow(iwc_g, 0.2214);
                    let b_term = 0.7957 * pow(iwc_g, 0.2535);
                    
                    // Sun & Rikus (1999) Equation 9: Base Effective Diameter (micrometers)
                    var d_eff_ice = a_term + b_term * (t_celsius + 190.0);
                    
                    // Sun & Rikus (1999) Equation 12: Low-temperature adjustment factor
                    let f_adj = 1.2351 + 0.0105 * t_celsius;
                    d_eff_ice *= f_adj;
                    
                    // Sun & Rikus (1999) Equation 8: Convert Hexagonal Diameter to Effective Radius (m)
                    // r_e = (3 * sqrt(3) / 8) * D_e
                    let r_eff_ice = (3.0 * sqrt(3.0) / 8.0) * d_eff_ice * 1e-6;
                    
                    // Extinction coefficient β = (3/2) * (IWC / (rho_w * r_eff))
                    beta += 1.5 * iwc / clamp(r_eff_ice, 10e-6, 150e-6);
                }
                
                output_val = beta * dens_factor;
            }
        }

        u_prev = u;
        v_prev = v;
        h_prev = h_m;
        
        let out_idx = (z * ny_t + y) * nx_t + x;
        b_out[out_idx] = output_val;
    }
}
