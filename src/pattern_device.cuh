// pattern_device.cuh — the procedural pattern VM, on-device (shared by both CUDA backends).
//
// This is the device twin of pattern.h's `patternEval`: the same postfix scalar-stack
// interpreter over the same POD `PatNode` programs, so a `pattern` in a scene evaluates
// bit-for-bit identically whether it is driven by the path tracer (render_cuda.cu) or by
// the preview rasterizer (raster_cuda.cu).
//
// It lives in its own header because BOTH backends need it and neither owns it. The one
// thing they disagree about is how a texture is stored on the device: render_cuda uploads
// spectral `DTexture` records (JH coefficients + a grayscale plane), while raster_cuda
// uploads flat linear-RGB `DTex` records for the preview skin. So the VM is templated on
// the texture-record type and reaches a texture only through
//
//     __device__ double TexT::patScalarAt(double u, double v) const
//
// which each backend implements against its own storage (both mirroring the host's
// Texture::scalarAt). Everything else the VM touches — PatGrid, PatScatter, the flat float
// data pool, the POV function library — is already shared __host__ __device__ code in
// pattern.h / pov_functions.h, so there is no second implementation to drift.
#pragma once

#include "pattern.h"   // PatNode, PatOp, PatGrid, PatScatter, patGridSample/patScatterSample

// One flat postfix PatNode pool holds every pattern in the scene back-to-back; a DPattern
// slices it by [off, off+n). Materials/primitives index patterns by their scene index, so
// the DPattern table is parallel to Scene::patterns.
struct DPattern { int off, n; };

// Everything the VM needs beyond the scalar variables: the scene's texture / grid /
// scatter tables and the flat float pool the latter two read their samples from. Bundled
// in a struct rather than threaded as loose parameters because the list grows and every
// growth would otherwise touch every call site.
//
// dPatEnvNoneT<T>() is the OUT-OF-SCOPE environment used at value sites the host compiler
// already refuses `tex:`/`grid:` at (implicit field formulas, medium density/ior), so such
// a node can never actually appear there; the null tables just make the VM total instead
// of undefined if one ever did.
template <class TexT>
struct DPatEnvT {
    const TexT*       tex;      int nTex;
    const PatGrid*    grids;    int nGrids;
    const PatScatter* scatters; int nScatters;
    const float*      dataPool; int dataPoolN;
};

template <class TexT>
__host__ __device__ inline DPatEnvT<TexT> dPatEnvNoneT() {
    DPatEnvT<TexT> e;
    e.tex = nullptr; e.nTex = 0;
    e.grids = nullptr; e.nGrids = 0;
    e.scatters = nullptr; e.nScatters = 0;
    e.dataPool = nullptr; e.dataPoolN = 0;
    return e;
}

// Deterministic integer-hash 3-D value noise. These used to be a hand-kept COPY of
// pattern.h's patHash3/patValueNoise, with a comment promising the two matched; they are
// now thin forwarders to the originals, which are `PATTERN_HD` (host+device) exactly like
// patWorley/patGabor/povNoise already were. Bit-identity between the CPU and GPU noise
// field is therefore structural rather than a promise two bodies have to keep — and it
// has to be, because `fnoise` (O8) sums this same lattice octave by octave and would
// amplify any drift by the octave count.
__device__ static inline double dPatHash3(int ix, int iy, int iz) {
    return patHash3(ix, iy, iz);
}
__device__ static inline double dPatValueNoise(double x, double y, double z) {
    return patValueNoise(x, y, z);
}

// Postfix scalar-stack evaluator (exact port of patternEval). PatNode/PatOp are the POD
// host types (pattern.h), uploaded verbatim; variables come in as scalar args. Pass
// dPatEnvNoneT<TexT>() where textures/tables are out of scope (field formulas, medium
// density/ior) — the host compiler rejects `tex:`/`grid:` at those sites, so such a node
// can never actually appear.
template <class TexT>
__device__ inline double dPatternEval(const PatNode* nodes, int n,
                                      double x, double y, double z, double f,
                                      double nx, double ny, double nz, double r,
                                      double u, double v, double curv, double cavity,
                                      double fw, const DPatEnvT<TexT>& env) {
    double st[64]; int sp = 0;
    double reg[PAT_CSE_REGS];   // CSE registers; StReg always precedes LdReg, so no init
    for (int i = 0; i < n; ++i) {
        const PatNode& nd = nodes[i];
        switch (nd.op) {
            case PatOp::Const:    st[sp++] = nd.a; break;
            case PatOp::VarX:     st[sp++] = x;  break;
            case PatOp::VarY:     st[sp++] = y;  break;
            case PatOp::VarZ:     st[sp++] = z;  break;
            case PatOp::VarF:     st[sp++] = f;  break;
            case PatOp::VarNx:    st[sp++] = nx; break;
            case PatOp::VarNy:    st[sp++] = ny; break;
            case PatOp::VarNz:    st[sp++] = nz; break;
            case PatOp::VarR:     st[sp++] = r;  break;
            case PatOp::VarU:     st[sp++] = u;  break;
            case PatOp::VarV:     st[sp++] = v;  break;
            case PatOp::VarCurv:  st[sp++] = curv; break; // mean curvature at the hit (O3); 0 wherever there is no surface (field/medium sites)
            case PatOp::VarCavity: st[sp++] = cavity; break; // hemispherical enclosure at the hit (O3 s2); 0 at field/medium sites, which have no surface
            case PatOp::VarFootprint: st[sp++] = fw; break;  // shading footprint diameter (O8 s2); 0 = unknown = unfiltered, which is every site but a mode-W/raster primary hit
            case PatOp::VarT:     st[sp++] = 0.0; break;  // flyby timeline: never in scope on-device (camera_curve exprs are consumed at load)
            case PatOp::Neg:      st[sp-1] = -st[sp-1]; break;
            case PatOp::Abs:      st[sp-1] = fabs(st[sp-1]); break;
            case PatOp::Sqrt:     st[sp-1] = sqrt(fmax(0.0, st[sp-1])); break;
            case PatOp::Sin:      st[sp-1] = sin(st[sp-1]); break;
            case PatOp::Cos:      st[sp-1] = cos(st[sp-1]); break;
            case PatOp::Tan:      st[sp-1] = tan(st[sp-1]); break;
            case PatOp::Exp:      st[sp-1] = exp(st[sp-1]); break;
            case PatOp::Log:      st[sp-1] = log(fmax(1e-300, st[sp-1])); break;
            case PatOp::Floor:    st[sp-1] = floor(st[sp-1]); break;
            case PatOp::Fract:    st[sp-1] = st[sp-1] - floor(st[sp-1]); break;
            case PatOp::Sign:     st[sp-1] = (st[sp-1] > 0.0) - (st[sp-1] < 0.0); break;
            case PatOp::Saturate: st[sp-1] = fmin(1.0, fmax(0.0, st[sp-1])); break;
            case PatOp::Add:      { double b = st[--sp]; st[sp-1] += b; break; }
            case PatOp::Sub:      { double b = st[--sp]; st[sp-1] -= b; break; }
            case PatOp::Mul:      { double b = st[--sp]; st[sp-1] *= b; break; }
            case PatOp::Div:      { double b = st[--sp]; st[sp-1] = (b != 0.0) ? st[sp-1] / b : 0.0; break; }
            case PatOp::Mod:      { double b = st[--sp]; st[sp-1] = (b != 0.0) ? st[sp-1] - b * floor(st[sp-1] / b) : 0.0; break; }
            case PatOp::Pow:      { double b = st[--sp]; st[sp-1] = pow(st[sp-1], b); break; }
            case PatOp::Min:      { double b = st[--sp]; st[sp-1] = fmin(st[sp-1], b); break; }
            case PatOp::Max:      { double b = st[--sp]; st[sp-1] = fmax(st[sp-1], b); break; }
            case PatOp::Atan2:    { double b = st[--sp]; st[sp-1] = atan2(st[sp-1], b); break; }
            case PatOp::Step:     { double b = st[--sp]; st[sp-1] = (b >= st[sp-1]) ? 1.0 : 0.0; break; }
            case PatOp::Clamp:    { double hi = st[--sp], lo = st[--sp]; st[sp-1] = fmin(hi, fmax(lo, st[sp-1])); break; }
            case PatOp::Mix:      { double t = st[--sp], b = st[--sp]; st[sp-1] = st[sp-1] + (b - st[sp-1]) * t; break; }
            case PatOp::Smoothstep: {
                double xx = st[--sp], e1 = st[--sp], e0 = st[sp-1];
                double tt = (e1 != e0) ? (xx - e0) / (e1 - e0) : 0.0;
                tt = fmin(1.0, fmax(0.0, tt));
                st[sp-1] = tt * tt * (3.0 - 2.0 * tt);
                break;
            }
            case PatOp::Noise:    { double zz = st[--sp], yy = st[--sp]; st[sp-1] = dPatValueNoise(st[sp-1], yy, zz); break; }
            case PatOp::DNoise: {   // POV vector noise, component in nd.a (POV_HD, double)
                double zz = st[--sp], yy = st[--sp], vout[3];
                povDNoise(st[sp-1], yy, zz, vout);
                st[sp-1] = vout[(int)nd.a];
                break;
            }
            case PatOp::DTurb: {    // POV vector turbulence (octave fBm of DNoise)
                double om = st[--sp], la = st[--sp], oc = st[--sp];
                double zz = st[--sp], yy = st[--sp], vout[3];
                povDTurbulence(st[sp-1], yy, zz, oc, la, om, vout);
                st[sp-1] = vout[(int)nd.a];
                break;
            }
            case PatOp::Worley: {   // cellular noise: a = 0 F1 / 1 F2 / 2 F2-F1 / 3 id
                double m = st[--sp], zz = st[--sp], yy = st[--sp], w[3];
                int mi = (int)floor(m + 0.5);
                if (mi < 0) mi = 0; if (mi > 2) mi = 2;
                patWorley(st[sp-1], yy, zz, mi, w);
                int sel = (int)nd.a;
                st[sp-1] = (sel == 3) ? w[2] : (sel == 2) ? (w[1] - w[0]) : w[sel];
                break;
            }
            case PatOp::Gabor: {    // anisotropic band-limited noise (GABOR_HD, double)
                double dz = st[--sp], dy = st[--sp], dx = st[--sp];
                double ff = st[--sp], zz = st[--sp], yy = st[--sp];
                st[sp-1] = patGabor(st[sp-1], yy, zz, ff, dx, dy, dz);
                break;
            }
            case PatOp::BlueNoise: { // Poisson-disk placement: a = 0 F1 / 1 F2 / 2 F2-F1 / 3 id
                double rr = st[--sp], zz = st[--sp], yy = st[--sp], b[3];
                patBlueNoise(st[sp-1], yy, zz, rr, b);
                int sel = (int)nd.a;
                st[sp-1] = (sel == 3) ? b[2] : (sel == 2) ? (b[1] - b[0]) : b[sel];
                break;
            }
            case PatOp::FNoise: {   // filtered (band-limited) fBm — the shared PATTERN_HD core
                double oc = st[--sp], ww = st[--sp], zz = st[--sp], yy = st[--sp];
                st[sp-1] = patFilteredNoise(st[sp-1], yy, zz, ww, oc);
                break;
            }
            case PatOp::PovFn: {
                int id = (int)nd.a;
                int na = povFnArity(id);
                double args[POV_FN_MAX_ARGS];
                for (int k = na - 1; k >= 0; --k) args[k] = st[--sp];
                st[sp++] = povFnEval(id, args);
                break;
            }
            case PatOp::Tex: {
                double vv = st[--sp];                       // args pushed as (u, v)
                int    ti = (int)nd.a;
                st[sp-1] = (env.tex && ti >= 0 && ti < env.nTex)
                             ? env.tex[ti].patScalarAt(st[sp-1], vv) : 0.0;
                break;
            }
            case PatOp::Grid: {
                // Arity is the GRID's own dimensionality; coordinates were pushed in
                // axis order, so pop them back to front. patGridSample is the SHARED
                // __host__ __device__ sampler from pattern.h — there is no device
                // re-implementation to drift from the host one.
                int gi = (int)nd.a;
                // A resolved index always names a live table (the host compiler only
                // accepts `grid:` where a table scope was in scope, and that scope is the
                // same Scene this device scene was built from), so this can only fire if a
                // call site passed an empty env — a wiring bug. Bail out of the WHOLE
                // program: the operand count is the table's own ndim, exactly what can't be
                // read here, so pushing a placeholder would leave the stack unbalanced and
                // silently return a COORDINATE as the result. Mirrors pattern.h.
                if (gi < 0 || gi >= env.nGrids || !env.grids) return 0.0;
                const PatGrid& g = env.grids[gi];
                int gnd = g.ndim < 1 ? 1 : (g.ndim > PAT_ND_MAX_DIM ? PAT_ND_MAX_DIM : g.ndim);
                double co[PAT_ND_MAX_DIM];
                for (int k = gnd - 1; k >= 0; --k) co[k] = st[--sp];
                st[sp++] = patGridSample(g, env.dataPool, env.dataPoolN, co);
                break;
            }
            case PatOp::Scatter: {
                // Same contract as Grid, sharing the same flat pool and the same shared
                // sampler — a scatter just resolves its value by inverse-distance blend
                // instead of a lattice walk.
                int si = (int)nd.a;
                if (si < 0 || si >= env.nScatters || !env.scatters) return 0.0;  // see Grid
                const PatScatter& s = env.scatters[si];
                int snd = s.ndim < 1 ? 1 : (s.ndim > PAT_ND_MAX_DIM ? PAT_ND_MAX_DIM : s.ndim);
                double co[PAT_ND_MAX_DIM];
                for (int k = snd - 1; k >= 0; --k) co[k] = st[--sp];
                st[sp++] = patScatterSample(s, env.dataPool, env.dataPoolN, co);
                break;
            }
            case PatOp::StReg:    reg[(int)nd.a] = st[sp-1]; break;   // save, keep on stack
            case PatOp::LdReg:    st[sp++] = reg[(int)nd.a]; break;   // reuse saved value
        }
    }
    return sp > 0 ? st[0] : 0.0;
}
