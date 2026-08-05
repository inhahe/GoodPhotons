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

// Deterministic integer-hash 3-D value noise; matches patHash3/patValueNoise so the GPU
// and CPU produce the same noise field. Output in [0,1].
__device__ static inline double dPatHash3(int ix, int iy, int iz) {
    unsigned int h = (unsigned int)ix * 374761393u + (unsigned int)iy * 668265263u
                   + (unsigned int)iz * 2147483647u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return (double)h / 4294967295.0;
}
__device__ static inline double dPatValueNoise(double x, double y, double z) {
    double fx = floor(x), fy = floor(y), fz = floor(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    double tx = x - fx, ty = y - fy, tz = z - fz;
    double ux = tx * tx * (3.0 - 2.0 * tx);
    double uy = ty * ty * (3.0 - 2.0 * ty);
    double uz = tz * tz * (3.0 - 2.0 * tz);
    double c000 = dPatHash3(ix,     iy,     iz);
    double c100 = dPatHash3(ix + 1, iy,     iz);
    double c010 = dPatHash3(ix,     iy + 1, iz);
    double c110 = dPatHash3(ix + 1, iy + 1, iz);
    double c001 = dPatHash3(ix,     iy,     iz + 1);
    double c101 = dPatHash3(ix + 1, iy,     iz + 1);
    double c011 = dPatHash3(ix,     iy + 1, iz + 1);
    double c111 = dPatHash3(ix + 1, iy + 1, iz + 1);
    double x00 = c000 + (c100 - c000) * ux;
    double x10 = c010 + (c110 - c010) * ux;
    double x01 = c001 + (c101 - c001) * ux;
    double x11 = c011 + (c111 - c011) * ux;
    double y0  = x00 + (x10 - x00) * uy;
    double y1  = x01 + (x11 - x01) * uy;
    return y0 + (y1 - y0) * uz;
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
                                      double u, double v,
                                      const DPatEnvT<TexT>& env) {
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
