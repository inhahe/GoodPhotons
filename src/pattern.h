// Procedural patterns: math-driven scalar fields for material properties.
//
// A Pattern maps a per-hit context (world point x,y,z; field value f; surface
// normal nx,ny,nz; radius r = |p|) to a single scalar. That scalar then drives a
// material property at the shading point — roughness, thin-film thickness, an
// emission/absorption scale, or the selection weight between two whole materials
// (so ANY property, including colour and material TYPE, can vary across a surface).
//
// Two authoring front-ends compile to the SAME representation:
//   (A) built-in GENERATORS — named patterns (axis, radial, bands, checker, noise,
//       field) with a few parameters; the parser expands them to postfix.
//   (B) an EXPRESSION — an infix math formula over the variables x y z f nx ny nz r
//       (e.g. "0.5 + 0.5*sin(10*x)") compiled by shunting-yard to postfix.
//
// Like the implicit field, a pattern is a FLAT ARRAY of POD nodes in POSTFIX order,
// evaluated by a tiny scalar stack — no pointers, no recursion, trivially GPU-
// portable (the CUDA backend iterates the same array).
#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstring>         // memcpy: bit-pattern keys in the CSE optimizer
#include <unordered_map>   // hash-consing map in the CSE optimizer
#include <algorithm>       // sort: CSE register-priority ordering
#include "linalg.h"
#include "pov_functions.h"   // exact POV-Ray internal isosurface functions (f_torus, ...)

// The grid sampler below is shared verbatim by the CPU evaluator and the CUDA
// pattern VM (render_cuda.cu includes this header), so it needs the same
// host/device annotation pov_functions.h already established for `povFnEval`.
#if defined(__CUDACC__)
  #ifndef PATTERN_HD
  #define PATTERN_HD __host__ __device__
  #endif
  #define PAT_FLOOR(x) ::floor(x)
  #define PAT_POW(x, y) ::pow(x, y)
#else
  #ifndef PATTERN_HD
  #define PATTERN_HD
  #endif
  #define PAT_FLOOR(x) std::floor(x)
  #define PAT_POW(x, y) std::pow(x, y)
#endif

// ---------------------------------------------------------------------------
// Postfix opcodes.
// ---------------------------------------------------------------------------
enum class PatOp : int {
    // nullary: push a value
    Const = 0,                       // push a (the node's literal)
    VarX, VarY, VarZ,                // world hit point
    VarF,                            // implicit field value at the hit (~0 on a surface)
    VarNx, VarNy, VarNz,             // surface normal (oriented against the ray)
    VarR,                            // radius = sqrt(x*x+y*y+z*z)
    VarU, VarV,                      // surface UV coordinates (mesh or native-primitive wrap)
    VarT,                            // flyby timeline t in [0,1] — ONLY in scope inside a
                                     // camera_curve record track (fov_from/roll_from/...);
                                     // deliberately placed AFTER VarV so patternHasFreeVars'
                                     // "surface intrinsic" range (VarX..VarV) excludes it.
    // unary: pop 1, push 1
    Neg, Abs, Sqrt, Sin, Cos, Tan, Exp, Log, Floor, Fract, Sign, Saturate,
    // binary: pop 2 (a below b), push 1
    Add, Sub, Mul, Div, Mod, Pow, Min, Max, Atan2, Step,
    // ternary: pop 3 (a,b,c bottom->top), push 1
    Clamp,        // clamp(x, lo, hi)
    Mix,          // lerp(a, b, t)
    Smoothstep,   // smoothstep(edge0, edge1, x)
    Noise,        // value noise of (x, y, z) in [0,1]
    // variadic built-in: an exact POV-Ray internal function. The node's `a` holds
    // the POV internal id (0..75); the evaluator pops povFnArity(id) args (the
    // first three are the coordinates) and pushes the returned scalar.
    PovFn,
    // texture sample: pops (u, v) and pushes the LINEAR grayscale value of a named
    // image texture there (Texture::scalarAt — the same sampler roughness/film maps
    // use). The node's `a` holds the Scene::textures index, resolved at compile time
    // from the authored `tex:<name>(u, v)` call. This is what makes an image usable
    // as a TERM INSIDE a formula ("procedural * photo"), as opposed to binding a
    // whole image to a slot. Sampling needs a texture table, which only exists in a
    // scene-shading context, so the compiler only accepts `tex:` where one is in
    // scope (see PatTexScope).
    Tex,
    // N-D sampled-array lookup: pops the grid's `ndim` coordinates (pushed in axis
    // order) and pushes the interpolated value of a named regular lattice. The node's
    // `a` holds the grid table index, resolved at compile time from the authored
    // `grid:<name>(c0, …)` call — the arity is the grid's OWN dimensionality, so
    // unlike every other op it is not a property of the function name alone.
    // This is the runtime half of the axis-labelled-array design (`[0 1](u)` = "an
    // array sampled along u"); a `grid` element is the named, reusable spelling of
    // that array, and the call IS the sample.
    Grid,
    // Scattered-sample lookup: the RAGGED sibling of Grid. Pops the scatter's `ndim`
    // query coordinates and pushes the Shepard inverse-distance blend of samples that
    // sit at arbitrary positions — no lattice, so it reads measured/irregular data a
    // Grid cannot represent. `a` holds the scatter-table index; the arity is likewise
    // the scatter's own dimensionality.
    Scatter,
    // ALBEDO input `a` — a *named input with no per-hit intrinsic* (ROADMAP_records.md
    // §3.2). Every other variable above is system-provided, so leaving one unbound means
    // "read it from the shading point", which is exactly "don't substitute". `a` has no
    // such source, so it is resolved at LOAD time instead — either to the expression a
    // use site binds it to (`gold(a=0.3*u)`) or, unbound, to the material's
    // `albedo_default` constant. It therefore never reaches the evaluator or the device;
    // the evaluator case below exists only to keep the switch exhaustive.
    //
    // Deliberately appended at the END of the enum, like Tex/Grid/Scatter before it, so
    // patternHasFreeVars' "surface intrinsic" range VarX..VarV is unperturbed — and
    // correctly so: once resolved, a program whose only variable was `a` IS a load-time
    // constant, so it stays legal at a constant value site.
    VarA,
    // Named-spectrum sample: pops a wavelength (nm) and pushes the value of a declared
    // `spectrum "<name>"` there. The node's `a` holds an index into the compiling
    // loader's own spectrum vector, resolved at compile time from `spec:<name>(w)`.
    //
    // This exists for ONE caller: a user-supplied RGB->spectral upsampler
    // (`upsample "<name>" { expr "…" }`, K1), where it is what lets an upsampler be a
    // *measured basis* — `r*spec:red(w) + g*spec:green(w) + b*spec:blue(w)` — and not
    // merely closed-form arithmetic. Like `tex:` it needs a table that exists in only
    // one context, so the compiler accepts it only where a PatSpecScope is in scope,
    // which is exactly an upsample body.
    //
    // It NEVER reaches the device, and cannot: an upsample program is consumed at LOAD
    // time (each authored colour is baked through it into an ordinary tabulated
    // Spectrum) and is never stored on a Material or in Scene::patterns, which are the
    // only things uploaded. That is a structural guarantee, not a convention — the same
    // one that keeps `albedo_default` loader-side.
    Spec,
    // Common-subexpression registers, emitted ONLY by patternOptimizeCSE (never by the
    // compiler or authors). StReg copies the top of stack into reg[a] WITHOUT popping
    // (the value is both the subtree's result and a saved copy); LdReg pushes reg[a].
    // The optimizer guarantees every LdReg is preceded (in program order) by the StReg
    // of the same register, so the evaluators' reg arrays need no zero-init. Both are
    // pure data movement — a load reproduces bit-for-bit what re-executing the stored
    // subtree would have produced, in the evaluator's own precision (double on the CPU
    // VMs, float in dPatternEvalF) — so a CSE'd program is exactly equivalent on every
    // backend. Appended at the END of the enum, like Tex/Grid/Scatter/VarA/Spec before
    // them, so patternHasFreeVars' VarX..VarV range and varName() are unperturbed.
    StReg,
    LdReg,
};

// Register-file size available to a CSE-optimized program (per evaluator invocation).
// Bounds the evaluators' stack-local `reg[]` arrays; the optimizer never assigns more.
constexpr int PAT_CSE_REGS = 32;

// Which variable vocabulary an expression is compiled against. The default surface
// vocabulary (x/y/z/f/n*/r/u/v, plus the scoped `t` and `a`) describes a point being
// shaded. An UPSAMPLE body has no shading point at all — it is a function from a colour
// and a wavelength to a spectral value — so it gets a DISJOINT vocabulary rather than a
// superset: `r`, `g`, `b` (the colour, linear sRGB) and `w` (wavelength, nm).
//
// Disjoint, not additive, and that is the whole point. `r` already means RADIUS in the
// surface vocabulary, so an upsampler that could see both would give one spelling two
// meanings depending on where it was written — the silent-wrong-value failure this
// codebase refuses. In Upsample mode every surface name is REJECTED BY NAME with a
// message saying so, so no spelling's meaning quietly depends on its context.
//
// Internally the four upsample variables reuse the VarX/VarY/VarZ/VarU opcode slots —
// a register assignment, not an aliasing of meaning: the surface names that normally
// occupy those slots cannot be written here, so no program can observe the overlap.
// That keeps the enum, `patternHasFreeVars`' VarX..VarV range, and the device switch
// untouched by a feature that never runs on the device.
enum class PatVarMode : int { Surface = 0, Upsample };

// One postfix node. POD (no std:: members) so it uploads to the GPU verbatim.
struct PatNode {
    PatOp  op = PatOp::Const;
    double a  = 0.0;   // literal for Const; POV internal id for PovFn; texture index
                       // for Tex; grid-table index for Grid
};

// ---------------------------------------------------------------------------
// N-D sampled array ("grid") — a regular lattice of scalars with a world box.
// ---------------------------------------------------------------------------
// A faithful port of loom's `data.Grid` + `interp.GridField` (tools/loom/loom):
// the samples sit on a FIXED regular lattice, so sampling is separable N-linear
// interpolation over 2^ndim corners. Deliberately POD-with-a-pointer so the same
// header + sampler compile for the GPU: the device copy just points `data` at a
// device allocation and everything else is by value.
//
// Axis order is C-order / row-major: axis 0 is the OUTERMOST (slowest-varying)
// index, matching loom's `_flatten_nested` (so `[[0 1 2][3 4 5]]` is shape (2,3)
// with axis 0 selecting the row). A query is `grid:<name>(c0, c1, …)` with one
// coordinate per axis, in that same order.
//
// Shared by BOTH N-D pattern tables (PatGrid and its ragged sibling PatScatter):
// it bounds the fixed-size coordinate buffers the samplers put on the stack, so it
// has to be one constant, not one per datatype.
enum : int { PAT_ND_MAX_DIM = 4 };

// Out-of-domain policy. Mirrors loom's `on_outside` — minus "raise", which is a
// load-time authoring guard in loom but would have to be a per-sample abort here;
// a renderer samples millions of times per frame and cannot throw, so ftrace's
// authoring guard is the domain check at load instead.
enum class PatGridOutside : int {
    Clamp = 0,      // edge-extend: pin to the boundary cell (default)
    Wrap,           // periodic, period hi-lo; sample n-1 aliases sample 0
    Extrapolate,    // keep the boundary cell but let the fraction run past [0,1]
};

// Numbers live in ONE shared pool (Scene::dataPool / its device copy) that every N-D
// table draws from, and a table refers to its run by OFFSET, never by pointer: a
// pointer into a std::vector would dangle the moment the pool grew, and the offset
// form is also exactly what the GPU upload wants (one flat device array + POD
// headers, regardless of how many tables or kinds of table a scene declares).
struct PatGrid {
    int ndim = 1;
    int shape[PAT_ND_MAX_DIM] = {1, 1, 1, 1};
    double lo[PAT_ND_MAX_DIM] = {0, 0, 0, 0};
    double hi[PAT_ND_MAX_DIM] = {0, 0, 0, 0};
    PatGridOutside outside = PatGridOutside::Clamp;
    int off = 0;                   // index of sample 0 within the shared pool
    int count = 0;                 // == product(shape); the sampler never reads past it
};

// Lower cell index + in-cell fraction on one axis. Device-safe (no std::, no throw).
// Twin of loom's `interp._cell_base_frac`.
PATTERN_HD inline void patGridCellFrac(const PatGrid& g, int axis, double coord,
                                       int& i0, double& fr) {
    const int n = g.shape[axis];
    if (n <= 1) { i0 = 0; fr = 0.0; return; }
    const double lo = g.lo[axis], hi = g.hi[axis];
    const double span = hi - lo;
    // Position in SAMPLE units (0 .. n-1), which is what the cell walk indexes.
    double p = (span != 0.0) ? (coord - lo) * (double)(n - 1) / span : 0.0;
    if (g.outside == PatGridOutside::Wrap) {
        const double per = (double)(n - 1);
        p -= PAT_FLOOR(p / per) * per;              // fold into [0, n-1)
        if (!(p < per)) p = 0.0;                    // numerical guard at the seam
        i0 = (int)PAT_FLOOR(p);
        if (i0 > n - 2) i0 = n - 2 < 0 ? 0 : n - 2;
        fr = p - (double)i0;
        return;
    }
    if (p <= 0.0) {
        i0 = 0;
        fr = (g.outside == PatGridOutside::Extrapolate && p < 0.0) ? p : 0.0;
        return;
    }
    if (p >= (double)(n - 1)) {
        i0 = n - 2;
        fr = (g.outside == PatGridOutside::Extrapolate && p > (double)(n - 1))
                 ? p - (double)(n - 2) : 1.0;
        return;
    }
    i0 = (int)PAT_FLOOR(p);
    fr = p - (double)i0;
}

// Separable N-linear sample over the 2^ndim corners of the containing cell.
// `pool` is the shared sample pool; `coords` holds one coordinate per axis, in axis
// order. `poolN` bounds every read so a malformed header can never walk off the end.
PATTERN_HD inline double patGridSample(const PatGrid& g, const float* pool, int poolN,
                                       const double* coords) {
    if (!pool || g.count <= 0) return 0.0;
    int    base[PAT_ND_MAX_DIM];
    double frac[PAT_ND_MAX_DIM];
    const int nd = (g.ndim < 1) ? 1 : (g.ndim > PAT_ND_MAX_DIM ? PAT_ND_MAX_DIM : g.ndim);
    for (int a = 0; a < nd; ++a) patGridCellFrac(g, a, coords[a], base[a], frac[a]);
    const bool wrap = (g.outside == PatGridOutside::Wrap);
    double acc = 0.0;
    const int corners = 1 << nd;
    for (int c = 0; c < corners; ++c) {
        double w = 1.0;
        int    flat = 0;
        for (int a = 0; a < nd; ++a) {
            const int n  = g.shape[a];
            const int up = (c >> a) & 1;
            w *= up ? frac[a] : (1.0 - frac[a]);
            int idx = base[a] + up;
            if (wrap && n > 1)      idx %= (n - 1);      // n-1 aliases 0
            else if (idx < 0)       idx = 0;
            else if (idx > n - 1)   idx = n - 1;
            flat = flat * n + idx;                       // C-order: axis 0 outermost
        }
        if (w == 0.0) continue;
        const int at = g.off + flat;
        if (flat >= 0 && flat < g.count && at >= 0 && at < poolN)
            acc += w * (double)pool[at];
    }
    return acc;
}

// ---------------------------------------------------------------------------
// N-D scattered samples ("scatter") — values at ARBITRARY positions, no lattice.
// ---------------------------------------------------------------------------
// The ragged sibling of PatGrid, and a faithful port of loom's `data.Scatter` +
// `interp.ScatterField`. A grid can only express data that already sits on a regular
// box; measurements rarely do — a handful of probe points, samples along a path, a
// few authored control values. A scatter stores each sample's own position and blends
// them by Shepard inverse-distance weighting, which needs no structure at all.
//
// Weight of sample i at query q is |q - p_i|^-power, i.e. (d²)^(-power/2); the result
// is the weighted mean, so the interpolant reproduces a CONSTANT field exactly (the
// weights are normalised) and reproduces each sample exactly at its own position (the
// eps test below, which also removes the 1/0 singularity there). Beyond the samples
// it flattens toward the global weighted mean rather than diverging — the natural
// counterpart to a grid's `clamp`.
//
// Positions and values interleave in the SAME shared pool a grid draws from, one
// sample per stride of (ndim + 1): [p0 … p_{ndim-1}, value]. That keeps the whole
// N-D-table story to a single flat float array on both backends.
struct PatScatter {
    int    ndim  = 1;
    int    count = 0;         // number of samples (each ndim+1 floats wide)
    int    off   = 0;         // index of sample 0's first coordinate in the pool
    double power = 2.0;       // Shepard exponent; 2 is loom's default (and the cheap path)
    double eps   = 1e-9;      // SQUARED-distance coincidence threshold
};

// Shepard inverse-distance sample. Device-safe; `poolN` bounds every read so a
// malformed header can never walk off the end. Twin of loom's `_shepard_weights`
// + `ScatterField._eval`.
PATTERN_HD inline double patScatterSample(const PatScatter& s, const float* pool,
                                          int poolN, const double* q) {
    if (!pool || s.count <= 0) return 0.0;
    const int nd = (s.ndim < 1) ? 1 : (s.ndim > PAT_ND_MAX_DIM ? PAT_ND_MAX_DIM : s.ndim);
    const int stride = nd + 1;
    const double half = 0.5 * s.power;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < s.count; ++i) {
        const int at = s.off + i * stride;
        if (at < 0 || at + stride > poolN) break;      // truncated run: stop, never read past
        double d2 = 0.0;
        for (int a = 0; a < nd; ++a) {
            const double d = q[a] - (double)pool[at + a];
            d2 += d * d;
        }
        const double val = (double)pool[at + nd];
        // Coincident with this sample: return it EXACTLY. This is both the correct
        // limit (its weight diverges) and what keeps the sampler finite at a sample.
        if (d2 <= s.eps) return val;
        const double w = (s.power == 2.0) ? 1.0 / d2 : PAT_POW(d2, -half);
        num += w * val;
        den += w;
    }
    return (den > 0.0) ? num / den : 0.0;
}

// Per-hit evaluation context (the pattern variables).
struct PatCtx {
    double x = 0, y = 0, z = 0;   // world point
    double f = 0;                 // implicit field value (0 for non-implicit hits)
    double nx = 0, ny = 0, nz = 0;// surface normal
    double r = 0;                 // radius |p|
    double u = 0, v = 0;          // surface UV (mesh interpolated or native-primitive wrap)
    double t = 0;                 // flyby timeline in [0,1] (camera_curve record tracks only)
    // PatOp::Tex sampler hook. pattern.h deliberately knows nothing about Texture
    // (texture.h drags in the spectral/upsampling machinery and is not something we
    // want every TU — least of all nvcc — to parse), so a sample goes through an
    // opaque callback installed by whoever owns the Scene: scene.h's bindPatTex.
    // Null => Tex reads 0; that can only happen at a site where the compiler already
    // refused to accept `tex:` in the first place.
    double (*texFn)(const void* self, int idx, double u, double v) = nullptr;
    const void* texSelf = nullptr;
    // N-D table headers (PatOp::Grid / PatOp::Scatter). Unlike a Texture, these are
    // plain POD over a flat float pool, so they need no opaque callback — the tables
    // are handed over directly and the very same samplers run on both backends.
    // Both kinds index the ONE shared pool below.
    const PatGrid* grids = nullptr;
    int nGrids = 0;
    const PatScatter* scatters = nullptr;
    int nScatters = 0;
    const float* dataPool = nullptr;
    int dataPoolN = 0;
    // PatOp::Spec sampler hook, for `spec:<name>(w)` inside an upsample body. Same
    // opaque-callback shape (and the same reason) as texFn: a Spectrum is a host
    // std::function living in spectrum.h, which pattern.h deliberately does not know
    // about. Null => Spec reads 0, which can only happen at a site where the compiler
    // already refused `spec:` outright.
    double (*specFn)(const void* self, int idx, double w) = nullptr;
    const void* specSelf = nullptr;
};

// The SCENE-OWNED half of a PatCtx, with none of the per-hit coordinates: just the N-D
// sampler tables. A shading site builds its PatCtx from a Hit and binds these in one go
// (scene.h's patCtxFromHit), but a *field* formula — an isosurface leaf, a medium
// density/ior program, a camera_curve driver — has no Hit at all: its evaluator makes a
// bare PatCtx per query, so it has to carry the tables separately and splice them in.
// Hence this struct, threaded through the field/medium evaluators as one parameter
// (never as loose pointers) for the same reason DPatEnv exists on the device: the list
// grows, and every growth would otherwise touch a dozen signatures.
//
// Deliberately NOT stored inside Implicit/Medium: these are pointers into Scene's
// vectors, and a Scene is copied and moved (buildCornell returns by value), so a cached
// copy would dangle. The owner passes them at the call.
struct PatTables {
    const PatGrid*    grids    = nullptr; int nGrids    = 0;
    const PatScatter* scatters = nullptr; int nScatters = 0;
    const float*      dataPool = nullptr; int dataPoolN = 0;
};
// Splice the tables into a freshly built PatCtx. Null is a no-op, which keeps every
// evaluator's `const PatTables* = nullptr` default meaning "no tables in scope".
inline void patBindTables(PatCtx& c, const PatTables* t) {
    if (!t) return;
    c.grids     = t->grids;    c.nGrids    = t->nGrids;
    c.scatters  = t->scatters; c.nScatters = t->nScatters;
    c.dataPool  = t->dataPool; c.dataPoolN = t->dataPoolN;
}

// Compile-time texture-name resolution for `tex:<name>(u, v)`. A value site passes
// one of these to compilePatternExpr exactly when a texture table will be in scope
// at evaluation time; passing nothing (the default) makes `tex:` a scope ERROR
// rather than a silent zero — so an implicit field formula, a medium density/ior
// program or a load-time constant site can never smuggle in a sample it could not
// actually perform. Kept as self+thunk (not std::function) to keep this header light.
struct PatTexScope {
    const void* self = nullptr;
    int (*lookup)(const void* self, const char* name) = nullptr;   // -> index, or -1
    int resolve(const char* n) const { return lookup ? lookup(self, n) : -1; }
};

// Compile-time name resolution for `spec:<name>(w)` — the named-spectrum sampler. Same
// shape and the same scope rule as PatTexScope: a site with no spectrum table at eval
// time passes nothing, which turns the sample into a legible compile error instead of a
// silent zero. In practice exactly one site passes it: an `upsample` body.
struct PatSpecScope {
    const void* self = nullptr;
    int (*lookup)(const void* self, const char* name) = nullptr;   // -> index, or -1
    int resolve(const char* n) const { return lookup ? lookup(self, n) : -1; }
};

// Which N-D table a `<kind>:<name>(…)` call names. Passed to the lookup below so ONE
// scope object serves every such datatype: adding the next one costs an enumerator,
// not another parameter on compilePatternExpr (and another edit at all of its call
// sites) — the same reasoning as DPatEnv on the GPU side.
enum class PatTableKind : int { Grid = 0, Scatter };

// Compile-time name resolution for the N-D table samplers `grid:<name>(c0, …)` and
// `scatter:<name>(c0, …)`. Same shape and the same scope rule as PatTexScope — a site
// that will have no table at eval time passes nothing, making the sample a legible
// ERROR instead of a silent zero. The lookup also reports the table's DIMENSIONALITY,
// because that is the call's arity: a 2-D grid must be called `grid:g(x, y)`, and
// getting it wrong is an authoring error the compiler can name precisely.
struct PatTableScope {
    const void* self = nullptr;
    int (*lookup)(const void* self, PatTableKind kind, const char* name, int* ndim) = nullptr;
    int resolve(PatTableKind kind, const char* n, int& ndim) const {
        ndim = 0;
        return lookup ? lookup(self, kind, n, &ndim) : -1;
    }
};

inline PatCtx makePatCtx(const Vec3& p, double f, const Vec3& n, double u = 0, double v = 0) {
    PatCtx c;
    c.x = p.x; c.y = p.y; c.z = p.z;
    c.f = f;
    c.nx = n.x; c.ny = n.y; c.nz = n.z;
    c.r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    c.u = u; c.v = v;
    return c;
}

// ---- 3-D value noise (hash lattice + trilinear smoothstep fade) -------------
// Deterministic integer hash so CPU and GPU agree bit-for-bit. Output in [0,1].
inline double patHash3(int ix, int iy, int iz) {
    uint32_t h = (uint32_t)ix * 374761393u + (uint32_t)iy * 668265263u
               + (uint32_t)iz * 2147483647u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return (double)h / 4294967295.0;   // [0,1]
}
inline double patValueNoise(double x, double y, double z) {
    double fx = std::floor(x), fy = std::floor(y), fz = std::floor(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    double tx = x - fx, ty = y - fy, tz = z - fz;
    // smoothstep fade
    double ux = tx * tx * (3.0 - 2.0 * tx);
    double uy = ty * ty * (3.0 - 2.0 * ty);
    double uz = tz * tz * (3.0 - 2.0 * tz);
    double c000 = patHash3(ix,     iy,     iz);
    double c100 = patHash3(ix + 1, iy,     iz);
    double c010 = patHash3(ix,     iy + 1, iz);
    double c110 = patHash3(ix + 1, iy + 1, iz);
    double c001 = patHash3(ix,     iy,     iz + 1);
    double c101 = patHash3(ix + 1, iy,     iz + 1);
    double c011 = patHash3(ix,     iy + 1, iz + 1);
    double c111 = patHash3(ix + 1, iy + 1, iz + 1);
    double x00 = c000 + (c100 - c000) * ux;
    double x10 = c010 + (c110 - c010) * ux;
    double x01 = c001 + (c101 - c001) * ux;
    double x11 = c011 + (c111 - c011) * ux;
    double y0  = x00 + (x10 - x00) * uy;
    double y1  = x01 + (x11 - x01) * uy;
    return y0 + (y1 - y0) * uz;
}

// ---- postfix evaluator ------------------------------------------------------
inline double patternEval(const PatNode* nodes, int n, const PatCtx& c) {
    double st[64];
    double reg[PAT_CSE_REGS];   // CSE registers; StReg always precedes LdReg, so no init
    int sp = 0;
    for (int i = 0; i < n; ++i) {
        const PatNode& nd = nodes[i];
        switch (nd.op) {
            case PatOp::Const:    st[sp++] = nd.a; break;
            case PatOp::VarX:     st[sp++] = c.x;  break;
            case PatOp::VarY:     st[sp++] = c.y;  break;
            case PatOp::VarZ:     st[sp++] = c.z;  break;
            case PatOp::VarF:     st[sp++] = c.f;  break;
            case PatOp::VarNx:    st[sp++] = c.nx; break;
            case PatOp::VarNy:    st[sp++] = c.ny; break;
            case PatOp::VarNz:    st[sp++] = c.nz; break;
            case PatOp::VarR:     st[sp++] = c.r;  break;
            case PatOp::VarU:     st[sp++] = c.u;  break;
            case PatOp::VarV:     st[sp++] = c.v;  break;
            case PatOp::VarT:     st[sp++] = c.t;  break;
            // `a` is resolved at load time (bound at the use site, or to the material's
            // albedo_default) and so is unreachable here; 0 keeps the switch total.
            case PatOp::VarA:     st[sp++] = 0.0;  break;
            case PatOp::Neg:      st[sp-1] = -st[sp-1]; break;
            case PatOp::Abs:      st[sp-1] = std::fabs(st[sp-1]); break;
            case PatOp::Sqrt:     st[sp-1] = std::sqrt(std::fmax(0.0, st[sp-1])); break;
            case PatOp::Sin:      st[sp-1] = std::sin(st[sp-1]); break;
            case PatOp::Cos:      st[sp-1] = std::cos(st[sp-1]); break;
            case PatOp::Tan:      st[sp-1] = std::tan(st[sp-1]); break;
            case PatOp::Exp:      st[sp-1] = std::exp(st[sp-1]); break;
            case PatOp::Log:      st[sp-1] = std::log(std::fmax(1e-300, st[sp-1])); break;
            case PatOp::Floor:    st[sp-1] = std::floor(st[sp-1]); break;
            case PatOp::Fract:    st[sp-1] = st[sp-1] - std::floor(st[sp-1]); break;
            case PatOp::Sign:     st[sp-1] = (st[sp-1] > 0.0) - (st[sp-1] < 0.0); break;
            case PatOp::Saturate: st[sp-1] = std::fmin(1.0, std::fmax(0.0, st[sp-1])); break;
            case PatOp::Add:      { double b = st[--sp]; st[sp-1] += b; break; }
            case PatOp::Sub:      { double b = st[--sp]; st[sp-1] -= b; break; }
            case PatOp::Mul:      { double b = st[--sp]; st[sp-1] *= b; break; }
            case PatOp::Div:      { double b = st[--sp]; st[sp-1] = (b != 0.0) ? st[sp-1] / b : 0.0; break; }
            case PatOp::Mod:      { double b = st[--sp]; st[sp-1] = (b != 0.0) ? st[sp-1] - b * std::floor(st[sp-1] / b) : 0.0; break; }
            case PatOp::Pow:      { double b = st[--sp]; st[sp-1] = std::pow(st[sp-1], b); break; }
            case PatOp::Min:      { double b = st[--sp]; st[sp-1] = std::fmin(st[sp-1], b); break; }
            case PatOp::Max:      { double b = st[--sp]; st[sp-1] = std::fmax(st[sp-1], b); break; }
            case PatOp::Atan2:    { double b = st[--sp]; st[sp-1] = std::atan2(st[sp-1], b); break; }
            case PatOp::Step:     { double b = st[--sp]; st[sp-1] = (b >= st[sp-1]) ? 1.0 : 0.0; break; }
            case PatOp::Clamp:    { double hi = st[--sp], lo = st[--sp]; st[sp-1] = std::fmin(hi, std::fmax(lo, st[sp-1])); break; }
            case PatOp::Mix:      { double t = st[--sp], b = st[--sp]; st[sp-1] = st[sp-1] + (b - st[sp-1]) * t; break; }
            case PatOp::Smoothstep: {
                double xx = st[--sp], e1 = st[--sp], e0 = st[sp-1];
                double tt = (e1 != e0) ? (xx - e0) / (e1 - e0) : 0.0;
                tt = std::fmin(1.0, std::fmax(0.0, tt));
                st[sp-1] = tt * tt * (3.0 - 2.0 * tt);
                break;
            }
            case PatOp::Noise:    { double zz = st[--sp], yy = st[--sp]; st[sp-1] = patValueNoise(st[sp-1], yy, zz); break; }
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
                st[sp-1] = c.texFn ? c.texFn(c.texSelf, (int)nd.a, st[sp-1], vv) : 0.0;
                break;
            }
            case PatOp::Spec:                               // one arg: the wavelength
                st[sp-1] = c.specFn ? c.specFn(c.specSelf, (int)nd.a, st[sp-1]) : 0.0;
                break;
            case PatOp::Grid: {
                int gi = (int)nd.a;
                // A resolved index always names a live table: the compiler accepts
                // `grid:` only where a PatTableScope was in scope, and that scope is the
                // same Scene these headers come from. So this guard can only fire if an
                // evaluation SITE forgot to bind them (patBindTables / bindPatData) — a
                // wiring bug, not an authoring one. Bail out of the whole program instead
                // of pushing a placeholder: the operand count is the table's own `ndim`,
                // which is exactly what can't be read here, so a push would leave the
                // stack unbalanced and quietly return a COORDINATE as the result.
                if (!c.grids || gi < 0 || gi >= c.nGrids) return 0.0;
                const PatGrid& g = c.grids[gi];
                int nd2 = g.ndim < 1 ? 1 : (g.ndim > PAT_ND_MAX_DIM ? PAT_ND_MAX_DIM : g.ndim);
                double co[PAT_ND_MAX_DIM];
                for (int k = nd2 - 1; k >= 0; --k) co[k] = st[--sp];  // pushed in axis order
                st[sp++] = patGridSample(g, c.dataPool, c.dataPoolN, co);
                break;
            }
            case PatOp::Scatter: {
                int si = (int)nd.a;
                if (!c.scatters || si < 0 || si >= c.nScatters) return 0.0;  // see Grid
                const PatScatter& sc = c.scatters[si];
                int nd2 = sc.ndim < 1 ? 1 : (sc.ndim > PAT_ND_MAX_DIM ? PAT_ND_MAX_DIM : sc.ndim);
                double co[PAT_ND_MAX_DIM];
                for (int k = nd2 - 1; k >= 0; --k) co[k] = st[--sp];  // pushed in axis order
                st[sp++] = patScatterSample(sc, c.dataPool, c.dataPoolN, co);
                break;
            }
            case PatOp::StReg:    reg[(int)nd.a] = st[sp-1]; break;   // save, keep on stack
            case PatOp::LdReg:    st[sp++] = reg[(int)nd.a]; break;   // reuse saved value
        }
    }
    return sp > 0 ? st[0] : 0.0;
}

// A named, storable pattern (owns its postfix program). Scene::patterns holds these.
struct Pattern {
    std::vector<PatNode> nodes;
    double eval(const PatCtx& c) const { return patternEval(nodes.data(), (int)nodes.size(), c); }
};

// ---------------------------------------------------------------------------
// (B) Expression compiler: infix math -> postfix, over the pattern variables.
// Supports: literals, constant `pi`; variables x y z f nx ny nz r u v; unary + -;
// binary + - * / % ^ (^ = pow, right-assoc); functions abs sqrt sin cos tan exp
// log floor fract sign saturate min max pow atan2 step clamp mix smoothstep noise;
// and `tex:<name>(u, v)` — the grayscale sample of a named image texture, so a photo
// can be a TERM inside a formula (`0.3 + 0.7*tex:grime(u,v)*sin(20*x)`).
// Returns false + fills `err` on a parse error. Shunting-yard with an operator and
// an output (postfix) queue; function arity is checked at the closing paren.
// ---------------------------------------------------------------------------
namespace pattern_detail {

struct Tok {
    enum Kind { Num, Var, Func, Op, LParen, RParen, Comma } kind;
    double num = 0;
    PatOp  var = PatOp::VarX;   // for Var
    std::string name;          // for Func
    char op = 0;               // for Op ('+','-','*','/','%','^','u' unary minus)
    int   texId = -1;          // for a `tex:<name>` Func: the resolved texture index
    int   tableId = -1;        // for a `grid:`/`scatter:<name>` Func: the table index
    int   tableDim = 0;        // ... and its dimensionality == the call's arity
};

inline bool isIdentStart(char c) { return std::isalpha((unsigned char)c) || c == '_'; }
inline bool isIdentCh(char c)    { return std::isalnum((unsigned char)c) || c == '_'; }

// Map an identifier to a variable opcode; returns false if it isn't a variable.
inline bool varOp(const std::string& s, PatOp& out) {
    if (s == "x")  { out = PatOp::VarX;  return true; }
    if (s == "y")  { out = PatOp::VarY;  return true; }
    if (s == "z")  { out = PatOp::VarZ;  return true; }
    if (s == "f")  { out = PatOp::VarF;  return true; }
    if (s == "nx") { out = PatOp::VarNx; return true; }
    if (s == "ny") { out = PatOp::VarNy; return true; }
    if (s == "nz") { out = PatOp::VarNz; return true; }
    if (s == "r")  { out = PatOp::VarR;  return true; }
    if (s == "u")  { out = PatOp::VarU;  return true; }
    if (s == "v")  { out = PatOp::VarV;  return true; }
    if (s == "a")  { out = PatOp::VarA;  return true; }   // albedo — resolved at load time
    return false;
}


// Function name -> (opcode, arity[, povId]). Returns false if not a known function.
// For exact POV-Ray internal functions (f_torus, f_heart, ...) out=PatOp::PovFn and
// povId is the POV internal id; for built-ins povId is left as -1.
inline bool texFuncName(const std::string& s) {
    return s.size() > 4 && s.compare(0, 4, "tex:") == 0;
}
inline bool gridFuncName(const std::string& s) {
    return s.size() > 5 && s.compare(0, 5, "grid:") == 0;
}
inline bool scatterFuncName(const std::string& s) {
    return s.size() > 8 && s.compare(0, 8, "scatter:") == 0;
}
inline bool specFuncName(const std::string& s) {
    return s.size() > 5 && s.compare(0, 5, "spec:") == 0;
}
// The `<kind>:` prefix length of an N-D table call, for stripping off the name.
inline size_t tablePrefixLen(PatTableKind k) { return k == PatTableKind::Grid ? 5 : 8; }
inline const char* tableKindName(PatTableKind k) { return k == PatTableKind::Grid ? "grid" : "scatter"; }

inline bool funcOp(const std::string& s, PatOp& out, int& arity, int& povId) {
    povId = -1;
    // `tex:<name>(u, v)` — sample a named image texture. The name is part of the
    // token (the tokenizer scans `tex:foo` as one identifier), so arity is fixed at
    // 2 and the index is resolved separately against the call site's PatTexScope.
    if (texFuncName(s)) { out = PatOp::Tex; arity = 2; return true; }
    // `spec:<name>(w)` — sample a named spectrum at a wavelength. Fixed arity 1 (a
    // spectrum is a function of one variable); the index resolves against the call
    // site's PatSpecScope, which only an upsample body supplies.
    if (specFuncName(s)) { out = PatOp::Spec; arity = 1; return true; }
    // `grid:<name>(c0, …)` / `scatter:<name>(c0, …)` — sample a named N-D table. Arity
    // is the TABLE's own ndim, which this name-only lookup cannot know, so it reports 0
    // and every caller that needs the real arity takes it from the token's resolved
    // `tableDim` instead.
    if (gridFuncName(s))    { out = PatOp::Grid;    arity = 0; return true; }
    if (scatterFuncName(s)) { out = PatOp::Scatter; arity = 0; return true; }
    struct F { const char* n; PatOp op; int ar; };
    static const F fs[] = {
        {"abs",PatOp::Abs,1},{"sqrt",PatOp::Sqrt,1},{"sin",PatOp::Sin,1},
        {"cos",PatOp::Cos,1},{"tan",PatOp::Tan,1},{"exp",PatOp::Exp,1},
        {"log",PatOp::Log,1},{"floor",PatOp::Floor,1},{"fract",PatOp::Fract,1},
        {"sign",PatOp::Sign,1},{"saturate",PatOp::Saturate,1},
        {"min",PatOp::Min,2},{"max",PatOp::Max,2},{"pow",PatOp::Pow,2},
        {"atan2",PatOp::Atan2,2},{"step",PatOp::Step,2},
        {"clamp",PatOp::Clamp,3},{"mix",PatOp::Mix,3},
        {"smoothstep",PatOp::Smoothstep,3},{"noise",PatOp::Noise,3},
    };
    for (const F& g : fs) if (s == g.n) { out = g.op; arity = g.ar; return true; }
    int id, ar;
    if (povFnLookup(s.c_str(), id, ar)) { out = PatOp::PovFn; arity = ar; povId = id; return true; }
    return false;
}
inline bool funcOp(const std::string& s, PatOp& out, int& arity) {
    int dummy; return funcOp(s, out, arity, dummy);
}

inline int opPrec(char c) {
    switch (c) {
        case 'u': return 5;              // unary minus
        case '^': return 4;
        case '*': case '/': case '%': return 3;
        case '+': case '-': return 2;
        default:  return 0;
    }
}
inline bool opRightAssoc(char c) { return c == '^' || c == 'u'; }
inline PatOp binOp(char c) {
    switch (c) {
        case '+': return PatOp::Add; case '-': return PatOp::Sub;
        case '*': return PatOp::Mul; case '/': return PatOp::Div;
        case '%': return PatOp::Mod; case '^': return PatOp::Pow;
        default:  return PatOp::Add;
    }
}

// Map an identifier to a variable opcode in the UPSAMPLE vocabulary. See PatVarMode:
// this vocabulary is disjoint from the surface one, and the opcode slots it borrows are
// unreachable by name here, so the reuse is invisible to any program.
inline bool upsampleVarOp(const std::string& s, PatOp& out) {
    if (s == "r") { out = PatOp::VarX; return true; }   // red   (linear sRGB)
    if (s == "g") { out = PatOp::VarY; return true; }   // green
    if (s == "b") { out = PatOp::VarZ; return true; }   // blue
    if (s == "w") { out = PatOp::VarU; return true; }   // wavelength, nm
    return false;
}

inline bool tokenize(const std::string& s, std::vector<Tok>& out, std::string& err,
                     bool allowT = false, const PatTexScope* tex = nullptr,
                     const PatTableScope* tables = nullptr, bool allowA = false,
                     PatVarMode vars = PatVarMode::Surface,
                     const PatSpecScope* specs = nullptr) {
    size_t i = 0, n = s.size();
    bool prevValue = false;   // was the previous token a value/RParen (for unary minus)
    while (i < n) {
        char c = s[i];
        if (std::isspace((unsigned char)c)) { ++i; continue; }
        if (std::isdigit((unsigned char)c) || (c == '.' && i + 1 < n && std::isdigit((unsigned char)s[i+1]))) {
            size_t j = i;
            while (j < n && (std::isdigit((unsigned char)s[j]) || s[j] == '.' ||
                             s[j] == 'e' || s[j] == 'E' ||
                             ((s[j] == '+' || s[j] == '-') && j > i && (s[j-1] == 'e' || s[j-1] == 'E'))))
                ++j;
            Tok t; t.kind = Tok::Num; t.num = std::atof(s.substr(i, j - i).c_str());
            out.push_back(t); i = j; prevValue = true; continue;
        }
        if (isIdentStart(c)) {
            size_t j = i;
            while (j < n && isIdentCh(s[j])) ++j;
            std::string id = s.substr(i, j - i);
            // A texture reference `tex:<name>` scans as ONE identifier, so a sample
            // reads as a plain two-argument call `tex:<name>(u, v)` and reuses the
            // very same `tex:` spelling that material slots already use for images.
            if ((id == "tex" || id == "grid" || id == "scatter" || id == "spec") &&
                j < n && s[j] == ':') {
                size_t e = j + 1;
                while (e < n && isIdentCh(s[e])) ++e;
                id = s.substr(i, e - i);
                j = e;
            }
            i = j;
            // skip spaces to see if a '(' follows -> function call
            size_t k = i; while (k < n && std::isspace((unsigned char)s[k])) ++k;
            bool isCall = (k < n && s[k] == '(');
            PatOp vop; PatOp fop; int ar;
            if (isCall && funcOp(id, fop, ar)) {
                Tok t; t.kind = Tok::Func; t.name = id;
                if (fop == PatOp::Tex) {
                    std::string nm = id.substr(4);
                    if (!tex) {
                        err = "texture sample '" + id + "' is out of scope here — a texture can only "
                              "be sampled where a scene texture table exists (material / pattern / "
                              "record expressions), not in an implicit field formula, a medium "
                              "density/ior program, or a load-time constant site";
                        return false;
                    }
                    t.texId = tex->resolve(nm.c_str());
                    if (t.texId < 0) { err = "unknown texture '" + nm + "' in " + id + "(u, v)"; return false; }
                }
                if (fop == PatOp::Spec) {
                    std::string nm = id.substr(5);
                    if (!specs) {
                        err = "spectrum sample '" + id + "' is out of scope here — a named "
                              "spectrum can only be sampled inside an `upsample` body, which is "
                              "the one place a wavelength is the free variable";
                        return false;
                    }
                    t.texId = specs->resolve(nm.c_str());
                    if (t.texId < 0) { err = "unknown spectrum '" + nm + "' in " + id + "(w) — "
                                             "`spec:` names a declared `spectrum \"<name>\"` block"; return false; }
                }
                if (fop == PatOp::Grid || fop == PatOp::Scatter) {
                    const PatTableKind kind = (fop == PatOp::Grid) ? PatTableKind::Grid
                                                                   : PatTableKind::Scatter;
                    const std::string what = tableKindName(kind);
                    std::string nm = id.substr(tablePrefixLen(kind));
                    if (!tables) {
                        err = what + " sample '" + id + "' is out of scope here — a " + what +
                              " can only be sampled where the scene's " + what + " tables are "
                              "reachable at evaluation time (material / pattern / record "
                              "expressions, isosurface and `function` field formulas, medium "
                              "density/ior programs, camera_curve drivers), not at a load-time "
                              "constant site";
                        return false;
                    }
                    t.tableId = tables->resolve(kind, nm.c_str(), t.tableDim);
                    if (t.tableId < 0) { err = "unknown " + what + " '" + nm + "' in " + id + "(...)"; return false; }
                }
                out.push_back(t);
                prevValue = false; continue;
            }
            if (texFuncName(id)) {
                err = "texture sample '" + id + "' must be called with coordinates, e.g. " + id + "(u, v)";
                return false;
            }
            if (specFuncName(id)) {
                err = "spectrum sample '" + id + "' must be called with a wavelength, e.g. " + id + "(w)";
                return false;
            }
            if (gridFuncName(id) || scatterFuncName(id)) {
                err = std::string(gridFuncName(id) ? "grid" : "scatter") + " sample '" + id +
                      "' must be called with one coordinate per axis, e.g. " + id + "(u)";
                return false;
            }
            if (id == "tex") { err = "texture sample needs a name: tex:<texture>(u, v)"; return false; }
            if (id == "spec") { err = "spectrum sample needs a name: spec:<spectrum>(w)"; return false; }
            if (id == "grid") { err = "grid sample needs a name: grid:<grid>(c0, ...)"; return false; }
            if (id == "scatter") { err = "scatter sample needs a name: scatter:<scatter>(c0, ...)"; return false; }
            if (id == "pi") {
                Tok t; t.kind = Tok::Num; t.num = 3.14159265358979323846;
                out.push_back(t); prevValue = true; continue;
            }
            // An UPSAMPLE body has a disjoint vocabulary (see PatVarMode), so it is decided
            // here — BEFORE the surface names below, which must not be reachable at all.
            // `pi` above is deliberately left shared: it is a constant, not a context.
            if (vars == PatVarMode::Upsample) {
                if (upsampleVarOp(id, vop)) {
                    Tok t; t.kind = Tok::Var; t.var = vop; out.push_back(t);
                    prevValue = true; continue;
                }
                // Name every surface variable explicitly rather than letting it fall through
                // to a bare "unknown identifier". The trap this guards is real and would
                // otherwise be silent-ish: `r` is RADIUS in the surface vocabulary and RED
                // here, so an author porting a formula between the two needs to be told the
                // spelling changed meaning, not just that something is unknown.
                PatOp dummy;
                if (varOp(id, dummy) || id == "t") {
                    err = "variable '" + id + "' is a surface/shading variable and has no meaning "
                          "in an `upsample` expression — there is no hit point here. An upsampler "
                          "sees only r, g, b (the colour, linear sRGB) and w (wavelength, nm)"
                          + (id == "r" ? std::string(" — note 'r' means RED here, not radius") : std::string());
                    return false;
                }
                err = "unknown identifier '" + id + "' — an `upsample` expression has r, g, b "
                      "(the colour, linear sRGB), w (wavelength in nm), pi, the usual functions, "
                      "and spec:<spectrum>(w)"; return false;
            }
            if (id == "t") {
                // The flyby timeline is in scope ONLY inside a camera_curve record track.
                // Everywhere else `t` is out of scope; report that specifically rather than
                // as a bare "unknown identifier" so the scope rule is legible.
                if (allowT) { Tok t; t.kind = Tok::Var; t.var = PatOp::VarT; out.push_back(t); prevValue = true; continue; }
                err = "variable 't' (flyby timeline) is only in scope inside a camera_curve "
                      "record track (fov_from/roll_from/zoom_from/fstop_from/focus_from)"; return false;
            }
            if (id == "a") {
                // The albedo input has no per-hit source, so unlike every other variable it
                // is only meaningful where a MATERIAL will later resolve it — bound at the
                // use site (`gold(a=0.3)`) or falling back to `albedo_default`. Outside that
                // scope it could only ever read as a silent 0, so say so instead.
                if (allowA) { Tok t; t.kind = Tok::Var; t.var = PatOp::VarA; out.push_back(t); prevValue = true; continue; }
                err = "input 'a' (albedo) is only in scope where a material can resolve it — "
                      "a pattern/texture-free expression in a `pattern` block, a record stop, "
                      "or a material slot"; return false;
            }
            if (varOp(id, vop)) {
                Tok t; t.kind = Tok::Var; t.var = vop; out.push_back(t);
                prevValue = true; continue;
            }
            err = "unknown identifier '" + id + "'"; return false;
        }
        if (c == '(') { Tok t; t.kind = Tok::LParen; out.push_back(t); ++i; prevValue = false; continue; }
        if (c == ')') { Tok t; t.kind = Tok::RParen; out.push_back(t); ++i; prevValue = true;  continue; }
        if (c == ',') { Tok t; t.kind = Tok::Comma;  out.push_back(t); ++i; prevValue = false; continue; }
        if (c == '+' || c == '-' || c == '*' || c == '/' || c == '%' || c == '^') {
            Tok t; t.kind = Tok::Op;
            t.op = (c == '-' && !prevValue) ? 'u' : c;   // leading '-' => unary
            if (c == '+' && !prevValue) { ++i; prevValue = false; continue; }  // unary + is a no-op
            out.push_back(t); ++i; prevValue = false; continue;
        }
        err = std::string("unexpected character '") + c + "'"; return false;
    }
    return true;
}

}  // namespace pattern_detail

// ---------------------------------------------------------------------------
// Named inputs: introspection + binding by substitution (ROADMAP_records.md §3.2/§3.3)
// ---------------------------------------------------------------------------
// A property is an expression over NAMED INPUTS, and "nothing is ever closed": any
// input can be rebound where the property is used. The mechanism is pure substitution,
// which postfix makes a straight SPLICE — a variable node is a leaf that pushes exactly
// one value, and so is a well-formed program, so swapping one for the other cannot
// disturb the surrounding stack discipline.
//
// The consequence is that a bound material is an ORDINARY material: its programs are
// concrete formulas in real per-hit variables, with no environment, no closure and no
// runtime indirection. Everything downstream — the CPU evaluator, the verbatim GPU
// upload, patternHasFreeVars, the record samplers — keeps working untouched, and a
// material nobody binds is bit-identical to before. That is what makes this feature
// additive rather than a re-plumbing of the shading path.

// Input NAME -> opcode. Published out of pattern_detail because binding a material's
// inputs at a use site (`gold(u=v)`) has to resolve an author-written name.
using pattern_detail::varOp;

// The name of a bindable input, or nullptr if `op` is not a variable. Inverse of varOp.
inline const char* varName(PatOp op) {
    switch (op) {
        case PatOp::VarX:  return "x";   case PatOp::VarY:  return "y";
        case PatOp::VarZ:  return "z";   case PatOp::VarF:  return "f";
        case PatOp::VarNx: return "nx";  case PatOp::VarNy: return "ny";
        case PatOp::VarNz: return "nz";  case PatOp::VarR:  return "r";
        case PatOp::VarU:  return "u";   case PatOp::VarV:  return "v";
        case PatOp::VarA:  return "a";   default: return nullptr;
    }
}

// True if `prog` reads the input `var` anywhere.
inline bool patternUsesVar(const std::vector<PatNode>& prog, PatOp var) {
    for (const PatNode& nd : prog) if (nd.op == var) return true;
    return false;
}

// Every input `prog` reads, appended to `out` without duplicates (order of first use, so
// diagnostics list inputs the way the author wrote them).
inline void patternCollectVars(const std::vector<PatNode>& prog, std::vector<PatOp>& out) {
    for (const PatNode& nd : prog) {
        if (!varName(nd.op)) continue;
        bool seen = false;
        for (PatOp o : out) if (o == nd.op) { seen = true; break; }
        if (!seen) out.push_back(nd.op);
    }
}

// One input binding: replace every read of `var` with the program `repl`.
struct PatBind {
    PatOp                var;
    std::vector<PatNode> repl;
};

// Rewrite `prog`, splicing each bound input's replacement in place of its variable node.
// Substitution is SIMULTANEOUS, not sequential: a replacement's own variable nodes are
// copied through untouched, so `gold(u=v, v=u)` swaps the two inputs instead of
// collapsing both to `u`. Bindings whose input never appears cost nothing.
inline std::vector<PatNode> patternSubstitute(const std::vector<PatNode>& prog,
                                              const std::vector<PatBind>& binds) {
    bool any = false;
    for (const PatBind& b : binds) if (patternUsesVar(prog, b.var)) { any = true; break; }
    if (!any) return prog;
    std::vector<PatNode> out;
    out.reserve(prog.size());
    for (const PatNode& nd : prog) {
        const std::vector<PatNode>* repl = nullptr;
        for (const PatBind& b : binds) if (b.var == nd.op) { repl = &b.repl; break; }
        if (repl) out.insert(out.end(), repl->begin(), repl->end());
        else      out.push_back(nd);
    }
    return out;
}

// Compile an infix expression string into a postfix pattern program.
// `allowT` publishes the flyby-timeline variable `t` as in-scope (camera_curve record
// tracks only). Default false: every other call site keeps `t` an out-of-scope error,
// so a surface/constant driver can never silently read the timeline.
// `tex` publishes the scene's named image textures for `tex:<name>(u, v)` samples;
// null (the default) makes any such sample a scope error — see PatTexScope.
// `tables` does the same for the N-D datatypes `grid:<name>(…)` / `scatter:<name>(…)`.
// `allowA` publishes the albedo input `a` (ROADMAP_records.md §3.2) — the one named input
// with no per-hit source, so it is only in scope where a material will resolve it at load
// time. It is LAST in the list on purpose: defaulting to false keeps every existing call
// site's scope exactly as it was, so only the handful of material-reachable sites opt in.
inline bool compilePatternExpr(const std::string& expr, std::vector<PatNode>& out,
                               std::string& err, bool allowT = false,
                               const PatTexScope* tex = nullptr,
                               const PatTableScope* tables = nullptr,
                               bool allowA = false,
                               PatVarMode vars = PatVarMode::Surface,
                               const PatSpecScope* specs = nullptr) {
    using namespace pattern_detail;
    std::vector<Tok> toks;
    if (!tokenize(expr, toks, err, allowT, tex, tables, allowA, vars, specs)) return false;

    std::vector<PatNode> queue;             // output (postfix)
    std::vector<Tok>     ops;               // operator stack (Op / Func / LParen)
    std::vector<int>     argCount;          // arg counter per open function
    std::vector<bool>    wasFunc;           // parallel to LParen: did a func precede it

    auto emitOp = [&](const Tok& t) {
        if (t.kind == Tok::Op) {
            if (t.op == 'u') { PatNode nd; nd.op = PatOp::Neg; queue.push_back(nd); }
            else             { PatNode nd; nd.op = binOp(t.op); queue.push_back(nd); }
        } else if (t.kind == Tok::Func) {
            PatOp op; int ar; int povId; funcOp(t.name, op, ar, povId);
            PatNode nd; nd.op = op;
            if      (op == PatOp::PovFn) nd.a = (double)povId;
            else if (op == PatOp::Tex)   nd.a = (double)t.texId;    // resolved at tokenize
            else if (op == PatOp::Spec)  nd.a = (double)t.texId;    // shares the resolved-index slot
            else if (op == PatOp::Grid || op == PatOp::Scatter)
                                         nd.a = (double)t.tableId;  // resolved at tokenize
            queue.push_back(nd);
        }
    };

    for (size_t ti = 0; ti < toks.size(); ++ti) {
        const Tok& t = toks[ti];
        switch (t.kind) {
            case Tok::Num:  { PatNode nd; nd.op = PatOp::Const; nd.a = t.num; queue.push_back(nd); break; }
            case Tok::Var:  { PatNode nd; nd.op = t.var; queue.push_back(nd); break; }
            case Tok::Func: ops.push_back(t); break;
            case Tok::Comma: {
                while (!ops.empty() && ops.back().kind != Tok::LParen) { emitOp(ops.back()); ops.pop_back(); }
                if (ops.empty()) { err = "misplaced comma"; return false; }
                if (!argCount.empty()) argCount.back()++;
                break;
            }
            case Tok::Op: {
                while (!ops.empty() && ops.back().kind == Tok::Op) {
                    char o2 = ops.back().op;
                    if ((opRightAssoc(t.op) && opPrec(t.op) < opPrec(o2)) ||
                        (!opRightAssoc(t.op) && opPrec(t.op) <= opPrec(o2))) {
                        emitOp(ops.back()); ops.pop_back();
                    } else break;
                }
                ops.push_back(t);
                break;
            }
            case Tok::LParen: {
                ops.push_back(t);
                bool fn = (!ops.empty() && ops.size() >= 2 && ops[ops.size()-2].kind == Tok::Func);
                wasFunc.push_back(fn);
                argCount.push_back(1);
                break;
            }
            case Tok::RParen: {
                while (!ops.empty() && ops.back().kind != Tok::LParen) { emitOp(ops.back()); ops.pop_back(); }
                if (ops.empty()) { err = "mismatched ')'"; return false; }
                ops.pop_back();   // discard LParen
                int args = argCount.empty() ? 0 : argCount.back();
                bool fn = wasFunc.empty() ? false : wasFunc.back();
                if (!argCount.empty()) argCount.pop_back();
                if (!wasFunc.empty())  wasFunc.pop_back();
                if (!ops.empty() && ops.back().kind == Tok::Func) {
                    PatOp op; int ar; funcOp(ops.back().name, op, ar);
                    // An N-D table's arity is its own dimensionality, not a property of
                    // the name, so it comes from the token resolved at tokenize time.
                    if (op == PatOp::Grid || op == PatOp::Scatter) ar = ops.back().tableDim;
                    if (fn && args != ar) {
                        err = "function '" + ops.back().name + "' expects " + std::to_string(ar) +
                              " arg(s), got " + std::to_string(args); return false;
                    }
                    emitOp(ops.back()); ops.pop_back();
                }
                break;
            }
        }
    }
    while (!ops.empty()) {
        if (ops.back().kind == Tok::LParen) { err = "mismatched '('"; return false; }
        emitOp(ops.back()); ops.pop_back();
    }
    if (queue.empty()) { err = "empty expression"; return false; }
    out = std::move(queue);
    return true;
}

// True if a compiled pattern program needs a per-hit shading context — either a
// surface intrinsic (x y z f nx ny nz r u v) or a texture sample (which needs the
// scene's texture table, present only while shading). Used by value sites that must
// be load-time constant (records stage 5a scope check): a constant site has no such
// context, so it admits only var-free drivers — `R.chan(u)` or a `tex:` sample there
// is a scope error rather than a silent zero.
inline bool patternHasFreeVars(const std::vector<PatNode>& prog) {
    for (const PatNode& nd : prog)
        if ((nd.op >= PatOp::VarX && nd.op <= PatOp::VarV) ||
            nd.op == PatOp::Tex || nd.op == PatOp::Grid ||
            nd.op == PatOp::Scatter) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Common-subexpression elimination over a compiled postfix program.
//
// Machine-generated field formulas (rotated/composed gyroids, exported CSG) repeat
// whole subtrees many times — the same rotated coordinate feeds three sin/cos pairs,
// the same argument arithmetic is spelled out per call. Since the pattern VM is the
// inner loop of the implicit-surface sphere-trace (every march step and every shadow
// ray bottoms out in patternEval / dPatternEval[F]), collapsing those repeats is a
// direct hot-path win on BOTH backends: the optimized program is what gets uploaded.
//
// Method: simulate the postfix stack, hash-consing every (op, payload, children)
// node into a DAG. Any interior node the DAG reaches >= 2 times becomes a register
// candidate; the top PAT_CSE_REGS by saved-work get a register. The program is then
// re-emitted by memoized post-order DFS: the first occurrence of a registered
// subtree is followed by StReg r (save, keep on stack); every later occurrence
// becomes a single LdReg r.
//
// WHY THIS IS EXACTLY SAFE (bit-identical, both precisions): every PatOp is a pure
// function of its operands, its payload and the fixed PatCtx, so re-executing an
// identical subtree within one evaluation yields identical bits — a register load
// returns precisely what the recomputation would have. No arithmetic is folded,
// reordered or changed; the emission order of the surviving nodes is the original
// post-order. The optimizer additionally refuses anything it cannot prove out:
// unknown arities (Grid/Scatter — table-dimension arity), already-optimized
// programs, malformed stacks, and as a final belt-and-braces check it bit-compares
// original vs optimized at a set of probe points before committing.
// ---------------------------------------------------------------------------

// Stack effect of one node: how many operands it pops and pushes. False when the
// arity cannot be read from the node alone (Grid/Scatter: the arity is the named
// table's own dimensionality, which lives outside the program).
inline bool patOpStackEffect(PatOp op, double a, int& pops, int& pushes) {
    pushes = 1;
    if (op >= PatOp::Const && op <= PatOp::VarT)  { pops = 0; return true; }
    if (op == PatOp::VarA)                        { pops = 0; return true; }
    if (op >= PatOp::Neg && op <= PatOp::Saturate){ pops = 1; return true; }
    if (op >= PatOp::Add && op <= PatOp::Step)    { pops = 2; return true; }
    if (op >= PatOp::Clamp && op <= PatOp::Noise) { pops = 3; return true; }
    if (op == PatOp::PovFn)                       { pops = povFnArity((int)a); return true; }
    if (op == PatOp::Tex)                         { pops = 2; return true; }
    if (op == PatOp::Spec)                        { pops = 1; return true; }
    if (op == PatOp::StReg)                       { pops = 0; pushes = 0; return true; }  // peeks
    if (op == PatOp::LdReg)                       { pops = 0; pushes = 1; return true; }
    return false;   // Grid / Scatter / anything future: bail out of optimizing
}

inline void patternOptimizeCSE(std::vector<PatNode>& prog) {
    const int n = (int)prog.size();
    if (n < 4) return;   // nothing worth sharing
    // ---- 1. postfix -> hash-consed DAG --------------------------------------
    struct DagNode { PatOp op; double a; std::vector<int> kids; int size; };
    std::vector<DagNode> dag; dag.reserve(n);
    std::unordered_map<std::string, int> cons;
    std::vector<int> stk;
    for (int i = 0; i < n; ++i) {
        const PatNode& nd = prog[i];
        if (nd.op == PatOp::StReg || nd.op == PatOp::LdReg) return;  // already optimized
        int pops, pushes;
        if (!patOpStackEffect(nd.op, nd.a, pops, pushes)) return;    // unknown arity
        if (pops > (int)stk.size()) return;                          // malformed program
        DagNode dn; dn.op = nd.op; dn.a = nd.a;
        dn.kids.assign(stk.end() - pops, stk.end());
        stk.resize(stk.size() - pops);
        dn.size = 1;
        for (int c : dn.kids) dn.size += dag[c].size;
        std::string key; key.reserve(12 + dn.kids.size() * 4);
        int opi = (int)dn.op;           key.append((const char*)&opi, 4);
        std::uint64_t ab; std::memcpy(&ab, &dn.a, 8); key.append((const char*)&ab, 8);
        for (int c : dn.kids)           key.append((const char*)&c, 4);
        auto it = cons.find(key);
        int id;
        if (it != cons.end()) id = it->second;
        else { id = (int)dag.size(); dag.push_back(std::move(dn)); cons.emplace(std::move(key), id); }
        stk.push_back(id);
    }
    if (stk.size() != 1) return;   // not a single-rooted expression: leave untouched
    const int root = stk[0];
    // ---- 2. edge-multiplicity (uses) over the DAG reachable from the root ----
    std::vector<int>  uses(dag.size(), 0);
    std::vector<char> reach(dag.size(), 0);
    {
        std::vector<int> w{root};
        while (!w.empty()) {
            int u2 = w.back(); w.pop_back();
            if (reach[u2]) continue;
            reach[u2] = 1;
            for (int c : dag[u2].kids) w.push_back(c);
        }
        for (int i = 0; i < (int)dag.size(); ++i)
            if (reach[i]) for (int c : dag[i].kids) uses[c]++;
    }
    // ---- 3. register allocation: interior nodes reached >= 2 times -----------
    struct Cand { int id; long long benefit; };
    std::vector<Cand> cands;
    for (int i = 0; i < (int)dag.size(); ++i)
        if (reach[i] && uses[i] >= 2 && !dag[i].kids.empty())
            cands.push_back({i, (long long)(uses[i] - 1) * dag[i].size});
    if (cands.empty()) return;     // no shared interior subtrees: nothing to do
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        if (a.benefit != b.benefit) return a.benefit > b.benefit;
        return a.id < b.id;        // deterministic tie-break
    });
    std::vector<int> regOf(dag.size(), -1);
    int nextReg = 0;
    for (const Cand& c : cands) {
        if (nextReg >= PAT_CSE_REGS) break;
        regOf[c.id] = nextReg++;
    }
    // ---- 4. re-emit: memoized post-order DFS ---------------------------------
    std::vector<PatNode> out; out.reserve(n);
    std::vector<char> emitted(dag.size(), 0);
    struct Frame { int id; int next; };
    std::vector<Frame> work; work.push_back({root, 0});
    while (!work.empty()) {
        Frame& fr = work.back();          // valid until the next push_back below
        const int id = fr.id;
        if (fr.next == 0 && emitted[id] && regOf[id] >= 0) {
            PatNode ld; ld.op = PatOp::LdReg; ld.a = (double)regOf[id];
            out.push_back(ld);
            work.pop_back();
            continue;
        }
        if (fr.next < (int)dag[id].kids.size()) {
            int c = dag[id].kids[fr.next];
            ++fr.next;                    // mutate BEFORE push_back may reallocate
            work.push_back({c, 0});
            continue;                     // fr is dead past this point
        }
        PatNode nd2; nd2.op = dag[id].op; nd2.a = dag[id].a;
        out.push_back(nd2);
        if (regOf[id] >= 0) {
            PatNode st2; st2.op = PatOp::StReg; st2.a = (double)regOf[id];
            out.push_back(st2);
            emitted[id] = 1;
        }
        work.pop_back();
    }
    if ((int)out.size() >= n) return;   // no shrink (all candidates beyond the reg cap)
    // ---- 5. safety: re-simulate the emitted program (depth + balance) --------
    {
        int sp = 0, maxSp = 0;
        for (const PatNode& nd : out) {
            int pops, pushes;
            if (!patOpStackEffect(nd.op, nd.a, pops, pushes)) return;
            if (pops > sp) return;
            sp += pushes - pops;
            if (sp > maxSp) maxSp = sp;
        }
        if (sp != 1 || maxSp > 64) return;   // evaluator stack is 64 deep
    }
    // ---- 6. belt-and-braces: bit-compare original vs optimized ---------------
    static const double probe[6][11] = {
        // x      y      z      f    nx     ny     nz     r      u      v      t
        { 0.31, -1.27,  2.63, 0.05, 0.27,  0.53, -0.80, 2.93,  0.37,  0.71, 0.13},
        {-2.11,  0.04, -0.57, -0.2, -0.7,  0.10,  0.70, 2.19,  0.93,  0.08, 0.77},
        { 5.02,  3.33, -4.19, 1.30, 0.57, -0.57,  0.59, 7.25,  0.11,  0.99, 0.42},
        {-0.02, -0.03,  0.01, 0.00, 0.00,  1.00,  0.00, 0.04,  0.50,  0.50, 0.00},
        { 12.7, -8.31,  0.66, -3.1, 0.80,  0.00, -0.60, 15.2,  0.66,  0.25, 1.00},
        {-0.99,  0.98, -0.97, 0.42, -0.5,  0.50, -0.70, 1.70,  0.01,  0.02, 0.55},
    };
    for (int p = 0; p < 6; ++p) {
        PatCtx c;
        c.x = probe[p][0]; c.y = probe[p][1]; c.z  = probe[p][2]; c.f = probe[p][3];
        c.nx = probe[p][4]; c.ny = probe[p][5]; c.nz = probe[p][6]; c.r = probe[p][7];
        c.u = probe[p][8]; c.v = probe[p][9]; c.t  = probe[p][10];
        double v0 = patternEval(prog.data(), n, c);
        double v1 = patternEval(out.data(), (int)out.size(), c);
        if (std::memcmp(&v0, &v1, 8) != 0) return;   // should be unreachable
    }
    prog = std::move(out);
}

// ---------------------------------------------------------------------------
// (A) Built-in generators -> postfix. Each returns a ready pattern program.
// These are thin sugar over the expression VM so both front-ends share evaluation.
// ---------------------------------------------------------------------------
namespace pattern_gen {

// helper builders
inline void pushConst(std::vector<PatNode>& o, double v) { PatNode n; n.op = PatOp::Const; n.a = v; o.push_back(n); }
inline void pushOp(std::vector<PatNode>& o, PatOp op)     { PatNode n; n.op = op; o.push_back(n); }

// axis: a coordinate (x|y|z) remapped by scale/offset -> value = coord*scale+offset
inline std::vector<PatNode> axis(PatOp coord, double scale, double offset) {
    std::vector<PatNode> o;
    pushOp(o, coord); pushConst(o, scale); pushOp(o, PatOp::Mul);
    pushConst(o, offset); pushOp(o, PatOp::Add);
    return o;
}

// radial: |p - center| * scale (a spherical gradient about `center`)
inline std::vector<PatNode> radial(const Vec3& c, double scale) {
    std::vector<PatNode> o;
    // sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2) * scale
    pushOp(o, PatOp::VarX); pushConst(o, c.x); pushOp(o, PatOp::Sub);
    pushConst(o, 2.0); pushOp(o, PatOp::Pow);
    pushOp(o, PatOp::VarY); pushConst(o, c.y); pushOp(o, PatOp::Sub);
    pushConst(o, 2.0); pushOp(o, PatOp::Pow); pushOp(o, PatOp::Add);
    pushOp(o, PatOp::VarZ); pushConst(o, c.z); pushOp(o, PatOp::Sub);
    pushConst(o, 2.0); pushOp(o, PatOp::Pow); pushOp(o, PatOp::Add);
    pushOp(o, PatOp::Sqrt);
    pushConst(o, scale); pushOp(o, PatOp::Mul);
    return o;
}

// bands: 0.5 + 0.5*sin(2*pi*freq * (coord) + phase) -> smooth stripes in [0,1]
inline std::vector<PatNode> bands(PatOp coord, double freq, double phase) {
    std::vector<PatNode> o;
    pushOp(o, coord); pushConst(o, 2.0 * 3.14159265358979323846 * freq); pushOp(o, PatOp::Mul);
    pushConst(o, phase); pushOp(o, PatOp::Add); pushOp(o, PatOp::Sin);
    pushConst(o, 0.5); pushOp(o, PatOp::Mul); pushConst(o, 0.5); pushOp(o, PatOp::Add);
    return o;
}

// checker: 3-D checkerboard parity in {0,1} at the given cell size.
// value = mod(floor(x/s)+floor(y/s)+floor(z/s), 2)
inline std::vector<PatNode> checker(double size) {
    std::vector<PatNode> o;
    double inv = (size != 0.0) ? 1.0 / size : 1.0;
    pushOp(o, PatOp::VarX); pushConst(o, inv); pushOp(o, PatOp::Mul); pushOp(o, PatOp::Floor);
    pushOp(o, PatOp::VarY); pushConst(o, inv); pushOp(o, PatOp::Mul); pushOp(o, PatOp::Floor); pushOp(o, PatOp::Add);
    pushOp(o, PatOp::VarZ); pushConst(o, inv); pushOp(o, PatOp::Mul); pushOp(o, PatOp::Floor); pushOp(o, PatOp::Add);
    pushConst(o, 2.0); pushOp(o, PatOp::Mod);
    return o;
}

// noise: value noise of (p*freq) in [0,1].
inline std::vector<PatNode> noise(double freq) {
    std::vector<PatNode> o;
    pushOp(o, PatOp::VarX); pushConst(o, freq); pushOp(o, PatOp::Mul);
    pushOp(o, PatOp::VarY); pushConst(o, freq); pushOp(o, PatOp::Mul);
    pushOp(o, PatOp::VarZ); pushConst(o, freq); pushOp(o, PatOp::Mul);
    pushOp(o, PatOp::Noise);
    return o;
}

// field: the raw implicit field value f (near 0 on a surface; useful volumetrically).
inline std::vector<PatNode> field(double scale) {
    std::vector<PatNode> o;
    pushOp(o, PatOp::VarF); pushConst(o, scale); pushOp(o, PatOp::Mul);
    return o;
}

}  // namespace pattern_gen
