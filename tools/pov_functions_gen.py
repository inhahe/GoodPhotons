#!/usr/bin/env python3
"""Generate src/pov_functions.h from POV-Ray's source/vm/fnintern.cpp.

We port the *pure-algebraic* internal isosurface functions verbatim (exact POV
formulas) into a single host+device evaluator `povFnEval(id, p)`, where
p[0..2] = x,y,z and p[3..] = the P0.. parameters.  Noise-/pattern-/pigment-
based functions (which need POV's Perlin noise + pattern engine) are excluded
and listed in EXCLUDE; the parser simply won't register them.

Reproducible: the POV-Ray source is fetched from GitHub (pinned raw URL) and
cached next to this script.  Run `python tools/pov_functions_gen.py`.
"""
import re, sys, os, urllib.request

# POV-Ray master; the internal-function table has been stable for years.
SRC_URL = ("https://raw.githubusercontent.com/POV-Ray/povray/master/"
           "source/vm/fnintern.cpp")
CACHE = os.path.join(os.path.dirname(__file__), ".pov_fnintern.cpp")
OUT = os.path.join(os.path.dirname(__file__), "..", "src", "pov_functions.h")

# internal ids that depend on POV's pattern / pigment / spline engines (not ported).
# The noise-based functions (29, 58, 59, 76, 78) ARE supported now via the exact
# povNoise() port in src/pov_noise.h; they use MANUAL_BODIES below (their POV source
# uses Vector3d / private_data caching that the plain text transform can't handle).
# Only f_pattern (77) stays excluded - it needs POV's full pattern/warp engine.
EXCLUDE = {77}

# Hand-written, POV-faithful bodies for the noise functions: same algebra as
# source/vm/fnintern.cpp, but with Vector3d ops spelled out on scalars and
# Noise(V, ngen) -> povNoise(x,y,z, ngen) (from pov_noise.h).  f_noise3d uses the
# scene default generator, which POV initializes to kNoiseGen_RangeCorrected (2).
MANUAL_BODIES = {
    76: """        // f_noise3d: POV Noise() with the scene-default generator (RangeCorrected=2)
        return povNoise(PARAM_X, PARAM_Y, PARAM_Z, 2);""",
    78: """        // f_noise_generator: generator selected by P0 (& 3)
        int ngen = (int)PARAM(0) & 3;
        return povNoise(PARAM_X, PARAM_Y, PARAM_Z, ngen);""",
    58: """        // f_ridge
        double px = PARAM_X, py = PARAM_Y, pz = PARAM_Z;
        int ngen = (int)PARAM(5) & 3;
        double Lambda = PARAM(0), l = Lambda;
        int Octaves = (int)PARAM(1);
        double Omega = PARAM(2), o = Omega;
        double off = PARAM(3), ridge = PARAM(4);
        double rscale = 1.0 / fmax(ridge, 1.0 - ridge);
        double scale  = 1.0 / fmax(off, 1.0 - off);
        double resid = PARAM(1) - (double)Octaves;
        double v = fabs(povNoise(px, py, pz, ngen) - ridge) * rscale;
        double value = (v - off);
        double tot = 1.0;
        for (int i = 2; i <= Octaves; i++) {
            double tx = px * l, ty = py * l, tz = pz * l;
            v = fabs(povNoise(tx, ty, tz, ngen) - ridge) * rscale;
            value += o * (v - off);
            tot += o; l *= Lambda; o *= Omega;
        }
        if (resid != 0.0) {
            double tx = px * l, ty = py * l, tz = pz * l;
            v = fabs(povNoise(tx, ty, tz, ngen) - ridge) * rscale;
            value += o * (v - off) * resid;
            tot += o * resid;
        }
        return value * scale / tot;""",
    59: """        // f_ridged_mf (exponent array computed inline instead of cached)
        double px = PARAM_X, py = PARAM_Y, pz = PARAM_Z;
        int ngen = (int)PARAM(5) & 3;
        double H = PARAM(0), Lambda = PARAM(1), offset = PARAM(3), gain = PARAM(4);
        double eastep = pow(Lambda, -H), eacur = 1.0;
        double signal = povNoise(px, py, pz, ngen) * 2.0 - 1.0;
        if (signal < 0.0) signal = -signal;
        signal = offset - signal; signal *= signal;
        double result = signal, weight = 1.0;
        for (int i = 1; i < PARAM(2); i++) {
            px *= Lambda; py *= Lambda; pz *= Lambda;
            weight = signal * gain;
            if (weight > 1.0) weight = 1.0;
            if (weight < 0.0) weight = 0.0;
            signal = povNoise(px, py, pz, ngen) * 2.0 - 1.0;
            if (signal < 0.0) signal = -signal;
            signal = offset - signal; signal *= signal; signal *= weight;
            eacur *= eastep;
            result += signal * eacur;
        }
        return result;""",
    29: """        // f_hetero_mf
        double vx = PARAM_X, vy = PARAM_Y, vz = PARAM_Z;
        int ngen = (int)PARAM(5) & 3;
        double signal = (povNoise(vx, vy, vz, ngen) * 2.0 - 1.0) + PARAM(3);
        vx *= PARAM(1); vy *= PARAM(1); vz *= PARAM(1);
        double p1_2_mp0 = pow(PARAM(1), -PARAM(0)), ea = p1_2_mp0;
        for (int i = 1; i < PARAM(2); i++) {
            double inc = ((povNoise(vx, vy, vz, ngen) * 2.0 - 1.0) + PARAM(3)) * ea;
            for (int q = (int)PARAM(4); q > 0; --q) inc *= signal;
            signal += inc;
            vx *= PARAM(1); vy *= PARAM(1); vz *= PARAM(1);
            ea *= p1_2_mp0;
        }
        double rem = PARAM(2) - (int)PARAM(2);
        if (rem != 0.0) {
            double inc = ((povNoise(vx, vy, vz, ngen) * 2.0 - 1.0) + PARAM(3)) * ea;
            signal += rem * inc * signal;
        }
        return signal;""",
}

if not os.path.exists(CACHE):
    print(f"fetching {SRC_URL}")
    urllib.request.urlretrieve(SRC_URL, CACHE)
src = open(CACHE, "r", encoding="utf-8", errors="replace").read()

# --- parse the trap table for (id -> total arg count) --------------------------
# lines like:  { f_torus,                   2 + 3 }, // 70
# Restrict to the main POVFPU_TrapTable (the S-table reuses ids 0,1,2 for
# f_pigment/f_transform/f_spline, which would otherwise clobber our entries).
_ts = src.index("POVFPU_TrapTable[]")
_te = src.index("POVFPU_TrapSTable[]")
trap_region = src[_ts:_te]
arity = {}
name_by_id = {}
for m in re.finditer(r"\{\s*(f_\w+)\s*,\s*(\d+)\s*\+\s*3\s*\}\s*,\s*//\s*(\d+)", trap_region):
    nm, np_, idn = m.group(1), int(m.group(2)), int(m.group(3))
    arity[idn] = np_ + 3
    name_by_id[idn] = nm

# --- parse each function body --------------------------------------------------
# DBL f_name(FPUContext *ctx, DBL *ptr, unsigned int[ fn]) // N \n { ... }
bodies = {}
pat = re.compile(
    r"DBL\s+(f_\w+)\s*\(FPUContext\s*\*ctx,\s*DBL\s*\*ptr,\s*unsigned int(?:\s+\w+)?\)\s*//\s*(\d+)\s*")
for m in pat.finditer(src):
    idn = int(m.group(2))
    # find the brace block following
    i = src.index("{", m.end())
    depth = 0
    j = i
    while j < len(src):
        if src[j] == "{": depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    body = src[i+1:j]
    bodies[idn] = body

def transform(body):
    b = body
    b = b.replace("(DBL)", "(double)")
    b = re.sub(r"\bDBL\b", "double", b)
    # std::min/std::max (doubles) -> fmin/fmax (device-safe)
    b = re.sub(r"\bmin\(", "fmin(", b)
    b = re.sub(r"\bmax\(", "fmax(", b)
    return b

supported = sorted(i for i in bodies if i not in EXCLUDE and i in arity)

# emit the switch
cases = []
for idn in supported:
    nm = name_by_id[idn]
    if idn in MANUAL_BODIES:
        body = MANUAL_BODIES[idn].strip("\n")
    else:
        body = transform(bodies[idn]).strip("\n")
    cases.append(f"    case {idn}: {{ // {nm}\n{body}\n    }}")
switch = "\n".join(cases)

# emit the name/arity table
tbl = []
for idn in supported:
    tbl.append(f'    {{ "{name_by_id[idn]}", {idn}, {arity[idn]} }},')
table = "\n".join(tbl)

# emit the device-safe arity-by-id switch
acases = []
for idn in supported:
    acases.append(f"    case {idn}: return {arity[idn]}; // {name_by_id[idn]}")
arity_switch = "\n".join(acases)

excluded_list = ", ".join(f"{name_by_id.get(i,'?')}({i})" for i in sorted(EXCLUDE))

TEMPLATE = r'''// pov_functions.h - EXACT ports of POV-Ray's internal isosurface functions.
//
// AUTO-GENERATED by tools/pov_functions_gen.py from POV-Ray source
// (source/vm/fnintern.cpp).  Do not edit by hand; re-run the generator.
//
// Each function is POV internal(N).  In SDL you call them as
//   f_name(x, y, z, P0, P1, ...)
// i.e. the first three args are the coordinates and the rest are parameters.
// Here povFnEval(id, p) takes p[0..2]=x,y,z and p[3..]=P0.. and returns the
// same DBL the POV function returns, using the identical algebra.
//
// The same evaluator runs on the CPU (pattern.h / patternEval) and the GPU
// (render_cuda.cu / dPatternEval), so results agree bit-for-bit across backends.
//
// EXCLUDED (need POV's Perlin-noise / pattern / pigment / spline engine, not yet
// ported): {excluded_list}.
#pragma once

#if defined(__CUDACC__)
  #ifndef POV_HD
  #define POV_HD __host__ __device__
  #endif
#else
  #ifndef POV_HD
  #define POV_HD
  #endif
#endif

#include <math.h>
// povNoise() - exact host+device port of POV's Perlin Noise(), used by the
// noise-based internal functions (f_noise3d, f_noise_generator, f_ridge,
// f_ridged_mf, f_hetero_mf).
#include "pov_noise.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_180
#define M_PI_180 (M_PI / 180.0)
#endif
#ifndef TWO_M_PI
#define TWO_M_PI (2.0 * M_PI)
#endif

// POV's ROT2D "surface of revolution" switch, verbatim (operates on the mutable
// PARAM_X/PARAM_Y locals and recomputes x2,y2).  p>0 enables the revolve.
#define ROT2D(p,d,ang) if (p>0) {x2=sqrt(x2+PARAM_Z*PARAM_Z)- d; th=(ang)*M_PI_180; \\
    if (th!=0){{ PARAM_X= x2*cos(th)-PARAM_Y*sin(th); PARAM_Y= x2*sin(th)+PARAM_Y*cos(th);}} else PARAM_X=x2; \\
    x2=PARAM_X*PARAM_X; y2=PARAM_Y*PARAM_Y;}}

// The largest arity (f_helical_torus) is 13 (3 coords + 10 params).
#define POV_FN_MAX_ARGS 13

// Evaluate POV internal function `id` with args _pp[] (_pp[0..2]=x,y,z,
// _pp[3..]=P0..).  The param is named `_pp` (not `p`) to avoid colliding with POV
// bodies that declare a local `p` (e.g. f_superellipsoid).
POV_HD inline double povFnEval(int id, const double* _pp) {{
    double X = _pp[0], Y = _pp[1], Z = _pp[2];
#define PARAM_X X
#define PARAM_Y Y
#define PARAM_Z Z
#define PARAM(i) (_pp[(i) + 3])
    switch (id) {{
{switch}
    default: break;
    }}
#undef PARAM_X
#undef PARAM_Y
#undef PARAM_Z
#undef PARAM
    return 0.0;
}}

#undef ROT2D

// Total argument count (including x,y,z) for POV internal function `id`, or 0 if
// unsupported.  Device-safe so the VM can pop the right number of stack slots.
POV_HD inline int povFnArity(int id) {
    switch (id) {
{arity_switch}
    default: return 0;
    }
}

// ---- name table (parser side) -----------------------------------------------
// Plain host inlines (not __device__).  No __CUDA_ARCH__ guard: under nvcc the
// host-side funcOp() in pattern.h references povFnLookup during the device pass
// too (it is parsed, though only host-compiled), so the symbol must stay visible.
#include <string.h>
struct PovFnInfo {{ const char* name; int id; int arity; }};
inline const PovFnInfo* povFnTable(int& count) {{
    static const PovFnInfo T[] = {{
{table}
    }};
    count = (int)(sizeof(T) / sizeof(T[0]));
    return T;
}}
// Look up an SDL name; on success fills id (POV internal id) and arity (total
// args including x,y,z).  Returns false if not a supported POV function.
inline bool povFnLookup(const char* name, int& id, int& arity) {{
    int n; const PovFnInfo* T = povFnTable(n);
    for (int i = 0; i < n; ++i)
        if (strcmp(name, T[i].name) == 0) {{ id = T[i].id; arity = T[i].arity; return true; }}
    return false;
}}
'''

hdr = TEMPLATE
# The template was authored with doubled braces / backslashes (f-string legacy);
# collapse them to single for real C, then splice the generated sections.
hdr = hdr.replace("{{", "{").replace("}}", "}").replace("\\\\", "\\")
hdr = hdr.replace("{switch}", switch)
hdr = hdr.replace("{arity_switch}", arity_switch)
hdr = hdr.replace("{table}", table)
hdr = hdr.replace("{excluded_list}", excluded_list)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(hdr)
print(f"wrote {OUT}")
print(f"supported: {len(supported)} functions; excluded: {sorted(EXCLUDE)}")
