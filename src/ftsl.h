// FTSL — the Forward-Tracer Scene Language loader.
//
// A small block-structured text format that populates the in-memory Scene + Camera
// from a file instead of a hand-written C++ builder. See docs/scene-language.md for
// the full design; this header implements Phase 1 (a loader for scenes the engine
// already renders) plus Phase 1e (full mesh transforms).
//
// Grammar (informal):
//   # line comment
//   spectrum "name" = <spectrum-expr>          # named reusable spectrum
//   material "name" { key value ...  key value }
//   material "name" { type mix  layer "child" w  layer "child2" w  ... }  # stochastic blend
//   sphere   { center x y z  radius r  material name }
//   quad     { origin x y z  u x y z  v x y z  material name }
//   triangle { v0 x y z  v1 x y z  v2 x y z  material name }
//   mesh "name" { file "p.obj"  material name  translate x y z  rotate x y z  scale x y z }
//   light area       { origin ...  u ...  v ...  normal ...  spd <spectrum-expr> }
//   light collimated { dir x y z  spd <spectrum-expr> }   # repeatable: N emitters
//   light sphere     { center x y z  radius r  spd <spectrum-expr> }  # glowing ball
//   light cylinder   { center x y z  axis x y z  length l  radius r  [caps on] spd … }  # tube/fluorescent (caps=closed capsule)
//   light spot       { origin x y z  dir x y z  inner_angle d  outer_angle d  spd … }
//   light env        { spd <spectrum-expr> }   # constant infinite environment
//   light env        { file "sky.hdr"  rotate d  intensity s }  # image-based (lat-long)
//   medium   { sigma_t v  albedo v  g v  rayleigh true }
//   camera "name" { eye ...  look_at ...  up ...  fov_y d  aperture r  focus d  mode B
//                   preset <archetype>  # cinema|pocket|portable|vintage|vintage-slr (fills optics)
//                   lens <mm>  fstop <N>  zoom <x>  # photographic authoring (overrides fov_y/aperture)
//                   projection <name> | fisheye [type]  # lens projection (rectilinear default; §8.5)
//                   film { res W H  format <name>  size <Wmm> <Hmm>  iso .. shutter .. exposure .. } }
//     preset picks a physically-plausible camera archetype (sensor+focal+f-stop); any
//     knob after it overrides. Works in mode A/C (real DOF) and pinhole modes (fov only).
//     zoom <x> multiplies the focal length (x>1 tele/narrower, x<1 wider). projection
//     picks the lens map: rectilinear (default), equidistant/fisheye, equisolid,
//     stereographic, orthographic — the fisheye modes allow fov_y >= 180.
//   camera_path "name" { ... key <t> <ex ey ez> [<lx ly lz>] [<fov>]  dolly_zoom }
//     per-keyframe fov animates a zoom; dolly_zoom holds the subject size (Vertigo).
//     lens <mm> sets field of view from focal length and film height; fstop <N> sets a
//     physically-correct aperture (radius = focal/2N) and, for modes A/C, seats the film
//     at the image distance so depth of field matches a real lens. film format presets:
//     full-frame(35mm) aps-c micro-four-thirds super35 medium-format(645) 6x6 6x7
//     large-format(4x5) 8x10 (see filmFormatMM). size gives an explicit W H in mm.
//   render   { photons N  device auto  mode B }
//
// Statements are newline-terminated; brace values (table {…}, film {…}) nest. A
// spectrum expression is any of: a number (constant), `blackbody K`, `gaussian
// center=.. sigma=.. amp=..`, `shortpass edge=.. slope=.. amp=..`, `ior n`,
// `rgb r g b`, `hsv h s v` / `hsl h s l` (hue in [0,1], wraps), `whitewall [r]`,
// `redwall`, `greenwall`, `glass:BK7|SF10`,
// `preset:<illuminant>`, `spectrum:<name>`, `file:<path>` (a measured CSV curve),
// or `table { λ:v λ:v … }`.
#pragma once
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <functional>
#include "scene.h"
#include "camera.h"
#include "spectrum.h"
#include "lights.h"
#include "materials.h"
#include "mesh.h"
#include "gltf.h"
#include "meshvoxel.h"   // solid voxelization for `medium { bounds { object "<mesh>" } }`
#include "fbx.h"
#include "upsample.h"
#include "color.h"
#include "sky.h"
#include "record_ladder.h"   // generalized record-stop delimiter ladder (J3b item 2)

namespace ftsl {

// A token is a number iff strtod consumes all of it (handles -1, 0.999, 1e30).
inline bool isNumber(const std::string& s) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end == s.c_str() + s.size();
}
inline double num(const std::string& s) { return std::strtod(s.c_str(), nullptr); }

// `<space>:<upsampler>` with a non-empty name. Split out so isColourHead and evalSpectrum
// cannot drift on what counts as one.
inline bool isCustomColourHead(const std::string& h) {
    return h.size() > 4 && h[3] == ':' &&
           (h.compare(0, 3, "rgb") == 0 || h.compare(0, 3, "hsv") == 0 ||
            h.compare(0, 3, "hsl") == 0);
}

// The arity-3 COLOUR HEADS: every spectrum expression of the form `<head> a b c`,
// across all three colour spaces and all the upsamplers/emission forms. Kept as one
// list because two very different sites need the same answer: `evalSpectrum` (which
// consumes them as a value head) and a record channel's inline-colour TAG (where the
// head is written once for the whole channel and each stop supplies only the triple).
// Splitting the list would let the two drift, and the failure mode is silent — a head
// the tag accepts but `evalSpectrum` doesn't would turn a colour stop into a mystery
// scalar-expression error pointing at the wrong token.
inline bool isColourHead(const std::string& h) {
    static const char* kHeads[] = {
        "rgb",      "hsv",      "hsl",
        "rgbline",  "hsvline",  "hslline",     // dominant-wavelength emission (K3)
        "rgbillum", "hsvillum", "hslillum",    // Jakob-Hanika illuminant (K1)
        "rgbsmits", "hsvsmits", "hslsmits",    // Smits 1999 reflectance (K1)
        "rgbbox",   "hsvbox",   "hslbox",      // calibrated 3-box reflectance (K1)
        "rgbmeng",  "hsvmeng",  "hslmeng",     // Meng 2015 smoothest reflectance (K1)
    };
    for (const char* k : kHeads) if (h == k) return true;
    // `rgb:<name>` / `hsv:<name>` / `hsl:<name>` — a USER-DECLARED upsampler (K1), named
    // by an `upsample "<name>"` block. A colon rather than yet another glued suffix
    // because the built-in suffixes are a closed set the reader can memorise while a user
    // name is open-ended, and `:` is already this grammar's namespace marker everywhere
    // else (`spectrum:`, `metal:`, `glass:`, `preset:`, `tex:`, `grid:`).
    //
    // Only the SHAPE is checked here; whether the name resolves is evalSpectrum's job.
    // That split matters: this predicate also gates a record channel's inline-colour tag,
    // and a head accepted here but rejected there would report an unknown upsampler as a
    // mystery scalar-expression error pointing at the wrong token.
    return isCustomColourHead(h);
}

// NOTE: the measured-SPD CSV loader (`loadSpdCsv`) that used to live here now lives
// in spectral_library.h as `speclib::loadSpdCsv` — a single implementation shared by
// the FTSL `file:` expression (via loadSpdFile below) and the named-preset library
// resolvers (metal/reflectance/illuminant). See data/README.md for the file format.

// Resolve a named glass dispersion from the spectral library (data/glass/<name>.glass),
// falling back to a constant index if that file is missing. Used for the built-in
// BK7/SF10 defaults that back `dielectric`'s default IOR and the lens presets — the
// dispersion DATA lives in files, but a sane default must survive a stripped data dir.
inline Spectrum glassOrDefault(const char* name, double fallbackN) {
    Spectrum s;
    if (resolveGlassIor(name, s)) return s;
    return iorConstant(fallbackN);
}

// ---------------------------------------------------------------------------
// Parse tree
// ---------------------------------------------------------------------------
struct Block;

// One entry inside a `[ … ]` bracket group at a value site: either a bare token or a
// nested group. Kept as a RAW tree by the grammar reducer, which therefore never has to
// decide what the brackets MEAN — that is the loader's job (applyBracketGroup below),
// and keeping the decision in exactly one place is what let the shared grammar replace
// the hand-written parser without either side re-deriving the interpretation.
struct BrItem {
    bool                isGroup = false;
    std::string         word;      // when !isGroup
    std::vector<BrItem> items;     // when isGroup
};

// An inline N-D array literal at a value site — `[0 1](u)`, `[[0 1 2][3 4 5]](u,v)`.
// The NESTING is the shape (axis 0 outermost, C order — the same layout a `grid`
// element's `data { … }` uses), so a shape need never be spelled. `call` is the raw
// text of the trailing sample call INCLUDING its parens, and is empty when the array
// was written UNSATURATED — legal to author, an error to render, with the fix in the
// message. The loader desugars each of these into an anonymous grid + pattern.
struct ArrayLit {
    std::vector<BrItem> items;
    std::string         call;
    int                 line = 0;
};

struct Value {
    std::vector<std::string> words;    // scalar / vector / expression tokens
    std::shared_ptr<Block> block;      // nested brace block (table/film/coat/body/...)
    std::shared_ptr<ArrayLit> array;   // inline `[ … ](coords)` array literal, if any
};
// What does a `[ … ]` group at a value site MEAN?  There are two answers — a record
// channel's STOP SELECTOR (`REC.chan[2]`) and an inline N-D ARRAY LITERAL (`[0 1](u)`) —
// and this ONE function is where the question is answered.  Keeping the decision at the
// LOADER level, out of the front end entirely, is deliberate: the grammar only has to
// *collect* the group's raw shape, never interpret it, which is what made swapping the
// front end out for the shared grammar a pure parse-tree exercise.
//
//   1. a trailing sample call    -> array literal (the call names the coordinates)
//   2. otherwise nesting present -> array literal, UNSATURATED (an error at load time,
//                                   but a parse-level array all the same)
//   3. otherwise something foldable before the group -> the record stop selector; append
//      the words to that token (`REC.chan` + `[2]`), exactly as before.  "Foldable" is a
//      dotted word, or — inside the `= rhs [i]` record-override form, where a selector is
//      the only thing brackets can possibly mean — any preceding token at all.
//   4. otherwise nothing before the group and every item a number -> array literal,
//      unsaturated (`roughness [0 1]` — the author forgot the `(u)`)
//   5. otherwise -> drop it (a stray bracket group; unchanged legacy behaviour)
inline void applyBracketGroup(Value& v, std::vector<BrItem> items,
                              const std::string& call, int line,
                              bool overrideForm = false) {
    bool nested = false, allNum = !items.empty();
    for (const auto& it : items) {
        if (it.isGroup) { nested = true; allNum = false; }
        else if (!isNumber(it.word)) allNum = false;
    }
    bool dotted = !v.words.empty() &&
                  (overrideForm || v.words.back().find('.') != std::string::npos);
    if (!call.empty() || nested || (v.words.empty() && allNum)) {
        v.array = std::make_shared<ArrayLit>();
        v.array->items = std::move(items);
        v.array->call  = call;
        v.array->line  = line;
    } else if (dotted) {
        std::string idx;
        for (const auto& it : items) idx += it.word;
        v.words.back() += "[" + idx + "]";
    }
}

// `used` records that some builder actually READ this statement. Nothing in the
// loader behaves differently because of it — it exists so that after a scene is
// built we can report the keys nobody looked at. An unknown key is otherwise
// silently ignored, which is the worst possible failure mode: a misspelt or
// drifted property (loom emitting `size` where ftrace wants `scale`, say) renders
// a *wrong image* rather than raising an error. It is `mutable` because reads go
// through `find(const Block&, ...)`, and "I was read" is not part of a block's
// logical value.
struct Stmt { std::string key; Value val; int line = 0; mutable bool used = false; };
struct Block {
    std::string type;                  // material / quad / film / table / ...
    std::string subtype;               // light "area" / "collimated"
    std::string name;                  // quoted name, if any
    std::vector<Stmt> stmts;           // newline-structured statements
    std::vector<std::string> words;    // flat token dump (for table/palette lists)
    // For type == "prefer": ordered alternative branches (`prefer { .. } else { .. }`).
    // Each branch is a list of ordinary top-level blocks; the loader picks the first
    // branch whose spliced scene is fully renderable in its chosen mode. Empty otherwise.
    std::vector<std::vector<Block>> branches;
};

}  // namespace ftsl  (temporarily closed so the GPDA front end can see
   //                  ftsl::Block/Stmt/Value at global scope)

// The ONE .ftsl front end: the shared grammar (tools/loom/loom/grammar/ftsl_scene.epeg,
// compiled to a GPDA graph) plus the reducer that turns its parse tree into the Block
// tree above.  Included here — after Block/Stmt/Value are defined — so the inline
// reducer can reference them; loadSource() below calls ftsl_gpda::parse().
// (Through 0.78 a hand-written recursive-descent Parser lived here too, kept behind
// `-legacy-parser` while the grammar proved itself; it was deleted in 0.79.0 after the
// corpus differ held at MATCH 2595/2595 across ten releases.)
#include "gpda/ftsl_frontend.hpp"

namespace ftsl {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
// Split "key=value" (named params for gaussian/shortpass). Returns false if no '='.
inline bool splitEq(const std::string& s, std::string& k, std::string& v) {
    auto p = s.find('=');
    if (p == std::string::npos) return false;
    k = s.substr(0, p); v = s.substr(p + 1);
    return true;
}

// Find the statement with a given key in a block; nullptr if absent.
inline const Stmt* find(const Block& b, const char* key) {
    for (const auto& s : b.stmts) if (s.key == key) { s.used = true; return &s; }
    return nullptr;
}
// Mark every statement with `key` as read. `find` only reaches the first one, so
// the loops that gather a REPEATED key (`point`, `look_point`, `fwd_at`, a light's
// `spd` stops, …) call this to account for the rest.
inline void markUsed(const Block& b, const char* key) {
    for (const auto& s : b.stmts) if (s.key == key) s.used = true;
}
// Gather every substring listed under a REPEATED, comma-splittable key (currently
// `skip_material`). FTSL's statement splitter starts a NEW statement at the second
// bareword, so `skip_material a b` would silently keep only `a` (and warn about a
// stray key `b`). Both spellings that survive the splitter are therefore accepted and
// unioned: repeat the statement (`skip_material a` / `skip_material b`) or comma-join
// one token (`skip_material a,b`). Marks every match read, like markUsed.
inline std::vector<std::string> wordListOf(const Block& b, const char* key) {
    std::vector<std::string> out;
    for (const auto& s : b.stmts) {
        if (s.key != key) continue;
        s.used = true;
        for (const std::string& w : s.val.words) {
            size_t start = 0;
            for (;;) {
                size_t c = w.find(',', start);
                std::string piece = w.substr(start, (c == std::string::npos) ? c : c - start);
                if (!piece.empty()) out.push_back(piece);
                if (c == std::string::npos) break;
                start = c + 1;
            }
        }
    }
    return out;
}
// Mark a whole block read, for bodies whose content is consumed as the flat
// `words` dump rather than key/value statements (`data { … }`, `palette { … }`,
// a table's rows). Their "keys" are just the first token of each line, so
// per-key accounting is meaningless there.
inline void markAllUsed(const Block& b) {
    for (const auto& s : b.stmts) s.used = true;
}
// Gather every statement no builder read, recursing into nested `{ }` bodies.
// `where` names the enclosing block for the message ("material \"gold\"", then
// "material \"gold\" > coat"). When a statement itself is unread we report only
// it and do NOT descend — its whole body is unread by construction, and listing
// each child would bury the one line the author actually has to fix.
inline void collectUnusedKeys(const Block& b, const std::string& where,
                              std::vector<std::string>& out) {
    for (const auto& s : b.stmts) {
        if (!s.used) {
            out.push_back(where + ": unknown key '" + s.key + "'" +
                          (s.line > 0 ? " on line " + std::to_string(s.line) : ""));
            continue;
        }
        if (s.val.block) collectUnusedKeys(*s.val.block, where + " > " + s.key, out);
    }
}
// Human-readable label for a top-level block, used by the message above.
inline std::string blockLabel(const Block& b) {
    std::string s = b.type;
    if (!b.subtype.empty()) s += " " + b.subtype;
    if (!b.name.empty()) s += " \"" + b.name + "\"";
    return s;
}
inline std::string strOf(const Block& b, const char* key, const std::string& dflt = "") {
    const Stmt* s = find(b, key);
    return (s && !s->val.words.empty()) ? s->val.words[0] : dflt;
}
inline bool vec3Of(const Block& b, const char* key, Vec3& out) {
    const Stmt* s = find(b, key);
    if (!s || s->val.words.size() < 3) return false;
    out = {num(s->val.words[0]), num(s->val.words[1]), num(s->val.words[2])};
    return true;
}
inline double dblOf(const Block& b, const char* key, double dflt) {
    const Stmt* s = find(b, key);
    return (s && !s->val.words.empty()) ? num(s->val.words[0]) : dflt;
}

// Named film / sensor formats -> physical (width, height) in millimetres, landscape
// orientation. Lets a camera say `film { format full-frame }` instead of `size 36 24`,
// matching how photographers pick a body/back. The key is normalised (lower-cased with
// spaces / underscores / hyphens stripped) so "medium format", "medium-format" and
// "MediumFormat" all match. Returns false for an unknown name.
inline bool filmFormatMM(const std::string& raw, double& w, double& h) {
    std::string k;
    for (char c : raw) { if (c==' '||c=='_'||c=='-') continue; k += (char)std::tolower((unsigned char)c); }
    struct F { const char* k; double w, h; };
    static const F tbl[] = {
        // 35 mm still / cine
        {"35mm",36,24}, {"fullframe",36,24}, {"135",36,24}, {"ff",36,24},
        {"halfframe",24,18},
        {"super35",24.89,18.66}, {"s35",24.89,18.66}, {"academy",21.95,16.0},
        // digital crop sensors
        {"apsc",23.6,15.6}, {"apsh",28.7,19.0},
        {"microfourthirds",17.3,13.0}, {"mft",17.3,13.0}, {"m43",17.3,13.0}, {"fourthirds",17.3,13.0},
        {"1inch",13.2,8.8}, {"1in",13.2,8.8},
        // medium format
        {"mediumformat",56,41.5}, {"645",56,41.5}, {"6x45",56,41.5},
        {"6x6",56,56}, {"6x7",70,56}, {"6x9",84,56},
        {"digitalmediumformat",43.8,32.9}, {"gfx",43.8,32.9},
        // large / sheet film
        {"largeformat",127,101.6}, {"4x5",127,101.6}, {"5x4",127,101.6},
        {"8x10",254,203.2},
    };
    for (const auto& f : tbl) if (k == f.k) { w = f.w; h = f.h; return true; }
    return false;
}

// Map a projection/fisheye name to a CameraProjection enum (-1 if unknown). Name
// matching is case/space/hyphen/underscore-insensitive.
inline int projectionFromName(const std::string& raw) {
    std::string k;
    for (char c : raw) { if (c==' '||c=='_'||c=='-') continue; k += (char)std::tolower((unsigned char)c); }
    if (k=="rectilinear" || k=="perspective" || k=="normal" || k=="pinhole") return CAM_RECTILINEAR;
    if (k=="equidistant" || k=="fisheye")                                     return CAM_EQUIDISTANT;
    if (k=="equisolid"   || k=="equalarea"  || k=="equisolidangle")           return CAM_EQUISOLID;
    if (k=="stereographic")                                                   return CAM_STEREOGRAPHIC;
    if (k=="orthographic"|| k=="ortho")                                       return CAM_ORTHOGRAPHIC;
    return -1;
}

// Catmull-Rom spline through control points (an INTERPOLATING spline: the curve
// passes through every control point). `g` is the global parameter in [0, nSeg],
// where nSeg = closed ? n : n-1; segment = floor(g) and the local t is its
// fraction. Segment `seg` blends points [seg-1, seg, seg+1, seg+2]; an open curve
// clamps the out-of-range neighbours (so the end tangents point straight down the
// last edge), a closed curve wraps them modulo n. Used by `camera_curve`.
//
// `alpha` selects the knot parameterization: 0 = UNIFORM (the classic form, tangent
// (P2-P0)/2 -- fast but OVERSHOOTS and can cusp/loop when control points are unevenly
// spaced), 0.5 = CENTRIPETAL (knots spaced by chord^0.5; provably no cusps or
// self-intersections, stays tight to the points -- the fix for a jerky flight through
// unevenly-spaced waypoints), 1 = CHORDAL (chord^1). alpha=0 is bit-identical to the
// original so existing scenes are unchanged.
inline Vec3 catmullRomAt(const std::vector<Vec3>& p, bool closed, double g, double alpha = 0.0) {
    int n = (int)p.size();
    if (n == 1) return p[0];
    int nSeg = closed ? n : n - 1;
    if (g < 0) g = 0;
    if (g > nSeg) g = nSeg;
    int seg = (int)std::floor(g);
    if (seg >= nSeg) seg = nSeg - 1;
    double t = g - seg;
    auto idx = [&](int i) -> const Vec3& {
        if (closed) { i = ((i % n) + n) % n; return p[(size_t)i]; }
        if (i < 0) i = 0;
        if (i > n - 1) i = n - 1;
        return p[(size_t)i];
    };
    const Vec3& P0 = idx(seg - 1);
    const Vec3& P1 = idx(seg);
    const Vec3& P2 = idx(seg + 1);
    const Vec3& P3 = idx(seg + 2);
    if (alpha <= 0.0) {
        double t2 = t * t, t3 = t2 * t;
        return (P1 * 2.0
              + (P2 - P0) * t
              + (P0 * 2.0 - P1 * 5.0 + P2 * 4.0 - P3) * t2
              + (P1 * 3.0 - P0 - P2 * 3.0 + P3) * t3) * 0.5;
    }
    // Non-uniform Catmull-Rom via the Barry-Goldman recursion. Knots advance by
    // chord^alpha; a small floor keeps duplicated endpoints (open-curve clamping) and
    // coincident control points from dividing by zero.
    auto knot = [&](double ti, const Vec3& a, const Vec3& b) {
        return ti + std::pow(std::max(length(b - a), 1e-9), alpha);
    };
    double k0 = 0.0;
    double k1 = knot(k0, P0, P1);
    double k2 = knot(k1, P1, P2);
    double k3 = knot(k2, P2, P3);
    double tt = k1 + t * (k2 - k1);                 // evaluation param within [k1,k2]
    auto lerp = [](const Vec3& a, const Vec3& b, double u) { return a * (1.0 - u) + b * u; };
    Vec3 A1 = lerp(P0, P1, (tt - k0) / (k1 - k0));
    Vec3 A2 = lerp(P1, P2, (tt - k1) / (k2 - k1));
    Vec3 A3 = lerp(P2, P3, (tt - k2) / (k3 - k2));
    Vec3 B1 = lerp(A1, A2, (tt - k0) / (k2 - k0));
    Vec3 B2 = lerp(A2, A3, (tt - k1) / (k3 - k1));
    return lerp(B1, B2, (tt - k1) / (k2 - k1));
}

// Rotate vector `v` about `axis` by `ang` radians (Rodrigues' rotation formula).
// `axis` is normalized internally; a zero-length axis returns `v` unchanged. Used
// by `camera_curve` to apply a per-frame `roll` (bank about the view direction).
inline Vec3 rotateAboutAxis(const Vec3& v, const Vec3& axis, double ang) {
    double al = length(axis);
    if (al < 1e-12) return v;
    Vec3 k = axis * (1.0 / al);
    double c = std::cos(ang), s = std::sin(ang);
    return v * c + cross(k, v) * s + k * (dot(k, v) * (1.0 - c));
}

// A piecewise-linear animation track over a normalized timeline t in [0,1]: a sorted
// list of `{t, value}` keyframes with flat clamping outside the first/last key. Used
// by `camera_curve` to animate scalar camera properties (roll, fov, zoom, f-stop,
// focus) frame-by-frame, mirroring how `density_at` keyframes camera speed.
struct ScalarTrack {
    struct Key { double t, v; };
    std::vector<Key> keys;
    bool active() const { return !keys.empty(); }
    void sort() { std::sort(keys.begin(), keys.end(),
                            [](const Key& a, const Key& b){ return a.t < b.t; }); }
    double sample(double t, double fallback) const {
        if (keys.empty()) return fallback;
        if (t <= keys.front().t) return keys.front().v;
        if (t >= keys.back().t)  return keys.back().v;
        for (size_t j = 0; j + 1 < keys.size(); ++j)
            if (t >= keys[j].t && t <= keys[j + 1].t) {
                double sp = keys[j + 1].t - keys[j].t;
                double f = (sp > 1e-12) ? (t - keys[j].t) / sp : 0.0;
                return keys[j].v + (keys[j + 1].v - keys[j].v) * f;
            }
        return keys.back().v;
    }
};

// A piecewise-linear 3-vector animation track over t in [0,1] (the Vec3 analogue of
// ScalarTrack), flat-clamped at the ends. Used by `camera_curve`'s orientation axes
// (`fwd_at`/`up_at`): a per-keyframe forward direction or up vector, interpolated
// component-wise (the caller normalizes / re-orthogonalizes the result).
struct Vec3Track {
    struct Key { double t; Vec3 v; };
    std::vector<Key> keys;
    bool active() const { return !keys.empty(); }
    void sort() { std::sort(keys.begin(), keys.end(),
                            [](const Key& a, const Key& b){ return a.t < b.t; }); }
    Vec3 sample(double t) const {
        if (keys.empty()) return Vec3{0, 0, 0};
        if (t <= keys.front().t) return keys.front().v;
        if (t >= keys.back().t)  return keys.back().v;
        for (size_t j = 0; j + 1 < keys.size(); ++j)
            if (t >= keys[j].t && t <= keys[j + 1].t) {
                double sp = keys[j + 1].t - keys[j].t;
                double f = (sp > 1e-12) ? (t - keys[j].t) / sp : 0.0;
                return keys[j].v * (1.0 - f) + keys[j + 1].v * f;
            }
        return keys.back().v;
    }
};

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------
// One authored camera. The Camera itself is built in main at the final resolution,
// so a CLI -r override stays consistent with the output film size. `res` is the
// camera's own film resolution (-1 = inherit the global/CLI res); `mode` is the
// per-camera measurement model (0 = inherit the global/CLI mode).
struct CamSpec {
    std::string name;
    Vec3   eye{0, 1, 3}, look{0, 1, 0}, up{0, 1, 0};
    double fov = 40.0, aperture = 0.02, focus = 0.0;
    char   mode = 0;             // 0 = not specified -> inherit global
    int    res  = -1;            // film WIDTH  in px (-1 = inherit global/CLI res)
    int    resY = -1;            // film HEIGHT in px (-1 = square: follow res)
    // Playback frame rate for a flyby (camera_path/orbit/curve). Purely an animation
    // *hint* consumed by the video-assembly tooling (showcase_flyby.py -> ffmpeg): it
    // does not affect how any still frame is rendered. 0 = not specified -> inherit the
    // scene-level `fps` default, else the tool's own default. Meaningless for a still.
    double fps = 0.0;

    // Lens projection (0 = rectilinear; see CameraProjection) and an optional zoom
    // multiplier on the focal length (1 = none; 2 = 2x tele, i.e. half the fov).
    int    projection = CAM_RECTILINEAR;
    double zoom = 1.0;

    // Physical film + photographic exposure (Phase 3a). filmW/H are the physical
    // sensor dimensions in millimetres (0 = unspecified -> 36x24 "full frame"
    // default is assumed only where a physical size is actually needed, e.g. to
    // turn an f-number into an aperture radius). `focal` is the derived focal
    // length in metres (internal units), from filmH + fov_y.
    double filmW_mm = 0.0, filmH_mm = 0.0, focal = 0.0;
    // Physical-optics camera (filled when `lens <mm>` or `fstop` is authored): the
    // film sits at the image distance and the thin lens has the real focal length,
    // so the f-number produces true depth of field in the catch modes (A/C). Both 0
    // => legacy camera (unit film distance, lens synthesised from `focus` by setFocus).
    double filmDist_m = 0.0, lensF_m = 0.0;
    // Photographic exposure controls. The film's radiometric scale is not absolute,
    // so images are always auto-exposed (99th-percentile anchor); these act as an
    // exposure *compensation* on top of that anchor:
    //   comp = exposure * (iso/100) * shutter    (each factor defaults to 1)
    // e.g. ISO 200 -> comp 2.0 -> one stop brighter than ISO 100. True absolute EV
    // needs absolute light power (a separate deferred feature). `exposure <= 0` here
    // means "not authored". Aperture is deliberately NOT folded into comp: in the
    // physical catch modes (A/C) a smaller aperture already darkens the image by
    // passing fewer photons, and in the splat mode (B) the aperture is virtual, so
    // an extra f-number term would double-count / be an artifact. See docs §8.1.
    double iso = 0.0, shutter = 0.0, exposure = 0.0;
    // Resolved exposure compensation (<= 0 => neutral auto-expose). Filled at load.
    double exposureMul = 0.0;

    // Exposure-lock across a `camera_path` (Phase 3a intermediate win): frames of a
    // path that authored `exposure_lock` share ONE auto-exposure anchor, so a dolly/
    // zoom doesn't flicker as scene brightness shifts. `pathGroup` (>=0) identifies the
    // owning path (all its frames share the value); -1 for a standalone `camera`.
    // `exposureLock` is set on every frame of a locked path. A CLI `-exposure-lock` can
    // additionally force a single shared anchor across *all* rendered cameras.
    //
    // Which frame's exposure the whole group locks to is chosen by the lock *selector*
    // (authored as an argument to `exposure_lock`; every frame of the path carries the
    // same resolved selector):
    //   EXPLOCK_AVERAGE `exposure_lock`            -> mean anchor over all frames (DEFAULT)
    //   EXPLOCK_FIRST   `exposure_lock first`      -> the path's first frame (expose for it)
    //   EXPLOCK_INDEX   `exposure_lock index N`    -> frame N (0-based; N<0 counts from end)
    //   EXPLOCK_NEAR    `exposure_lock near X Y Z` -> the frame whose eye is nearest (X,Y,Z)
    //   EXPLOCK_CAMERA  `exposure_lock <name>`     -> a separately-defined camera "<name>"
    enum { EXPLOCK_FIRST = 0, EXPLOCK_INDEX, EXPLOCK_NEAR, EXPLOCK_CAMERA, EXPLOCK_AVERAGE };
    int  pathGroup   = -1;
    bool exposureLock = false;
    int  expLockSel   = EXPLOCK_FIRST;   // which frame the group meters from (enum above)
    int  expLockIndex = 0;               // EXPLOCK_INDEX: frame index (may be negative)
    Vec3 expLockPoint{0, 0, 0};          // EXPLOCK_NEAR: metering viewpoint
    std::string expLockCam;              // EXPLOCK_CAMERA: name of the metering camera

    // Physical multi-element lens (the "mesh-lens" camera), built from a `lens { ... }`
    // block. When set, main renders this camera through the backward realistic-camera
    // path (mode R), tracing rays through the real glass interfaces. Null => the
    // analytic pinhole/thin-lens camera above.
    std::shared_ptr<LensSystem> lens;
};

// Round-trip record of an authored `camera_curve`'s CONTROL POINTS (not the expanded
// per-frame cameras), captured at load so the interactive editor (-explore / -fly) can
// seed itself from an existing curve and edit it in place. Positions are in internal
// units (metres), matching everything the viewer works with.
struct AuthoredCurve {
    std::string         name;
    std::vector<Vec3>   eyes;      // `point` control points, in file order
    std::vector<Vec3>   fwds;      // per-point unit look direction (from look curve / look_at / tangent)
    std::vector<double> density;   // per-point rho (internal units); empty => uniform speed
    Vec3   up{0, 1, 0};
    double fov = 40.0;             // fov_y in degrees
    char   mode = 0;               // 0 = inherit
    bool   closed = false;
};

struct Loaded {
    Scene scene;
    // All authored cameras, in file order. Phase 3a: any number of `camera` blocks
    // accumulate; main renders the CLI-selected one, or all of them.
    std::vector<CamSpec> cameras;
    // Control points of every authored `camera_curve` (for the in-viewer editor's
    // round-trip load; see AuthoredCurve). Empty for scenes with no curve.
    std::vector<AuthoredCurve> authoredCurves;
    // Mirror of the FIRST camera (kept so the pre-Phase-3a single-camera code paths
    // and defaults keep working unchanged).
    bool hasCamera = false;
    Vec3 camEye{0, 1, 3}, camLook{0, 1, 0}, camUp{0, 1, 0};
    double camFov = 40.0, camAperture = 0.02, camFocus = 0.0;
    char mode = 'B';
    char defaultMode = 0;        // scene { default_mode X }: fallback mode for cameras that
                                 //   don't author their own `mode` (0 = not specified). Unlike
                                 //   `mode` above (which trails the last camera/render block),
                                 //   this is a stable, camera-immune default.
    // `mode W` is not a transport mode of its own: it is mode R plus the deterministic
    // Whitted estimators (backward.h `whitted`). Normalise it to 'R' AT PARSE TIME and
    // raise this flag, so nothing downstream ever has to know about a 'W' — the letter
    // reaching the render dispatch used to fall through to the forward default and
    // silently produce a black image. main.cpp ORs this into g_whitted. It is a
    // whole-run switch (like -heroc), so one camera authoring `mode W` turns it on for
    // the run; mixing W and R cameras in one file is not meaningful.
    bool whitted = false;
    double defaultFps = 0.0;     // scene { fps N }: default flyby playback fps (0 = not specified)
    long long photons = -1;      // -1 = not specified (CLI default wins)
    int res = -1;                // -1 = not specified
    // render { max_bounce N }: path depth the SCENE needs, -1 = not specified. This is a
    // property of the geometry, not of the operator's taste, so a scene that needs it must
    // be able to say so: mode D defaults to 8 path edges, and a thin-walled glass shell
    // with another tube inside it (the gallery's Klein bottle) presents ~8 dielectric
    // interfaces on one line of sight, so at the default the innermost tube renders as a
    // solid BLACK plug — truncated paths, not a material bug. A CLI `-max-bounce` still wins.
    int maxBounce = -1;
    std::string device;          // empty = not specified
    std::string out;             // empty = not specified
    // Keys no builder read (see collectUnusedKeys). Carried on Loaded rather than
    // printed inside build() because `prefer { } else { }` builds several candidate
    // scenes and discards all but one — only the accepted candidate's warnings are
    // the author's problem.
    std::vector<std::string> unknownKeys;
};

// Normalise an authored mode letter: `W` is mode R plus the deterministic Whitted
// estimators, not a transport mode of its own, so it never survives the parse.
// See Loaded::whitted.
inline char normMode(const std::string& md, Loaded& L) {
    char m = md[0];
    if (m == 'W' || m == 'w') { L.whitted = true; return 'R'; }
    return m;
}

class Builder {
public:
    std::string err;

    bool build(std::vector<Block>& blocks, Loaded& L) {
        records_ = &L.scene.records;   // stable handle for record refs at value sites (records added in Pass 1d)
        loadedRef_ = &L;               // ditto for §3.2 material-property refs (which may apply a material)
        gridsRef_    = &L.scene.grids;      // ditto for N-D table arity lookups
        scattersRef_ = &L.scene.scatters;   // (both kinds are added in Pass 1a)
        // Pass 0: global scene settings — the length unit and spectral range. All
        // authored lengths are scaled to the internal unit (metres) at load time,
        // so a scene authored in cm and one in m render identically.
        for (const auto& b : blocks) {
            if (b.type != "scene") continue;
            std::string u = strOf(b, "units", "meters");
            if      (u == "meters" || u == "metres" || u == "m")        L_ = 1.0;
            else if (u == "centimeters" || u == "cm")                   L_ = 0.01;
            else if (u == "millimeters" || u == "mm")                   L_ = 0.001;
            else if (u == "inches" || u == "in")                        L_ = 0.0254;
            else if (u == "feet" || u == "ft")                          L_ = 0.3048;
            else { fail("unknown units '" + u + "' (meters|centimeters|millimeters|inches|feet)"); return false; }
            const Stmt* sp = find(b, "spectral");
            if (sp && sp->val.words.size() >= 3) {
                double lo = num(sp->val.words[0]), hi = num(sp->val.words[1]);
                binWidth_ = num(sp->val.words[2]);
                if (binWidth_ <= 0) binWidth_ = 1.0;
                if (lo != LAMBDA_MIN || hi != LAMBDA_MAX)
                    std::fprintf(stderr, "[ftsl] warning: spectral range %g..%g nm requested but the "
                                 "engine range is fixed at %g..%g nm (widening is not yet supported); "
                                 "only the bin width (%g nm) is applied.\n", lo, hi, LAMBDA_MIN, LAMBDA_MAX, binWidth_);
            }
            // Scene-level defaults. `default_mode X` gives a stable fallback render mode for
            // any camera that doesn't author its own `mode` (see effMode in main). `fps N`
            // is the default flyby playback rate the video tooling uses when a camera_curve/
            // path/orbit doesn't set its own `fps`. Both are pure defaults — a per-camera
            // `mode`/`fps` and the CLI still override them.
            std::string dm = strOf(b, "default_mode");
            if (!dm.empty()) L.defaultMode = normMode(dm, L);
            double dfps = dblOf(b, "fps", 0.0);
            if (dfps > 0.0) L.defaultFps = dfps;
        }

        // Pass 1: collect named spectra (resolve refs lazily), materials, camera.
        for (const auto& b : blocks)
            if (b.type == "spectrum") {
                spectraBlocks_[b.name] = &b;
                // A `spectrum "x" = <expr>` parses to one synthetic `=` statement, read
                // lazily only if something references the spectrum. Count it as read at
                // DECLARATION time: an unreferenced spectrum is pointless but legal, and
                // is not an unknown-key problem.
                markUsed(b, "=");
            }

        // Named RGB->spectral upsamplers (K1): `upsample "name" { expr "<f(r,g,b,w)>" }`.
        // Collected here, beside spectra and for the same reason: an upsampler is only
        // ever reached BY NAME from an `rgb:<name>` colour head, so it is compiled lazily
        // on first use. Declaration order therefore does not matter, and an unreferenced
        // upsampler is legal (pointless, but not an unknown-key problem).
        for (const auto& b : blocks)
            if (b.type == "upsample") {
                upsampleBlocks_[b.name] = &b;
                markUsed(b, "expr");
            }

        // Pass 1-arr: desugar inline `[ … ](coords)` array literals into anonymous
        // `grid` + `pattern` block pairs, APPENDED to `blocks`. Running it before Pass 1a
        // is what makes an inline literal work at every slot that already accepts a
        // `pattern:` — the rest of the loader never learns the syntax exists.
        if (!desugarArrays(blocks)) return false;

        // Pass 1a: N-D data tables (regular `grid`s and ragged `scatter`s). FIRST of the
        // table passes, because a procedural `texture { rgb "…" }` bakes during Pass 1b
        // and may sample `grid:<name>(…)` / `scatter:<name>(…)`.
        for (const auto& b : blocks) {
            if (b.type != "grid") continue;
            if (!addGrid(b, L)) return false;
        }
        for (const auto& b : blocks) {
            if (b.type != "scatter") continue;
            if (!addScatter(b, L)) return false;
        }

        // Pass 1b: image textures (must exist before materials that bind them).
        for (const auto& b : blocks) {
            if (b.type != "texture") continue;
            if (!addTexture(b, L)) return false;
        }

        // Pass 1c: procedural patterns (must exist before materials that bind them).
        for (const auto& b : blocks) {
            if (b.type != "pattern") continue;
            if (!addPattern(b, L)) return false;
        }

        // Pass 1d: parametric records (must exist before materials that reference them).
        for (const auto& b : blocks) {
            if (b.type != "record") continue;
            if (!addRecord(b, L)) return false;
        }

        // Pass 2: materials (must exist before geometry references them).
        for (const auto& b : blocks) {
            if (b.type != "material") continue;
            if (b.name.empty()) { fail("material needs a \"name\""); return false; }
            int id = (int)L.scene.mats.size();
            Material m = buildMaterial(b, L);
            if (!err.empty()) return false;
            L.scene.mats.push_back(m);
            matIndex_[b.name] = id;
            // §3.3: the fallback for a free `a` at any use site that doesn't bind it.
            if (find(b, "albedo_default")) albedoDefault_[id] = dblOf(b, "albedo_default", 1.0);
        }

        // Pass 2b: resolve Mix / Layered body child references (now that every name
        // is known). A Layered material's `layer "name" weight` list is its body,
        // resolved by the same second pass as a mix.
        for (const auto& b : blocks) {
            if (b.type != "material") continue;
            int id = matIndex_[b.name];
            MatType t = L.scene.mats[id].type;
            if (t != MatType::Mix && t != MatType::Layered) continue;
            if (!resolveMixChildren(b, id, L)) return false;
        }

        // Pass 2.5: mesh assets (shared instanced geometry). Loaded before the
        // geometry pass so a `mesh_instance { of "name" }` (top-level or inside a
        // group) can reference any asset regardless of authoring order.
        for (const auto& b : blocks) {
            if (b.type == "mesh_asset") { if (!addMeshAsset(b, L)) return false; }
        }

        // Pass 3: geometry, lights, medium, camera, render.
        // `medium` blocks are DEFERRED to a second sweep so `bounds { object "name" }`
        // can reference any named sphere / isosurface / mesh regardless of authoring
        // order (the object registries are populated by the geometry builders below).
        bool haveLight = false;
        std::vector<const Block*> mediaBlocks;
        for (const auto& b : blocks) {
            if      (b.type == "sphere")   { if (!addSphere(b, L)) return false; }
            else if (b.type == "quad")     { if (!addQuad(b, L)) return false; }
            else if (b.type == "triangle") { if (!addTriangle(b, L)) return false; }
            else if (b.type == "mesh")     { if (!addMesh(b, L)) return false; }
            else if (b.type == "mesh_instance") { if (!addMeshInstance(b, L)) return false; }
            else if (b.type == "isosurface") { if (!addIsosurface(b, L)) return false; }
            else if (b.type == "light")    { if (!addLight(b, L, b.subtype)) return false; haveLight = true; }
            else if (b.type == "group")    { if (!addGroup(b, L, Affine::identity(), haveLight)) return false; }
            else if (b.type == "medium")   { mediaBlocks.push_back(&b); }
            else if (b.type == "camera")   { if (!addCamera(b, L)) return false; }
            else if (b.type == "camera_path") { if (!addCameraPath(b, L)) return false; }
            else if (b.type == "camera_orbit") { if (!addCameraOrbit(b, L)) return false; }
            else if (b.type == "camera_curve") { if (!addCameraCurve(b, L)) return false; }
            else if (b.type == "render")   { if (!applyRender(b, L)) return false; }
            else if (b.type == "scene" || b.type == "spectrum" || b.type == "material" ||
                     b.type == "texture" || b.type == "pattern" || b.type == "record" ||
                     b.type == "grid" || b.type == "scatter" ||
                     b.type == "upsample" ||
                     b.type == "mesh_asset") { /* handled */ }
            else { fail("unknown top-level block '" + b.type + "'"); return false; }
        }
        // Deferred medium sweep (object-name bounds resolve against the registries).
        for (const Block* mb : mediaBlocks) { if (!addMedium(*mb, L)) return false; }
        // Every consumer of a shape-only mesh's triangles has now run, so drop them
        // before the BVH is built (see stripShapeOnlyMeshes).
        stripShapeOnlyMeshes(L);
        // A scene is lit if it has an explicit `light` block OR any emitter was
        // registered implicitly — e.g. an emissive mesh (a material with `emit` bound
        // to a mesh registers a Mesh area light in addMesh) — OR it contains a
        // self-illuminating volume (a medium with a `temperature` grid + `emission`),
        // i.e. "fire": the hot voxels are the only light source.
        bool haveVolumeEmission = false;
        for (const Medium& m : L.scene.media)
            if (m.emissive()) { haveVolumeEmission = true; break; }
        if (!haveLight && L.scene.emitters.empty() && !haveVolumeEmission) {
            fail("scene has no light: add a 'light' block, an emissive ('emit') mesh, "
                 "or an emissive volume ('temperature' + 'emission blackbody')");
            return false;
        }
        // Catch errors recorded via fail() inside add* helpers that returned true
        // without re-checking `err` (e.g. an unknown `spd preset:`/`spectrum:` name
        // in a light or material silently falls back otherwise). Any recorded error
        // is fatal — surface it instead of rendering a wrong scene.
        if (!err.empty()) return false;

        // Every property has now been read by whichever pass wanted it, so anything
        // still unmarked is a key nothing in the loader understands — a typo, a
        // property put on the wrong block, or an emitter that has drifted from the
        // grammar. Collect them for the caller to report; silently ignoring them is
        // how a misspelt key turns into a wrong image instead of a message.
        for (const auto& b : blocks)
            collectUnusedKeys(b, blockLabel(b), L.unknownKeys);

        // build() finalizes tris/BVH and the emitter set (per-emitter samplers were
        // built in addLight; finalizeEmitters computes powers, the selection CDF,
        // and the combined backward wavelength sampler).
        L.scene.build();
        // build() -> finalizeEmitters() has now adopted each emitter's emitPat from the
        // material on its geometry, so this is the first moment the SHAPE of every
        // emission pattern's emitter is known. Reject the shapes that cannot honour one.
        if (!checkEmitPatsSupported(L)) return false;
        return true;
    }

    // Remove every `mesh { shape_only yes }` group's triangles from Scene::tris.
    //
    // Runs after the deferred medium sweep (the only current consumer of a shape-only
    // mesh's geometry) and before Scene::build(), so the stripped triangles never reach
    // the BVH: they are neither intersected nor drawn, and cost nothing at render time.
    //
    // Safety of renumbering: Scene::tris is referenced by INDEX only through
    // MeshGroup::triStart/triCount, which this function fixes up. Mesh area lights COPY
    // their triangles into Emitter::meshTris (Scene::addMeshLight) rather than keeping a
    // range, and `shape_only` is refused on an emissive material anyway; materials are
    // referenced per-triangle by id, not by position. Everything else that indexes tris
    // (the BVH, hit records) is built later.
    void stripShapeOnlyMeshes(Loaded& L) {
        if (shapeOnlyGroups_.empty()) return;
        // Mark the doomed triangles by group so one linear compaction handles any number
        // of ranges, in any authoring order, without repeated erases.
        std::vector<char> drop(L.scene.tris.size(), 0);
        size_t dropped = 0;
        for (size_t gi : shapeOnlyGroups_) {
            MeshGroup& g = L.scene.meshGroups[gi];
            size_t end = std::min(g.triStart + g.triCount, L.scene.tris.size());
            for (size_t t = g.triStart; t < end; ++t) { if (!drop[t]) { drop[t] = 1; ++dropped; } }
        }
        if (dropped == 0) return;
        // Prefix count of removed triangles, so each surviving group's new triStart is
        // just its old one minus however many were removed before it.
        std::vector<size_t> removedBefore(L.scene.tris.size() + 1, 0);
        for (size_t t = 0; t < L.scene.tris.size(); ++t)
            removedBefore[t + 1] = removedBefore[t] + (drop[t] ? 1 : 0);
        for (MeshGroup& g : L.scene.meshGroups) {
            if (g.blasId >= 0) continue;                      // instanced: not in Scene::tris
            if (g.shapeOnly) { g.triStart = 0; g.triCount = 0; continue; }
            g.triStart -= removedBefore[std::min(g.triStart, L.scene.tris.size())];
        }
        std::vector<Tri> kept;
        kept.reserve(L.scene.tris.size() - dropped);
        for (size_t t = 0; t < L.scene.tris.size(); ++t)
            if (!drop[t]) kept.push_back(L.scene.tris[t]);
        L.scene.tris.swap(kept);
        for (size_t gi : shapeOnlyGroups_)
            std::fprintf(stderr, "[mesh] shape_only \"%s\": geometry consumed as a shape and "
                         "removed from the scene (not rendered)\n",
                         L.scene.meshGroups[gi].name.c_str());
        std::fprintf(stderr, "[mesh] shape_only: %zu triangle(s) stripped, %zu remain\n",
                     dropped, L.scene.tris.size());
    }

    // An emission pattern is read from BOTH sides of transport — emission-on-hit (a
    // PatCtx built from a Hit) and the Le at an emitter-sampled point (a PatCtx built
    // from Emitter::samplePoint) — and MIS combines the two. If they disagree even
    // slightly the render is BIASED, not merely noisy, so a pattern is only legal on the
    // shapes where samplePoint can report exactly the (u,v) a hit would interpolate:
    // Quad (bilinear parameters, matched by addAreaLight's two UV'd tris) and Mesh (the
    // EmitTri's barycentric UVs, the same interpolation geometry.h does). A sphere/tube/
    // spot/env emitter has no such correspondence, so refuse loudly rather than render
    // a wrong image — the same rule checkSlotPatSupported applies to reflect/transmit.
    bool checkEmitPatsSupported(Loaded& L) {
        for (const auto& e : L.scene.emitters) {
            if (e.emitPat < 0) continue;
            if (e.shape == EmitterShape::Quad || e.shape == EmitterShape::Mesh) continue;
            const char* what = (e.shape == EmitterShape::Sphere)   ? "sphere"
                             : (e.shape == EmitterShape::Cylinder) ? "cylinder"
                             : (e.shape == EmitterShape::Spot)     ? "spot"
                             : (e.shape == EmitterShape::Env)      ? "env"
                             : (e.shape == EmitterShape::Sun)      ? "sun" : "this";
            fail(std::string("an emit pattern is not supported on a ") + what +
                 " light — only quad and mesh emitters sample a (u,v) that matches the "
                 "one emission-on-hit interpolates, and a mismatch would bias the image");
            return false;
        }
        return true;
    }

private:
    std::unordered_map<std::string, const Block*> spectraBlocks_;
    std::unordered_map<std::string, int> matIndex_;

    // ---- K1: user-supplied named RGB->spectral upsamplers -----------------------------
    // `upsample "name" { expr "<f of r,g,b,w>" }`, referenced as `rgb:<name> r g b`.
    // The declaration blocks, and the COMPILED program per name (compiled once on first
    // use, then reused for every colour that names it — an upsampler is typically named
    // by many colours, and the compile is pure overhead after the first).
    // Both the program and the spectrum table are held by shared_ptr because the
    // Spectrum this produces must OUTLIVE the Builder (it ends up on a Material, which
    // the Scene owns), while N colours naming the same upsampler should share one copy
    // of each rather than carrying their own.
    std::unordered_map<std::string, const Block*>                             upsampleBlocks_;
    std::unordered_map<std::string, std::shared_ptr<const std::vector<PatNode>>> upsampleProg_;
    // Spectra reachable from an upsample body as `spec:<name>(w)`. A resolved index is
    // into this vector, which is APPEND-ONLY — that is what makes an index handed out at
    // compile time still valid after a later name resolves, even though the vector
    // reallocates. Shared with every closure produced, so a `spec:` sample stays live.
    std::shared_ptr<std::vector<Spectrum>> upsampleSpecs_ =
        std::make_shared<std::vector<Spectrum>>();
    std::unordered_map<std::string, int>  upsampleSpecIndex_;

    // Resolve `spec:<name>` to an index, evaluating the named spectrum block on first
    // reference. Returns -1 for an unknown name, which the tokenizer turns into a compile
    // error naming the spectrum.
    //
    // NOTE the const_cast: a PatSpecScope lookup is a C function pointer taking `const
    // void*`, but resolution genuinely MUTATES — it memoises the evaluated spectrum.
    // (texScopeThunk_ has the same shape but only reads.) Resolving lazily like this is
    // the point: pre-evaluating every spectrum in the scene just in case an upsampler
    // wanted one would do real work for scenes that declare no upsampler at all.
    static int specScopeThunk_(const void* self, const char* name) {
        Builder* B = const_cast<Builder*>(static_cast<const Builder*>(self));
        auto hit = B->upsampleSpecIndex_.find(name);
        if (hit != B->upsampleSpecIndex_.end()) return hit->second;
        auto bit = B->spectraBlocks_.find(name);
        if (bit == B->spectraBlocks_.end()) return -1;
        const Stmt* e = find(*bit->second, "=");
        if (!e) return -1;
        Spectrum s = B->evalSpectrum(e->val, 1);
        int idx = (int)B->upsampleSpecs_->size();
        B->upsampleSpecs_->push_back(std::move(s));
        B->upsampleSpecIndex_[name] = idx;
        return idx;
    }
    // Sampler hook. `self` is the SPECTRUM VECTOR, not the Builder — that is what lets a
    // produced Spectrum outlive the loader.
    static double specSampleThunk_(const void* self, int idx, double w) {
        const auto& v = *static_cast<const std::vector<Spectrum>*>(self);
        return (idx >= 0 && idx < (int)v.size() && v[idx]) ? v[idx](w) : 0.0;
    }
    PatSpecScope specScope_{ this, &Builder::specScopeThunk_ };

    // Turn a colour into a Spectrum through a named upsampler.
    Spectrum applyUpsample(const std::string& name, const Vec3& c) {
        auto bit = upsampleBlocks_.find(name);
        if (bit == upsampleBlocks_.end()) {
            fail("unknown upsampler '" + name + "' — declare it as `upsample \"" + name +
                 "\" { expr \"…\" }`");
            return constantSpectrum(0.0);
        }
        auto pit = upsampleProg_.find(name);
        if (pit == upsampleProg_.end()) {
            const Stmt* e = find(*bit->second, "expr");
            if (!e || e->val.words.empty()) {
                fail("upsample '" + name + "': no `expr`"); return constantSpectrum(0.0);
            }
            std::string expr;
            for (size_t k = 0; k < e->val.words.size(); ++k) { if (k) expr += " "; expr += e->val.words[k]; }
            std::vector<PatNode> prog; std::string perr;
            // PatVarMode::Upsample swaps in the r/g/b/w vocabulary and locks the surface
            // one out; `&specScope_` is what makes `spec:<name>(w)` legal. No texScope_ or
            // tableScope_: there is no hit here, so `tex:` stays a compile error — and a
            // `grid:`/`scatter:` sample, while conceivable, would be a second spelling for
            // what `spec:` already does better (a spectrum knows its own wavelength
            // domain; a grid would make the author restate it).
            if (!compilePatternExpr(expr, prog, perr, /*allowT=*/false, nullptr, nullptr,
                                    /*allowA=*/false, PatVarMode::Upsample, &specScope_)) {
                fail("upsample '" + name + "': " + perr); return constantSpectrum(0.0);
            }
            pit = upsampleProg_.emplace(
                name, std::make_shared<const std::vector<PatNode>>(std::move(prog))).first;
        }
        // The closure evaluates the program at each queried wavelength, exactly like the
        // built-in upsamplers evaluate their own fits — deliberately NOT pre-tabulated.
        // A user upsampler is free to be a narrow emission line or any other sharp
        // feature, and baking it to a fixed grid here would quietly band-limit it; the
        // renderer already tabulates spectra where it needs to (the device tables), at a
        // resolution it chooses.
        //
        // Captures are all by value and own their targets, so the result is independent
        // of this Builder: `prog` and `specs` are shared_ptrs, and `specSelf` points at
        // the shared vector rather than at the loader.
        auto prog  = pit->second;
        auto specs = upsampleSpecs_;
        Vec3 col = c;
        return [prog, specs, col](double w) -> double {
            PatCtx q;
            q.x = col.x; q.y = col.y; q.z = col.z;   // r, g, b  (see PatVarMode)
            q.u = w;                                 // w = wavelength, nm
            q.specFn  = &Builder::specSampleThunk_;
            q.specSelf = specs.get();
            double v = patternEval(prog->data(), (int)prog->size(), q);
            return std::isfinite(v) ? v : 0.0;
        };
    }

    // §3.3 material application. `albedoDefault_` is the per-material fallback for the
    // named input `a` — the one input with NO per-hit intrinsic, so an unbound `a` has
    // to be resolved at LOAD time (mirrors loom's Material.albedo_default, same 1.0
    // default). Kept loader-side rather than on Material because Material is uploaded
    // to the device and `a` never survives past load. `applyCache_` memoises
    // "<matIdx>(<args>)" -> materialised index so one application shared by N objects
    // builds ONE material.
    std::unordered_map<int, double>      albedoDefault_;
    std::unordered_map<std::string, int> applyCache_;

    std::unordered_map<std::string, int> textureIndex_;   // texture name -> Scene::textures index
    std::unordered_map<std::string, int> patternIndex_;   // pattern name -> Scene::patterns index

    // Generated-block name (`__arrN`) -> the AUTHOR-facing site that produced it.
    // `desugarArrays` mints anonymous `grid`/`pattern` blocks the author never named, so
    // any error raised while BUILDING one must be re-attributed: a message about
    // `pattern '__arr3'` names a symbol that appears nowhere in the scene file. Populated
    // by `desugarOne`, consulted by `genWho` below.
    std::unordered_map<std::string, std::string> genSite_;
    // "pattern 'foo'" for an authored block; "line 12: `reflect`: the inline array literal"
    // for a generated one. Every fail() that can fire on a generated block goes through it.
    std::string genWho(const char* kind, const std::string& name) const {
        auto it = genSite_.find(name);
        if (it != genSite_.end()) return it->second;
        return std::string(kind) + " '" + name + "'";
    }

    // Texture scope for `tex:<name>(u, v)` samples inside a pattern expression. Passed
    // to compilePatternExpr ONLY at value sites that are evaluated with a shading
    // context (material / pattern / record-driver expressions); leaving it off at the
    // others (implicit field formulas, medium density/ior, load-time constant sites)
    // is what turns an unperformable sample into a clear compile error. Resolution
    // reads textureIndex_, which Pass 1b fills before patterns / records / materials
    // are built — so file order does not matter for those. The one ordering rule is
    // texture-samples-texture: a procedural `texture { rgb "…" }` bakes DURING Pass
    // 1b, so it can only sample textures declared ABOVE it (and never itself, which
    // therefore fails cleanly as "unknown texture" rather than recursing).
    static int texScopeThunk_(const void* self, const char* name) {
        const auto& idx = static_cast<const Builder*>(self)->textureIndex_;
        auto it = idx.find(name);
        return (it == idx.end()) ? -1 : it->second;
    }
    PatTexScope texScope_{ this, &Builder::texScopeThunk_ };

    // N-D table scope for `grid:<name>(c0, …)` and `scatter:<name>(c0, …)` samples
    // inside a pattern expression. Same deal as texScope_, with one extra job: it also
    // reports the table's DIMENSIONALITY, because the call arity is a property of the
    // table itself (a 2-D grid takes two coordinates), not of the function name — so
    // the compiler can only check the argument count once the name resolves. Filled by
    // the data pass, which runs before textures/patterns/records, so authoring order
    // doesn't matter for those.
    std::unordered_map<std::string, int> gridIndex_;         // name -> Scene::grids index
    std::unordered_map<std::string, int> scatterIndex_;      // name -> Scene::scatters index
    const std::vector<PatGrid>*    gridsRef_    = nullptr;   // -> L.scene.grids    (ndim report)
    const std::vector<PatScatter>* scattersRef_ = nullptr;   // -> L.scene.scatters (ndim report)
    static int tableScopeThunk_(const void* self, PatTableKind kind, const char* name, int* ndim) {
        const Builder* bl = static_cast<const Builder*>(self);
        if (kind == PatTableKind::Grid) {
            auto it = bl->gridIndex_.find(name);
            if (it == bl->gridIndex_.end()) return -1;
            if (ndim && bl->gridsRef_ && it->second < (int)bl->gridsRef_->size())
                *ndim = (*bl->gridsRef_)[it->second].ndim;
            return it->second;
        }
        auto it = bl->scatterIndex_.find(name);
        if (it == bl->scatterIndex_.end()) return -1;
        if (ndim && bl->scattersRef_ && it->second < (int)bl->scattersRef_->size())
            *ndim = (*bl->scattersRef_)[it->second].ndim;
        return it->second;
    }
    PatTableScope tableScope_{ this, &Builder::tableScopeThunk_ };
    std::unordered_map<std::string, int> recordIndex_;    // record name  -> Scene::records index
    const std::vector<Record>* records_ = nullptr;        // -> L.scene.records (set in build; for record refs at value sites)
    // -> the Loaded being built. Needed by materialPropRef, which reaches value sites
    // (evalSpectrum / dblParam / bindScalarPattern) that were never handed a `Loaded&`,
    // yet may APPEND to Scene::mats and Scene::patterns via applyMaterial. Deliberately a
    // pointer to the owner, never to an element: the vectors reallocate.
    Loaded* loadedRef_ = nullptr;
    std::unordered_map<std::string, Spectrum> spdFileCache_; // path -> loaded measured SPD

    // Named-object registries for `medium { bounds { object "name" } }` resolution.
    // Populated by the geometry builders during Pass 3; read by addMedium afterwards.
    struct NamedSphere { Vec3 center; double radius; };
    std::unordered_map<std::string, NamedSphere> sphereByName_;   // named sphere -> world center/radius
    std::unordered_map<std::string, int>         implicitByName_; // named isosurface -> Scene::implicits index
    std::unordered_map<std::string, Aabb>        meshAabbByName_; // named mesh -> world AABB
    std::unordered_map<std::string, int>         blasIndex_;      // mesh_asset name -> Scene::blasList index
    // `mesh { shape_only yes }` groups (indices into Scene::meshGroups), removed from
    // Scene::tris by stripShapeOnlyMeshes() once the deferred medium sweep has read them.
    std::vector<size_t>                          shapeOnlyGroups_;

    double L_ = 1.0;              // authored length -> internal metres
    double binWidth_ = 1.0;      // spectral sampling bin width (nm)

    // Scale an authored position/length into internal (metre) units.
    Vec3 P(const Vec3& v) const { return v * L_; }
    double Len(double d) const { return d * L_; }

    void fail(const std::string& m) { if (err.empty()) err = m; }

    int matId(const std::string& name) {
        auto it = matIndex_.find(name);
        if (it == matIndex_.end()) { fail("unknown material '" + name + "'"); return 0; }
        return it->second;
    }

    // Map a slot keyword to its RecSlot id, or -1 if it isn't a record-fillable slot.
    static int recSlotId(const std::string& name) {
        if (name == "reflect")   return REC_SLOT_REFLECT;
        if (name == "roughness") return REC_SLOT_ROUGHNESS;
        return -1;
    }

    // Install one slot binding, applying last-write-wins: drop any existing binding for
    // the same slot first (so a later `from`/assignment overrides an earlier one).
    static void setBinding(Material& m, const RecBinding& rb) {
        auto& v = m.recBindings;
        v.erase(std::remove_if(v.begin(), v.end(),
                               [&](const RecBinding& e) { return e.slot == rb.slot; }),
                v.end());
        v.push_back(rb);
    }

    // Apply `from R(drv)`: bind every slot whose name matches a channel of record
    // `recIdx` (the kind must match the slot — reflect wants a Spectrum channel,
    // roughness a Scalar channel), each sampled at the shared driver `drv`. Lenient:
    // an unmatched channel is ignored, an unfilled slot keeps its constant.
    static void applyFrom(Material& m, int recIdx, const std::vector<PatNode>& drv,
                          const Record& rec) {
        int ci = rec.channelIndex("reflect");
        if (ci >= 0 && rec.channels[ci].kind == ChanKind::Spectrum) {
            RecBinding rb; rb.slot = REC_SLOT_REFLECT; rb.recordIndex = recIdx;
            rb.channel = ci; rb.driver = drv; setBinding(m, rb);
        }
        int ri = rec.channelIndex("roughness");
        if (ri >= 0 && rec.channels[ri].kind == ChanKind::Scalar) {
            RecBinding rb; rb.slot = REC_SLOT_ROUGHNESS; rb.recordIndex = recIdx;
            rb.channel = ri; rb.driver = drv; setBinding(m, rb);
        }
    }

    // Synthesize a record-driven material (§records §2.2): a default (diffuse) material
    // whose slots are filled per-hit by record `recIdx` sampled at driver `driverExpr`
    // (the inline `material R(driver)` form — equivalent to a lone `from R(driver)`).
    // Returns the new material's index in Scene::mats, or -1 on error.
    int buildRecordMaterial(int recIdx, const std::string& driverExpr, Loaded& L) {
        Material m;   // default type: diffuse
        m.reflect = constantSpectrum(0.75);
        std::vector<PatNode> drv;
        std::string cerr;
        if (!compilePatternExpr(driverExpr, drv, cerr, /*allowT=*/false, &texScope_,
                                &tableScope_, /*allowA=*/true)) {
            fail("record material driver '" + driverExpr + "': " + cerr);
            return -1;
        }
        applyFrom(m, recIdx, drv, L.scene.records[recIdx]);
        int id = (int)L.scene.mats.size();
        L.scene.mats.push_back(std::move(m));
        return id;
    }

    // ===================== §3.3 materials as parameterized bundles ==============
    // A material is a bundle of slot->expression bindings, and is itself a FUNCTION:
    // its free-input set is the union of its slots' free inputs. Applying it at a use
    // site — `material gold(u=v, a=1)` — binds those inputs across the whole bundle at
    // once (ROADMAP_records.md §3.3), which is §3.2's per-property rebinding lifted to
    // bundle granularity.
    //
    // Implemented as BINDING BY SUBSTITUTION. Every slot program is POSTFIX, so a
    // variable node pushes exactly one value and so does a well-formed argument
    // program: binding is a pure splice (patternSubstitute). No environment, no
    // closure, no runtime indirection — an applied material is just another Material,
    // so the device upload, the CPU evaluator and every sampler stay untouched. That
    // is what keeps the feature additive: a material nobody applies is bit-identical
    // to before, and `applyMaterial` returns the ORIGINAL index for a no-op call.

    // The material's free-input set: the union of the inputs read by its pattern slots
    // and by its record bindings' drivers, in order of first use (so diagnostics list
    // inputs the way the author wrote them). Keep the slot list in sync with
    // Material's `*Pat` members (scene.h).
    static void materialFreeInputs(const Material& m, const Loaded& L,
                                   std::vector<PatOp>& out) {
        out.clear();
        const int pats[6] = { m.roughnessPat, m.filmThicknessPat, m.mixWeightPat,
                              m.reflectPat,   m.transmitPat,      m.emitPat };
        for (int pi : pats)
            if (pi >= 0 && pi < (int)L.scene.patterns.size())
                patternCollectVars(L.scene.patterns[pi].nodes, out);
        for (const RecBinding& rb : m.recBindings) patternCollectVars(rb.driver, out);
    }

    // An argument list MAY contain spaces (`f(0.5*u + 0.5*v)`, `gold(a=1, u=v)`): the
    // shared grammar lexes a balanced paren group as part of one WORD, so the value is
    // no longer truncated at the first space. Reaching a "missing ')'" now means the
    // parens really are unbalanced — say so plainly.
    static std::string parenHint(const std::string& raw) {
        if (raw.find(')') != std::string::npos) return "";
        return " — the parentheses are unbalanced (a group must close on the same line)";
    }

    // The fallback value of the named input `a` for material `matIdx` — its authored
    // `albedo_default <x>`, or 1.0.
    double albedoDefaultFor(int matIdx) const {
        auto it = albedoDefault_.find(matIdx);
        return (it == albedoDefault_.end()) ? 1.0 : it->second;
    }

    static std::string trimWs(const std::string& s) {
        size_t a = 0, b = s.size();
        while (a < b && isspace((unsigned char)s[a])) ++a;
        while (b > a && isspace((unsigned char)s[b - 1])) --b;
        return s.substr(a, b - a);
    }

    // Split a material application's argument text into (name, expr) pieces; an empty
    // name means a POSITIONAL argument. Uses the same delimiter ladder as record stops
    // (§3.1) — a top-level comma and a run of whitespace are equally valid separators,
    // so `gold(u=v, a=1)` and `gold(u=v a=1)` are identical. `(` `)` and `[` `]` nest,
    // so an argument may itself be a call.
    //
    // Boundaries are found from the `=` signs rather than from the whitespace, because
    // an argument is an ARBITRARY EXPRESSION that may contain spaces (`gold(u = v * 2)`).
    // Every top-level `=` starts a named argument whose name is the identifier just to
    // its left, so the argument begins at that identifier. This is unambiguous: the
    // pattern language has NO comparison operators, so a top-level `=` can only ever
    // mean a binding and never continues an expression.
    bool parseBindArgs(const std::string& text,
                       std::vector<std::pair<std::string, std::string>>& out,
                       std::string& e) {
        out.clear();
        const size_t n = text.size();
        auto isIdent  = [](char c) { return isalnum((unsigned char)c) || c == '_'; };
        auto isIdent0 = [](char c) { return isalpha((unsigned char)c) || c == '_'; };

        std::vector<size_t> starts;      // where each argument begins
        starts.push_back(0);
        int depth = 0;
        for (size_t i = 0; i < n; ++i) {
            char c = text[i];
            if (c == '(' || c == '[') { ++depth; continue; }
            if (c == ')' || c == ']') {
                if (--depth < 0) { e = "unbalanced ')' in argument list"; return false; }
                continue;
            }
            if (depth) continue;
            if (c == ',') { starts.push_back(i + 1); continue; }
            if (c != '=') continue;
            size_t j = i;                                   // back over spaces before `=`
            while (j > 0 && isspace((unsigned char)text[j - 1])) --j;
            size_t k = j;                                   // back over the input name
            while (k > 0 && isIdent(text[k - 1])) --k;
            if (k == j)          { e = "'=' with no input name on its left"; return false; }
            if (!isIdent0(text[k])) {
                e = "invalid input name '" + text.substr(k, j - k) + "'"; return false;
            }
            starts.push_back(k);
        }
        if (depth) { e = "unbalanced '(' in argument list"; return false; }

        std::sort(starts.begin(), starts.end());
        starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
        for (size_t s = 0; s < starts.size(); ++s) {
            size_t b = starts[s], en = (s + 1 < starts.size()) ? starts[s + 1] : n;
            std::string seg = trimWs(text.substr(b, en - b));
            if (!seg.empty() && seg.back() == ',') seg = trimWs(seg.substr(0, seg.size() - 1));
            if (seg.empty()) continue;
            // A named piece always carries its `=` at top level (that `=` is what cut here).
            size_t eq = std::string::npos;
            int d = 0;
            for (size_t i = 0; i < seg.size(); ++i) {
                char c = seg[i];
                if (c == '(' || c == '[') ++d;
                else if (c == ')' || c == ']') --d;
                else if (c == '=' && d == 0) { eq = i; break; }
            }
            if (eq == std::string::npos) { out.emplace_back(std::string(), seg); continue; }
            std::string nm = trimWs(seg.substr(0, eq)), ex = trimWs(seg.substr(eq + 1));
            if (nm.empty()) { e = "'=' with no input name on its left"; return false; }
            if (ex.empty()) { e = "input '" + nm + "' bound to an empty expression"; return false; }
            out.emplace_back(nm, ex);
        }
        return true;
    }

    // Apply material `matIdx` at a use site, binding its named inputs. Clones the
    // material, splices each argument program into every slot that reads the bound
    // input, resolves any still-free `a` (albedo — the one input with no per-hit
    // intrinsic) to the material's `albedo_default`, and returns the new index.
    // Returns `matIdx` UNCHANGED when the application binds nothing, so an ordinary
    // use costs nothing. Results are memoised on (material, argument text) so
    // `gold(u=v)` shared by 500 spheres builds ONE material, not 500.
    int applyMaterial(int matIdx, const std::string& argText, Loaded& L) {
        std::string key = std::to_string(matIdx) + "(" + trimWs(argText) + ")";
        auto cit = applyCache_.find(key);
        if (cit != applyCache_.end()) return cit->second;

        std::vector<std::pair<std::string, std::string>> args;
        std::string e;
        if (!parseBindArgs(argText, args, e)) {
            fail("material application '" + argText + "': " + e); return -1;
        }
        std::vector<PatOp> freeIn;
        materialFreeInputs(L.scene.mats[matIdx], L, freeIn);

        std::vector<PatBind> binds;
        auto alreadyBound = [&](PatOp v) {
            for (const PatBind& b : binds) if (b.var == v) return true;
            return false;
        };
        // NAMED arguments are resolved first, whatever order they were written in, so a
        // positional argument sees only the inputs still unbound — loom's
        // `free_inputs() - set(binds)` rule (Material.apply, tools/loom/loom/scene.py).
        // That makes `gold(0.5, a=1)` legal for a two-input bundle: `a` is taken by name,
        // leaving exactly one free input for the positional.
        std::vector<std::pair<std::string, std::string>> ordered;
        for (auto& a : args) if (!a.first.empty()) ordered.push_back(a);
        size_t nPositional = 0;
        for (auto& a : args) if (a.first.empty()) { ordered.push_back(a); ++nPositional; }
        if (nPositional > 1) {
            fail("material application '" + argText +
                 "': at most one positional argument (bind the rest by name)");
            return -1;
        }
        for (auto& a : ordered) {
            PatOp var;
            if (a.first.empty()) {
                // Positional. Fragile with several inputs, so §3.3 allows it only when
                // exactly one input is still free (matching `RECORD(driver)`).
                std::vector<PatOp> rest;
                for (PatOp v : freeIn) if (!alreadyBound(v)) rest.push_back(v);
                if (rest.size() != 1) {
                    std::string names;
                    for (size_t i = 0; i < rest.size(); ++i)
                        names += (i ? ", " : "") + std::string(varName(rest[i]));
                    fail("material application '" + argText + "': a positional argument "
                         "needs exactly one still-free input, but there are " +
                         std::to_string(rest.size()) +
                         (rest.empty() ? "" : " (" + names + ")") +
                         " — bind by name");
                    return -1;
                }
                var = rest[0];
            } else if (!varOp(a.first, var)) {
                fail("material application '" + argText + "': '" + a.first +
                     "' is not a bindable input"); return -1;
            }
            if (alreadyBound(var)) {
                fail("material application '" + argText + "': input '" +
                     std::string(varName(var)) + "' bound twice"); return -1;
            }
            // The RHS is evaluated in the CONSUMER's scope, so `a` is not in scope here
            // (the consumer is geometry, which has no albedo to offer).
            PatBind pb; pb.var = var;
            std::string cerr;
            if (!compilePatternExpr(a.second, pb.repl, cerr, /*allowT=*/false,
                                    &texScope_, &tableScope_)) {
                fail("material application '" + argText + "': binding '" +
                     std::string(varName(var)) + "=" + a.second + "': " + cerr);
                return -1;
            }
            binds.push_back(std::move(pb));
        }

        // `a` is the one named input with NO per-hit intrinsic, so an unbound `a` must
        // be resolved at LOAD time or it would silently read 0. Fall back to the
        // material's `albedo_default` (mirrors loom's Material.albedo_default).
        bool aFree = false;
        for (PatOp v : freeIn) if (v == PatOp::VarA) { aFree = true; break; }
        if (aFree && !alreadyBound(PatOp::VarA)) {
            PatBind pb; pb.var = PatOp::VarA;
            PatNode nd; nd.op = PatOp::Const; nd.a = albedoDefaultFor(matIdx);
            pb.repl.push_back(nd);
            binds.push_back(std::move(pb));
        }
        if (binds.empty()) { applyCache_[key] = matIdx; return matIdx; }

        Material m = L.scene.mats[matIdx];
        int* slots[6] = { &m.roughnessPat, &m.filmThicknessPat, &m.mixWeightPat,
                          &m.reflectPat,   &m.transmitPat,      &m.emitPat };
        for (int* pi : slots) {
            if (*pi < 0 || *pi >= (int)L.scene.patterns.size()) continue;
            const std::vector<PatNode>& src = L.scene.patterns[*pi].nodes;
            bool touched = false;
            for (const PatBind& b : binds) if (patternUsesVar(src, b.var)) { touched = true; break; }
            if (!touched) continue;                      // share the original program
            Pattern np; np.nodes = patternSubstitute(src, binds);
            *pi = (int)L.scene.patterns.size();
            L.scene.patterns.push_back(std::move(np));
        }
        for (RecBinding& rb : m.recBindings) rb.driver = patternSubstitute(rb.driver, binds);

        int id = (int)L.scene.mats.size();
        L.scene.mats.push_back(std::move(m));
        applyCache_[key] = id;
        return id;
    }

    // Resolve a geometry block's `material` field to a Scene::mats index. Accepts both
    // a plain material name and the inline record form `RECORD(driver)` (§2.2). Sets
    // `err` (via fail) and returns -1 on any problem; if the field is absent, fails
    // with "<geom> needs a material". `optional` suppresses the absent-field error and
    // returns -1 silently (for an optional per-instance override).
    int matFieldId(const Block& b, Loaded& L, const char* geom, bool optional = false) {
        const Stmt* s = find(b, "material");
        if (!s || s->val.words.empty()) {
            if (!optional) fail(std::string(geom) + " needs a material");
            return -1;
        }
        // Reconstruct the raw field text (a spaced driver expression spans >1 token).
        std::string raw = s->val.words[0];
        for (size_t k = 1; k < s->val.words.size(); ++k) raw += " " + s->val.words[k];
        size_t lp = raw.find('(');
        if (lp != std::string::npos) {
            std::string name = raw.substr(0, lp);
            auto rit = recordIndex_.find(name);
            if (rit != recordIndex_.end()) {
                size_t rp = raw.rfind(')');
                if (rp == std::string::npos || rp <= lp) {
                    fail("record material '" + raw + "': missing ')'" + parenHint(raw));
                    return -1;
                }
                std::string driver = raw.substr(lp + 1, rp - lp - 1);
                return buildRecordMaterial(rit->second, driver, L);
            }
            // Not a record: a §3.3 material application, `gold(u=v, a=1)`.
            auto mit = matIndex_.find(trimWs(name));
            if (mit != matIndex_.end()) {
                size_t rp = raw.rfind(')');
                if (rp == std::string::npos || rp <= lp) {
                    fail("material application '" + raw + "': missing ')'" + parenHint(raw));
                    return -1;
                }
                return applyMaterial(mit->second, raw.substr(lp + 1, rp - lp - 1), L);
            }
        }
        int id = lookupMaterial(raw, L);
        if (id < 0) { fail("unknown material '" + raw + "'"); return -1; }
        return id;
    }

    // Resolve a plain material NAME to a usable Scene::mats index. Every by-name
    // reference goes through here so that a material with a free `a` resolves it to
    // its `albedo_default` exactly once (applyMaterial memoises the empty application),
    // instead of silently reading 0. Returns -1 if the name is unknown.
    int lookupMaterial(const std::string& name, Loaded& L) {
        auto it = matIndex_.find(name);
        if (it == matIndex_.end()) return -1;
        return applyMaterial(it->second, "", L);
    }

    // ---- §3.2 per-property access: `MATERIAL.slot` / `MATERIAL.slot(args)` -------
    // The value of ONE property of an already-declared material, readable at any value
    // site that takes that property's type. `gold.reflect` reads the slot with every
    // named input at its default; `gold.reflect(u=v)` / `gold.reflect(a=0.3)` rebinds
    // first. Rebinding goes through `applyMaterial`, so it is the *same* machinery §3.3
    // uses at a geometry `material` field — including the `a` -> `albedo_default`
    // fallback and the (material, argtext) memo, so `gold.reflect(u=v)` written in five
    // places builds one applied material, not five.
    //
    // WHY THE SLOT KEYWORD IS THE HANDLE. §3.2 writes a property as `<type/slot keyword>
    // ["name"] = <value>` and makes the quoted name OPTIONAL, because the slot keyword
    // alone already binds the property to its slot; the name exists only to mint an
    // external dot-handle. ftrace properties are spelled with the slot keyword and have
    // never carried a quoted name — every ftrace property is already anonymous, so the
    // "naming is optional" arm is the ftrace status quo, vacuously. With no names to use
    // as handles, the handle here IS the slot keyword: `gold.reflect`, `gold.roughness`.
    struct MatProp {
        bool        ok       = false;                    // resolved (false => fail() was set)
        bool        spectral = false;                    // spectral slot (else scalar)
        Spectrum    spec     = constantSpectrum(0.0);
        double      scalar   = 0.0;
        int         pat      = -1;                       // companion per-hit pattern, or -1
        std::string slot;
    };

    // ---- a table sampled DIRECTLY at a value site --------------------------------
    // `roughness grid:bumps(u,v)`, `reflect scatter:swatch(u,v)` — the NAMED counterpart
    // of an inline array literal, and the same thing semantically: an inline `[0 1](u)`
    // desugars to exactly this expression wrapped in an anonymous `pattern`, so the two
    // spellings must produce the same binding. That symmetry is the whole feature; the
    // workaround it retires is writing the one-line `pattern` wrapper by hand.
    //
    // A value site is NOT an expression site, which is why this needs code at all: ftrace's
    // expression compiler has always read `grid:name(args)` as a call, but only *inside* a
    // pattern body. At a value site the token previously reached the spectrum reader and
    // died as "unrecognized spectrum expression".
    //
    // The **scoped** spelling is the one accepted, deliberately. A bare `ramp(u)` at a value
    // site already means "apply the material `ramp`" (§7.6 bundles), so accepting it for
    // tables too would make the meaning depend on which namespace happens to hold the name —
    // and a scene could change meaning by gaining a material. `grid:` / `scatter:` cannot
    // collide with that.
    //
    // Returns the new pattern's index, or -1 with fail() already set. Callers test the
    // prefix themselves (it is two string compares) so that each can phrase its own
    // refusal when its slot has nowhere to put a per-hit value.
    static bool isTableCallHead(const std::string& t) {
        return t.rfind("grid:", 0) == 0 || t.rfind("scatter:", 0) == 0;
    }
    int tableCallPattern(const std::string& tok, const std::string& who) {
        if (!loadedRef_) {                    // no scene to append the pattern to
            fail(who + ": `" + tok + "` cannot be used here");
            return -1;
        }
        const bool isGrid = (tok[0] == 'g');
        const std::string kind = isGrid ? "grid" : "scatter";
        // The call is required for the same reason an array literal's is: a table is a
        // function of its coordinates, and a slot holds a value, so naming one without
        // saying where it is read is incomplete rather than defaulted. Refusing beats
        // inventing `(u)`, which would silently pick an axis for a 2-D table.
        if (tok.find('(') == std::string::npos || tok.back() != ')') {
            fail(who + ": `" + tok + "` names a " + kind + " but does not sample it — a "
                 "table is read AT coordinates, so write `" + tok + "(u)`, one coordinate "
                 "per axis (`" + tok + "(u,v)` for a 2-D table)");
            return -1;
        }
        Pattern p;
        std::string perr;
        // `a` is allowed for the same reason a named `pattern` block allows it: the axis can
        // be left free here and rebound where the material is USED (`mat(a=u)`), which is
        // exactly the deferral route an inline literal spells `[0 1](a)`.
        if (!compilePatternExpr(tok, p.nodes, perr, false, &texScope_,
                                &tableScope_, /*allowA=*/true)) {
            fail(who + " `" + tok + "`: " + perr);
            return -1;
        }
        Loaded& L = *loadedRef_;
        L.scene.patterns.push_back(std::move(p));
        return (int)L.scene.patterns.size() - 1;
    }

    // Recognise + resolve. Returns FALSE when `tok` is not this form at all (the caller
    // falls through to its other spellings); TRUE when it was recognised, in which case
    // `out` is filled or `fail()` has been set. Deliberately keyed on "the head names a
    // known MATERIAL", exactly like the record refs below key on "names a known record",
    // so the two dotted forms never contend: a name cannot be both.
    bool materialPropRef(const std::string& tok, MatProp& out) {
        if (!loadedRef_) return false;
        Loaded& L = *loadedRef_;
        size_t dot = tok.find('.');
        if (dot == std::string::npos || dot == 0) return false;
        std::string head = tok.substr(0, dot);
        // Records win a name clash, so `R.chan` keeps meaning the shipped record ref at
        // EVERY value site — including the ones (bindScalarPattern) that reach materials
        // before records. Without this, declaring a material named after a record would
        // silently change what an existing scene's `R.chan` resolves to.
        if (recordIndex_.count(head)) return false;
        auto mit = matIndex_.find(head);
        if (mit == matIndex_.end()) return false;        // not a material -> not our form
        // `slot` or `slot(args)`; the arg list may contain spaces (balanced-paren lexing).
        std::string rest = tok.substr(dot + 1), slot = trimWs(rest), args;
        size_t lp = rest.find('(');
        if (lp != std::string::npos) {
            size_t rp = rest.rfind(')');
            if (rp == std::string::npos || rp <= lp) {
                fail("material property '" + tok + "': missing ')'" + parenHint(tok)); return true;
            }
            std::string tail = trimWs(rest.substr(rp + 1));
            if (!tail.empty()) {
                fail("material property '" + tok + "': unexpected '" + tail + "' after ')'"); return true;
            }
            slot = trimWs(rest.substr(0, lp));
            args = rest.substr(lp + 1, rp - lp - 1);
        }
        if (slot.empty()) { fail("material property '" + tok + "': no property after '.'"); return true; }

        int src = applyMaterial(mit->second, args, L);   // -1 => applyMaterial already failed
        if (src < 0 || src >= (int)L.scene.mats.size()) return true;
        const Material& sm = L.scene.mats[src];
        out.slot = slot;
        // A record-DRIVEN slot has no load-time value at all: the constant sitting in the
        // field is a placeholder the per-hit record sampler overwrites. Reading it would
        // hand the consumer a number the source material never actually uses.
        auto recDriven = [&](int recSlot) {
            for (const RecBinding& rb : sm.recBindings) if (rb.slot == recSlot) return true;
            return false;
        };
        auto refuseRec = [&](int recSlot) {
            if (!recDriven(recSlot)) return false;
            fail("material property '" + tok + "': '" + head + "." + slot + "' is driven by a "
                 "record, so it has no load-time value — reference the record channel "
                 "directly instead");
            return true;
        };
        // Likewise a texture-bound slot: the value is an image sampled at the hit UV, and
        // a property reference carries a spectrum + a pattern, not a texture binding.
        auto refuseTex = [&](int tex) {
            if (tex < 0) return false;
            fail("material property '" + tok + "': '" + head + "." + slot + "' is bound to a "
                 "texture, which a property reference cannot carry — bind the texture at "
                 "the use site instead");
            return true;
        };
        if (slot == "reflect") {
            if (refuseRec(REC_SLOT_REFLECT) || refuseTex(sm.reflectTex)) return true;
            out.spectral = true; out.spec = sm.reflect;  out.pat = sm.reflectPat;
        } else if (slot == "transmit") {
            out.spectral = true; out.spec = sm.transmit; out.pat = sm.transmitPat;
        } else if (slot == "emit") {
            out.spectral = true; out.spec = sm.emit;     out.pat = sm.emitPat;
        } else if (slot == "ior") {
            out.spectral = true; out.spec = sm.ior;
        } else if (slot == "absorb") {
            out.spectral = true; out.spec = sm.absorb;
        } else if (slot == "roughness") {
            if (refuseRec(REC_SLOT_ROUGHNESS) || refuseTex(sm.roughnessTex)) return true;
            out.scalar = sm.roughness;      out.pat = sm.roughnessPat;
        } else if (slot == "film_thickness") {
            if (refuseTex(sm.filmThicknessTex)) return true;
            out.scalar = sm.filmThickness;  out.pat = sm.filmThicknessPat;
        } else if (slot == "film_ior") {
            out.scalar = sm.filmIor;
        } else if (slot == "groove_spacing") {
            out.scalar = sm.grooveSpacing;
        } else if (slot == "yield") {
            out.scalar = sm.fluoYield;
        } else {
            fail("material property '" + tok + "': material '" + head + "' has no property '" +
                 slot + "' (spectral: reflect, transmit, emit, ior, absorb; scalar: roughness, "
                 "film_thickness, film_ior, groove_spacing, yield)");
            return true;
        }
        out.ok = true;
        return true;
    }

    // Combine two per-hit scalar patterns into one. Both spellings a pattern index can
    // arrive from — the source slot of a property reference and the consumer's own
    // `<slot>_map` — mean "a multiplier on whatever the slot otherwise evaluates to", so
    // the composition of the two IS their product. Appending `[a…, b…, Mul]` is valid
    // postfix because each program pushes exactly one value (the same invariant that
    // makes §3.3's substitution a pure splice). Returns the surviving index when only
    // one is real, so the common case allocates nothing.
    int composePatterns(int a, int b, Loaded& L) {
        if (a < 0) return b;
        if (b < 0) return a;
        if (a >= (int)L.scene.patterns.size() || b >= (int)L.scene.patterns.size()) return a;
        Pattern np;
        const std::vector<PatNode>& pa = L.scene.patterns[a].nodes;
        const std::vector<PatNode>& pb = L.scene.patterns[b].nodes;
        np.nodes.reserve(pa.size() + pb.size() + 1);
        np.nodes.insert(np.nodes.end(), pa.begin(), pa.end());
        np.nodes.insert(np.nodes.end(), pb.begin(), pb.end());
        PatNode mul; mul.op = PatOp::Mul;
        np.nodes.push_back(mul);
        int id = (int)L.scene.patterns.size();
        L.scene.patterns.push_back(std::move(np));
        return id;
    }

    // Records stage 5a: resolve a record channel reference used as a CONSTANT spectrum
    // value — `RECORD.channel[i]` (the channel's i-th stop colour) or `RECORD.channel(c)`
    // (sample the colour channel at a constant driver `c`). Accepted anywhere a spectrum
    // is read (light `spd`, top-level `spectrum`, material `reflect`/`transmit`/…).
    // Returns true if `tok` names a known record — the form was recognised, so `out` is
    // filled or `fail` is set — and false if `tok` isn't a record ref (the caller then
    // falls through to the other spectrum forms). The `(c)` driver must be a load-time
    // constant: a per-hit driver like `R.chan(u)` is a scope error at a value site (a
    // constant site publishes no per-hit variables), matching the 5a free-variable rule.
    bool recordConstSpectrumRef(const std::string& tok, Spectrum& out) {
        size_t dot = tok.find('.');
        if (dot == std::string::npos) return false;
        std::string head = tok.substr(0, dot);
        auto rit = recordIndex_.find(head);
        if (rit == recordIndex_.end()) return false;                 // not a record -> not our form
        if (!records_ || rit->second >= (int)records_->size()) return false;
        const Record& rec = (*records_)[rit->second];
        std::string rest = tok.substr(dot + 1);                      // channel[i] | channel(c) | channel
        out = constantSpectrum(0);
        auto colourChannel = [&](const std::string& chan, int& ci) -> bool {
            ci = rec.channelIndex(chan);
            if (ci < 0) { fail("record ref '" + tok + "': record '" + head + "' has no channel '" + chan + "'"); return false; }
            if (rec.channels[ci].kind != ChanKind::Spectrum) {
                fail("record ref '" + tok + "': channel '" + chan + "' is scalar, not a colour channel"); return false;
            }
            return true;
        };
        // Stop selector: `channel[i]`.
        size_t lb = rest.find('[');
        if (lb != std::string::npos) {
            size_t rb = rest.rfind(']');
            std::string idxs = (rb != std::string::npos && rb > lb) ? rest.substr(lb + 1, rb - lb - 1) : "";
            if (idxs.empty() || !isNumber(idxs)) { fail("record ref '" + tok + "': bad stop selector"); return true; }
            int ci; if (!colourChannel(rest.substr(0, lb), ci)) return true;
            const RecChannel& ch = rec.channels[ci];
            int i = (int)num(idxs);
            if (i < 0 || i >= (int)ch.stops.size()) {
                fail("record ref '" + tok + "': stop index " + idxs + " out of range (0.." +
                     std::to_string((int)ch.stops.size() - 1) + ")"); return true;
            }
            out = ch.stops[i].color;
            return true;
        }
        // Sample form: `channel(c)` with a constant driver c.
        size_t lp = rest.find('(');
        if (lp != std::string::npos) {
            size_t rp = rest.rfind(')');
            if (rp == std::string::npos || rp <= lp) { fail("record ref '" + tok + "': malformed `channel(constant)`"); return true; }
            std::string cexpr = rest.substr(lp + 1, rp - lp - 1);
            int ci; if (!colourChannel(rest.substr(0, lp), ci)) return true;
            std::vector<PatNode> drv; std::string cerr;
            if (!compilePatternExpr(cexpr, drv, cerr)) { fail("record ref '" + tok + "' driver '" + cexpr + "': " + cerr); return true; }
            if (patternHasFreeVars(drv)) {
                fail("record ref '" + tok + "': driver must be a constant here — no per-hit variables "
                     "(x/y/z/u/v/…) are in scope at this value site"); return true;
            }
            PatCtx zero{};
            double c = drv.empty() ? 0.0 : patternEval(drv.data(), (int)drv.size(), zero);
            out = recSampleSpectrum(rec, rec.channels[ci], c);
            return true;
        }
        // Bare `RECORD.channel` with no selector/sample: ambiguous at a constant site.
        fail("record ref '" + tok + "': use `RECORD.channel[i]` (a stop) or `RECORD.channel(constant)` at a value site");
        return true;
    }

    // Records stage 5a (scalar sibling of recordConstSpectrumRef): resolve a record
    // channel reference used as a CONSTANT scalar value — `RECORD.channel[i]` (the
    // channel's i-th stop value) or `RECORD.channel(c)` (sample the scalar channel at a
    // constant driver `c`). Returns true if `tok` names a known record (recognised — `out`
    // filled or `fail` set) and false if it isn't a record ref (caller falls through to
    // numeric parse). Same free-variable scope rule as the spectrum path: the `(c)` driver
    // must be constant, and a stop whose expression carries per-hit variables is not a
    // constant here, so it is rejected rather than silently evaluated at a zero context.
    bool recordConstScalarRef(const std::string& tok, double& out) {
        size_t dot = tok.find('.');
        if (dot == std::string::npos) return false;
        std::string head = tok.substr(0, dot);
        auto rit = recordIndex_.find(head);
        if (rit == recordIndex_.end()) return false;                 // not a record -> not our form
        if (!records_ || rit->second >= (int)records_->size()) return false;
        const Record& rec = (*records_)[rit->second];
        std::string rest = tok.substr(dot + 1);                      // channel[i] | channel(c) | channel
        out = 0.0;
        auto scalarChannel = [&](const std::string& chan, int& ci) -> bool {
            ci = rec.channelIndex(chan);
            if (ci < 0) { fail("record ref '" + tok + "': record '" + head + "' has no channel '" + chan + "'"); return false; }
            if (rec.channels[ci].kind != ChanKind::Scalar) {
                fail("record ref '" + tok + "': channel '" + chan + "' is a colour channel, not scalar"); return false;
            }
            return true;
        };
        auto stopIsConst = [&](const RecStop& st) { return !patternHasFreeVars(st.expr); };
        // Stop selector: `channel[i]`.
        size_t lb = rest.find('[');
        if (lb != std::string::npos) {
            size_t rb = rest.rfind(']');
            std::string idxs = (rb != std::string::npos && rb > lb) ? rest.substr(lb + 1, rb - lb - 1) : "";
            if (idxs.empty() || !isNumber(idxs)) { fail("record ref '" + tok + "': bad stop selector"); return true; }
            int ci; if (!scalarChannel(rest.substr(0, lb), ci)) return true;
            const RecChannel& ch = rec.channels[ci];
            int i = (int)num(idxs);
            if (i < 0 || i >= (int)ch.stops.size()) {
                fail("record ref '" + tok + "': stop index " + idxs + " out of range (0.." +
                     std::to_string((int)ch.stops.size() - 1) + ")"); return true;
            }
            if (!stopIsConst(ch.stops[i])) {
                fail("record ref '" + tok + "': stop " + idxs + " is an expression with per-hit "
                     "variables — not a constant at this value site"); return true;
            }
            PatCtx zero{};
            const std::vector<PatNode>& e = ch.stops[i].expr;
            out = e.empty() ? 0.0 : patternEval(e.data(), (int)e.size(), zero);
            return true;
        }
        // Sample form: `channel(c)` with a constant driver c.
        size_t lp = rest.find('(');
        if (lp != std::string::npos) {
            size_t rp = rest.rfind(')');
            if (rp == std::string::npos || rp <= lp) { fail("record ref '" + tok + "': malformed `channel(constant)`"); return true; }
            std::string cexpr = rest.substr(lp + 1, rp - lp - 1);
            int ci; if (!scalarChannel(rest.substr(0, lp), ci)) return true;
            const RecChannel& ch = rec.channels[ci];
            std::vector<PatNode> drv; std::string cerr;
            if (!compilePatternExpr(cexpr, drv, cerr)) { fail("record ref '" + tok + "' driver '" + cexpr + "': " + cerr); return true; }
            if (patternHasFreeVars(drv)) {
                fail("record ref '" + tok + "': driver must be a constant here — no per-hit variables "
                     "(x/y/z/u/v/…) are in scope at this value site"); return true;
            }
            for (const RecStop& st : ch.stops) {
                if (!stopIsConst(st)) {
                    fail("record ref '" + tok + "': channel '" + ch.name + "' has expression stops with "
                         "per-hit variables — not a constant curve at this value site"); return true;
                }
            }
            PatCtx zero{};
            double c = drv.empty() ? 0.0 : patternEval(drv.data(), (int)drv.size(), zero);
            out = recSampleScalar(rec, ch, c, zero);
            return true;
        }
        // Bare `RECORD.channel` with no selector/sample: ambiguous at a constant site.
        fail("record ref '" + tok + "': use `RECORD.channel[i]` (a stop) or `RECORD.channel(constant)` at a value site");
        return true;
    }

    // ---- spectrum evaluation ----
    Spectrum evalSpectrum(const Value& v, int depth = 0) {
        if (depth > 16) { fail("spectrum reference cycle"); return constantSpectrum(0); }
        // table { λ:v … } — an inline piecewise curve. An optional `interp=cubic`
        // (monotone PCHIP, no overshoot) / `interp=linear` (default) flag among the
        // entries picks the interpolant; the rest are `λ:value` control points.
        if (v.block && v.block->type == "table") {
            std::vector<std::pair<double, double>> pairs;
            bool cubic = false;
            markAllUsed(*v.block);   // a flat `λ:value` list, not key/value statements
            for (const auto& w : v.block->words) {
                if (w.rfind("interp=", 0) == 0) { cubic = interpIsCubic(w); continue; }
                auto p = w.find(':');
                if (p == std::string::npos) { fail("table entry '" + w + "' not λ:value"); continue; }
                pairs.push_back({num(w.substr(0, p)), num(w.substr(p + 1))});
            }
            return cubic ? tabulatedSpectrumMono(std::move(pairs))
                         : tabulatedSpectrum(std::move(pairs));
        }
        const auto& w = v.words;
        if (w.empty()) { fail("empty spectrum expression"); return constantSpectrum(0); }
        const std::string& h = w[0];
        // Optional interpolation modifier for file/table curves, e.g.
        // `absorb file:data/red.csv interp=cubic`. Default (absent) = linear.
        bool curveCubic = false;
        for (const auto& t : w) if (t.rfind("interp=", 0) == 0) curveCubic = interpIsCubic(t);

        if (isNumber(h) && w.size() == 1) return constantSpectrum(num(h));

        // Records stage 5a: a record colour channel used as a constant value —
        // `RECORD.channel[i]` / `RECORD.channel(const)`. Fires only when h's head names
        // a known record; otherwise falls through to the ordinary spectrum forms below.
        if (w.size() == 1) {
            // A table sample at a PATTERN-LESS spectral site (`ior`, `absorb`, a light's
            // `spd`, a top-level `spectrum`). The pattern-aware slots intercept it in
            // patternedSpectrumParam and never arrive here, so reaching this point means the
            // slot holds one spectrum evaluated at load time and has nowhere to put a per-hit
            // sample — the same refusal a pattern-carrying material property gets below.
            if (isTableCallHead(h)) {
                fail("`" + h + "`: a table is sampled per hit, but this slot holds one "
                     "spectrum fixed at load time — the per-hit spectral slots are "
                     "`reflect`, `transmit` and `emit` (and their `_map` companions)");
                return constantSpectrum(0);
            }
            Spectrum rs;
            if (recordConstSpectrumRef(h, rs)) return rs;
            // §3.2 per-property access — `MATERIAL.slot` / `MATERIAL.slot(args)`. This is
            // the PATTERN-LESS spectral site (`ior`, `absorb`, a light's `spd`, a
            // top-level `spectrum`): it can hold a spectrum but has nowhere to put a
            // per-hit multiplier, so a pattern-carrying source is refused rather than
            // silently flattened. The pattern-aware slots (reflect/transmit/emit) are
            // intercepted earlier, in patternedSpectrumParam, and never reach here.
            MatProp mp;
            if (materialPropRef(h, mp)) {
                if (!mp.ok) return constantSpectrum(0);            // fail() already set
                if (!mp.spectral) {
                    fail("material property '" + h + "': '" + mp.slot + "' is a scalar "
                         "property, but a spectrum is needed here");
                    return constantSpectrum(0);
                }
                if (mp.pat >= 0)
                    fail("material property '" + h + "': '" + mp.slot + "' carries a per-hit "
                         "pattern, which this slot cannot apply — reference a material whose "
                         + mp.slot + " is a plain spectrum");
                return mp.spec;
            }
        }

        if (h == "blackbody")  return blackbody(w.size() > 1 ? num(w[1]) : 6500.0);
        if (h == "ior")        return iorConstant(w.size() > 1 ? num(w[1]) : 1.5);
        if (h == "whitewall")  return whiteWall(w.size() > 1 ? num(w[1]) : 0.75);
        if (h == "redwall")    return redWall();
        if (h == "greenwall")  return greenWall();
        // `gaussian center=560 sigma=25 amp=1` / `shortpass edge=480 slope=0.2 amp=1`,
        // or POSITIONALLY: `gaussian 560 25 1` / `shortpass 480 0.2 1`. Both forms are
        // documented, and the positional one used to be silently DROPPED (the loop
        // `continue`d on anything without an `=`), so `shortpass 470 0.2 1.0` became
        // shortPass(0, 0, 1.0) — a flat 0.5 absorption — and `gaussian 600 30 1.0`
        // became gaussianBand(0, 0, 1.0), which is identically ZERO (sigma = 0 makes
        // exp(-inf)), silently killing the dye in scenes/layered.ftsl. Positional and
        // keyed args may be mixed (a key overrides the slot it names); an unrecognised
        // key is now a hard error rather than a no-op, so this cannot recur.
        if (h == "gaussian" || h == "shortpass") {
            const bool isG = (h == "gaussian");
            double a = 0, b = 0, c = 1.0;   // gaussian: center,sigma,amp ; shortpass: edge,slope,amp
            int pos = 0;
            for (size_t k = 1; k < w.size(); ++k) {
                std::string key, val;
                if (splitEq(w[k], key, val)) {
                    double x = num(val);
                    if      (key == "center" || key == "edge")  a = x;
                    else if (key == "sigma"  || key == "slope") b = x;
                    else if (key == "amp")                      c = x;
                    else {
                        fail(h + ": unknown parameter '" + key + "' (expected " +
                             (isG ? "center/sigma/amp" : "edge/slope/amp") + ")");
                        return constantSpectrum(0);
                    }
                } else {                                  // positional
                    double x = num(w[k]);
                    if      (pos == 0) a = x;
                    else if (pos == 1) b = x;
                    else if (pos == 2) c = x;
                    else {
                        fail(h + ": too many arguments ('" + w[k] + "'); expected at most " +
                             (isG ? "center sigma amp" : "edge slope amp"));
                        return constantSpectrum(0);
                    }
                    ++pos;
                }
            }
            // sigma/slope 0 is not a usable band (gaussian collapses to identically
            // zero, shortpass to a flat amp/2), and is far likelier to be a typo than
            // an intent. Catch it here rather than letting it render as black.
            if (!(b > 0.0)) {
                fail(h + ": " + (isG ? "sigma" : "slope") + " must be > 0 (got " +
                     std::to_string(b) + ")");
                return constantSpectrum(0);
            }
            return isG ? gaussianBand(a, b, c) : shortPass(a, b, c);
        }
        // `rgb r g b` / `hsv h s v` / `hsl h s l` — a colour, upsampled to a smooth
        // reflectance via the Jakob-Hanika fit. hue in [0,1] (turns, wraps); s/v/l in
        // [0,1] (l = lightness). The `…line` heads (`rgbline`/`hsvline`/`hslline`)
        // instead take the K3 dominant-wavelength *emission* form: `rgbline r g b [sigma]`
        // emits a narrow band at the colour's dominant wavelength (near-monochromatic, so
        // glass disperses it), width from saturation or the explicit `sigma` (nm). The
        // `…illum` heads (`rgbillum`/`hsvillum`/`hslillum`) take the K1 Jakob-Hanika
        // *illuminant* upsample: a smooth, full-spectrum *emission* SPD (A·sigmoid) whose
        // integral under the bare CIE observer reproduces the colour — the emitter analogue
        // of `rgb`, right for coloured lights (`spd rgbillum 1 0.6 0.2`). Meant for lights;
        // accepted anywhere a spectrum is. (Head keywords, not trailing modifiers, because
        // the parser stops a value at the next bareword.)
        // The `…smits` heads (`rgbsmits`/…) take the classic Smits 1999 basis, the
        // `…box` heads (`rgbbox`/…) a plain calibrated 3-box, and the `…meng` heads
        // (`rgbmeng`/…) the Meng 2015 smoothest-spectrum grid, instead of the default
        // Jakob-Hanika fit — selectable alternative upsamplers (K1). `…meng` is the
        // one to reach for when a reflectance will be re-illuminated by a strongly
        // non-D65 light or dispersed, since it is the *smoothest* spectrum of that
        // colour rather than a shape-constrained fit.
        {
            bool isLine  = (h == "rgbline"  || h == "hsvline"  || h == "hslline");
            bool isIllum = (h == "rgbillum" || h == "hsvillum" || h == "hslillum");
            bool isSmits = (h == "rgbsmits" || h == "hsvsmits" || h == "hslsmits");
            bool isBox   = (h == "rgbbox"   || h == "hsvbox"   || h == "hslbox");
            bool isMeng  = (h == "rgbmeng"  || h == "hsvmeng"  || h == "hslmeng");
            bool isUser  = isCustomColourHead(h);   // `rgb:<name>` — user `upsample` block
            if (isColourHead(h)) {
                if (w.size() < 4) { fail(h + " needs 3 components"); return constantSpectrum(0); }
                std::string space = (isLine || isIllum || isSmits || isBox || isMeng || isUser)
                                        ? h.substr(0, 3) : h;
                Vec3 c;
                if      (space == "rgb") c = {num(w[1]), num(w[2]), num(w[3])};
                else if (space == "hsv") c = hsvToRgb(num(w[1]), num(w[2]), num(w[3]));
                else                     c = hslToRgb(num(w[1]), num(w[2]), num(w[3]));
                if (isLine) {
                    double sigma = (w.size() > 4 && isNumber(w[4])) ? num(w[4]) : -1.0;
                    return rgbToLineEmission(c.x, c.y, c.z, sigma);
                }
                // A user upsampler runs AFTER the space conversion, so it always sees
                // linear sRGB in (r, g, b) regardless of which of the three heads was
                // written — same contract as every built-in, so `hsv:mine 0.3 1 1` and
                // the equivalent `rgb:mine …` cannot disagree.
                if (isUser) return applyUpsample(h.substr(4), c);
                if (isIllum) return rgbToIlluminantJH(c.x, c.y, c.z);
                if (isSmits) return rgbToReflectanceSmits(c.x, c.y, c.z);
                if (isBox)   return rgbToReflectanceBox(c.x, c.y, c.z);
                if (isMeng)  return rgbToReflectanceMeng(c.x, c.y, c.z);
                return rgbToReflectanceJH(c.x, c.y, c.z);
            }
        }
        if (h.rfind("glass:", 0) == 0) {
            std::string g = h.substr(6);
            Spectrum ior;
            if (resolveGlassIor(g, ior)) return ior;
            fail("unknown glass '" + g + "'"); return glassOrDefault("BK7", 1.5168);
        }
        if (h.rfind("metal:", 0) == 0) {
            std::string mname = h.substr(6);
            Spectrum r;
            if (resolveMetalReflectance(mname, r)) return r;
            fail("unknown metal '" + mname + "'"); return constantSpectrum(0.9);
        }
        if (h.rfind("reflectance:", 0) == 0) {
            std::string rname = h.substr(12);
            Spectrum r;
            if (resolveNaturalReflectance(rname, r)) return r;
            fail("unknown reflectance '" + rname + "'"); return constantSpectrum(0.5);
        }
        if (h.rfind("filter:", 0) == 0) {
            std::string fname = h.substr(7);
            Spectrum t;
            if (resolveFilterTransmittance(fname, t)) return t;
            fail("unknown filter '" + fname + "'"); return constantSpectrum(0.5);
        }
        if (h.rfind("preset:", 0) == 0)  return resolvePreset(h.substr(7));
        if (h.rfind("file:", 0) == 0)    return loadSpdFile(h.substr(5), curveCubic);
        if (h.rfind("spectrum:", 0) == 0) {
            std::string nm = h.substr(9);
            auto it = spectraBlocks_.find(nm);
            if (it == spectraBlocks_.end()) { fail("unknown spectrum '" + nm + "'"); return constantSpectrum(0); }
            const Stmt* e = find(*it->second, "=");
            if (!e) { fail("spectrum '" + nm + "' has no value"); return constantSpectrum(0); }
            return evalSpectrum(e->val, depth + 1);
        }
        // Common authoring trap: a bare numeric run like `absorb 3 0.5 0.3` looks like
        // a per-channel colour but isn't a spectrum expression (only a lone scalar, a
        // tagged colour, or a named/ref spectrum parse). Point straight at the fix.
        if (isNumber(h) && w.size() >= 3) {
            bool allNum = true;
            for (const auto& t : w) if (!isNumber(t)) { allNum = false; break; }
            if (allNum) {
                fail("unrecognized spectrum expression '" + h + "' — a bare numeric triple "
                     "isn't a spectrum; tag it, e.g. `rgb " + w[0] + " " + w[1] + " " + w[2] + "`");
                return constantSpectrum(0);
            }
        }
        // Reaching here with a `texture:` prefix means the slot has no texture path.
        // Only the two Lambertian families bind a reflect texture (bindReflectTexture);
        // the specular ones read the reflect slot directly and have nowhere to sample a
        // UV from (see reflectSlot in scene.h). Say THAT, rather than making the author
        // wonder why a texture name isn't a valid spectrum.
        if (h.rfind("texture:", 0) == 0) {
            fail("'" + h + "': this slot takes a spectrum, not a texture — only the "
                 "`reflect` slot of a `diffuse`/`translucent` material binds an image "
                 "albedo. Use `pattern:<name>` for a procedural drive, or a uniform "
                 "spectrum here");
            return constantSpectrum(0);
        }
        fail("unrecognized spectrum expression '" + h + "'");
        return constantSpectrum(0);
    }

    // `interp=<mode>` -> is it the monotone-cubic (PCHIP) mode? Accepts cubic / pchip /
    // monotone as synonyms; anything else (incl. `linear`) is the piecewise-linear default.
    static bool interpIsCubic(const std::string& tok) {
        std::string v = (tok.rfind("interp=", 0) == 0) ? tok.substr(7) : tok;
        return v == "cubic" || v == "pchip" || v == "monotone";
    }

    // Load a measured SPD/reflectance from an external data file: `spd file:<path>`
    // (or `reflect file:<path>`). Reads the CSV/whitespace table mirrored under data/
    // into a tabulated `Spectrum` — piecewise-linear by default, or monotone-cubic
    // (no overshoot) when `cubic` is set (`… interp=cubic`). Paths resolve relative to
    // the current working directory (same convention as `texture`/`mesh` file refs),
    // and repeated references to the same (path, interp) share one cached curve.
    Spectrum loadSpdFile(const std::string& path, bool cubic = false) {
        std::string key = cubic ? path + "\x01cubic" : path;
        auto it = spdFileCache_.find(key);
        if (it != spdFileCache_.end()) return it->second;
        std::vector<std::pair<double, double>> pairs;
        std::string ferr;
        if (!speclib::loadSpdCsv(path, pairs, ferr)) { fail(ferr); return constantSpectrum(0); }
        // Coverage check: warn (once per key) if the file fails to cover the
        // perceptually significant band (~400..700 nm, where >99.9% of the CIE
        // observer's response lives), since sampling outside the file just holds the
        // nearest endpoint flat (deliberate — no extrapolation). Guarding the visible
        // core rather than the full 360..830 render range keeps this high-signal:
        // standard 380..780 / 400..700 datasets stay quiet, but a genuinely narrow
        // file (whose clamped tails would visibly distort colour) is flagged.
        double lo = pairs.front().first, hi = pairs.front().first;
        for (const auto& p : pairs) { lo = std::min(lo, p.first); hi = std::max(hi, p.first); }
        const double visLo = 400.0, visHi = 700.0, eps = 0.5;
        if (lo > visLo + eps || hi < visHi - eps)
            std::fprintf(stderr, "[ftsl] warning: spectrum file '%s' covers only %.0f..%.0f nm — "
                         "it misses part of the visible band (~%.0f..%.0f nm); values outside the "
                         "file are held flat at the nearest endpoint (no extrapolation).\n",
                         path.c_str(), lo, hi, visLo, visHi);
        Spectrum s = cubic ? tabulatedSpectrumMono(std::move(pairs))
                           : tabulatedSpectrum(std::move(pairs));
        spdFileCache_[key] = s;
        return s;
    }

    // Illuminant presets. Delegates to the shared resolver in lights.h (the same one
    // the `-light` CLI flag uses) so the two never drift apart.
    Spectrum resolvePreset(const std::string& nm) {
        Spectrum s;
        if (resolveLightPreset(nm, s)) return s;
        fail("unknown preset '" + nm + "'"); return blackbody(6500.0);
    }

    // Fetch a spectral-typed material parameter (inline expr or spectrum:ref).
    Spectrum spectrumParam(const Block& b, const char* key, Spectrum dflt) {
        const Stmt* s = find(b, key);
        if (!s) return dflt;
        return evalSpectrum(s->val);
    }

    // Records stage 5a: scalar-slot reader that also accepts a constant record channel
    // ref (`RECORD.channel[i]` / `RECORD.channel(const)`) — the scalar analogue of
    // spectrumParam. Absent key -> default; a plain number -> that number; a token whose
    // head names a record -> recordConstScalarRef (which fills the value or sets fail).
    // This is the chokepoint that gives material scalar slots (roughness, film_ior, …)
    // record-ref support without each call site knowing about records.
    double dblParam(const Block& b, const char* key, double dflt) {
        const Stmt* s = find(b, key);
        if (!s || s->val.words.empty()) return dflt;
        const std::string& w0 = s->val.words[0];
        // Reaching dblParam with a table sample in hand means THIS slot has nowhere to put a
        // per-hit value: the slots that can hold one try bindScalarPattern first and take it
        // there. Refused rather than silently read as `num("grid:…") == 0`.
        if (isTableCallHead(w0)) {
            fail(std::string("`") + key + " " + w0 + "`: a table is sampled per hit, but '" +
                 key + "' takes one number fixed at load time — the per-hit slots are "
                 "`roughness`, `film_thickness_map`, `weight_map`, and the `_map` companion "
                 "of a spectral slot");
            return dflt;
        }
        if (w0.find('.') != std::string::npos && !isNumber(w0)) {
            double rv;
            if (recordConstScalarRef(w0, rv)) return rv;             // record ref (or a fail was set)
            // §3.2 per-property access at a scalar slot: `roughness gold.roughness`,
            // `film_ior coat.film_ior`. A pattern-carrying source is refused here rather
            // than dropped — the slots that CAN hold a per-hit pattern try
            // bindScalarPattern first (which takes the pattern), so reaching dblParam
            // with one in hand means this particular slot has nowhere to put it.
            MatProp mp;
            if (materialPropRef(w0, mp)) {
                if (!mp.ok) return dflt;                             // fail() already set
                if (mp.spectral) {
                    fail(std::string(key) + " " + w0 + ": '" + mp.slot + "' is a spectral "
                         "property, but this slot takes a scalar");
                    return dflt;
                }
                if (mp.pat >= 0)
                    fail(std::string(key) + " " + w0 + ": '" + mp.slot + "' carries a per-hit "
                         "pattern, which the '" + key + "' slot cannot apply — write it on the "
                         "matching '_map' slot instead");
                return mp.scalar;
            }
        }
        return num(w0);
    }

    // ---- textures ----
    // A `texture "name" { file "path" [encoding srgb|linear] [filter nearest|
    // bilinear] [wrap repeat|clamp|mirror] }` block loads an image into
    // Scene::textures and records its name -> index. Reflectance coefficients are
    // precomputed here so per-hit sampling is a cheap bilerp+sigmoid.
    bool addTexture(const Block& b, Loaded& L) {
        if (b.name.empty()) { fail("texture needs a \"name\""); return false; }
        if (textureIndex_.count(b.name)) { fail("duplicate texture name '" + b.name + "'"); return false; }
        Texture tex;
        tex.name = b.name;
        // Filter + wrap are common to both a file image and a procedural skin.
        std::string flt = strOf(b, "filter", "bilinear");
        if      (flt == "bilinear") tex.filter = TexFilter::Bilinear;
        else if (flt == "nearest")  tex.filter = TexFilter::Nearest;
        else { fail("texture '" + b.name + "': unknown filter '" + flt + "' (nearest|bilinear)"); return false; }
        std::string wr = strOf(b, "wrap", "repeat");
        if      (wr == "repeat") tex.wrap = TexWrap::Repeat;
        else if (wr == "clamp")  tex.wrap = TexWrap::Clamp;
        else if (wr == "mirror") tex.wrap = TexWrap::Mirror;
        else { fail("texture '" + b.name + "': unknown wrap '" + wr + "' (repeat|clamp|mirror)"); return false; }

        const Stmt* rgbS = find(b, "rgb");
        if (rgbS) {
            // Procedural (function-defined) UV-space skin (E1): three ftsl expressions
            // r(u,v) g(u,v) b(u,v) over the surface UV, baked once to a `res`x`res`
            // LINEAR RGB grid at load, then treated as an ordinary texture — so the
            // whole existing UV-wrap / Jakob-Hanika / triplanar / GPU / raster pipeline
            // (and the `reflect texture:<name>` binding) applies unchanged with no
            // per-hit fit cost. The expressions are functions of u,v (and constants);
            // the world-space pattern variables x y z f nx ny nz r are 0 here since a
            // UV image carries no world position.
            if (rgbS->val.words.size() < 3) {
                fail("texture '" + b.name + "': rgb needs three quoted exprs: rgb \"r(u,v)\" \"g(u,v)\" \"b(u,v)\""); return false;
            }
            std::vector<PatNode> pr, pg, pb; std::string perr;
            if (!compilePatternExpr(rgbS->val.words[0], pr, perr, false, &texScope_, &tableScope_)) { fail("texture '" + b.name + "' rgb r: " + perr); return false; }
            if (!compilePatternExpr(rgbS->val.words[1], pg, perr, false, &texScope_, &tableScope_)) { fail("texture '" + b.name + "' rgb g: " + perr); return false; }
            if (!compilePatternExpr(rgbS->val.words[2], pb, perr, false, &texScope_, &tableScope_)) { fail("texture '" + b.name + "' rgb b: " + perr); return false; }
            int res = (int)dblOf(b, "res", 512.0);
            if (res < 1) res = 1; else if (res > 8192) res = 8192;
            tex.encoding = TexEncoding::Linear;   // expr outputs are linear albedo already
            tex.w = res; tex.h = res;
            tex.rgb.assign((size_t)res * res, Vec3{0, 0, 0});
            auto cl = [](double t) { return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t); };
            // Bind the scene's pattern tables once; only (u,v) vary per texel.
            PatCtx c; bindPatScene(c, L.scene);
            for (int y = 0; y < res; ++y) {
                // Invert v to match sampleRgb's (1-v) flip so f(u,v) reads back at the
                // surface UV (top-left storage, v=0 at image bottom / OBJ convention).
                double v = 1.0 - (y + 0.5) / res;
                for (int x = 0; x < res; ++x) {
                    double u = (x + 0.5) / res;
                    c.u = u; c.v = v;
                    double rr = patternEval(pr.data(), (int)pr.size(), c);
                    double gg = patternEval(pg.data(), (int)pg.size(), c);
                    double bb = patternEval(pb.data(), (int)pb.size(), c);
                    tex.rgb[(size_t)y * res + x] = Vec3{cl(rr), cl(gg), cl(bb)};
                }
            }
        } else {
            std::string file = strOf(b, "file");
            if (file.empty()) { fail("texture '" + b.name + "' needs a file (or an `rgb \"r\" \"g\" \"b\"` expr triple)"); return false; }
            std::string enc = strOf(b, "encoding", "srgb");
            if      (enc == "srgb")   tex.encoding = TexEncoding::sRGB;
            else if (enc == "linear") tex.encoding = TexEncoding::Linear;
            else { fail("texture '" + b.name + "': unknown encoding '" + enc + "' (srgb|linear)"); return false; }
            std::string terr;
            if (!tex.load(file, terr)) { fail("texture '" + b.name + "': " + terr); return false; }
            // Optional indexed-spectral palette (§9.3): `palette { 0 spectrum:navy 1 ... }`.
            // The nested block's flat word dump is (index, spectrum-ref) pairs in order; we
            // resolve each ref to a Spectrum now and size the palette to max-index+1. The
            // texture's red channel then selects an entry per texel (nearest, no upsample).
            if (const Stmt* ps = find(b, "palette")) {
                if (!ps->val.block) { fail("texture '" + b.name + "': palette needs a { } body"); return false; }
                markAllUsed(*ps->val.block);   // a flat (index, spectrum-ref) list
                const auto& w = ps->val.block->words;
                if (w.empty() || (w.size() % 2) != 0) {
                    fail("texture '" + b.name + "': palette needs (index spectrum) pairs"); return false;
                }
                std::vector<std::pair<int, Spectrum>> entries;
                int maxIdx = -1;
                for (size_t k = 0; k + 1 < w.size(); k += 2) {
                    int idx = std::atoi(w[k].c_str());
                    if (idx < 0 || idx > 255) { fail("texture '" + b.name + "': palette index out of 0..255"); return false; }
                    Value ref; ref.words.push_back(w[k + 1]);
                    entries.emplace_back(idx, evalSpectrum(ref));
                    if (idx > maxIdx) maxIdx = idx;
                }
                tex.palette.assign((size_t)maxIdx + 1, constantSpectrum(0.0));
                for (auto& e : entries) tex.palette[(size_t)e.first] = e.second;
            }
        }
        tex.buildReflCoeff();   // precompute Jakob-Hanika reflectance coefficients (skipped for palette maps)
        int id = (int)L.scene.textures.size();
        L.scene.textures.push_back(std::move(tex));
        textureIndex_[b.name] = id;
        return true;
    }

    // If the material's `reflect` statement is `texture:<name>`, bind the texture to
    // the material (spatially-varying diffuse albedo) and return true. Otherwise the
    // caller falls back to spectrumParam for a uniform reflectance.
    bool bindReflectTexture(const Block& b, Material& m) {
        const Stmt* s = find(b, "reflect");
        if (!s || s->val.words.empty()) return false;
        const std::string& w0 = s->val.words[0];
        if (w0.rfind("texture:", 0) != 0) return false;
        std::string nm = w0.substr(8);
        auto it = textureIndex_.find(nm);
        if (it == textureIndex_.end()) { fail("reflect references unknown texture '" + nm + "'"); return false; }
        m.reflectTex = it->second;
        return true;
    }

    // The `reflect` slot, pattern-aware — the spectrum-slot half of §4's procedural
    // drives. Two spellings, one runtime field (`Material::reflectPat`, a per-hit scalar
    // multiplier on whatever the slot otherwise evaluates to):
    //
    //   reflect pattern:p                       greyscale albedo p(hit)  — the pattern is
    //   reflect [0 1](u)                        ALONE in the slot, so the base spectrum
    //                                           becomes a flat 1.0 and the multiply IS
    //                                           the answer. (An inline array literal
    //                                           desugars to the `pattern:` form, which is
    //                                           what makes `reflect [0 1](u)` work.)
    //   reflect rgb .8 .2 .2                    that tint, modulated per hit — colour from
    //   reflect_map pattern:p                   the spectrum, variation from the pattern.
    //   reflect texture:t / reflect_map p       likewise for an image albedo.
    //
    // Returning the base spectrum (rather than assigning) keeps every material type's own
    // default intact; call it wherever `spectrumParam(b, "reflect", …)` used to be called.
    // `m.type` must already be set — the honoured-family check below reads it.
    // The shared mechanism, keyed on the slot name: `<key> pattern:p` puts the pattern
    // ALONE in the slot (base becomes a flat 1.0), `<mapKey> pattern:p` modulates whatever
    // `<key>` otherwise says. Identical for `reflect`/`reflect_map` and `transmit`/
    // `transmit_map`, hence one implementation.
    Spectrum patternedSpectrumParam(const Block& b, const char* key, const char* mapKey,
                                    int& patOut, const Spectrum& dflt) {
        bindScalarPattern(b, mapKey, patOut);
        const Stmt* s = find(b, key);
        // `reflect grid:ramp(u)` — same slot semantics as `reflect pattern:p`: the sampled
        // table goes ALONE into the slot and the base spectrum becomes flat 1.0, so the
        // table's own values are the greyscale albedo. Handled here rather than in
        // evalSpectrum because this is the only spectral site with somewhere to put a
        // per-hit multiplier.
        if (s && !s->val.words.empty() && isTableCallHead(s->val.words[0])) {
            if (!bindScalarPattern(b, key, patOut)) return dflt;   // fail() already set
            return constantSpectrum(1.0);
        }
        if (s && !s->val.words.empty() && s->val.words[0].rfind("pattern:", 0) == 0) {
            if (!bindScalarPattern(b, key, patOut)) return dflt;   // unknown name: failed
            return constantSpectrum(1.0);
        }
        // §3.2 per-property access at a PATTERN-AWARE spectral slot: `reflect gold.reflect`
        // / `reflect gold.reflect(u=v)`. This is the only site that can carry BOTH halves
        // of the source slot — the base spectrum and its per-hit multiplier — which is why
        // the hook lives here rather than in evalSpectrum. If the consumer also wrote its
        // own `<key>_map`, the two are COMPOSED (both spellings mean "a multiplier on
        // whatever the slot otherwise evaluates to", so their composition is the product)
        // instead of one silently clobbering the other.
        if (s && s->val.words.size() == 1 && loadedRef_) {
            MatProp mp;
            if (materialPropRef(s->val.words[0], mp)) {
                if (!mp.ok) return dflt;                           // fail() already set
                if (!mp.spectral) {
                    fail(std::string(key) + " " + s->val.words[0] + ": '" + mp.slot +
                         "' is a scalar property, but this slot takes a spectrum");
                    return dflt;
                }
                patOut = composePatterns(patOut, mp.pat, *loadedRef_);
                return mp.spec;
            }
        }
        return spectrumParam(b, key, dflt);
    }

    Spectrum reflectParam(const Block& b, Material& m, const Spectrum& dflt) {
        return patternedSpectrumParam(b, "reflect", "reflect_map", m.reflectPat, dflt);
    }

    // Same for the transmit slot (read through transmitSlot() by Filter's gel
    // transmittance and DiffuseTransmit's back-lobe albedo).
    Spectrum transmitParam(const Block& b, Material& m, const Spectrum& dflt) {
        return patternedSpectrumParam(b, "transmit", "transmit_map", m.transmitPat, dflt);
    }

    // Which material families route their reflect slot through diffuseReflectance() /
    // reflectSlot(), the only two accessors that apply reflectPat. Anything else reads
    // `m.reflect` directly (Fluorescent's fluoroWeights, for instance, which has no hit
    // to evaluate a pattern at), so a pattern there would be silently dropped — and the
    // flat-1.0 base that a lone `reflect pattern:` leaves behind would then render as
    // albedo 1.0, a WRONG image rather than a missing effect. Hence the loader refuses it.
    static bool reflectPatHonoured(MatType t) {
        return t == MatType::Diffuse   || t == MatType::DiffuseTransmit ||
               t == MatType::Mirror    || t == MatType::HalfMirror      ||
               t == MatType::Glossy    || t == MatType::Grating;
    }

    // The transmit slot is only ever READ by these two: a colored gel's T(lambda) and a
    // translucent's back-hemisphere albedo. Every other type leaves `transmit` at its 0
    // default and never looks at it, so a pattern there is meaningless.
    static bool transmitPatHonoured(MatType t) {
        return t == MatType::Filter || t == MatType::DiffuseTransmit;
    }

    // Refuse a slot pattern on a family that would not apply it.
    void checkSlotPatSupported(const Block& b, const char* key, const char* mapKey,
                               bool honoured, int& patOut, const char* families) {
        if (honoured) return;
        const Stmt* s = find(b, key);
        const bool patInSlot = s && !s->val.words.empty() &&
                               s->val.words[0].rfind("pattern:", 0) == 0;
        if (!patInSlot && !find(b, mapKey)) return;
        fail(std::string("a ") + key + " pattern is not supported on this material type — "
             "only the families whose " + key + " slot goes through the shared per-hit "
             "accessor (" + families + ") apply one; elsewhere it would be silently ignored");
        patOut = -1;
    }

    // Run from the COMMON tail of buildMaterial (not from reflectParam/transmitParam) so
    // it also catches the types that never read the slot at all — otherwise `reflect_map`
    // on, say, a thinfilm would be dropped in silence, which reads as "it worked".
    void checkSlotPatsSupported(const Block& b, Material& m) {
        checkSlotPatSupported(b, "reflect", "reflect_map", reflectPatHonoured(m.type),
                              m.reflectPat,
                              "diffuse, translucent, mirror, halfmirror, glossy, grating");
        checkSlotPatSupported(b, "transmit", "transmit_map", transmitPatHonoured(m.type),
                              m.transmitPat, "translucent, filter");
    }

    // If `<key>`'s value is `texture:<name>`, bind that texture's grayscale value to a
    // NON-albedo scalar material parameter (spec §9.4) and return true; otherwise
    // false (the caller reads a numeric value instead). Used for roughness /
    // film-thickness maps. Sampled via Texture::scalarAt at the hit UV.
    bool bindScalarTexture(const Block& b, const char* key, int& texOut) {
        const Stmt* s = find(b, key);
        if (!s || s->val.words.empty()) return false;
        const std::string& w0 = s->val.words[0];
        if (w0.rfind("texture:", 0) != 0) return false;
        std::string nm = w0.substr(8);
        auto it = textureIndex_.find(nm);
        if (it == textureIndex_.end()) {
            fail(std::string(key) + " references unknown texture '" + nm + "'"); return false;
        }
        texOut = it->second;
        return true;
    }

    // If `<key>`'s value is `pattern:<name>`, bind a procedural pattern to a scalar
    // material parameter (§4) and return true; otherwise false (the caller reads a
    // numeric value or tries bindScalarTexture instead). Patterns are evaluated at
    // the hit's (x,y,z,f,normal,r) — the mechanism that gives UV-less implicit
    // surfaces spatially-varying roughness / thickness / material selection.
    bool bindScalarPattern(const Block& b, const char* key, int& patOut) {
        const Stmt* s = find(b, key);
        if (!s || s->val.words.empty()) return false;
        const std::string& w0 = s->val.words[0];
        // A table sampled at a value site IS a pattern, so it binds here exactly like
        // `pattern:<name>` does — which is what makes `roughness grid:bumps(u,v)` and the
        // inline `roughness [ … ](u,v)` the same statement written two ways.
        if (isTableCallHead(w0)) {
            int p = tableCallPattern(w0, std::string("`") + key + "`");
            if (p < 0) return false;                          // fail() already set
            patOut = p;
            return true;
        }
        // §3.2 per-property access carrying a pattern: `roughness gold.roughness` where
        // gold's roughness is itself pattern-driven. Reported as "handled" ONLY when the
        // source actually has a pattern, so a plain-constant source falls through to
        // dblParam and still reads its value — which is what makes the call sites'
        // existing `bindScalarPattern(...) else dblParam(...)` ladder do the right thing
        // for both shapes without any of them knowing this form exists.
        if (w0.rfind("pattern:", 0) != 0 && w0.find('.') != std::string::npos &&
            !isNumber(w0) && loadedRef_) {
            MatProp mp;
            if (materialPropRef(w0, mp)) {
                if (!mp.ok || mp.spectral || mp.pat < 0) return false;
                patOut = composePatterns(patOut, mp.pat, *loadedRef_);
                return true;
            }
            return false;
        }
        if (w0.rfind("pattern:", 0) != 0) return false;
        std::string nm = w0.substr(8);
        auto it = patternIndex_.find(nm);
        if (it == patternIndex_.end()) {
            fail(std::string(key) + " references unknown pattern '" + nm + "'"); return false;
        }
        patOut = it->second;
        return true;
    }

    // ---- patterns ----
    // A `pattern "name" { ... }` block compiles a procedural scalar field into
    // Scene::patterns. Two authoring forms:
    //   * an infix formula:  `expr "0.5 + 0.5*sin(20*x)"`  (MUST be quoted; compiled
    //     by the shunting-yard evaluator over the variables x y z f nx ny nz r), or
    //   * a named generator via `type <gen>` + params (mirrors material syntax):
    //       type axis    axis <x|y|z>  [scale <s>] [offset <o>]
    //       type radial  [center <x y z>] [scale <s>]
    //       type bands   axis <x|y|z>  [freq <f>] [phase <p>]
    //       type checker [size <s>]
    //       type noise   [freq <f>]
    //       type field   [scale <s>]
    bool addPattern(const Block& b, Loaded& L) {
        if (b.name.empty()) { fail("pattern needs a \"name\""); return false; }
        if (patternIndex_.count(b.name)) { fail("duplicate pattern name '" + b.name + "'"); return false; }
        Pattern pat;
        auto axisOp = [&](const std::string& a, PatOp& out) -> bool {
            if (a == "x") { out = PatOp::VarX; return true; }
            if (a == "y") { out = PatOp::VarY; return true; }
            if (a == "z") { out = PatOp::VarZ; return true; }
            fail("pattern '" + b.name + "': axis must be x|y|z"); return false;
        };
        if (const Stmt* es = find(b, "expr")) {
            // The quoted expression arrives as a single token (spaces preserved).
            std::string expr;
            for (size_t k = 0; k < es->val.words.size(); ++k) { if (k) expr += " "; expr += es->val.words[k]; }
            std::string perr;
            // `a` is legal here: a named pattern stays unresolved until a material uses
            // it, and applyMaterial then resolves `a` against THAT material.
            if (!compilePatternExpr(expr, pat.nodes, perr, false, &texScope_,
                                    &tableScope_, /*allowA=*/true)) {
                fail(genWho("pattern", b.name) + ": " + perr); return false;
            }
        } else {
            std::string g = strOf(b, "type", "");
            if (g == "axis") {
                PatOp coord; if (!axisOp(strOf(b, "axis", "x"), coord)) return false;
                pat.nodes = pattern_gen::axis(coord, dblOf(b, "scale", 1.0), dblOf(b, "offset", 0.0));
            } else if (g == "radial") {
                Vec3 c{0, 0, 0}; vec3Of(b, "center", c);
                pat.nodes = pattern_gen::radial(c, dblOf(b, "scale", 1.0));
            } else if (g == "bands") {
                PatOp coord; if (!axisOp(strOf(b, "axis", "x"), coord)) return false;
                pat.nodes = pattern_gen::bands(coord, dblOf(b, "freq", 1.0), dblOf(b, "phase", 0.0));
            } else if (g == "checker") {
                pat.nodes = pattern_gen::checker(dblOf(b, "size", 1.0));
            } else if (g == "noise") {
                pat.nodes = pattern_gen::noise(dblOf(b, "freq", 1.0));
            } else if (g == "field") {
                pat.nodes = pattern_gen::field(dblOf(b, "scale", 1.0));
            } else {
                fail("pattern '" + b.name + "' needs `expr \"...\"` or `type <axis|radial|bands|checker|noise|field>`");
                return false;
            }
        }
        int id = (int)L.scene.patterns.size();
        L.scene.patterns.push_back(std::move(pat));
        patternIndex_[b.name] = id;
        return true;
    }

    // ---- inline array literals: `[0 1](u)` / `[[0 1 2][3 4 5]](u,v)` ----
    // The shorthand for "a tiny lookup table, written where it is used". It is pure
    // SUGAR: each literal becomes an anonymous `grid` + a one-line `pattern` that samples
    // it, and the value site is rewritten to `pattern:<gen>`. That is why the literal
    // works in every slot that already takes a pattern without any of those slots
    // knowing the syntax exists — and why the two front ends only ever have to carry the
    // literal's raw shape (ftsl::BrItem), never its meaning.
    //
    // The NESTING is the shape (axis 0 outermost, C order — the same layout `grid`'s
    // `data { … }` uses), so a shape need never be spelled; the array must be rectangular.
    // The domain is the UNIT box per axis (lo 0, hi 1), NOT the `grid` element's default
    // index lattice: an inline literal has no domain of its own and is overwhelmingly
    // read at normalized coordinates (`u`, `v`), whereas a standalone `grid` is a data
    // container whose sample spacing is the meaningful thing.

    // Walk the raw bracket tree in C order, proving it RECTANGULAR (every group at a
    // given depth the same length) while collecting the per-axis extents.
    bool flattenArray(const std::vector<BrItem>& items, int depth,
                      std::vector<int>& shape, std::vector<std::string>& out,
                      const std::string& who, const char* raggedHint = "") {
        if (items.empty()) { fail(who + ": empty `[ ]` group"); return false; }
        const bool grouped = items[0].isGroup;
        for (const auto& it : items) {
            if (it.isGroup != grouped) {
                fail(who + ": axis " + std::to_string(depth) +
                     " mixes numbers with nested `[ … ]` groups");
                return false;
            }
        }
        if ((int)shape.size() == depth) shape.push_back((int)items.size());
        else if (shape[depth] != (int)items.size()) {
            fail(who + ": ragged — axis " + std::to_string(depth) + " has both " +
                 std::to_string(shape[depth]) + " and " + std::to_string(items.size()) +
                 " entries; every group at the same nesting level must be the same length" +
                 raggedHint);
            return false;
        }
        if (!grouped) {
            for (const auto& it : items) {
                if (!isNumber(it.word)) {
                    fail(who + ": non-numeric entry '" + it.word + "' in an inline array");
                    return false;
                }
                out.push_back(it.word);
            }
            return true;
        }
        for (const auto& it : items)
            if (!flattenArray(it.items, depth + 1, shape, out, who)) return false;
        return true;
    }

    // Split a sample call's `( … )` text into its top-level, comma-separated arguments,
    // and report whether any of them carries a top-level `=` (the keyword `formal=driver`
    // form). Top-level means "not inside a nested paren/bracket group", so a composed
    // coordinate keeps its own commas to itself. As in `parseBindArgs`, a top-level `=`
    // is unambiguously a binding because the pattern language has no comparison operators.
    // `kwAt` is the index of the FIRST keyword argument, or -1.
    static void splitCallArgs(const std::string& call, std::vector<std::string>& out,
                              int& kwAt, std::string& kwFormal) {
        out.clear(); kwAt = -1; kwFormal.clear();
        if (call.size() <= 2) return;                       // `()` — no arguments at all
        const std::string in = call.substr(1, call.size() - 2);
        int depth = 0; size_t start = 0;
        std::vector<std::string> raw;
        for (size_t i = 0; i <= in.size(); ++i) {
            if (i == in.size() || (in[i] == ',' && depth == 0)) {
                raw.push_back(trimWs(in.substr(start, i - start)));
                start = i + 1;
                continue;
            }
            char c = in[i];
            if (c == '(' || c == '[') ++depth;
            else if (c == ')' || c == ']') --depth;
        }
        for (size_t k = 0; k < raw.size(); ++k) {
            const std::string& seg = raw[k];
            out.push_back(seg);
            if (kwAt >= 0) continue;
            int d = 0;
            for (size_t i = 0; i < seg.size(); ++i) {
                char c = seg[i];
                if (c == '(' || c == '[') ++d;
                else if (c == ')' || c == ']') --d;
                else if (c == '=' && d == 0) {
                    kwAt = (int)k;
                    kwFormal = trimWs(seg.substr(0, i));
                    break;
                }
            }
        }
    }

    // Parse an array literal's `[ … ]` TEXT back into the same BrItem tree the grammar
    // builds at a value site. Needed because a literal COMPOSED inside another one's
    // sample call — `[0 1]([0.2 0.8](u))` — reaches the loader as raw characters inside a
    // single PARENWORD token, not as a reduced bracket group: the lexer deliberately does
    // not treat `(` as a delimiter (that is exactly what keeps an expression like
    // `sin(2*pi*u)` one token), so everything between a call's parens is text by
    // construction. The splitting rule here is the tokenizer's own — whitespace separates
    // entries, `[`/`]` nest, and a paren group is held together so `[f(a b) 2]` keeps its
    // call whole — which is what makes a composed literal read identically to a written-out
    // one. On success `i` sits one past the closing `]`; on failure `err` says why.
    static bool parseArrayText(const std::string& t, size_t& i,
                               std::vector<BrItem>& out, std::string& err) {
        ++i;                                         // past the '['
        for (;;) {
            while (i < t.size() && (t[i] == ' ' || t[i] == '\t')) ++i;
            if (i >= t.size()) { err = "unbalanced `[`"; return false; }
            if (t[i] == ']') { ++i; return true; }
            if (t[i] == '[') {
                BrItem g; g.isGroup = true;
                if (!parseArrayText(t, i, g.items, err)) return false;
                out.push_back(std::move(g));
                continue;
            }
            const size_t s = i;
            int d = 0;
            while (i < t.size()) {
                const char c = t[i];
                if (c == '(') ++d;
                else if (c == ')') {
                    if (d == 0) { err = "unbalanced `)`"; return false; }
                    --d;
                } else if (d == 0 && (c == ' ' || c == '\t' || c == '[' || c == ']')) break;
                ++i;
            }
            BrItem w; w.word = t.substr(s, i - s);
            out.push_back(std::move(w));
        }
    }

    static void addGenStmt(Block& blk, const char* key, std::vector<std::string> words, int line) {
        Stmt t; t.key = key; t.line = line; t.val.words = std::move(words);
        blk.words.push_back(t.key);
        for (const auto& w : t.val.words) blk.words.push_back(w);
        blk.stmts.push_back(std::move(t));
    }

    // Flatten ONE literal into an anonymous `grid __arrN` block (appended to `gen`) and
    // check its sample call against the nesting. Shared by the two places a literal can
    // appear: at a VALUE SITE (desugarOne, which additionally wraps the grid in a
    // `pattern` so a statement has a name to reference) and COMPOSED inside another
    // literal's sample call (desugarNestedLiterals), which needs the grid ALONE — because
    // `grid:NAME(coords)` is already a legal pattern-expression term, so a composed
    // literal costs one block instead of two and needs no name at the ftsl level at all.
    // `call` is always the AUTHOR's text: arity and shape are checked, and errors phrased,
    // against what they wrote — never against the rewritten text the composer produces.
    bool buildArrayGrid(const std::vector<BrItem>& items, const std::string& call,
                        const std::string& who, int line,
                        std::vector<Block>& gen, int& n, std::string& outName) {
        const std::string nm = "__arr" + std::to_string(n++);
        std::vector<int> shape;
        std::vector<std::string> flat;
        if (!flattenArray(items, 0, shape, flat, who,
                          " (use a named `scatter` element for irregular data)")) return false;
        if ((int)shape.size() > PAT_ND_MAX_DIM) {
            fail(who + ": " + std::to_string(shape.size()) + " nested axes exceeds the " +
                 std::to_string((int)PAT_ND_MAX_DIM) + "-D limit");
            return false;
        }
        // Arity and argument SHAPE are checked HERE, not left to the generated grid sample,
        // so the message can talk about what the author wrote (`[…](u)`) instead of a name
        // they never chose.
        {
            std::vector<std::string> args;
            int kwAt = -1; std::string kwFormal;
            splitCallArgs(call, args, kwAt, kwFormal);
            // Emptiness before arity: `(u,)` is a stray comma, not a 2-D call, and saying so
            // beats "the array is 1-D but the call gives 2 coordinates".
            for (size_t k = 0; k < args.size(); ++k) {
                if (!args[k].empty()) continue;
                fail(who + ": axis " + std::to_string(k) + " of the sample call `" + call +
                     "` is empty — every axis needs a coordinate expression");
                return false;
            }
            if ((int)args.size() != (int)shape.size()) {
                fail(who + ": the array is " + std::to_string(shape.size()) +
                     "-D but its sample call `" + call + "` gives " +
                     std::to_string(args.size()) + " coordinate(s) — one per nesting level");
                return false;
            }
            // --- PINNED SEMANTICS: an inline literal's axes carry no names ------------
            // `formal=driver` binds a name belonging to the CALLEE. A material, a material
            // property and a named pattern all have callee-side input names, so `mat(a=u)`
            // / `src.reflect(u=v)` are meaningful there. An inline array literal has no
            // such namespace: its axes are positional and anonymous, and the names in its
            // own tuple are DRIVERS (coordinate expressions), not formals. Accepting
            // `[0 1](a=u)` would therefore have to invent a per-material default for `a`,
            // which two literals in one material could contradict — so it is refused, and
            // the message names the two spellings that actually do the two things an
            // author can mean.
            if (kwAt >= 0) {
                fail(who + ": `" + args[kwAt] + "` — an inline array literal's axes are "
                     "positional and unnamed, so a `formal=driver` argument has no formal "
                     "to bind. Its sample call takes DRIVERS: write `[…](" +
                     trimWs(args[kwAt].substr(args[kwAt].find('=') + 1)) +
                     ")` to spend the axis here, or `[…](" + kwFormal +
                     ")` to leave it free and rebind it where the material is USED — "
                     "`material mat(" + args[kwAt] + ")`");
                return false;
            }
        }

        Block g;
        g.type = "grid";
        g.name = nm;
        {
            std::vector<std::string> sh;
            for (int d : shape) sh.push_back(std::to_string(d));
            addGenStmt(g, "shape", sh, line);
            addGenStmt(g, "lo", std::vector<std::string>(shape.size(), "0"), line);
            addGenStmt(g, "hi", std::vector<std::string>(shape.size(), "1"), line);
            addGenStmt(g, "data", flat, line);
        }
        gen.push_back(std::move(g));
        // Re-attribute anything that goes wrong inside the generated block (an unknown
        // identifier in a coordinate, a `tex:` out of scope, …) back to the literal the
        // author actually wrote — `__arrN` is a name they never chose and cannot search
        // for. The grid and its wrapping pattern share the one name, so one entry covers
        // both.
        genSite_[nm] = who + ": the inline array literal's sample call `" + call + "`";
        outName = nm;
        return true;
    }

    // Rewrite a sample call's text, replacing every array literal COMPOSED inside it with
    // a reference to its own freshly-generated grid, so that `[0 1]([0.2 0.8](u))` ends up
    // as `grid:__arr1(grid:__arr0(u))` — TODO.md's `coord = NAME | NUMBER | value`, where
    // a coordinate may itself be a sampled value. Recursion is on the call text, so the
    // composition nests to any depth.
    //
    // The inner grids are emitted BEFORE the outer block that references them; ordering is
    // not actually load-bearing (the data pass registers every grid before any pattern
    // compiles), but emitting a definition before its use keeps the generated scene
    // readable when dumped.
    //
    // This is also the only place that can diagnose a malformed composed literal: the lexer
    // captures the whole call as one PARENWORD without balance-checking the brackets inside
    // it (see ftsl_scene.epeg), precisely so that the complaint can be phrased against the
    // author's own source instead of surfacing as a token that mysteriously fails to match.
    bool desugarNestedLiterals(const std::string& text, const std::string& who, int line,
                               std::vector<Block>& gen, int& n, std::string& out) {
        out.clear();
        for (size_t i = 0; i < text.size(); ) {
            if (text[i] != '[') { out += text[i++]; continue; }
            const size_t at = i;
            // The pattern language has NO bracket syntax of its own, so a `[` here always
            // opens a composed literal — but one written flush against an identifier
            // (`f[0 1](u)`) is a typo, not a composition. Caught before the substitution
            // rather than after, because the rewrite would otherwise glue the author's
            // token to the generated name and report an "unknown identifier `fgrid`" that
            // appears nowhere in their file.
            if (at > 0) {
                const char p = text[at - 1];
                if (std::isalnum((unsigned char)p) || p == '_' || p == '.' || p == ':') {
                    fail(who + ": an array literal composed into a sample call has to stand "
                         "on its own as a coordinate, but this one directly follows `" +
                         std::string(1, p) + "` — separate them, or drop the stray text");
                    return false;
                }
            }
            std::vector<BrItem> items;
            std::string err;
            if (!parseArrayText(text, i, items, err)) {
                fail(who + ": " + err + " in the array literal composed into the sample call "
                     "at `" + text.substr(at) + "`");
                return false;
            }
            const std::string lit = text.substr(at, i - at);
            if (i >= text.size() || text[i] != '(') {
                fail(who + ": the array literal `" + lit + "` composed into a sample call "
                     "needs its own trailing call naming the coordinates it is read at — "
                     "e.g. `" + lit + "(u)`");
                return false;
            }
            const size_t cs = i;
            int d = 0;
            for (; i < text.size(); ++i) {
                if (text[i] == '(') ++d;
                else if (text[i] == ')' && --d == 0) { ++i; break; }
            }
            if (d != 0) {
                fail(who + ": unbalanced `(` in the sample call of the composed array "
                     "literal `" + lit + "` at `" + text.substr(cs) + "`");
                return false;
            }
            const std::string innerCall = text.substr(cs, i - cs);
            std::string innerOut;
            if (!desugarNestedLiterals(innerCall, who, line, gen, n, innerOut)) return false;
            std::string nm;
            if (!buildArrayGrid(items, innerCall, who, line, gen, n, nm)) return false;
            out += "grid:" + nm + innerOut;
        }
        return true;
    }

    // Turn ONE literal into its `grid` + `pattern` pair (appended to `gen`) and rewrite
    // the statement's value to reference the generated pattern.
    bool desugarOne(Stmt& s, std::vector<Block>& gen, int& n) {
        const ArrayLit& a = *s.val.array;
        const std::string who = "line " + std::to_string(a.line) + ": `" + s.key + "`";
        if (a.call.empty()) {
            // Two ways to finish an array, and the message names both: SPEND the axis here
            // (`(u)`), or leave it as a FORMAL for whoever uses the material (`(a)`, bound
            // at the use site by `mat(a=u)`). The second is the "unsaturated, completed by
            // the user" case from the design — in ftrace it is spelled by naming `a`, the
            // one input with no per-hit intrinsic, rather than by omitting the call.
            fail(who + ": an inline array literal needs a trailing sample call naming the "
                 "coordinates it is read at — e.g. `[0 1](u)`, or `[[0 1][2 3]](u,v)` for "
                 "2-D. To leave the choice to whoever USES this material, name the free "
                 "input instead — `[0 1](a)` — and bind it at the use site with "
                 "`material mat(a=u)`. Write the call with nothing between it and the `]`.");
            return false;
        }
        if (!s.val.words.empty()) {
            fail(who + ": an inline array literal must be the whole value, but it follows '" +
                 s.val.words.back() + "'");
            return false;
        }
        // A literal composed INSIDE this one's sample call becomes its own grid first, and
        // the call text is rewritten to reference it. The rewritten text is used ONLY for
        // the generated expression: every message stays phrased against `a.call`, which is
        // what the author actually wrote.
        std::string emitCall;
        if (!desugarNestedLiterals(a.call, who, a.line, gen, n, emitCall)) return false;

        std::string nm;
        if (!buildArrayGrid(a.items, a.call, who, a.line, gen, n, nm)) return false;

        Block p;
        p.type = "pattern";
        p.name = nm;
        addGenStmt(p, "expr", {"grid:" + nm + emitCall}, a.line);
        gen.push_back(std::move(p));

        s.val.array.reset();
        s.val.words.push_back("pattern:" + nm);
        return true;
    }

    // The mirror of the case above: an array literal composed into a NAMED table's sample
    // call, `reflect grid:ramp([0.2 0.8](u))`. Here there is no `ArrayLit` at all — the
    // whole thing lexed as one WORD — so the rewrite happens on the token text, and what
    // comes out (`grid:ramp(grid:__arrN(u))`) is compiled by `tableCallPattern` later.
    // Both directions of composition therefore run through the one `desugarNestedLiterals`.
    bool desugarTableCall(Stmt& s, std::vector<Block>& gen, int& n) {
        std::string& w0 = s.val.words[0];
        const std::string who = "line " + std::to_string(s.line) + ": `" + s.key + "`";
        std::string out;
        if (!desugarNestedLiterals(w0, who, s.line, gen, n, out)) return false;
        w0 = out;
        return true;
    }

    bool desugarArrays(std::vector<Block>& blocks) {
        std::vector<Block> gen;
        int n = 0;
        // A block's flat `words` dump is a mirror of its statements (key, then value
        // words), so any block whose statements changed has to be re-mirrored — some
        // bodies (`palette`, `data`) are read through `words` rather than `stmts`.
        std::function<bool(Block&)> visit = [&](Block& b) -> bool {
            // A `grid`/`scatter` element's own `data [ … ]` is that element's samples, not
            // a value-site literal: nesting there is the element's shape and there is no
            // sample call to make. Its bracket group is read by addGrid / addScatter.
            if (b.type == "grid" || b.type == "scatter") return true;
            bool touched = false;
            for (auto& s : b.stmts) {
                if (s.val.array) { if (!desugarOne(s, gen, n)) return false; touched = true; }
                else if (!s.val.words.empty() && isTableCallHead(s.val.words[0]) &&
                         s.val.words[0].find('[') != std::string::npos) {
                    if (!desugarTableCall(s, gen, n)) return false;
                    touched = true;
                }
                if (s.val.block) { if (!visit(*s.val.block)) return false; }
            }
            if (touched && !b.words.empty()) {
                b.words.clear();
                for (const auto& s : b.stmts) {
                    b.words.push_back(s.key);
                    for (const auto& w : s.val.words) b.words.push_back(w);
                }
            }
            return true;
        };
        for (auto& b : blocks) if (!visit(b)) return false;
        for (auto& g : gen) blocks.push_back(std::move(g));
        return true;
    }

    // ---- N-D sampled arrays (`grid`) ----
    //   grid "name" {
    //       shape 3 4                 # sample counts per axis, axis 0 outermost (C order)
    //       lo 0 0                    # box corner: absent -> zeros, one number -> broadcast
    //       hi 1 1                    # absent -> unit-spacing index lattice, one -> isotropic
    //       outside clamp             # clamp (default) | wrap | extrapolate
    //       data { 0 1 2  3 4 5  … }  # product(shape) numbers, C order
    //   }
    // Sampled from any pattern expression as `grid:<name>(c0, …)` with one coordinate
    // per axis. Coordinates are the grid's OWN units and are NOT scaled by the scene's
    // `units` setting — like `pattern`'s `scale`/`size`, a grid is unit-agnostic; author
    // `lo`/`hi` in whatever the expression feeding it produces (metres, u/v, …).
    static bool parseGridOutside(const std::string& s, PatGridOutside& out) {
        if (s == "clamp" || s == "edge")   { out = PatGridOutside::Clamp;       return true; }
        if (s == "wrap"  || s == "repeat") { out = PatGridOutside::Wrap;        return true; }
        if (s == "extrapolate" || s == "extend") { out = PatGridOutside::Extrapolate; return true; }
        return false;
    }

    bool addGrid(const Block& b, Loaded& L) {
        if (b.name.empty()) { fail("grid needs a \"name\""); return false; }
        if (gridIndex_.count(b.name)) { fail("duplicate grid name '" + b.name + "'"); return false; }
        const std::string who = genWho("grid", b.name);

        // Samples: a `data { … }` brace body (the flat-word list form `palette {}` uses),
        // an inline `data 1 2 3` line, or a BRACKETED `data [[0 1 2][3 4 5]]` whose nesting
        // IS the shape (loom's data.Grid constructor rule — a shape you cannot get wrong,
        // because it is not written down twice). In the two flat forms, nesting/newlines
        // inside the body are pure formatting and C order is the shape.
        const Stmt* ds = find(b, "data");
        if (!ds) { fail(who + " needs a `data { … }` list of numbers"); return false; }
        std::vector<int> shape;            // filled from the nesting when `data` is bracketed
        std::vector<float> samples;
        if (ds->val.array) {
            if (!ds->val.array->call.empty()) {
                fail(who + ": `data` is this grid's own samples, so it takes no `" +
                     ds->val.array->call + "` sample call — the call belongs at a value site "
                     "that reads the grid, e.g. `grid:" + b.name + "(u)`");
                return false;
            }
            std::vector<std::string> words;
            if (!flattenArray(ds->val.array->items, 0, shape, words, who + " `data`")) return false;
            // Only NESTING carries a shape. A single flat `data [0 1 2 3]` is just the
            // bracketed spelling of the flat list, so an explicit `shape` still folds it.
            if (shape.size() < 2) shape.clear();
            samples.reserve(words.size());
            for (const auto& w : words) samples.push_back((float)num(w));
        } else {
            if (ds->val.block) markAllUsed(*ds->val.block);   // a flat numeric sample list
            const std::vector<std::string>& dw = ds->val.block ? ds->val.block->words : ds->val.words;
            samples.reserve(dw.size());
            for (const auto& w : dw) {
                if (!isNumber(w)) { fail(who + ": non-numeric sample '" + w + "' in `data`"); return false; }
                samples.push_back((float)num(w));
            }
        }
        if (samples.empty()) { fail(who + ": `data` is empty"); return false; }

        // Shape. Absent means "1-D, as long as the data" — the common ramp/LUT case. An
        // explicit `shape` is how a FLAT list is folded; alongside bracketed data it is
        // redundant, so it is accepted only when it agrees (and named when it doesn't).
        if (const Stmt* ss = find(b, "shape")) {
            std::vector<int> given;
            for (const auto& w : ss->val.words) {
                if (!isNumber(w)) { fail(who + ": non-numeric `shape` entry '" + w + "'"); return false; }
                int n = (int)num(w);
                if (n < 1) { fail(who + ": `shape` entries must be >= 1"); return false; }
                given.push_back(n);
            }
            if (!shape.empty() && given != shape) {
                auto join = [](const std::vector<int>& v) {
                    std::string s;
                    for (size_t k = 0; k < v.size(); ++k) { if (k) s += " "; s += std::to_string(v[k]); }
                    return s;
                };
                fail(who + ": `shape " + join(given) + "` disagrees with the nesting of `data`, "
                     "which is " + join(shape) + " — with bracketed data the shape is the nesting, "
                     "so just drop the `shape` line");
                return false;
            }
            if (shape.empty()) shape = given;
        }
        if (shape.empty()) shape.push_back((int)samples.size());
        if ((int)shape.size() > PAT_ND_MAX_DIM) {
            fail(who + ": " + std::to_string(shape.size()) + " axes exceeds the " +
                 std::to_string((int)PAT_ND_MAX_DIM) + "-D limit");
            return false;
        }
        long long need = 1;
        for (int n : shape) need *= (long long)n;
        if (need != (long long)samples.size()) {
            fail(who + ": `shape` wants " + std::to_string(need) + " samples but `data` has " +
                 std::to_string(samples.size()));
            return false;
        }

        PatGrid g;
        g.ndim = (int)shape.size();
        for (int a = 0; a < g.ndim; ++a) g.shape[a] = shape[a];

        // Read an axis-vector setting: absent -> `have=false`, one number -> broadcast,
        // exactly ndim numbers -> per-axis. Anything else is an authoring error.
        auto axisVec = [&](const char* key, double* out, bool& have, bool& scalar) -> bool {
            have = false; scalar = false;
            const Stmt* s = find(b, key);
            if (!s) return true;
            std::vector<double> v;
            for (const auto& w : s->val.words) {
                if (!isNumber(w)) { fail(who + std::string(": non-numeric `") + key + "` entry '" + w + "'"); return false; }
                v.push_back(num(w));
            }
            if (v.empty()) return true;
            if ((int)v.size() == 1) { scalar = true; for (int a = 0; a < g.ndim; ++a) out[a] = v[0]; }
            else if ((int)v.size() == g.ndim) { for (int a = 0; a < g.ndim; ++a) out[a] = v[a]; }
            else {
                fail(who + std::string(": `") + key + "` needs 1 or " + std::to_string(g.ndim) + " numbers");
                return false;
            }
            have = true;
            return true;
        };

        bool haveLo = false, loScalar = false, haveHi = false, hiScalar = false;
        double lo[PAT_ND_MAX_DIM] = {0, 0, 0, 0};
        double hi[PAT_ND_MAX_DIM] = {0, 0, 0, 0};
        if (!axisVec("lo", lo, haveLo, loScalar)) return false;
        if (!axisVec("hi", hi, haveHi, hiScalar)) return false;
        (void)loScalar;
        // `hi` defaults mirror loom's Grid._resolve_hi:
        //   absent  -> the unit-spacing INDEX lattice, hi[a] = lo[a] + shape[a] - 1
        //   scalar  -> an ISOTROPIC lattice: spacing is set by axis 0, other axes follow
        //   vector  -> the exact box
        if (!haveHi) {
            for (int a = 0; a < g.ndim; ++a) hi[a] = lo[a] + double(g.shape[a] - 1);
        } else if (hiScalar) {
            const double h = (g.shape[0] > 1) ? (hi[0] - lo[0]) / double(g.shape[0] - 1) : 0.0;
            for (int a = 0; a < g.ndim; ++a) hi[a] = lo[a] + h * double(g.shape[a] - 1);
        }
        for (int a = 0; a < g.ndim; ++a) {
            if (g.shape[a] > 1 && hi[a] == lo[a]) {
                fail(who + ": axis " + std::to_string(a) + " has " + std::to_string(g.shape[a]) +
                     " samples but a zero-width extent (lo == hi)");
                return false;
            }
            g.lo[a] = lo[a];
            g.hi[a] = hi[a];
        }

        std::string os = strOf(b, "outside", "clamp");
        if (!parseGridOutside(os, g.outside)) {
            fail(who + ": unknown `outside` '" + os + "' (clamp|wrap|extrapolate)");
            return false;
        }

        // Samples go into ONE shared pool; the header points at its run by OFFSET, never
        // by pointer — the pool grows as later grids load, and that is also the exact
        // flat layout the GPU uploads.
        g.off   = (int)L.scene.dataPool.size();
        g.count = (int)samples.size();
        L.scene.dataPool.insert(L.scene.dataPool.end(), samples.begin(), samples.end());
        int id = (int)L.scene.grids.size();
        L.scene.grids.push_back(g);
        gridIndex_[b.name] = id;
        return true;
    }

    // ---- N-D scattered samples (the ragged sibling of `grid`) ----
    //   scatter "name" {
    //       dim   2                   # coordinates per sample; absent -> 1
    //       power 2                   # Shepard exponent (absent -> 2); higher = tighter
    //       eps   1e-9                # squared-distance "this IS that sample" threshold
    //       data {                    # (dim + 1) numbers per sample: position…, value
    //           0 0   0.1
    //           1 0   0.9
    //           0.5 1 0.4
    //       }
    //   }
    // Sampled as `scatter:<name>(c0, …)` with one coordinate per dimension, exactly like
    // a grid — and like a grid, its coordinates are its OWN units, unscaled by `units`.
    // Use this where the data does NOT sit on a lattice; a `grid` is both smaller and
    // cheaper (O(2^ndim) vs O(count) per sample) whenever the data actually is regular.
    bool addScatter(const Block& b, Loaded& L) {
        if (b.name.empty()) { fail("scatter needs a \"name\""); return false; }
        if (scatterIndex_.count(b.name)) { fail("duplicate scatter name '" + b.name + "'"); return false; }
        const std::string who = "scatter '" + b.name + "'";

        PatScatter s;
        if (const Stmt* ds = find(b, "dim")) {
            if (ds->val.words.empty() || !isNumber(ds->val.words[0])) {
                fail(who + ": `dim` needs a number"); return false;
            }
            s.ndim = (int)num(ds->val.words[0]);
            if (s.ndim < 1 || s.ndim > PAT_ND_MAX_DIM) {
                fail(who + ": `dim` must be 1.." + std::to_string((int)PAT_ND_MAX_DIM));
                return false;
            }
        }

        if (const Stmt* ps = find(b, "power")) {
            if (ps->val.words.empty() || !isNumber(ps->val.words[0])) {
                fail(who + ": `power` needs a number"); return false;
            }
            s.power = num(ps->val.words[0]);
            if (!(s.power > 0.0)) { fail(who + ": `power` must be > 0"); return false; }
        }
        if (const Stmt* es = find(b, "eps")) {
            if (es->val.words.empty() || !isNumber(es->val.words[0])) {
                fail(who + ": `eps` needs a number"); return false;
            }
            s.eps = num(es->val.words[0]);
            if (s.eps < 0.0) { fail(who + ": `eps` must be >= 0"); return false; }
        }

        // Samples: same flat-word `data { … }` body a grid uses, but read in strides of
        // (dim + 1) — the position's coordinates followed by that sample's value. One
        // interleaved list keeps a sample's position and value visually together, which
        // is the whole point of a scatter (they are not separable the way a lattice is).
        // Brackets are also accepted, and there the nesting carries the same meaning it
        // does for a grid: `data [[0 0 0.1][1 0 0.9]]` is one group PER SAMPLE, so the
        // stride is checked per group and a miscount names the offending sample.
        const Stmt* ds = find(b, "data");
        if (!ds) { fail(who + " needs a `data { … }` list of numbers"); return false; }
        const int stride = s.ndim + 1;
        std::vector<float> flat;
        if (ds->val.array) {
            if (!ds->val.array->call.empty()) {
                fail(who + ": `data` is this scatter's own samples, so it takes no `" +
                     ds->val.array->call + "` sample call — the call belongs at a value site "
                     "that reads it, e.g. `scatter:" + b.name + "(u)`");
                return false;
            }
            std::vector<int> shape;
            std::vector<std::string> words;
            if (!flattenArray(ds->val.array->items, 0, shape, words, who + " `data`")) return false;
            if (shape.size() > 2) {
                fail(who + ": `data` nests " + std::to_string(shape.size()) + " deep, but a "
                     "scatter's samples are a flat list or ONE group per sample");
                return false;
            }
            if (shape.size() == 2 && shape[1] != stride) {
                fail(who + ": each `data` group has " + std::to_string(shape[1]) + " numbers but "
                     "dim+1 = " + std::to_string(stride) + " (each sample is " +
                     std::to_string(s.ndim) + " coordinate(s) then its value)");
                return false;
            }
            flat.reserve(words.size());
            for (const auto& w : words) flat.push_back((float)num(w));
        } else {
            if (ds->val.block) markAllUsed(*ds->val.block);   // a flat numeric sample list
            const std::vector<std::string>& dw = ds->val.block ? ds->val.block->words : ds->val.words;
            flat.reserve(dw.size());
            for (const auto& w : dw) {
                if (!isNumber(w)) { fail(who + ": non-numeric entry '" + w + "' in `data`"); return false; }
                flat.push_back((float)num(w));
            }
        }
        if (flat.empty()) { fail(who + ": `data` is empty"); return false; }
        if ((int)(flat.size() % (size_t)stride) != 0) {
            fail(who + ": `data` has " + std::to_string(flat.size()) + " numbers, which is not a " +
                 "multiple of dim+1 = " + std::to_string(stride) + " (each sample is " +
                 std::to_string(s.ndim) + " coordinate(s) then its value)");
            return false;
        }

        s.count = (int)(flat.size() / (size_t)stride);
        s.off   = (int)L.scene.dataPool.size();
        L.scene.dataPool.insert(L.scene.dataPool.end(), flat.begin(), flat.end());
        int id = (int)L.scene.scatters.size();
        L.scene.scatters.push_back(s);
        scatterIndex_[b.name] = id;
        return true;
    }

    // ---- parametric records (§records) ----
    // Parse a record domain: either one hyphen-joined token "LO-HI" (the common form,
    // e.g. `range 0-1`) or two number tokens (`range 0 1`). Requires HI > LO.
    static bool parseRecordDomain(const std::vector<std::string>& w, double& lo, double& hi) {
        if (w.size() == 2 && isNumber(w[0]) && isNumber(w[1])) {
            lo = num(w[0]); hi = num(w[1]); return hi > lo;
        }
        if (w.size() == 1) {
            const std::string& s = w[0];
            // Split at a '-' that yields two numbers, skipping a leading sign and any
            // exponent marker (so "1e-3-2", "-1-1", "0.5-2.5" all parse).
            for (size_t k = 1; k < s.size(); ++k) {
                if (s[k] != '-') continue;
                char p = s[k - 1];
                if (p == 'e' || p == 'E' || p == '+' || p == '-') continue;
                std::string a = s.substr(0, k), b = s.substr(k + 1);
                if (isNumber(a) && isNumber(b)) { lo = num(a); hi = num(b); return hi > lo; }
            }
        }
        return false;
    }

    // Assign domain positions to a channel's stops: an author `p:<pos>` pins a stop;
    // the first/last unpinned stops anchor to lo/hi; each interior run of unpinned
    // stops spreads evenly between its fixed neighbours.
    static void redistributeStops(RecChannel& ch, double lo, double hi) {
        const int n = (int)ch.stops.size();
        std::vector<char> fixed(n, 0);
        for (int i = 0; i < n; ++i) fixed[i] = ch.stops[i].pinned ? 1 : 0;
        if (n == 1) { if (!fixed[0]) ch.stops[0].pos = lo; return; }
        if (!fixed[0])     { ch.stops[0].pos = lo;     fixed[0] = 1; }
        if (!fixed[n - 1]) { ch.stops[n - 1].pos = hi; fixed[n - 1] = 1; }
        int a = 0;
        while (a < n) {
            if (!fixed[a]) { ++a; continue; }
            int b = a + 1;
            while (b < n && !fixed[b]) ++b;
            if (b < n && b > a + 1) {
                double pa = ch.stops[a].pos, pb = ch.stops[b].pos;
                int gaps = b - a;
                for (int j = a + 1; j < b; ++j)
                    ch.stops[j].pos = pa + (pb - pa) * double(j - a) / double(gaps);
            }
            a = b;
        }
    }

    // Read one record channel line's stops — the **generalized stop grammar** (J3b
    // item 2, `src/record_ladder.h`). Three delimiters form a fixed precedence ladder:
    // whitespace binds tightest (juxtaposition -> a vector), comma looser (a new stop),
    // brackets are its parentheses. So structure comes from the delimiters alone and
    // the channel's arity only *validates*:
    //
    //     reflect  spectrum:steel spectrum:gold      # 2 colour-ref stops (as always)
    //     rough    0 0.4 1                           # 3 scalar stops    (as always)
    //     reflect  rgb 0 0 0, 1 1 1                  # 2 inline-colour stops (J3b item 1)
    //     reflect  rgb [0 0 0] [1 1 1]               # ...the same thing, bracket form
    //     tint     0 0 0,                            # ONE arity-3 stop (trailing comma)
    //
    // The whole thing is a strict ADDITIVE SUPERSET: a line using none of the ladder
    // delimiters and carrying no colour tag takes a path that is behaviourally the old
    // one — every word its own stop — so no record in the tree can reparse differently.
    // That is deliberate rather than incidental: records are data, and a silent change
    // in how an existing gradient is chopped into stops would show up only as a subtly
    // wrong render, which is exactly the class of regression worth designing out.
    //
    // A leading colour head (`rgb`/`hsv`/`hsl` and every upsampler variant — see
    // `isColourHead`) is the channel-level INLINE-COLOUR TAG: it fixes arity 3, so each
    // comma group (or the lone group) is one colour written in place, instead of a chain
    // of `spectrum:<name>` refs that each need their own top-level declaration.
    bool parseChannelStops(const std::string& recName, RecChannel& ch,
                           const std::vector<std::string>& words) {
        auto bad = [&](const std::string& m) {
            fail("record '" + recName + "' channel '" + ch.name + "': " + m);
            return false;
        };
        if (words.empty()) return bad("has no stops");

        // A leading colour head tags the whole channel and is not itself a stop.
        size_t first = 0;
        if (isColourHead(words[0])) {
            ch.space = words[0];
            first = 1;
            if (words.size() == 1) return bad("'" + ch.space + "' tag with no stops");
        }
        const std::vector<std::string> body(words.begin() + first, words.end());

        std::vector<std::string> toks;
        recladder::tokenize(body, toks);

        // Fast path: no ladder delimiter and no tag -> the plain whitespace list, read
        // exactly as it always was (a `p:<pos>` word pins the stop that follows it).
        if (ch.space.empty() && !recladder::usesLadder(toks)) {
            bool havePin = false; double pinPos = 0.0;
            for (const auto& w : toks) {
                if (w.rfind("p:", 0) == 0) {
                    if (!isNumber(w.substr(2))) return bad("bad p:<pos> '" + w + "'");
                    havePin = true; pinPos = num(w.substr(2));
                    continue;
                }
                RecStop st; st.token = w;
                if (havePin) { st.pinned = true; st.pos = pinPos; havePin = false; }
                ch.stops.push_back(std::move(st));
            }
            if (havePin) return bad("trailing p:<pos> with no value");
            if (ch.stops.empty()) return bad("has no stops");
            return true;
        }

        recladder::Value v;
        bool trailingComma = false;
        std::string lerr;
        if (!recladder::parse(toks, v, trailingComma, lerr)) return bad(lerr);

        // Turn the parsed ladder into a stop list. A leading `p:<pos>` component pins
        // the stop it introduces (the ladder is purely structural, so the pin prefix
        // stays an orthogonal concern peeled off here).
        auto stopFrom = [&](const recladder::Value& g, RecStop& st) -> bool {
            std::vector<std::string> comps;
            if (g.isLeaf) comps.push_back(g.leaf);
            else for (const auto& c : g.items) {
                if (!c.isLeaf) return bad("stop is nested more than two levels deep");
                comps.push_back(c.leaf);
            }
            if (!comps.empty() && comps[0].rfind("p:", 0) == 0) {
                if (!isNumber(comps[0].substr(2))) return bad("bad p:<pos> '" + comps[0] + "'");
                st.pinned = true; st.pos = num(comps[0].substr(2));
                comps.erase(comps.begin());
                if (comps.empty()) return bad("p:<pos> with no value");
            }
            st.token = comps[0];
            if (comps.size() > 1) st.comps = comps;
            return true;
        };

        const int depth = v.depth();
        if (depth >= 2) {                       // a group per stop — the general shape
            for (const auto& g : v.items) {
                RecStop st;
                if (!stopFrom(g, st)) return false;
                ch.stops.push_back(std::move(st));
            }
        } else if (!ch.space.empty() || trailingComma) {
            // One juxtaposed run that is ONE stop, not N: either the arity-3 tag says so
            // (`reflect rgb .5 .5 .5`) or the author's trailing comma does (`tint 0 0 0,`).
            RecStop st;
            if (!stopFrom(v, st)) return false;
            ch.stops.push_back(std::move(st));
        } else {
            // Untagged, comma-free, but bracketed — `rough [0] [1]`. Brackets around a
            // single value are idempotent, so this is still the whitespace reading.
            if (v.isLeaf) { RecStop st; if (!stopFrom(v, st)) return false; ch.stops.push_back(std::move(st)); }
            else for (const auto& g : v.items) {
                RecStop st;
                if (!stopFrom(g, st)) return false;
                ch.stops.push_back(std::move(st));
            }
        }
        if (ch.stops.empty()) return bad("has no stops");

        // Validate arity against the channel's kind. ftrace materializes exactly two
        // kinds of channel — scalar and colour — so an arity-D vector channel has no
        // destination here even though the grammar happily describes one. Say that
        // outright rather than failing later with a confusing per-stop message.
        for (const auto& st : ch.stops) {
            const size_t arity = st.comps.empty() ? 1 : st.comps.size();
            if (!ch.space.empty()) {
                if (arity != 3)
                    return bad("'" + ch.space + "' stops need 3 components, got " +
                               std::to_string(arity));
            } else if (arity != 1) {
                return bad("arity-" + std::to_string(arity) + " vector stops have no "
                           "destination in ftrace — tag the channel with a colour head "
                           "(e.g. `" + ch.name + " rgb " + st.comps[0] + " " + st.comps[1] +
                           " " + st.comps[2 % st.comps.size()] + ", ...`) to make it a "
                           "colour channel, or write one value per stop for a scalar one");
            }
        }
        return true;
    }

    // Build one Record from a `NAME = range LO-HI [ ... ]` block: parse the domain,
    // interp, channels and stops; redistribute positions (stage 1); then compile each
    // stop into a scalar pattern program or a resolved colour + linear-RGB (stage 2).
    bool addRecord(const Block& b, Loaded& L) {
        if (b.name.empty()) { fail("record needs a name"); return false; }
        if (recordIndex_.count(b.name)) { fail("duplicate record name '" + b.name + "'"); return false; }
        Record rec;
        rec.name = b.name;
        bool haveRange = false;
        // Every line of a record body is meaningful — `range`, `interp`, or a CHANNEL
        // whose name the author invents — so there is no such thing as an unknown key
        // here and the whole body counts as read.
        markAllUsed(b);
        for (const auto& s : b.stmts) {
            if (s.key == "range") {
                if (!parseRecordDomain(s.val.words, rec.lo, rec.hi)) {
                    fail("record '" + rec.name + "': bad `range` (need LO-HI or LO HI with HI>LO)");
                    return false;
                }
                haveRange = true;
                continue;
            }
            if (s.key == "interp") {
                const std::string& m = s.val.words.empty() ? std::string() : s.val.words[0];
                if      (m == "nearest") rec.interp = RecInterp::Nearest;
                else if (m == "linear")  rec.interp = RecInterp::Linear;
                else if (m == "smooth")  rec.interp = RecInterp::Smooth;
                else { fail("record '" + rec.name + "': interp must be nearest|linear|smooth"); return false; }
                continue;
            }
            // Otherwise: a channel line, read by the generalized stop grammar.
            RecChannel ch;
            ch.name = s.key;
            if (!parseChannelStops(rec.name, ch, s.val.words)) return false;
            redistributeStops(ch, rec.lo, rec.hi);
            rec.channels.push_back(std::move(ch));
        }
        if (!haveRange) { fail("record '" + rec.name + "' has no `range LO-HI`"); return false; }
        if (rec.channels.empty()) { fail("record '" + rec.name + "' has no channels"); return false; }
        // Validate: pinned positions in [lo,hi] and non-decreasing per channel.
        for (const auto& ch : rec.channels) {
            for (const auto& st : ch.stops) {
                if (st.pos < rec.lo - 1e-9 || st.pos > rec.hi + 1e-9) {
                    fail("record '" + rec.name + "' channel '" + ch.name + "': stop position " +
                         std::to_string(st.pos) + " is outside the domain");
                    return false;
                }
            }
            for (size_t i = 1; i < ch.stops.size(); ++i) {
                if (ch.stops[i].pos < ch.stops[i - 1].pos - 1e-12) {
                    fail("record '" + rec.name + "' channel '" + ch.name + "': stop positions must be non-decreasing");
                    return false;
                }
            }
        }
        // Stage 2: compile stop tokens. A channel is a COLOUR channel two ways — either
        // it carries an inline-colour TAG (`reflect rgb 0 0 0, 1 1 1`), or its stops are
        // prefixed spectrum refs (`spectrum:steel` / `metal:copper` — anything with a
        // ':'). Otherwise every stop is a SCALAR pattern expression (a literal, or math
        // over intrinsics x y z nx ny nz r u v f + functions). A channel must be
        // homogeneous. Note the two colour forms CONVERGE here: a tagged stop is handed
        // to the very same `evalSpectrum` as `rgb 0 0 0` written at any other value
        // site, so it inherits every upsampler and colour space for free and nothing
        // downstream (the JH coeff bake, the CPU sampler, the GPU upload) can tell the
        // two apart.
        for (auto& ch : rec.channels) {
            if (!ch.space.empty()) {
                ch.kind = ChanKind::Spectrum;
            } else {
                bool anyColour = false, anyScalar = false;
                for (const auto& st : ch.stops)
                    (st.token.find(':') != std::string::npos ? anyColour : anyScalar) = true;
                if (anyColour && anyScalar) {
                    fail("record '" + rec.name + "' channel '" + ch.name +
                         "': mixes colour (spectrum:...) and scalar stops");
                    return false;
                }
                ch.kind = anyColour ? ChanKind::Spectrum : ChanKind::Scalar;
            }
            for (auto& st : ch.stops) {
                if (ch.kind == ChanKind::Scalar) {
                    std::string cerr;
                    if (!compilePatternExpr(st.token, st.expr, cerr, false, &texScope_, &tableScope_)) {
                        fail("record '" + rec.name + "' channel '" + ch.name +
                             "': bad stop expression '" + st.token + "': " + cerr);
                        return false;
                    }
                } else {
                    Value v;
                    if (ch.space.empty()) v.words = { st.token };
                    else {
                        v.words.push_back(ch.space);
                        for (const auto& c : st.comps) v.words.push_back(c);
                    }
                    st.color = evalSpectrum(v);
                    if (!err.empty()) return false;   // evalSpectrum already set the message
                    st.rgb = reflectanceToLinearSrgbD65(st.color);
                }
            }
        }
        recBakeSpectrumChannels(rec);   // colour channels -> per-domain JH coeff LUT
        int id = (int)L.scene.records.size();
        recordIndex_[rec.name] = id;
        L.scene.records.push_back(std::move(rec));
        return true;
    }

    // ---- materials ----
    // True if a material block is a §records override block: it either bulk-imports a
    // record (`from R(d)`) or assigns a slot from a record/expression (`slot = …`, whose
    // parsed value begins with a lone `=` token). Such blocks are built by
    // buildRecordOverrideMaterial instead of the ordinary key→value path.
    static bool isRecordOverrideBlock(const Block& b) {
        for (const auto& s : b.stmts) {
            if (s.key == "from") return true;
            if (!s.val.words.empty() && s.val.words[0] == "=") return true;
        }
        return false;
    }

    // Build a material from a §records override block (stage 4): an ordered list of
    // `type <kind>`, `from R(driver)`, and `slot = <rhs>` statements. Statements are
    // processed top→bottom and each write to a slot overrides earlier ones (last-write-
    // wins). RHS forms: a scalar expression (scalar slots), a bare imported-channel
    // name, `REC.chan` (driven by the most recent `from REC(...)`), or a constant
    // selector `REC.chan[i]` / `self.chan[i]` (the channel's i-th stop).
    Material buildRecordOverrideMaterial(const Block& b, Loaded& L) {
        Material m;                          // default type: diffuse
        m.reflect = constantSpectrum(0.75);
        // Base type (optional) — lets a record drive e.g. a glossy base.
        if (find(b, "type")) {
            std::string type = strOf(b, "type", "diffuse");
            if      (type == "diffuse")   m.type = MatType::Diffuse;
            else if (type == "glossy")  { m.type = MatType::Glossy; m.reflect = constantSpectrum(0.75); }
            else { fail("record-override material: unsupported base `type " + type +
                        "` (only diffuse/glossy)"); return m; }
        }
        // Channels imported by a `from` (name -> its source), for `slot = channel` and
        // `slot = self.chan[i]` lookups; plus each record's most-recent `from` driver
        // (for a bare `slot = REC.chan` that needs a driver).
        struct ImportedChan { int recIdx, chanIdx; std::vector<PatNode> driver; };
        std::unordered_map<std::string, ImportedChan> imported;
        std::unordered_map<std::string, std::pair<int, std::vector<PatNode>>> fromDriver;  // recName -> (idx,drv)

        // This loop is exhaustive — every statement is `type`, `from`, or a `slot = rhs`
        // assignment, and an unrecognised slot hard-fails below — so nothing here can be
        // an unread key.
        markAllUsed(b);
        for (const auto& s : b.stmts) {
            if (s.key == "type") continue;                       // already handled
            if (s.key == "from") {
                if (s.val.words.empty()) { fail("`from` needs RECORD(driver)"); return m; }
                std::string raw = s.val.words[0];
                for (size_t k = 1; k < s.val.words.size(); ++k) raw += " " + s.val.words[k];
                size_t lp = raw.find('('), rp = raw.rfind(')');
                if (lp == std::string::npos || rp == std::string::npos || rp <= lp) {
                    fail("`from " + raw + "`: expected RECORD(driver)"); return m;
                }
                std::string rname = raw.substr(0, lp);
                std::string dexpr = raw.substr(lp + 1, rp - lp - 1);
                auto rit = recordIndex_.find(rname);
                if (rit == recordIndex_.end()) { fail("`from`: unknown record '" + rname + "'"); return m; }
                std::vector<PatNode> drv; std::string cerr;
                if (!compilePatternExpr(dexpr, drv, cerr, false, &texScope_,
                                        &tableScope_, /*allowA=*/true)) {
                    fail("`from " + rname + "` driver '" + dexpr + "': " + cerr); return m;
                }
                const Record& rec = L.scene.records[rit->second];
                applyFrom(m, rit->second, drv, rec);
                for (int c = 0; c < (int)rec.channels.size(); ++c)
                    imported[rec.channels[c].name] = { rit->second, c, drv };
                fromDriver[rname] = { rit->second, drv };
                continue;
            }
            // Otherwise: a `slot = <rhs>` assignment. val.words == ["=", rhs?].
            const std::string& slot = s.key;
            int slotId = recSlotId(slot);
            if (slotId < 0) { fail("record-override material: '" + slot +
                                   "' is not a record-fillable slot (reflect|roughness)"); return m; }
            if (s.val.words.size() < 2 || s.val.words[0] != "=") {
                fail("record-override material: `" + slot + " = <value>` needs a right-hand side"); return m;
            }
            const std::string& rhs = s.val.words[1];
            bool ok = true;

            // Split a trailing `[i]` stop selector, if any: `REC.chan[i]` / `self.chan[i]`.
            std::string base = rhs; int selStop = -1;
            size_t lb = rhs.find('[');
            if (lb != std::string::npos) {
                size_t rb = rhs.rfind(']');
                if (rb == std::string::npos || rb <= lb || !isNumber(rhs.substr(lb + 1, rb - lb - 1))) {
                    fail("record-override `" + slot + " = " + rhs + "`: bad stop selector"); return m;
                }
                selStop = (int)num(rhs.substr(lb + 1, rb - lb - 1));
                base = rhs.substr(0, lb);
            }

            // A channel reference (`REC.chan`, `self.chan`, or a bare imported-channel
            // name) is a *simple* token — identifier chars plus a single dotted qualifier
            // — never a numeric literal or an expression (which carries parens/operators
            // or a decimal point). This keeps `roughness = sin(v*3.14159)` an expression
            // rather than mis-reading `3.14159`'s dot as `REC.chan`. A `[i]` selector was
            // already stripped, so its presence forces the reference interpretation.
            auto isSimpleRef = [](const std::string& s) {
                if (s.empty()) return false;
                for (char c : s)
                    if (!(std::isalnum((unsigned char)c) || c == '_' || c == '.')) return false;
                return true;
            };
            bool refForm = selStop >= 0 || (isSimpleRef(base) && !isNumber(base));

            // Resolve the base reference to (recordIndex, channelIndex, driver). A dotted
            // `REC.chan` names a record channel directly; `self.chan` / a bare name looks
            // up an imported channel. A bare name that is not an imported channel falls
            // through to a scalar expression (scalar slots only).
            int recIdx = -1, chanIdx = -1; std::vector<PatNode> drv; bool haveSrc = false;
            size_t dot = base.find('.');
            if (refForm && dot != std::string::npos) {
                std::string head = base.substr(0, dot), chan = base.substr(dot + 1);
                if (head == "self") {
                    auto ic = imported.find(chan);
                    if (ic == imported.end()) { fail("record-override `self." + chan +
                        "`: no channel '" + chan + "' imported by a preceding `from`"); return m; }
                    recIdx = ic->second.recIdx; chanIdx = ic->second.chanIdx; drv = ic->second.driver;
                } else {
                    auto rit = recordIndex_.find(head);
                    if (rit == recordIndex_.end()) { fail("record-override: unknown record '" + head + "'"); return m; }
                    recIdx = rit->second;
                    chanIdx = L.scene.records[recIdx].channelIndex(chan);
                    if (chanIdx < 0) { fail("record-override: record '" + head +
                        "' has no channel '" + chan + "'"); return m; }
                    // Driver: reuse the most recent `from head(...)` in this block. Not
                    // needed for a constant stop selector.
                    auto fd = fromDriver.find(head);
                    if (fd != fromDriver.end()) drv = fd->second.second;
                    else if (selStop < 0) { fail("record-override `" + base +
                        "`: needs a driver — add `from " + head + "(<driver>)` first"); return m; }
                }
                haveSrc = true;
            } else if (refForm) {
                // A bare simple reference: an imported-channel name.
                auto ic = imported.find(base);
                if (ic != imported.end()) {
                    recIdx = ic->second.recIdx; chanIdx = ic->second.chanIdx; drv = ic->second.driver;
                    haveSrc = true;
                } else if (selStop >= 0) {
                    fail("record-override `" + rhs + "`: no channel '" + base +
                         "' imported by a preceding `from`"); return m;
                }
                // else: an unqualified identifier that is not an imported channel — let it
                // fall through to the expression compiler (e.g. a bare intrinsic like `u`).
            }

            RecBinding rb; rb.slot = slotId; rb.selStop = selStop;
            if (haveSrc) {
                const RecChannel& ch = L.scene.records[recIdx].channels[chanIdx];
                if (slotId == REC_SLOT_REFLECT && ch.kind != ChanKind::Spectrum) {
                    fail("record-override `reflect = " + rhs + "`: channel '" + ch.name +
                         "' is scalar, not a colour channel"); return m;
                }
                if (slotId == REC_SLOT_ROUGHNESS && ch.kind != ChanKind::Scalar) {
                    fail("record-override `roughness = " + rhs + "`: channel '" + ch.name +
                         "' is a colour channel, not scalar"); return m;
                }
                rb.recordIndex = recIdx; rb.channel = chanIdx; rb.driver = std::move(drv);
            } else {
                // Not a channel reference: a direct scalar expression (scalar slots only).
                if (slotId == REC_SLOT_REFLECT) {
                    fail("record-override `reflect = " + rhs + "`: expected a colour channel "
                         "(a spectrum channel of a record), not an expression"); return m;
                }
                if (selStop >= 0) { fail("record-override `" + slot + " = " + rhs +
                    "`: a stop selector needs a record channel"); return m; }
                std::string cerr;
                if (!compilePatternExpr(rhs, rb.driver, cerr, false, &texScope_,
                                        &tableScope_, /*allowA=*/true)) {
                    fail("record-override `" + slot + " = " + rhs + "`: " + cerr); return m;
                }
                rb.recordIndex = -1;
            }
            setBinding(m, rb);
            (void)ok;
        }
        return m;
    }

    Material buildMaterial(const Block& b, Loaded& L) {
        if (isRecordOverrideBlock(b)) return buildRecordOverrideMaterial(b, L);
        Material m;
        // Built-in whole-material recipe: `preset <name>` fills a complete material
        // (metal / glass / iridescent film). A few common knobs may still be
        // overridden afterwards so a preset can be lightly retuned.
        if (find(b, "preset")) {
            std::string pname = strOf(b, "preset", "");
            if (!resolveMaterialPreset(pname, m)) { fail("unknown material preset '" + pname + "'"); return m; }
            if (find(b, "roughness")) {
                if (!bindScalarPattern(b, "roughness", m.roughnessPat) &&
                    !bindScalarTexture(b, "roughness", m.roughnessTex))
                    m.roughness = dblParam(b, "roughness", m.roughness);
            }
            if (find(b, "film_ior"))       m.filmIor       = dblParam(b, "film_ior", m.filmIor);
            if (find(b, "film_thickness")) m.filmThickness = dblParam(b, "film_thickness", m.filmThickness);
            if (!bindScalarPattern(b, "film_thickness_map", m.filmThicknessPat))
                bindScalarTexture(b, "film_thickness_map", m.filmThicknessTex);
            if (find(b, "reflect") || find(b, "reflect_map"))
                m.reflect = reflectParam(b, m, m.reflect);
            if (find(b, "ior"))            m.ior           = spectrumParam(b, "ior", m.ior);
            checkSlotPatsSupported(b, m);
            return m;
        }
        std::string type = strOf(b, "type", "diffuse");
        if (type == "diffuse") {
            m.type = MatType::Diffuse;
            // `reflect texture:<name>` binds a spatially-varying albedo; otherwise a
            // uniform reflectance spectrum (or a pattern — see reflectParam). A bound
            // texture leaves m.reflect as the fallback used where UVs are unavailable
            // (e.g. the CUDA bake path); `reflect_map` still modulates it.
            if (bindReflectTexture(b, m)) { bindScalarPattern(b, "reflect_map", m.reflectPat); m.reflect = constantSpectrum(0.75); }
            else                          m.reflect = reflectParam(b, m, constantSpectrum(0.75));
        } else if (type == "translucent" || type == "diffuse_transmit") {
            // Two-lobe Lambertian: `reflect` (front-hemisphere diffuse albedo) +
            // `transmit` (back-hemisphere diffuse albedo). Both non-specular, so a
            // directly-viewed solid is visible in mode B. reflect+transmit is clamped
            // to <= 1 per wavelength at render time (the remainder is absorbed).
            m.type = MatType::DiffuseTransmit;
            if (bindReflectTexture(b, m)) { bindScalarPattern(b, "reflect_map", m.reflectPat); m.reflect = constantSpectrum(0.5); }
            else                          m.reflect = reflectParam(b, m, constantSpectrum(0.4));
            m.transmit = transmitParam(b, m, constantSpectrum(0.4));
        } else if (type == "dielectric") {
            m.type = MatType::Dielectric;
            m.ior = spectrumParam(b, "ior", glassOrDefault("BK7", 1.5168));
            // Frosted/rough transmission: 0 (default) = perfectly clear glass, bit-
            // identical to before; >0 roughens both the reflected and refracted lobes.
            // `roughness pattern:<name>` (§4) or `texture:<name>` binds a per-hit map.
            if (bindScalarPattern(b, "roughness", m.roughnessPat)) m.roughness = 0.2;
            else if (bindScalarTexture(b, "roughness", m.roughnessTex)) m.roughness = 0.2;
            else m.roughness = dblParam(b, "roughness", 0.0);
            // Interior absorption sigma_a(lambda) per metre travelled inside the glass
            // (Beer-Lambert tint). 0 (default) = colorless. e.g. `absorb rgb 3 0.5 0.3`
            // (a tagged RGB triple, upsampled to a spectrum) gives green-tinted glass.
            // NOTE: the `rgb` tag is required — a bare triple (`absorb 3 0.5 0.3`) is NOT
            // a valid spectrum expression; only a scalar, a tagged colour (`rgb`/`xyz`/…),
            // or a named/ref spectrum parse. See evalSpectrum below.
            m.absorb = spectrumParam(b, "absorb", constantSpectrum(0.0));
        } else if (type == "mirror") {
            m.type = MatType::Mirror;
            m.reflect = reflectParam(b, m, constantSpectrum(0.95));
        } else if (type == "halfmirror") {
            m.type = MatType::HalfMirror;
            m.reflect = reflectParam(b, m, constantSpectrum(0.5));
        } else if (type == "filter") {
            // Colored gel / Wratten filter: a thin non-scattering absorber. A photon
            // passes straight through, surviving with probability `transmit`(lambda) —
            // the per-wavelength transmittance T(lambda) in [0,1] — and is absorbed
            // otherwise. No reflection, no refraction. Feed T from a measured curve
            // (`transmit file:data/filter/rosco-red.csv` / `transmit filter:red-25`)
            // or a primitive (`transmit gaussian center=630 sigma=25`).
            m.type = MatType::Filter;
            m.transmit = transmitParam(b, m, constantSpectrum(0.5));
        } else if (type == "glossy") {
            m.type = MatType::Glossy;
            m.reflect = reflectParam(b, m, constantSpectrum(0.9));
            // `roughness pattern:<name>` (§4) / `texture:<name>` binds a per-hit
            // roughness map (grayscale = roughness directly, both 0..1); else a constant.
            if (bindScalarPattern(b, "roughness", m.roughnessPat)) m.roughness = 0.2;
            else if (bindScalarTexture(b, "roughness", m.roughnessTex)) m.roughness = 0.2;
            else m.roughness = dblParam(b, "roughness", 0.2);
        } else if (type == "thinfilm") {
            m.type = MatType::ThinFilm;
            m.ior = spectrumParam(b, "ior", iorConstant(1.5));
            m.filmIor = dblParam(b, "film_ior", 1.30);
            // `film_thickness <nm>` is the peak/scale; `film_thickness_map texture:<n>`
            // binds a 0..1 profile scaled by it (spatially-varying iridescence, §9.4).
            m.filmThickness = dblParam(b, "film_thickness", 300.0);
            if (!bindScalarPattern(b, "film_thickness_map", m.filmThicknessPat))
                bindScalarTexture(b, "film_thickness_map", m.filmThicknessTex);
            // Substrate extinction kappa (spectral): 0 = transparent dielectric
            // (lossless, default). Non-zero -> absorbing/metallic substrate giving
            // opaque structural colour; a spectral kappa (e.g. a gaussian) tints it
            // like a real metal (gold, copper).
            m.substrateK = spectrumParam(b, "substrate_k", constantSpectrum(0.0));
        } else if (type == "grating") {
            m.type = MatType::Grating;
            m.reflect = reflectParam(b, m, constantSpectrum(0.9));
            m.grooveSpacing = dblParam(b, "groove_spacing", 1000.0);
            Vec3 gd{0, 1, 0}; vec3Of(b, "groove_dir", gd); m.grooveDir = gd;
            m.gratingMaxOrder = (int)dblParam(b, "max_order", 3);
        } else if (type == "fluorescent") {
            m.type = MatType::Fluorescent;
            m.reflect = reflectParam(b, m, constantSpectrum(0.1));
            m.fluoAbsorb = spectrumParam(b, "absorb", shortPass(490.0, 0.15, 1.0));
            m.fluoEmit = spectrumParam(b, "emit", gaussianBand(560.0, 25.0, 1.0));
            m.fluoYield = dblParam(b, "yield", 1.0);
            m.fluoEmitSampler.build(m.fluoEmit, 1.0);
        } else if (type == "multilayer") {
            // Multilayer thin-film stack (Bragg / dichroic). Substrate index/kappa
            // via ior / substrate_k; the stack is an ordered list of `layer <n> <k>
            // <thickness_nm>` statements, layer 0 outermost (nearest the incident
            // air). Evaluated with the Abeles characteristic-matrix method.
            m.type = MatType::Multilayer;
            m.ior = spectrumParam(b, "ior", iorConstant(1.5));
            m.substrateK = spectrumParam(b, "substrate_k", constantSpectrum(0.0));
            for (const auto& s : b.stmts) {
                if (s.key != "layer") continue;
                s.used = true;
                if (s.val.words.size() < 3) { fail("multilayer 'layer' needs: <n> <k> <thickness_nm>"); return m; }
                m.layerN.push_back(num(s.val.words[0]));
                m.layerK.push_back(num(s.val.words[1]));
                m.layerThick.push_back(num(s.val.words[2]));
            }
            if (m.layerN.empty()) { fail("multilayer material has no 'layer' entries"); return m; }
        } else if (type == "mix") {
            // Stochastic mix of named child materials. Children are resolved to
            // indices in a second pass (they may be declared later in the file);
            // here we only mark the type — resolveMixChildren() fills the lists.
            m.type = MatType::Mix;
        } else if (type == "layered") {
            // Physical two-layer stack (§3.2): a specular coat interface over a
            // weighted body. On a hit a photon reflects off the coat with prob R
            // (Fresnel / thin-film Airy / manual constant), else it enters and one
            // body lobe is chosen (the `layer "name" weight` list, resolved in the
            // second pass exactly like a mix). Coat R + body weights partition the
            // photon so the surface stays energy-consistent.
            m.type = MatType::Layered;
            m.ior = spectrumParam(b, "ior", iorConstant(1.5));   // body/effective index
            const Stmt* cs = find(b, "coat");
            if (!cs || !cs->val.block) { fail("layered material needs a coat { } block"); return m; }
            const Block& cb = *cs->val.block;
            // Coat reflectance model: fresnel (default) | thinfilm | manual.
            std::string cmodel = strOf(cb, "reflectance", "fresnel");
            if      (cmodel == "fresnel")  m.coatModel = 0;
            else if (cmodel == "thinfilm") m.coatModel = 1;
            else if (cmodel == "manual")   m.coatModel = 2;
            else { fail("layered coat reflectance must be fresnel|thinfilm|manual"); return m; }
            // Coat interface roughness (glossy lobe on the reflected ray); grayscale
            // roughness_map allowed just like a glossy material.
            if (bindScalarPattern(cb, "roughness", m.roughnessPat)) m.roughness = 0.05;
            else if (bindScalarTexture(cb, "roughness", m.roughnessTex)) m.roughness = 0.05;
            else m.roughness = dblOf(cb, "roughness", 0.05);
            // Fresnel/thinfilm read the coat index from `ior` (coat over body index
            // m.ior); manual uses a flat specular fraction.
            if (find(cb, "ior")) m.ior = spectrumParam(cb, "ior", m.ior);
            m.filmIor = dblOf(cb, "film_ior", 1.30);
            m.filmThickness = dblOf(cb, "film_thickness", 300.0);
            bindScalarTexture(cb, "film_thickness_map", m.filmThicknessTex);
            if (m.coatModel == 2) m.coatSpecular = dblOf(cb, "specular", 0.05);
        } else {
            fail("unknown material type '" + type + "'");
        }
        checkSlotPatsSupported(b, m);
        // Nested-dielectric priority (§ nested dielectrics): `priority <N>` — integer,
        // higher wins where dielectric solids overlap. Common to every material type
        // (only consulted for dielectric-like ones); unset => the ahead-of-time audit
        // warns if this material overlaps another dielectric without a priority.
        if (find(b, "priority")) m.priority = (int)std::lround(dblOf(b, "priority", 0.0));
        // Emissive surfaces (§ mesh area lights): any material may carry an `emit`
        // spectrum, turning every triangle it is bound to into a light that radiates
        // `emit(lambda)` from its front face. This drives emission-on-hit generically
        // (m.isLight + m.emit, already consumed by every renderer); a mesh bound to
        // such a material is additionally registered as a Mesh area emitter in addMesh
        // so NEE / forward emission sample it. The SPD is radiance per unit solid angle
        // per unit area (absolute if the scene is absolute); a mesh block's optional
        // `power`/`lumens` rescales it there to hit a target flux over the mesh area.
        // `emit` takes the same two pattern spellings as `reflect`/`transmit` (0.80.0):
        // `emit pattern:p` makes the pattern the whole emission profile (greyscale, base
        // spectrum flat 1.0), `emit_map pattern:p` modulates whatever `emit` otherwise
        // says. Unlike those two the pattern is read from BOTH sides of transport —
        // emission-on-hit and the Le at an emitter-sampled point — so it is only legal
        // where the two provably agree on (u,v); checkEmitPatSupported (run after the
        // scene is built, when the emitter shapes are known) enforces that.
        // EXCEPT on a `fluorescent` material, where `emit` was already consumed above as the
        // RERADIATION profile (`fluoEmit` — the Stokes-shifted emission SHAPE, normalised by
        // its own integral at every use site), not as self-emission. Letting the generic block
        // also run made every fluorescent surface a self-luminous absolute-radiance light of
        // its own emission band, which on the CPU tracers put a `gaussian center=560` dye pane
        // ~8200× above a 0.5-albedo floor's radiance under the same lamp — impossible at
        // `yield <= 1` — while the GPU (which never uploaded that `emit` slot) rendered only
        // the elastic base. That is the whole of the CPU/GPU fluorescence divergence.
        // `emit_map` has no meaning on a fluorescent either, so say so rather than drop it.
        if (m.type == MatType::Fluorescent) {
            if (find(b, "emit_map"))
                fail("a fluorescent material's 'emit' is its reradiation spectrum, not surface "
                     "emission, so 'emit_map' is not supported here");
        } else if (find(b, "emit") || find(b, "emit_map")) {
            m.emit = patternedSpectrumParam(b, "emit", "emit_map", m.emitPat,
                                            constantSpectrum(0.0));
            m.isLight = true;
        }
        // Tangent-space NORMAL MAP (C6): `normal_map texture:<name> [strength <s>]`.
        // Common to every material type. Binds a declared (linear-encoded) texture as
        // the material's normal map and reads an optional perturbation strength (default
        // 1). The shading normal is perturbed at closestHit via the surface TBN frame.
        bindNormalTexture(b, m, L);
        return m;
    }

    // Bind a `normal_map texture:<name> [strength <s>]` on a material (C6). The value
    // must reference a declared texture (loaded `encoding linear` — a normal map is raw
    // vector data, not colour). Sets m.normalTex + m.normalStrength; warns if the map is
    // not linear-encoded (a common authoring mistake that de-gammas the vectors). The
    // optional `strength <s>` is a trailing token of the same statement.
    void bindNormalTexture(const Block& b, Material& m, Loaded& L) {
        const Stmt* s = find(b, "normal_map");
        if (!s || s->val.words.empty()) return;
        const std::string& w0 = s->val.words[0];
        std::string nm = (w0.rfind("texture:", 0) == 0) ? w0.substr(8) : w0;  // bare name tolerated
        auto it = textureIndex_.find(nm);
        if (it == textureIndex_.end()) { fail("normal_map references unknown texture '" + nm + "'"); return; }
        m.normalTex = it->second;
        if (it->second >= 0 && it->second < (int)L.scene.textures.size() &&
            L.scene.textures[it->second].encoding != TexEncoding::Linear)
            std::fprintf(stderr, "[ftsl] warning: normal_map texture '%s' is not "
                         "'encoding linear' — normal vectors will be de-gammaed\n", nm.c_str());
        // Optional trailing `strength <s>` token on the normal_map statement.
        for (size_t i = 1; i + 1 < s->val.words.size(); ++i)
            if (s->val.words[i] == "strength") m.normalStrength = num(s->val.words[i + 1]);
    }

    // Second material pass: resolve a Mix material's `layer "name" weight` entries
    // to child indices + weights. Called after every material name is registered,
    // so a mix may reference children declared before OR after it. Nested mixes are
    // rejected to keep resolution single-step (and the CUDA CDF bounded).
    // Takes the child material by INDEX, not by reference: resolving a layer name goes
    // through lookupMaterial(), which may append to L.scene.mats (a material with a
    // free `a` materialises its albedo_default clone on first use), and a `Material&`
    // held across that would dangle on reallocation.
    bool resolveMixChildren(const Block& b, int matIdx, Loaded& L) {
        double sum = 0.0;
        std::vector<int>    kids;
        std::vector<double> wts;
        for (const auto& s : b.stmts) {
            if (s.key != "layer") continue;
            s.used = true;
            if (s.val.words.size() < 2) { fail("mix 'layer' needs a material name and a weight"); return false; }
            const std::string& cname = s.val.words[0];
            double w = num(s.val.words[1]);
            int child = lookupMaterial(cname, L);
            if (child < 0) { fail("mix layer references unknown material '" + cname + "'"); return false; }
            if (L.scene.mats[child].type == MatType::Mix ||
                L.scene.mats[child].type == MatType::Layered) {
                fail("layer '" + cname + "' is itself a mix/layered (nesting is not allowed)"); return false;
            }
            if (w < 0.0) { fail("mix layer weight must be >= 0"); return false; }
            kids.push_back(child);
            wts.push_back(w);
            sum += w;
        }
        Material& m = L.scene.mats[matIdx];   // safe: no more mats appends below
        m.mixChildren = std::move(kids);
        m.mixWeights  = std::move(wts);
        if (m.mixChildren.empty()) { fail("mix material has no 'layer' entries"); return false; }
        if (sum > 1.0 + 1e-9) { fail("mix layer weights sum to " + std::to_string(sum) + " (> 1)"); return false; }
        // Optional per-hit blend mask: `weight_map pattern:<name>` (§4, math-driven
        // spatial selection — the key mechanism for per-xyz material choice on an
        // implicit surface) or `weight_map texture:<name>` (§9.4, UV map) drives the
        // selection weight of child 0 (child 1 = 1 - map). Only meaningful for a
        // 2-child mix — reject otherwise so the semantics stay unambiguous.
        if (find(b, "weight_map")) {
            if (m.mixChildren.size() != 2) {
                fail("mix weight_map requires exactly 2 layers (a binary A/B blend)"); return false;
            }
            if (!bindScalarPattern(b, "weight_map", m.mixWeightPat))
                bindScalarTexture(b, "weight_map", m.mixWeightTex);
        }
        return true;
    }

    // ---- geometry ----
    // Every geometry/light builder takes an authored-space affine `xf` (identity
    // for top-level primitives; the composed transform of the enclosing `group`
    // chain otherwise). Authored coordinates are transformed by `xf` FIRST, then
    // P()/Len() fold in the unit scale (→ metres). With xf = identity the result
    // is bit-identical to the pre-group path.
    // Tessellate a sphere (authored center `c`, radius `r`) baked through a
    // non-uniform / sheared affine `xf` into a smooth-normal triangle mesh, so a
    // squashed / skewed sphere renders as the ellipsoid (or sheared quadric) the
    // analytic primitive can't represent. Positions go through the affine; shading
    // normals go through its inverse-transpose (`applyNormal`) so the surface stays
    // smooth under non-uniform scale. Only the fallback — a uniform-scaled sphere
    // keeps the fast analytic path in addSphere().
    void addTessellatedSphere(Loaded& L, const Affine& xf, const Vec3& c,
                              double r, int id) {
        const int nlat = 48;   // latitude bands  (theta 0..PI)
        const int nlon = 96;   // longitude steps (phi   0..2PI)
        auto dirOf = [&](int i, int j) -> Vec3 {
            double theta = PI * (double)i / nlat;
            double phi   = 2.0 * PI * (double)j / nlon;
            double st = std::sin(theta), ct = std::cos(theta);
            return Vec3{st * std::cos(phi), ct, st * std::sin(phi)};  // y-up unit dir
        };
        auto pos = [&](const Vec3& d) { return P(xf.apply(c + d * r)); };
        auto nrm = [&](const Vec3& d) { return normalize(xf.applyNormal(d)); };
        auto uv  = [&](int i, int j) { return Vec3{(double)j / nlon, (double)i / nlat, 0.0}; };
        for (int i = 0; i < nlat; ++i) {
            for (int j = 0; j < nlon; ++j) {
                Vec3 d00 = dirOf(i, j),     d01 = dirOf(i, j + 1);
                Vec3 d10 = dirOf(i + 1, j), d11 = dirOf(i + 1, j + 1);
                if (i != 0) {                       // skip degenerate north-pole tri
                    Tri t{pos(d00), pos(d10), pos(d11), id, -1, {}};
                    t.n0 = nrm(d00); t.n1 = nrm(d10); t.n2 = nrm(d11);
                    t.uv0 = uv(i, j); t.uv1 = uv(i + 1, j); t.uv2 = uv(i + 1, j + 1);
                    L.scene.tris.push_back(t);
                }
                if (i != nlat - 1) {                // skip degenerate south-pole tri
                    Tri t{pos(d00), pos(d11), pos(d01), id, -1, {}};
                    t.n0 = nrm(d00); t.n1 = nrm(d11); t.n2 = nrm(d01);
                    t.uv0 = uv(i, j); t.uv1 = uv(i + 1, j + 1); t.uv2 = uv(i, j + 1);
                    L.scene.tris.push_back(t);
                }
            }
        }
    }
    bool addSphere(const Block& b, Loaded& L, const Affine& xf = Affine::identity()) {
        Vec3 c{0, 0, 0}; vec3Of(b, "center", c);
        double r = dblOf(b, "radius", 1.0);
        int id = matFieldId(b, L, "sphere"); if (id < 0) return false;
        // A sphere stays an analytic sphere only under translate + rotation + UNIFORM
        // scale. A non-uniform scale (or a shear) would make it an ellipsoid / sheared
        // quadric the analytic primitive can't represent, so tessellate it into a
        // smooth-normal mesh baked through the affine instead of failing.
        bool nonUniform = false; double s = xf.uniformScale(nonUniform);
        if (nonUniform) {
            addTessellatedSphere(L, xf, c, r, id);
            // (A tessellated sphere has no analytic center/radius, so it is not
            // registered in sphereByName_ — it can't serve as an analytic fog bound.)
            return true;
        }
        Vec3 wc = P(xf.apply(c));
        double wr = Len(r) * s;
        L.scene.spheres.push_back(Sphere{wc, wr, id});
        if (!b.name.empty()) sphereByName_[b.name] = NamedSphere{wc, wr};
        return true;
    }
    bool addQuad(const Block& b, Loaded& L, const Affine& xf = Affine::identity()) {
        Vec3 o{0, 0, 0}, u{1, 0, 0}, v{0, 0, 1};
        vec3Of(b, "origin", o); vec3Of(b, "u", u); vec3Of(b, "v", v);
        int id = matFieldId(b, L, "quad"); if (id < 0) return false;
        Vec3 a = P(xf.apply(o)), bb = P(xf.apply(o + u)),
             cc = P(xf.apply(o + u + v)), dd = P(xf.apply(o + v));
        // UVs span the parallelogram: origin=(0,0), +u=(1,0), +v=(0,1). The two
        // triangles share the o and o+u+v corners; assign matching corner UVs so a
        // bound texture maps continuously across the quad.
        Tri t1{a, bb, cc, id, -1, {}};
        t1.uv0 = {0, 0, 0}; t1.uv1 = {1, 0, 0}; t1.uv2 = {1, 1, 0};
        Tri t2{a, cc, dd, id, -1, {}};
        t2.uv0 = {0, 0, 0}; t2.uv1 = {1, 1, 0}; t2.uv2 = {0, 1, 0};
        L.scene.tris.push_back(t1);
        L.scene.tris.push_back(t2);
        return true;
    }
    bool addTriangle(const Block& b, Loaded& L, const Affine& xf = Affine::identity()) {
        Vec3 v0{0, 0, 0}, v1{1, 0, 0}, v2{0, 1, 0};
        vec3Of(b, "v0", v0); vec3Of(b, "v1", v1); vec3Of(b, "v2", v2);
        int id = matFieldId(b, L, "triangle"); if (id < 0) return false;
        L.scene.tris.push_back(Tri{P(xf.apply(v0)), P(xf.apply(v1)), P(xf.apply(v2)), id, -1, {}});
        return true;
    }
    bool addMesh(const Block& b, Loaded& L, const Affine& parentXf = Affine::identity()) {
        std::string file = strOf(b, "file");
        if (file.empty()) { fail("mesh needs a file"); return false; }
        int id = matFieldId(b, L, "mesh"); if (id < 0) return false;
        MeshXform mx;
        vec3Of(b, "translate", mx.translate);
        vec3Of(b, "rotate", mx.rotDeg);
        // scale accepts a single uniform value or a vec3.
        const Stmt* sc = find(b, "scale");
        if (sc) {
            if (sc->val.words.size() >= 3)
                mx.scale = {num(sc->val.words[0]), num(sc->val.words[1]), num(sc->val.words[2])};
            else if (!sc->val.words.empty()) {
                double k = num(sc->val.words[0]); mx.scale = {k, k, k};
            }
        }
        // Compose the enclosing group's authored-space transform with the mesh's
        // own local transform, then fold the unit scale into the OUTPUT so the
        // baked verts land in metres (scaling an affine's linear part and
        // translation by L_ scales its result by L_). With parentXf = identity
        // this reproduces the old `scale*=L_; translate*=L_` path exactly.
        Affine xf = parentXf.compose(mx.toAffine());
        for (double& e : xf.m) e *= L_;
        xf.t = xf.t * L_;
        // `uv use_mesh` reads texture coordinates from the OBJ's `vt` records (needed
        // for textured materials); the default keeps the Tri fallback UVs.
        // `uv planar|spherical|cylindrical [x|y|z]` instead synthesizes UVs at load
        // time from the world-space vertex positions (spec §9.2 procedural
        // projections), for meshes without their own `vt` coordinates. The optional
        // second token picks the projection/up axis (default y).
        const std::string uvMode = strOf(b, "uv");
        bool loadUV = (uvMode == "use_mesh");
        UvProjection uvProj = parseUvProjection(uvMode);
        // The projection/up axis is an optional value continuation on the `uv`
        // statement. It must be a `key=val` param (`uv planar axis=x`) — a bareword
        // (`uv planar x`) would start a NEW statement the parser never folds back in,
        // so the axis would be silently ignored. Default y.
        int uvAxis = 1;   // y up by default
        if (const Stmt* uvs = find(b, "uv"); uvs && uvMode != "triplanar") {
            for (size_t i = 1; i < uvs->val.words.size(); ++i) {
                std::string k, a;
                if (!splitEq(uvs->val.words[i], k, a) || k != "axis") continue;
                if (a == "x") uvAxis = 0; else if (a == "z") uvAxis = 2; else uvAxis = 1;
            }
        }
        // `uv triplanar [<s>|scale=<s>]` (spec §9.2) can't be baked into per-vertex
        // UVs — it blends three world-axis projections per hit, weighted by the
        // surface normal — so it's carried on the bound material as a world-to-texture
        // scale (repeats per world unit) and applied in diffuseReflectance / dDiffuseRho.
        // The scale must be a *value continuation* the parser keeps on the `uv`
        // statement: a bare number (`uv triplanar 4`) or a `key=val` param
        // (`uv triplanar scale=4`). A bareword `scale` would instead start a NEW
        // statement and collide with the mesh's own `scale` transform, so it is not
        // accepted. Default scale 1.0.
        if (uvMode == "triplanar") {
            double tpScale = 1.0;
            if (const Stmt* uvs = find(b, "uv")) {
                const auto& w = uvs->val.words;
                for (size_t i = 1; i < w.size(); ++i) {
                    std::string k, val;
                    if (splitEq(w[i], k, val)) { if (k == "scale") tpScale = num(val); }
                    else if (isNumber(w[i]))   tpScale = num(w[i]);
                }
            }
            if (id >= 0 && id < (int)L.scene.mats.size()) L.scene.mats[id].triplanarScale = tpScale;
            loadUV = false; uvProj = UvProjection::None;
        }
        // `usemtl use_names` switches material per OBJ `usemtl` group by matching the
        // group name to an FTSL material of the same name (unknown -> the mesh's
        // default `material`). Two-token maps can't survive the statement splitter,
        // so name-matching is the grammar-friendly convention (mirrors `uv use_mesh`).
        bool useNames = (strOf(b, "usemtl") == "use_names");
        MtlResolver resolver = [this, &L](const std::string& nm) -> int {
            return lookupMaterial(nm, L);   // resolves a free `a` to albedo_default
        };
        // `shape_only yes` — load this mesh's triangles as a SHAPE REFERENCE only and
        // keep them out of the rendered scene. The motivating case is fog cut to an
        // imported silhouette: `medium { bounds { object "cloud" } }` bakes the mesh
        // into an occupancy lattice, after which the 1.8M-triangle shell is not only
        // useless but actively wrong — it would render as a solid surface wrapped
        // around the fog it was supposed to define, and would cost a BVH it never
        // needs. There is no material that means "not there" (a `filter` with
        // transmit 1 still consumes a bounce at every crossing), so this is a load-time
        // property of the mesh, applied once the shape has been consumed.
        std::string soStr = strOf(b, "shape_only");
        bool shapeOnly = !soStr.empty() &&
                         !(soStr == "no" || soStr == "off" || soStr == "false" || soStr == "0");
        if (shapeOnly && b.name.empty()) {
            fail("mesh `shape_only` needs a \"name\" — the whole point is for something "
                 "else (e.g. `medium { bounds { object \"…\" } }`) to reference the shape");
            return false;
        }
        if (shapeOnly && id >= 0 && id < (int)L.scene.mats.size() && L.scene.mats[id].isLight) {
            fail("mesh \"" + b.name + "\": `shape_only` cannot be combined with an emissive "
                 "(`emit`) material — a light with no surface would be silently dropped");
            return false;
        }
        size_t triStart = L.scene.tris.size();
        // Dispatch by file extension: .gltf/.glb use the glTF loader (which imports
        // its own pbrMetallicRoughness materials by default; `import_materials no`
        // forces the FTSL-assigned `material` on every primitive). Everything else
        // is an OBJ. Extension match is case-insensitive.
        std::string ext;
        if (size_t dot = file.find_last_of('.'); dot != std::string::npos) {
            ext = file.substr(dot);
            for (char& c : ext) c = (char)std::tolower((unsigned char)c);
        }
        if (ext == ".gltf" || ext == ".glb") {
            bool importMats = (strOf(b, "import_materials") != "no");
            // `skip_material <substr>[,<substr>…]` (glTF/GLB only), repeatable: drop
            // every primitive whose glTF material name contains one of these, matched
            // case-insensitively. Asset-store models routinely bundle a ground plane
            // or studio backdrop into the same file as the subject, and there is no
            // way to subtract geometry after loading. NOTE: list several by repeating
            // the statement or comma-joining them — a space-separated `a b` would be
            // split into a separate statement by the parser (see wordListOf).
            std::vector<std::string> skipMats = wordListOf(b, "skip_material");
            std::string gerr;
            if (loadGltf(L.scene, file.c_str(), id, xf, importMats, gerr, skipMats) == 0
                && !gerr.empty()) {
                fail("mesh: " + gerr); return false;
            }
        } else if (ext == ".fbx") {
            // Autodesk FBX via the vendored ufbx bridge. `uv use_mesh` pulls the file's
            // first UV set; procedural UV projections and crease smoothing are OBJ-only.
            std::string ferr;
            if (loadFbx(L.scene, file.c_str(), id, xf, loadUV, ferr) == 0 && !ferr.empty()) {
                fail("mesh: " + ferr); return false;
            }
        } else {
            // `smooth [<deg>]` (OBJ only): when the mesh has no `vn`, auto-generate
            // smooth shading normals, merging faces across edges softer than <deg>
            // (default 40°) and leaving sharper creases faceted. Authored `vn` wins.
            double creaseAngleDeg = -1.0;
            if (const Stmt* sm = find(b, "smooth")) {
                creaseAngleDeg = 40.0;
                if (!sm->val.words.empty() && isNumber(sm->val.words[0]))
                    creaseAngleDeg = num(sm->val.words[0]);
            }
            loadObj(L.scene, file.c_str(), id, xf, loadUV, useNames ? &resolver : nullptr,
                    uvProj, uvAxis, creaseAngleDeg);
        }
        // Record the object as a named mesh group (for -check-watertight): the range of
        // world triangles this block just appended. Unnamed blocks get a synthesized
        // "mesh#N" label so the report can still point at them.
        if (L.scene.tris.size() > triStart) {
            MeshGroup g;
            g.name = b.name.empty() ? ("mesh#" + std::to_string(L.scene.meshGroups.size())) : b.name;
            g.triStart = triStart;
            g.triCount = L.scene.tris.size() - triStart;
            g.blasId   = -1;
            g.matId    = id;
            g.shapeOnly = shapeOnly;
            L.scene.meshGroups.push_back(std::move(g));
            // A shape-only mesh exists purely to hand its silhouette to something else
            // (today: a `medium { bounds { object "<name>" } }` containment bake). Its
            // triangles have to survive until the deferred medium sweep has read them,
            // so the actual removal happens later, in stripShapeOnlyMeshes().
            if (shapeOnly) shapeOnlyGroups_.push_back(L.scene.meshGroups.size() - 1);
        }
        // Emissive mesh → register a Mesh area light (§ mesh area lights). When the
        // bound material carries an `emit` spectrum, the triangles just appended form
        // one area light: emission-on-hit already works via m.isLight/m.emit, and this
        // adds the emitter so NEE / forward emission sample the mesh. An optional
        // `power`/`lumens` on the mesh block rescales the SPD to a target flux over the
        // mesh's total area; to keep emission-on-hit consistent (and not disturb the
        // material if it is shared with a non-emissive or differently-scaled mesh), the
        // scaled case clones the material, rebinds this range's triangles to the clone,
        // and registers the emitter against the clone. Only the single-material OBJ/FBX
        // path is handled (glTF meshes that import their own materials are not auto-lit).
        if (id >= 0 && id < (int)L.scene.mats.size() && L.scene.mats[id].isLight &&
            L.scene.tris.size() > triStart) {
            size_t triEnd = L.scene.tris.size();
            // Orient a CLOSED emissive mesh outward. Emission is one-sided (radiates
            // along each triangle's geometric normal / front face only), so an
            // inward-wound imported shell — e.g. torus.obj, all faces wound toward the
            // interior — would emit into its own hollow and look black from outside.
            // Detect closure via the signed volume about the centroid: for a closed
            // shell |V| = enclosed volume and sign follows the winding (negative =
            // inward); for a planar/open sheet V≈0. If V is negative AND large enough to
            // be a real enclosed volume (thresholded against area^1.5 so open meshes are
            // never touched), reverse every triangle's winding so its front face — and
            // thus its emission — points OUTWARD. This runs before emitter registration
            // so both the per-Tri geometric normals used by emission-on-hit and the
            // addMeshLight sampler normals come out consistent.
            {
                size_t n = triEnd - triStart;
                Vec3 cen{0, 0, 0};
                for (size_t t = triStart; t < triEnd; ++t) {
                    const Tri& tr = L.scene.tris[t];
                    cen = cen + tr.v0 + tr.v1 + tr.v2;
                }
                cen = cen / (3.0 * (double)n);
                double vol = 0.0, area2 = 0.0;
                for (size_t t = triStart; t < triEnd; ++t) {
                    const Tri& tr = L.scene.tris[t];
                    Vec3 a = tr.v0 - cen, bb = tr.v1 - cen, cc = tr.v2 - cen;
                    vol += dot(a, cross(bb, cc));
                    area2 += length(cross(tr.v1 - tr.v0, tr.v2 - tr.v0));
                }
                vol /= 6.0;
                double area = 0.5 * area2;
                if (vol < -1e-6 * std::pow(area, 1.5)) {
                    for (size_t t = triStart; t < triEnd; ++t) {
                        Tri& tr = L.scene.tris[t];
                        std::swap(tr.v1, tr.v2);
                        std::swap(tr.uv1, tr.uv2);
                        std::swap(tr.n1, tr.n2);
                        tr.finalize();
                    }
                }
            }
            if (find(b, "power") || find(b, "lumens")) {
                // Total front-facing area of the range → power law geomW = area*PI.
                double area = 0.0;
                for (size_t t = triStart; t < triEnd; ++t) {
                    const Tri& tr = L.scene.tris[t];
                    area += 0.5 * length(cross(tr.v1 - tr.v0, tr.v2 - tr.v0));
                }
                Spectrum scaled = absPower(b, L.scene.mats[id].emit, area * PI, L);
                Material clone = L.scene.mats[id];
                clone.emit = scaled;
                int newId = (int)L.scene.mats.size();
                L.scene.mats.push_back(std::move(clone));
                for (size_t t = triStart; t < triEnd; ++t) L.scene.tris[t].matId = newId;
                L.scene.addMeshLight(triStart, triEnd - triStart, scaled, binWidth_, newId);
            } else {
                L.scene.addMeshLight(triStart, triEnd - triStart,
                                     L.scene.mats[id].emit, binWidth_, id);
            }
        }
        // Record the loaded mesh's world AABB for object-name fog bounds. This is no
        // longer the *shape* of such a bound — `bounds { object "..." }` solid-voxelizes
        // the mesh and carves the silhouette (see the Mesh branch of addMedium) — it is
        // the TRACKING bound: delta/ratio tracking still clips to a box, and this is it.
        if (!b.name.empty() && L.scene.tris.size() > triStart) {
            Aabb box; bool first = true;
            for (size_t t = triStart; t < L.scene.tris.size(); ++t) {
                const Tri& tr = L.scene.tris[t];
                for (const Vec3& v : {tr.v0, tr.v1, tr.v2}) {
                    if (first) { box.lo = v; box.hi = v; first = false; } else box.expand(v);
                }
            }
            meshAabbByName_[b.name] = box;
        }
        return true;
    }

    // ---- mesh_asset (shared instanced geometry) ----
    // `mesh_asset "name" { file "asset.obj|gltf|glb"  material <m>  [import_materials no]
    //  [skip_material <substr>[,…]]  [uv use_mesh]  [usemtl use_names] }` loads a mesh ONCE into its own local
    //  (authored) space as a BLAS (Scene::blasList). It bakes NO world transform and
    //  emits NO triangles into Scene::tris — placement is done by `mesh_instance`,
    //  which references the asset by name. Multiple instances share this one BLAS,
    //  so N copies cost N affines rather than N triangle sets.
    bool addMeshAsset(const Block& b, Loaded& L) {
        if (b.name.empty()) { fail("mesh_asset needs a name: mesh_asset \"name\" { ... }"); return false; }
        if (blasIndex_.count(b.name)) { fail("duplicate mesh_asset name '" + b.name + "'"); return false; }
        std::string file = strOf(b, "file");
        if (file.empty()) { fail("mesh_asset '" + b.name + "' needs a file"); return false; }
        int id = matFieldId(b, L, "mesh_asset"); if (id < 0) return false;

        bool loadUV = (strOf(b, "uv") == "use_mesh");
        bool useNames = (strOf(b, "usemtl") == "use_names");
        MtlResolver resolver = [this, &L](const std::string& nm) -> int {
            return lookupMaterial(nm, L);   // resolves a free `a` to albedo_default
        };
        // Load into local space (identity transform) at the END of Scene::tris, then
        // move those triangles out into a private BLAS. The unit scale is NOT folded in
        // here — it is applied per-instance so one asset can serve differently-scaled
        // placements.
        Affine xf = Affine::identity();
        size_t start = L.scene.tris.size();
        std::string ext;
        if (size_t dot = file.find_last_of('.'); dot != std::string::npos) {
            ext = file.substr(dot);
            for (char& c : ext) c = (char)std::tolower((unsigned char)c);
        }
        if (ext == ".gltf" || ext == ".glb") {
            bool importMats = (strOf(b, "import_materials") != "no");
            std::string gerr;
            std::vector<std::string> skipMats = wordListOf(b, "skip_material");  // see the mesh block
            if (loadGltf(L.scene, file.c_str(), id, xf, importMats, gerr, skipMats) == 0
                && !gerr.empty()) {
                fail("mesh_asset: " + gerr); return false;
            }
        } else if (ext == ".fbx") {
            std::string ferr;
            if (loadFbx(L.scene, file.c_str(), id, xf, loadUV, ferr) == 0 && !ferr.empty()) {
                fail("mesh_asset: " + ferr); return false;
            }
        } else {
            double creaseAngleDeg = -1.0;
            if (const Stmt* sm = find(b, "smooth")) {
                creaseAngleDeg = 40.0;
                if (!sm->val.words.empty() && isNumber(sm->val.words[0]))
                    creaseAngleDeg = num(sm->val.words[0]);
            }
            loadObj(L.scene, file.c_str(), id, xf, loadUV, useNames ? &resolver : nullptr,
                    UvProjection::None, 1, creaseAngleDeg);
        }
        Blas blas;
        blas.tris.assign(L.scene.tris.begin() + start, L.scene.tris.end());
        L.scene.tris.resize(start);
        if (blas.tris.empty()) { fail("mesh_asset '" + b.name + "' loaded no triangles"); return false; }
        blas.build();
        int blasId = (int)L.scene.blasList.size();
        L.scene.blasList.push_back(std::move(blas));
        blasIndex_[b.name] = blasId;
        // Named mesh group backed by the shared BLAS (for -check-watertight).
        MeshGroup g; g.name = b.name; g.blasId = blasId; g.matId = id;
        L.scene.meshGroups.push_back(std::move(g));
        return true;
    }

    // ---- mesh_instance (place a shared mesh_asset) ----
    // `mesh_instance { of "asset-name"  [translate ..] [rotate ..] [scale ..]
    //  [material <m>] }` places a `mesh_asset` into the world via an affine (composed
    //  with the enclosing group's transform, then the scene's unit scale). `material`
    //  overrides the asset's own per-triangle materials for this placement; without it
    //  the asset's materials (glTF-imported or the asset's fallback) are used.
    bool addMeshInstance(const Block& b, Loaded& L, const Affine& parentXf = Affine::identity()) {
        std::string of = strOf(b, "of");
        if (of.empty()) { fail("mesh_instance needs `of \"asset-name\"`"); return false; }
        auto it = blasIndex_.find(of);
        if (it == blasIndex_.end()) { fail("mesh_instance: unknown mesh_asset '" + of + "'"); return false; }
        MeshXform mx;
        vec3Of(b, "translate", mx.translate);
        vec3Of(b, "rotate", mx.rotDeg);
        const Stmt* sc = find(b, "scale");
        if (sc) {
            if (sc->val.words.size() >= 3)
                mx.scale = {num(sc->val.words[0]), num(sc->val.words[1]), num(sc->val.words[2])};
            else if (!sc->val.words.empty()) {
                double k = num(sc->val.words[0]); mx.scale = {k, k, k};
            }
        }
        // world = (parent group) ∘ (instance local); then fold in the unit scale so the
        // placement lands in metres (mirrors addMesh's transform construction).
        Affine xf = parentXf.compose(mx.toAffine());
        for (double& e : xf.m) e *= L_;
        xf.t = xf.t * L_;

        int matOverride = -1;
        if (find(b, "material")) {
            matOverride = matFieldId(b, L, "instance", /*optional=*/true);
            if (matOverride < 0 && !err.empty()) return false;
        }

        MeshInstance inst;
        inst.blasId = it->second;
        inst.toWorld = xf;
        inst.toLocal = xf.inverse();
        inst.matOverride = matOverride;
        L.scene.instances.push_back(inst);
        return true;
    }

    // ---- group (transform hierarchy) ----
    // A `group { translate .. rotate .. scale .. <child prims / nested groups> }`
    // node. The group's own transform composes with its parent's (parent applied
    // last: world = parentXf ∘ localXf), and every child primitive is baked into
    // world space with that composed transform — so at render time the scene is
    // still a flat list of world-space prims (no scene graph, no per-instance
    // cost). Nested groups recurse; a light anywhere in the tree sets haveLight.
    // See known-issues.md for the deferred true-instancing (shared-geometry) path.
    bool addGroup(const Block& b, Loaded& L, const Affine& parentXf, bool& haveLight) {
        Vec3 tr{0, 0, 0}, rot{0, 0, 0}, scl{1, 1, 1};
        vec3Of(b, "translate", tr);
        vec3Of(b, "rotate", rot);
        const Stmt* sc = find(b, "scale");
        if (sc) {
            if (sc->val.words.size() >= 3)
                scl = {num(sc->val.words[0]), num(sc->val.words[1]), num(sc->val.words[2])};
            else if (!sc->val.words.empty()) {
                double k = num(sc->val.words[0]); scl = {k, k, k};
            }
        }
        // `shear a b c` (all optional, default 0) is a unit-diagonal upper-triangular
        // shear applied in the group's LOCAL frame (innermost, before scale/rotate):
        //   x' = x + a*y + b*z    y' = y + c*z    z' = z
        // so a shears X along Y, b shears X along Z, c shears Y along Z. It composes
        // as world = parent ∘ TRS ∘ Shear. Analytic spheres reject any shear (they
        // would become ellipsoids, see addSphere); meshes/quads/triangles take it.
        Vec3 shr{0, 0, 0};
        vec3Of(b, "shear", shr);
        Affine localXf = affineFromTRS(tr, rot, scl);
        if (shr.x != 0.0 || shr.y != 0.0 || shr.z != 0.0) {
            Affine sh;                 // identity, then fill the strict-upper triangle
            sh.m[1] = shr.x;           // x += a*y
            sh.m[2] = shr.y;           // x += b*z
            sh.m[5] = shr.z;           // y += c*z
            localXf = localXf.compose(sh);   // apply shear first, then scale/rotate
        }
        Affine world = parentXf.compose(localXf);
        // Child primitives are nested brace blocks; the transform-only statements
        // (translate/rotate/scale) carry no block and are skipped here.
        for (const auto& s : b.stmts) {
            const Block* cb = s.val.block.get();
            if (!cb) continue;
            s.used = true;   // a child block: dispatched below, or rejected by name
            if      (s.key == "sphere")   { if (!addSphere(*cb, L, world)) return false; }
            else if (s.key == "quad")     { if (!addQuad(*cb, L, world)) return false; }
            else if (s.key == "triangle") { if (!addTriangle(*cb, L, world)) return false; }
            else if (s.key == "mesh")     { if (!addMesh(*cb, L, world)) return false; }
            else if (s.key == "mesh_instance") { if (!addMeshInstance(*cb, L, world)) return false; }
            else if (s.key == "isosurface") { if (!addIsosurface(*cb, L, world)) return false; }
            else if (s.key == "light")    { if (!addLight(*cb, L, (cb->type == "light" ? std::string() : cb->type), world)) return false; haveLight = true; }
            else if (s.key == "group")    { if (!addGroup(*cb, L, world, haveLight)) return false; }
            else { fail("unknown block '" + s.key + "' inside group (allowed: sphere, quad, triangle, mesh, mesh_instance, isosurface, light, group)"); return false; }
        }
        return true;
    }

    // ---- isosurface / metaballs / (smooth) CSG ----
    // An `isosurface { material <m>  <one field element> }` builds an Implicit whose
    // field is a flat postfix expression (see implicit.h). A field element is either
    // a LEAF analytic SDF (sphere/box/torus/cylinder/plane) or a COMBINATOR
    // (union/intersect/difference and their smooth_* variants, plus `blob` = smooth
    // union) whose nested children are themselves field elements. Each element may
    // carry translate/rotate/scale that composes down the tree (a mini scene graph),
    // and smooth combinators take a blend radius `k` (authored length) that fillets
    // the seam — this is what makes metaballs merge and gives rounded booleans.

    // Compose the authored-space transform for a field block: parentXf ∘ TRS(this).
    Affine fieldXf(const Block& b, const Affine& parentXf) {
        Vec3 tr{0, 0, 0}, rot{0, 0, 0}; double sc = 1.0;
        vec3Of(b, "translate", tr); vec3Of(b, "rotate", rot);
        const Stmt* scs = find(b, "scale");
        if (scs && !scs->val.words.empty()) sc = num(scs->val.words[0]);
        return parentXf.compose(affineFromTRS(tr, rot, Vec3{sc, sc, sc}));
    }

    // Build one analytic-SDF leaf. `authoredXf` is the leaf's local->world transform
    // in authored units; the global unit scale L_ folds in here (world = L_·authored),
    // and the leaf's uniform scale becomes the field's distance multiplier. Params
    // (radius, half-extents, ...) stay in authored units and are rescaled at eval.
    bool addFieldLeaf(FieldOp op, const Block& b, const Affine& authoredXf,
                      std::vector<FieldNode>& out, bool ellipsoid = false) {
        // Fold the authored center into the transform, then rebase to metres. An
        // ellipsoid is a unit sphere with a non-uniform LEAF pre-scale (its radii)
        // baked into the local->world map.
        Vec3 center{0, 0, 0}; vec3Of(b, "center", center);
        Vec3 preScale{1, 1, 1};
        if (ellipsoid) vec3Of(b, "radius", preScale);   // rx, ry, rz
        Affine A = authoredXf.compose(affineFromTRS(center, Vec3{0, 0, 0}, preScale));
        Affine L2W;
        for (int k = 0; k < 9; ++k) L2W.m[k] = L_ * A.m[k];
        L2W.t = A.t * L_;
        // Conservative world-distance factor: the SMALLEST per-axis scale of the
        // local->world map (its columns are R*S, so column norms == the axis scales,
        // exactly, since we only build translate/rotate/scale — no shear). Multiplying
        // the local SDF by this underestimates true world distance, which keeps the
        // field a valid Lipschitz-1 SDF under NON-UNIFORM scale too (sphere->ellipsoid,
        // squashed box/torus/cone, ...): sphere-tracing just takes shorter steps along
        // the stretched axis, and gradient normals stay correct via the chain rule.
        double sx = std::sqrt(L2W.m[0]*L2W.m[0] + L2W.m[3]*L2W.m[3] + L2W.m[6]*L2W.m[6]);
        double sy = std::sqrt(L2W.m[1]*L2W.m[1] + L2W.m[4]*L2W.m[4] + L2W.m[7]*L2W.m[7]);
        double sz = std::sqrt(L2W.m[2]*L2W.m[2] + L2W.m[5]*L2W.m[5] + L2W.m[8]*L2W.m[8]);
        double s  = std::fmin(sx, std::fmin(sy, sz));
        FieldNode nd; nd.op = op; nd.scale = s; nd.inv = L2W.inverse();
        switch (op) {
            case FieldOp::Sphere:
                nd.p[0] = ellipsoid ? 1.0 : dblOf(b, "radius", 1.0);   // radii live in the transform
                break;
            case FieldOp::Box: {
                Vec3 size{1, 1, 1}; vec3Of(b, "size", size);
                nd.p[0] = size.x * 0.5; nd.p[1] = size.y * 0.5; nd.p[2] = size.z * 0.5;
                nd.p[3] = dblOf(b, "round", 0.0);           // corner rounding radius (0 = sharp)
                break;
            }
            case FieldOp::Torus:
                nd.p[0] = dblOf(b, "major", 1.0);
                nd.p[1] = dblOf(b, "minor", 0.25);
                break;
            case FieldOp::Cylinder:
                nd.p[0] = dblOf(b, "radius", 0.5);
                nd.p[1] = dblOf(b, "height", 1.0) * 0.5;    // half-height (axis = local y)
                break;
            case FieldOp::Cone: {
                // `radius`/`radius2` = bottom/top radii; a pure cone omits radius2 (top=0).
                nd.p[0] = dblOf(b, "radius",  0.5);         // bottom radius (y = -h)
                nd.p[1] = dblOf(b, "radius2", 0.0);         // top radius    (y = +h)
                nd.p[2] = dblOf(b, "height", 1.0) * 0.5;    // half-height (axis = local y)
                break;
            }
            case FieldOp::Plane: {
                Vec3 n{0, 1, 0}; vec3Of(b, "normal", n);
                double ln = length(n); if (ln > 0) n = n / ln;
                nd.p[0] = n.x; nd.p[1] = n.y; nd.p[2] = n.z;
                nd.p[3] = dblOf(b, "offset", 0.0);
                break;
            }
            default: break;
        }
        out.push_back(nd);
        return true;
    }

    // Build an arbitrary-expression leaf: `function { expr "f(x,y,z)" }`. The infix
    // formula (variables x y z, plus r = |p|) is compiled by the SAME shunting-yard
    // used for procedural patterns, and its postfix program is appended to the
    // isosurface's shared exprNodes pool; the FieldNode records the (offset, count)
    // slice. Unlike an analytic leaf the value is NOT a distance, so `scale` stays 1
    // (no world-distance rescale) and the isosurface must supply a container box +
    // Lipschitz bound (see addIsosurface). `authoredXf` maps the leaf-local frame the
    // expression is written in to world (metre rebase folded in).
    bool addFunctionLeaf(const Block& b, const Affine& authoredXf,
                         std::vector<FieldNode>& out, std::vector<PatNode>& exprPool) {
        const Stmt* es = find(b, "expr");
        if (!es || es->val.words.empty()) { fail("function leaf needs `expr \"f(x,y,z)\"`"); return false; }
        std::string expr;
        for (size_t k = 0; k < es->val.words.size(); ++k) { if (k) expr += " "; expr += es->val.words[k]; }
        std::vector<PatNode> prog; std::string perr;
        // `&tableScope_` (but NOT texScope_): the field evaluators are handed the scene's
        // grid:/scatter: tables, so a `function` leaf can BE a measured volume or height
        // field — `expr "grid:terrain(x, z) - y"`. There is no shading context here, so
        // `tex:` (which needs u,v from a hit) stays unavailable and fails to compile.
        if (!compilePatternExpr(expr, prog, perr, /*allowT=*/false, nullptr, &tableScope_)) {
            fail("function expr: " + perr); return false;
        }
        // CSE the compiled program: field formulas are the sphere-trace's inner loop
        // (every march step + shadow ray runs this program), and machine-generated
        // exprs repeat whole subtrees. Bit-identical by construction (see pattern.h).
        patternOptimizeCSE(prog);
        Affine L2W;
        for (int k = 0; k < 9; ++k) L2W.m[k] = L_ * authoredXf.m[k];
        L2W.t = authoredXf.t * L_;
        FieldNode nd; nd.op = FieldOp::Expr;
        nd.inv   = L2W.inverse();
        nd.scale = 1.0;
        nd.exprOff = (int)exprPool.size();
        nd.exprN   = (int)prog.size();
        for (const auto& pn : prog) exprPool.push_back(pn);
        out.push_back(nd);
        return true;
    }

    // Recursively emit a field element's postfix nodes. `parentXf` is the composed
    // authored transform of the enclosing element(s).
    bool buildFieldStmt(const Stmt& st, const Affine& parentXf, std::vector<FieldNode>& out,
                        std::vector<PatNode>& exprPool) {
        const Block* b = st.val.block.get();
        if (!b) { fail("field element '" + st.key + "' needs a { } block"); return false; }
        const std::string& k = st.key;
        // Leaves — the element's own translate/rotate/scale wrap the primitive.
        Affine xf = fieldXf(*b, parentXf);
        if (k == "function")  return addFunctionLeaf(*b, xf, out, exprPool);
        if (k == "sphere")    return addFieldLeaf(FieldOp::Sphere,   *b, xf, out);
        if (k == "ellipsoid") return addFieldLeaf(FieldOp::Sphere,   *b, xf, out, /*ellipsoid=*/true);
        if (k == "box")       return addFieldLeaf(FieldOp::Box,      *b, xf, out);
        if (k == "torus")     return addFieldLeaf(FieldOp::Torus,    *b, xf, out);
        if (k == "cylinder")  return addFieldLeaf(FieldOp::Cylinder, *b, xf, out);
        if (k == "cone")      return addFieldLeaf(FieldOp::Cone,     *b, xf, out);
        if (k == "plane")     return addFieldLeaf(FieldOp::Plane,    *b, xf, out);
        // Combinators — fold N children pairwise in postfix order.
        FieldOp op; bool smooth = false;
        if      (k == "union")             op = FieldOp::Union;
        else if (k == "intersect" || k == "intersection") op = FieldOp::Intersect;
        else if (k == "difference" || k == "subtract")    op = FieldOp::Difference;
        else if (k == "smooth_union")      { op = FieldOp::SmoothUnion;      smooth = true; }
        else if (k == "smooth_intersect" || k == "smooth_intersection") { op = FieldOp::SmoothIntersect; smooth = true; }
        else if (k == "smooth_difference" || k == "smooth_subtract")    { op = FieldOp::SmoothDifference; smooth = true; }
        else if (k == "blob")              { op = FieldOp::SmoothUnion;      smooth = true; }
        else { fail("unknown field element '" + k + "' in isosurface (leaves: sphere/"
                    "ellipsoid/box/torus/cylinder/cone/plane; combinators: union/intersect/"
                    "difference, smooth_union/smooth_intersect/smooth_difference, blob)"); return false; }
        double kBlend = 0.0;
        const Stmt* kk = find(*b, "k");
        if (kk && !kk->val.words.empty()) kBlend = num(kk->val.words[0]) * L_;   // authored -> metres
        (void)smooth;
        int nChild = 0;
        for (const auto& cs : b->stmts) {
            if (!cs.val.block) continue;              // transform-only / k stmts carry no block
            cs.used = true;   // a nested field element, validated by name on recursion
            if (!buildFieldStmt(cs, xf, out, exprPool)) return false;
            if (++nChild >= 2) { FieldNode c; c.op = op; c.p[0] = kBlend; out.push_back(c); }
        }
        if (nChild < 1) { fail("'" + k + "' needs at least one child primitive"); return false; }
        return true;
    }

    bool addIsosurface(const Block& b, Loaded& L, const Affine& parentXf = Affine::identity()) {
        int id = matFieldId(b, L, "isosurface"); if (id < 0) return false;
        // The enclosing group's transform (identity at top level) composes OUTSIDE the
        // isosurface's own translate/rotate/scale, so a settled `group { translate ..
        // rotate .. <isosurface> }` rest pose bakes into the field's local->world map.
        Affine rootXf = fieldXf(b, parentXf);
        std::vector<FieldNode> nodes;
        std::vector<PatNode> exprPool;
        int nRoot = 0;
        for (const auto& cs : b.stmts) {
            if (!cs.val.block) continue;              // skip material/translate/rotate/scale
            // The container box is read below (only the `function { }` branch requires
            // one, but it is always meaningful), not treated as a field element here.
            if (cs.key == "contained_by") { cs.used = true; continue; }
            cs.used = true;   // a field element, validated by name in buildFieldStmt
            if (!buildFieldStmt(cs, rootXf, nodes, exprPool)) return false;
            ++nRoot;
        }
        if (nRoot != 1) {
            fail("isosurface must contain exactly one root field element (a leaf or a "
                 "CSG combinator); wrap multiple shapes in a union { ... }");
            return false;
        }
        Implicit im;
        im.nodes = std::move(nodes);
        im.exprNodes = std::move(exprPool);
        im.matId = id;
        if (fieldHasExpr(im.nodes)) {
            // An arbitrary expression field has no analytic bound and its value is not
            // a signed distance, so the user must supply a container box (marched only
            // inside it) and we need a Lipschitz bound to size safe march steps.
            const Stmt* cb = find(b, "contained_by");
            if (!cb || !cb->val.block) {
                fail("an isosurface using function { } needs a `contained_by { min <x y z>  max <x y z> }` box");
                return false;
            }
            const Block& cbb = *cb->val.block;
            // Container shape. `contained_by { sphere { center x y z  radius r } }`
            // clips the ray along a smooth curved boundary (an unbounded surface's
            // unavoidable cut then reads as a rounded edge, not hard box facets);
            // otherwise a `min`/`max` axis-aligned box.
            Aabb box;
            const Stmt* sph = find(cbb, "sphere");
            if (sph && sph->val.block) {
                const Block& sb = *sph->val.block;
                Vec3 ctr{0, 0, 0}; vec3Of(sb, "center", ctr);
                double rad = dblOf(sb, "radius", 1.0);
                // World center; radius scaled by the leaf transform (exact for uniform
                // scale, a conservative bounding sphere under rotation/shear).
                Vec3 cw = rootXf.apply(ctr) * L_;
                double rx = length(rootXf.apply(ctr + Vec3{rad, 0, 0}) * L_ - cw);
                double ry = length(rootXf.apply(ctr + Vec3{0, rad, 0}) * L_ - cw);
                double rz = length(rootXf.apply(ctr + Vec3{0, 0, rad}) * L_ - cw);
                double rw = std::fmax(rx, std::fmax(ry, rz));
                im.container    = Container::Sphere;
                im.sphereCenter = cw;
                im.sphereRadius = rw;
                box.lo = cw - Vec3{rw, rw, rw};
                box.hi = cw + Vec3{rw, rw, rw};
            } else {
                Vec3 mn{-1,-1,-1}, mx{1,1,1};
                vec3Of(cbb, "min", mn);
                vec3Of(cbb, "max", mx);
                // Transform all 8 authored corners into world (metre-rebased), take the AABB.
                bool first = true;
                for (int c = 0; c < 8; ++c) {
                    Vec3 corner{(c&1)?mx.x:mn.x, (c&2)?mx.y:mn.y, (c&4)?mx.z:mn.z};
                    Vec3 w = rootXf.apply(corner) * L_;
                    if (first) { box.lo = w; box.hi = w; first = false; } else box.expand(w);
                }
                im.container = Container::Box;
                // That AABB is right for the BVH but WRONG to clip the ray to whenever
                // rootXf rotates the box: the AABB of a tilted box pokes outside the
                // authored container, and `max_gradient` only bounds the field INSIDE
                // it. Marching the excess makes the first |f|/max_gradient step
                // arbitrarily long and the sphere-trace steps straight over the object
                // (see the Container comment in implicit.h). So when the map is not a
                // positive-diagonal (axis-preserving) one, keep the box oriented and
                // clip in its own frame; the marched region is then a rigid motion of
                // the authored one, where the authored bound provably still holds.
                Affine boxL2W;
                for (int k = 0; k < 9; ++k) boxL2W.m[k] = L_ * rootXf.m[k];
                boxL2W.t = rootXf.t * L_;
                bool axisAligned = true;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        double v = boxL2W.m[i * 3 + j];
                        if (i == j) { if (v <= 0.0) axisAligned = false; }
                        else if (v != 0.0) axisAligned = false;
                    }
                im.boxOriented = !axisAligned;
                im.boxInv      = boxL2W.inverse();
                im.boxLo = mn; im.boxHi = mx;
            }
            im.bounds = box;
            // Cap policy: a container-clipped solid is sealed with a face of the
            // isosurface material by default; `open` (or `open on`) omits those caps,
            // leaving the surface's cut edge / a see-through opening. `open off`
            // forces the default. Only affects surfaces that reach the container.
            const Stmt* openS = find(b, "open");
            std::string openV = strOf(b, "open", "");
            bool isOpen = (openS != nullptr) &&
                          !(openV == "off" || openV == "false" || openV == "no" || openV == "0");
            im.capped = !isOpen;
            double mg = dblOf(b, "max_gradient", 0.0);
            // The Lipschitz probe must see the SAME field the marcher will: a
            // `grid:`-sampling formula evaluated without the tables reads 0 everywhere,
            // whose estimated slope is 0 — an unusably tiny bound. The data pass runs
            // before geometry, so the tables are already loaded here.
            const PatTables tabs = L.scene.patTables();
            // Probe the region the marcher will actually walk. For an oriented box that
            // is the box itself, not its (larger, steeper) world AABB — see the note on
            // estimateFieldLipschitz. Sphere containers already bound their own region.
            Affine boxL2W;
            for (int k = 0; k < 9; ++k) boxL2W.m[k] = L_ * rootXf.m[k];
            boxL2W.t = rootXf.t * L_;
            Aabb  probe = box;
            const Affine* probeXf = nullptr;
            if (im.boxOriented) { probe.lo = im.boxLo; probe.hi = im.boxHi; probeXf = &boxL2W; }
            im.lipschitz = (mg > 0.0) ? mg
                                      : 1.3 * estimateFieldLipschitz(im.nodes, im.exprNodes, probe,
                                                                     /*grid=*/24, &tabs, probeXf);
            double acc = dblOf(b, "accuracy", 0.0);
            // Step floor also sizes off the true container — the OBB's own world
            // diagonal, not its AABB's — so a rotated piece keeps exactly the march
            // resolution it had before a rest pose was baked onto it.
            double stepDiag = im.boxOriented ? length(boxL2W.applyDir(im.boxHi - im.boxLo))
                                             : length(box.hi - box.lo);
            im.minStep = (acc > 0.0) ? acc * L_ : implicitMinStepForDiag(stepDiag);
        } else {
            im.lipschitz = 1.0;                       // SDF leaves + smin/CSG stay unit-Lipschitz
            im.bounds = implicitBounds(im.nodes);
            im.minStep = implicitMinStep(im.bounds);
        }
        // Ray-march strategy (default `adaptive`). `sample` = fixed-step POV-Ray-style
        // marching, for fields whose Lipschitz bound can't be trusted; the fixed world
        // step comes from `samples <n>` (n intervals across the box diagonal), else from
        // `accuracy`, else a 256-sample default.
        std::string meth = strOf(b, "method", "adaptive");
        if (meth == "sample" || meth == "fixed") {
            im.method = MarchMethod::Sample;
            double diag = length(im.bounds.hi - im.bounds.lo);
            double ns   = dblOf(b, "samples", 0.0);
            double acc  = dblOf(b, "accuracy", 0.0);
            im.sampleStep = (ns > 0.0)  ? diag / ns
                          : (acc > 0.0) ? acc * L_
                                        : diag / 256.0;
        } else if (meth != "adaptive") {
            fail("isosurface `method` must be `adaptive` or `sample` (got '" + meth + "')");
            return false;
        }
        // Root refinement once a sign change is bracketed (default `bisect`).
        std::string ref = strOf(b, "refine", "bisect");
        if (ref == "regula_falsi" || ref == "falsi" || ref == "secant") im.refine = RootRefine::RegulaFalsi;
        else if (ref == "bisect") im.refine = RootRefine::Bisect;
        else { fail("isosurface `refine` must be `bisect` or `regula_falsi` (got '" + ref + "')"); return false; }
        // Procedural UV wrap for pattern/expression materials on this native surface:
        // `uv planar|spherical|cylindrical [axis=x|y|z]`. Exposes `u`,`v` to material
        // expressions using the SAME projection meshes use (geometry.h projectUV),
        // referenced to the primitive's world bounds. Default up/projection axis is y.
        const std::string uvMode = strOf(b, "uv");
        im.uvProj = parseUvProjection(uvMode);
        im.uvAxis = 1;
        if (const Stmt* uvs = find(b, "uv")) {
            for (size_t i = 1; i < uvs->val.words.size(); ++i) {
                std::string k, a;
                if (!splitEq(uvs->val.words[i], k, a) || k != "axis") continue;
                if (a == "x") im.uvAxis = 0; else if (a == "z") im.uvAxis = 2; else im.uvAxis = 1;
            }
        }
        im.uvBounds = im.bounds;
        im.uvBoundsSet = true;
        im.name = b.name;                         // authored name -> -export-mesh group name
        L.scene.implicits.push_back(std::move(im));
        if (!b.name.empty()) implicitByName_[b.name] = (int)L.scene.implicits.size() - 1;
        return true;
    }

    // ---- lights ----
    // Absolute light power. If a `light` block authored an absolute flux — `power
    // <watts>` (radiometric radiant flux) or `lumens <lm>` (photometric luminous
    // flux) — scale its SPD so the emitter's total emitted power
    // (emitIntegral*geomW) equals that flux, and flag the scene absolute so
    // writeFilm uses a fixed photographic exposure rather than per-image auto-
    // exposure. `geomW` is the emitter's geometric weight (area*PI for surface
    // emitters, spotOmega for a spot) so `power = emitIntegral*geomW` — the same law
    // finalizeEmitters() applies. Radiant flux uses the SPD integral directly;
    // luminous flux uses Phi_v = 683 * geomW * INT SPD(lambda)*V(lambda) dlambda with
    // cieY() as the CIE photopic V(lambda) (peak ~1). Both integrals use the same
    // midpoint/binWidth_ quadrature as EmissionSampler so the scaling is exact.
    // `power` wins if both are given. Returns the (possibly scaled) SPD.
    Spectrum absPower(const Block& b, Spectrum spd, double geomW, Loaded& L) {
        const Stmt* pw = find(b, "power");
        const Stmt* lm = find(b, "lumens");
        if (!pw && !lm) return spd;
        double rawInt = 0.0, vInt = 0.0;
        int n = (int)((LAMBDA_MAX - LAMBDA_MIN) / binWidth_);
        for (int i = 0; i < n; ++i) {
            double w = LAMBDA_MIN + (i + 0.5) * binWidth_;
            double s = std::max(0.0, spd(w)) * binWidth_;
            rawInt += s; vInt += s * cieY(w);
        }
        double k = 0.0;
        if (pw) {
            double watts = dblOf(b, "power", 0.0);
            double rawFlux = rawInt * geomW;               // watts of the unscaled SPD
            k = (rawFlux > 0.0) ? watts / rawFlux : 0.0;
        } else {
            double lumens = dblOf(b, "lumens", 0.0);
            double denom = 683.0 * geomW * vInt;           // lm per unit SPD scale
            k = (denom > 0.0) ? lumens / denom : 0.0;
        }
        L.scene.absolute = true;
        return [spd, k](double w) { return spd(w) * k; };
    }

    // Each `light` block registers one Emitter. Multiple light blocks accumulate;
    // the forward tracer selects among them power-weighted and the backward
    // reference sums over them (see scene.h / render.h / backward.h).
    bool addLight(const Block& b, Loaded& L, std::string subtype,
                  const Affine& xf = Affine::identity()) {
        // New unified header form `NAME = light { kind <subtype>  ... }` carries the
        // light subtype in a `kind` property rather than a bareword after the KIND.
        // When no bareword subtype was parsed (empty), fall back to the property.
        // Old default point/area lights have no `kind`, so they stay empty as before.
        if (subtype.empty()) subtype = strOf(b, "kind", "");
        // A light block's emission slot is spelled `spd`, so its pattern spellings are
        // `spd pattern:p` (the pattern IS the profile) and `spd_map pattern:p` (modulate
        // the authored SPD) — the same pair a material spells `emit` / `emit_map`. Only
        // the default rectangular quad below can carry one: it is the one light subtype
        // whose emitter samples a (u,v) that provably matches the geometry it drops into
        // the scene. Note `power`/`lumens` still normalise the UNPATTERNED SPD, so a
        // pattern that averages 0.5 emits half the requested flux (see Material::emitPat).
        int spdPat = -1;
        Spectrum spd = patternedSpectrumParam(b, "spd", "spd_map", spdPat, blackbody(6500.0));
        // Everything NOT in this list falls through to the rectangular quad at the
        // bottom (`` and `area` both spell the default), which is the only subtype that
        // can honour a pattern.
        if (spdPat >= 0 && (subtype == "collimated" || subtype == "sphere" ||
                            subtype == "cylinder" || subtype == "spot" ||
                            subtype == "env" || subtype == "sun")) {
            fail("an spd pattern is only supported on the default rectangular area light — "
                 "a '" + subtype + "' light samples positions that no surface (u,v) "
                 "corresponds to, so the pattern would be silently ignored");
            return false;
        }
        // Uniform scale of the enclosing group chain (spheres/pencils scale by it;
        // a non-uniform scale is only meaningful for the flat quad/mesh emitters).
        bool nonUniform = false; double s = xf.uniformScale(nonUniform);
        if (subtype == "collimated") {
            Vec3 dir{0, 0, -1}; vec3Of(b, "dir", dir);
            Vec3 beam = normalize(xf.applyDir(dir));
            // A thin pencil cross-section at the given origin (3 cm pencil).
            Vec3 o{0.5, 0.5, 0.95}; vec3Of(b, "origin", o);
            Vec3 t, bt; onb(beam, t, bt);
            double w = Len(0.03) * s;
            Vec3 U = t * w, V = bt * w;
            // The emitter quad is sampled corner-anchored (origin + u*[0,1] + v*[0,1]),
            // so pass a corner offset by -half(U+V): that makes `origin` (the aim point)
            // the CENTRE of the beam footprint, not a corner. For a bare 3 cm pencil the
            // offset is negligible, but a group-scaled beam (w = 0.03*scale) would
            // otherwise sit entirely on the +u/+v side of the aim point, lighting only
            // half the intended footprint (hard seam through the aim point).
            Vec3 corner = P(xf.apply(o)) - U * 0.5 - V * 0.5;
            spd = absPower(b, spd, (w * w) * PI, L);
            L.scene.addAreaLight(corner, U, V, beam, w * w, spd, binWidth_,
                                 /*collimated*/true, beam);
            return true;
        }
        if (subtype == "sphere") {
            // Spherical area light: a glowing ball. Also add an emissive sphere to
            // geometry so photons that strike it are absorbed and it is visible in
            // the photon-catch camera modes (mirrors the area-light quad below).
            if (nonUniform) { fail("sphere light under non-uniform scale would be an ellipsoid; use uniform scale"); return false; }
            Vec3 c{0.5, 0.7, 0.5}; vec3Of(b, "center", c);
            double rad = Len(dblOf(b, "radius", 0.1)) * s;
            Vec3 cw = P(xf.apply(c));
            spd = absPower(b, spd, (4.0 * PI * rad * rad) * PI, L);
            Material lm; lm.reflect = constantSpectrum(0.0); lm.emit = spd; lm.isLight = true;
            int id = (int)L.scene.mats.size(); L.scene.mats.push_back(lm);
            L.scene.spheres.push_back(Sphere{cw, rad, id});
            L.scene.addSphereLight(cw, rad, spd, binWidth_, /*matId*/id);
            return true;
        }
        if (subtype == "cylinder") {
            // Cylindrical area light: a glowing tube (fluorescent lamp). The LATERAL
            // surface emits; by default the end caps are omitted (matching the analytic
            // 2*PI*r*L sampling area and a real tube's non-emissive metal ends). With
            // `caps on` the two end discs also emit (a closed glowing capsule) -- both
            // added to the sampling area (see addCylinderLight) and tessellated as
            // emissive fans below. We tessellate the wall into emissive triangles so
            // the tube is visible and absorbs returning photons (mirrors how the sphere
            // light drops an emissive sphere into geometry). `center` is the tube
            // midpoint, `axis` its direction (default +Y), `length`/`radius` its size,
            // and `segments` (default 48) the wall tessellation fineness.
            if (nonUniform) { fail("cylinder light under non-uniform scale would be an elliptic cylinder; use uniform scale"); return false; }
            Vec3 c{0.5, 0.5, 0.5}; vec3Of(b, "center", c);
            Vec3 dir{0, 1, 0}; vec3Of(b, "axis", dir);
            double len = Len(dblOf(b, "length", 0.5)) * s;
            double rad = Len(dblOf(b, "radius", 0.05)) * s;
            int segs = (int)dblOf(b, "segments", 48.0);
            if (segs < 3) segs = 3;
            std::string capsStr = strOf(b, "caps", "off");
            bool caps = (capsStr == "on" || capsStr == "true" || capsStr == "yes");
            double cylArea = 2.0 * PI * rad * len + (caps ? 2.0 * PI * rad * rad : 0.0);
            spd = absPower(b, spd, cylArea * PI, L);
            Vec3 axisW = normalize(xf.applyDir(dir)) * len;   // world axis vector (|.| = len)
            Vec3 baseW = P(xf.apply(c)) - axisW * 0.5;        // base-cap center
            Material lm; lm.reflect = constantSpectrum(0.0); lm.emit = spd; lm.isLight = true;
            int id = (int)L.scene.mats.size(); L.scene.mats.push_back(lm);
            // Tessellate the lateral wall. onb(normalize(axisW),...) here matches the
            // basis addCylinderLight computes, so facets align with the sampled radius.
            Vec3 au = normalize(axisW); Vec3 t, bt; onb(au, t, bt);
            Vec3 topW = baseW + axisW;                          // top-cap center
            for (int i = 0; i < segs; ++i) {
                double a0 = 2.0 * PI * i / segs, a1 = 2.0 * PI * (i + 1) / segs;
                Vec3 r0 = t * std::cos(a0) + bt * std::sin(a0);
                Vec3 r1 = t * std::cos(a1) + bt * std::sin(a1);
                Vec3 b0 = baseW + r0 * rad, b1 = baseW + r1 * rad;
                Vec3 p0 = b0 + axisW, p1 = b1 + axisW;
                L.scene.tris.push_back(Tri{b0, b1, p1, id, -1, {}});   // outward winding
                L.scene.tris.push_back(Tri{b0, p1, p0, id, -1, {}});
                if (caps) {
                    // Emissive end-disc fans: base normal -au (winding b1,b0,center),
                    // top normal +au (winding p0,p1,center).
                    L.scene.tris.push_back(Tri{b1, b0, baseW, id, -1, {}});
                    L.scene.tris.push_back(Tri{p0, p1, topW, id, -1, {}});
                }
            }
            L.scene.addCylinderLight(baseW, axisW, rad, spd, binWidth_, /*matId*/id, caps);
            return true;
        }
        if (subtype == "spot") {
            // Point spotlight: a cone about `dir`, smoothstep penumbra between the
            // inner and outer half-angles (degrees). No emissive geometry (a point).
            Vec3 o{0.5, 0.99, 0.5}; vec3Of(b, "origin", o);
            Vec3 dir{0, -1, 0}; vec3Of(b, "dir", dir);
            double inner = dblOf(b, "inner_angle", 20.0);
            double outer = dblOf(b, "outer_angle", 30.0);
            if (outer < inner) outer = inner;            // outer cone must enclose inner
            const double d2r = PI / 180.0;
            double cosInner = std::cos(inner * d2r), cosOuter = std::cos(outer * d2r);
            spd = absPower(b, spd, PI * (2.0 - cosInner - cosOuter), L);
            L.scene.addSpotLight(P(xf.apply(o)), normalize(xf.applyDir(dir)), cosInner, cosOuter, spd, binWidth_);
            return true;
        }
        if (subtype == "sun") {
            // Distant directional sun: an infinitely-far disc of angular radius
            // `angle`/2 about `dir`, so every point of the scene sees it in the same
            // direction at the same radiance. Aim it either with `dir` (pointing TO the
            // sun, like the sky block's `sun_dir`) or with `elevation`/`azimuth` in
            // degrees. `angle` is the full angular DIAMETER in degrees (default 0.53,
            // the real sun); widening it only softens shadows, because `spd` is the
            // perpendicular IRRADIANCE, which is what fixes the exposure.
            Vec3 dir{0.3, 0.6, 0.2};
            if (!vec3Of(b, "dir", dir)) {
                double el = dblOf(b, "elevation", 45.0) * PI / 180.0;
                double az = dblOf(b, "azimuth", 0.0) * PI / 180.0;
                dir = Vec3{std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az)};
            }
            if (find(b, "power") || find(b, "lumens")) {
                fail("sun light: absolute `power`/`lumens` is not supported (a distant "
                     "light's flux depends on the scene's cross-section); use `spd` "
                     "(perpendicular irradiance) or `intensity` instead");
                return false;
            }
            double halfAng = 0.5 * dblOf(b, "angle", 0.53) * PI / 180.0;
            if (halfAng <= 0.0) {
                fail("sun light: `angle` (angular diameter, degrees) must be > 0"); return false;
            }
            if (halfAng >= PI * 0.5) {
                fail("sun light: `angle` must be under 180 degrees (it is a cone, not a "
                     "whole sphere — use an `env` light for that)"); return false;
            }
            double inten = dblOf(b, "intensity", 1.0);
            Spectrum irr = (inten == 1.0) ? spd
                                          : Spectrum([spd, inten](double w) { return spd(w) * inten; });
            L.scene.addSunLight(normalize(xf.applyDir(dir)), halfAng, irr, binWidth_);
            return true;
        }
        if (subtype == "env") {
            // Environment light. With a `file` it is an image-based (lat-long) env:
            // each texel is upsampled to a physical emission spectrum and directions
            // are importance-sampled from the map's luminance. `rotate` spins the map
            // about the vertical axis (degrees); `intensity` scales its brightness.
            // Without a `file` it is a constant env: uniform radiance `spd` from every
            // direction. Either way it is sized by the scene bounds in Scene::build().
            if (find(b, "power") || find(b, "lumens")) {
                fail("env light: absolute `power`/`lumens` is not supported (the env's "
                     "phase-space weight depends on scene bounds); use `intensity` or "
                     "scale the `spd` instead"); return false;
            }
            // Analytic physical sky (Preetham): `sky preetham` / `kind preetham`, or
            // simply the presence of `turbidity` / `sun_dir` / `sun_elevation`. Bakes
            // an equirectangular Preetham daylight sky (with a spectrally attenuated
            // solar disk) into an EnvMap, so it lights the scene exactly like an HDRI.
            std::string kind = strOf(b, "kind"); if (kind.empty()) kind = strOf(b, "sky");
            bool isSky = (kind == "preetham" || kind == "sky") ||
                         find(b, "turbidity") || find(b, "sun_dir") || find(b, "sun_elevation");
            if (isSky) {
                Vec3 sunDir{0.3, 0.6, 0.2};
                if (!vec3Of(b, "sun_dir", sunDir)) {
                    // Elevation (deg above horizon) + azimuth (deg from +x toward +z).
                    double el = dblOf(b, "sun_elevation", 45.0) * PI / 180.0;
                    double az = dblOf(b, "sun_azimuth", 0.0) * PI / 180.0;
                    sunDir = Vec3{std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az)};
                }
                double turb = dblOf(b, "turbidity", 2.5);
                double gAlb = dblOf(b, "ground_albedo", 0.3);
                double inten = dblOf(b, "intensity", 1.0);
                int res = (int)dblOf(b, "res", 1024.0);
                if (res < 16) res = 16; if (res > 8192) res = 8192;
                int sw = res, sh = res / 2;
                // `sun_disk on` (default) bakes the solar disk into the map, as before.
                // `sun_disk off` leaves the map as pure skylight; `sun_disk separate`
                // ALSO registers an equivalent first-class `sun` emitter, which is the
                // fast-converging form: the ~10^5x brighter sun is then sampled
                // directly instead of having to be found inside 6.8e-5 sr of texels.
                std::string diskStr = strOf(b, "sun_disk", "on");
                bool diskOn = (diskStr == "on" || diskStr == "true" || diskStr == "yes" ||
                               diskStr == "baked");
                bool diskSep = (diskStr == "separate" || diskStr == "light" ||
                                diskStr == "emitter");
                bool diskOff = (diskStr == "off" || diskStr == "false" || diskStr == "no" ||
                                diskStr == "none");
                if (!diskOn && !diskSep && !diskOff) {
                    fail("env sky: `sun_disk` must be one of on / off / separate (got '" +
                         diskStr + "')"); return false;
                }
                sky::SunDisk disk;
                std::vector<Vec3> img = sky::generatePreethamSky(sw, sh, sunDir, turb, gAlb,
                                                                 inten, diskOn, &disk);
                auto map = std::make_shared<EnvMap>();
                std::string eerr;
                if (!map->buildFromRgb(img, sw, sh, dblOf(b, "rotate", 0.0), 1.0, eerr)) {
                    fail("env sky: " + eerr); return false;
                }
                L.scene.addEnvLight(std::move(map), binWidth_);
                if (diskSep && disk.present) {
                    // The map was rotated about +y by `rotate`; rotate the sun with it so
                    // the separate emitter still lands where the sky says it does.
                    double rot = dblOf(b, "rotate", 0.0) * PI / 180.0;
                    double cs = std::cos(rot), sn = std::sin(rot);
                    Vec3 d = disk.dir;
                    Vec3 dRot{d.x * cs - d.z * sn, d.y, d.x * sn + d.z * cs};
                    L.scene.addSunLight(dRot, disk.halfAngle, disk.irradiance, binWidth_);
                }
                return true;
            }
            std::string file = strOf(b, "file");
            if (!file.empty()) {
                auto map = std::make_shared<EnvMap>();
                std::string eerr;
                if (!map->load(file, dblOf(b, "rotate", 0.0), dblOf(b, "intensity", 1.0), eerr)) {
                    fail("env light: " + eerr); return false;
                }
                L.scene.addEnvLight(std::move(map), binWidth_);
                return true;
            }
            L.scene.addEnvLight(spd, binWidth_);
            return true;
        }
        // Default: rectangular area light. Also add the emissive quad to geometry so
        // photons landing back on it are absorbed (matches buildCornell).
        Vec3 o{0, 1, 0}, u{1, 0, 0}, v{0, 0, 1}, nrm{0, -1, 0};
        vec3Of(b, "origin", o); vec3Of(b, "u", u); vec3Of(b, "v", v);
        if (!vec3Of(b, "normal", nrm)) nrm = normalize(cross(u, v));
        // Transform origin as a point and the u/v edge vectors as directions, then
        // fold in the unit scale. The emitter area is recomputed from the actual
        // transformed edges (exact for any affine). The emission normal is the
        // authored normal carried by the direction map (exact for rotation +
        // uniform scale; see known-issues.md for the non-uniform-scale caveat).
        Vec3 os = P(xf.apply(o)), us = xf.applyDir(u) * L_, vs = xf.applyDir(v) * L_;
        Vec3 nw = normalize(xf.applyDir(nrm));
        spd = absPower(b, spd, length(cross(us, vs)) * PI, L);
        Material lm; lm.reflect = constantSpectrum(0.0); lm.emit = spd; lm.isLight = true;
        lm.emitPat = spdPat;
        int id = (int)L.scene.mats.size(); L.scene.mats.push_back(lm);
        Vec3 a = os, bb = os + us, cc = os + us + vs, dd = os + vs;
        // UVs must equal the emitter's own (u,v) parameterisation — Emitter::samplePoint
        // returns the bilinear (u1,u2) of `origin + us*u1 + vs*u2`, so corner a is (0,0),
        // bb is (1,0), cc is (1,1) and dd is (0,1). The first tri's defaults already say
        // that; the second's did NOT (it inherited (0,0),(1,0),(1,1) for corners a,cc,dd),
        // which put a seam down the diagonal of any UV-driven emission pattern — and,
        // before this change, of any texture applied to an area light's quad.
        L.scene.tris.push_back(Tri{a, bb, cc, id, -1, {}});
        L.scene.tris.push_back(Tri{a, cc, dd, id, -1, {},
                                   Vec3{0, 0, 0}, Vec3{1, 1, 0}, Vec3{0, 1, 0}});
        L.scene.addAreaLight(os, us, vs, nw, length(cross(us, vs)), spd, binWidth_,
                             /*collimated*/false, /*beamDir*/{1, 0, 0}, /*matId*/id);
        return true;
    }

    // ---- medium ----
    // Each `medium { }` block appends one independent region to Scene::media. Several
    // may be authored (overlapping or disjoint boxes/spheres/heterogeneous blobs) and
    // the forward tracer superposes them (extinction adds). Backward/BDPT modes use
    // only the first as a global homogeneous haze (see Scene::backwardMedium()).
    bool addMedium(const Block& b, Loaded& L) {
        Medium med;
        med.enabled = true;
        med.g = dblOf(b, "g", 0.0);
        bool rayleigh = strOf(b, "rayleigh") == "true" || strOf(b, "rayleigh") == "1";
        // Extinction coefficients are per-length (1/authored-unit); divide by L_ to
        // convert to the internal 1/metre so fog reads the same regardless of unit.
        const double invL = 1.0 / L_;
        const Stmt* sa = find(b, "sigma_a");
        const Stmt* ss = find(b, "sigma_s");
        if (sa || ss) {
            Spectrum a = sa ? evalSpectrum(sa->val) : constantSpectrum(0.0);
            Spectrum s = ss ? evalSpectrum(ss->val) : constantSpectrum(0.0);
            med.sigma_a = [a, invL](double w) { return a(w) * invL; };
            med.sigma_s = [s, invL](double w) { return s(w) * invL; };
        } else {
            double sigmaT = dblOf(b, "sigma_t", 0.0) * invL;
            double albedo = dblOf(b, "albedo", 0.9);
            double s_s = albedo * sigmaT, s_a = (1.0 - albedo) * sigmaT;
            if (rayleigh) {
                med.sigma_s = [s_s](double w) { double r = 550.0 / w; double r2 = r * r; return s_s * r2 * r2; };
                med.sigma_a = constantSpectrum(s_a);
            } else {
                med.sigma_s = constantSpectrum(s_s);
                med.sigma_a = constantSpectrum(s_a);
            }
        }

        // ---- Optional spatial bound (localized / per-object fog) -----------------
        // `bounds { min <x y z>  max <x y z> }` confines the fog to an AABB, while
        // `bounds { center <x y z>  radius <r> }` confines it to a SPHERE — e.g. the
        // whole inside of a glass sphere: author the same center/radius as the sphere
        // geometry and the fog fills exactly that region. (`contained_by` is an alias.)
        // Authored positions/radii are unit-scaled to metres. A photon's fog interaction
        // is clipped to the ray's overlap with the region.
        const Stmt* bd = find(b, "bounds");
        if (!bd) bd = find(b, "contained_by");
        if (bd && bd->val.block) {
            const Block& bb = *bd->val.block;
            // `bounds { object "name" }` shapes the fog to a NAMED scene object:
            //   • sphere     -> exact analytic sphere bound (center/radius)
            //   • isosurface -> field membership (fog fills the field's interior,
            //                   carved per-point by fieldEval inside the field AABB)
            //   • mesh       -> TRUE containment: the mesh's triangles are solid-voxelized
            //                   into an occupancy lattice (meshvoxel.h) and membership is a
            //                   trilinear sample of it, so the fog takes the imported
            //                   silhouette. `voxels <n>` sets the longest-axis resolution.
            if (const std::string onm = strOf(bb, "object"); !onm.empty()) {
                // `feather` softens a baked occupancy LATTICE, which only the mesh branch
                // produces — a sphere and an isosurface are carved analytically per point and
                // have no lattice to erode. Say that, rather than letting the generic
                // unknown-key warning imply the keyword does not exist.
                const bool isMeshBound = (sphereByName_.find(onm) == sphereByName_.end())
                                      && (implicitByName_.find(onm) == implicitByName_.end());
                if (!isMeshBound && find(bb, "feather")) {
                    fail("medium `bounds { object \"" + onm + "\" feather … }`: `feather` "
                         "applies only to a MESH bound (it softens the voxelized occupancy "
                         "lattice). A sphere or isosurface bound is carved analytically — "
                         "shape its edge with the `density` field instead.");
                    return false;
                }
                if (auto sit = sphereByName_.find(onm); sit != sphereByName_.end()) {
                    const NamedSphere& ns = sit->second;
                    med.bounded = true;
                    med.boundShape = MediumBound::Sphere;
                    med.bcenter = ns.center;
                    med.bradius = ns.radius;
                    med.bmin = ns.center - Vec3{ns.radius, ns.radius, ns.radius};
                    med.bmax = ns.center + Vec3{ns.radius, ns.radius, ns.radius};
                } else if (auto iit = implicitByName_.find(onm); iit != implicitByName_.end()) {
                    const Implicit& im = L.scene.implicits[iit->second];
                    med.bounded = true;
                    med.boundShape = MediumBound::Implicit;
                    med.boundField = im.nodes;         // world-space field program
                    med.boundFieldExpr = im.exprNodes; // shared expression pool
                    med.bmin = im.bounds.lo;
                    med.bmax = im.bounds.hi;
                    // Inside-sign auto-detect: SDF/CSG fields are negative inside, so a
                    // point deep in the AABB (its center) reads f<0 => inside == (f<0).
                    Vec3 ctr = (im.bounds.lo + im.bounds.hi) * 0.5;
                    const PatTables btabs = L.scene.patTables();   // field may sample grid:/scatter:
                    med.boundInsideNeg = (im.eval(ctr, &btabs) <= 0.0);
                } else if (auto mit = meshAabbByName_.find(onm); mit != meshAabbByName_.end()) {
                    // TRUE mesh containment. Find the triangle range this named block
                    // appended and bake it to an occupancy lattice; the AABB is kept only
                    // as the tracking bound (delta/ratio tracking still clips to a box,
                    // the lattice then carves the silhouette out of it).
                    const MeshGroup* grp = nullptr;
                    for (const MeshGroup& mg : L.scene.meshGroups)
                        if (mg.name == onm) { grp = &mg; break; }
                    if (!grp || grp->triCount == 0) {
                        // An INSTANCED mesh (mesh_asset -> blasList) keeps its triangles in
                        // object space behind a BLAS rather than in Scene::tris, so there is
                        // no world-space range here to voxelize. Say which case this is.
                        fail("medium `bounds { object \"" + onm + "\" }`: that mesh has no "
                             "world triangles to voxelize" +
                             std::string((grp && grp->blasId >= 0)
                                 ? " (it is an INSTANCED mesh_asset; give the fog a "
                                   "non-instanced `mesh` block to shape it to)"
                                 : ""));
                        return false;
                    }
                    int vres = (int)dblOf(bb, "voxels", 160.0);
                    if (vres < 8 || vres > 1024) {
                        fail("medium `bounds { object \"" + onm + "\" voxels N }`: N must be "
                             "8..1024");
                        return false;
                    }
                    VdbGrid occ = meshvox::voxelizeSolid(L.scene.tris.data(), grp->triStart,
                                                         grp->triCount, vres);
                    if (occ.empty()) {
                        fail("medium `bounds { object \"" + onm + "\" }`: voxelization "
                             "produced an empty lattice (degenerate mesh?)");
                        return false;
                    }
                    const double frac = meshvox::solidFraction(occ);
                    std::fprintf(stderr,
                        "[medium] mesh bound \"%s\": %d x %d x %d lattice, %.1f%% solid "
                        "(%zu tris)\n", onm.c_str(), occ.nx, occ.ny, occ.nz, frac * 100.0,
                        grp->triCount);
                    // `feather <metres>` softens the silhouette: density ramps from 0 at the
                    // mesh surface to full only that far INSIDE it, instead of the one-voxel
                    // trilinear step the raw occupancy gives. Authored in world units so it
                    // is independent of `voxels`; converted to voxels here because that is
                    // what the distance transform measures in.
                    const double feath = dblOf(bb, "feather", 0.0);
                    if (feath < 0.0) {
                        fail("medium `bounds { object \"" + onm + "\" feather D }`: D is a "
                             "distance in metres and cannot be negative");
                        return false;
                    }
                    if (feath > 0.0) {
                        // Voxel edge = extent along the longest axis / that axis' voxel count.
                        const Vec3 ex = occ.wmax - occ.wmin;
                        const int nmx = std::max(occ.nx, std::max(occ.ny, occ.nz));
                        const double vh = std::max({ex.x, ex.y, ex.z}) / std::max(1, nmx - 1);
                        const double fv = (vh > 0.0) ? feath / vh : 0.0;
                        if (fv < 1.0)
                            std::fprintf(stderr,
                                "[medium] NOTE: mesh bound \"%s\" feather %.3g m is %.2f voxel(s) "
                                "at `voxels %d` — below one voxel it cannot resolve a ramp; "
                                "raise `voxels` or `feather`.\n", onm.c_str(), feath, fv, vres);
                        meshvox::featherGrid(occ, fv);
                        std::fprintf(stderr,
                            "[medium] mesh bound \"%s\": feathered %.3g m (%.1f voxels) inward\n",
                            onm.c_str(), feath, fv);
                    }
                    // A mesh that voxelizes to nothing would silently render as no fog at
                    // all, which reads as "the medium block did nothing" rather than as a
                    // geometry problem. Say so instead of leaving the author guessing.
                    if (frac < 1e-4)
                        std::fprintf(stderr,
                            "[medium] WARNING: mesh bound \"%s\" is essentially empty — the "
                            "mesh may be open (not a closed solid) or too thin for `voxels "
                            "%d`; the fog will be invisible.\n", onm.c_str(), vres);
                    med.bounded = true;
                    med.boundShape = MediumBound::Mesh;
                    med.boundGrid = std::make_shared<VdbGrid>(std::move(occ));
                    med.bmin = med.boundGrid->wmin;
                    med.bmax = med.boundGrid->wmax;
                } else {
                    fail("medium `bounds { object \"" + onm + "\" }` names no sphere, "
                         "isosurface, or mesh (objects must have a \"name\")");
                    return false;
                }
            } else if (find(bb, "center") || find(bb, "radius")) {  // sphere-shaped region
                Vec3 ctr{0, 0, 0};
                vec3Of(bb, "center", ctr);
                double rad = Len(dblOf(bb, "radius", 0.0));
                ctr = P(ctr);
                if (rad <= 0.0) { fail("medium `bounds { center .. radius .. }` needs a positive radius"); return false; }
                med.bounded = true;
                med.boundShape = MediumBound::Sphere;
                med.bcenter = ctr;
                med.bradius = rad;
                med.bmin = ctr - Vec3{rad, rad, rad};  // AABB for the majorant grid
                med.bmax = ctr + Vec3{rad, rad, rad};
            } else {                                              // axis-aligned box region
                Vec3 mn{0, 0, 0}, mx{0, 0, 0};
                vec3Of(bb, "min", mn);
                vec3Of(bb, "max", mx);
                mn = P(mn); mx = P(mx);
                for (int a = 0; a < 3; ++a) if ((&mn.x)[a] > (&mx.x)[a]) std::swap((&mn.x)[a], (&mx.x)[a]);
                med.bounded = true;
                med.boundShape = MediumBound::Box;
                med.bmin = mn;
                med.bmax = mx;
            }
        }

        // ---- Optional heterogeneous density field --------------------------------
        // `density pattern:<name>` (a named pattern) or `density "<expr>"` (inline
        // infix formula over world x y z r, §6.1). Multiplies sigma_a/sigma_s per
        // point (>= 0) so the fog forms blobs with soft, formula-defined boundaries.
        if (const Stmt* ds = find(b, "density")) {
            if (!ds->val.words.empty()) {
                const std::string& w0 = ds->val.words[0];
                // `density vdb:"cloud.nvdb"` (or `vdb:cloud.nvdb`) — import a real
                // NanoVDB FloatGrid as the density field. Baked to a dense grid on
                // load; the grid's world AABB seeds the medium bound and its peak
                // value the delta-tracking majorant.
                if (w0 == "vdb:" || w0.rfind("vdb:", 0) == 0) {
                    std::string path = (w0 == "vdb:")
                        ? (ds->val.words.size() > 1 ? ds->val.words[1] : std::string())
                        : w0.substr(4);
                    if (path.empty()) { fail("medium `density vdb:` needs a file path"); return false; }
                    auto grid = std::make_shared<VdbGrid>();
                    std::string verr;
                    if (!loadVdbGrid(path, *grid, verr)) {
                        fail("medium density: " + verr); return false;
                    }
                    med.vdb = grid;
                    std::fprintf(stderr,
                        "[vdb] density '%s': %dx%dx%d, world AABB [%.3g %.3g %.3g]..[%.3g %.3g %.3g], peak %.4g\n",
                        path.c_str(), grid->nx, grid->ny, grid->nz,
                        grid->wmin.x, grid->wmin.y, grid->wmin.z,
                        grid->wmax.x, grid->wmax.y, grid->wmax.z, (double)grid->maxVal);
                    // Seed the bound from the grid's world AABB unless one is authored.
                    if (!med.bounded) {
                        med.bounded = true;
                        med.boundShape = MediumBound::Box;
                        med.bmin = grid->wmin;
                        med.bmax = grid->wmax;
                    }
                    double dmax = dblOf(b, "density_max", 0.0);
                    med.densityMax = (dmax > 0.0) ? dmax
                                                  : std::max(1e-6, (double)grid->maxVal * 1.05);
                    // NOTE: fall through (no early return) so a `temperature vdb:` /
                    // `emission` block below can add volumetric blackbody emission to
                    // this same imported volume (fire).
                } else {
                std::vector<PatNode> prog;
                if (w0.rfind("pattern:", 0) == 0) {
                    std::string nm = w0.substr(8);
                    auto it = patternIndex_.find(nm);
                    if (it == patternIndex_.end()) {
                        fail("medium density references unknown pattern '" + nm + "'"); return false;
                    }
                    prog = L.scene.patterns[it->second].nodes;
                } else {
                    std::string expr;
                    for (size_t k = 0; k < ds->val.words.size(); ++k) { if (k) expr += " "; expr += ds->val.words[k]; }
                    std::string perr;
                    // `&tableScope_`, no texScope_: densityAt is handed the scene's tables
                    // (see Medium::densityAt), so `density "grid:rho(x, y, z)"` samples a
                    // measured volume without going through `vdb:`; there is no hit here,
                    // so `tex:` remains a compile error.
                    if (!compilePatternExpr(expr, prog, perr, /*allowT=*/false, nullptr, &tableScope_)) {
                        fail("medium density: " + perr); return false;
                    }
                }
                med.density = std::move(prog);

                // Majorant for delta/ratio tracking: explicit `density_max`, else a
                // grid estimate over the bound (×1.3 safety), mirroring isosurface's
                // `max_gradient`. A heterogeneous medium needs a finite region to
                // estimate over, so require either `bounds` or an explicit `density_max`.
                double dmax = dblOf(b, "density_max", 0.0);
                if (dmax <= 0.0) {
                    if (!med.bounded) {
                        fail("a `medium` with a `density` field needs `bounds { min .. max .. }` "
                             "or an explicit `density_max <v>` (the delta-tracking majorant)");
                        return false;
                    }
                    const Vec3& lo = med.bmin;
                    const Vec3& hi = med.bmax;
                    const int NS = 24;
                    double peak = 0.0;
                    const PatTables tabs = L.scene.patTables();
                    for (int iz = 0; iz <= NS; ++iz)
                    for (int iy = 0; iy <= NS; ++iy)
                    for (int ix = 0; ix <= NS; ++ix) {
                        Vec3 p{ lo.x + (hi.x - lo.x) * ix / NS,
                                lo.y + (hi.y - lo.y) * iy / NS,
                                lo.z + (hi.z - lo.z) * iz / NS };
                        // Real tables: the majorant must be estimated from the SAME field
                        // the renderer samples, or a `grid:`-driven density would majorise
                        // to 0 and the medium would vanish. The data pass runs before this
                        // one, so L.scene.grids/scatters/dataPool are already populated.
                        // densityFieldAt, not densityAt: the membership carve of an
                        // implicit/mesh bound only multiplies by 0 or 1, so the uncarved
                        // field peak is the conservative majorant, and probing the carved
                        // one on a 25^3 grid can miss a thin shape and majorise to 0.
                        peak = std::max(peak, med.densityFieldAt(p, &tabs));
                    }
                    dmax = 1.3 * peak;
                }
                med.densityMax = (dmax > 0.0) ? dmax : 1.0;
                }   // end else (formula density)
            }
        }

        // ---- Optional volumetric blackbody EMISSION (fire) -----------------------
        // `temperature vdb:"fire.vdb"` (or `vdb:fire.vdb`) imports a second grid
        // giving per-voxel temperature in Kelvin; combined with `emission blackbody`
        // it makes the medium a self-illuminating volumetric emitter (hot voxels
        // glow, Planck-shaped by their temperature). `emission_scale <v>` scales the
        // glow (default 1). A fire `.vdb` typically carries both a "density" and a
        // "temperature" float grid, so the usual authoring is:
        //     medium { density vdb:fire.vdb  temperature vdb:fire.vdb
        //              emission blackbody  emission_scale 2.0 }
        // where each `vdb:` selects the like-named grid out of the multi-grid file.
        if (const Stmt* ts = find(b, "temperature")) {
            if (ts->val.words.empty()) { fail("medium `temperature` needs a `vdb:<file>`"); return false; }
            const std::string& w0 = ts->val.words[0];
            if (!(w0 == "vdb:" || w0.rfind("vdb:", 0) == 0)) {
                fail("medium `temperature` only supports an imported grid: `temperature vdb:<file>`");
                return false;
            }
            std::string path = (w0 == "vdb:")
                ? (ts->val.words.size() > 1 ? ts->val.words[1] : std::string())
                : w0.substr(4);
            if (path.empty()) { fail("medium `temperature vdb:` needs a file path"); return false; }
            auto tgrid = std::make_shared<VdbGrid>();
            std::string verr;
            // Select the "temperature" grid by name out of the (possibly multi-grid) file.
            if (!loadVdbGrid(path, *tgrid, verr, "temperature")) {
                fail("medium temperature: " + verr); return false;
            }
            med.temperature = tgrid;
            med.tempPeak = std::max(1e-6, (double)tgrid->maxVal);
            std::fprintf(stderr,
                "[vdb] temperature '%s': %dx%dx%d, world AABB [%.3g %.3g %.3g]..[%.3g %.3g %.3g], peak %.4gK\n",
                path.c_str(), tgrid->nx, tgrid->ny, tgrid->nz,
                tgrid->wmin.x, tgrid->wmin.y, tgrid->wmin.z,
                tgrid->wmax.x, tgrid->wmax.y, tgrid->wmax.z, (double)tgrid->maxVal);
            // If nothing else seeded the medium's bound, use the temperature grid's AABB.
            if (!med.bounded) {
                med.bounded = true;
                med.boundShape = MediumBound::Box;
                med.bmin = tgrid->wmin;
                med.bmax = tgrid->wmax;
            }
        }
        // `emission blackbody` (the only model today) turns emission on; it is also
        // implied whenever a `temperature` grid is present. `emission_scale` tunes it.
        if (const Stmt* es = find(b, "emission")) {
            std::string kind = es->val.words.empty() ? std::string("blackbody") : es->val.words[0];
            if (kind != "blackbody") {
                fail("medium `emission " + kind + "` is not a known emission model (use `blackbody`)");
                return false;
            }
            if (!med.temperature) {
                fail("medium `emission blackbody` needs a `temperature vdb:<file>` grid to emit from");
                return false;
            }
        }
        med.emissionScale = dblOf(b, "emission_scale", med.emissionScale);
        med.emitKelvin    = dblOf(b, "emission_kelvin", med.emitKelvin);

        // ---- Optional gradient-index (GRIN) refractive field n(x,y,z) ------------
        // `ior pattern:<name>` (a named pattern) or `ior "<expr>"` (inline infix
        // formula over world x y z r, §6.1) — the local refractive index. When set,
        // rays bend through the region (Eikonal march) instead of going straight.
        // A GRIN region must be bounded (the march needs a finite region to enter),
        // and the march step is `ior_step <v>` world units (default: 1/64 of the
        // smallest bound extent). EXPERIMENTAL: CPU backward tracer only for now.
        if (const Stmt* is = find(b, "ior")) {
            if (!med.bounded) {
                fail("a `medium` with an `ior` (gradient-index) field needs `bounds { .. }` "
                     "so the ray-bending march has a finite region to enter");
                return false;
            }
            std::vector<PatNode> prog;
            if (!is->val.words.empty() && is->val.words[0].rfind("pattern:", 0) == 0) {
                std::string nm = is->val.words[0].substr(8);
                auto it = patternIndex_.find(nm);
                if (it == patternIndex_.end()) {
                    fail("medium ior references unknown pattern '" + nm + "'"); return false;
                }
                prog = L.scene.patterns[it->second].nodes;
            } else {
                std::string expr;
                for (size_t k = 0; k < is->val.words.size(); ++k) { if (k) expr += " "; expr += is->val.words[k]; }
                std::string perr;
                // `&tableScope_`, no texScope_: the Eikonal marcher hands nAt/gradNAt the
                // scene's tables, so a MEASURED index volume can bend rays —
                // `ior "1 + grid:n(x, y, z)"` (an atmospheric profile, a GRIN lens blank).
                if (!compilePatternExpr(expr, prog, perr, /*allowT=*/false, nullptr, &tableScope_)) {
                    fail("medium ior: " + perr); return false;
                }
            }
            med.ior = std::move(prog);
            // March step: explicit `ior_step`, else 1/64 of the smallest bound extent.
            double step = dblOf(b, "ior_step", 0.0);
            if (step <= 0.0) {
                double ext;
                if (med.boundShape == MediumBound::Sphere) ext = 2.0 * med.bradius;
                else ext = std::min(med.bmax.x - med.bmin.x,
                             std::min(med.bmax.y - med.bmin.y, med.bmax.z - med.bmin.z));
                step = (ext > 0.0) ? ext / 64.0 : 0.01;
            }
            med.iorStep = step;
        }

        // ---- Optional angular phase model (HG lobe vs. spectral rainbow) ---------
        // Default (no `phase` statement, or `phase hg`) keeps the smooth single-`g`
        // Henyey-Greenstein lobe (`med.g` above). `phase rainbow { .. }` swaps in the
        // physically-tabulated Airy water-droplet phase (rainbow.h) so a fog/haze
        // actually shows a primary + secondary bow, dispersion, Alexander's dark band
        // and supernumeraries. Its physical features are ON BY DEFAULT; the block
        // knobs are overrides (turn a feature off, or retune it):
        //   droplet_um <r>       droplet radius in microns (default 500 = 0.5mm rain;
        //                        ~10 -> a broad desaturated fogbow).
        //   secondary on|off     the p=3 secondary bow (default on).
        //   supernumerary on|off the Airy side-maxima / supernumerary arcs (default on).
        //   strength <s>         relative weight of the bows over the forward haze (default 1).
        //   forward_g <g>        HG anisotropy of the smooth forward-scatter background (default 0.55).
        //   secondary_ratio <v>  secondary brightness vs. primary (default 0.43).
        // The droplet index n(lambda) defaults to water's Cauchy fit; if this medium
        // also carries a scalar `ior` it does NOT feed the droplet optics (the GRIN
        // `ior` field is a spatial bend, unrelated to per-droplet dispersion).
        if (const Stmt* ph = find(b, "phase")) {
            // `phase rainbow { .. }` — the subtype bareword before `{` is consumed as the
            // nested block's TYPE (parseValue, ~line 203), NOT left in val.words. So read
            // the kind from val.words[0] (the block-less forms `phase hg` / `phase rainbow`)
            // and fall back to the block's type when a `{ .. }` body is present.
            std::string kind = ph->val.words.empty() ? std::string() : ph->val.words[0];
            if (kind.empty() && ph->val.block && ph->val.block->type != "phase")
                kind = ph->val.block->type;
            auto truthy = [](const std::string& s) {
                return s == "on" || s == "true" || s == "1" || s == "yes";
            };
            auto falsy = [](const std::string& s) {
                return s == "off" || s == "false" || s == "0" || s == "no";
            };
            if (kind == "rainbow") {
                rainbow::Params prm;
                const Block* pb = ph->val.block.get();
                if (pb) {
                    double dropUm = dblOf(*pb, "droplet_um", prm.dropletRadius_m * 1e6);
                    if (dropUm <= 0.0) { fail("medium `phase rainbow` needs a positive `droplet_um`"); return false; }
                    prm.dropletRadius_m = dropUm * 1e-6;
                    prm.rainbowStrength = dblOf(*pb, "strength", prm.rainbowStrength);
                    prm.gForward        = dblOf(*pb, "forward_g", prm.gForward);
                    prm.secondaryRatio  = dblOf(*pb, "secondary_ratio", prm.secondaryRatio);
                    if (const Stmt* s = find(*pb, "secondary")) {
                        std::string v = s->val.words.empty() ? "on" : s->val.words[0];
                        if (falsy(v)) prm.secondary = false; else if (truthy(v)) prm.secondary = true;
                    }
                    if (const Stmt* s = find(*pb, "supernumerary")) {
                        std::string v = s->val.words.empty() ? "on" : s->val.words[0];
                        if (falsy(v)) prm.supernumerary = false; else if (truthy(v)) prm.supernumerary = true;
                    }
                }
                auto rp = std::make_shared<rainbow::RainbowPhase>();
                rp->build(prm);
                med.rainbowPhase = rp;
            } else if (kind == "hg" || kind.empty()) {
                // explicit HG (or `phase` with no argument): default lobe, nothing to do.
            } else {
                fail("medium `phase " + kind + "` is not a known phase model (use `hg` or `rainbow`)");
                return false;
            }
        }

        L.scene.media.push_back(std::move(med));
        return true;
    }

    // Derive the optical quantities (focal length, resolved fov, aperture radius and
    // physical-optics film distance) from the photographic controls, and write them into
    // `cs`. `fovDeg` is the BASE vertical fov before zoom; `lensMM` a prime focal length
    // in mm (>0 overrides fov); `zoom` a focal-length multiplier; `fstopN` the f-number;
    // `hmm` the film height in mm; `focus_m` the focus distance in metres. Factored out of
    // readFilmExposure so `camera_curve` can re-derive it per frame when fov/zoom/fstop/
    // focus are animated by keyframe tracks (see ScalarTrack). Pure function of its args.
    static void deriveCameraOptics(CamSpec& cs, double fovDeg, double lensMM,
                                   double zoom, double fstopN, double hmm, double focus_m) {
        const double DEG = 3.141592653589793 / 180.0;
        if (zoom <= 0.0) zoom = 1.0;
        cs.zoom = zoom;
        double focalMM;
        if (lensMM > 0.0) focalMM = lensMM;
        else { double th = std::tan(0.5 * fovDeg * DEG); focalMM = (th > 1e-9) ? hmm / (2.0 * th) : 0.0; }
        focalMM *= zoom;
        cs.focal = focalMM / 1000.0;
        // fov follows the (zoomed) focal length when a lens/zoom is in play; otherwise it
        // stays the authored base fov.
        if (lensMM > 0.0 || zoom != 1.0)
            cs.fov = (focalMM > 1e-9) ? 2.0 * std::atan(hmm / (2.0 * focalMM)) / DEG : fovDeg;
        else
            cs.fov = fovDeg;
        // f-stop: N = f / (2*apertureR) -> apertureR = f / (2N). Overrides any `aperture`
        // radius. Aperture radius is an internal (metre) length.
        if (fstopN > 0.0 && cs.focal > 0.0) cs.aperture = cs.focal / (2.0 * fstopN);
        // Physical-optics camera: when a lens/f-stop is authored, put the film at the real
        // image distance and give the thin lens the true focal length, so the f-number
        // yields correct depth of field in the catch modes (A/C). Thin-lens law
        // 1/so + 1/si = 1/f; focus 0 (or beyond hyperfocal) means infinity -> si = f.
        if ((lensMM > 0.0 || fstopN > 0.0) && cs.focal > 0.0) {
            double f = cs.focal, so = focus_m;
            cs.filmDist_m = (so > f) ? 1.0 / (1.0 / f - 1.0 / so) : f;
            cs.lensF_m    = f;
        }
    }

    // A camera archetype preset: physically-plausible optics for a real camera *type*,
    // filled BEFORE the block's own knobs so any dial (`lens`, `fstop`, `film{size}`)
    // still overrides it — exactly like `material { preset gold }`. One preset serves
    // both worlds: in the finite-lens catch modes (A/C) the sensor size + focal + f-stop
    // give real depth of field; in the pinhole/backward modes (R/B/U) the same sensor +
    // focal still set the correct field of view and the aperture collapses to a point.
    struct CamPreset { double filmW_mm = 0, filmH_mm = 0, lensMM = 0, fstop = 0; };

    // Resolve a camera archetype name -> CamPreset. Names are normalised (lowercased,
    // spaces/underscores/hyphens stripped) so `vintage-slr`, `vintage slr`, `vintageslr`
    // all match. Specs are drawn from the reference archetypes in `cameras/`. Returns
    // false for an unknown name.
    static bool resolveCameraPreset(const std::string& raw, CamPreset& p) {
        std::string k;
        for (char c : raw) { if (c==' '||c=='_'||c=='-') continue; k += (char)std::tolower((unsigned char)c); }
        // cinema: Blackmagic-style cine ("35 T2.1", "4K") — Super35, 35mm, ~T2.1.
        if (k=="cinema"||k=="cine"||k=="cinemacamera")      { p={24.6,13.8,35.0,2.1}; return true; }
        // pocket: Sony RX0-style rugged compact — 1" sensor, ~24mm-equiv wide, deep DOF.
        if (k=="pocket"||k=="compact"||k=="pocketcamera")   { p={13.2, 8.8, 8.8,4.0}; return true; }
        // portable: full-frame mirrorless with a bright ~35mm prime.
        if (k=="portable"||k=="mirrorless"||k=="portablecamera") { p={36.0,24.0,35.0,1.8}; return true; }
        // vintage: purple folding rangefinder (FED/Zorki) — 35mm film, ~50mm, collapsible.
        if (k=="vintage"||k=="rangefinder"||k=="vintagecamera")  { p={36.0,24.0,50.0,3.5}; return true; }
        // vintage-slr: classic 35mm SLR with a fast 50mm normal.
        if (k=="vintageslr"||k=="slr"||k=="vintageslrcamera")    { p={36.0,24.0,50.0,1.4}; return true; }
        return false;
    }

    // Read the film sub-block + photographic exposure/f-stop/lens controls shared by
    // `camera` and `camera_path`, and resolve the film size (named format or explicit
    // mm), the focal length (from `lens <mm>` or `fov_y`), the f-stop -> aperture
    // radius, the physical-optics film distance, and the manual exposure multiplier.
    // `cs.fov` must already be set. Returns false only on an unknown film format.
    bool readFilmExposure(const Block& b, CamSpec& cs) {
        // Camera archetype preset (`preset <name>`) fills default optics first, so the
        // block's own knobs below override it. Applies to camera/path/orbit/curve alike.
        CamPreset preset;
        bool hasPreset = false;
        {
            std::string pn = strOf(b, "preset");
            if (!pn.empty()) {
                if (!resolveCameraPreset(pn, preset)) {
                    fail("unknown camera preset '" + pn + "' (cinema, pocket, portable, "
                         "vintage, vintage-slr)");
                    return false;
                }
                hasPreset = true;
                cs.filmW_mm = preset.filmW_mm; cs.filmH_mm = preset.filmH_mm;
            }
        }
        const Stmt* film = find(b, "film");
        if (film && film->val.block) {
            const Block& fb = *film->val.block;
            const Stmt* r = find(fb, "res");
            if (r && !r->val.words.empty()) {
                cs.res  = (int)num(r->val.words[0]);
                // `res W H` gives a non-square film; `res W` stays square (resY=W).
                cs.resY = (r->val.words.size() >= 2) ? (int)num(r->val.words[1]) : cs.res;
            }
            // Named sensor/film format -> physical size in mm (e.g. `format full-frame`,
            // `format medium-format`, `format 4x5`). Words are joined so a spaced
            // "medium format" also works. An explicit `size w h` below overrides it.
            const Stmt* fmt = find(fb, "format");
            if (fmt && !fmt->val.words.empty()) {
                std::string joined;
                for (const auto& w : fmt->val.words) joined += w;
                double fw = 0.0, fh = 0.0;
                if (!filmFormatMM(joined, fw, fh)) {
                    fail("unknown film format '" + joined + "' (try: full-frame, aps-c, "
                         "micro-four-thirds, super35, medium-format, 6x6, 6x7, large-format, 4x5, 8x10)");
                    return false;
                }
                cs.filmW_mm = fw; cs.filmH_mm = fh;
            }
            const Stmt* sz = find(fb, "size");     // explicit physical sensor, millimetres
            if (sz && sz->val.words.size() >= 2) {
                cs.filmW_mm = num(sz->val.words[0]);
                cs.filmH_mm = num(sz->val.words[1]);
            }
            cs.iso      = dblOf(fb, "iso", 0.0);
            cs.shutter  = dblOf(fb, "shutter", 0.0);
            cs.exposure = dblOf(fb, "exposure", 0.0);
        }
        // Focal length. Photographers pick a lens (mm) far more often than a fov, so
        // `lens <mm>` is honoured first: fov_y = 2*atan(filmH/(2*focal)) (overrides any
        // fov_y). Otherwise derive the focal length from fov_y and the film height:
        // fov_y = 2*atan(filmH/(2f)) -> f = filmH / (2 tan(fov/2)). Fall back to a 35mm
        // full-frame 24mm height when no physical size is authored.
        double hmm = (cs.filmH_mm > 0.0) ? cs.filmH_mm : 24.0;
        // `lens`/`fstop` default to the archetype preset's values when one is named, so
        // the preset supplies focal length + aperture unless the block overrides them.
        double lensMM = dblOf(b, "lens", hasPreset ? preset.lensMM : 0.0);  // focal length in mm
        // `zoom <x>` multiplies the focal length (x>1 = tele/narrower fov; x<1 = wider).
        // It is the animatable "zoom ring" and composes on top of `lens`/`fov_y`.
        double zoom  = dblOf(b, "zoom", 1.0);
        double fstop = dblOf(b, "fstop", hasPreset ? preset.fstop : 0.0);
        // Resolve focal/fov/aperture/film-distance. cs.focus is already in metres (Len-scaled).
        deriveCameraOptics(cs, cs.fov, lensMM, zoom, fstop, hmm, cs.focus);
        // Manual exposure multiplier (see CamSpec). Active iff any control authored.
        if (cs.exposure > 0.0 || cs.iso > 0.0 || cs.shutter > 0.0) {
            double base = (cs.exposure > 0.0) ? cs.exposure : 1.0;
            double isoF = (cs.iso     > 0.0) ? cs.iso / 100.0 : 1.0;
            double shF  = (cs.shutter > 0.0) ? cs.shutter     : 1.0;
            cs.exposureMul = base * isoF * shF;
        }
        return true;
    }

    // Parse a camera's lens projection: `projection <name>` or the `fisheye [type]`
    // shorthand (bare `fisheye` -> equisolid, the common consumer default).
    bool readProjection(const Block& b, CamSpec& cs) {
        const Stmt* pj = find(b, "projection");
        const Stmt* fe = find(b, "fisheye");
        if (pj && !pj->val.words.empty()) {
            int p = projectionFromName(pj->val.words[0]);
            if (p < 0) { fail("unknown projection '" + pj->val.words[0] + "' (rectilinear, "
                              "equidistant/fisheye, equisolid, stereographic, orthographic)"); return false; }
            cs.projection = p;
        } else if (fe) {
            int p = fe->val.words.empty() ? CAM_EQUISOLID : projectionFromName(fe->val.words[0]);
            if (p < 0) { fail("unknown fisheye type '" + fe->val.words[0] + "' (equidistant, "
                              "equisolid, stereographic, orthographic)"); return false; }
            cs.projection = p;
        }
        return true;
    }

    // Resolve a lens-surface `ior` word: a glass name (`glass:BK7`, `BK7`, `SF10`,
    // `flint`, ...) -> its Sellmeier dispersion; otherwise a bare number -> constant
    // index. Air is index 1.
    static Spectrum lensIorOf(const std::string& raw) {
        std::string name = raw;
        if (name.rfind("glass:", 0) == 0) name = name.substr(6);
        Spectrum g;
        if (resolveGlassIor(name, g)) return g;
        return iorConstant(std::atof(raw.c_str()));
    }

    // Parse an optional physical lens: `lens { preset <name> | surface <r> <t> <ior>
    // <semi_ap> [stop] ... | focal <mm> fstop <N> glass <name> }`. Radii/thicknesses/
    // apertures are millimetres (the lens's own units). Builds the LensSystem, sets
    // the sensor size from the camera film, and autofocuses at `cs.focus`.
    bool readLens(const Block& b, CamSpec& cs) {
        const Stmt* ls = find(b, "lens");
        if (!ls || !ls->val.block) return true;    // no lens block (a scalar `lens <mm>` is handled elsewhere)
        const Block& lb = *ls->val.block;

        double focalMM = (cs.focal > 0.0) ? cs.focal * 1000.0 : 50.0;
        focalMM = dblOf(lb, "focal", focalMM);
        double fstop = dblOf(lb, "fstop", 0.0);
        if (fstop <= 0.0) fstop = dblOf(b, "fstop", 0.0);
        if (fstop <= 0.0) fstop = 2.8;
        std::string glassName = strOf(lb, "glass", "BK7");

        auto sys = std::make_shared<LensSystem>();
        // 1) explicit surfaces take priority (paste any real prescription).
        int nSurf = 0;
        for (const auto& s : lb.stmts) {
            if (s.key != "surface") continue;
            s.used = true;
            const auto& wds = s.val.words;
            if (wds.size() < 4) { fail("camera '" + cs.name + "' lens: `surface` needs "
                                       "<radius_mm> <thickness_mm> <ior> <semi_aperture_mm> [stop]"); return false; }
            LensSurface e;
            e.radius    = num(wds[0]);
            e.thickness = num(wds[1]);
            e.ior       = lensIorOf(wds[2]);
            e.aperture  = num(wds[3]);
            if (wds.size() >= 5 && wds[4] == "stop") { e.isStop = true; }
            sys->surf.push_back(e);
            ++nSurf;
        }
        if (nSurf > 0) {
            sys->finalize();
            sys->name = "custom";
        } else {
            // 2) a named preset (singlet/achromat/doublet/telephoto/wide), else default
            //    to an achromatic doublet at the derived focal length + f-number.
            std::string preset = strOf(lb, "preset", "");
            std::string pk;
            for (char c : preset) { if (c==' '||c=='_'||c=='-') continue; pk += (char)std::tolower((unsigned char)c); }
            if ((pk == "singlet" || pk == "biconvex" || pk == "simple")) {
                Spectrum g;
                if (!resolveGlassIor((glassName.rfind("glass:",0)==0?glassName.substr(6):glassName), g))
                    g = glassOrDefault("BK7", 1.5168);
                *sys = makeSinglet(focalMM, fstop, g);
            } else if (!preset.empty()) {
                if (!resolveLensPreset(preset, focalMM, fstop, *sys)) {
                    fail("camera '" + cs.name + "' lens: unknown preset '" + preset +
                         "' (singlet, achromat/doublet, telephoto, wide)"); return false;
                }
            } else {
                *sys = makeAchromat(focalMM, fstop, glassOrDefault("BK7", 1.5168), glassOrDefault("SF10", 1.7283));
            }
        }
        // Sensor size from the camera film (default full-frame 36x24 mm).
        sys->filmW_mm = (cs.filmW_mm > 0.0) ? cs.filmW_mm : 36.0;
        sys->filmH_mm = (cs.filmH_mm > 0.0) ? cs.filmH_mm : 24.0;
        sys->focusAt(cs.focus);                    // cs.focus is metres (0 => infinity)
        cs.lens = sys;
        return true;
    }

    // ---- camera ----
    bool addCamera(const Block& b, Loaded& L) {
        CamSpec cs;
        cs.name = b.name.empty() ? ("cam" + std::to_string(L.cameras.size())) : b.name;
        vec3Of(b, "eye", cs.eye); vec3Of(b, "look_at", cs.look); vec3Of(b, "up", cs.up);
        cs.eye = P(cs.eye); cs.look = P(cs.look);   // up is a direction: unscaled
        cs.fov = dblOf(b, "fov_y", 40.0);
        cs.aperture = Len(dblOf(b, "aperture", 0.02));
        cs.focus = Len(dblOf(b, "focus", 0.0));
        if (!readFilmExposure(b, cs)) return false;   // film{res,size/format,iso,...}, lens, fstop, zoom
        if (!readProjection(b, cs)) return false;     // projection/fisheye
        if (!readLens(b, cs)) return false;           // optional physical `lens { ... }` block
        std::string md = strOf(b, "mode");
        if (!md.empty()) cs.mode = normMode(md, L);
        L.cameras.push_back(cs);

        // Mirror the first camera into the flat fields + global mode/res (defaults
        // that the rest of the loader and CLI-override logic still read).
        if (!L.hasCamera) {
            L.camEye = cs.eye; L.camLook = cs.look; L.camUp = cs.up;
            L.camFov = cs.fov; L.camAperture = cs.aperture; L.camFocus = cs.focus;
            if (cs.mode) L.mode = cs.mode;
            if (cs.res > 0) L.res = cs.res;
            L.hasCamera = true;
        }
        return true;
    }

    // A `camera_path` expands into a sequence of CamSpec frames sharing look_at/up/
    // fov_y/mode/aperture/focus/film, with the eye (and optionally look_at) linearly
    // interpolated across keyframes. Grammar (numbers only, so the parser keeps each
    // key on one statement):
    //   camera_path "dolly" {
    //       look_at 0 1 0   up 0 1 0   fov_y 40   mode B   frames 60
    //       film { res 256 256 }
    //       key <t> <ex> <ey> <ez> [<lx> <ly> <lz>]   # >= 2 keys, t in [0,1]
    //       ...
    //   }
    // Frame i (0..frames-1) samples t = i/(frames-1); its output name is
    // "<path><i>" (zero-padded), so the multi-camera loop writes one file per frame.
    // Parse an `exposure_lock [selector]` statement (shared by camera_path/orbit/curve)
    // into the selector fields of `cs`. Returns whether the lock is enabled. The selector
    // words are: (none)/on/true/1/first -> FIRST; off/false/0 -> disabled; average/avg/mean
    // -> AVERAGE; `index N`/`frame N` -> INDEX; `near X Y Z` -> NEAR; anything else (a
    // quoted or bare word) -> CAMERA metering from a separately-defined camera of that name.
    bool parseExposureLock(const Block& b, CamSpec& cs) {
        const Stmt* el = find(b, "exposure_lock");
        if (!el) return false;
        const auto& w = el->val.words;
        // A bare `exposure_lock` (or `on`/`true`/`1`) defaults to metering the AVERAGE of
        // the whole path — a robust choice that won't expose the entire flythrough for one
        // possibly-atypical opening frame. `first` is the explicit "expose for frame 0".
        if (w.empty()) { cs.expLockSel = CamSpec::EXPLOCK_AVERAGE; return true; }
        const std::string v0 = w[0];
        if (v0 == "off" || v0 == "false" || v0 == "0") return false;
        if (v0 == "on" || v0 == "true" || v0 == "1") {
            cs.expLockSel = CamSpec::EXPLOCK_AVERAGE; return true;
        }
        if (v0 == "first") { cs.expLockSel = CamSpec::EXPLOCK_FIRST; return true; }
        if (v0 == "average" || v0 == "avg" || v0 == "mean") {
            cs.expLockSel = CamSpec::EXPLOCK_AVERAGE; return true;
        }
        if (v0 == "index" || v0 == "frame") {
            cs.expLockSel = CamSpec::EXPLOCK_INDEX;
            cs.expLockIndex = (w.size() >= 2) ? (int)num(w[1]) : 0;
            return true;
        }
        if (v0 == "near") {
            cs.expLockSel = CamSpec::EXPLOCK_NEAR;
            if (w.size() >= 4) cs.expLockPoint = P(Vec3{num(w[1]), num(w[2]), num(w[3])});
            else fail("exposure_lock near needs: near X Y Z");
            return true;
        }
        if (v0 == "camera" || v0 == "cam") {
            // Explicit `exposure_lock camera "name"` — the name follows the keyword.
            if (w.size() >= 2) { cs.expLockSel = CamSpec::EXPLOCK_CAMERA; cs.expLockCam = w[1]; return true; }
            fail("exposure_lock camera needs a name: exposure_lock camera \"name\" "
                 "(or just exposure_lock \"name\")");
            return true;
        }
        cs.expLockSel = CamSpec::EXPLOCK_CAMERA;   // a camera name given directly (quoted or bare)
        cs.expLockCam = v0;
        return true;
    }

    bool addCameraPath(const Block& b, Loaded& L) {
        std::string base = b.name.empty() ? ("path" + std::to_string(L.cameras.size())) : b.name;
        CamSpec shared;
        vec3Of(b, "look_at", shared.look); vec3Of(b, "up", shared.up);
        shared.look = P(shared.look);
        shared.fov = dblOf(b, "fov_y", 40.0);
        shared.aperture = Len(dblOf(b, "aperture", 0.02));
        shared.focus = Len(dblOf(b, "focus", 0.0));
        std::string md = strOf(b, "mode"); if (!md.empty()) shared.mode = normMode(md, L);
        shared.fps = dblOf(b, "fps", 0.0);   // playback hint for the flyby (0 = inherit scene default)
        if (!readFilmExposure(b, shared)) return false;   // film{res,size/format,...}, lens, fstop, zoom
        if (!readProjection(b, shared)) return false;     // projection/fisheye
        int frames = (int)dblOf(b, "frames", 0.0);
        if (frames < 1) { fail("camera_path '" + base + "' needs frames >= 1"); return false; }

        // Dolly-zoom (Vertigo) mode: hold the subject's on-screen size constant by
        // trading fov against distance. The subject is each frame's look_at point;
        // the reference size is anchored on the first frame. Enabled by a bare
        // `dolly_zoom` (or `dolly_zoom on`); `off`/`false`/`0` disables.
        bool dolly = false;
        if (const Stmt* dz = find(b, "dolly_zoom")) {
            if (dz->val.words.empty()) dolly = true;
            else { const std::string& v = dz->val.words[0]; dolly = !(v=="off"||v=="false"||v=="0"); }
        }
        const double DEG = 3.141592653589793 / 180.0;

        // Exposure-lock: a bare `exposure_lock` (or a selector — see parseExposureLock)
        // makes every frame of this path share ONE auto-exposure anchor (no flicker);
        // `off`/`false`/`0` disables (the default). The selector fields land on `shared`
        // so they copy into every frame's CamSpec. The group id is this path's starting
        // index in L.cameras — unique because paths occupy disjoint contiguous ranges.
        bool pathLock = parseExposureLock(b, shared);
        const int pathGroup = (int)L.cameras.size();

        // Collect keyframes (t, eye, optional look_at, optional fov), sorted by t.
        // Field count disambiguates: 4=t,eye  5=t,eye,fov  7=t,eye,look  8=t,eye,look,fov.
        struct Key { double t; Vec3 eye, look; bool hasLook; double fov; };
        std::vector<Key> keys;
        for (const auto& s : b.stmts) {
            if (s.key != "key") continue;
            s.used = true;
            const auto& w = s.val.words;
            size_t n = w.size();
            if (n != 4 && n != 5 && n != 7 && n != 8) {
                fail("camera_path key needs: t ex ey ez [lx ly lz] [fov_deg]"); return false;
            }
            Key k;
            k.t = num(w[0]);
            k.eye = P(Vec3{num(w[1]), num(w[2]), num(w[3])});
            k.hasLook = (n >= 7);
            k.look = k.hasLook ? P(Vec3{num(w[4]), num(w[5]), num(w[6])}) : shared.look;
            k.fov = shared.fov;                        // default: the shared/path fov
            if (n == 5) k.fov = num(w[4]);             // t,eye,fov
            else if (n == 8) k.fov = num(w[7]);        // t,eye,look,fov
            keys.push_back(k);
        }
        if (keys.size() < 2) { fail("camera_path '" + base + "' needs >= 2 keys"); return false; }
        std::sort(keys.begin(), keys.end(), [](const Key& a, const Key& b2){ return a.t < b2.t; });

        double refHalf = -1.0;   // dolly-zoom reference: dist * tan(fov/2), set on frame 0
        int pad = 1; for (int f = frames - 1; f >= 10; f /= 10) ++pad;   // zero-pad width
        for (int i = 0; i < frames; ++i) {
            double t = (frames == 1) ? keys.front().t : keys.front().t +
                       (keys.back().t - keys.front().t) * ((double)i / (frames - 1));
            // Piecewise-linear lookup of the bracketing keyframes for this t.
            const Key* a = &keys.front(); const Key* c = &keys.back();
            for (size_t j = 0 ; j + 1 < keys.size(); ++j)
                if (t >= keys[j].t && t <= keys[j + 1].t) { a = &keys[j]; c = &keys[j + 1]; break; }
            double span = c->t - a->t;
            double f = (span > 1e-12) ? (t - a->t) / span : 0.0;
            CamSpec cs = shared;
            cs.eye  = a->eye  + (c->eye  - a->eye)  * f;
            cs.look = a->look + (c->look - a->look) * f;
            cs.fov  = a->fov  + (c->fov  - a->fov)  * f;   // per-keyframe zoom
            if (dolly) {                                   // override fov to hold subject size
                double di = length(cs.eye - cs.look);
                if (refHalf < 0.0) refHalf = di * std::tan(0.5 * cs.fov * DEG);   // anchor on frame 0
                if (di > 1e-9) cs.fov = 2.0 * std::atan(refHalf / di) / DEG;
            }
            char num5[8]; std::snprintf(num5, sizeof(num5), "%0*d", pad, i);
            cs.name = base + num5;
            cs.pathGroup = pathGroup;
            cs.exposureLock = pathLock;
            L.cameras.push_back(cs);
            if (!L.hasCamera) {
                L.camEye = cs.eye; L.camLook = cs.look; L.camUp = cs.up;
                L.camFov = cs.fov; L.camAperture = cs.aperture; L.camFocus = cs.focus;
                if (cs.mode) L.mode = cs.mode;
                if (cs.res > 0) L.res = cs.res;
                L.hasCamera = true;
            }
        }
        return true;
    }

    // A `camera_orbit` expands into N CamSpec frames whose eye rides a circle around a
    // fixed `center` (also the default look_at), producing a turntable / fly-around that
    // stitches straight into an MP4. The circle lies in the plane perpendicular to `axis`
    // (default y -> circle in the xz-plane); `radius` is its radius and `height` offsets
    // the eye along the axis from the centre. The sweep runs `sweep_deg` degrees (default
    // 360) starting at `start_deg`; a full 360 sweep is sampled at i/frames (frame N ==
    // frame 0, so it is NOT duplicated) to make a seamless loop, while a partial sweep
    // spans its endpoints via i/(frames-1). All frames share look_at/up/fov/mode/film/lens.
    // Grammar:
    //   camera_orbit "spin" {
    //       center 0.40 0.37 0.45   radius 0.45   height -0.05   axis y
    //       up 0 1 0   fov_y 58   mode R   frames 120
    //       look_at 0.40 0.37 0.45           # optional, defaults to center
    //       start_deg 0   sweep_deg 360       # optional
    //       exposure_lock                     # optional (flicker-free)
    //       film { res 900 900 }
    //   }
    bool addCameraOrbit(const Block& b, Loaded& L) {
        std::string base = b.name.empty() ? ("orbit" + std::to_string(L.cameras.size())) : b.name;
        CamSpec shared;
        vec3Of(b, "up", shared.up);
        shared.fov = dblOf(b, "fov_y", 40.0);
        shared.aperture = Len(dblOf(b, "aperture", 0.02));
        shared.focus = Len(dblOf(b, "focus", 0.0));
        std::string md = strOf(b, "mode"); if (!md.empty()) shared.mode = normMode(md, L);
        shared.fps = dblOf(b, "fps", 0.0);   // playback hint for the flyby (0 = inherit scene default)
        if (!readFilmExposure(b, shared)) return false;   // film{res,size/format,...}, lens, fstop, zoom
        if (!readProjection(b, shared)) return false;     // projection/fisheye
        if (!readLens(b, shared)) return false;           // optional physical `lens { ... }` block

        Vec3 center{0, 0, 0};
        if (!vec3Of(b, "center", center)) { fail("camera_orbit '" + base + "' needs a `center`"); return false; }
        center = P(center);
        Vec3 look = center;                               // look_at defaults to the orbit centre
        if (find(b, "look_at")) { Vec3 lv{0,0,0}; vec3Of(b, "look_at", lv); look = P(lv); }
        shared.look = look;

        double radius = Len(dblOf(b, "radius", 0.0));
        if (radius <= 0.0) { fail("camera_orbit '" + base + "' needs radius > 0"); return false; }
        double height = Len(dblOf(b, "height", 0.0));     // offset along the axis from centre
        int frames = (int)dblOf(b, "frames", 0.0);
        if (frames < 1) { fail("camera_orbit '" + base + "' needs frames >= 1"); return false; }
        double startDeg = dblOf(b, "start_deg", 0.0);
        double sweepDeg = dblOf(b, "sweep_deg", 360.0);

        // Orbit axis + an orthonormal basis (U, W) spanning the plane perpendicular to it.
        std::string axisW = strOf(b, "axis", "y");
        Vec3 A = (axisW == "x") ? Vec3{1, 0, 0} : (axisW == "z") ? Vec3{0, 0, 1} : Vec3{0, 1, 0};
        Vec3 ref = (std::fabs(A.y) < 0.9) ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
        Vec3 U = normalize(cross(ref, A));
        Vec3 W = normalize(cross(A, U));

        bool pathLock = parseExposureLock(b, shared);
        const int pathGroup = (int)L.cameras.size();
        const double DEG = 3.141592653589793 / 180.0;

        bool fullLoop = std::fabs(std::fabs(sweepDeg) - 360.0) < 1e-6;
        int pad = 1; for (int f = frames - 1; f >= 10; f /= 10) ++pad;   // zero-pad width
        for (int i = 0; i < frames; ++i) {
            double frac = (frames <= 1) ? 0.0
                        : fullLoop    ? ((double)i / frames)
                                      : ((double)i / (frames - 1));
            double ang = (startDeg + sweepDeg * frac) * DEG;
            CamSpec cs = shared;
            cs.eye = center + A * height + (U * std::cos(ang) + W * std::sin(ang)) * radius;
            char num5[8]; std::snprintf(num5, sizeof(num5), "%0*d", pad, i);
            cs.name = base + num5;
            cs.pathGroup = pathGroup;
            cs.exposureLock = pathLock;
            L.cameras.push_back(cs);
            if (!L.hasCamera) {
                L.camEye = cs.eye; L.camLook = cs.look; L.camUp = cs.up;
                L.camFov = cs.fov; L.camAperture = cs.aperture; L.camFocus = cs.focus;
                if (cs.mode) L.mode = cs.mode;
                if (cs.res > 0) L.res = cs.res;
                L.hasCamera = true;
            }
        }
        return true;
    }

    // A `camera_curve` expands into N CamSpec frames whose eye rides a smooth 3D
    // Catmull-Rom spline through the authored `point` control points (the curve passes
    // THROUGH each). Where cameras sit along the curve is set either by a fixed count
    // (`frames N`, uniform arc-length spacing) or by a DENSITY — cameras per unit
    // length — that may itself vary along the curve, giving the camera a "speed": high
    // density = many closely-spaced frames = slow motion through that stretch; low
    // density = fast. `density <rho>` is constant; `density_at <t> <rho>` keyframes it
    // (piecewise-linear over the normalized position t in [0,1], t=0 first point, t=1
    // last). The number of cameras placed with local spacing 1/rho follows from
    // integrating rho over arc length; `frames N` (if also given) instead fixes the
    // count and only uses the density to DISTRIBUTE those N cameras. Orientation:
    //   look tangent    (default) — aim along the direction of travel
    //       min_reach <frac>       (default 0.5) fold defence: floor the horizontal reach used
    //                              for pitch so a hairpin/U-turn can't rake the view into the
    //                              ceiling or floor. `0` disables (legacy behaviour).
    //       look_smooth <n>        (default 0/off) Gaussian sigma in frames; temporally smooths
    //                              the look direction so a fold's fast pan is spread out.
    //   look_at x y z             — a fixed target for every frame
    //   look curve + look_point.. — aim at a SECOND Catmull-Rom spline, sampled in step
    // Roll and the lens scalars can be ANIMATED per frame over the normalized timeline
    // t in [0,1] (t=0 first frame, t=1 last), each keyframed by `<name>_at <t> <value>`
    // (piecewise-linear, flat-clamped outside the ends, exactly like `density_at`) or set
    // constant with the bare keyword:
    //   roll <deg> | roll_at t deg      — bank about the view axis (the third orientation DOF)
    //   fov_at t deg                    — animate vertical field of view (a fov "zoom")
    //   zoom_at t x                     — animate the focal-length multiplier
    //   fstop_at t N                    — animate the f-number (aperture / depth of field)
    //   focus_at t dist                 — animate the focus distance (authored units)
    // (Lens PROJECTION / fisheye is a discrete mode, not a continuous track — set it once
    // for the whole flight with `projection`/`fisheye`.) `closed` loops the curve
    // (seamless). Frames share up/mode/film/lens and any un-animated lens scalars.
    // Grammar:
    //   camera_curve "fly" {
    //       point 0 1 3   point 1 1 1   point 2 1 3   point 1 1 5   # >= 2 control points
    //       up 0 1 0   fov_y 50   mode R   frames 90
    //       density 20                       # OR density_at 0 5  density_at 0.5 40  density_at 1 5
    //       look tangent                     # OR look_at 1 1 3   OR look curve + look_point ...
    //       roll_at 0 0   roll_at 0.5 20   roll_at 1 0        # bank into the turn and back
    //       fstop_at 0 8   fstop_at 1 1.4                     # rack the aperture open
    //       focus_at 0 5   focus_at 1 1.5                     # pull focus toward the camera
    //       closed                           # seamless loop (its OWN line — see note below)
    //       exposure_lock                    # freeze frame 0's exposure across all frames
    //       film { res 900 600 }
    //   }
    // NOTE: each value-less flag keyword (`closed`, `exposure_lock`) must be on its OWN
    // line. A statement's value is always the next token, so `closed exposure_lock` parses
    // as `closed` with the *value* "exposure_lock" (the grammar can't tell that apart from
    // `material white`), silently dropping the second flag.
    bool addCameraCurve(const Block& b, Loaded& L) {
        std::string base = b.name.empty() ? ("curve" + std::to_string(L.cameras.size())) : b.name;
        CamSpec shared;
        vec3Of(b, "up", shared.up);
        shared.fov = dblOf(b, "fov_y", 40.0);
        shared.aperture = Len(dblOf(b, "aperture", 0.02));
        shared.focus = Len(dblOf(b, "focus", 0.0));
        std::string md = strOf(b, "mode"); if (!md.empty()) shared.mode = normMode(md, L);
        shared.fps = dblOf(b, "fps", 0.0);   // playback hint for the flyby (0 = inherit scene default)
        if (!readFilmExposure(b, shared)) return false;   // film{res,size/format,...}, lens, fstop, zoom
        if (!readProjection(b, shared)) return false;     // projection/fisheye
        if (!readLens(b, shared)) return false;           // optional physical `lens { ... }` block

        // Control points (>= 2), in file order; the spline passes through each.
        std::vector<Vec3> pts;
        for (const auto& s : b.stmts) {
            if (s.key != "point") continue;
            s.used = true;
            if (s.val.words.size() < 3) { fail("camera_curve '" + base + "' point needs x y z"); return false; }
            pts.push_back(P(Vec3{num(s.val.words[0]), num(s.val.words[1]), num(s.val.words[2])}));
        }
        if (pts.size() < 2) { fail("camera_curve '" + base + "' needs >= 2 `point` control points"); return false; }

        bool closed = false;
        if (const Stmt* c = find(b, "closed")) {
            if (c->val.words.empty()) closed = true;
            else { const std::string& v = c->val.words[0]; closed = !(v == "off" || v == "false" || v == "0"); }
        }

        // Spline knot parameterization: `spline uniform|centripetal|chordal` (or a raw
        // alpha number). Default UNIFORM preserves every existing scene bit-for-bit;
        // CENTRIPETAL (alpha 0.5) removes the overshoot/looping that uniform Catmull-Rom
        // produces when waypoints are unevenly spaced -- the fix for a jerky room flight.
        double splineAlpha = 0.0;
        if (const Stmt* sp = find(b, "spline")) {
            if (sp->val.words.empty()) { fail("camera_curve '" + base + "' spline needs: uniform|centripetal|chordal|<alpha>"); return false; }
            const std::string& v = sp->val.words[0];
            if      (v == "uniform")     splineAlpha = 0.0;
            else if (v == "centripetal") splineAlpha = 0.5;
            else if (v == "chordal")     splineAlpha = 1.0;
            else                         splineAlpha = num(v);   // raw alpha
            if (splineAlpha < 0.0) splineAlpha = 0.0;
        }

        // Density = cameras per unit LENGTH (1/authored-unit -> 1/metre). `density_at`
        // keyframes it (piecewise-linear over normalized position t in [0,1]); a bare
        // `density` is constant. Either drives BOTH the camera count (integral of rho
        // over the curve) and the local spacing (1/rho).
        struct DKey { double t, rho; };
        std::vector<DKey> dkeys;
        for (const auto& s : b.stmts) {
            if (s.key != "density_at") continue;
            s.used = true;
            if (s.val.words.size() < 2) { fail("camera_curve '" + base + "' density_at needs: <t> <rho>"); return false; }
            dkeys.push_back({num(s.val.words[0]), num(s.val.words[1]) / L_});
        }
        std::sort(dkeys.begin(), dkeys.end(), [](const DKey& a, const DKey& b2){ return a.t < b2.t; });
        double constDensity = find(b, "density") ? dblOf(b, "density", 0.0) / L_ : -1.0;
        bool haveDensity = !dkeys.empty() || constDensity > 0.0;

        int framesReq = (int)dblOf(b, "frames", 0.0);
        if (framesReq < 1 && !haveDensity) {
            fail("camera_curve '" + base + "' needs `frames N` or a `density`/`density_at <t> <rho>`");
            return false;
        }

        auto densityAt = [&](double u) -> double {
            if (!dkeys.empty()) {
                if (u <= dkeys.front().t) return dkeys.front().rho;
                if (u >= dkeys.back().t)  return dkeys.back().rho;
                for (size_t j = 0; j + 1 < dkeys.size(); ++j)
                    if (u >= dkeys[j].t && u <= dkeys[j + 1].t) {
                        double sp = dkeys[j + 1].t - dkeys[j].t;
                        double f = (sp > 1e-12) ? (u - dkeys[j].t) / sp : 0.0;
                        return dkeys[j].rho + (dkeys[j + 1].rho - dkeys[j].rho) * f;
                    }
                return dkeys.back().rho;
            }
            return (constDensity > 0.0) ? constDensity : 1.0;
        };

        // Densely sample the spline; build a cumulative "count" table C(g) = INT rho ds.
        // For a constant/absent density this is just arc length, so the same inversion
        // yields uniform arc-length spacing.
        int nSeg = closed ? (int)pts.size() : (int)pts.size() - 1;
        int M = std::max(64, 64 * nSeg);
        std::vector<double> sampG((size_t)M + 1), sampC((size_t)M + 1), sampS((size_t)M + 1);
        Vec3 prev = catmullRomAt(pts, closed, 0.0, splineAlpha);
        sampG[0] = 0.0; sampC[0] = 0.0; sampS[0] = 0.0;
        for (int k = 1; k <= M; ++k) {
            double g = nSeg * (double)k / M;
            Vec3 pcur = catmullRomAt(pts, closed, g, splineAlpha);
            double ds = length(pcur - prev);
            double rho = densityAt(g / nSeg);
            sampC[k] = sampC[k - 1] + rho * ds;
            sampS[k] = sampS[k - 1] + ds;      // pure arc length (density-free), for look-ahead
            sampG[k] = g;
            prev = pcur;
        }
        double Cmax = sampC[M];
        if (Cmax <= 0.0) { fail("camera_curve '" + base + "' has zero length or density"); return false; }

        int N = (framesReq >= 1) ? framesReq : std::max(1, (int)std::llround(Cmax));

        auto invertC = [&](double target) -> double {   // count-value -> global param g
            if (target <= 0.0) return 0.0;
            if (target >= Cmax) return (double)nSeg;
            int lo = 0, hi = M;
            while (lo + 1 < hi) { int mid = (lo + hi) / 2; (sampC[(size_t)mid] <= target ? lo : hi) = mid; }
            double c0 = sampC[(size_t)lo], c1 = sampC[(size_t)lo + 1];
            double f = (c1 > c0) ? (target - c0) / (c1 - c0) : 0.0;
            return sampG[(size_t)lo] + (sampG[(size_t)lo + 1] - sampG[(size_t)lo]) * f;
        };

        // Pure arc-length reparameterization (density-free) for the tangent look-ahead.
        double Smax = sampS[M];
        auto arcAtG = [&](double g) -> double {          // g -> arc length s
            if (g <= 0.0) return 0.0;
            if (g >= (double)nSeg) return Smax;
            double kf = g / (double)nSeg * (double)M;     // sampG is linear in k
            int lo = (int)kf; if (lo < 0) lo = 0; if (lo > M - 1) lo = M - 1;
            double f = kf - lo;
            return sampS[(size_t)lo] + (sampS[(size_t)lo + 1] - sampS[(size_t)lo]) * f;
        };
        auto gAtArc = [&](double s) -> double {          // arc length s -> g
            if (s <= 0.0) return 0.0;
            if (s >= Smax) return (double)nSeg;
            int lo = 0, hi = M;
            while (lo + 1 < hi) { int mid = (lo + hi) / 2; (sampS[(size_t)mid] <= s ? lo : hi) = mid; }
            double s0 = sampS[(size_t)lo], s1 = sampS[(size_t)lo + 1];
            double f = (s1 > s0) ? (s - s0) / (s1 - s0) : 0.0;
            return sampG[(size_t)lo] + (sampG[(size_t)lo + 1] - sampG[(size_t)lo]) * f;
        };

        // Orientation. Gather optional look-curve control points first.
        std::vector<Vec3> lookPts;
        for (const auto& s : b.stmts) {
            if (s.key != "look_point") continue;
            s.used = true;
            if (s.val.words.size() < 3) { fail("camera_curve '" + base + "' look_point needs x y z"); return false; }
            lookPts.push_back(P(Vec3{num(s.val.words[0]), num(s.val.words[1]), num(s.val.words[2])}));
        }
        std::string lookKw = strOf(b, "look", "tangent");
        bool lookCurve = (!lookPts.empty() || lookKw == "curve" || lookKw == "look_curve" || find(b, "look_curve"));
        bool lookFixed = !lookCurve && (find(b, "look_at") || lookKw == "look_at");
        if (lookCurve && lookPts.size() < 2) {
            fail("camera_curve '" + base + "' look curve needs >= 2 `look_point` control points"); return false;
        }
        Vec3 fixedLook{0, 0, 0};
        if (lookFixed) { vec3Of(b, "look_at", fixedLook); fixedLook = P(fixedLook); }
        int lookSeg = lookCurve ? (closed ? (int)lookPts.size() : (int)lookPts.size() - 1) : 0;

        // Tangent-mode robustness knobs (fold defence). The tangent look aims at a point a
        // fixed arc-length ahead; where the path FOLDS (a U-turn / hairpin) the horizontal
        // reach of that chord collapses toward zero, so even a small height difference gets
        // amplified by asin(dy/L) into a steep pitch and the camera rakes up into the
        // ceiling (or down into the floor). Two defences, both only touching frames that are
        // actually near a fold — well-conditioned frames (incl. legitimately steep dives that
        // keep their horizontal reach) are left byte-identical:
        //   min_reach <frac>   floor the horizontal reach used for the PITCH at
        //                      frac * (look-ahead chord length); default 0.5, `0` disables.
        //   look_smooth <n>    Gaussian sigma (in frames) for temporal smoothing of the look
        //                      direction (yaw+pitch, wrap-aware for closed loops), spreading a
        //                      fold's unavoidable fast pan over more frames; default 0 (off).
        double minReachFrac = dblOf(b, "min_reach",   0.5);
        double lookSmooth   = dblOf(b, "look_smooth", 0.0);

        // ---- Animatable orientation + lens tracks ----------------------------------
        // Roll (bank about the view axis) and the lens scalars (fov_y, zoom, f-stop,
        // focus) can each be keyframed by `<name>_at <t> <value>` over the normalized
        // timeline t in [0,1] (mirroring `density_at`), or set constant with the bare
        // keyword. A bare keyword doubles as the flat fallback for its track.
        bool trkOk = true;
        auto readTrack = [&](const char* atKey) -> ScalarTrack {
            ScalarTrack tk;
            for (const auto& s : b.stmts) {
                if (s.key != atKey) continue;
                s.used = true;
                if (s.val.words.size() < 2) {
                    fail("camera_curve '" + base + "' " + atKey + " needs: <t> <value>");
                    trkOk = false; continue;
                }
                tk.keys.push_back({num(s.val.words[0]), num(s.val.words[1])});
            }
            tk.sort();
            return tk;
        };
        ScalarTrack rollTrk  = readTrack("roll_at");
        ScalarTrack fovTrk   = readTrack("fov_at");
        ScalarTrack zoomTrk  = readTrack("zoom_at");
        ScalarTrack fstopTrk = readTrack("fstop_at");
        ScalarTrack focusTrk = readTrack("focus_at");
        if (!trkOk) return false;

        // ---- Record-driven tracks (stage 5b) ---------------------------------------
        // `<scalar>_from RECORD.channel[(<driver in t>)]` drives an animatable camera
        // scalar from a parametric-record channel sampled over the flyby timeline t in
        // [0,1] — the "records-as-keyframe-tracks" form: a named, reusable curve bank in
        // place of hand-placed `<scalar>_at` keyframes. The driver defaults to the bare
        // timeline `t` (so `fov_from zoom.fov` walks the channel start->end across the
        // flight) but may be any expression in `t` (`fov_from zoom.fov(1-t)` reverses it,
        // `(t*t)` eases in, …). `t` is the ONLY variable in scope here: a surface intrinsic
        // (x/y/z/u/v/…) in the driver — or in the channel's stops — is an out-of-scope
        // error. Record and `_at` track are mutually exclusive per scalar; a record wins.
        struct RecTrack {
            int recIdx = -1, chanIdx = -1;
            std::vector<PatNode> driver;
            bool active() const { return recIdx >= 0; }
        };
        bool recOk = true;
        auto readRecTrack = [&](const char* key, RecTrack& rt) {
            const Stmt* s = find(b, key);
            if (!s || s->val.words.empty()) return;
            // Rejoin the value tokens (a driver written with spaces, `zoom.fov(1 - t)`,
            // splits across words); strip whitespace — the RECORD.channel head has none and
            // the pattern compiler ignores it inside the driver.
            std::string joined = s->val.words[0];
            for (size_t k = 1; k < s->val.words.size(); ++k) joined += s->val.words[k];
            std::string tok; for (char c : joined) if (!std::isspace((unsigned char)c)) tok += c;
            std::string headChan = tok, dexpr = "t";       // default driver = the raw timeline
            size_t lp = tok.find('(');
            if (lp != std::string::npos) {
                size_t rp = tok.rfind(')');
                if (rp == std::string::npos || rp <= lp) {
                    fail("camera_curve '" + base + "' " + key + " '" + tok + "': malformed `RECORD.channel(driver)`");
                    recOk = false; return;
                }
                headChan = tok.substr(0, lp);
                dexpr    = tok.substr(lp + 1, rp - lp - 1);
            }
            size_t dot = headChan.find('.');
            if (dot == std::string::npos) {
                fail("camera_curve '" + base + "' " + key + " needs `RECORD.channel[(driver)]`");
                recOk = false; return;
            }
            std::string rname = headChan.substr(0, dot), chan = headChan.substr(dot + 1);
            auto rit = recordIndex_.find(rname);
            if (rit == recordIndex_.end()) {
                fail("camera_curve '" + base + "' " + key + ": unknown record '" + rname + "'");
                recOk = false; return;
            }
            const Record& rec = L.scene.records[(size_t)rit->second];
            int ci = rec.channelIndex(chan);
            if (ci < 0) {
                fail("camera_curve '" + base + "' " + key + ": record '" + rname + "' has no channel '" + chan + "'");
                recOk = false; return;
            }
            if (rec.channels[(size_t)ci].kind != ChanKind::Scalar) {
                fail("camera_curve '" + base + "' " + key + ": channel '" + chan + "' is a colour channel, not scalar");
                recOk = false; return;
            }
            for (const RecStop& st : rec.channels[(size_t)ci].stops)
                if (patternHasFreeVars(st.expr)) {
                    fail("camera_curve '" + base + "' " + key + ": channel '" + chan + "' has stop expressions with "
                         "per-hit surface variables — only the flyby timeline `t` is in scope here");
                    recOk = false; return;
                }
            std::vector<PatNode> drv; std::string cerr;
            // `&tableScope_`: recSample below binds the scene's tables, so a flyby track
            // can be driven by MEASURED data over the timeline — `fov_from lens.fov(grid:zoom(t))`.
            if (!compilePatternExpr(dexpr, drv, cerr, /*allowT=*/true, nullptr, &tableScope_)) {
                fail("camera_curve '" + base + "' " + key + " driver '" + dexpr + "': " + cerr);
                recOk = false; return;
            }
            if (patternHasFreeVars(drv)) {
                fail("camera_curve '" + base + "' " + key + " driver '" + dexpr +
                     "': references a surface variable — only the flyby timeline `t` is in scope here");
                recOk = false; return;
            }
            rt.recIdx = rit->second; rt.chanIdx = ci; rt.driver = std::move(drv);
        };
        RecTrack fovRec, rollRec, zoomRec, fstopRec, focusRec;
        readRecTrack("fov_from",   fovRec);
        readRecTrack("roll_from",  rollRec);
        readRecTrack("zoom_from",  zoomRec);
        readRecTrack("fstop_from", fstopRec);
        readRecTrack("focus_from", focusRec);
        if (!recOk) return false;
        // Sample a record track at timeline position `fr`: evaluate its driver (in t), then
        // the record channel at that driven value. Stops are constant (checked above), so
        // the PatCtx only needs `t`.
        auto recSample = [&](const RecTrack& rt, double fr) -> double {
            PatCtx c{}; c.t = fr;
            bindPatData(c, L.scene);   // grid:/scatter: in the driver (see the compile site)
            double d = rt.driver.empty() ? fr : patternEval(rt.driver.data(), (int)rt.driver.size(), c);
            const Record& rec = L.scene.records[(size_t)rt.recIdx];
            return recSampleScalar(rec, rec.channels[(size_t)rt.chanIdx], d, c);
        };

        // ---- Orientation axes (camera_curve bridge, milestone M13) ------------------
        // Full 3-DOF camera rotation is authored as two independent axes; the third is
        // derived by the camera basis (`Camera::lookAt`: right = forward x up, then up is
        // re-orthogonalized). So here we only produce, per frame, a FORWARD direction (into
        // `cs.look = eye + forward`) and a reference UP (into `cs.up`).
        //   * forward (2 DOF): `fwd_at <t> x y z` direction track  >  aim-point
        //     (`look_at`/`look_curve`)  >  path tangent (today's default).
        //   * up (1 DOF): `up_at <t> x y z` vector track  >  reference up + `roll`.
        // Each axis has an optional reference `frame world|travel`: `world` uses fixed world
        // axes (today's behavior); `travel` uses the curve's rotation-minimizing frame (RMF,
        // parallel-transported — banks into turns, no torsion flips; closed loops distribute
        // the holonomy twist for a seamless seam). A `fwd_at`/`up_at` vector is interpreted
        // in that frame's basis (x=right, y=up, z=forward) when `travel`, else as a world
        // direction. Curve-level `frame` sets the default for both; `fwd_frame`/`up_frame`
        // override per axis. With none of these authored the block is byte-identical to the
        // legacy tangent-look + world-up + `roll_at` path.
        bool vecTrkOk = true;
        auto readVecTrack = [&](const char* atKey) -> Vec3Track {
            Vec3Track tk;
            for (const auto& s : b.stmts) {
                if (s.key != atKey) continue;
                s.used = true;
                if (s.val.words.size() < 4) {
                    fail("camera_curve '" + base + "' " + atKey + " needs: <t> <x> <y> <z>");
                    vecTrkOk = false; continue;
                }
                tk.keys.push_back({num(s.val.words[0]),
                                   Vec3{num(s.val.words[1]), num(s.val.words[2]), num(s.val.words[3])}});
            }
            tk.sort();
            return tk;
        };
        Vec3Track fwdTrk = readVecTrack("fwd_at");
        Vec3Track upTrk  = readVecTrack("up_at");
        if (!vecTrkOk) return false;

        auto parseFrameKw = [&](const char* key, bool def, bool& out) -> bool {
            const Stmt* s = find(b, key);
            if (!s) { out = def; return true; }
            const std::string& v = s->val.words.empty() ? std::string() : s->val.words[0];
            if      (v == "world")  out = false;
            else if (v == "travel") out = true;
            else { fail("camera_curve '" + base + "' " + key + " must be world|travel"); return false; }
            return true;
        };
        bool defTravel = false, fwdFrameTravel = false, upFrameTravel = false;
        if (!parseFrameKw("frame",     false,     defTravel))     return false;
        if (!parseFrameKw("fwd_frame", defTravel, fwdFrameTravel)) return false;
        if (!parseFrameKw("up_frame",  defTravel, upFrameTravel))  return false;
        // The RMF is only needed when some axis actually references the travel frame: a
        // `fwd_at`/`up_at` vector *in* travel mode, or an up axis whose reference (the
        // default up when no `up_at`) is the travel frame. `fwd_at world` / `up_at world`
        // never touch it, so a legacy curve skips the whole precompute.
        bool needRMF = (fwdTrk.active() && fwdFrameTravel) || upFrameTravel;

        // Base (constant) values a track falls back to and that the whole-flight optics
        // were derived from. Captured from the same keywords readFilmExposure consumed so
        // per-frame re-derivation starts from the authored baseline (never double-applies).
        double rollConst  = dblOf(b, "roll",  0.0);
        double baseFovDeg = dblOf(b, "fov_y", 40.0);
        double baseLensMM = dblOf(b, "lens",  0.0);
        double baseZoom   = dblOf(b, "zoom",  1.0);
        double baseFstop  = dblOf(b, "fstop", 0.0);
        double hmm        = (shared.filmH_mm > 0.0) ? shared.filmH_mm : 24.0;
        double baseFocus  = shared.focus;   // metres (Len-scaled)
        bool haveRoll   = rollTrk.active() || find(b, "roll") || rollRec.active();
        bool haveOptics = fovTrk.active() || zoomTrk.active() ||
                          fstopTrk.active() || focusTrk.active() ||
                          fovRec.active() || zoomRec.active() ||
                          fstopRec.active() || focusRec.active();
        const double DEG = 3.141592653589793 / 180.0;

        bool pathLock = parseExposureLock(b, shared);
        const int pathGroup = (int)L.cameras.size();

        // ---- Tangent look-direction pre-pass (fold-robust) -------------------------
        // Computed ahead of the main frame loop because temporal smoothing needs the whole
        // sequence. For look_curve / look_at modes this stays empty and the loop uses those
        // targets directly. The min_reach floor prevents a folded chord's collapsing
        // horizontal reach from raking the pitch into the ceiling/floor; look_smooth then
        // spreads a fold's unavoidable fast pan over neighbouring frames.
        std::vector<Vec3> tangentDirs;
        if (!lookCurve && !lookFixed) {
            const double lookAheadFrac = 0.045;                 // shared with the note above
            const double hMin = std::max(0.0, minReachFrac) * lookAheadFrac * Smax;
            std::vector<double> yawA((size_t)N), pitA((size_t)N);
            for (int i = 0; i < N; ++i) {
                double fr = closed ? ((double)i / N) : (N == 1 ? 0.5 : (double)i / (N - 1));
                double g = invertC(fr * Cmax);
                Vec3 eye = catmullRomAt(pts, closed, g, splineAlpha);
                double sHere = arcAtG(g);
                double sTgt  = sHere + lookAheadFrac * Smax;
                double gTgt  = closed ? gAtArc(std::fmod(sTgt, Smax)) : gAtArc(std::min(sTgt, Smax));
                Vec3 tan = catmullRomAt(pts, closed, gTgt, splineAlpha) - eye;
                if (length(tan) <= 1e-9) {   // degenerate (end of an open curve): look forward from behind
                    Vec3 a = catmullRomAt(pts, closed, std::max(0.0, g - (double)nSeg / (M * 4.0)), splineAlpha);
                    tan = eye - a;
                }
                double h = std::sqrt(tan.x * tan.x + tan.z * tan.z);
                yawA[(size_t)i] = std::atan2(tan.x, tan.z);          // bearing in xz
                pitA[(size_t)i] = std::atan2(tan.y, std::max(h, hMin));   // floored-reach pitch
            }
            auto rebuild = [&](const std::vector<double>& yawS, const std::vector<double>& pitS) {
                tangentDirs.resize((size_t)N);
                for (int i = 0; i < N; ++i) {
                    double cp = std::cos(pitS[(size_t)i]);
                    Vec3 d{ cp * std::sin(yawS[(size_t)i]), std::sin(pitS[(size_t)i]), cp * std::cos(yawS[(size_t)i]) };
                    tangentDirs[(size_t)i] = (length(d) > 1e-12) ? normalize(d) : Vec3{0, 0, -1};
                }
            };
            if (lookSmooth > 1e-6 && N >= 3) {
                const double PI = 3.141592653589793, TWO_PI = 6.283185307179586;
                int r = std::max(1, (int)std::lround(3.0 * lookSmooth));
                std::vector<double> w((size_t)(2 * r + 1));
                for (int k = -r; k <= r; ++k)
                    w[(size_t)(k + r)] = std::exp(-(double)(k * k) / (2.0 * lookSmooth * lookSmooth));
                std::vector<double> yawS((size_t)N), pitS((size_t)N);
                for (int i = 0; i < N; ++i) {
                    double ay = 0, ap = 0, ws = 0;
                    for (int k = -r; k <= r; ++k) {
                        int j = i + k, jj;
                        if (closed) jj = ((j % N) + N) % N; else jj = std::min(std::max(j, 0), N - 1);
                        double vy = yawA[(size_t)jj];
                        while (vy - yawA[(size_t)i] >  PI) vy -= TWO_PI;   // wrap-aware: nearest branch
                        while (vy - yawA[(size_t)i] < -PI) vy += TWO_PI;
                        double wk = w[(size_t)(k + r)];
                        ay += vy * wk; ap += pitA[(size_t)jj] * wk; ws += wk;
                    }
                    yawS[(size_t)i] = ay / ws; pitS[(size_t)i] = ap / ws;
                }
                rebuild(yawS, pitS);
            } else {
                rebuild(yawA, pitA);
            }
        }

        // ---- Rotation-minimizing frame (RMF) pre-pass ------------------------------
        // Parallel-transport an "up" reference along the curve so the travel frame twists
        // as little as possible (Bishop/RMF, not Frenet — no torsion flips). Built with the
        // double-reflection method (Wang, Jüttler, Zheng & Liu 2008), which is exact to
        // second order and robust at inflections. For a CLOSED loop the transported frame
        // generally does not return to itself (holonomy); the residual twist is measured and
        // distributed linearly along the loop so the seam is seamless (same idea as the
        // sweep engine's closed-spine frame, DESIGN.md §7a). Frames align with the same `fr`
        // sampling as the render loop. rmfRight[i] = tangent x rmfUp is filled for basis use.
        std::vector<Vec3> rmfTan, rmfUp, rmfRight;
        if (needRMF) {
            rmfTan.resize((size_t)N); rmfUp.resize((size_t)N); rmfRight.resize((size_t)N);
            auto frAt   = [&](int i){ return closed ? ((double)i / N) : (N == 1 ? 0.5 : (double)i / (N - 1)); };
            auto tanAtG = [&](double g) -> Vec3 {
                double eps = (nSeg > 0) ? (double)nSeg / (M * 2.0) : 1e-3;
                Vec3 a = catmullRomAt(pts, closed, closed ? g - eps : std::max(0.0, g - eps), splineAlpha);
                Vec3 c = catmullRomAt(pts, closed, closed ? g + eps : std::min((double)nSeg, g + eps), splineAlpha);
                Vec3 d = c - a;
                return (length(d) > 1e-12) ? normalize(d) : Vec3{0, 0, -1};
            };
            std::vector<Vec3> eyeAt((size_t)N);
            for (int i = 0; i < N; ++i) {
                double g = invertC(frAt(i) * Cmax);
                eyeAt[(size_t)i] = catmullRomAt(pts, closed, g, splineAlpha);
                rmfTan[(size_t)i] = tanAtG(g);
            }
            // Seed: project the world reference up perpendicular to the first tangent; if the
            // path starts dead-vertical (up ~parallel to tangent), fall back to a world axis.
            auto perp = [&](const Vec3& ref, const Vec3& t) -> Vec3 {
                Vec3 r = ref - t * dot(ref, t);
                if (length(r) < 1e-9) { Vec3 alt = (std::fabs(t.x) < 0.9) ? Vec3{1,0,0} : Vec3{0,1,0};
                                        r = alt - t * dot(alt, t); }
                return normalize(r);
            };
            rmfUp[0] = perp(shared.up, rmfTan[0]);
            // Double-reflection transport of the reference vector r along the samples.
            auto transport = [&](const Vec3& r_i, const Vec3& x_i, const Vec3& t_i,
                                 const Vec3& x_j, const Vec3& t_j) -> Vec3 {
                Vec3 v1 = x_j - x_i; double c1 = dot(v1, v1);
                if (c1 < 1e-24) return r_i;                       // coincident samples: no rotation
                Vec3 rL = r_i - v1 * (2.0 / c1 * dot(v1, r_i));
                Vec3 tL = t_i - v1 * (2.0 / c1 * dot(v1, t_i));
                Vec3 v2 = t_j - tL; double c2 = dot(v2, v2);
                if (c2 < 1e-24) return rL;
                return rL - v2 * (2.0 / c2 * dot(v2, rL));
            };
            for (int i = 1; i < N; ++i)
                rmfUp[(size_t)i] = normalize(transport(rmfUp[(size_t)i-1],
                                    eyeAt[(size_t)i-1], rmfTan[(size_t)i-1],
                                    eyeAt[(size_t)i],   rmfTan[(size_t)i]));
            if (closed && N >= 2) {
                // Transport once more from the last frame back onto frame 0's tangent to read
                // the holonomy angle between the wrapped-around up and the seed up.
                Vec3 wrap = normalize(transport(rmfUp[(size_t)N-1], eyeAt[(size_t)N-1], rmfTan[(size_t)N-1],
                                                 eyeAt[0], rmfTan[0]));
                Vec3 t0 = rmfTan[0], u0 = rmfUp[0], r0 = cross(t0, u0);
                double ang = std::atan2(dot(wrap, r0), dot(wrap, u0));   // signed twist about t0
                // Distribute -ang * (i/N) so frame N would land exactly on the seed.
                for (int i = 0; i < N; ++i)
                    rmfUp[(size_t)i] = normalize(rotateAboutAxis(rmfUp[(size_t)i], rmfTan[(size_t)i],
                                                                 -ang * ((double)i / N)));
            }
            for (int i = 0; i < N; ++i)
                rmfRight[(size_t)i] = normalize(cross(rmfTan[(size_t)i], rmfUp[(size_t)i]));
        }

        // ---- Round-trip capture: record this curve's CONTROL POINTS for the editor ----
        // The in-viewer camera_curve editor seeds its `editPts` from this so an existing
        // curve can be loaded and edited in place (rather than starting from an empty
        // editor). Per control point we store the eye, a unit look direction (sampled from
        // whichever orientation mode the curve uses), and — if a density was authored — the
        // local rho so the editor's speed track round-trips too.
        {
            AuthoredCurve ac;
            ac.name   = base;
            ac.up     = shared.up;
            ac.fov    = shared.fov;
            ac.mode   = shared.mode;
            ac.closed = closed;
            ac.eyes   = pts;
            const int nPts = (int)pts.size();
            ac.fwds.reserve((size_t)nPts);
            for (int i = 0; i < nPts; ++i) {
                double gi = (double)i;                                  // control point i sits at g = i
                double ui = (nSeg > 0) ? gi / (double)nSeg : 0.0;       // its normalized timeline position
                Vec3 dir{0, 0, -1};
                if (lookCurve && lookPts.size() >= 2) {
                    dir = catmullRomAt(lookPts, closed, ui * (double)lookSeg, splineAlpha) - pts[(size_t)i];
                } else if (lookFixed) {
                    dir = fixedLook - pts[(size_t)i];
                } else {                                                // tangent: central difference along the eye spline
                    double eps = (nSeg > 0) ? (double)nSeg / 256.0 : 1e-3;
                    Vec3 a = catmullRomAt(pts, closed, std::max(0.0, gi - eps), splineAlpha);
                    Vec3 c = catmullRomAt(pts, closed, std::min((double)nSeg, gi + eps), splineAlpha);
                    dir = c - a;
                }
                ac.fwds.push_back((length(dir) > 1e-9) ? normalize(dir) : Vec3{0, 0, -1});
            }
            if (haveDensity) {
                ac.density.reserve((size_t)nPts);
                for (int i = 0; i < nPts; ++i)
                    ac.density.push_back(densityAt((nSeg > 0) ? (double)i / (double)nSeg : 0.0));
            }
            L.authoredCurves.push_back(std::move(ac));
        }

        int pad = 1; for (int f = N - 1; f >= 10; f /= 10) ++pad;   // zero-pad width
        for (int i = 0; i < N; ++i) {
            // A closed loop samples i/N (frame N == frame 0, not duplicated); an open
            // curve spans both endpoints via i/(N-1).
            double fr = closed ? ((double)i / N)
                               : (N == 1 ? 0.5 : (double)i / (N - 1));
            double g = invertC(fr * Cmax);
            CamSpec cs = shared;
            cs.eye = catmullRomAt(pts, closed, g, splineAlpha);
            // Forward axis: default aim (look_curve / look_at / tangent), then an optional
            // `fwd_at` direction override (world vector, or travel-frame components).
            if (lookCurve) {
                cs.look = catmullRomAt(lookPts, closed, fr * lookSeg, splineAlpha);
            } else if (lookFixed) {
                cs.look = fixedLook;
            } else {   // tangent: aim a fixed arc-length ahead. A differential finite-difference
                // tangent is hypersensitive to local spline wiggle where control points cluster
                // (e.g. the channel-threading zigzag), so aiming at an absolute point a fair
                // arc-distance ahead averages that wiggle out into a smooth "flying down the
                // path" motion. Direction (plus the fold-robust min_reach / look_smooth
                // treatment) is precomputed in tangentDirs above.
                cs.look = cs.eye + tangentDirs[(size_t)i];
            }
            if (fwdTrk.active()) {
                Vec3 fv = fwdTrk.sample(fr);
                Vec3 fwd = fwdFrameTravel
                    ? (rmfRight[(size_t)i] * fv.x + rmfUp[(size_t)i] * fv.y + rmfTan[(size_t)i] * fv.z)
                    : fv;
                if (length(fwd) > 1e-12) cs.look = cs.eye + normalize(fwd);
            }
            // Per-frame lens: re-derive optics from the animated fov/zoom/f-stop/focus.
            // cs starts as `shared`, so restore its aperture before re-deriving in case a
            // static f-stop had already set it (a live fstop track will overwrite it).
            if (haveOptics) {
                // Per scalar: a record track (5b) wins over an `_at` keyframe track, which
                // wins over the authored base constant.
                double fov = fovRec.active()   ? recSample(fovRec,   fr) : fovTrk.sample(fr, baseFovDeg);
                double zm  = zoomRec.active()  ? recSample(zoomRec,  fr) : zoomTrk.sample(fr, baseZoom);
                double fs  = fstopRec.active() ? recSample(fstopRec, fr) : fstopTrk.sample(fr, baseFstop);
                double fo  = focusRec.active() ? Len(recSample(focusRec, fr))
                                               : (focusTrk.active() ? Len(focusTrk.sample(fr, 0.0)) : baseFocus);
                cs.aperture = shared.aperture;
                cs.focus    = fo;
                deriveCameraOptics(cs, fov, baseLensMM, zm, fs, hmm, fo);
            }
            // Up axis: an explicit `up_at` vector wins; otherwise the reference up (world
            // `up`, or the travel-frame RMF up) with `roll` banked on top about the view
            // ray. camera.h re-orthogonalizes this reference against forward, so it need not
            // be exactly perpendicular. Byte-identical to the legacy roll path when no
            // `up_at`/travel frame is authored (reference up == shared.up).
            {
                Vec3 w = cs.look - cs.eye;
                bool wok = length(w) > 1e-12;
                if (upTrk.active()) {
                    Vec3 uv = upTrk.sample(fr);
                    Vec3 up = upFrameTravel
                        ? (rmfRight[(size_t)i] * uv.x + rmfUp[(size_t)i] * uv.y + rmfTan[(size_t)i] * uv.z)
                        : uv;
                    if (length(up) > 1e-12) cs.up = up;
                } else {
                    Vec3 refUp = upFrameTravel ? rmfUp[(size_t)i] : shared.up;
                    if (haveRoll && wok) {
                        double rollDeg = rollRec.active() ? recSample(rollRec, fr) : rollTrk.sample(fr, rollConst);
                        cs.up = rotateAboutAxis(refUp, normalize(w), rollDeg * DEG);
                    } else {
                        cs.up = refUp;
                    }
                }
            }
            char num5[8]; std::snprintf(num5, sizeof(num5), "%0*d", pad, i);
            cs.name = base + num5;
            cs.pathGroup = pathGroup;
            cs.exposureLock = pathLock;
            L.cameras.push_back(cs);
            if (!L.hasCamera) {
                L.camEye = cs.eye; L.camLook = cs.look; L.camUp = cs.up;
                L.camFov = cs.fov; L.camAperture = cs.aperture; L.camFocus = cs.focus;
                if (cs.mode) L.mode = cs.mode;
                if (cs.res > 0) L.res = cs.res;
                L.hasCamera = true;
            }
        }
        return true;
    }

    // ---- render controls (overridable by CLI later) ----
    bool applyRender(const Block& b, Loaded& L) {
        const Stmt* p = find(b, "photons");
        if (p && !p->val.words.empty()) L.photons = std::atoll(p->val.words[0].c_str());
        std::string dev = strOf(b, "device");
        if (!dev.empty()) L.device = dev;
        std::string md = strOf(b, "mode");
        if (!md.empty()) L.mode = normMode(md, L);
        std::string o = strOf(b, "out");
        if (!o.empty()) L.out = o;
        const Stmt* r = find(b, "res");
        if (r && !r->val.words.empty()) L.res = (int)num(r->val.words[0]);
        const Stmt* mb = find(b, "max_bounce");
        if (mb && !mb->val.words.empty()) {
            int n = (int)num(mb->val.words[0]);
            if (n < 1) { fail("render: max_bounce must be >= 1"); return false; }
            L.maxBounce = n;
        }
        return true;
    }
};

// A caller-supplied capability predicate for `prefer{}/else{}` resolution: given a
// freshly-built scene it returns a reason string if the scene is NOT renderable (some
// feature unsupported by the mode it would render in), or nullptr if it is fine. main.cpp
// supplies this using its per-mode support gates (BDPT/VCM/fisheye). Empty => no filtering
// (the first / most-preferred branch always wins).
using SupportFn = std::function<const char*(const Loaded&)>;

// Splice a flat block list: each top-level `prefer` node (at preferIdx[k]) is replaced by
// the blocks of its choice[k]-th branch; all other blocks pass through unchanged.
inline std::vector<Block> flattenPrefer(const std::vector<Block>& blocks,
                                        const std::vector<size_t>& preferIdx,
                                        const std::vector<int>& choice) {
    std::vector<Block> flat;
    size_t k = 0;
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (k < preferIdx.size() && preferIdx[k] == i) {
            const auto& branch = blocks[i].branches[(size_t)choice[k]];
            for (const auto& bb : branch) flat.push_back(bb);
            ++k;
        } else {
            flat.push_back(blocks[i]);
        }
    }
    return flat;
}

// Load an FTSL file, populating `L`. Returns false and sets `err` on any error. When the
// scene contains `prefer{}/else{}` blocks, `supported` chooses which branch renders (see
// SupportFn): the first branch whose spliced scene is fully renderable wins, falling back
// to the last branch when none are (a loud mode error then fires at render time).
// Load an FTSL scene from an in-memory source string (rather than a file). `nameForMsgs`
// is used only for diagnostics (grammar-shim path label, error messages). This is the
// shared core of `load()`; it also backs the synthesized quick-viewer scene (a bare
// `ftrace foo.glb` builds an auto-lit scene string and loads it through here).
inline bool loadSource(const std::string& src, const std::string& nameForMsgs,
                       Loaded& L, std::string& err,
                       const SupportFn& supported = {}) {
    // Report the keys nothing in the loader read. A warning rather than an error:
    // the check is new, and an old scene carrying a stale property should still
    // render — but it must SAY so, because the alternative (today's behaviour) is
    // that the property silently does nothing and the author blames the renderer.
    auto reportUnknownKeys = [&](const Loaded& out) {
        for (const std::string& w : out.unknownKeys)
            std::fprintf(stderr, "[ftsl] warning: %s: %s\n", nameForMsgs.c_str(), w.c_str());
    };

    // The shared grammar (src/gpda/ftsl_frontend.hpp) is the only front end.
    std::vector<Block> blocks;
    if (!ftsl_gpda::parse(src, blocks, err)) return false;

    // Collect top-level `prefer` nodes. The common case (none) is the original fast path.
    std::vector<size_t> preferIdx;
    for (size_t i = 0; i < blocks.size(); ++i)
        if (blocks[i].type == "prefer") preferIdx.push_back(i);

    if (preferIdx.empty()) {
        Builder bld;
        if (!bld.build(blocks, L)) { err = bld.err; return false; }
        reportUnknownKeys(L);
        return true;
    }

    // Validate: every prefer must have >=1 branch and must not nest another prefer inside
    // a branch (use flat `else` chaining instead — keeps resolution non-circular).
    for (size_t idx : preferIdx) {
        if (blocks[idx].branches.empty()) { err = "'prefer' has no branches"; return false; }
        for (const auto& branch : blocks[idx].branches)
            for (const auto& bb : branch)
                if (bb.type == "prefer") {
                    err = "nested 'prefer' inside a branch is not supported; use "
                          "'prefer { } else { } else { }' chaining instead";
                    return false;
                }
    }

    // Try-build a candidate (fresh Builder each time). Returns:
    //   built=false           -> the branch's scene has a real authoring error (buildErr set)
    //   built=true, reason==0  -> renderable
    //   built=true, reason!=0  -> builds but the mode can't render some feature
    struct Trial { bool built; std::string buildErr; const char* reason; };
    auto tryBuild = [&](const std::vector<int>& ch, Loaded& out) -> Trial {
        std::vector<Block> flat = flattenPrefer(blocks, preferIdx, ch);
        Builder bld;
        if (!bld.build(flat, out)) return {false, bld.err, nullptr};
        return {true, {}, supported ? supported(out) : nullptr};
    };

    // Greedy per-node resolution (nodes fixed left-to-right; the realistic case is a
    // single node). For each node pick the first branch that yields a renderable scene,
    // else keep the last branch.
    //
    // Single-node fast path: with exactly one `prefer`, the trial that resolves the
    // node IS the final scene (its flattened block list == the resolved one), so we
    // keep that trial's `Loaded` and skip a redundant final rebuild — which for a
    // heavy scene would otherwise re-parse and RE-LOAD every mesh a second time.
    const bool singleNode = (preferIdx.size() == 1);
    std::vector<int> choice(preferIdx.size(), 0);
    Loaded accepted;
    bool haveAccepted = false;
    for (size_t j = 0; j < preferIdx.size(); ++j) {
        int nb = (int)blocks[preferIdx[j]].branches.size();
        int chosen = nb - 1;
        for (int c = 0; c < nb; ++c) {
            choice[j] = c;
            Loaded trial;
            Trial t = tryBuild(choice, trial);
            const bool renderable = (t.built && t.reason == nullptr);
            // For a single node, whichever branch we end on (first renderable, or the
            // last as fallback) is `chosen`, and `trial` currently holds its build.
            if (singleNode && (renderable || c == nb - 1)) {
                accepted = std::move(trial);
                haveAccepted = true;
            }
            if (renderable) { chosen = c; break; }   // renderable -> take it
            if (c < nb - 1) {
                const char* why = t.built ? t.reason : t.buildErr.c_str();
                std::fprintf(stderr, "[prefer] branch %d rejected (%s); trying the next\n",
                             c + 1, why ? why : "unrenderable");
            }
        }
        choice[j] = chosen;
    }

    if (singleNode && haveAccepted) {
        L = std::move(accepted);
    } else {
        // Multi-node: rebuild once with the fully-resolved choices across all nodes.
        std::vector<Block> flat = flattenPrefer(blocks, preferIdx, choice);
        Builder bld;
        if (!bld.build(flat, L)) { err = bld.err; return false; }
    }
    for (size_t j = 0; j < preferIdx.size(); ++j)
        std::printf("[prefer] using branch %d of %d\n",
                    choice[j] + 1, (int)blocks[preferIdx[j]].branches.size());
    reportUnknownKeys(L);
    return true;
}

inline bool load(const std::string& path, Loaded& L, std::string& err,
                 const SupportFn& supported = {}) {
    std::ifstream f(path);
    if (!f) { err = "cannot open scene file: " + path; return false; }
    std::stringstream ss; ss << f.rdbuf();
    std::string src = ss.str();
    return loadSource(src, path, L, err, supported);
}

} // namespace ftsl
