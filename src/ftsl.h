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
//                   lens <mm>  fstop <N>  zoom <x>  # photographic authoring (overrides fov_y/aperture)
//                   projection <name> | fisheye [type]  # lens projection (rectilinear default; §8.5)
//                   film { res W H  format <name>  size <Wmm> <Hmm>  iso .. shutter .. exposure .. } }
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
// `rgb r g b`, `whitewall [r]`, `redwall`, `greenwall`, `glass:BK7|SF10`,
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
#include "scene.h"
#include "camera.h"
#include "spectrum.h"
#include "lights.h"
#include "materials.h"
#include "mesh.h"
#include "upsample.h"

namespace ftsl {

// A token is a number iff strtod consumes all of it (handles -1, 0.999, 1e30).
inline bool isNumber(const std::string& s) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end == s.c_str() + s.size();
}
inline double num(const std::string& s) { return std::strtod(s.c_str(), nullptr); }

// Load a measured spectrum from a CSV/whitespace file into (wavelength_nm, value)
// pairs. This is the runtime "measured-SPD loader": the ingestion point for the
// authoritative data mirrored under data/ (see data/README.md). Format is liberal —
// lines beginning with '#' are comments, fields are separated by comma OR whitespace,
// and any line whose first two fields do not both parse as numbers (e.g. a
// `wavelength_nm,relative_power` header row) is skipped. The first numeric field is
// the wavelength in nanometres, the second is the (relative or absolute) value; extra
// columns are ignored. Values are taken verbatim — an emission SPD's absolute scale is
// irrelevant (the power law renormalises it), and a reflectance file should already be
// in 0..1. Returns false with `err` set on an unreadable/empty file.
inline bool loadSpdCsv(const std::string& path,
                       std::vector<std::pair<double, double>>& out,
                       std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open spectrum file: " + path; return false; }
    out.clear();
    std::string line;
    while (std::getline(f, line)) {
        // Strip a trailing CR (CRLF files) and an inline '#' comment.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        // Turn commas into spaces so a single stream reader handles both delimiters.
        for (char& c : line) if (c == ',' || c == '\t') c = ' ';
        std::istringstream ss(line);
        std::string a, b;
        if (!(ss >> a >> b)) continue;                 // blank / single-field line
        if (!isNumber(a) || !isNumber(b)) continue;    // header row or junk -> skip
        out.push_back({num(a), num(b)});
    }
    if (out.empty()) { err = "spectrum file has no numeric rows: " + path; return false; }
    return true;
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
enum class Tok { Word, String, LBrace, RBrace, Newline, End };
struct Token { Tok kind; std::string text; int line; };

inline std::vector<Token> tokenize(const std::string& src) {
    std::vector<Token> out;
    int line = 1;
    size_t i = 0, n = src.size();
    while (i < n) {
        char c = src[i];
        if (c == '\n') { out.push_back({Tok::Newline, "\n", line}); ++line; ++i; continue; }
        if (c == '\r' || c == ' ' || c == '\t') { ++i; continue; }
        if (c == '#') { while (i < n && src[i] != '\n') ++i; continue; }
        if (c == '{') { out.push_back({Tok::LBrace, "{", line}); ++i; continue; }
        if (c == '}') { out.push_back({Tok::RBrace, "}", line}); ++i; continue; }
        if (c == '"') {
            ++i; std::string s;
            while (i < n && src[i] != '"') { if (src[i] == '\n') ++line; s += src[i++]; }
            if (i < n) ++i;   // closing quote
            out.push_back({Tok::String, s, line});
            continue;
        }
        // Bareword: accrete until whitespace/brace/comment/quote/newline.
        std::string w;
        while (i < n) {
            char d = src[i];
            if (d == ' ' || d == '\t' || d == '\r' || d == '\n' ||
                d == '{' || d == '}' || d == '#' || d == '"') break;
            w += d; ++i;
        }
        out.push_back({Tok::Word, w, line});
    }
    out.push_back({Tok::End, "", line});
    return out;
}

// ---------------------------------------------------------------------------
// Parse tree
// ---------------------------------------------------------------------------
struct Block;
struct Value {
    std::vector<std::string> words;    // scalar / vector / expression tokens
    std::shared_ptr<Block> block;      // nested brace block (table/film/coat/body/...)
};
struct Stmt { std::string key; Value val; int line = 0; };
struct Block {
    std::string type;                  // material / quad / film / table / ...
    std::string subtype;               // light "area" / "collimated"
    std::string name;                  // quoted name, if any
    std::vector<Stmt> stmts;           // newline-structured statements
    std::vector<std::string> words;    // flat token dump (for table/palette lists)
};

struct Parser {
    std::vector<Token> t;
    size_t i = 0;
    std::string err;

    const Token& cur() const { return t[i]; }
    bool is(Tok k) const { return t[i].kind == k; }
    void adv() { if (t[i].kind != Tok::End) ++i; }
    void skipNewlines() { while (is(Tok::Newline)) adv(); }
    void fail(const std::string& m) { if (err.empty()) err = "line " + std::to_string(cur().line) + ": " + m; }

    // Read the value part of a statement. This must work when several `key value`
    // pairs share one line (e.g. `quad { origin 0 0 0  u 1 0 0  material white }`),
    // so a value cannot simply run to the newline. Rule: take the first token
    // unconditionally (a value always has one — a number, a name, or a spectrum
    // keyword), then keep consuming *continuation* tokens — numbers or `key=val`
    // named params (gaussian/shortpass) — and stop at the next bareword, which
    // begins the next statement's key. A trailing `{` opens a nested brace block
    // (table/film/…) whose type is the preceding word, or the statement key.
    void parseValue(const std::string& key, Value& v) {
        if (is(Tok::Word) || is(Tok::String)) { v.words.push_back(cur().text); adv(); }
        while (is(Tok::Word)) {
            const std::string& tx = cur().text;
            bool cont = isNumber(tx) || tx.find('=') != std::string::npos;
            if (!cont) break;
            v.words.push_back(tx); adv();
        }
        if (is(Tok::LBrace)) {
            std::string btype = key;
            if (!v.words.empty()) { btype = v.words.back(); v.words.pop_back(); }
            v.block = std::make_shared<Block>();
            v.block->type = btype;
            parseBraceBody(*v.block);
        }
    }

    // Parse "{ ... }" body into stmts (+ flat words). Assumes cur() == LBrace.
    void parseBraceBody(Block& b) {
        adv();   // consume '{'
        while (!is(Tok::RBrace) && !is(Tok::End)) {
            if (is(Tok::Newline)) { adv(); continue; }
            Stmt s; s.line = cur().line;
            s.key = cur().text; adv();
            b.words.push_back(s.key);
            parseValue(s.key, s.val);
            for (const auto& w : s.val.words) b.words.push_back(w);
            b.stmts.push_back(std::move(s));
        }
        if (is(Tok::RBrace)) adv();
        else fail("unterminated '{'");
    }

    // Parse the whole file into a list of top-level blocks.
    std::vector<Block> parseTop() {
        std::vector<Block> blocks;
        skipNewlines();
        while (!is(Tok::End) && err.empty()) {
            if (!is(Tok::Word)) { fail("expected a block type"); break; }
            Block b; b.type = cur().text; adv();
            if (is(Tok::String)) { b.name = cur().text; adv(); }
            // Optional bareword subtype (light area / light collimated), but not '='.
            if (is(Tok::Word) && cur().text != "=") { b.subtype = cur().text; adv(); }
            if (b.type == "spectrum") {
                if (is(Tok::Word) && cur().text == "=") adv();
                else { fail("spectrum declaration needs '='"); break; }
                Stmt s; s.key = "="; s.line = cur().line;
                parseValue("=", s.val);
                b.stmts.push_back(std::move(s));
            } else {
                if (!is(Tok::LBrace)) { fail("expected '{' after " + b.type); break; }
                parseBraceBody(b);
            }
            blocks.push_back(std::move(b));
            skipNewlines();
        }
        return blocks;
    }
};

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
    for (const auto& s : b.stmts) if (s.key == key) return &s;
    return nullptr;
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
inline Vec3 catmullRomAt(const std::vector<Vec3>& p, bool closed, double g) {
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
    double t2 = t * t, t3 = t2 * t;
    return (P1 * 2.0
          + (P2 - P0) * t
          + (P0 * 2.0 - P1 * 5.0 + P2 * 4.0 - P3) * t2
          + (P1 * 3.0 - P0 - P2 * 3.0 + P3) * t3) * 0.5;
}

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
    // path that authored `exposure_lock` share the auto-exposure anchor computed from
    // the first frame, so a dolly/zoom doesn't flicker as scene brightness shifts.
    // `pathGroup` (>=0) identifies the owning path (all its frames share the value);
    // -1 for a standalone `camera`. `exposureLock` is set on every frame of a locked
    // path. A CLI `-exposure-lock` can additionally force a single shared anchor
    // across *all* rendered cameras regardless of these fields.
    int  pathGroup   = -1;
    bool exposureLock = false;

    // Physical multi-element lens (the "mesh-lens" camera), built from a `lens { ... }`
    // block. When set, main renders this camera through the backward realistic-camera
    // path (mode R), tracing rays through the real glass interfaces. Null => the
    // analytic pinhole/thin-lens camera above.
    std::shared_ptr<LensSystem> lens;
};

struct Loaded {
    Scene scene;
    // All authored cameras, in file order. Phase 3a: any number of `camera` blocks
    // accumulate; main renders the CLI-selected one, or all of them.
    std::vector<CamSpec> cameras;
    // Mirror of the FIRST camera (kept so the pre-Phase-3a single-camera code paths
    // and defaults keep working unchanged).
    bool hasCamera = false;
    Vec3 camEye{0, 1, 3}, camLook{0, 1, 0}, camUp{0, 1, 0};
    double camFov = 40.0, camAperture = 0.02, camFocus = 0.0;
    char mode = 'B';
    long long photons = -1;      // -1 = not specified (CLI default wins)
    int res = -1;                // -1 = not specified
    std::string device;          // empty = not specified
    std::string out;             // empty = not specified
};

class Builder {
public:
    std::string err;

    bool build(const std::vector<Block>& blocks, Loaded& L) {
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
        }

        // Pass 1: collect named spectra (resolve refs lazily), materials, camera.
        for (const auto& b : blocks)
            if (b.type == "spectrum") spectraBlocks_[b.name] = &b;

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

        // Pass 2: materials (must exist before geometry references them).
        for (const auto& b : blocks) {
            if (b.type != "material") continue;
            if (b.name.empty()) { fail("material needs a \"name\""); return false; }
            int id = (int)L.scene.mats.size();
            Material m = buildMaterial(b);
            if (!err.empty()) return false;
            L.scene.mats.push_back(m);
            matIndex_[b.name] = id;
        }

        // Pass 2b: resolve Mix / Layered body child references (now that every name
        // is known). A Layered material's `layer "name" weight` list is its body,
        // resolved by the same second pass as a mix.
        for (const auto& b : blocks) {
            if (b.type != "material") continue;
            int id = matIndex_[b.name];
            MatType t = L.scene.mats[id].type;
            if (t != MatType::Mix && t != MatType::Layered) continue;
            if (!resolveMixChildren(b, L.scene.mats[id], L)) return false;
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
                     b.type == "texture" || b.type == "pattern") { /* handled */ }
            else { fail("unknown top-level block '" + b.type + "'"); return false; }
        }
        // Deferred medium sweep (object-name bounds resolve against the registries).
        for (const Block* mb : mediaBlocks) { if (!addMedium(*mb, L)) return false; }
        if (!haveLight) { fail("scene has no 'light' block"); return false; }
        // Catch errors recorded via fail() inside add* helpers that returned true
        // without re-checking `err` (e.g. an unknown `spd preset:`/`spectrum:` name
        // in a light or material silently falls back otherwise). Any recorded error
        // is fatal — surface it instead of rendering a wrong scene.
        if (!err.empty()) return false;

        // build() finalizes tris/BVH and the emitter set (per-emitter samplers were
        // built in addLight; finalizeEmitters computes powers, the selection CDF,
        // and the combined backward wavelength sampler).
        L.scene.build();
        return true;
    }

private:
    std::unordered_map<std::string, const Block*> spectraBlocks_;
    std::unordered_map<std::string, int> matIndex_;
    std::unordered_map<std::string, int> textureIndex_;   // texture name -> Scene::textures index
    std::unordered_map<std::string, int> patternIndex_;   // pattern name -> Scene::patterns index
    std::unordered_map<std::string, Spectrum> spdFileCache_; // path -> loaded measured SPD

    // Named-object registries for `medium { bounds { object "name" } }` resolution.
    // Populated by the geometry builders during Pass 3; read by addMedium afterwards.
    struct NamedSphere { Vec3 center; double radius; };
    std::unordered_map<std::string, NamedSphere> sphereByName_;   // named sphere -> world center/radius
    std::unordered_map<std::string, int>         implicitByName_; // named isosurface -> Scene::implicits index
    std::unordered_map<std::string, Aabb>        meshAabbByName_; // named mesh -> world AABB

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

    // ---- spectrum evaluation ----
    Spectrum evalSpectrum(const Value& v, int depth = 0) {
        if (depth > 16) { fail("spectrum reference cycle"); return constantSpectrum(0); }
        // table { λ:v … }
        if (v.block && v.block->type == "table") {
            std::vector<std::pair<double, double>> pairs;
            for (const auto& w : v.block->words) {
                auto p = w.find(':');
                if (p == std::string::npos) { fail("table entry '" + w + "' not λ:value"); continue; }
                pairs.push_back({num(w.substr(0, p)), num(w.substr(p + 1))});
            }
            return tabulatedSpectrum(std::move(pairs));
        }
        const auto& w = v.words;
        if (w.empty()) { fail("empty spectrum expression"); return constantSpectrum(0); }
        const std::string& h = w[0];

        if (isNumber(h) && w.size() == 1) return constantSpectrum(num(h));

        if (h == "blackbody")  return blackbody(w.size() > 1 ? num(w[1]) : 6500.0);
        if (h == "ior")        return iorConstant(w.size() > 1 ? num(w[1]) : 1.5);
        if (h == "whitewall")  return whiteWall(w.size() > 1 ? num(w[1]) : 0.75);
        if (h == "redwall")    return redWall();
        if (h == "greenwall")  return greenWall();
        if (h == "gaussian" || h == "shortpass") {
            double a = 0, b = 0, c = 1.0;   // gaussian: center,sigma,amp ; shortpass: edge,slope,amp
            for (size_t k = 1; k < w.size(); ++k) {
                std::string key, val;
                if (!splitEq(w[k], key, val)) continue;
                double x = num(val);
                if      (key == "center" || key == "edge")  a = x;
                else if (key == "sigma"  || key == "slope") b = x;
                else if (key == "amp")                      c = x;
            }
            return (h == "gaussian") ? gaussianBand(a, b, c) : shortPass(a, b, c);
        }
        if (h == "rgb") {
            if (w.size() < 4) { fail("rgb needs 3 components"); return constantSpectrum(0); }
            return rgbToReflectanceJH(num(w[1]), num(w[2]), num(w[3]));
        }
        if (h.rfind("glass:", 0) == 0) {
            std::string g = h.substr(6);
            Spectrum ior;
            if (resolveGlassIor(g, ior)) return ior;
            fail("unknown glass '" + g + "'"); return iorBK7();
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
        if (h.rfind("preset:", 0) == 0)  return resolvePreset(h.substr(7));
        if (h.rfind("file:", 0) == 0)    return loadSpdFile(h.substr(5));
        if (h.rfind("spectrum:", 0) == 0) {
            std::string nm = h.substr(9);
            auto it = spectraBlocks_.find(nm);
            if (it == spectraBlocks_.end()) { fail("unknown spectrum '" + nm + "'"); return constantSpectrum(0); }
            const Stmt* e = find(*it->second, "=");
            if (!e) { fail("spectrum '" + nm + "' has no value"); return constantSpectrum(0); }
            return evalSpectrum(e->val, depth + 1);
        }
        fail("unrecognized spectrum expression '" + h + "'");
        return constantSpectrum(0);
    }

    // Load a measured SPD/reflectance from an external data file: `spd file:<path>`
    // (or `reflect file:<path>`). Reads the CSV/whitespace table mirrored under data/
    // into a piecewise-linear `tabulatedSpectrum`. Paths resolve relative to the
    // current working directory (same convention as `texture`/`mesh` file refs), and
    // repeated references to the same path share one cached curve.
    Spectrum loadSpdFile(const std::string& path) {
        auto it = spdFileCache_.find(path);
        if (it != spdFileCache_.end()) return it->second;
        std::vector<std::pair<double, double>> pairs;
        std::string ferr;
        if (!loadSpdCsv(path, pairs, ferr)) { fail(ferr); return constantSpectrum(0); }
        Spectrum s = tabulatedSpectrum(std::move(pairs));
        spdFileCache_[path] = s;
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

    // ---- textures ----
    // A `texture "name" { file "path" [encoding srgb|linear] [filter nearest|
    // bilinear] [wrap repeat|clamp|mirror] }` block loads an image into
    // Scene::textures and records its name -> index. Reflectance coefficients are
    // precomputed here so per-hit sampling is a cheap bilerp+sigmoid.
    bool addTexture(const Block& b, Loaded& L) {
        if (b.name.empty()) { fail("texture needs a \"name\""); return false; }
        if (textureIndex_.count(b.name)) { fail("duplicate texture name '" + b.name + "'"); return false; }
        std::string file = strOf(b, "file");
        if (file.empty()) { fail("texture '" + b.name + "' needs a file"); return false; }
        Texture tex;
        tex.name = b.name;
        std::string enc = strOf(b, "encoding", "srgb");
        if      (enc == "srgb")   tex.encoding = TexEncoding::sRGB;
        else if (enc == "linear") tex.encoding = TexEncoding::Linear;
        else { fail("texture '" + b.name + "': unknown encoding '" + enc + "' (srgb|linear)"); return false; }
        std::string flt = strOf(b, "filter", "bilinear");
        if      (flt == "bilinear") tex.filter = TexFilter::Bilinear;
        else if (flt == "nearest")  tex.filter = TexFilter::Nearest;
        else { fail("texture '" + b.name + "': unknown filter '" + flt + "' (nearest|bilinear)"); return false; }
        std::string wr = strOf(b, "wrap", "repeat");
        if      (wr == "repeat") tex.wrap = TexWrap::Repeat;
        else if (wr == "clamp")  tex.wrap = TexWrap::Clamp;
        else if (wr == "mirror") tex.wrap = TexWrap::Mirror;
        else { fail("texture '" + b.name + "': unknown wrap '" + wr + "' (repeat|clamp|mirror)"); return false; }
        std::string terr;
        if (!tex.load(file, terr)) { fail("texture '" + b.name + "': " + terr); return false; }
        // Optional indexed-spectral palette (§9.3): `palette { 0 spectrum:navy 1 ... }`.
        // The nested block's flat word dump is (index, spectrum-ref) pairs in order; we
        // resolve each ref to a Spectrum now and size the palette to max-index+1. The
        // texture's red channel then selects an entry per texel (nearest, no upsample).
        if (const Stmt* ps = find(b, "palette")) {
            if (!ps->val.block) { fail("texture '" + b.name + "': palette needs a { } body"); return false; }
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
            if (!compilePatternExpr(expr, pat.nodes, perr)) {
                fail("pattern '" + b.name + "': " + perr); return false;
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

    // ---- materials ----
    Material buildMaterial(const Block& b) {
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
                    m.roughness = dblOf(b, "roughness", m.roughness);
            }
            if (find(b, "film_ior"))       m.filmIor       = dblOf(b, "film_ior", m.filmIor);
            if (find(b, "film_thickness")) m.filmThickness = dblOf(b, "film_thickness", m.filmThickness);
            if (!bindScalarPattern(b, "film_thickness_map", m.filmThicknessPat))
                bindScalarTexture(b, "film_thickness_map", m.filmThicknessTex);
            if (find(b, "reflect"))        m.reflect       = spectrumParam(b, "reflect", m.reflect);
            if (find(b, "ior"))            m.ior           = spectrumParam(b, "ior", m.ior);
            return m;
        }
        std::string type = strOf(b, "type", "diffuse");
        if (type == "diffuse") {
            m.type = MatType::Diffuse;
            // `reflect texture:<name>` binds a spatially-varying albedo; otherwise a
            // uniform reflectance spectrum. A bound texture leaves m.reflect as the
            // fallback used where UVs are unavailable (e.g. the CUDA bake path).
            if (bindReflectTexture(b, m)) m.reflect = constantSpectrum(0.75);
            else                          m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.75));
        } else if (type == "translucent" || type == "diffuse_transmit") {
            // Two-lobe Lambertian: `reflect` (front-hemisphere diffuse albedo) +
            // `transmit` (back-hemisphere diffuse albedo). Both non-specular, so a
            // directly-viewed solid is visible in mode B. reflect+transmit is clamped
            // to <= 1 per wavelength at render time (the remainder is absorbed).
            m.type = MatType::DiffuseTransmit;
            if (bindReflectTexture(b, m)) m.reflect = constantSpectrum(0.5);
            else                          m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.4));
            m.transmit = spectrumParam(b, "transmit", constantSpectrum(0.4));
        } else if (type == "dielectric") {
            m.type = MatType::Dielectric;
            m.ior = spectrumParam(b, "ior", iorBK7());
            // Frosted/rough transmission: 0 (default) = perfectly clear glass, bit-
            // identical to before; >0 roughens both the reflected and refracted lobes.
            // `roughness pattern:<name>` (§4) or `texture:<name>` binds a per-hit map.
            if (bindScalarPattern(b, "roughness", m.roughnessPat)) m.roughness = 0.2;
            else if (bindScalarTexture(b, "roughness", m.roughnessTex)) m.roughness = 0.2;
            else m.roughness = dblOf(b, "roughness", 0.0);
            // Interior absorption sigma_a(lambda) per metre travelled inside the glass
            // (Beer-Lambert tint). 0 (default) = colorless. e.g. `absorb 3 0.5 0.3`
            // (per-channel, upsampled) gives green-tinted glass.
            m.absorb = spectrumParam(b, "absorb", constantSpectrum(0.0));
        } else if (type == "mirror") {
            m.type = MatType::Mirror;
            m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.95));
        } else if (type == "halfmirror") {
            m.type = MatType::HalfMirror;
            m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.5));
        } else if (type == "glossy") {
            m.type = MatType::Glossy;
            m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.9));
            // `roughness pattern:<name>` (§4) / `texture:<name>` binds a per-hit
            // roughness map (grayscale = roughness directly, both 0..1); else a constant.
            if (bindScalarPattern(b, "roughness", m.roughnessPat)) m.roughness = 0.2;
            else if (bindScalarTexture(b, "roughness", m.roughnessTex)) m.roughness = 0.2;
            else m.roughness = dblOf(b, "roughness", 0.2);
        } else if (type == "thinfilm") {
            m.type = MatType::ThinFilm;
            m.ior = spectrumParam(b, "ior", iorConstant(1.5));
            m.filmIor = dblOf(b, "film_ior", 1.30);
            // `film_thickness <nm>` is the peak/scale; `film_thickness_map texture:<n>`
            // binds a 0..1 profile scaled by it (spatially-varying iridescence, §9.4).
            m.filmThickness = dblOf(b, "film_thickness", 300.0);
            if (!bindScalarPattern(b, "film_thickness_map", m.filmThicknessPat))
                bindScalarTexture(b, "film_thickness_map", m.filmThicknessTex);
            // Substrate extinction kappa (spectral): 0 = transparent dielectric
            // (lossless, default). Non-zero -> absorbing/metallic substrate giving
            // opaque structural colour; a spectral kappa (e.g. a gaussian) tints it
            // like a real metal (gold, copper).
            m.substrateK = spectrumParam(b, "substrate_k", constantSpectrum(0.0));
        } else if (type == "grating") {
            m.type = MatType::Grating;
            m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.9));
            m.grooveSpacing = dblOf(b, "groove_spacing", 1000.0);
            Vec3 gd{0, 1, 0}; vec3Of(b, "groove_dir", gd); m.grooveDir = gd;
            m.gratingMaxOrder = (int)dblOf(b, "max_order", 3);
        } else if (type == "fluorescent") {
            m.type = MatType::Fluorescent;
            m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.1));
            m.fluoAbsorb = spectrumParam(b, "absorb", shortPass(490.0, 0.15, 1.0));
            m.fluoEmit = spectrumParam(b, "emit", gaussianBand(560.0, 25.0, 1.0));
            m.fluoYield = dblOf(b, "yield", 1.0);
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
        return m;
    }

    // Second material pass: resolve a Mix material's `layer "name" weight` entries
    // to child indices + weights. Called after every material name is registered,
    // so a mix may reference children declared before OR after it. Nested mixes are
    // rejected to keep resolution single-step (and the CUDA CDF bounded).
    bool resolveMixChildren(const Block& b, Material& m, Loaded& L) {
        double sum = 0.0;
        for (const auto& s : b.stmts) {
            if (s.key != "layer") continue;
            if (s.val.words.size() < 2) { fail("mix 'layer' needs a material name and a weight"); return false; }
            const std::string& cname = s.val.words[0];
            double w = num(s.val.words[1]);
            auto it = matIndex_.find(cname);
            if (it == matIndex_.end()) { fail("mix layer references unknown material '" + cname + "'"); return false; }
            if (L.scene.mats[it->second].type == MatType::Mix ||
                L.scene.mats[it->second].type == MatType::Layered) {
                fail("layer '" + cname + "' is itself a mix/layered (nesting is not allowed)"); return false;
            }
            if (w < 0.0) { fail("mix layer weight must be >= 0"); return false; }
            m.mixChildren.push_back(it->second);
            m.mixWeights.push_back(w);
            sum += w;
        }
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
    bool addSphere(const Block& b, Loaded& L, const Affine& xf = Affine::identity()) {
        Vec3 c{0, 0, 0}; vec3Of(b, "center", c);
        double r = dblOf(b, "radius", 1.0);
        std::string mat = strOf(b, "material");
        if (mat.empty()) { fail("sphere needs a material"); return false; }
        int id = matId(mat); if (!err.empty()) return false;
        // A sphere stays a sphere only under translate + rotation + UNIFORM scale;
        // a non-uniform scale would make it an ellipsoid the analytic primitive
        // cannot represent (see known-issues.md — true instancing/quadrics).
        bool nonUniform = false; double s = xf.uniformScale(nonUniform);
        if (nonUniform) { fail("sphere under non-uniform scale would be an ellipsoid; use translate + uniform scale (or a mesh)"); return false; }
        Vec3 wc = P(xf.apply(c));
        double wr = Len(r) * s;
        L.scene.spheres.push_back(Sphere{wc, wr, id});
        if (!b.name.empty()) sphereByName_[b.name] = NamedSphere{wc, wr};
        return true;
    }
    bool addQuad(const Block& b, Loaded& L, const Affine& xf = Affine::identity()) {
        Vec3 o{0, 0, 0}, u{1, 0, 0}, v{0, 0, 1};
        vec3Of(b, "origin", o); vec3Of(b, "u", u); vec3Of(b, "v", v);
        std::string mat = strOf(b, "material");
        if (mat.empty()) { fail("quad needs a material"); return false; }
        int id = matId(mat); if (!err.empty()) return false;
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
        std::string mat = strOf(b, "material");
        if (mat.empty()) { fail("triangle needs a material"); return false; }
        int id = matId(mat); if (!err.empty()) return false;
        L.scene.tris.push_back(Tri{P(xf.apply(v0)), P(xf.apply(v1)), P(xf.apply(v2)), id, -1, {}});
        return true;
    }
    bool addMesh(const Block& b, Loaded& L, const Affine& parentXf = Affine::identity()) {
        std::string file = strOf(b, "file");
        if (file.empty()) { fail("mesh needs a file"); return false; }
        std::string mat = strOf(b, "material");
        if (mat.empty()) { fail("mesh needs a material"); return false; }
        int id = matId(mat); if (!err.empty()) return false;
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
        MtlResolver resolver = [this](const std::string& nm) -> int {
            auto it = matIndex_.find(nm);
            return (it == matIndex_.end()) ? -1 : it->second;
        };
        size_t triStart = L.scene.tris.size();
        loadObj(L.scene, file.c_str(), id, xf, loadUV, useNames ? &resolver : nullptr,
                uvProj, uvAxis);
        // Record the loaded mesh's world AABB for object-name fog bounds (a mesh bound
        // is approximated by its box — true containment is deferred, see known-issues).
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
        Affine world = parentXf.compose(affineFromTRS(tr, rot, scl));
        // Child primitives are nested brace blocks; the transform-only statements
        // (translate/rotate/scale) carry no block and are skipped here.
        for (const auto& s : b.stmts) {
            const Block* cb = s.val.block.get();
            if (!cb) continue;
            if      (s.key == "sphere")   { if (!addSphere(*cb, L, world)) return false; }
            else if (s.key == "quad")     { if (!addQuad(*cb, L, world)) return false; }
            else if (s.key == "triangle") { if (!addTriangle(*cb, L, world)) return false; }
            else if (s.key == "mesh")     { if (!addMesh(*cb, L, world)) return false; }
            else if (s.key == "light")    { if (!addLight(*cb, L, cb->type, world)) return false; haveLight = true; }
            else if (s.key == "group")    { if (!addGroup(*cb, L, world, haveLight)) return false; }
            else { fail("unknown block '" + s.key + "' inside group (allowed: sphere, quad, triangle, mesh, light, group)"); return false; }
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
        if (!compilePatternExpr(expr, prog, perr)) { fail("function expr: " + perr); return false; }
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
            if (!buildFieldStmt(cs, xf, out, exprPool)) return false;
            if (++nChild >= 2) { FieldNode c; c.op = op; c.p[0] = kBlend; out.push_back(c); }
        }
        if (nChild < 1) { fail("'" + k + "' needs at least one child primitive"); return false; }
        return true;
    }

    bool addIsosurface(const Block& b, Loaded& L) {
        std::string mat = strOf(b, "material");
        if (mat.empty()) { fail("isosurface needs a material"); return false; }
        int id = matId(mat); if (!err.empty()) return false;
        Affine rootXf = fieldXf(b, Affine::identity());
        std::vector<FieldNode> nodes;
        std::vector<PatNode> exprPool;
        int nRoot = 0;
        for (const auto& cs : b.stmts) {
            if (!cs.val.block) continue;              // skip material/translate/rotate/scale
            if (cs.key == "contained_by") continue;   // container box, not a field element
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
            Vec3 mn{-1,-1,-1}, mx{1,1,1};
            vec3Of(cbb, "min", mn);
            vec3Of(cbb, "max", mx);
            // Transform all 8 authored corners into world (metre-rebased) and take the AABB.
            Aabb box; bool first = true;
            for (int c = 0; c < 8; ++c) {
                Vec3 corner{(c&1)?mx.x:mn.x, (c&2)?mx.y:mn.y, (c&4)?mx.z:mn.z};
                Vec3 w = rootXf.apply(corner) * L_;
                if (first) { box.lo = w; box.hi = w; first = false; } else box.expand(w);
            }
            im.bounds = box;
            double mg = dblOf(b, "max_gradient", 0.0);
            im.lipschitz = (mg > 0.0) ? mg
                                      : 1.3 * estimateFieldLipschitz(im.nodes, im.exprNodes, box);
            double acc = dblOf(b, "accuracy", 0.0);
            im.minStep = (acc > 0.0) ? acc * L_ : implicitMinStep(box);
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
    bool addLight(const Block& b, Loaded& L, const std::string& subtype,
                  const Affine& xf = Affine::identity()) {
        Spectrum spd = spectrumParam(b, "spd", blackbody(6500.0));
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
            spd = absPower(b, spd, (w * w) * PI, L);
            L.scene.addAreaLight(P(xf.apply(o)), t * w, bt * w, beam, w * w, spd, binWidth_,
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
        int id = (int)L.scene.mats.size(); L.scene.mats.push_back(lm);
        Vec3 a = os, bb = os + us, cc = os + us + vs, dd = os + vs;
        L.scene.tris.push_back(Tri{a, bb, cc, id, -1, {}});
        L.scene.tris.push_back(Tri{a, cc, dd, id, -1, {}});
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
            //   • mesh       -> the mesh's world AABB (box approximation; true mesh
            //                   containment is deferred — see known-issues.md)
            if (const std::string onm = strOf(bb, "object"); !onm.empty()) {
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
                    med.boundInsideNeg = (im.eval(ctr) <= 0.0);
                } else if (auto mit = meshAabbByName_.find(onm); mit != meshAabbByName_.end()) {
                    const Aabb& box = mit->second;
                    med.bounded = true;
                    med.boundShape = MediumBound::Box;
                    med.bmin = box.lo;
                    med.bmax = box.hi;
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
                    if (!compilePatternExpr(expr, prog, perr)) {
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
                    for (int iz = 0; iz <= NS; ++iz)
                    for (int iy = 0; iy <= NS; ++iy)
                    for (int ix = 0; ix <= NS; ++ix) {
                        Vec3 p{ lo.x + (hi.x - lo.x) * ix / NS,
                                lo.y + (hi.y - lo.y) * iy / NS,
                                lo.z + (hi.z - lo.z) * iz / NS };
                        peak = std::max(peak, med.densityAt(p));
                    }
                    dmax = 1.3 * peak;
                }
                med.densityMax = (dmax > 0.0) ? dmax : 1.0;
            }
        }
        L.scene.media.push_back(std::move(med));
        return true;
    }

    // Read the film sub-block + photographic exposure/f-stop/lens controls shared by
    // `camera` and `camera_path`, and resolve the film size (named format or explicit
    // mm), the focal length (from `lens <mm>` or `fov_y`), the f-stop -> aperture
    // radius, the physical-optics film distance, and the manual exposure multiplier.
    // `cs.fov` must already be set. Returns false only on an unknown film format.
    bool readFilmExposure(const Block& b, CamSpec& cs) {
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
        const double DEG = 3.141592653589793 / 180.0;
        double lensMM = dblOf(b, "lens", 0.0);     // focal length in mm (physical, unit-independent)
        // `zoom <x>` multiplies the focal length (x>1 = tele/narrower fov; x<1 = wider).
        // It is the animatable "zoom ring" and composes on top of `lens`/`fov_y`.
        double zoom = dblOf(b, "zoom", 1.0); if (zoom <= 0.0) zoom = 1.0;
        cs.zoom = zoom;
        double focalMM;
        if (lensMM > 0.0) focalMM = lensMM;
        else { double th = std::tan(0.5 * cs.fov * DEG); focalMM = (th > 1e-9) ? hmm / (2.0 * th) : 0.0; }
        focalMM *= zoom;
        cs.focal = focalMM / 1000.0;
        if (lensMM > 0.0 || zoom != 1.0)           // fov follows the (zoomed) focal length
            cs.fov = (focalMM > 1e-9) ? 2.0 * std::atan(hmm / (2.0 * focalMM)) / DEG : cs.fov;
        // f-stop authoring: N = f / (2*apertureR) -> apertureR = f / (2N). Overrides
        // any `aperture` radius. Aperture radius is an internal (metre) length.
        double fstop = dblOf(b, "fstop", 0.0);
        if (fstop > 0.0 && cs.focal > 0.0) cs.aperture = cs.focal / (2.0 * fstop);
        // Physical-optics camera: when a lens/f-stop is authored, put the film at the
        // real image distance and give the thin lens the true focal length, so the
        // f-number yields correct depth of field in the catch modes (A/C). Thin-lens
        // law 1/so + 1/si = 1/f: a focus plane at so (metres) images at si; focus 0
        // (or beyond hyperfocal) means infinity -> si = f. Legacy cameras (no lens /
        // f-stop) leave these 0 and keep the unit-film-distance behaviour.
        if ((lensMM > 0.0 || fstop > 0.0) && cs.focal > 0.0) {
            double f = cs.focal, so = cs.focus;    // cs.focus already in metres (Len-scaled)
            cs.filmDist_m = (so > f) ? 1.0 / (1.0 / f - 1.0 / so) : f;
            cs.lensF_m    = f;
        }
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
                    g = iorBK7();
                *sys = makeSinglet(focalMM, fstop, g);
            } else if (!preset.empty()) {
                if (!resolveLensPreset(preset, focalMM, fstop, *sys)) {
                    fail("camera '" + cs.name + "' lens: unknown preset '" + preset +
                         "' (singlet, achromat/doublet, telephoto, wide)"); return false;
                }
            } else {
                *sys = makeAchromat(focalMM, fstop, iorBK7(), iorSF10());
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
        if (!md.empty()) cs.mode = md[0];
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
    bool addCameraPath(const Block& b, Loaded& L) {
        std::string base = b.name.empty() ? ("path" + std::to_string(L.cameras.size())) : b.name;
        CamSpec shared;
        vec3Of(b, "look_at", shared.look); vec3Of(b, "up", shared.up);
        shared.look = P(shared.look);
        shared.fov = dblOf(b, "fov_y", 40.0);
        shared.aperture = Len(dblOf(b, "aperture", 0.02));
        shared.focus = Len(dblOf(b, "focus", 0.0));
        std::string md = strOf(b, "mode"); if (!md.empty()) shared.mode = md[0];
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

        // Exposure-lock: a bare `exposure_lock` (or `exposure_lock on`) makes every
        // frame of this path share the auto-exposure anchor from frame 0 (no
        // flicker); `off`/`false`/`0` disables (the default). The group id is this
        // path's starting index in L.cameras — unique because paths occupy disjoint
        // contiguous ranges.
        bool pathLock = false;
        if (const Stmt* el = find(b, "exposure_lock")) {
            if (el->val.words.empty()) pathLock = true;
            else { const std::string& v = el->val.words[0]; pathLock = !(v=="off"||v=="false"||v=="0"); }
        }
        const int pathGroup = (int)L.cameras.size();

        // Collect keyframes (t, eye, optional look_at, optional fov), sorted by t.
        // Field count disambiguates: 4=t,eye  5=t,eye,fov  7=t,eye,look  8=t,eye,look,fov.
        struct Key { double t; Vec3 eye, look; bool hasLook; double fov; };
        std::vector<Key> keys;
        for (const auto& s : b.stmts) {
            if (s.key != "key") continue;
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
        std::string md = strOf(b, "mode"); if (!md.empty()) shared.mode = md[0];
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

        bool pathLock = false;
        if (const Stmt* el = find(b, "exposure_lock")) {
            if (el->val.words.empty()) pathLock = true;
            else { const std::string& v = el->val.words[0]; pathLock = !(v == "off" || v == "false" || v == "0"); }
        }
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
    //   look_at x y z             — a fixed target for every frame
    //   look curve + look_point.. — aim at a SECOND Catmull-Rom spline, sampled in step
    // `closed` loops the curve (seamless). All frames share up/fov/mode/film/lens.
    // Grammar:
    //   camera_curve "fly" {
    //       point 0 1 3   point 1 1 1   point 2 1 3   point 1 1 5   # >= 2 control points
    //       up 0 1 0   fov_y 50   mode R   frames 90
    //       density 20                       # OR density_at 0 5  density_at 0.5 40  density_at 1 5
    //       look tangent                     # OR look_at 1 1 3   OR look curve + look_point ...
    //       closed   exposure_lock
    //       film { res 900 600 }
    //   }
    bool addCameraCurve(const Block& b, Loaded& L) {
        std::string base = b.name.empty() ? ("curve" + std::to_string(L.cameras.size())) : b.name;
        CamSpec shared;
        vec3Of(b, "up", shared.up);
        shared.fov = dblOf(b, "fov_y", 40.0);
        shared.aperture = Len(dblOf(b, "aperture", 0.02));
        shared.focus = Len(dblOf(b, "focus", 0.0));
        std::string md = strOf(b, "mode"); if (!md.empty()) shared.mode = md[0];
        if (!readFilmExposure(b, shared)) return false;   // film{res,size/format,...}, lens, fstop, zoom
        if (!readProjection(b, shared)) return false;     // projection/fisheye
        if (!readLens(b, shared)) return false;           // optional physical `lens { ... }` block

        // Control points (>= 2), in file order; the spline passes through each.
        std::vector<Vec3> pts;
        for (const auto& s : b.stmts) {
            if (s.key != "point") continue;
            if (s.val.words.size() < 3) { fail("camera_curve '" + base + "' point needs x y z"); return false; }
            pts.push_back(P(Vec3{num(s.val.words[0]), num(s.val.words[1]), num(s.val.words[2])}));
        }
        if (pts.size() < 2) { fail("camera_curve '" + base + "' needs >= 2 `point` control points"); return false; }

        bool closed = false;
        if (const Stmt* c = find(b, "closed")) {
            if (c->val.words.empty()) closed = true;
            else { const std::string& v = c->val.words[0]; closed = !(v == "off" || v == "false" || v == "0"); }
        }

        // Density = cameras per unit LENGTH (1/authored-unit -> 1/metre). `density_at`
        // keyframes it (piecewise-linear over normalized position t in [0,1]); a bare
        // `density` is constant. Either drives BOTH the camera count (integral of rho
        // over the curve) and the local spacing (1/rho).
        struct DKey { double t, rho; };
        std::vector<DKey> dkeys;
        for (const auto& s : b.stmts) {
            if (s.key != "density_at") continue;
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
        std::vector<double> sampG((size_t)M + 1), sampC((size_t)M + 1);
        Vec3 prev = catmullRomAt(pts, closed, 0.0);
        sampG[0] = 0.0; sampC[0] = 0.0;
        for (int k = 1; k <= M; ++k) {
            double g = nSeg * (double)k / M;
            Vec3 pcur = catmullRomAt(pts, closed, g);
            double ds = length(pcur - prev);
            double rho = densityAt(g / nSeg);
            sampC[k] = sampC[k - 1] + rho * ds;
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

        // Orientation. Gather optional look-curve control points first.
        std::vector<Vec3> lookPts;
        for (const auto& s : b.stmts) {
            if (s.key != "look_point") continue;
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

        bool pathLock = false;
        if (const Stmt* el = find(b, "exposure_lock")) {
            if (el->val.words.empty()) pathLock = true;
            else { const std::string& v = el->val.words[0]; pathLock = !(v == "off" || v == "false" || v == "0"); }
        }
        const int pathGroup = (int)L.cameras.size();

        int pad = 1; for (int f = N - 1; f >= 10; f /= 10) ++pad;   // zero-pad width
        for (int i = 0; i < N; ++i) {
            // A closed loop samples i/N (frame N == frame 0, not duplicated); an open
            // curve spans both endpoints via i/(N-1).
            double fr = closed ? ((double)i / N)
                               : (N == 1 ? 0.5 : (double)i / (N - 1));
            double g = invertC(fr * Cmax);
            CamSpec cs = shared;
            cs.eye = catmullRomAt(pts, closed, g);
            if (lookCurve) {
                cs.look = catmullRomAt(lookPts, closed, fr * lookSeg);
            } else if (lookFixed) {
                cs.look = fixedLook;
            } else {   // tangent: aim one finite-difference step ahead along the curve
                double dg = (double)nSeg / (M * 4.0);
                Vec3 a = catmullRomAt(pts, closed, g - dg);
                Vec3 c = catmullRomAt(pts, closed, g + dg);
                Vec3 tan = c - a;
                cs.look = cs.eye + ((length(tan) > 1e-12) ? normalize(tan) : Vec3{0, 0, -1});
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
        if (!md.empty()) L.mode = md[0];
        std::string o = strOf(b, "out");
        if (!o.empty()) L.out = o;
        const Stmt* r = find(b, "res");
        if (r && !r->val.words.empty()) L.res = (int)num(r->val.words[0]);
        return true;
    }
};

// Load an FTSL file, populating `L`. Returns false and sets `err` on any error.
inline bool load(const std::string& path, Loaded& L, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open scene file: " + path; return false; }
    std::stringstream ss; ss << f.rdbuf();
    std::string src = ss.str();

    Parser p; p.t = tokenize(src);
    std::vector<Block> blocks = p.parseTop();
    if (!p.err.empty()) { err = p.err; return false; }

    Builder bld;
    if (!bld.build(blocks, L)) { err = bld.err; return false; }
    return true;
}

} // namespace ftsl
