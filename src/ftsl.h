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
#include "fbx.h"
#include "upsample.h"
#include "color.h"

namespace ftsl {

// A token is a number iff strtod consumes all of it (handles -1, 0.999, 1e30).
inline bool isNumber(const std::string& s) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end == s.c_str() + s.size();
}
inline double num(const std::string& s) { return std::strtod(s.c_str(), nullptr); }

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
// Tokenizer
// ---------------------------------------------------------------------------
enum class Tok { Word, String, LBrace, RBrace, LBracket, RBracket, Newline, End };
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
        if (c == '[') { out.push_back({Tok::LBracket, "[", line}); ++i; continue; }
        if (c == ']') { out.push_back({Tok::RBracket, "]", line}); ++i; continue; }
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
                d == '{' || d == '}' || d == '[' || d == ']' || d == '#' || d == '"') break;
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
    // For type == "prefer": ordered alternative branches (`prefer { .. } else { .. }`).
    // Each branch is a list of ordinary top-level blocks; the loader picks the first
    // branch whose spliced scene is fully renderable in its chosen mode. Empty otherwise.
    std::vector<std::vector<Block>> branches;
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
        // Record-override assignment: `slot = <rhs>` (used inside a `material "m" { … }`
        // record block, §records stage 4). A standalone `=` first token never begins a
        // normal statement value, so this is unambiguous. The RHS is a single token
        // (an expression, a channel name, or `REC.chan`), optionally followed by a
        // `[i]` stop selector — which we fold back into the RHS token (`REC.chan[i]`) so
        // the bracket tokens don't leak into the generic brace-body statement stream.
        if (is(Tok::Word) && cur().text == "=") {
            v.words.push_back("="); adv();
            if (is(Tok::Word) || is(Tok::String)) { v.words.push_back(cur().text); adv(); }
            if (is(Tok::LBracket)) {
                adv();
                std::string idx;
                while (is(Tok::Word)) { idx += cur().text; adv(); }
                if (is(Tok::RBracket)) adv(); else fail("record-override selector missing ']'");
                if (!v.words.empty()) v.words.back() += "[" + idx + "]";
            }
            return;
        }
        bool firstWasString = false;
        if (is(Tok::Word) || is(Tok::String)) {
            firstWasString = is(Tok::String);
            v.words.push_back(cur().text); adv();
        }
        while (is(Tok::Word) || is(Tok::String)) {
            // A quoted string never begins a new statement (statement keys are always
            // barewords), so a String that follows the value's tokens is part of THIS
            // value — e.g. a name argument: `exposure_lock camera "meter"`. (Without this
            // the string was silently swallowed as a stray statement key.) Barewords only
            // continue when they are numbers or `key=val` named params; a plain bareword
            // still ends the value (it begins the next statement's key).
            if (is(Tok::Word)) {
                const std::string& tx = cur().text;
                bool cont = isNumber(tx) || tx.find('=') != std::string::npos;
                if (!cont) break;
            }
            v.words.push_back(cur().text); adv();
        }
        // Records stage 5a: a trailing `[i]` stop selector on a record-channel ref
        // (`RECORD.channel[i]`) at an ordinary value site — fold the bracket tokens back
        // into the preceding word so they don't leak into the brace-body statement stream
        // (the `=` override path above already does this). Only when the preceding word
        // looks like a dotted reference, so a stray `[` elsewhere still surfaces as an error.
        if (is(Tok::LBracket) && !v.words.empty() && v.words.back().find('.') != std::string::npos) {
            adv();
            std::string idx;
            while (is(Tok::Word)) { idx += cur().text; adv(); }
            if (is(Tok::RBracket)) adv(); else fail("record selector missing ']'");
            v.words.back() += "[" + idx + "]";
        }
        if (is(Tok::LBrace)) {
            std::string btype = key, bname;
            if (!v.words.empty()) {
                // A single *quoted* word before `{` is the block's NAME (e.g. a nested
                // `mesh "klein_a" { ... }` inside a group), so the type stays = key.
                // A bareword before `{` is instead the block's TYPE (e.g. `table { }`,
                // `light sphere { }`) — the historical behaviour for subtyped blocks.
                if (v.words.size() == 1 && firstWasString) { bname = v.words.back(); v.words.pop_back(); }
                else { btype = v.words.back(); v.words.pop_back(); }
            }
            v.block = std::make_shared<Block>();
            v.block->type = btype;
            v.block->name = bname;
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

    // Parse a record body `[ <lines> ]` into stmts (one per non-empty line: a channel
    // name key + its stop tokens). Unlike a brace body, NEWLINES delimit statements and
    // a value holds many barewords (the stops), so this does NOT use parseValue. Assumes
    // cur() == LBracket.
    void parseRecordBody(Block& b) {
        adv();  // consume '['
        while (!is(Tok::RBracket) && !is(Tok::End)) {
            if (is(Tok::Newline)) { adv(); continue; }
            if (!is(Tok::Word)) { fail("record line must start with a channel name"); return; }
            Stmt s; s.line = cur().line;
            s.key = cur().text; adv();
            while (is(Tok::Word) || is(Tok::String)) { s.val.words.push_back(cur().text); adv(); }
            b.stmts.push_back(std::move(s));
        }
        if (is(Tok::RBracket)) adv();
        else fail("unterminated '[' in record");
    }

    // Parse `NAME = range LO-HI [ ... ]` (the leading `NAME = range` already consumed;
    // `name` is NAME). Stores the domain as a stmt `range <tokens>` plus one stmt per
    // channel line, all under a Block of type "record".
    bool parseRecord(Block& b, const std::string& name) {
        b.type = "record";
        b.name = name;
        Stmt dom; dom.key = "range"; dom.line = cur().line;
        while (is(Tok::Word)) { dom.val.words.push_back(cur().text); adv(); }
        b.stmts.push_back(std::move(dom));
        skipNewlines();
        if (!is(Tok::LBracket)) { fail("record '" + name + "' needs '[ ... ]' after `range LO-HI`"); return false; }
        parseRecordBody(b);
        return err.empty();
    }

    // Parse ONE top-level block (or a `prefer { } else { }` construct) starting at
    // cur() (which must be a Word). Fills `b`; returns false on a parse error.
    bool parseOneTopBlock(Block& b) {
        if (!is(Tok::Word)) { fail("expected a block type"); return false; }
        b.type = cur().text; adv();
        if (b.type == "prefer") return parsePrefer(b);
        // Parametric record: `NAME = range LO-HI [ ... ]` (ROADMAP_records.md). Detected
        // by a bare first word immediately followed by '=' then `range` — spectrum uses
        // a *quoted* name before '=', so there is no collision.
        if (is(Tok::Word) && cur().text == "=") {
            std::string name = b.type;   // the leading bare word is the binding NAME
            adv();  // consume '='
            if (is(Tok::Word) && cur().text == "range") {
                adv();  // consume 'range'
                return parseRecord(b, name);
            }
            // Unified element header `NAME = KIND { ... }` (the loom-emitted form):
            // the word after '=' is the block KIND (material/texture/camera/light/
            // isosurface/pattern/geometry) and NAME is its name. Body parses exactly
            // like the legacy `KIND "name" { ... }`. Accepted alongside the legacy
            // spelling during the grammar transition (J3c); one spelling once ftrace's
            // front-end is ported from the shared grammar.
            if (is(Tok::Word)) {
                b.type = cur().text; adv();          // KIND
                b.name = name;
                // Optional bareword subtype (`sun = light point { }`); the new grammar
                // prefers a `kind` property instead, but accept both here.
                if (is(Tok::Word) && cur().text != "=") { b.subtype = cur().text; adv(); }
                if (!is(Tok::LBrace)) { fail("expected '{' after `" + name + " = " + b.type + "`"); return false; }
                parseBraceBody(b);
                return true;
            }
            fail("unknown '=' declaration '" + name + "' (expected `= range` or `= KIND { ... }`)");
            return false;
        }
        if (is(Tok::String)) { b.name = cur().text; adv(); }
        // Optional bareword subtype (light area / light collimated), but not '='.
        if (is(Tok::Word) && cur().text != "=") { b.subtype = cur().text; adv(); }
        if (b.type == "spectrum") {
            if (is(Tok::Word) && cur().text == "=") adv();
            else { fail("spectrum declaration needs '='"); return false; }
            Stmt s; s.key = "="; s.line = cur().line;
            parseValue("=", s.val);
            b.stmts.push_back(std::move(s));
        } else {
            if (!is(Tok::LBrace)) { fail("expected '{' after " + b.type); return false; }
            parseBraceBody(b);
        }
        return true;
    }

    // Parse a `{ <top-level blocks> }` list (cur() must be LBrace). Consumes the braces.
    // Used for each branch of a `prefer`/`else` construct.
    std::vector<Block> parseBlockList() {
        std::vector<Block> list;
        adv();   // consume '{'
        skipNewlines();
        while (!is(Tok::RBrace) && !is(Tok::End) && err.empty()) {
            Block b;
            if (!parseOneTopBlock(b)) break;
            list.push_back(std::move(b));
            skipNewlines();
        }
        if (is(Tok::RBrace)) adv();
        else fail("unterminated 'prefer'/'else' block");
        return list;
    }

    // Parse `prefer { .. } else { .. } else { .. }` (b.type already == "prefer").
    // Each brace group becomes one ordered branch in b.branches; `else` chains flatly.
    bool parsePrefer(Block& b) {
        skipNewlines();
        if (!is(Tok::LBrace)) { fail("'prefer' needs '{ ... }'"); return false; }
        b.branches.push_back(parseBlockList());
        if (!err.empty()) return false;
        for (;;) {
            skipNewlines();
            if (is(Tok::Word) && cur().text == "else") {
                adv(); skipNewlines();
                if (!is(Tok::LBrace)) { fail("'else' needs '{ ... }'"); return false; }
                b.branches.push_back(parseBlockList());
                if (!err.empty()) return false;
            } else break;
        }
        return true;
    }

    // Parse the whole file into a list of top-level blocks.
    std::vector<Block> parseTop() {
        std::vector<Block> blocks;
        skipNewlines();
        while (!is(Tok::End) && err.empty()) {
            Block b;
            if (!parseOneTopBlock(b)) break;
            blocks.push_back(std::move(b));
            skipNewlines();
        }
        return blocks;
    }
};

}  // namespace ftsl  (temporarily closed so the validation shim can see
   //                  ftsl::Block/Stmt/Value at global scope)

// J3c grammar-validation shim: the shared .ftsl grammar (GPDA) run alongside the
// hand-written Parser above.  Included here — after Block/Stmt/Value/Parser are
// defined — so its inline reducer/differ can reference them.  Non-authoritative;
// see the header for details.
#include "gpda/ftsl_shim.hpp"

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
    double defaultFps = 0.0;     // scene { fps N }: default flyby playback fps (0 = not specified)
    long long photons = -1;      // -1 = not specified (CLI default wins)
    int res = -1;                // -1 = not specified
    std::string device;          // empty = not specified
    std::string out;             // empty = not specified
};

class Builder {
public:
    std::string err;

    bool build(const std::vector<Block>& blocks, Loaded& L) {
        records_ = &L.scene.records;   // stable handle for record refs at value sites (records added in Pass 1d)
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
            if (!dm.empty()) L.defaultMode = dm[0];
            double dfps = dblOf(b, "fps", 0.0);
            if (dfps > 0.0) L.defaultFps = dfps;
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
                     b.type == "mesh_asset") { /* handled */ }
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
    std::unordered_map<std::string, int> recordIndex_;    // record name  -> Scene::records index
    const std::vector<Record>* records_ = nullptr;        // -> L.scene.records (set in build; for record refs at value sites)
    std::unordered_map<std::string, Spectrum> spdFileCache_; // path -> loaded measured SPD

    // Named-object registries for `medium { bounds { object "name" } }` resolution.
    // Populated by the geometry builders during Pass 3; read by addMedium afterwards.
    struct NamedSphere { Vec3 center; double radius; };
    std::unordered_map<std::string, NamedSphere> sphereByName_;   // named sphere -> world center/radius
    std::unordered_map<std::string, int>         implicitByName_; // named isosurface -> Scene::implicits index
    std::unordered_map<std::string, Aabb>        meshAabbByName_; // named mesh -> world AABB
    std::unordered_map<std::string, int>         blasIndex_;      // mesh_asset name -> Scene::blasList index

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
        if (!compilePatternExpr(driverExpr, drv, cerr)) {
            fail("record material driver '" + driverExpr + "': " + cerr);
            return -1;
        }
        applyFrom(m, recIdx, drv, L.scene.records[recIdx]);
        int id = (int)L.scene.mats.size();
        L.scene.mats.push_back(std::move(m));
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
                    fail("record material '" + raw + "': missing ')'");
                    return -1;
                }
                std::string driver = raw.substr(lp + 1, rp - lp - 1);
                return buildRecordMaterial(rit->second, driver, L);
            }
        }
        auto it = matIndex_.find(raw);
        if (it == matIndex_.end()) { fail("unknown material '" + raw + "'"); return -1; }
        return it->second;
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
            Spectrum rs;
            if (recordConstSpectrumRef(h, rs)) return rs;
        }

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
        {
            bool isLine  = (h == "rgbline"  || h == "hsvline"  || h == "hslline");
            bool isIllum = (h == "rgbillum" || h == "hsvillum" || h == "hslillum");
            if (h == "rgb" || h == "hsv" || h == "hsl" || isLine || isIllum) {
                if (w.size() < 4) { fail(h + " needs 3 components"); return constantSpectrum(0); }
                std::string space = (isLine || isIllum) ? h.substr(0, 3) : h;
                Vec3 c;
                if      (space == "rgb") c = {num(w[1]), num(w[2]), num(w[3])};
                else if (space == "hsv") c = hsvToRgb(num(w[1]), num(w[2]), num(w[3]));
                else                     c = hslToRgb(num(w[1]), num(w[2]), num(w[3]));
                if (isLine) {
                    double sigma = (w.size() > 4 && isNumber(w[4])) ? num(w[4]) : -1.0;
                    return rgbToLineEmission(c.x, c.y, c.z, sigma);
                }
                if (isIllum) return rgbToIlluminantJH(c.x, c.y, c.z);
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
        if (w0.find('.') != std::string::npos && !isNumber(w0)) {
            double rv;
            if (recordConstScalarRef(w0, rv)) return rv;             // record ref (or a fail was set)
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
            if (!compilePatternExpr(rgbS->val.words[0], pr, perr)) { fail("texture '" + b.name + "' rgb r: " + perr); return false; }
            if (!compilePatternExpr(rgbS->val.words[1], pg, perr)) { fail("texture '" + b.name + "' rgb g: " + perr); return false; }
            if (!compilePatternExpr(rgbS->val.words[2], pb, perr)) { fail("texture '" + b.name + "' rgb b: " + perr); return false; }
            int res = (int)dblOf(b, "res", 512.0);
            if (res < 1) res = 1; else if (res > 8192) res = 8192;
            tex.encoding = TexEncoding::Linear;   // expr outputs are linear albedo already
            tex.w = res; tex.h = res;
            tex.rgb.assign((size_t)res * res, Vec3{0, 0, 0});
            auto cl = [](double t) { return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t); };
            for (int y = 0; y < res; ++y) {
                // Invert v to match sampleRgb's (1-v) flip so f(u,v) reads back at the
                // surface UV (top-left storage, v=0 at image bottom / OBJ convention).
                double v = 1.0 - (y + 0.5) / res;
                for (int x = 0; x < res; ++x) {
                    double u = (x + 0.5) / res;
                    PatCtx c; c.u = u; c.v = v;
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

    // Build one Record from a `NAME = range LO-HI [ ... ]` block: parse the domain,
    // interp, channels and stops; redistribute positions (stage 1); then compile each
    // stop into a scalar pattern program or a resolved colour + linear-RGB (stage 2).
    bool addRecord(const Block& b, Loaded& L) {
        if (b.name.empty()) { fail("record needs a name"); return false; }
        if (recordIndex_.count(b.name)) { fail("duplicate record name '" + b.name + "'"); return false; }
        Record rec;
        rec.name = b.name;
        bool haveRange = false;
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
            // Otherwise: a channel line. Its words are the stops, with optional `p:<pos>`
            // prefixes pinning the position of the following value.
            RecChannel ch;
            ch.name = s.key;
            bool havePin = false; double pinPos = 0.0;
            for (const auto& w : s.val.words) {
                if (w.rfind("p:", 0) == 0) {
                    if (!isNumber(w.substr(2))) {
                        fail("record '" + rec.name + "' channel '" + ch.name + "': bad p:<pos> '" + w + "'");
                        return false;
                    }
                    havePin = true; pinPos = num(w.substr(2));
                    continue;
                }
                RecStop st; st.token = w;
                if (havePin) { st.pinned = true; st.pos = pinPos; havePin = false; }
                ch.stops.push_back(std::move(st));
            }
            if (havePin) {
                fail("record '" + rec.name + "' channel '" + ch.name + "': trailing p:<pos> with no value");
                return false;
            }
            if (ch.stops.empty()) {
                fail("record '" + rec.name + "' channel '" + ch.name + "' has no stops");
                return false;
            }
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
        // Stage 2: compile stop tokens. A stop is a COLOUR iff its token is a prefixed
        // spectrum ref (contains ':', e.g. spectrum:steel / metal:copper / rgb:... );
        // otherwise it is a SCALAR pattern expression (a literal, or math over
        // intrinsics x y z nx ny nz r u v f + functions). A channel must be homogeneous.
        for (auto& ch : rec.channels) {
            bool anyColour = false, anyScalar = false;
            for (const auto& st : ch.stops)
                (st.token.find(':') != std::string::npos ? anyColour : anyScalar) = true;
            if (anyColour && anyScalar) {
                fail("record '" + rec.name + "' channel '" + ch.name +
                     "': mixes colour (spectrum:...) and scalar stops");
                return false;
            }
            ch.kind = anyColour ? ChanKind::Spectrum : ChanKind::Scalar;
            for (auto& st : ch.stops) {
                if (ch.kind == ChanKind::Scalar) {
                    std::string cerr;
                    if (!compilePatternExpr(st.token, st.expr, cerr)) {
                        fail("record '" + rec.name + "' channel '" + ch.name +
                             "': bad stop expression '" + st.token + "': " + cerr);
                        return false;
                    }
                } else {
                    Value v; v.words = { st.token };
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
                if (!compilePatternExpr(dexpr, drv, cerr)) {
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
                if (!compilePatternExpr(rhs, rb.driver, cerr)) {
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
            m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.95));
        } else if (type == "halfmirror") {
            m.type = MatType::HalfMirror;
            m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.5));
        } else if (type == "filter") {
            // Colored gel / Wratten filter: a thin non-scattering absorber. A photon
            // passes straight through, surviving with probability `transmit`(lambda) —
            // the per-wavelength transmittance T(lambda) in [0,1] — and is absorbed
            // otherwise. No reflection, no refraction. Feed T from a measured curve
            // (`transmit file:data/filter/rosco-red.csv` / `transmit filter:red-25`)
            // or a primitive (`transmit gaussian center=630 sigma=25`).
            m.type = MatType::Filter;
            m.transmit = spectrumParam(b, "transmit", constantSpectrum(0.5));
        } else if (type == "glossy") {
            m.type = MatType::Glossy;
            m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.9));
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
            m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.9));
            m.grooveSpacing = dblParam(b, "groove_spacing", 1000.0);
            Vec3 gd{0, 1, 0}; vec3Of(b, "groove_dir", gd); m.grooveDir = gd;
            m.gratingMaxOrder = (int)dblParam(b, "max_order", 3);
        } else if (type == "fluorescent") {
            m.type = MatType::Fluorescent;
            m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.1));
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
        // Nested-dielectric priority (§ nested dielectrics): `priority <N>` — integer,
        // higher wins where dielectric solids overlap. Common to every material type
        // (only consulted for dielectric-like ones); unset => the ahead-of-time audit
        // warns if this material overlaps another dielectric without a priority.
        if (find(b, "priority")) m.priority = (int)std::lround(dblOf(b, "priority", 0.0));
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
        MtlResolver resolver = [this](const std::string& nm) -> int {
            auto it = matIndex_.find(nm);
            return (it == matIndex_.end()) ? -1 : it->second;
        };
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
            std::string gerr;
            if (loadGltf(L.scene, file.c_str(), id, xf, importMats, gerr) == 0 && !gerr.empty()) {
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
            L.scene.meshGroups.push_back(std::move(g));
        }
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

    // ---- mesh_asset (shared instanced geometry) ----
    // `mesh_asset "name" { file "asset.obj|gltf|glb"  material <m>  [import_materials no]
    //  [uv use_mesh]  [usemtl use_names] }` loads a mesh ONCE into its own local
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
        MtlResolver resolver = [this](const std::string& nm) -> int {
            auto it = matIndex_.find(nm);
            return (it == matIndex_.end()) ? -1 : it->second;
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
            if (loadGltf(L.scene, file.c_str(), id, xf, importMats, gerr) == 0 && !gerr.empty()) {
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
                    L.scene.media.push_back(std::move(med));
                    return true;
                }
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
                if (!compilePatternExpr(expr, prog, perr)) {
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
        std::string md = strOf(b, "mode"); if (!md.empty()) shared.mode = md[0];
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
        std::string md = strOf(b, "mode"); if (!md.empty()) shared.mode = md[0];
        shared.fps = dblOf(b, "fps", 0.0);   // playback hint for the flyby (0 = inherit scene default)
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
            if (!compilePatternExpr(dexpr, drv, cerr, /*allowT=*/true)) {
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
        if (!md.empty()) L.mode = md[0];
        std::string o = strOf(b, "out");
        if (!o.empty()) L.out = o;
        const Stmt* r = find(b, "res");
        if (r && !r->val.words.empty()) L.res = (int)num(r->val.words[0]);
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
    Parser p; p.t = tokenize(src);
    std::vector<Block> blocks = p.parseTop();
    if (!p.err.empty()) { err = p.err; return false; }

    // J3c validation shim (non-authoritative); see load() below and src/gpda/ftsl_shim.hpp.
    ftsl_shim::validate(src, blocks, nameForMsgs);

    // Collect top-level `prefer` nodes. The common case (none) is the original fast path.
    std::vector<size_t> preferIdx;
    for (size_t i = 0; i < blocks.size(); ++i)
        if (blocks[i].type == "prefer") preferIdx.push_back(i);

    if (preferIdx.empty()) {
        Builder bld;
        if (!bld.build(blocks, L)) { err = bld.err; return false; }
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
