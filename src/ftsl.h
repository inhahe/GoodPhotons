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
//   light spot       { origin x y z  dir x y z  inner_angle d  outer_angle d  spd … }
//   medium   { sigma_t v  albedo v  g v  rayleigh true }
//   camera "name" { eye ...  look_at ...  up ...  fov_y d  aperture r  focus d  mode B
//                   film { res W H } }
//   render   { photons N  device auto  mode B }
//
// Statements are newline-terminated; brace values (table {…}, film {…}) nest. A
// spectrum expression is any of: a number (constant), `blackbody K`, `gaussian
// center=.. sigma=.. amp=..`, `shortpass edge=.. slope=.. amp=..`, `ior n`,
// `rgb r g b`, `whitewall [r]`, `redwall`, `greenwall`, `glass:BK7|SF10`,
// `preset:<illuminant>`, `spectrum:<name>`, or `table { λ:v λ:v … }`.
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
    int    res  = -1;            // -1 = not specified -> inherit global

    // Physical film + photographic exposure (Phase 3a). filmW/H are the physical
    // sensor dimensions in millimetres (0 = unspecified -> 36x24 "full frame"
    // default is assumed only where a physical size is actually needed, e.g. to
    // turn an f-number into an aperture radius). `focal` is the derived focal
    // length in metres (internal units), from filmH + fov_y.
    double filmW_mm = 0.0, filmH_mm = 0.0, focal = 0.0;
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

        // Pass 2b: resolve Mix child references (now that every name is known).
        for (const auto& b : blocks) {
            if (b.type != "material") continue;
            int id = matIndex_[b.name];
            if (L.scene.mats[id].type != MatType::Mix) continue;
            if (!resolveMixChildren(b, L.scene.mats[id], L)) return false;
        }

        // Pass 3: geometry, lights, medium, camera, render.
        bool haveLight = false;
        for (const auto& b : blocks) {
            if      (b.type == "sphere")   { if (!addSphere(b, L)) return false; }
            else if (b.type == "quad")     { if (!addQuad(b, L)) return false; }
            else if (b.type == "triangle") { if (!addTriangle(b, L)) return false; }
            else if (b.type == "mesh")     { if (!addMesh(b, L)) return false; }
            else if (b.type == "light")    { if (!addLight(b, L)) return false; haveLight = true; }
            else if (b.type == "medium")   { if (!addMedium(b, L)) return false; }
            else if (b.type == "camera")   { if (!addCamera(b, L)) return false; }
            else if (b.type == "camera_path") { if (!addCameraPath(b, L)) return false; }
            else if (b.type == "render")   { if (!applyRender(b, L)) return false; }
            else if (b.type == "scene" || b.type == "spectrum" || b.type == "material" || b.type == "texture") { /* handled */ }
            else { fail("unknown top-level block '" + b.type + "'"); return false; }
        }
        if (!haveLight) { fail("scene has no 'light' block"); return false; }

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
            if (g == "BK7") return iorBK7();
            if (g == "SF10") return iorSF10();
            fail("unknown glass '" + g + "'"); return iorBK7();
        }
        if (h.rfind("preset:", 0) == 0)  return resolvePreset(h.substr(7));
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

    // Illuminant presets, mirroring src/main.cpp resolveLight.
    Spectrum resolvePreset(const std::string& nm) {
        if (nm.rfind("bb", 0) == 0 && nm.size() > 2) { double k = num(nm.substr(2)); if (k > 0) return blackbody(k); }
        if (nm == "sun")                       return sunlight();
        if (nm == "daylight" || nm == "d65")   return daylight(6504.0);
        if (nm == "a" || nm == "incandescent") return illuminantA();
        if (nm == "led")                       return ledWhite(0.3);
        if (nm == "led-warm")                  return ledWhite(1.0);
        if (nm == "fluorescent" || nm == "cfl") return fluorescent();
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
        tex.buildReflCoeff();   // precompute Jakob-Hanika reflectance coefficients
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

    // ---- materials ----
    Material buildMaterial(const Block& b) {
        Material m;
        std::string type = strOf(b, "type", "diffuse");
        if (type == "diffuse") {
            m.type = MatType::Diffuse;
            // `reflect texture:<name>` binds a spatially-varying albedo; otherwise a
            // uniform reflectance spectrum. A bound texture leaves m.reflect as the
            // fallback used where UVs are unavailable (e.g. the CUDA bake path).
            if (bindReflectTexture(b, m)) m.reflect = constantSpectrum(0.75);
            else                          m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.75));
        } else if (type == "dielectric") {
            m.type = MatType::Dielectric;
            m.ior = spectrumParam(b, "ior", iorBK7());
        } else if (type == "mirror") {
            m.type = MatType::Mirror;
            m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.95));
        } else if (type == "halfmirror") {
            m.type = MatType::HalfMirror;
            m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.5));
        } else if (type == "glossy") {
            m.type = MatType::Glossy;
            m.reflect = spectrumParam(b, "reflect", constantSpectrum(0.9));
            m.roughness = dblOf(b, "roughness", 0.2);
        } else if (type == "thinfilm") {
            m.type = MatType::ThinFilm;
            m.ior = spectrumParam(b, "ior", iorConstant(1.5));
            m.filmIor = dblOf(b, "film_ior", 1.30);
            m.filmThickness = dblOf(b, "film_thickness", 300.0);
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
        } else if (type == "mix") {
            // Stochastic mix of named child materials. Children are resolved to
            // indices in a second pass (they may be declared later in the file);
            // here we only mark the type — resolveMixChildren() fills the lists.
            m.type = MatType::Mix;
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
            if (L.scene.mats[it->second].type == MatType::Mix) {
                fail("mix layer '" + cname + "' is itself a mix (nesting is not allowed)"); return false;
            }
            if (w < 0.0) { fail("mix layer weight must be >= 0"); return false; }
            m.mixChildren.push_back(it->second);
            m.mixWeights.push_back(w);
            sum += w;
        }
        if (m.mixChildren.empty()) { fail("mix material has no 'layer' entries"); return false; }
        if (sum > 1.0 + 1e-9) { fail("mix layer weights sum to " + std::to_string(sum) + " (> 1)"); return false; }
        return true;
    }

    // ---- geometry ----
    bool addSphere(const Block& b, Loaded& L) {
        Vec3 c{0, 0, 0}; vec3Of(b, "center", c);
        double r = dblOf(b, "radius", 1.0);
        std::string mat = strOf(b, "material");
        if (mat.empty()) { fail("sphere needs a material"); return false; }
        int id = matId(mat); if (!err.empty()) return false;
        L.scene.spheres.push_back(Sphere{P(c), Len(r), id});
        return true;
    }
    bool addQuad(const Block& b, Loaded& L) {
        Vec3 o{0, 0, 0}, u{1, 0, 0}, v{0, 0, 1};
        vec3Of(b, "origin", o); vec3Of(b, "u", u); vec3Of(b, "v", v);
        std::string mat = strOf(b, "material");
        if (mat.empty()) { fail("quad needs a material"); return false; }
        int id = matId(mat); if (!err.empty()) return false;
        Vec3 a = P(o), bb = P(o + u), cc = P(o + u + v), dd = P(o + v);
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
    bool addTriangle(const Block& b, Loaded& L) {
        Vec3 v0{0, 0, 0}, v1{1, 0, 0}, v2{0, 1, 0};
        vec3Of(b, "v0", v0); vec3Of(b, "v1", v1); vec3Of(b, "v2", v2);
        std::string mat = strOf(b, "material");
        if (mat.empty()) { fail("triangle needs a material"); return false; }
        int id = matId(mat); if (!err.empty()) return false;
        L.scene.tris.push_back(Tri{P(v0), P(v1), P(v2), id, -1, {}});
        return true;
    }
    bool addMesh(const Block& b, Loaded& L) {
        std::string file = strOf(b, "file");
        if (file.empty()) { fail("mesh needs a file"); return false; }
        std::string mat = strOf(b, "material");
        if (mat.empty()) { fail("mesh needs a material"); return false; }
        int id = matId(mat); if (!err.empty()) return false;
        MeshXform xf;
        vec3Of(b, "translate", xf.translate);
        vec3Of(b, "rotate", xf.rotDeg);
        // scale accepts a single uniform value or a vec3.
        const Stmt* sc = find(b, "scale");
        if (sc) {
            if (sc->val.words.size() >= 3)
                xf.scale = {num(sc->val.words[0]), num(sc->val.words[1]), num(sc->val.words[2])};
            else if (!sc->val.words.empty()) {
                double k = num(sc->val.words[0]); xf.scale = {k, k, k};
            }
        }
        // Fold the unit scale into the transform: both the scaled local verts and the
        // translation live in authored units, so multiply both by L_ to reach metres.
        xf.scale = xf.scale * L_;
        xf.translate = xf.translate * L_;
        // `uv use_mesh` reads texture coordinates from the OBJ's `vt` records (needed
        // for textured materials); the default keeps the Tri fallback UVs.
        bool loadUV = (strOf(b, "uv") == "use_mesh");
        // `usemtl use_names` switches material per OBJ `usemtl` group by matching the
        // group name to an FTSL material of the same name (unknown -> the mesh's
        // default `material`). Two-token maps can't survive the statement splitter,
        // so name-matching is the grammar-friendly convention (mirrors `uv use_mesh`).
        bool useNames = (strOf(b, "usemtl") == "use_names");
        MtlResolver resolver = [this](const std::string& nm) -> int {
            auto it = matIndex_.find(nm);
            return (it == matIndex_.end()) ? -1 : it->second;
        };
        loadObj(L.scene, file.c_str(), id, xf, loadUV, useNames ? &resolver : nullptr);
        return true;
    }

    // ---- lights ----
    // Each `light` block registers one Emitter. Multiple light blocks accumulate;
    // the forward tracer selects among them power-weighted and the backward
    // reference sums over them (see scene.h / render.h / backward.h).
    bool addLight(const Block& b, Loaded& L) {
        Spectrum spd = spectrumParam(b, "spd", blackbody(6500.0));
        if (b.subtype == "collimated") {
            Vec3 dir{0, 0, -1}; vec3Of(b, "dir", dir);
            Vec3 beam = normalize(dir);
            // A thin pencil cross-section at the given origin (3 cm pencil).
            Vec3 o{0.5, 0.5, 0.95}; vec3Of(b, "origin", o);
            Vec3 t, bt; onb(beam, t, bt);
            double w = Len(0.03);
            L.scene.addAreaLight(P(o), t * w, bt * w, beam, w * w, spd, binWidth_,
                                 /*collimated*/true, beam);
            return true;
        }
        if (b.subtype == "sphere") {
            // Spherical area light: a glowing ball. Also add an emissive sphere to
            // geometry so photons that strike it are absorbed and it is visible in
            // the photon-catch camera modes (mirrors the area-light quad below).
            Vec3 c{0.5, 0.7, 0.5}; vec3Of(b, "center", c);
            double rad = Len(dblOf(b, "radius", 0.1));
            Material lm; lm.reflect = constantSpectrum(0.0); lm.emit = spd; lm.isLight = true;
            int id = (int)L.scene.mats.size(); L.scene.mats.push_back(lm);
            L.scene.spheres.push_back(Sphere{P(c), rad, id});
            L.scene.addSphereLight(P(c), rad, spd, binWidth_);
            return true;
        }
        if (b.subtype == "spot") {
            // Point spotlight: a cone about `dir`, smoothstep penumbra between the
            // inner and outer half-angles (degrees). No emissive geometry (a point).
            Vec3 o{0.5, 0.99, 0.5}; vec3Of(b, "origin", o);
            Vec3 dir{0, -1, 0}; vec3Of(b, "dir", dir);
            double inner = dblOf(b, "inner_angle", 20.0);
            double outer = dblOf(b, "outer_angle", 30.0);
            if (outer < inner) outer = inner;            // outer cone must enclose inner
            const double d2r = PI / 180.0;
            double cosInner = std::cos(inner * d2r), cosOuter = std::cos(outer * d2r);
            L.scene.addSpotLight(P(o), normalize(dir), cosInner, cosOuter, spd, binWidth_);
            return true;
        }
        // Default: rectangular area light. Also add the emissive quad to geometry so
        // photons landing back on it are absorbed (matches buildCornell).
        Vec3 o{0, 1, 0}, u{1, 0, 0}, v{0, 0, 1}, nrm{0, -1, 0};
        vec3Of(b, "origin", o); vec3Of(b, "u", u); vec3Of(b, "v", v);
        if (!vec3Of(b, "normal", nrm)) nrm = normalize(cross(u, v));
        Vec3 os = P(o), us = u * L_, vs = v * L_;
        Material lm; lm.reflect = constantSpectrum(0.0); lm.emit = spd; lm.isLight = true;
        int id = (int)L.scene.mats.size(); L.scene.mats.push_back(lm);
        Vec3 a = os, bb = os + us, cc = os + us + vs, dd = os + vs;
        L.scene.tris.push_back(Tri{a, bb, cc, id, -1, {}});
        L.scene.tris.push_back(Tri{a, cc, dd, id, -1, {}});
        L.scene.addAreaLight(os, us, vs, normalize(nrm), length(cross(us, vs)), spd, binWidth_);
        return true;
    }

    // ---- medium ----
    bool addMedium(const Block& b, Loaded& L) {
        L.scene.medium.enabled = true;
        L.scene.medium.g = dblOf(b, "g", 0.0);
        bool rayleigh = strOf(b, "rayleigh") == "true" || strOf(b, "rayleigh") == "1";
        // Extinction coefficients are per-length (1/authored-unit); divide by L_ to
        // convert to the internal 1/metre so fog reads the same regardless of unit.
        const double invL = 1.0 / L_;
        const Stmt* sa = find(b, "sigma_a");
        const Stmt* ss = find(b, "sigma_s");
        if (sa || ss) {
            Spectrum a = sa ? evalSpectrum(sa->val) : constantSpectrum(0.0);
            Spectrum s = ss ? evalSpectrum(ss->val) : constantSpectrum(0.0);
            L.scene.medium.sigma_a = [a, invL](double w) { return a(w) * invL; };
            L.scene.medium.sigma_s = [s, invL](double w) { return s(w) * invL; };
        } else {
            double sigmaT = dblOf(b, "sigma_t", 0.0) * invL;
            double albedo = dblOf(b, "albedo", 0.9);
            double s_s = albedo * sigmaT, s_a = (1.0 - albedo) * sigmaT;
            if (rayleigh) {
                L.scene.medium.sigma_s = [s_s](double w) { double r = 550.0 / w; double r2 = r * r; return s_s * r2 * r2; };
                L.scene.medium.sigma_a = constantSpectrum(s_a);
            } else {
                L.scene.medium.sigma_s = constantSpectrum(s_s);
                L.scene.medium.sigma_a = constantSpectrum(s_a);
            }
        }
        return true;
    }

    // Read the film sub-block + photographic exposure/f-stop controls shared by
    // `camera` and `camera_path`, and resolve the derived focal length, f-stop ->
    // aperture radius, and manual exposure multiplier. `cs.fov` must already be set.
    void readFilmExposure(const Block& b, CamSpec& cs) {
        const Stmt* film = find(b, "film");
        if (film && film->val.block) {
            const Block& fb = *film->val.block;
            const Stmt* r = find(fb, "res");
            if (r && !r->val.words.empty()) cs.res = (int)num(r->val.words[0]);
            const Stmt* sz = find(fb, "size");     // physical sensor, millimetres
            if (sz && sz->val.words.size() >= 2) {
                cs.filmW_mm = num(sz->val.words[0]);
                cs.filmH_mm = num(sz->val.words[1]);
            }
            cs.iso      = dblOf(fb, "iso", 0.0);
            cs.shutter  = dblOf(fb, "shutter", 0.0);
            cs.exposure = dblOf(fb, "exposure", 0.0);
        }
        // Focal length (metres) from the vertical fov and physical film height.
        // fov_y = 2*atan(filmH/(2f)) -> f = filmH / (2 tan(fov/2)). Fall back to a
        // 35mm full-frame 24mm height when no physical size is authored (only used
        // where a real length is needed, i.e. f-stop -> aperture).
        double hmm = (cs.filmH_mm > 0.0) ? cs.filmH_mm : 24.0;
        double th  = std::tan(0.5 * cs.fov * 3.141592653589793 / 180.0);
        cs.focal = (th > 1e-9) ? (hmm / 1000.0) / (2.0 * th) : 0.0;
        // f-stop authoring: N = f / (2*apertureR) -> apertureR = f / (2N). Overrides
        // any `aperture` radius. Aperture radius is an internal (metre) length.
        double fstop = dblOf(b, "fstop", 0.0);
        if (fstop > 0.0 && cs.focal > 0.0) cs.aperture = cs.focal / (2.0 * fstop);
        // Manual exposure multiplier (see CamSpec). Active iff any control authored.
        if (cs.exposure > 0.0 || cs.iso > 0.0 || cs.shutter > 0.0) {
            double base = (cs.exposure > 0.0) ? cs.exposure : 1.0;
            double isoF = (cs.iso     > 0.0) ? cs.iso / 100.0 : 1.0;
            double shF  = (cs.shutter > 0.0) ? cs.shutter     : 1.0;
            cs.exposureMul = base * isoF * shF;
        }
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
        readFilmExposure(b, cs);   // film{res,size,iso,shutter,exposure}, fstop
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
        readFilmExposure(b, shared);   // film{res,size,iso,shutter,exposure}, fstop
        int frames = (int)dblOf(b, "frames", 0.0);
        if (frames < 1) { fail("camera_path '" + base + "' needs frames >= 1"); return false; }

        // Collect keyframes (t, eye, optional look_at), sorted by t.
        struct Key { double t; Vec3 eye, look; bool hasLook; };
        std::vector<Key> keys;
        for (const auto& s : b.stmts) {
            if (s.key != "key") continue;
            const auto& w = s.val.words;
            if (w.size() < 4) { fail("camera_path key needs: t ex ey ez [lx ly lz]"); return false; }
            Key k;
            k.t = num(w[0]);
            k.eye = P(Vec3{num(w[1]), num(w[2]), num(w[3])});
            k.hasLook = (w.size() >= 7);
            k.look = k.hasLook ? P(Vec3{num(w[4]), num(w[5]), num(w[6])}) : shared.look;
            keys.push_back(k);
        }
        if (keys.size() < 2) { fail("camera_path '" + base + "' needs >= 2 keys"); return false; }
        std::sort(keys.begin(), keys.end(), [](const Key& a, const Key& b2){ return a.t < b2.t; });

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
            char num5[8]; std::snprintf(num5, sizeof(num5), "%0*d", pad, i);
            cs.name = base + num5;
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
