// loom `CurveDrive` JSON sidecar — the E2 "channel a" config, shared by ftrace's
// interactive curve editor (`-explore -anim <file.json>`) and loom's `loom.anim`.
//
// E2 splits authoring into two channels:
//   (a) the WHOLE-VIDEO config — an N-dimensional control-point curve plus the
//       channel->scene-variable bindings that say what each curve dimension drives.
//       That config lives in this JSON sidecar, written atomically by whichever side
//       edited it last, and is what this header reads and writes.
//   (b) the PER-FRAME live values — newline-delimited JSON over a pipe (see
//       loom/anim.py `LiveSession` / `serve_live`), which never touches disk.
// Keeping (a) on disk in a plain, human-diffable format is what makes the round trip
// honest: loom can propose a drive, the C++ editor reshapes it in 3D, loom reads the
// reshaped one back, and neither side owns the file.
//
// The schema mirrors `loom.anim.CurveDrive.to_dict()` exactly (SIDECAR_VERSION 1):
//   { "version": 1, "name": str, "mode": "animation"|"flyby", "dims": int,
//     "closed": bool, "points": [[float x dims], ...],
//     "bindings": [{"channel": int, "target": str, "mode": "pin"|"mod",
//                   "gain": float, "kind": "additive"|"gain"|"bipolar"}, ...] }
// `load` enforces the same invariants `CurveDrive.__init__` does (dims >= 1, >= 2
// points, every point exactly `dims` wide, every binding channel < dims), so ftrace
// never accepts — or writes — a sidecar loom would reject.
#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <system_error>
#include <filesystem>
#include "third_party/json.h"

namespace curvedrive {

// Bump in lockstep with loom.anim.SIDECAR_VERSION on any breaking shape change.
static const int kSidecarVersion = 1;

// Authoring modes (loom.anim.MODE_*).
static const char* const kModeFlyby     = "flyby";       // channels collapse to camera pose
static const char* const kModeAnimation = "animation";   // any channel -> any variable

// One channel -> scene-variable edge (loom.anim.ChannelBinding / the E5 influence model).
struct Binding {
    int         channel = 0;            // curve dimension index
    std::string target;                 // scene variable name
    std::string mode   = "pin";         // "pin" (replace) | "mod" (accumulate)
    double      gain   = 1.0;
    std::string kind   = "additive";    // "additive" | "gain" | "bipolar"
};

// The sidecar as a whole.
struct Drive {
    int         dims   = 3;
    std::string name   = "drive";
    std::string mode   = kModeAnimation;
    bool        closed = false;
    std::vector<std::vector<double>> points;   // each exactly `dims` wide
    std::vector<Binding>             bindings;

    // Distinct driven variables, in first-seen order (loom's CurveDrive.targets()).
    std::vector<std::string> targets() const {
        std::vector<std::string> seen;
        for (const auto& b : bindings) {
            bool have = false;
            for (const auto& s : seen) if (s == b.target) { have = true; break; }
            if (!have) seen.push_back(b.target);
        }
        return seen;
    }
    // Bindings on one channel (several channels may drive one target and vice versa).
    std::vector<const Binding*> onChannel(int ch) const {
        std::vector<const Binding*> out;
        for (const auto& b : bindings) if (b.channel == ch) out.push_back(&b);
        return out;
    }
};

// ---- JSON writing ---------------------------------------------------------------
namespace detail {

// Shortest representation that still round-trips through strtod, so an editor pass
// that only moved point 3 leaves every other coordinate byte-identical.
inline std::string num(double v) {
    if (!std::isfinite(v)) return "0";   // NaN/+-inf have no JSON spelling; write a neutral 0
    char buf[40];
    for (int prec = 6; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof buf, "%.*g", prec, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    return buf;
}

inline std::string str(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) { char e[8]; std::snprintf(e, sizeof e, "\\u%04x", c); o += e; }
                else o += (char)c;
        }
    }
    return o + "\"";
}

}  // namespace detail

// Serialise with the same two-space indentation `json.dumps(..., indent=2)` uses, except
// that a control point is kept on ONE line (loom's writer explodes every coordinate onto
// its own line). Both are valid JSON and each side reads the other; the one-line form
// just makes "the editor moved point 7" a one-line diff instead of a `dims`-line one.
inline std::string toJson(const Drive& d) {
    std::string o;
    o += "{\n";
    o += "  \"version\": " + std::to_string(kSidecarVersion) + ",\n";
    o += "  \"name\": " + detail::str(d.name) + ",\n";
    o += "  \"mode\": " + detail::str(d.mode) + ",\n";
    o += "  \"dims\": " + std::to_string(d.dims) + ",\n";
    o += std::string("  \"closed\": ") + (d.closed ? "true" : "false") + ",\n";
    o += "  \"points\": [";
    if (d.points.empty()) o += "]";
    else {
        o += "\n";
        for (size_t i = 0; i < d.points.size(); ++i) {
            o += "    [";
            for (size_t c = 0; c < d.points[i].size(); ++c) {
                if (c) o += ", ";
                o += detail::num(d.points[i][c]);
            }
            o += "]";
            o += (i + 1 < d.points.size()) ? ",\n" : "\n";
        }
        o += "  ]";
    }
    o += ",\n  \"bindings\": [";
    if (d.bindings.empty()) o += "]";
    else {
        o += "\n";
        for (size_t i = 0; i < d.bindings.size(); ++i) {
            const Binding& b = d.bindings[i];
            o += "    {\n";
            o += "      \"channel\": " + std::to_string(b.channel) + ",\n";
            o += "      \"target\": " + detail::str(b.target) + ",\n";
            o += "      \"mode\": " + detail::str(b.mode) + ",\n";
            o += "      \"gain\": " + detail::num(b.gain) + ",\n";
            o += "      \"kind\": " + detail::str(b.kind) + "\n";
            o += "    }";
            o += (i + 1 < d.bindings.size()) ? ",\n" : "\n";
        }
        o += "  ]";
    }
    o += "\n}";
    return o;
}

// ---- validation -----------------------------------------------------------------
// The exact invariants loom's CurveDrive constructor enforces. Checked on BOTH load
// and save: refusing to write a bad sidecar is what keeps the round trip closed.
inline bool validate(const Drive& d, std::string& err) {
    if (d.dims < 1) { err = "dims must be >= 1"; return false; }
    if (d.points.size() < 2) { err = "need >= 2 control points"; return false; }
    for (size_t i = 0; i < d.points.size(); ++i)
        if ((int)d.points[i].size() != d.dims) {
            err = "control point " + std::to_string(i) + " has " +
                  std::to_string(d.points[i].size()) + " coords, expected dims=" +
                  std::to_string(d.dims);
            return false;
        }
    if (d.mode != kModeFlyby && d.mode != kModeAnimation) {
        err = "mode must be \"flyby\" or \"animation\", got \"" + d.mode + "\""; return false;
    }
    for (const auto& b : d.bindings) {
        if (b.channel < 0 || b.channel >= d.dims) {
            err = "binding channel " + std::to_string(b.channel) + " out of range for dims " +
                  std::to_string(d.dims) + " (target \"" + b.target + "\")";
            return false;
        }
        if (b.mode != "pin" && b.mode != "mod") {
            err = "binding mode must be \"pin\" or \"mod\", got \"" + b.mode + "\""; return false;
        }
        if (b.kind != "additive" && b.kind != "gain" && b.kind != "bipolar") {
            err = "binding kind must be additive/gain/bipolar, got \"" + b.kind + "\""; return false;
        }
    }
    return true;
}

// ---- disk ------------------------------------------------------------------------
inline bool load(const std::string& path, Drive& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) { err = "cannot open " + path; return false; }
    std::ostringstream ss; ss << f.rdbuf();
    std::string text = ss.str();

    minijson::Value root;
    minijson::Parser p(text);
    if (!p.parse(root, err) || !root.isObject()) {
        if (err.empty()) err = "not a JSON object";
        return false;
    }
    int ver = root.intAt("version", kSidecarVersion);
    if (ver != kSidecarVersion) {
        err = "sidecar version " + std::to_string(ver) + " != supported " +
              std::to_string(kSidecarVersion);
        return false;
    }
    Drive d;
    d.dims = root.intAt("dims", 3);
    if (const minijson::Value* v = root.find("name"))   d.name   = v->asString(d.name);
    if (const minijson::Value* v = root.find("mode"))   d.mode   = v->asString(d.mode);
    if (const minijson::Value* v = root.find("closed")) d.closed = v->asBool(false);
    if (const minijson::Value* pts = root.find("points")) {
        if (!pts->isArray()) { err = "\"points\" must be an array"; return false; }
        for (const auto& p3 : pts->arr) {
            if (!p3.isArray()) { err = "each control point must be an array"; return false; }
            std::vector<double> row;
            row.reserve(p3.arr.size());
            for (const auto& c : p3.arr) row.push_back(c.asNumber(0.0));
            d.points.push_back(std::move(row));
        }
    }
    if (const minijson::Value* bs = root.find("bindings")) {
        if (!bs->isArray()) { err = "\"bindings\" must be an array"; return false; }
        for (const auto& bv : bs->arr) {
            Binding b;
            b.channel = bv.intAt("channel", 0);
            b.gain    = bv.numAt("gain", 1.0);
            if (const minijson::Value* v = bv.find("target")) b.target = v->asString("");
            if (const minijson::Value* v = bv.find("mode"))   b.mode   = v->asString(b.mode);
            if (const minijson::Value* v = bv.find("kind"))   b.kind   = v->asString(b.kind);
            if (b.target.empty()) { err = "binding is missing a \"target\""; return false; }
            d.bindings.push_back(std::move(b));
        }
    }
    if (!validate(d, err)) return false;
    out = std::move(d);
    return true;
}

// Atomic write (temp file + rename), so a reader — loom, or a second ftrace — never
// sees a half-written config. Mirrors CurveDrive.save()'s mkstemp + os.replace.
inline bool save(const std::string& path, const Drive& d, std::string& err) {
    if (!validate(d, err)) return false;
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f.good()) { err = "cannot write " + tmp; return false; }
        f << toJson(d) << "\n";
        f.flush();
        if (!f.good()) { err = "write failed for " + tmp; return false; }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        err = "cannot replace " + path;
        return false;
    }
    return true;
}

}  // namespace curvedrive
