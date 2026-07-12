// Spectral asset library: loads measured/tabulated spectral DATA from external files
// under data/, instead of compiling it into the binary. This is the single home for
// the four "pick a named real-world spectrum" resolvers that used to be baked tables
// in spectrum.h / materials.h / lights.h:
//
//   resolveGlassIor(name)          -> data/glass/<name>.glass   (dispersion coeffs)
//   resolveMetalReflectance(name)  -> data/metal/<name>.csv     (reflectance R(lambda))
//   resolveNaturalReflectance(name)-> data/reflectance/<name>.csv
//   resolveTabulatedIlluminant(name)-> data/illuminant/<name>.csv (measured SPDs)
//
// Only DATA moved out; the ALGORITHMS stay native — glass files carry Sellmeier /
// Cauchy coefficients that this loader feeds to the native `sellmeier()`/`cauchy()`
// evaluators in spectrum.h, and the curve files are ingested by `tabulatedSpectrum()`.
// Parametric sources (blackbody, LED, gas-discharge line models, iridescent recipes)
// are algorithms and remain in code.
//
// A category is just a directory of files. The lookup key is the lowercased filename
// stem; a file may declare extra names with a `# aliases: a b c` header line. So the
// library is drop-in extensible: add a file to data/<category>/ and it resolves by
// name with no rebuild. Paths resolve relative to the library root (default "data",
// i.e. the current working directory), matching FTSL's `file:` / mesh / texture refs.
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include "spectrum.h"

namespace speclib {

// Library root directory (holds glass/, metal/, reflectance/, illuminant/). Default
// "data" resolves relative to the process working directory. Overridable if a future
// CLI flag wants an alternate asset root.
inline std::string& root() { static std::string r = "data"; return r; }
inline void setRoot(const std::string& r) { root() = r; }

inline std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
inline bool isNumberTok(const std::string& s) {
    if (s.empty()) return false;
    char* e = nullptr; std::strtod(s.c_str(), &e); return e == s.c_str() + s.size();
}

// Load a two-column spectrum file (wavelength_nm,value) into sorted-ready pairs. The
// format is liberal: '#' comments, comma OR whitespace delimiters, non-numeric header
// rows skipped. Same contract as the FTSL `file:` loader (which forwards here).
inline bool loadSpdCsv(const std::string& path,
                       std::vector<std::pair<double, double>>& out, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open spectrum file: " + path; return false; }
    out.clear();
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto h = line.find('#'); if (h != std::string::npos) line.erase(h);
        for (char& c : line) if (c == ',' || c == '\t') c = ' ';
        std::istringstream ss(line); std::string a, b;
        if (!(ss >> a >> b)) continue;
        if (!isNumberTok(a) || !isNumberTok(b)) continue;
        out.push_back({std::strtod(a.c_str(), nullptr), std::strtod(b.c_str(), nullptr)});
    }
    if (out.empty()) { err = "spectrum file has no numeric rows: " + path; return false; }
    return true;
}

// Build (once, cached) a name->path index for a category directory: canonical key is
// the lowercased filename stem, plus any `# aliases:` declared in a file's header.
inline const std::unordered_map<std::string, std::string>& index(const std::string& category) {
    static std::unordered_map<std::string, std::unordered_map<std::string, std::string>> cache;
    auto cit = cache.find(category);
    if (cit != cache.end()) return cit->second;

    std::unordered_map<std::string, std::string> idx;
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path dir = fs::path(root()) / category;
    if (fs::exists(dir, ec) && fs::is_directory(dir, ec)) {
        for (const auto& e : fs::directory_iterator(dir, ec)) {
            if (ec || !e.is_regular_file()) continue;
            std::string path = e.path().string();
            idx[lower(e.path().stem().string())] = path;      // canonical = filename stem
            std::ifstream f(path); std::string line; int n = 0; // scan header for aliases
            while (std::getline(f, line) && n++ < 12) {
                if (line.empty() || line[0] != '#') { if (n > 1) break; else continue; }
                auto p = line.find("aliases:");
                if (p == std::string::npos) continue;
                std::istringstream ss(line.substr(p + 8)); std::string a;
                while (ss >> a) idx[lower(a)] = path;
            }
        }
    }
    return (cache[category] = std::move(idx));
}

inline bool findFile(const std::string& category, const std::string& name, std::string& path) {
    const auto& idx = index(category);
    auto it = idx.find(lower(name));
    if (it == idx.end()) return false;
    path = it->second; return true;
}

// Resolve + cache a measured CURVE (metal / reflectance / illuminant) as a
// piecewise-linear Spectrum. Repeated references to the same file share one curve.
inline bool loadCurve(const std::string& category, const std::string& name, Spectrum& out) {
    std::string path;
    if (!findFile(category, name, path)) return false;
    static std::unordered_map<std::string, Spectrum> cache;
    auto it = cache.find(path);
    if (it != cache.end()) { out = it->second; return true; }
    std::vector<std::pair<double, double>> pairs; std::string err;
    if (!loadSpdCsv(path, pairs, err)) return false;
    Spectrum s = tabulatedSpectrum(std::move(pairs));
    cache[path] = s; out = s; return true;
}

// Resolve + cache a GLASS dispersion file into an IOR Spectrum. The file names a native
// evaluator by `form` and supplies its coefficients:
//   form sellmeier / B b1 b2 b3 / C c1 c2 c3      -> sellmeier(B..,C..)
//   form cauchy     / A a / B b                    -> cauchy(a, b)
//   form constant   / n 1.5                        -> iorConstant(n)
inline bool loadGlass(const std::string& name, Spectrum& out) {
    std::string path;
    if (!findFile("glass", name, path)) return false;
    static std::unordered_map<std::string, Spectrum> cache;
    auto it = cache.find(path);
    if (it != cache.end()) { out = it->second; return true; }
    std::ifstream f(path);
    if (!f) return false;
    std::string form, line;
    std::vector<double> B, C, A, N;
    while (std::getline(f, line)) {
        auto h = line.find('#'); if (h != std::string::npos) line.erase(h);
        std::istringstream ss(line); std::string key;
        if (!(ss >> key)) continue;
        if (key == "form") { ss >> form; continue; }
        std::vector<double>* dst = (key == "B") ? &B : (key == "C") ? &C
                                 : (key == "A") ? &A : (key == "n") ? &N : nullptr;
        if (dst) { double v; while (ss >> v) dst->push_back(v); }
    }
    Spectrum s;
    if (form == "sellmeier" && B.size() >= 3 && C.size() >= 3)
        s = sellmeier(B[0], B[1], B[2], C[0], C[1], C[2]);
    else if (form == "cauchy" && A.size() >= 1 && B.size() >= 1)
        s = cauchy(A[0], B[0]);
    else if (form == "constant" && N.size() >= 1)
        s = iorConstant(N[0]);
    else return false;
    cache[path] = s; out = s; return true;
}

} // namespace speclib

// ---------------------------------------------------------------------------
// Global resolvers — SAME names/signatures as the former baked-in tables, so every
// call site (`if (resolveGlassIor(name, out)) ...`) is unchanged; only the data
// source moved from compiled arrays to data/ files. Each returns false on an unknown
// name so the caller keeps its existing error/fallback behaviour.
// ---------------------------------------------------------------------------
inline bool resolveGlassIor(const std::string& name, Spectrum& out)          { return speclib::loadGlass(name, out); }
inline bool resolveMetalReflectance(const std::string& name, Spectrum& out)  { return speclib::loadCurve("metal", name, out); }
inline bool resolveNaturalReflectance(const std::string& name, Spectrum& out){ return speclib::loadCurve("reflectance", name, out); }
inline bool resolveTabulatedIlluminant(const std::string& name, Spectrum& out){ return speclib::loadCurve("illuminant", name, out); }
