// 2D image textures for spatially-varying material parameters (Phase 3b).
//
// A Texture holds an image decoded to LINEAR values, plus (for reflectance use)
// per-texel Jakob-Hanika sigmoid coefficients so a bound albedo can be turned into
// a physical reflectance spectrum at any wavelength without a per-hit fit. It is
// sampled at a surface (u,v) with a wrap mode and nearest/bilinear filtering.
//
// The loader has a small built-in path for the formats the project itself produces
// (PPM P6/P3, 8-bit sRGB or linear; PFM Pf/PF float, always linear) and defers every
// other format to the vendored stb_image (PNG/JPG/BMP/TGA + Radiance .hdr). To keep
// the 8k-line stb header out of every TU (especially nvcc), we forward-declare just
// the handful of stbi_* functions we call here; the implementation is compiled once
// in src/stb_image_impl.cpp. Everything downstream (sampling, coefficient precompute,
// the render/backward/ftsl plumbing) is format-agnostic.
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include "linalg.h"
#include "color.h"
#include "upsample.h"
#include "spectrum.h"

// stb_image API (implementation lives in src/stb_image_impl.cpp). Declared here so
// this header stays light and CUDA never parses the full stb_image.h.
extern "C" {
    unsigned char* stbi_load(const char* filename, int* x, int* y, int* channels, int desired);
    float*         stbi_loadf(const char* filename, int* x, int* y, int* channels, int desired);
    void           stbi_image_free(void* retval_from_stbi_load);
    int            stbi_is_hdr(const char* filename);
    const char*    stbi_failure_reason(void);
}

enum class TexEncoding { sRGB, Linear };
enum class TexFilter   { Nearest, Bilinear };
enum class TexWrap     { Repeat, Clamp, Mirror };

struct Texture {
    std::string name;
    int w = 0, h = 0;
    std::vector<Vec3> rgb;                    // decoded LINEAR rgb, row-major, top-left origin
    std::vector<std::array<double, 3>> coeff; // per-texel JH reflectance coefficients (built on demand)
    // Indexed-spectral palette (spec §9.3): when non-empty, this texture is an INDEX
    // map — the red channel quantized to 0..255 selects a named reflectance spectrum
    // from `palette` (resolved at parse time) instead of an RGB colour to upsample.
    // Indices never interpolate (nearest only). GPU falls back to CPU for such maps.
    std::vector<Spectrum> palette;
    bool hasPalette() const { return !palette.empty(); }
    TexEncoding encoding = TexEncoding::sRGB;
    TexFilter   filter   = TexFilter::Bilinear;
    TexWrap     wrap     = TexWrap::Repeat;

    bool valid() const { return w > 0 && h > 0 && (int)rgb.size() == w * h; }

    // Precompute the Jakob-Hanika reflectance coefficients per texel (once). Called
    // when a texture is bound to a reflectance parameter so per-hit sampling is a
    // cheap coefficient bilerp + sigmoid evaluation rather than a Gauss-Newton fit.
    void buildReflCoeff() {
        if (!valid() || hasPalette() || coeff.size() == (size_t)w * h) return;
        coeff.resize((size_t)w * h);
        for (size_t i = 0; i < rgb.size(); ++i)
            coeff[i] = upsample::fit(rgb[i].x, rgb[i].y, rgb[i].z);
    }

    // ---- sampling -----------------------------------------------------------
    int wrapIndex(int i, int n) const {
        switch (wrap) {
            case TexWrap::Clamp:  return (i < 0) ? 0 : (i >= n ? n - 1 : i);
            case TexWrap::Mirror: {
                int period = 2 * n;
                int m = ((i % period) + period) % period;
                return (m < n) ? m : (period - 1 - m);
            }
            case TexWrap::Repeat:
            default: { int m = i % n; return (m < 0) ? m + n : m; }
        }
    }

    // Map (u,v) in texture space to the four surrounding texel indices + weights.
    // v is flipped so v=0 is the bottom of the image (standard OBJ/UV convention).
    void bilerpSetup(double u, double v, int& x0, int& y0, int& x1, int& y1,
                     double& fx, double& fy) const {
        double tu = u * w - 0.5;
        double tv = (1.0 - v) * h - 0.5;
        double flx = std::floor(tu), fly = std::floor(tv);
        fx = tu - flx; fy = tv - fly;
        x0 = wrapIndex((int)flx,     w); x1 = wrapIndex((int)flx + 1, w);
        y0 = wrapIndex((int)fly,     h); y1 = wrapIndex((int)fly + 1, h);
    }

    Vec3 sampleRgb(double u, double v) const {
        if (!valid()) return Vec3{0.5, 0.5, 0.5};
        if (filter == TexFilter::Nearest) {
            int x = wrapIndex((int)std::floor(u * w), w);
            int y = wrapIndex((int)std::floor((1.0 - v) * h), h);
            return rgb[(size_t)y * w + x];
        }
        int x0, y0, x1, y1; double fx, fy;
        bilerpSetup(u, v, x0, y0, x1, y1, fx, fy);
        const Vec3& c00 = rgb[(size_t)y0 * w + x0];
        const Vec3& c10 = rgb[(size_t)y0 * w + x1];
        const Vec3& c01 = rgb[(size_t)y1 * w + x0];
        const Vec3& c11 = rgb[(size_t)y1 * w + x1];
        Vec3 a = c00 * (1 - fx) + c10 * fx;
        Vec3 b = c01 * (1 - fx) + c11 * fx;
        return a * (1 - fy) + b * fy;
    }

    // Scalar (grayscale) sample of the LINEAR image, for NON-colour parameters —
    // roughness, film thickness, mix weight, etc. (spec §9.4). Averages the three
    // linear channels (grayscale maps carry r=g=b, so this is exact for them; a
    // colour map degrades gracefully to its luminance-ish mean). No Jakob-Hanika
    // upsampling — the value is used directly as the scalar parameter, so such maps
    // should be authored `encoding linear`. Mirrored on the GPU by dTexScalarAt.
    double scalarAt(double u, double v) const {
        Vec3 c = sampleRgb(u, v);
        return (c.x + c.y + c.z) * (1.0 / 3.0);
    }

    // Reflectance at (u,v,lambda): bilerp the JH coefficients (the standard
    // Jakob-Hanika interpolation) then evaluate the sigmoid. Requires buildReflCoeff().
    // Nearest-index palette lookup: the red channel (already LINEAR, 0..1) is the
    // index / 255. Indices are categorical, so this never bilerps — it snaps to the
    // nearest texel and clamps the index into the palette. Returns the selected
    // spectrum's reflectance at lambda.
    double paletteReflectanceAt(double u, double v, double lambda) const {
        int x = wrapIndex((int)std::floor(u * w), w);
        int y = wrapIndex((int)std::floor((1.0 - v) * h), h);
        int idx = (int)std::lround(rgb[(size_t)y * w + x].x * 255.0);
        if (idx < 0) idx = 0;
        if (idx >= (int)palette.size()) idx = (int)palette.size() - 1;
        return palette[idx] ? palette[idx](lambda) : 0.0;
    }

    double reflectanceAt(double u, double v, double lambda) const {
        if (hasPalette()) return paletteReflectanceAt(u, v, lambda);
        if (coeff.empty()) return 0.5;
        auto at = [&](int x, int y) -> const std::array<double, 3>& {
            return coeff[(size_t)y * w + x];
        };
        if (filter == TexFilter::Nearest) {
            int x = wrapIndex((int)std::floor(u * w), w);
            int y = wrapIndex((int)std::floor((1.0 - v) * h), h);
            return upsample::reflAt(at(x, y), lambda);
        }
        int x0, y0, x1, y1; double fx, fy;
        bilerpSetup(u, v, x0, y0, x1, y1, fx, fy);
        std::array<double, 3> c;
        for (int k = 0; k < 3; ++k) {
            double a = at(x0, y0)[k] * (1 - fx) + at(x1, y0)[k] * fx;
            double b = at(x0, y1)[k] * (1 - fx) + at(x1, y1)[k] * fx;
            c[k] = a * (1 - fy) + b * fy;
        }
        return upsample::reflAt(c, lambda);
    }

    // Triplanar (box) projection reflectance at a world hit. Samples the texture
    // from the three world axes — the plane perpendicular to X at (z,y), to Y at
    // (x,z), to Z at (x,y), each world coordinate multiplied by `scale` (texture
    // repeats per world unit) — and blends the three by the surface normal, weighted
    // by |n|^4 componentwise (sharp seams, distortion-free). No per-vertex UVs are
    // used; this is the renderer-side mapping for un-UV'd/organic meshes (spec §9.2).
    // Mirrored on the GPU by dTexReflTriplanar in render_cuda.cu.
    double reflectanceTriplanar(const Vec3& p, const Vec3& n, double scale, double lambda) const {
        double ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
        double wx = ax * ax * ax * ax, wy = ay * ay * ay * ay, wz = az * az * az * az;
        double s = wx + wy + wz;
        if (s <= 0.0) return reflectanceAt(p.x * scale, p.y * scale, lambda);
        wx /= s; wy /= s; wz /= s;
        double r = 0.0;
        if (wx > 0.0) r += wx * reflectanceAt(p.z * scale, p.y * scale, lambda);
        if (wy > 0.0) r += wy * reflectanceAt(p.x * scale, p.z * scale, lambda);
        if (wz > 0.0) r += wz * reflectanceAt(p.x * scale, p.y * scale, lambda);
        return r;
    }

    // Triplanar (box) projection of the LINEAR RGB albedo at a world hit — the RGB
    // twin of reflectanceTriplanar, used by the solid-shaded preview rasterizer (which
    // shades from linear albedo, not per-wavelength reflectance). Same |n|^4 axis blend.
    Vec3 sampleRgbTriplanar(const Vec3& p, const Vec3& n, double scale) const {
        double ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
        double wx = ax * ax * ax * ax, wy = ay * ay * ay * ay, wz = az * az * az * az;
        double s = wx + wy + wz;
        if (s <= 0.0) return sampleRgb(p.x * scale, p.y * scale);
        wx /= s; wy /= s; wz /= s;
        Vec3 c{0, 0, 0};
        if (wx > 0.0) c = c + sampleRgb(p.z * scale, p.y * scale) * wx;
        if (wy > 0.0) c = c + sampleRgb(p.x * scale, p.z * scale) * wy;
        if (wz > 0.0) c = c + sampleRgb(p.x * scale, p.y * scale) * wz;
        return c;
    }

    // ---- loading ------------------------------------------------------------
    bool load(const std::string& path, std::string& err) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { err = "cannot open texture file: " + path; return false; }
        char m0 = 0, m1 = 0; f.get(m0); f.get(m1);
        // Our own PPM/PFM paths (also what the renderer writes); stb handles the rest.
        if (m0 == 'P' && (m1 == '6' || m1 == '3')) { f.seekg(0); return loadPPM(f, m1 == '6', err); }
        if (m0 == 'P' && (m1 == 'F' || m1 == 'f')) { f.seekg(0); return loadPFM(f, err); }
        f.close();
        return loadSTB(path, err);   // PNG / JPG / BMP / TGA / HDR via stb_image
    }

  private:
    // Skip whitespace and '#' comment lines, then read one integer token.
    static bool nextInt(std::istream& s, int& out) {
        int c;
        for (;;) {
            c = s.get();
            if (c == EOF) return false;
            if (c == '#') { while (c != '\n' && c != EOF) c = s.get(); continue; }
            if (std::isspace(c)) continue;
            break;
        }
        std::string tok(1, (char)c);
        while ((c = s.get()) != EOF && !std::isspace(c)) tok.push_back((char)c);
        out = std::atoi(tok.c_str());
        return true;
    }

    void storeLinear(double r, double g, double b) {
        if (encoding == TexEncoding::sRGB)
            rgb.push_back(Vec3{srgbToLinear(r), srgbToLinear(g), srgbToLinear(b)});
        else
            rgb.push_back(Vec3{r, g, b});
    }

    bool loadPPM(std::istream& s, bool binary, std::string& err) {
        s.get(); s.get();   // 'P','6'/'3'
        int maxv = 255;
        if (!nextInt(s, w) || !nextInt(s, h) || !nextInt(s, maxv) || w <= 0 || h <= 0) {
            err = "malformed PPM header"; return false;
        }
        rgb.clear(); rgb.reserve((size_t)w * h);
        double inv = 1.0 / (double)maxv;
        if (binary) {
            // nextInt() already consumed the single whitespace after maxval, so the
            // binary pixel block starts at the current position (no extra get()).
            std::vector<unsigned char> buf((size_t)w * h * 3);
            s.read((char*)buf.data(), (std::streamsize)buf.size());
            if (s.gcount() != (std::streamsize)buf.size()) { err = "short PPM pixel data"; return false; }
            for (size_t i = 0; i < buf.size(); i += 3)
                storeLinear(buf[i] * inv, buf[i + 1] * inv, buf[i + 2] * inv);
        } else {
            for (int i = 0; i < w * h; ++i) {
                int r, g, b;
                if (!nextInt(s, r) || !nextInt(s, g) || !nextInt(s, b)) { err = "short P3 data"; return false; }
                storeLinear(r * inv, g * inv, b * inv);
            }
        }
        return true;
    }

    // PFM: "PF" (colour) or "Pf" (grayscale), then W H, then a scale line whose
    // sign gives endianness (negative = little-endian). Pixels are float, bottom-up,
    // and always linear -> force the encoding to Linear.
    bool loadPFM(std::istream& s, std::string& err) {
        std::string magic; s >> magic;
        bool colour = (magic == "PF");
        s >> w >> h;
        double scale; s >> scale;
        s.get();   // one whitespace after the scale line
        bool little = (scale < 0);
        int comp = colour ? 3 : 1;
        std::vector<float> buf((size_t)w * h * comp);
        s.read((char*)buf.data(), (std::streamsize)(buf.size() * sizeof(float)));
        if (s.gcount() != (std::streamsize)(buf.size() * sizeof(float))) { err = "short PFM data"; return false; }
        if (!little) {   // swap big-endian floats
            for (auto& x : buf) {
                uint32_t u; std::memcpy(&u, &x, 4);
                u = (u >> 24) | ((u >> 8) & 0xFF00) | ((u << 8) & 0xFF0000) | (u << 24);
                std::memcpy(&x, &u, 4);
            }
        }
        encoding = TexEncoding::Linear;
        rgb.assign((size_t)w * h, Vec3{0, 0, 0});
        // PFM rows run bottom-to-top; flip into our top-left-origin storage.
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                size_t src = ((size_t)y * w + x) * comp;
                Vec3 c = colour ? Vec3{buf[src], buf[src + 1], buf[src + 2]}
                                : Vec3{buf[src], buf[src], buf[src]};
                rgb[(size_t)(h - 1 - y) * w + x] = c;
            }
        }
        return true;
    }

    // PNG / JPG / BMP / TGA / HDR via stb_image (top-left origin, so no row flip).
    // Radiance .hdr files are linear float; stbi_loadf returns them directly and we
    // force `encoding` to Linear. LDR formats decode to 8-bit RGB and honour the
    // authored `encoding` (srgb by default → linearized per texel via storeLinear).
    bool loadSTB(const std::string& path, std::string& err) {
        int n = 0;
        if (stbi_is_hdr(path.c_str())) {
            float* data = stbi_loadf(path.c_str(), &w, &h, &n, 3);
            if (!data) { err = "stb_image: " + std::string(stbi_failure_reason() ?
                                 stbi_failure_reason() : "load failed") + " (" + path + ")"; return false; }
            encoding = TexEncoding::Linear;   // .hdr is scene-linear
            rgb.assign((size_t)w * h, Vec3{0, 0, 0});
            for (size_t i = 0; i < (size_t)w * h; ++i)
                rgb[i] = Vec3{data[i * 3], data[i * 3 + 1], data[i * 3 + 2]};
            stbi_image_free(data);
            return true;
        }
        unsigned char* data = stbi_load(path.c_str(), &w, &h, &n, 3);
        if (!data) { err = "stb_image: " + std::string(stbi_failure_reason() ?
                             stbi_failure_reason() : "load failed") + " (" + path + ")"; return false; }
        rgb.clear(); rgb.reserve((size_t)w * h);
        const double inv = 1.0 / 255.0;
        for (size_t i = 0; i < (size_t)w * h; ++i)
            storeLinear(data[i * 3] * inv, data[i * 3 + 1] * inv, data[i * 3 + 2] * inv);
        stbi_image_free(data);
        return true;
    }
};
