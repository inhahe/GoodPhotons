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
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include "linalg.h"
#include "color.h"
#include "upsample.h"
#include "spectrum.h"
#include "stochtile.h"

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
    // One linear-sRGB colour per palette entry, so a palette map can answer sampleRgb()
    // without a spectral evaluation. Spectra are exact but useless to anything that wants
    // a colour (the raster preview, thumbnails), and the old behaviour there was to read
    // the raw INDEX out of the red channel and shade with it — index 3 of 12 previewing as
    // near-black. Built once by buildReflCoeff(), which every bound texture already calls.
    std::vector<Vec3> paletteRgb;
    TexEncoding encoding = TexEncoding::sRGB;
    TexFilter   filter   = TexFilter::Bilinear;
    TexWrap     wrap     = TexWrap::Repeat;

    // Histogram-preserving stochastic tiling (O7, `tiling stochastic`). See stochtile.h
    // for the operator. Enabling it adds TWO Gaussianized plane sets, one per space the
    // renderer actually asks a texture to be filtered in:
    //
    //   gaussRgb   (3ch) — linear RGB. Used by sampleRgb, normal maps, the raster
    //                      preview AND the spectral path, which blends here and then
    //                      converts the blended colour to a spectrum through the shared
    //                      upsample::coeffLut(). Blending the Jakob-Hanika coefficients
    //                      instead was the first implementation and is wrong — see the
    //                      measurements in stochtile.h. Doing it this way also means the
    //                      preview and the render run the identical operator on the
    //                      identical planes, so they agree by construction.
    //   gaussGray  (1ch) — the scalar mean, for roughness / weight maps and `tex:`.
    //                      Not the mean of the three blended colour channels: three
    //                      independent inverse transforms averaged is not the inverse
    //                      transform of the mean, and the device backends rank the gray
    //                      plane separately, so deriving it would put CPU and GPU on
    //                      different values.
    //
    // Each plane is float: the values are quantiles, so the extra mantissa of a double
    // buys nothing, and this already costs 16 bytes/texel on top of the source.
    StochTile          stoch;
    std::vector<float> gaussRgb,  lutRgb;      // 3*w*h  /  3*STOCH_LUT_N
    std::vector<float> gaussGray, lutGray;     // 1*w*h  /  1*STOCH_LUT_N
    bool stochastic() const { return stoch.on != 0 && !gaussRgb.empty(); }
    int  filterCode() const { return filter == TexFilter::Nearest ? 0 : 1; }

    bool valid() const { return w > 0 && h > 0 && (int)rgb.size() == w * h; }

    // Precompute the Jakob-Hanika reflectance coefficients per texel (once). Called
    // when a texture is bound to a reflectance parameter so per-hit sampling is a
    // cheap coefficient bilerp + sigmoid evaluation rather than a Gauss-Newton fit.
    //
    // Deliberately upsample::fitMany rather than a per-texel loop over upsample::fit:
    // the fit is a few microseconds, which at megapixel resolutions is SECONDS of dead
    // time during scene load with nothing on screen. fitMany dedups bit-equal texels
    // (an 8-bit image has far fewer distinct colours than texels) and threads what is
    // left; the coefficients it writes are bit-identical to the serial loop's.
    //
    // Returns false only if a clean stop was requested mid-fit (`ftrace -stop`, Ctrl-C).
    // The coefficient table is then partial, so it is cleared and the caller must fail
    // the scene load — sampling a half-fitted texture would shade garbage.
    [[nodiscard]] bool buildReflCoeff() {
        if (hasPalette()) { buildPaletteRgb(); return true; }
        if (!valid() || coeff.size() == (size_t)w * h) return true;
        coeff.resize((size_t)w * h);
        if (!upsample::fitMany(rgb.data(), rgb.size(), coeff.data())) {
            coeff.clear(); coeff.shrink_to_fit();
            return false;
        }
        return true;
    }

    // Rank-transform one plane set into interleaved Gaussianized floats plus the
    // per-channel inverse LUT (O7). `get(i, c)` reads channel c of texel i.
    //
    // The transform is the empirical rank, not a fitted Gaussian: texel of rank r among n
    // gets the quantile at (r + 1/2)/n, which maps ANY histogram — bimodal, clipped,
    // spiky, a two-colour tile pattern — onto N(1/2, 1/6) exactly. The LUT inverts it by
    // walking the same sorted array at uniformly spaced values of G, so T^-1(T(x)) == x to
    // within the LUT's linear interpolation. Ties are broken by texel index so the result
    // is deterministic regardless of the sort implementation.
    template <class Get>
    static void gaussianizePlanes(size_t n, int nch, Get get,
                                  std::vector<float>& gauss, std::vector<float>& lut) {
        gauss.assign(n * (size_t)nch, 0.0f);
        lut.assign((size_t)nch * STOCH_LUT_N, 0.0f);
        if (n == 0) return;
        std::vector<size_t> ord(n);
        std::vector<double> val(n);
        for (int c = 0; c < nch; ++c) {
            for (size_t i = 0; i < n; ++i) { ord[i] = i; val[i] = get(i, c); }
            std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
                return val[a] != val[b] ? (val[a] < val[b]) : (a < b);
            });
            for (size_t r = 0; r < n; ++r)
                gauss[ord[r] * (size_t)nch + c] = (float)stochForward(r, n);
            float* L = lut.data() + (size_t)c * STOCH_LUT_N;
            for (int j = 0; j < STOCH_LUT_N; ++j) {
                const double g = STOCH_G_LO + (STOCH_G_HI - STOCH_G_LO) *
                                              ((double)j / (double)(STOCH_LUT_N - 1));
                const double p = stochNormalCdf((g - STOCH_G_MEAN) / STOCH_G_SIGMA);
                const double t = p * (double)n - 0.5;   // the continuous rank G asks for
                if (t <= 0.0)                    L[j] = (float)val[ord[0]];
                else if (t >= (double)(n - 1))   L[j] = (float)val[ord[n - 1]];
                else {
                    const double f = std::floor(t);
                    const size_t i0 = (size_t)f;
                    const double fr = t - f;
                    L[j] = (float)(val[ord[i0]] * (1.0 - fr) + val[ord[i0 + 1]] * fr);
                }
            }
        }
    }

    // Build the Gaussianized planes (O7). Both are transforms of the linear-RGB texels,
    // so unlike buildReflCoeff this has no ordering dependency on anything else.
    void buildStochastic() {
        if (!stoch.on || !valid() || hasPalette()) return;
        const size_t n = (size_t)w * h;
        gaussianizePlanes(n, 3, [&](size_t i, int c) {
            return c == 0 ? rgb[i].x : (c == 1 ? rgb[i].y : rgb[i].z);
        }, gaussRgb, lutRgb);
        gaussianizePlanes(n, 1, [&](size_t i, int) {
            return (rgb[i].x + rgb[i].y + rgb[i].z) * (1.0 / 3.0);
        }, gaussGray, lutGray);
        (void)upsample::coeffLut();   // force the shared RGB->coefficient table now, so
                                      // the cost lands in load rather than first hit
    }

    // Integrate each palette spectrum against the CIE curves under an equal-energy
    // illuminant -> the colour that reflectance would appear as. At most 256 entries, so
    // this is trivially cheap and done eagerly rather than lazily (sampleRgb is const and
    // runs on many threads).
    void buildPaletteRgb() {
        if (palette.empty() || paletteRgb.size() == palette.size()) return;
        paletteRgb.resize(palette.size());
        for (size_t i = 0; i < palette.size(); ++i) {
            Vec3 xyz{0, 0, 0};
            double wsum = 0.0;
            if (palette[i]) {
                for (double lam = LAMBDA_MIN; lam <= LAMBDA_MAX; lam += 5.0) {
                    double v = palette[i](lam);
                    xyz = xyz + Vec3(cieX(lam), cieY(lam), cieZ(lam)) * v;
                    wsum += cieY(lam);
                }
            }
            Vec3 c = (wsum > 0.0) ? xyzToLinearSrgb(xyz / wsum) : Vec3{0, 0, 0};
            paletteRgb[i] = Vec3{std::max(0.0, c.x), std::max(0.0, c.y), std::max(0.0, c.z)};
        }
    }

    // Nearest-index palette lookup returning the entry's preview COLOUR. Categorical, so
    // it never bilerps — the same snap-and-clamp paletteReflectanceAt does.
    Vec3 paletteRgbAt(double u, double v) const {
        int x = wrapIndex((int)std::floor(u * w), w);
        int y = wrapIndex((int)std::floor((1.0 - v) * h), h);
        int idx = (int)std::lround(rgb[(size_t)y * w + x].x * 255.0);
        if (idx < 0) idx = 0;
        if (idx >= (int)paletteRgb.size()) idx = (int)paletteRgb.size() - 1;
        return paletteRgb.empty() ? Vec3{0.5, 0.5, 0.5} : paletteRgb[idx];
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
        // An INDEX map's texels are palette indices, not colours: resolve through the
        // palette rather than returning the index as if it were a shade of grey.
        if (hasPalette()) return paletteRgbAt(u, v);
        if (stochastic()) {   // O7: three offset taps blended in RGB-Gaussian space
            double o[3];
            stochSample(stoch, gaussRgb.data(), lutRgb.data(), 3, w, h, filterCode(), u, v, o);
            return Vec3{o[0], o[1], o[2]};
        }
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
        // O7: the scalar plane gets its OWN rank transform rather than averaging the three
        // stochastic colour channels — a roughness map wants its own histogram preserved,
        // and the device's dTexScalarAt reads a gray plane too, so this is also what keeps
        // the two backends identical. (For a grey source the two agree anyway.)
        if (stoch.on && !gaussGray.empty()) {
            double o[1];
            stochSample(stoch, gaussGray.data(), lutGray.data(), 1, w, h, filterCode(), u, v, o);
            return o[0];
        }
        Vec3 c = sampleRgb(u, v);
        return (c.x + c.y + c.z) * (1.0 / 3.0);
    }

    // Tangent-space normal at (u,v) for normal mapping (C6). The image stores an
    // encoded normal: RGB in [0,1] maps to a vector in [-1,1]^3 (the usual
    // n = 2*c - 1). Normal maps carry raw vector data, so the texture MUST be loaded
    // `encoding linear` (no sRGB de-gamma) — then `rgb` already holds the [0,1]
    // components and we just remap+normalize. Returned in tangent space (x=U, y=V,
    // z=surface normal); the caller rotates it into world via the TBN frame. Mirrored
    // on the GPU by dTexNormalAt.
    Vec3 sampleNormalTS(double u, double v) const {
        Vec3 c = sampleRgb(u, v);
        Vec3 n{2.0 * c.x - 1.0, 2.0 * c.y - 1.0, 2.0 * c.z - 1.0};
        double l = std::sqrt(dot(n, n));
        return (l > 1e-12) ? n * (1.0 / l) : Vec3{0, 0, 1};
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
        if (stochastic()) {
            // O7: blend the three crops in linear RGB — the same planes, lattice and
            // weights the preview uses — then convert that colour to a spectrum. See
            // stochtile.h for why this is not done in coefficient space.
            double c[3];
            stochSample(stoch, gaussRgb.data(), lutRgb.data(), 3, w, h, filterCode(),
                        u, v, c);
            double cs[3];
            stochJhCoeff(upsample::coeffLut().data(), c[0], c[1], c[2], cs);
            return stochReflAt(cs, lambda);
        }
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
