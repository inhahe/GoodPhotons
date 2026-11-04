// Temporal denoising of a shot (ROADMAP 11 stage D): an SVGF-style accumulator (Schied et al.
// 2017, "Spatiotemporal Variance-Guided Filtering"), run over a shot's frames in order, each
// frame read as its render and its guide pass (`-write-aovs`).
//
// Per frame:
//   1. The illumination is the render divided by the albedo guide (where there is one), so the
//      filter smooths lighting and leaves texture alone; the result is multiplied back at the end.
//   2. Each pixel's history is the previous frame's accumulated illumination at the place the
//      pixel's surface sat a frame earlier (its backward motion vector), read bilinearly from the
//      taps that ARE that surface: a tap counts only if the previous frame's world position there
//      lies within a share of the depth of where this point sat a frame earlier (the guide pass's
//      prevposition), and their normals agree. Taps that fail -- a disocclusion, the edge of the
//      frame -- start the history afresh. The history is then clamped, channel by channel, to
//      this frame's illumination about the pixel (the mean of its 3x3 neighbours on the same
//      surface, +- clampGamma standard deviations): a still surface whose light changed -- a
//      moving shadow crossing it -- keeps no ghost of the light it had (the paper has no such
//      guard). What the clamp took off shortens the history and its moments alike (anti-lag).
//   3. The illumination and its luminance's first two moments are blended into the history, the
//      new frame's share 1/n while the history is young (n frames long) and at least `alpha`
//      after, so the history converges and still follows lighting that changes. The variance --
//      of the accumulated mean, what the filter smooths -- comes from the moments over the
//      history's length once four frames have accumulated, and from a 7x7 neighbourhood before.
//   4. An a-trous wavelet filter (5x5 B3-spline, steps 1, 2, 4, ...) smooths the accumulated
//      illumination, its weights stopping at depth, normal and luminance edges -- the last scaled
//      by the local standard deviation, so a pixel whose history is converged keeps its detail.
//      The first pass's output is what the next frame's history reads (the feedback that lets
//      the history itself be denoised), as in the paper.
//
// CPU, header-only. Images are W x H in the film's row order (row 0 at the bottom); motion
// vectors in pixels, x right and y up, as dguide writes them.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include "geometry.h"
#include "parallel.h"

namespace temporal {

// One frame as the denoiser reads it.
struct Frame {
    int W = 0, H = 0;
    std::vector<Vec3> color;        // the render, scene-linear
    std::vector<Vec3> albedo;       // dguide's guides (-write-aovs)
    std::vector<Vec3> normal;
    std::vector<Vec3> position;     // world position at the frame's instant; 0 with depth 0
    std::vector<Vec3> prevPosition; // where that point sat a frame earlier
    std::vector<Vec3> motionBack;   // pixels to where it sat a frame earlier
    std::vector<float> depth;       // along the camera's axis; 0 where nothing was met
};

struct SvgfOptions {
    // Tuned on tools/temporal_compare.py's shot (a closed room, a ball rolling through its own
    // contact shadow, 4-8 spp): three passes where the paper's real-time 1 spp takes five, the
    // clamp and anti-lag where it had none (see the header).
    double alpha = 0.2;          // a converged history's share of each new frame, colour
    double alphaMoments = 0.2;   // ... and luminance moments
    int    iterations = 3;       // a-trous passes
    double sigmaL = 4.0;         // luminance edge-stopping, in local standard deviations
    double sigmaN = 128.0;       // normal edge-stopping exponent
    double sigmaZ = 1.0;         // depth edge-stopping, in multiples of the local depth gradient
    double reprojTol = 0.02;     // a history tap is the same surface within this share of depth
    double maxHistory = 64.0;    // frames a history may count (its share never below 1/this)
    double clampGamma = 0.75;    // history clamped to the frame's local mean +- this many deviations
    int    clampRadius = 1;      // ... over (2r+1)^2 pixels of the same surface; 0: no clamp
    double antiLag = 1.0;        // a clamped history's length and moments cut by this x its move
};

inline double luminance(const Vec3& c) { return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z; }

class Svgf {
public:
    explicit Svgf(SvgfOptions o = SvgfOptions{}) : opt_(o) {}

    // Denoise `f`, the shot's next frame, reading and then updating the history.
    std::vector<Vec3> process(const Frame& f) {
        const int W = f.W, H = f.H;
        const size_t N = (size_t)W * (size_t)H;
        if (W != W_ || H != H_) reset(W, H);
        // 1. Demodulate.
        std::vector<Vec3> alb(N), illum(N);
        for (size_t i = 0; i < N; ++i) {
            const Vec3 a = f.albedo.empty() ? Vec3{1, 1, 1} : f.albedo[i];
            alb[i] = Vec3{a.x > 1e-3 ? a.x : 1.0, a.y > 1e-3 ? a.y : 1.0, a.z > 1e-3 ? a.z : 1.0};
            illum[i] = Vec3{f.color[i].x / alb[i].x, f.color[i].y / alb[i].y, f.color[i].z / alb[i].z};
        }
        // The box each pixel's history is clamped to: the mean of this frame's illumination about
        // it, per channel, +- clampGamma standard deviations, over the neighbours on its surface.
        std::vector<Vec3> boxLo, boxHi;
        if (opt_.clampRadius > 0 && havePrev_) {
            boxLo.resize(N); boxHi.resize(N);
            const int r = opt_.clampRadius;
            ft::parallelFor((size_t)H, 1, [&](size_t row) {
                const int py = (int)row;
                for (int px = 0; px < W; ++px) {
                    const size_t i = (size_t)py * W + px;
                    Vec3 s1{0, 0, 0}, s2{0, 0, 0};
                    double ws = 0.0;
                    for (int dy = -r; dy <= r; ++dy)
                        for (int dx = -r; dx <= r; ++dx) {
                            const int x = px + dx, y = py + dy;
                            if (x < 0 || y < 0 || x >= W || y >= H) continue;
                            const size_t j = (size_t)y * W + x;
                            if (!sameSurface(f, i, j)) continue;
                            const Vec3& c = illum[j];
                            s1 = s1 + c;
                            s2 = s2 + Vec3{c.x * c.x, c.y * c.y, c.z * c.z};
                            ws += 1.0;
                        }
                    const Vec3 mu = s1 * (1.0 / ws), m2 = s2 * (1.0 / ws);
                    const Vec3 sd{std::sqrt(std::max(0.0, m2.x - mu.x * mu.x)),
                                  std::sqrt(std::max(0.0, m2.y - mu.y * mu.y)),
                                  std::sqrt(std::max(0.0, m2.z - mu.z * mu.z))};
                    boxLo[i] = mu - sd * opt_.clampGamma;
                    boxHi[i] = mu + sd * opt_.clampGamma;
                }
            });
        }
        // 2-3. Reproject and accumulate.
        std::vector<Vec3> acc(N);
        std::vector<double> m1(N), m2(N), len(N), var(N);
        ft::parallelFor((size_t)H, 1, [&](size_t row) {
            const int py = (int)row;
            for (int px = 0; px < W; ++px) {
                const size_t i = (size_t)py * W + px;
                const double l = luminance(illum[i]);
                Vec3 hc{0, 0, 0};
                double hm1 = 0.0, hm2 = 0.0, hl = 0.0, wsum = 0.0;
                if (havePrev_ && f.depth[i] > 0.0f) {
                    const double qx = px + 0.5 + f.motionBack[i].x - 0.5;
                    const double qy = py + 0.5 + f.motionBack[i].y - 0.5;
                    const int x0 = (int)std::floor(qx), y0 = (int)std::floor(qy);
                    const double fx = qx - x0, fy = qy - y0;
                    const Vec3& pp = f.prevPosition.empty() ? f.position[i] : f.prevPosition[i];
                    const double tol = opt_.reprojTol * (double)f.depth[i];
                    for (int t = 0; t < 4; ++t) {
                        const int x = x0 + (t & 1), y = y0 + (t >> 1);
                        if (x < 0 || y < 0 || x >= W || y >= H) continue;
                        const size_t j = (size_t)y * W + x;
                        if (!(prevDepth_[j] > 0.0f)) continue;
                        if (length(prevPosition_[j] - pp) > tol) continue;
                        if (dot(prevNormal_[j], f.normal[i]) < 0.9 * length(prevNormal_[j]) * length(f.normal[i]))
                            continue;
                        const double w = ((t & 1) ? fx : 1.0 - fx) * ((t >> 1) ? fy : 1.0 - fy);
                        hc = hc + histIllum_[j] * w;
                        hm1 += histM1_[j] * w; hm2 += histM2_[j] * w; hl += histLen_[j] * w;
                        wsum += w;
                    }
                }
                if (wsum > 1e-3) {
                    hc = hc * (1.0 / wsum); hm1 /= wsum; hm2 /= wsum; hl /= wsum;
                    if (!boxLo.empty()) {
                        const Vec3 raw = hc;
                        hc = Vec3{std::clamp(hc.x, boxLo[i].x, boxHi[i].x), std::clamp(hc.y, boxLo[i].y, boxHi[i].y),
                                  std::clamp(hc.z, boxLo[i].z, boxHi[i].z)};
                        // Anti-lag: the share of its luminance the clamp took off is how stale the
                        // history is. Its length shrinks by it, so the new frames weigh more; its
                        // moments move with the clamped mean and their spread shrinks alike, or the
                        // change itself would read as noise and widen the filter over it.
                        if (opt_.antiLag > 0.0) {
                            const double lr = luminance(raw), lc = luminance(hc);
                            const double moved = std::fabs(lr - lc) /
                                                 std::max(1e-30, std::max(std::fabs(lr), std::fabs(lc)));
                            const double keep = std::max(0.0, 1.0 - opt_.antiLag * moved);
                            const double v = std::max(0.0, hm2 - hm1 * hm1) * keep;
                            hl *= keep;
                            hm1 += lc - lr;
                            hm2 = hm1 * hm1 + v;
                        }
                    }
                    const double n = std::min(hl + 1.0, opt_.maxHistory);
                    const double a = std::max(opt_.alpha, 1.0 / n);
                    const double am = std::max(opt_.alphaMoments, 1.0 / n);
                    acc[i] = hc * (1.0 - a) + illum[i] * a;
                    m1[i] = hm1 * (1.0 - am) + l * am;
                    m2[i] = hm2 * (1.0 - am) + l * l * am;
                    len[i] = n;
                } else {
                    acc[i] = illum[i]; m1[i] = l; m2[i] = l * l; len[i] = 1.0;
                }
            }
        });
        // The variance: temporal from four frames on, else spatial over a 7x7 neighbourhood.
        ft::parallelFor((size_t)H, 1, [&](size_t row) {
            const int py = (int)row;
            for (int px = 0; px < W; ++px) {
                const size_t i = (size_t)py * W + px;
                // Of the accumulated mean, which is what the filter smooths: a sample's variance over
                // the frames the history holds -- at most (2 - alpha) / alpha of them, all an
                // exponential average with a floor of alpha on each new frame is worth.
                if (len[i] >= 4.0) {
                    const double frames = std::min(len[i], (2.0 - opt_.alpha) / opt_.alpha);
                    var[i] = std::max(0.0, m2[i] - m1[i] * m1[i]) / frames;
                    continue;
                }
                double s1 = 0.0, s2 = 0.0, ws = 0.0;
                for (int dy = -3; dy <= 3; ++dy)
                    for (int dx = -3; dx <= 3; ++dx) {
                        const int x = px + dx, y = py + dy;
                        if (x < 0 || y < 0 || x >= W || y >= H) continue;
                        const size_t j = (size_t)y * W + x;
                        if ((f.depth[j] > 0.0f) != (f.depth[i] > 0.0f)) continue;
                        s1 += m1[j]; s2 += m2[j]; ws += 1.0;
                    }
                var[i] = ws > 0.0 ? std::max(0.0, s2 / ws - (s1 / ws) * (s1 / ws)) / len[i] : 0.0;
            }
        });
        // 4. The a-trous passes.
        std::vector<Vec3> cur = acc, nxt(N);
        std::vector<double> v = var, vn(N);
        for (int it = 0; it < opt_.iterations; ++it) {
            atrous(f, 1 << it, cur, v, nxt, vn);
            std::swap(cur, nxt);
            std::swap(v, vn);
            if (it == 0) histIllum_ = cur;   // the history the next frame reads
        }
        if (opt_.iterations == 0) histIllum_ = cur;
        histM1_ = m1; histM2_ = m2; histLen_ = len;
        prevPosition_ = f.position; prevNormal_ = f.normal; prevDepth_ = f.depth;
        havePrev_ = true;
        // Remodulate.
        std::vector<Vec3> out(N);
        for (size_t i = 0; i < N; ++i)
            out[i] = Vec3{cur[i].x * alb[i].x, cur[i].y * alb[i].y, cur[i].z * alb[i].z};
        return out;
    }

private:
    // Whether pixel j of `f` shows the surface pixel i does: both meet something (or neither), at
    // a depth within a tenth of i's, and with normals that agree.
    static bool sameSurface(const Frame& f, size_t i, size_t j) {
        if (i == j) return true;
        if ((f.depth[i] > 0.0f) != (f.depth[j] > 0.0f)) return false;
        if (!(f.depth[i] > 0.0f)) return true;
        if (std::fabs(f.depth[j] - f.depth[i]) > 0.1f * f.depth[i]) return false;
        return dot(f.normal[i], f.normal[j]) >= 0.9 * length(f.normal[i]) * length(f.normal[j]);
    }

    void reset(int W, int H) {
        W_ = W; H_ = H; havePrev_ = false;
        histIllum_.clear(); histM1_.clear(); histM2_.clear(); histLen_.clear();
        prevPosition_.clear(); prevNormal_.clear(); prevDepth_.clear();
    }

    // One a-trous pass at `step` pixels: `in`/`var` filtered into `out`/`varOut`.
    void atrous(const Frame& f, int step, const std::vector<Vec3>& in, const std::vector<double>& var,
                std::vector<Vec3>& out, std::vector<double>& varOut) const {
        const int W = f.W, H = f.H;
        static const double kH[5] = {1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16};
        ft::parallelFor((size_t)H, 1, [&](size_t row) {
            const int py = (int)row;
            for (int px = 0; px < W; ++px) {
                const size_t i = (size_t)py * W + px;
                const double zi = f.depth[i];
                // The depth's change per pixel here, for the depth weight's scale.
                const auto zAt = [&](int x, int y) {
                    x = std::clamp(x, 0, W - 1); y = std::clamp(y, 0, H - 1);
                    return (double)f.depth[(size_t)y * W + x];
                };
                const double gz = std::max(std::fabs(zAt(px + 1, py) - zAt(px - 1, py)),
                                           std::fabs(zAt(px, py + 1) - zAt(px, py - 1))) * 0.5;
                // The local standard deviation, blurred 3x3 (the paper's g3x3), for the luminance
                // weight.
                double vs = 0.0, vw = 0.0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int x = px + dx, y = py + dy;
                        if (x < 0 || y < 0 || x >= W || y >= H) continue;
                        const double k = (dx == 0 ? 0.5 : 0.25) * (dy == 0 ? 0.5 : 0.25);
                        vs += k * var[(size_t)y * W + x]; vw += k;
                    }
                const double sd = std::sqrt(std::max(0.0, vs / vw));
                const double li = luminance(in[i]);
                const Vec3& ni = f.normal[i];
                Vec3 cs{0, 0, 0};
                double ws = 0.0, vsum = 0.0;
                for (int dy = -2; dy <= 2; ++dy)
                    for (int dx = -2; dx <= 2; ++dx) {
                        const int x = px + dx * step, y = py + dy * step;
                        if (x < 0 || y < 0 || x >= W || y >= H) continue;
                        const size_t j = (size_t)y * W + x;
                        const double h = kH[dx + 2] * kH[dy + 2];
                        double w = 1.0;
                        if (dx != 0 || dy != 0) {
                            if ((f.depth[j] > 0.0f) != (zi > 0.0)) continue;
                            const double dist = std::sqrt((double)(dx * dx + dy * dy)) * step;
                            const double wz = std::exp(-std::fabs(f.depth[j] - zi) /
                                                       (opt_.sigmaZ * gz * dist + 1e-6));
                            const double nn = std::max(0.0, dot(f.normal[j], ni)) /
                                              std::max(1e-12, length(f.normal[j]) * length(ni));
                            const double wn = std::pow(nn, opt_.sigmaN);
                            const double wl = std::exp(-std::fabs(luminance(in[j]) - li) /
                                                       (opt_.sigmaL * sd + 1e-10));
                            w = wz * wn * wl;
                        }
                        cs = cs + in[j] * (h * w);
                        ws += h * w;
                        vsum += h * h * w * w * var[j];
                    }
                out[i] = ws > 0.0 ? cs * (1.0 / ws) : in[i];
                varOut[i] = ws > 0.0 ? vsum / (ws * ws) : var[i];
            }
        });
    }

    SvgfOptions opt_;
    int W_ = 0, H_ = 0;
    bool havePrev_ = false;
    std::vector<Vec3> histIllum_;
    std::vector<double> histM1_, histM2_, histLen_;
    std::vector<Vec3> prevPosition_, prevNormal_;
    std::vector<float> prevDepth_;
};

}  // namespace temporal
