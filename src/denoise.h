// Post-render denoiser for Monte-Carlo speckle. Operates on the LINEAR sRGB image,
// after the film's XYZ has been converted but before exposure and the sRGB transfer
// curve, which is the only place the maths is meaningful: noise is symmetric in linear
// radiance and badly skewed once a gamma curve has squashed the highlights.
//
// ---------------------------------------------------------------------------------
// WHY THIS FILTER AND NOT AN AI DENOISER
//
// The noise this renderer actually produces in its hardest scenes is not generic MC
// grain, it is CHROMATIC SPECKLE, and it has a specific cause. A spectral path carries
// ONE wavelength; the hero-wavelength bundle (`-heroc`) that normally amortises a path
// over N wavelengths is switched off by participating media, and a dispersive
// refraction terminates the bundle's secondaries anyway. So every sample landing in a
// dispersive caustic deposits a fully SATURATED colour, and the pixel only becomes
// white-ish once enough samples of enough different wavelengths have averaged together.
//
// That means the variance is concentrated in CHROMA, not in luminance: the luminance
// estimate converges at the usual 1/spp while the hue is still bouncing between
// saturated red, green and blue. Which is very good news, because human spatial acuity
// in chroma is roughly a third of that in luma — the reason every video codec on earth
// subsamples chroma and no one notices. Blurring chroma hard while leaving luma almost
// untouched removes nearly all of the visible artifact and costs nearly nothing that
// the eye can see.
//
// So the filter is an edge-aware a-trous wavelet (the standard SVGF cascade: a 5x5
// B3-spline kernel applied at strides 1, 2, 4, 8, ... which buys a wide support for a
// small number of taps) applied TO CHROMA ONLY. Luma comes out bit-identical. That is a
// stronger claim than "filtered lightly", and it is deliberate: it means turning the
// denoiser on cannot cost a single edge, wire, caustic rim or speck of geometric detail,
// so there is never a reason to weigh it against sharpness. See Params for the
// measurements that ruled the luma channel out, and for why it is still available.
//
// The filter is guided by LUMA. Chroma must not be its own edge-stop: the whole point is
// that the chroma signal is the unreliable one, so letting it gate its own filter would
// preserve exactly the speckle we are trying to remove. Luma is the right guide anyway,
// because a real material boundary is a luma edge.
//
// The cost of this choice is honest and worth stating: a genuinely fine-scale colour
// signal — the rainbow fringing at the rim of a dispersive caustic, say — is a chroma
// detail, and a wide chroma blur will soften it. That is the one thing this filter can
// damage, which is why `chroma` and `levels` are exposed; lower them if the fringes
// matter more than the speckle.
//
// A trained denoiser (OIDN, OptiX) would do better on the luma channel, but it wants
// albedo and normal auxiliary buffers that this film does not carry, and it is a large
// binary dependency. This is ~200 lines, no dependency, and it targets the dominant
// artifact directly.
// ---------------------------------------------------------------------------------
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <thread>
#include "linalg.h"

namespace denoise {

// Filter parameters. `luma`/`chroma` are edge-stopping *tolerances*, each an independent
// multiple of the local luma sigma: larger = more smoothing, 0 disables that channel
// group entirely.
//
// LUMA DEFAULTS TO OFF, which is not the obvious choice and is worth justifying, because
// it was measured rather than assumed. Against an 8000 spp reference of the same frame,
// filtering luma at the strength that visibly cleans the image made the result WORSE than
// doing nothing: luma RMSE 17.2 -> 24.2 and PSNR -2.4 dB, because at these tolerances the
// filter cannot tell a real wire, edge or caustic rim from a noise spike and eats all of
// them. That is the honest consequence of the premise in the header comment: luma is the
// channel that is ALREADY converging at 1/spp, so there is little there to win and a lot
// of real detail to lose. Chroma alone leaves luma RMSE at exactly 100% of the baseline
// by construction and takes the chroma error down, which is a strict improvement.
//
// So `-denoise` is chroma-only, and luma smoothing is available but must be asked for
// (`-denoise-luma <t>`) with the understanding that it trades detail for smoothness.
//
// `levels` and `chroma` were picked the same way, by sweeping them against that 8000 spp
// reference rather than by eye (PSNR gain over the unfiltered frame, higher is better):
//
//     levels   2      3      4      5      7          chroma  0.5    1.0    2.0    4.0
//     dPSNR  +1.97  +1.83  +1.51  +1.00  -0.26        dPSNR  +1.93  +1.93  +1.83  +1.75
//
// The optimum is a PLATEAU at 2-3 levels, not the wide 5-level cascade SVGF uses, and by
// 7 levels the filter is a net loss. That is the honest consequence of chroma-only
// filtering: with no luma term to hold the edges, a very wide chroma support starts
// bleeding colour across real material boundaries -- gold into white, caustic into floor
// -- and that costs more than the speckle it removes. 3 levels (a 16-pixel support) sits
// at the chroma-error minimum and within 0.15 dB of the best PSNR. `chroma` is flat
// across a factor of 8, so it is a genuinely insensitive knob; 2.0 is mid-plateau.
struct Params {
    double luma      = 0.0;    // luma tolerance in local sigma (0 = leave luma alone)
    double chroma    = 2.0;    // chroma tolerance in local sigma
    int    levels    = 3;      // a-trous levels; support is 2^levels taps wide
    double fireflies = 0.0;    // >0: clamp isolated outliers to this multiple of the
                               //     neighbourhood's 2nd-brightest luma (0 = off)
    // fireflies counts: `-denoise-chroma 0 -fireflies 3` is a legitimate ask (clamp the
    // outliers, filter nothing), and without it here that combination would silently no-op.
    bool valid() const { return fireflies > 0.0 || (levels > 0 && (luma > 0.0 || chroma > 0.0)); }
};

// Rec.709 luma of a LINEAR sRGB triple. Matches the primaries xyzToLinearSrgb emits,
// so this is the true luminance of the pixel rather than a rough average.
inline double lumaOf(const Vec3& c) { return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z; }

// Reversible luma/chroma split. Y is the real luminance above; the two chroma terms are
// simple opponent differences (a YCoCg-style basis expressed against Rec.709 luma), and
// the inverse is exact because we keep Y and reconstruct the third channel from it.
//
// Storing chroma as a RATIO to luma rather than as a difference matters: MC noise in an
// image is multiplicative (a pixel twice as bright has roughly twice the absolute
// error), so a difference-based chroma would carry the luma's dynamic range into the
// chroma channel and make one tolerance impossible to set for both a dim floor and a
// blown caustic. Dividing by luma makes the chroma channel scale-free, so a single
// tolerance works across ~4 decades of brightness.
//
// The ratio is how chroma is STORED and reconstructed. It is emphatically not how it is
// AVERAGED — the filter multiplies by Y again and divides by the filtered Y, because the
// mean of a ratio whose denominator is the noisy quantity is not the quantity we want.
// See the luma-weighted gather in apply(); getting this wrong turned per-pixel speckle
// into coherent colour blobs.
struct YCC { double y, a, b; };

inline YCC toYcc(const Vec3& c) {
    double y = lumaOf(c);
    if (!(y > 1e-12)) return {y <= 0.0 ? 0.0 : y, 0.0, 0.0};
    return {y, (c.x - c.y) / y, (c.z - c.y) / y};
}

inline Vec3 fromYcc(const YCC& p) {
    if (!(p.y > 1e-12)) return {std::max(0.0, p.y), std::max(0.0, p.y), std::max(0.0, p.y)};
    // Solve  0.2126R + 0.7152G + 0.0722B = y   with  R-G = a*y,  B-G = b*y.
    // => G(0.2126+0.7152+0.0722) + y(0.2126a + 0.0722b) = y  => G = y(1 - 0.2126a - 0.0722b).
    double g = p.y * (1.0 - 0.2126 * p.a - 0.0722 * p.b);
    double r = g + p.a * p.y;
    double bl = g + p.b * p.y;

    // A filtered chroma can land outside the positive octant -- averaging two hues gives
    // a valid hue, but averaging a hue with a ratio derived from a nearly black pixel can
    // not. Clamping the offending channel at 0 is the obvious repair and it is wrong
    // here: clamping ADDS light, so it breaks the one property the rest of this file goes
    // to some trouble to guarantee. Measured, it moved the frame's mean luma by 1.3% and
    // its luma RMSE by ~5% in a filter that is supposed to leave luma bit-identical.
    //
    // Instead, desaturate toward the achromatic point along the segment from (y,y,y) to
    // the requested colour, stopping at the gamut boundary. Every channel is affine in
    // that parameter and luma equals y at BOTH ends, so luma equals y everywhere along it
    // -- the projection is exactly luminance preserving, and it degrades saturation
    // (which was the untrustworthy part) rather than brightness.
    double t = 1.0;
    const double ch[3] = {r, g, bl};
    for (int i = 0; i < 3; ++i)
        if (ch[i] < 0.0) t = std::min(t, p.y / (p.y - ch[i]));
    if (t < 1.0) {
        r  = p.y + t * (r - p.y);
        g  = p.y + t * (g - p.y);
        bl = p.y + t * (bl - p.y);
    }
    return {std::max(0.0, r), std::max(0.0, g), std::max(0.0, bl)};
}

// The 5-tap B3-spline row whose outer product is the 5x5 a-trous kernel.
static const double kB3[5] = {1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16};

// Per-pixel local luma scale, used to turn the relative tolerances into absolute ones.
// A single global tolerance cannot work on an image that spans a black sky and a clipped
// caustic; this makes the edge-stop adapt to how bright the neighbourhood is.
inline std::vector<double> localScale(const std::vector<double>& y, int W, int H) {
    std::vector<double> s((size_t)W * H);
    for (int j = 0; j < H; ++j) for (int i = 0; i < W; ++i) {
        double mn = 1e30, mx = -1e30, sum = 0.0; int n = 0;
        for (int dj = -1; dj <= 1; ++dj) for (int di = -1; di <= 1; ++di) {
            int x = i + di, jj = j + dj;
            if (x < 0 || x >= W || jj < 0 || jj >= H) continue;
            double v = y[(size_t)jj * W + x];
            mn = std::min(mn, v); mx = std::max(mx, v); sum += v; ++n;
        }
        double mean = n ? sum / n : 0.0;
        // Half the local peak-to-peak, floored by a small fraction of the mean so a
        // perfectly flat but noisy-in-chroma region still gets a usable tolerance.
        s[(size_t)j * W + i] = std::max(0.5 * (mx - mn), 0.05 * mean) + 1e-9;
    }
    return s;
}

// Suppress isolated fireflies: a pixel far brighter than every one of its 8 neighbours
// is a single lucky path, not a feature, because a real highlight is at least a couple
// of pixels wide after the camera filter. Clamping to a multiple of the SECOND
// brightest neighbour (not the brightest) keeps the filter from eating the leading edge
// of a genuine thin highlight, where exactly one neighbour is legitimately bright.
inline void clampFireflies(std::vector<Vec3>& img, int W, int H, double k) {
    if (!(k > 0.0)) return;
    std::vector<Vec3> src = img;
    for (int j = 0; j < H; ++j) for (int i = 0; i < W; ++i) {
        double n0 = 0.0, n1 = 0.0;              // brightest, 2nd brightest neighbour
        for (int dj = -1; dj <= 1; ++dj) for (int di = -1; di <= 1; ++di) {
            if (!di && !dj) continue;
            int x = i + di, y = j + dj;
            if (x < 0 || x >= W || y < 0 || y >= H) continue;
            double v = lumaOf(src[(size_t)y * W + x]);
            if (v > n0) { n1 = n0; n0 = v; } else if (v > n1) { n1 = v; }
        }
        size_t idx = (size_t)j * W + i;
        double c = lumaOf(src[idx]);
        double lim = k * n1;
        if (c > lim && lim > 0.0) img[idx] = src[idx] * (lim / c);   // keep the hue
    }
}

// The main entry point. `img` is linear sRGB, W*H, row-major, and is filtered in place.
inline void apply(std::vector<Vec3>& img, int W, int H, const Params& p) {
    if (!p.valid() || W < 3 || H < 3) return;
    const size_t N = (size_t)W * H;
    if (img.size() != N) return;

    clampFireflies(img, W, H, p.fireflies);
    // The clamp is independent of the wavelet cascade, so `-fireflies k` on its own has
    // nothing left to do here and must not pay for a full no-op pass over the image.
    if (p.levels <= 0 || (!(p.luma > 0.0) && !(p.chroma > 0.0))) return;

    // Split once; filter Y, a, b as three scalar fields sharing one luma-derived weight.
    std::vector<double> Y(N), A(N), B(N);
    for (size_t i = 0; i < N; ++i) { YCC c = toYcc(img[i]); Y[i] = c.y; A[i] = c.a; B[i] = c.b; }

    // The edge-stop GUIDE is a 3x3 pre-smoothed copy of the luma, never the raw luma.
    // Guiding on the noisy signal makes every noise spike look like an edge, which both
    // preserves the noise and (see below) skews the filter. It also stays FIXED across
    // the whole cascade: re-deriving it from the partially filtered luma would let level
    // 0's smoothing decide where level 4 is allowed to smooth, which compounds into
    // blotches.
    std::vector<double> guide(N);
    for (int j = 0; j < H; ++j) for (int i = 0; i < W; ++i) {
        double s = 0; int n = 0;
        for (int dj = -1; dj <= 1; ++dj) for (int di = -1; di <= 1; ++di) {
            int x = i + di, y = j + dj;
            if (x < 0 || x >= W || y < 0 || y >= H) continue;
            s += Y[(size_t)y * W + x]; ++n;
        }
        guide[(size_t)j * W + i] = n ? s / n : Y[(size_t)j * W + i];
    }
    const std::vector<double> scale = localScale(guide, W, H);

    std::vector<double> Yo(N), Ao(N), Bo(N), DL(N), DC(N);
    for (int level = 0; level < p.levels; ++level) {
        const int stride = 1 << level;
        // The edge-stop must LOOSEN as the support widens, or the outer levels reject
        // every tap and contribute nothing. sqrt(stride) is the usual compromise: wide
        // enough to keep averaging, tight enough that level 4 does not cross a wall.
        const double sl = p.luma   * std::sqrt((double)stride);
        const double sc = p.chroma * std::sqrt((double)stride);

        // ---------------------------------------------------------------------------
        // ENERGY CONSERVATION. A plain edge-aware gather — out_i = sum_k w_ik in_k / sum_k
        // w_ik — is row-normalised but NOT column-normalised, and on the heavy-tailed
        // distribution MC noise actually has, that regresses each pixel toward the local
        // MODE rather than the local mean. Measured on a 120 spp gallery_rain frame it
        // cost 30% of the image's total luminance over five levels, which is unacceptable
        // in a renderer whose whole point is that the numbers mean something.
        //
        // Two changes make it exact instead. First, the tolerance entering the weight is
        // the GEOMETRIC MEAN of the two pixels' tolerances, tol_i*tol_k, rather than the
        // centre pixel's — which makes w genuinely symmetric, w_ik == w_ki. Second, luma
        // is SCATTERED rather than gathered: each pixel divides its value by its own
        // weight sum and distributes it,
        //        out_i = sum_k w_ik * (in_k / D_k),      D_k = sum_j w_kj
        // whose total is sum_k (in_k/D_k) * sum_i w_ik = sum_k in_k exactly, because
        // symmetry gives sum_i w_ik = D_k. Verified: 100.0000% of input luminance, and it
        // denoises BETTER than the biased gather (local CV 0.243 vs 0.316).
        //
        // Chroma stays a normalised GATHER. It carries no energy to conserve — fromYcc
        // reconstructs the luma exactly whatever the chroma is — and a gather keeps each
        // pixel's hue inside its neighbourhood's range instead of letting a scatter ring
        // past it.
        //
        // BORDERS. Out-of-range taps are MIRRORED, not dropped. Dropping them is still
        // energy-exact (if i's tap at k is in range then k's tap at i is too, so w stays
        // symmetric) but it breaks the other property we need: rows near an edge then sum
        // to less than 1, and since the scatter divides by the NEIGHBOUR's row sum D_k, a
        // deficient border pixel hands out more than it holds and brightens the interior
        // beside it. A flat grey image came back with a visible frame around it. Mirroring
        // fixes that — no tap is ever discarded, so every row sums to 1 — and it keeps the
        // operator symmetric, because the folded matrix is Toeplitz-plus-Hankel and both
        // are symmetric for an even kernel. The filter is then doubly stochastic on flat
        // input: energy exact AND a constant is a fixed point.
        //
        // It must be HALF-sample mirroring (reflect(-1) = 0), not whole-sample
        // (reflect(-1) = 1). Whole-sample reflection fixes the two end samples, so the
        // Hankel term lands on top of the Toeplitz term there and that column is counted
        // twice while its transpose is not: measured, w stopped being symmetric and 0.06%
        // of the energy leaked. Half-sample reflection is a free involution with no fixed
        // point, so the two terms never collide.
        // ---------------------------------------------------------------------------
        auto reflect = [](int x, int n) {
            if (n <= 1) return 0;
            const int period = 2 * n;
            x %= period;
            if (x < 0) x += period;
            return x < n ? x : period - 1 - x;
        };
        auto forEachTap = [&](int i, int j, auto&& fn) {
            const size_t idx = (size_t)j * W + i;
            for (int dj = -2; dj <= 2; ++dj) for (int di = -2; di <= 2; ++di) {
                int x = reflect(i + di * stride, W), y = reflect(j + dj * stride, H);
                size_t k = (size_t)y * W + x;
                const double hs = kB3[di + 2] * kB3[dj + 2];
                const double d = guide[k] - guide[idx];
                const double dd = d * d;
                // Symmetric tolerances: geometric mean of the two endpoints'.
                const double tL = sl * sl * scale[idx] * scale[k];
                const double tC = sc * sc * scale[idx] * scale[k];
                fn(k, hs * std::exp(-dd / (2.0 * tL + 1e-30)),
                      hs * std::exp(-dd / (2.0 * tC + 1e-30)));
            }
        };

        // Row-banded across cores: at 1280x720 x 5 levels this is ~460 M exp() calls,
        // which single-threaded would add seconds to every periodic image write and to
        // every live-window refresh. Bands are read-only on the inputs and disjoint on
        // the outputs, so no synchronisation is needed inside a pass.
        auto parallel = [&](auto&& body) {
            unsigned nt = std::max(1u, std::thread::hardware_concurrency());
            nt = std::min<unsigned>(nt, (unsigned)std::max(1, H / 8));
            if (nt <= 1) { body(0, H); return; }
            std::vector<std::thread> ts; ts.reserve(nt);
            for (unsigned t = 0; t < nt; ++t) {
                int j0 = (int)((size_t)H * t / nt), j1 = (int)((size_t)H * (t + 1) / nt);
                ts.emplace_back(body, j0, j1);
            }
            for (auto& t : ts) t.join();
        };

        // Pass 1: the per-pixel weight sums D, needed by BOTH the luma scatter (as the
        // source divisor) and the chroma gather (as its normaliser).
        parallel([&](int j0, int j1) {
            for (int j = j0; j < j1; ++j) for (int i = 0; i < W; ++i) {
                double dl = 0, dc = 0;
                forEachTap(i, j, [&](size_t, double wl, double wc) { dl += wl; dc += wc; });
                DL[(size_t)j * W + i] = dl; DC[(size_t)j * W + i] = dc;
            }
        });

        // Pass 2: luma scatter (read neighbours' Y/D), chroma LUMA-WEIGHTED gather.
        //
        // Chroma is carried as a ratio to luma, so the obvious gather -- average the
        // neighbours' ratios -- is wrong, and wrong in a way that is easy to miss because
        // it still conserves energy and still passes a flat-field test. The mean of a
        // ratio is not the ratio of the means, and here the denominator is the noisy
        // quantity: a pixel that happened to catch one dim deposit has a tiny Y and
        // therefore an enormous |a|, so a handful of near-black pixels dominate the
        // average and drag a whole neighbourhood to a saturated hue. Measured on a 120spp
        // gallery_rain frame this turned per-pixel rainbow speckle into coherent purple
        // and orange BLOBS -- less obviously noisy, further from the truth.
        //
        // The fix is to average the numerator and the denominator separately and divide
        // once, which is the luma-weighted mean hue:
        //        a_out = sum_k w_ik (R-G)_k / sum_k w_ik Y_k
        // Bright pixels, which are the ones whose hue is actually well determined, now
        // dominate; a black pixel contributes nothing to either sum instead of dominating
        // both. This is the same reason one averages colours, not chromaticities.
        parallel([&](int j0, int j1) {
            for (int j = j0; j < j1; ++j) for (int i = 0; i < W; ++i) {
                const size_t idx = (size_t)j * W + i;
                double sy = 0, sa = 0, sb = 0, syc = 0;
                forEachTap(i, j, [&](size_t k, double wl, double wc) {
                    if (DL[k] > 0) sy += wl * (Y[k] / DL[k]);
                    sa  += wc * A[k] * Y[k];   // w * (R-G)
                    sb  += wc * B[k] * Y[k];   // w * (B-G)
                    syc += wc * Y[k];
                });
                Yo[idx] = (p.luma > 0.0) ? sy : Y[idx];
                Ao[idx] = (p.chroma > 0.0 && syc > 1e-12) ? sa / syc : A[idx];
                Bo[idx] = (p.chroma > 0.0 && syc > 1e-12) ? sb / syc : B[idx];
            }
        });
        Y.swap(Yo); A.swap(Ao); B.swap(Bo);
    }

    for (size_t i = 0; i < N; ++i) img[i] = fromYcc({Y[i], A[i], B[i]});
}

}  // namespace denoise
