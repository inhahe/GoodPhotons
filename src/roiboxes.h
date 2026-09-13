#pragma once
// -roiboxes — derive per-material screen-space ROIs from the renderer's OWN primary
// visibility, for any scene, exactly.
//
// WHY THIS EXISTS. "Score per-ROI, never whole-frame" is the measurement rule this
// project keeps re-learning, and for a long time it was unenforceable on almost every
// scene: `scraps/gallery_rain.rois` was the ONLY ROI file in the repo, because building
// one meant reading primitive centres out of the .ftsl by hand, projecting them through
// the camera, and then checking each box against the projected footprint of everything
// nearer to it. That process took three drafts and its own header documents two ways it
// silently produced a box on the wrong object. Nobody was going to repeat it per scene,
// so cross-scene work quietly fell back to whole-frame — and a gather-radius sweep was
// then compared against a per-ROI sweep on another scene as though the two numbers meant
// the same thing. Four mechanisms were proposed and withdrawn off the back of that.
//
// THE FIX IS TO STOP REIMPLEMENTING VISIBILITY. A renderer already resolves exactly
// which surface each pixel sees, including occlusion, and it does so with the same
// camera, the same resolution and the same geometry the render will use. So the ROI is
// not computed by reasoning about the scene — it is READ OFF a primary-ray pass. A box
// derived this way cannot land on the cap next door, because the pixels in it are, by
// construction, the pixels showing that material.
//
// A BOUNDING BOX IS STILL NOT AUTOMATICALLY AN ROI, and this is the trap that survives
// the change. A material used in two places has a bbox spanning both and everything
// between them. So the mask is split into CONNECTED COMPONENTS and only the largest is
// reported, and every box carries the two numbers that say whether to trust it:
//
//   purity — the fraction of pixels INSIDE the box that actually show the material.
//            A low purity means the box is mostly other things; the score would be
//            measuring them.
//   share  — the fraction of the material's pixels that live in the reported component.
//            A low share means the material is scattered and one box cannot represent
//            it; that material needs a hand-placed ROI or none at all.
//
// Both are printed always, and a box failing either threshold is emitted COMMENTED OUT
// with the reason, so a bad ROI cannot be picked up by accident from this tool's output.
//
// THE Y ORIGIN IS DERIVED, NOT ASSUMED, and the first version of this file got it wrong.
// `Camera::genRay` maps py = 0 to sy = -1, i.e. to -v: row 0 is the image BOTTOM. The
// .rois format and tools/roi_score.py both measure y downward from the TOP, so emitting
// raster rows directly produces boxes that are correct in x, plausible-looking, and
// VERTICALLY MIRRORED -- the failure mode that is hardest to notice, because every box
// still lands on some real object and the numbers all look reasonable. It was caught by
// a by-construction impossibility: the tool reported materials at y = 0.0 on a frame
// whose top 17 % is provably empty sky, and it put the creature BELOW the plinth cap it
// stands on. So rather than hardcode the convention that was just gotten wrong, the row
// order is measured off the camera itself: trace the first and last row and see which
// one points further along the camera's own up vector. That stays correct if genRay's
// film mapping ever changes, which a comment asserting "row 0 is the bottom" would not.
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>

struct RoiBox {
    std::string name;
    int  x0 = 0, y0 = 0, x1 = 0, y1 = 0;   // pixel bounds of the largest component, inclusive
    long long px = 0;        // pixels of this material inside that component
    long long total = 0;     // pixels of this material anywhere in the frame
    int  comps = 0;          // how many disjoint regions the material occupies
    double purity() const {
        const double a = (double)(x1 - x0 + 1) * (double)(y1 - y0 + 1);
        return a > 0 ? (double)px / a : 0.0;
    }
    double share() const { return total > 0 ? (double)px / (double)total : 0.0; }
};

// One pixel-centre camera ray per pixel; the material each pixel actually sees.
// `-1` means the ray escaped (sky) or hit a sensor, neither of which is an ROI.
// Single-threaded on purpose: this is a diagnostic that runs once at preview
// resolution, and a deterministic serial pass is worth more here than the speed.
inline std::vector<int> roiMaterialImage(const Scene& scene, const Camera& cam,
                                         int resX, int resY) {
    const size_t n = (size_t)resX * (size_t)resY;
    std::vector<int> mid(n, -1);
    for (int py = 0; py < resY; ++py)
        for (int px = 0; px < resX; ++px) {
            Ray r = cam.genRay(px, py, 0.5, 0.5);
            // Same call the composite classifier uses: a `hide_camera` flat must not
            // stand in for what the render will actually show behind it.
            Hit h = scene.closestHit(r, 1e-6, nullptr, /*skipHair=*/false,
                                     /*skipCamHidden=*/true);
            mid[(size_t)py * resX + px] = (h.valid && h.sensorId < 0) ? h.matId : -1;
        }
    return mid;
}

inline std::vector<RoiBox> roiBoxesCompute(const Scene& scene, const Camera& cam,
                                           int resX, int resY) {
    const size_t n = (size_t)resX * (size_t)resY;
    const std::vector<int> mid = roiMaterialImage(scene, cam, resX, resY);

    // Flood-fill connected components of equal material id (4-connectivity), one pass
    // over the whole image rather than one pass per material.
    struct Comp { int mat; long long px; int x0, y0, x1, y1; };
    std::vector<Comp> comps;
    std::vector<int>  lab(n, -1);
    std::vector<int>  stack;
    for (size_t i = 0; i < n; ++i) {
        if (mid[i] < 0 || lab[i] >= 0) continue;
        const int m  = mid[i];
        const int ci = (int)comps.size();
        comps.push_back({m, 0, resX, resY, -1, -1});
        stack.clear();
        stack.push_back((int)i);
        lab[i] = ci;
        while (!stack.empty()) {
            const int q = stack.back(); stack.pop_back();
            const int qx = q % resX, qy = q / resX;
            Comp& c = comps[ci];
            ++c.px;
            if (qx < c.x0) c.x0 = qx;
            if (qy < c.y0) c.y0 = qy;
            if (qx > c.x1) c.x1 = qx;
            if (qy > c.y1) c.y1 = qy;
            const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
            for (int k = 0; k < 4; ++k) {
                const int nx = qx + dx[k], ny = qy + dy[k];
                if (nx < 0 || ny < 0 || nx >= resX || ny >= resY) continue;
                const size_t j = (size_t)ny * (size_t)resX + (size_t)nx;
                if (lab[j] >= 0 || mid[j] != m) continue;
                lab[j] = ci;
                stack.push_back((int)j);
            }
        }
    }

    // Per material: the largest component, plus the totals that say whether it is
    // representative.
    std::vector<RoiBox> out;
    const int nMat = (int)scene.mats.size();
    std::vector<long long> total((size_t)nMat, 0);
    std::vector<int>       count((size_t)nMat, 0);
    std::vector<int>       best((size_t)nMat, -1);
    for (size_t c = 0; c < comps.size(); ++c) {
        const int m = comps[c].mat;
        if (m < 0 || m >= nMat) continue;
        total[m] += comps[c].px;
        ++count[m];
        if (best[m] < 0 || comps[c].px > comps[best[m]].px) best[m] = (int)c;
    }
    for (int m = 0; m < nMat; ++m) {
        if (best[m] < 0) continue;
        const char* nm = scene.matNameFor(m);
        if (!nm) continue;              // unnamed materials cannot be referred to later
        const Comp& c = comps[best[m]];
        RoiBox b;
        b.name  = nm;
        b.x0 = c.x0; b.y0 = c.y0; b.x1 = c.x1; b.y1 = c.y1;
        b.px = c.px; b.total = total[m]; b.comps = count[m];
        out.push_back(b);
    }
    std::sort(out.begin(), out.end(),
              [](const RoiBox& a, const RoiBox& b) { return a.name < b.name; });
    return out;
}

// Print a ready-to-use .rois file on stdout. FIELD ORDER IS `name x0 y0 x1 y1`, x FIRST,
// in [0,1] fractions of width/height — matching tools/roi_score.py's parser, which reads
// x first and yields silent one-pixel boxes if fed y first.
inline int roiBoxesReport(const Scene& scene, const Camera& cam, int resX, int resY,
                          const char* camName, const char* sceneFile,
                          double minPurity, double minShare, long long minPx) {
    const std::vector<RoiBox> boxes = roiBoxesCompute(scene, cam, resX, resY);
    // Ask the camera which raster row is the top, instead of asserting it (see above).
    const Ray rLo = cam.genRay(resX / 2, 0, 0.5, 0.5);
    const Ray rHi = cam.genRay(resX / 2, resY - 1, 0.5, 0.5);
    const bool row0IsTop = dot(rLo.d, cam.v) > dot(rHi.d, cam.v);
    std::printf("# ROIs derived from ftrace's own primary visibility -- not hand-placed.\n");
    std::printf("#   scene   %s\n", sceneFile ? sceneFile : "(built-in)");
    std::printf("#   camera  %s at %dx%d\n",
                (camName && *camName) ? camName : "(default)", resX, resY);
    std::printf("# Each box is the bounding box of the LARGEST connected region showing that\n");
    std::printf("# material, so a material used in several places does not get a box spanning\n");
    std::printf("# all of them. Two numbers say whether the box is usable:\n");
    std::printf("#   purity = pixels in the box that really show the material\n");
    std::printf("#   share  = the material's pixels that live in this one region\n");
    std::printf("# A box below purity %.2f, share %.2f or %lld px is commented out below --\n",
                minPurity, minShare, minPx);
    std::printf("# it is reported so you know the material was seen, not so you can score it.\n");
    std::printf("#\n");
    std::printf("# %-26s %7s %7s %7s %7s   %6s %6s %6s %5s\n",
                "name", "x0", "y0", "x1", "y1", "px", "purity", "share", "regs");
    int usable = 0, rejected = 0;
    for (const RoiBox& b : boxes) {
        const double pur = b.purity(), sh = b.share();
        const bool ok = pur >= minPurity && sh >= minShare && b.px >= minPx;
        char why[96];
        why[0] = '\0';
        if (!ok) {
            if (b.px < minPx)         std::snprintf(why, sizeof why, "  <- too small");
            else if (pur < minPurity) std::snprintf(why, sizeof why, "  <- impure box");
            else                      std::snprintf(why, sizeof why, "  <- %d scattered regions", b.comps);
        }
        // +1 on the far edge: the bounds are inclusive pixel indices, and the consumer
        // slices [x0*W, x1*W), so the last pixel row/column would otherwise be dropped.
        // y is emitted TOP-DOWN whatever the raster order is (see the header).
        const double yTop = row0IsTop ? (double)b.y0 / resY
                                      : (double)(resY - 1 - b.y1) / resY;
        const double yBot = row0IsTop ? (double)(b.y1 + 1) / resY
                                      : (double)(resY - b.y0) / resY;
        std::printf("%s%-26s %7.5f %7.5f %7.5f %7.5f   %6lld %6.3f %6.3f %5d%s\n",
                    ok ? "  " : "# ", b.name.c_str(),
                    (double)b.x0 / resX, yTop,
                    (double)(b.x1 + 1) / resX, yBot,
                    b.px, pur, sh, b.comps, why);
        if (ok) ++usable; else ++rejected;
    }
    std::printf("#\n# %d usable, %d rejected, %d materials visible.\n",
                usable, rejected, (int)boxes.size());
    if (usable == 0)
        std::printf("# No usable ROI. Either the camera sees none of the named materials, or\n"
                    "# every one of them is scattered -- check the camera before scoring.\n");
    return 0;
}

// -roi-audit <file.rois> — say what an EXISTING ROI file's boxes are actually looking at.
//
// The boxes a measurement campaign rests on are usually hand-placed, and a hand-placed box
// can be right about the object and still wrong about the pixels: fur, foliage and any
// thin structure are sub-pixel and interleaved with whatever is behind them, so a box
// squarely on a fur coat can be mostly background. Nothing in a radiance number says so.
// This prints, per box, what fraction of its pixels each material actually occupies, so a
// box can be checked against what it was supposed to sample.
inline int roiAuditReport(const Scene& scene, const Camera& cam, int resX, int resY,
                          const char* roisPath) {
    std::FILE* f = std::fopen(roisPath, "rb");
    if (!f) { std::fprintf(stderr, "[roi-audit] cannot open %s\n", roisPath); return 1; }
    const std::vector<int> mid = roiMaterialImage(scene, cam, resX, resY);
    const Ray rLo = cam.genRay(resX / 2, 0, 0.5, 0.5);
    const Ray rHi = cam.genRay(resX / 2, resY - 1, 0.5, 0.5);
    const bool row0IsTop = dot(rLo.d, cam.v) > dot(rHi.d, cam.v);
    const int nMat = (int)scene.mats.size();

    std::printf("[roi-audit] %s against %s at %dx%d\n", roisPath,
                scene.matNames.empty() ? "(unnamed materials)" : "the visible materials",
                resX, resY);
    std::printf("%-16s %6s  %-24s %7s   %s\n",
                "roi", "px", "dominant material", "share", "rest");
    char line[1024];
    while (std::fgets(line, sizeof line, f)) {
        const char* p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') continue;
        char nm[128];
        double x0, y0, x1, y1;
        if (std::sscanf(p, "%127s %lf %lf %lf %lf", nm, &x0, &y0, &x1, &y1) != 5) continue;

        // .rois y is TOP-DOWN; map it back to this camera's raster row order.
        int rt0 = (int)(y0 * resY), rt1 = (int)(y1 * resY);
        if (rt1 <= rt0) rt1 = rt0 + 1;
        int ry0, ry1;                                   // inclusive raster rows
        if (row0IsTop) { ry0 = rt0; ry1 = rt1 - 1; }
        else           { ry0 = resY - rt1; ry1 = resY - 1 - rt0; }
        int rx0 = (int)(x0 * resX), rx1 = (int)(x1 * resX) - 1;
        if (rx1 < rx0) rx1 = rx0;
        if (ry0 < 0) ry0 = 0; if (ry1 > resY - 1) ry1 = resY - 1;
        if (rx0 < 0) rx0 = 0; if (rx1 > resX - 1) rx1 = resX - 1;

        std::vector<long long> cnt((size_t)nMat + 1, 0);   // [nMat] = escaped (sky)
        long long tot = 0;
        for (int y = ry0; y <= ry1; ++y)
            for (int x = rx0; x <= rx1; ++x) {
                const int m = mid[(size_t)y * resX + x];
                ++cnt[(m < 0 || m >= nMat) ? (size_t)nMat : (size_t)m];
                ++tot;
            }
        if (tot <= 0) continue;
        int b1 = 0;
        for (int m = 1; m <= nMat; ++m) if (cnt[m] > cnt[b1]) b1 = m;
        const char* bn = (b1 == nMat) ? "(sky/escaped)" : scene.matNameFor(b1);
        char rest[256]; rest[0] = '\0';
        int used = 0;
        for (int k = 0; k < 3; ++k) {
            int b = -1;
            for (int m = 0; m <= nMat; ++m)
                if (m != b1 && cnt[m] > 0 && (b < 0 || cnt[m] > cnt[b])) b = m;
            if (b < 0 || cnt[b] == 0) break;
            const char* n2 = (b == nMat) ? "(sky)" : scene.matNameFor(b);
            used += std::snprintf(rest + used, sizeof rest - (size_t)used, "%s%s %.0f%%",
                                  used ? ", " : "", n2 ? n2 : "?",
                                  100.0 * (double)cnt[b] / (double)tot);
            cnt[b] = 0;
            if (used >= (int)sizeof rest - 24) break;
        }
        const double share = (double)cnt[b1] / (double)tot;
        std::printf("%-16s %6lld  %-24s %6.1f%%   %s%s\n", nm, tot, bn ? bn : "(unnamed)",
                    100.0 * share, rest,
                    share < 0.5 ? "   <- the box is mostly NOT its dominant material" : "");
    }
    std::fclose(f);
    return 0;
}
