// Ahead-of-time nested-dielectric PRIORITY audit (read-only, runs at scene load).
//
// Where two DIELECTRIC solids overlap in space, the renderer must know which medium
// fills the overlap so it can pick the correct exterior/interior IOR at the shared
// boundary. The nested-dielectric model (Schmidt & Budge 2002) resolves this with an
// integer `priority` per dielectric: the higher priority wins the overlap. If two
// overlapping dielectrics of DIFFERENT media don't have priorities that strictly rank
// them (one or both unset, or equal), the exterior IOR there is ambiguous and the render
// will be wrong. This audit flags exactly those cases BEFORE rendering.
//
// Detection is by BOUNDING-VOLUME overlap of the objects' world AABBs. This is
// conservative in the right direction: each dielectric solid is bounded by its AABB, so
// two solids can only truly overlap where their AABBs overlap — AABB-overlap therefore
// never misses a real overlap. It can over-report (AABBs overlap but the surfaces don't),
// which is acceptable for a warning. For isosurfaces the AABB is the container box/sphere
// bound the field is clipped to, so the same guarantee holds (the surface lives inside
// its container). Objects of the SAME material (same medium/IOR) are skipped: their
// overlap region is that one medium regardless of who "wins", so there is no ambiguity.
#pragma once
#include "scene.h"
#include <string>
#include <vector>

namespace pri {

// One dielectric solid's conservative world AABB + its material's priority state.
struct DielVol {
    std::string label;   // human-readable id, e.g. "sphere #2" / "isosurface #0"
    Aabb        box;     // conservative world bounds
    int         matId;
    bool        hasPri;
    int         pri;
};

// Strict AABB interpenetration (shared face only does NOT count as an overlapping medium).
inline bool aabbOverlap(const Aabb& a, const Aabb& b) {
    return a.lo.x < b.hi.x && a.hi.x > b.lo.x &&
           a.lo.y < b.hi.y && a.hi.y > b.lo.y &&
           a.lo.z < b.hi.z && a.hi.z > b.lo.z;
}

// Collect every dielectric solid with a clean per-object world AABB, then pairwise-test
// for overlapping DIFFERENT-material solids that lack a disambiguating priority. Returns
// one warning string per offending pair (empty => nothing to warn about).
inline std::vector<std::string> audit(const Scene& sc) {
    auto isDiel = [&](int m) {
        return m >= 0 && m < (int)sc.mats.size() && sc.mats[m].type == MatType::Dielectric;
    };
    auto matName = [&](int m) { return std::string("material #") + std::to_string(m); };

    std::vector<DielVol> vols;
    auto push = [&](std::string label, const Aabb& box, int matId) {
        const Material& mt = sc.mats[matId];
        vols.push_back({std::move(label), box, matId, mt.hasPriority(), mt.priority});
    };

    // Analytic spheres — the canonical glass-in-glass / gem-in-water case.
    for (size_t i = 0; i < sc.spheres.size(); ++i) {
        const Sphere& s = sc.spheres[i];
        if (!isDiel(s.matId)) continue;
        Aabb b; b.expand(s.c - Vec3{s.r, s.r, s.r}); b.expand(s.c + Vec3{s.r, s.r, s.r});
        push("sphere #" + std::to_string(i), b, s.matId);
    }
    // Isosurfaces — AABB is the container the field is clipped to (box or sphere bound).
    for (size_t i = 0; i < sc.implicits.size(); ++i) {
        const Implicit& im = sc.implicits[i];
        if (!isDiel(im.matId)) continue;
        push("isosurface #" + std::to_string(i), im.bounds, im.matId);
    }
    // Named triangle-run mesh objects (blasId < 0: geometry lives in Scene::tris). A
    // dielectric mesh's world AABB is the union of its triangle boxes.
    for (const MeshGroup& g : sc.meshGroups) {
        if (g.blasId >= 0 || !isDiel(g.matId)) continue;   // BLAS instances handled below
        Aabb b;
        for (size_t t = g.triStart; t < g.triStart + g.triCount && t < sc.tris.size(); ++t) {
            b.expand(sc.tris[t].v0); b.expand(sc.tris[t].v1); b.expand(sc.tris[t].v2);
        }
        push("mesh '" + g.name + "'", b, g.matId);
    }
    // Instanced dielectric meshes: world AABB = the BLAS local bounds transformed by the
    // instance placement (8 corners). Covers glass assets dropped in via mesh_asset.
    for (size_t i = 0; i < sc.instances.size(); ++i) {
        const MeshInstance& inst = sc.instances[i];
        if (inst.blasId < 0 || inst.blasId >= (int)sc.blasList.size()) continue;
        int mid = inst.matOverride;   // instances usually override the BLAS material
        if (!isDiel(mid)) continue;
        const Aabb& lb = sc.blasList[inst.blasId].localBounds;
        Aabb wb;
        for (int c = 0; c < 8; ++c) {
            Vec3 corner{(c & 1) ? lb.hi.x : lb.lo.x,
                        (c & 2) ? lb.hi.y : lb.lo.y,
                        (c & 4) ? lb.hi.z : lb.lo.z};
            wb.expand(inst.toWorld.apply(corner));
        }
        push("mesh instance #" + std::to_string(i), wb, mid);
    }

    std::vector<std::string> warns;
    for (size_t i = 0; i < vols.size(); ++i)
        for (size_t j = i + 1; j < vols.size(); ++j) {
            const DielVol& a = vols[i];
            const DielVol& b = vols[j];
            if (a.matId == b.matId) continue;          // same medium -> no IOR ambiguity
            if (!aabbOverlap(a.box, b.box)) continue;  // volumes don't overlap
            // Ambiguous unless BOTH have priorities AND they strictly rank.
            bool ranked = a.hasPri && b.hasPri && a.pri != b.pri;
            if (ranked) continue;
            std::string why;
            if (!a.hasPri && !b.hasPri)      why = "neither sets `priority`";
            else if (!a.hasPri)              why = a.label + " has no `priority`";
            else if (!b.hasPri)              why = b.label + " has no `priority`";
            else                             why = "both use priority " + std::to_string(a.pri) + " (a tie)";
            warns.push_back(a.label + " (" + matName(a.matId) + ") overlaps " +
                            b.label + " (" + matName(b.matId) + ") but " + why +
                            " — the exterior IOR in the overlap is ambiguous; add a distinct "
                            "`priority <N>` to each (higher wins).");
        }
    return warns;
}

}  // namespace pri
