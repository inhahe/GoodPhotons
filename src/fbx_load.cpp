// FBX mesh loader implementation — confines the vendored ufbx library to this one
// translation unit (see src/fbx.h for the rationale and the public signature).
#define _CRT_SECURE_NO_WARNINGS   // assetbytes.h reads files with plain C stdio
#include "fbx.h"
#include "scene.h"
#include "geometry.h"
#include "linalg.h"
#include "assetbytes.h"
#include "upsample.h"   // rgbToReflectanceJH: base colour -> reflectance spectrum

#include <cmath>
#include <cstdio>
#include <vector>
#include <unordered_map>

#include "third_party/ufbx.h"

namespace {

inline Vec3 v3(const ufbx_vec3& v) { return Vec3{(double)v.x, (double)v.y, (double)v.z}; }

// Map a ufbx matrix onto an ftrace Affine (both are row-major 3x4 world maps: a
// 3x3 linear part + translation). ufbx_matrix stores three column vectors cols[0..2]
// (the transformed basis) plus a translation col `cols[3]`, so the linear entry
// m[row*3 + col] = cols[col].component[row].
inline Affine toAffine(const ufbx_matrix& M) {
    Affine a;
    a.m[0] = M.m00; a.m[1] = M.m01; a.m[2] = M.m02;
    a.m[3] = M.m10; a.m[4] = M.m11; a.m[5] = M.m12;
    a.m[6] = M.m20; a.m[7] = M.m21; a.m[8] = M.m22;
    a.t = Vec3{M.m03, M.m13, M.m23};
    return a;
}

// Turn one ufbx material into an ftrace one. ufbx normalises both property sets for us:
// `pbr` is populated for modern shaders (Arnold/glTF-style), `fbx` for the legacy
// Lambert/Phong ones, and it fills the PBR view from the legacy properties where it can —
// so reading `pbr` first and falling back to `fbx` covers both eras.
//
// The mapping mirrors the OBJ .mtl and glTF importers exactly, including the 0.5 cuts:
// these are single-BSDF materials, so a partially transmissive or partially metallic
// surface has to be called one thing or the other.
inline Material fbxToMaterial(const ufbx_material* fm) {
    Material m;
    double r = 0.8, g = 0.8, b = 0.8;
    if (fm->pbr.base_color.has_value) {
        r = fm->pbr.base_color.value_vec3.x;
        g = fm->pbr.base_color.value_vec3.y;
        b = fm->pbr.base_color.value_vec3.z;
    } else if (fm->fbx.diffuse_color.has_value) {
        r = fm->fbx.diffuse_color.value_vec3.x;
        g = fm->fbx.diffuse_color.value_vec3.y;
        b = fm->fbx.diffuse_color.value_vec3.z;
    }
    m.reflect = rgbToReflectanceJH(r, g, b);

    // Transmission: the PBR slot when present, else the legacy transparency factor.
    double transmission = 0.0;
    if (fm->pbr.transmission_factor.has_value)      transmission = fm->pbr.transmission_factor.value_real;
    else if (fm->fbx.transparency_factor.has_value) transmission = fm->fbx.transparency_factor.value_real;
    const double metalness = fm->pbr.metalness.has_value ? fm->pbr.metalness.value_real : 0.0;
    const double roughness = fm->pbr.roughness.has_value ? fm->pbr.roughness.value_real : 0.5;

    if (transmission >= 0.5) {
        m.type = MatType::Dielectric;
        const double ior = (fm->pbr.specular_ior.has_value && fm->pbr.specular_ior.value_real > 1.0)
                         ? fm->pbr.specular_ior.value_real : 1.5;
        m.ior = iorConstant(ior);
        // transmission_color is a per-channel filter with no distance attached (as .mtl's
        // Tf is), so record it as one scene unit of glass: absorb = -ln(colour), refDist 1.
        double tr = 1.0, tg = 1.0, tb = 1.0;
        if (fm->pbr.transmission_color.has_value) {
            tr = fm->pbr.transmission_color.value_vec3.x;
            tg = fm->pbr.transmission_color.value_vec3.y;
            tb = fm->pbr.transmission_color.value_vec3.z;
        } else if (fm->fbx.transparency_color.has_value) {
            tr = fm->fbx.transparency_color.value_vec3.x;
            tg = fm->fbx.transparency_color.value_vec3.y;
            tb = fm->fbx.transparency_color.value_vec3.z;
        }
        if (tr < 1.0 || tg < 1.0 || tb < 1.0) {
            auto sigma = [](double c) {
                c = std::min(1.0, std::max(1e-6, c));
                return -std::log(c);
            };
            m.absorb = rgbToReflectanceJH(sigma(tr), sigma(tg), sigma(tb));
            m.absorbRefDist = 1.0;
        }
    } else if (metalness >= 0.5) {
        m.type = MatType::Glossy;
        m.roughness = std::max(0.02, roughness);
    } else {
        // Deliberately no legacy-Phong -> Glossy rule. A tight, strong FBX highlight does
        // NOT mean metal: exporters write Phong defaults (specular factor ~0.5, a large
        // exponent) for perfectly ordinary diffuse surfaces, so keying off them would turn
        // most FBX exports into mirrors. Without a metalness channel the honest answer is
        // diffuse.
        m.type = MatType::Diffuse;
    }
    return m;
}

}  // namespace

int loadFbx(Scene& s, const char* path, int matId, const Affine& xf,
            bool loadUV, std::string& err, bool importMaterials) {
    // Normalize the handedness to the engine's convention (right-handed Y-up) and
    // synthesize normals for meshes that ship without them so shading always works.
    // We deliberately do NOT apply unit conversion: FBX's native unit is centimetres,
    // so ufbx's target_unit_meters would bake a 0.01 scale into geometry_to_world and
    // shrink the mesh 100x. Like the OBJ and glTF paths, we consume the file's raw
    // vertex coordinates and let the `mesh { scale ... }` block control final size.
    ufbx_load_opts opts;
    std::memset(&opts, 0, sizeof(opts));
    opts.target_axes = ufbx_axes_right_handed_y_up;
    opts.generate_missing_normals = true;

    ufbx_error uerr;
    // ufbx insists on opening the file itself, so the search path (assetbytes.h) has
    // to be applied to the name before it goes in — this is the one mesh format that
    // does not come through assetbytes::readFile and so does not get it for free.
    const std::string real = assetbytes::resolve(path ? path : "");
    ufbx_scene* scene = ufbx_load_file(real.c_str(), &opts, &uerr);
    if (!scene) {
        char buf[512];
        ufbx_format_error(buf, sizeof(buf), &uerr);
        err = std::string("ufbx: ") + buf;
        if (uerr.type == UFBX_ERROR_FILE_NOT_FOUND)
            err += " — " + assetbytes::describeOpenFailure(path ? path : "");
        return 0;
    }

    // Import the file's materials once, up front: ufbx hands out stable
    // `ufbx_material*` pointers, so a pointer -> scene-material-id table is all the
    // per-face lookup below needs. Faces whose material is missing or unknown keep the
    // caller's fallback, exactly as the OBJ and glTF paths do.
    std::unordered_map<const ufbx_material*, int> matTable;
    if (importMaterials) {
        for (size_t mi = 0; mi < scene->materials.count; ++mi) {
            const ufbx_material* fm = scene->materials.data[mi];
            if (!fm) continue;
            matTable[fm] = (int)s.mats.size();
            s.mats.push_back(fbxToMaterial(fm));
        }
        if (!matTable.empty())
            std::printf("loadFbx: %s -> %zu material(s)\n", path, matTable.size());
    }

    int added = 0;
    size_t nVertsTotal = 0;
    std::vector<uint32_t> triIdx;   // scratch buffer for per-face triangulation

    // Walk every node; each node that instances a mesh contributes triangles baked
    // through that node's geometry_to_world transform (ufbx already resolves the full
    // parent hierarchy + geometric transform for us), then the caller's authored xf.
    for (size_t ni = 0; ni < scene->nodes.count; ++ni) {
        ufbx_node* node = scene->nodes.data[ni];
        if (!node || node->is_root || !node->mesh) continue;
        const ufbx_mesh* mesh = node->mesh;

        // geometry_to_world maps mesh-local vertices to world; normals map by its
        // inverse-transpose (ufbx_matrix_for_normals). Compose ufbx's world map with
        // the caller's authored-space affine.
        Affine g2w = toAffine(node->geometry_to_world);
        Affine world = xf.compose(g2w);

        nVertsTotal += mesh->num_vertices;
        if (mesh->max_face_triangles == 0) continue;
        triIdx.resize(mesh->max_face_triangles * 3);

        const bool haveN  = mesh->vertex_normal.exists;
        const bool haveUV = loadUV && mesh->vertex_uv.exists;

        for (size_t fi = 0; fi < mesh->faces.count; ++fi) {
            ufbx_face face = mesh->faces.data[fi];
            if (face.num_indices < 3) continue;   // skip points/lines/empty faces
            // Per-face material: face_material indexes the MESH's own material list,
            // which is what makes a multi-material FBX come in as several materials
            // rather than one.
            int faceMat = matId;
            if (!matTable.empty() && fi < mesh->face_material.count) {
                const uint32_t mi = mesh->face_material.data[fi];
                if (mi < mesh->materials.count) {
                    auto it = matTable.find(mesh->materials.data[mi]);
                    if (it != matTable.end()) faceMat = it->second;
                }
            }
            uint32_t ntri = ufbx_triangulate_face(triIdx.data(), triIdx.size(), mesh, face);
            for (uint32_t t = 0; t < ntri; ++t) {
                uint32_t ia = triIdx[t * 3 + 0];
                uint32_t ib = triIdx[t * 3 + 1];
                uint32_t ic = triIdx[t * 3 + 2];

                Vec3 p0 = world.apply(v3(mesh->vertex_position[ia]));
                Vec3 p1 = world.apply(v3(mesh->vertex_position[ib]));
                Vec3 p2 = world.apply(v3(mesh->vertex_position[ic]));

                Tri tri{p0, p1, p2, faceMat, -1, {}};

                if (haveN) {
                    auto wn = [&](uint32_t i) {
                        Vec3 n = world.applyNormal(v3(mesh->vertex_normal[i]));
                        double l = std::sqrt(dot(n, n));
                        return l > 1e-18 ? n * (1.0 / l) : Vec3{0, 0, 0};
                    };
                    tri.n0 = wn(ia); tri.n1 = wn(ib); tri.n2 = wn(ic);
                }
                if (haveUV) {
                    auto uvAt = [&](uint32_t i) {
                        ufbx_vec2 uv = mesh->vertex_uv[i];
                        return Vec3{(double)uv.x, (double)uv.y, 0.0};
                    };
                    tri.uv0 = uvAt(ia); tri.uv1 = uvAt(ib); tri.uv2 = uvAt(ic);
                }
                s.tris.push_back(tri);
                ++added;
            }
        }
    }

    ufbx_free_scene(scene);
    std::printf("loadFbx: %s -> %d verts, %d tris (mat %d)\n",
                path, (int)nVertsTotal, added, matId);
    return added;
}
