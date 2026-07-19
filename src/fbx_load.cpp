// FBX mesh loader implementation — confines the vendored ufbx library to this one
// translation unit (see src/fbx.h for the rationale and the public signature).
#include "fbx.h"
#include "scene.h"
#include "geometry.h"
#include "linalg.h"

#include <cmath>
#include <cstdio>
#include <vector>

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

}  // namespace

int loadFbx(Scene& s, const char* path, int matId, const Affine& xf,
            bool loadUV, std::string& err) {
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
    ufbx_scene* scene = ufbx_load_file(path, &opts, &uerr);
    if (!scene) {
        char buf[512];
        ufbx_format_error(buf, sizeof(buf), &uerr);
        err = std::string("ufbx: ") + buf;
        return 0;
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
            uint32_t ntri = ufbx_triangulate_face(triIdx.data(), triIdx.size(), mesh, face);
            for (uint32_t t = 0; t < ntri; ++t) {
                uint32_t ia = triIdx[t * 3 + 0];
                uint32_t ib = triIdx[t * 3 + 1];
                uint32_t ic = triIdx[t * 3 + 2];

                Vec3 p0 = world.apply(v3(mesh->vertex_position[ia]));
                Vec3 p1 = world.apply(v3(mesh->vertex_position[ib]));
                Vec3 p2 = world.apply(v3(mesh->vertex_position[ic]));

                Tri tri{p0, p1, p2, matId, -1, {}};

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
