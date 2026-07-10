// Minimal Wavefront OBJ loader. Reads v / f (triangulating polygons via a fan);
// ignores vt/vn/mtl for now (surface appearance comes from the assigned material,
// and geometric normals are recomputed in Scene::build). Dependency-free — enough
// to drop real meshes into a scene and stress the BVH. A richer loader (normals,
// per-face materials, .mtl) can replace this later.
#pragma once
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "geometry.h"
#include "scene.h"

// Parse the leading vertex index of an OBJ face token ("12", "12/3", "12/3/4",
// "12//4"). OBJ indices are 1-based; negative means relative to the current end.
inline int objVertexIndex(const std::string& tok, int vertexCount) {
    int idx = std::atoi(tok.c_str());
    if (idx > 0)  return idx - 1;
    if (idx < 0)  return vertexCount + idx;   // relative
    return -1;
}

// Load an OBJ into the scene as triangles of material `matId`, transformed by
// world = translate + scale * local. Returns the number of triangles added
// (0 on failure). Call before Scene::build().
inline int loadObj(Scene& s, const char* path, int matId,
                   Vec3 translate = {0, 0, 0}, double scale = 1.0) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "loadObj: cannot open %s\n", path); return 0; }

    std::vector<Vec3> verts;
    int added = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && line[1] == ' ') {
            std::istringstream ss(line.substr(2));
            Vec3 v; ss >> v.x >> v.y >> v.z;
            verts.push_back(translate + v * scale);
        } else if (line[0] == 'f' && line[1] == ' ') {
            std::istringstream ss(line.substr(2));
            std::vector<int> idx;
            std::string tok;
            while (ss >> tok) {
                int vi = objVertexIndex(tok, (int)verts.size());
                if (vi >= 0 && vi < (int)verts.size()) idx.push_back(vi);
            }
            // Fan-triangulate the polygon (0, k, k+1).
            for (size_t k = 1; k + 1 < idx.size(); ++k) {
                s.tris.push_back(Tri{verts[idx[0]], verts[idx[k]], verts[idx[k + 1]], matId, -1, {}});
                ++added;
            }
        }
    }
    std::printf("loadObj: %s -> %d verts, %d tris (mat %d)\n",
                path, (int)verts.size(), added, matId);
    return added;
}
