// Autodesk FBX mesh loader — a thin bridge over the vendored single-file `ufbx`
// library (src/third_party/ufbx.{h,c}, MIT / public-domain). Declared here with a
// lightweight signature so ftsl.h can dispatch `.fbx` files to it WITHOUT pulling
// the 220-KB ufbx.h into every translation unit — the implementation
// (src/fbx_load.cpp) confines ufbx to a single TU, exactly like src/vdbgrid.cpp
// does for NanoVDB and src/stb_image_impl.cpp for stb.
//
// The loader walks every mesh instance in the FBX scene, bakes each triangulated
// face into Scene::tris (world positions via ufbx's geometry_to_world, per-vertex
// shading normals and UVs into the same Tri slots the OBJ/glTF paths fill), then
// applies the caller's authored-space affine `xf` on top. ufbx normalizes the
// scene to metres and a right-handed Y-up frame at load, so FBX content lands in
// the same convention as the rest of the engine.
#pragma once
#include <string>
#include "geometry.h"
#include "linalg.h"

struct Scene;

// Load `path` (an .fbx / .fbx-family file) into `s`, assigning material `matId` to
// every triangle. `xf` is the mesh block's authored-space transform (applied on top
// of ufbx's own world transform). `loadUV` requests per-vertex UVs from the file's
// first UV set (otherwise the Tri fallback UVs are kept). Returns the number of
// triangles appended; on failure returns 0 and fills `err`.
int loadFbx(Scene& s, const char* path, int matId, const Affine& xf,
            bool loadUV, std::string& err);
