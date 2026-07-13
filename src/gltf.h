// glTF 2.0 / GLB mesh loader. Parses the JSON document (via src/third_party/json.h),
// resolves buffers (GLB BIN chunk, external .bin, or base64 data URIs), walks the
// node hierarchy composing transforms, and bakes each mesh primitive's triangles
// (POSITION / NORMAL / TEXCOORD_0 + indices) into Scene::tris — reusing the same
// per-vertex normal/UV slots the OBJ path fills, so smooth shading and texturing come
// for free. glTF `pbrMetallicRoughness` materials are mapped onto the renderer's
// spectral BSDFs (baseColor -> upsampled reflectance; metallic -> glossy tint;
// roughness -> lobe width), each created once and referenced by primitive.
//
// Scope / limitations (see known-issues.md): triangles only (primitive.mode 4);
// POSITION/NORMAL/TEXCOORD_0 attributes; no skinning/morph targets, no sparse
// accessors, no KHR material extensions (transmission/clearcoat/etc.), no textures
// (only factor colors), no animation. Enough to drop static Fab/Sketchfab/Blender
// glTF+GLB models into a scene, smooth-shaded, with plausible materials.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <functional>
#include "geometry.h"
#include "scene.h"
#include "linalg.h"
#include "upsample.h"
#include "third_party/json.h"

namespace gltfimpl {

// Read an entire file into a byte vector. Returns false on open failure.
inline bool readFileBytes(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize n = f.tellg();
    if (n < 0) return false;
    f.seekg(0);
    out.resize((size_t)n);
    if (n > 0) f.read(reinterpret_cast<char*>(out.data()), n);
    return true;
}

// Directory prefix of a path (including trailing slash), for resolving relative
// buffer URIs. Returns "" when the path has no directory component.
inline std::string dirOf(const std::string& path) {
    size_t p = path.find_last_of("/\\");
    return (p == std::string::npos) ? std::string() : path.substr(0, p + 1);
}

// Decode a base64 string (data-URI payloads). Ignores whitespace; stops at '='.
inline bool base64Decode(const std::string& in, std::vector<uint8_t>& out) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    int buf = 0, bits = 0;
    for (char c : in) {
        if (c == '=' ) break;
        int v = val(c);
        if (v < 0) continue;   // skip whitespace/newlines
        buf = (buf << 6) | v; bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((uint8_t)((buf >> bits) & 0xFF)); }
    }
    return true;
}

// Resolve a glTF buffer URI to bytes: base64 data URI, or an external file relative
// to `baseDir`. An empty URI (GLB) is filled by the caller from the BIN chunk.
inline bool resolveBufferUri(const std::string& uri, const std::string& baseDir,
                             std::vector<uint8_t>& out) {
    const std::string dataPrefix = "data:";
    if (uri.rfind(dataPrefix, 0) == 0) {
        size_t comma = uri.find(',');
        if (comma == std::string::npos) return false;
        return base64Decode(uri.substr(comma + 1), out);
    }
    return readFileBytes(baseDir + uri, out);
}

// Component byte size for a glTF accessor.componentType.
inline int compByteSize(int ct) {
    switch (ct) {
        case 5120: case 5121: return 1;  // (u)byte
        case 5122: case 5123: return 2;  // (u)short
        case 5125: case 5126: return 4;  // uint / float
        default: return 0;
    }
}
// Number of components for a glTF accessor.type string.
inline int typeNumComp(const std::string& t) {
    if (t == "SCALAR") return 1;
    if (t == "VEC2")   return 2;
    if (t == "VEC3")   return 3;
    if (t == "VEC4")   return 4;
    if (t == "MAT2")   return 4;
    if (t == "MAT3")   return 9;
    if (t == "MAT4")   return 16;
    return 0;
}

struct BufferView { int buffer = -1; size_t offset = 0; size_t length = 0; size_t stride = 0; };
struct Accessor {
    int bufferView = -1; size_t offset = 0; int componentType = 0;
    size_t count = 0; int numComp = 0; bool normalized = false;
};

// Parsed context shared by the accessor readers.
struct Doc {
    minijson::Value root;
    std::vector<std::vector<uint8_t>> buffers;
    std::vector<BufferView> views;
    std::vector<Accessor>   accessors;
};

// Read one component of an accessor element as a double, converting from the stored
// componentType and applying integer normalization when requested.
inline double readComp(const uint8_t* p, int ct, bool norm) {
    switch (ct) {
        case 5126: { float f; std::memcpy(&f, p, 4); return (double)f; }
        case 5125: { uint32_t v; std::memcpy(&v, p, 4); return norm ? (double)v / 4294967295.0 : (double)v; }
        case 5123: { uint16_t v; std::memcpy(&v, p, 2); return norm ? (double)v / 65535.0 : (double)v; }
        case 5122: { int16_t v;  std::memcpy(&v, p, 2); return norm ? std::max((double)v / 32767.0, -1.0) : (double)v; }
        case 5121: { uint8_t v = *p;                    return norm ? (double)v / 255.0 : (double)v; }
        case 5120: { int8_t v = (int8_t)*p;             return norm ? std::max((double)v / 127.0, -1.0) : (double)v; }
        default: return 0.0;
    }
}

// Read a float-valued accessor (positions/normals/texcoords) into a flat array of
// count*numComp doubles. Returns false on malformed indices.
inline bool readAccessorFloat(const Doc& d, int accIdx, std::vector<double>& out, int& numComp) {
    if (accIdx < 0 || accIdx >= (int)d.accessors.size()) return false;
    const Accessor& a = d.accessors[accIdx];
    numComp = a.numComp;
    if (a.bufferView < 0 || a.bufferView >= (int)d.views.size()) return false;
    const BufferView& bv = d.views[a.bufferView];
    if (bv.buffer < 0 || bv.buffer >= (int)d.buffers.size()) return false;
    const std::vector<uint8_t>& buf = d.buffers[bv.buffer];
    int csz = compByteSize(a.componentType);
    if (csz == 0 || numComp == 0) return false;
    size_t stride = bv.stride ? bv.stride : (size_t)csz * numComp;
    size_t base = bv.offset + a.offset;
    out.resize(a.count * numComp);
    for (size_t i = 0; i < a.count; ++i) {
        size_t eltOff = base + i * stride;
        for (int c = 0; c < numComp; ++c) {
            size_t off = eltOff + (size_t)c * csz;
            if (off + csz > buf.size()) return false;
            out[i * numComp + c] = readComp(buf.data() + off, a.componentType, a.normalized);
        }
    }
    return true;
}

// Read a scalar index accessor into uint32.
inline bool readAccessorIndices(const Doc& d, int accIdx, std::vector<uint32_t>& out) {
    if (accIdx < 0 || accIdx >= (int)d.accessors.size()) return false;
    const Accessor& a = d.accessors[accIdx];
    if (a.bufferView < 0 || a.bufferView >= (int)d.views.size()) return false;
    const BufferView& bv = d.views[a.bufferView];
    if (bv.buffer < 0 || bv.buffer >= (int)d.buffers.size()) return false;
    const std::vector<uint8_t>& buf = d.buffers[bv.buffer];
    int csz = compByteSize(a.componentType);
    if (csz == 0) return false;
    size_t stride = bv.stride ? bv.stride : (size_t)csz;
    size_t base = bv.offset + a.offset;
    out.resize(a.count);
    for (size_t i = 0; i < a.count; ++i) {
        size_t off = base + i * stride;
        if (off + csz > buf.size()) return false;
        out[i] = (uint32_t)readComp(buf.data() + off, a.componentType, false);
    }
    return true;
}

// Build an Affine from a glTF node: either its column-major `matrix` (16) or a
// translation/rotation(quat)/scale triple.
inline Affine nodeLocalAffine(const minijson::Value& node) {
    Affine a;
    if (const minijson::Value* m = node.find("matrix"); m && m->isArray() && m->arr.size() == 16) {
        double col[16];
        for (int i = 0; i < 16; ++i) col[i] = m->arr[i].asNumber();
        // glTF matrices are column-major: element(row r, col c) = col[c*4 + r].
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                a.m[r * 3 + c] = col[c * 4 + r];
        a.t = Vec3{col[12], col[13], col[14]};
        return a;
    }
    Vec3 T{0, 0, 0}, S{1, 1, 1};
    double qx = 0, qy = 0, qz = 0, qw = 1;
    if (const minijson::Value* t = node.find("translation"); t && t->arr.size() == 3)
        T = Vec3{t->arr[0].asNumber(), t->arr[1].asNumber(), t->arr[2].asNumber()};
    if (const minijson::Value* r = node.find("rotation"); r && r->arr.size() == 4) {
        qx = r->arr[0].asNumber(); qy = r->arr[1].asNumber();
        qz = r->arr[2].asNumber(); qw = r->arr[3].asNumber();
    }
    if (const minijson::Value* sc = node.find("scale"); sc && sc->arr.size() == 3)
        S = Vec3{sc->arr[0].asNumber(), sc->arr[1].asNumber(), sc->arr[2].asNumber()};
    // Rotation matrix from the unit quaternion (x,y,z,w).
    double xx = qx*qx, yy = qy*qy, zz = qz*qz;
    double xy = qx*qy, xz = qx*qz, yz = qy*qz;
    double wx = qw*qx, wy = qw*qy, wz = qw*qz;
    double R[9] = {
        1 - 2*(yy+zz),   2*(xy - wz),   2*(xz + wy),
        2*(xy + wz),     1 - 2*(xx+zz), 2*(yz - wx),
        2*(xz - wy),     2*(yz + wx),   1 - 2*(xx+yy)
    };
    // M = R * diag(S): column c scaled by S[c]. t = T.
    double sArr[3] = {S.x, S.y, S.z};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            a.m[r * 3 + c] = R[r * 3 + c] * sArr[c];
    a.t = T;
    return a;
}

}  // namespace gltfimpl

// Load a glTF (.gltf) or GLB (.glb) into `s` as world-space triangles placed by `xf`.
// `fallbackMat` is used for primitives lacking a material (or all primitives when
// `importMaterials` is false). Returns the number of triangles added (0 on failure,
// with a message in `err`). Call before Scene::build().
inline int loadGltf(Scene& s, const char* path, int fallbackMat, const Affine& xf,
                    bool importMaterials, std::string& err) {
    using namespace gltfimpl;
    std::string spath(path);
    std::vector<uint8_t> file;
    if (!readFileBytes(spath, file)) { err = "cannot open " + spath; return 0; }

    Doc doc;
    std::vector<uint8_t> glbBin;   // GLB BIN chunk (buffer 0 with no URI)
    bool haveGlbBin = false;

    // --- GLB container: 12-byte header + JSON chunk + optional BIN chunk ----------
    if (file.size() >= 12 && std::memcmp(file.data(), "glTF", 4) == 0) {
        auto rd32 = [&](size_t o) { uint32_t v; std::memcpy(&v, file.data() + o, 4); return v; };
        uint32_t total = rd32(8);
        if (total > file.size()) total = (uint32_t)file.size();
        size_t off = 12;
        std::string jsonText;
        while (off + 8 <= total) {
            uint32_t clen = rd32(off);
            uint32_t ctype = rd32(off + 4);
            size_t cdata = off + 8;
            if (cdata + clen > file.size()) break;
            if (ctype == 0x4E4F534A) {  // 'JSON'
                jsonText.assign((const char*)file.data() + cdata, clen);
            } else if (ctype == 0x004E4942) {  // 'BIN\0'
                glbBin.assign(file.begin() + cdata, file.begin() + cdata + clen);
                haveGlbBin = true;
            }
            off = cdata + clen;
            if (clen % 4) off += 4 - (clen % 4);   // chunks are 4-byte aligned
        }
        if (jsonText.empty()) { err = "GLB has no JSON chunk"; return 0; }
        std::string jerr;
        if (!minijson::parse(jsonText, doc.root, jerr)) { err = "GLB JSON: " + jerr; return 0; }
    } else {
        // Plain .gltf JSON text.
        std::string jsonText((const char*)file.data(), file.size());
        std::string jerr;
        if (!minijson::parse(jsonText, doc.root, jerr)) { err = "glTF JSON: " + jerr; return 0; }
    }
    if (!doc.root.isObject()) { err = "glTF root is not an object"; return 0; }

    // --- buffers ------------------------------------------------------------------
    std::string baseDir = dirOf(spath);
    if (const minijson::Value* bufs = doc.root.find("buffers"); bufs && bufs->isArray()) {
        doc.buffers.resize(bufs->arr.size());
        for (size_t i = 0; i < bufs->arr.size(); ++i) {
            const minijson::Value& bn = bufs->arr[i];
            const minijson::Value* uri = bn.find("uri");
            if (uri && uri->isString()) {
                if (!resolveBufferUri(uri->str, baseDir, doc.buffers[i])) {
                    err = "cannot resolve buffer uri: " + uri->str; return 0;
                }
            } else if (haveGlbBin && i == 0) {
                doc.buffers[i] = glbBin;   // GLB embedded buffer
            } else {
                err = "buffer has no uri and no GLB BIN"; return 0;
            }
        }
    }
    // --- bufferViews --------------------------------------------------------------
    if (const minijson::Value* bvs = doc.root.find("bufferViews"); bvs && bvs->isArray()) {
        doc.views.resize(bvs->arr.size());
        for (size_t i = 0; i < bvs->arr.size(); ++i) {
            const minijson::Value& v = bvs->arr[i];
            BufferView bv;
            bv.buffer = v.intAt("buffer", -1);
            bv.offset = (size_t)v.numAt("byteOffset", 0);
            bv.length = (size_t)v.numAt("byteLength", 0);
            bv.stride = (size_t)v.numAt("byteStride", 0);
            doc.views[i] = bv;
        }
    }
    // --- accessors ----------------------------------------------------------------
    if (const minijson::Value* accs = doc.root.find("accessors"); accs && accs->isArray()) {
        doc.accessors.resize(accs->arr.size());
        for (size_t i = 0; i < accs->arr.size(); ++i) {
            const minijson::Value& v = accs->arr[i];
            Accessor a;
            a.bufferView    = v.intAt("bufferView", -1);
            a.offset        = (size_t)v.numAt("byteOffset", 0);
            a.componentType = v.intAt("componentType", 0);
            a.count         = (size_t)v.numAt("count", 0);
            const minijson::Value* tp = v.find("type");
            a.numComp       = tp && tp->isString() ? typeNumComp(tp->str) : 0;
            const minijson::Value* nm = v.find("normalized");
            a.normalized    = nm && nm->asBool(false);
            doc.accessors[i] = a;
        }
    }

    // --- materials: map each glTF material index -> a scene material id ------------
    std::vector<int> matMap;   // gltf material index -> scene mat id
    if (importMaterials) {
        if (const minijson::Value* mats = doc.root.find("materials"); mats && mats->isArray()) {
            matMap.resize(mats->arr.size(), fallbackMat);
            for (size_t i = 0; i < mats->arr.size(); ++i) {
                const minijson::Value& mj = mats->arr[i];
                double r = 0.8, g = 0.8, b = 0.8;
                double metallic = 1.0, roughness = 1.0;
                if (const minijson::Value* pmr = mj.find("pbrMetallicRoughness")) {
                    if (const minijson::Value* bc = pmr->find("baseColorFactor");
                        bc && bc->isArray() && bc->arr.size() >= 3) {
                        r = bc->arr[0].asNumber(0.8);
                        g = bc->arr[1].asNumber(0.8);
                        b = bc->arr[2].asNumber(0.8);
                    }
                    metallic  = pmr->numAt("metallicFactor", 1.0);
                    roughness = pmr->numAt("roughnessFactor", 1.0);
                }
                Material m;
                m.reflect = rgbToReflectanceJH(r, g, b);
                // Heuristic map onto the spectral BSDFs: metals -> glossy tinted by the
                // base color, dielectrics -> diffuse. Roughness carries straight over.
                if (metallic >= 0.5) {
                    m.type = MatType::Glossy;
                    m.roughness = std::max(0.02, roughness);
                } else {
                    m.type = MatType::Diffuse;
                }
                int id = (int)s.mats.size();
                s.mats.push_back(m);
                matMap[i] = id;
            }
        }
    }
    auto resolveMat = [&](int gltfMatIdx) -> int {
        if (importMaterials && gltfMatIdx >= 0 && gltfMatIdx < (int)matMap.size())
            return matMap[gltfMatIdx];
        return fallbackMat;
    };

    const minijson::Value* meshes = doc.root.find("meshes");
    const minijson::Value* nodes  = doc.root.find("nodes");
    if (!meshes || !meshes->isArray()) { err = "glTF has no meshes"; return 0; }

    int added = 0;
    int skippedNonTri = 0;

    // Emit one mesh's primitives, transformed by `world` (already includes xf).
    std::function<void(int, const Affine&)> emitMesh =
        [&](int meshIdx, const Affine& world) {
        if (meshIdx < 0 || meshIdx >= (int)meshes->arr.size()) return;
        const minijson::Value& mesh = meshes->arr[meshIdx];
        const minijson::Value* prims = mesh.find("primitives");
        if (!prims || !prims->isArray()) return;
        for (const minijson::Value& prim : prims->arr) {
            int mode = prim.intAt("mode", 4);
            if (mode != 4) { ++skippedNonTri; continue; }   // only TRIANGLES
            const minijson::Value* attrs = prim.find("attributes");
            if (!attrs || !attrs->isObject()) continue;
            int posAcc = attrs->intAt("POSITION", -1);
            int nrmAcc = attrs->intAt("NORMAL", -1);
            int uvAcc  = attrs->intAt("TEXCOORD_0", -1);
            if (posAcc < 0) continue;
            std::vector<double> pos, nrm, uv;
            int pc = 0, nc = 0, uc = 0;
            if (!readAccessorFloat(doc, posAcc, pos, pc) || pc < 3) continue;
            bool hasN = (nrmAcc >= 0) && readAccessorFloat(doc, nrmAcc, nrm, nc) && nc >= 3;
            bool hasUV = (uvAcc >= 0) && readAccessorFloat(doc, uvAcc, uv, uc) && uc >= 2;
            size_t vcount = pos.size() / pc;
            int matId = resolveMat(prim.intAt("material", -1));

            auto vertPos = [&](uint32_t vi) {
                return world.apply(Vec3{pos[vi*pc+0], pos[vi*pc+1], pos[vi*pc+2]});
            };
            auto vertNrm = [&](uint32_t vi) -> Vec3 {
                if (!hasN) return Vec3{0, 0, 0};
                Vec3 n = world.applyNormal(Vec3{nrm[vi*nc+0], nrm[vi*nc+1], nrm[vi*nc+2]});
                double l = std::sqrt(dot(n, n));
                return l > 1e-18 ? n * (1.0 / l) : Vec3{0, 0, 0};
            };
            auto vertUV = [&](uint32_t vi) -> Vec3 {
                if (!hasUV) return Vec3{0, 0, 0};
                return Vec3{uv[vi*uc+0], uv[vi*uc+1], 0};
            };
            auto emitTri = [&](uint32_t a, uint32_t bIdx, uint32_t c) {
                if (a >= vcount || bIdx >= vcount || c >= vcount) return;
                Tri t{vertPos(a), vertPos(bIdx), vertPos(c), matId, -1, {}};
                if (hasUV) { t.uv0 = vertUV(a); t.uv1 = vertUV(bIdx); t.uv2 = vertUV(c); }
                if (hasN)  { t.n0 = vertNrm(a); t.n1 = vertNrm(bIdx); t.n2 = vertNrm(c); }
                s.tris.push_back(t);
                ++added;
            };

            int idxAcc = prim.intAt("indices", -1);
            if (idxAcc >= 0) {
                std::vector<uint32_t> idx;
                if (!readAccessorIndices(doc, idxAcc, idx)) continue;
                for (size_t i = 0; i + 2 < idx.size(); i += 3)
                    emitTri(idx[i], idx[i+1], idx[i+2]);
            } else {
                for (uint32_t i = 0; i + 2 < (uint32_t)vcount; i += 3)
                    emitTri(i, i+1, i+2);
            }
        }
    };

    // Walk the node hierarchy, composing transforms (xf placed on top). Cycle-guarded
    // by a visited set (malformed files could otherwise recurse forever).
    std::vector<char> visited(nodes && nodes->isArray() ? nodes->arr.size() : 0, 0);
    std::function<void(int, const Affine&)> walk = [&](int nodeIdx, const Affine& parent) {
        if (!nodes || nodeIdx < 0 || nodeIdx >= (int)nodes->arr.size()) return;
        if (visited[nodeIdx]) return;
        visited[nodeIdx] = 1;
        const minijson::Value& node = nodes->arr[nodeIdx];
        Affine world = parent.compose(nodeLocalAffine(node));
        if (const minijson::Value* mi = node.find("mesh"); mi && mi->isNumber())
            emitMesh(mi->asInt(), world);
        if (const minijson::Value* ch = node.find("children"); ch && ch->isArray())
            for (const minijson::Value& c : ch->arr)
                if (c.isNumber()) walk(c.asInt(), world);
        visited[nodeIdx] = 0;   // allow the same node under different parents (instancing)
    };

    // Roots: the active scene's node list, else scene 0, else every node, else — if
    // there are no nodes at all — every mesh at the identity (some exporters omit the
    // scene graph for a single mesh).
    bool walkedAny = false;
    if (nodes && nodes->isArray() && !nodes->arr.empty()) {
        const minijson::Value* scenesV = doc.root.find("scenes");
        int sceneIdx = doc.root.intAt("scene", 0);
        const minijson::Value* rootList = nullptr;
        if (scenesV && scenesV->isArray() && sceneIdx >= 0 && sceneIdx < (int)scenesV->arr.size())
            rootList = scenesV->arr[sceneIdx].find("nodes");
        if (rootList && rootList->isArray()) {
            for (const minijson::Value& n : rootList->arr)
                if (n.isNumber()) { walk(n.asInt(), xf); walkedAny = true; }
        } else {
            for (size_t i = 0; i < nodes->arr.size(); ++i) { walk((int)i, xf); walkedAny = true; }
        }
    }
    if (!walkedAny) {
        for (size_t i = 0; i < meshes->arr.size(); ++i) emitMesh((int)i, xf);
    }

    std::printf("loadGltf: %s -> %d tris%s%s\n", path, added,
                (importMaterials && !matMap.empty()) ? " [glTF materials]" : "",
                skippedNonTri ? " (skipped non-triangle primitives)" : "");
    if (added == 0 && err.empty()) err = "no triangles loaded (unsupported primitive layout?)";
    return added;
}
