// glTF 2.0 / GLB mesh loader. Parses the JSON document (via src/third_party/json.h),
// resolves buffers (GLB BIN chunk, external .bin, or base64 data URIs), walks the
// node hierarchy composing transforms, and bakes each mesh primitive's triangles
// (POSITION / NORMAL / TEXCOORD_0 + indices) into Scene::tris — reusing the same
// per-vertex normal/UV slots the OBJ path fills, so smooth shading and texturing come
// for free. glTF `pbrMetallicRoughness` materials are mapped onto the renderer's
// spectral BSDFs (baseColor -> upsampled reflectance; metallic -> glossy tint;
// roughness -> lobe width), each created once and referenced by primitive.
//
// TEXTURES are imported too: baseColorTexture -> Material::reflectTex (de-gamma'd),
// metallicRoughnessTexture -> Material::roughnessTex (its green plane) plus the mean
// metalness that decides diffuse-vs-metal, normalTexture -> Material::normalTex. Images
// are decoded straight out of the GLB's BIN chunk / a data URI / a sibling file, and
// area-averaged down to kGltfMaxTexDim so an 8K atlas does not cost gigabytes.
//
// Scope / limitations (see known-issues.md): triangles only (primitive.mode 4);
// POSITION/NORMAL/TEXCOORD_0 attributes; no skinning/morph targets, no sparse
// accessors, the KHR material extensions that describe GLASS (ior / transmission /
// volume / dispersion — see the material loop; clearcoat, sheen etc. are still ignored),
// only TEXCOORD_0 (a map on texCoord 1 is sampled with set 0's UVs) and no
// KHR_texture_transform, no occlusion/emissive maps, no animation. Enough to drop static
// Fab/Sketchfab/Blender/Meshy glTF+GLB models into a scene, smooth-shaded and painted.
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
#include "assetbytes.h"
#include "third_party/json.h"

#include <chrono>

// How an imported glTF DIELECTRIC carries the specular lobe glTF gives it:
//   2 = `layered`  (default) the physical coat -- a Fresnel interface over the diffuse body,
//                  so the lobe ramps toward grazing incidence as it should. Modes M, R, W and
//                  A/B/C render it, on BOTH backends since 0.317.0. **Mode D / J / U cannot
//                  render a layered material at all** (a pre-existing gate, not a device one),
//                  which is what `mix` below is for.
//   1 = `mix`      the 0.316.0 stack: an uncoloured glossy lobe at constant weight F0 over the
//                  body. No angular ramp, but every mode can render it.
//   0 = `off`      a flat `diffuse`, the pre-0.316.0 import.
// Metals (`metallic >= 0.5`) are unaffected by all three.
namespace gltfimp { inline int dielectricSpecular = 2; }
// `-import-metal mix`: honour a metallicRoughness map's metalness PER TEXEL, as a two-lobe
// body chosen by the map, instead of typing the whole material by the map's mean. OFF by
// default because the assets this would change are AI-generator exports whose metalness is a
// mid-grey nobody intended physically (Alice's map: max 0.612, i.e. no metal anywhere, yet a
// 0.281 mean makes the dress visibly metallic when honoured). Turning it on is a look
// decision about someone's asset, so it is the author's to make.
namespace gltfimp { inline bool metalMixImport = false; }

namespace gltfimpl {

// Read an entire file into a byte vector. Returns false on open failure.
inline bool readFileBytes(const std::string& authored, std::vector<uint8_t>& out) {
    // External buffer/image URIs are already resolved against the .gltf's own
    // directory by the caller; `resolve` adds the scene search path on top, which
    // matters when the .gltf itself was found somewhere other than the cwd.
    const std::string path = assetbytes::resolve(authored);
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

// ---------------------------------------------------------------------------------
// TEXTURES.  A glTF material's colour usually does NOT live in `baseColorFactor` --
// every DCC and every AI generator exports factor (1,1,1) and puts the whole look in
// `baseColorTexture`. Reading only the factor therefore does not lose "a nuance", it
// turns a painted character into a white blank, which is exactly what a Meshy export
// used to import as (metallicFactor is 1.0 in those files too, so the blank was also
// a MIRROR). The three maps below are the ones that change what a surface IS:
//
//   baseColorTexture         -> Material::reflectTex   (sRGB; the albedo)
//   metallicRoughnessTexture -> Material::roughnessTex (linear; G channel only) and,
//                               via its MEAN metallic, the diffuse/glossy decision
//   normalTexture            -> Material::normalTex    (linear tangent-space)
//
// Everything else glTF can attach (occlusion, emissive, clearcoat/sheen maps) is still
// ignored -- see the file header.
struct TexRef { int index = -1; int texCoord = 0; };

// Read a glTF textureInfo object (`{"index":n,"texCoord":k}`) out of `parent[key]`.
inline TexRef texRefOf(const minijson::Value* parent, const char* key) {
    TexRef t;
    if (!parent) return t;
    if (const minijson::Value* v = parent->find(key); v && v->isObject()) {
        t.index = v->intAt("index", -1);
        t.texCoord = v->intAt("texCoord", 0);
    }
    return t;
}

// What a decoded image is going to be USED for. The same glTF image can legitimately
// be wanted twice with different decodes (a colour map is sRGB, a data map is linear),
// and metallicRoughness has to be split into a single channel before it can drive a
// scalar parameter -- so the cache below is keyed on (glTF texture index, role), not
// on the image alone.
// Metalness is the B plane of the same metallicRoughness image the Roughness role reads --
// a separate binding because that role BROADCASTS G over all three channels (scalarAt
// averages them), destroying B in the process. Only bound when a material actually needs a
// per-texel metal/dielectric split (see metalFrac below), so the second decode is paid for
// exactly by the materials it rescues.
enum class TexRole { Color, Roughness, Normal, Metalness };

// Largest texture dimension kept on import. A Texture stores LINEAR doubles (24 B per
// texel) and a reflectance map builds a Jakob-Hanika coefficient table beside it (another
// 24 B), so an 8192x8192 base-colour atlas -- which is what AI mesh generators emit as a
// matter of course -- would cost 3.2 GB of host RAM before a single ray is traced, then
// the same again on the way to the device. Area-averaging it down to 2048 leaves 200 MB
// and still gives 4.2 M texels to an asset that covers a few tens of thousands of pixels
// in any frame it appears in; the filtering happens in linear light after decode, so it
// is a correct box downsample rather than a resample of gamma-encoded bytes. Raising this
// is safe but buys nothing until an asset genuinely fills the frame.
inline constexpr int kGltfMaxTexDim = 2048;

// Area-average an already-decoded LINEAR image so neither dimension exceeds `maxDim`.
// Source rectangles are computed with integer arithmetic so they tile the source exactly
// (no texel counted twice, none dropped) even when the ratio is not an integer.
inline void downscaleTexture(Texture& t, int maxDim) {
    if (!t.valid() || (t.w <= maxDim && t.h <= maxDim)) return;
    const double sc = (double)maxDim / (double)std::max(t.w, t.h);
    const int nw = std::max(1, (int)std::lround(t.w * sc));
    const int nh = std::max(1, (int)std::lround(t.h * sc));
    std::vector<Vec3> out((size_t)nw * nh);
    for (int y = 0; y < nh; ++y) {
        const int y0 = (int)((int64_t)y * t.h / nh);
        const int y1 = std::max(y0 + 1, (int)((int64_t)(y + 1) * t.h / nh));
        for (int x = 0; x < nw; ++x) {
            const int x0 = (int)((int64_t)x * t.w / nw);
            const int x1 = std::max(x0 + 1, (int)((int64_t)(x + 1) * t.w / nw));
            Vec3 acc{0, 0, 0};
            for (int sy = y0; sy < y1; ++sy)
                for (int sx = x0; sx < x1; ++sx)
                    acc = acc + t.rgb[(size_t)sy * t.w + sx];
            out[(size_t)y * nw + x] = acc * (1.0 / (double)((y1 - y0) * (x1 - x0)));
        }
    }
    t.w = nw; t.h = nh;
    t.rgb.swap(out);
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

// Decode one entry of the document's `images` array into `out` under encoding `enc`.
// glTF gives an image either as a bufferView (the GLB case: the JPEG/PNG bytes sit
// inside the BIN chunk, which is already resolved into doc.buffers) or as a URI, which
// may itself be a base64 data payload or an external file beside the document. All three
// end in Texture::loadMemory, so there is one decode path and no temporary files.
// TEXTURE-DECODE ACCOUNTING (FTRACE_LOADSTATS=1). `assets` in the load profile covers both
// reading/parsing a mesh file and decoding the images its materials reference, and those have
// very different fixes. thread_local for the same reason ftsl.h's asset/accel timers are:
// a parallel loader must not cross-contaminate. Reset by ftsl.h at the top of each load.
inline thread_local double g_texDecodeMs = 0.0;
// ...and the per-texel spectral upsampling that follows it (Texture::buildReflCoeff ->
// upsample::fitMany). A separate counter because the two have entirely different fixes: one is
// an image codec, the other a Jakob-Hanika fit whose dedup pass is serial.
inline thread_local double g_texFitMs = 0.0;

inline bool decodeGltfImage(const Doc& doc, const minijson::Value* imagesArr,
                            int imageIdx, const std::string& baseDir,
                            TexEncoding enc, Texture& out, std::string& err) {
    // Whole-function scope: the decode is the body, and an early-out costs nothing to time.
    const auto _texT0 = std::chrono::steady_clock::now();
    struct TexT { const std::chrono::steady_clock::time_point& t0;
                  ~TexT() { g_texDecodeMs += std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0).count(); } } _texT{_texT0};
    if (!imagesArr || !imagesArr->isArray() ||
        imageIdx < 0 || imageIdx >= (int)imagesArr->arr.size()) {
        err = "glTF image index " + std::to_string(imageIdx) + " out of range";
        return false;
    }
    const minijson::Value& img = imagesArr->arr[imageIdx];
    const std::string what = "glTF image " + std::to_string(imageIdx);
    // Set BEFORE decoding: Texture::storeLinear consults `encoding` per texel, so a
    // colour map must already be marked sRGB when its bytes are de-gamma'd, and a data
    // map (roughness, normals) must already be marked Linear so its bytes are NOT.
    out.encoding = enc;
    if (const minijson::Value* bvv = img.find("bufferView"); bvv && bvv->isNumber()) {
        const int bvi = bvv->asInt();
        if (bvi < 0 || bvi >= (int)doc.views.size()) { err = "bad bufferView in " + what; return false; }
        const BufferView& bv = doc.views[bvi];
        if (bv.buffer < 0 || bv.buffer >= (int)doc.buffers.size()) { err = "bad buffer in " + what; return false; }
        const std::vector<uint8_t>& buf = doc.buffers[bv.buffer];
        if (bv.offset + bv.length > buf.size()) { err = "truncated " + what; return false; }
        return out.loadMemory(buf.data() + bv.offset, bv.length, what, err);
    }
    if (const minijson::Value* uri = img.find("uri"); uri && uri->isString()) {
        std::vector<uint8_t> bytes;
        if (!resolveBufferUri(uri->str, baseDir, bytes) || bytes.empty()) {
            err = "cannot resolve image uri: " + uri->str; return false;
        }
        return out.loadMemory(bytes.data(), bytes.size(), what, err);
    }
    err = what + " has neither bufferView nor uri";
    return false;
}

}  // namespace gltfimpl

// Load a glTF (.gltf) or GLB (.glb) into `s` as world-space triangles placed by `xf`.
// `fallbackMat` is used for primitives lacking a material (or all primitives when
// `importMaterials` is false). `skipMaterials` drops every primitive whose glTF
// material NAME contains one of the given substrings (case-insensitive) -- see the
// filter block below for why. Returns the number of triangles added (0 on failure,
// with a message in `err`). Call before Scene::build().
// FTRACE_LOADSTATS=1: one `[loadstats] asset` line per glTF load. The aggregate `assets` phase
// cannot say WHICH file it is spending its time in, and this scene's four assets span 1.6 MB to
// 111.6 MB, so the aggregate is not actionable on its own.
inline bool gltfLoadStatsOn() {
    static const bool on = [] {
        const char* e = std::getenv("FTRACE_LOADSTATS");
        return e && std::atoi(e) != 0;
    }();
    return on;
}

inline int loadGltf(Scene& s, const char* path, int fallbackMat, const Affine& xf,
                    bool importMaterials, std::string& err,
                    const std::vector<std::string>& skipMaterials = {}) {
    using namespace gltfimpl;
    const auto _gltfT0 = std::chrono::steady_clock::now();
    const double _gltfTex0 = gltfimpl::g_texDecodeMs;   // running total; differenced at the end
    const double _gltfFit0 = gltfimpl::g_texFitMs;
    // Resolve the document ONCE, up front: `dirOf(spath)` below becomes the base for
    // every external buffer/image URI, so it has to be the directory the document was
    // actually found in, not the one the scene happened to name it relative to.
    const std::string authored(path);
    std::string spath = assetbytes::resolve(authored);
    std::vector<uint8_t> file;
    if (!readFileBytes(spath, file)) {
        err = "cannot open " + authored + ": " + assetbytes::describeOpenFailure(authored);
        return 0;
    }

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

    // --- material-name skip filter --------------------------------------------------
    // Asset-store GLBs routinely bundle a backdrop with the subject -- a ground plane,
    // a studio sweep, a display pedestal -- as just another primitive of the same file.
    // There is no way to subtract geometry after the fact, so a scene that wants only
    // the subject has to drop those primitives at LOAD time. Matched case-insensitively
    // as a SUBSTRING of the glTF material's `name`, and done independently of
    // `importMaterials` because the names are in the document either way.
    std::vector<char> matSkip;
    int skippedByMat = 0;
    if (!skipMaterials.empty()) {
        if (const minijson::Value* mats = doc.root.find("materials"); mats && mats->isArray()) {
            matSkip.assign(mats->arr.size(), 0);
            for (size_t i = 0; i < mats->arr.size(); ++i) {
                const minijson::Value* nm = mats->arr[i].find("name");
                if (!nm || !nm->isString()) continue;
                std::string lower = nm->str;
                for (char& c : lower) c = (char)std::tolower((unsigned char)c);
                for (const std::string& pat : skipMaterials) {
                    std::string p = pat;
                    for (char& c : p) c = (char)std::tolower((unsigned char)c);
                    if (!p.empty() && lower.find(p) != std::string::npos) { matSkip[i] = 1; break; }
                }
            }
        }
    }

    // --- image textures --------------------------------------------------------------
    // Decoded lazily and memoised, because one atlas is routinely referenced by several
    // materials and an 8192^2 JPEG costs seconds to decode. The key is not the image
    // alone: the SAME image legitimately wants two different decodes (a colour map is
    // sRGB, a data map is linear), and the factor that glTF multiplies the map by is
    // baked into the texels here (ftrace's texture bindings replace the constant, they
    // do not scale it), so two materials sharing a map with different factors must get
    // two textures. Hence (glTF texture index, role, factor) -> Scene::textures id.
    const minijson::Value* texturesArr = doc.root.find("textures");
    const minijson::Value* imagesArr   = doc.root.find("images");
    const minijson::Value* samplersArr = doc.root.find("samplers");
    struct TexCacheEnt { int gltfTex; int role; double p0, p1, p2; int sceneTex; double meanMetal, meanRough, metalFrac; };
    std::vector<TexCacheEnt> texCache;
    bool texStopped = false;   // buildReflCoeff refused: `ftrace -stop` during scene load

    // Bind glTF texture `ti` in `role` (with glTF's per-material factor `p0..p2` folded
    // in) to a Scene::textures slot, returning its id or -1. `outMeanMetal`, when given,
    // receives the mean of the map's BLUE channel -- for a metallicRoughness map that is
    // the mean metallic, which is the only thing the single-BSDF material choice below
    // can act on (ftrace has no per-texel metal/dielectric blend).
    auto bindTex = [&](int ti, TexRole role, double p0, double p1, double p2,
                       double* outMeanMetal, double* outMeanRough = nullptr,
                       double* outMetalFrac = nullptr) -> int {
        if (texStopped) return -1;
        if (!texturesArr || !texturesArr->isArray() || ti < 0 || ti >= (int)texturesArr->arr.size())
            return -1;
        for (const TexCacheEnt& e : texCache)
            if (e.gltfTex == ti && e.role == (int)role &&
                e.p0 == p0 && e.p1 == p1 && e.p2 == p2) {
                if (outMeanMetal) *outMeanMetal = e.meanMetal;
                if (outMeanRough) *outMeanRough = e.meanRough;
                if (outMetalFrac) *outMetalFrac = e.metalFrac;
                return e.sceneTex;
            }
        const minijson::Value& tj = texturesArr->arr[ti];
        const int src = tj.intAt("source", -1);
        if (src < 0) return -1;

        Texture tex;
        const TexEncoding enc = (role == TexRole::Color) ? TexEncoding::sRGB : TexEncoding::Linear;
        std::string terr;
        if (!decodeGltfImage(doc, imagesArr, src, baseDir, enc, tex, terr)) {
            std::fprintf(stderr, "[gltf] %s: %s (texture ignored)\n", authored.c_str(), terr.c_str());
            return -1;
        }
        downscaleTexture(tex, kGltfMaxTexDim);

        // --- sampler: glTF's GL enums -> ftrace's wrap/filter -------------------------
        if (const int si = tj.intAt("sampler", -1);
            si >= 0 && samplersArr && samplersArr->isArray() && si < (int)samplersArr->arr.size()) {
            const minijson::Value& sm = samplersArr->arr[si];
            auto wrapOf = [](int e) {
                return e == 33071 ? TexWrap::Clamp : (e == 33648 ? TexWrap::Mirror : TexWrap::Repeat);
            };
            // ftrace has a single wrap mode per texture, so an S/T pair that disagrees
            // has to collapse; S wins, which is what every real asset agrees on anyway.
            tex.wrap   = wrapOf(sm.intAt("wrapS", 10497));
            tex.filter = (sm.intAt("magFilter", 9729) == 9728) ? TexFilter::Nearest
                                                              : TexFilter::Bilinear;
        }

        double meanMetal = 0.0, meanRoughOut = p0, metalFrac = 0.0;
        if (role == TexRole::Metalness) {
            // B * metallicFactor, broadcast so scalarAt (which averages RGB) reads metalness.
            const size_t n = tex.rgb.size();
            double acc = 0.0, above = 0.0;
            for (Vec3& c : tex.rgb) {
                const double mv = std::min(1.0, std::max(0.0, c.z * p0));
                acc += mv;
                if (mv >= 0.5) above += 1.0;
                c = Vec3{mv, mv, mv};
            }
            meanMetal = n ? acc / (double)n : 0.0;
            metalFrac = n ? above / (double)n : 0.0;
        } else if (role == TexRole::Roughness) {
            // glTF packs occlusion/roughness/metalness into R/G/B of one image. ftrace's
            // scalarAt averages the three channels, so the G plane has to be broadcast to
            // all three or a rough surface would read as (0 + rough + metal)/3. The mean
            // metallic is harvested first, before B is overwritten.
            const size_t n = tex.rgb.size();
            double aboveHalf = 0.0;
            for (const Vec3& c : tex.rgb) {
                const double mv = std::min(1.0, std::max(0.0, c.z * p1));   // p1 carries metallicFactor
                meanMetal += mv;
                if (mv >= 0.5) aboveHalf += 1.0;
            }
            meanMetal = n ? meanMetal / (double)n : 0.0;
            metalFrac = n ? aboveHalf / (double)n : 0.0;
            double meanR = 0.0;
            for (Vec3& c : tex.rgb) {
                const double g = std::min(1.0, std::max(0.0, c.y * p0));
                meanR += g;
                c = Vec3{g, g, g};
            }
            // The map's MEAN roughness, harvested like the mean metalness above and used as
            // the material's constant. The factor alone is the wrong representative: a
            // Meshy-class export writes `roughnessFactor 1.0` and puts the real value in the
            // map, so every consumer that cannot sample the texture -- the preview
            // rasterizers among them -- saw a fully rough surface and drew a lobe so broad
            // it was invisible. Alice's map means 0.25 (satin); the factor said 1.0.
            meanRoughOut = n ? meanR / (double)n : p0;
        } else if (role == TexRole::Color) {
            if (p0 != 1.0 || p1 != 1.0 || p2 != 1.0)
                for (Vec3& c : tex.rgb) c = Vec3{c.x * p0, c.y * p1, c.z * p2};
            {
                const auto _fitT0 = std::chrono::steady_clock::now();
                const bool okFit = tex.buildReflCoeff();
                g_texFitMs += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - _fitT0).count();
                if (!okFit) { texStopped = true; return -1; }
            }
        }

        tex.name = authored + "#tex" + std::to_string(ti) +
                   (role == TexRole::Color ? ":color"
                    : role == TexRole::Roughness ? ":rough"
                    : role == TexRole::Metalness ? ":metal" : ":normal");
        const int id = (int)s.textures.size();
        s.textures.push_back(std::move(tex));
        texCache.push_back(TexCacheEnt{ti, (int)role, p0, p1, p2, id, meanMetal, meanRoughOut, metalFrac});
        if (outMeanMetal) *outMeanMetal = meanMetal;
        if (outMeanRough) *outMeanRough = meanRoughOut;
        if (outMetalFrac) *outMetalFrac = metalFrac;
        return id;
    };

    // --- materials: map each glTF material index -> a scene material id ------------
    std::vector<int> matMap;   // gltf material index -> scene mat id
    if (importMaterials) {
        if (const minijson::Value* mats = doc.root.find("materials"); mats && mats->isArray()) {
            matMap.resize(mats->arr.size(), fallbackMat);
            for (size_t i = 0; i < mats->arr.size(); ++i) {
                const minijson::Value& mj = mats->arr[i];
                double r = 0.8, g = 0.8, b = 0.8;
                double metallic = 1.0, roughness = 1.0;
                TexRef baseTex, mrTex;
                if (const minijson::Value* pmr = mj.find("pbrMetallicRoughness")) {
                    if (const minijson::Value* bc = pmr->find("baseColorFactor");
                        bc && bc->isArray() && bc->arr.size() >= 3) {
                        r = bc->arr[0].asNumber(0.8);
                        g = bc->arr[1].asNumber(0.8);
                        b = bc->arr[2].asNumber(0.8);
                    }
                    metallic  = pmr->numAt("metallicFactor", 1.0);
                    roughness = pmr->numAt("roughnessFactor", 1.0);
                    baseTex = texRefOf(pmr, "baseColorTexture");
                    mrTex   = texRefOf(pmr, "metallicRoughnessTexture");
                }
                const TexRef nrmTex = texRefOf(&mj, "normalTexture");
                double normalScale = 1.0;
                if (const minijson::Value* nt = mj.find("normalTexture"); nt && nt->isObject())
                    normalScale = nt->numAt("scale", 1.0);

                // Bind the maps BEFORE the BSDF is chosen: `metallic` below is the number
                // that decides diffuse-vs-metal, and in a textured material the factor is
                // only half of it -- Meshy and friends emit metallicFactor 1.0 with the
                // real (near-zero) metalness in the map's blue channel, so reading the
                // factor alone turns a painted character into a mirror.
                int reflectTexId = -1, roughTexId = -1, normalTexId = -1;
                if (baseTex.index >= 0) {
                    reflectTexId = bindTex(baseTex.index, TexRole::Color, r, g, b, nullptr);
                    if (reflectTexId >= 0) { r = g = b = 1.0; }   // factor folded into the texels
                }
                const double metalFactor0 = metallic;   // the raw metallicFactor; the map's mean replaces it below
                double metalFrac = 0.0;          // fraction of texels that are genuinely metal
                if (mrTex.index >= 0) {
                    double meanMetal = metallic, meanRough = roughness;
                    roughTexId = bindTex(mrTex.index, TexRole::Roughness, roughness, metallic, 0.0,
                                         &meanMetal, &meanRough, &metalFrac);
                    if (roughTexId >= 0) {
                        // `meanMetal` already carries metallicFactor (passed as p1), so this is
                        // an assignment, not a second multiply -- squaring it would read a
                        // half-metal export as quarter-metal.
                        metallic = meanMetal;
                        // The map's mean becomes the material's CONSTANT roughness. The
                        // factor is the wrong representative when the real value is in the
                        // map (`roughnessFactor 1.0` + a 0.25 map is the Meshy house style),
                        // and every consumer that cannot sample the texture reads the
                        // constant: the preview rasterizers drew a fully-rough lobe, so
                        // broad it was invisible, on a surface the render showed as satin.
                        roughness = meanRough;
                    }
                }
                if (nrmTex.index >= 0)
                    normalTexId = bindTex(nrmTex.index, TexRole::Normal, 0.0, 0.0, 0.0, nullptr);
                if (texStopped) { err = "scene load stopped"; return 0; }
                // --- KHR material extensions: where GLASS actually lives ---------------
                // A transmissive glTF material says almost nothing in its core block — a
                // coloured gem is routinely `baseColorFactor [1,1,1,1]`, with the tint in
                // KHR_materials_volume and the transparency in KHR_materials_transmission.
                // Ignoring these does not lose a nuance, it turns every gem into an opaque
                // white ball.
                double khrIor = 1.5, khrTransmission = 0.0, khrDispersion = 0.0;
                double attC[3] = {1.0, 1.0, 1.0};
                double attDist = 0.0;                 // 0 == absent == glTF's +infinity
                if (const minijson::Value* ext = mj.find("extensions")) {
                    if (const minijson::Value* e = ext->find("KHR_materials_ior"))
                        khrIor = e->numAt("ior", 1.5);
                    if (const minijson::Value* e = ext->find("KHR_materials_transmission"))
                        khrTransmission = e->numAt("transmissionFactor", 0.0);
                    if (const minijson::Value* e = ext->find("KHR_materials_dispersion"))
                        khrDispersion = e->numAt("dispersion", 0.0);
                    if (const minijson::Value* e = ext->find("KHR_materials_volume")) {
                        attDist = e->numAt("attenuationDistance", 0.0);
                        if (const minijson::Value* ac = e->find("attenuationColor");
                            ac && ac->isArray() && ac->arr.size() >= 3) {
                            attC[0] = ac->arr[0].asNumber(1.0);
                            attC[1] = ac->arr[1].asNumber(1.0);
                            attC[2] = ac->arr[2].asNumber(1.0);
                        }
                    }
                }
                Material m;
                bool wantCoat = false;   // dielectric: add glTF's specular lobe over the body
                // MIXED METALNESS (0.339.0, TODO item 2). glTF's metalness is per TEXEL and a
                // material here is one BSDF, so typing the surface by the map's mean renders a
                // genuinely metallic region as a 4 % dielectric (Alice: mean 0.28, p90 0.53).
                // When the map is neither mostly-metal nor mostly-dielectric, the body becomes
                // TWO lobes chosen per texel by the map. The window is deliberately wide: a map
                // that is 99 % one thing is typed as that one thing and imports bit-identically.
                const bool metalMix = (gltfimp::metalMixImport && mrTex.index >= 0 &&
                                       khrTransmission < 0.5 &&
                                       metalFrac > 0.02 && metalFrac < 0.98);
                int metalTexId = -1;
                if (!gltfimp::metalMixImport && mrTex.index >= 0 && khrTransmission < 0.5 &&
                    metalFrac > 0.02 && metalFrac < 0.98)
                    std::fprintf(stderr, "[gltf] %s: metalness varies across the map (%.1f%% of texels >= 0.5, "
                                         "mean %.3f) but the material is typed by the mean. `-import-metal mix` "
                                         "honours it per texel instead.\n", authored.c_str(), 100.0 * metalFrac, metallic);
                if (metalMix) {
                    metalTexId = bindTex(mrTex.index, TexRole::Metalness, metalFactor0, 0.0, 0.0, nullptr);
                    if (metalTexId < 0) {
                        std::fprintf(stderr, "[gltf] %s: metalness map could not be bound -- "
                                             "falling back to a single BSDF typed by the mean\n", authored.c_str());
                    }
                }
                m.reflect = rgbToReflectanceJH(r, g, b);
                // Heuristic map onto the spectral BSDFs: transmissive -> dielectric,
                // metals -> glossy tinted by the base color, everything else -> diffuse.
                // The 0.5 cut on transmission mirrors the one already used on metallic:
                // these are single-BSDF materials, so a partially transmissive surface has
                // to be called one thing or the other.
                if (khrTransmission >= 0.5) {
                    m.type = MatType::Dielectric;
                    m.roughness = std::max(0.0, roughness);
                    // KHR_materials_dispersion states its strength as 20/Abbe, so an Abbe
                    // number falls straight out of it. Turn that into the two-term Cauchy
                    // n(l) = A + B/l^2 that reproduces the same n_d and the same F-to-C
                    // spread: B from V = (n_d-1)/(n_F-n_C), then A so n(587.6nm) == n_d.
                    if (khrDispersion > 1e-6) {
                        const double V  = 20.0 / khrDispersion;         // Abbe number
                        const double lF = 0.4861, lC = 0.6563, lD = 0.5876;   // micrometres
                        const double B  = (khrIor - 1.0) /
                                          (V * (1.0 / (lF * lF) - 1.0 / (lC * lC)));
                        const double A  = khrIor - B / (lD * lD);
                        m.ior = cauchy(A, B);
                    } else {
                        m.ior = iorConstant(khrIor);
                    }
                    // KHR_materials_volume: transmittance over a distance d is
                    // attenuationColor^(d/attenuationDistance) — Beer-Lambert with
                    // sigma = -ln(attenuationColor)/attenuationDistance, which is exactly
                    // what ftrace's `absorb` is (a coefficient per metre). An absent or
                    // infinite attenuationDistance means no absorption at all.
                    if (attDist > 0.0) {
                        auto sigma = [attDist](double c) {
                            c = std::min(1.0, std::max(1e-6, c));       // ln(0) guard
                            return -std::log(c) / attDist;
                        };
                        const double sr = sigma(attC[0]), sg = sigma(attC[1]), sb = sigma(attC[2]);
                        if (sr > 1e-9 || sg > 1e-9 || sb > 1e-9) {
                            m.absorb = rgbToReflectanceJH(sr, sg, sb);
                            m.absorbRefDist = attDist;   // preview hint; see Material
                        }
                    }
                } else if (metalMix) {
                    // MIXED metalness (0.339.0): neither answer is right for the whole
                    // surface, so the body becomes two lobes picked per texel by the map.
                    // Built as the dielectric below -- the metal lobe is spliced in after the
                    // coat, where the body id is known.
                    m.type = MatType::Diffuse;
                    wantCoat = gltfimp::dielectricSpecular;
                } else if (metallic >= 0.5) {
                    m.type = MatType::Glossy;
                    m.roughness = std::max(0.02, roughness);
                } else {
                    // A DIELECTRIC. glTF gives it a specular lobe as well as an albedo -- F0 =
                    // ((n-1)/(n+1))^2, 4 % at the default ior 1.5, carrying the SAME roughness
                    // map -- and dropping it is what imports a satin dress as chalk. The body is
                    // built here exactly as before; the lobe is added over it below, once the
                    // maps are bound, so the two share them.
                    m.type = MatType::Diffuse;
                    wantCoat = gltfimp::dielectricSpecular;   // 0 none / 1 mix / 2 layered
                }
                // Bind the maps. reflectTex REPLACES the constant `reflect` spectrum at
                // each hit (the factor is already folded into its texels above), so it is
                // safe on every BSDF; roughnessTex is inert on a Diffuse material but is
                // bound anyway, so that a material re-typed later still has its data.
                m.reflectTex  = reflectTexId;
                m.roughnessTex = roughTexId;
                if (normalTexId >= 0) {
                    m.normalTex = normalTexId;
                    m.normalStrength = normalScale;
                }
                // THE DIELECTRIC'S SPECULAR LOBE (0.316.0). Expressed as a two-lobe `mix` --
                // an uncoloured glossy lobe selected with probability F0, the diffuse body with
                // 1-F0 -- because that is what every backend can already render. `layered`, the
                // physical coat, would be the better model and is NOT usable here: it has no
                // device branch, so cudaForwardSupported rejects the whole scene and a mode-M
                // flyby would silently fall back to the CPU tracer. The price is the Fresnel
                // ANGULAR RAMP: a mix weight is a constant, so the lobe stays at F0 instead of
                // rising toward grazing incidence, and the silhouette rim sheen is missing.
                // Mix weights are selection probabilities that are NOT reweighted, so the two
                // lobes partition each photon exactly and energy is conserved by construction.
                if (wantCoat) {
                    // The body is the material built above, unchanged; what differs is how the
                    // lobe is put over it. See gltfimp::dielectricSpecular for the two forms and
                    // why the default is the physical one.
                    Material body = m;
                    const int bodyId = (int)s.mats.size();
                    s.mats.push_back(body);
                    if (wantCoat >= 2) {
                        // A REAL COAT: a Fresnel interface of index `ior`, which gives both the
                        // 4 % normal-incidence reflectance AND the ramp toward grazing that a
                        // constant mix weight cannot express.
                        Material lay;
                        lay.type = MatType::Layered;
                        lay.coatModel = 0;                       // Fresnel dielectric interface
                        lay.ior = iorConstant(khrIor);           // KHR_materials_ior, else 1.5
                        lay.roughness = std::max(0.02, roughness);
                        lay.roughnessTex = roughTexId;           // glTF's roughness drives the coat
                        if (normalTexId >= 0) {
                            lay.normalTex = normalTexId;         // the coat follows the same bumps
                            lay.normalStrength = normalScale;
                        }
                        lay.mixChildren = {bodyId};
                        lay.mixWeights  = {1.0};
                        m = lay;
                    } else {
                        // The flat-weight stack, for the modes that refuse `layered`. The 4 % lives
                        // in the mix weight, so the lobe itself is white -- folding F0 in twice
                        // would square it.
                        const double f0 = ((khrIor - 1.0) / (khrIor + 1.0)) *
                                          ((khrIor - 1.0) / (khrIor + 1.0));
                        Material coat;
                        coat.type = MatType::Glossy;
                        coat.reflect = rgbToReflectanceJH(1.0, 1.0, 1.0);
                        coat.roughness = std::max(0.02, roughness);
                        coat.roughnessTex = roughTexId;
                        if (normalTexId >= 0) {
                            coat.normalTex = normalTexId;
                            coat.normalStrength = normalScale;
                        }
                        const int coatId = (int)s.mats.size();
                        s.mats.push_back(coat);
                        Material mix;
                        mix.type = MatType::Mix;
                        mix.mixChildren = {coatId, bodyId};
                        mix.mixWeights  = {f0, 1.0 - f0};
                        m = mix;
                    }
                }
                // THE METAL LOBE (0.339.0). `m` is now the dielectric as built above -- a
                // `layered` coat over a one-lobe body, or a flat 2-lobe mix, or the bare body.
                // The metal goes in BESIDE the diffuse body, chosen per texel by the map.
                // ONE level of compound is what both resolvers support (host mixResolveChild /
                // device dResolveCompound resolve a single step), so the mix must live in the
                // body-lobe list of the layered stack rather than wrapping it -- wrapping would
                // nest Mix over Layered, which neither backend unwraps, and the dielectric would
                // silently shade through the `default:` branch.
                if (metalMix && metalTexId >= 0) {
                    Material metal;
                    metal.type = MatType::Glossy;
                    metal.reflect = rgbToReflectanceJH(r, g, b);   // glTF: base colour IS the metal's tint
                    metal.reflectTex = reflectTexId;
                    metal.roughness = std::max(0.02, roughness);
                    metal.roughnessTex = roughTexId;
                    if (normalTexId >= 0) { metal.normalTex = normalTexId; metal.normalStrength = normalScale; }
                    const int metalId = (int)s.mats.size();
                    s.mats.push_back(metal);
                    // Constant weights beside the map, for any consumer that cannot sample a
                    // texture (the preview rasterizers): the map's own mean is the honest one.
                    const double wm = std::min(1.0, std::max(0.0, metallic));
                    int bodyId = -1;
                    if (m.type == MatType::Layered && m.mixChildren.size() == 1) {
                        bodyId = m.mixChildren[0];                  // the coat stays over both lobes
                        m.mixChildren = {metalId, bodyId};
                        m.mixWeights  = {wm, 1.0 - wm};
                        m.mixWeightTex = metalTexId;
                    } else if (m.type == MatType::Mix && m.mixChildren.size() == 2) {
                        // The flat-weight coat form (-gltf-specular mix): its 4 % white lobe is
                        // the thing that has to go, because a 2-child mix has exactly one weight
                        // slot and metal-vs-dielectric is the bigger of the two errors by far.
                        bodyId = m.mixChildren[1];
                        m.mixChildren = {metalId, bodyId};
                        m.mixWeights  = {wm, 1.0 - wm};
                        m.mixWeightTex = metalTexId;
                    } else {
                        Material body = m;                          // no coat at all: plain A/B
                        bodyId = (int)s.mats.size();
                        s.mats.push_back(body);
                        Material mx;
                        mx.type = MatType::Mix;
                        mx.mixChildren = {metalId, bodyId};
                        mx.mixWeights  = {wm, 1.0 - wm};
                        mx.mixWeightTex = metalTexId;
                        m = mx;
                    }
                    std::fprintf(stderr, "[gltf] %s: metalness map is mixed (%.1f%% of texels metal, "
                                         "mean %.3f) -- body split into a metal lobe and a dielectric one, "
                                         "chosen per texel\n", authored.c_str(), 100.0 * metalFrac, metallic);
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
            int colAcc = attrs->intAt("COLOR_0", -1);
            if (posAcc < 0) continue;
            std::vector<double> pos, nrm, uv;
            int pc = 0, nc = 0, uc = 0;
            if (!readAccessorFloat(doc, posAcc, pos, pc) || pc < 3) continue;
            bool hasN = (nrmAcc >= 0) && readAccessorFloat(doc, nrmAcc, nrm, nc) && nc >= 3;
            bool hasUV = (uvAcc >= 0) && readAccessorFloat(doc, uvAcc, uv, uc) && uc >= 2;
            // COLOR_0 is glTF's per-vertex colour. It is already LINEAR (unlike a PLY's
            // display-space bytes) and may be VEC3 or VEC4 — the alpha is ignored, since
            // ftrace has no per-vertex opacity to put it in.
            std::vector<double> vcol; int cc = 0;
            bool hasVC = (colAcc >= 0) && readAccessorFloat(doc, colAcc, vcol, cc) && cc >= 3;
            size_t vcount = pos.size() / pc;
            int gltfMat = prim.intAt("material", -1);
            if (gltfMat >= 0 && gltfMat < (int)matSkip.size() && matSkip[gltfMat]) {
                ++skippedByMat; continue;   // `skip_material` -- bundled backdrop, not the subject
            }
            int matId = resolveMat(gltfMat);

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
                // glTF's UV origin is the image's TOP-left (v grows downward); ftrace's
                // is the BOTTOM-left (Texture::sampleRgb indexes row (1-v)*h of top-left
                // storage), which is the OBJ/OpenGL convention the rest of the loader
                // family already uses. Without this flip every glTF texture renders
                // vertically mirrored -- and mirrored on a UV atlas is not "upside down",
                // it is each shell landing on a DIFFERENT shell's pixels, i.e. noise.
                // Flipping here rather than flipping the image also keeps the derived
                // tangent frames (geometry.h) consistent, so normal maps stay correct.
                return Vec3{uv[vi*uc+0], 1.0 - uv[vi*uc+1], 0};
            };
            auto emitTri = [&](uint32_t a, uint32_t bIdx, uint32_t c) {
                if (a >= vcount || bIdx >= vcount || c >= vcount) return;
                Tri t{vertPos(a), vertPos(bIdx), vertPos(c), matId, -1, {}};
                if (hasUV) { t.uv0 = vertUV(a); t.uv1 = vertUV(bIdx); t.uv2 = vertUV(c); }
                if (hasN)  { t.n0 = vertNrm(a); t.n1 = vertNrm(bIdx); t.n2 = vertNrm(c); }
                if (hasVC) {
                    t.vcol = (int)(s.vertColors.size() / 3);
                    for (uint32_t vi : {a, bIdx, c})
                        for (int k = 0; k < 3; ++k)
                            s.vertColors.push_back((float)std::max(0.0, vcol[vi * cc + k]));
                }
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

    char skipNote[64] = {0};
    if (skippedByMat)
        std::snprintf(skipNote, sizeof skipNote, " (skip_material dropped %d prims)", skippedByMat);
    // Report the texture count: a glTF that silently imported ZERO maps is the failure
    // mode that looks like a lighting bug (a painted character arrives as a white blank),
    // so the load line has to say whether any arrived.
    char matNote[64] = {0};
    if (importMaterials && !matMap.empty()) {
        if (texCache.empty()) std::snprintf(matNote, sizeof matNote, " [glTF materials]");
        else std::snprintf(matNote, sizeof matNote, " [glTF materials, %d textures]",
                           (int)texCache.size());
    }
    if (gltfLoadStatsOn())
        std::fprintf(stderr, "[loadstats] asset %s: %.0f ms (decode %.0f, spectral fit %.0f), %d tris\n",
                     path,
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - _gltfT0).count(),
                     gltfimpl::g_texDecodeMs - _gltfTex0,
                     gltfimpl::g_texFitMs - _gltfFit0, added);
    std::printf("loadGltf: %s -> %d tris%s%s%s\n", path, added, matNote,
                skippedNonTri ? " (skipped non-triangle primitives)" : "", skipNote);
    if (added == 0 && err.empty()) err = "no triangles loaded (unsupported primitive layout?)";
    return added;
}
