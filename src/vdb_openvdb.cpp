// vdb_openvdb.cpp — native OpenVDB `.vdb` reader (no OpenVDB / NanoVDB dependency).
//
// Parses the OpenVDB file container (header, grid descriptor, per-grid
// compression / metadata / transform / tree) and bakes the first `float` grid
// into the renderer's dense NanoVDB-free VdbGrid. The tree is the standard
// `float 5_4_3` layout (RootNode → Internal<5> → Internal<4> → Leaf<3>); leaf and
// tile buffers are read with OpenVDB's ACTIVE_MASK + BLOSC compression and (when
// the grid was saved as half) 16-bit HalfFloat storage.
//
// The BLOSC codec used by every mainstream writer (Houdini, and OpenVDB's own
// `-l lz4` export) is LZ4, which we decode with the vendored single-file LZ4
// (src/third_party/lz4.*) plus a compact, exact reimplementation of blosc1's
// chunk/block/byte-shuffle framing (validated bit-for-bit against python-blosc on
// the official smoke/sphere/cube samples). Other blosc codecs (BloscLZ / Zlib /
// Zstd) and ZIP compression are reported with a clear "re-export with LZ4"
// message rather than silently producing garbage — see known-issues.md.
//
// The serialization layout is transcribed directly from the OpenVDB source
// (io/Archive.cc, io/Compression.h/.cc, io/GridDescriptor.cc, tree/{Root,Internal,
// Leaf}Node.h, math/{Transform,Maps}); see the C4 notes in TODO.md.

#define _CRT_SECURE_NO_WARNINGS

#include "vdbgrid.h"

extern "C" {
#include "third_party/lz4.h"
}

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cctype>
#include <memory>
#include <string>
#include <vector>

namespace {

// ---- OpenVDB file-format constants ---------------------------------------
constexpr uint32_t V_GRID_INSTANCING       = 216;
constexpr uint32_t V_NODE_MASK_COMPRESSION = 222;  // metadata byte + grid compression uint32

constexpr uint32_t COMPRESS_ZIP         = 0x1;
constexpr uint32_t COMPRESS_ACTIVE_MASK = 0x2;
constexpr uint32_t COMPRESS_BLOSC       = 0x4;

// io::readCompressedValues metadata flags (Compression.h)
enum : int8_t {
    NO_MASK_OR_INACTIVE_VALS   = 0,
    NO_MASK_AND_MINUS_BG       = 1,
    NO_MASK_AND_ONE_INACTIVE   = 2,
    MASK_AND_NO_INACTIVE_VALS  = 3,
    MASK_AND_ONE_INACTIVE_VAL  = 4,
    MASK_AND_TWO_INACTIVE_VALS = 5,
    NO_MASK_AND_ALL_VALS       = 6,
};

// blosc1 header flags (blosc.h)
constexpr uint8_t BLOSC_DOSHUFFLE    = 0x01;
constexpr uint8_t BLOSC_MEMCPYED     = 0x02;
constexpr uint8_t BLOSC_DOBITSHUFFLE = 0x04;
constexpr int     BLOSC_MIN_BUFFERSIZE = 128;
constexpr int     BLOSC_MAX_SPLITS     = 16;

// A parse failure carries its message via this exception so the recursive
// descent can bail out cleanly; loadOpenVDBGrid() catches it and fills `err`.
struct VdbError { std::string msg; };
[[noreturn]] void fail(const std::string& m) { throw VdbError{ m }; }

// ---- little-endian byte cursor over the whole file -----------------------
struct Cursor {
    const uint8_t* b;
    size_t n;
    size_t p = 0;
    Cursor(const uint8_t* buf, size_t len, size_t pos = 0) : b(buf), n(len), p(pos) {}

    void need(size_t k) const { if (p + k > n) fail("unexpected end of .vdb file"); }
    const uint8_t* take(size_t k) { need(k); const uint8_t* q = b + p; p += k; return q; }

    template <class T> T scalar() { need(sizeof(T)); T v; std::memcpy(&v, b + p, sizeof(T)); p += sizeof(T); return v; }
    uint32_t u32() { return scalar<uint32_t>(); }
    int32_t  i32() { return scalar<int32_t>(); }
    int64_t  i64() { return scalar<int64_t>(); }
    uint8_t  u8()  { return scalar<uint8_t>(); }
    int8_t   i8()  { return scalar<int8_t>(); }
    float    f32() { return scalar<float>(); }
    double   f64() { return scalar<double>(); }

    std::string str() {  // io::readString: uint32 length + chars (no NUL)
        uint32_t len = u32();
        const uint8_t* q = take(len);
        return std::string(reinterpret_cast<const char*>(q), len);
    }
    void skipMeta() {   // Metadata::read: name, typeName, uint32 size, size bytes
        (void)str(); (void)str();
        uint32_t size = u32();
        take(size);
    }
};

inline int popcountBytes(const uint8_t* mask, size_t bytes) {
    int c = 0;
    for (size_t i = 0; i < bytes; ++i) {
        uint8_t x = mask[i];
        while (x) { x &= (uint8_t)(x - 1); ++c; }
    }
    return c;
}
inline bool bitOn(const uint8_t* mask, int n) { return (mask[n >> 3] >> (n & 7)) & 1; }

inline float halfToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else {  // subnormal → normalize
            exp = 1;
            while (!(man & 0x400)) { man <<= 1; --exp; }
            man &= 0x3FF;
            bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {  // Inf / NaN
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float f; std::memcpy(&f, &bits, 4); return f;
}

// ---- blosc1 single-chunk decompressor (LZ4 codec + memcpy + byte shuffle) --
// Reproduces c-blosc's blosc_decompress for the subset written by OpenVDB.
void bloscDecodeChunk(const uint8_t* data, size_t dataLen, uint8_t* out, size_t outBytes) {
    if (dataLen < 16) fail("blosc: truncated chunk header");
    uint8_t flags    = data[2];
    uint8_t typesize = data[3];
    uint32_t nbytes, blocksize, cbytes;
    std::memcpy(&nbytes,    data + 4,  4);
    std::memcpy(&blocksize, data + 8,  4);
    std::memcpy(&cbytes,    data + 12, 4);
    if (nbytes != outBytes) fail("blosc: size mismatch");

    if (flags & BLOSC_MEMCPYED) {                  // stored uncompressed
        if (16 + (size_t)nbytes > dataLen) fail("blosc: truncated memcpy chunk");
        std::memcpy(out, data + 16, nbytes);
        return;
    }
    if (flags & BLOSC_DOBITSHUFFLE)
        fail("blosc: bit-shuffled .vdb not supported (re-export with byte shuffle / LZ4)");
    const uint32_t compformat = (uint32_t)((flags & 0xE0) >> 5);
    if (compformat != 1)   // 0=BloscLZ 1=LZ4 3=Zlib 4=Zstd
        fail("blosc: unsupported codec (only LZ4 is built in) — re-export the .vdb "
             "with blosc LZ4 compression");
    const bool doshuffle = (flags & BLOSC_DOSHUFFLE) && typesize > 1;
    const bool dontSplit = (flags & 0x10) != 0;

    if (blocksize == 0) fail("blosc: zero block size");
    uint32_t nblocks = nbytes / blocksize;
    uint32_t leftover = nbytes % blocksize;
    if (leftover) ++nblocks;

    const uint8_t* bstarts = data + 16;            // nblocks × int32 offsets
    if (16 + (size_t)nblocks * 4 > dataLen) fail("blosc: truncated block table");

    std::vector<uint8_t> tmp;                       // shuffled scratch (per block)
    for (uint32_t bi = 0; bi < nblocks; ++bi) {
        uint32_t bsize = blocksize;
        bool leftoverBlock = false;
        if (bi == nblocks - 1 && leftover > 0) { bsize = leftover; leftoverBlock = true; }

        int nsplits = 1;
        if (!dontSplit && typesize <= BLOSC_MAX_SPLITS &&
            (int)(blocksize / typesize) >= BLOSC_MIN_BUFFERSIZE && !leftoverBlock)
            nsplits = typesize;
        uint32_t neblock = bsize / (uint32_t)nsplits;

        int32_t off;
        std::memcpy(&off, bstarts + (size_t)bi * 4, 4);
        if (off < 0 || (size_t)off > dataLen) fail("blosc: bad block offset");
        size_t sp = (size_t)off;

        uint8_t* dst = doshuffle ? (tmp.resize(bsize), tmp.data()) : (out + (size_t)bi * blocksize);
        uint8_t* d = dst;
        for (int j = 0; j < nsplits; ++j) {
            if (sp + 4 > dataLen) fail("blosc: truncated split header");
            int32_t cb; std::memcpy(&cb, data + sp, 4); sp += 4;
            if (cb < 0 || sp + (size_t)cb > dataLen) fail("blosc: bad split size");
            if ((uint32_t)cb == neblock) {          // this split stored raw
                std::memcpy(d, data + sp, neblock);
            } else {
                int got = LZ4_decompress_safe(reinterpret_cast<const char*>(data + sp),
                                              reinterpret_cast<char*>(d),
                                              cb, (int)neblock);
                if (got != (int)neblock) fail("blosc: LZ4 decode error");
            }
            sp += (size_t)cb;
            d  += neblock;
        }
        if (doshuffle) {                            // un-byte-shuffle tmp → out block
            uint8_t* o = out + (size_t)bi * blocksize;
            uint32_t nelem = bsize / typesize;
            for (uint32_t t = 0; t < typesize; ++t)
                for (uint32_t k = 0; k < nelem; ++k)
                    o[k * typesize + t] = tmp[t * nelem + k];
            for (uint32_t x = nelem * typesize; x < bsize; ++x)  // trailing bytes verbatim
                o[x] = tmp[x];
        }
    }
}

// io::bloscFromStream — read the int64 length prefix, then decode into `out`.
void bloscFromStream(Cursor& c, uint8_t* out, size_t outBytes) {
    int64_t ncomp = c.i64();
    if (ncomp <= 0) {                               // negative → stored uncompressed
        const uint8_t* q = c.take((size_t)(-ncomp));
        if ((size_t)(-ncomp) != outBytes) fail("blosc: uncompressed size mismatch");
        std::memcpy(out, q, outBytes);
        return;
    }
    const uint8_t* q = c.take((size_t)ncomp);
    bloscDecodeChunk(q, (size_t)ncomp, out, outBytes);
}

// io::readData — read `count` values (BLOSC/ZIP/none) as raw bytes into `out`.
void readRawData(Cursor& c, uint8_t* out, size_t nbytes, uint32_t compression) {
    if (compression & COMPRESS_BLOSC)      bloscFromStream(c, out, nbytes);
    else if (compression & COMPRESS_ZIP)   fail("ZIP-compressed .vdb not supported (re-export with LZ4)");
    else                                   std::memcpy(out, c.take(nbytes), nbytes);
}

// io::readCompressedValues<float> — decode `destCount` float values honoring the
// grid's ACTIVE_MASK + (half)float compression, using `valueMask` to restore the
// unsaved inactive voxels. Returns dest as a float vector of length destCount.
std::vector<float> readCompressedValues(Cursor& c, int destCount, const uint8_t* valueMask,
                                        bool fromHalf, uint32_t compression, float background,
                                        uint32_t fileVer) {
    const bool maskCompressed = (compression & COMPRESS_ACTIVE_MASK) != 0;
    int8_t metadata = NO_MASK_AND_ALL_VALS;
    if (fileVer >= V_NODE_MASK_COMPRESSION) metadata = c.i8();

    float inactive1 = background;
    float inactive0 = (metadata == NO_MASK_OR_INACTIVE_VALS) ? background : -background;
    if (metadata == NO_MASK_AND_ONE_INACTIVE || metadata == MASK_AND_ONE_INACTIVE_VAL ||
        metadata == MASK_AND_TWO_INACTIVE_VALS) {
        inactive0 = c.f32();                        // inactive vals are full float, even when half
        if (metadata == MASK_AND_TWO_INACTIVE_VALS) inactive1 = c.f32();
    }
    const uint8_t* selectionMask = nullptr;
    if (metadata == MASK_AND_NO_INACTIVE_VALS || metadata == MASK_AND_ONE_INACTIVE_VAL ||
        metadata == MASK_AND_TWO_INACTIVE_VALS) {
        selectionMask = c.take((size_t)destCount / 8);
    }

    int tempCount = destCount;
    if (maskCompressed && metadata != NO_MASK_AND_ALL_VALS)
        tempCount = popcountBytes(valueMask, (size_t)destCount / 8);

    std::vector<float> temp((size_t)tempCount);
    if (tempCount >= 1) {
        if (fromHalf) {
            std::vector<uint16_t> half((size_t)tempCount);
            readRawData(c, reinterpret_cast<uint8_t*>(half.data()),
                        (size_t)tempCount * 2, compression);
            for (int i = 0; i < tempCount; ++i) temp[(size_t)i] = halfToFloat(half[(size_t)i]);
        } else {
            readRawData(c, reinterpret_cast<uint8_t*>(temp.data()),
                        (size_t)tempCount * 4, compression);
        }
    }

    if (maskCompressed && metadata != NO_MASK_AND_ALL_VALS && tempCount != destCount) {
        std::vector<float> dest((size_t)destCount);
        int ti = 0;
        for (int di = 0; di < destCount; ++di) {
            if (bitOn(valueMask, di)) dest[(size_t)di] = temp[(size_t)ti++];
            else dest[(size_t)di] = (selectionMask && bitOn(selectionMask, di)) ? inactive1 : inactive0;
        }
        return dest;
    }
    return temp;   // tempCount == destCount
}

// ---- tree nodes (float 5_4_3) --------------------------------------------
struct Leaf {
    int32_t origin[3];
    std::vector<uint8_t> valueMask;   // 64 bytes (512 bits)
    std::vector<float>   values;      // 512
};
struct Internal {
    int32_t origin[3];
    int log2dim;          // 5 (top) or 4
    int childTotal;       // child's voxel-dim log2 (7 for I5, 3 for I4)
    int numValues;        // 1<<(3*log2dim)
    bool childIsLeaf;
    std::vector<uint8_t> childMask, valueMask;
    std::vector<float>   values;      // tile values
    // children in ascending offset order (matches beginChildOn):
    std::vector<std::pair<int, std::unique_ptr<Internal>>> childInternals;
    std::vector<std::pair<int, std::unique_ptr<Leaf>>>     childLeaves;
};

inline std::vector<uint8_t> readMask(Cursor& c, size_t bytes) {
    const uint8_t* q = c.take(bytes);
    return std::vector<uint8_t>(q, q + bytes);
}

// InternalNode::readTopology: childMask, valueMask, tile values, then recurse into
// each active child's topology.
std::unique_ptr<Internal> readInternalTopology(Cursor& c, int32_t ox, int32_t oy, int32_t oz,
                                               int log2dim, int childTotal, bool childIsLeaf,
                                               bool fromHalf, uint32_t compression,
                                               float background, uint32_t fileVer) {
    auto node = std::make_unique<Internal>();
    node->origin[0] = ox; node->origin[1] = oy; node->origin[2] = oz;
    node->log2dim = log2dim; node->childTotal = childTotal;
    node->numValues = 1 << (3 * log2dim);
    node->childIsLeaf = childIsLeaf;
    const size_t maskBytes = (size_t)node->numValues / 8;
    node->childMask = readMask(c, maskBytes);
    node->valueMask = readMask(c, maskBytes);
    node->values = readCompressedValues(c, node->numValues, node->valueMask.data(),
                                        fromHalf, compression, background, fileVer);
    const int mask = (1 << log2dim) - 1;
    for (int off = 0; off < node->numValues; ++off) {
        if (!bitOn(node->childMask.data(), off)) continue;
        // offsetToLocalCoord: x = off>>2L, y = (off>>L)&mask, z = off&mask
        int i = (off >> (2 * log2dim)) & mask;
        int j = (off >> log2dim) & mask;
        int k = off & mask;
        int32_t cx = ox + (i << childTotal);
        int32_t cy = oy + (j << childTotal);
        int32_t cz = oz + (k << childTotal);
        if (childIsLeaf) {
            auto leaf = std::make_unique<Leaf>();
            leaf->origin[0] = cx; leaf->origin[1] = cy; leaf->origin[2] = cz;
            leaf->valueMask = readMask(c, 64);   // 8^3 bits
            node->childLeaves.emplace_back(off, std::move(leaf));
        } else {
            node->childInternals.emplace_back(
                off, readInternalTopology(c, cx, cy, cz, 4, 3, true,
                                          fromHalf, compression, background, fileVer));
        }
    }
    return node;
}

// InternalNode::readBuffers: recurse; a leaf re-reads its value mask then its voxels.
void readInternalBuffers(Cursor& c, Internal* node, bool fromHalf, uint32_t compression,
                         float background, uint32_t fileVer) {
    for (auto& kv : node->childInternals)
        readInternalBuffers(c, kv.second.get(), fromHalf, compression, background, fileVer);
    for (auto& kv : node->childLeaves) {
        Leaf* leaf = kv.second.get();
        leaf->valueMask = readMask(c, 64);   // value mask written again in the buffer pass
        leaf->values = readCompressedValues(c, 512, leaf->valueMask.data(),
                                            fromHalf, compression, background, fileVer);
    }
}

// Invert a row-major 3x3 matrix; false if (near-)singular.
bool invert3x3(const double m[9], double out[9]) {
    double a = m[0], b = m[1], cc = m[2], d = m[3], e = m[4], f = m[5],
           g = m[6], h = m[7], i = m[8];
    double A = e*i - f*h, B = -(d*i - f*g), C = d*h - e*g;
    double det = a*A + b*B + cc*C;
    if (det > -1e-30 && det < 1e-30) return false;
    double inv = 1.0 / det;
    out[0] = A * inv;              out[1] = -(b*i - cc*h) * inv;  out[2] =  (b*f - cc*e) * inv;
    out[3] = B * inv;              out[4] =  (a*i - cc*g) * inv;  out[5] = -(a*f - cc*d) * inv;
    out[6] = C * inv;              out[7] = -(a*h - b*g) * inv;   out[8] =  (a*e - b*d) * inv;
    return true;
}

// Transform::read → map type name + map data. Fill `A` (index→world linear, row
// major) and `T` (world of index origin). Handles the linear maps OpenVDB emits.
void readTransform(Cursor& c, double A[9], double T[3]) {
    std::string mt = c.str();
    for (int i = 0; i < 9; ++i) A[i] = 0;
    T[0] = T[1] = T[2] = 0;
    auto vec3 = [&](double v[3]) { v[0] = c.f64(); v[1] = c.f64(); v[2] = c.f64(); };
    if (mt == "UniformScaleMap" || mt == "ScaleMap") {
        double s[3]; vec3(s);
        for (int i = 0; i < 12; ++i) (void)c.f64();   // voxelSize, inverse, inv², inv2x
        A[0] = s[0]; A[4] = s[1]; A[8] = s[2];
    } else if (mt == "UniformScaleTranslateMap" || mt == "ScaleTranslateMap") {
        double t[3], s[3]; vec3(t); vec3(s);
        for (int i = 0; i < 12; ++i) (void)c.f64();
        A[0] = s[0]; A[4] = s[1]; A[8] = s[2];
        T[0] = t[0]; T[1] = t[1]; T[2] = t[2];
    } else if (mt == "TranslationMap") {
        double t[3]; vec3(t);
        A[0] = A[4] = A[8] = 1.0;
        T[0] = t[0]; T[1] = t[1]; T[2] = t[2];
    } else if (mt == "AffineMap" || mt == "UnitaryMap") {
        double m[16]; for (int i = 0; i < 16; ++i) m[i] = c.f64();
        // OpenVDB is row-vector (world = index·M); the column-vector linear map is Mᵀ.
        A[0] = m[0]; A[1] = m[4]; A[2] = m[8];
        A[3] = m[1]; A[4] = m[5]; A[5] = m[9];
        A[6] = m[2]; A[7] = m[6]; A[8] = m[10];
        T[0] = m[12]; T[1] = m[13]; T[2] = m[14];
    } else {
        fail("unsupported .vdb transform map '" + mt + "'");
    }
}

}  // namespace

bool loadOpenVDBGrid(const std::string& path, VdbGrid& out, std::string& err,
                     const std::string& wantName) {
    // OpenVDB "unique names" append a record-separator (0x1e) + instance index to
    // disambiguate duplicates; strip it to recover the authored grid name, and
    // compare case-insensitively so `temperature`/`Temperature` both match.
    auto baseName = [](std::string s) {
        auto rs = s.find('\x1e'); if (rs != std::string::npos) s.resize(rs);
        return s;
    };
    auto iequals = [](const std::string& a, const std::string& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
        return true;
    };
    // Slurp the whole file (samples are a few MB; guarded against huge inputs).
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) { err = "cannot open '" + path + "'"; return false; }
    std::fseek(fp, 0, SEEK_END);
    long len = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (len <= 0) { std::fclose(fp); err = "empty file: '" + path + "'"; return false; }
    std::vector<uint8_t> buf((size_t)len);
    if (std::fread(buf.data(), 1, (size_t)len, fp) != (size_t)len) {
        std::fclose(fp); err = "short read: '" + path + "'"; return false;
    }
    std::fclose(fp);

    try {
        Cursor c(buf.data(), buf.size());
        int64_t magic = c.i64();
        if ((uint64_t)(magic & 0xFFFFFFFF) != 0x56444220ull) fail("not an OpenVDB file");
        uint32_t fileVer = c.u32();
        (void)c.u32(); (void)c.u32();            // library major/minor
        (void)c.u8();                             // hasGridOffsets (assume 1)
        if (fileVer >= 218) c.take(36);           // BOOST uuid (ASCII)
        // file-level metamap
        uint32_t nMeta = c.u32();
        for (uint32_t i = 0; i < nMeta; ++i) c.skipMeta();
        uint32_t gridCount = c.u32();
        if (gridCount == 0) fail("no grids in file");
        if (fileVer < V_NODE_MASK_COMPRESSION)
            fail("legacy .vdb (pre-5.0) not supported — re-export with a modern OpenVDB");

        // Bake the requested float grid (or the first, when wantName is empty);
        // scan descriptors for one whose type is a float tree (Tree_float_5_4_3
        // [+_HalfFloat]) and, if a name was requested, whose name matches.
        for (uint32_t gi = 0; gi < gridCount; ++gi) {
            std::string gname = baseName(c.str());  // unique name (RS-suffix stripped)
            std::string gtype = c.str();            // grid type
            bool fromHalf = false;
            const std::string half = "_HalfFloat";
            if (gtype.size() > half.size() &&
                gtype.compare(gtype.size() - half.size(), half.size(), half) == 0) {
                fromHalf = true;
                gtype.resize(gtype.size() - half.size());
            }
            if (fileVer >= V_GRID_INSTANCING) (void)c.str();   // instance parent
            int64_t gridPos = c.i64(); (void)c.i64(); int64_t endPos = c.i64();

            // The grid descriptors are NOT contiguous in the stream: OpenVDB's writer
            // lays out {descriptor, grid body} per grid, so the NEXT descriptor begins
            // at THIS grid's endPos, not immediately after these offsets. When we skip a
            // grid (name mismatch or non-float) we must therefore seek the cursor to
            // endPos before reading the following descriptor, or it reads garbage.
            // Name filter: when a specific grid is requested, skip the others.
            if (!wantName.empty() && !iequals(gname, wantName)) {
                if (gi + 1 == gridCount)
                    fail("grid '" + wantName + "' not found in '" + path + "'");
                c.p = (size_t)endPos;
                continue;
            }
            if (gtype != "Tree_float_5_4_3") {
                if (!wantName.empty())
                    fail("grid '" + wantName + "' is not a float grid (type '" + gtype + "')");
                if (gi + 1 == gridCount)
                    fail("no float grid found (last grid type '" + gtype +
                         "'); only Tree_float_5_4_3 is supported");
                c.p = (size_t)endPos;
                continue;   // skip non-float grid, try the next descriptor
            }

            // --- parse the grid body at gridPos --------------------------------
            Cursor g(buf.data(), buf.size(), (size_t)gridPos);
            uint32_t compression = g.u32();       // grid compression (v ≥ 222)
            uint32_t nGMeta = g.u32();
            for (uint32_t i = 0; i < nGMeta; ++i) g.skipMeta();
            double A[9], T[3];
            readTransform(g, A, T);

            int32_t bufferCount = g.i32();        // Tree bufferCount (== 1)
            if (bufferCount != 1) fail("multi-buffer tree not supported");
            float background = g.f32();
            uint32_t numTiles = g.u32();
            uint32_t numChildren = g.u32();

            // Track active-voxel index bounds while filling the dense grid; do a
            // first pass to bound the box, then a second to fill (memory-friendly).
            std::vector<std::unique_ptr<Internal>> roots;

            // Root tiles (constant active/inactive regions at root level).
            struct RootTile { int32_t c[3]; float v; bool active; };
            std::vector<RootTile> tiles;
            tiles.reserve(numTiles);
            for (uint32_t i = 0; i < numTiles; ++i) {
                RootTile t;
                t.c[0] = g.i32(); t.c[1] = g.i32(); t.c[2] = g.i32();
                t.v = g.f32(); t.active = g.u8() != 0;
                tiles.push_back(t);
            }
            for (uint32_t i = 0; i < numChildren; ++i) {
                int32_t cx = g.i32(), cy = g.i32(), cz = g.i32();
                // top internal: log2dim=5, child (Internal<4>) voxel-dim log2 = 4+3 = 7
                roots.push_back(readInternalTopology(g, cx, cy, cz, 5, 7, false,
                                                     fromHalf, compression, background, fileVer));
            }
            for (auto& r : roots)
                readInternalBuffers(g, r.get(), fromHalf, compression, background, fileVer);

            // --- collect active voxels, compute index bbox --------------------
            long long lo[3] = { (long long)1e18,  (long long)1e18,  (long long)1e18 };
            long long hi[3] = { (long long)-1e18, (long long)-1e18, (long long)-1e18 };
            auto grow = [&](long long x, long long y, long long z) {
                if (x < lo[0]) lo[0] = x;  if (x > hi[0]) hi[0] = x;
                if (y < lo[1]) lo[1] = y;  if (y > hi[1]) hi[1] = y;
                if (z < lo[2]) lo[2] = z;  if (z > hi[2]) hi[2] = z;
            };
            // Gather leaves (and active internal tiles) via a walk.
            struct ActiveTile { long long c[3]; long long dim; float v; };
            std::vector<Leaf*> leaves;
            std::vector<ActiveTile> activeTiles;
            std::vector<Internal*> stack;
            for (auto& r : roots) stack.push_back(r.get());
            while (!stack.empty()) {
                Internal* nd = stack.back(); stack.pop_back();
                // active tiles at this internal level (valueMask on, childMask off)
                long long tdim = (long long)1 << nd->childTotal;
                int mk = (1 << nd->log2dim) - 1;
                for (int off = 0; off < nd->numValues; ++off) {
                    if (bitOn(nd->childMask.data(), off)) continue;
                    if (!bitOn(nd->valueMask.data(), off)) continue;
                    float v = nd->values[(size_t)off];
                    if (!(v > 0.0f)) continue;   // fog density ≥ 0
                    int i = (off >> (2 * nd->log2dim)) & mk;
                    int j = (off >> nd->log2dim) & mk;
                    int k = off & mk;
                    long long bx = nd->origin[0] + ((long long)i << nd->childTotal);
                    long long by = nd->origin[1] + ((long long)j << nd->childTotal);
                    long long bz = nd->origin[2] + ((long long)k << nd->childTotal);
                    activeTiles.push_back({ { bx, by, bz }, tdim, v });
                    grow(bx, by, bz); grow(bx + tdim - 1, by + tdim - 1, bz + tdim - 1);
                }
                for (auto& kv : nd->childInternals) stack.push_back(kv.second.get());
                for (auto& kv : nd->childLeaves) {
                    Leaf* lf = kv.second.get();
                    for (int off = 0; off < 512; ++off) {
                        if (!bitOn(lf->valueMask.data(), off)) continue;
                        float v = lf->values[(size_t)off];
                        if (!(v > 0.0f)) continue;
                        long long x = lf->origin[0] + ((off >> 6) & 7);
                        long long y = lf->origin[1] + ((off >> 3) & 7);
                        long long z = lf->origin[2] + (off & 7);
                        grow(x, y, z);
                    }
                    leaves.push_back(lf);
                }
            }
            if (hi[0] < lo[0]) fail("empty grid (no positive active voxels)");

            long long nx = hi[0] - lo[0] + 1, ny = hi[1] - lo[1] + 1, nz = hi[2] - lo[2] + 1;
            unsigned long long voxels = (unsigned long long)nx * ny * nz;
            if (voxels > 512ull * 1024 * 1024) {
                char b[128];
                std::snprintf(b, sizeof(b), "%lldx%lldx%lld = %.1fM", nx, ny, nz, voxels / 1e6);
                fail("grid too large to dense-bake (" + std::string(b) + " voxels)");
            }

            out.nx = (int)nx; out.ny = (int)ny; out.nz = (int)nz;
            out.imin = Vec3((double)lo[0], (double)lo[1], (double)lo[2]);
            out.data.assign((size_t)voxels, floatToHalfBits(0.0f));
            float mx = 0.0f;
            auto put = [&](long long x, long long y, long long z, float v) {
                long long i = x - lo[0], j = y - lo[1], k = z - lo[2];
                if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return;
                out.data[((size_t)k * ny + j) * nx + i] = floatToHalfBits(v);
                if (v > mx) mx = v;
            };
            // active constant tiles first, then leaves (leaves are the fine detail)
            for (auto& t : activeTiles)
                for (long long dz = 0; dz < t.dim; ++dz)
                for (long long dy = 0; dy < t.dim; ++dy)
                for (long long dx = 0; dx < t.dim; ++dx)
                    put(t.c[0] + dx, t.c[1] + dy, t.c[2] + dz, t.v);
            for (Leaf* lf : leaves) {
                for (int off = 0; off < 512; ++off) {
                    if (!bitOn(lf->valueMask.data(), off)) continue;
                    float v = lf->values[(size_t)off];
                    if (v < 0.0f) v = 0.0f;
                    put(lf->origin[0] + ((off >> 6) & 7),
                        lf->origin[1] + ((off >> 3) & 7),
                        lf->origin[2] + (off & 7), v);
                }
            }
            // fp16 round-to-nearest can nudge a stored value above the exact max by
            // up to one half-ULP (~2^-11 rel); bump the majorant so it stays a valid
            // upper bound for delta/ratio tracking.
            out.maxVal = mx * 1.001f;

            // world→index affine + world AABB of the active box.
            if (!invert3x3(A, out.ainv)) fail("degenerate grid transform");
            out.w0 = Vec3(T[0], T[1], T[2]);
            double wlo[3] = { 1e300, 1e300, 1e300 }, whi[3] = { -1e300, -1e300, -1e300 };
            for (int cxi = 0; cxi <= 1; ++cxi)
            for (int cyi = 0; cyi <= 1; ++cyi)
            for (int czi = 0; czi <= 1; ++czi) {
                double ix = (double)(cxi ? hi[0] + 1 : lo[0]);
                double iy = (double)(cyi ? hi[1] + 1 : lo[1]);
                double iz = (double)(czi ? hi[2] + 1 : lo[2]);
                double wx = A[0]*ix + A[1]*iy + A[2]*iz + T[0];
                double wy = A[3]*ix + A[4]*iy + A[5]*iz + T[1];
                double wz = A[6]*ix + A[7]*iy + A[8]*iz + T[2];
                if (wx < wlo[0]) wlo[0] = wx;  if (wx > whi[0]) whi[0] = wx;
                if (wy < wlo[1]) wlo[1] = wy;  if (wy > whi[1]) whi[1] = wy;
                if (wz < wlo[2]) wlo[2] = wz;  if (wz > whi[2]) whi[2] = wz;
            }
            out.wmin = Vec3(wlo[0], wlo[1], wlo[2]);
            out.wmax = Vec3(whi[0], whi[1], whi[2]);
            (void)endPos;
            return true;
        }
        fail("no float grid found in '" + path + "'");
    } catch (const VdbError& e) {
        err = e.msg + " ('" + path + "')";
        return false;
    }
    return false;
}
