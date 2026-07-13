// vdbgrid.cpp — the ONLY translation unit that includes NanoVDB.h.
//
// Reads an uncompressed .nvdb file (as written by NanoVDB's
// io::writeUncompressedGrids, or a raw grid buffer) and bakes its first
// FloatGrid into the dense, NanoVDB-free VdbGrid consumed by the rest of the
// renderer. We deliberately parse the tiny file container by hand (FileHeader /
// FileMetaData are public in the core header) so the renderer only needs to
// vendor the single self-contained NanoVDB.h — no GridHandle/HostBuffer/IO
// headers, no zlib/blosc.
//
// The dense bake covers the grid's active index bounding box on its own integer
// voxel lattice; the world->index affine is recovered by probing indexToWorld at
// four points and inverting, so any affine map (including rotation/shear) is
// handled and the CPU/GPU trilinear samplers stay NanoVDB-agnostic.

#define _CRT_SECURE_NO_WARNINGS   // std::fopen etc. (single-TU loader, plain C stdio)

// NanoVDB is a large third-party template header; silence its warnings locally so
// the renderer's /W4 stays clean without polluting the vendored file.
#if defined(_MSC_VER)
#  pragma warning(push, 0)
#endif
#include "third_party/nanovdb/NanoVDB.h"
#if defined(_MSC_VER)
#  pragma warning(pop)
#endif

#include "vdbgrid.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

namespace {

// Aligned host allocation (NanoVDB grids require NANOVDB_DATA_ALIGNMENT bytes).
void* alignedAlloc(size_t n) {
#if defined(_MSC_VER)
    return _aligned_malloc(n, NANOVDB_DATA_ALIGNMENT);
#else
    size_t rounded = ((n + NANOVDB_DATA_ALIGNMENT - 1) / NANOVDB_DATA_ALIGNMENT) * NANOVDB_DATA_ALIGNMENT;
    return std::aligned_alloc(NANOVDB_DATA_ALIGNMENT, rounded);
#endif
}
void alignedFree(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

// Invert a row-major 3x3 matrix. Returns false if (near-)singular.
bool invert3x3(const double m[9], double out[9]) {
    double a = m[0], b = m[1], c = m[2];
    double d = m[3], e = m[4], f = m[5];
    double g = m[6], h = m[7], i = m[8];
    double A =  (e*i - f*h), B = -(d*i - f*g), C =  (d*h - e*g);
    double det = a*A + b*B + c*C;
    if (det > -1e-30 && det < 1e-30) return false;
    double inv = 1.0 / det;
    out[0] = A * inv;
    out[1] = -(b*i - c*h) * inv;
    out[2] =  (b*f - c*e) * inv;
    out[3] = B * inv;
    out[4] =  (a*i - c*g) * inv;
    out[5] = -(a*f - c*d) * inv;
    out[6] = C * inv;
    out[7] = -(a*h - b*g) * inv;
    out[8] =  (a*e - b*d) * inv;
    return true;
}

} // namespace

bool loadVdbGrid(const std::string& path, VdbGrid& out, std::string& err) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) { err = "cannot open '" + path + "'"; return false; }

    // Disambiguate the two accepted layouts the same way NanoVDB's own reader does:
    // read the first 40 bytes as GridData; if that looks like a valid grid the file
    // is a RAW grid buffer, otherwise it is a FILE CONTAINER (FileHeader, then
    // {FileMetaData, name, grid}...). Note the file-container magic is
    // NANOVDB_MAGIC_NUMBER in default builds — same first 8 bytes as a grid — so the
    // discriminator must be GridData::isValid() (which also checks the version
    // field), not the magic alone.
    void* gridMem = nullptr;
    uint64_t gridSize = 0;

    nanovdb::GridData probe{};
    if (std::fread(&probe, 1, 40, fp) != 40) {
        std::fclose(fp); err = "file too small: '" + path + "'"; return false;
    }
    std::fseek(fp, 0, SEEK_SET);

    const bool isRaw = (probe.mMagic == NANOVDB_MAGIC_GRID) || probe.isValid();
    if (isRaw) {
        // Raw grid buffer: GridData starts at offset 0; mGridSize is its length.
        gridSize = probe.mGridSize;
        gridMem = alignedAlloc((size_t)gridSize);
        if (!gridMem || std::fread(gridMem, 1, (size_t)gridSize, fp) != gridSize) {
            if (gridMem) alignedFree(gridMem);
            std::fclose(fp); err = "truncated raw grid: '" + path + "'"; return false;
        }
    } else {
        // File container: FileHeader, then per grid {FileMetaData, name, grid bytes}.
        nanovdb::io::FileHeader head{};
        if (std::fread(&head, 1, sizeof(head), fp) != sizeof(head) || !head.isValid()) {
            std::fclose(fp); err = "not a NanoVDB file or raw grid: '" + path + "'"; return false;
        }
        if (head.codec != nanovdb::io::Codec::NONE) {
            std::fclose(fp);
            err = "compressed .nvdb not supported (re-export uncompressed): '" + path + "'";
            return false;
        }
        if (head.gridCount == 0) {
            std::fclose(fp); err = "no grids in '" + path + "'"; return false;
        }
        nanovdb::io::FileMetaData meta{};
        if (std::fread(&meta, 1, sizeof(meta), fp) != sizeof(meta)) {
            std::fclose(fp); err = "truncated grid metadata: '" + path + "'"; return false;
        }
        if (meta.nameSize) std::fseek(fp, (long)meta.nameSize, SEEK_CUR);
        gridSize = meta.gridSize;
        gridMem = alignedAlloc((size_t)gridSize);
        if (!gridMem || std::fread(gridMem, 1, (size_t)gridSize, fp) != gridSize) {
            if (gridMem) alignedFree(gridMem);
            std::fclose(fp); err = "truncated grid data: '" + path + "'"; return false;
        }
    }
    std::fclose(fp);

    auto* base = reinterpret_cast<const nanovdb::GridData*>(gridMem);
    if (base->mGridType != nanovdb::GridType::Float) {
        alignedFree(gridMem);
        err = "only float grids supported (grid is " +
              std::string(nanovdb::toStr(base->mGridType)) + "): '" + path + "'";
        return false;
    }
    auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(gridMem);

    // --- Recover the world->index affine by probing indexToWorld ---------------
    using V3 = nanovdb::Vec3d;
    V3 w000 = grid->indexToWorld(V3(0, 0, 0));
    V3 wx   = grid->indexToWorld(V3(1, 0, 0));
    V3 wy   = grid->indexToWorld(V3(0, 1, 0));
    V3 wz   = grid->indexToWorld(V3(0, 0, 1));
    // Columns of the index->world linear map A (world = A*index + w000).
    double A[9] = {
        wx[0]-w000[0], wy[0]-w000[0], wz[0]-w000[0],
        wx[1]-w000[1], wy[1]-w000[1], wz[1]-w000[1],
        wx[2]-w000[2], wy[2]-w000[2], wz[2]-w000[2],
    };
    if (!invert3x3(A, out.ainv)) {
        alignedFree(gridMem);
        err = "degenerate grid transform: '" + path + "'";
        return false;
    }
    out.w0 = Vec3(w000[0], w000[1], w000[2]);

    // --- Dense bake over the active index bounding box -------------------------
    auto ib = grid->tree().bbox();           // inclusive CoordBBox in index space
    nanovdb::Coord lo = ib.min(), hi = ib.max();
    int nx = hi[0] - lo[0] + 1;
    int ny = hi[1] - lo[1] + 1;
    int nz = hi[2] - lo[2] + 1;
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        alignedFree(gridMem);
        err = "empty grid (no active voxels): '" + path + "'";
        return false;
    }
    // Guard against pathological memory blow-ups from a huge sparse volume.
    const uint64_t voxels = (uint64_t)nx * ny * nz;
    if (voxels > 512ull * 1024 * 1024) {
        alignedFree(gridMem);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%dx%dx%d = %.1fM", nx, ny, nz, voxels / 1e6);
        err = "grid too large to dense-bake (" + std::string(buf) +
              " voxels): '" + path + "'";
        return false;
    }

    out.nx = nx; out.ny = ny; out.nz = nz;
    out.imin = Vec3(lo[0], lo[1], lo[2]);
    out.data.assign((size_t)voxels, 0.0f);

    auto acc = grid->getAccessor();
    float mx = 0.0f;
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        float v = acc.getValue(nanovdb::Coord(lo[0] + i, lo[1] + j, lo[2] + k));
        if (v < 0.0f) v = 0.0f;             // density must be >= 0
        out.data[(size_t(k) * ny + j) * nx + i] = v;
        if (v > mx) mx = v;
    }
    out.maxVal = mx;

    auto wb = grid->worldBBox();
    out.wmin = Vec3(wb.min()[0], wb.min()[1], wb.min()[2]);
    out.wmax = Vec3(wb.max()[0], wb.max()[1], wb.max()[2]);

    alignedFree(gridMem);
    return true;
}
