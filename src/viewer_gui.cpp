#include "viewer_gui.h"

#ifndef _WIN32
// -------- Non-Windows stub: the native viewer needs Win32 + D3D11 --------------
#include <cstdio>
int runGroomGui(const std::string&, bool) {
    std::fprintf(stderr, "error: -groom is only available on Windows builds.\n");
    return 1;
}
int groomSectionsReport(const std::string&) {
    std::fprintf(stderr, "error: -groom-sections is only available on Windows builds.\n");
    return 1;
}
int runViewerGui(const std::string&, const std::string&, bool, bool, int) {
    std::fprintf(stderr, "error: -viewer is only available on Windows builds.\n");
    return 1;
}

#else
// =============================== Win32 + D3D11 =================================
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <tchar.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "imgui.h"
#include "implot.h"                // ImPlot: F3 strip charts
#include "imnodes.h"               // imnodes: F5 modulator-DAG panel
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx11.h"
#include "third_party/json.h"      // minijson: the vendored JSON parser
#include "loomlink.h"              // the shared `python -m loom.<server>` child link
#include "parallel.h"              // ft::stopRequested — cooperative `ftrace -stop <pid>`
#include <map>
#include <unordered_map>
#include <functional>
#include <thread>
#include <mutex>                   // F4 item 2: the live re-introspection job queue
#include <condition_variable>
#include <cstring>
#include <cstdlib>                 // strtoul: parsing the pid out of a scratch dir name
#include <cwctype>
#include <memory>                  // shared_ptr: the live channel's in-memory payload
#include <array>
#include <filesystem>
#include "assetbytes.h"            // asset bytes handed to the loader instead of paths

// Bridge to ftrace's own scene loader + GPU field raymarcher (F7 primary path).
// The viewer IS the ftrace binary, so it can parse loom's emitted `.ftsl` with the
// exact loader main() uses and render the real isosurface field in-process via
// renderIsoPreviewCuda — the `-raster-gpu` preview kernel that sphere-traces the
// field's bytecode with NO tessellation (the static marching-cubes mesh in the
// sidecar is only a fallback). These headers are plain-C++ (main.cpp includes them
// under MSVC too); the raymarch itself is guarded by HAVE_CUDA below.
#include "ftsl.h"
#include "groom.h"               // the hair-authoring tool's curve model (Phase 2)
#include "render_cuda.h"

#include <d3dcompiler.h>           // the mesh pane's z-buffered shaders (runtime-compiled)

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

// ImGui's Win32 backend provides this handler; declare it (the header guards it
// behind a macro we don't want to define project-wide).
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg,
                                                             WPARAM wParam, LPARAM lParam);

// --------------------------------------------------------------------------
// D3D11 device / swap-chain plumbing (adapted from the ImGui dx11 example)
// --------------------------------------------------------------------------
static ID3D11Device*           g_pd3dDevice        = nullptr;
static ID3D11DeviceContext*    g_pd3dDeviceContext = nullptr;
static IDXGISwapChain*         g_pSwapChain        = nullptr;
static ID3D11RenderTargetView* g_mainRTV           = nullptr;

static void CreateRenderTarget() {
    ID3D11Texture2D* back = nullptr;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&back));
    if (back) {
        g_pd3dDevice->CreateRenderTargetView(back, nullptr, &g_mainRTV);
        back->Release();
    }
}
static void CleanupRenderTarget() {
    if (g_mainRTV) { g_mainRTV->Release(); g_mainRTV = nullptr; }
}

static bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount        = 2;
    sd.BufferDesc.Width   = 0;
    sd.BufferDesc.Height  = 0;
    sd.BufferDesc.Format  = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator   = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags              = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow       = hWnd;
    sd.SampleDesc.Count   = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed           = TRUE;
    sd.SwapEffect         = DXGI_SWAP_EFFECT_DISCARD;

    UINT flags = 0;
    D3D_FEATURE_LEVEL fl;
    const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };
    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, levels, 2,
        D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &fl, &g_pd3dDeviceContext);
    if (hr == DXGI_ERROR_UNSUPPORTED)  // fall back to WARP if no hardware device
        hr = D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_WARP, nullptr, flags, levels, 2,
            D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &fl, &g_pd3dDeviceContext);
    if (FAILED(hr)) return false;
    CreateRenderTarget();
    return true;
}
static void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain)        { g_pSwapChain->Release();        g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice)        { g_pd3dDevice->Release();        g_pd3dDevice = nullptr; }
}

static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;
    switch (msg) {
    case WM_SIZE:
        if (g_pd3dDevice && wParam != SIZE_MINIMIZED) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam),
                                        DXGI_FORMAT_UNKNOWN, 0);
            CreateRenderTarget();
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) return 0;  // disable ALT app menu
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}

// --------------------------------------------------------------------------
// Sidecar model (a thin view over the parsed minijson tree)
// --------------------------------------------------------------------------
namespace {

using loomlink::utf8ToWide;   // shared with the child-process link (loomlink.h)

// Render a JSON scalar (string OR number) as a display string. loom emits dataset
// ids as integer node ids, so a plain asString() would fall back to the default.
std::string scalarStr(const minijson::Value* v, const char* dflt = "-") {
    if (!v) return dflt;
    if (v->isString()) return v->str;
    if (v->isNumber()) {
        double n = v->num;
        char buf[32];
        if (n == (double)(long long)n) std::snprintf(buf, sizeof buf, "%lld", (long long)n);
        else                           std::snprintf(buf, sizeof buf, "%g", n);
        return buf;
    }
    if (v->type == minijson::Value::Bool) return v->b ? "true" : "false";
    return dflt;
}

bool readFile(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    out = ss.str();
    return true;
}

// A parsed sidecar plus small conveniences. The full minijson tree is kept so
// panels can read whatever fields they need without a rigid struct mirror.
struct Sidecar {
    minijson::Value root;
    std::string     err;
    bool            ok = false;

    bool load(const std::string& path) {
        std::string text;
        if (!readFile(path, text)) { err = "cannot open " + path; return false; }
        minijson::Value v;
        if (!minijson::parse(text, v, err)) return false;
        return adopt(std::move(v));
    }
    // Take over an already-parsed tree. The live channel gets its sidecar as an
    // object *inside* loom's ack, so by the time it reaches here the parse has
    // already happened (on the bridge's worker thread); re-serialising it only to
    // re-parse it would be pure waste. Moves — this tree is ~900 KB on a real scene.
    bool adopt(minijson::Value v) {
        root = std::move(v);
        if (!root.isObject()) { err = "sidecar root is not an object"; return false; }
        ok = true;
        return true;
    }
    const minijson::Value* arr(const char* key) const {
        const minijson::Value* v = root.find(key);
        return (v && v->isArray()) ? v : nullptr;
    }
    // Absolute path to the scene's `.ftsl` source loom emitted next to the sidecar
    // (F7). Empty when the sidecar predates the source key or loom skipped it — the
    // viewer then shows only the static sidecar geometry (no live raymarch).
    std::string source() const {
        const minijson::Value* v = root.find("source");
        return (v && v->isString()) ? v->str : std::string();
    }
    // Absolute path of the loom file the `build()` came from (F4 item 2). This is the
    // sidecar's provenance, and it is what lets `-viewer <sidecar>` reopen the LIVE
    // re-introspection channel without being told the scene file a second time.
    std::string buildFile() const {
        const minijson::Value* v = root.find("build");
        return (v && v->isString()) ? v->str : std::string();
    }
};

// A tacked-on channel (TrackedPath track) sampled along the curve parameter — the
// source for F3's per-channel strip charts. Stored flat, `dim` scalars per sample.
struct ChannelGeom {
    std::string        name;
    int                dim    = 1;
    bool               scalar = true;
    std::vector<float> samp;         // flat, dim scalars per sample
    int                n      = 0;   // number of samples (matches the polyline)
};

// A curve dataset kept at full N-D. Points are stored flat, `dim` scalars each,
// so the viewer can pick any 3 of N dims to display (§F2's 3-of-N projection).
struct CurveGeom {
    std::string        id;
    bool               closed = false;
    int                dim    = 3;   // dimensionality of each stored point
    std::vector<float> poly;         // flat, dim scalars per sampled point
    int                polyN  = 0;   // number of polyline points
    std::vector<float> ctrl;         // flat, dim scalars per control point
    int                ctrlN  = 0;   // number of control points
    std::vector<ChannelGeom> channels;  // tracked-path tacked-on channels (F3)
};

// Pull a flat N-D array out of a JSON array-of-arrays. Every row is padded/kept to
// `dim` scalars (dim = the widest row seen). Returns the row count.
static int flattenPtsND(const minijson::Value* a, std::vector<float>& out, int dim) {
    out.clear();
    if (!a || !a->isArray()) return 0;
    for (const auto& p : a->arr) {
        for (int k = 0; k < dim; ++k) {
            float v = 0.0f;
            if (p.isArray() && k < (int)p.arr.size()) v = (float)p.arr[k].asNumber(0.0);
            out.push_back(v);
        }
    }
    return (int)a->arr.size();
}

static int rowWidth(const minijson::Value* a) {
    int w = 0;
    if (a && a->isArray())
        for (const auto& p : a->arr)
            if (p.isArray()) w = std::max(w, (int)p.arr.size());
    return w;
}

// Collect every dataset that carries polyline geometry (paths / tracked paths).
static std::vector<CurveGeom> collectCurves(const Sidecar& sc) {
    std::vector<CurveGeom> curves;
    const minijson::Value* ds = sc.arr("datasets");
    if (!ds) return curves;
    for (const auto& d : ds->arr) {
        const minijson::Value* poly = d.find("polyline");
        if (!poly || !poly->isArray()) continue;
        CurveGeom g;
        g.id     = scalarStr(d.find("id"), "");
        g.closed = d.find("closed") ? d.find("closed")->asBool(false) : false;
        int dim  = std::max({ 3, rowWidth(poly), rowWidth(d.find("control_points")),
                              d.intAt("dim", 0) });
        g.dim    = dim;
        g.polyN  = flattenPtsND(poly, g.poly, dim);
        g.ctrlN  = flattenPtsND(d.find("control_points"), g.ctrl, dim);
        // tracked-path channels (F3): each track sampled along the same parameter
        const minijson::Value* chans = d.find("channels");
        if (chans && chans->isArray()) {
            for (const auto& ch : chans->arr) {
                ChannelGeom cg;
                cg.name   = scalarStr(ch.find("name"), "");
                cg.dim    = std::max(1, ch.intAt("dim", 1));
                cg.scalar = ch.find("scalar") ? ch.find("scalar")->asBool(true) : true;
                cg.n      = flattenPtsND(ch.find("samples"), cg.samp, cg.dim);
                g.channels.push_back(std::move(cg));
            }
        }
        curves.push_back(std::move(g));
    }
    return curves;
}

// --------------------------------------------------------------------------
// F6 — scatter + grid field datasets. Both collapse to a common form: a list of
// sample points (each an N-D position + a channel-vector value). A grid also keeps
// its per-axis coordinates + shape so extra dims beyond the 3 shown can be sliced.
// --------------------------------------------------------------------------
struct FieldPoint {
    std::vector<float> pos;    // dim scalars
    std::vector<float> val;    // valueDim channel scalars
    std::vector<int>   idx;    // per-axis lattice index (grid only; empty for scatter)
};
struct FieldGeom {
    std::string              id;
    std::string              kind;       // "scatter" | "grid"
    int                      dim = 0;    // position dimensionality
    int                      valueDim = 1;
    std::vector<std::string> channels;   // names (may be empty)
    std::vector<FieldPoint>  points;
    std::vector<int>         shape;      // grid: samples per axis (empty for scatter)
    bool                     isGrid = false;
};

static void readFlatVecs(const minijson::Value* a, std::vector<std::vector<float>>& out) {
    out.clear();
    if (!a || !a->isArray()) return;
    for (const auto& row : a->arr) {
        std::vector<float> v;
        if (row.isArray()) for (const auto& x : row.arr) v.push_back((float)x.asNumber(0.0));
        else               v.push_back((float)row.asNumber(0.0));
        out.push_back(std::move(v));
    }
}

static std::vector<FieldGeom> collectFields(const Sidecar& sc) {
    std::vector<FieldGeom> fields;
    const minijson::Value* ds = sc.arr("datasets");
    if (!ds) return fields;
    for (const auto& d : ds->arr) {
        std::string kind = scalarStr(d.find("kind"), "");
        bool isGrid = (kind == "grid"), isScat = (kind == "scatter");
        if (!isGrid && !isScat) continue;
        FieldGeom f;
        f.id       = scalarStr(d.find("id"), "");
        f.kind     = kind;
        f.isGrid   = isGrid;
        f.valueDim = std::max(1, d.intAt("value_dim", 1));
        if (const minijson::Value* ch = d.find("channels"); ch && ch->isArray())
            for (const auto& c : ch->arr) f.channels.push_back(c.asString(""));

        std::vector<std::vector<float>> vals;
        readFlatVecs(d.find("values"), vals);

        if (isScat) {
            std::vector<std::vector<float>> pts;
            readFlatVecs(d.find("points"), pts);
            f.dim = pts.empty() ? 0 : (int)pts[0].size();
            for (size_t i = 0; i < pts.size(); ++i) {
                FieldPoint fp;
                fp.pos = pts[i];
                if (i < vals.size()) fp.val = vals[i];
                f.points.push_back(std::move(fp));
            }
        } else {  // grid: reconstruct node positions from axes + shape (C order)
            std::vector<std::vector<float>> axes;
            readFlatVecs(d.find("axes"), axes);
            if (const minijson::Value* sh = d.find("shape"); sh && sh->isArray())
                for (const auto& s : sh->arr) f.shape.push_back(std::max(1, s.asInt(1)));
            int ndim = (int)f.shape.size();
            f.dim = ndim;
            std::vector<int> strides(ndim, 1);
            for (int a = ndim - 2; a >= 0; --a) strides[a] = strides[a + 1] * f.shape[a + 1];
            for (size_t i = 0; i < vals.size(); ++i) {
                FieldPoint fp;
                fp.idx.resize(ndim);
                fp.pos.resize(ndim);
                for (int a = 0; a < ndim; ++a) {
                    int k = (strides[a] ? ((int)i / strides[a]) % f.shape[a] : 0);
                    fp.idx[a] = k;
                    fp.pos[a] = (a < (int)axes.size() && k < (int)axes[a].size())
                                    ? axes[a][k] : (float)k;
                }
                fp.val = vals[i];
                f.points.push_back(std::move(fp));
            }
        }
        fields.push_back(std::move(f));
    }
    return fields;
}

// --------------------------------------------------------------------------
// F4 — tessellated geometry. A `swept_mesh` object carries a `mesh` key
// (vertices / faces / uvs) baked by loom; a `strand` object carries a fiber
// centreline + per-sample radius and is tubed here (see `strandToMesh`). Either
// way the viewer draws a shaded, depth-buffered triangle surface in a 3-D orbit
// pane.
// --------------------------------------------------------------------------
struct MeshGeom {
    std::string        id, name, material;
    std::vector<float> verts;   // flat xyz (3 per vertex)
    int                nverts = 0;
    std::vector<int>   faces;   // flat index triples
    int                nfaces = 0;
    std::vector<float> uvs;     // flat uv (2 per vertex), may be empty
    std::vector<float> normals; // flat xyz (3 per vertex), may be empty (authored)
    // Crease angle in DEGREES the object asked for, mirroring `mesh { smooth <deg> }`
    // in the emitted ftsl; 0 == flat. Authored `normals` win over it, exactly as
    // OBJ `vn` wins over `smooth` in src/mesh.h. See buildMeshPaneVerts().
    double             smoothDeg = 0.0;
};

// ftrace's own default crease angle (src/ftsl.h: a bare `smooth` means 40 deg).
// Used for geometry the pane tubes itself, which has no authored `smooth`.
static const double MESH_PANE_CREASE_DEG = 40.0;

// --------------------------------------------------------------------------
// Per-vertex shading normals for the mesh pane, computed the way ftrace's own
// loader computes them (src/mesh.h) so the preview creases where the render will.
//
// The pane used to have no normals at all: the pixel shader rebuilt a FACE normal
// from screen-space derivatives of the view position. That can only ever show
// facets — and worse, ddx/ddy are evaluated over 2x2 pixel quads, so every quad
// straddling a triangle edge differentiates across two different triangles and
// emits a garbage normal, painting a one-pixel band of wrong shading along every
// single edge in the mesh. That is what read as "jagged seams" on loom's sweeps.
//
// Mirrors src/mesh.h exactly, and for the same reasons:
//   * weld by POSITION first (eps = 1e-6 * bounding diagonal), because a swept
//     surface's seam ring is two coincident vertices that must share a normal;
//   * accumulate face normals weighted by the CORNER'S INTERIOR ANGLE (Thurmer &
//     Wuthrich), which is tessellation-independent where area weighting is not;
//   * only merge a neighbour whose normal is within the crease angle, so a cap rim
//     or a hard fold stays sharp instead of being smeared round;
//   * fall back to the face normal when a corner has no qualifying neighbours.
// The one thing it must do that mesh.h does not is re-index: mesh.h stores normals
// per triangle CORNER, while a vertex buffer is indexed, so corners of the same
// vertex that ended up with different normals (either side of a crease) are split
// into separate vertices here.
// --------------------------------------------------------------------------
struct MeshPaneVert { float x, y, z, u, v, nx, ny, nz; };

static void buildMeshPaneVerts(const MeshGeom& m,
                               std::vector<MeshPaneVert>& outV,
                               std::vector<uint32_t>&     outI) {
    outV.clear();
    outI.clear();
    const bool hasUv = (int)m.uvs.size() >= 2 * m.nverts;
    auto uvOf = [&](int i, int c) { return hasUv ? m.uvs[(size_t)i * 2 + c] : 0.0f; };
    auto pos  = [&](int i, int c) { return (double)m.verts[(size_t)i * 3 + c]; };

    // Valid triangles only — a malformed face is skipped, exactly as before.
    std::vector<int> tri;
    tri.reserve((size_t)m.nfaces * 3);
    for (int f = 0; f < m.nfaces; ++f) {
        int a = m.faces[(size_t)f * 3 + 0], b = m.faces[(size_t)f * 3 + 1],
            c = m.faces[(size_t)f * 3 + 2];
        if (a < 0 || b < 0 || c < 0 || a >= m.nverts || b >= m.nverts || c >= m.nverts)
            continue;
        tri.push_back(a); tri.push_back(b); tri.push_back(c);
    }
    const int nt = (int)tri.size() / 3;

    // Authored normals win, exactly as OBJ `vn` wins over `smooth` in mesh.h; so
    // does "no smoothing asked for", which keeps the flat look but now via a real
    // per-vertex normal rather than a derivative guess.
    if (!m.normals.empty() || m.smoothDeg <= 0.0) {
        const bool authored = !m.normals.empty();
        if (authored) {
            outV.reserve((size_t)m.nverts);
            for (int i = 0; i < m.nverts; ++i) {
                double nx = m.normals[(size_t)i * 3 + 0], ny = m.normals[(size_t)i * 3 + 1],
                       nz = m.normals[(size_t)i * 3 + 2];
                double l = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (l > 1e-12) { nx /= l; ny /= l; nz /= l; } else { nx = ny = 0.0; nz = 1.0; }
                outV.push_back({ m.verts[(size_t)i * 3 + 0], m.verts[(size_t)i * 3 + 1],
                                 m.verts[(size_t)i * 3 + 2], uvOf(i, 0), uvOf(i, 1),
                                 (float)nx, (float)ny, (float)nz });
            }
            outI.assign(tri.begin(), tri.end());
            return;
        }
        // Flat: one vertex per corner carrying its face normal. Duplicating is the
        // point — a shared vertex cannot hold two faces' normals.
        outV.reserve((size_t)nt * 3);
        outI.reserve((size_t)nt * 3);
        for (int f = 0; f < nt; ++f) {
            const int a = tri[(size_t)f * 3 + 0], b = tri[(size_t)f * 3 + 1], c = tri[(size_t)f * 3 + 2];
            double e0[3], e1[3], n[3];
            for (int k = 0; k < 3; ++k) { e0[k] = pos(b, k) - pos(a, k); e1[k] = pos(c, k) - pos(a, k); }
            n[0] = e0[1] * e1[2] - e0[2] * e1[1];
            n[1] = e0[2] * e1[0] - e0[0] * e1[2];
            n[2] = e0[0] * e1[1] - e0[1] * e1[0];
            double l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (l > 1e-20) { n[0] /= l; n[1] /= l; n[2] /= l; } else { n[0] = n[1] = 0.0; n[2] = 1.0; }
            const int ids[3] = { a, b, c };
            for (int cc = 0; cc < 3; ++cc) {
                int i = ids[cc];
                outI.push_back((uint32_t)outV.size());
                outV.push_back({ m.verts[(size_t)i * 3 + 0], m.verts[(size_t)i * 3 + 1],
                                 m.verts[(size_t)i * 3 + 2], uvOf(i, 0), uvOf(i, 1),
                                 (float)n[0], (float)n[1], (float)n[2] });
            }
        }
        return;
    }

    // ---- crease-limited angle-weighted smoothing (the mesh.h algorithm) ----
    std::vector<double> fn((size_t)nt * 3, 0.0);   // face normals
    std::vector<double> ang((size_t)nt * 3, 0.0);  // per-corner interior angle
    for (int f = 0; f < nt; ++f) {
        const int idx[3] = { tri[(size_t)f * 3 + 0], tri[(size_t)f * 3 + 1], tri[(size_t)f * 3 + 2] };
        double p[3][3];
        for (int c = 0; c < 3; ++c) for (int k = 0; k < 3; ++k) p[c][k] = pos(idx[c], k);
        double e0[3], e1[3], n[3];
        for (int k = 0; k < 3; ++k) { e0[k] = p[1][k] - p[0][k]; e1[k] = p[2][k] - p[0][k]; }
        n[0] = e0[1] * e1[2] - e0[2] * e1[1];
        n[1] = e0[2] * e1[0] - e0[0] * e1[2];
        n[2] = e0[0] * e1[1] - e0[1] * e1[0];
        double l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (l > 1e-20) { n[0] /= l; n[1] /= l; n[2] /= l; } else { n[0] = n[1] = 0.0; n[2] = 1.0; }
        for (int k = 0; k < 3; ++k) fn[(size_t)f * 3 + k] = n[k];
        for (int c = 0; c < 3; ++c) {
            double a[3], b[3];
            for (int k = 0; k < 3; ++k) {
                a[k] = p[(c + 1) % 3][k] - p[c][k];
                b[k] = p[(c + 2) % 3][k] - p[c][k];
            }
            double la = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
            double lb = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
            double t = 0.0;
            if (la > 1e-20 && lb > 1e-20) {
                double d = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (la * lb);
                t = std::acos(d < -1.0 ? -1.0 : (d > 1.0 ? 1.0 : d));
            }
            ang[(size_t)f * 3 + c] = t;
        }
    }

    // Weld by quantized position: a sweep's seam is two coincident vertices that
    // MUST share a normal, and an OBJ/sidecar can duplicate a position freely.
    double lo[3] = { 1e300, 1e300, 1e300 }, hi[3] = { -1e300, -1e300, -1e300 };
    for (int i = 0; i < m.nverts; ++i)
        for (int k = 0; k < 3; ++k) {
            double v = pos(i, k);
            lo[k] = std::min(lo[k], v); hi[k] = std::max(hi[k], v);
        }
    double dsq = 0.0;
    for (int k = 0; k < 3; ++k) { double d = hi[k] - lo[k]; if (d > 0.0) dsq += d * d; }
    const double eps  = std::max(std::sqrt(dsq) * 1e-6, 1e-12);
    const double invE = 1.0 / eps;

    struct QKey { long long a, b, c; bool operator==(const QKey& o) const { return a == o.a && b == o.b && c == o.c; } };
    struct QHash {
        size_t operator()(const QKey& k) const {
            unsigned long long h = 1469598103934665603ull;
            auto mix = [&](long long x) {
                unsigned long long u = (unsigned long long)x;
                for (int i = 0; i < 8; ++i) { h ^= (u & 0xff); h *= 1099511628211ull; u >>= 8; }
            };
            mix(k.a); mix(k.b); mix(k.c);
            return (size_t)h;
        }
    };
    std::unordered_map<QKey, int, QHash> weld;
    weld.reserve((size_t)m.nverts * 2);
    std::vector<int> rep((size_t)m.nverts, 0);
    int nw = 0;
    for (int i = 0; i < m.nverts; ++i) {
        QKey k{ (long long)std::llround(pos(i, 0) * invE),
                (long long)std::llround(pos(i, 1) * invE),
                (long long)std::llround(pos(i, 2) * invE) };
        auto it = weld.find(k);
        if (it == weld.end()) { weld.emplace(k, nw); rep[i] = nw++; }
        else                  rep[i] = it->second;
    }

    // CSR adjacency: welded vertex -> the triangle corners that touch it.
    std::vector<int> voff((size_t)nw + 1, 0), vcorner((size_t)nt * 3, 0);
    for (int c = 0; c < nt * 3; ++c) ++voff[(size_t)rep[tri[(size_t)c]] + 1];
    for (int i = 0; i < nw; ++i) voff[(size_t)i + 1] += voff[(size_t)i];
    {
        std::vector<int> cur(voff.begin(), voff.end() - 1);
        for (int c = 0; c < nt * 3; ++c) vcorner[(size_t)cur[(size_t)rep[tri[(size_t)c]]]++] = c;
    }

    const double cosThresh = std::cos(m.smoothDeg * 3.14159265358979323846 / 180.0);

    // Split corners whose smoothed normals differ (either side of a crease) into
    // separate vertices; corners that agree collapse back onto one.
    struct SKey { int v; long long a, b, c; bool operator==(const SKey& o) const { return v == o.v && a == o.a && b == o.b && c == o.c; } };
    struct SHash {
        size_t operator()(const SKey& k) const {
            unsigned long long h = 1469598103934665603ull;
            auto mix = [&](long long x) {
                unsigned long long u = (unsigned long long)x;
                for (int i = 0; i < 8; ++i) { h ^= (u & 0xff); h *= 1099511628211ull; u >>= 8; }
            };
            mix(k.v); mix(k.a); mix(k.b); mix(k.c);
            return (size_t)h;
        }
    };
    std::unordered_map<SKey, uint32_t, SHash> emitted;
    emitted.reserve((size_t)m.nverts * 2);
    outV.reserve((size_t)m.nverts);
    outI.reserve((size_t)nt * 3);

    for (int c = 0; c < nt * 3; ++c) {
        const int f = c / 3;
        const int i = tri[(size_t)c];
        const double* fni = &fn[(size_t)f * 3];
        double s[3] = { 0.0, 0.0, 0.0 };
        const int w = rep[i];
        for (int a = voff[(size_t)w]; a < voff[(size_t)w + 1]; ++a) {
            const int cid = vcorner[(size_t)a];
            const double* fnj = &fn[(size_t)(cid / 3) * 3];
            if (fni[0] * fnj[0] + fni[1] * fnj[1] + fni[2] * fnj[2] < cosThresh) continue;
            const double wgt = ang[(size_t)cid];
            for (int k = 0; k < 3; ++k) s[k] += fnj[k] * wgt;
        }
        double l = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
        if (l > 1e-12) { for (int k = 0; k < 3; ++k) s[k] /= l; }
        else           { for (int k = 0; k < 3; ++k) s[k] = fni[k]; }

        SKey key{ i, (long long)std::llround(s[0] * 10000.0),
                     (long long)std::llround(s[1] * 10000.0),
                     (long long)std::llround(s[2] * 10000.0) };
        auto it = emitted.find(key);
        if (it != emitted.end()) { outI.push_back(it->second); continue; }
        uint32_t at = (uint32_t)outV.size();
        outV.push_back({ m.verts[(size_t)i * 3 + 0], m.verts[(size_t)i * 3 + 1],
                         m.verts[(size_t)i * 3 + 2], uvOf(i, 0), uvOf(i, 1),
                         (float)s[0], (float)s[1], (float)s[2] });
        emitted.emplace(key, at);
        outI.push_back(at);
    }
}

// A loom `Strand` ships NO triangles: it emits ftrace's native `curve` primitive,
// which the renderer flattens into a watertight chain of round cones itself. This
// pane is a triangle rasteriser, so the fiber is tubed *here*, from the sidecar's
// spine samples + per-sample radius: one ring of STRAND_SIDES vertices per sample,
// swept along a rotation-minimising frame so the tube doesn't corkscrew round a
// bend. Preview geometry only — ftrace still renders the analytic cones, never these.
static const int STRAND_SIDES = 10;

static bool strandToMesh(const minijson::Value& s, MeshGeom& g) {
    std::vector<std::vector<float>> pv, rv;
    readFlatVecs(s.find("points"), pv);
    readFlatVecs(s.find("radii"), rv);
    const int n = (int)pv.size();
    if (n < 2) return false;
    const minijson::Value* cl = s.find("closed");
    const bool closed = cl && cl->asBool(false);

    std::vector<Vec3>   P(n);
    std::vector<double> R(n, 0.0);
    for (int i = 0; i < n; ++i) {
        const std::vector<float>& v = pv[i];
        P[i] = Vec3(v.size() > 0 ? v[0] : 0.0f, v.size() > 1 ? v[1] : 0.0f,
                    v.size() > 2 ? v[2] : 0.0f);
        R[i] = (i < (int)rv.size() && !rv[i].empty()) ? rv[i][0] : 0.0;
    }
    // Per-sample tangent: a central difference, wrapped on a closed fiber and
    // one-sided at an open fiber's two ends.
    std::vector<Vec3> T(n);
    for (int i = 0; i < n; ++i) {
        Vec3 d;
        if (closed)           d = P[(i + 1) % n] - P[(i + n - 1) % n];
        else if (i == 0)      d = P[1] - P[0];
        else if (i == n - 1)  d = P[n - 1] - P[n - 2];
        else                  d = P[i + 1] - P[i - 1];
        double L = length(d);
        T[i] = (L > 1e-12) ? d / L : Vec3(0, 0, 1);
    }
    // Parallel transport: carry the reference vector forward by the same rotation
    // that takes T[i-1] to T[i]. A fixed reference (say world up) would make the
    // ring shear wherever the fiber turns; this keeps consecutive rings aligned.
    auto transport = [](const Vec3& u, const Vec3& t0, const Vec3& t1) {
        Vec3   ax = cross(t0, t1);
        double sn = length(ax), cs = dot(t0, t1);
        Vec3   r  = u;
        if (sn > 1e-12) {                                   // Rodrigues about t0 x t1
            ax = ax / sn;
            double a = std::atan2(sn, cs), c = std::cos(a), si = std::sin(a);
            r = u * c + cross(ax, u) * si + ax * (dot(ax, u) * (1.0 - c));
        }
        r = r - t1 * dot(r, t1);                            // undo accumulated drift
        double L = length(r);
        if (L < 1e-9) { Vec3 b; onb(t1, r, b); L = length(r); }
        return r / L;
    };
    std::vector<Vec3> U(n);
    { Vec3 b; onb(T[0], U[0], b); }
    for (int i = 1; i < n; ++i) U[i] = transport(U[i - 1], T[i - 1], T[i]);
    if (closed && n > 2) {
        // Transporting once more across the seam does NOT land back on U[0] — that
        // residual angle is the frame's holonomy, and left alone it becomes a single
        // sheared band of triangles at one joint. Spread it evenly along the loop so
        // the tube closes on itself exactly (c[n-1] - c[0] == the mismatch).
        Vec3   w   = transport(U[n - 1], T[n - 1], T[0]);
        double ang = std::atan2(dot(cross(w, U[0]), T[0]), dot(w, U[0]));
        for (int i = 0; i < n; ++i) {
            double a = ang * ((double)i / (double)(n - 1));
            double c = std::cos(a), si = std::sin(a);
            U[i] = U[i] * c + cross(T[i], U[i]) * si;
        }
    }

    // A closed fiber emits one extra ring that repeats sample 0 — the positions are
    // identical (so the tube is still closed) but it carries u=1, which keeps the
    // texture from folding back over the last span.
    const int rings = closed ? n + 1 : n;
    const int K     = STRAND_SIDES;
    g.verts.reserve((size_t)rings * K * 3);
    g.uvs.reserve((size_t)rings * K * 2);
    for (int i = 0; i < rings; ++i) {
        int   j = i % n;
        Vec3  V = cross(T[j], U[j]);
        float u = (rings > 1) ? (float)i / (float)(rings - 1) : 0.0f;
        for (int k = 0; k < K; ++k) {
            double a = 2.0 * 3.14159265358979323846 * (double)k / (double)K;
            Vec3   p = P[j] + (U[j] * std::cos(a) + V * std::sin(a)) * R[j];
            g.verts.push_back((float)p.x);
            g.verts.push_back((float)p.y);
            g.verts.push_back((float)p.z);
            g.uvs.push_back(u);
            g.uvs.push_back((float)k / (float)K);
        }
    }
    for (int i = 0; i + 1 < rings; ++i)
        for (int k = 0; k < K; ++k) {
            int k1 = (k + 1) % K;
            int a = i * K + k, b = i * K + k1, c = (i + 1) * K + k1, d = (i + 1) * K + k;
            g.faces.push_back(a); g.faces.push_back(b); g.faces.push_back(c);
            g.faces.push_back(a); g.faces.push_back(c); g.faces.push_back(d);
        }
    if (!closed) {                       // flat caps, so an open fiber isn't a straw
        for (int e = 0; e < 2; ++e) {
            int  j    = e ? n - 1 : 0;
            int  ring = e ? (rings - 1) * K : 0;
            int  ctr  = (int)g.verts.size() / 3;
            g.verts.push_back((float)P[j].x);
            g.verts.push_back((float)P[j].y);
            g.verts.push_back((float)P[j].z);
            g.uvs.push_back(e ? 1.0f : 0.0f); g.uvs.push_back(0.5f);
            for (int k = 0; k < K; ++k) {
                int k1 = (k + 1) % K;
                g.faces.push_back(ctr); g.faces.push_back(ring + k); g.faces.push_back(ring + k1);
            }
        }
    }
    g.nverts = (int)g.verts.size() / 3;
    g.nfaces = (int)g.faces.size() / 3;
    // A tube is a smooth surface that a 10-gon only approximates, so shade it as
    // one. ftrace's default crease angle does the right thing on both features
    // here: the ~36 deg step around the profile is under it and gets merged, the
    // 90 deg rim where a flat cap meets the tube is over it and stays sharp.
    g.smoothDeg = MESH_PANE_CREASE_DEG;
    return g.nfaces > 0;
}

static std::vector<MeshGeom> collectMeshes(const Sidecar& sc) {
    std::vector<MeshGeom> meshes;
    const minijson::Value* objs = sc.arr("objects");
    if (!objs) return meshes;
    // objects may nest (Groups) — walk recursively
    std::function<void(const minijson::Value&)> visit = [&](const minijson::Value& o) {
        if (const minijson::Value* ch = o.find("children"); ch && ch->isArray())
            for (const auto& c : ch->arr) visit(c);
        MeshGeom g;
        g.id       = scalarStr(o.find("id"), "");
        g.name     = scalarStr(o.find("name"), "");
        g.material = scalarStr(o.find("material"), "");
        // A fiber has no baked mesh — tube its centreline so it shares this pane
        // with the swept surfaces instead of being invisible here.
        if (const minijson::Value* st = o.find("strand"); st && st->isObject()) {
            if (strandToMesh(*st, g)) meshes.push_back(std::move(g));
            return;
        }
        const minijson::Value* m = o.find("mesh");
        if (!m || !m->isObject()) return;
        if (const minijson::Value* v = m->find("vertices"); v && v->isArray())
            for (const auto& p : v->arr) {
                for (int k = 0; k < 3; ++k)
                    g.verts.push_back(p.isArray() && k < (int)p.arr.size()
                                          ? (float)p.arr[k].asNumber(0.0) : 0.0f);
            }
        g.nverts = (int)g.verts.size() / 3;
        if (const minijson::Value* f = m->find("faces"); f && f->isArray())
            for (const auto& t : f->arr)
                if (t.isArray() && t.arr.size() >= 3)
                    for (int k = 0; k < 3; ++k) g.faces.push_back(t.arr[k].asInt(0));
        g.nfaces = (int)g.faces.size() / 3;
        if (const minijson::Value* u = m->find("uvs"); u && u->isArray())
            for (const auto& p : u->arr)
                for (int k = 0; k < 2; ++k)
                    g.uvs.push_back(p.isArray() && k < (int)p.arr.size()
                                        ? (float)p.arr[k].asNumber(0.0) : 0.0f);
        // Shading normals: authored ones if loom shipped them (the .ftmesh format
        // carries a normals block), otherwise the crease angle it asked for, which
        // the pane resolves into per-vertex normals itself — see buildMeshPaneVerts.
        if (const minijson::Value* nn = m->find("normals"); nn && nn->isArray())
            for (const auto& p : nn->arr)
                for (int k = 0; k < 3; ++k)
                    g.normals.push_back(p.isArray() && k < (int)p.arr.size()
                                            ? (float)p.arr[k].asNumber(0.0) : 0.0f);
        if ((int)g.normals.size() < 3 * g.nverts) g.normals.clear();
        if (const minijson::Value* sm = m->find("smooth")) g.smoothDeg = sm->asNumber(0.0);
        meshes.push_back(std::move(g));
    };
    for (const auto& o : objs->arr) visit(o);
    return meshes;
}

// --------------------------------------------------------------------------
// F4 — skins: the sidecar's `textures` + `materials` decoded into real GPU
// textures the mesh pane samples at the mesh UVs (replacing the UV-checker
// placeholder). Decoding goes through ftrace's OWN `Texture` (image files) and
// pattern VM (procedural `rgb "r(u,v)" …` skins baked exactly the way
// `FtslLoader::addTexture` bakes them), so the preview shows the same pixels the
// renderer would — no second decoder to drift out of sync.
// --------------------------------------------------------------------------
struct Skin {
    std::string               name;
    Texture                   tex;            // decoded LINEAR rgb (ftrace's own)
    ID3D11Texture2D*          d3d = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    std::string               err;            // non-empty => unusable, shown in the UI
    std::string               kind;           // "image" | "formula"
};

// Largest edge we upload. A skin may legitimately be 8192² (the ftsl `res` cap),
// which is 256 MB of RGBA — far more than a preview pane can show. Anything bigger
// is resampled down through the texture's own sampler (so its filter/wrap apply).
static const int SKIN_MAX_EDGE = 2048;

struct SkinLib {
    std::vector<Skin>                    skins;
    std::unordered_map<std::string, int> byName;      // texture name -> index
    std::unordered_map<std::string, int> byMaterial;  // material name -> index
    int nOk = 0;

    // `tex:<name>(u,v)` inside a procedural skin resolves against the images decoded
    // BEFORE it, mirroring ftrace's rule that a procedural texture can only sample
    // images declared above it (both bake in sidecar/file order).
    static int lookupThunk(const void* self, const char* name) {
        const SkinLib* L = (const SkinLib*)self;
        auto it = L->byName.find(name);
        if (it == L->byName.end()) return -1;
        return L->skins[it->second].tex.valid() ? it->second : -1;
    }
    static double sampleThunk(const void* self, int idx, double u, double v) {
        const SkinLib* L = (const SkinLib*)self;
        if (idx < 0 || idx >= (int)L->skins.size()) return 0.0;
        return L->skins[idx].tex.scalarAt(u, v);
    }

    int forMaterial(const std::string& m) const {
        auto it = byMaterial.find(m);
        return (it == byMaterial.end()) ? -1 : it->second;
    }
    const Skin* skinFor(const std::string& material) const {
        int i = forMaterial(material);
        return (i < 0) ? nullptr : &skins[i];
    }

    void release() {
        for (auto& s : skins) {
            if (s.srv) { s.srv->Release(); s.srv = nullptr; }
            if (s.d3d) { s.d3d->Release(); s.d3d = nullptr; }
        }
        skins.clear(); byName.clear(); byMaterial.clear(); nOk = 0;
    }

    // Decode one `textures[]` entry into `sk.tex` (or set sk.err).
    void decode(const minijson::Value& t, Skin& sk, const std::string& baseDir) {
        auto pick = [&](const char* key, const char* dflt) {
            const minijson::Value* v = t.find(key);
            return (v && v->isString()) ? v->str : std::string(dflt);
        };
        std::string flt = pick("filter", "bilinear");
        sk.tex.filter = (flt == "nearest") ? TexFilter::Nearest : TexFilter::Bilinear;
        std::string wr = pick("wrap", "repeat");
        sk.tex.wrap = (wr == "clamp")  ? TexWrap::Clamp
                    : (wr == "mirror") ? TexWrap::Mirror : TexWrap::Repeat;
        sk.tex.name = sk.name;

        if (sk.kind == "image") {
            std::string file = pick("file", "");
            if (file.empty()) { sk.err = "image skin has no `file`"; return; }
            sk.tex.encoding = (pick("encoding", "srgb") == "linear")
                                  ? TexEncoding::Linear : TexEncoding::sRGB;
            // Paths in the sidecar are as the loom script authored them (usually
            // relative to where it ran). Try verbatim first, then next to the sidecar.
            std::string e1, e2;
            if (!sk.tex.load(file, e1)) {
                bool rel = file.size() < 2 || (file[1] != ':' && file[0] != '/' && file[0] != '\\');
                if (rel && !baseDir.empty() && sk.tex.load(baseDir + file, e2)) return;
                sk.err = e1;
            }
            return;
        }
        if (sk.kind != "formula") { sk.err = "unsupported skin kind '" + sk.kind + "'"; return; }

        // Procedural skin: bake the three UV expressions to a res x res LINEAR grid,
        // byte-for-byte the same loop as FtslLoader::addTexture (ftsl.h).
        const char* chan[3] = { "r", "g", "b" };
        std::vector<PatNode> prog[3];
        PatTexScope scope{ this, &SkinLib::lookupThunk };
        for (int k = 0; k < 3; ++k) {
            std::string expr = pick(chan[k], "0");
            std::string perr;
            if (!compilePatternExpr(expr, prog[k], perr, false, &scope)) {
                sk.err = std::string(chan[k]) + ": " + perr;
                return;
            }
        }
        const minijson::Value* rv = t.find("res");
        int res = (rv && rv->isNumber()) ? (int)rv->num : 512;
        res = std::min(std::max(res, 1), SKIN_MAX_EDGE);
        sk.tex.encoding = TexEncoding::Linear;   // expr outputs are linear albedo
        sk.tex.w = sk.tex.h = res;
        sk.tex.rgb.assign((size_t)res * res, Vec3{0, 0, 0});
        auto cl = [](double q) { return q < 0.0 ? 0.0 : (q > 1.0 ? 1.0 : q); };
        for (int y = 0; y < res; ++y) {
            double v = 1.0 - (y + 0.5) / res;    // matches sampleRgb's (1-v) flip
            for (int x = 0; x < res; ++x) {
                PatCtx c;
                c.u = (x + 0.5) / res; c.v = v;
                c.texFn = &SkinLib::sampleThunk; c.texSelf = this;
                sk.tex.rgb[(size_t)y * res + x] =
                    Vec3{ cl(patternEval(prog[0].data(), (int)prog[0].size(), c)),
                          cl(patternEval(prog[1].data(), (int)prog[1].size(), c)),
                          cl(patternEval(prog[2].data(), (int)prog[2].size(), c)) };
            }
        }
    }

    // Upload a decoded skin as an sRGB-encoded RGBA8 texture. `Texture::rgb` is
    // LINEAR (that is what the renderer wants); the ImGui blit path is a plain
    // pass-through to an 8-bit backbuffer, so gamma-encode here or every skin shows
    // up washed-out dark.
    bool upload(Skin& sk, ID3D11Device* dev, ID3D11DeviceContext* ctx) {
        if (!sk.tex.valid()) return false;
        int W = std::min(sk.tex.w, SKIN_MAX_EDGE), H = std::min(sk.tex.h, SKIN_MAX_EDGE);
        std::vector<uint8_t> px((size_t)W * H * 4);
        for (int y = 0; y < H; ++y) {
            // v runs the other way (sampleRgb flips it), so row 0 here is row 0 there.
            double v = 1.0 - (y + 0.5) / H;
            for (int x = 0; x < W; ++x) {
                Vec3 c = sk.tex.sampleRgb((x + 0.5) / W, v);
                uint8_t* d = &px[((size_t)y * W + x) * 4];
                d[0] = (uint8_t)std::lround(std::clamp(srgbGamma(c.x), 0.0, 1.0) * 255.0);
                d[1] = (uint8_t)std::lround(std::clamp(srgbGamma(c.y), 0.0, 1.0) * 255.0);
                d[2] = (uint8_t)std::lround(std::clamp(srgbGamma(c.z), 0.0, 1.0) * 255.0);
                d[3] = 255;
            }
        }
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_IMMUTABLE;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA sd = {};
        sd.pSysMem = px.data();
        sd.SysMemPitch = (UINT)W * 4;
        if (dev->CreateTexture2D(&td, &sd, &sk.d3d) != S_OK) { sk.err = "CreateTexture2D failed"; return false; }
        if (dev->CreateShaderResourceView(sk.d3d, nullptr, &sk.srv) != S_OK) {
            sk.d3d->Release(); sk.d3d = nullptr;
            sk.err = "CreateShaderResourceView failed";
            return false;
        }
        (void)ctx;
        return true;
    }

    void build(const Sidecar& sc, const std::string& baseDir,
               ID3D11Device* dev, ID3D11DeviceContext* ctx) {
        release();
        if (const minijson::Value* ts = sc.arr("textures")) {
            for (const auto& t : ts->arr) {
                if (!t.isObject()) continue;
                Skin sk;
                sk.name = scalarStr(t.find("name"), "");
                sk.kind = scalarStr(t.find("kind"), "");
                if (sk.name.empty()) continue;
                decode(t, sk, baseDir);
                // Register the NAME before uploading so a later procedural skin's
                // `tex:` can resolve it (lookupThunk re-checks tex.valid()).
                int idx = (int)skins.size();
                skins.push_back(std::move(sk));
                byName[skins[idx].name] = idx;
                if (skins[idx].err.empty() && upload(skins[idx], dev, ctx)) ++nOk;
            }
        }
        // A mesh names a MATERIAL; the skin it wears is that material's `texture:`
        // binding, which the sidecar resolved for us (loom's `_describe_material`).
        if (const minijson::Value* ms = sc.arr("materials")) {
            for (const auto& m : ms->arr) {
                if (!m.isObject()) continue;
                std::string mn = scalarStr(m.find("name"), "");
                const minijson::Value* tv = m.find("texture");
                if (mn.empty() || !tv || !tv->isString()) continue;
                auto it = byName.find(tv->str);
                if (it != byName.end()) byMaterial[mn] = it->second;
            }
        }
    }
};

} // namespace

// --------------------------------------------------------------------------
// F4 item 2 — live parameter state, declared up here because the geometry panes
// below carry the "rotate into a parameter dimension" gesture. The transport that
// fills this in (LoomLink / LoomBridge) lives further down, next to the entry point.
//
// The distinction this whole feature turns on: the three dims a pane SHOWS can be
// re-projected for free by rotating the view, but a parameter dimension is not in
// the geometry at all — moving along it means loom has to re-derive (re-tessellate)
// the scene. So the spatial orbit stays on the left mouse button and is instant,
// and the parameter sweep is on the right button and costs a round trip.
// --------------------------------------------------------------------------

// One live control. `toJson` is what actually gets sent, so a param the build
// declared as an int stays an int (JSON has no such distinction; loom tells us).
struct LiveParam {
    std::string name;
    int  kind = 0;            // 0 float, 1 int, 2 bool, 3 opaque (shown read-only)
    double num = 0.0;
    bool   bval = false;
    std::string text;         // opaque/str params: echoed back verbatim
    double speed = 0.01;      // drag sensitivity, seeded from the default's magnitude

    bool continuous() const { return kind == 0 || kind == 1; }

    std::string toJson() const {
        char b[64];
        switch (kind) {
        case 1: std::snprintf(b, sizeof b, "%lld", (long long)llround(num)); return b;
        case 2: return bval ? "true" : "false";
        case 3: return text;
        default: std::snprintf(b, sizeof b, "%.10g", num); return b;
        }
    }
};

struct LivePanel {
    bool up = false;                 // the bridge started and the link is serving
    std::string startErr;            // ...or why it isn't
    int  frame = 0, frames = 1;      // the clock, itself a parameter dimension
    std::vector<LiveParam> params;
    int  sweep = -1;                 // index into params: the axis a canvas drag moves
    bool autoApply = true;           // re-derive on every change vs. on the button
    // Latest-wins accounting, and the whole point of the panel's counter line: `posted`
    // is how many jobs the UI handed to the bridge, `baked` how many loom actually ran,
    // `appliedSeq` which job the panes are showing. posted > baked is the mechanism
    // working (a fast drag collapses to one bake), not jobs being lost.
    long long posted = 0, baked = 0, appliedSeq = 0;
    double lastMs = 0.0;
    std::string lastErr;
    // --- F8(a) paced play -------------------------------------------------
    // The clock advances only when a bake LANDS, never on a wall-clock timer. The
    // bridge is latest-wins on a one-slot pending job (the rule that makes a drag
    // cost one bake), so a play loop that posted on a timer would have most of its
    // frames superseded before they ran and would show a stutter of whichever ones
    // won the slot -- not playback. Pacing to the bake rate plays every frame, and
    // the fps readout states the rate honestly rather than pretending to be 30.
    bool playing = false;
    // Force the "play just started" priming post even though `playing` was set
    // before the first draw (the -play flag), where there is no rising edge to see.
    bool primePlay = false;
    bool loopPlay = true;         // wrap at the end vs. stop there
    bool pingpong = false;        // bounce instead of wrapping
    int  dir = 1;                 // +1 / -1, flipped by ping-pong
    // MEASURED playback rate (EMA), not 1000/lastMs. `lastMs` times loom's bake alone;
    // the viewer then parses a multi-MB sidecar, rebuilds mesh buffers and re-inits the
    // raymarch pane on the UI thread, and that adoption cost is routinely the larger
    // half. Deriving fps from the bake overstated real playback by ~10x here, which is
    // precisely the "pretend it's 30" this feature was supposed to avoid.
    double    playFps = 0.0;
    double    msPlayPeriod = 0.0;   // the EMA `playFps` is derived from -- see below
    long long lastAdvanceQpc = 0;
    // Per-stage adoption cost, so "why is play slow?" is answered by measurement
    // rather than by guessing at the FTSL round trip. Milliseconds, last frame.
    double msSidecar = 0.0;   // parse the introspection JSON + rebuild DAG/skins
    double msFtsl    = 0.0;   // parse the .ftsl + load its mesh assets
    double msRender  = 0.0;   // the Render pane's synchronous CUDA raymarch
    // ...and where THAT went. Split out because the three phases scale with different
    // things (scene size / pixels / pixels) and because only `kernel` is SM-bound: a
    // foreign process saturating the card inflates that phase ALONE. An earlier profile
    // of this pane blamed the raymarch while another program held 90% of the GPU; with
    // the split, such a contaminated reading is self-evident instead of plausible.
    double msRenderUpload = 0.0;  // marshal the WHOLE scene + H2D (scene size; CPU+DMA)
    double msRenderKernel = 0.0;  // the raymarch itself           (pixels; SM-bound)
    double msRenderRead   = 0.0;  // D2H + host tone map           (pixels; mostly CPU)
    // ...and the same treatment for msSidecar, which the n=50 profile showed is the
    // single biggest term in a played frame (90.5 of 215 ms). Same rule as above: it
    // is a sum of four unrelated costs -- a JSON parse that scales with sidecar bytes,
    // geometry collection that scales with tessellation, a DAG rebuild that scales with
    // node count, and a GPU skin rebuild that scales with TEXELS and is pure waste when
    // the texture set has not changed. Ranking those by intuition is exactly the error
    // that produced three wrong diagnoses here, so they are measured separately.
    double msAdoptJson  = 0.0;  // Sidecar::load  -- minijson parse of the whole file
    double msAdoptGeom  = 0.0;  // curves/strips/fields/meshes collection
    double msAdoptDag   = 0.0;  // collectDag + layout carry-over
    double msAdoptSkins = 0.0;  // skins.release() + skins.build() (decode + D3D11 upload)
    // ...and the same for msFtsl, now the largest term. `assets` and `accel` are nested
    // inside `build`, so the residual (build - assets - accel) is the Builder's own work.
    // The interesting question this answers: of the per-frame .ftsl round-trip, how much
    // is text parsing (which a direct mesh handoff would delete outright) versus BVH
    // construction (which it would NOT -- that has to happen for any new geometry).
    double msFtslParse  = 0.0;  // source text -> Block tree (ftsl_gpda::parse)
    double msFtslBuild  = 0.0;  // Block tree -> Scene, INCLUDING assets + accel below
    double msFtslAssets = 0.0;  // of build: mesh files read+parsed from disk (obj/gltf/fbx)
    double msFtslAccel  = 0.0;  // of build: BVH construction (per-asset Blas + Scene::build)
    // F8(b): the cost of showing a frame from the prebake cache INSTEAD of baking it --
    // two O(1) state swaps plus the derived-view rebuilds that a fresh adoption would
    // also have done (the skin atlas, the raymarch pane's scene upload). It exists as
    // its own term because a cached frame pays NONE of bake/sidecar/ftsl, and leaving
    // those reading their last uncached values would print a breakdown whose parts sum
    // to twice the frame time it is printed beside. They are zeroed on a cache hit and
    // this replaces them; on a real bake the reverse happens.
    double msCache = 0.0;
    // Set by the Render tab each UI frame it actually draws. Needed because the
    // Live panel is drawn BEFORE the Render pane, so zeroing msRender when a bake
    // lands would blank it every frame during play -- it would always read 0 and
    // wrongly exonerate the raymarch. Instead the value persists and is cleared
    // only once the tab has genuinely stopped drawing.
    bool renderTabDrew = false;
};

// A scoped wall-clock stopwatch that adds into a double (milliseconds).
struct MsTimer {
    double*   sink;
    long long t0;
    explicit MsTimer(double* d) : sink(d) {
        LARGE_INTEGER q; QueryPerformanceCounter(&q); t0 = q.QuadPart;
    }
    // Close the interval early and detach, for a phase whose end does not line up with
    // a scope (e.g. a block that declares locals the next phase must still see). Safe
    // to call more than once; the destructor then does nothing.
    void stop() {
        LARGE_INTEGER q, f;
        QueryPerformanceCounter(&q); QueryPerformanceFrequency(&f);
        if (sink && f.QuadPart) *sink = 1000.0 * double(q.QuadPart - t0) / double(f.QuadPart);
        sink = nullptr;
    }
    ~MsTimer() { if (sink) stop(); }
};

// Step the clock one frame in the current play direction, honouring loop/ping-pong.
static void liveAdvanceClock(LivePanel& lp) {
    if (lp.frames <= 1) return;
    int next = lp.frame + lp.dir;
    if (lp.pingpong) {
        if (next >= lp.frames)  { next = lp.frames - 2; lp.dir = -1; }
        else if (next < 0)      { next = 1;             lp.dir = +1; }
    } else if (next >= lp.frames) {
        if (lp.loopPlay) next = 0;
        else { next = lp.frames - 1; lp.playing = false; }
    } else if (next < 0) {
        next = lp.loopPlay ? lp.frames - 1 : 0;
    }
    lp.frame = std::min(std::max(next, 0), lp.frames - 1);

    LARGE_INTEGER f, now;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&now);
    if (lp.lastAdvanceQpc && f.QuadPart) {
        double dt = double(now.QuadPart - lp.lastAdvanceQpc) / double(f.QuadPart);
        if (dt > 1e-9) {
            // Smooth the PERIOD and invert at the end, not the rate. Averaging rates
            // is the wrong mean: with the §F8(b) pacer the interval alternates between
            // two and three vblanks (33.3 / 50.0 ms at 60 Hz), whose true average is
            // 41.7 ms = 24 fps -- but averaging 30 and 20 fps gives 25. That 6% flatter
            // it reports the faster it is asked to go, and the readout exists precisely
            // so the requested rate can be checked against the delivered one.
            lp.msPlayPeriod = lp.msPlayPeriod > 0.0 ? lp.msPlayPeriod * 0.8 + dt * 1000.0 * 0.2
                                                    : dt * 1000.0;
            lp.playFps = 1000.0 / lp.msPlayPeriod;
        }
    }
    lp.lastAdvanceQpc = now.QuadPart;
}

// The canvas gesture: right-drag sweeps the chosen parameter axis. Call it directly
// after the pane's InvisibleButton (it reads that item's active state). Returns true
// when the value actually moved, which is what schedules a re-derivation.
static bool liveSweepDrag(LivePanel* lp) {
    if (!lp || !lp->up) return false;
    if (lp->sweep < 0 || lp->sweep >= (int)lp->params.size()) return false;
    if (!ImGui::IsItemActive() || !ImGui::IsMouseDragging(ImGuiMouseButton_Right)) return false;
    float dx = ImGui::GetIO().MouseDelta.x;
    if (dx == 0.0f) return false;
    LiveParam& p = lp->params[lp->sweep];
    double before = p.num;
    p.num += dx * p.speed;
    if (p.kind == 1) p.num = (double)llround(p.num);
    return p.num != before;          // an int axis only ticks once per whole step
}

// The line under a pane's banner naming the sweep axis, so the gesture is discoverable.
// Deliberately NOT SameLine'd onto the banner: the banners are already near the width
// of the right-hand column, so appending to them pushed the hint — the part that says
// which key the drag actually turns — off the right edge at any normal window size.
static void liveSweepHint(const LivePanel* lp) {
    if (!lp || !lp->up) return;
    if (lp->sweep >= 0 && lp->sweep < (int)lp->params.size()) {
        const LiveParam& p = lp->params[lp->sweep];
        ImGui::TextColored(ImVec4(0.5f, 0.9f, 1.0f, 1.0f), "right-drag sweeps %s = %.4g",
                           p.name.c_str(), p.num);
    } else {
        ImGui::TextDisabled("(no sweep axis - pick one in Live)");
    }
}

// --------------------------------------------------------------------------
// Curve pane: a simple orthographic projection drawn with ImDrawList.
// (Slice C generalizes this to 3-of-N dim selection, rotation, and stereo.)
// --------------------------------------------------------------------------
enum StereoMode { STEREO_MONO = 0, STEREO_ANAGLYPH, STEREO_WALL, STEREO_CROSS };

struct OrbitView {
    float yaw   = 0.6f;   // radians
    float pitch = 0.4f;
    float zoom  = 1.0f;
    int   dx = 0, dy = 1, dz = 2;   // which of the N dims map to screen X/Y/Z
    int   maxDim = 3;               // widest curve dimensionality in the scene
    float index = 0.0f;             // 0..1 position of the highlighted index marker
    int   stereo = STEREO_MONO;     // mono / anaglyph / side-by-side
    float sep    = 0.10f;           // stereo eye-yaw separation (radians)
    double sx0 = 0.0, sx1 = 1.0;    // shared/linked X range for the F3 strip charts
};

// Project a 3-vector (already the 3 selected dims, centered) to screen space, with
// an extra yaw offset for the stereo eye separation.
static ImVec2 project3(float x, float y, float z, const OrbitView& v, float yawOff,
                       ImVec2 center, float scale) {
    float cy = std::cos(v.yaw + yawOff), sy = std::sin(v.yaw + yawOff);
    float cx = std::cos(v.pitch),        sx = std::sin(v.pitch);
    float x1 =  cy * x + sy * z;
    float z1 = -sy * x + cy * z;
    float y1 =  cx * y - sx * z1;
    return ImVec2(center.x + x1 * scale * v.zoom,
                  center.y - y1 * scale * v.zoom);
}

// Pull the 3 selected dims out of a flat N-D point, minus the centering offset.
static void pick3(const float* p, int dim, const OrbitView& v, const float* mid,
                  float& x, float& y, float& z) {
    x = (v.dx < dim ? p[v.dx] : 0.0f) - mid[0];
    y = (v.dy < dim ? p[v.dy] : 0.0f) - mid[1];
    z = (v.dz < dim ? p[v.dz] : 0.0f) - mid[2];
}

static void drawCurvePane(const std::vector<CurveGeom>& curves, OrbitView& view) {
    // ---- controls: 3-of-N dim pickers + index marker slider ----
    ImGui::TextUnformatted("Curves - drag to orbit, wheel to zoom");
    if (view.maxDim > 3) {
        ImGui::SameLine();
        ImGui::TextDisabled("(showing 3 of %d dims)", view.maxDim);
    }
    auto dimCombo = [&](const char* label, int& sel) {
        ImGui::SetNextItemWidth(70);
        std::string cur = "d" + std::to_string(sel);
        if (ImGui::BeginCombo(label, cur.c_str())) {
            for (int k = 0; k < view.maxDim; ++k) {
                std::string it = "d" + std::to_string(k);
                if (ImGui::Selectable(it.c_str(), sel == k)) sel = k;
            }
            ImGui::EndCombo();
        }
    };
    dimCombo("X", view.dx); ImGui::SameLine();
    dimCombo("Y", view.dy); ImGui::SameLine();
    dimCombo("Z", view.dz); ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    ImGui::SliderFloat("index", &view.index, 0.0f, 1.0f, "%.3f");
    // stereo controls
    ImGui::SetNextItemWidth(150);
    const char* modes[] = { "mono", "anaglyph (R/cyan)", "wall-eyed L|R", "cross-eyed R|L" };
    ImGui::Combo("stereo", &view.stereo, modes, 4);
    if (view.stereo != STEREO_MONO) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(150);
        ImGui::SliderFloat("sep", &view.sep, 0.0f, 0.4f, "%.3f rad");
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.y < 80.0f) avail.y = 80.0f;
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("curve_canvas", avail);
    bool hovered = ImGui::IsItemHovered();
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        ImVec2 d = ImGui::GetIO().MouseDelta;
        view.yaw   += d.x * 0.01f;
        view.pitch += d.y * 0.01f;
    }
    if (hovered) {
        float w = ImGui::GetIO().MouseWheel;
        if (w != 0.0f) view.zoom *= (1.0f + w * 0.1f);
    }
    if (view.zoom < 0.05f) view.zoom = 0.05f;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 br(origin.x + avail.x, origin.y + avail.y);
    dl->AddRectFilled(origin, br, IM_COL32(18, 18, 22, 255));
    dl->PushClipRect(origin, br, true);

    // auto-fit scale from the union bounds of the 3 selected dims
    float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
    int seldim[3] = { view.dx, view.dy, view.dz };
    for (const auto& c : curves)
        for (int i = 0; i < c.polyN; ++i) {
            const float* p = &c.poly[(size_t)i * c.dim];
            for (int k = 0; k < 3; ++k)
                if (seldim[k] < c.dim) {
                    lo[k] = std::min(lo[k], p[seldim[k]]);
                    hi[k] = std::max(hi[k], p[seldim[k]]);
                }
        }
    float ext = 1.0f;
    for (int k = 0; k < 3; ++k) if (hi[k] > lo[k]) ext = std::max(ext, hi[k] - lo[k]);
    float mid[3] = { 0, 0, 0 };
    for (int k = 0; k < 3; ++k) if (hi[k] >= lo[k]) mid[k] = 0.5f * (lo[k] + hi[k]);

    const ImU32 palette[] = {
        IM_COL32(120, 200, 255, 255), IM_COL32(255, 180, 120, 255),
        IM_COL32(160, 255, 160, 255), IM_COL32(255, 140, 200, 255),
    };

    // Draw all curves for one eye into one sub-viewport. `tint != 0` forces a single
    // colour (anaglyph); otherwise each curve uses the palette. `yawOff` is the eye
    // separation. `center`/`scale` are per-viewport so side-by-side splits the canvas.
    auto drawEye = [&](ImVec2 center, float scale, float yawOff, ImU32 tint) {
        int ci = 0;
        for (const auto& c : curves) {
            ImU32 col = tint ? tint : palette[ci % 4]; ++ci;
            ImVec2 prev; bool have = false;
            for (int i = 0; i < c.polyN; ++i) {
                float x, y, z;
                pick3(&c.poly[(size_t)i * c.dim], c.dim, view, mid, x, y, z);
                ImVec2 s = project3(x, y, z, view, yawOff, center, scale);
                if (have) dl->AddLine(prev, s, col, 1.6f);
                prev = s; have = true;
            }
            ImU32 dotc = tint ? tint : IM_COL32(90, 90, 110, 255);
            int markers = 8;
            for (int m = 0; m <= markers; ++m) {
                if (c.closed && m == markers) break;
                int i = (int)((float)m / markers * (c.polyN - 1) + 0.5f);
                if (i < 0 || i >= c.polyN) continue;
                float x, y, z;
                pick3(&c.poly[(size_t)i * c.dim], c.dim, view, mid, x, y, z);
                ImVec2 s = project3(x, y, z, view, yawOff, center, scale);
                dl->AddCircleFilled(s, 2.2f, dotc);
            }
            if (c.polyN > 0) {   // highlighted index dot
                int i = (int)(view.index * (c.polyN - 1) + 0.5f);
                i = std::max(0, std::min(c.polyN - 1, i));
                float x, y, z;
                pick3(&c.poly[(size_t)i * c.dim], c.dim, view, mid, x, y, z);
                ImVec2 s = project3(x, y, z, view, yawOff, center, scale);
                ImU32 hc = tint ? tint : IM_COL32(255, 240, 80, 255);
                dl->AddCircleFilled(s, 4.5f, hc);
                dl->AddCircle(s, 6.5f, tint ? tint : IM_COL32(255, 240, 80, 160), 0, 1.5f);
            }
            ImU32 sq = tint ? tint : IM_COL32(255, 255, 255, 220);
            for (int i = 0; i < c.ctrlN; ++i) {   // control points
                float x, y, z;
                pick3(&c.ctrl[(size_t)i * c.dim], c.dim, view, mid, x, y, z);
                ImVec2 s = project3(x, y, z, view, yawOff, center, scale);
                dl->AddRectFilled(ImVec2(s.x - 2, s.y - 2), ImVec2(s.x + 2, s.y + 2), sq);
            }
        }
    };

    const ImU32 RED  = IM_COL32(230, 40, 40, 255);
    const ImU32 CYAN = IM_COL32(40, 220, 220, 255);
    float half = view.sep * 0.5f;
    if (view.stereo == STEREO_MONO) {
        ImVec2 center(origin.x + avail.x * 0.5f, origin.y + avail.y * 0.5f);
        float scale = 0.4f * std::min(avail.x, avail.y) / (0.5f * ext + 1e-3f);
        drawEye(center, scale, 0.0f, 0);
    } else if (view.stereo == STEREO_ANAGLYPH) {
        ImVec2 center(origin.x + avail.x * 0.5f, origin.y + avail.y * 0.5f);
        float scale = 0.4f * std::min(avail.x, avail.y) / (0.5f * ext + 1e-3f);
        drawEye(center, scale, -half, RED);   // left eye  -> red
        drawEye(center, scale, +half, CYAN);  // right eye -> cyan
    } else {  // side-by-side: split the canvas into two half-width viewports
        float hw = avail.x * 0.5f;
        float scale = 0.4f * std::min(hw, avail.y) / (0.5f * ext + 1e-3f);
        ImVec2 cL(origin.x + hw * 0.5f,        origin.y + avail.y * 0.5f);
        ImVec2 cR(origin.x + hw + hw * 0.5f,   origin.y + avail.y * 0.5f);
        dl->AddLine(ImVec2(origin.x + hw, origin.y), ImVec2(origin.x + hw, br.y),
                    IM_COL32(60, 60, 70, 255));
        if (view.stereo == STEREO_WALL) {     // L|R
            drawEye(cL, scale, -half, 0);
            drawEye(cR, scale, +half, 0);
        } else {                              // cross-eyed R|L
            drawEye(cL, scale, +half, 0);
            drawEye(cR, scale, -half, 0);
        }
    }
    dl->PopClipRect();
}

// --------------------------------------------------------------------------
// F3 — scroll-locked strip charts (ImPlot). One chart per curve dimension and
// one per tacked-on channel component, all sharing a linked X axis (paging
// scrolls every chart together) and a draggable shared index marker.
// --------------------------------------------------------------------------
struct StripSeries {
    std::string        label;
    std::vector<float> x;   // normalized curve parameter 0..1 (so charts align)
    std::vector<float> y;
};

static std::vector<StripSeries> buildStrips(const std::vector<CurveGeom>& curves) {
    std::vector<StripSeries> out;
    int ci = 0;
    for (const auto& c : curves) {
        std::string cid = "#" + (c.id.empty() ? std::to_string(ci) : c.id);
        // one series per spatial dimension of the polyline
        for (int d = 0; d < c.dim; ++d) {
            StripSeries s;
            s.label = cid + " d" + std::to_string(d);
            s.x.resize(c.polyN); s.y.resize(c.polyN);
            for (int i = 0; i < c.polyN; ++i) {
                s.x[i] = c.polyN > 1 ? (float)i / (c.polyN - 1) : 0.0f;
                s.y[i] = c.poly[(size_t)i * c.dim + d];
            }
            out.push_back(std::move(s));
        }
        // one series per tacked-on channel component
        for (const auto& ch : c.channels) {
            for (int comp = 0; comp < ch.dim; ++comp) {
                StripSeries s;
                s.label = cid + " " + ch.name;
                if (ch.dim > 1) s.label += "[" + std::to_string(comp) + "]";
                s.x.resize(ch.n); s.y.resize(ch.n);
                for (int i = 0; i < ch.n; ++i) {
                    s.x[i] = ch.n > 1 ? (float)i / (ch.n - 1) : 0.0f;
                    s.y[i] = ch.samp[(size_t)i * ch.dim + comp];
                }
                out.push_back(std::move(s));
            }
        }
        ++ci;
    }
    return out;
}

static void drawStripCharts(const std::vector<StripSeries>& strips, OrbitView& view) {
    if (strips.empty()) {
        ImGui::TextDisabled("(no per-dimension / channel series to chart)");
        return;
    }
    ImGui::TextUnformatted("Strip charts - scroll-locked; drag the yellow index line");
    int n = (int)strips.size();
    ImVec2 avail = ImGui::GetContentRegionAvail();
    // fill when few charts, but never below a readable height (child then scrolls)
    float rowH = std::max(70.0f, avail.y / n);
    ImVec4 lineCol(0.47f, 0.78f, 1.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        const StripSeries& s = strips[i];
        std::string title = s.label + "##strip" + std::to_string(i);
        ImPlotFlags pf = ImPlotFlags_NoLegend | ImPlotFlags_NoMenus | ImPlotFlags_NoMouseText;
        if (ImPlot::BeginPlot(title.c_str(), ImVec2(-1, rowH - 6), pf)) {
            // link the X axis across every chart → they scroll/zoom together
            ImPlot::SetupAxisLinks(ImAxis_X1, &view.sx0, &view.sx1);
            ImPlotAxisFlags xf = ImPlotAxisFlags_NoGridLines |
                                 (i == n - 1 ? 0 : ImPlotAxisFlags_NoTickLabels);
            ImPlot::SetupAxes(nullptr, nullptr, xf, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, 1.0, ImGuiCond_Once);
            ImPlotSpec spec;
            spec.LineColor  = lineCol;
            spec.LineWeight = 1.4f;
            ImPlot::PlotLine(s.label.c_str(), s.x.data(), s.y.data(), (int)s.x.size(), spec);
            // the shared index marker (bidirectional with the 3-D pane's index dot)
            double idx = view.index;
            if (ImPlot::DragLineX(9001, &idx, ImVec4(1.0f, 0.94f, 0.3f, 1.0f), 1.5f))
                view.index = (float)std::min(1.0, std::max(0.0, idx));
            ImPlot::EndPlot();
        }
    }
}

// --------------------------------------------------------------------------
// F6 — field pane: scatter / grid sample points in a 3-D orbit view, colour-mapped
// by a selectable channel (or ch0/1/2 -> RGB), click-to-inspect, and per-extra-dim
// slice sliders for N-D grids (dims not shown collapse to a chosen lattice index).
// --------------------------------------------------------------------------
struct FieldView {
    float yaw = 0.6f, pitch = 0.4f, zoom = 1.0f;
    int   dx = 0, dy = 1, dz = 2;   // which position dims map to screen X/Y/Z
    int   maxDim = 3;
    int   colorMode = 0;            // 0 = scalar heatmap of `channel`, 1 = ch0/1/2 -> RGB
    int   channel = 0;             // heatmap channel index
    std::vector<int> slice;         // per-dim chosen lattice index (grids); size maxDim
    int   picked = -1;              // last clicked point (global index over the flat list)
    int   pickedField = -1;
};

// heat colour ramp (blue -> cyan -> green -> yellow -> red) for a 0..1 value
static ImU32 heat(float t) {
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    float r = std::min(1.0f, std::max(0.0f, 1.5f - std::fabs(4 * t - 3)));
    float g = std::min(1.0f, std::max(0.0f, 1.5f - std::fabs(4 * t - 2)));
    float b = std::min(1.0f, std::max(0.0f, 1.5f - std::fabs(4 * t - 1)));
    return IM_COL32((int)(r * 255), (int)(g * 255), (int)(b * 255), 255);
}

static void drawFieldPane(const std::vector<FieldGeom>& fields, FieldView& view) {
    ImGui::TextUnformatted("Fields - drag to orbit, wheel to zoom, click a point to inspect");
    if (view.maxDim > 3) {
        ImGui::SameLine();
        ImGui::TextDisabled("(showing 3 of %d dims)", view.maxDim);
    }
    auto dimCombo = [&](const char* label, int& sel) {
        ImGui::SetNextItemWidth(60);
        std::string cur = "d" + std::to_string(sel);
        if (ImGui::BeginCombo(label, cur.c_str())) {
            for (int k = 0; k < view.maxDim; ++k) {
                std::string it = "d" + std::to_string(k);
                if (ImGui::Selectable(it.c_str(), sel == k)) sel = k;
            }
            ImGui::EndCombo();
        }
    };
    dimCombo("X", view.dx); ImGui::SameLine();
    dimCombo("Y", view.dy); ImGui::SameLine();
    dimCombo("Z", view.dz); ImGui::SameLine();

    // channel / colour controls
    int maxVDim = 1;
    for (const auto& f : fields) maxVDim = std::max(maxVDim, f.valueDim);
    ImGui::SetNextItemWidth(140);
    const char* cmodes[] = { "heatmap channel", "ch0/1/2 -> RGB" };
    ImGui::Combo("colour", &view.colorMode, cmodes, 2);
    if (view.colorMode == 0 && maxVDim > 1) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        if (view.channel >= maxVDim) view.channel = 0;
        ImGui::SliderInt("chan", &view.channel, 0, maxVDim - 1);
    }

    // N-D grid slice sliders: for any dim not on screen, pick a lattice index to show
    if ((int)view.slice.size() < view.maxDim) view.slice.resize(view.maxDim, 0);
    if (view.maxDim > 3) {
        // widest per-dim lattice extent across grids (so the slider range is sensible)
        std::vector<int> ext(view.maxDim, 1);
        for (const auto& f : fields)
            for (int a = 0; a < (int)f.shape.size() && a < view.maxDim; ++a)
                ext[a] = std::max(ext[a], f.shape[a]);
        for (int a = 0; a < view.maxDim; ++a) {
            if (a == view.dx || a == view.dy || a == view.dz) continue;
            if (ext[a] <= 1) continue;
            ImGui::SetNextItemWidth(160);
            std::string lbl = "slice d" + std::to_string(a);
            if (view.slice[a] >= ext[a]) view.slice[a] = ext[a] - 1;
            ImGui::SliderInt(lbl.c_str(), &view.slice[a], 0, ext[a] - 1);
        }
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.y < 80.0f) avail.y = 80.0f;
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("field_canvas", avail);
    bool hovered = ImGui::IsItemHovered();
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        ImVec2 d = ImGui::GetIO().MouseDelta;
        view.yaw   += d.x * 0.01f;
        view.pitch += d.y * 0.01f;
    }
    if (hovered) {
        float w = ImGui::GetIO().MouseWheel;
        if (w != 0.0f) view.zoom *= (1.0f + w * 0.1f);
    }
    if (view.zoom < 0.05f) view.zoom = 0.05f;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 br(origin.x + avail.x, origin.y + avail.y);
    dl->AddRectFilled(origin, br, IM_COL32(16, 18, 20, 255));
    dl->PushClipRect(origin, br, true);

    // Reuse the curve pane's projection via a throwaway OrbitView with the same angles.
    OrbitView ov; ov.yaw = view.yaw; ov.pitch = view.pitch; ov.zoom = view.zoom;
    ov.dx = view.dx; ov.dy = view.dy; ov.dz = view.dz;

    int seldim[3] = { view.dx, view.dy, view.dz };
    auto visible = [&](const FieldGeom& f, const FieldPoint& p) -> bool {
        if (!f.isGrid || view.maxDim <= 3) return true;
        for (int a = 0; a < (int)p.idx.size(); ++a) {
            if (a == view.dx || a == view.dy || a == view.dz) continue;
            if (a < (int)view.slice.size() && p.idx[a] != view.slice[a]) return false;
        }
        return true;
    };

    // auto-fit bounds over the (visible) sample positions
    float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
    for (const auto& f : fields)
        for (const auto& p : f.points) {
            if (!visible(f, p)) continue;
            for (int k = 0; k < 3; ++k)
                if (seldim[k] < (int)p.pos.size()) {
                    lo[k] = std::min(lo[k], p.pos[seldim[k]]);
                    hi[k] = std::max(hi[k], p.pos[seldim[k]]);
                }
        }
    float ext = 1.0f;
    for (int k = 0; k < 3; ++k) if (hi[k] > lo[k]) ext = std::max(ext, hi[k] - lo[k]);
    float mid[3] = { 0, 0, 0 };
    for (int k = 0; k < 3; ++k) if (hi[k] >= lo[k]) mid[k] = 0.5f * (lo[k] + hi[k]);
    ImVec2 center(origin.x + avail.x * 0.5f, origin.y + avail.y * 0.5f);
    float scale = 0.4f * std::min(avail.x, avail.y) / (0.5f * ext + 1e-3f);

    // per-channel value range (for the heatmap normalisation)
    float vlo = 1e30f, vhi = -1e30f;
    for (const auto& f : fields)
        for (const auto& p : f.points) {
            if (!visible(f, p)) continue;
            int c = std::min(view.channel, (int)p.val.size() - 1);
            if (c >= 0) { vlo = std::min(vlo, p.val[c]); vhi = std::max(vhi, p.val[c]); }
        }
    float vspan = (vhi > vlo) ? (vhi - vlo) : 1.0f;

    // draw points, tracking the nearest to the mouse for click-to-inspect
    int   bestField = -1, bestPt = -1; float bestD2 = 1e30f;
    ImVec2 mouse = ImGui::GetIO().MousePos;
    for (int fi = 0; fi < (int)fields.size(); ++fi) {
        const FieldGeom& f = fields[fi];
        for (int pi = 0; pi < (int)f.points.size(); ++pi) {
            const FieldPoint& p = f.points[pi];
            if (!visible(f, p)) continue;
            float x, y, z;
            pick3(p.pos.data(), (int)p.pos.size(), ov, mid, x, y, z);
            ImVec2 s = project3(x, y, z, ov, 0.0f, center, scale);
            ImU32 col;
            if (view.colorMode == 1) {   // ch0/1/2 -> RGB
                float r = p.val.size() > 0 ? p.val[0] : 0.0f;
                float g = p.val.size() > 1 ? p.val[1] : 0.0f;
                float b = p.val.size() > 2 ? p.val[2] : 0.0f;
                auto cl = [](float u){ return (int)(std::min(1.0f, std::max(0.0f, u)) * 255); };
                col = IM_COL32(cl(r), cl(g), cl(b), 255);
            } else {
                int c = std::min(view.channel, (int)p.val.size() - 1);
                float t = (c >= 0) ? (p.val[c] - vlo) / vspan : 0.5f;
                col = heat(t);
            }
            float rad = (f.isGrid ? 3.0f : 4.0f);
            bool sel = (fi == view.pickedField && pi == view.picked);
            dl->AddCircleFilled(s, rad, col);
            dl->AddCircle(s, rad + 1.5f, sel ? IM_COL32(255, 240, 80, 255)
                                             : IM_COL32(0, 0, 0, 140), 0, 1.2f);
            float d2 = (s.x - mouse.x) * (s.x - mouse.x) + (s.y - mouse.y) * (s.y - mouse.y);
            if (d2 < bestD2) { bestD2 = d2; bestField = fi; bestPt = pi; }
        }
    }
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && bestD2 < 14.0f * 14.0f) {
        view.pickedField = bestField;
        view.picked = bestPt;
    }
    dl->PopClipRect();

    // inspector line for the picked sample
    if (view.pickedField >= 0 && view.pickedField < (int)fields.size()) {
        const FieldGeom& f = fields[view.pickedField];
        if (view.picked >= 0 && view.picked < (int)f.points.size()) {
            const FieldPoint& p = f.points[view.picked];
            std::string pos = "(";
            for (size_t k = 0; k < p.pos.size(); ++k)
                pos += (k ? ", " : "") + std::string(std::to_string(p.pos[k]));
            pos += ")";
            std::string val;
            for (size_t k = 0; k < p.val.size(); ++k) {
                std::string nm = (k < f.channels.size() && !f.channels[k].empty())
                                     ? f.channels[k] : ("c" + std::to_string(k));
                val += (k ? "  " : "") + nm + "=" + std::to_string(p.val[k]);
            }
            ImGui::TextWrapped("#%s[%d]  pos %s  |  %s",
                               f.id.c_str(), view.picked, pos.c_str(), val.c_str());
        }
    } else {
        ImGui::TextDisabled("(click a sample point to inspect its position & channels)");
    }
}

// --------------------------------------------------------------------------
// F4 — the mesh pane's z-buffered D3D11 renderer.
//
// The pane used to sort triangles back-to-front by centroid depth and hand them
// to ImGui's draw list — a painter's algorithm. That is simply wrong for
// interpenetrating geometry, which is exactly what loom's swept / blobby
// surfaces produce (two tubes crossing, a skin passing through a spine), and it
// re-sorted every triangle on the UI thread every frame. The viewer is already
// running on a D3D11 device, so the honest fix is a real depth buffer: upload
// the tessellation once into a vertex/index buffer, draw it into an offscreen
// render target that has a depth-stencil view, and show that target with
// ImGui::Image — the same trick the Render pane uses for its raymarch.
//
// Shading is two-sided lambert, 0.30 + 0.70*|n.z| with n in the rotated view
// basis, from the INTERPOLATED per-vertex normal (buildMeshPaneVerts above). It
// used to reconstruct a flat face normal per pixel from screen-space derivatives
// of the view position instead, which was wrong twice over: it could only ever
// show facets, and ddx/ddy work on 2x2 pixel quads, so a quad straddling a
// triangle edge differentiated across two triangles and produced a garbage normal
// — a one-pixel band of wrong shading along every edge, which is what loom's
// sweeps looked like ("jagged seams"). A real NORMAL element fixes both.
// The UV checker is likewise evaluated per-pixel at the interpolated UV instead of
// once at the triangle centroid — the whole point of a UV checker is to show UV
// distortion *within* a face, which a flat centroid sample cannot do.
//
// The target is 4x multisampled and resolved before the pane shows it: the mesh is
// a hard-edged silhouette against a flat background, where aliasing is at its most
// visible and MSAA at its cheapest.
// --------------------------------------------------------------------------
struct MeshGpu {
    // pipeline objects (created once, on first use)
    ID3D11VertexShader*      vs      = nullptr;
    ID3D11PixelShader*       ps      = nullptr;
    ID3D11InputLayout*       layout  = nullptr;
    ID3D11Buffer*            cb      = nullptr;
    ID3D11RasterizerState*   rsSolid = nullptr;
    ID3D11RasterizerState*   rsWire  = nullptr;
    ID3D11RasterizerState*   rsBack  = nullptr;   // solid, depth pushed a hair AWAY: lines lying on the
                                                  // surface still pass the depth test (groom grid outlines)
    ID3D11RasterizerState*   rsFront = nullptr;   // solid, depth pulled a hair TOWARD the eye: a surface lying on
                                                  // another (a groom's scalp on the skin) wins instead of flickering
    ID3D11DepthStencilState* dsSolid = nullptr;   // LESS, writes depth
    ID3D11DepthStencilState* dsWire  = nullptr;   // LESS_EQUAL, no depth write
    ID3D11BlendState*        blend   = nullptr;
    ID3D11BlendState*        blendNoColor = nullptr;   // colour writes off: a depth-only pre-pass (the groom
                                                       // tool's hidden-line grid outlines, 0.373.0)
    ID3D11SamplerState*      samp    = nullptr;
    bool                     pipeReady = false;

    // geometry (rebuilt only when the sidecar hands over a new tessellation)
    ID3D11Buffer* vb = nullptr;
    ID3D11Buffer* ib = nullptr;
    struct Range { UINT firstIndex = 0, indexCount = 0; INT baseVertex = 0; };
    std::vector<Range> ranges;      // one per MeshGeom, parallel to `meshes`
    unsigned geomGen = ~0u;         // MeshView::geomGen the buffers were built from
    bool     geomReady = false;
    float    mid[3] = { 0, 0, 0 };  // union-bounds centre / extents, baked with the upload
    float    ext = 1.0f, diag = 1.0f;

    // Offscreen colour + depth target, resized to the pane. Drawing goes into the
    // MULTISAMPLED pair (msTex/depthTex) and is resolved down into colorTex, which
    // is the single-sample texture ImGui samples — an MSAA texture cannot be bound
    // as a shader resource, so the resolve is not optional.
    ID3D11Texture2D*          colorTex = nullptr;   // resolve destination, SRV
    ID3D11ShaderResourceView* srv      = nullptr;
    ID3D11Texture2D*          msTex    = nullptr;   // MSAA colour, render target
    ID3D11RenderTargetView*   rtv      = nullptr;
    ID3D11Texture2D*          depthTex = nullptr;   // MSAA depth
    ID3D11DepthStencilView*   dsv      = nullptr;
    UINT samples = 1;               // what the device actually granted (4 if it can)
    int texW = 0, texH = 0;

    std::string err;                // non-empty => the pane says so instead of drawing

    using Vert = MeshPaneVert;   // position + uv + shading normal
    // Must match the cbuffer in the shader below (176 B, a multiple of 16). Zero-initialised it
    // draws exactly as before 0.374.0: no slab, no rims, no hue mode.
    struct CB {
        float mvp[16];
        float rot0[4], rot1[4], rot2[4];   // xyz = view-basis row, w = -(row . mid)
        float baseColor[4];
        float opts[4];                     // x = shade on, y = colour mode, z = hue flags (mode 4), w = lines per hue cycle
        float slab[4];                     // xyz = a plane's normal (world), w = n.p at its centre (groom slice, 0.374.0)
        float extra[4];                    // x = slab half-thickness (0: none), y = rim strength, z/w = depth range for hue
    };

    void releaseGeom() {
        if (vb) { vb->Release(); vb = nullptr; }
        if (ib) { ib->Release(); ib = nullptr; }
        ranges.clear();
        geomReady = false;
        geomGen = ~0u;
    }
    void releaseTargets() {
        if (srv)      { srv->Release();      srv = nullptr; }
        if (rtv)      { rtv->Release();      rtv = nullptr; }
        if (msTex)    { msTex->Release();    msTex = nullptr; }
        if (colorTex) { colorTex->Release(); colorTex = nullptr; }
        if (dsv)      { dsv->Release();      dsv = nullptr; }
        if (depthTex) { depthTex->Release(); depthTex = nullptr; }
        texW = texH = 0;
    }
    void release() {
        releaseGeom();
        releaseTargets();
        if (samp)    { samp->Release();    samp = nullptr; }
        if (blend)   { blend->Release();   blend = nullptr; }
        if (blendNoColor) { blendNoColor->Release(); blendNoColor = nullptr; }
        if (dsWire)  { dsWire->Release();  dsWire = nullptr; }
        if (dsSolid) { dsSolid->Release(); dsSolid = nullptr; }
        if (rsWire)  { rsWire->Release();  rsWire = nullptr; }
        if (rsBack)  { rsBack->Release();  rsBack = nullptr; }
        if (rsFront) { rsFront->Release(); rsFront = nullptr; }
        if (rsSolid) { rsSolid->Release(); rsSolid = nullptr; }
        if (cb)      { cb->Release();      cb = nullptr; }
        if (layout)  { layout->Release();  layout = nullptr; }
        if (ps)      { ps->Release();      ps = nullptr; }
        if (vs)      { vs->Release();      vs = nullptr; }
        pipeReady = false;
    }

    bool buildPipeline(ID3D11Device* dev) {
        if (pipeReady) return true;
        if (!dev) { err = "no D3D11 device"; return false; }

        static const char* kVS = R"HLSL(
cbuffer CB : register(b0) {
    row_major float4x4 mvp;
    float4 rot0, rot1, rot2;
    float4 baseColor;
    float4 opts;
    float4 slab;
    float4 extra;
};
struct VSIn  { float3 p : POSITION; float2 uv : TEXCOORD0; float3 n : NORMAL; };
struct VSOut { float4 pos : SV_Position; float3 vn : TEXCOORD1; float2 uv : TEXCOORD0; float3 wp : TEXCOORD2; };
VSOut main(VSIn i) {
    VSOut o;
    o.pos = mul(mvp, float4(i.p, 1.0));
    // shading normal rotated into the view basis. The rows are orthonormal, so the
    // basis is its own inverse-transpose and the normal transforms like a point
    // minus the translation (rot*.w, which positions do carry, is dropped here).
    o.vn  = float3(dot(rot0.xyz, i.n), dot(rot1.xyz, i.n), dot(rot2.xyz, i.n));
    o.uv  = i.uv;
    o.wp  = i.p;            // world position, for the groom tool's slab (the geometry is in world space)
    return o;
}
)HLSL";

        static const char* kPS = R"HLSL(
cbuffer CB : register(b0) {
    row_major float4x4 mvp;
    float4 rot0, rot1, rot2;
    float4 baseColor;
    float4 opts;
    float4 slab;
    float4 extra;
};
Texture2D    tex0  : register(t0);
SamplerState samp0 : register(s0);
struct VSOut { float4 pos : SV_Position; float3 vn : TEXCOORD1; float2 uv : TEXCOORD0; float3 wp : TEXCOORD2; };
// The groom tool's colours (0.374.0). None is red: red is the guides' own colour.
// Per contour line: six colours in turn, far apart, so neighbouring lines never match.
static const float3 kLineCol[6] = { float3(1.00, 0.55, 0.10), float3(1.00, 0.95, 0.25), float3(0.30, 0.95, 0.35),
                                    float3(0.20, 0.85, 1.00), float3(0.40, 0.50, 1.00), float3(0.95, 0.40, 1.00) };
// By depth: bright yellow near, through green, teal and blue, to a dim purple far -- the hue and
// the brightness both say how far away a thing is.
float3 depthRamp(float t) {
    const float3 c0 = float3(1.00, 0.92, 0.30), c1 = float3(0.55, 0.92, 0.30), c2 = float3(0.20, 0.78, 0.62),
                 c3 = float3(0.25, 0.48, 0.95), c4 = float3(0.42, 0.24, 0.66);
    float s = saturate(t) * 4.0;
    if (s < 1.0) return lerp(c0, c1, s);
    if (s < 2.0) return lerp(c1, c2, s - 1.0);
    if (s < 3.0) return lerp(c2, c3, s - 2.0);
    return lerp(c3, c4, s - 3.0);
}
float4 main(VSOut i) : SV_Target {
    // the groom tool's SLAB: only what lies within extra.x of the plane is drawn
    if (extra.x > 0.0 && abs(dot(slab.xyz, i.wp) - slab.w) > extra.x) discard;
    float3 base = baseColor.rgb;
    int mode = (int)opts.y;
    if (mode == 4) {
        // flag 1: a colour per contour line (uv.x = the line's plane index); flag 2: by depth, over
        // the depth range extra.z..extra.w; both: the line's colour, dimmer with depth
        int flags = (int)opts.z;
        float t = saturate((i.pos.z - extra.z) / max(extra.w - extra.z, 1e-6));
        if ((flags & 1) != 0) {
            int k = (int)floor(i.uv.x + 0.5);
            base = kLineCol[((k % 6) + 6) % 6];
            if ((flags & 2) != 0) base *= lerp(1.0, 0.32, t);
        } else if ((flags & 2) != 0) {
            base = depthRamp(t);
        }
    } else if (mode == 2) {
        // UV checker, per-pixel (8 cells across the unit square)
        float2 c = floor(i.uv * 8.0);
        float  s = frac((c.x + c.y) * 0.5);
        base = (s > 0.25) ? float3(210.0, 210.0, 220.0) / 255.0
                          : float3( 90.0,  95.0, 110.0) / 255.0;
    } else if (mode == 3) {
        // v is flipped because Texture::sampleRgb treats v=0 as the image BOTTOM
        // while the uploaded D3D texture has v=0 at its top row.
        base *= tex0.Sample(samp0, float2(i.uv.x, 1.0 - i.uv.y)).rgb;
    }
    float sh = 1.0;
    float a = baseColor.a;
    if (opts.x > 0.5) {
        float3 n = normalize(i.vn);
        sh = 0.30 + 0.70 * abs(n.z);          // two-sided lambert, headlight along z
    }
    if (extra.y > 0.0) {
        // BRIGHT EDGES (the groom tool's see-through parts, 0.374.0): opaque and lit where the
        // surface is seen edge-on, clear where it faces you -- every fold, curl and lock edge
        // becomes an outline, and what is inside stays visible (an x-ray / Fresnel look)
        float3 n = normalize(i.vn);
        float rim = pow(saturate(1.0 - abs(n.z)), 2.5) * extra.y;
        a  = lerp(a, 1.0, rim);
        sh = lerp(sh, 1.2, rim);
    }
    return float4(base * sh, a);
}
)HLSL";

        auto fail = [&](const char* what, ID3DBlob* e) {
            err = what;
            if (e) { err += ": "; err.append((const char*)e->GetBufferPointer()); e->Release(); }
            release();
            return false;
        };
        ID3DBlob* vsb = nullptr; ID3DBlob* psb = nullptr; ID3DBlob* eb = nullptr;
        if (FAILED(D3DCompile(kVS, strlen(kVS), nullptr, nullptr, nullptr, "main", "vs_4_0", 0, 0, &vsb, &eb)))
            return fail("mesh VS compile failed", eb);
        if (FAILED(dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs))) {
            vsb->Release(); return fail("CreateVertexShader failed", nullptr);
        }
        if (FAILED(D3DCompile(kPS, strlen(kPS), nullptr, nullptr, nullptr, "main", "ps_4_0", 0, 0, &psb, &eb))) {
            vsb->Release(); return fail("mesh PS compile failed", eb);
        }
        if (FAILED(dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps))) {
            vsb->Release(); psb->Release(); return fail("CreatePixelShader failed", nullptr);
        }
        psb->Release();
        const D3D11_INPUT_ELEMENT_DESC il[] = {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 20, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };
        HRESULT hr = dev->CreateInputLayout(il, 3, vsb->GetBufferPointer(), vsb->GetBufferSize(), &layout);
        vsb->Release();
        if (FAILED(hr)) return fail("CreateInputLayout failed", nullptr);

        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = sizeof(CB);
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(dev->CreateBuffer(&bd, nullptr, &cb))) return fail("CreateBuffer(cb) failed", nullptr);

        D3D11_RASTERIZER_DESC rd = {};
        rd.FillMode = D3D11_FILL_SOLID;
        rd.CullMode = D3D11_CULL_NONE;          // surfaces are drawn two-sided
        rd.DepthClipEnable = TRUE;
        rd.MultisampleEnable = TRUE;            // ignored when the target is 1x
        if (FAILED(dev->CreateRasterizerState(&rd, &rsSolid))) return fail("rasterizer(solid) failed", nullptr);
        // the same, biased AWAY from the eye (constant + slope-scaled, like a polygon offset): a
        // surface drawn with it lets the lines lying exactly on it through a LESS_EQUAL test
        // (biasing the surface, a triangle, is surer than biasing the lines)
        rd.DepthBias = 2000;
        rd.SlopeScaledDepthBias = 2.0f;
        if (FAILED(dev->CreateRasterizerState(&rd, &rsBack))) return fail("rasterizer(back) failed", nullptr);
        rd.DepthBias = -2000;
        rd.SlopeScaledDepthBias = -2.0f;
        if (FAILED(dev->CreateRasterizerState(&rd, &rsFront))) return fail("rasterizer(front) failed", nullptr);
        rd.FillMode = D3D11_FILL_WIREFRAME;
        // The wire pass draws the SAME triangles, so pull it a hair toward the eye;
        // LESS_EQUAL alone would still lose to rasterization rounding on the edges.
        rd.DepthBias = -800;
        rd.SlopeScaledDepthBias = -1.0f;
        if (FAILED(dev->CreateRasterizerState(&rd, &rsWire))) return fail("rasterizer(wire) failed", nullptr);

        D3D11_DEPTH_STENCIL_DESC dd = {};
        dd.DepthEnable = TRUE;
        dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        dd.DepthFunc = D3D11_COMPARISON_LESS;
        if (FAILED(dev->CreateDepthStencilState(&dd, &dsSolid))) return fail("depth state(solid) failed", nullptr);
        dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        dd.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
        if (FAILED(dev->CreateDepthStencilState(&dd, &dsWire))) return fail("depth state(wire) failed", nullptr);

        D3D11_BLEND_DESC bl = {};
        bl.RenderTarget[0].BlendEnable = TRUE;
        bl.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
        bl.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
        bl.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        bl.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
        bl.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
        bl.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        bl.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        if (FAILED(dev->CreateBlendState(&bl, &blend))) return fail("CreateBlendState failed", nullptr);
        bl.RenderTarget[0].BlendEnable = FALSE;
        bl.RenderTarget[0].RenderTargetWriteMask = 0;      // depth only
        if (FAILED(dev->CreateBlendState(&bl, &blendNoColor))) return fail("CreateBlendState(no colour) failed", nullptr);

        D3D11_SAMPLER_DESC sd = {};
        sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        sd.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
        sd.MaxLOD = D3D11_FLOAT32_MAX;
        if (FAILED(dev->CreateSamplerState(&sd, &samp))) return fail("CreateSamplerState failed", nullptr);

        err.clear();
        pipeReady = true;
        return true;
    }

    // Highest sample count <= 4 the device supports for BOTH the colour and depth
    // formats, so a driver that can't do 4x silently gets 2x or 1x rather than a
    // failed pane. 4x is where MSAA's quality/cost curve knees for a silhouette.
    static UINT pickSamples(ID3D11Device* dev) {
        for (UINT s = 4; s > 1; s >>= 1) {
            UINT qc = 0, qd = 0;
            if (FAILED(dev->CheckMultisampleQualityLevels(DXGI_FORMAT_R8G8B8A8_UNORM, s, &qc)) || !qc) continue;
            if (FAILED(dev->CheckMultisampleQualityLevels(DXGI_FORMAT_D32_FLOAT, s, &qd)) || !qd) continue;
            return s;
        }
        return 1;
    }

    bool ensureTargets(ID3D11Device* dev, int W, int H) {
        if (W < 1) W = 1;
        if (H < 1) H = 1;
        if (colorTex && texW == W && texH == H) return true;
        releaseTargets();
        samples = pickSamples(dev);

        D3D11_TEXTURE2D_DESC td = {};
        td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        // The resolve destination. It keeps BIND_RENDER_TARGET so the samples==1
        // path can render straight into it and skip the resolve entirely.
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &colorTex))) { err = "mesh colour target failed"; return false; }
        if (FAILED(dev->CreateShaderResourceView(colorTex, nullptr, &srv))) { releaseTargets(); err = "mesh SRV failed"; return false; }

        if (samples > 1) {
            td.SampleDesc.Count = samples;
            td.BindFlags = D3D11_BIND_RENDER_TARGET;   // MSAA can't also be an SRV
            if (FAILED(dev->CreateTexture2D(&td, nullptr, &msTex))) { releaseTargets(); err = "mesh MSAA target failed"; return false; }
            if (FAILED(dev->CreateRenderTargetView(msTex, nullptr, &rtv))) { releaseTargets(); err = "mesh RTV failed"; return false; }
        } else {
            if (FAILED(dev->CreateRenderTargetView(colorTex, nullptr, &rtv))) { releaseTargets(); err = "mesh RTV failed"; return false; }
        }

        td.SampleDesc.Count = samples;
        td.Format = DXGI_FORMAT_D32_FLOAT;
        td.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &depthTex))) { releaseTargets(); err = "mesh depth target failed"; return false; }
        if (FAILED(dev->CreateDepthStencilView(depthTex, nullptr, &dsv))) { releaseTargets(); err = "mesh DSV failed"; return false; }
        texW = W; texH = H;
        return true;
    }

    // One interleaved vertex buffer + one index buffer for the whole sidecar, with a
    // per-mesh (firstIndex, count, baseVertex) range so each mesh is still its own
    // draw call (it needs its own skin and tint).
    bool uploadGeometry(ID3D11Device* dev, const std::vector<MeshGeom>& meshes, unsigned gen) {
        releaseGeom();
        std::vector<Vert>     verts;
        std::vector<uint32_t> idx;
        std::vector<Vert>     mv;     // scratch, reused across meshes
        std::vector<uint32_t> mi_;
        ranges.resize(meshes.size());
        for (size_t mi = 0; mi < meshes.size(); ++mi) {
            const MeshGeom& m = meshes[mi];
            Range r;
            r.baseVertex = (INT)verts.size();
            r.firstIndex = (UINT)idx.size();
            // Shading normals are resolved here (crease-smoothed, which can split a
            // vertex), so this owns both the vertices and the indices for the mesh.
            buildMeshPaneVerts(m, mv, mi_);
            verts.insert(verts.end(), mv.begin(), mv.end());
            idx.insert(idx.end(), mi_.begin(), mi_.end());
            r.indexCount = (UINT)idx.size() - r.firstIndex;
            ranges[mi] = r;
        }
        // Union bounds, computed once with the upload rather than per frame: they
        // depend only on the tessellation, and an orbit must not re-scan 5M verts.
        float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
        for (const auto& m : meshes)
            for (int i = 0; i < m.nverts; ++i)
                for (int k = 0; k < 3; ++k) {
                    float v = m.verts[(size_t)i * 3 + k];
                    lo[k] = std::min(lo[k], v); hi[k] = std::max(hi[k], v);
                }
        ext = 1.0f; diag = 0.0f;
        for (int k = 0; k < 3; ++k) {
            float d = (hi[k] > lo[k]) ? (hi[k] - lo[k]) : 0.0f;
            ext = std::max(ext, d);
            diag += d * d;
            mid[k] = (hi[k] >= lo[k]) ? 0.5f * (lo[k] + hi[k]) : 0.0f;
        }
        diag = 0.5f * std::sqrt(diag) + 1e-3f;   // depth half-range in the rotated basis

        geomGen = gen;
        if (verts.empty() || idx.empty()) { geomReady = true; return true; }   // nothing to draw, but valid

        D3D11_BUFFER_DESC bd = {};
        D3D11_SUBRESOURCE_DATA sd = {};
        bd.ByteWidth = (UINT)(verts.size() * sizeof(Vert));
        bd.Usage = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        sd.pSysMem = verts.data();
        if (FAILED(dev->CreateBuffer(&bd, &sd, &vb))) { err = "mesh vertex buffer failed"; releaseGeom(); return false; }
        bd.ByteWidth = (UINT)(idx.size() * sizeof(uint32_t));
        bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
        sd.pSysMem = idx.data();
        if (FAILED(dev->CreateBuffer(&bd, &sd, &ib))) { err = "mesh index buffer failed"; releaseGeom(); return false; }
        geomGen = gen;
        geomReady = true;
        return true;
    }
};

// --------------------------------------------------------------------------
// F4 — mesh pane: SweptMesh tessellated surfaces, plus Strand fibers tubed from
// their centreline, as a shaded, z-buffered triangle mesh. Orbiting the 3 spatial
// dims is a view-only re-projection (no re-tessellation, exactly as the F4 rule
// specifies for isometries of the shown dims). Colour: flat lambert shading,
// per-object tint, a UV checker, or the material's real skin sampled per-pixel at
// the interpolated mesh UVs.
// --------------------------------------------------------------------------
struct MeshView {
    float yaw = 0.6f, pitch = 0.4f, zoom = 1.0f;
    bool  shade = true;         // flat lambert lighting
    bool  wire = false;         // wireframe overlay
    int   colorBy = 3;          // 0 grey, 1 per-object tint, 2 UV checker, 3 texture
    // Bumped whenever loom hands over a NEW tessellation; the GPU buffers are
    // rebuilt only when it changes, so an orbit costs nothing but a cbuffer write.
    unsigned geomGen = 0;
    MeshGpu  gpu;
};

static bool drawMeshPane(const std::vector<MeshGeom>& meshes, MeshView& view,
                         const SkinLib& skins, LivePanel* live,
                         ID3D11Device* dev, ID3D11DeviceContext* ctx) {
    bool swept = false;   // the parameter axis moved -> the surface must be re-baked
    ImGui::TextUnformatted("Meshes - drag to orbit, wheel to zoom (view-only re-projection)");
    liveSweepHint(live);
    ImGui::Checkbox("shade", &view.shade); ImGui::SameLine();
    ImGui::Checkbox("wireframe", &view.wire); ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    const char* cmodes[] = { "grey", "per-object tint", "UV checker", "texture" };
    ImGui::Combo("colour", &view.colorBy, cmodes, 4);
    if (view.colorBy == 3) {
        ImGui::SameLine();
        if (skins.skins.empty())
            ImGui::TextDisabled("(no textures in the sidecar - falls back to grey)");
        else
            ImGui::TextDisabled("(%d/%d skin(s) ready)", skins.nOk, (int)skins.skins.size());
    }

    // Reserve the footer rows (stats + one line per broken skin) BEFORE sizing the
    // canvas — the canvas otherwise eats the whole remaining height and pushes them
    // out of the window, which is exactly where a skin's error message must not go.
    int footer = 1;
    for (const auto& sk : skins.skins) if (!sk.err.empty()) ++footer;
    ImVec2 avail = ImGui::GetContentRegionAvail();
    avail.y -= footer * ImGui::GetTextLineHeightWithSpacing();
    if (avail.y < 80.0f) avail.y = 80.0f;
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("mesh_canvas", avail,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    bool hovered = ImGui::IsItemHovered();
    swept = liveSweepDrag(live);   // right-drag: rotate INTO the parameter dimension
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        ImVec2 d = ImGui::GetIO().MouseDelta;
        view.yaw   += d.x * 0.01f;
        view.pitch += d.y * 0.01f;
    }
    if (hovered) { float w = ImGui::GetIO().MouseWheel; if (w != 0.0f) view.zoom *= (1.0f + w * 0.1f); }
    if (view.zoom < 0.05f) view.zoom = 0.05f;

    MeshGpu& gpu = view.gpu;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 br(origin.x + avail.x, origin.y + avail.y);

    // Pipeline once, geometry once per tessellation, targets once per pane size.
    bool ok = gpu.buildPipeline(dev);
    if (ok && (!gpu.geomReady || gpu.geomGen != view.geomGen))
        ok = gpu.uploadGeometry(dev, meshes, view.geomGen);
    if (ok) ok = gpu.ensureTargets(dev, (int)(avail.x + 0.5f), (int)(avail.y + 0.5f));

    if (ok) {
        // ---- the orthographic orbit projection, as one 4x4 -------------------
        // Rows of the rotation taking a world point to (X screen-right, Y up,
        // Z toward the viewer) — the exact basis the old CPU projector used.
        float cy = std::cos(view.yaw),   sy = std::sin(view.yaw);
        float cx = std::cos(view.pitch), sx = std::sin(view.pitch);
        const float R[3][3] = {
            {  cy,        0.0f,  sy      },
            {  sx * sy,   cx,   -sx * cy },
            { -cx * sy,   sx,    cx * cy },
        };
        float scale = 0.42f * std::min(avail.x, avail.y) / (0.5f * gpu.ext + 1e-3f);
        float s  = scale * view.zoom;
        // The pane's own pixel box IS the render target, so the screen mapping
        // collapses to a pure scale: the centre of the box is NDC (0,0).
        float ax = (avail.x > 0.0f) ? 2.0f * s / avail.x : 0.0f;
        float ay = (avail.y > 0.0f) ? 2.0f * s / avail.y : 0.0f;
        float kz = 0.5f / gpu.diag;      // rotated Z in [-diag,+diag] -> depth 0(near)..1(far)
        auto dotMid = [&](int r) {
            return R[r][0] * gpu.mid[0] + R[r][1] * gpu.mid[1] + R[r][2] * gpu.mid[2];
        };
        MeshGpu::CB c = {};
        const float rowScale[3] = { ax, ay, -kz };
        for (int r = 0; r < 3; ++r)
            for (int k = 0; k < 3; ++k) c.mvp[r * 4 + k] = rowScale[r] * R[r][k];
        c.mvp[0 * 4 + 3] = -ax * dotMid(0);
        c.mvp[1 * 4 + 3] = -ay * dotMid(1);
        c.mvp[2 * 4 + 3] =  kz * dotMid(2) + 0.5f;
        c.mvp[3 * 4 + 3] = 1.0f;
        for (int r = 0; r < 3; ++r) {
            float* dst = (r == 0) ? c.rot0 : (r == 1) ? c.rot1 : c.rot2;
            dst[0] = R[r][0]; dst[1] = R[r][1]; dst[2] = R[r][2]; dst[3] = -dotMid(r);
        }

        auto setCB = [&](const float rgba[4], float shadeOn, float mode) {
            for (int k = 0; k < 4; ++k) c.baseColor[k] = rgba[k];
            c.opts[0] = shadeOn; c.opts[1] = mode; c.opts[2] = c.opts[3] = 0.0f;
            D3D11_MAPPED_SUBRESOURCE ms;
            if (ctx->Map(gpu.cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms) == S_OK) {
                std::memcpy(ms.pData, &c, sizeof(c));
                ctx->Unmap(gpu.cb, 0);
            }
        };

        // ---- render the pane offscreen, with a real depth buffer -------------
        const float clearCol[4] = { 14 / 255.0f, 16 / 255.0f, 20 / 255.0f, 1.0f };
        ctx->ClearRenderTargetView(gpu.rtv, clearCol);
        ctx->ClearDepthStencilView(gpu.dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);
        if (gpu.vb && gpu.ib) {
            ID3D11RenderTargetView* rtvs[1] = { gpu.rtv };
            ctx->OMSetRenderTargets(1, rtvs, gpu.dsv);
            D3D11_VIEWPORT vp = {};
            vp.Width = (float)gpu.texW; vp.Height = (float)gpu.texH; vp.MaxDepth = 1.0f;
            ctx->RSSetViewports(1, &vp);
            UINT stride = sizeof(MeshGpu::Vert), voff = 0;
            ctx->IASetInputLayout(gpu.layout);
            ctx->IASetVertexBuffers(0, 1, &gpu.vb, &stride, &voff);
            ctx->IASetIndexBuffer(gpu.ib, DXGI_FORMAT_R32_UINT, 0);
            ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx->VSSetShader(gpu.vs, nullptr, 0);
            ctx->PSSetShader(gpu.ps, nullptr, 0);
            ctx->GSSetShader(nullptr, nullptr, 0);
            ctx->HSSetShader(nullptr, nullptr, 0);
            ctx->DSSetShader(nullptr, nullptr, 0);
            ctx->VSSetConstantBuffers(0, 1, &gpu.cb);
            ctx->PSSetConstantBuffers(0, 1, &gpu.cb);
            ctx->PSSetSamplers(0, 1, &gpu.samp);
            const float bf[4] = { 0, 0, 0, 0 };
            ctx->OMSetBlendState(gpu.blend, bf, 0xffffffff);
            ctx->RSSetState(gpu.rsSolid);
            ctx->OMSetDepthStencilState(gpu.dsSolid, 0);

            static const float tints[4][3] = {
                { 150 / 255.0f, 190 / 255.0f, 235 / 255.0f },
                { 235 / 255.0f, 175 / 255.0f, 130 / 255.0f },
                { 160 / 255.0f, 225 / 255.0f, 165 / 255.0f },
                { 225 / 255.0f, 155 / 255.0f, 200 / 255.0f },
            };
            size_t n = std::min(gpu.ranges.size(), meshes.size());
            for (size_t mi = 0; mi < n; ++mi) {
                const MeshGpu::Range& r = gpu.ranges[mi];
                if (!r.indexCount) continue;
                // Which skin this mesh wears (mesh -> material -> texture). A mesh
                // with no UVs can't be textured however good its skin, so it stays grey.
                ID3D11ShaderResourceView* skinSrv = nullptr;
                if (view.colorBy == 3) {
                    const Skin* sk = skins.skinFor(meshes[mi].material);
                    if (sk && sk->srv && (int)meshes[mi].uvs.size() >= 2 * meshes[mi].nverts)
                        skinSrv = sk->srv;
                }
                float rgba[4] = { 180 / 255.0f, 185 / 255.0f, 195 / 255.0f, 1.0f };
                float mode = 0.0f;
                if (skinSrv) {
                    // White base modulated by the lambert term, so the shading scales
                    // the skin instead of replacing it (what the old vertex colour did).
                    rgba[0] = rgba[1] = rgba[2] = 1.0f;
                    mode = 3.0f;
                } else if (view.colorBy == 1) {
                    for (int k = 0; k < 3; ++k) rgba[k] = tints[mi % 4][k];
                } else if (view.colorBy == 2) {
                    mode = 2.0f;
                }
                setCB(rgba, view.shade ? 1.0f : 0.0f, mode);
                ID3D11ShaderResourceView* srvs[1] = { skinSrv };
                ctx->PSSetShaderResources(0, 1, srvs);
                ctx->DrawIndexed(r.indexCount, r.firstIndex, r.baseVertex);
            }

            if (view.wire) {
                // A second, depth-tested wireframe pass: nearer faces hide farther
                // edges for real now, instead of relying on the fill/wire interleave
                // that the painter's-algorithm version needed.
                ctx->RSSetState(gpu.rsWire);
                ctx->OMSetDepthStencilState(gpu.dsWire, 0);
                ID3D11ShaderResourceView* none[1] = { nullptr };
                ctx->PSSetShaderResources(0, 1, none);
                const float wireCol[4] = { 30 / 255.0f, 30 / 255.0f, 36 / 255.0f, 120 / 255.0f };
                setCB(wireCol, 0.0f, 0.0f);
                for (size_t mi = 0; mi < n; ++mi) {
                    const MeshGpu::Range& r = gpu.ranges[mi];
                    if (r.indexCount) ctx->DrawIndexed(r.indexCount, r.firstIndex, r.baseVertex);
                }
            }

            // Unbind before ImGui samples this very texture as an SRV later in the frame.
            ID3D11ShaderResourceView* none[1] = { nullptr };
            ctx->PSSetShaderResources(0, 1, none);
            ID3D11RenderTargetView* noRtv[1] = { nullptr };
            ctx->OMSetRenderTargets(1, noRtv, nullptr);
        }
        // Collapse the multisampled target into the single-sample texture ImGui
        // draws. Outside the vb/ib guard so an empty scene still shows the clear.
        if (gpu.msTex && gpu.colorTex)
            ctx->ResolveSubresource(gpu.colorTex, 0, gpu.msTex, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
    }

    if (ok && gpu.srv) {
        dl->AddImage((ImTextureID)(intptr_t)gpu.srv, origin, br);
    } else {
        dl->AddRectFilled(origin, br, IM_COL32(14, 16, 20, 255));
        if (!gpu.err.empty())
            dl->AddText(ImVec2(origin.x + 8.0f, origin.y + 8.0f),
                        IM_COL32(240, 140, 110, 255), gpu.err.c_str());
    }

    int totalTris = 0, totalV = 0;
    for (const auto& m : meshes) { totalTris += m.nfaces; totalV += m.nverts; }
    ImGui::Text("%d mesh(es), %d verts, %d tris", (int)meshes.size(), totalV, totalTris);

    // Any skin that failed to decode is a real authoring error (a missing file, a bad
    // formula) — surface it here rather than silently drawing grey.
    for (const auto& sk : skins.skins)
        if (!sk.err.empty())
            ImGui::TextColored(ImVec4(0.95f, 0.55f, 0.45f, 1.0f),
                               "skin '%s' (%s): %s", sk.name.c_str(),
                               sk.kind.c_str(), sk.err.c_str());
    return swept;
}

// --------------------------------------------------------------------------
// The panels
// --------------------------------------------------------------------------
static void drawObjectsPanel(const Sidecar& sc) {
    const minijson::Value* objs = sc.arr("objects");
    if (!objs) { ImGui::TextDisabled("(no objects)"); return; }
    if (ImGui::BeginTable("objects", 4,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupColumn("name");
        ImGui::TableSetupColumn("kind");
        ImGui::TableSetupColumn("material");
        ImGui::TableSetupColumn("datasets");
        ImGui::TableHeadersRow();
        for (const auto& o : objs->arr) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const minijson::Value* nm = o.find("name");
            ImGui::TextUnformatted(nm && nm->isString() ? nm->str.c_str() : "-");
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(o.find("kind") ? o.find("kind")->asString("-").c_str() : "-");
            ImGui::TableNextColumn();
            const minijson::Value* mt = o.find("material");
            ImGui::TextUnformatted(mt && mt->isString() ? mt->str.c_str() : "-");
            ImGui::TableNextColumn();
            const minijson::Value* dv = o.find("datasets");
            if (dv && dv->isArray() && !dv->arr.empty()) {
                std::string s;
                for (size_t i = 0; i < dv->arr.size(); ++i)
                    s += (i ? "," : "") + scalarStr(&dv->arr[i]);
                ImGui::TextUnformatted(s.c_str());
            } else {
                ImGui::TextDisabled("-");
            }
        }
        ImGui::EndTable();
    }
}

static void drawDatasetsPanel(const Sidecar& sc) {
    const minijson::Value* ds = sc.arr("datasets");
    if (!ds) { ImGui::TextDisabled("(no datasets)"); return; }
    if (ImGui::BeginTable("datasets", 3,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupColumn("id");
        ImGui::TableSetupColumn("kind");
        ImGui::TableSetupColumn("detail");
        ImGui::TableHeadersRow();
        for (const auto& d : ds->arr) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(scalarStr(d.find("id")).c_str());
            ImGui::TableNextColumn();
            std::string kind = d.find("kind") ? d.find("kind")->asString("-") : "-";
            ImGui::TextUnformatted(kind.c_str());
            ImGui::TableNextColumn();
            char buf[128] = {0};
            if (kind == "grid") {
                const minijson::Value* sh = d.find("shape");
                std::string s = "shape=[";
                if (sh && sh->isArray())
                    for (size_t i = 0; i < sh->arr.size(); ++i) {
                        char t[24]; std::snprintf(t, sizeof t, "%d%s",
                            sh->arr[i].asInt(0), i + 1 < sh->arr.size() ? "," : "");
                        s += t;
                    }
                s += "]";
                ImGui::TextUnformatted(s.c_str());
            } else if (kind == "path" || kind == "tracked_path") {
                std::snprintf(buf, sizeof buf, "count=%d dim=%d %s",
                    d.intAt("count", 0), d.intAt("dim", 0),
                    (d.find("closed") && d.find("closed")->asBool(false)) ? "closed" : "open");
                ImGui::TextUnformatted(buf);
            } else if (kind == "scatter") {
                std::snprintf(buf, sizeof buf, "count=%d dim=%d",
                    d.intAt("count", 0), d.intAt("dim", 0));
                ImGui::TextUnformatted(buf);
            } else {
                ImGui::TextUnformatted("-");
            }
        }
        ImGui::EndTable();
    }
}

static void drawScenePanel(const Sidecar& sc) {
    const minijson::Value* cam = sc.root.find("camera");
    if (cam && cam->isObject())
        ImGui::Text("camera: %s", cam->find("class") ? cam->find("class")->asString("?").c_str() : "?");
    const minijson::Value* frame = sc.root.find("frame");
    if (frame && frame->isObject())
        ImGui::Text("frame: %d / %d", frame->intAt("frame", 0), frame->intAt("frames", 0));
    ImGui::Text("sidecar version: %d", sc.root.intAt("version", 0));
    const minijson::Value* lights = sc.arr("lights");
    if (lights) {
        ImGui::Separator();
        ImGui::Text("lights: %d", (int)lights->arr.size());
        for (const auto& l : lights->arr)
            ImGui::BulletText("%s", l.find("kind") ? l.find("kind")->asString("?").c_str() : "?");
    }
    const minijson::Value* dag = sc.root.find("dag");
    if (dag && dag->isObject()) {
        const minijson::Value* nodes = dag->find("nodes");
        const minijson::Value* edges = dag->find("edges");
        ImGui::Separator();
        ImGui::Text("DAG: %d nodes, %d edges",
            nodes && nodes->isArray() ? (int)nodes->arr.size() : 0,
            edges && edges->isArray() ? (int)edges->arr.size() : 0);
    }
}

// --------------------------------------------------------------------------
// F5 — modulator-DAG panel (imnodes). Each node shows its op + stable id; each
// edge is a link into the destination's labelled parameter pin (so you can read
// which input of a node's function each upstream modulator feeds).
//
// E5 axis annotation (sidecar v2). A modulator is typed by its **free axes**, and
// an influence edge carries a pin/mod mode + gain — neither is recoverable from
// the op name alone, so loom projects both into the sidecar and we surface them:
//   * a node shows its axis set as `{s,t}` (∅ for a constant, which broadcasts
//     everywhere) plus the extras that make the model legible — a Target's
//     declared quantity kind, the axis a Reduce consumes, and, on the two bridge
//     nodes, the value-site's scope (`t from clock, s pinned`).
//   * an edge into a Target reads `mod[0] x0.8` / `pin[1] x0.25` on its input pin
//     instead of an anonymous `in0`.
// Everything is optional: a v1 sidecar simply renders as before.
// --------------------------------------------------------------------------
struct DagNode {
    int id = 0;
    std::string op, label;
    std::string axes;      // "{s,t}" / "{}"     — empty when unannotated
    std::string detail;    // "gain target" / "reduce s (sum)" / site scope
};
struct DagEdge {
    int src = 0, dst = 0;
    std::string param;
    std::string mode;      // "pin" / "mod" — empty for a plain input edge
    double gain = 1.0;
};

// "{s,t}" from a JSON array of axis names ("{}" when empty — the broadcast case).
static std::string axisSetStr(const minijson::Value* v) {
    if (!v || !v->isArray()) return "";
    std::string s = "{";
    for (size_t i = 0; i < v->arr.size(); ++i) {
        if (i) s += ",";
        s += v->arr[i].asString("?");
    }
    return s + "}";
}

// The one-line "what kind of node is this, in E5 terms" caption.
static std::string dagDetail(const minijson::Value& n) {
    const minijson::Value* site = n.find("site");
    if (site && site->isString()) {                    // Lower / LowerVec bridge
        std::string s = scalarStr(n.find("clock_axis"), "t") + " from clock";
        std::string bound = axisSetStr(n.find("bound_axes"));
        if (!bound.empty() && bound != "{}") s += ", " + bound + " pinned";
        std::string src = axisSetStr(n.find("source_axes"));
        if (!src.empty()) s += "  <- " + src;
        return s;
    }
    if (const minijson::Value* k = n.find("target_kind"))
        return k->asString("?") + " target (neutral "
               + scalarStr(n.find("neutral"), "?") + ")";
    if (const minijson::Value* r = n.find("reduces"))
        return "reduce " + r->asString("?") + " ("
               + scalarStr(n.find("reduce_op"), "?") + ", "
               + scalarStr(n.find("samples"), "?") + " samples)";
    if (const minijson::Value* c = n.find("channel"))
        return "channel " + c->asString("?");
    return "";
}
struct DagGraph {
    std::vector<DagNode> nodes;
    std::vector<DagEdge> edges;
    std::vector<ImVec2>  pos;             // grid-space position per node (parallel to nodes)
    std::vector<ImVec2>  realSize;        // node rects imnodes actually produced
    bool   sizesValid = false;            // realSize populated (after one drawn frame)
    ImVec2 extent = ImVec2(0.0f, 0.0f);   // laid-out bounding box, grid space
    float  measuredFont = 0.0f;           // font size the measure ran at (re-measure on DPI change)
    float  measuredAvailH = -1.0f;        // pane height the wrap was measured against
    float  zoom = 1.0f;                   // font/padding scale — a real zoom (imnodes has none)
    int    fitFrames = 0;                 // frames left of an iterative "fit the whole graph" solve
    bool   fitted = false;                // last fit converged: keep it fitted across resizes
    float  fitCanvasW = 0.0f;             // canvas width that fit was solved for
    bool laidOut  = false;                // grid positions applied to imnodes yet?
    bool maximized = false;               // show the graph full-window instead of in the side column
};

// imnodes id namespaces (node ids from loom are small; keep pins/links clear of them)
static const int DAG_OUT_BASE = 1 << 20;   // output pin id = base + node id
static const int DAG_IN_BASE  = 1 << 21;   // input  pin id = base + edge index

static DagGraph collectDag(const Sidecar& sc) {
    DagGraph g;
    const minijson::Value* dag = sc.root.find("dag");
    if (!dag || !dag->isObject()) return g;
    const minijson::Value* nodes = dag->find("nodes");
    const minijson::Value* edges = dag->find("edges");
    if (nodes && nodes->isArray())
        for (const auto& n : nodes->arr) {
            DagNode dn;
            dn.id     = n.intAt("id", 0);
            dn.op     = n.find("op") ? n.find("op")->asString("?") : "?";
            dn.label  = scalarStr(n.find("label"), "");
            dn.axes   = axisSetStr(n.find("axes"));      // "" when unannotated
            dn.detail = dagDetail(n);
            g.nodes.push_back(std::move(dn));
        }
    if (edges && edges->isArray())
        for (const auto& e : edges->arr) {
            DagEdge de;
            de.src   = e.intAt("src", 0);
            de.dst   = e.intAt("dst", 0);
            de.param = scalarStr(e.find("param"), "in");
            const minijson::Value* m = e.find("mode");
            if (m && m->isString()) { de.mode = m->str; de.gain = e.numAt("gain", 1.0); }
            g.edges.push_back(std::move(de));
        }
    return g;
}

// Size a node box the way imnodes will: the node grows to its widest ImGui item and
// to the sum of its rows. Mirroring the exact lines drawDagPanel emits (title, label,
// axes, detail, one row per input pin) means the layout below can leave real gaps
// instead of the old fixed 230x95 grid pitch, which overlapped as soon as a node had
// several input pins and broke outright at >100% DPI (bigger text, same pitch).
static ImVec2 dagNodeSize(const DagGraph& g, const DagNode& n,
                          const std::vector<int>* inEdges) {
    const float lineH = ImGui::GetTextLineHeightWithSpacing();
    char buf[512];
    const bool titled = !n.label.empty() && n.label != n.op;
    snprintf(buf, sizeof buf, "%s  #%d", n.op.c_str(), n.id);
    float w = ImGui::CalcTextSize(buf).x;
    int   rows = 1;                                    // title bar
    if (titled) {
        snprintf(buf, sizeof buf, "= %s", n.label.c_str());
        w = std::max(w, ImGui::CalcTextSize(buf).x); ++rows;
    }
    if (!n.axes.empty()) {
        snprintf(buf, sizeof buf, "axes %s", n.axes.c_str());
        w = std::max(w, ImGui::CalcTextSize(buf).x); ++rows;
    }
    if (!n.detail.empty()) {
        w = std::max(w, ImGui::CalcTextSize(n.detail.c_str()).x); ++rows;
    }
    if (inEdges)
        for (int ei : *inEdges) {
            const DagEdge& e = g.edges[ei];
            if (e.mode.empty()) snprintf(buf, sizeof buf, "%s", e.param.c_str());
            else                snprintf(buf, sizeof buf, "%s x%g", e.param.c_str(), e.gain);
            w = std::max(w, ImGui::CalcTextSize(buf).x); ++rows;
        }
    const ImVec2 pad = ImNodes::GetStyle().NodePadding;
    // + pin circles either side, + the (empty) output attribute's own row
    return ImVec2(w + pad.x * 2.0f + lineH * 1.6f,
                  rows * lineH + pad.y * 2.0f + lineH * 0.5f);
}

// Longest-path layering (level = max over incoming edges of src level + 1) so the
// graph reads left→right from leaves (constants/oscillators) to the params they drive.
// Purely a measurement pass — no imnodes calls, so it can run before the editor
// begins and hand the caller a real extent to size the pane with.
//
// `availH` is the height the caller can actually show. A level wider than that wraps
// into side-by-side sub-columns instead of running off the bottom: a typical loom DAG
// is mostly leaves (a `field.viewer.json` here has 60 of its 78 nodes at level 0), so
// the old one-column-per-level layout was ~6600 px tall and no pane could ever show it.
static void measureDag(DagGraph& g, float availH) {
    g.pos.assign(g.nodes.size(), ImVec2(0.0f, 0.0f));
    g.extent = ImVec2(0.0f, 0.0f);
    g.measuredFont  = ImGui::GetFontSize();
    g.measuredAvailH = availH;
    g.laidOut = false;                                  // positions still need applying
    if (g.nodes.empty()) return;

    std::unordered_map<int, std::vector<int>> incoming;  // dst -> [src...]
    for (const auto& e : g.edges) incoming[e.dst].push_back(e.src);
    std::unordered_map<int, int> level;
    std::unordered_map<int, int> visiting;
    std::function<int(int)> lvl = [&](int id) -> int {
        auto it = level.find(id);
        if (it != level.end()) return it->second;
        if (visiting[id]) return 0;          // cycle guard (shouldn't happen in a DAG)
        visiting[id] = 1;
        int mx = 0;
        auto in = incoming.find(id);
        if (in != incoming.end())
            for (int s : in->second) mx = std::max(mx, lvl(s) + 1);
        visiting[id] = 0;
        level[id] = mx;
        return mx;
    };
    std::unordered_map<int, std::vector<int>> inEdges;   // node id -> [edge index...]
    for (int i = 0; i < (int)g.edges.size(); ++i) inEdges[g.edges[i].dst].push_back(i);

    std::map<int, std::vector<int>> byLevel;             // level -> [node index...]
    for (size_t i = 0; i < g.nodes.size(); ++i) byLevel[lvl(g.nodes[i].id)].push_back((int)i);

    // Real rects once imnodes has drawn a frame; the text estimate only has to carry
    // the very first layout (it can't know imnodes' own padding or the DPI scaling).
    std::vector<ImVec2> sz(g.nodes.size());
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        if (g.sizesValid && g.realSize[i].x > 0.0f && g.realSize[i].y > 0.0f) {
            sz[i] = g.realSize[i];
            continue;
        }
        auto it = inEdges.find(g.nodes[i].id);
        sz[i] = dagNodeSize(g, g.nodes[i], it == inEdges.end() ? nullptr : &it->second);
    }

    const float lineH  = ImGui::GetTextLineHeightWithSpacing();
    const float colGap = lineH * 2.2f, rowGap = lineH * 0.9f;
    const float budget = std::max(availH, lineH * 8.0f);   // never wrap after one node
    float x = 0.0f;
    for (const auto& lv : byLevel) {
        float y = 0.0f, colW = 0.0f;
        for (int idx : lv.second) {
            if (y > 0.0f && y + sz[idx].y > budget) {      // wrap into a sub-column
                x += colW + colGap;
                y = 0.0f; colW = 0.0f;
            }
            g.pos[idx] = ImVec2(x, y);
            y += sz[idx].y + rowGap;
            colW = std::max(colW, sz[idx].x);
            g.extent.y = std::max(g.extent.y, y - rowGap);
        }
        g.extent.x = std::max(g.extent.x, x + colW);
        x += colW + colGap;
    }
}

// Re-measure only when the graph, the text metrics (DPI / font scale) or the height
// we have to fill changed.
static void dagEnsureMeasured(DagGraph& g, float availH) {
    if (g.pos.size() != g.nodes.size() || g.measuredFont != ImGui::GetFontSize() ||
        std::fabs(g.measuredAvailH - availH) > 1.0f)
        measureDag(g, availH);
}

// Non-graph vertical cost of the pane: child border/padding + the hint line + slack.
static float dagChrome() {
    return ImGui::GetStyle().WindowPadding.y * 2.0f + ImGui::GetTextLineHeightWithSpacing() * 2.0f;
}

// availH < 0 means "wrap to whatever this canvas actually is" — used by the maximized
// view, where the canvas is a function of the window alone, so measuring against it
// can't feed back into the pane's own size.
static void drawDagPanel(DagGraph& g, float availH) {
    if (g.nodes.empty()) { ImGui::TextDisabled("(no modulator DAG)"); return; }
    ImGui::TextDisabled("%d nodes, %d edges - drag to pan, wheel to zoom (%.0f%%)",
                        (int)g.nodes.size(), (int)g.edges.size(), g.zoom * 100.0f);
    const float canvasW = ImGui::GetContentRegionAvail().x;
    const float canvasH = ImGui::GetContentRegionAvail().y;
    if (availH < 0.0f)   // maximized: wrap to the canvas we were actually given
        availH = canvasH - ImGui::GetTextLineHeightWithSpacing();

    // Wheel zoom. imnodes has no zoom of its own, but scaling the font and the paddings
    // shrinks the nodes for real, and the layout follows because it re-wraps from the
    // rects imnodes reports. The pane is NoScrollbar|NoScrollWithMouse, so the wheel is
    // ours and never scrolls the column behind it.
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows)) {
        const float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f) {
            g.zoom = std::min(std::max(g.zoom * std::pow(1.12f, wheel), 0.15f), 3.0f);
            g.fitFrames = 0;
            g.fitted = false;             // the user is driving the zoom now
        }
    }
    // A pane resize invalidates a previous fit — re-solve so "show me all of it" stays
    // true when the window changes size or the panel is docked/maximized.
    if (g.fitted && (std::fabs(g.measuredAvailH - availH) > 1.0f ||
                     std::fabs(g.fitCanvasW - canvasW) > 1.0f))
        g.fitFrames = 16;
    dagEnsureMeasured(g, availH);
    // per-node incoming edges (each becomes a labelled input pin)
    std::unordered_map<int, std::vector<int>> inEdges;   // node id -> [edge index...]
    for (int i = 0; i < (int)g.edges.size(); ++i) inEdges[g.edges[i].dst].push_back(i);

    const ImGuiStyle& st = ImGui::GetStyle();
    const ImVec2 nodePad = ImNodes::GetStyle().NodePadding;
    ImGui::PushFont(nullptr, st.FontSizeBase * g.zoom);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                        ImVec2(st.ItemSpacing.x * g.zoom, st.ItemSpacing.y * g.zoom));
    ImNodes::PushStyleVar(ImNodesStyleVar_NodePadding,
                          ImVec2(nodePad.x * g.zoom, nodePad.y * g.zoom));
    ImNodes::BeginNodeEditor();
    if (!g.laidOut) {               // must be inside Begin/EndNodeEditor
        for (size_t i = 0; i < g.nodes.size(); ++i)
            ImNodes::SetNodeGridSpacePos(g.nodes[i].id, g.pos[i]);
    }
    for (const auto& n : g.nodes) {
        ImNodes::BeginNode(n.id);
        ImNodes::BeginNodeTitleBar();
        if (!n.label.empty() && n.label != n.op)
            ImGui::Text("%s  #%d", n.op.c_str(), n.id);
        else
            ImGui::Text("%s #%d", n.op.c_str(), n.id);
        ImNodes::EndNodeTitleBar();
        if (!n.label.empty() && n.label != n.op)
            ImGui::TextDisabled("= %s", n.label.c_str());
        if (!n.axes.empty())            // E5: the node's free axes
            ImGui::TextDisabled("axes %s", n.axes.c_str());
        if (!n.detail.empty())          // target kind / reduced axis / site scope
            ImGui::TextDisabled("%s", n.detail.c_str());
        // one labelled input pin per incoming edge (the param it feeds; an E5
        // influence edge also shows its pin/mod mode and gain)
        auto it = inEdges.find(n.id);
        if (it != inEdges.end())
            for (int ei : it->second) {
                const DagEdge& e = g.edges[ei];
                ImNodes::BeginInputAttribute(DAG_IN_BASE + ei);
                if (e.mode.empty())
                    ImGui::TextUnformatted(e.param.c_str());
                else            // param is already "mod[i]"/"pin[i]"; add the gain
                    ImGui::Text("%s x%g", e.param.c_str(), e.gain);
                ImNodes::EndInputAttribute();
            }
        ImNodes::BeginOutputAttribute(DAG_OUT_BASE + n.id);
        ImNodes::EndOutputAttribute();
        ImNodes::EndNode();
    }
    for (int i = 0; i < (int)g.edges.size(); ++i)
        ImNodes::Link(i, DAG_OUT_BASE + g.edges[i].src, DAG_IN_BASE + i);
    ImNodes::EndNodeEditor();
    g.laidOut = true;
    ImNodes::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopFont();

    // imnodes now knows each node's true rect (its own padding, the DPI-scaled font,
    // the pin rows). Adopt those and re-wrap once — otherwise the first-frame text
    // estimate decides the packing and a column can overhang the bottom of the pane.
    //
    // ...but only when the editor actually drew. The pane lives at the bottom of a
    // scrolling side column, so it is routinely clipped to zero height; imgui then sets
    // SkipItems on the canvas and every ImGui::Text inside a node returns without
    // measuring anything. imnodes still reports a rect — the node origin expanded by
    // NodePadding — and adopting *that* would silently re-pack the graph at 16 px per
    // node, so the layout is wrong the moment the user scrolls the pane into view. A
    // node always draws at least its title, so a content width of zero means "not
    // measured", never "an empty node".
    if (g.realSize.size() != g.nodes.size()) g.realSize.assign(g.nodes.size(), ImVec2(0.0f, 0.0f));
    const float minRealW = nodePad.x * 2.0f * g.zoom + 1.0f;
    std::vector<ImVec2> fresh(g.nodes.size());
    bool measured = true;
    for (size_t i = 0; i < g.nodes.size() && measured; ++i) {
        fresh[i] = ImNodes::GetNodeDimensions(g.nodes[i].id);
        measured = fresh[i].x > minRealW && fresh[i].y > 0.0f;
    }
    bool sizeChanged = false;
    if (measured)
        for (size_t i = 0; i < g.nodes.size(); ++i)
            if (std::fabs(fresh[i].x - g.realSize[i].x) > 1.0f ||
                std::fabs(fresh[i].y - g.realSize[i].y) > 1.0f) {
                g.realSize[i] = fresh[i];
                sizeChanged = true;
            }
    if (sizeChanged) { g.sizesValid = true; g.pos.clear(); }   // re-measure next frame

    // "fit": iterate zoom towards the scale at which the whole graph is on screen.
    // One shot isn't enough — a smaller zoom lets more nodes stack per column, which
    // changes the wrap and so the width — so it converges over a few (invisible) frames.
    // Each step waits for the layout to settle (node rects stable, positions current),
    // otherwise it compounds a correction that hasn't taken effect yet and collapses the
    // graph to a speck.
    const bool settled = measured && !sizeChanged && g.pos.size() == g.nodes.size();
    if (g.fitFrames > 0 && settled && g.extent.x > 1.0f && g.extent.y > 1.0f) {
        --g.fitFrames;
        // Only the width is a real constraint: the wrap already pins the height to the
        // pane, so extent.y ~= availH at every zoom and its ratio says nothing. Step
        // towards canvasW with a square-root damping, because zooming in also costs
        // sub-columns (width grows faster than the zoom does).
        const float s = canvasW / g.extent.x;
        if (s < 0.98f || s > 1.03f) {
            const float step = std::min(std::max(std::sqrt(s), 0.6f), 1.5f);
            // 0.30 is the floor (labels stop being readable) and 1.0 the ceiling (fit
            // shows the graph, it doesn't magnify it).
            g.zoom = std::min(std::max(g.zoom * step, 0.30f), 1.0f);
            g.pos.clear();
        } else {
            g.fitFrames = 0;
            g.fitted = true;
            g.fitCanvasW = canvasW;
        }
        ImNodes::EditorContextResetPanning(ImVec2(0.0f, 0.0f));
    }
}

// --------------------------------------------------------------------------
// Render pane (F7 primary path): raymarch the real field in-process.
//
// The viewer parses loom's emitted `.ftsl` (Sidecar::source) with ftrace's own
// ftsl::load, then renders it through renderIsoPreviewCuda — the same `-raster-gpu`
// preview kernel `-explore`/`-fly` and stills use, which sphere-traces the
// isosurface bytecode with NO tessellation. An orbit camera around the scene bounds
// drives it; each rendered RGB frame is blitted into a D3D11 texture shown with
// ImGui::Image. Re-rendering happens only when the camera moves (dirty), so an idle
// pane is free. Compiled only with CUDA (renderIsoPreviewCuda lives in the .cu).
// --------------------------------------------------------------------------
#ifdef HAVE_CUDA
struct RenderPane {
    // orbit camera around the scene bounding sphere
    float yaw = 0.6f, pitch = 0.3f;   // radians
    float distMul = 2.6f;             // eye distance = radius * distMul
    float fov = 40.0f;                // vertical fov, degrees
    Vec3  center{0, 0, 0};
    float radius = 1.0f;
    bool  inited = false;

    // last raymarched frame -> a dynamic D3D11 texture shown with ImGui::Image
    ID3D11Texture2D*          tex = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    int  texW = 0, texH = 0;
    bool dirty = true;
    int  resLong = 640;               // square raymarch resolution (long edge)
    // Playback resolution. The raymarch is synchronous on the UI thread and scales
    // with res^2, so at the full 640 it costs ~440 ms and is ~63% of a played frame
    // -- it, not the .ftsl round trip, is what makes play slow. Dropping to 256
    // while playing is ~6x less work, and matches what the -explore viewer already
    // does: degrade while moving, refine once settled (here, once paused).
    int  resPlay = 256;
    bool lowRes  = false;             // the res the current texture was traced at
    std::string status;
    // Phase split of the last raymarch (upload / kernel / readback). Fed to the Live
    // panel so the play breakdown can say WHICH part of the raymarch costs, rather
    // than leaving one opaque number to be over-interpreted.
    IsoPreviewTiming lastTiming;

    void initFrom(const Scene& s) {
        center = s.sceneCenter;
        radius = (s.sceneRadius > 0.0) ? (float)s.sceneRadius : 1.0f;
        inited = true;
        dirty  = true;
    }
    void release() {
        if (srv) { srv->Release(); srv = nullptr; }
        if (tex) { tex->Release(); tex = nullptr; }
        texW = texH = 0;
    }

    Camera camera(int W, int H) const {
        float cp = std::cos(pitch), sp = std::sin(pitch);
        float cy = std::cos(yaw),   sy = std::sin(yaw);
        Vec3 dir{ (double)(cp * sy), (double)sp, (double)(cp * cy) };  // center -> eye
        Vec3 eye = center + dir * (double)(radius * distMul);
        Camera c;
        c.lookAt(eye, center, Vec3{0, 1, 0}, fov, W, H);
        return c;
    }

    bool upload(const std::vector<uint8_t>& rgb, int W, int H,
                ID3D11Device* dev, ID3D11DeviceContext* ctx) {
        if ((int)rgb.size() < W * H * 3) return false;
        if (!tex || texW != W || texH != H) {
            release();
            D3D11_TEXTURE2D_DESC td = {};
            td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1;
            td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            td.SampleDesc.Count = 1;
            td.Usage = D3D11_USAGE_DYNAMIC;
            td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            if (dev->CreateTexture2D(&td, nullptr, &tex) != S_OK) return false;
            if (dev->CreateShaderResourceView(tex, nullptr, &srv) != S_OK) { release(); return false; }
            texW = W; texH = H;
        }
        D3D11_MAPPED_SUBRESOURCE ms;
        if (ctx->Map(tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms) != S_OK) return false;
        for (int y = 0; y < H; ++y) {
            uint8_t* dst = (uint8_t*)ms.pData + (size_t)y * ms.RowPitch;
            const uint8_t* src = &rgb[(size_t)y * W * 3];
            for (int x = 0; x < W; ++x) {
                dst[x * 4 + 0] = src[x * 3 + 0];
                dst[x * 4 + 1] = src[x * 3 + 1];
                dst[x * 4 + 2] = src[x * 3 + 2];
                dst[x * 4 + 3] = 255;
            }
        }
        ctx->Unmap(tex, 0);
        return true;
    }

    void render(const Scene& s, ID3D11Device* dev, ID3D11DeviceContext* ctx,
                bool draft = false) {
        int W = draft ? std::min(resPlay, resLong) : resLong, H = W;
        lowRes = (W != resLong);
        Camera cam = camera(W, H);
        unsigned hw = std::thread::hardware_concurrency();
        int nThreads = hw ? (int)hw : 4;
        lastTiming = IsoPreviewTiming{};
        std::vector<uint8_t> img =
            renderIsoPreviewCuda(s, cam, W, H, nThreads, 1.0, true, nullptr, &lastTiming);
        if (img.empty()) { status = "raymarch unavailable (no CUDA device or unsupported scene)"; return; }
        if (upload(img, W, H, dev, ctx)) { status.clear(); dirty = false; }
        else status = "D3D11 texture upload failed";
    }
};

// The Render tab body: orbit controls + the blitted raymarch image.
static bool drawRenderPane(RenderPane& rp, const Scene& scene, bool sceneOk,
                           const std::string& sceneErr,
                           ID3D11Device* dev, ID3D11DeviceContext* ctx,
                           LivePanel* live) {
    bool swept = false;   // the parameter axis moved -> loom must re-derive the field
    if (!sceneOk) {
        ImGui::TextWrapped("No live scene to raymarch.");
        if (!sceneErr.empty()) ImGui::TextWrapped("(%s)", sceneErr.c_str());
        ImGui::TextWrapped("The sidecar carries no `source` .ftsl (older loom, or "
                           "emit_source was off). Re-save it with a current loom to "
                           "enable the in-process field raymarch.");
        return false;
    }
    if (!rp.inited) rp.initFrom(scene);

    ImGui::TextUnformatted("GPU field raymarch (renderIsoPreviewCuda) - drag to orbit, wheel to zoom");
    liveSweepHint(live);
    ImGui::SetNextItemWidth(120);
    if (ImGui::SliderInt("res", &rp.resLong, 128, 1024)) rp.dirty = true;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110);
    ImGui::SliderInt("play res", &rp.resPlay, 64, 1024);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Resolution to raymarch at while the clock is PLAYING.\n"
                          "This trace is synchronous and scales with res^2, so it is\n"
                          "normally the largest single cost of a played frame; lower\n"
                          "this to play faster. Full `res` is restored when you pause.");
    if (rp.lowRes) {
        ImGui::SameLine();
        ImGui::TextDisabled("(draft %d)", std::min(rp.resPlay, rp.resLong));
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    if (ImGui::SliderFloat("fov", &rp.fov, 10.0f, 110.0f, "%.0f deg")) rp.dirty = true;
    ImGui::SameLine();
    if (ImGui::Button("re-render")) rp.dirty = true;
    if (!rp.status.empty()) { ImGui::SameLine(); ImGui::TextColored(ImVec4(1, 0.6f, 0.4f, 1), "%s", rp.status.c_str()); }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.y < 80.0f) avail.y = 80.0f;
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("render_canvas", avail,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    swept = liveSweepDrag(live);   // right-drag: rotate INTO the parameter dimension
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        ImVec2 d = ImGui::GetIO().MouseDelta;
        rp.yaw   -= d.x * 0.01f;
        rp.pitch += d.y * 0.01f;
        const float lim = 1.55f;   // keep the up vector well-defined
        if (rp.pitch >  lim) rp.pitch =  lim;
        if (rp.pitch < -lim) rp.pitch = -lim;
        rp.dirty = true;
    }
    if (ImGui::IsItemHovered()) {
        float w = ImGui::GetIO().MouseWheel;
        if (w != 0.0f) { rp.distMul *= (1.0f - w * 0.1f); if (rp.distMul < 0.2f) rp.distMul = 0.2f; rp.dirty = true; }
    }

    if (live) live->renderTabDrew = true;
    // Trace at draft res while the clock is playing, full res once it settles. The
    // moment play stops, the image on screen is a draft, so ask for one more trace
    // -- otherwise pausing would leave you inspecting a deliberately coarse frame.
    const bool draft = (live && live->playing);
    if (!draft && rp.lowRes) rp.dirty = true;
    if (rp.dirty) {
        // Timed because a landed bake calls initFrom, which sets `dirty`, so the
        // whole scene is re-raymarched synchronously on the UI thread at res^2 on
        // every played frame -- a cost paid only while this tab is open.
        if (live) {
            { MsTimer _t(&live->msRender); rp.render(scene, dev, ctx, draft); }
            // Carry the phase split up alongside the total. Copied after the timer
            // closes so `msRender` stays the authoritative wall-clock figure and the
            // three parts are only ever a breakdown OF it, never a substitute.
            live->msRenderUpload = rp.lastTiming.msUpload;
            live->msRenderKernel = rp.lastTiming.msKernel;
            live->msRenderRead   = rp.lastTiming.msReadback;
        }
        else      { rp.render(scene, dev, ctx, draft); }
    }

    // Fit the square texture into the pane, centered, preserving aspect.
    if (rp.srv && rp.texW > 0 && rp.texH > 0) {
        float side = std::min(avail.x, avail.y);
        ImVec2 img0(origin.x + 0.5f * (avail.x - side), origin.y + 0.5f * (avail.y - side));
        ImVec2 img1(img0.x + side, img0.y + side);
        ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)rp.srv, img0, img1);
    } else {
        ImGui::GetWindowDrawList()->AddRectFilled(
            origin, ImVec2(origin.x + avail.x, origin.y + avail.y), IM_COL32(18, 18, 22, 255));
    }
    return swept;
}
#endif // HAVE_CUDA

// --------------------------------------------------------------------------
// F4 item 2 — the live re-introspection link and its latest-wins job queue.
//
// A frozen sidecar can *display* geometry but cannot **re-derive** it. Orbiting
// the three shown spatial dims is a view-only re-projection (drawMeshPane says so
// in its own banner), but rotating into a **parameter dimension** — moving along
// one of the build's declared keyword params, or along the clock — changes the
// geometry itself, so the surface has to be re-tessellated by loom. That is what
// this section wires up: the C++ half of the §F4/§F7 channel whose loom half is
// `loom.viewer.ViewerSession` / `serve_viewer`.
//
//   * LoomLink   — spawns `python -m loom.viewer <scene.py>` and does one
//                  newline-delimited-JSON request/ack round trip over its pipes.
//                  Touched ONLY by the worker thread once the bridge is running.
//   * LoomBridge — that worker thread plus a **one-slot** pending job. Posting
//                  overwrites whatever was queued, so a fast param drag leaves at
//                  most one job in flight and one waiting; the values swept through
//                  in between are dropped rather than queued into a backlog the
//                  user would then have to sit through frame by frame. That is the
//                  latest-wins rule, and it is the whole reason this is a queue and
//                  not a plain synchronous call.
//   * LivePanel  — the UI: connection state, a clock scrub, one control per
//                  declared param, and a chosen **sweep axis** that the mesh /
//                  render canvas drags along with the right mouse button.
//
// Re-derivation is not free (a marching-cubes IsoMesh bake is comfortably a
// second), so the UI never blocks on it: it posts, keeps drawing the geometry it
// already has, and folds a result in on whatever frame it lands.
// --------------------------------------------------------------------------

// The transport itself (the pipes, the PYTHONPATH-augmented child environment, one
// JSON round trip per call) is shared with the fly editor's E2 live channel and now
// lives in loomlink.h; what stays here is only the viewer's own job policy.
using loomlink::jsonEsc;
using LoomLink = loomlink::Link;

// One re-derivation request. `params` values are raw JSON text so any declared type
// round-trips unchanged (an int stays `8`, not `8.0` — see loom's `types` ack).
struct LoomJob {
    long long seq = 0;
    int  frame = 0, frames = 1;
    std::vector<std::pair<std::string, std::string>> params;
    bool wantSidecar = true;    // re-introspect: curves / fields / MESH geometry
    bool wantSource  = true;    // re-emit .ftsl: the Render tab's raymarched field
    // Fingerprint of the params this job was posted at (see playCacheKey). Carried
    // only so the result can be matched back against the prebake cache it belongs to;
    // nothing in the request line uses it.
    std::string key;
};

// One finished re-derivation, entire in memory. NOTHING here names a file.
//
// It used to: loom wrote a `.json` and a `.ftsl` (plus the mesh assets the `.ftsl`
// referenced) into a scratch directory and the viewer opened them back. Measured
// 2026-08-06, that cost ~17 ms of a 130 ms frame — not I/O but Windows Defender's
// on-access scan, which charges a flat ~8 ms to open a file another process wrote a
// millisecond ago and cannot be avoided by writing faster. loom and ftrace already
// hold a pipe open between them, so the bytes come down that instead.
//
// Handed to the UI thread by `shared_ptr` and never copied: the sidecar tree alone is
// ~900 KB, and deep-copying it under the bridge's lock once a frame would give back
// most of what this change is buying.
struct LoomPayload {
    minijson::Value     sidecar;              // parsed off the ack, on the worker
    bool                hasSidecar = false;
    std::string         source;               // .ftsl text
    bool                hasSource  = false;
    assetbytes::Overlay assets;               // mesh bytes the source names by path
};

struct LoomResult {
    long long   seq = 0;
    bool        ok  = false;
    std::string err;
    std::shared_ptr<LoomPayload> payload;
    double      ms = 0.0;
    // Which clock frame this bake IS. Echoed back off the job because the prebake pass
    // (F8b) files each result into a cache SLOT, and the UI's own `lp.frame` has usually
    // moved on by the time a result lands — filing by "wherever the clock is now" would
    // scramble the cache in exactly the case that matters, a prebake running ahead of
    // the display.
    int         frame = 0;
    // ...and the parameter fingerprint it was baked at, so a result that was in flight
    // when a control moved is recognised as belonging to the old scene and dropped
    // rather than filed into the new cache under the frame number it happens to share.
    std::string key;
};

struct LoomBridge {
    LoomBridge() = default;
    // Owns a thread, a child process and three handles: not copyable, and destroying it
    // MUST stop the worker. `runViewerGui` has early returns after the bridge is
    // started (the D3D-device failure path), and ~std::thread on a joinable thread
    // calls std::terminate — an "the device didn't come up" message would have become
    // an abort instead.
    LoomBridge(const LoomBridge&) = delete;
    LoomBridge& operator=(const LoomBridge&) = delete;
    ~LoomBridge() { stop(); }

    // ---- UI thread ----
    bool start(const std::string& scenePy, std::string& err) {
        if (!link_.start("loom.viewer", scenePy, {}, err)) return false;
        // Ask for the controls synchronously, before the worker owns the link.
        minijson::Value ack;
        if (!link_.call("{\"cmd\":\"params\"}", ack, err)) { link_.stop(); return false; }
        if (const minijson::Value* p = ack.find("params"); p && p->isObject())
            for (const auto& kv : p->obj) paramDefaults_.push_back({kv.first, kv.second});
        if (const minijson::Value* t = ack.find("types"); t && t->isObject())
            for (const auto& kv : t->obj) paramTypes_[kv.first] = kv.second.asString("float");
        if (!makeTempDir(err)) { link_.stop(); return false; }
        worker_ = std::thread([this] { workerMain(); });
        return true;
    }

    // Idempotent: the destructor calls it too, and an explicit stop() before the
    // viewer's normal teardown is still the common path.
    void stop() {
        if (worker_.joinable()) {
            { std::lock_guard<std::mutex> lk(m_); quit_ = true; }
            cv_.notify_one();
            worker_.join();
        }
        link_.stop();
        if (!tempDir_.empty()) {
            // Nothing is written there any more (the live channel is all in-memory),
            // but a directory left by an OLDER ftrace under this same pid name would
            // otherwise never be collected, and the sweep is one syscall on an empty
            // dir. Files first: RemoveDirectory fails on a non-empty one.
            WIN32_FIND_DATAA fd{};
            HANDLE h = FindFirstFileA((tempDir_ + "\\*").c_str(), &fd);
            if (h != INVALID_HANDLE_VALUE) {
                do {
                    if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
                    DeleteFileA((tempDir_ + "\\" + fd.cFileName).c_str());
                } while (FindNextFileA(h, &fd));
                FindClose(h);
            }
            RemoveDirectoryA(tempDir_.c_str());
            tempDir_.clear();
        }
    }

    // LATEST WINS: this overwrites any job that has not started yet.
    void post(LoomJob j) {
        std::lock_guard<std::mutex> lk(m_);
        j.seq = ++seq_;
        pending_ = std::move(j);
        hasPending_ = true;
        cv_.notify_one();
    }

    // Moves: the payload is ~1 MB and the bridge must not keep a second reference to
    // it alive until the next bake happens to overwrite the slot.
    bool take(LoomResult& out) {
        std::lock_guard<std::mutex> lk(m_);
        if (!hasResult_) return false;
        out = std::move(result_);
        result_ = LoomResult{};
        hasResult_ = false;
        return true;
    }

    // "a re-derivation is happening or is about to" — drives the UI's spinner and
    // keeps a drag from being reported as idle between two jobs.
    bool busy() const {
        std::lock_guard<std::mutex> lk(m_);
        return running_ || hasPending_;
    }
    bool linkUp() const {
        std::lock_guard<std::mutex> lk(m_);
        return !dead_;
    }
    std::string deadReason() const {
        std::lock_guard<std::mutex> lk(m_);
        return deadErr_;
    }
    const std::vector<std::pair<std::string, minijson::Value>>& paramDefaults() const {
        return paramDefaults_;
    }
    std::string paramType(const std::string& name) const {
        auto it = paramTypes_.find(name);
        return it == paramTypes_.end() ? std::string("float") : it->second;
    }
    const std::string& command() const { return link_.cmdline; }

private:
    // Delete `ftrace_viewer_<pid>` directories left behind by viewers that died without
    // running stop() — a crash, or the user killing the process. `stop()` handles the
    // orderly exit, but nothing can clean up after a kill except the *next* run, and a
    // scene bake drops a multi-megabyte sidecar plus its mesh assets each time, so
    // without this %TEMP% accumulates them for as long as the machine stands. A PID is
    // reused eventually, hence the liveness probe rather than an age heuristic:
    // OpenProcess failing with ERROR_INVALID_PARAMETER is Windows saying "no such pid".
    static void sweepOrphanTempDirs(const char* tmp) {
        char pat[MAX_PATH + 64];
        std::snprintf(pat, sizeof pat, "%sftrace_viewer_*", tmp);
        WIN32_FIND_DATAA fd{};
        HANDLE h = FindFirstFileA(pat, &fd);
        if (h == INVALID_HANDLE_VALUE) return;
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
            const char* pidTxt = std::strrchr(fd.cFileName, '_');
            if (!pidTxt || !pidTxt[1]) continue;
            const DWORD pid = (DWORD)std::strtoul(pidTxt + 1, nullptr, 10);
            if (pid == 0 || pid == GetCurrentProcessId()) continue;
            HANDLE ph = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
            if (ph) { CloseHandle(ph); continue; }               // still running: leave it
            if (GetLastError() != ERROR_INVALID_PARAMETER) continue;  // exists, just not ours
            std::string dir = std::string(tmp) + fd.cFileName;
            WIN32_FIND_DATAA f2{};
            HANDLE h2 = FindFirstFileA((dir + "\\*").c_str(), &f2);
            if (h2 != INVALID_HANDLE_VALUE) {
                do {
                    if (f2.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
                    DeleteFileA((dir + "\\" + f2.cFileName).c_str());
                } while (FindNextFileA(h2, &f2));
                FindClose(h2);
            }
            RemoveDirectoryA(dir.c_str());
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }

    // The live channel writes nothing, so this directory is now only a NAMING scheme:
    // loom builds its mesh paths under it and the same strings become the `file`
    // arguments in the emitted `.ftsl`, which is what keys the byte overlay ftrace
    // loads from. It is deliberately still per-pid and still swept, because older
    // builds *did* write here and because a `mesh_format: "obj"` caller could ask for
    // files again. Not created: nothing needs it to exist.
    bool makeTempDir(std::string& err) {
        char tmp[MAX_PATH + 1];
        DWORD n = GetTempPathA(MAX_PATH, tmp);
        if (n == 0 || n > MAX_PATH) { err = "GetTempPath failed"; return false; }
        sweepOrphanTempDirs(tmp);
        char dir[MAX_PATH + 64];
        std::snprintf(dir, sizeof dir, "%sftrace_viewer_%lu", tmp, GetCurrentProcessId());
        tempDir_ = dir;
        return true;
    }

    // `extra` is spliced in before the closing brace, for the per-command fields.
    std::string requestLine(const char* cmd, const LoomJob& j,
                            const std::string& extra = std::string()) {
        std::string s = "{\"cmd\":\"";
        s += cmd;
        s += "\",\"clock\":{\"frame\":" + std::to_string(j.frame)
           + ",\"frames\":" + std::to_string(j.frames) + "},\"params\":{";
        for (size_t i = 0; i < j.params.size(); ++i) {
            if (i) s += ",";
            s += "\"" + jsonEsc(j.params[i].first) + "\":" + j.params[i].second;
        }
        s += "}" + extra + "}";
        return s;
    }

    // Move a named object out of an ack rather than copying it. The sidecar is ~900 KB
    // of parsed tree; `find()` hands back a const pointer, and taking a copy of that
    // would undo the whole point of parsing it exactly once.
    static bool stealMember(minijson::Value& ack, const char* key, minijson::Value& out) {
        if (ack.type != minijson::Value::Object) return false;
        for (auto& kv : ack.obj)
            if (kv.first == key) { out = std::move(kv.second); return true; }
        return false;
    }

    void workerMain() {
        for (;;) {
            LoomJob job;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this] { return quit_ || hasPending_; });
                if (quit_) return;
                job = pending_;
                hasPending_ = false;     // whatever else was posted meanwhile is gone
                running_ = true;
            }
            LoomResult r;
            r.seq = job.seq;
            r.frame = job.frame;
            r.key = job.key;
            r.payload = std::make_shared<LoomPayload>();
            LARGE_INTEGER f, t0, t1;
            QueryPerformanceFrequency(&f);
            QueryPerformanceCounter(&t0);
            std::string err;
            bool ok = true;
            minijson::Value ack;
            // Both requests come back INLINE — no `out`, so loom writes nothing and the
            // reply carries the payload. The sidecar rides in the ack's JSON (already
            // parsed by the time `call` returns, on this thread, off the UI's); the
            // meshes ride as binary attachments after the ack line, because base64 in
            // JSON would cost a 4/3 blowup plus an encode and a decode for bytes that
            // are already exactly what the loader wants.
            if (ok && job.wantSidecar) {
                ok = link_.call(requestLine("introspect", job), ack, err);
                if (ok) r.payload->hasSidecar =
                            stealMember(ack, "sidecar", r.payload->sidecar);
            }
            if (ok && job.wantSource) {
                std::vector<loomlink::Blob> blobs;
                // `assets_dir` no longer points anywhere real; it is only how loom
                // *names* the meshes, and those names are what the overlay is keyed by.
                std::string extra = ",\"assets_dir\":\"" + jsonEsc(tempDir_)
                                  + "\",\"assets\":\"inline\"";
                ok = link_.call(requestLine("emit", job, extra), ack, err, &blobs);
                if (ok) {
                    minijson::Value src;
                    if (stealMember(ack, "source", src) && src.isString()) {
                        r.payload->source    = std::move(src.str);
                        r.payload->hasSource = true;
                    }
                    for (loomlink::Blob& b : blobs)
                        r.payload->assets.put(b.name, std::move(b.bytes));
                }
            }
            QueryPerformanceCounter(&t1);
            r.ms = f.QuadPart ? 1000.0 * double(t1.QuadPart - t0.QuadPart) / double(f.QuadPart) : 0.0;
            r.ok = ok;
            r.err = err;
            {
                std::lock_guard<std::mutex> lk(m_);
                // A result must never overwrite a FRESHER one the UI has not read yet;
                // with one job in flight at a time that can't happen, but the guard
                // makes the invariant explicit rather than incidental. Superseding an
                // unread result now just drops its shared_ptr — there is no scratch
                // file left over for anyone to have to collect.
                if (!hasResult_ || r.seq >= result_.seq) {
                    result_ = std::move(r);
                    hasResult_ = true;
                }
                running_ = false;
                if (!ok && !link_.alive()) { dead_ = true; deadErr_ = err; }
            }
        }
    }

    LoomLink link_;
    std::thread worker_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    LoomJob    pending_;
    LoomResult result_;
    bool hasPending_ = false, hasResult_ = false, running_ = false, quit_ = false;
    bool dead_ = false;
    std::string deadErr_;
    long long seq_ = 0;
    std::string tempDir_;
    std::vector<std::pair<std::string, minijson::Value>> paramDefaults_;
    std::map<std::string, std::string> paramTypes_;
};

// Seed the controls from what loom advertised.
static void liveSeedParams(LivePanel& lp, const LoomBridge& br) {
    for (const auto& kv : br.paramDefaults()) {
        LiveParam p;
        p.name = kv.first;
        const std::string ty = br.paramType(kv.first);
        const minijson::Value& v = kv.second;
        if (ty == "bool")       { p.kind = 2; p.bval = v.asBool(false); }
        else if (ty == "int")   { p.kind = 1; p.num  = v.asNumber(0.0); }
        else if (ty == "float") { p.kind = 0; p.num  = v.asNumber(0.0); }
        else if (ty == "str")   { p.kind = 3; p.text = "\"" + jsonEsc(v.asString("")) + "\""; }
        else                    { p.kind = 3; p.text = "null"; }
        // A drag should cross the interesting range in a screen-width of travel, so
        // scale it to the default's own magnitude (and never to exactly zero).
        double mag = std::abs(p.num);
        p.speed = (p.kind == 1) ? std::max(1.0, mag * 0.02)
                                : std::max(1e-4, mag * 0.005);
        lp.params.push_back(std::move(p));
    }
    // Default the sweep axis to the first continuous control — the one "rotating into
    // a parameter dimension" actually means something for.
    for (size_t i = 0; i < lp.params.size(); ++i)
        if (lp.params[i].kind == 0 || lp.params[i].kind == 1) { lp.sweep = (int)i; break; }
}

// The fingerprint of everything a bake depends on EXCEPT the clock: every declared
// parameter's value, plus the clock length. Built from `toJson()` — the exact text
// that goes down the wire — so two states that bake identically fingerprint
// identically. It is what a prebaked cache (F8b) is a cache OF, and it is stamped onto
// each job so a result that was in flight across a parameter change can be recognised
// as belonging to the old scene and dropped instead of filed.
static std::string playCacheKey(const LivePanel& lp) {
    std::string k = "n=" + std::to_string(lp.frames);
    for (const auto& p : lp.params) { k += '\x1f'; k += p.name; k += '='; k += p.toJson(); }
    return k;
}

static LoomJob liveJobAt(const LivePanel& lp, int frame, bool wantSidecar, bool wantSource) {
    LoomJob j;
    j.frame = frame;
    j.frames = std::max(1, lp.frames);
    j.wantSidecar = wantSidecar;
    j.wantSource = wantSource;
    j.key = playCacheKey(lp);
    for (const auto& p : lp.params) j.params.push_back({p.name, p.toJson()});
    return j;
}

static LoomJob liveJob(const LivePanel& lp, bool wantSidecar, bool wantSource) {
    return liveJobAt(lp, lp.frame, wantSidecar, wantSource);
}

// --------------------------------------------------------------------------
// F8(b) — PREBAKED PLAY.
//
// F8(a) plays at the bake rate: the clock steps only when a result lands, which is
// honest but slow (measured 9.75 fps at v0.148.0 on `scatter_modulated_sweep.py`, of
// which ~42 ms/frame is loom's own Python bake and ~23 ms is adopting the result).
// You cannot judge MOTION at 10 fps, which is what a viewer of an animated scene is
// for. So: walk the whole clock ONCE, keep every frame, then play out of memory.
//
// WHAT IS CACHED, and why it is the adopted state rather than the payload. The
// obvious cache is loom's reply (the sidecar tree + the `.ftsl` text + mesh bytes),
// which is what the bridge already hands over. That only deletes the `bake 42`, and
// leaves `sidecar 2 + ftsl 21` to be paid again on every replay of every frame —
// worse, `Sidecar::adopt` and `ftsl::loadSource` both CONSUME what they are given, so
// replaying a cached payload would mean deep-copying a ~900 KB tree per frame just to
// have something to consume. Caching the ADOPTED products instead — the parsed
// sidecar, the collected curves/strips/fields/meshes, the DAG layout and the built
// `ftsl::Loaded` scene with its BVH — makes a replayed frame cost nothing but pointer
// swaps, and the only per-frame work left is drawing.
//
// HOW IT AVOIDS COPYING ANY OF THAT. Every cached member is a vector-of-vectors or a
// Scene; copying one per displayed frame would give back most of the win, and turning
// the viewer's ~8 pane-state locals into pointers into the cache would be a wide
// refactor of a long function. Instead the cache holds the SAME types as the live
// locals and frames are exchanged by `std::swap`, which is O(1) for all of them. The
// invariant that makes that safe is one line: **the live locals hold frame `liveIdx`,
// and slot `liveIdx` is empty.** Showing frame k is then always the same two swaps —
// park the live state back into its own slot, then swap slot k into the live state.
// --------------------------------------------------------------------------

// One cached clock frame: exactly the viewer state that a bake re-derives.
struct PlayFrame {
    bool                     have = false;
    Sidecar                  sc;
    std::vector<CurveGeom>   curves;
    std::vector<StripSeries> strips;
    std::vector<FieldGeom>   fields;
    std::vector<MeshGeom>    meshes;
    DagGraph                 dag;
    ftsl::Loaded             loaded;
    bool                     sceneOk = false;
    std::string              sceneErr;
    size_t                   bytes = 0;     // estimate; see playFrameBytes
};

struct PlayCache {
    std::vector<PlayFrame> f;
    // Everything a bake depends on besides the clock. A cache built at one set of
    // parameters is not a cache of the scene the user is now looking at, so any change
    // to a control (or to `frames`, which re-times the whole clock) drops it. Comparing
    // a fingerprint rather than trying to notice each edit means a control added later
    // cannot silently escape the check.
    std::string key;
    size_t      bytes = 0;
    int         capMB = 1024;        // stop prebaking here; the estimate is conservative
    bool        baking = false;
    int         bakeNext = 0;        // frame the next prebake post asks for
    int         bakeHave = 0;        // frames stored (including the one held live)
    bool        capped = false;      // prebake stopped early: the cache covers a prefix
    bool        projected = false;   // the up-front size projection has been printed once
    int         liveIdx = -1;        // the frame the LIVE locals hold; that slot is empty
    // Cached playback is paced by a wall clock, not by the bake — that is the whole
    // point. 0 means "as fast as it will go", which is the honest way to measure the
    // ceiling.
    float       targetFps = 24.0f;
    long long   lastStepQpc = 0;

    void drop() {
        f.clear(); key.clear(); bytes = 0; baking = false;
        bakeNext = 0; bakeHave = 0; capped = false; projected = false;
        liveIdx = -1; lastStepQpc = 0;
    }
    bool holds(int i) const { return i >= 0 && i < (int)f.size() && f[i].have; }
    // Can frame `i` be shown without going to loom? Either it is in a slot, or the
    // live locals are already showing it.
    bool covers(int i) const { return holds(i) || (i >= 0 && i == liveIdx); }
    // A prefix cache is still useful (that is the point of the cap), so "usable" is
    // not "complete" — it is "the frame we want is in it".
    bool active(const std::string& k, int frames) const {
        return !f.empty() && (int)f.size() == frames && key == k && bakeHave > 0;
    }
};

// Deep size of a parsed JSON tree. Needed because the sidecar is the single largest
// thing in a cached frame and its cost is invisible from the outside — `sizeof(Value)`
// says nothing about a 900 KB document. Counts the vectors' own storage plus every
// string's heap buffer; ignores allocator overhead, so it reads low, which is the
// right direction for a cap.
static size_t jsonTreeBytes(const minijson::Value& v) {
    size_t n = sizeof(minijson::Value) + v.str.capacity();
    for (const auto& c : v.arr) n += jsonTreeBytes(c);
    for (const auto& kv : v.obj) n += kv.first.capacity() + jsonTreeBytes(kv.second);
    return n;
}

// Estimated resident size of one cached frame. Every term is a real allocation the
// cache is holding onto; what is NOT counted is small and fixed (names, the DAG's
// layout, the Loaded camera list), so the number is a floor. Shown as an estimate and
// used for the cap, never for anything that has to be exact.
static size_t playFrameBytes(const PlayFrame& pf) {
    size_t n = jsonTreeBytes(pf.sc.root);
    for (const auto& c : pf.curves) {
        n += c.poly.capacity() * 4 + c.ctrl.capacity() * 4;
        for (const auto& ch : c.channels) n += ch.samp.capacity() * 4;
    }
    for (const auto& s : pf.strips) n += (s.x.capacity() + s.y.capacity()) * 4;
    for (const auto& f : pf.fields)
        for (const auto& p : f.points)
            n += p.pos.capacity() * 4 + p.val.capacity() * 4 + p.idx.capacity() * 4;
    for (const auto& m : pf.meshes)
        n += m.verts.capacity() * 4 + m.uvs.capacity() * 4 + m.faces.capacity() * 4
           + m.normals.capacity() * 4;
    const Scene& s = pf.loaded.scene;
    n += s.tris.capacity() * sizeof(Tri)
       + s.spheres.capacity() * sizeof(Sphere)
       + s.implicits.capacity() * sizeof(Implicit)
       + s.curveSegs.capacity() * sizeof(CurveSeg)
       + s.instances.capacity() * sizeof(MeshInstance)
       + s.dataPool.capacity() * sizeof(float)
       + s.bvh.nodes.capacity() * sizeof(BvhNode)
       + s.bvh.primIdx.capacity() * sizeof(int);
    for (const auto& b : s.blasList)
        n += b.tris.capacity() * sizeof(Tri)
           + b.bvh.nodes.capacity() * sizeof(BvhNode)
           + b.bvh.primIdx.capacity() * sizeof(int);
    for (const auto& t : s.textures)
        n += t.rgb.capacity() * sizeof(Vec3) + t.coeff.capacity() * sizeof(double) * 3;
    // The DAG last. Small next to the geometry, and easy to leave out for that reason --
    // but a cache is capped by this number, so anything omitted here is budget the cap
    // silently overshoots by. Counted rather than argued about.
    n += pf.dag.nodes.capacity() * sizeof(DagNode)
       + pf.dag.edges.capacity() * sizeof(DagEdge)
       + (pf.dag.pos.capacity() + pf.dag.realSize.capacity()) * sizeof(ImVec2);
    for (const auto& d : pf.dag.nodes)
        n += d.op.capacity() + d.label.capacity() + d.axes.capacity() + d.detail.capacity();
    for (const auto& e : pf.dag.edges) n += e.param.capacity() + e.mode.capacity();
    return n;
}

// The left-column "Live (loom)" section. Returns true when something the geometry
// depends on moved this frame.
static bool drawLivePanel(LivePanel& lp, LoomBridge& br, PlayCache& pc) {
    bool changed = false;
    if (!lp.up) {
        ImGui::TextWrapped("Not connected - the viewer is showing the static sidecar.");
        if (!lp.startErr.empty())
            ImGui::TextColored(ImVec4(1, 0.6f, 0.4f, 1), "%s", lp.startErr.c_str());
        ImGui::TextDisabled("Pass -loom <scene.py> (or use a sidecar carrying a `build` "
                            "key) to re-derive geometry live.");
        return false;
    }
    if (!br.linkUp()) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "loom link lost");
        std::string why = br.deadReason();
        if (!why.empty()) ImGui::TextWrapped("%s", why.c_str());
        return false;
    }
    ImGui::TextDisabled("%s", br.command().c_str());
    if (br.busy()) { ImGui::SameLine(); ImGui::TextColored(ImVec4(0.5f, 0.9f, 1, 1), "[re-deriving]"); }
    ImGui::Text("posted %lld / baked %lld", lp.posted, lp.baked);
    ImGui::SameLine();
    ImGui::TextDisabled("(last #%lld, %.0f ms)", lp.appliedSeq, lp.lastMs);
    if (!lp.lastErr.empty())
        ImGui::TextColored(ImVec4(1, 0.6f, 0.4f, 1), "%s", lp.lastErr.c_str());

    // `changed` = a control moved; `forced` = the user asked for it outright. With
    // auto off, a drag still updates the displayed value but costs no bake until the
    // button is pressed — which is the point of the switch on a slow scene.
    bool forced = false;
    ImGui::Checkbox("auto", &lp.autoApply);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("re-derive on every change; off = only on `re-derive now`");
    ImGui::SameLine();
    if (ImGui::Button("re-derive now")) forced = true;

    // --- the clock, which is a parameter dimension like any other ---
    ImGui::SetNextItemWidth(140);
    if (ImGui::SliderInt("frame", &lp.frame, 0, std::max(0, lp.frames - 1))) changed = true;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90);
    if (ImGui::DragInt("frames", &lp.frames, 1.0f, 1, 100000)) {
        if (lp.frames < 1) lp.frames = 1;
        if (lp.frame >= lp.frames) lp.frame = lp.frames - 1;
        changed = true;
    }

    // --- transport (F8a): play is paced by the bake, not by a timer ---
    // Starting play must post once to prime the loop: the clock only advances when a
    // result lands, so with nothing in flight nothing would ever land and play would
    // sit still. Hence `forced` on the leading edge.
    const bool wasPlaying = lp.playing;
    // A one-frame timeline has nowhere to advance to, so play would be a button that
    // silently does nothing -- the worst kind. Say why instead. `frames` comes from the
    // sidecar's clock, so the usual cause is a sidecar saved without one.
    const bool playable = lp.frames > 1;
    if (!playable) { lp.playing = false; ImGui::BeginDisabled(); }
    if (ImGui::Button(lp.playing ? "pause" : "play")) lp.playing = !lp.playing;
    if (!playable) ImGui::EndDisabled();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(playable
            ? "space; the clock advances one frame per completed bake"
            : "frames = 1: nothing to play. Raise `frames`, or save the sidecar with a\n"
              "clock (ViewerModel.save_sidecar(path, Clock.at_frame(0, N))).");
    ImGui::SameLine();
    if (ImGui::Button("|<")) { lp.frame = 0; lp.dir = 1; changed = true; }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("rewind to frame 0");
    ImGui::SameLine();
    ImGui::Checkbox("loop", &lp.loopPlay);
    ImGui::SameLine();
    ImGui::Checkbox("ping-pong", &lp.pingpong);

    // --- F8(b): prebake the whole clock, then play out of memory ---------------
    {
        const std::string key = playCacheKey(lp);
        const bool valid = pc.active(key, lp.frames);
        if (pc.baking) {
            if (ImGui::Button("cancel")) { pc.baking = false; }
            ImGui::SameLine();
            const float frac = lp.frames > 0 ? (float)pc.bakeHave / (float)lp.frames : 0.0f;
            char lab[64];
            std::snprintf(lab, sizeof lab, "%d/%d", pc.bakeHave, lp.frames);
            ImGui::ProgressBar(frac, ImVec2(140, 0), lab);
            ImGui::SameLine();
            ImGui::TextDisabled("%.0f MB", pc.bytes / 1048576.0);
        } else {
            if (!playable) ImGui::BeginDisabled();
            if (ImGui::Button(valid ? "re-prebake" : "prebake")) {
                pc.drop();
                pc.key = key;
                pc.f.assign((size_t)std::max(1, lp.frames), PlayFrame{});
                pc.baking = true;
                pc.bakeNext = 0;
            }
            if (!playable) ImGui::EndDisabled();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Walk the clock once, keeping every frame's adopted geometry and\n"
                    "scene in memory; play (and scrubbing) then costs no bake at all.\n"
                    "Dropped whenever a parameter or `frames` changes -- a cache built\n"
                    "at other values is not a cache of what you are looking at.");
            if (valid) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.6f, 1), "cached %d/%d, %.0f MB%s",
                                   pc.bakeHave, lp.frames, pc.bytes / 1048576.0,
                                   pc.capped ? " (cap)" : "");
            } else if (!pc.f.empty()) {
                ImGui::SameLine();
                ImGui::TextDisabled("(cache stale)");
            }
        }
        ImGui::SetNextItemWidth(90);
        ImGui::SliderFloat("fps", &pc.targetFps, 0.0f, 120.0f,
                           pc.targetFps <= 0.0f ? "free" : "%.0f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Playback rate once the clock is cached. `free` (0) runs as\n"
                              "fast as the draw allows, which is how you measure the ceiling.\n"
                              "Uncached play is still paced by the bake and ignores this.");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90);
        ImGui::DragInt("cap MB", &pc.capMB, 16.0f, 64, 65536);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Stop prebaking once the cache is this big. A partial cache is\n"
                              "still used -- the frames it holds play from memory and the rest\n"
                              "fall back to baking, so a long clock degrades instead of failing.");
    }

    // Keyboard: space toggles, arrows step. Guarded on WantTextInput so typing a
    // value into a drag field doesn't scrub the clock out from under the edit.
    if (!ImGui::GetIO().WantTextInput) {
        if (playable && ImGui::IsKeyPressed(ImGuiKey_Space)) lp.playing = !lp.playing;
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow) && lp.frames > 1) {
            lp.frame = (lp.frame + 1) % lp.frames; changed = true;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) && lp.frames > 1) {
            lp.frame = (lp.frame + lp.frames - 1) % lp.frames; changed = true;
        }
    }
    if (lp.playing && (!wasPlaying || lp.primePlay)) {
        forced = true;                 // prime the paced loop
        lp.playFps = 0.0;              // and don't average across the pause
        lp.lastAdvanceQpc = 0;
        lp.primePlay = false;
    }
    if (lp.playing) {
        ImGui::SameLine();
        // Say WHICH kind of playback this is. The two rates are not comparable — one is
        // loom's bake rate, the other is the draw rate — and a bare number would invite
        // exactly the confusion F8(a)'s fps readout was added to prevent.
        const char* src = pc.covers(lp.frame) ? " (cached)" : "";
        if (lp.playFps > 0.0)
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.6f, 1), "playing %.1f fps%s",
                               lp.playFps, src);
        else
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.6f, 1), "playing...");
    }
    // Where the frame time actually goes. Worth showing rather than leaving to be
    // guessed at: the intuition is that the .ftsl round trip dominates, and on a
    // modest mesh it does not -- the Render pane's synchronous raymarch does.
    {
        const double acc = lp.lastMs + lp.msSidecar + lp.msFtsl + lp.msCache + lp.msRender;
        // `bake + sidecar + ftsl` and `cache` are mutually exclusive by construction --
        // a frame is either derived or replayed -- so only the live half is shown. The
        // point of the swap is that the row it prints gets SHORTER.
        const bool cached = lp.msCache > 0.0;
        // Show the MEASURED period beside the parts, and the residual explicitly.
        // A breakdown that silently omits the gap between "what I timed" and "what
        // it actually costs" is the same dishonesty as deriving fps from the bake.
        if (lp.playing && lp.playFps > 0.0) {
            const double period = 1000.0 / lp.playFps;
            const double other  = (period - acc > 0.0 ? period - acc : 0.0);
            if (cached)
                ImGui::TextDisabled("frame %.0f ms = cache %.2f + raymarch %.0f + other %.0f",
                                    period, lp.msCache, lp.msRender, other);
            else
                ImGui::TextDisabled(
                    "frame %.0f ms = bake %.0f + sidecar %.0f + ftsl %.0f + raymarch %.0f + other %.0f",
                    period, lp.lastMs, lp.msSidecar, lp.msFtsl, lp.msRender, other);
        } else if (cached) {
            ImGui::TextDisabled("cache %.2f + raymarch %.0f = %.0f ms", lp.msCache, lp.msRender, acc);
        } else {
            ImGui::TextDisabled("bake %.0f + sidecar %.0f + ftsl %.0f + raymarch %.0f = %.0f ms",
                                lp.lastMs, lp.msSidecar, lp.msFtsl, lp.msRender, acc);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "bake     loom: build() + emit + introspect (its own process)\n"
                "sidecar  parse the introspection JSON, rebuild the DAG and skin buffers\n"
                "ftsl     parse the .ftsl and load its mesh assets\n"
                "cache    replaces all three on a prebaked frame: two state swaps plus\n"
                "         the skin/scene re-point a fresh adoption would also have done\n"
                "raymarch the Render tab re-tracing the scene on the UI thread.\n"
                "         Only charged while that tab is open -- switch to Meshes\n"
                "         to play without it.\n"
                "other    the residual against the measured play period: IPC with the\n"
                "         loom process, writing/reading the sidecar + OBJ through the\n"
                "         filesystem, and the wait for vblank.");
        // Break the raymarch open. Without this the single `raymarch` number invites
        // exactly one wrong conclusion -- that the .ftsl round trip is secondary --
        // which cannot be checked, because the three phases inside it scale with
        // different things and only one of them is affected by other GPU users.
        if (lp.msRender > 0.0) {
            ImGui::TextDisabled("   raymarch %.0f = upload %.0f + kernel %.0f + readback %.0f",
                                lp.msRender, lp.msRenderUpload, lp.msRenderKernel,
                                lp.msRenderRead);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "upload   re-marshal the WHOLE scene (tris, BVH, materials, every\n"
                    "         texel) and push it across PCIe -- every frame, even when\n"
                    "         only the camera moved. Scales with SCENE size, not pixels.\n"
                    "kernel   the raymarch. Scales with PIXELS (see `play res`). This is\n"
                    "         the ONLY phase another process on the GPU can inflate, so\n"
                    "         if it dwarfs the rest, check the card is actually idle\n"
                    "         before concluding the raymarch is the bottleneck.\n"
                    "readback D2H of accum/z/emissive + the host tone map. Pixels; CPU.\n"
                    "\n"
                    "Compare `upload` against bake+sidecar+ftsl to see whether caching\n"
                    "the scene on the device would actually buy anything for THIS scene.");
        }
    }

    // --- the build's declared params ---
    if (lp.params.empty()) {
        ImGui::TextDisabled("(the build declares no keyword params)");
    } else {
        for (size_t i = 0; i < lp.params.size(); ++i) {
            LiveParam& p = lp.params[i];
            ImGui::PushID((int)i);
            bool isAxis = ((int)i == lp.sweep);
            if (p.kind == 0 || p.kind == 1) {
                if (ImGui::RadioButton("##axis", isAxis)) lp.sweep = isAxis ? -1 : (int)i;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("make this the canvas sweep axis (right-drag to rotate into it)");
                ImGui::SameLine();
            } else {
                ImGui::Dummy(ImVec2(ImGui::GetFrameHeight(), 0));
                ImGui::SameLine();
            }
            ImGui::SetNextItemWidth(150);
            if (p.kind == 2) {
                if (ImGui::Checkbox(p.name.c_str(), &p.bval)) changed = true;
            } else if (p.kind == 3) {
                ImGui::LabelText(p.name.c_str(), "%s", p.text.c_str());
            } else if (p.kind == 1) {
                int v = (int)llround(p.num);
                if (ImGui::DragInt(p.name.c_str(), &v, (float)p.speed)) { p.num = v; changed = true; }
            } else {
                float v = (float)p.num;
                if (ImGui::DragFloat(p.name.c_str(), &v, (float)p.speed, 0.0f, 0.0f, "%.4g")) {
                    p.num = v; changed = true;
                }
            }
            ImGui::PopID();
        }
    }
    return forced || (lp.autoApply && changed);
}

// --------------------------------------------------------------------------
// Entry point
// --------------------------------------------------------------------------

// =====================================================================================
// THE GROOM TOOL -- `ftrace -groom <scene.ftsl>`  (0.329.0, Phase 1: view)
// =====================================================================================
// Opens the viewer shell on a plain .ftsl scene and shows, in one orbitable pane, the
// scene's own meshes (solid or wireframe -- the loader is asked to KEEP shape-only meshes,
// because the scalp a groom roots on is exactly one of those), every named `curve` node as a
// polyline in ITS LEVEL'S COLOUR (0 a strand, 1 a curve of strands, 2 a curve of those ...),
// its control points as crosses, and -- on demand, because they hide the curves -- the strands
// every `fur` block actually generated, read straight from the loaded scene so what you see is
// what the renderer would trace. Phase 2 adds authoring on top of this view.
namespace groom {

static const float kLevelColours[6][4] = {
    { 1.00f, 0.28f, 0.22f, 1.0f },   // level 0: strands (the guides)      red
    { 0.25f, 0.90f, 0.35f, 1.0f },   // level 1: a curve of strands        green
    { 0.35f, 0.60f, 1.00f, 1.0f },   // level 2: a curve of those          blue
    { 1.00f, 0.85f, 0.25f, 1.0f },   // level 3                            yellow
    { 0.95f, 0.40f, 0.95f, 1.0f },   // level 4                            magenta
    { 0.35f, 0.95f, 0.95f, 1.0f },   // level 5 and up                     cyan
};
static const float* levelColour(int level) { return kLevelColours[std::min(std::max(level, 0), 5)]; }

// A SECTION (0.372.0): what the Sections panel lists and the pane draws one at a time -- one
// material's triangles within a mesh block (a glTF part: "root.4"), a whole block from a format
// with no parts, or a part the scene SKIPS (`skip_material`), which the loader keeps out of the
// render and hands the tool as reference geometry (ftsl::keepSectionsRef) so a groom can be shaped
// inside it -- Alice2's sculpted hair, root.1. Any section can be hidden or made see-through; a
// see-through one is drawn blended, and picks and hovering pass through it to what is behind.
struct Section {
    std::string label;             // "alice2 / root.4"
    int   group = -1;              // its Scene::meshGroups index; -1 for a reference section
    bool  reference = false;       // not in the scene: `ref` holds its triangles
    std::vector<uint32_t> tris;    // its triangles in Scene::tris (an in-scene section)
    std::vector<Tri> ref;          // or its own (a reference section)
    bool  visible = true;
    float opacity = 1.0f;          // < 1: drawn blended; picked and hovered THROUGH
    bool  grid = false;            // drawn as a GRID OUTLINE (0.373.0): contour lines of the surface
    bool  roots = false;           // roots of new strands land here (default: the fur blocks' `on` meshes)
    Vec3  lo{0, 0, 0}, hi{0, 0, 0};   // its bounds (depth ranges for the hues, the slice's travel)
    bool  seeThrough() const { return !visible || opacity < 0.999f; }
    size_t triCount() const { return reference ? ref.size() : tris.size(); }
    const Tri& tri(const Scene& sc, size_t i) const { return reference ? ref[i] : sc.tris[tris[i]]; }
};

// The sections as pane geometry, one MeshGeom each and in the same order (so the GPU's draw
// ranges line up with them): independent triangles (the scene stores them flat), shading normals
// where the mesh had them.
static std::vector<MeshGeom> meshesFromSections(const Scene& sc, const std::vector<Section>& secs) {
    std::vector<MeshGeom> out;
    out.reserve(secs.size());
    for (const Section& s : secs) {
        MeshGeom m; m.name = s.label; m.id = s.label;
        const size_t n = s.triCount();
        m.verts.reserve(n * 9); m.normals.reserve(n * 9); m.faces.reserve(n * 3);
        for (size_t i = 0; i < n; ++i) {
            const Tri& tr = s.tri(sc, i);
            const Vec3* v[3] = { &tr.v0, &tr.v1, &tr.v2 };
            const Vec3* nn[3] = { &tr.n0, &tr.n1, &tr.n2 };
            Vec3 gn = tr.gn;
            if (dot(gn, gn) < 1e-18) { gn = cross(tr.v1 - tr.v0, tr.v2 - tr.v0); if (dot(gn, gn) > 1e-30) gn = normalize(gn); }
            for (int k = 0; k < 3; ++k) {
                m.verts.push_back((float)v[k]->x); m.verts.push_back((float)v[k]->y); m.verts.push_back((float)v[k]->z);
                const Vec3 nv = (dot(*nn[k], *nn[k]) > 1e-18) ? *nn[k] : gn;
                m.normals.push_back((float)nv.x); m.normals.push_back((float)nv.y); m.normals.push_back((float)nv.z);
                m.faces.push_back((int)m.faces.size());
            }
        }
        m.nverts = (int)m.verts.size() / 3; m.nfaces = (int)m.faces.size() / 3;
        m.smoothDeg = 0.0;
        out.push_back(std::move(m));               // even if empty: indices stay parallel to `secs`
    }
    return out;
}

// Line geometry: batches of segments, one colour each, in one immutable vertex buffer that is
// rebuilt only when a toggle, a selection or a point changes (a groom is a quarter-million segments).
// `slab`: the batch is cut by the groom tool's slab when the guides are (0.374.0) -- not the
// selected strand, which stays whole, nor the bald zones.
struct LineBatch { std::vector<MeshPaneVert> v; float rgba[4] = { 1, 1, 1, 1 }; bool slab = true; };
struct LinesGpu {
    ID3D11Buffer* vb = nullptr;
    UINT count = 0;
    std::vector<UINT> first, num;
    std::vector<std::array<float, 4>> colour;
    std::vector<char> slab;
    bool dirty = true;
    void release() { if (vb) { vb->Release(); vb = nullptr; } count = 0; first.clear(); num.clear(); colour.clear(); slab.clear(); }
    bool upload(ID3D11Device* dev, const std::vector<LineBatch>& batches) {
        release();
        std::vector<MeshPaneVert> all;
        for (const LineBatch& b : batches) {
            first.push_back((UINT)all.size()); num.push_back((UINT)b.v.size());
            colour.push_back({ b.rgba[0], b.rgba[1], b.rgba[2], b.rgba[3] });
            slab.push_back(b.slab ? 1 : 0);
            all.insert(all.end(), b.v.begin(), b.v.end());
        }
        dirty = false;
        if (all.empty()) return true;
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = (UINT)(all.size() * sizeof(MeshPaneVert));
        bd.Usage = D3D11_USAGE_IMMUTABLE;
        bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        D3D11_SUBRESOURCE_DATA sd = {}; sd.pSysMem = all.data();
        if (FAILED(dev->CreateBuffer(&bd, &sd, &vb))) { count = 0; return false; }
        count = (UINT)all.size();
        return true;
    }
};
static inline MeshPaneVert lineVert(const Vec3& p) {
    MeshPaneVert v; v.x = (float)p.x; v.y = (float)p.y; v.z = (float)p.z; v.u = v.v = 0.0f; v.nx = v.ny = 0.0f; v.nz = 1.0f; return v;
}
static void addSegment(LineBatch& b, const Vec3& a, const Vec3& c) { b.v.push_back(lineVert(a)); b.v.push_back(lineVert(c)); }
static void addCross(LineBatch& b, const Vec3& p, double h) {
    addSegment(b, p - Vec3{h, 0, 0}, p + Vec3{h, 0, 0});
    addSegment(b, p - Vec3{0, h, 0}, p + Vec3{0, h, 0});
    addSegment(b, p - Vec3{0, 0, h}, p + Vec3{0, 0, h});
}

// ---- the pane's camera, kept per frame so a pixel can become a ray and a point a pixel --------
// Orthographic: NDC x = ax * R0.(p - mid), y = ay * R1.(p - mid); R2 points at the viewer.
struct PaneCam {
    float R[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
    float mid[3] = { 0, 0, 0 };
    float ax = 1.0f, ay = 1.0f, s = 1.0f, diag = 1.0f;
    ImVec2 origin, avail;
    bool valid = false;
    Vec3 right()  const { return Vec3{ R[0][0], R[0][1], R[0][2] }; }
    Vec3 up()     const { return Vec3{ R[1][0], R[1][1], R[1][2] }; }
    Vec3 toward() const { return Vec3{ R[2][0], R[2][1], R[2][2] }; }   // out of the screen
    ImVec2 toScreen(const Vec3& p) const {
        const float d[3] = { (float)p.x - mid[0], (float)p.y - mid[1], (float)p.z - mid[2] };
        const float X = R[0][0] * d[0] + R[0][1] * d[1] + R[0][2] * d[2];
        const float Y = R[1][0] * d[0] + R[1][1] * d[1] + R[1][2] * d[2];
        return ImVec2(origin.x + (ax * X + 1.0f) * 0.5f * avail.x, origin.y + (1.0f - ay * Y) * 0.5f * avail.y);
    }
    void ray(const ImVec2& px, Vec3& o, Vec3& d) const {
        const float nx = (px.x - origin.x) / std::max(avail.x, 1.0f) * 2.0f - 1.0f;
        const float ny = 1.0f - (px.y - origin.y) / std::max(avail.y, 1.0f) * 2.0f;
        const Vec3 m{ mid[0], mid[1], mid[2] };
        o = m + right() * (double)(nx / ax) + up() * (double)(ny / ay) + toward() * (double)(diag * 4.0f);
        d = toward() * -1.0;
    }
    double worldPerPixel() const { return 1.0 / std::max(s, 1e-6f); }
};

// ---- picking: a pixel ray against a mesh group's own triangles -------------------------------
// The scene's BVH answers "what is hit" but not "which mesh"; a groom picks on ONE mesh (the
// fur's `on` target, ~10k triangles), so its triangle range is simply walked.
static bool rayTri(const Vec3& o, const Vec3& d, const Tri& t, double& tOut) {
    const Vec3 e1 = t.v1 - t.v0, e2 = t.v2 - t.v0;
    const Vec3 pv = cross(d, e2);
    const double det = dot(e1, pv);
    if (std::fabs(det) < 1e-18) return false;
    const double inv = 1.0 / det;
    const Vec3 tv = o - t.v0;
    const double u = dot(tv, pv) * inv;
    if (u < 0.0 || u > 1.0) return false;
    const Vec3 qv = cross(tv, e1);
    const double v = dot(d, qv) * inv;
    if (v < 0.0 || u + v > 1.0) return false;
    const double tt = dot(e2, qv) * inv;
    if (tt <= 0.0) return false;
    tOut = tt;
    return true;
}
struct Pick { bool hit = false; Vec3 p{0, 0, 0}, n{0, 1, 0}; double t = 0.0; int group = -1; int sec = -1; };
// The nearest hit of a ray on one section's triangles (the normal turned to face the viewer).
// With a slab (0.374.0), only hits inside it: what the slab cuts away cannot be clicked.
static Pick pickSection(const Scene& sc, const Section& s, const Vec3& o, const Vec3& d,
                        const Vec3* slabN = nullptr, double slabD = 0.0, double slabHalf = 0.0) {
    Pick best;
    const size_t n = s.triCount();
    for (size_t i = 0; i < n; ++i) {
        const Tri& tr = s.tri(sc, i);
        double t;
        if (!rayTri(o, d, tr, t) || (best.hit && t >= best.t)) continue;
        if (slabN && std::fabs(dot(*slabN, o + d * t) - slabD) > slabHalf) continue;
        best.hit = true; best.t = t;
        Vec3 nv = tr.gn;
        if (dot(nv, nv) < 1e-18) { nv = cross(tr.v1 - tr.v0, tr.v2 - tr.v0); if (dot(nv, nv) > 1e-30) nv = normalize(nv); }
        if (dot(nv, d) > 0.0) nv = nv * -1.0;       // facing the viewer
        best.n = nv;
    }
    if (best.hit) best.p = o + d * best.t;
    return best;
}

struct GroomState {
    std::string   scenePath;
    ftsl::Loaded  loaded;
    bool          ok = false;
    std::string   err;
    std::vector<MeshGeom> meshes;
    MeshView      view;
    LinesGpu      lines;
    PaneCam       cam;
    // toggles (any change marks the line buffer dirty)
    bool showMesh = true, showCurves = true, showPoints = true, showHair = false, showRoots = false;
    bool levelOn[8] = { true, true, true, true, true, true, true, true };
    int  maxLevel = 0;
    double ext = 1.0;               // extent of what is framed, for cross sizes
    // Framing: what the orbit centres on and scales to. 0 = the GROOM (every curve's points and
    // the fur's `on` mesh -- what one is authoring), 1 = every mesh. A floor or a room in the
    // scene would otherwise push the head into a corner of the pane.
    int   frameMode = 0;
    float frameMid[3] = { 0, 0, 0 }, frameExt = 1.0f, frameDiag = 1.0f;

    // ---- authoring (Phase 2): the curve tree as written, and what is being edited
    groom::Model model;
    std::unordered_map<int, Affine> xfOf;        // model node id -> authored->world (the loader's transform for its file/group)
    std::unordered_map<int, char>   nodeOn;      // visibility per node id (absent = shown)
    std::unordered_map<std::string, const ftsl::Loaded::HairCurveInfo*> recByName;   // the loader's flattened strands, for placed nodes
    int  selNode = -1, selPt = -1;               // the selected node (model id) and point index
    int  hoverNode = -1, hoverPt = -1;
    std::string target;                          // the roots sections' labels, joined by ", " (for the status line)
    bool pickAny = false;                        // or on any mesh
    // ---- sections (0.372.0): parts of the meshes, each hideable and see-through
    std::vector<Section> sections;               // parallel to `meshes` (one MeshGeom each) and the GPU ranges
    std::vector<char>    triPass;                // per Scene::tris: 1 = its section is hidden or see-through
    bool  sectionsDirty = true;                  // triPass is out of date (a visibility / opacity changed)
    // Where a strand's NEXT point goes (its root always goes on a roots section): 0 on surfaces
    // (the roots sections, or any with "any mesh"), 1 IN THE AIR at the view depth of the point it
    // follows, 2 INSIDE section `insideSec`, `insideDepth` of the way from where the pixel's ray
    // meets it to the next surface behind (0.373.0 made this a choice; 0.372.0 had only 2)
    int   placeMode = 1;
    int   insideSec = -1;
    float insideDepth = 0.5f;
    double dragFrac = 0.5;                       // a point dragged inside keeps its fraction of that interval
    // grid outlines (0.373.0): sections with `grid` drawn as contour lines -- the surface sliced by
    // planes `frameExt / gridLines` apart along the ticked axes; hidden-line unless `gridXray`
    float gridLines = 40.0f;
    bool  gridAxis[3] = { true, true, true };
    bool  gridXray = false;
    LinesGpu gridGpu;                            // one batch per gridded section
    std::vector<int> gridSecOf;                  // batch -> section
    bool  gridDirty = true;
    // reading a see-through part (0.374.0): hues, bright edges, and a movable slab
    bool  gridHueLine = false;                   // a colour per contour line, six in turn (none red: the guides are red)
    bool  hueDepth = false;                      // colour by depth, bright yellow near .. dim purple far: grid lines and see-through fills
    bool  rims = true;                           // see-through parts opaque and bright where seen edge-on
    bool  sliceOn = false;                       // show only a SLAB of the see-through and gridded parts
    int   sliceAxis = 1;                         // across 0 x, 1 y (height), 2 z, 3 the view direction
    float slicePos = 0.5f, sliceThick = 0.06f;   // its centre across the parts' extent (0..1); its thickness (fraction)
    bool  sliceGuides = true;                    // slab the guides too (the selected strand is always drawn whole)
    bool  sliceSolids = false;                   // and the solid parts: a scan-like slice (clicks and hovering see only the slab of them)
    Vec3  slabN{0, 1, 0};                        // this frame's slab: |n.p - slabD| <= slabHalf (slabHalf 0: none)
    double slabD = 0.0, slabHalf = 0.0;
    LinesGpu sliceGpu;                           // the parts' cross-section at the slab's centre
    std::string sliceKey;                        // the plane sliceGpu was built for
    // sketching (0.373.0): drag from a roots section to draw a new strand along the drag
    bool   sketchMode = false, sketching = false;
    ImVec2 sketchLast;
    float  sketchStep = 30.0f;                   // pixels of drag per sketched point
    // box selection of points (0.373.0): Shift-drag on empty space
    bool   boxing = false;
    ImVec2 boxA, boxB;
    std::vector<std::pair<int, int>> selPts;     // (node id, point index), the box-selected points
    bool   showHelp = false;
    bool addOnClick = true;                      // a click on the surface plots a point on the selected strand
    std::vector<groom::Model> undo;
    std::string status;
    bool stale = false;                          // the fur shown is from before an edit (placed strands preview live)
    groom::Preview preview;                      // every named node flattened through the loader's recursion, rebuilt with the lines
    std::vector<int> multi;                      // the Ctrl-click selection (node ids), for grouping
    // ---- fur (Phase 4)
    int   pickBald = -1, pickBaldStmt = -1;      // the fur entry / `bald` statement taking the next surface click as its centre
    bool  showBald = true;                       // bald zones as wire spheres
    int   renderSeconds = 60;
    char  renderMode[8] = "M";
    std::string renderNote;
    // a point being dragged
    bool   dragging = false, dragMoved = false;
    int    dragNode = -1, dragPt = -1;
    int    dragMode = 0;                         // 0 slide on the surface, 1 along the normal, 2 in the screen plane,
                                                 // 3 inside the "grow inside" section, at its depth fraction
    Vec3   dragN{0, 1, 0}, dragP0{0, 0, 0};
    ImVec2 dragMouse0, pressPos;
    bool   pressedEmpty = false;                 // the press began on nothing (an orbit, or a click that plots)
};

static bool nodeVisible(const GroomState& g, int id) { auto it = g.nodeOn.find(id); return it == g.nodeOn.end() || it->second != 0; }
// A picked or dragged point is kept to the micron: hair needs no more, and the file stays readable.
static Vec3 snapMicron(const Vec3& v) { return Vec3{ std::round(v.x * 1e6) / 1e6, std::round(v.y * 1e6) / 1e6, std::round(v.z * 1e6) / 1e6 }; }
static Affine xfOf(const GroomState& g, int id) { auto it = g.xfOf.find(id); return it == g.xfOf.end() ? Affine::identity() : it->second; }
static void assignXf(GroomState& g, groom::Node& n, const Affine& xf);
static bool isMulti(const GroomState& g, int id) { return std::find(g.multi.begin(), g.multi.end(), id) != g.multi.end(); }
static void toggleMulti(GroomState& g, int id) {
    auto it = std::find(g.multi.begin(), g.multi.end(), id);
    if (it == g.multi.end()) g.multi.push_back(id); else g.multi.erase(it);
}

template <class F> static void forEachEntryCurve(std::vector<groom::Entry>& list, F& f) {
    for (groom::Entry& e : list) { if (e.fur) continue; if (e.group) forEachEntryCurve(e.items, f); else f(e.curve); }
}
template <class F> static void forEachTopCurve(groom::Model& m, F f) { for (groom::FileModel& fm : m.files) forEachEntryCurve(fm.entries, f); }

// Is section `s` cut by the SLAB (0.374.0)? The see-through and gridded parts are (the ones you
// look into), and with "solid parts too" every part -- a scan-like slice of the whole scene. What
// a slab cuts away is gone for clicks and hovering too: only hits inside the slab count.
static bool slabbed(const GroomState& g, const Section& s) {
    return g.slabHalf > 0.0 && s.visible && (g.sliceSolids || s.grid || s.opacity < 0.999f);
}
static bool inSlab(const GroomState& g, const Vec3& p) { return g.slabHalf <= 0.0 || std::fabs(dot(g.slabN, p) - g.slabD) <= g.slabHalf; }

// The status line's name for where roots go: the labels of the sections ticked `roots`.
static void updateRootsLabel(GroomState& g) {
    g.target.clear();
    for (const Section& s : g.sections) if (s.roots) g.target += (g.target.empty() ? "" : ", ") + s.label;
    if (g.target.empty()) g.target = "no roots section -- tick 'roots' on one in Sections, or 'any mesh'";
}

// The surface under a pixel: a ROOTS section's own triangles, and -- for a click -- a check that
// nothing else is nearer (a pick through the face onto the back of the scalp is refused; the
// doll's own coincident triangles are not "nearer"). During a drag only the roots are walked.
// Walked per SECTION: a hidden or see-through one is looked through -- never picked, never in
// the way -- which is what lets a root be placed on the scalp under a translucent hair shell. A
// roots section is picked even when see-through (the scalp you root on may be made faint too). A
// reference section (not in the scene) takes no surface points; "inside a section" places into it.
static Pick pickSurfaceRay(GroomState& g, const Vec3& o, const Vec3& d, bool checkOcclusion) {
    const Scene& sc = g.loaded.scene;
    Pick best;
    for (size_t si = 0; si < g.sections.size(); ++si) {
        const Section& s = g.sections[si];
        if (s.reference || !s.visible || (s.opacity < 0.999f && !s.roots)) continue;
        const bool isTarget = s.roots;
        if (!g.pickAny && !isTarget && !checkOcclusion) continue;
        Pick p = pickSection(sc, s, o, d, slabbed(g, s) ? &g.slabN : nullptr, g.slabD, g.slabHalf);
        if (!p.hit) continue;
        p.group = s.group; p.sec = (int)si;
        const double tol = 1e-6 * (1.0 + std::fabs(p.t));
        if (!best.hit || p.t < best.t - tol) best = p;
        else if (isTarget && std::fabs(p.t - best.t) <= tol) best = p;   // coincident with a roots part: it wins
    }
    if (best.hit && !g.pickAny && (best.sec < 0 || !g.sections[(size_t)best.sec].roots)) best.hit = false;
    return best;
}
static Pick pickSurface(GroomState& g, const ImVec2& px, bool checkOcclusion) {
    Vec3 o, d;
    g.cam.ray(px, o, d);
    return pickSurfaceRay(g, o, d, checkOcclusion);
}

// Which scene triangles the eye looks THROUGH: those of a hidden or see-through section (1) --
// and (2) those of a solid one a slab cuts ("solid parts too"), which block only inside the slab.
static void updateTriPass(GroomState& g) {
    g.triPass.assign(g.loaded.scene.tris.size(), 0);
    for (const Section& s : g.sections) {
        if (s.reference) continue;
        const char v = s.seeThrough() ? 1 : (g.sliceOn && g.sliceSolids) ? 2 : 0;
        if (v) for (uint32_t t : s.tris) if (t < g.triPass.size()) g.triPass[t] = v;
    }
    g.sectionsDirty = false;
}

// Is anything the eye would SEE between `o` and `o + dir*maxDist`? The scene BVH as in
// Scene::occludedSkipHair (hair never blocks), except that a triangle of a hidden or see-through
// section does not block either -- so a point inside a translucent hair shell stays grabbable.
static bool occludedForHover(const GroomState& g, const Vec3& o, const Vec3& dir, double maxDist) {
    const Scene& sc = g.loaded.scene;
    const double tmin = 1e-6;
    const double seg = maxDist - tmin;
    if (!(seg > 0.0)) return false;
    Ray r{o, dir};
    const size_t nT = sc.tris.size(), nS = sc.spheres.size(), nI = sc.implicits.size(), nC = sc.curveSegs.size();
    const TriShear sh = makeTriShear(r.d);
    const CurveRay cray = nC ? makeCurveRay(r.d) : CurveRay{};
    const PatTables tabs = sc.patTables();
    const auto leaf = [&](int prim) {
        Hit h; h.t = seg;
        if (prim < (int)nT) {
            const char tp = (size_t)prim < g.triPass.size() ? g.triPass[(size_t)prim] : 0;
            if (tp == 1) return false;                                                          // see-through
            if (!intersectTri(sh, r, sc.tris[prim], tmin, h, sc.vcolData())) return false;
            return tp != 2 || inSlab(g, r.o + r.d * h.t);                                       // cut by the slab
        }
        if (prim < (int)(nT + nS))      return intersectSphere(r, sc.spheres[prim - nT], tmin, h);
        if (prim < (int)(nT + nS + nI)) return intersectImplicit(r, sc.implicits[prim - nT - nS], tmin, h, &tabs, /*anyHit=*/true);
        if (prim < (int)(nT + nS + nI + nC)) {
            const CurveSeg& cs = sc.curveSegs[prim - nT - nS - nI];
            if (sc.isHairCurve(cs)) return false;
            return intersectCurveSeg(cray, r, cs, curveMin(r, tmin), h, /*anyHit=*/true);
        }
        const MeshInstance& inst = sc.instances[prim - nT - nS - nI - nC];
        Ray lr{inst.toLocal.apply(r.o), inst.toLocal.applyDir(r.d)};
        return sc.blasList[inst.blasId].occludedLocal(lr, tmin, seg);
    };
    return sc.bvh.traverseAny(r, tmin, seg, leaf);
}

// GROW INSIDE: the depth interval a ray spends inside section `si`. Its hits, in order, pair up
// into RUNS inside it (in at one, out at the next; an odd last hit is an open sheet, its surface
// alone). The interval is the first run, ended early by anything solid inside it (the scalp under
// a hair shell) -- and refused when something solid is in front of it: a sculpted shell runs on
// inside the head behind the face, and a click on the face must not put a point in there.
// With a SLAB cutting the section (0.374.0) it is the first run that crosses the slab, clipped to
// it: the part of the section the eye actually sees there. False when there is none.
static bool insideInterval(const GroomState& g, const Vec3& o, const Vec3& d, int si, double& t0, double& t1) {
    if (si < 0 || si >= (int)g.sections.size()) return false;
    const Scene& sc = g.loaded.scene;
    const Section& S = g.sections[(size_t)si];
    std::vector<double> raw;
    const size_t n = S.triCount();
    for (size_t i = 0; i < n; ++i) { double t; if (rayTri(o, d, S.tri(sc, i), t)) raw.push_back(t); }
    if (raw.empty()) return false;
    std::sort(raw.begin(), raw.end());
    const double eps = 1e-7 * (1.0 + std::fabs(raw[0])) + 1e-5 * g.ext;   // a face's own duplicates
    std::vector<double> hits;
    for (double t : raw) if (hits.empty() || t > hits.back() + eps) hits.push_back(t);
    // the nearest solid surface on the ray (inside the slab, for a solid part the slab cuts)
    double solid = 1e300;
    for (size_t k = 0; k < g.sections.size(); ++k) {
        const Section& s = g.sections[k];
        if ((int)k == si || s.reference || s.seeThrough()) continue;
        const bool cut = slabbed(g, s);
        const size_t m = s.triCount();
        for (size_t i = 0; i < m; ++i) {
            double t;
            if (rayTri(o, d, s.tri(sc, i), t) && t < solid && (!cut || inSlab(g, o + d * t))) solid = t;
        }
    }
    // the stretch of the ray inside the slab
    double sa = -1e300, sb = 1e300;
    if (slabbed(g, S)) {
        const double nd = dot(g.slabN, d), no = dot(g.slabN, o);
        if (std::fabs(nd) < 1e-12) { if (std::fabs(no - g.slabD) > g.slabHalf) return false; }
        else {
            const double ta = (g.slabD - g.slabHalf - no) / nd, tb = (g.slabD + g.slabHalf - no) / nd;
            sa = std::min(ta, tb); sb = std::max(ta, tb);
        }
    }
    for (size_t k = 0; k < hits.size(); k += 2) {
        const double a = hits[k];
        double b = (k + 1 < hits.size()) ? hits[k + 1] : a;
        if (solid < a - eps) return false;                      // this run and all behind it are hidden
        if (solid > a + eps && solid < b) b = solid;
        const double ca = std::max(a, sa), cb = std::min(b, sb);
        if (ca <= cb) { t0 = ca; t1 = cb; return true; }
    }
    return false;
}
static bool insidePoint(const GroomState& g, const Vec3& o, const Vec3& d, double frac, Vec3& p) {
    double t0, t1;
    if (!insideInterval(g, o, d, g.insideSec, t0, t1)) return false;
    p = o + d * (t0 + std::clamp(frac, 0.0, 1.0) * (t1 - t0));
    return true;
}

// The sections of a freshly loaded scene: each mesh group split by the glTF material names the
// loader recorded (Loaded::toolSections), the rest of a group as one section, then the parts the
// scene skips as reference sections. Nothing here knows any particular model; the defaults come
// from the scene itself:
//   * roots: the sections of the meshes the scene's `fur` blocks grow on (`fur { on "scalp" }`),
//     else the first mesh;
//   * a part the scene SKIPS (`skip_material`) starts faint (see-through, its folds drawn by the
//     bright edges; 0.373.0 gave it a grid outline, which proved harder to read) -- the reason to
//     show a part the render leaves out is to groom inside it -- and, on a first load, is what
//     new points are placed inside.
// A reload keeps every section's settings (matched by label) and the placement choice.
static void buildSections(GroomState& g) {
    struct Kept { bool visible; float opacity; bool grid, roots; };
    std::map<std::string, Kept> kept;
    for (const Section& s : g.sections) kept[s.label] = { s.visible, s.opacity, s.grid, s.roots };
    const bool firstBuild = g.sections.empty();
    const std::string insideLabel = (g.insideSec >= 0 && g.insideSec < (int)g.sections.size()) ? g.sections[(size_t)g.insideSec].label : std::string();
    const Scene& sc = g.loaded.scene;
    g.sections.clear();
    for (size_t gi = 0; gi < sc.meshGroups.size(); ++gi) {
        const MeshGroup& mg = sc.meshGroups[gi];
        if (mg.blasId >= 0 || mg.triCount == 0) continue;
        const size_t end = std::min(mg.triStart + mg.triCount, sc.tris.size());
        std::vector<char> covered(end - mg.triStart, 0);
        std::map<std::string, size_t> byName;
        for (const ftsl::Loaded::ToolSection& ts : g.loaded.toolSections) {
            if (ts.skipped || ts.group != (int)gi) continue;
            auto it = byName.find(ts.material);
            if (it == byName.end()) {
                Section s; s.group = (int)gi;
                s.label = mg.name + " / " + (ts.material.empty() ? std::string("(unnamed material)") : ts.material);
                it = byName.emplace(ts.material, g.sections.size()).first;
                g.sections.push_back(std::move(s));
            }
            Section& s = g.sections[it->second];
            for (size_t t = std::max(ts.triStart, mg.triStart); t < ts.triStart + ts.triCount && t < end; ++t) {
                s.tris.push_back((uint32_t)t); covered[t - mg.triStart] = 1;
            }
        }
        Section rest; rest.group = (int)gi;
        rest.label = byName.empty() ? mg.name : mg.name + " / (rest)";
        for (size_t t = mg.triStart; t < end; ++t) if (!covered[t - mg.triStart]) rest.tris.push_back((uint32_t)t);
        if (!rest.tris.empty()) g.sections.push_back(std::move(rest));
    }
    std::map<std::string, size_t> refByName;
    for (const ftsl::Loaded::ToolSection& ts : g.loaded.toolSections) {
        if (!ts.skipped || ts.ref.empty()) continue;
        const std::string label = (ts.mesh.empty() ? std::string("mesh") : ts.mesh) + " / " +
                                  (ts.material.empty() ? std::string("(unnamed material)") : ts.material) + "  [skipped by the scene]";
        auto it = refByName.find(label);
        if (it == refByName.end()) {
            Section s; s.reference = true; s.label = label; s.opacity = 0.12f;   // faint; with bright edges (0.374.0)
            it = refByName.emplace(label, g.sections.size()).first;
            g.sections.push_back(std::move(s));
        }
        Section& s = g.sections[it->second];
        s.ref.insert(s.ref.end(), ts.ref.begin(), ts.ref.end());
    }
    for (Section& s : g.sections) {
        s.lo = Vec3{ 1e30, 1e30, 1e30 }; s.hi = Vec3{ -1e30, -1e30, -1e30 };
        for (size_t i = 0; i < s.triCount(); ++i) {
            const Tri& t = s.tri(sc, i);
            for (const Vec3* v : { &t.v0, &t.v1, &t.v2 }) {
                s.lo.x = std::min(s.lo.x, v->x); s.lo.y = std::min(s.lo.y, v->y); s.lo.z = std::min(s.lo.z, v->z);
                s.hi.x = std::max(s.hi.x, v->x); s.hi.y = std::max(s.hi.y, v->y); s.hi.z = std::max(s.hi.z, v->z);
            }
        }
        if (s.hi.x < s.lo.x) s.lo = s.hi = Vec3{ 0, 0, 0 };
    }
    // default roots: the fur blocks' `on` meshes, else the first mesh
    for (Section& s : g.sections) {
        if (s.reference || s.group < 0) continue;
        for (const auto& fi : g.loaded.furInfos)
            if (sc.meshGroups[(size_t)s.group].name == fi.on) { s.roots = true; break; }
    }
    if (std::none_of(g.sections.begin(), g.sections.end(), [](const Section& s) { return s.roots; }))
        for (Section& s : g.sections) if (!s.reference) { s.roots = true; break; }
    g.insideSec = -1;
    for (size_t i = 0; i < g.sections.size(); ++i) {
        auto it = kept.find(g.sections[i].label);
        if (it != kept.end()) {
            g.sections[i].visible = it->second.visible; g.sections[i].opacity = it->second.opacity;
            g.sections[i].grid = it->second.grid;       g.sections[i].roots = it->second.roots;
        }
        if (!insideLabel.empty() && g.sections[i].label == insideLabel) g.insideSec = (int)i;
    }
    // first load of a scene that skips a part: new points go inside it
    if (firstBuild && g.insideSec < 0)
        for (size_t i = 0; i < g.sections.size(); ++i)
            if (g.sections[i].reference) { g.insideSec = (int)i; g.placeMode = 2; break; }
    if (g.placeMode == 2 && g.insideSec < 0) g.placeMode = 1;
    updateRootsLabel(g);
    g.sectionsDirty = true;
    g.gridDirty = true;
}

static void groomComputeFrame(GroomState& g) {
    float alo[3] = { 1e30f, 1e30f, 1e30f }, ahi[3] = { -1e30f, -1e30f, -1e30f };   // everything drawn
    float glo[3] = { 1e30f, 1e30f, 1e30f }, ghi[3] = { -1e30f, -1e30f, -1e30f };   // the groom
    auto grow = [](float* lo, float* hi, float x, float y, float z) {
        lo[0] = std::min(lo[0], x); hi[0] = std::max(hi[0], x);
        lo[1] = std::min(lo[1], y); hi[1] = std::max(hi[1], y);
        lo[2] = std::min(lo[2], z); hi[2] = std::max(hi[2], z);
    };
    for (size_t mi = 0; mi < g.meshes.size(); ++mi) {
        const MeshGeom& m = g.meshes[mi];
        const bool on = mi < g.sections.size() && g.sections[mi].roots;   // a roots section
        for (int i = 0; i < m.nverts; ++i) {
            const float* v = &m.verts[(size_t)i * 3];
            grow(alo, ahi, v[0], v[1], v[2]);
            if (on) grow(glo, ghi, v[0], v[1], v[2]);
        }
    }
    for (const auto& c : g.loaded.hairCurves)
        for (const ftsl::CurveStrand& s : c.strands)
            for (const Vec3& p : s.pts) {
                grow(glo, ghi, (float)p.x, (float)p.y, (float)p.z);
                grow(alo, ahi, (float)p.x, (float)p.y, (float)p.z);
            }
    const bool haveGroom = ghi[0] >= glo[0], haveAll = ahi[0] >= alo[0];
    const float* lo = (g.frameMode == 0 && haveGroom) ? glo : alo;
    const float* hi = (g.frameMode == 0 && haveGroom) ? ghi : ahi;
    if (!haveAll) { g.frameMid[0] = g.frameMid[1] = g.frameMid[2] = 0.0f; g.frameExt = g.frameDiag = 1.0f; g.ext = 1.0; return; }
    g.frameExt = 1e-3f;
    for (int k = 0; k < 3; ++k) { g.frameExt = std::max(g.frameExt, hi[k] - lo[k]); g.frameMid[k] = 0.5f * (lo[k] + hi[k]); }
    // The depth range still has to cover EVERYTHING drawn, whatever the pane centres on: the
    // union box's half-diagonal plus the offset of its centre from the frame's.
    float ad2 = 0.0f, off2 = 0.0f;
    for (int k = 0; k < 3; ++k) {
        const float d = ahi[k] - alo[k], o = 0.5f * (alo[k] + ahi[k]) - g.frameMid[k];
        ad2 += d * d; off2 += o * o;
    }
    g.frameDiag = 0.5f * std::sqrt(ad2) + std::sqrt(off2) + 1e-3f;
    g.ext = g.frameExt;
}

// ---- the model -> lines ------------------------------------------------------------------------
// A strand draws its polyline through its authored points (transformed to world) and a cross at
// each point. A curve of curves draws its PATH -- the Catmull-Rom through its children's roots,
// what `count` places along -- and, when it places, the loader's flattened instances (from the
// last load: dimmed once an edit has made them stale). Children draw themselves; a reference
// draws nothing (its definition does).
static void drawNodeLines(GroomState& g, groom::Node& n, int level, std::vector<LineBatch>& out, double cross,
                          const groom::DrawBasis& db) {
    if (n.ref || !nodeVisible(g, n.id)) return;
    const Affine xf = xfOf(g, n.id);
    const bool leaf = n.kids.empty();
    if (level >= 0 && level < 8 && g.levelOn[level]) {
        LineBatch b;
        const float* col = levelColour(level);
        const bool dim = (g.selNode >= 0 && n.id != g.selNode);
        b.slab = (n.id != g.selNode);                  // the strand being edited is never cut by the slab
        for (int k = 0; k < 4; ++k) b.rgba[k] = col[k];
        if (dim) { b.rgba[0] *= 0.35f; b.rgba[1] *= 0.35f; b.rgba[2] *= 0.35f; }
        if (leaf) {
            // The strand AS THE RENDERER WILL TESSELLATE IT (groom.h leafPolyline), not the
            // control polygon. The default basis is catmull_rom at 4 cones per span, so the
            // straight-segment drawing this replaced was showing a shape no render produces.
            std::vector<Vec3> w; w.reserve(n.pts.size());
            for (const groom::Pt& p : n.pts) w.push_back(xf.apply(p.p));
            std::vector<Vec3> poly;
            groom::leafPolyline(n, db, poly);
            if (poly.size() >= 2) {
                for (Vec3& q : poly) q = xf.apply(q);
                for (size_t k = 1; k < poly.size(); ++k) addSegment(b, poly[k - 1], poly[k]);
                // the control polygon stays, faint, as the editing hint it always was: with
                // handles off the curve (catmull_rom passes through them, bezier does not)
                // you cannot grab a point you cannot see.
                if (g.showPoints && w.size() >= 2) {
                    LineBatch hull;
                    hull.slab = b.slab;
                    for (int k = 0; k < 3; ++k) hull.rgba[k] = b.rgba[k] * 0.30f;
                    hull.rgba[3] = 1.0f;
                    for (size_t k = 1; k < w.size(); ++k) addSegment(hull, w[k - 1], w[k]);
                    if (!hull.v.empty()) out.push_back(std::move(hull));
                }
            } else {
                // not a valid chain in this basis (bezier wants 3k+1, bspline >= 4): show the
                // polygon, so the point count is visibly the reason rather than a blank pane.
                for (size_t k = 1; k < w.size(); ++k) addSegment(b, w[k - 1], w[k]);
            }
            if (g.showPoints) for (const Vec3& p : w) addCross(b, p, cross);
        } else {
            std::vector<Vec3> roots;
            for (const groom::Node& k : n.kids) { Vec3 r; if (groom::rootOf(g.model, k, r)) roots.push_back(xf.apply(r)); }
            std::vector<Vec3> path;                       // the node's path, sampled 16 per segment
            if (roots.size() >= 2) {
                const bool closed = n.closed();
                const int nSeg = closed ? (int)roots.size() : (int)roots.size() - 1;
                const int M = nSeg * 16;
                const double alpha = n.alpha();
                path.reserve((size_t)M + 1);
                for (int k = 0; k <= M; ++k) path.push_back(ftsl::catmullRomAt(roots, closed, nSeg * (double)k / M, alpha));
                for (size_t k = 1; k < path.size(); ++k) addSegment(b, path[k - 1], path[k]);
            }
            if (g.showPoints) for (const Vec3& r : roots) addCross(b, r, cross * 1.6);
            // `density_at` keys as ticks on the path at their arc-length fraction, sized by rho
            const auto keys = groom::densityKeys(n);
            if (!keys.empty() && path.size() >= 2) {
                std::vector<double> cum(path.size(), 0.0);
                for (size_t k = 1; k < path.size(); ++k) cum[k] = cum[k - 1] + length(path[k] - path[k - 1]);
                double maxRho = 1e-300;
                for (const auto& kv : keys) maxRho = std::max(maxRho, kv.second);
                for (const auto& kv : keys) {
                    const double s = std::min(std::max(kv.first, 0.0), 1.0) * cum.back();
                    size_t k = 1;
                    while (k + 1 < path.size() && cum[k] < s) ++k;
                    const double seg = cum[k] - cum[k - 1], u = (seg > 1e-15) ? (s - cum[k - 1]) / seg : 0.0;
                    const Vec3 p = path[k - 1] + (path[k] - path[k - 1]) * u;
                    addCross(b, p, cross * (1.0 + 3.0 * std::max(kv.second, 0.0) / maxRho));
                }
            }
            if (n.placed() && !n.name.empty()) {
                // the instances `count` / density place, from the live preview (the file's frame -> world)
                auto it = g.preview.byName.find(n.name);
                if (it != g.preview.byName.end()) {
                    LineBatch inst;
                    inst.slab = b.slab;
                    for (int k = 0; k < 3; ++k) inst.rgba[k] = b.rgba[k] * 0.7f;
                    inst.rgba[3] = 1.0f;
                    // through the loader's tessellator, like the leaf above: these are the
                    // BLENDED CONTROL points, and the render splines them at the top basis.
                    std::vector<Vec3> poly;
                    for (const ftsl::CurveStrand& s : it->second.strands) {
                        groom::strandPolyline(s, db, poly);
                        for (size_t k = 1; k < poly.size(); ++k)
                            addSegment(inst, xf.apply(poly[k - 1]), xf.apply(poly[k]));
                    }
                    if (!inst.v.empty()) out.push_back(std::move(inst));
                }
            }
        }
        if (!b.v.empty()) out.push_back(std::move(b));
    }
    for (groom::Node& k : n.kids) if (!k.ref) drawNodeLines(g, k, groom::levelOf(g.model, k), out, cross, db);
}

static void groomBuildLines(GroomState& g, std::vector<LineBatch>& out) {
    out.clear();
    groom::buildPreview(g.model, g.preview);      // the placed instances, through the loader's own recursion
    const double cross = 0.006 * g.ext;
    if (g.showCurves)
        // `basis`/`segments`/`spline` come from the OUTERMOST node, so they are resolved here
        // and handed down -- the same rule the loader tessellates by.
        forEachTopCurve(g.model, [&](groom::Node& n) {
            drawNodeLines(g, n, groom::levelOf(g.model, n), out, cross, groom::drawBasisOf(n));
        });
    const Scene& sc = g.loaded.scene;
    if (g.showHair || g.showRoots) {
        for (const auto& fi : g.loaded.furInfos) {
            LineBatch hair, roots;
            const float f = g.stale ? 0.45f : 1.0f;
            hair.rgba[0] = 0.86f * f; hair.rgba[1] = 0.72f * f; hair.rgba[2] = 0.45f * f; hair.rgba[3] = 1.0f;
            roots.rgba[0] = roots.rgba[1] = roots.rgba[2] = f; roots.rgba[3] = 1.0f;
            const int cEnd = std::min((int)sc.curves.size(), fi.firstCurve + (int)fi.strands);
            for (int ci = fi.firstCurve; ci < cEnd; ++ci) {
                const Curve& c = sc.curves[(size_t)ci];
                if (c.segCount <= 0) continue;
                if (g.showHair)
                    for (int k = 0; k < c.segCount; ++k) {
                        const CurveSeg& s = sc.curveSegs[(size_t)c.firstSeg + (size_t)k];
                        addSegment(hair, s.p0, s.p1);
                    }
                if (g.showRoots) addCross(roots, sc.curveSegs[(size_t)c.firstSeg].p0, cross * 0.5);
            }
            if (!hair.v.empty()) out.push_back(std::move(hair));
            if (!roots.v.empty()) out.push_back(std::move(roots));
        }
    }
    if (g.showBald) {
        // `bald <x> <y> <z> <r>` zones as wire spheres (the named-sphere form is the scene's to draw)
        LineBatch bz;
        bz.slab = false;
        bz.rgba[0] = 1.0f; bz.rgba[1] = 0.55f; bz.rgba[2] = 0.15f; bz.rgba[3] = 1.0f;
        groom::forEachFur(g.model, [&](groom::FileModel&, groom::Entry& e) {
            for (const ftsl::Stmt& s : e.furBlock.stmts) {
                if (s.key != "bald" || s.val.words.size() < 4 || !groom::isNumberTok(s.val.words[0])) continue;
                const Vec3 c{ std::atof(s.val.words[0].c_str()), std::atof(s.val.words[1].c_str()), std::atof(s.val.words[2].c_str()) };
                const double r = std::atof(s.val.words[3].c_str());
                if (r <= 0.0) continue;
                for (int axis = 0; axis < 3; ++axis) {
                    Vec3 prev{0, 0, 0};
                    for (int k = 0; k <= 48; ++k) {
                        const double a = 2.0 * 3.14159265358979 * k / 48.0, ca = std::cos(a) * r, sa = std::sin(a) * r;
                        const Vec3 p = (axis == 0) ? Vec3{ c.x, c.y + ca, c.z + sa } : (axis == 1) ? Vec3{ c.x + ca, c.y, c.z + sa } : Vec3{ c.x + ca, c.y + sa, c.z };
                        if (k) addSegment(bz, prev, p);
                        prev = p;
                    }
                }
            }
        });
        if (!bz.v.empty()) out.push_back(std::move(bz));
    }
}

// ---- editing ----------------------------------------------------------------------------------
static void pushUndo(GroomState& g) {
    g.undo.push_back(g.model);
    if (g.undo.size() > 100) g.undo.erase(g.undo.begin());
}
static void select(GroomState& g, int node, int pt) {
    if (g.selNode != node || g.selPt != pt) g.lines.dirty = true;
    g.selNode = node; g.selPt = pt;
}
static void markEdited(GroomState& g, int nodeId) {
    groom::Where w = groom::whereIs(g.model, nodeId);
    if (w.file) w.file->dirty = true;
    g.lines.dirty = true;
    if (!g.loaded.furInfos.empty()) g.stale = true;
}
static bool isReferenced(groom::Model& m, const std::string& name) {
    if (name.empty()) return false;
    std::vector<groom::Node*> all; groom::collectNodes(m, all);
    for (const groom::Node* n : all) if (n->ref && n->name == name) return true;
    return false;
}
// A new hair file beside a scene that has no writable one, included from the scene.
static groom::FileModel* ensureGroomFile(GroomState& g) {
    namespace fs = std::filesystem;
    const fs::path root = fs::path(g.scenePath);
    const fs::path out = root.parent_path() / (root.stem().string() + "_groom.ftsl");
    for (groom::FileModel& f : g.model.files) if (f.path == out.string()) return &f;
    {
        std::ofstream a(g.scenePath, std::ios::app);
        if (!a) { g.status = "cannot append an include to " + g.scenePath; return nullptr; }
        a << "\n# hair authored with ftrace -groom\ninclude \"" << out.filename().string() << "\"\n";
    }
    groom::FileModel fm;
    fm.path = out.string(); fm.writable = true; fm.dirty = true;
    fm.leading = "# hair authored with ftrace -groom for " + root.filename().string() + "\n";
    g.model.files.push_back(std::move(fm));
    g.status = "created " + out.string() + " and included it from " + g.scenePath;
    return &g.model.files.back();
}
static void newStrand(GroomState& g) {
    groom::FileModel* fm = nullptr;
    groom::Entry* container = nullptr;
    if (g.selNode >= 0) {
        groom::Where w = groom::whereIs(g.model, g.selNode);
        if (w.file && w.file->writable) { fm = w.file; container = w.container; }
    }
    if (!fm) for (groom::FileModel& f : g.model.files) if (f.writable) { fm = &f; break; }
    if (!fm) fm = ensureGroomFile(g);
    if (!fm) return;
    Affine xf = Affine::identity();
    const std::vector<groom::Entry>& sib = container ? container->items : fm->entries;
    for (const groom::Entry& e : sib) if (!e.group && !e.fur) { xf = xfOf(g, e.curve.id); break; }
    pushUndo(g);
    groom::Node& n = groom::newCurve(g.model, *fm, container, groom::freshName(g.model, "strand"));
    g.xfOf[n.id] = xf;
    select(g, n.id, -1);
    g.status = "new strand \"" + n.name + "\" in " + fm->path + " -- click the surface to plot its points";
    g.lines.dirty = true;
}
static void addPointAt(GroomState& g, const Pick& pk) {
    groom::Node* n = (g.selNode >= 0) ? groom::findNode(g.model, g.selNode) : nullptr;
    if (!n || !n->kids.empty() || n->ref) { g.status = "select a strand (or press N for a new one) to plot points on it"; return; }
    pushUndo(g);
    groom::Pt p;
    p.p = snapMicron(xfOf(g, n->id).inverse().apply(pk.p));
    p.edited = true;
    const int at = (g.selPt >= 0 && g.selPt < (int)n->pts.size()) ? g.selPt + 1 : (int)n->pts.size();
    n->pts.insert(n->pts.begin() + at, p);
    g.selPt = at;
    markEdited(g, n->id);
    char buf[160];
    std::snprintf(buf, sizeof buf, "plotted point %d of \"%s\" at (%.4f %.4f %.4f)", at, n->name.c_str(), pk.p.x, pk.p.y, pk.p.z);
    g.status = buf;
}
static void deletePoint(GroomState& g) {
    groom::Node* n = (g.selNode >= 0) ? groom::findNode(g.model, g.selNode) : nullptr;
    if (!n || g.selPt < 0 || g.selPt >= (int)n->pts.size()) return;
    pushUndo(g);
    n->pts.erase(n->pts.begin() + g.selPt);
    if (n->pts.empty()) g.selPt = -1; else g.selPt = std::min(g.selPt, (int)n->pts.size() - 1);
    markEdited(g, n->id);
    g.status = "deleted a point of \"" + n->name + "\"";
}
static int eraseCurveAndRefs(GroomState& g, int id);
static void deleteStrand(GroomState& g) {
    groom::Node* n = (g.selNode >= 0) ? groom::findNode(g.model, g.selNode) : nullptr;
    if (!n) return;
    const std::string name = n->name;
    const int id = n->id;
    const bool referenced = isReferenced(g.model, name);
    pushUndo(g);
    if (!eraseCurveAndRefs(g, id)) { g.undo.pop_back(); g.status = "could not delete \"" + name + "\""; return; }
    g.selNode = g.selPt = -1;
    g.multi.erase(std::remove(g.multi.begin(), g.multi.end(), id), g.multi.end());
    g.selPts.erase(std::remove_if(g.selPts.begin(), g.selPts.end(), [id](const std::pair<int, int>& pr) { return pr.first == id; }), g.selPts.end());
    g.lines.dirty = true; g.stale = true;
    g.status = "deleted \"" + name + "\"" + (referenced ? " (and took it out of the curves that listed it)" : "");
}
// The Ctrl-click selection, validated for referencing by name: nodes at file (or group) level,
// all in ONE file, so `curve "name"` resolves when that file loads.
static bool multiAsNames(GroomState& g, groom::Where& where, std::vector<int>& unnamed) {
    unnamed.clear();
    where = groom::Where();
    for (int id : g.multi) {
        groom::Node* n = groom::findNode(g.model, id);
        if (!n) continue;
        groom::Where w = groom::whereIs(g.model, id);
        if (!w.entry || w.entry->curve.id != id) { g.status = "\"" + n->name + "\" is nested inside another curve; only curves at file (or group) level can be referenced by name"; return false; }
        if (!where.file) where = w;
        else if (w.file != where.file) { g.status = "the selection spans two files (" + where.file->path + ", " + w.file->path + "); group within one"; return false; }
        if (n->name.empty()) unnamed.push_back(id);
    }
    if (!where.file) { g.status = "Ctrl-click curves in the tree to select what to group"; return false; }
    if (!where.file->writable) { g.status = where.file->path + " is not writable (" + where.file->why + ")"; return false; }
    return true;
}
static void nameUnnamed(GroomState& g, const std::vector<int>& unnamed) {
    for (int id : unnamed) { groom::Node* n = groom::findNode(g.model, id); if (n && n->name.empty()) n->name = groom::freshName(g.model, "curve"); }
}
static void groupMulti(GroomState& g) {
    groom::Where w; std::vector<int> unnamed;
    if (!multiAsNames(g, w, unnamed)) return;
    pushUndo(g);
    nameUnnamed(g, unnamed);
    std::vector<std::string> names;
    for (int id : g.multi) if (groom::Node* n = groom::findNode(g.model, id)) names.push_back(n->name);
    const Affine xf = xfOf(g, g.multi.front());
    groom::Node& grp = groom::newGroupOf(g.model, *w.file, w.container, groom::freshName(g.model, "curve"), names);
    assignXf(g, grp, xf);
    const int id = grp.id;
    g.multi.clear();
    select(g, id, -1);
    markEdited(g, id);
    g.status = "grouped " + std::to_string(names.size()) + " curve(s) into \"" + grp.name + "\" -- give it a count to place instances along its path";
}
static void addMultiAsChildren(GroomState& g, int parentId) {
    groom::Where w; std::vector<int> unnamed;
    if (!multiAsNames(g, w, unnamed)) return;
    groom::Where pw = groom::whereIs(g.model, parentId);
    if (pw.file != w.file) { g.status = "the selection is in another file than the parent"; return; }
    pushUndo(g);
    nameUnnamed(g, unnamed);
    groom::Node* p = groom::findNode(g.model, parentId);
    if (!p) return;
    int added = 0;
    for (int id : g.multi) {
        groom::Node* n = groom::findNode(g.model, id);
        if (!n || n == p) continue;
        groom::Node r; r.id = g.model.nextId++; r.ref = true; r.name = n->name;
        g.xfOf[r.id] = xfOf(g, parentId);
        p->kids.push_back(std::move(r));
        ++added;
    }
    if (pw.entry && pw.entry->curve.id == parentId) groom::moveEntryToEnd(g.model, parentId);   // after everything it references
    g.multi.clear();
    markEdited(g, parentId);
    g.status = "added " + std::to_string(added) + " child(ren) to \"" + p->name + "\"";
}
static void undoLast(GroomState& g) {
    if (g.undo.empty()) { g.status = "nothing to undo"; return; }
    g.model = std::move(g.undo.back());
    g.undo.pop_back();
    g.multi.clear(); g.selPts.clear();
    for (groom::FileModel& f : g.model.files) if (f.writable) f.dirty = true;   // the disk may hold a later save
    if (g.selNode >= 0 && !groom::findNode(g.model, g.selNode)) g.selNode = g.selPt = -1;
    g.lines.dirty = true;
    g.status = "undone";
}
static void groomSave(GroomState& g) {
    std::string err;
    std::vector<std::string> written;
    if (!groom::saveModel(g.model, err, &written)) { g.status = "save failed: " + err; return; }
    if (written.empty()) { g.status = "nothing to save"; return; }
    std::string list;
    for (const std::string& w : written) list += (list.empty() ? "" : ", ") + w;
    g.status = "saved " + list + (g.stale ? " -- reload to refresh placed strands and fur" : "");
}

// ---- fur (Phase 4): the block's statements, bald zones placed on the surface, a real render ----
static std::vector<std::string> splitWords(const char* text) {
    std::vector<std::string> out;
    std::string cur;
    for (const char* p = text; *p; ++p) {
        if (*p == ' ' || *p == '\t') { if (!cur.empty()) { out.push_back(cur); cur.clear(); } }
        else cur += *p;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}
static void markFur(GroomState& g, groom::Entry& e) { e.furDirty = true; g.lines.dirty = true; g.stale = true; }
static void placeBald(GroomState& g, const ImVec2& mouse) {
    groom::Entry* target = nullptr;
    groom::forEachFur(g.model, [&](groom::FileModel&, groom::Entry& e) { if (e.id == g.pickBald) target = &e; });
    g.pickBald = -1;
    if (!target || g.pickBaldStmt < 0 || g.pickBaldStmt >= (int)target->furBlock.stmts.size()) { g.status = "the bald zone being placed is gone"; return; }
    const bool any = g.pickAny;
    g.pickAny = true;                                    // a bald zone is usually on the face, not the scalp
    const Pick pk = pickSurface(g, mouse, true);
    g.pickAny = any;
    if (!pk.hit) { g.status = "no surface under the click"; return; }
    pushUndo(g);
    ftsl::Stmt& s = target->furBlock.stmts[(size_t)g.pickBaldStmt];
    const Vec3 c = snapMicron(pk.p);
    while (s.val.words.size() < 4) s.val.words.push_back("0");
    s.val.words[0] = groom::fmtNum(c.x); s.val.words[1] = groom::fmtNum(c.y); s.val.words[2] = groom::fmtNum(c.z);
    markFur(g, *target);
    char buf[160];
    std::snprintf(buf, sizeof buf, "bald zone centred at (%.4f %.4f %.4f); its radius is the fourth number", c.x, c.y, c.z);
    g.status = buf;
}
// The saved scene rendered by a real ftrace from the pane's framing, with the live window.
static void spawnRender(GroomState& g) {
    groomSave(g);
    if (g.status.rfind("save failed", 0) == 0) { g.renderNote = g.status; return; }
    namespace fs = std::filesystem;
    wchar_t exe[MAX_PATH] = L"";
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    const std::string stem = fs::path(g.scenePath).stem().string();
    std::error_code ec;
    fs::create_directories("png/groom", ec);
    const std::string out = "png/groom/" + stem + "_groom.png", log = "png/groom/" + stem + "_groom.log";
    const Vec3 mid{ g.frameMid[0], g.frameMid[1], g.frameMid[2] };
    const double zoom = std::max(g.view.zoom, 0.05f);
    const double dist = 3.0 * g.frameExt / zoom;
    const Vec3 eye = mid + g.cam.toward() * dist;
    const double fov = 2.0 * std::atan(0.6 * g.frameExt / zoom / dist) * 180.0 / 3.14159265358979;
    char view[256];
    std::snprintf(view, sizeof view, "%.6g,%.6g,%.6g/%.6g,%.6g,%.6g/%.4g", eye.x, eye.y, eye.z, mid.x, mid.y, mid.z, fov);
    const int secs = std::max(1, g.renderSeconds);
    std::string mode = g.renderMode;
    if (mode.empty()) mode = "M";
    // the pane's aspect, 640 px wide (the ad-hoc view camera would otherwise default to 256^2)
    const int rw = 640, rh = std::max(64, (int)std::lround(640.0 * (g.cam.avail.y > 1.0f ? g.cam.avail.y / std::max(g.cam.avail.x, 1.0f) : 1.0f)));
    const std::string args = " -in \"" + g.scenePath + "\" -mode " + mode + " -time " + std::to_string(secs) + " -view " + view +
                             " -r " + std::to_string(rw) + " " + std::to_string(rh) +
                             " -window -keepwindow -interval 10 -o \"" + out + "\"";
    std::wstring wcmd = L"\"" + std::wstring(exe) + L"\"" + utf8ToWide(args);
    std::vector<wchar_t> mut(wcmd.begin(), wcmd.end());
    mut.push_back(L'\0');
    SECURITY_ATTRIBUTES sa = { sizeof sa, nullptr, TRUE };
    HANDLE hLog = CreateFileA(log.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    STARTUPINFOW si = {};
    si.cb = sizeof si;
    if (hLog != INVALID_HANDLE_VALUE) { si.dwFlags = STARTF_USESTDHANDLES; si.hStdOutput = hLog; si.hStdError = hLog; si.hStdInput = GetStdHandle(STD_INPUT_HANDLE); }
    PROCESS_INFORMATION pi = {};
    const BOOL ok = CreateProcessW(nullptr, mut.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    if (hLog != INVALID_HANDLE_VALUE) CloseHandle(hLog);
    if (!ok) { g.renderNote = "could not start the render (error " + std::to_string(GetLastError()) + ")"; return; }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    const std::string pid = std::to_string(pi.dwProcessId);
    g.renderNote = "rendering as pid " + pid + " -> " + out + " (log: " + log + "); the live window stays up when it finishes -- close it, or `ftrace -stop " + pid + "`";
}
static void drawFurSection(GroomState& g) {
    if (!ImGui::CollapsingHeader("Fur", ImGuiTreeNodeFlags_DefaultOpen)) return;
    if (ImGui::Checkbox("show bald zones", &g.showBald)) g.lines.dirty = true;
    int nfur = 0;
    groom::forEachFur(g.model, [&](groom::FileModel& fm, groom::Entry& e) {
        ++nfur;
        ImGui::PushID(e.id);
        ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.3f, 1.0f), "fur \"%s\"%s", e.name.c_str(), e.furDirty ? "  *modified*" : "");
        const ftsl::Loaded::FurInfo* fi = nullptr;
        for (const auto& f : g.loaded.furInfos) if (f.name == e.name) fi = &f;
        if (fi) ImGui::TextDisabled("last load: %lld strands on \"%s\"%s", fi->strands, fi->on.c_str(), g.stale ? "  (stale: save + reload regenerates)" : "");
        ImGui::TextDisabled("in %s%s", fm.path.c_str(), fm.writable ? "" : " (patched in place on save)");
        for (size_t i = 0; i < e.furBlock.stmts.size(); ++i) {
            ftsl::Stmt& s = e.furBlock.stmts[i];
            ImGui::PushID((int)i);
            std::string joined;
            for (const std::string& w : s.val.words) joined += (joined.empty() ? "" : " ") + w;
            char buf[256];
            std::snprintf(buf, sizeof buf, "%s", joined.c_str());
            ImGui::SetNextItemWidth(ImGui::GetFontSize() * 11.0f);
            const bool changed = ImGui::InputText(s.key.c_str(), buf, sizeof buf);
            if (ImGui::IsItemActivated()) pushUndo(g);
            if (changed) { s.val.words = splitWords(buf); markFur(g, e); }
            ImGui::SameLine();
            if (ImGui::SmallButton("x")) { pushUndo(g); e.furBlock.stmts.erase(e.furBlock.stmts.begin() + (std::ptrdiff_t)i); markFur(g, e); ImGui::PopID(); break; }
            if (s.key == "bald") {
                ImGui::SameLine();
                const bool picking = (g.pickBald == e.id && g.pickBaldStmt == (int)i);
                if (ImGui::SmallButton(picking ? "click the surface..." : "pick centre")) {
                    g.pickBald = e.id; g.pickBaldStmt = (int)i;
                    g.status = "click the surface to place the bald zone's centre (its radius is the 4th number)";
                }
            }
            ImGui::PopID();
        }
        static char key[32] = "", val[160] = "";
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 6.0f); ImGui::InputText("##k", key, sizeof key); ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 9.0f); ImGui::InputText("##v", val, sizeof val); ImGui::SameLine();
        if (ImGui::SmallButton("add statement") && key[0]) {
            pushUndo(g);
            ftsl::Stmt s; s.key = key; s.val.words = splitWords(val);
            e.furBlock.stmts.push_back(std::move(s));
            markFur(g, e);
            key[0] = 0; val[0] = 0;
        }
        if (ImGui::SmallButton("+ bald zone (then click its centre)")) {
            pushUndo(g);
            ftsl::Stmt s; s.key = "bald"; s.val.words = { "0", "0", "0", groom::fmtNum(0.15 * g.ext) };
            e.furBlock.stmts.push_back(std::move(s));
            g.pickBald = e.id; g.pickBaldStmt = (int)e.furBlock.stmts.size() - 1;
            markFur(g, e);
            g.status = "click the surface to place the new bald zone's centre";
        }
        ImGui::PopID();
    });
    if (!nfur) ImGui::TextDisabled("(no fur blocks in this scene)");
    ImGui::Separator();
    ImGui::TextUnformatted("render the saved scene from this view (a real ftrace, live window):");
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 5.0f); ImGui::InputInt("seconds", &g.renderSeconds); ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 3.0f); ImGui::InputText("mode", g.renderMode, sizeof g.renderMode);
    if (ImGui::Button("save + render")) spawnRender(g);
    if (!g.renderNote.empty()) ImGui::TextWrapped("%s", g.renderNote.c_str());
}

// ---- loading: the scene through the renderer's loader, the curve tree through the parser --------
static bool groomParseModel(GroomState& g) {
    std::string err;
    if (!groom::parseSceneModel(g.scenePath, g.model, err)) { g.status = err; return false; }
    return true;
}
static void assignXf(GroomState& g, groom::Node& n, const Affine& xf) { g.xfOf[n.id] = xf; for (groom::Node& k : n.kids) assignXf(g, k, xf); }
static void resolveXfList(GroomState& g, std::vector<groom::Entry>& list) {
    // A named curve takes the transform the loader applied to it; an unnamed or new one takes
    // its named siblings' (they share the group).
    Affine known = Affine::identity();
    for (groom::Entry& e : list)
        if (!e.group && !e.fur && !e.curve.name.empty()) { auto it = g.recByName.find(e.curve.name); if (it != g.recByName.end()) { known = it->second->xf; break; } }
    for (groom::Entry& e : list) {
        if (e.fur) continue;
        if (e.group) { resolveXfList(g, e.items); continue; }
        auto it = e.curve.name.empty() ? g.recByName.end() : g.recByName.find(e.curve.name);
        assignXf(g, e.curve, it != g.recByName.end() ? it->second->xf : known);
    }
}
static bool groomLoadScene(GroomState& g) {
    g.loaded = ftsl::Loaded();
    g.meshes.clear(); g.recByName.clear(); g.xfOf.clear(); g.model = groom::Model(); g.undo.clear();
    g.selNode = g.selPt = g.hoverNode = g.hoverPt = -1;
    g.multi.clear(); g.selPts.clear();
    g.stale = false; g.dragging = false; g.sketching = false; g.boxing = false;
    g.sliceKey = "(reload)";                              // the parts' cross-section is rebuilt from the new geometry
    ftsl::keepShapeOnlyRef() = true;                      // the scalp must be drawable and pickable
    ftsl::keepSectionsRef() = true;                       // the parts by name, and the parts the scene skips
    g.ok = ftsl::load(g.scenePath, g.loaded, g.err);
    ftsl::keepSectionsRef() = false;
    ftsl::keepShapeOnlyRef() = false;
    if (!g.ok) { std::fprintf(stderr, "[groom] could not load '%s': %s\n", g.scenePath.c_str(), g.err.c_str()); return false; }
    buildSections(g);
    g.meshes = meshesFromSections(g.loaded.scene, g.sections);
    for (const auto& r : g.loaded.hairCurves) g.recByName[r.name] = &r;
    if (!groomParseModel(g)) std::fprintf(stderr, "[groom] the authored curve tree could not be read: %s\n", g.status.c_str());
    for (groom::FileModel& fm : g.model.files) resolveXfList(g, fm.entries);
    // (the roots -- where a strand's first point lands -- are per section: buildSections ticks the
    // fur blocks' `on` meshes by default, and the Sections panel lets any part be ticked instead)
    g.maxLevel = 0;
    forEachTopCurve(g.model, [&](groom::Node& n) { g.maxLevel = std::max(g.maxLevel, groom::levelOf(g.model, n)); });
    groomComputeFrame(g);
    g.view.geomGen++;
    g.lines.dirty = true;
    std::fprintf(stderr, "[groom] %zu section(s), %zu named curve(s) (max level %d), %zu fur block(s); framing %.3g m about (%.3f %.3f %.3f); picks on \"%s\"\n",
                 g.meshes.size(), g.loaded.hairCurves.size(), g.maxLevel, g.loaded.furInfos.size(),
                 g.frameExt, g.frameMid[0], g.frameMid[1], g.frameMid[2], g.target.c_str());
    for (const auto& m : g.meshes) {
        float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
        for (int i = 0; i < m.nverts; ++i) for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], m.verts[(size_t)i * 3 + k]); hi[k] = std::max(hi[k], m.verts[(size_t)i * 3 + k]); }
        std::fprintf(stderr, "[groom]   section \"%s\": %d tris, x %.3f..%.3f  y %.3f..%.3f  z %.3f..%.3f\n",
                     m.name.c_str(), m.nfaces, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
    }
    for (const groom::FileModel& fm : g.model.files) {
        std::fprintf(stderr, "[groom]   file %s: %zu top-level block(s)%s%s\n", fm.path.c_str(), fm.entries.size(),
                     fm.writable ? ", writable" : ", NOT writable: ", fm.writable ? "" : fm.why.c_str());
    }
    return true;
}

// ---- sections drawn as GRID OUTLINES (0.373.0) ---------------------------------------------------
static const float kSectionTints[4][3] = { { 0.62f, 0.66f, 0.72f }, { 0.78f, 0.62f, 0.50f }, { 0.58f, 0.76f, 0.60f }, { 0.76f, 0.58f, 0.72f } };
static void sectionTint(const GroomState& g, size_t si, float alpha, float out[4]) {
    const bool refSec = si < g.sections.size() && g.sections[si].reference;
    const float* t = kSectionTints[si % 4];
    out[0] = refSec ? 0.95f : t[0]; out[1] = refSec ? 0.80f : t[1]; out[2] = refSec ? 0.42f : t[2]; out[3] = alpha;
}
static double axisOf(const Vec3& v, int ax) { return ax == 0 ? v.x : (ax == 1 ? v.y : v.z); }
// Every gridded section sliced by the planes x = k s, y = k s, z = k s (the ticked axes), s = the
// framed extent / `gridLines`: the CONTOUR LINES of its surface. A folded shape reads from its
// contours where a see-through fill shows only a haze. One batch per section (`gridSecOf`); the
// colour is chosen when drawing, from the section's tint and opacity.
static void buildGridLines(GroomState& g, std::vector<LineBatch>& out) {
    out.clear(); g.gridSecOf.clear();
    const Scene& sc = g.loaded.scene;
    const double s = (double)g.frameExt / std::max(4.0, (double)g.gridLines);
    for (size_t si = 0; si < g.sections.size(); ++si) {
        const Section& S = g.sections[si];
        if (!S.grid) continue;                            // (hidden ones are skipped when drawing, so a
        LineBatch b;                                      //  show/hide or a colour needs no rebuild)
        const size_t n = S.triCount();
        for (size_t i = 0; i < n; ++i) {
            const Tri& t = S.tri(sc, i);
            const Vec3 v[3] = { t.v0, t.v1, t.v2 };
            for (int ax = 0; ax < 3; ++ax) {
                if (!g.gridAxis[ax]) continue;
                const double c0 = axisOf(v[0], ax), c1 = axisOf(v[1], ax), c2 = axisOf(v[2], ax);
                const long long k0 = (long long)std::ceil(std::min({ c0, c1, c2 }) / s);
                const long long k1 = (long long)std::floor(std::max({ c0, c1, c2 }) / s);
                for (long long k = k0; k <= k1; ++k) {
                    const double c = (double)k * s;
                    double dd[3] = { c0 - c, c1 - c, c2 - c };
                    for (double& x : dd) if (x == 0.0) x = 1e-12;      // a vertex ON the plane counts as above it
                    Vec3 p[2]; int np = 0;
                    for (int e = 0; e < 3 && np < 2; ++e) {
                        const int a = e, bb = (e + 1) % 3;
                        if ((dd[a] < 0.0) != (dd[bb] < 0.0)) p[np++] = v[a] + (v[bb] - v[a]) * (dd[a] / (dd[a] - dd[bb]));
                    }
                    if (np == 2) {
                        // uv = (the line's plane index, its axis): what "a hue per line" colours by
                        MeshPaneVert q0 = lineVert(p[0]), q1 = lineVert(p[1]);
                        q0.u = q1.u = (float)k; q0.v = q1.v = (float)ax;
                        b.v.push_back(q0); b.v.push_back(q1);
                    }
                }
            }
        }
        if (!b.v.empty()) { out.push_back(std::move(b)); g.gridSecOf.push_back((int)si); }
    }
}

// ---- the SLAB (0.374.0): only a slice of the see-through and gridded parts is drawn ------------
// Across x, y, z or the view direction; its centre travels over those parts' extent along that
// direction (slicePos 0..1) and its thickness is a fraction of it. Recomputed every frame: a slab
// across the view turns with the orbit.
static void updateSlab(GroomState& g) {
    g.slabHalf = 0.0;
    if (!g.sliceOn) return;
    const Vec3 n = g.sliceAxis == 0 ? Vec3{ 1, 0, 0 } : g.sliceAxis == 1 ? Vec3{ 0, 1, 0 } : g.sliceAxis == 2 ? Vec3{ 0, 0, 1 } : g.cam.toward();
    // it travels across the parts you look into (see-through or gridded); with none of those and
    // "solid parts too", across every shown part
    double lo = 1e300, hi = -1e300;
    for (int pass = 0; pass < 2 && !(hi > lo); ++pass) {
        if (pass == 1 && !g.sliceSolids) break;
        for (const Section& s : g.sections) {
            if (!s.visible || (pass == 0 && !(s.grid || s.opacity < 0.999f))) continue;
            for (int c = 0; c < 8; ++c) {
                const Vec3 p{ (c & 1) ? s.hi.x : s.lo.x, (c & 2) ? s.hi.y : s.lo.y, (c & 4) ? s.hi.z : s.lo.z };
                const double t = dot(n, p);
                lo = std::min(lo, t); hi = std::max(hi, t);
            }
        }
    }
    if (!(hi > lo)) return;
    g.slabN = n;
    g.slabD = lo + (double)std::clamp(g.slicePos, 0.0f, 1.0f) * (hi - lo);
    g.slabHalf = 0.5 * (double)std::clamp(g.sliceThick, 0.002f, 1.0f) * (hi - lo);
}
// The cut the slab's centre plane makes through the parts it slices: their CROSS-SECTION, drawn
// bright -- in a horizontal slab through a hairdo, the outline of every lock at that height.
static void buildSliceLines(GroomState& g, std::vector<LineBatch>& out) {
    out.clear();
    if (g.slabHalf <= 0.0) return;
    const Scene& sc = g.loaded.scene;
    LineBatch b;
    b.slab = false;
    b.rgba[0] = 1.0f; b.rgba[1] = 1.0f; b.rgba[2] = 1.0f; b.rgba[3] = 1.0f;
    for (const Section& S : g.sections) {
        if (!slabbed(g, S)) continue;
        const size_t n = S.triCount();
        for (size_t i = 0; i < n; ++i) {
            const Tri& t = S.tri(sc, i);
            const Vec3 v[3] = { t.v0, t.v1, t.v2 };
            double dd[3];
            for (int k = 0; k < 3; ++k) { dd[k] = dot(g.slabN, v[k]) - g.slabD; if (dd[k] == 0.0) dd[k] = 1e-12; }
            if ((dd[0] < 0.0) == (dd[1] < 0.0) && (dd[1] < 0.0) == (dd[2] < 0.0)) continue;
            Vec3 p[2]; int np = 0;
            for (int e = 0; e < 3 && np < 2; ++e) {
                const int a = e, c = (e + 1) % 3;
                if ((dd[a] < 0.0) != (dd[c] < 0.0)) p[np++] = v[a] + (v[c] - v[a]) * (dd[a] / (dd[a] - dd[c]));
            }
            if (np == 2) addSegment(b, p[0], p[1]);
        }
    }
    if (!b.v.empty()) out.push_back(std::move(b));
}

// ---- placing points (0.373.0): where a click puts the selected strand's next point ---------------
// Its ROOT (first point) always goes on a roots section, through anything see-through. After that,
// by `placeMode`: 0 on surfaces (the roots, or any with "any mesh"), 1 IN THE AIR -- on the plane
// facing the viewer through the point it follows, so a strand is drawn where it is clicked and
// then shaped by orbiting and dragging -- or 2 INSIDE `insideSec`, `insideDepth` of the way in.
static bool placeNextPoint(GroomState& g, const ImVec2& mouse, groom::Node* n, Vec3& out, std::string& why) {
    if (!n || !n->kids.empty() || n->ref) { why = "select a strand first (click one of its points, or its name in the Curves tree), or press N for a new one"; return false; }
    const bool root = n->pts.empty();
    Vec3 o, d; g.cam.ray(mouse, o, d);
    if (root || g.placeMode == 0) {
        const Pick pk = pickSurfaceRay(g, o, d, true);
        if (!pk.hit) {
            why = root ? "a strand's ROOT goes on a roots section (" + g.target + ") -- click on it"
                       : "no surface under the click (on " + g.target + "; tick 'any mesh' to use any)";
            return false;
        }
        out = pk.p; return true;
    }
    if (g.placeMode == 2) {
        if (insidePoint(g, o, d, g.insideDepth, out)) return true;
        why = "no \"" + ((g.insideSec >= 0 && g.insideSec < (int)g.sections.size()) ? g.sections[(size_t)g.insideSec].label : std::string("?")) +
              "\" under the click (or something solid is in front of it)";
        return false;
    }
    const int after = (g.selNode == n->id && g.selPt >= 0 && g.selPt < (int)n->pts.size()) ? g.selPt : (int)n->pts.size() - 1;
    const Vec3 ref = xfOf(g, n->id).apply(n->pts[(size_t)after].p);
    out = o + d * dot(ref - o, d);
    return true;
}

// ---- sketching (0.373.0): a drag from a roots section draws a new strand along the drag ----------
static void beginSketch(GroomState& g, const Pick& root, const ImVec2& mouse) {
    const int before = g.selNode;
    newStrand(g);                                        // selects it; one undo covers the whole sketch
    groom::Node* n = (g.selNode >= 0 && g.selNode != before) ? groom::findNode(g.model, g.selNode) : nullptr;
    if (!n || !n->pts.empty()) return;                   // newStrand could not make one (it has said why)
    groom::Pt p; p.p = snapMicron(xfOf(g, n->id).inverse().apply(root.p)); p.edited = true;
    n->pts.push_back(p); g.selPt = 0;
    markEdited(g, n->id);
    g.sketching = true; g.sketchLast = mouse;
    g.status = "sketching \"" + n->name + "\" -- release the button to finish";
}
static void sketchAppend(GroomState& g, const ImVec2& mouse) {
    groom::Node* n = (g.selNode >= 0) ? groom::findNode(g.model, g.selNode) : nullptr;
    if (!n) { g.sketching = false; return; }
    Vec3 p; std::string why;
    if (!placeNextPoint(g, mouse, n, p, why)) return;    // e.g. the drag left the section: skip this sample
    groom::Pt q; q.p = snapMicron(xfOf(g, n->id).inverse().apply(p)); q.edited = true;
    n->pts.push_back(q); g.selPt = (int)n->pts.size() - 1;
    markEdited(g, n->id);
}

// ---- selecting several, and deleting (0.373.0) ------------------------------------------------------
// Shift-drag on empty space: the VISIBLE points inside the box (a point behind a solid surface is
// not taken); Ctrl held as well adds to what is already selected.
static void finishBoxSelect(GroomState& g, bool add) {
    const float x0 = std::min(g.boxA.x, g.boxB.x), x1 = std::max(g.boxA.x, g.boxB.x);
    const float y0 = std::min(g.boxA.y, g.boxB.y), y1 = std::max(g.boxA.y, g.boxB.y);
    if (!add) g.selPts.clear();
    if (!(g.showCurves && g.showPoints && g.levelOn[0])) { g.status = "points are hidden -- tick 'curves', 'points' and level 0 to box-select them"; return; }
    if (g.sectionsDirty) updateTriPass(g);
    const Vec3 toward = g.cam.toward();
    const double farD = 4.0 * g.cam.diag, tol = 0.02 * g.ext;
    forEachTopCurve(g.model, [&](groom::Node& top) {
        std::vector<groom::Node*> all; groom::collectNodes(top, all);
        for (groom::Node* n : all) {
            if (n->ref || !n->kids.empty() || !nodeVisible(g, n->id)) continue;
            const Affine xf = xfOf(g, n->id);
            const bool cut = g.sliceGuides && g.slabHalf > 0.0 && n->id != g.selNode;   // the slab hides it
            for (size_t i = 0; i < n->pts.size(); ++i) {
                const Vec3 pw = xf.apply(n->pts[i].p);
                if (cut && !inSlab(g, pw)) continue;
                const ImVec2 s = g.cam.toScreen(pw);
                if (s.x < x0 || s.x > x1 || s.y < y0 || s.y > y1) continue;
                if (occludedForHover(g, pw + toward * farD, toward * -1.0, farD - tol)) continue;
                const std::pair<int, int> key{ n->id, (int)i };
                if (std::find(g.selPts.begin(), g.selPts.end(), key) == g.selPts.end()) g.selPts.push_back(key);
            }
        }
    });
    g.status = "selected " + std::to_string(g.selPts.size()) + " point(s) -- Del deletes them, Esc clears";
}
// A curve and every reference to it by name (a guide listed in a `curve "group" { curve "g" }`):
// deleting a guide takes it out of its group too, rather than refusing because it is referenced.
static int eraseCurveAndRefs(GroomState& g, int id) {
    groom::Node* n = groom::findNode(g.model, id);
    if (!n) return 0;
    const std::string name = n->name;
    if (!name.empty()) {
        std::vector<groom::Node*> all; groom::collectNodes(g.model, all);
        std::vector<int> refs;
        for (groom::Node* r : all) if (r->ref && r->name == name) refs.push_back(r->id);
        for (int rid : refs) {
            groom::Where rw = groom::whereIs(g.model, rid);
            if (groom::eraseNode(g.model, rid) && rw.file) rw.file->dirty = true;
        }
    }
    groom::Where w = groom::whereIs(g.model, id);
    if (!groom::eraseNode(g.model, id)) return 0;
    if (w.file) w.file->dirty = true;
    return 1;
}
// Del: the box-selected points if any, else the Ctrl-selected curves, else the selected point.
// A strand left with fewer than 2 points is removed with them: the loader refuses a 1-point curve.
static void deleteSelected(GroomState& g) {
    if (!g.selPts.empty()) {
        pushUndo(g);
        std::map<int, std::vector<int>> byNode;
        for (const auto& pr : g.selPts) byNode[pr.first].push_back(pr.second);
        int count = 0, removed = 0;
        std::vector<int> emptied;
        for (auto& kv : byNode) {
            groom::Node* node = groom::findNode(g.model, kv.first);
            if (!node) continue;
            std::sort(kv.second.rbegin(), kv.second.rend());
            kv.second.erase(std::unique(kv.second.begin(), kv.second.end()), kv.second.end());
            for (int i : kv.second) if (i >= 0 && i < (int)node->pts.size()) { node->pts.erase(node->pts.begin() + i); ++count; }
            markEdited(g, node->id);
            if (node->pts.size() < 2) emptied.push_back(node->id);
        }
        for (int id : emptied) removed += eraseCurveAndRefs(g, id);
        g.selPts.clear(); g.selPt = -1;
        if (g.selNode >= 0 && !groom::findNode(g.model, g.selNode)) g.selNode = -1;
        g.lines.dirty = true;
        g.status = "deleted " + std::to_string(count) + " point(s)" +
                   (removed ? "; " + std::to_string(removed) + " strand(s) left with under 2 points were removed" : std::string());
        return;
    }
    if (!g.multi.empty()) {
        pushUndo(g);
        int count = 0;
        const std::vector<int> ids = g.multi;
        for (int id : ids) count += eraseCurveAndRefs(g, id);
        g.multi.clear(); g.selNode = g.selPt = -1;
        g.lines.dirty = true; g.stale = true;
        g.status = "deleted " + std::to_string(count) + " curve(s) (and their references in groups)";
        return;
    }
    deletePoint(g);
}
// every strand (a curve with points, not a curve of curves) that is shown
static void selectAllStrands(GroomState& g) {
    g.multi.clear(); g.selPts.clear();
    std::vector<groom::Node*> all; groom::collectNodes(g.model, all);
    for (groom::Node* n : all) if (!n->ref && n->kids.empty() && nodeVisible(g, n->id)) g.multi.push_back(n->id);
    g.status = "selected " + std::to_string(g.multi.size()) + " strand(s) -- Del deletes them, G groups them, Esc clears";
}
// the box-selected points -> the strands they belong to (to delete or group whole strands)
static void selectStrandsOfPoints(GroomState& g) {
    std::vector<int> ids;
    for (const auto& pr : g.selPts) if (std::find(ids.begin(), ids.end(), pr.first) == ids.end()) ids.push_back(pr.first);
    g.selPts.clear();
    g.multi = ids;
    g.status = "selected the " + std::to_string(ids.size()) + " strand(s) those points belong to -- Del deletes them, G groups them, Esc clears";
}
static void clearSelection(GroomState& g) {
    g.selPts.clear(); g.multi.clear(); g.boxing = false;
    g.status = "selection cleared";
}

// ---- the pane: the scene, the curves, the hair; and the mouse on it -------------------------------
static void groomHandleInput(GroomState& g, bool hovered) {
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos;
    const float pickPx = 8.0f * std::max(io.FontGlobalScale, 1.0f);
    // hover: the nearest point on screen that the eye can actually see -- a point on the far
    // side of the head projects onto the near side, and grabbing it would drag the wrong
    // strand. A fresh (empty) strand, or Alt, turns hovering off so a click plots.
    g.hoverNode = g.hoverPt = -1;
    if (g.sectionsDirty) updateTriPass(g);                 // which triangles the eye looks through
    groom::Node* selN = (g.selNode >= 0) ? groom::findNode(g.model, g.selNode) : nullptr;
    const bool freshStrand = selN && !selN->ref && selN->kids.empty() && selN->pts.empty();
    // a box being dragged out (Shift-drag on empty space): the points inside it on release
    if (g.boxing) {
        g.boxB = mouse;
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) { g.boxing = false; finishBoxSelect(g, io.KeyCtrl); }
        return;
    }
    // a strand being sketched: a point every `sketchStep` pixels of the drag, until release
    if (g.sketching) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            const float dx = mouse.x - g.sketchLast.x, dy = mouse.y - g.sketchLast.y;
            const float step = g.sketchStep * std::max(io.FontGlobalScale, 1.0f);
            if (dx * dx + dy * dy >= step * step) { sketchAppend(g, mouse); g.sketchLast = mouse; }
        } else {
            g.sketching = false;
            groom::Node* n = (g.selNode >= 0) ? groom::findNode(g.model, g.selNode) : nullptr;
            const float dx = mouse.x - g.pressPos.x, dy = mouse.y - g.pressPos.y;
            if (n && n->pts.size() < 2 && dx * dx + dy * dy >= 16.0f) {   // a short drag still ends where released
                sketchAppend(g, mouse);
                n = (g.selNode >= 0) ? groom::findNode(g.model, g.selNode) : nullptr;
            }
            if (n && n->pts.size() < 2)
                g.status = "planted the root of \"" + n->name + "\" -- drag further to sketch, or click to add its next points";
            else if (n)
                g.status = "sketched \"" + n->name + "\": " + std::to_string(n->pts.size()) + " points -- drag them to adjust, or click to extend it";
        }
        return;
    }
    if (hovered && !g.dragging && !io.KeyAlt && !freshStrand && g.pickBald < 0 && g.showCurves && g.showPoints && g.levelOn[0]) {
        float bestD = pickPx * pickPx;
        const Vec3 toward = g.cam.toward();
        const double farD = 4.0 * g.cam.diag, tol = 0.02 * g.ext;
        forEachTopCurve(g.model, [&](groom::Node& top) {
            std::vector<groom::Node*> all; groom::collectNodes(top, all);
            for (groom::Node* n : all) {
                if (n->ref || !n->kids.empty() || !nodeVisible(g, n->id)) continue;
                const Affine xf = xfOf(g, n->id);
                const bool cut = g.sliceGuides && g.slabHalf > 0.0 && n->id != g.selNode;   // the slab hides it
                for (size_t i = 0; i < n->pts.size(); ++i) {
                    const Vec3 pw = xf.apply(n->pts[i].p);
                    if (cut && !inSlab(g, pw)) continue;
                    const ImVec2 s = g.cam.toScreen(pw);
                    const float dx = s.x - mouse.x, dy = s.y - mouse.y, d2 = dx * dx + dy * dy;
                    if (d2 >= bestD) continue;
                    // behind a surface the eye sees (not a hidden or see-through section)
                    if (occludedForHover(g, pw + toward * farD, toward * -1.0, farD - tol)) continue;
                    bestD = d2; g.hoverNode = n->id; g.hoverPt = (int)i;
                }
            }
        });
    }
    // press
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        g.pressPos = mouse;
        if (g.hoverNode >= 0) {
            groom::Node* n = groom::findNode(g.model, g.hoverNode);
            if (n && g.hoverPt < (int)n->pts.size()) {
                // a point outside the box selection starts over (Del must not take points you forgot)
                if (!g.selPts.empty() && std::find(g.selPts.begin(), g.selPts.end(), std::make_pair(g.hoverNode, g.hoverPt)) == g.selPts.end())
                    g.selPts.clear();
                select(g, g.hoverNode, g.hoverPt);
                g.dragging = true; g.dragMoved = false;
                g.dragNode = g.hoverNode; g.dragPt = g.hoverPt; g.dragMouse0 = mouse;
                g.dragP0 = xfOf(g, n->id).apply(n->pts[(size_t)g.hoverPt].p);
                const Pick pk = pickSurface(g, g.cam.toScreen(g.dragP0), false);
                g.dragN = pk.hit ? pk.n : g.cam.toward();
                // How it moves: Shift along the surface normal, Ctrl in the screen plane; otherwise a
                // ROOT slides on its roots section and a later point moves the way new points are
                // placed -- on surfaces, in the screen plane (in the air), or through the "inside"
                // section keeping the depth it has now (its fraction of the interval under the cursor)
                if (io.KeyShift)      g.dragMode = 1;
                else if (io.KeyCtrl)  g.dragMode = 2;
                else if (g.hoverPt == 0 || g.placeMode == 0) g.dragMode = 0;
                else if (g.placeMode == 2 && g.insideSec >= 0) {
                    Vec3 o, dd; g.cam.ray(g.cam.toScreen(g.dragP0), o, dd);
                    double t0, t1;
                    g.dragFrac = (insideInterval(g, o, dd, g.insideSec, t0, t1) && t1 > t0)
                                     ? std::clamp((dot(g.dragP0 - o, dd) - t0) / (t1 - t0), 0.0, 1.0)
                                     : (double)g.insideDepth;
                    g.dragMode = 3;
                } else g.dragMode = 2;
                g.pressedEmpty = false;
            }
        } else if (io.KeyShift) {
            // Shift-drag on empty space: box-select points (Ctrl as well: add to the selection)
            g.boxing = true; g.boxA = g.boxB = mouse;
            g.pressedEmpty = false;
            return;
        } else if (g.sketchMode && g.pickBald < 0) {
            // SKETCH: a press on a roots section starts a new strand there, drawn along the drag
            const Pick pk = pickSurface(g, mouse, true);
            if (pk.hit) { beginSketch(g, pk, mouse); g.pressedEmpty = false; return; }
            g.pressedEmpty = true;                                // off the roots: an orbit, as usual
        } else {
            g.pressedEmpty = true;
        }
    }
    // drag a point
    if (g.dragging) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            const ImVec2 delta(mouse.x - g.dragMouse0.x, mouse.y - g.dragMouse0.y);
            groom::Node* n = groom::findNode(g.model, g.dragNode);
            if (!n || g.dragPt < 0 || g.dragPt >= (int)n->pts.size()) { g.dragging = false; return; }
            if (!g.dragMoved && delta.x * delta.x + delta.y * delta.y < 9.0f) return;
            if (!g.dragMoved) { pushUndo(g); g.dragMoved = true; }
            const double wpp = g.cam.worldPerPixel();
            Vec3 pw;
            if (g.dragMode == 0) {
                const Pick pk = pickSurface(g, mouse, false);
                if (!pk.hit) return;                         // off the surface: the point stays
                pw = pk.p;
            } else if (g.dragMode == 1) {
                pw = g.dragP0 + g.dragN * (-(double)delta.y * wpp);
            } else if (g.dragMode == 2) {
                pw = g.dragP0 + g.cam.right() * ((double)delta.x * wpp) - g.cam.up() * ((double)delta.y * wpp);
            } else {
                Vec3 o, dd; g.cam.ray(mouse, o, dd);
                if (!insidePoint(g, o, dd, g.dragFrac, pw)) return;   // off the section: the point stays
            }
            groom::Pt& p = n->pts[(size_t)g.dragPt];
            p.p = snapMicron(xfOf(g, n->id).inverse().apply(pw));
            p.edited = true;
            markEdited(g, n->id);
        } else {
            g.dragging = false;
            if (g.dragMoved) {
                groom::Node* n = groom::findNode(g.model, g.dragNode);
                g.status = std::string(g.dragMode == 0 ? "slid" : g.dragMode == 1 ? "lifted" : g.dragMode == 2 ? "moved" : "moved (inside)") + " point " + std::to_string(g.dragPt) + " of \"" + (n ? n->name : std::string("?")) + "\"";
            }
        }
        return;
    }
    // a click on the surface (a press that did not orbit) plots a point
    if (g.pressedEmpty && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        g.pressedEmpty = false;
        const float dx = mouse.x - g.pressPos.x, dy = mouse.y - g.pressPos.y;
        if (hovered && g.pickBald >= 0 && dx * dx + dy * dy < 9.0f) { placeBald(g, mouse); return; }
        if (hovered && g.addOnClick && dx * dx + dy * dy < 9.0f) {
            // the selected strand's next point: its root on a roots section, then by "place new points"
            groom::Node* sn = (g.selNode >= 0) ? groom::findNode(g.model, g.selNode) : nullptr;
            Pick pk; std::string why;
            if (placeNextPoint(g, mouse, sn, pk.p, why)) { pk.hit = true; addPointAt(g, pk); }
            else g.status = why;
        }
    }
}

// What the next click / drag in the pane will do, in words, above the pane (0.373.0).
static std::string paneHint(GroomState& g) {
    const char* nav = g.sliceOn ? "  |  drag empty space or right-drag: orbit, wheel: zoom, Ctrl+wheel: move the slice, F1: help"
                                : "  |  drag empty space or right-drag: orbit, wheel: zoom, F1: help";
    if (g.pickBald >= 0) return std::string("click the surface to place the bald zone's centre") + nav;
    if (g.sketchMode)
        return "SKETCH: press on a roots section (" + g.target + ") and drag -- a new strand follows the drag, its points placed " +
               (g.placeMode == 0 ? "on surfaces" : g.placeMode == 1 ? "in the air" : "inside the chosen section") + nav;
    groom::Node* n = (g.selNode >= 0) ? groom::findNode(g.model, g.selNode) : nullptr;
    if (!n || n->ref || !n->kids.empty())
        return std::string("press N for a new strand, or click a point to select its strand; Shift-drag: box-select points") + nav;
    if (n->pts.empty()) return "click on a roots section (" + g.target + ") to plant the ROOT of \"" + n->name + "\"" + nav;
    const std::string after = (g.selPt >= 0 && g.selPt < (int)n->pts.size()) ? "after point " + std::to_string(g.selPt) : "at the end";
    std::string where;
    if (g.placeMode == 0) where = "ON the surface you click";
    else if (g.placeMode == 1) where = "IN THE AIR where you click (at the depth of the point it follows; orbit to see it, then drag it)";
    else where = "INSIDE \"" + ((g.insideSec >= 0 && g.insideSec < (int)g.sections.size()) ? g.sections[(size_t)g.insideSec].label : std::string("?")) + "\" where you click";
    return "\"" + n->name + "\": a click adds a point " + after + ", " + where + "; drag a point to move it" + nav;
}

static void drawGroomPane(GroomState& g, ID3D11Device* dev, ID3D11DeviceContext* ctx) {
    MeshView& view = g.view;
    ImGui::TextWrapped("%s", paneHint(g).c_str());
    // the status line under the view, measured first so the view leaves it the room it wraps to
    std::string statusLine;
    {
        int totalTris = 0; for (const auto& m : g.meshes) totalTris += m.nfaces;
        const char* placeWords[] = { "on surfaces", "in the air", "inside " };
        const bool insideOk = g.insideSec >= 0 && g.insideSec < (int)g.sections.size();
        statusLine = "roots on " + g.target + (g.pickAny ? " or any mesh" : "") + "  |  new points " + placeWords[std::clamp(g.placeMode, 0, 2)] +
                     ((g.placeMode == 2 && insideOk) ? g.sections[(size_t)g.insideSec].label : std::string());
        if (!g.selPts.empty()) statusLine += "  |  " + std::to_string(g.selPts.size()) + " point(s) selected";
        else if (!g.multi.empty()) statusLine += "  |  " + std::to_string(g.multi.size()) + " curve(s) selected";
        if (g.stale) statusLine += "  |  placed strands and fur are STALE (save + reload)";
        statusLine += "  |  " + std::to_string(g.meshes.size()) + " section(s), " + std::to_string(totalTris) + " tris, " +
                      std::to_string(g.loaded.scene.curves.size()) + " strand(s) of fur";
    }
    ImVec2 avail = ImGui::GetContentRegionAvail();
    avail.y -= ImGui::CalcTextSize(statusLine.c_str(), nullptr, false, avail.x).y + ImGui::GetStyle().ItemSpacing.y;
    if (avail.y < 80.0f) avail.y = 80.0f;
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("groom_canvas", avail, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const bool hovered = ImGui::IsItemHovered();
    if (!g.dragging && g.pressedEmpty && ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        ImVec2 d = ImGui::GetIO().MouseDelta;
        view.yaw += d.x * 0.01f; view.pitch += d.y * 0.01f;
    }
    // the right button always orbits (what a left drag does is taken by sketching and box selection)
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
        ImVec2 d = ImGui::GetIO().MouseDelta;
        view.yaw += d.x * 0.01f; view.pitch += d.y * 0.01f;
    }
    if (hovered) {
        const float w = ImGui::GetIO().MouseWheel;
        if (w != 0.0f) {
            if (g.sliceOn && ImGui::GetIO().KeyCtrl) g.slicePos = std::clamp(g.slicePos + w * 0.02f, 0.0f, 1.0f);   // Ctrl+wheel: the slab
            else view.zoom *= (1.0f + w * 0.1f);
        }
    }
    if (view.zoom < 0.05f) view.zoom = 0.05f;
    MeshGpu& gpu = view.gpu;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 br(origin.x + avail.x, origin.y + avail.y);
    bool ok = gpu.buildPipeline(dev);
    if (ok && (!gpu.geomReady || gpu.geomGen != view.geomGen)) ok = gpu.uploadGeometry(dev, g.meshes, view.geomGen);
    if (ok && g.lines.dirty) { std::vector<LineBatch> batches; groomBuildLines(g, batches); ok = g.lines.upload(dev, batches); }
    if (ok && g.gridDirty) { std::vector<LineBatch> batches; buildGridLines(g, batches); ok = g.gridGpu.upload(dev, batches); g.gridDirty = false; }
    if (ok) ok = gpu.ensureTargets(dev, (int)(avail.x + 0.5f), (int)(avail.y + 0.5f));
    if (ok) {
        float cy = std::cos(view.yaw),   sy = std::sin(view.yaw);
        float cx = std::cos(view.pitch), sx = std::sin(view.pitch);
        const float R[3][3] = { { cy, 0.0f, sy }, { sx * sy, cx, -sx * cy }, { -cx * sy, sx, cx * cy } };
        // Orthographic about the FRAME (groomComputeFrame), not the meshes' union box the
        // upload baked: a scene's floor must not decide where the head sits in the pane.
        float scale = 0.42f * std::min(avail.x, avail.y) / (0.5f * g.frameExt + 1e-3f);
        float s  = scale * view.zoom;
        float ax = (avail.x > 0.0f) ? 2.0f * s / avail.x : 0.0f;
        float ay = (avail.y > 0.0f) ? 2.0f * s / avail.y : 0.0f;
        float kz = 0.5f / g.frameDiag;
        auto dotMid = [&](int r) { return R[r][0] * g.frameMid[0] + R[r][1] * g.frameMid[1] + R[r][2] * g.frameMid[2]; };
        // keep the camera for picking
        for (int r = 0; r < 3; ++r) for (int k = 0; k < 3; ++k) g.cam.R[r][k] = R[r][k];
        for (int k = 0; k < 3; ++k) g.cam.mid[k] = g.frameMid[k];
        g.cam.ax = ax; g.cam.ay = ay; g.cam.s = s; g.cam.diag = g.frameDiag; g.cam.origin = origin; g.cam.avail = avail; g.cam.valid = true;
        MeshGpu::CB c = {};
        const float rowScale[3] = { ax, ay, -kz };
        for (int r = 0; r < 3; ++r) for (int k = 0; k < 3; ++k) c.mvp[r * 4 + k] = rowScale[r] * R[r][k];
        c.mvp[0 * 4 + 3] = -ax * dotMid(0);
        c.mvp[1 * 4 + 3] = -ay * dotMid(1);
        c.mvp[2 * 4 + 3] =  kz * dotMid(2) + 0.5f;
        c.mvp[3 * 4 + 3] = 1.0f;
        for (int r = 0; r < 3; ++r) {
            float* dst = (r == 0) ? c.rot0 : (r == 1) ? c.rot1 : c.rot2;
            dst[0] = R[r][0]; dst[1] = R[r][1]; dst[2] = R[r][2]; dst[3] = -dotMid(r);
        }
        // this frame's slab (a slab across the view turns with the orbit), and the cross-section
        // at its centre, rebuilt when the plane or the parts it cuts change
        updateSlab(g);
        {
            std::string key;
            if (g.slabHalf > 0.0) {
                char kb[160];
                std::snprintf(kb, sizeof kb, "%.9g %.9g %.9g %.9g ", g.slabN.x, g.slabN.y, g.slabN.z, g.slabD);
                key = kb;
                for (const Section& sec : g.sections) key += slabbed(g, sec) ? '1' : '0';
            }
            if (key != g.sliceKey) {
                std::vector<LineBatch> batches; buildSliceLines(g, batches);
                g.sliceGpu.upload(dev, batches);
                g.sliceKey = key;
            }
        }
        // setCB: a plain draw -- the colour, shading on/off, a colour mode. setCBx: the groom tool's
        // extras (0.374.0) -- the slab (cut to it), bright edges, and the hue modes (mode 4, flags
        // 1 per line / 2 by depth over z0..z1)
        auto setCBx = [&](const float rgba[4], float shadeOn, float mode, bool slab, float rim, int hueFlags, float z0, float z1) {
            for (int k = 0; k < 4; ++k) c.baseColor[k] = rgba[k];
            c.opts[0] = shadeOn; c.opts[1] = mode; c.opts[2] = (float)hueFlags; c.opts[3] = 0.0f;
            const bool cut = slab && g.slabHalf > 0.0;
            c.slab[0] = (float)g.slabN.x; c.slab[1] = (float)g.slabN.y; c.slab[2] = (float)g.slabN.z; c.slab[3] = (float)g.slabD;
            c.extra[0] = cut ? (float)g.slabHalf : 0.0f; c.extra[1] = rim; c.extra[2] = z0; c.extra[3] = z1;
            D3D11_MAPPED_SUBRESOURCE ms;
            if (ctx->Map(gpu.cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms) == S_OK) { std::memcpy(ms.pData, &c, sizeof(c)); ctx->Unmap(gpu.cb, 0); }
        };
        auto setCB = [&](const float rgba[4], float shadeOn, float mode) { setCBx(rgba, shadeOn, mode, false, 0.0f, 0, 0.0f, 1.0f); };
        // a section's depth range in the pane's depth units (its bounds' corners), for hue by depth
        auto depthRange = [&](size_t si, float& z0, float& z1) {
            z0 = 1e30f; z1 = -1e30f;
            const Section& s = g.sections[si];
            for (int k = 0; k < 8; ++k) {
                const float px = (float)((k & 1) ? s.hi.x : s.lo.x), py = (float)((k & 2) ? s.hi.y : s.lo.y), pz = (float)((k & 4) ? s.hi.z : s.lo.z);
                const float z = c.mvp[8] * px + c.mvp[9] * py + c.mvp[10] * pz + c.mvp[11];
                z0 = std::min(z0, z); z1 = std::max(z1, z);
            }
        };
        auto secSlab = [&](size_t mi) { return mi < g.sections.size() && slabbed(g, g.sections[mi]); };
        const float clearCol[4] = { 14 / 255.0f, 16 / 255.0f, 20 / 255.0f, 1.0f };
        ctx->ClearRenderTargetView(gpu.rtv, clearCol);
        ctx->ClearDepthStencilView(gpu.dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);
        ID3D11RenderTargetView* rtvs[1] = { gpu.rtv };
        ctx->OMSetRenderTargets(1, rtvs, gpu.dsv);
        D3D11_VIEWPORT vp = {}; vp.Width = (float)gpu.texW; vp.Height = (float)gpu.texH; vp.MaxDepth = 1.0f;
        ctx->RSSetViewports(1, &vp);
        UINT stride = sizeof(MeshGpu::Vert), voff = 0;
        ctx->IASetInputLayout(gpu.layout);
        ctx->VSSetShader(gpu.vs, nullptr, 0); ctx->PSSetShader(gpu.ps, nullptr, 0);
        ctx->GSSetShader(nullptr, nullptr, 0); ctx->HSSetShader(nullptr, nullptr, 0); ctx->DSSetShader(nullptr, nullptr, 0);
        ctx->VSSetConstantBuffers(0, 1, &gpu.cb); ctx->PSSetConstantBuffers(0, 1, &gpu.cb);
        ctx->PSSetSamplers(0, 1, &gpu.samp);
        const float bf[4] = { 0, 0, 0, 0 };
        ctx->OMSetBlendState(gpu.blend, bf, 0xffffffff);
        ID3D11ShaderResourceView* none[1] = { nullptr };
        ctx->PSSetShaderResources(0, 1, none);
        // a section's colour: the tint cycle, and a hair gold for a reference part (a sculpted
        // hairdo the scene skips is the usual one) -- sectionTint
        auto secVisible = [&](size_t mi) { return mi >= g.sections.size() || g.sections[mi].visible; };
        auto secOpacity = [&](size_t mi) { return mi < g.sections.size() ? g.sections[mi].opacity : 1.0f; };
        auto secGrid    = [&](size_t mi) { return mi < g.sections.size() && g.sections[mi].grid; };
        if (g.showMesh && gpu.vb && gpu.ib) {
            ctx->IASetVertexBuffers(0, 1, &gpu.vb, &stride, &voff);
            ctx->IASetIndexBuffer(gpu.ib, DXGI_FORMAT_R32_UINT, 0);
            ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx->OMSetDepthStencilState(gpu.dsSolid, 0);
            // 1. the solid sections (written to depth, so they hide what is behind them); a gridded
            // one a hair farther than it is, so its own outline lines (drawn last) are not hidden by it
            for (size_t mi = 0; mi < gpu.ranges.size(); ++mi) {
                const MeshGpu::Range& r = gpu.ranges[mi];
                if (!r.indexCount || !secVisible(mi) || secOpacity(mi) < 0.999f) continue;
                // a roots section (a scalp lying on the skin) is pulled a hair forward so it shows
                // whole instead of flickering through the surface it lies on (0.374.0)
                const bool rootsSec = mi < g.sections.size() && g.sections[mi].roots;
                ctx->RSSetState(secGrid(mi) ? gpu.rsBack : rootsSec ? gpu.rsFront : gpu.rsSolid);
                float rgba[4]; sectionTint(g, mi, 1.0f, rgba);
                setCBx(rgba, view.shade ? 1.0f : 0.0f, 0.0f, secSlab(mi), 0.0f, 0, 0.0f, 1.0f);
                ctx->DrawIndexed(r.indexCount, r.firstIndex, r.baseVertex);
            }
            if (view.wire) {
                ctx->RSSetState(gpu.rsWire);
                ctx->OMSetDepthStencilState(gpu.dsWire, 0);
                const float wireCol[4] = { 30 / 255.0f, 30 / 255.0f, 36 / 255.0f, 120 / 255.0f };
                for (size_t mi = 0; mi < gpu.ranges.size(); ++mi) {
                    const MeshGpu::Range& r = gpu.ranges[mi];
                    if (!r.indexCount || !secVisible(mi)) continue;
                    setCBx(wireCol, 0.0f, 0.0f, secSlab(mi), 0.0f, 0, 0.0f, 1.0f);
                    ctx->DrawIndexed(r.indexCount, r.firstIndex, r.baseVertex);
                }
            }
            // 2. the see-through sections (0.372.0): blended over what is solid, depth-tested but NOT
            // written -- so the curves drawn next are not hidden by a shell they sit inside, and a
            // strand grown inside a translucent hairdo stays in plain view. With bright edges
            // (0.374.0) opaque where seen edge-on; with hue by depth, bright yellow near .. dim purple far.
            ctx->RSSetState(gpu.rsSolid);
            ctx->OMSetDepthStencilState(gpu.dsWire, 0);
            for (size_t mi = 0; mi < gpu.ranges.size(); ++mi) {
                const MeshGpu::Range& r = gpu.ranges[mi];
                const float op = secOpacity(mi);
                if (!r.indexCount || !secVisible(mi) || op >= 0.999f) continue;
                if (op <= 0.001f && !g.rims) continue;              // wholly clear: nothing to draw
                float rgba[4]; sectionTint(g, mi, op, rgba);
                float z0 = 0.0f, z1 = 1.0f;
                if (g.hueDepth) depthRange(mi, z0, z1);
                setCBx(rgba, view.shade ? 1.0f : 0.0f, g.hueDepth ? 4.0f : 0.0f, secSlab(mi), g.rims ? 1.0f : 0.0f,
                       g.hueDepth ? 2 : 0, z0, z1);
                ctx->DrawIndexed(r.indexCount, r.firstIndex, r.baseVertex);
            }
        }
        if (g.lines.vb && g.lines.count) {
            ctx->IASetVertexBuffers(0, 1, &g.lines.vb, &stride, &voff);
            ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
            ctx->RSSetState(gpu.rsSolid);
            ctx->OMSetDepthStencilState(gpu.dsWire, 0);          // tested, not written
            for (size_t bi = 0; bi < g.lines.first.size(); ++bi) {
                if (!g.lines.num[bi]) continue;
                // unlit: the batch colour; cut by the slab when the guides are (never the selected strand)
                setCBx(g.lines.colour[bi].data(), 0.0f, 0.0f, g.sliceGuides && bi < g.lines.slab.size() && g.lines.slab[bi], 0.0f, 0, 0.0f, 1.0f);
                ctx->Draw(g.lines.num[bi], g.lines.first[bi]);
            }
        }
        // the slab's centre cut (0.374.0): the parts' cross-section there, in white, over the rest --
        // three pixels wide (a D3D11 line is one): drawn five times, shifted a pixel each way
        if (g.showMesh && g.slabHalf > 0.0 && g.sliceGpu.vb && g.sliceGpu.count) {
            ctx->IASetVertexBuffers(0, 1, &g.sliceGpu.vb, &stride, &voff);
            ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
            ctx->RSSetState(gpu.rsSolid);
            ctx->OMSetDepthStencilState(gpu.dsWire, 0);
            const float m3 = c.mvp[3], m7 = c.mvp[7];
            const float px = 2.0f / (float)std::max(gpu.texW, 1), py = 2.0f / (float)std::max(gpu.texH, 1);
            static const float offs[5][2] = { { 0, 0 }, { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
            for (const auto& o : offs) {
                c.mvp[3] = m3 + o[0] * px; c.mvp[7] = m7 + o[1] * py;
                for (size_t bi = 0; bi < g.sliceGpu.first.size(); ++bi) {
                    if (!g.sliceGpu.num[bi]) continue;
                    setCB(g.sliceGpu.colour[bi].data(), 0.0f, 0.0f);
                    ctx->Draw(g.sliceGpu.num[bi], g.sliceGpu.first[bi]);
                }
            }
            c.mvp[3] = m3; c.mvp[7] = m7;
        }
        // 3. GRID OUTLINES (0.373.0), last: the contour lines of the gridded sections. Hidden-line
        // unless x-ray -- a depth-only pass of the see-through gridded sections first (pushed a hair
        // back, so a line on the surface passes where one behind it does not), so each shape shows
        // only its near side's contours. After the curves, so that depth does not hide the strands
        // grown inside a shell; a front contour is drawn over them, as it should be.
        if (g.showMesh && g.gridGpu.vb && g.gridGpu.count) {
            if (!g.gridXray && gpu.vb && gpu.ib) {
                ctx->IASetVertexBuffers(0, 1, &gpu.vb, &stride, &voff);
                ctx->IASetIndexBuffer(gpu.ib, DXGI_FORMAT_R32_UINT, 0);
                ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                ctx->RSSetState(gpu.rsBack);
                ctx->OMSetDepthStencilState(gpu.dsSolid, 0);
                ctx->OMSetBlendState(gpu.blendNoColor, bf, 0xffffffff);
                const float none4[4] = { 0, 0, 0, 0 };
                for (size_t mi = 0; mi < gpu.ranges.size(); ++mi) {
                    const MeshGpu::Range& r = gpu.ranges[mi];
                    if (!r.indexCount || !secVisible(mi) || !secGrid(mi) || secOpacity(mi) >= 0.999f) continue;
                    setCBx(none4, 0.0f, 0.0f, secSlab(mi), 0.0f, 0, 0.0f, 1.0f);   // the slab cuts the depth too
                    ctx->DrawIndexed(r.indexCount, r.firstIndex, r.baseVertex);
                }
                ctx->OMSetBlendState(gpu.blend, bf, 0xffffffff);
            }
            ctx->IASetVertexBuffers(0, 1, &g.gridGpu.vb, &stride, &voff);
            ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
            ctx->RSSetState(gpu.rsSolid);
            ctx->OMSetDepthStencilState(gpu.dsWire, 0);
            for (size_t bi = 0; bi < g.gridGpu.first.size() && bi < g.gridSecOf.size(); ++bi) {
                const size_t si = (size_t)g.gridSecOf[bi];
                if (!g.gridGpu.num[bi] || si >= g.sections.size() || !g.sections[si].visible || !g.sections[si].grid) continue;
                // the section's own colour: darkened over an opaque fill, brightened over a faint one --
                // or (0.374.0) a hue per line and / or a hue by depth
                float rgba[4]; sectionTint(g, si, 0.85f, rgba);
                const bool overFill = g.sections[si].opacity >= 0.5f;
                for (int k = 0; k < 3; ++k) rgba[k] = overFill ? rgba[k] * 0.30f : std::min(1.0f, rgba[k] * 1.1f + 0.05f);
                const int flags = (g.gridHueLine ? 1 : 0) | (g.hueDepth ? 2 : 0);
                float z0 = 0.0f, z1 = 1.0f;
                if (g.hueDepth) depthRange(si, z0, z1);
                setCBx(rgba, 0.0f, flags ? 4.0f : 0.0f, secSlab(si), 0.0f, flags, z0, z1);
                ctx->Draw(g.gridGpu.num[bi], g.gridGpu.first[bi]);
            }
        }
        ID3D11RenderTargetView* noRtv[1] = { nullptr };
        ctx->OMSetRenderTargets(1, noRtv, nullptr);
        if (gpu.msTex && gpu.colorTex) ctx->ResolveSubresource(gpu.colorTex, 0, gpu.msTex, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
    }
    if (ok && gpu.srv) dl->AddImage((ImTextureID)(intptr_t)gpu.srv, origin, br);
    else {
        dl->AddRectFilled(origin, br, IM_COL32(14, 16, 20, 255));
        if (!gpu.err.empty()) dl->AddText(ImVec2(origin.x + 8.0f, origin.y + 8.0f), IM_COL32(240, 140, 110, 255), gpu.err.c_str());
    }
    if (g.cam.valid) {
        groomHandleInput(g, hovered);
        // the selected and hovered points, as overlay marks (no line rebuild on a hover)
        const float rr = 5.0f * std::max(ImGui::GetIO().FontGlobalScale, 1.0f);
        dl->PushClipRect(origin, br, true);
        auto mark = [&](int node, int pt, ImU32 col, bool filled) {
            groom::Node* n = (node >= 0) ? groom::findNode(g.model, node) : nullptr;
            if (!n || pt < 0 || pt >= (int)n->pts.size()) return;
            const ImVec2 s = g.cam.toScreen(xfOf(g, n->id).apply(n->pts[(size_t)pt].p));
            if (filled) dl->AddCircleFilled(s, rr, col); else dl->AddCircle(s, rr * 1.4f, col, 0, 2.0f);
        };
        // the box-selected points (0.373.0), and the box being dragged out
        for (const auto& pr : g.selPts) mark(pr.first, pr.second, IM_COL32(90, 220, 255, 235), true);
        if (g.boxing) {
            const ImVec2 a(std::min(g.boxA.x, g.boxB.x), std::min(g.boxA.y, g.boxB.y)), b(std::max(g.boxA.x, g.boxB.x), std::max(g.boxA.y, g.boxB.y));
            dl->AddRectFilled(a, b, IM_COL32(90, 220, 255, 40));
            dl->AddRect(a, b, IM_COL32(90, 220, 255, 200), 0.0f, 0, 1.5f);
        }
        mark(g.hoverNode, g.hoverPt, IM_COL32(255, 230, 90, 255), false);
        mark(g.selNode, g.selPt, IM_COL32(255, 255, 255, 230), true);
        dl->PopClipRect();
    }
    ImGui::TextWrapped("%s", statusLine.c_str());
}

// ---- the panel: toggles, the selection being edited, the tree, the fur ----------------------------
static void treeNode(GroomState& g, groom::Node& n, int depth) {
    if (depth > 16) return;
    groom::Node* shown = &n;
    if (n.ref) {
        shown = groom::findDef(g.model, n.name);
        if (!shown) { ImGui::TextDisabled("    curve \"%s\" (no such definition)", n.name.c_str()); return; }
    }
    const int level = groom::levelOf(g.model, *shown);
    const float* col = levelColour(level);
    ImGui::PushID(n.id);
    bool on = nodeVisible(g, shown->id);
    if (ImGui::Checkbox("##on", &on)) { g.nodeOn[shown->id] = on ? 1 : 0; g.lines.dirty = true; }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("show / hide this curve in the pane -- NOT a selection.\n"
                          "To select several curves, Ctrl-click their NAMES (they turn highlighted, marked *);\n"
                          "then Del deletes them and G groups them. Shift-drag in the pane selects points.");
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(col[0], col[1], col[2], 1.0f));
    ImGuiTreeNodeFlags fl = ImGuiTreeNodeFlags_OpenOnArrow | (shown->kids.empty() ? ImGuiTreeNodeFlags_Leaf : 0)
                          | ((g.selNode == shown->id || isMulti(g, shown->id)) ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_DefaultOpen;
    char label[256];
    std::string extra;
    if (shown->kids.empty()) extra = std::to_string(shown->pts.size()) + (shown->pts.size() == 1 ? " point" : " points");
    else {
        extra = std::to_string(shown->kids.size()) + (shown->kids.size() == 1 ? " child" : " children");
        if (shown->placed()) {
            auto it = g.recByName.find(shown->name);
            if (it != g.recByName.end()) extra += ", " + std::to_string(it->second->strands.size()) + " strands" + (g.stale ? " (stale)" : "");
        }
    }
    std::snprintf(label, sizeof label, "%s  [L%d]%s%s%s%s  %s",
                  shown->name.empty() ? "(unnamed)" : shown->name.c_str(), level,
                  shown->count() > 0 ? "  count" : "", (shown->find("density") || shown->find("density_at")) ? "  density" : "",
                  shown->closed() ? "  closed" : "", shown->rendered() ? "" : "  (definition)", extra.c_str());
    bool open = ImGui::TreeNodeEx("node", fl, "%s%s", isMulti(g, shown->id) ? "* " : "", label);
    ImGui::PopStyleColor();
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
        if (ImGui::GetIO().KeyCtrl) { toggleMulti(g, shown->id); g.selPts.clear(); }
        else select(g, (g.selNode == shown->id) ? -1 : shown->id, -1);
    }
    if (open) {
        for (groom::Node& k : shown->kids) treeNode(g, k, depth + 1);
        ImGui::TreePop();
    }
    ImGui::PopID();
}

// A curve of curves' own parameters: what places instances along its path, and its children.
static void drawNodeParams(GroomState& g, groom::Node& n) {
    const float w5 = ImGui::GetFontSize() * 5.0f;
    // WHAT THIS NODE DOES WITH ITS CHILDREN, stated before `count` is edited -- because
    // `count` is the field that switches it, and the switch is NOT additive. With no
    // count/density the node is a plain group and its children pass through bit-for-bit
    // (ftsl.h flattenCurveNode: "the instances ARE the children"). The moment it places,
    // the children stop being output and become the control cage: the path is the spline
    // through their roots and the output is `count` samples along it. So a level added on
    // top does not refine the level below, it consumes it -- which is why the tool says so
    // here rather than leaving "one level up, the next colour" to imply otherwise.
    if (!n.kids.empty()) {
        // The numbers, not just the words: what this node emits now, and what `count` would
        // replace it with. `M` (strands per child) is what a placing node multiplies, so a
        // group of 20 single strands becoming `count 9` really does end at 9, and the panel
        // says 20 -> 9 rather than leaving the user to discover it in the render.
        size_t now = 0;
        if (!n.name.empty()) {
            auto pit = g.preview.byName.find(n.name);
            if (pit != g.preview.byName.end()) now = pit->second.strands.size();
        }
        const size_t C = n.kids.size();
        if (n.placed()) {
            ImGui::TextWrapped("PLACES: its %zu child curve(s) are the control cage, NOT output --"
                               " this node emits the %zu strand(s) instead, spaced along the path"
                               " through their roots by arc length.", C, now);
        } else {
            const int cnt = (n.count() > 0) ? n.count() : 1;
            ImGui::TextWrapped("GROUPS: no count/density, so its %zu child curve(s) render as"
                               " themselves -- %zu strand(s), bit-for-bit. Setting count N here"
                               " REPLACES them with N x %zu; it is not an extra layer of detail.",
                               C, now, C ? now / C : 0);
            (void)cnt;
        }
        ImGui::Separator();
    }
    int count = n.count();
    ImGui::SetNextItemWidth(w5);
    if (ImGui::InputInt("count", &count)) {
        if (count < 0) count = 0;
        pushUndo(g);
        if (count > 0) groom::setStmt(n, "count", { std::to_string(count) }); else groom::removeStmts(n, "count");
        markEdited(g, n.id);
    }
    ImGui::SameLine();
    bool cl = n.closed();
    if (ImGui::Checkbox("closed", &cl)) { pushUndo(g); if (cl) groom::setStmt(n, "closed", {}); else groom::removeStmts(n, "closed"); markEdited(g, n.id); }
    ImGui::SameLine();
    const char* splines[] = { "uniform", "centripetal", "chordal" };
    const std::string sw = groom::splineWord(n);
    int si = (sw == "centripetal") ? 1 : (sw == "chordal") ? 2 : 0;
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8.0f);
    if (ImGui::Combo("spline", &si, splines, 3)) { pushUndo(g); if (si == 0) groom::removeStmts(n, "spline"); else groom::setStmt(n, "spline", { splines[si] }); markEdited(g, n.id); }
    double dens = groom::constDensity(n);
    ImGui::SetNextItemWidth(w5);
    if (ImGui::InputDouble("density (per m; 0 = none)", &dens, 0, 0, "%.4g")) {
        pushUndo(g);
        if (dens > 0.0) groom::setStmt(n, "density", { groom::fmtNum(dens) }); else groom::removeStmts(n, "density");
        markEdited(g, n.id);
    }
    auto keys = groom::densityKeys(n);
    bool kch = false;
    ImGui::TextUnformatted("density_at keys (t along the path 0..1, strands per metre):");
    ImGui::SameLine();
    if (ImGui::SmallButton("+ key")) { keys.push_back({ keys.empty() ? 0.0 : std::min(1.0, keys.back().first + 0.25), 10.0 }); kch = true; }
    for (size_t i = 0; i < keys.size(); ++i) {
        ImGui::PushID((int)i);
        ImGui::SetNextItemWidth(w5);
        if (ImGui::InputDouble("t", &keys[i].first, 0, 0, "%.3g")) kch = true;
        ImGui::SameLine(); ImGui::SetNextItemWidth(w5);
        if (ImGui::InputDouble("rho", &keys[i].second, 0, 0, "%.4g")) kch = true;
        ImGui::SameLine();
        if (ImGui::SmallButton("x")) { keys.erase(keys.begin() + (std::ptrdiff_t)i); kch = true; ImGui::PopID(); break; }
        ImGui::PopID();
    }
    if (kch) { pushUndo(g); groom::setDensityKeys(n, keys); markEdited(g, n.id); }
    ImGui::TextUnformatted("children (the path runs through their roots, in this order):");
    for (size_t i = 0; i < n.kids.size(); ++i) {
        ImGui::PushID(1000 + (int)i);
        const groom::Node& k = n.kids[i];
        ImGui::BulletText("%s%s", k.name.empty() ? "(inline)" : k.name.c_str(), k.ref ? "" : "  (inline)");
        ImGui::SameLine();
        if (ImGui::SmallButton("up") && i > 0) { pushUndo(g); std::swap(n.kids[i], n.kids[i - 1]); markEdited(g, n.id); }
        ImGui::SameLine();
        if (ImGui::SmallButton("down") && i + 1 < n.kids.size()) { pushUndo(g); std::swap(n.kids[i], n.kids[i + 1]); markEdited(g, n.id); }
        ImGui::SameLine();
        if (ImGui::SmallButton("remove")) { pushUndo(g); n.kids.erase(n.kids.begin() + (std::ptrdiff_t)i); markEdited(g, n.id); ImGui::PopID(); break; }
        ImGui::PopID();
    }
    if (!g.multi.empty()) {
        char bl[80]; std::snprintf(bl, sizeof bl, "add the %zu Ctrl-selected as children", g.multi.size());
        if (ImGui::Button(bl)) addMultiAsChildren(g, n.id);
    }
}

// Narrow-panel layout (0.373.0): the next item goes on this line if it fits, else on the next --
// the panel is a third of the window, and at 150 % scaling a row of buttons ran off its edge.
static void flowNext(float w) { ImGui::SameLine(); if (ImGui::GetContentRegionAvail().x < w) ImGui::NewLine(); }
static float btnW(const char* s) { return ImGui::CalcTextSize(s, nullptr, true).x + 2.0f * ImGui::GetStyle().FramePadding.x; }
static float chkW(const char* s) { return ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x + ImGui::CalcTextSize(s, nullptr, true).x; }
static void disabledWrapped(const char* s) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("%s", s);
    ImGui::PopStyleColor();
}

// The Sections panel (0.372.0): every part of every mesh, each hideable and see-through. A part
// the scene SKIPS is listed too (drawn, never rendered): that is how a sculpted hairdo the render
// replaces with strands shows up here as the shape to groom inside.
static void drawSectionsSection(GroomState& g) {
    if (!ImGui::CollapsingHeader("Sections", ImGuiTreeNodeFlags_DefaultOpen)) return;
    disabledWrapped("the meshes' parts, read from the scene file (hover here for more)");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Each row is one part: a glTF material within a mesh (\"alice2 / root.4\"), or a whole mesh\n"
                          "(an OBJ, a primitive). The list and every default come from the scene file:\n"
                          "  * a part the scene SKIPS (`skip_material`) is loaded for reference only -- never\n"
                          "    rendered -- and starts faint, its folds shown by bright edges: the shape to groom inside;\n"
                          "  * 'roots' starts ticked on the meshes the fur blocks grow on (`fur { on \"...\" }`).\n"
                          "Any of it can be changed here, for any part.");
    const float sliderW = ImGui::GetFontSize() * 3.2f;
    bool changed = false, gridChanged = false, rootsChanged = false;
    if (g.sections.empty()) ImGui::TextDisabled("(no meshes)");
    else if (ImGui::BeginTable("sections", 5, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableSetupColumn("show");
        ImGui::TableSetupColumn("opacity");
        ImGui::TableSetupColumn("grid");
        ImGui::TableSetupColumn("roots");
        ImGui::TableSetupColumn("part", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < g.sections.size(); ++i) {
            Section& s = g.sections[i];
            ImGui::PushID((int)i);
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            changed |= ImGui::Checkbox("##vis", &s.visible);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("show / hide (a hidden part is looked through)");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(sliderW);
            changed |= ImGui::SliderFloat("##op", &s.opacity, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("opacity: below 1 the part is see-through, and clicks and hovering pass through it\n(0 with 'grid' ticked: the outline alone)");
            ImGui::TableSetColumnIndex(2);
            gridChanged |= ImGui::Checkbox("##grid", &s.grid);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("grid: draw the part as an OUTLINE -- its surface's contour lines, cut by evenly spaced\nplanes -- so a folded shape reads clearly even when see-through (settings below)");
            ImGui::TableSetColumnIndex(3);
            rootsChanged |= ImGui::Checkbox("##roots", &s.roots);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("roots: a new strand's FIRST point (its root) is planted on the parts ticked here\n(by default the meshes the scene's fur blocks grow on)");
            ImGui::TableSetColumnIndex(4);
            if (s.reference) ImGui::TextColored(ImVec4(0.95f, 0.80f, 0.42f, 1.0f), "%s", s.label.c_str());
            else ImGui::TextUnformatted(s.label.c_str());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s\n%zu triangles%s%s", s.label.c_str(), s.triCount(),
                                  s.reference ? "\nskipped by the scene (`skip_material`): shown here for reference, never rendered" : "",
                                  s.roots ? "\na roots section: new strands are rooted here" : "");
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    // READING A SEE-THROUGH PART (0.374.0): edges, hues, and a slab
    ImGui::Checkbox("bright edges", &g.rims);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("See-through parts turn opaque and bright where the surface is seen EDGE-ON, and stay clear\n"
                          "where it faces you: every fold, curl and lock edge becomes an outline, and what is inside\n"
                          "stays visible. With a part's opacity at 0, the edges alone.");
    flowNext(chkW("hue by depth"));
    ImGui::Checkbox("hue by depth", &g.hueDepth);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Colour see-through parts and grid lines by their DISTANCE from you: bright yellow near, then\n"
                          "green, teal and blue, to a dim purple far. A line that dives behind a lock changes colour on the\n"
                          "way, and stacked layers separate. The colours follow the view as you orbit.");
    if (std::any_of(g.sections.begin(), g.sections.end(), [](const Section& s) { return s.grid; })) {
        ImGui::TextUnformatted("grid:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(sliderW * 1.4f);
        if (ImGui::SliderFloat("lines", &g.gridLines, 8.0f, 200.0f, "%.0f", ImGuiSliderFlags_Logarithmic)) gridChanged = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("how many contour lines fit across the framed extent (their spacing: extent / lines)");
        static const char* axisNames[3] = { "x", "y", "z" };
        for (int k = 0; k < 3; ++k) {
            flowNext(chkW(axisNames[k]));
            if (ImGui::Checkbox(axisNames[k], &g.gridAxis[k])) gridChanged = true;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("lines where the planes %s = constant cut the surface", axisNames[k]);
        }
        flowNext(chkW("hue per line"));
        ImGui::Checkbox("hue per line", &g.gridHueLine);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Each contour line its own colour -- six in turn (orange, yellow, green, cyan, blue, violet;\n"
                              "never red, the guides' colour) -- so neighbouring lines never match and one line can be\n"
                              "followed through a tangle. With 'hue by depth' as well, each line dims with distance.\n"
                              "Clearest with one direction of lines only (untick two of x / y / z).");
        flowNext(chkW("far side too"));
        ImGui::Checkbox("far side too", &g.gridXray);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("off: only the lines on the near side of each part (hidden-line); on: the far side's too");
    }
    if (ImGui::Checkbox("slice", &g.sliceOn)) g.sectionsDirty = true;   // what blocks hovering changes
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Show only a SLAB of the see-through and gridded parts, with their cross-section at its centre\n"
                          "in white: in a horizontal slab through a hairdo, the outline of every lock at that height.\n"
                          "Ctrl+wheel in the view moves it through the part. A point placed 'inside a section' goes\n"
                          "inside the slab -- the part you can see.");
    if (g.sliceOn) {
        static const char* across[4] = { "x", "y (height)", "z", "your view (depth)" };
        flowNext(ImGui::GetFontSize() * 7.0f);
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 7.0f);
        ImGui::Combo("across", &g.sliceAxis, across, 4);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("the direction the slab is thin in; 'your view' is a slab of depth, turning as you orbit");
        ImGui::SetNextItemWidth(sliderW * 2.0f);
        ImGui::SliderFloat("at", &g.slicePos, 0.0f, 1.0f, "%.3f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("where the slab's centre is, across the parts it cuts (Ctrl+wheel in the view)");
        flowNext(sliderW * 1.6f + chkW("thick"));
        ImGui::SetNextItemWidth(sliderW * 1.6f);
        ImGui::SliderFloat("thick", &g.sliceThick, 0.005f, 0.5f, "%.3f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("the slab's thickness, as a fraction of the parts' extent across it");
        flowNext(chkW("guides too"));
        ImGui::Checkbox("guides too", &g.sliceGuides);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("cut the guide curves by the slab as well (the selected strand is always drawn whole)");
        flowNext(chkW("solid parts too"));
        if (ImGui::Checkbox("solid parts too", &g.sliceSolids)) g.sectionsDirty = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("cut every part by the slab, the head and body too -- a slice of the whole scene, like a scan;\n"
                              "with 'look along it', the head's outline and every lock's around it. What is cut away\n"
                              "cannot be clicked or hovered.");
        if (g.sliceAxis < 3) {
            flowNext(btnW("look along it"));
            if (ImGui::Button("look along it")) {
                // turn the view to face the slab: its cross-section then reads as a flat outline
                // (toward = (-cos p sin y, sin p, cos p cos y): +y from above, +x from the side, +z from the front)
                if (g.sliceAxis == 1)      { g.view.pitch = 1.5707963f; }
                else if (g.sliceAxis == 0) { g.view.pitch = 0.0f; g.view.yaw = -1.5707963f; }
                else                       { g.view.pitch = 0.0f; g.view.yaw = 0.0f; }
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("turn the view to look straight through the slab: for a slab across y, from above --\nthe cross-section of every lock around the head, like a scan");
        }
    }
    if (changed) g.sectionsDirty = true;
    if (gridChanged) g.gridDirty = true;
    if (rootsChanged) updateRootsLabel(g);
}

static void drawEditSection(GroomState& g) {
    if (!ImGui::CollapsingHeader("Edit", ImGuiTreeNodeFlags_DefaultOpen)) return;
    if (ImGui::Button("new strand (N)")) newStrand(g);
    flowNext(btnW("undo (Ctrl+Z)"));
    if (ImGui::Button("undo (Ctrl+Z)")) undoLast(g);
    flowNext(btnW("save (Ctrl+S)"));
    if (ImGui::Button("save (Ctrl+S)")) groomSave(g);
    flowNext(btnW("reload"));
    if (ImGui::Button("reload")) { g.status = "reloading..."; groomLoadScene(g); if (g.ok) g.status = "reloaded"; }
    const float fieldW = ImGui::GetFontSize() * 6.0f;
    // WHERE NEW POINTS GO (0.373.0). A strand is a chain of points from its root to its tip: the
    // root on a roots section (the scalp), every later point off it -- in the air, or inside a part.
    ImGui::TextUnformatted("place new points (after the root):");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("A strand is a chain of points from ROOT to TIP. Its first click (after N) plants the root on a\n"
                          "roots section (the scalp; tick 'roots' in Sections for others). Every later click adds the next\n"
                          "point -- after the selected one -- placed as chosen here. Dragging a point moves it the same way.");
    ImGui::Indent();
    ImGui::RadioButton("in the air", &g.placeMode, 1);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Where you click, at the same depth (distance from you) as the point it follows: draw the\n"
                          "strand's shape as seen from here, then orbit and drag points to move them in depth.\n"
                          "A dragged point moves in the screen plane.");
    flowNext(chkW("on surfaces"));
    ImGui::RadioButton("on surfaces", &g.placeMode, 0);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("On the roots sections under the click (any visible mesh with 'any mesh' ticked):\n"
                          "for strands that lie ON a surface. A dragged point slides on it.");
    flowNext(chkW("inside a section"));
    // a part one places INSIDE has to be seen (and clicked) through: made so when it is chosen
    auto seeThrough = [&](int i) {
        if (i < 0 || i >= (int)g.sections.size()) return;
        Section& s = g.sections[(size_t)i];
        if (!s.visible || s.opacity >= 0.999f) { s.visible = true; s.opacity = 0.35f; g.sectionsDirty = true; }
    };
    {
        const bool haveInside = g.insideSec >= 0 && g.insideSec < (int)g.sections.size();
        if (ImGui::RadioButton("inside a section", g.placeMode == 2)) {
            if (!haveInside)                      // choose one: the first see-through part, else the first part
                for (size_t i = 0; i < g.sections.size(); ++i)
                    if (g.sections[i].reference || g.sections[i].opacity < 0.999f) { g.insideSec = (int)i; break; }
            if (g.insideSec < 0 && !g.sections.empty()) g.insideSec = 0;
            if (g.insideSec >= 0) { g.placeMode = 2; seeThrough(g.insideSec); }
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Inside the part chosen below -- e.g. a sculpted hairdo the render skips: 'depth' of the way\n"
                              "from where the pixel's ray enters it (0) to the next surface behind (1: its far side, or the\n"
                              "scalp under it). A dragged point keeps its depth. The part is made see-through when chosen.");
    }
    if (g.placeMode == 2) {
        const bool on = g.insideSec >= 0 && g.insideSec < (int)g.sections.size();
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::BeginCombo("##inside", on ? g.sections[(size_t)g.insideSec].label.c_str() : "(choose a section)")) {
            for (size_t i = 0; i < g.sections.size(); ++i) {
                ImGui::PushID((int)i);
                if (ImGui::Selectable(g.sections[i].label.c_str(), g.insideSec == (int)i)) { g.insideSec = (int)i; seeThrough((int)i); }
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("the part new points are placed inside");
        ImGui::SetNextItemWidth(fieldW);
        ImGui::SliderFloat("depth", &g.insideDepth, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("0 = where the ray enters the part, 1 = at the next surface behind it");
    }
    ImGui::Unindent();
    ImGui::Checkbox("sketch", &g.sketchMode);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Sketch strands: press on a roots section and drag -- a new strand is drawn along the drag,\n"
                          "a point every 'spacing' pixels, placed as chosen above (so 'inside a section' sketches a\n"
                          "strand through the inside of the part). Right-drag orbits while sketching.");
    if (g.sketchMode) {
        flowNext(fieldW + ImGui::GetStyle().ItemInnerSpacing.x + ImGui::CalcTextSize("spacing (px)").x);
        ImGui::SetNextItemWidth(fieldW);
        ImGui::SliderFloat("spacing (px)", &g.sketchStep, 8.0f, 120.0f, "%.0f");
    }
    flowNext(chkW("click plots"));
    ImGui::Checkbox("click plots", &g.addOnClick);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("a click in the view adds a point to the selected strand (untick to only select and drag)");
    flowNext(chkW("any mesh"));
    ImGui::Checkbox("any mesh", &g.pickAny);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("roots, and 'on surfaces' points, may land on any visible mesh, not only the roots sections");
    flowNext(ImGui::CalcTextSize("(Alt: never grab a point)").x);
    ImGui::TextDisabled("(Alt: never grab a point)");
    // SELECTING SEVERAL, AND DELETING (0.373.0)
    if (!g.selPts.empty()) {
        char b1[64]; std::snprintf(b1, sizeof b1, "delete the %zu selected points (Del)", g.selPts.size());
        if (ImGui::Button(b1)) deleteSelected(g);
        flowNext(btnW("select their strands"));
        if (ImGui::Button("select their strands")) selectStrandsOfPoints(g);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("select the whole strands these points belong to (to delete or group them)");
        flowNext(btnW("clear (Esc)"));
        if (ImGui::Button("clear (Esc)")) clearSelection(g);
    } else if (!g.multi.empty()) {
        char b1[64]; std::snprintf(b1, sizeof b1, "delete the %zu selected curves (Del)", g.multi.size());
        if (ImGui::Button(b1)) deleteSelected(g);
        flowNext(btnW("group them (G)"));
        if (ImGui::Button("group them (G)")) groupMulti(g);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("group the selected curves into a curve of curves");
        flowNext(btnW("clear (Esc)"));
        if (ImGui::Button("clear (Esc)")) clearSelection(g);
    } else {
        if (ImGui::Button("select all strands (Ctrl+A)")) selectAllStrands(g);
        disabledWrapped("or Ctrl-click curve names in Curves; Shift-drag in the view selects points");
    }
    groom::Node* n = (g.selNode >= 0) ? groom::findNode(g.model, g.selNode) : nullptr;
    if (!n) {
        disabledWrapped("nothing selected: press N for a new strand, or click a curve in the tree or a point in the view");
    } else {
        groom::Where w = groom::whereIs(g.model, n->id);
        const int level = groom::levelOf(g.model, *n);
        const float* col = levelColour(level);
        ImGui::TextColored(ImVec4(col[0], col[1], col[2], 1.0f), "%s  [L%d]  %s", n->name.empty() ? "(unnamed)" : n->name.c_str(), level,
                           n->kids.empty() ? "strand" : "curve of curves");
        if (w.file) ImGui::TextDisabled("in %s%s%s", w.file->path.c_str(), w.container ? "  group " : "", w.container ? w.container->name.c_str() : "");
        if (w.file && !w.file->writable) ImGui::TextColored(ImVec4(1, 0.6f, 0.4f, 1), "not saveable: the file %s", w.file->why.c_str());
        if (n->kids.empty()) {
            if (g.selPt >= 0 && g.selPt < (int)n->pts.size()) {
                groom::Pt& p = n->pts[(size_t)g.selPt];
                ImGui::Text("point %d of %zu (authored coordinates)", g.selPt, n->pts.size());
                double v[3] = { p.p.x, p.p.y, p.p.z };
                bool ch = false;
                const float coordW = fieldW + ImGui::GetStyle().ItemInnerSpacing.x + ImGui::CalcTextSize("x").x;
                ImGui::SetNextItemWidth(fieldW); ch |= ImGui::InputDouble("x", &v[0], 0, 0, "%.6g"); if (ImGui::IsItemActivated()) pushUndo(g); flowNext(coordW);
                ImGui::SetNextItemWidth(fieldW); ch |= ImGui::InputDouble("y", &v[1], 0, 0, "%.6g"); if (ImGui::IsItemActivated()) pushUndo(g); flowNext(coordW);
                ImGui::SetNextItemWidth(fieldW); ch |= ImGui::InputDouble("z", &v[2], 0, 0, "%.6g"); if (ImGui::IsItemActivated()) pushUndo(g);
                if (ch) { p.p = Vec3{ v[0], v[1], v[2] }; p.edited = true; markEdited(g, n->id); }
                bool hr = p.haveR;
                if (ImGui::Checkbox("own radius (r=)", &hr)) { pushUndo(g); p.haveR = hr; if (hr && p.r <= 0.0) p.r = 0.001; p.edited = true; markEdited(g, n->id); }
                if (p.haveR) {
                    ImGui::SameLine(); ImGui::SetNextItemWidth(fieldW);
                    double r = p.r;
                    if (ImGui::InputDouble("r", &r, 0, 0, "%.6g")) { p.r = r; p.edited = true; markEdited(g, n->id); }
                    if (ImGui::IsItemActivated()) pushUndo(g);
                }
                if (ImGui::Button("delete point (Del)")) deletePoint(g);
                flowNext(btnW("delete strand"));
            } else {
                ImGui::TextWrapped("%zu point(s) -- %s; click one in the pane to edit it", n->pts.size(),
                                   n->pts.empty() ? "click a roots section in the pane to plant its root" : "a click in the pane adds the next point at the end");
            }
            if (ImGui::Button("delete strand")) deleteStrand(g);
        } else {
            drawNodeParams(g, *n);
            if (ImGui::Button("delete curve")) deleteStrand(g);
        }
    }
    if (!g.preview.err.empty()) ImGui::TextColored(ImVec4(1, 0.6f, 0.4f, 1), "the loader would refuse this: %s", g.preview.err.c_str());
    if (!g.status.empty()) ImGui::TextWrapped("%s", g.status.c_str());
}

static void drawGroomPanel(GroomState& g) {
    if (ImGui::Button("Help (F1)")) g.showHelp = true;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("how everything here works: strands, placing points, sections, grids, selecting, saving");
    ImGui::SameLine();
    ImGui::TextWrapped("scene: %s", std::filesystem::path(g.scenePath).filename().string().c_str());
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", g.scenePath.c_str());
    if (!g.ok) { ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "load failed: %s", g.err.c_str()); return; }
    ImGui::Separator();
    bool d = false;
    d |= ImGui::Checkbox("mesh", &g.showMesh);        flowNext(chkW("wireframe"));
    d |= ImGui::Checkbox("wireframe", &g.view.wire);  flowNext(chkW("shade"));
    d |= ImGui::Checkbox("shade", &g.view.shade);
    d |= ImGui::Checkbox("curves", &g.showCurves);    flowNext(chkW("points"));
    d |= ImGui::Checkbox("points", &g.showPoints);    flowNext(chkW("hair"));
    d |= ImGui::Checkbox("hair", &g.showHair);        flowNext(chkW("roots"));
    d |= ImGui::Checkbox("roots", &g.showRoots);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("the grown fur's roots, as small crosses ('hair' draws the fur itself)");
    ImGui::TextUnformatted("frame:"); ImGui::SameLine();
    bool f = false;
    f |= ImGui::RadioButton("groom", &g.frameMode, 0); ImGui::SameLine();
    f |= ImGui::RadioButton("all", &g.frameMode, 1);   ImGui::SameLine();
    if (ImGui::SmallButton("reset view")) { g.view.yaw = 0.6f; g.view.pitch = 0.4f; g.view.zoom = 1.0f; }
    if (f) { groomComputeFrame(g); d = true; g.gridDirty = true; }   // cross sizes and grid spacing follow the frame
    // Edit first (0.373.0): it is used on every click; the parts and levels are set once
    drawEditSection(g);
    drawSectionsSection(g);
    if (ImGui::CollapsingHeader("Levels", ImGuiTreeNodeFlags_DefaultOpen)) {
        // A level is a node's HEIGHT in the tree (groom.h levelOf, computed bottom-up), not a
        // refinement pass you add. Saying so here because the colour-per-level display reads
        // as incremental and the semantics are the opposite: see drawNodeParams.
        ImGui::TextDisabled("a level is a node's height in the tree, not a refinement pass");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Level N means: the deepest child below this node is level N-1.\n"
                              "A node with no count/density is a pass-through group -- its children\n"
                              "still render as themselves. A node that PLACES consumes them: they\n"
                              "become the control cage its count samples along, so the curves you\n"
                              "drew below are keys, not hair. Choose the authoring depth before\n"
                              "drawing -- adding a placing level reinterprets everything under it.");
        for (int L = 0; L <= g.maxLevel && L < 8; ++L) {
            const float* col = levelColour(L);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(col[0], col[1], col[2], 1.0f));
            char lab[64]; std::snprintf(lab, sizeof lab, L == 0 ? "level 0: strands (control points)"
                                                               : "level %d: built FROM level %d", L, L - 1);
            bool on = g.levelOn[L];
            if (ImGui::Checkbox(lab, &on)) { g.levelOn[L] = on; d = true; }
            ImGui::PopStyleColor();
        }
    }
    if (d) g.lines.dirty = true;
    if (ImGui::CollapsingHeader("Curves", ImGuiTreeNodeFlags_DefaultOpen)) {
        // roots of the tree: curves nobody references by name
        std::map<std::string, int> referenced;
        std::vector<groom::Node*> all; groom::collectNodes(g.model, all);
        for (const groom::Node* n : all) if (n->ref) referenced[n->name] = 1;
        int shown = 0;
        forEachTopCurve(g.model, [&](groom::Node& n) {
            if (!n.name.empty() && referenced.count(n.name)) return;
            treeNode(g, n, 0); ++shown;
        });
        if (!shown) ImGui::TextDisabled("(no curves in this scene -- press N to start a strand)");
    }
    drawFurSection(g);
    if (ImGui::CollapsingHeader("Files")) {
        for (const groom::FileModel& fm : g.model.files)
            ImGui::BulletText("%s%s%s%s", fm.path.c_str(), fm.dirty ? "  *modified*" : "", fm.writable ? "" : "  (not writable: ", fm.writable ? "" : (fm.why + ")").c_str());
    }
}

static void groomHotkeys(GroomState& g) {
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::IsKeyPressed(ImGuiKey_F1)) g.showHelp = !g.showHelp;
    if (io.WantTextInput) return;
    if (ImGui::IsKeyPressed(ImGuiKey_Delete)) deleteSelected(g);
    if (ImGui::IsKeyPressed(ImGuiKey_N) && !io.KeyCtrl) newStrand(g);
    if (ImGui::IsKeyPressed(ImGuiKey_G) && !io.KeyCtrl) groupMulti(g);
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Z)) undoLast(g);
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S)) groomSave(g);
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_A)) selectAllStrands(g);
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        // one thing at a time: the box being dragged, the selection, then the point, then the strand
        if (g.boxing) g.boxing = false;
        else if (!g.selPts.empty() || !g.multi.empty()) clearSelection(g);
        else if (g.selPt >= 0) select(g, g.selNode, -1);
        else select(g, -1, -1);
    }
}

// ---- the Help window (0.373.0): F1, or the Help button ---------------------------------------------
static void helpPara(const char* text) { ImGui::TextWrapped("%s", text); ImGui::Spacing(); }
static void helpBullet(const char* text) { ImGui::Bullet(); ImGui::SameLine(); ImGui::TextWrapped("%s", text); }
static void drawGroomHelp(GroomState& g) {
    const float fs = std::max(ImGui::GetIO().FontGlobalScale, 1.0f);
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowBgAlpha(1.0f);                   // opaque: it is read, not glanced through
    ImGui::SetNextWindowSize(ImVec2(std::min(820.0f * fs, vp->WorkSize.x * 0.9f), std::min(700.0f * fs, vp->WorkSize.y * 0.9f)), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f, vp->WorkPos.y + vp->WorkSize.y * 0.5f), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    if (!ImGui::Begin("ftrace -groom: help (F1 opens and closes it)", &g.showHelp, ImGuiWindowFlags_NoCollapse)) { ImGui::End(); return; }
    const ImGuiTreeNodeFlags open = ImGuiTreeNodeFlags_DefaultOpen;
    if (ImGui::CollapsingHeader("What this tool does", open)) {
        helpPara("ftrace -groom edits the hair of an FTSL scene. The rendered hair is grown by the scene's `fur` blocks from "
                 "GUIDE CURVES: each guide is a STRAND -- a chain of points from its ROOT on the scalp to its TIP -- and the fur "
                 "fills in thousands of hairs between the guides. This tool shows the scene's meshes and its guides; you add, "
                 "move and delete guide points, and Ctrl+S saves them back into the scene's curve files. 'reload' (Edit panel) "
                 "regrows the fur from what was saved.");
        helpPara("Nothing here is specific to one model: the parts listed, which one is the scalp and which one is shown "
                 "see-through all come from the scene file (see Sections), and every one of them can be changed.");
    }
    if (ImGui::CollapsingHeader("The view", open)) {
        helpBullet("Drag empty space, or right-drag anywhere: orbit.   Mouse wheel: zoom.");
        helpBullet("The line above the view always says what a click will do next.");
        helpBullet("The line below it: the roots sections, where new points go, what is selected, and whether the fur is stale.");
        helpBullet("frame: 'groom' centres on the scalp and the guides, 'all' on everything; 'reset view' straightens it.");
        helpBullet("Guides are coloured by level: strands red, curves of strands green, then blue, yellow ...");
        helpBullet("Selected point: white dot. Point under the mouse: yellow ring. Box-selected points: cyan dots.");
        helpBullet("Top checkboxes: mesh, wireframe, shading, curves, their points, the grown hair ('hair') and its roots.");
        ImGui::Spacing();
    }
    if (ImGui::CollapsingHeader("Making a strand (a guide)", open)) {
        helpPara("1. Press N (or 'new strand'). A new, empty strand is selected.");
        helpPara("2. Click on a ROOTS section -- normally the scalp -- to plant its root, the point the hair grows from. "
                 "Only roots sections take a root (see Sections; 'any mesh' lifts that). A see-through part in front, "
                 "such as a sculpted hairdo, is clicked through.");
        helpPara("3. Click again for each next point, from root to tip. Where each goes is set by 'place new points' (Edit):");
        ImGui::Indent();
        helpPara("in the air -- where you click, at the same depth (distance from you) as the point it follows. Draw the "
                 "strand's shape as you see it, then orbit to look from the side and drag points to set their depth.");
        helpPara("on surfaces -- on the roots sections under the click (any visible mesh with 'any mesh'): for hair lying on a surface.");
        helpPara("inside a section -- inside the part chosen beside it (for instance a sculpted hairdo the render skips), "
                 "'depth' of the way from where the click's ray enters the part (0) to the next surface behind (1). "
                 "The part is made see-through when chosen.");
        ImGui::Unindent();
        helpPara("A new point goes AFTER the selected point: select a point in the middle and click to insert one there. "
                 "The renderer needs at least 2 points in a strand.");
        helpPara("SKETCH instead: tick 'sketch', press on the scalp and drag. A new strand follows the drag, a point every "
                 "'spacing' pixels, each placed as 'place new points' says -- so with 'inside a section' the strand runs "
                 "through the inside of the part. Right-drag still orbits.");
    }
    if (ImGui::CollapsingHeader("Moving points", open)) {
        helpBullet("Click a point to select it and its strand; drag it to move it.");
        helpBullet("A root slides on its roots section. A later point moves the way new points are placed:");
        ImGui::Indent(); ImGui::TextWrapped("in the screen plane (in the air), through the section's inside keeping its depth (inside), or sliding on surfaces."); ImGui::Unindent();
        helpBullet("Shift-drag a point: along the surface normal (lift it off or push it in). Ctrl-drag: in the screen plane.");
        helpBullet("Hold Alt to click through the points (to add a point where one is drawn).");
        helpBullet("Edit shows the selected point's coordinates to type in, and 'own radius' for its thickness.");
        ImGui::Spacing();
    }
    if (ImGui::CollapsingHeader("Selecting several, and deleting", open)) {
        helpBullet("Shift-drag on empty space: select the points in the box (cyan). Ctrl+Shift-drag adds to them.");
        ImGui::Indent(); ImGui::TextWrapped("Only points you can see are taken; a point behind a solid surface is not."); ImGui::Unindent();
        helpBullet("In the Curves tree, Ctrl-click curve NAMES to select several curves (highlighted, marked *).");
        ImGui::Indent(); ImGui::TextWrapped("The checkbox beside a name only shows or hides that curve in the view -- it is not a selection."); ImGui::Unindent();
        helpBullet("'select all strands' (Ctrl+A): every strand that is shown.");
        helpBullet("Del deletes the selected points if there are any; otherwise the selected curves; otherwise the one selected point.");
        ImGui::Indent(); ImGui::TextWrapped("A strand left with fewer than 2 points is removed with them. Deleting a guide also takes it out of "
                                            "any curve of curves that lists it."); ImGui::Unindent();
        helpBullet("'select their strands' turns selected points into their whole strands (to delete or group whole guides).");
        helpBullet("Esc clears the selection. Ctrl+Z undoes, up to 100 steps.");
        ImGui::Spacing();
    }
    if (ImGui::CollapsingHeader("Sections: parts, see-through, scalp", open)) {
        helpPara("The Sections panel lists the parts of the scene's meshes: each glTF material within a mesh (\"alice2 / root.4\"), "
                 "or a whole mesh when its format has no parts (an OBJ). Each row: show | opacity | grid | roots | name.");
        helpBullet("opacity below 1 makes a part see-through; clicks and the mouse pass through it to what is behind.");
        helpBullet("grid draws the part as an outline (below).");
        helpBullet("roots: a new strand's root is planted on the parts ticked here.");
        helpPara("Where the defaults come from -- the scene file, nothing built in:");
        ImGui::Indent();
        helpPara("A part the scene SKIPS (`skip_material \"...\"` in its mesh block) is left out of the render, but the tool "
                 "loads it for reference: it is listed in gold as '[skipped by the scene]', starts faint (its folds shown by "
                 "bright edges), "
                 "and new points are placed inside it. That is how a sculpted hairdo, replaced in the render by real hair, "
                 "becomes the shape to groom inside.");
        helpPara("'roots' starts ticked on the meshes the scene's fur blocks grow on (`fur { on \"scalp\" ... }`); with no fur "
                 "block, on the first mesh.");
        ImGui::Unindent();
    }
    if (ImGui::CollapsingHeader("Seeing the shape of a see-through part", open)) {
        helpPara("A folded, see-through surface such as a sculpted hairdo is hard to read: every layer shows at once. "
                 "Four tools, under the Sections list, each on its own or together:");
        helpBullet("bright edges (on by default): the part turns opaque and bright where its surface is seen EDGE-ON and "
                   "stays clear where it faces you, so every fold, curl and lock edge becomes an outline while what is "
                   "inside stays visible. With the part's opacity at 0 you see the edges alone.");
        helpBullet("hue by depth: see-through parts and grid lines are coloured by their distance from you -- bright yellow "
                   "near, then green, teal and blue, to a dim purple far. Stacked layers separate by colour, and a line that "
                   "dives behind a lock changes colour on the way. The colours follow the view as you orbit.");
        helpBullet("grid: tick it on a part to draw its contour lines -- where evenly spaced planes (x, y, z = constant) cut "
                   "its surface. 'lines' sets how many fit across the framed extent, x / y / z choose the plane families. "
                   "'hue per line' gives each line its own colour, six in turn (none red: red is the guides' colour), so "
                   "neighbours never match and one line can be followed through a tangle -- clearest with one direction of "
                   "lines only; with 'hue by depth' too, each line dims with distance. 'far side too' also draws the lines "
                   "behind the part's near surface.");
        helpBullet("slice: shows only a SLAB of the see-through and gridded parts, with their cross-section at its centre in "
                   "white -- in a horizontal slab through a hairdo, the outline of every lock at that height. Choose what it "
                   "cuts across (x, y, z, or your view: a slab of depth), where it is ('at', or Ctrl+wheel in the view) and "
                   "how thick. 'guides too' cuts the guides as well (the selected strand is always drawn whole); 'solid "
                   "parts too' cuts the head and body as well, a slice of the whole scene like a scan. 'look along it' turns "
                   "the view to look straight through the slab -- for a slab across y, from above. What a slab cuts away "
                   "cannot be clicked or hovered, and a point placed 'inside a section' goes inside the slab: the part you "
                   "can see.");
        ImGui::Spacing();
        helpPara("A way to work: slice across y, tick 'solid parts too', press 'look along it', and Ctrl+wheel down through "
                 "the hair -- the head is an outline and every lock a white outline around it, like a scan. Follow a lock "
                 "down, placing a strand's points inside it at each height (the view can be turned back to the side at "
                 "any time; the slab stays where it is).");
    }
    if (ImGui::CollapsingHeader("Curves tree and levels")) {
        helpPara("Every curve in the scene's files. A strand (level 0) is a chain of points. A curve of curves (level 1 and up) "
                 "either GROUPS its children (they render as themselves) or, with count or density, PLACES copies along the "
                 "path through its children's roots -- its children then become the control cage, not hair. G groups the "
                 "selected curves into a new curve of curves. The Levels panel shows or hides each level. Click a name to "
                 "select that curve; the arrow opens it.");
    }
    if (ImGui::CollapsingHeader("Fur, and rendering")) {
        helpPara("The scene's fur blocks, statement by statement: edit a value, remove it (x), or add one. A bald zone is a "
                 "sphere where no hair grows: 'pick centre', then click the surface. 'save + render' saves and starts a real "
                 "ftrace render from this view, in its own window.");
    }
    if (ImGui::CollapsingHeader("Saving", open)) {
        helpPara("Ctrl+S (or 'save') writes the guides back to the files they came from. A scene with no curve file gets one, "
                 "<scene>_groom.ftsl, included from the scene. The fur shown is from the last load: after an edit it is marked "
                 "STALE -- save, then 'reload', to regrow it.");
    }
    if (ImGui::CollapsingHeader("Keys and mouse", open)) {
        static const char* keys[][2] = {
            { "N", "new strand" },
            { "click", "the selected strand's next point (its root, first)" },
            { "drag a point", "move it  (Shift: along the normal, Ctrl: in the screen plane)" },
            { "Shift-drag", "box-select points  (Ctrl+Shift: add to them)" },
            { "drag empty / right-drag", "orbit;  wheel: zoom" },
            { "Ctrl+wheel", "move the slice (with 'slice' ticked)" },
            { "Alt", "click through the points" },
            { "Del", "delete the selected points, else the selected curves, else the selected point" },
            { "Esc", "clear the selection, then the point, then the strand" },
            { "Ctrl+A", "select all strands" },
            { "G", "group the selected curves into a curve of curves" },
            { "Ctrl+Z / Ctrl+S", "undo / save" },
            { "F1", "this help" },
        };
        if (ImGui::BeginTable("keys", 2, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)) {
            for (const auto& k : keys) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(k[0]);
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted(k[1]);
            }
            ImGui::EndTable();
        }
    }
    ImGui::End();
}

}  // namespace groom

int groomSectionsReport(const std::string& scenePath) {
    groom::GroomState g;
    g.scenePath = scenePath;
    if (!groom::groomLoadScene(g)) return 1;        // prints the sections with their bounds
    groom::updateTriPass(g);
    for (const groom::Section& S : g.sections)
        std::printf("[groom-sections] %-48s %8zu tris  opacity %.2f%s%s%s\n", S.label.c_str(), S.triCount(), S.opacity,
                    S.reference ? "  reference (skipped by the scene)" : "", S.grid ? "  [grid]" : "", S.roots ? "  [roots]" : "");
    const int defMode = g.placeMode, defInside = g.insideSec;
    static const char* modeNames[] = { "on surfaces", "in the air", "inside a section" };
    std::printf("[groom-sections] new points go %s%s%s; roots on: %s\n", modeNames[std::clamp(defMode, 0, 2)],
                defMode == 2 ? " -- " : "", defMode == 2 ? g.sections[(size_t)defInside].label.c_str() : "", g.target.c_str());
    // the grid outlines (0.373.0) -- of the gridded sections, and of each skipped part, which since
    // 0.374.0 starts without one (bright edges instead) but is what a grid is for
    {
        std::vector<char> hadGrid;
        for (groom::Section& S : g.sections) { hadGrid.push_back(S.grid ? 1 : 0); if (S.reference) S.grid = true; }
        std::vector<groom::LineBatch> gb;
        double ms = 0.0;
        { MsTimer t(&ms); groom::buildGridLines(g, gb); }
        for (size_t i = 0; i < g.sections.size(); ++i) g.sections[i].grid = hadGrid[i] != 0;
        for (size_t b = 0; b < gb.size() && b < g.gridSecOf.size(); ++b)
            std::printf("[groom-sections] grid outline of \"%s\": %zu segments, planes %.4g m apart (built in %.1f ms)\n",
                        g.sections[(size_t)g.gridSecOf[b]].label.c_str(), gb[b].v.size() / 2,
                        (double)g.frameExt / std::max(4.0, (double)g.gridLines), ms);
    }
    // For each part the scene skips, probe rays through its middle along the axes: the interval
    // "grow inside" finds, where depth 0 / 0.5 / 1 puts a point, and where the same pixel's click
    // would put a ROOT (on the roots mesh, through the see-through part) -- or what blocks it.
    for (size_t si = 0; si < g.sections.size(); ++si) {
        const groom::Section& S = g.sections[si];
        if (!S.reference || S.ref.empty()) continue;
        Vec3 lo{1e30, 1e30, 1e30}, hi{-1e30, -1e30, -1e30};
        for (const Tri& t : S.ref)
            for (const Vec3* v : { &t.v0, &t.v1, &t.v2 }) {
                lo.x = std::min(lo.x, v->x); lo.y = std::min(lo.y, v->y); lo.z = std::min(lo.z, v->z);
                hi.x = std::max(hi.x, v->x); hi.y = std::max(hi.y, v->y); hi.z = std::max(hi.z, v->z);
            }
        const Vec3 c = (lo + hi) * 0.5;
        const double R = std::sqrt(dot(hi - lo, hi - lo));
        g.insideSec = (int)si;
        static const Vec3 dirs[] = { {0, 0, -1}, {0, 0, 1}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0} };
        static const char* names[] = { "from the front (+z)", "from behind (-z)", "from her left (+x)", "from her right (-x)", "from above (+y)" };
        for (int k = 0; k < 5; ++k) {
            const Vec3 d = dirs[k];
            const Vec3 o = c - d * (2.0 * R);
            double t0, t1;
            if (!groom::insideInterval(g, o, d, (int)si, t0, t1)) {
                std::printf("[groom-sections]   probe %s: no interval (the ray misses \"%s\", or something solid is in front of it)\n",
                            names[k], S.label.c_str());
                continue;
            }
            const Vec3 p0 = o + d * t0, pm = o + d * (t0 + 0.5 * (t1 - t0)), p1 = o + d * t1;
            const groom::Pick root = groom::pickSurfaceRay(g, o, d, true);
            std::printf("[groom-sections]   probe %s: enters at (%.4f %.4f %.4f), next surface at (%.4f %.4f %.4f), "
                        "%.4f m inside; depth 0.5 -> (%.4f %.4f %.4f); a root click there %s",
                        names[k], p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, t1 - t0, pm.x, pm.y, pm.z,
                        root.hit ? "lands on the roots mesh" : "finds no roots mesh (something solid is in front, or the ray misses it)");
            if (root.hit) std::printf(" at (%.4f %.4f %.4f)", root.p.x, root.p.y, root.p.z);
            std::printf("\n");
        }
    }
    // A strand placed the way the pane's clicks place one (0.373.0): a camera looking straight
    // down on the groom (a 1000-px pane across the frame), the root clicked at the pixel nearest
    // the centre whose click reaches a roots section, then the next point 20 px to the side in
    // each of the three modes -- through placeNextPoint, the code a click runs.
    {
        groom::PaneCam& cam = g.cam;
        const float R[3][3] = { { 1, 0, 0 }, { 0, 0, -1 }, { 0, 1, 0 } };      // right, up, toward (+y: from above)
        for (int r = 0; r < 3; ++r) for (int k = 0; k < 3; ++k) cam.R[r][k] = R[r][k];
        for (int k = 0; k < 3; ++k) cam.mid[k] = g.frameMid[k];
        cam.origin = ImVec2(0.0f, 0.0f); cam.avail = ImVec2(1000.0f, 1000.0f);
        cam.ax = cam.ay = 2.0f / std::max(g.frameExt, 1e-3f); cam.s = 1000.0f / std::max(g.frameExt, 1e-3f);
        cam.diag = g.frameDiag; cam.valid = true;
        groom::Node n; n.id = -12345;                                     // a scratch strand, not in the model
        Vec3 p; std::string why = "no pixel of the middle half reaches a roots section";
        ImVec2 rootPx(-1.0f, -1.0f);
        float bestD2 = 1e30f;
        for (int iy = 0; iy <= 20; ++iy)
            for (int ix = 0; ix <= 20; ++ix) {
                const ImVec2 px(250.0f + 25.0f * ix, 250.0f + 25.0f * iy);
                const float d2 = (px.x - 500.0f) * (px.x - 500.0f) + (px.y - 500.0f) * (px.y - 500.0f);
                if (d2 >= bestD2) continue;
                Vec3 q; std::string w;
                if (groom::placeNextPoint(g, px, &n, q, w)) { bestD2 = d2; rootPx = px; p = q; }
                else if (rootPx.x < 0.0f && px.x == 500.0f && px.y == 500.0f) why = w;
            }
        if (rootPx.x < 0.0f) {
            std::printf("[groom-sections] placement from above: no root click lands (%s)\n", why.c_str());
        } else {
            groom::Pt rp; rp.p = p; n.pts.push_back(rp);
            std::printf("[groom-sections] placement from above: root clicked at pixel (%.0f, %.0f) of 1000x1000 -> (%.4f %.4f %.4f)\n",
                        rootPx.x, rootPx.y, p.x, p.y, p.z);
            g.insideSec = defInside;
            for (int mode = 0; mode < 3; ++mode) {
                if (mode == 2 && (defInside < 0 || defInside >= (int)g.sections.size())) continue;
                g.placeMode = mode;
                Vec3 q; std::string w2;
                const bool ok = groom::placeNextPoint(g, ImVec2(rootPx.x + 20.0f, rootPx.y), &n, q, w2);
                if (ok) std::printf("[groom-sections]   next point %-16s -> (%.4f %.4f %.4f)%s\n", modeNames[mode], q.x, q.y, q.z,
                                    mode == 1 ? (std::fabs(q.y - p.y) < 1e-9 ? "  (the root's depth: ok)" : "  (NOT the root's depth)") : "");
                else std::printf("[groom-sections]   next point %-16s -> refused: %s\n", modeNames[mode], w2.c_str());
            }
        }
        // A SLAB (0.374.0): one across y through the middle of the part points go inside; every
        // "inside" click from above that lands must land INSIDE the slab (the part the eye sees)
        if (defInside >= 0 && defInside < (int)g.sections.size()) {
            g.placeMode = 2; g.insideSec = defInside;
            g.sliceOn = true; g.sliceAxis = 1; g.slicePos = 0.5f; g.sliceThick = 0.06f;
            groom::updateSlab(g);
            groom::Node m; m.id = -12346;
            groom::Pt any; m.pts.push_back(any);                         // a strand with a root: clicks go inside
            int landed = 0, within = 0;
            for (int iy = 0; iy <= 20; ++iy)
                for (int ix = 0; ix <= 20; ++ix) {
                    Vec3 q; std::string w;
                    if (!groom::placeNextPoint(g, ImVec2(50.0f * ix, 50.0f * iy), &m, q, w)) continue;
                    ++landed;
                    if (std::fabs(dot(g.slabN, q) - g.slabD) <= g.slabHalf * (1.0 + 1e-9)) ++within;
                }
            std::printf("[groom-sections] with a slab across y at 0.5 (y %.4f +- %.4f): %d of 441 clicks from above land inside \"%s\", "
                        "%d of them inside the slab%s\n", g.slabD, g.slabHalf, landed, g.sections[(size_t)defInside].label.c_str(), within,
                        (landed > 0 && within == landed) ? " (ok)" : (landed ? " (NOT ALL)" : ""));
            std::vector<groom::LineBatch> sb;
            groom::buildSliceLines(g, sb);
            std::printf("[groom-sections] its cross-section at y %.4f: %zu segments\n", g.slabD, sb.empty() ? (size_t)0 : sb[0].v.size() / 2);
            g.sliceOn = false; g.slabHalf = 0.0;
        }
        g.placeMode = defMode; g.insideSec = defInside;
    }
    return 0;
}

// A TEST HOOK (0.374.0), not a user feature: FTRACE_GROOM_VIEW="key=value;..." presets view
// options at startup, so the pane can be captured in each mode (tools/gui_peek.ps1) without
// clicking -- synthetic mouse input cannot reach a window that does not have the focus.
static void groomPresetView(groom::GroomState& g) {
    const char* env = std::getenv("FTRACE_GROOM_VIEW");
    if (!env || !*env) return;
    std::string s = env;
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t end = std::min(s.find(';', pos), s.size());
        const std::string item = s.substr(pos, end - pos);
        pos = end + 1;
        const size_t eq = item.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = item.substr(0, eq);
        const double v = std::atof(item.c_str() + eq + 1);
        if      (k == "rims")        g.rims = v != 0.0;
        else if (k == "hueDepth")    g.hueDepth = v != 0.0;
        else if (k == "hueLine")     g.gridHueLine = v != 0.0;
        else if (k == "farSide")     g.gridXray = v != 0.0;
        else if (k == "slice")       { g.sliceOn = v != 0.0; g.sectionsDirty = true; }
        else if (k == "sliceAxis")   g.sliceAxis = std::clamp((int)v, 0, 3);
        else if (k == "slicePos")    g.slicePos = (float)v;
        else if (k == "sliceThick")  g.sliceThick = (float)v;
        else if (k == "sliceGuides") g.sliceGuides = v != 0.0;
        else if (k == "sliceSolids") { g.sliceSolids = v != 0.0; g.sectionsDirty = true; }
        else if (k == "curves")      g.showCurves = v != 0.0;
        else if (k == "yaw")         g.view.yaw = (float)v;
        else if (k == "pitch")       g.view.pitch = (float)v;
        else if (k == "zoom")        g.view.zoom = (float)v;
        else if (k == "grid" || k == "opacity") {           // for every reference (skipped) part
            for (groom::Section& sec : g.sections)
                if (sec.reference) { if (k == "grid") sec.grid = v != 0.0; else sec.opacity = (float)v; }
            g.gridDirty = true; g.sectionsDirty = true;
        }
    }
    g.lines.dirty = true;
    std::fprintf(stderr, "[groom] FTRACE_GROOM_VIEW: %s\n", env);
}

int runGroomGui(const std::string& scenePath, bool minimized) {
    groom::GroomState g;
    g.scenePath = scenePath;
    groom::groomLoadScene(g);
    groomPresetView(g);
    ImGui_ImplWin32_EnableDpiAwareness();
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0, 0, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, L"FtraceGroom", nullptr };
    RegisterClassExW(&wc);
    std::wstring title = utf8ToWide("ftrace \xF0\x9F\xAA\x9F groom");
    HWND hwnd = CreateWindowW(wc.lpszClassName, title.c_str(), WS_OVERLAPPEDWINDOW, 80, 80, 1400, 860, nullptr, nullptr, wc.hInstance, nullptr);
    if (!CreateDeviceD3D(hwnd)) {
        CleanupDeviceD3D(); UnregisterClassW(wc.lpszClassName, wc.hInstance);
        std::fprintf(stderr, "error: -groom: failed to create D3D11 device.\n");
        return 1;
    }
    ShowWindow(hwnd, minimized ? SW_SHOWMINNOACTIVE : SW_SHOWDEFAULT); UpdateWindow(hwnd);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();
    { float dpi = ImGui_ImplWin32_GetDpiScaleForHwnd(hwnd); if (dpi > 1.0f) { ImGui::GetStyle().ScaleAllSizes(dpi); ImGui::GetIO().FontGlobalScale = dpi; } }
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);
    bool done = false;
    while (!done) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg); DispatchMessage(&msg);
            if (msg.message == WM_QUIT) done = true;
        }
        if (done || ft::stopRequested()) break;
        // minimized: nothing to draw, and Present would not wait for a vsync -- idle instead of
        // spinning a core and the GPU on a window nobody can see
        if (IsIconic(hwnd)) { Sleep(50); continue; }

        ImGui_ImplDX11_NewFrame(); ImGui_ImplWin32_NewFrame(); ImGui::NewFrame();
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos); ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::Begin("groom", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus);
        float leftW = ImGui::GetContentRegionAvail().x * 0.36f;
        ImGui::BeginChild("groom_left", ImVec2(leftW, 0), true);
        groom::drawGroomPanel(g);
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("groom_right", ImVec2(0, 0), true);
        if (g.ok) groom::drawGroomPane(g, g_pd3dDevice, g_pd3dDeviceContext);
        ImGui::EndChild();
        if (g.ok) groom::groomHotkeys(g);
        ImGui::End();
        if (g.showHelp) groom::drawGroomHelp(g);
        ImGui::Render();
        const float clear[4] = { 0.06f, 0.06f, 0.08f, 1.0f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRTV, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRTV, clear);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_pSwapChain->Present(1, 0);
    }
    g.lines.release();
    g.gridGpu.release();
    g.sliceGpu.release();
    g.view.gpu.release();
    ImGui_ImplDX11_Shutdown(); ImGui_ImplWin32_Shutdown(); ImGui::DestroyContext();
    CleanupDeviceD3D(); DestroyWindow(hwnd); UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return 0;
}

int runViewerGui(const std::string& sidecarPath, const std::string& loomScene,
                 bool startPlaying, bool startPrebake, int prebakeCapMB) {
    Sidecar sc;
    if (!sc.load(sidecarPath)) {
        std::fprintf(stderr, "error: -viewer: %s\n", sc.err.c_str());
        return 1;
    }
    std::vector<CurveGeom> curves = collectCurves(sc);
    std::vector<StripSeries> strips = buildStrips(curves);
    DagGraph dag = collectDag(sc);
    std::vector<FieldGeom> fields = collectFields(sc);
    std::vector<MeshGeom> meshes = collectMeshes(sc);

    OrbitView view;
    for (const auto& c : curves) view.maxDim = std::max(view.maxDim, c.dim);
    FieldView fview;
    for (const auto& f : fields) fview.maxDim = std::max(fview.maxDim, f.dim);
    MeshView mview;

    // F7 primary path: parse loom's emitted `.ftsl` (Sidecar::source) with ftrace's
    // own loader so the Render tab can raymarch the real field in-process. Failure is
    // non-fatal — the viewer still shows the static sidecar geometry.
    ftsl::Loaded loaded;
    bool sceneOk = false;
    std::string sceneErr;
    const std::string sourcePath = sc.source();
    if (!sourcePath.empty()) {
        if (ftsl::load(sourcePath, loaded, sceneErr)) sceneOk = true;
        else std::fprintf(stderr, "[viewer] could not load scene '%s': %s\n",
                          sourcePath.c_str(), sceneErr.c_str());
    }
#ifdef HAVE_CUDA
    RenderPane rpane;
    const bool liveWantSource = true;
#else
    // No raymarch pane to feed, so don't make loom emit an .ftsl nobody reads.
    const bool liveWantSource = false;
#endif

    // --- F4 item 2: the live re-introspection channel -------------------------
    // Which loom file to talk to: an explicit `-loom` wins, else the sidecar's own
    // `build` provenance key (loom records the file its `build()` came from). Neither
    // present — or python/loom not importable — is NOT an error: the viewer stays
    // frozen on the static sidecar and the Live panel says exactly why.
    LoomBridge bridge;
    LivePanel  live;
    {
        if (const minijson::Value* fr = sc.root.find("frame"); fr && fr->isObject()) {
            live.frame  = fr->intAt("frame", 0);
            live.frames = std::max(1, fr->intAt("frames", 1));
        }
        const std::string scenePy = loomScene.empty() ? sc.buildFile() : loomScene;
        if (!scenePy.empty()) {
            if (bridge.start(scenePy, live.startErr)) {
                live.up = true;
                liveSeedParams(live, bridge);
            } else {
                std::fprintf(stderr, "[viewer] live channel unavailable: %s\n",
                             live.startErr.c_str());
            }
        }
    }

    // --- window ---
    // MUST precede window creation. Without it Windows DPI-*virtualizes* the process on
    // a scaled display: the swapchain is created at the logical client size and the
    // compositor upscales it, so every glyph and every 1-px mesh wireframe comes out
    // blurry. Opting in makes the backbuffer native-resolution; the style/font scale
    // below then restores the intended physical size (see just after ImGui::Begin-time
    // setup) so the UI is crisp rather than merely small.
    ImGui_ImplWin32_EnableDpiAwareness();
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0, 0,
                       GetModuleHandle(nullptr), nullptr, nullptr, nullptr,
                       nullptr, L"FtraceViewer", nullptr };
    RegisterClassExW(&wc);
    std::wstring title = utf8ToWide("ftrace \xF0\x9F\xAA\x9F loom viewer");  // 🪟
    HWND hwnd = CreateWindowW(wc.lpszClassName, title.c_str(),
                              WS_OVERLAPPEDWINDOW, 80, 80, 1280, 800,
                              nullptr, nullptr, wc.hInstance, nullptr);
    if (!CreateDeviceD3D(hwnd)) {
        CleanupDeviceD3D();
        UnregisterClassW(wc.lpszClassName, wc.hInstance);
        std::fprintf(stderr, "error: -viewer: failed to create D3D11 device.\n");
        return 1;
    }
    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();                // F3 strip charts
    ImNodes::CreateContext();               // F5 modulator-DAG panel
    ImGui::GetIO().IniFilename = nullptr;   // don't litter an imgui.ini in the CWD
    ImGui::StyleColorsDark();
    // Now that the backbuffer is native-resolution (EnableDpiAwareness above), scale
    // the whole UI by the monitor's content scale so 13 px of font stays the same
    // PHYSICAL size it was before — crisp instead of upscaled, not microscopic.
    {
        float dpi = ImGui_ImplWin32_GetDpiScaleForHwnd(hwnd);
        if (dpi > 1.0f) {
            ImGui::GetStyle().ScaleAllSizes(dpi);
            ImGui::GetStyle().FontScaleDpi = dpi;
        }
    }
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    // F4 skins — needs the device, so it happens after CreateDeviceD3D. Relative
    // image paths in the sidecar fall back to the sidecar's own directory.
    // baseDir is hoisted out because a live re-derivation rebuilds the skins against
    // the SAME directory — a re-derived sidecar has no file of its own at all (it comes
    // down the pipe), and its image paths were always relative to the original scene.
    std::string baseDir;
    {
        size_t cut = sidecarPath.find_last_of("/\\");
        if (cut != std::string::npos) baseDir = sidecarPath.substr(0, cut + 1);
    }
    SkinLib skins;
    skins.build(sc, baseDir, g_pd3dDevice, g_pd3dDeviceContext);

    // Fold a freshly re-derived sidecar into the panes (F4 item 2). What loom re-derived
    // is the GEOMETRY; the VIEW is ours, so orbit / zoom / dim selection / tab choice are
    // deliberately preserved — a parameter sweep that snapped the camera back to its
    // default on every bake would be unusable.
    auto adoptSidecar = [&](minijson::Value&& tree) -> bool {
        Sidecar ns;
        {
            MsTimer _t(&live.msAdoptJson);
            if (!ns.adopt(std::move(tree))) { live.lastErr = "sidecar: " + ns.err; return false; }
        }
        std::vector<int> oldIds;
        for (const auto& n : dag.nodes) oldIds.push_back(n.id);

        sc = std::move(ns);
        {
            MsTimer _t(&live.msAdoptGeom);
            curves = collectCurves(sc);
            strips = buildStrips(curves);
            fields = collectFields(sc);
            meshes = collectMeshes(sc);
            ++mview.geomGen;   // a NEW tessellation -> the mesh pane must re-upload its buffers
            for (const auto& c : curves) view.maxDim  = std::max(view.maxDim,  c.dim);
            for (const auto& f : fields) fview.maxDim = std::max(fview.maxDim, f.dim);
        }

        MsTimer _tdag(&live.msAdoptDag);
        DagGraph nd = collectDag(sc);
        std::vector<int> newIds;
        for (const auto& n : nd.nodes) newIds.push_back(n.id);
        // Same node set = the same graph with new values, so keep the layout the user
        // panned/zoomed to. A different node set is a different graph: re-lay it out,
        // carrying over only the docked-vs-maximized choice (which is about the window,
        // not the graph).
        if (newIds == oldIds) {
            nd.pos            = dag.pos;
            nd.realSize       = dag.realSize;
            nd.sizesValid     = dag.sizesValid;
            nd.extent         = dag.extent;
            nd.measuredFont   = dag.measuredFont;
            nd.measuredAvailH = dag.measuredAvailH;
            nd.zoom           = dag.zoom;
            nd.fitted         = dag.fitted;
            nd.fitCanvasW     = dag.fitCanvasW;
            nd.laidOut        = dag.laidOut;
        }
        nd.maximized = dag.maximized;
        dag = std::move(nd);
        _tdag.stop();

        {
            MsTimer _t(&live.msAdoptSkins);
            skins.release();
            skins.build(sc, baseDir, g_pd3dDevice, g_pd3dDeviceContext);
        }
        return true;
    };

    // --- F8(b) prebaked play ------------------------------------------------
    // See the PlayCache commentary above for why the cache holds ADOPTED state and
    // why frames move by swapping. `swapWith` is the one primitive: it exchanges the
    // live pane state with a slot, in O(1), and the invariant it maintains is that the
    // live locals hold frame `cache.liveIdx` while slot `liveIdx` sits empty.
    PlayCache cache;
    auto swapWith = [&](PlayFrame& pf) {
        std::swap(sc,       pf.sc);
        std::swap(curves,   pf.curves);
        std::swap(strips,   pf.strips);
        std::swap(fields,   pf.fields);
        std::swap(meshes,   pf.meshes);
        std::swap(dag,      pf.dag);
        std::swap(loaded,   pf.loaded);
        std::swap(sceneOk,  pf.sceneOk);
        std::swap(sceneErr, pf.sceneErr);
    };
    // Park whatever the live locals are showing back into its own slot, leaving the
    // live state holding the (empty) contents of that slot. Called before adopting a
    // fresh bake and before swapping a different cached frame in, so a frame is never
    // in two places and never in none.
    auto parkLive = [&]() {
        if (cache.liveIdx < 0 || cache.liveIdx >= (int)cache.f.size()) { cache.liveIdx = -1; return; }
        PlayFrame& slot = cache.f[cache.liveIdx];
        swapWith(slot);
        slot.have = true;
        cache.liveIdx = -1;
    };
    // Show a cached frame. Cheap enough to call from the scrub path as well as from
    // play — which is the second thing the cache buys: once prebaked, dragging the
    // frame slider is instant instead of one bake per stop.
    auto showCached = [&](int k) {
        if (!cache.holds(k)) return false;
        MsTimer _t(&live.msCache);
        // This frame pays no bake, no sidecar adoption and no .ftsl round trip, so the
        // last-measured values for all three are now describing work that did not
        // happen. Clear them: a breakdown line whose terms sum to twice the frame time
        // beside them is worse than no breakdown, because it reads as a real profile.
        live.lastMs = live.msSidecar = live.msFtsl = 0.0;
        live.msAdoptJson = live.msAdoptGeom = live.msAdoptDag = live.msAdoptSkins = 0.0;
        live.msFtslParse = live.msFtslBuild = live.msFtslAssets = live.msFtslAccel = 0.0;
        parkLive();
        PlayFrame& slot = cache.f[k];
        swapWith(slot);
        // The slot is now "empty" in the bookkeeping sense but still owns whatever the
        // locals held a moment ago -- deliberately. Freeing it here would cost a Scene
        // teardown on every step of playback; leaving it means the vacated slot doubles
        // as the buffer `parkLive` swaps back into, so a whole loop of playback does no
        // allocation at all. Costs one extra frame's worth of memory, once.
        slot.have = false;
        cache.liveIdx = k;
        // The panes below are not told "the frame changed" by the swap itself: the mesh
        // pane keys its GPU buffers off `geomGen`, the raymarch pane off its own dirty
        // flag, and the skins are built from `sc`. All three have to be re-pointed here
        // exactly as a fresh adoption would.
        ++mview.geomGen;
        skins.release();
        skins.build(sc, baseDir, g_pd3dDevice, g_pd3dDeviceContext);
#ifdef HAVE_CUDA
        if (sceneOk) rpane.initFrom(loaded.scene);
#endif
        return true;
    };

    // `-play`: open with the transport already running. Only meaningful once the
    // clock has somewhere to go and there is a live channel to re-derive through --
    // a frozen sidecar has no frames to bake, so silently "playing" it would be a lie.
    if (startPlaying) {
        if (live.up && live.frames > 1) { live.playing = true; live.primePlay = true; }
        else std::fprintf(stderr, "[play] ignoring -play: %s\n",
                          !live.up ? "no live loom channel (-loom, or a sidecar `build` key)"
                                   : "the sidecar advertises frames = 1 (saved without a clock)");
    }

    // `-prebake`: start the §F8(b) walk on open. Exactly what the panel's button does,
    // hoisted to the command line so a cached play can be MEASURED from a script --
    // the whole point of the cache is a frame rate, and a frame rate you can only get
    // to by clicking into a window is a frame rate nobody records. Ordered after
    // `-play` on purpose: the prebake owns the bridge while it runs and the transport
    // simply waits, so `-prebake -play` starts playing the instant the walk finishes.
    if (prebakeCapMB > 0) cache.capMB = std::max(1, prebakeCapMB);
    if (startPrebake) {
        if (live.up && live.frames > 1) {
            cache.key      = playCacheKey(live);
            cache.f.assign((size_t)live.frames, PlayFrame{});
            cache.baking   = true;
            cache.bakeNext = 0;
            std::printf("[prebake] walking %d frames (cap %d MB)\n", live.frames, cache.capMB);
        } else {
            std::fprintf(stderr, "[prebake] ignoring -prebake: %s\n",
                         !live.up ? "no live loom channel (-loom, or a sidecar `build` key)"
                                  : "the sidecar advertises frames = 1 (saved without a clock)");
        }
    }

    bool done = false;
    bool firstFrame = true;   // one-shot: default-select the primary geometry tab
    while (!done) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) done = true;
        }
        // `ftrace -stop <pid>` must reach EVERY ftrace process, not just renders — it is
        // the one sanctioned alternative to `taskkill /F`, and a viewer holding the exe
        // open is exactly what makes a rebuild fail and tempts the force-kill. A render
        // polls the flag at its chunk boundary; the GUI has no chunk, so poll once per
        // frame (~free) and shut down through the ordinary exit path, releasing D3D11,
        // the loom child process and the window the same way closing it by hand does.
        if (!done && ft::stopRequested()) {
            std::printf("[stop] external stop requested — closing the loom viewer.\n");
            std::fflush(stdout);
            done = true;
        }
        if (done) break;

        // Set when paced play steps the clock below; OR'd into `livePost` so the next
        // frame's bake is requested through the single post site like any other change.
        bool livePlayPost = false;
        // Fold in whatever loom finished since the last frame. The UI never waits on a
        // bake — it keeps drawing the geometry it already has and adopts the new one on
        // whatever frame it lands.
        if (live.up) {
            LoomResult r;
            if (bridge.take(r)) {
                live.appliedSeq = r.seq;
                ++live.baked;
                live.lastMs  = r.ms;
                live.lastErr = r.ok ? std::string() : r.err;
                live.msCache = 0.0;   // a real bake: the cache term is the one now stale
                // Before the live locals are overwritten by this bake, put the frame
                // they are currently holding back in its slot. Without this, adopting
                // over a cached frame would destroy it in place and the cache would
                // quietly develop holes exactly where playback had already been.
                if (cache.liveIdx >= 0) parkLive();
                if (r.ok && r.payload) {
                    if (r.payload->hasSidecar) {
                        MsTimer _t(&live.msSidecar);
                        adoptSidecar(std::move(r.payload->sidecar));
                    }
                    if (r.payload->hasSource) {
                        ftsl::Loaded nl;
                        std::string  nerr;
                        ftsl::LoadTiming lt;
                        MsTimer _t(&live.msFtsl);
                        // The overlay is why nothing here opens a file: every `mesh`
                        // the emitted source names came down the pipe with it.
                        if (ftsl::loadSource(r.payload->source, "<loom live>", nl, nerr,
                                             {}, &lt, &r.payload->assets)) {
                            live.msFtslParse  = lt.msParse;
                            live.msFtslBuild  = lt.msBuild;
                            live.msFtslAssets = lt.msAssets;
                            live.msFtslAccel  = lt.msAccel;
                            loaded  = std::move(nl);
                            sceneOk = true;
                            sceneErr.clear();
#ifdef HAVE_CUDA
                            // Re-frame on the new bounds but keep yaw/pitch/dist: the
                            // user's orbit survives the sweep (initFrom touches only
                            // center/radius, and marks the pane dirty).
                            rpane.initFrom(loaded.scene);
#endif
                        } else {
                            // A re-derived scene ftrace cannot load is a real error and
                            // must be said out loud, not silently left on stale geometry.
                            sceneOk      = false;
                            sceneErr     = nerr;
                            live.lastErr = "ftsl: " + nerr;
                        }
                    }
                }
                r.payload.reset();   // last reference: the ~1 MB frame goes here

                // F8(b): the live locals now hold frame `r.frame`, freshly adopted.
                // Claim the slot for it (the state is not COPIED there -- it stays
                // live, and `parkLive` will move it home when something else needs the
                // locals). Three things have to hold before a claim is legitimate:
                //
                //   ok    -- caching a FAILED frame would replay the failure forever
                //            with no way to notice it had ever been transient.
                //   key   -- a result that was in flight when a control moved belongs
                //            to the OLD scene. Without this test it would be filed
                //            into the new cache under the frame number it happens to
                //            share, and play back as a frame from a scene the user has
                //            already left. This is the reason LoomJob/LoomResult carry
                //            a key at all.
                //   cap   -- an on-demand bake must not push the cache past the budget
                //            the user set. The prebake walk is exempt because it stops
                //            ITSELF at the cap one frame later, and refusing the frame
                //            it is standing on would spin it forever.
                const size_t capBytes = (size_t)std::max(1, cache.capMB) * 1048576ull;
                if (r.ok && !cache.f.empty() && r.key == cache.key
                    && r.frame >= 0 && r.frame < (int)cache.f.size()
                    && !cache.f[r.frame].have
                    && (cache.baking || cache.bytes < capBytes)) {
                    // Size the frame just baked. It is in the LIVE locals, not in its
                    // slot, so measure it there -- via a park/unpark round trip, which
                    // is two O(1) swaps and keeps `playFrameBytes` a pure function of a
                    // PlayFrame rather than a second copy of the same field list. Done
                    // for on-demand bakes too and not just for the prebake walk, so the
                    // byte count can never drift away from what is actually held.
                    const int k = r.frame;
                    cache.liveIdx = k;
                    ++cache.bakeHave;
                    parkLive();
                    cache.f[k].bytes = playFrameBytes(cache.f[k]);
                    cache.bytes += cache.f[k].bytes;
                    // Unpark by hand rather than through showCached: this frame is
                    // already ON SCREEN and everything derived from it (the skin
                    // atlas, the CUDA pane's scene upload) is still valid, so going
                    // the long way round would rebuild all of it for nothing -- once
                    // per frame of the walk, which is exactly where a prebake can
                    // least afford it.
                    swapWith(cache.f[k]);
                    cache.f[k].have = false;
                    cache.liveIdx   = k;
                }

                if (cache.baking) {
                    // Project the whole clock's cost as soon as the per-frame cost is
                    // known, and say so UP FRONT. Otherwise a short cap announces itself
                    // only once it has already been hit, which reads as a status line
                    // rather than as a thing to act on -- and the consequence is a silent
                    // performance cliff mid-loop, where the cached prefix plays at the
                    // target fps and the rest drops to whatever a live bake costs. Four
                    // frames is enough for a stable average (frames of one clock differ
                    // in tessellation, not in kind) while still landing early in a walk.
                    if (!cache.projected && cache.bakeHave >= 4) {
                        cache.projected = true;
                        const double perFrame = cache.bytes / 1048576.0 / cache.bakeHave;
                        const double total    = perFrame * live.frames;
                        if (total > cache.capMB) {
                            // Round the suggestion up with headroom: landing exactly on
                            // the cap is the one case that still ends capped.
                            const int want = (int)(total * 1.1) + 1;
                            std::printf("[prebake] ~%.1f MB/frame x %d frames = ~%.0f MB, "
                                        "over the %d MB cap: only ~%d frames will cache and "
                                        "play will stutter past there. Use -prebake-cap %d "
                                        "for the whole clock.\n",
                                        perFrame, live.frames, total, cache.capMB,
                                        (int)(cache.capMB / perFrame), want);
                        } else {
                            std::printf("[prebake] ~%.1f MB/frame x %d frames = ~%.0f MB, "
                                        "fits the %d MB cap\n",
                                        perFrame, live.frames, total, cache.capMB);
                        }
                        std::fflush(stdout);
                    }
                    // Stop at the cap rather than at the end of the clock if the cap
                    // comes first: a prefix cache still plays from memory as far as it
                    // goes, which beats refusing to cache a long clock at all.
                    if (cache.bytes >= capBytes) {
                        cache.capped = true;
                        cache.baking = false;
                        const double perFrame = cache.bakeHave
                            ? cache.bytes / 1048576.0 / cache.bakeHave : 0.0;
                        const int want = perFrame > 0.0
                            ? (int)(perFrame * live.frames * 1.1) + 1 : 0;
                        std::printf("[prebake] cap %d MB reached at frame %d/%d "
                                    "(%.0f MB); the rest will bake on demand"
                                    " -- re-run with -prebake-cap %d to cache all %d\n",
                                    cache.capMB, cache.bakeHave, live.frames,
                                    cache.bytes / 1048576.0, want, live.frames);
                    } else if (cache.bakeNext >= live.frames) {
                        cache.baking = false;
                        std::printf("[prebake] %d frames cached, %.0f MB (%.1f MB/frame)\n",
                                    cache.bakeHave, cache.bytes / 1048576.0,
                                    cache.bakeHave ? cache.bytes / 1048576.0 / cache.bakeHave : 0.0);
                    }
                    // A prebake owns the bridge until it finishes: the clock does not
                    // move and no play post is made. Advancing the display clock here
                    // as well would race the walk and leave the cache half-filled.
                } else if (live.playing) {
                    // F8(a): a bake landed, so the clock may take its next step. Doing
                    // it HERE -- rather than on a timer -- is what makes UNCACHED play
                    // show every frame instead of only the ones that won the
                    // latest-wins slot. A failed bake still advances: stalling on a bad
                    // frame would look like a hang, and the error is already on screen.
                    liveAdvanceClock(live);
                    livePlayPost = true;
                }
            }
        }

        // --- F8(b) prebake driver: one outstanding job at a time ---------------
        // Serial on purpose. The bridge is latest-wins on a ONE-slot pending job, so
        // posting the whole range up front would bake the last frame and discard the
        // other N-1 -- the same trap paced play was built to avoid. Walking it one
        // landing at a time costs nothing extra (loom is the bottleneck either way)
        // and makes the progress bar mean what it says.
        if (cache.baking && live.up && bridge.linkUp() && !bridge.busy()
            && cache.bakeNext < live.frames) {
            bridge.post(liveJobAt(live, cache.bakeNext, /*wantSidecar=*/true, liveWantSource));
            ++live.posted;
            ++cache.bakeNext;
        } else if (cache.baking && (!live.up || !bridge.linkUp())) {
            cache.baking = false;      // the link died mid-walk; keep the prefix
            cache.capped = true;
        }

        // --- F8(b) cached playback: paced by a wall clock, not by loom ----------
        // This is the whole payoff. When the frame the clock wants is already in the
        // cache there is no bake to wait for, so the clock is free to advance on real
        // time -- `targetFps`, or as fast as the draw allows at 0. The fps readout is
        // still MEASURED (liveAdvanceClock stamps it), so what it reports is the rate
        // actually achieved rather than the rate requested.
        if (live.playing && !cache.baking && cache.covers(live.frame)) {
            bool step = true;
            if (cache.targetFps > 0.0f) {
                LARGE_INTEGER f, now;
                QueryPerformanceFrequency(&f);
                QueryPerformanceCounter(&now);
                const long long per = f.QuadPart > 0
                                    ? (long long)(double(f.QuadPart) / (double)cache.targetFps)
                                    : 0;
                if (!cache.lastStepQpc || !per) {
                    cache.lastStepQpc = now.QuadPart;     // first step starts the clock
                } else {
                    step = (now.QuadPart - cache.lastStepQpc) >= per;
                    if (step) {
                        // Advance the DEADLINE by exactly one period instead of
                        // resetting it to now. Resetting discards the overshoot, and
                        // since this test is only reached once per UI frame that
                        // quantises the achievable rate to the display's own refresh:
                        // on a 60 Hz vsync, asking for 24 waits three vblanks every
                        // time and delivers a rock-steady 20 (measured, and initially
                        // mistaken for the cache being the bottleneck). Carrying the
                        // remainder makes the wait alternate 2 and 3 vblanks and
                        // average out at the 24 that was asked for.
                        cache.lastStepQpc += per;
                        // ...but never bank more than one period of debt. A hitch, a
                        // drag, or the end of a long prebake would otherwise be repaid
                        // as a burst of frames at draw rate -- a visible lurch, and the
                        // opposite of the steady playback this whole feature is for.
                        if (now.QuadPart - cache.lastStepQpc > per)
                            cache.lastStepQpc = now.QuadPart;
                    }
                }
            }
            if (step) liveAdvanceClock(live);
        }
        // Whatever moved the clock -- play, the slider, an arrow key -- if the cache
        // has that frame, show it from memory and skip the bake entirely. Checked
        // every UI frame rather than only on a change, because a prebake landing can
        // make a frame available that the clock is already sitting on.
        if (!cache.baking && cache.holds(live.frame)) showCached(live.frame);

        // Playing off the END of a prefix cache -- the cap stopped the walk short, or
        // the user hit play on a scene that was never prebaked at all. Nothing above
        // moved the clock (the wall-clock stepper only runs on frames the cache
        // covers) and F8(a)'s advance-on-landing cannot fire either, because no bake
        // is in flight to land. Without this the clock would simply stop and play
        // would look like a hang. So fall back to bake-paced play for the frames the
        // cache does not hold, and let the two schemes meet in the middle: the cached
        // prefix runs at `targetFps`, the tail runs at whatever loom can do.
        //
        // Guarded on the bridge being IDLE so a slow bake is waited for rather than
        // re-posted every UI frame -- with latest-wins, a repost of the same frame
        // would queue one redundant bake behind the one already running.
        if (live.playing && !cache.baking && !cache.covers(live.frame)
            && live.up && bridge.linkUp() && !bridge.busy())
            livePlayPost = true;

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        // one full-window layout
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::Begin("loom viewer", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::Text("sidecar: %s", sidecarPath.c_str());
        ImGui::Separator();

        // Something the geometry depends on moved this frame. Seeded from the paced-play
        // step because `drawLivePanel` is inside a CollapsingHeader: seeding it there
        // instead would make collapsing the panel silently stop playback.
        bool livePost = livePlayPost;

        float leftW = ImGui::GetContentRegionAvail().x * 0.42f;
        ImGui::BeginChild("left", ImVec2(leftW, 0), true);
        if (ImGui::CollapsingHeader("Live (loom)",
                                    live.up ? ImGuiTreeNodeFlags_DefaultOpen : 0))
            livePost = drawLivePanel(live, bridge, cache) || livePost;
        if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
            drawScenePanel(sc);
        if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen))
            drawObjectsPanel(sc);
        if (ImGui::CollapsingHeader("Datasets", ImGuiTreeNodeFlags_DefaultOpen))
            drawDatasetsPanel(sc);
        if (ImGui::CollapsingHeader("Modulator DAG", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button(dag.maximized ? "dock" : "maximize")) {
                dag.maximized = !dag.maximized;
                // Going full-window is the "let me see all of it" gesture, so fit there;
                // coming back to the narrow column, readable 100% beats a thumbnail.
                if (dag.maximized) dag.fitFrames = 16;
                else { dag.zoom = 1.0f; dag.fitFrames = 0; dag.fitted = false; dag.pos.clear(); }
            }
            ImGui::SameLine();
            if (ImGui::Button("fit")) dag.fitFrames = 16;       // zoom until it all shows
            ImGui::SameLine();
            if (ImGui::Button("100%")) {
                dag.zoom = 1.0f; dag.fitFrames = 0; dag.fitted = false; dag.pos.clear();
                ImNodes::EditorContextResetPanning(ImVec2(0.0f, 0.0f));
            }
            ImGui::SameLine();
            if (ImGui::Button("re-layout")) {
                dag.pos.clear();                   // re-measure at the current pane size
                ImNodes::EditorContextResetPanning(ImVec2(0.0f, 0.0f));
            }
            if (dag.maximized) {
                ImGui::TextDisabled("(shown full-window - Esc to dock)");
            } else {
                // imnodes wants its own non-scrolling area (it pans on drag itself).
                // The pane takes what is left of the side column (so nothing is cut off
                // the bottom and the column doesn't have to scroll) but no more than the
                // graph needs — the layout wraps itself to whatever height it gets.
                const float colAvail  = ImGui::GetContentRegionAvail().y;
                const float dockAvail = std::max(200.0f, colAvail - dagChrome());
                dagEnsureMeasured(dag, dockAvail);
                const float h = std::min(dag.extent.y + dagChrome(), dockAvail + dagChrome());
                ImGui::BeginChild("dagpane", ImVec2(0, h), ImGuiChildFlags_Borders,
                                  ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
                drawDagPanel(dag, dockAvail);
                ImGui::EndChild();
            }
        }
        ImGui::EndChild();

        ImGui::SameLine();
        ImGui::BeginChild("right", ImVec2(0, 0), true);
        // Curves and Fields each get a tab. A tab is shown only when its kind is
        // present, so whichever exists is the default-selected one (no empty tabs).
        bool haveCurves = !curves.empty();
        bool curvesTab = haveCurves || (fields.empty() && meshes.empty());
#ifdef HAVE_CUDA
        const bool haveRender = sceneOk;
#else
        const bool haveRender = false;
#endif
        if (ImGui::BeginTabBar("rightTabs")) {
#ifdef HAVE_CUDA
            if (haveRender) {
                // F7 primary path: the in-process raymarch of the real field is the
                // point of the viewer, so it opens selected.
                ImGuiTabItemFlags rf = firstFrame ? ImGuiTabItemFlags_SetSelected : 0;
                if (ImGui::BeginTabItem("Render", nullptr, rf)) {
                    if (drawRenderPane(rpane, loaded.scene, sceneOk, sceneErr,
                                       g_pd3dDevice, g_pd3dDeviceContext,
                                       live.up ? &live : nullptr) && live.autoApply)
                        livePost = true;
                    ImGui::EndTabItem();
                }
            }
#endif
            if (curvesTab && ImGui::BeginTabItem("Curves")) {
                if (!strips.empty()) {
                    float paneH = ImGui::GetContentRegionAvail().y * 0.58f;
                    ImGui::BeginChild("curvepane", ImVec2(0, paneH), false);
                    drawCurvePane(curves, view);
                    ImGui::EndChild();
                    ImGui::BeginChild("stripcharts", ImVec2(0, 0), false);
                    drawStripCharts(strips, view);
                    ImGui::EndChild();
                } else {
                    drawCurvePane(curves, view);
                }
                ImGui::EndTabItem();
            }
            if (!fields.empty() && ImGui::BeginTabItem("Fields")) {
                drawFieldPane(fields, fview);
                ImGui::EndTabItem();
            }
            if (!meshes.empty()) {
                // a swept-mesh scene is "about" its surface, so open on Meshes even
                // though the internal spine curves also populate the Curves tab —
                // unless the live Render tab is present, which takes priority.
                ImGuiTabItemFlags mf = (firstFrame && !haveRender) ? ImGuiTabItemFlags_SetSelected : 0;
                if (ImGui::BeginTabItem("Meshes", nullptr, mf)) {
                    if (drawMeshPane(meshes, mview, skins, live.up ? &live : nullptr,
                                     g_pd3dDevice, g_pd3dDeviceContext)
                        && live.autoApply)
                        livePost = true;
                    ImGui::EndTabItem();
                }
            }
            ImGui::EndTabBar();
        }
        ImGui::EndChild();

        ImGui::End();

        // Maximized DAG: the side column can never be tall enough for a wide graph
        // (and imnodes has no zoom here), so give it the whole window on demand.
        if (dag.maximized) {
            ImGui::SetNextWindowPos(vp->WorkPos);
            ImGui::SetNextWindowSize(vp->WorkSize);
            bool open = true;
            ImGui::Begin("Modulator DAG", &open,
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings);
            if (ImGui::Button("dock")) dag.maximized = false;
            ImGui::SameLine();
            if (ImGui::Button("fit")) dag.fitFrames = 16;
            ImGui::SameLine();
            if (ImGui::Button("100%")) {
                dag.zoom = 1.0f; dag.fitFrames = 0; dag.fitted = false; dag.pos.clear();
                ImNodes::EditorContextResetPanning(ImVec2(0.0f, 0.0f));
            }
            ImGui::SameLine();
            if (ImGui::Button("re-layout")) {
                dag.pos.clear();                    // force a re-measure at the canvas size
                ImNodes::EditorContextResetPanning(ImVec2(0.0f, 0.0f));
            }
            ImGui::BeginChild("dagpanefull", ImVec2(0, 0), ImGuiChildFlags_Borders,
                              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            drawDagPanel(dag, -1.0f);               // wrap to the real canvas height
            ImGui::EndChild();
            ImGui::End();
            if (!open || ImGui::IsKeyPressed(ImGuiKey_Escape)) dag.maximized = false;
        }

        // F8(b): anything that is not the clock invalidates the cache, because the
        // cache is of THESE parameters at every frame. Compared as a fingerprint (see
        // playCacheKey) rather than by watching individual controls, so a control added
        // later cannot silently escape the check. The frames the cache holds are
        // released here and not lazily, so the memory goes back at the moment the user
        // changes something rather than at the next prebake.
        {
            const std::string key = playCacheKey(live);
            // Safe to drop outright: by the invariant the frame on screen lives in the
            // LOCALS, not in a slot, so the display survives the cache going away.
            if (!cache.f.empty() && key != cache.key) cache.drop();
        }

        // At most one post per frame, and posting OVERWRITES any job that has not
        // started: a fast sweep drag therefore costs one bake of wherever the user
        // ends up, not one bake per intermediate frame. That is the latest-wins rule.
        //
        // Two things suppress it. A prebake owns the bridge (its own serial driver
        // above posts instead), and a frame the cache can already show needs no bake at
        // all -- which is what makes a prebaked scrub instant rather than one round
        // trip per stop.
        if (livePost && live.up && bridge.linkUp()
            && !cache.baking && !cache.covers(live.frame)) {
            bridge.post(liveJob(live, /*wantSidecar=*/true, liveWantSource));
            ++live.posted;
        }

        // The Render tab has stopped drawing (collapsed, or another tab selected),
        // so its cost is no longer being paid -- stop reporting the stale figure.
        if (!live.renderTabDrew) {
            live.msRender = 0.0;
            live.msRenderUpload = live.msRenderKernel = live.msRenderRead = 0.0;
        }
        live.renderTabDrew = false;

        // Echo the same breakdown to stdout about once a second while playing. The
        // panel shows it live, but a printed trace is what you can actually diff
        // between builds, capture from a script, or read back after the fact.
        if (live.playing && live.playFps > 0.0) {
            static double lastLog = 0.0;
            const double now = ImGui::GetTime();
            if (now - lastLog > 1.0) {
                lastLog = now;
                const double period = 1000.0 / live.playFps;
                const double acc = live.lastMs + live.msSidecar + live.msFtsl
                                 + live.msCache + live.msRender;
                const double other = (period - acc > 0.0 ? period - acc : 0.0);
                // Same split as the panel: a prebaked frame pays `cache` INSTEAD of
                // bake/sidecar/ftsl, and printing the three it did not pay would make
                // the log unusable for exactly the comparison it exists to support --
                // uncached play against cached play.
                if (live.msCache > 0.0)
                    std::printf("[play] %5.1f fps  %6.1f ms = cache %.2f + raymarch %.0f "
                                "+ other %.0f   (prebaked)\n",
                                live.playFps, period, live.msCache, live.msRender, other);
                else
                    std::printf("[play] %5.1f fps  %6.1f ms = bake %.0f + sidecar %.0f + "
                                "ftsl %.0f + raymarch %.0f + other %.0f\n",
                                live.playFps, period, live.lastMs, live.msSidecar,
                                live.msFtsl, live.msRender, other);
                // The raymarch broken open, on its own line. A printed trace is what
                // gets diffed between builds and quoted afterwards, so it must carry
                // the same detail as the panel -- a bare `raymarch N` in the log is
                // what let an SM-contention artefact get read as a real ranking.
                if (live.msRender > 0.0) {
                    std::printf("[play]        raymarch %.0f = upload %.0f (scene) + "
                                "kernel %.0f (pixels) + readback %.0f\n",
                                live.msRender, live.msRenderUpload,
                                live.msRenderKernel, live.msRenderRead);
                }
                // ...and sidecar adoption, the biggest term of all, on the same terms.
                if (live.msSidecar > 0.0) {
                    std::printf("[play]        sidecar %.0f = json %.0f + geom %.0f + "
                                "dag %.0f + skins %.0f\n",
                                live.msSidecar, live.msAdoptJson, live.msAdoptGeom,
                                live.msAdoptDag, live.msAdoptSkins);
                }
                // ...and the .ftsl reload. `rest` is the Builder's own work with the two
                // nested phases (asset file loading, BVH build) taken back out.
                if (live.msFtsl > 0.0) {
                    std::printf("[play]        ftsl %.0f = parse %.0f + assets %.0f + "
                                "accel %.0f + rest %.0f\n",
                                live.msFtsl, live.msFtslParse, live.msFtslAssets,
                                live.msFtslAccel,
                                live.msFtslBuild - live.msFtslAssets - live.msFtslAccel);
                }
                std::fflush(stdout);
            }
        }

        ImGui::Render();
        const float clear[4] = { 0.10f, 0.10f, 0.12f, 1.0f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRTV, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRTV, clear);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_pSwapChain->Present(1, 0);  // vsync
        firstFrame = false;
    }

    // Ask loom to quit and join the worker BEFORE tearing D3D down — a result landing
    // mid-shutdown would otherwise rebuild skins against a released device.
    bridge.stop();
#ifdef HAVE_CUDA
    rpane.release();   // free the raymarch texture before the D3D device goes away
#endif
    skins.release();   // ditto for the F4 skin textures
    mview.gpu.release();   // and the mesh pane's shaders / buffers / offscreen target
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImNodes::DestroyContext();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    CleanupDeviceD3D();
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return 0;
}

#endif // _WIN32
