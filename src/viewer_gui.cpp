#include "viewer_gui.h"

#ifndef _WIN32
// -------- Non-Windows stub: the native viewer needs Win32 + D3D11 --------------
#include <cstdio>
int runViewerGui(const std::string&) {
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
#include <map>
#include <unordered_map>
#include <functional>
#include <thread>

// Bridge to ftrace's own scene loader + GPU field raymarcher (F7 primary path).
// The viewer IS the ftrace binary, so it can parse loom's emitted `.ftsl` with the
// exact loader main() uses and render the real isosurface field in-process via
// renderIsoPreviewCuda — the `-raster-gpu` preview kernel that sphere-traces the
// field's bytecode with NO tessellation (the static marching-cubes mesh in the
// sidecar is only a fallback). These headers are plain-C++ (main.cpp includes them
// under MSVC too); the raymarch itself is guarded by HAVE_CUDA below.
#include "ftsl.h"
#include "render_cuda.h"

#pragma comment(lib, "d3d11.lib")

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

std::wstring utf8ToWide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    if (n <= 0) return std::wstring(s.begin(), s.end());
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}

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
        if (!minijson::parse(text, root, err)) return false;
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
// F4 — SweptMesh tessellated geometry. Each swept_mesh object carries a `mesh`
// key (vertices / faces / uvs) baked by loom; the viewer draws it as a shaded,
// depth-sorted triangle surface in a 3-D orbit pane.
// --------------------------------------------------------------------------
struct MeshGeom {
    std::string        id, name, material;
    std::vector<float> verts;   // flat xyz (3 per vertex)
    int                nverts = 0;
    std::vector<int>   faces;   // flat index triples
    int                nfaces = 0;
    std::vector<float> uvs;     // flat uv (2 per vertex), may be empty
};

static std::vector<MeshGeom> collectMeshes(const Sidecar& sc) {
    std::vector<MeshGeom> meshes;
    const minijson::Value* objs = sc.arr("objects");
    if (!objs) return meshes;
    // objects may nest (Groups) — walk recursively
    std::function<void(const minijson::Value&)> visit = [&](const minijson::Value& o) {
        if (const minijson::Value* ch = o.find("children"); ch && ch->isArray())
            for (const auto& c : ch->arr) visit(c);
        const minijson::Value* m = o.find("mesh");
        if (!m || !m->isObject()) return;
        MeshGeom g;
        g.id       = scalarStr(o.find("id"), "");
        g.name     = scalarStr(o.find("name"), "");
        g.material = scalarStr(o.find("material"), "");
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
        meshes.push_back(std::move(g));
    };
    for (const auto& o : objs->arr) visit(o);
    return meshes;
}

} // namespace

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
// F4 — mesh pane: SweptMesh tessellated surfaces as a shaded, depth-sorted
// triangle mesh. Orbiting the 3 spatial dims is a view-only re-projection (no
// re-tessellation, exactly as the F4 rule specifies for isometries of the shown
// dims). Colour: flat lambert shading, per-object tint, or a UV checker.
// --------------------------------------------------------------------------
struct MeshView {
    float yaw = 0.6f, pitch = 0.4f, zoom = 1.0f;
    bool  shade = true;         // flat lambert lighting
    bool  wire = false;         // wireframe overlay
    int   colorBy = 0;          // 0 = grey, 1 = per-object tint, 2 = UV checker
};

static void drawMeshPane(const std::vector<MeshGeom>& meshes, MeshView& view) {
    ImGui::TextUnformatted("Meshes - drag to orbit, wheel to zoom (view-only re-projection)");
    ImGui::Checkbox("shade", &view.shade); ImGui::SameLine();
    ImGui::Checkbox("wireframe", &view.wire); ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    const char* cmodes[] = { "grey", "per-object tint", "UV checker" };
    ImGui::Combo("colour", &view.colorBy, cmodes, 3);

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.y < 80.0f) avail.y = 80.0f;
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("mesh_canvas", avail);
    bool hovered = ImGui::IsItemHovered();
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        ImVec2 d = ImGui::GetIO().MouseDelta;
        view.yaw   += d.x * 0.01f;
        view.pitch += d.y * 0.01f;
    }
    if (hovered) { float w = ImGui::GetIO().MouseWheel; if (w != 0.0f) view.zoom *= (1.0f + w * 0.1f); }
    if (view.zoom < 0.05f) view.zoom = 0.05f;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 br(origin.x + avail.x, origin.y + avail.y);
    dl->AddRectFilled(origin, br, IM_COL32(14, 16, 20, 255));
    dl->PushClipRect(origin, br, true);

    // shared rotation basis: rotate each world vertex to (X screen-right, Y up, Z toward viewer)
    float cy = std::cos(view.yaw),   sy = std::sin(view.yaw);
    float cx = std::cos(view.pitch), sx = std::sin(view.pitch);
    auto rot = [&](float x, float y, float z, float& X, float& Y, float& Z) {
        float x1 =  cy * x + sy * z;
        float z1 = -sy * x + cy * z;
        X = x1;
        Y = cx * y - sx * z1;
        Z = sx * y + cx * z1;   // depth toward viewer
    };

    // union bounds (centre + extent) over all meshes
    float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
    for (const auto& m : meshes)
        for (int i = 0; i < m.nverts; ++i)
            for (int k = 0; k < 3; ++k) {
                float v = m.verts[(size_t)i * 3 + k];
                lo[k] = std::min(lo[k], v); hi[k] = std::max(hi[k], v);
            }
    float ext = 1.0f;
    for (int k = 0; k < 3; ++k) if (hi[k] > lo[k]) ext = std::max(ext, hi[k] - lo[k]);
    float mid[3] = { 0, 0, 0 };
    for (int k = 0; k < 3; ++k) if (hi[k] >= lo[k]) mid[k] = 0.5f * (lo[k] + hi[k]);
    ImVec2 center(origin.x + avail.x * 0.5f, origin.y + avail.y * 0.5f);
    float scale = 0.42f * std::min(avail.x, avail.y) / (0.5f * ext + 1e-3f);

    const ImU32 tints[] = {
        IM_COL32(150, 190, 235, 255), IM_COL32(235, 175, 130, 255),
        IM_COL32(160, 225, 165, 255), IM_COL32(225, 155, 200, 255),
    };

    // rotate every vertex once, project to screen + keep depth
    struct SV { ImVec2 s; float d; float X, Y, Z; };
    // collect all triangles across meshes into one depth-sorted list (painter's algo)
    struct Tri { int mi; ImVec2 a, b, c; float depth; float shade; float ua, va, ub, vb, uc, vc; };
    std::vector<Tri> tris;
    std::vector<std::vector<SV>> proj(meshes.size());
    for (size_t mi = 0; mi < meshes.size(); ++mi) {
        const MeshGeom& m = meshes[mi];
        proj[mi].resize(m.nverts);
        for (int i = 0; i < m.nverts; ++i) {
            float X, Y, Z;
            rot(m.verts[(size_t)i*3+0] - mid[0], m.verts[(size_t)i*3+1] - mid[1],
                m.verts[(size_t)i*3+2] - mid[2], X, Y, Z);
            SV sv;
            sv.X = X; sv.Y = Y; sv.Z = Z; sv.d = Z;
            sv.s = ImVec2(center.x + X * scale * view.zoom, center.y - Y * scale * view.zoom);
            proj[mi][i] = sv;
        }
        for (int f = 0; f < m.nfaces; ++f) {
            int ia = m.faces[(size_t)f*3+0], ib = m.faces[(size_t)f*3+1], ic = m.faces[(size_t)f*3+2];
            if (ia < 0 || ib < 0 || ic < 0 || ia >= m.nverts || ib >= m.nverts || ic >= m.nverts) continue;
            const SV& A = proj[mi][ia]; const SV& B = proj[mi][ib]; const SV& C = proj[mi][ic];
            // face normal in rotated space -> Z component = facing the viewer
            float ux = B.X - A.X, uy = B.Y - A.Y, uz = B.Z - A.Z;
            float vx = C.X - A.X, vy = C.Y - A.Y, vz = C.Z - A.Z;
            float nx = uy*vz - uz*vy, ny = uz*vx - ux*vz, nz = ux*vy - uy*vx;
            float nl = std::sqrt(nx*nx + ny*ny + nz*nz) + 1e-9f;
            float facing = nz / nl;                       // -1..1, +1 = toward viewer
            Tri t;
            t.mi = (int)mi;
            t.a = A.s; t.b = B.s; t.c = C.s;
            t.depth = (A.d + B.d + C.d) / 3.0f;
            t.shade = 0.30f + 0.70f * std::fabs(facing);  // two-sided lambert
            if ((int)m.uvs.size() >= 2 * m.nverts) {
                t.ua = m.uvs[(size_t)ia*2]; t.va = m.uvs[(size_t)ia*2+1];
                t.ub = m.uvs[(size_t)ib*2]; t.vb = m.uvs[(size_t)ib*2+1];
                t.uc = m.uvs[(size_t)ic*2]; t.vc = m.uvs[(size_t)ic*2+1];
            } else { t.ua = t.va = t.ub = t.vb = t.uc = t.vc = 0.0f; }
            tris.push_back(t);
        }
    }
    // back-to-front so nearer triangles overdraw farther ones
    std::sort(tris.begin(), tris.end(), [](const Tri& p, const Tri& q){ return p.depth < q.depth; });

    for (const Tri& t : tris) {
        ImU32 base;
        if (view.colorBy == 1)        base = tints[t.mi % 4];
        else if (view.colorBy == 2) {  // UV checker at the triangle centroid
            float u = (t.ua + t.ub + t.uc) / 3.0f, v = (t.va + t.vb + t.vc) / 3.0f;
            int cu = (int)std::floor(u * 8.0f), cv = (int)std::floor(v * 8.0f);
            bool on = ((cu + cv) & 1) != 0;
            base = on ? IM_COL32(210, 210, 220, 255) : IM_COL32(90, 95, 110, 255);
        } else                        base = IM_COL32(180, 185, 195, 255);
        float s = view.shade ? t.shade : 1.0f;
        int r = (int)(((base >> IM_COL32_R_SHIFT) & 0xFF) * s);
        int g = (int)(((base >> IM_COL32_G_SHIFT) & 0xFF) * s);
        int b = (int)(((base >> IM_COL32_B_SHIFT) & 0xFF) * s);
        dl->AddTriangleFilled(t.a, t.b, t.c, IM_COL32(r, g, b, 255));
        if (view.wire)
            dl->AddTriangle(t.a, t.b, t.c, IM_COL32(30, 30, 36, 120), 1.0f);
    }
    dl->PopClipRect();

    int totalTris = 0, totalV = 0;
    for (const auto& m : meshes) { totalTris += m.nfaces; totalV += m.nverts; }
    ImGui::Text("%d mesh(es), %d verts, %d tris", (int)meshes.size(), totalV, totalTris);
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
// --------------------------------------------------------------------------
struct DagNode { int id = 0; std::string op, label; };
struct DagEdge { int src = 0, dst = 0; std::string param; };
struct DagGraph {
    std::vector<DagNode> nodes;
    std::vector<DagEdge> edges;
    bool laidOut = false;   // grid positions applied on the first frame only
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
            dn.id    = n.intAt("id", 0);
            dn.op    = n.find("op") ? n.find("op")->asString("?") : "?";
            dn.label = scalarStr(n.find("label"), "");
            g.nodes.push_back(std::move(dn));
        }
    if (edges && edges->isArray())
        for (const auto& e : edges->arr) {
            DagEdge de;
            de.src   = e.intAt("src", 0);
            de.dst   = e.intAt("dst", 0);
            de.param = scalarStr(e.find("param"), "in");
            g.edges.push_back(std::move(de));
        }
    return g;
}

// Longest-path layering (level = max over incoming edges of src level + 1) so the
// graph reads left→right from leaves (constants/oscillators) to the params they drive.
static void layoutDag(DagGraph& g) {
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
    std::map<int, int> rowInLevel;
    for (const auto& n : g.nodes) {
        int L = lvl(n.id);
        int row = rowInLevel[L]++;
        ImNodes::SetNodeGridSpacePos(n.id, ImVec2((float)L * 230.0f, (float)row * 95.0f));
    }
}

static void drawDagPanel(DagGraph& g) {
    if (g.nodes.empty()) { ImGui::TextDisabled("(no modulator DAG)"); return; }
    ImGui::TextDisabled("%d nodes, %d edges - drag to pan, scroll to zoom",
                        (int)g.nodes.size(), (int)g.edges.size());
    // per-node incoming edges (each becomes a labelled input pin)
    std::unordered_map<int, std::vector<int>> inEdges;   // node id -> [edge index...]
    for (int i = 0; i < (int)g.edges.size(); ++i) inEdges[g.edges[i].dst].push_back(i);

    ImNodes::BeginNodeEditor();
    if (!g.laidOut) layoutDag(g);   // must be inside Begin/EndNodeEditor
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
        // one labelled input pin per incoming edge (the param it feeds)
        auto it = inEdges.find(n.id);
        if (it != inEdges.end())
            for (int ei : it->second) {
                ImNodes::BeginInputAttribute(DAG_IN_BASE + ei);
                ImGui::TextUnformatted(g.edges[ei].param.c_str());
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
    std::string status;

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

    void render(const Scene& s, ID3D11Device* dev, ID3D11DeviceContext* ctx) {
        int W = resLong, H = resLong;
        Camera cam = camera(W, H);
        unsigned hw = std::thread::hardware_concurrency();
        int nThreads = hw ? (int)hw : 4;
        std::vector<uint8_t> img =
            renderIsoPreviewCuda(s, cam, W, H, nThreads, 1.0, true, nullptr);
        if (img.empty()) { status = "raymarch unavailable (no CUDA device or unsupported scene)"; return; }
        if (upload(img, W, H, dev, ctx)) { status.clear(); dirty = false; }
        else status = "D3D11 texture upload failed";
    }
};

// The Render tab body: orbit controls + the blitted raymarch image.
static void drawRenderPane(RenderPane& rp, const Scene& scene, bool sceneOk,
                           const std::string& sceneErr,
                           ID3D11Device* dev, ID3D11DeviceContext* ctx) {
    if (!sceneOk) {
        ImGui::TextWrapped("No live scene to raymarch.");
        if (!sceneErr.empty()) ImGui::TextWrapped("(%s)", sceneErr.c_str());
        ImGui::TextWrapped("The sidecar carries no `source` .ftsl (older loom, or "
                           "emit_source was off). Re-save it with a current loom to "
                           "enable the in-process field raymarch.");
        return;
    }
    if (!rp.inited) rp.initFrom(scene);

    ImGui::TextUnformatted("GPU field raymarch (renderIsoPreviewCuda) - drag to orbit, wheel to zoom");
    ImGui::SetNextItemWidth(120);
    if (ImGui::SliderInt("res", &rp.resLong, 128, 1024)) rp.dirty = true;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    if (ImGui::SliderFloat("fov", &rp.fov, 10.0f, 110.0f, "%.0f deg")) rp.dirty = true;
    ImGui::SameLine();
    if (ImGui::Button("re-render")) rp.dirty = true;
    if (!rp.status.empty()) { ImGui::SameLine(); ImGui::TextColored(ImVec4(1, 0.6f, 0.4f, 1), "%s", rp.status.c_str()); }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.y < 80.0f) avail.y = 80.0f;
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("render_canvas", avail);
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

    if (rp.dirty) rp.render(scene, dev, ctx);

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
}
#endif // HAVE_CUDA

// --------------------------------------------------------------------------
// Entry point
// --------------------------------------------------------------------------
int runViewerGui(const std::string& sidecarPath) {
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
#endif

    // --- window ---
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
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    bool done = false;
    bool firstFrame = true;   // one-shot: default-select the primary geometry tab
    while (!done) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) done = true;
        }
        if (done) break;

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

        float leftW = ImGui::GetContentRegionAvail().x * 0.42f;
        ImGui::BeginChild("left", ImVec2(leftW, 0), true);
        if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
            drawScenePanel(sc);
        if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen))
            drawObjectsPanel(sc);
        if (ImGui::CollapsingHeader("Datasets", ImGuiTreeNodeFlags_DefaultOpen))
            drawDatasetsPanel(sc);
        if (ImGui::CollapsingHeader("Modulator DAG", ImGuiTreeNodeFlags_DefaultOpen)) {
            // imnodes wants its own non-scrolling area (it pans on drag itself)
            ImGui::BeginChild("dagpane", ImVec2(0, 360), true);
            drawDagPanel(dag);
            ImGui::EndChild();
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
                    drawRenderPane(rpane, loaded.scene, sceneOk, sceneErr,
                                   g_pd3dDevice, g_pd3dDeviceContext);
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
                    drawMeshPane(meshes, mview);
                    ImGui::EndTabItem();
                }
            }
            ImGui::EndTabBar();
        }
        ImGui::EndChild();

        ImGui::End();

        ImGui::Render();
        const float clear[4] = { 0.10f, 0.10f, 0.12f, 1.0f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRTV, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRTV, clear);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_pSwapChain->Present(1, 0);  // vsync
        firstFrame = false;
    }

#ifdef HAVE_CUDA
    rpane.release();   // free the raymarch texture before the D3D device goes away
#endif
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
