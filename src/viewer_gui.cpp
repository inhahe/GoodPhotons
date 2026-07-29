#include "viewer_gui.h"

#ifndef _WIN32
// -------- Non-Windows stub: the native viewer needs Win32 + D3D11 --------------
#include <cstdio>
int runViewerGui(const std::string&, const std::string&) {
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
#include <map>
#include <unordered_map>
#include <functional>
#include <thread>
#include <mutex>                   // F4 item 2: the live re-introspection job queue
#include <condition_variable>
#include <cstring>
#include <cstdlib>                 // strtoul: parsing the pid out of a scratch dir name
#include <cwctype>

// Bridge to ftrace's own scene loader + GPU field raymarcher (F7 primary path).
// The viewer IS the ftrace binary, so it can parse loom's emitted `.ftsl` with the
// exact loader main() uses and render the real isosurface field in-process via
// renderIsoPreviewCuda — the `-raster-gpu` preview kernel that sphere-traces the
// field's bytecode with NO tessellation (the static marching-cubes mesh in the
// sidecar is only a fallback). These headers are plain-C++ (main.cpp includes them
// under MSVC too); the raymarch itself is guarded by HAVE_CUDA below.
#include "ftsl.h"
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
};

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
// Shading is kept identical to the old CPU path: flat two-sided lambert
// 0.30 + 0.70*|n.z| with n the FACE normal in the rotated view basis. The GPU
// recovers that per-pixel from screen-space derivatives of the view-space
// position; under the orthographic projection used here that is exact, not an
// approximation. The one deliberate improvement is the UV checker, which is now
// evaluated per-pixel at the interpolated UV instead of once at the triangle
// centroid — the whole point of a UV checker is to show UV distortion *within* a
// face, which a flat centroid sample cannot do.
// --------------------------------------------------------------------------
struct MeshGpu {
    // pipeline objects (created once, on first use)
    ID3D11VertexShader*      vs      = nullptr;
    ID3D11PixelShader*       ps      = nullptr;
    ID3D11InputLayout*       layout  = nullptr;
    ID3D11Buffer*            cb      = nullptr;
    ID3D11RasterizerState*   rsSolid = nullptr;
    ID3D11RasterizerState*   rsWire  = nullptr;
    ID3D11DepthStencilState* dsSolid = nullptr;   // LESS, writes depth
    ID3D11DepthStencilState* dsWire  = nullptr;   // LESS_EQUAL, no depth write
    ID3D11BlendState*        blend   = nullptr;
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

    // offscreen colour + depth target, resized to the pane
    ID3D11Texture2D*          colorTex = nullptr;
    ID3D11RenderTargetView*   rtv      = nullptr;
    ID3D11ShaderResourceView* srv      = nullptr;
    ID3D11Texture2D*          depthTex = nullptr;
    ID3D11DepthStencilView*   dsv      = nullptr;
    int texW = 0, texH = 0;

    std::string err;                // non-empty => the pane says so instead of drawing

    struct Vert { float x, y, z, u, v; };
    // Must match the cbuffer in the shader below (144 B, a multiple of 16).
    struct CB {
        float mvp[16];
        float rot0[4], rot1[4], rot2[4];   // xyz = view-basis row, w = -(row . mid)
        float baseColor[4];
        float opts[4];                     // x = shade on, y = colour mode
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
        if (dsWire)  { dsWire->Release();  dsWire = nullptr; }
        if (dsSolid) { dsSolid->Release(); dsSolid = nullptr; }
        if (rsWire)  { rsWire->Release();  rsWire = nullptr; }
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
};
struct VSIn  { float3 p : POSITION; float2 uv : TEXCOORD0; };
struct VSOut { float4 pos : SV_Position; float3 vp : TEXCOORD1; float2 uv : TEXCOORD0; };
VSOut main(VSIn i) {
    VSOut o;
    o.pos = mul(mvp, float4(i.p, 1.0));
    // view-space position, used ONLY for the flat face normal via ddx/ddy
    o.vp  = float3(dot(rot0.xyz, i.p) + rot0.w,
                   dot(rot1.xyz, i.p) + rot1.w,
                   dot(rot2.xyz, i.p) + rot2.w);
    o.uv  = i.uv;
    return o;
}
)HLSL";

        static const char* kPS = R"HLSL(
cbuffer CB : register(b0) {
    row_major float4x4 mvp;
    float4 rot0, rot1, rot2;
    float4 baseColor;
    float4 opts;
};
Texture2D    tex0  : register(t0);
SamplerState samp0 : register(s0);
struct VSOut { float4 pos : SV_Position; float3 vp : TEXCOORD1; float2 uv : TEXCOORD0; };
float4 main(VSOut i) : SV_Target {
    float3 base = baseColor.rgb;
    int mode = (int)opts.y;
    if (mode == 2) {
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
    if (opts.x > 0.5) {
        float3 n = normalize(cross(ddx(i.vp), ddy(i.vp)));
        sh = 0.30 + 0.70 * abs(n.z);          // two-sided lambert, flat per face
    }
    return float4(base * sh, baseColor.a);
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
        };
        HRESULT hr = dev->CreateInputLayout(il, 2, vsb->GetBufferPointer(), vsb->GetBufferSize(), &layout);
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
        if (FAILED(dev->CreateRasterizerState(&rd, &rsSolid))) return fail("rasterizer(solid) failed", nullptr);
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

    bool ensureTargets(ID3D11Device* dev, int W, int H) {
        if (W < 1) W = 1;
        if (H < 1) H = 1;
        if (colorTex && texW == W && texH == H) return true;
        releaseTargets();
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &colorTex))) { err = "mesh colour target failed"; return false; }
        if (FAILED(dev->CreateRenderTargetView(colorTex, nullptr, &rtv))) { releaseTargets(); err = "mesh RTV failed"; return false; }
        if (FAILED(dev->CreateShaderResourceView(colorTex, nullptr, &srv))) { releaseTargets(); err = "mesh SRV failed"; return false; }
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
        ranges.resize(meshes.size());
        for (size_t mi = 0; mi < meshes.size(); ++mi) {
            const MeshGeom& m = meshes[mi];
            Range r;
            r.baseVertex = (INT)verts.size();
            r.firstIndex = (UINT)idx.size();
            bool hasUv = (int)m.uvs.size() >= 2 * m.nverts;
            for (int i = 0; i < m.nverts; ++i) {
                Vert v;
                v.x = m.verts[(size_t)i * 3 + 0];
                v.y = m.verts[(size_t)i * 3 + 1];
                v.z = m.verts[(size_t)i * 3 + 2];
                v.u = hasUv ? m.uvs[(size_t)i * 2 + 0] : 0.0f;
                v.v = hasUv ? m.uvs[(size_t)i * 2 + 1] : 0.0f;
                verts.push_back(v);
            }
            for (int f = 0; f < m.nfaces; ++f) {
                int f0 = m.faces[(size_t)f * 3 + 0], f1 = m.faces[(size_t)f * 3 + 1],
                    f2 = m.faces[(size_t)f * 3 + 2];
                if (f0 < 0 || f1 < 0 || f2 < 0 || f0 >= m.nverts || f1 >= m.nverts || f2 >= m.nverts)
                    continue;   // a malformed face is skipped, exactly as before
                idx.push_back((uint32_t)f0); idx.push_back((uint32_t)f1); idx.push_back((uint32_t)f2);
            }
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
// F4 — mesh pane: SweptMesh tessellated surfaces as a shaded, z-buffered
// triangle mesh. Orbiting the 3 spatial dims is a view-only re-projection (no
// re-tessellation, exactly as the F4 rule specifies for isometries of the shown
// dims). Colour: flat lambert shading, per-object tint, a UV checker, or the
// material's real skin sampled per-pixel at the interpolated mesh UVs.
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
};

struct LoomResult {
    long long   seq = 0;
    bool        ok  = false;
    std::string err;
    std::string sidecarPath;    // temp file loom wrote (empty when not requested)
    std::string sourcePath;
    double      ms = 0.0;
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
            std::lock_guard<std::mutex> lk(m_);
            for (const auto& f : temps_) DeleteFileA(f.c_str());
            temps_.clear();
            // Sweep the whole scratch directory, not just the files we named: emitting
            // an .ftsl also drops the mesh assets it references (loom's `asset_path`
            // writes .obj next to `out`), and RemoveDirectory fails on a non-empty dir.
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

    bool take(LoomResult& out) {
        std::lock_guard<std::mutex> lk(m_);
        if (!hasResult_) return false;
        out = result_;
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

    // temp scratch files the UI has finished reading
    void reap(const std::string& path) {
        if (path.empty()) return;
        DeleteFileA(path.c_str());
        std::lock_guard<std::mutex> lk(m_);
        forgetLocked(path);
    }

private:
    // `temps_` is the outstanding-scratch-file set, not a log: a long sweep posts
    // hundreds of jobs, so entries must leave it as the files are deleted or it (and
    // the %TEMP% directory it mirrors) would grow without bound for the session.
    void forgetLocked(const std::string& path) {
        for (size_t i = 0; i < temps_.size(); ++i)
            if (temps_[i] == path) { temps_[i] = temps_.back(); temps_.pop_back(); return; }
    }

    void dropLocked(const std::string& path) {
        if (path.empty()) return;
        DeleteFileA(path.c_str());
        forgetLocked(path);
    }

    // Delete `ftrace_viewer_<pid>` directories left behind by viewers that died without
    // running stop() — a crash, or the user killing the process. `stop()` handles the
    // orderly exit, but nothing can clean up after a kill except the *next* run, and a
    // scene bake drops a multi-megabyte sidecar plus its .obj assets each time, so
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

    bool makeTempDir(std::string& err) {
        char tmp[MAX_PATH + 1];
        DWORD n = GetTempPathA(MAX_PATH, tmp);
        if (n == 0 || n > MAX_PATH) { err = "GetTempPath failed"; return false; }
        sweepOrphanTempDirs(tmp);
        char dir[MAX_PATH + 64];
        std::snprintf(dir, sizeof dir, "%sftrace_viewer_%lu", tmp, GetCurrentProcessId());
        if (!CreateDirectoryA(dir, nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) {
            err = "cannot create the viewer scratch directory"; return false;
        }
        tempDir_ = dir;
        return true;
    }

    // Called on the worker thread, so the bookkeeping must take the lock (tempDir_ is
    // written once, before the worker exists, and is only cleared after it is joined).
    std::string scratch(long long seq, const char* ext) {
        char b[MAX_PATH + 64];
        std::snprintf(b, sizeof b, "%s\\live_%lld%s", tempDir_.c_str(), seq, ext);
        std::string p = b;
        { std::lock_guard<std::mutex> lk(m_); temps_.push_back(p); }
        return p;
    }

    std::string requestLine(const char* cmd, const LoomJob& j, const std::string& out) {
        std::string s = "{\"cmd\":\"";
        s += cmd;
        s += "\",\"clock\":{\"frame\":" + std::to_string(j.frame)
           + ",\"frames\":" + std::to_string(j.frames) + "},\"params\":{";
        for (size_t i = 0; i < j.params.size(); ++i) {
            if (i) s += ",";
            s += "\"" + jsonEsc(j.params[i].first) + "\":" + j.params[i].second;
        }
        s += "},\"out\":\"" + jsonEsc(out) + "\"}";
        return s;
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
            LARGE_INTEGER f, t0, t1;
            QueryPerformanceFrequency(&f);
            QueryPerformanceCounter(&t0);
            std::string err;
            bool ok = true;
            minijson::Value ack;
            // A failed request may still have left a partial file behind: drop it here
            // rather than let it sit in temps_ until the viewer exits.
            if (ok && job.wantSidecar) {
                std::string out = scratch(job.seq, ".json");
                ok = link_.call(requestLine("introspect", job, out), ack, err);
                if (ok) r.sidecarPath = out;
                else { std::lock_guard<std::mutex> lk(m_); dropLocked(out); }
            }
            if (ok && job.wantSource) {
                std::string out = scratch(job.seq, ".ftsl");
                ok = link_.call(requestLine("emit", job, out), ack, err);
                if (ok) r.sourcePath = out;
                else { std::lock_guard<std::mutex> lk(m_); dropLocked(out); }
            }
            QueryPerformanceCounter(&t1);
            r.ms = f.QuadPart ? 1000.0 * double(t1.QuadPart - t0.QuadPart) / double(f.QuadPart) : 0.0;
            r.ok = ok;
            r.err = err;
            {
                std::lock_guard<std::mutex> lk(m_);
                // A result must never overwrite a FRESHER one the UI has not read yet;
                // with one job in flight at a time that can't happen, but the guard
                // makes the invariant explicit rather than incidental.
                if (!hasResult_ || r.seq >= result_.seq) {
                    // Superseding an unread result: the UI will never call reap() for
                    // its files, so they have to go here or a fast sweep leaves one
                    // scratch pair per skipped bake behind in %TEMP%.
                    if (hasResult_) {
                        dropLocked(result_.sidecarPath);
                        dropLocked(result_.sourcePath);
                    }
                    result_ = r;
                    hasResult_ = true;
                } else {
                    dropLocked(r.sidecarPath);
                    dropLocked(r.sourcePath);
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
    std::vector<std::string> temps_;
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

static LoomJob liveJob(const LivePanel& lp, bool wantSidecar, bool wantSource) {
    LoomJob j;
    j.frame = lp.frame;
    j.frames = std::max(1, lp.frames);
    j.wantSidecar = wantSidecar;
    j.wantSource = wantSource;
    for (const auto& p : lp.params) j.params.push_back({p.name, p.toJson()});
    return j;
}

// The left-column "Live (loom)" section. Returns true when something the geometry
// depends on moved this frame.
static bool drawLivePanel(LivePanel& lp, LoomBridge& br) {
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
int runViewerGui(const std::string& sidecarPath, const std::string& loomScene) {
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
    // the SAME directory — loom's scratch sidecar lives in %TEMP%, but the image paths
    // in it are still relative to the original scene, not to the scratch file.
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
    auto adoptSidecar = [&](const std::string& path) -> bool {
        Sidecar ns;
        if (!ns.load(path)) { live.lastErr = "sidecar: " + ns.err; return false; }
        std::vector<int> oldIds;
        for (const auto& n : dag.nodes) oldIds.push_back(n.id);

        sc = std::move(ns);
        curves = collectCurves(sc);
        strips = buildStrips(curves);
        fields = collectFields(sc);
        meshes = collectMeshes(sc);
        ++mview.geomGen;   // a NEW tessellation -> the mesh pane must re-upload its buffers
        for (const auto& c : curves) view.maxDim  = std::max(view.maxDim,  c.dim);
        for (const auto& f : fields) fview.maxDim = std::max(fview.maxDim, f.dim);

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

        skins.release();
        skins.build(sc, baseDir, g_pd3dDevice, g_pd3dDeviceContext);
        return true;
    };

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
                if (r.ok) {
                    if (!r.sidecarPath.empty()) adoptSidecar(r.sidecarPath);
                    if (!r.sourcePath.empty()) {
                        ftsl::Loaded nl;
                        std::string  nerr;
                        if (ftsl::load(r.sourcePath, nl, nerr)) {
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
                bridge.reap(r.sidecarPath);
                bridge.reap(r.sourcePath);
            }
        }

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

        bool livePost = false;    // something the geometry depends on moved this frame

        float leftW = ImGui::GetContentRegionAvail().x * 0.42f;
        ImGui::BeginChild("left", ImVec2(leftW, 0), true);
        if (ImGui::CollapsingHeader("Live (loom)",
                                    live.up ? ImGuiTreeNodeFlags_DefaultOpen : 0))
            livePost = drawLivePanel(live, bridge);
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

        // At most one post per frame, and posting OVERWRITES any job that has not
        // started: a fast sweep drag therefore costs one bake of wherever the user
        // ends up, not one bake per intermediate frame. That is the latest-wins rule.
        if (livePost && live.up && bridge.linkUp()) {
            bridge.post(liveJob(live, /*wantSidecar=*/true, liveWantSource));
            ++live.posted;
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
