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
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx11.h"
#include "third_party/json.h"      // minijson: the vendored JSON parser

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
};

// A curve dataset flattened to draw-ready 3-D polylines.
struct CurveGeom {
    std::string              id;
    bool                     closed = false;
    std::vector<float>       poly;   // flat xyz triples (sampled polyline)
    std::vector<float>       ctrl;   // flat xyz triples (control points)
};

// Pull a flat xyz array out of a JSON array-of-arrays; pads/truncates to 3.
static void flattenPts(const minijson::Value* a, std::vector<float>& out) {
    out.clear();
    if (!a || !a->isArray()) return;
    for (const auto& p : a->arr) {
        float xyz[3] = {0, 0, 0};
        if (p.isArray())
            for (int k = 0; k < 3 && k < (int)p.arr.size(); ++k)
                xyz[k] = (float)p.arr[k].asNumber(0.0);
        out.push_back(xyz[0]); out.push_back(xyz[1]); out.push_back(xyz[2]);
    }
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
        flattenPts(poly, g.poly);
        flattenPts(d.find("control_points"), g.ctrl);
        curves.push_back(std::move(g));
    }
    return curves;
}

} // namespace

// --------------------------------------------------------------------------
// Curve pane: a simple orthographic projection drawn with ImDrawList.
// (Slice C generalizes this to 3-of-N dim selection, rotation, and stereo.)
// --------------------------------------------------------------------------
struct OrbitView {
    float yaw   = 0.6f;   // radians
    float pitch = 0.4f;
    float zoom  = 1.0f;
};

static ImVec2 project(const float* p, const OrbitView& v, ImVec2 center, float scale) {
    // rotate about Y (yaw) then X (pitch), orthographic drop of Z
    float cy = std::cos(v.yaw),   sy = std::sin(v.yaw);
    float cx = std::cos(v.pitch), sx = std::sin(v.pitch);
    float x = p[0], y = p[1], z = p[2];
    float x1 =  cy * x + sy * z;
    float z1 = -sy * x + cy * z;
    float y1 =  cx * y - sx * z1;
    return ImVec2(center.x + x1 * scale * v.zoom,
                  center.y - y1 * scale * v.zoom);
}

static void drawCurvePane(const std::vector<CurveGeom>& curves, OrbitView& view) {
    ImGui::TextUnformatted("Curves (orthographic preview — drag to orbit, wheel to zoom)");
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
    ImVec2 center(origin.x + avail.x * 0.5f, origin.y + avail.y * 0.5f);
    dl->AddRectFilled(origin, ImVec2(origin.x + avail.x, origin.y + avail.y),
                      IM_COL32(18, 18, 22, 255));

    // auto-fit scale from the union bounds of all curves
    float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
    for (const auto& c : curves)
        for (size_t i = 0; i + 2 < c.poly.size(); i += 3)
            for (int k = 0; k < 3; ++k) {
                lo[k] = std::min(lo[k], c.poly[i + k]);
                hi[k] = std::max(hi[k], c.poly[i + k]);
            }
    float ext = 1.0f;
    for (int k = 0; k < 3; ++k) ext = std::max(ext, hi[k] - lo[k]);
    float scale = 0.4f * std::min(avail.x, avail.y) / (0.5f * ext + 1e-3f);
    // recenter data about its midpoint
    float mid[3] = { 0, 0, 0 };
    for (int k = 0; k < 3; ++k) mid[k] = 0.5f * (lo[k] + hi[k]);

    const ImU32 palette[] = {
        IM_COL32(120, 200, 255, 255), IM_COL32(255, 180, 120, 255),
        IM_COL32(160, 255, 160, 255), IM_COL32(255, 140, 200, 255),
    };
    int ci = 0;
    for (const auto& c : curves) {
        ImU32 col = palette[ci % 4]; ++ci;
        ImVec2 prev; bool have = false;
        for (size_t i = 0; i + 2 < c.poly.size(); i += 3) {
            float p[3] = { c.poly[i] - mid[0], c.poly[i+1] - mid[1], c.poly[i+2] - mid[2] };
            ImVec2 s = project(p, view, center, scale);
            if (have) dl->AddLine(prev, s, col, 1.6f);
            prev = s; have = true;
        }
        // control points as small squares
        for (size_t i = 0; i + 2 < c.ctrl.size(); i += 3) {
            float p[3] = { c.ctrl[i] - mid[0], c.ctrl[i+1] - mid[1], c.ctrl[i+2] - mid[2] };
            ImVec2 s = project(p, view, center, scale);
            dl->AddRectFilled(ImVec2(s.x - 2, s.y - 2), ImVec2(s.x + 2, s.y + 2),
                              IM_COL32(255, 255, 255, 220));
        }
    }
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
// Entry point
// --------------------------------------------------------------------------
int runViewerGui(const std::string& sidecarPath) {
    Sidecar sc;
    if (!sc.load(sidecarPath)) {
        std::fprintf(stderr, "error: -viewer: %s\n", sc.err.c_str());
        return 1;
    }
    std::vector<CurveGeom> curves = collectCurves(sc);

    // --- window ---
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0, 0,
                       GetModuleHandle(nullptr), nullptr, nullptr, nullptr,
                       nullptr, L"FtraceViewer", nullptr };
    RegisterClassExW(&wc);
    HWND hwnd = CreateWindowW(wc.lpszClassName, L"ftrace \xF0\x9F\xAA\x9F loom viewer",
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
    ImGui::GetIO().IniFilename = nullptr;   // don't litter an imgui.ini in the CWD
    ImGui::StyleColorsDark();
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    OrbitView view;
    bool done = false;
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
        ImGui::EndChild();

        ImGui::SameLine();
        ImGui::BeginChild("right", ImVec2(0, 0), true);
        drawCurvePane(curves, view);
        ImGui::EndChild();

        ImGui::End();

        ImGui::Render();
        const float clear[4] = { 0.10f, 0.10f, 0.12f, 1.0f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRTV, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRTV, clear);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_pSwapChain->Present(1, 0);  // vsync
    }

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    CleanupDeviceD3D();
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return 0;
}

#endif // _WIN32
