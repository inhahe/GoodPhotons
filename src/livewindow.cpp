#include "livewindow.h"

// Shared by both builds: on a headless build there is no window to minimize, but the
// setter must still link so callers need no platform guard.
static bool g_lwStartMinimized = false;
void LiveWindow::setStartMinimized(bool on) { g_lwStartMinimized = on; }
bool LiveWindow::startMinimized() { return g_lwStartMinimized; }

#ifndef _WIN32
// -------- Non-Windows stub: -window is a no-op (headless builds unaffected) --------
struct LiveWindow::Impl {};
LiveWindow::LiveWindow(int, int, const char*) : impl_(nullptr) {}
LiveWindow::~LiveWindow() {}
void LiveWindow::update(int, int, const std::vector<uint8_t>&) {}
bool LiveWindow::renderShared(int, int, const std::function<bool(void*, void*)>&) { return false; }
void LiveWindow::setTitle(const std::string&) {}
bool LiveWindow::closed() const { return false; }
NavInput LiveWindow::drainNav() { return {}; }
bool LiveWindow::clientSize(int&, int&) const { return false; }
void LiveWindow::enablePanel(int, double, const char*) {}
void LiveWindow::setPanelState(int, bool, bool, const char*) {}
void LiveWindow::setPathCount(int) {}
void LiveWindow::setEditState(bool, int) {}
void LiveWindow::setSpeedLabel(double) {}
void LiveWindow::enableBindRow(const std::vector<std::string>&, int) {}
void LiveWindow::setBindState(const std::vector<std::string>&, const char*) {}

#else
// ------------------------------- Win32 GDI window ----------------------------------
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <windowsx.h>          // GET_X_LPARAM / GET_Y_LPARAM
#include <commctrl.h>          // trackbar (msctls_trackbar32) for the timeline
#include <d3d11.h>             // the image area's swap chain (replaces the GDI StretchDIBits tail)
#include <dxgi1_2.h>           // CreateSwapChainForHwnd / flip-model presentation
#include <d3dcompiler.h>       // the two present shaders, compiled at runtime
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <cstdlib>             // strtod / atoi for the speed inputs
#include <algorithm>
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

// Convert a UTF-8 byte string to UTF-16 for the Win32 *W APIs. The old code did a
// naive `assign(begin, end)` byte-widen, which mangles any non-ASCII: an em dash
// "—" (UTF-8 0xE2 0x80 0x94) became THREE junk wchars, so the title bar showed
// "ftrace <3 garbage glyphs> live preview". MultiByteToWideChar decodes it properly.
static std::wstring utf8ToWide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    if (n <= 0) return std::wstring(s.begin(), s.end());   // fall back to byte-widen
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}

// The inverse of utf8ToWide: needed to read a name back OUT of a combo box (the loom bind row's
// slot list), since a scene variable's name is UTF-8 everywhere else in ftrace.
static std::string wideToUtf8(const std::wstring& w) {
    if (w.empty()) return std::string();
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    if (n <= 0) {
        // Unreachable for valid UTF-16, but a bind-row slot name must never come back EMPTY
        // (that would read as "(none)" and silently unbind the channel), so salvage the ASCII
        // subset rather than returning nothing. The cast is explicit: this narrowing is the
        // intent, not an accident.
        std::string s;
        s.reserve(w.size());
        for (wchar_t c : w) s.push_back((c < 0x80) ? (char)c : '?');
        return s;
    }
    std::string s((size_t)n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], n, nullptr, nullptr);
    return s;
}

// =====================================================================================
//                      D3D11 presenter for the live image area
// =====================================================================================
// Why this exists: the old present path was pure CPU + GDI, and it cost far more than
// the render it was displaying. Per frame it did
//
//   1. a scalar per-pixel RGB8 -> BGRA repack in LiveWindow::update(), and
//   2. CreateCompatibleDC/Bitmap + FillRect + SetStretchBltMode(HALFTONE) +
//      StretchDIBits + BitBlt in WM_PAINT,
//
// and because paint() held the same mutex update() needs, step 2 sat squarely on the
// render thread's critical path. Measured on this machine (scraps/tailbench.cpp,
// reproducing both steps byte-for-byte):
//
//   image -> client            repack    HALFTONE blit
//   1920x1920 -> 1264x1264     5.07 ms       23.99 ms
//   1264x1264 -> 1264x1264     2.35 ms        7.83 ms
//   1920x1080 -> 1920x1080     2.94 ms       12.37 ms
//
// — i.e. 10-29 ms of host work behind a GPU raster frame that takes 9.12 ms in total.
// So the tail, not the render, was the frame-rate limiter.
//
// The replacement uploads the renderer's RGB8 bytes untouched (no repack: there is no
// RGB8 DXGI format, so the buffer goes up as an R8 texture 3x as wide and a pixel
// shader deswizzles it into an RGBA8 image texture), then presents that texture through
// a flip-model swap chain with a letterboxed viewport and a linear sampler — the GPU
// doing the scale GDI's HALFTONE was doing on the CPU.
//
// It is also the prerequisite for the zero-copy path: once D3D owns the image texture,
// the CUDA rasterizer can write it directly (cudaGraphicsD3D11RegisterResource +
// surf2Dwrite) and the device->host download disappears too.
//
// Threading: every entry point locks `mtx`, because ID3D11DeviceContext is not
// thread-safe and both the render thread (upload+present from update()) and the UI
// thread (re-present on resize/expose) drive it.
//
// Fallback: any failure here leaves `ok` false and LiveWindow keeps using the original
// GDI path, which is retained in full anyway — WM_PRINTCLIENT capture (PrintWindow
// cannot see swap-chain content) still goes through it, fed by a GPU readback.

// One shared vertex shader: a full-screen triangle generated from SV_VertexID, so no
// vertex/index buffer and no input layout is needed.
static const char* kPresentVS =
    "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
    "VSOut main(uint id : SV_VertexID) {\n"
    "    VSOut o;\n"
    "    float2 t = float2((id << 1) & 2, id & 2);\n"   // (0,0) (2,0) (0,2)
    "    o.uv  = t;\n"
    "    o.pos = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);\n"
    "    return o;\n"
    "}\n";

// Pass 1: RGB8 -> RGBA8. The source is bound as an R8 texture of width 3*W, so pixel
// (x,y) of the image is bytes (3x, 3x+1, 3x+2) of row y. Load() (not Sample) because
// the addressing is exact texel indexing, and filtering across the interleaved channels
// would be meaningless.
static const char* kDeswizzlePS =
    "Texture2D<float> packed : register(t0);\n"
    "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
    "float4 main(VSOut i) : SV_Target {\n"
    "    int3 p = int3((int)i.pos.x * 3, (int)i.pos.y, 0);\n"
    "    return float4(packed.Load(p),\n"
    "                  packed.Load(p + int3(1, 0, 0)),\n"
    "                  packed.Load(p + int3(2, 0, 0)), 1.0);\n"
    "}\n";

// Pass 2: the image texture stretched into the letterboxed viewport, bilinear.
static const char* kBlitPS =
    "Texture2D    img   : register(t0);\n"
    "SamplerState samp0 : register(s0);\n"
    "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
    "float4 main(VSOut i) : SV_Target { return float4(img.Sample(samp0, i.uv).rgb, 1.0); }\n";

template <class T> static void relCom(T*& p) { if (p) { p->Release(); p = nullptr; } }

struct LivePresenter {
    std::mutex               mtx;                 // guards EVERYTHING below (ctx is not thread-safe)
    bool                     ok = false;          // pipeline usable; false => caller must use GDI
    ID3D11Device*            dev  = nullptr;
    ID3D11DeviceContext*     ctx  = nullptr;
    IDXGISwapChain1*         swap = nullptr;
    ID3D11RenderTargetView*  backRTV = nullptr;
    int                      swapW = 0, swapH = 0;
    ID3D11VertexShader*      vs = nullptr;
    ID3D11PixelShader*       psDeswizzle = nullptr, *psBlit = nullptr;
    ID3D11SamplerState*      samp = nullptr;
    ID3D11RasterizerState*   rsNoCull = nullptr;
    // Upload staging: the renderer's RGB8 rows, as an R8 texture 3x as wide.
    ID3D11Texture2D*         packTex = nullptr;
    ID3D11ShaderResourceView* packSRV = nullptr;
    int                      packW = 0, packH = 0;
    // The image itself, in the only layout D3D can filter. Also the surface CUDA will
    // eventually write into directly.
    ID3D11Texture2D*         imgTex = nullptr;
    ID3D11RenderTargetView*  imgRTV = nullptr;
    ID3D11ShaderResourceView* imgSRV = nullptr;
    int                      imgW = 0, imgH = 0;
    ID3D11Texture2D*         readTex = nullptr;   // STAGING copy, allocated only when a capture asks

    bool init(HWND hview);
    void release();
    bool ensureImg(int w, int h);
    bool ensurePack(int w, int h);
    bool ensureSwap(int w, int h);
    void fullScreenPass(ID3D11PixelShader* ps, ID3D11ShaderResourceView* srv,
                        ID3D11RenderTargetView* rtv,
                        float vx, float vy, float vw, float vh, bool clear);
    bool uploadHost(int w, int h, const uint8_t* rgb);
    bool present(int viewW, int viewH);
    bool readbackBgra(std::vector<uint8_t>& bgra, int& w, int& h);
};

static ID3DBlob* compileShader(const char* src, const char* target) {
    ID3DBlob* code = nullptr; ID3DBlob* err = nullptr;
    HRESULT hr = D3DCompile(src, strlen(src), nullptr, nullptr, nullptr, "main", target,
                            D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err);
    if (err) err->Release();
    if (FAILED(hr)) { relCom(code); return nullptr; }
    return code;
}

bool LivePresenter::init(HWND hview) {
    if (!hview) return false;
    D3D_FEATURE_LEVEL want[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1,
                                 D3D_FEATURE_LEVEL_10_0 };
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                                   want, (UINT)std::size(want), D3D11_SDK_VERSION,
                                   &dev, nullptr, &ctx);
    if (FAILED(hr)) { release(); return false; }

    // Reach the factory through the device's own adapter, so the swap chain is created
    // on the same GPU the device lives on (matters on hybrid iGPU/dGPU laptops).
    IDXGIDevice*  dxgiDev = nullptr;
    IDXGIAdapter* adapter = nullptr;
    IDXGIFactory2* factory = nullptr;
    if (FAILED(dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDev)) ||
        FAILED(dxgiDev->GetAdapter(&adapter)) ||
        FAILED(adapter->GetParent(__uuidof(IDXGIFactory2), (void**)&factory))) {
        relCom(factory); relCom(adapter); relCom(dxgiDev); release(); return false;
    }
    DXGI_SWAP_CHAIN_DESC1 sd{};
    sd.Width  = 0; sd.Height = 0;                  // 0 => track the HWND's client size
    sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 2;
    sd.Scaling     = DXGI_SCALING_STRETCH;
    sd.SwapEffect  = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.AlphaMode   = DXGI_ALPHA_MODE_IGNORE;
    hr = factory->CreateSwapChainForHwnd(dev, hview, &sd, nullptr, nullptr, &swap);
    if (FAILED(hr)) {
        // Pre-Win10 (or a driver without flip-discard): fall back to the bitblt model.
        sd.SwapEffect  = DXGI_SWAP_EFFECT_DISCARD;
        sd.BufferCount = 1;
        sd.Scaling     = DXGI_SCALING_STRETCH;
        hr = factory->CreateSwapChainForHwnd(dev, hview, &sd, nullptr, nullptr, &swap);
    }
    // Alt-Enter must not turn the preview into an exclusive-fullscreen surprise.
    factory->MakeWindowAssociation(hview, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES);
    relCom(factory); relCom(adapter); relCom(dxgiDev);
    if (FAILED(hr)) { release(); return false; }

    ID3DBlob* bvs = compileShader(kPresentVS,   "vs_4_0");
    ID3DBlob* bd  = compileShader(kDeswizzlePS, "ps_4_0");
    ID3DBlob* bb  = compileShader(kBlitPS,      "ps_4_0");
    bool shOk = bvs && bd && bb &&
        SUCCEEDED(dev->CreateVertexShader(bvs->GetBufferPointer(), bvs->GetBufferSize(), nullptr, &vs)) &&
        SUCCEEDED(dev->CreatePixelShader (bd->GetBufferPointer(),  bd->GetBufferSize(),  nullptr, &psDeswizzle)) &&
        SUCCEEDED(dev->CreatePixelShader (bb->GetBufferPointer(),  bb->GetBufferSize(),  nullptr, &psBlit));
    relCom(bvs); relCom(bd); relCom(bb);
    if (!shOk) { release(); return false; }

    D3D11_SAMPLER_DESC sda{};
    sda.Filter   = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sda.AddressU = sda.AddressV = sda.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sda.MaxLOD   = D3D11_FLOAT32_MAX;
    if (FAILED(dev->CreateSamplerState(&sda, &samp))) { release(); return false; }

    // The full-screen triangle's winding depends on nothing but SV_VertexID, so rather
    // than reason about it, disable culling.
    D3D11_RASTERIZER_DESC rd{};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    rd.DepthClipEnable = TRUE;
    if (FAILED(dev->CreateRasterizerState(&rd, &rsNoCull))) { release(); return false; }

    ok = true;
    return true;
}

void LivePresenter::release() {
    ok = false;
    relCom(readTex);
    relCom(imgSRV); relCom(imgRTV); relCom(imgTex); imgW = imgH = 0;
    relCom(packSRV); relCom(packTex); packW = packH = 0;
    relCom(rsNoCull); relCom(samp);
    relCom(psBlit); relCom(psDeswizzle); relCom(vs);
    relCom(backRTV); swapW = swapH = 0;
    relCom(swap);
    if (ctx) { ctx->ClearState(); ctx->Flush(); }
    relCom(ctx); relCom(dev);
}

// The image texture: RGBA8, render-targetable (pass 1 writes it) and samplable (pass 2
// reads it). Recreated only when the render resolution changes.
bool LivePresenter::ensureImg(int w, int h) {
    if (imgTex && imgW == w && imgH == h) return true;
    relCom(readTex);                      // its size is tied to the image's
    relCom(imgSRV); relCom(imgRTV); relCom(imgTex);
    imgW = imgH = 0;
    D3D11_TEXTURE2D_DESC td{};
    td.Width = (UINT)w; td.Height = (UINT)h; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(dev->CreateTexture2D(&td, nullptr, &imgTex))) return false;
    if (FAILED(dev->CreateRenderTargetView(imgTex, nullptr, &imgRTV)) ||
        FAILED(dev->CreateShaderResourceView(imgTex, nullptr, &imgSRV))) {
        relCom(imgSRV); relCom(imgRTV); relCom(imgTex); return false;
    }
    imgW = w; imgH = h;
    return true;
}

// The upload staging texture: DYNAMIC R8, 3*W wide, so the renderer's tightly-packed
// RGB8 rows go up as-is with one memcpy per row and no channel shuffling on the CPU.
bool LivePresenter::ensurePack(int w, int h) {
    if (packTex && packW == w && packH == h) return true;
    relCom(packSRV); relCom(packTex); packW = packH = 0;
    D3D11_TEXTURE2D_DESC td{};
    td.Width = (UINT)w; td.Height = (UINT)h; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DYNAMIC;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(dev->CreateTexture2D(&td, nullptr, &packTex))) return false;
    if (FAILED(dev->CreateShaderResourceView(packTex, nullptr, &packSRV))) {
        relCom(packTex); return false;
    }
    packW = w; packH = h;
    return true;
}

bool LivePresenter::ensureSwap(int w, int h) {
    if (backRTV && swapW == w && swapH == h) return true;
    relCom(backRTV);                                   // must drop every buffer reference first
    DXGI_SWAP_CHAIN_DESC1 sd{};
    // Compare against the buffers' ACTUAL size rather than a cached one: at creation the
    // swap chain sized itself from the HWND, so the first acquire usually needs no resize.
    if (SUCCEEDED(swap->GetDesc1(&sd)) && ((int)sd.Width != w || (int)sd.Height != h)) {
        if (FAILED(swap->ResizeBuffers(0, (UINT)w, (UINT)h, DXGI_FORMAT_UNKNOWN, 0))) return false;
    }
    ID3D11Texture2D* back = nullptr;
    if (FAILED(swap->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&back))) return false;
    HRESULT hr = dev->CreateRenderTargetView(back, nullptr, &backRTV);
    back->Release();
    if (FAILED(hr)) return false;
    swapW = w; swapH = h;
    return true;
}

// One full-screen-triangle pass. `clear` blacks the target first (the letterbox bars).
void LivePresenter::fullScreenPass(ID3D11PixelShader* ps, ID3D11ShaderResourceView* srv,
                                   ID3D11RenderTargetView* rtv,
                                   float vx, float vy, float vw, float vh, bool clear) {
    ID3D11ShaderResourceView* noSrv = nullptr;
    ctx->OMSetRenderTargets(1, &rtv, nullptr);
    if (clear) { const float black[4] = {0, 0, 0, 1}; ctx->ClearRenderTargetView(rtv, black); }
    D3D11_VIEWPORT vp{ vx, vy, vw, vh, 0.0f, 1.0f };
    ctx->RSSetViewports(1, &vp);
    ctx->RSSetState(rsNoCull);
    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(vs, nullptr, 0);
    ctx->PSSetShader(ps, nullptr, 0);
    ctx->PSSetShaderResources(0, 1, &srv);
    ctx->PSSetSamplers(0, 1, &samp);
    ctx->OMSetBlendState(nullptr, nullptr, 0xffffffff);
    ctx->OMSetDepthStencilState(nullptr, 0);
    ctx->Draw(3, 0);
    ctx->PSSetShaderResources(0, 1, &noSrv);      // never leave the target bound as input
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
}

// Hand the renderer's RGB8 frame to the GPU. Caller holds mtx.
bool LivePresenter::uploadHost(int w, int h, const uint8_t* rgb) {
    if (!ok || w <= 0 || h <= 0) return false;
    if (!ensurePack(w * 3, h) || !ensureImg(w, h)) return false;
    D3D11_MAPPED_SUBRESOURCE ms{};
    if (FAILED(ctx->Map(packTex, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) return false;
    const size_t rowBytes = (size_t)w * 3;
    if (ms.RowPitch == rowBytes) {
        memcpy(ms.pData, rgb, rowBytes * (size_t)h);       // one shot when the pitch matches
    } else {
        uint8_t* dst = (uint8_t*)ms.pData;
        for (int y = 0; y < h; ++y) memcpy(dst + (size_t)y * ms.RowPitch, rgb + (size_t)y * rowBytes, rowBytes);
    }
    ctx->Unmap(packTex, 0);
    fullScreenPass(psDeswizzle, packSRV, imgRTV, 0.0f, 0.0f, (float)w, (float)h, false);
    return true;
}

// Draw the image texture, aspect-fit and letterboxed, into the swap chain. Caller holds mtx.
bool LivePresenter::present(int viewW, int viewH) {
    if (!ok || viewW <= 0 || viewH <= 0) return false;
    if (!ensureSwap(viewW, viewH)) return false;
    if (imgW > 0 && imgH > 0) {
        double s = std::min((double)viewW / imgW, (double)viewH / imgH);
        int dw = std::max(1, (int)(imgW * s)), dh = std::max(1, (int)(imgH * s));
        fullScreenPass(psBlit, imgSRV, backRTV,
                       (float)((viewW - dw) / 2), (float)((viewH - dh) / 2),
                       (float)dw, (float)dh, true);
    } else {
        const float black[4] = {0, 0, 0, 1};
        ctx->ClearRenderTargetView(backRTV, black);
    }
    // Present(0): never block the render thread on vblank. DWM composites, so this does
    // not tear; at worst a frame is superseded before it is shown.
    HRESULT hr = swap->Present(0, 0);
    if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) { ok = false; return false; }
    return true;
}

// Pull the current image back to host BGRA for a GDI capture (WM_PRINTCLIENT). Rare —
// PrintWindow cannot see swap-chain content, so this is how off-screen grabs of the
// preview keep working now that the image never touches host memory in BGRA form.
bool LivePresenter::readbackBgra(std::vector<uint8_t>& bgra, int& w, int& h) {
    if (!ok || !imgTex || imgW <= 0 || imgH <= 0) return false;
    if (!readTex) {
        D3D11_TEXTURE2D_DESC td{};
        imgTex->GetDesc(&td);
        td.Usage = D3D11_USAGE_STAGING;
        td.BindFlags = 0;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        td.MiscFlags = 0;
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &readTex))) return false;
    }
    ctx->CopyResource(readTex, imgTex);
    D3D11_MAPPED_SUBRESOURCE ms{};
    if (FAILED(ctx->Map(readTex, 0, D3D11_MAP_READ, 0, &ms))) return false;
    bgra.resize((size_t)imgW * imgH * 4);
    for (int y = 0; y < imgH; ++y) {
        const uint8_t* src = (const uint8_t*)ms.pData + (size_t)y * ms.RowPitch;
        uint8_t*       dst = bgra.data() + (size_t)y * imgW * 4;
        for (int x = 0; x < imgW; ++x) {
            dst[x * 4 + 0] = src[x * 4 + 2];   // B
            dst[x * 4 + 1] = src[x * 4 + 1];   // G
            dst[x * 4 + 2] = src[x * 4 + 0];   // R
            dst[x * 4 + 3] = 255;
        }
    }
    ctx->Unmap(readTex, 0);
    w = imgW; h = imgH;
    return true;
}

// ---- Control-panel constants ----
// Child-window command IDs (WM_COMMAND LOWORD) + the trackbar. Kept out of the low
// range Windows reserves for standard dialog buttons.
enum {
    ID_CLIP = 1001, ID_RESET, ID_PATH, ID_PLAY,
    ID_TIMELINE, ID_STRIDE, ID_RATE, ID_SW_UPDATE, ID_SW_SEC,
    // ---- curve-editor row ----
    ID_REC, ID_ADDPT, ID_INSPT, ID_DELPT, ID_SAVE, ID_TOL, ID_RAW,
    // ---- paint-mode controls (speed + orientation painting) ----
    ID_PAINT, ID_FLAT,
    // ---- loom bind row (drive channel -> named scene variable) ----
    ID_BCH, ID_BSLOT, ID_BIND, ID_BCLEAR, ID_BDIMS
};
static const int kRowH   = 28;              // one panel row: button height (24) + 4px gap
static const int kPanelH = 92;              // control-strip height (px) WITHOUT the bind row: buttons + timeline + editor rows
// Marshal cross-thread panel ops onto the window's own message-pump thread.
#define WM_MKPANEL      (WM_APP + 1)        // build the control panel (params staged in Impl)
#define WM_SETPATHCOUNT (WM_APP + 2)        // retune the timeline range/visibility (wParam = new count)
#define WM_MKBINDROW    (WM_APP + 3)        // build + reveal the loom bind row (params staged in Impl)

struct LiveWindow::Impl {
    std::thread          ui;
    std::mutex           mtx;          // guards bgra / imgW / imgH
    std::vector<uint8_t> bgra;         // imgW*imgH*4, top-down BGRA (GDI DIB order)
    int                  imgW = 0, imgH = 0;
    std::atomic<bool>    dirty{false};
    std::atomic<bool>    closedFlag{false};
    // Window handle: created on the UI thread, nulled on WM_DESTROY. Atomic + null-on-destroy
    // so cross-thread callers (setTitle/clientSize/enablePanel/setPathCount and the dtor) never
    // marshal to a STALE handle after the window closes — Windows recycles HWND values, so a
    // since-reused handle belonging to another window/thread must never receive our WM_CLOSE/
    // WM_SETTEXT. Readers load() once and null-check before use.
    std::atomic<HWND>    hwnd{nullptr};
    // ---- D3D11 presenter (the image area only; the control strip stays GDI) ----
    // The swap chain lives on its OWN child window covering the image area, because a
    // flip-model swap chain and overlapping GDI child controls cannot share one HWND.
    // The child is hit-test transparent (HTTRANSPARENT), so every mouse message still
    // reaches the parent's fly-camera handlers with unchanged client coordinates.
    HWND                 hview = nullptr;          // created/destroyed on the UI thread
    LivePresenter        pres;
    std::atomic<bool>    d3dOk{false};             // presenter live => GDI image path is skipped
    std::atomic<int>     viewW{0}, viewH{0};       // child client size (UI thread writes, presenter reads)
    int                  initW = 0, initH = 0;
    int                  minW = 640, minH = 300;   // readable floor so the title bar stays legible
    std::wstring         title;
    HANDLE               readyEvent = nullptr;
    // ---- Fly-camera input state (guarded by inMtx unless noted) ----
    std::mutex           inMtx;                     // guards the look/wheel accumulators + one-shots
    double               lookX = 0.0, lookY = 0.0;  // hover-look turn RATE: cursor offset from centre, dead-zoned, -1..+1
    double               wheelAcc = 0.0;            // plain wheel notches since drain (dolly move)
    double               wheelSpeedAcc = 0.0;       // Ctrl+wheel notches since drain (step-size adjust)
    bool                 resetReq = false;          // '0' / Home pressed since last drain (one-shot)
    bool                 printReq = false;          // 'P' pressed since last drain (one-shot)
    bool                 collideReq = false;        // 'C' pressed since last drain (one-shot)
    bool                 traceReq = false;          // 'T' pressed since last drain (one-shot)
    // Held-key throttle state — atomics so WM_KEYUP on the UI thread and drainNav on the
    // render thread can race freely without the inMtx.
    std::atomic<bool>    keyFwd{false};             // Space / '+' currently held -> fly forward
    std::atomic<bool>    keyBack{false};            // Shift / '-' currently held -> fly backward
    // Mouse-look is HOVER-look with RATE (joystick) steering: while the cursor is over the client
    // area, its offset from the window centre sets a TURN RATE (dead-zoned near centre so the view
    // can rest). The cursor stays VISIBLE and free — we never hide, clip, or capture it — and
    // steering stops the moment the pointer leaves the window. `looking` tracks whether the cursor
    // is inside; `tracking` is whether we've armed WM_MOUSELEAVE for the current hover.
    std::atomic<bool>    looking{false};            // cursor currently inside client (steering live)
    bool                 tracking  = false;         // WM_MOUSELEAVE requested for this hover

    // ---- Control panel (optional strip below the image) ----
    std::atomic<bool>    hasPanel{false};           // panel built & child HWNDs valid (release/acquire)
    int                  panelH = 0;                // reserved strip height (0 = no panel); UI thread
    int                  pathCount = 0;             // cameras on the timeline (0 = no path controls)
    HWND hClip=nullptr, hReset=nullptr, hPath=nullptr, hPlay=nullptr, hTimeline=nullptr,
         hStrideLbl=nullptr, hStride=nullptr, hRateLbl=nullptr, hRate=nullptr,
         hSwUpdate=nullptr, hSwSec=nullptr;         // child controls (set on UI thread pre-hasPanel)
    // ---- curve-editor row controls ----
    HWND hRec=nullptr, hAddPt=nullptr, hInsPt=nullptr, hDelPt=nullptr, hSave=nullptr,
         hPtLbl=nullptr, hRaw=nullptr, hTolLbl=nullptr, hTol=nullptr;
    // ---- paint-mode controls (on the timeline row, shown with the path group) ----
    HWND hPaint=nullptr, hFlat=nullptr, hSpdLbl=nullptr;
    // ---- loom bind row (row 4; built on demand by enableBindRow, absent otherwise) ----
    HWND hBLbl=nullptr, hBCh=nullptr, hBArrow=nullptr, hBSlot=nullptr, hBind=nullptr,
         hBClear=nullptr, hBDimsLbl=nullptr, hBDims=nullptr, hBStat=nullptr;
    std::atomic<bool>    hasBindRow{false};         // bind-row child HWNDs valid (release/acquire)
    // Channel count the channel combo is currently populated for. Atomic because setChannelCombo
    // is reached from the UI thread (buildBindRow) AND the render thread (setBindState, when the
    // drive's channel count changes) — the combo messages themselves marshal, but this bookkeeping
    // int would otherwise be a plain cross-thread read/write.
    std::atomic<int>     bindDims{0};
    HFONT panelFont = nullptr;
    // Staged enablePanel() params (set under inMtx before WM_MKPANEL is sent).
    int                  reqPathCount = 0; double reqDefFps = 0.0; std::string reqCollide;
    // Panel outputs (guarded by inMtx): one-shot button edges + current input values.
    bool                 pathReq = false;           // "Path" toggle pressed (one-shot)
    bool                 playReq = false;           // "Play/Pause" pressed (one-shot)
    int                  scrubReq = -1;             // timeline dragged/jumped to index (>=0), else -1
    int                  strideVal = 1;             // "cameras / screen update" input (current)
    double               rateVal   = 30.0;          // "cameras / second" input (current)
    bool                 rateModeVal = true;        // switch: true = per-sec, false = per-update
    // ---- Curve-editor outputs (guarded by inMtx): one-shot button edges + current inputs ----
    bool                 recReq = false, addReq = false, insReq = false, delReq = false, saveReq = false;
    double               tolVal = -1.0;             // simplify tolerance (world units; <0 = unset/unchanged)
    bool                 rawVal = false;            // "raw" checkbox: keep every sample vs. simplify
    bool                 paintVal = false;          // "Paint" checkbox: speed/orientation painting on (persistent)
    bool                 flatReq = false;           // "Flat" button: reset painted speed (one-shot)
    // ---- Bind-row outputs (guarded by inMtx) ----
    bool                 bindReq = false, bclearReq = false;   // Bind / Unbind edges (one-shot)
    int                  bindDimsVal = 0;           // channel-count box (current value; 0 = absent/unchanged)
    // Combo selections are CACHED on CBN_SELCHANGE rather than read at drain time: drainNav runs
    // on the RENDER thread, and CB_GETCURSEL would be a blocking SendMessage into the UI thread
    // on every single frame just to learn something that changes only when the user clicks.
    int                  bindChVal = -1;            // channel combo selection (>=0), else -1
    std::string          bindSlotVal;               // slot combo selection ("" = the "(none)" entry)
    // Staged enableBindRow() params (set under inMtx before WM_MKBINDROW is sent).
    std::vector<std::string> reqSlots; int reqDims = 0;

    static LRESULT CALLBACK WndProc(HWND h, UINT msg, WPARAM wp, LPARAM lp);
    static LRESULT CALLBACK ViewProc(HWND h, UINT msg, WPARAM wp, LPARAM lp);
    void threadMain();
    void makeView(HWND parent);                     // create the image child + init D3D (UI thread)
    void layoutView(HWND parent);                   // fit the child to the image area (UI thread)
    void presentNow();                              // re-present the last frame (any thread)
    void paint(HDC hdc, const RECT& client, bool forCapture);
    void endLook();                                 // cursor left / focus lost: stop steering cleanly
    void buildPanel(HWND h);                        // create child controls + grow window (UI thread)
    void layoutPanel(HWND h);                       // position child controls in the strip (UI thread)
    void applyPathCount(int pc);                    // retune/show/hide the timeline group (UI thread)
    void showPathGroup(bool vis);                   // toggle visibility of the path (timeline) controls
    void buildBindRow(HWND h);                      // create the loom bind row + grow window (UI thread)
    void setChannelCombo(int dims);                 // repopulate the channel combo for `dims` (UI thread)
};

// Stop hover-look steering: called when the cursor leaves the client area or the window loses
// focus. Zeroes the turn rate (so the view stops) and clears the "inside"/tracking flags. The
// cursor is never hidden/clipped, so there is nothing to restore.
void LiveWindow::Impl::endLook() {
    looking.store(false);
    tracking = false;
    std::lock_guard<std::mutex> lk(inMtx);
    lookX = lookY = 0.0;
}

// Build the control-panel child windows and grow the window by kPanelH so the image area is
// unchanged. Runs on the UI thread (via WM_MKPANEL). Reads the staged reqPathCount/reqDefFps/
// reqCollide. Sets hasPanel=true LAST, after every HWND is valid, so the render thread's
// setPanelState/paint see a fully-formed panel.
void LiveWindow::Impl::buildPanel(HWND h) {
    if (hasPanel.load()) return;
    INITCOMMONCONTROLSEX icc{ sizeof(icc), ICC_BAR_CLASSES };
    InitCommonControlsEx(&icc);                     // register msctls_trackbar32
    int pc; double defFps; std::string collide;
    { std::lock_guard<std::mutex> lk(inMtx); pc = reqPathCount; defFps = reqDefFps; collide = reqCollide; }
    pathCount = pc;
    panelH    = kPanelH;
    panelFont = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
    HINSTANCE hi = (HINSTANCE)GetWindowLongPtrW(h, GWLP_HINSTANCE);
    auto mk = [&](const wchar_t* cls, const wchar_t* txt, DWORD style, int id) -> HWND {
        HWND c = CreateWindowExW(0, cls, txt, WS_CHILD | WS_VISIBLE | style,
                                 0, 0, 10, 10, h, (HMENU)(INT_PTR)id, hi, nullptr);
        if (c) SendMessageW(c, WM_SETFONT, (WPARAM)panelFont, TRUE);
        return c;
    };
    std::wstring clipTxt = utf8ToWide("Clip: " + collide);
    hClip  = mk(L"BUTTON", clipTxt.c_str(), BS_PUSHBUTTON, ID_CLIP);
    hReset = mk(L"BUTTON", L"Reset", BS_PUSHBUTTON, ID_RESET);
    // Path/timeline group is ALWAYS created (so authoring a curve can reveal it later via
    // setPathCount), then hidden when there is no path yet (pathCount < 2).
    // Path (lock-to-path) toggle doubles as the timeline enable — same option, per spec.
    hPath = mk(L"BUTTON", L"Path lock", BS_AUTOCHECKBOX | BS_PUSHLIKE, ID_PATH);
    hPlay = mk(L"BUTTON", L"Play", BS_PUSHBUTTON, ID_PLAY);
    hStrideLbl = mk(L"STATIC", L"cams/upd:", SS_RIGHT | SS_CENTERIMAGE, 0);
    hStride    = mk(L"EDIT", L"1", ES_NUMBER | ES_RIGHT | WS_BORDER, ID_STRIDE);
    hRateLbl   = mk(L"STATIC", L"cams/s:", SS_RIGHT | SS_CENTERIMAGE, 0);
    wchar_t rbuf[32]; swprintf(rbuf, 32, L"%g", (defFps > 0.0 ? defFps : 30.0));
    hRate      = mk(L"EDIT", rbuf, ES_RIGHT | WS_BORDER, ID_RATE);
    // Speed-model switch: two radio buttons; default = per-sec (real-time playback at fps).
    hSwUpdate  = mk(L"BUTTON", L"per upd", BS_AUTORADIOBUTTON | WS_GROUP, ID_SW_UPDATE);
    hSwSec     = mk(L"BUTTON", L"per sec", BS_AUTORADIOBUTTON, ID_SW_SEC);
    SendMessageW(hSwSec, BM_SETCHECK, BST_CHECKED, 0);
    // Timeline trackbar: one tick per camera (page = ~5%).
    hTimeline  = mk(L"msctls_trackbar32", L"", TBS_HORZ | TBS_AUTOTICKS, ID_TIMELINE);
    int rng = std::max(1, pathCount - 1);
    SendMessageW(hTimeline, TBM_SETRANGE, TRUE, MAKELPARAM(0, rng));
    SendMessageW(hTimeline, TBM_SETPAGESIZE, 0, (LPARAM)std::max(1, pathCount / 20));
    SendMessageW(hTimeline, TBM_SETPOS, TRUE, 0);
    // Paint-mode controls (live on the timeline row, right of the trackbar): a Paint toggle
    // (wheel paints speed / mouse paints orientation along the path), a Flat reset, and a
    // local-speed readout. Shown/hidden with the rest of the path group.
    hPaint  = mk(L"BUTTON", L"Paint", BS_AUTOCHECKBOX | BS_PUSHLIKE, ID_PAINT);
    hFlat   = mk(L"BUTTON", L"Flat",  BS_PUSHBUTTON, ID_FLAT);
    hSpdLbl = mk(L"STATIC", L"1.00x", SS_CENTER | SS_CENTERIMAGE, 0);
    showPathGroup(pathCount >= 2);
    // ---- Curve-editor row: author/record a camera_curve, then Save it ----
    hRec   = mk(L"BUTTON", L"Rec",   BS_PUSHBUTTON, ID_REC);
    hAddPt = mk(L"BUTTON", L"+Pt",   BS_PUSHBUTTON, ID_ADDPT);
    hInsPt = mk(L"BUTTON", L"Ins",   BS_PUSHBUTTON, ID_INSPT);
    hDelPt = mk(L"BUTTON", L"Del",   BS_PUSHBUTTON, ID_DELPT);
    hPtLbl = mk(L"STATIC", L"pts: 0", SS_LEFT | SS_CENTERIMAGE, 0);
    hRaw   = mk(L"BUTTON", L"raw",   BS_AUTOCHECKBOX, ID_RAW);
    hTolLbl= mk(L"STATIC", L"tol:",  SS_RIGHT | SS_CENTERIMAGE, 0);
    hTol   = mk(L"EDIT",   L"0",     ES_RIGHT | WS_BORDER, ID_TOL);
    hSave  = mk(L"BUTTON", L"Save",  BS_PUSHBUTTON, ID_SAVE);
    // Grow the window by the strip height so the image keeps its size.
    RECT wr; GetWindowRect(h, &wr);
    SetWindowPos(h, nullptr, 0, 0, wr.right - wr.left, (wr.bottom - wr.top) + panelH,
                 SWP_NOMOVE | SWP_NOZORDER);
    layoutPanel(h);
    hasPanel.store(true);                           // publish: children are all valid now
    InvalidateRect(h, nullptr, TRUE);
}

// Populate the channel combo with 0..dims-1, preserving the current selection where it still
// exists (shrinking the drive can delete the selected channel; then fall back to the last one).
// UI thread only.
void LiveWindow::Impl::setChannelCombo(int dims) {
    if (!hBCh || dims <= 0 || dims == bindDims.load()) return;
    int sel = (int)SendMessageW(hBCh, CB_GETCURSEL, 0, 0);
    SendMessageW(hBCh, CB_RESETCONTENT, 0, 0);
    for (int c = 0; c < dims; ++c) {
        wchar_t b[16]; swprintf(b, 16, L"ch %d", c);
        SendMessageW(hBCh, CB_ADDSTRING, 0, (LPARAM)b);
    }
    // CB_SETCURSEL does NOT raise CBN_SELCHANGE (only a user pick does), so the cached selection
    // has to be updated by hand — otherwise a shrunk drive would leave `bindChVal` pointing at a
    // channel that no longer exists and Bind would target it.
    int keep = std::min(std::max(sel, 0), dims - 1);
    SendMessageW(hBCh, CB_SETCURSEL, (WPARAM)keep, 0);
    { std::lock_guard<std::mutex> lk(inMtx); bindChVal = keep; }
    bindDims.store(dims);
}

// Build the loom bind row (row 4) and grow the window by one row so the image keeps its size.
// Runs on the UI thread (via WM_MKBINDROW) and reads the staged reqSlots/reqDims. Sets
// hasBindRow=true LAST, after every HWND is valid, so setBindState sees a fully-formed row.
void LiveWindow::Impl::buildBindRow(HWND h) {
    if (!hasPanel.load() || hasBindRow.load()) return;
    std::vector<std::string> slots; int dims;
    { std::lock_guard<std::mutex> lk(inMtx); slots = reqSlots; dims = reqDims; }
    HINSTANCE hi = (HINSTANCE)GetWindowLongPtrW(h, GWLP_HINSTANCE);
    auto mk = [&](const wchar_t* cls, const wchar_t* txt, DWORD style, int id) -> HWND {
        HWND c = CreateWindowExW(0, cls, txt, WS_CHILD | WS_VISIBLE | style,
                                 0, 0, 10, 10, h, (HMENU)(INT_PTR)id, hi, nullptr);
        if (c) SendMessageW(c, WM_SETFONT, (WPARAM)panelFont, TRUE);
        return c;
    };
    hBLbl  = mk(L"STATIC", L"loom:", SS_LEFT | SS_CENTERIMAGE, 0);
    // The combos are DROPDOWNLIST (pick-only): a channel index and a slot name must both be
    // things the drive/scene actually has, so free text would only invite typos loom rejects.
    // Their creation height is the DROPPED height, not the closed one — hence the tall value.
    hBCh   = mk(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL, ID_BCH);
    hBArrow= mk(L"STATIC", L"\u2192", SS_CENTER | SS_CENTERIMAGE, 0);
    hBSlot = mk(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL, ID_BSLOT);
    hBind  = mk(L"BUTTON", L"Bind",   BS_PUSHBUTTON, ID_BIND);
    hBClear= mk(L"BUTTON", L"Unbind", BS_PUSHBUTTON, ID_BCLEAR);
    hBDimsLbl = mk(L"STATIC", L"chans:", SS_RIGHT | SS_CENTERIMAGE, 0);
    wchar_t db[16]; swprintf(db, 16, L"%d", dims);
    hBDims = mk(L"EDIT", db, ES_NUMBER | ES_RIGHT | WS_BORDER, ID_BDIMS);
    hBStat = mk(L"STATIC", L"", SS_LEFT | SS_CENTERIMAGE | SS_ENDELLIPSIS, 0);
    // Slot pick-list: "(none)" first, so Bind can also express "this channel drives nothing"
    // without reaching for Unbind.
    SendMessageW(hBSlot, CB_ADDSTRING, 0, (LPARAM)L"(none)");
    for (const std::string& s : slots) {
        std::wstring w = utf8ToWide(s);
        SendMessageW(hBSlot, CB_ADDSTRING, 0, (LPARAM)w.c_str());
    }
    SendMessageW(hBSlot, CB_SETCURSEL, 0, 0);
    setChannelCombo(dims);
    { std::lock_guard<std::mutex> lk(inMtx); bindDimsVal = dims; }
    // Grow by one row so the image area is unchanged (same contract as buildPanel).
    panelH += kRowH;
    // Publish BEFORE laying out: layoutPanel skips row 4 unless hasBindRow is set, so storing it
    // afterwards (the order buildPanel uses for hasPanel) would leave every bind-row control
    // parked at its 10x10 creation rect in the window's top-left corner. Safe here for the same
    // reason it is safe there — every child HWND above is already valid; only the *positions*
    // are still pending, and nothing outside this thread can observe them before we return.
    hasBindRow.store(true);
    RECT wr; GetWindowRect(h, &wr);
    SetWindowPos(h, nullptr, 0, 0, wr.right - wr.left, (wr.bottom - wr.top) + kRowH,
                 SWP_NOMOVE | SWP_NOZORDER);
    layoutPanel(h);
    InvalidateRect(h, nullptr, TRUE);
}

// Position the panel children within the bottom strip. Row 1 = buttons + speed inputs + switch;
// row 2 = the full-width timeline. Called on build and on every WM_SIZE. UI thread only.
void LiveWindow::Impl::layoutPanel(HWND h) {
    if (!panelH) return;
    RECT cr; GetClientRect(h, &cr);
    int W = cr.right - cr.left, H = cr.bottom - cr.top;
    int top = H - panelH;
    const int pad = 5, bh = 24;
    int row1 = top + 5, row2 = top + 5 + (bh + 4), row3 = top + 5 + 2 * (bh + 4);
    int x = pad;
    auto place = [&](HWND c, int w, int y, int height) {
        if (c) MoveWindow(c, x, y, w, height, TRUE);
        x += w + pad;
    };
    // Row 1: collision/reset + the path (timeline) group. The group is positioned even when
    // hidden, so revealing it later (setPathCount) needs no relayout.
    place(hClip, 84, row1, bh);
    place(hReset, 56, row1, bh);
    place(hPath, 66, row1, bh);
    place(hPlay, 56, row1, bh);
    place(hStrideLbl, 58, row1, bh);
    place(hStride, 40, row1, bh);
    place(hRateLbl, 50, row1, bh);
    place(hRate, 48, row1, bh);
    place(hSwUpdate, 66, row1, bh);
    place(hSwSec, 62, row1, bh);
    // Row 2: the timeline, with the paint controls docked at the right end.
    const int paintW = 52, flatW = 44, spdW = 52;
    int rightBlock = paintW + flatW + spdW + 3 * pad;   // reserved on the right for paint tools
    int tlW = std::max(1, W - 2 * pad - rightBlock);
    if (hTimeline) MoveWindow(hTimeline, pad, row2, tlW, bh, TRUE);
    x = pad + tlW + pad;
    place(hPaint, paintW, row2, bh);
    place(hFlat,  flatW,  row2, bh);
    place(hSpdLbl, spdW,  row2, bh);
    // Row 3: the curve-editor toolset.
    x = pad;
    place(hRec,   56, row3, bh);
    place(hAddPt, 48, row3, bh);
    place(hInsPt, 48, row3, bh);
    place(hDelPt, 48, row3, bh);
    place(hPtLbl, 60, row3, bh);
    place(hRaw,   52, row3, bh);
    place(hTolLbl,34, row3, bh);
    place(hTol,   56, row3, bh);
    place(hSave,  56, row3, bh);
    // Row 4 (only when the loom live channel exists): channel -> slot binding + a status line.
    if (hasBindRow.load()) {
        int row4 = top + 5 + 3 * (bh + 4);
        x = pad;
        place(hBLbl,    38, row4, bh);
        // A combo's height is its DROPPED height; the closed box is one line tall regardless.
        place(hBCh,     70, row4, bh + 120);
        place(hBArrow,  16, row4, bh);
        place(hBSlot,  130, row4, bh + 120);
        place(hBind,    50, row4, bh);
        place(hBClear,  62, row4, bh);
        place(hBDimsLbl,44, row4, bh);
        place(hBDims,   40, row4, bh);
        // The status readout takes whatever width is left (it is SS_ENDELLIPSIS, so a long
        // loom error truncates cleanly instead of overrunning the strip).
        if (hBStat) MoveWindow(hBStat, x, row4, std::max(40, W - pad - x), bh, TRUE);
    }
}

// Show/hide the path (timeline) controls as a group — the timeline only makes sense once a
// curve with >= 2 cameras exists (loaded or authored). Called on build and from applyPathCount.
void LiveWindow::Impl::showPathGroup(bool vis) {
    int sw = vis ? SW_SHOW : SW_HIDE;
    HWND grp[] = { hPath, hPlay, hStrideLbl, hStride, hRateLbl, hRate, hSwUpdate, hSwSec, hTimeline,
                   hPaint, hFlat, hSpdLbl };
    for (HWND c : grp) if (c) ShowWindow(c, sw);
}

// Retune the timeline to a new camera count and show/hide the path group accordingly. Runs on
// the UI thread (via WM_SETPATHCOUNT) so the trackbar messages and ShowWindow are thread-safe.
void LiveWindow::Impl::applyPathCount(int pc) {
    pathCount = pc;
    if (hTimeline) {
        int rng = std::max(1, pc - 1);
        SendMessageW(hTimeline, TBM_SETRANGE, TRUE, MAKELPARAM(0, rng));
        SendMessageW(hTimeline, TBM_SETPAGESIZE, 0, (LPARAM)std::max(1, pc / 20));
    }
    showPathGroup(pc >= 2);
}

// ---- The image child window -----------------------------------------------------------
// Nothing but a surface for the swap chain. It answers WM_NCHITTEST with HTTRANSPARENT so
// the hit test falls through to the parent (same thread), which keeps all of the fly-camera
// mouse handling — hover-look, wheel dolly, the dead zone measured from the image centre —
// working on unchanged coordinates: the child sits at the parent client origin, so the two
// coordinate spaces coincide over the image area.
LRESULT CALLBACK LiveWindow::Impl::ViewProc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_CREATE) {
        auto cs = reinterpret_cast<CREATESTRUCTW*>(lp);
        SetWindowLongPtrW(h, GWLP_USERDATA, (LONG_PTR)cs->lpCreateParams);
        return 0;
    }
    auto self = reinterpret_cast<Impl*>(GetWindowLongPtrW(h, GWLP_USERDATA));
    switch (msg) {
        case WM_NCHITTEST:  return HTTRANSPARENT;    // mouse belongs to the parent
        case WM_ERASEBKGND: return 1;                // the swap chain owns every pixel
        case WM_PAINT: {
            PAINTSTRUCT ps; BeginPaint(h, &ps); EndPaint(h, &ps);
            // A flip-model swap chain keeps showing its last frame through occlusion and
            // moves, so an expose costs nothing — but after a RESIZE the buffers must be
            // rebuilt, and the render thread may be seconds away from its next frame.
            if (self) self->presentNow();
            return 0;
        }
        case WM_SIZE:
            if (self) { self->viewW.store(LOWORD(lp)); self->viewH.store(HIWORD(lp)); }
            return 0;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

// Create the image child and bring up D3D on it. Runs on the UI thread, before the ctor
// is unblocked, so the render thread cannot race the first frame against initialisation.
// Any failure simply leaves d3dOk false and the original GDI path in charge.
void LiveWindow::Impl::makeView(HWND parent) {
    // Escape hatch: FTRACE_LIVE_GDI=1 forces the original CPU/GDI present path. Useful for
    // A/B measurement, and as a way out if a driver/remote-session ever makes the swap
    // chain misbehave on a machine where the GDI path still works.
    if (const char* e = std::getenv("FTRACE_LIVE_GDI")) { if (*e && *e != '0') return; }
    WNDCLASSEXW wc{};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = ViewProc;
    wc.hInstance     = GetModuleHandleW(nullptr);
    wc.hCursor       = LoadCursorW(nullptr, (LPCWSTR)IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.lpszClassName = L"FtraceLiveView";
    RegisterClassExW(&wc);
    RECT cr; GetClientRect(parent, &cr);
    int cw = std::max(1, (int)(cr.right - cr.left));
    int ch = std::max(1, (int)(cr.bottom - cr.top) - panelH);
    hview = CreateWindowExW(0, wc.lpszClassName, L"", WS_CHILD | WS_VISIBLE,
                            0, 0, cw, ch, parent, nullptr, wc.hInstance, this);
    if (!hview) return;
    viewW.store(cw); viewH.store(ch);
    if (pres.init(hview)) {
        d3dOk.store(true);
    } else {
        DestroyWindow(hview);                       // fall all the way back to GDI
        hview = nullptr;
    }
}

// Keep the child exactly over the image area (client minus the control strip).
void LiveWindow::Impl::layoutView(HWND parent) {
    if (!hview) return;
    RECT cr; GetClientRect(parent, &cr);
    int cw = std::max(1, (int)(cr.right - cr.left));
    int ch = std::max(1, (int)(cr.bottom - cr.top) - panelH);
    MoveWindow(hview, 0, 0, cw, ch, TRUE);
}

// Re-draw the last uploaded frame. Safe from either thread: LivePresenter serialises the
// device context internally, and this never sends a message, so it cannot deadlock against
// the UI thread.
void LiveWindow::Impl::presentNow() {
    if (!d3dOk.load()) return;
    std::lock_guard<std::mutex> lk(pres.mtx);
    if (pres.ok) pres.present(viewW.load(), viewH.load());
}

// `forCapture` distinguishes the two remaining GDI callers. On screen, when the presenter
// is live, the image area belongs to the child window and the parent must not draw over it
// (it is clipped out by WS_CLIPCHILDREN anyway) — only the control strip is painted here.
// A capture (WM_PRINTCLIENT / PrintWindow) is different: it renders into someone else's DC,
// which the swap chain is invisible to, so the full image has to be drawn the old way —
// pulling the pixels back off the GPU first if that is where they live.
void LiveWindow::Impl::paint(HDC hdc, const RECT& client, bool forCapture) {
    int cw = client.right - client.left;
    int ch = (client.bottom - client.top) - panelH;   // image area = client minus the control strip
    if (!forCapture && d3dOk.load()) {
        if (panelH > 0 && cw > 0) {
            RECT strip{0, std::max(0, ch), cw, client.bottom - client.top};
            FillRect(hdc, &strip, GetSysColorBrush(COLOR_BTNFACE));
        }
        return;
    }
    if (forCapture && d3dOk.load()) {
        // Refresh the host mirror from the GPU so the DIB below has something current.
        std::vector<uint8_t> px; int rw = 0, rh = 0;
        bool got = false;
        { std::lock_guard<std::mutex> lk(pres.mtx); got = pres.readbackBgra(px, rw, rh); }
        if (got) {
            std::lock_guard<std::mutex> lk(mtx);
            bgra.swap(px); imgW = rw; imgH = rh;
        }
    }
    if (cw <= 0 || ch <= 0) {
        // Degenerate (window dragged shorter than the panel): just fill with the toolbar face.
        if (panelH > 0) {
            RECT strip{0, 0, cw, client.bottom - client.top};
            FillRect(hdc, &strip, GetSysColorBrush(COLOR_BTNFACE));
        }
        return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    // Double-buffer the IMAGE area through a memory DC so the letterbox fill + stretch blit
    // land in one BitBlt (no flicker; WM_ERASEBKGND is suppressed).
    HDC     mem = CreateCompatibleDC(hdc);
    HBITMAP bmp = CreateCompatibleBitmap(hdc, cw, ch);
    HBITMAP old = (HBITMAP)SelectObject(mem, bmp);
    RECT full{0, 0, cw, ch};
    FillRect(mem, &full, (HBRUSH)GetStockObject(BLACK_BRUSH));
    if (imgW > 0 && imgH > 0 && !bgra.empty()) {
        double s  = std::min((double)cw / imgW, (double)ch / imgH);   // aspect fit
        int    dw = std::max(1, (int)(imgW * s)), dh = std::max(1, (int)(imgH * s));
        int    dx = (cw - dw) / 2, dy = (ch - dh) / 2;
        BITMAPINFO bi{};
        bi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
        bi.bmiHeader.biWidth       = imgW;
        bi.bmiHeader.biHeight      = -imgH;        // negative => top-down rows
        bi.bmiHeader.biPlanes      = 1;
        bi.bmiHeader.biBitCount    = 32;           // BGRA: scanlines are DWORD-aligned
        bi.bmiHeader.biCompression = BI_RGB;
        SetStretchBltMode(mem, HALFTONE);
        SetBrushOrgEx(mem, 0, 0, nullptr);
        StretchDIBits(mem, dx, dy, dw, dh, 0, 0, imgW, imgH,
                      bgra.data(), &bi, DIB_RGB_COLORS, SRCCOPY);
    }
    BitBlt(hdc, 0, 0, cw, ch, mem, 0, 0, SRCCOPY);
    SelectObject(mem, old);
    DeleteObject(bmp);
    DeleteDC(mem);
    // Fill the control-strip background (behind/between the child controls) with the toolbar face.
    if (panelH > 0) {
        RECT strip{0, ch, cw, ch + panelH};
        FillRect(hdc, &strip, GetSysColorBrush(COLOR_BTNFACE));
    }
}

LRESULT CALLBACK LiveWindow::Impl::WndProc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_CREATE) {
        auto cs = reinterpret_cast<CREATESTRUCTW*>(lp);
        SetWindowLongPtrW(h, GWLP_USERDATA, (LONG_PTR)cs->lpCreateParams);
        return 0;
    }
    auto self = reinterpret_cast<Impl*>(GetWindowLongPtrW(h, GWLP_USERDATA));
    switch (msg) {
        case WM_TIMER:
            if (self && self->dirty.exchange(false)) InvalidateRect(h, nullptr, FALSE);
            return 0;
        case WM_ERASEBKGND:
            return 1;                               // painted fully in WM_PAINT
        case WM_PAINT: {
            PAINTSTRUCT ps; HDC hdc = BeginPaint(h, &ps);
            RECT cr; GetClientRect(h, &cr);
            if (self) self->paint(hdc, cr, false);
            EndPaint(h, &ps);
            return 0;
        }
        case WM_PRINTCLIENT: {
            // Render the current frame into the caller's DC so PrintWindow() captures the
            // live image even when the window is occluded (used for off-screen grabs).
            // PrintWindow cannot see swap-chain content, so this always takes the GDI path.
            if (self) { RECT cr; GetClientRect(h, &cr); self->paint((HDC)wp, cr, true); }
            return 0;
        }
        case WM_MKPANEL:
            if (self) self->buildPanel(h);           // build controls + grow window (UI thread)
            return 0;
        case WM_SETPATHCOUNT:
            if (self && self->hasPanel.load()) { self->applyPathCount((int)wp); InvalidateRect(h, nullptr, FALSE); }
            return 0;
        case WM_MKBINDROW:
            if (self) self->buildBindRow(h);          // build the loom bind row + grow window (UI thread)
            return 0;
        case WM_SIZE:
            if (self) {
                self->layoutPanel(h);                // reflow the control strip to the new width
                self->layoutView(h);                 // and keep the D3D child over the image area
            }
            InvalidateRect(h, nullptr, FALSE);
            return 0;
        case WM_MOUSEMOVE:
            // Hover-look (rate / joystick): while the cursor is over the IMAGE area, its offset
            // from the image centre sets a TURN RATE. A central dead zone reports zero (the view
            // holds still so you can see the scene); beyond it the rate ramps to ±1 at the image
            // edge, so holding the pointer to one side keeps the view turning that way — you can
            // look a full circle without the cursor leaving the window. The cursor stays visible
            // and free (no hide/clip/warp). We arm WM_MOUSELEAVE so we know when it exits. Over the
            // control strip (below the image) steering is neutral, so reaching a button doesn't turn.
            if (self) {
                RECT cr; GetClientRect(h, &cr);
                int imgH = (cr.bottom - cr.top) - self->panelH;   // image area excludes the strip
                double halfW = (cr.right - cr.left) * 0.5, halfH = imgH * 0.5;
                int mx = GET_X_LPARAM(lp), my = GET_Y_LPARAM(lp);
                bool inImage = (my < imgH);
                double nx = (inImage && halfW > 0.5) ? (mx - halfW) / halfW : 0.0;   // -1 (left) .. +1 (right)
                double ny = (inImage && halfH > 0.5) ? (my - halfH) / halfH : 0.0;   // -1 (top)  .. +1 (bottom)
                const double dz = 0.15;                                 // central neutral dead zone
                auto shape = [dz](double v) -> double {                 // dead-zone + rescale to full-edge = ±1
                    double a = v < 0 ? -v : v;
                    if (a <= dz) return 0.0;
                    double t = (a - dz) / (1.0 - dz);
                    if (t > 1.0) t = 1.0;
                    return v < 0 ? -t : t;
                };
                if (!self->tracking) {
                    TRACKMOUSEEVENT tme{ sizeof(tme), TME_LEAVE, h, 0 };
                    TrackMouseEvent(&tme);
                    self->tracking = true;
                    self->looking.store(true);
                }
                std::lock_guard<std::mutex> lk(self->inMtx);
                self->lookX = shape(nx);
                self->lookY = shape(ny);
            }
            return 0;
        case WM_MOUSELEAVE:
            // Cursor left the client area: stop steering until it comes back.
            if (self) self->endLook();
            return 0;
        case WM_MOUSEWHEEL:
            // One detent (120 units) = one notch. Plain wheel DOLLIES the camera a few fly-steps
            // per notch (+ve/wheel-up = forward, -ve = back; the coarse-vs-fine split lives in
            // main.cpp's kWheelDolly); Ctrl+wheel adjusts the STEP SIZE instead (up = bigger
            // steps). Both are feedback-locked — each notch is one bounded, fully rendered move,
            // so you can never overshoot into geometry between frames.
            if (self) {
                double notches = (double)GET_WHEEL_DELTA_WPARAM(wp) / 120.0;
                bool   ctrl    = (GET_KEYSTATE_WPARAM(wp) & MK_CONTROL) != 0;
                std::lock_guard<std::mutex> lk(self->inMtx);
                if (ctrl) self->wheelSpeedAcc += notches;
                else      self->wheelAcc      += notches;
            }
            return 0;
        case WM_COMMAND:
            // Control-panel buttons / edits. Button clicks raise the same one-shot edges the
            // keyboard uses (Clip == 'C', Reset == '0'); the radio pair sets the speed switch;
            // the edit boxes cache their parsed value on every change so drainNav reads current.
            if (self) {
                int id = LOWORD(wp), code = HIWORD(wp);
                switch (id) {
                    case ID_CLIP:  { std::lock_guard<std::mutex> lk(self->inMtx); self->collideReq = true; } break;
                    case ID_RESET: { std::lock_guard<std::mutex> lk(self->inMtx); self->resetReq   = true; } break;
                    case ID_PATH:  { std::lock_guard<std::mutex> lk(self->inMtx); self->pathReq     = true; } break;
                    case ID_PLAY:  { std::lock_guard<std::mutex> lk(self->inMtx); self->playReq     = true; } break;
                    case ID_SW_UPDATE:
                        if (code == BN_CLICKED) { std::lock_guard<std::mutex> lk(self->inMtx); self->rateModeVal = false; }
                        break;
                    case ID_SW_SEC:
                        if (code == BN_CLICKED) { std::lock_guard<std::mutex> lk(self->inMtx); self->rateModeVal = true; }
                        break;
                    case ID_STRIDE:
                        if (code == EN_CHANGE && self->hStride) {
                            wchar_t b[32]; GetWindowTextW(self->hStride, b, 32);
                            int v = _wtoi(b);
                            if (v >= 1) { std::lock_guard<std::mutex> lk(self->inMtx); self->strideVal = v; }
                        }
                        break;
                    case ID_RATE:
                        if (code == EN_CHANGE && self->hRate) {
                            wchar_t b[32]; GetWindowTextW(self->hRate, b, 32);
                            double v = wcstod(b, nullptr);
                            if (v > 0.0) { std::lock_guard<std::mutex> lk(self->inMtx); self->rateVal = v; }
                        }
                        break;
                    // ---- curve-editor buttons: one-shot edges the render loop acts on ----
                    case ID_REC:   { std::lock_guard<std::mutex> lk(self->inMtx); self->recReq  = true; } break;
                    case ID_ADDPT: { std::lock_guard<std::mutex> lk(self->inMtx); self->addReq  = true; } break;
                    case ID_INSPT: { std::lock_guard<std::mutex> lk(self->inMtx); self->insReq  = true; } break;
                    case ID_DELPT: { std::lock_guard<std::mutex> lk(self->inMtx); self->delReq  = true; } break;
                    case ID_SAVE:  { std::lock_guard<std::mutex> lk(self->inMtx); self->saveReq = true; } break;
                    case ID_RAW:
                        if (code == BN_CLICKED && self->hRaw) {
                            bool on = SendMessageW(self->hRaw, BM_GETCHECK, 0, 0) == BST_CHECKED;
                            std::lock_guard<std::mutex> lk(self->inMtx); self->rawVal = on;
                        }
                        break;
                    case ID_TOL:
                        if (code == EN_CHANGE && self->hTol) {
                            wchar_t b[32]; GetWindowTextW(self->hTol, b, 32);
                            double v = wcstod(b, nullptr);
                            if (v >= 0.0) { std::lock_guard<std::mutex> lk(self->inMtx); self->tolVal = v; }
                        }
                        break;
                    case ID_PAINT:
                        if (code == BN_CLICKED && self->hPaint) {
                            bool on = SendMessageW(self->hPaint, BM_GETCHECK, 0, 0) == BST_CHECKED;
                            std::lock_guard<std::mutex> lk(self->inMtx); self->paintVal = on;
                        }
                        break;
                    case ID_FLAT:  { std::lock_guard<std::mutex> lk(self->inMtx); self->flatReq = true; } break;
                    // ---- loom bind row: cache the two combo selections, raise the button edges ----
                    case ID_BCH:
                        if (code == CBN_SELCHANGE && self->hBCh) {
                            int sel = (int)SendMessageW(self->hBCh, CB_GETCURSEL, 0, 0);
                            std::lock_guard<std::mutex> lk(self->inMtx);
                            self->bindChVal = (sel == CB_ERR) ? -1 : sel;
                        }
                        break;
                    case ID_BSLOT:
                        if (code == CBN_SELCHANGE && self->hBSlot) {
                            int sel = (int)SendMessageW(self->hBSlot, CB_GETCURSEL, 0, 0);
                            std::string name;                       // index 0 is "(none)"
                            if (sel != CB_ERR && sel > 0) {
                                int n = (int)SendMessageW(self->hBSlot, CB_GETLBTEXTLEN, (WPARAM)sel, 0);
                                if (n > 0) {
                                    std::wstring w((size_t)n + 1, L'\0');
                                    SendMessageW(self->hBSlot, CB_GETLBTEXT, (WPARAM)sel, (LPARAM)&w[0]);
                                    w.resize((size_t)n);
                                    name = wideToUtf8(w);
                                }
                            }
                            std::lock_guard<std::mutex> lk(self->inMtx);
                            self->bindSlotVal = name;
                        }
                        break;
                    case ID_BIND:   { std::lock_guard<std::mutex> lk(self->inMtx); self->bindReq   = true; } break;
                    case ID_BCLEAR: { std::lock_guard<std::mutex> lk(self->inMtx); self->bclearReq = true; } break;
                    case ID_BDIMS:
                        if (code == EN_CHANGE && self->hBDims) {
                            wchar_t b[32]; GetWindowTextW(self->hBDims, b, 32);
                            int v = _wtoi(b);
                            // 0 is what an empty box reads as (mid-edit); it means "unchanged",
                            // never "a drive with no channels".
                            if (v >= 1) { std::lock_guard<std::mutex> lk(self->inMtx); self->bindDimsVal = v; }
                        }
                        break;
                    default: break;
                }
            }
            return 0;
        case WM_HSCROLL:
            // Timeline trackbar dragged / paged / arrowed: record the new camera index so the
            // render loop jumps the view there (and engages path mode). SB_ENDSCROLL still reports
            // the final position, so a drag ends on the exact frame the user released on.
            if (self && self->hTimeline && (HWND)lp == self->hTimeline) {
                int pos = (int)SendMessageW(self->hTimeline, TBM_GETPOS, 0, 0);
                std::lock_guard<std::mutex> lk(self->inMtx);
                self->scrubReq = pos;
            }
            return 0;
        case WM_KEYDOWN:
            // Unified fly-camera controls. Space or '+' (held) fly forward; Shift or '-'
            // (held) fly backward — you always travel where you look (or the exact
            // opposite when reversing). Mouse-look steers. Wheel throttles the speed.
            // '0'/Home reset the camera, 'P' prints a paste-ready camera block, 'C' cycles
            // the collision mode (slide/stop/noclip). The movement keys are layout-independent
            // (Space/Shift and the +/- keys land in the same place on QWERTY, Dvorak, etc.).
            if (self) {
                switch (wp) {
                    case VK_SPACE: case VK_OEM_PLUS: case VK_ADD:
                        self->keyFwd.store(true);  break;   // fly forward
                    case VK_SHIFT: case VK_OEM_MINUS: case VK_SUBTRACT:
                        self->keyBack.store(true); break;   // fly backward
                    case '0': case VK_HOME:
                        { std::lock_guard<std::mutex> lk(self->inMtx); self->resetReq = true; } break;
                    case 'P':
                        { std::lock_guard<std::mutex> lk(self->inMtx); self->printReq = true; } break;
                    case 'C':
                        { std::lock_guard<std::mutex> lk(self->inMtx); self->collideReq = true; } break;
                    case 'T':
                        { std::lock_guard<std::mutex> lk(self->inMtx); self->traceReq = true; } break;
                    default: break;
                }
            }
            return 0;
        case WM_KEYUP:
            // Clear the held-throttle state when the fly keys are released.
            if (self) {
                switch (wp) {
                    case VK_SPACE: case VK_OEM_PLUS: case VK_ADD:
                        self->keyFwd.store(false);  break;
                    case VK_SHIFT: case VK_OEM_MINUS: case VK_SUBTRACT:
                        self->keyBack.store(false); break;
                    default: break;
                }
            }
            return 0;
        case WM_KILLFOCUS:
            // Losing focus (Alt-Tab, click-away) must stop steering and drop any held
            // throttle, else the keys would appear "stuck" down.
            if (self) {
                self->endLook();
                self->keyFwd.store(false);
                self->keyBack.store(false);
            }
            return 0;
        case WM_GETMINMAXINFO:
            // Keep the window from being dragged smaller than a readable floor, so the
            // title bar (source -> destination) stays legible. The image itself is
            // aspect-fit + letterboxed into whatever size the window is, so a wide
            // minimum just adds black margins to a tall/square preview. (This can arrive
            // before WM_CREATE sets USERDATA, so tolerate a null self.)
            if (self) {
                auto mmi = reinterpret_cast<MINMAXINFO*>(lp);
                RECT r{0, 0, self->minW, self->minH};
                AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
                int minWpx = r.right - r.left, minHpx = r.bottom - r.top;
                if (self->panelH > 0) {
                    minHpx += self->panelH;             // room for the control strip below the image
                    if (minWpx < 700) minWpx = 700;     // wide enough for the button row not to clip
                }
                mmi->ptMinTrackSize.x = minWpx;
                mmi->ptMinTrackSize.y = minHpx;
            }
            return 0;
        case WM_CLOSE:
            if (self) {
                self->endLook();
                self->closedFlag.store(true);
                // Retire the presenter BEFORE the child HWND dies. Clearing d3dOk stops new
                // presents; taking and dropping the presenter lock then waits out any that
                // is already in flight on the render thread, so the swap chain is never
                // driven at a destroyed window.
                self->d3dOk.store(false);
                { std::lock_guard<std::mutex> lk(self->pres.mtx); self->pres.release(); }
                self->hview = nullptr;               // destroyed with the parent below
            }
            DestroyWindow(h);
            return 0;
        case WM_DESTROY:
            // The handle is now invalid: publish null so no cross-thread caller (or the dtor)
            // marshals to this — or a recycled — HWND after we return.
            if (self) self->hwnd.store(nullptr);
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

void LiveWindow::Impl::threadMain() {
    WNDCLASSEXW wc{};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = GetModuleHandleW(nullptr);
    wc.hCursor       = LoadCursorW(nullptr, (LPCWSTR)IDC_ARROW);   // integer resource id
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.lpszClassName = L"FtraceLiveWindow";
    RegisterClassExW(&wc);                          // benign if already registered

    RECT  r{0, 0, initW, initH};
    // WS_CLIPCHILDREN: the image area is a child window hosting the swap chain, so the
    // parent must never paint through it (that would fight the presenter and flicker).
    DWORD style = WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN;
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    int ww = r.right - r.left, wh = r.bottom - r.top;

    HWND hw = CreateWindowExW(0, wc.lpszClassName, title.c_str(), style,
                              CW_USEDEFAULT, CW_USEDEFAULT, ww, wh,
                              nullptr, nullptr, wc.hInstance, this);
    hwnd.store(hw);
    if (hw) {
        makeView(hw);                              // D3D child over the image area (may fail -> GDI)
        // SW_SHOWMINNOACTIVE, not SW_MINIMIZE: the latter would still activate the window
        // first (stealing focus for an instant, and dropping whatever the user was typing
        // into). SHOWMINNOACTIVE goes straight to the taskbar without ever taking the
        // foreground, which is the entire point of -window-min.
        ShowWindow(hw, g_lwStartMinimized ? SW_SHOWMINNOACTIVE : SW_SHOWNORMAL);
        UpdateWindow(hw);
        SetTimer(hw, 1, 33, nullptr);              // ~30 fps repaint poll (GDI fallback path)
    }
    if (readyEvent) SetEvent(readyEvent);          // unblock the ctor
    if (!hw) { closedFlag.store(true); return; }

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    closedFlag.store(true);
}

LiveWindow::LiveWindow(int w, int h, const char* title) {
    impl_ = new Impl();
    // Open the window at the render's OWN resolution/aspect (clamped to fit on-screen),
    // so the client area matches the image exactly — no letterbox bars on any side. The
    // old code forced a fixed 720-wide floor, which pillarboxed anything narrower (e.g.
    // a 640px render opened in a 720px window with 40px black bars each side).
    const int mw = 1600, mh = 900;
    double s = std::min(1.0, std::min((double)mw / std::max(1, w),
                                      (double)mh / std::max(1, h)));
    impl_->initW = std::max(1, (int)(w * s));
    impl_->initH = std::max(1, (int)(h * s));
    // Minimum drag size: a readable floor (~320px tall) scaled to KEEP the image's own
    // aspect, so shrinking the window never re-introduces letterbox bars and never
    // exceeds the initial image-sized window. The title bar stays legible.
    double fs = std::min(1.0, 320.0 / std::max(1, impl_->initH));
    impl_->minW = std::max(1, (int)(impl_->initW * fs));
    impl_->minH = std::max(1, (int)(impl_->initH * fs));
    std::string t = title ? title : "ftrace";
    impl_->title = utf8ToWide(t);                  // proper UTF-8 -> UTF-16
    impl_->readyEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    impl_->ui = std::thread([this] { impl_->threadMain(); });
    if (impl_->readyEvent) WaitForSingleObject(impl_->readyEvent, 3000);
}

LiveWindow::~LiveWindow() {
    if (!impl_) return;
    // Ask the window to close only if it is still alive (user hasn't already closed it, which
    // would have nulled hwnd on WM_DESTROY). Posting to a stale/recycled handle at exit is
    // exactly the kind of cross-thread hazard that can fault on shutdown.
    HWND hw = impl_->hwnd.load();
    if (hw) PostMessageW(hw, WM_CLOSE, 0, 0);
    if (impl_->ui.joinable()) impl_->ui.join();
    // Normally WM_CLOSE already retired the presenter; this covers the case where the UI
    // thread never got that far (window creation failed, or it was destroyed some other
    // way). release() is idempotent.
    impl_->pres.release();
    if (impl_->readyEvent) CloseHandle(impl_->readyEvent);
    delete impl_;
}

void LiveWindow::update(int w, int h, const std::vector<uint8_t>& rgb) {
    if (!impl_ || w <= 0 || h <= 0) return;
    if ((size_t)w * h * 3 > rgb.size()) return;
    if (impl_->d3dOk.load()) {
        // GPU path: the bytes go up exactly as the renderer produced them (one memcpy per
        // row into a mapped R8 texture), a shader turns RGB8 into RGBA8, and the swap chain
        // does the letterboxed scale. No repack, no DIB, no StretchDIBits — and no host
        // BGRA mirror, which is why WM_PRINTCLIENT reads back from the GPU instead.
        std::lock_guard<std::mutex> lk(impl_->pres.mtx);
        if (impl_->pres.ok && impl_->pres.uploadHost(w, h, rgb.data())) {
            impl_->pres.present(impl_->viewW.load(), impl_->viewH.load());
            return;
        }
        // Upload failed (device lost / OOM): drop to GDI for good rather than freeze the
        // preview on a stale frame.
        impl_->d3dOk.store(false);
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        impl_->imgW = w; impl_->imgH = h;
        impl_->bgra.resize((size_t)w * h * 4);
        for (size_t i = 0, n = (size_t)w * h; i < n; ++i) {
            impl_->bgra[i * 4 + 0] = rgb[i * 3 + 2];   // B
            impl_->bgra[i * 4 + 1] = rgb[i * 3 + 1];   // G
            impl_->bgra[i * 4 + 2] = rgb[i * 3 + 0];   // R
            impl_->bgra[i * 4 + 3] = 255;              // X
        }
    }
    impl_->dirty.store(true);
}

// Zero-copy frame: hand the caller the presenter's own image texture, let it fill it on the
// GPU, then present. The device lock is held across the whole callback — the CUDA interop it
// runs inside drives D3D's immediate context, which the UI thread also uses to repaint.
bool LiveWindow::renderShared(int w, int h, const std::function<bool(void*, void*)>& fn) {
    if (!impl_ || w <= 0 || h <= 0 || !fn) return false;
    if (!impl_->d3dOk.load()) return false;
    LivePresenter& p = impl_->pres;
    std::lock_guard<std::mutex> lk(p.mtx);
    if (!p.ok) return false;
    // ensureImg may hand back a DIFFERENT texture than last frame (resolution change); the
    // caller detects that by pointer and re-registers its interop mapping.
    if (!p.ensureImg(w, h)) { impl_->d3dOk.store(false); return false; }
    if (!fn(p.dev, p.imgTex)) return false;   // caller falls back to update(); texture is stale
    return p.present(impl_->viewW.load(), impl_->viewH.load());
}

void LiveWindow::setTitle(const std::string& utf8) {
    HWND hw = impl_ ? impl_->hwnd.load() : nullptr;
    if (!hw) return;
    // SetWindowTextW marshals a WM_SETTEXT to the window's own thread, so this is safe
    // to call from the render thread. Skip the OS call when the text is unchanged.
    std::wstring w = utf8ToWide(utf8);
    if (w == impl_->title) return;
    impl_->title = w;
    SetWindowTextW(hw, impl_->title.c_str());
}

bool LiveWindow::closed() const { return impl_ && impl_->closedFlag.load(); }

NavInput LiveWindow::drainNav() {
    if (!impl_) return {};
    NavInput n;
    // Held-key throttle + capture state read straight from the atomics (current state).
    n.fwd     = impl_->keyFwd.load();
    n.back    = impl_->keyBack.load();
    n.looking = impl_->looking.load();
    std::lock_guard<std::mutex> lk(impl_->inMtx);
    // Hover-look turn rate is PERSISTENT state (the current cursor offset): read but do NOT
    // clear, so the view keeps turning between drains while the pointer is held off-centre.
    n.lookX = impl_->lookX; n.lookY = impl_->lookY;
    // Accumulated wheel notches + one-shot edges: read-and-clear under the lock.
    n.wheel = impl_->wheelAcc; n.wheelSpeed = impl_->wheelSpeedAcc;
    n.reset  = impl_->resetReq; n.print = impl_->printReq;
    n.cycleCollide = impl_->collideReq; n.toggleTrace = impl_->traceReq;
    // Control-panel outputs: one-shot button edges (read-and-clear) + current input values.
    n.togglePath = impl_->pathReq;  n.togglePlay = impl_->playReq;  n.scrubTo = impl_->scrubReq;
    n.stride = impl_->strideVal;    n.camPerSec = impl_->rateVal;   n.rateMode = impl_->rateModeVal;
    // Curve-editor outputs: one-shot button edges (read-and-clear) + current authoring inputs.
    n.recToggle = impl_->recReq;    n.addPoint = impl_->addReq;     n.insPoint = impl_->insReq;
    n.delPoint  = impl_->delReq;    n.saveCurve = impl_->saveReq;
    n.simplifyTol = impl_->tolVal;  n.rawRecord = impl_->rawVal;
    // Paint-mode outputs: persistent checkbox state + one-shot Flat edge.
    n.paintMode = impl_->paintVal;  n.speedReset = impl_->flatReq;
    // Loom bind-row outputs: cached combo selections + one-shot Bind/Unbind edges.
    n.bindChannel = impl_->bindChVal;  n.bindTarget = impl_->bindSlotVal;
    n.bindApply   = impl_->bindReq;    n.bindClear  = impl_->bclearReq;
    n.dimsReq     = impl_->bindDimsVal;
    impl_->wheelAcc = impl_->wheelSpeedAcc = 0.0;
    impl_->resetReq = impl_->printReq = impl_->collideReq = impl_->traceReq = false;
    impl_->pathReq = impl_->playReq = false;
    impl_->scrubReq = -1;
    impl_->recReq = impl_->addReq = impl_->insReq = impl_->delReq = impl_->saveReq = false;
    impl_->flatReq = false;
    impl_->bindReq = impl_->bclearReq = false;
    return n;
}

bool LiveWindow::clientSize(int& w, int& h) const {
    HWND hw = impl_ ? impl_->hwnd.load() : nullptr;
    if (!hw) return false;
    RECT cr;
    if (!GetClientRect(hw, &cr)) return false;
    int cw = cr.right - cr.left, ch = (cr.bottom - cr.top) - impl_->panelH;  // image area only
    if (cw <= 0 || ch <= 0) return false;
    w = cw; h = ch;
    return true;
}

void LiveWindow::enablePanel(int pathCount, double defFps, const char* collideLabel) {
    HWND hw = impl_ ? impl_->hwnd.load() : nullptr;
    if (!hw || impl_->hasPanel.load()) return;
    {
        std::lock_guard<std::mutex> lk(impl_->inMtx);
        impl_->reqPathCount = pathCount;
        impl_->reqDefFps    = defFps;
        impl_->reqCollide   = collideLabel ? collideLabel : "slide";
        impl_->rateVal      = (defFps > 0.0) ? defFps : 30.0;   // seed the cam/sec input
        impl_->strideVal    = 1;
        impl_->rateModeVal  = true;                             // default switch = per-sec
    }
    // Build on the window's own thread (synchronous, so the child HWNDs exist on return).
    SendMessageW(hw, WM_MKPANEL, 0, 0);
}

void LiveWindow::setPanelState(int idx, bool playing, bool pathMode, const char* collideLabel) {
    if (!impl_ || !impl_->hasPanel.load()) return;
    // These USER32 calls marshal to the window thread; setting them does not re-emit the
    // matching NavInput edge (TBM_SETPOS/BM_SETCHECK/SetWindowText raise no WM_HSCROLL/WM_COMMAND).
    if (impl_->hTimeline) SendMessageW(impl_->hTimeline, TBM_SETPOS, TRUE, (LPARAM)idx);
    if (impl_->hPlay)     SetWindowTextW(impl_->hPlay, playing ? L"Pause" : L"Play");
    if (impl_->hPath)     SendMessageW(impl_->hPath, BM_SETCHECK, pathMode ? BST_CHECKED : BST_UNCHECKED, 0);
    if (impl_->hClip && collideLabel) {
        std::wstring w = utf8ToWide(std::string("Clip: ") + collideLabel);
        SetWindowTextW(impl_->hClip, w.c_str());
    }
}

void LiveWindow::setPathCount(int pathCount) {
    HWND hw = impl_ ? impl_->hwnd.load() : nullptr;
    if (!hw || !impl_->hasPanel.load()) return;
    // Marshal to the UI thread: retunes the trackbar range + shows/hides the path group.
    SendMessageW(hw, WM_SETPATHCOUNT, (WPARAM)pathCount, 0);
}

void LiveWindow::setEditState(bool recording, int pointCount) {
    if (!impl_ || !impl_->hasPanel.load()) return;
    if (impl_->hRec)   SetWindowTextW(impl_->hRec, recording ? L"Stop" : L"Rec");
    if (impl_->hPtLbl) { wchar_t b[32]; swprintf(b, 32, L"pts: %d", pointCount);
                         SetWindowTextW(impl_->hPtLbl, b); }
}

void LiveWindow::setSpeedLabel(double speedX) {
    if (!impl_ || !impl_->hasPanel.load() || !impl_->hSpdLbl) return;
    wchar_t b[32]; swprintf(b, 32, L"%.2fx", speedX);
    SetWindowTextW(impl_->hSpdLbl, b);   // marshals to the UI thread; no feedback edge
}

void LiveWindow::enableBindRow(const std::vector<std::string>& slotNames, int dims) {
    HWND hw = impl_ ? impl_->hwnd.load() : nullptr;
    if (!hw || !impl_->hasPanel.load() || impl_->hasBindRow.load()) return;
    {
        std::lock_guard<std::mutex> lk(impl_->inMtx);
        impl_->reqSlots = slotNames;
        impl_->reqDims  = std::max(1, dims);
    }
    // Build on the window's own thread (synchronous, so the child HWNDs exist on return).
    SendMessageW(hw, WM_MKBINDROW, 0, 0);
}

void LiveWindow::setBindState(const std::vector<std::string>& targets, const char* status) {
    if (!impl_ || !impl_->hasBindRow.load()) return;
    // Retune the channel list first: a drive that grew/shrank changes what can be selected.
    // (SendMessage marshals to the UI thread, so touching the combo from here is safe.)
    int dims = (int)targets.size();
    if (dims > 0 && dims != impl_->bindDims.load()) {
        impl_->setChannelCombo(dims);
        wchar_t db[16]; swprintf(db, 16, L"%d", dims);
        SetWindowTextW(impl_->hBDims, db);   // EN_CHANGE re-caches the same value: harmless
    }
    // The status line carries the CURRENT channel's binding as well as the link health, so the
    // row answers "what does this channel do right now?" without a second lookup.
    std::string line;
    int ch;
    { std::lock_guard<std::mutex> lk(impl_->inMtx); ch = impl_->bindChVal; }
    if (ch >= 0 && ch < dims)
        // NOTE the arrow is written as explicit UTF-8 BYTES, not a \u2192 escape: this is a
        // NARROW literal that utf8ToWide later decodes, and MSVC would try to transcode the
        // escape into the cp1252 execution charset (warning C4566) and emit a junk byte.
        line = "ch " + std::to_string(ch) + (targets[(size_t)ch].empty()
                   ? " unbound" : " \xE2\x86\x92 " + targets[(size_t)ch]);
    if (status && *status) line += (line.empty() ? "" : "   |   ") + std::string(status);
    std::wstring w = utf8ToWide(line);
    SetWindowTextW(impl_->hBStat, w.c_str());
}

#endif // _WIN32
