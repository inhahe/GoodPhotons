#include "livewindow.h"

#ifndef _WIN32
// -------- Non-Windows stub: -window is a no-op (headless builds unaffected) --------
struct LiveWindow::Impl {};
LiveWindow::LiveWindow(int, int, const char*) : impl_(nullptr) {}
LiveWindow::~LiveWindow() {}
void LiveWindow::update(int, int, const std::vector<uint8_t>&) {}
void LiveWindow::setTitle(const std::string&) {}
bool LiveWindow::closed() const { return false; }

#else
// ------------------------------- Win32 GDI window ----------------------------------
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <algorithm>

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

struct LiveWindow::Impl {
    std::thread          ui;
    std::mutex           mtx;          // guards bgra / imgW / imgH
    std::vector<uint8_t> bgra;         // imgW*imgH*4, top-down BGRA (GDI DIB order)
    int                  imgW = 0, imgH = 0;
    std::atomic<bool>    dirty{false};
    std::atomic<bool>    closedFlag{false};
    HWND                 hwnd = nullptr;
    int                  initW = 0, initH = 0;
    std::wstring         title;
    HANDLE               readyEvent = nullptr;

    static LRESULT CALLBACK WndProc(HWND h, UINT msg, WPARAM wp, LPARAM lp);
    void threadMain();
    void paint(HDC hdc, const RECT& client);
};

void LiveWindow::Impl::paint(HDC hdc, const RECT& client) {
    int cw = client.right - client.left, ch = client.bottom - client.top;
    if (cw <= 0 || ch <= 0) return;
    std::lock_guard<std::mutex> lk(mtx);
    // Double-buffer through a memory DC so the letterbox fill + stretch blit land in
    // one BitBlt (no flicker; WM_ERASEBKGND is suppressed).
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
            if (self) self->paint(hdc, cr);
            EndPaint(h, &ps);
            return 0;
        }
        case WM_SIZE:
            InvalidateRect(h, nullptr, FALSE);
            return 0;
        case WM_CLOSE:
            if (self) self->closedFlag.store(true);
            DestroyWindow(h);
            return 0;
        case WM_DESTROY:
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
    DWORD style = WS_OVERLAPPEDWINDOW;
    AdjustWindowRect(&r, style, FALSE);
    int ww = r.right - r.left, wh = r.bottom - r.top;

    hwnd = CreateWindowExW(0, wc.lpszClassName, title.c_str(), style,
                           CW_USEDEFAULT, CW_USEDEFAULT, ww, wh,
                           nullptr, nullptr, wc.hInstance, this);
    if (hwnd) {
        ShowWindow(hwnd, SW_SHOWNORMAL);
        UpdateWindow(hwnd);
        SetTimer(hwnd, 1, 33, nullptr);            // ~30 fps repaint poll
    }
    if (readyEvent) SetEvent(readyEvent);          // unblock the ctor
    if (!hwnd) { closedFlag.store(true); return; }

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    closedFlag.store(true);
}

LiveWindow::LiveWindow(int w, int h, const char* title) {
    impl_ = new Impl();
    // Clamp the initial window to a sane on-screen size, preserving aspect.
    const int mw = 1600, mh = 900;
    double s = std::min(1.0, std::min((double)mw / std::max(1, w),
                                      (double)mh / std::max(1, h)));
    impl_->initW = std::max(160, (int)(w * s));
    impl_->initH = std::max(90,  (int)(h * s));
    std::string t = title ? title : "ftrace";
    impl_->title = utf8ToWide(t);                  // proper UTF-8 -> UTF-16
    impl_->readyEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    impl_->ui = std::thread([this] { impl_->threadMain(); });
    if (impl_->readyEvent) WaitForSingleObject(impl_->readyEvent, 3000);
}

LiveWindow::~LiveWindow() {
    if (!impl_) return;
    if (impl_->hwnd) PostMessageW(impl_->hwnd, WM_CLOSE, 0, 0);
    if (impl_->ui.joinable()) impl_->ui.join();
    if (impl_->readyEvent) CloseHandle(impl_->readyEvent);
    delete impl_;
}

void LiveWindow::update(int w, int h, const std::vector<uint8_t>& rgb) {
    if (!impl_ || w <= 0 || h <= 0) return;
    if ((size_t)w * h * 3 > rgb.size()) return;
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

void LiveWindow::setTitle(const std::string& utf8) {
    if (!impl_ || !impl_->hwnd) return;
    // SetWindowTextW marshals a WM_SETTEXT to the window's own thread, so this is safe
    // to call from the render thread. Skip the OS call when the text is unchanged.
    std::wstring w = utf8ToWide(utf8);
    if (w == impl_->title) return;
    impl_->title = w;
    SetWindowTextW(impl_->hwnd, impl_->title.c_str());
}

bool LiveWindow::closed() const { return impl_ && impl_->closedFlag.load(); }

#endif // _WIN32
