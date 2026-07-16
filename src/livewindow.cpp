#include "livewindow.h"

#ifndef _WIN32
// -------- Non-Windows stub: -window is a no-op (headless builds unaffected) --------
struct LiveWindow::Impl {};
LiveWindow::LiveWindow(int, int, const char*) : impl_(nullptr) {}
LiveWindow::~LiveWindow() {}
void LiveWindow::update(int, int, const std::vector<uint8_t>&) {}
void LiveWindow::setTitle(const std::string&) {}
bool LiveWindow::closed() const { return false; }
NavInput LiveWindow::drainNav() { return {}; }
bool LiveWindow::clientSize(int&, int&) const { return false; }

#else
// ------------------------------- Win32 GDI window ----------------------------------
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <windowsx.h>          // GET_X_LPARAM / GET_Y_LPARAM
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
    int                  minW = 640, minH = 300;   // readable floor so the title bar stays legible
    std::wstring         title;
    HANDLE               readyEvent = nullptr;
    // ---- Fly-camera input state (guarded by inMtx unless noted) ----
    std::mutex           inMtx;                     // guards the look/wheel accumulators + one-shots
    double               lookDx = 0.0, lookDy = 0.0;// accumulated mouse-look deltas (client px) since drain
    double               wheelAcc = 0.0;            // accumulated wheel notches since last drain
    bool                 resetReq = false;          // '0' / Home pressed since last drain (one-shot)
    bool                 printReq = false;          // 'P' pressed since last drain (one-shot)
    // Held-key throttle state — atomics so WM_KEYUP on the UI thread and drainNav on the
    // render thread can race freely without the inMtx.
    std::atomic<bool>    keyFwd{false};             // Space / '+' currently held -> fly forward
    std::atomic<bool>    keyBack{false};            // Shift / '-' currently held -> fly backward
    // Mouse-look capture: while `looking`, the OS cursor is hidden and re-centred every
    // move so the user can turn without limit. Esc releases; a click re-captures.
    std::atomic<bool>    looking{false};            // mouse-look currently captured
    bool                 recentring = false;        // true while we ignore the SetCursorPos echo event

    static LRESULT CALLBACK WndProc(HWND h, UINT msg, WPARAM wp, LPARAM lp);
    void threadMain();
    void paint(HDC hdc, const RECT& client);
    void setCapture(HWND h, bool on);               // enter/leave mouse-look (hide+centre cursor)
};

// Enter or leave mouse-look capture: hide/show the cursor, confine it (so it can't leave
// the window while turning), and re-centre it. Called only on the UI thread.
void LiveWindow::Impl::setCapture(HWND h, bool on) {
    if (on == looking.load()) return;
    looking.store(on);
    if (on) {
        RECT cr; GetClientRect(h, &cr);
        POINT c{ (cr.right - cr.left) / 2, (cr.bottom - cr.top) / 2 };
        ClientToScreen(h, &c);
        recentring = true;
        SetCursorPos(c.x, c.y);
        while (ShowCursor(FALSE) >= 0) {}            // hide (counter may start > 0)
        RECT sr = cr; POINT tl{cr.left, cr.top}, br{cr.right, cr.bottom};
        ClientToScreen(h, &tl); ClientToScreen(h, &br);
        sr.left = tl.x; sr.top = tl.y; sr.right = br.x; sr.bottom = br.y;
        ClipCursor(&sr);                            // keep the cursor inside while turning
        ::SetCapture(h);
    } else {
        ClipCursor(nullptr);
        while (ShowCursor(TRUE) < 0) {}             // show
        if (GetCapture() == h) ReleaseCapture();
    }
}

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
        case WM_PRINTCLIENT: {
            // Render the current frame into the caller's DC so PrintWindow() captures the
            // live image even when the window is occluded (used for off-screen grabs).
            if (self) { RECT cr; GetClientRect(h, &cr); self->paint((HDC)wp, cr); }
            return 0;
        }
        case WM_SIZE:
            InvalidateRect(h, nullptr, FALSE);
            return 0;
        case WM_LBUTTONDOWN:
            // A click (re)captures mouse-look: the cursor is hidden and re-centred so the
            // user can turn freely. Esc releases it (to resize/close the window).
            if (self) self->setCapture(h, true);
            return 0;
        case WM_MOUSEMOVE:
            // While mouse-look is captured, accumulate the raw motion away from the client
            // centre as a steering delta, then warp the cursor back to the centre so the
            // next move measures a fresh delta (unlimited turning). The SetCursorPos warp
            // generates its own WM_MOUSEMOVE, which we skip via the `recentring` flag.
            if (self && self->looking.load()) {
                if (self->recentring) { self->recentring = false; return 0; }
                RECT cr; GetClientRect(h, &cr);
                int cx = (cr.right - cr.left) / 2, cy = (cr.bottom - cr.top) / 2;
                int mx = GET_X_LPARAM(lp), my = GET_Y_LPARAM(lp);
                int dx = mx - cx, dy = my - cy;
                if (dx || dy) {
                    { std::lock_guard<std::mutex> lk(self->inMtx);
                      self->lookDx += dx; self->lookDy += dy; }
                    POINT c{cx, cy}; ClientToScreen(h, &c);
                    self->recentring = true;
                    SetCursorPos(c.x, c.y);
                }
            }
            return 0;
        case WM_MOUSEWHEEL:
            // One detent (120 units) = one notch of fly-SPEED: +ve (wheel up) = faster.
            if (self) {
                double notches = (double)GET_WHEEL_DELTA_WPARAM(wp) / 120.0;
                std::lock_guard<std::mutex> lk(self->inMtx);
                self->wheelAcc += notches;
            }
            return 0;
        case WM_KEYDOWN:
            // Unified fly-camera controls. Space or '+' (held) fly forward; Shift or '-'
            // (held) fly backward — you always travel where you look (or the exact
            // opposite when reversing). Mouse-look steers. Wheel throttles the speed.
            // '0'/Home reset the camera, 'P' prints a paste-ready camera block, Esc
            // releases the captured cursor. These are layout-independent (Space/Shift and
            // the +/- keys land in the same place on QWERTY, Dvorak, etc.).
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
                    case VK_ESCAPE:
                        self->setCapture(h, false); break;  // release cursor
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
            // Losing focus (Alt-Tab, click-away) must release the cursor and drop any held
            // throttle, else the keys would appear "stuck" down.
            if (self) {
                self->setCapture(h, false);
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
                mmi->ptMinTrackSize.x = r.right - r.left;
                mmi->ptMinTrackSize.y = r.bottom - r.top;
            }
            return 0;
        case WM_CLOSE:
            if (self) { self->setCapture(h, false); self->closedFlag.store(true); }
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

NavInput LiveWindow::drainNav() {
    if (!impl_) return {};
    NavInput n;
    // Held-key throttle + capture state read straight from the atomics (current state).
    n.fwd     = impl_->keyFwd.load();
    n.back    = impl_->keyBack.load();
    n.looking = impl_->looking.load();
    // Accumulated look/wheel deltas + one-shot edges: read-and-clear under the lock.
    std::lock_guard<std::mutex> lk(impl_->inMtx);
    n.lookDx = impl_->lookDx; n.lookDy = impl_->lookDy; n.wheel = impl_->wheelAcc;
    n.reset  = impl_->resetReq; n.print = impl_->printReq;
    impl_->lookDx = impl_->lookDy = impl_->wheelAcc = 0.0;
    impl_->resetReq = impl_->printReq = false;
    return n;
}

bool LiveWindow::clientSize(int& w, int& h) const {
    if (!impl_ || !impl_->hwnd) return false;
    RECT cr;
    if (!GetClientRect(impl_->hwnd, &cr)) return false;
    int cw = cr.right - cr.left, ch = cr.bottom - cr.top;
    if (cw <= 0 || ch <= 0) return false;
    w = cw; h = ch;
    return true;
}

#endif // _WIN32
