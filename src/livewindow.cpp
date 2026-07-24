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
void LiveWindow::enablePanel(int, double, const char*) {}
void LiveWindow::setPanelState(int, bool, bool, const char*) {}
void LiveWindow::setPathCount(int) {}
void LiveWindow::setEditState(bool, int) {}
void LiveWindow::setSpeedLabel(double) {}

#else
// ------------------------------- Win32 GDI window ----------------------------------
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <windowsx.h>          // GET_X_LPARAM / GET_Y_LPARAM
#include <commctrl.h>          // trackbar (msctls_trackbar32) for the timeline
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <cstdlib>             // strtod / atoi for the speed inputs
#include <algorithm>
#pragma comment(lib, "comctl32.lib")

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

// ---- Control-panel constants ----
// Child-window command IDs (WM_COMMAND LOWORD) + the trackbar. Kept out of the low
// range Windows reserves for standard dialog buttons.
enum {
    ID_CLIP = 1001, ID_RESET, ID_PATH, ID_PLAY,
    ID_TIMELINE, ID_STRIDE, ID_RATE, ID_SW_UPDATE, ID_SW_SEC,
    // ---- curve-editor row ----
    ID_REC, ID_ADDPT, ID_INSPT, ID_DELPT, ID_SAVE, ID_TOL, ID_RAW,
    // ---- paint-mode controls (speed + orientation painting) ----
    ID_PAINT, ID_FLAT
};
static const int kPanelH = 92;              // reserved control-strip height (px): buttons + timeline + editor rows
// Marshal cross-thread panel ops onto the window's own message-pump thread.
#define WM_MKPANEL      (WM_APP + 1)        // build the control panel (params staged in Impl)
#define WM_SETPATHCOUNT (WM_APP + 2)        // retune the timeline range/visibility (wParam = new count)

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

    static LRESULT CALLBACK WndProc(HWND h, UINT msg, WPARAM wp, LPARAM lp);
    void threadMain();
    void paint(HDC hdc, const RECT& client);
    void endLook();                                 // cursor left / focus lost: stop steering cleanly
    void buildPanel(HWND h);                        // create child controls + grow window (UI thread)
    void layoutPanel(HWND h);                       // position child controls in the strip (UI thread)
    void applyPathCount(int pc);                    // retune/show/hide the timeline group (UI thread)
    void showPathGroup(bool vis);                   // toggle visibility of the path (timeline) controls
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

void LiveWindow::Impl::paint(HDC hdc, const RECT& client) {
    int cw = client.right - client.left;
    int ch = (client.bottom - client.top) - panelH;   // image area = client minus the control strip
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
        case WM_MKPANEL:
            if (self) self->buildPanel(h);           // build controls + grow window (UI thread)
            return 0;
        case WM_SETPATHCOUNT:
            if (self && self->hasPanel.load()) { self->applyPathCount((int)wp); InvalidateRect(h, nullptr, FALSE); }
            return 0;
        case WM_SIZE:
            if (self) self->layoutPanel(h);          // reflow the control strip to the new width
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
            if (self) { self->endLook(); self->closedFlag.store(true); }
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
    DWORD style = WS_OVERLAPPEDWINDOW;
    AdjustWindowRect(&r, style, FALSE);
    int ww = r.right - r.left, wh = r.bottom - r.top;

    HWND hw = CreateWindowExW(0, wc.lpszClassName, title.c_str(), style,
                              CW_USEDEFAULT, CW_USEDEFAULT, ww, wh,
                              nullptr, nullptr, wc.hInstance, this);
    hwnd.store(hw);
    if (hw) {
        ShowWindow(hw, SW_SHOWNORMAL);
        UpdateWindow(hw);
        SetTimer(hw, 1, 33, nullptr);              // ~30 fps repaint poll
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
    impl_->wheelAcc = impl_->wheelSpeedAcc = 0.0;
    impl_->resetReq = impl_->printReq = impl_->collideReq = impl_->traceReq = false;
    impl_->pathReq = impl_->playReq = false;
    impl_->scrubReq = -1;
    impl_->recReq = impl_->addReq = impl_->insReq = impl_->delReq = impl_->saveReq = false;
    impl_->flatReq = false;
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

#endif // _WIN32
