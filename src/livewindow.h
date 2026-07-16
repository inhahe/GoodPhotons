// Live-preview window — a real OS window that displays the current render frame,
// refreshed on demand from the render loop (the -window CLI flag). Unlike the
// -preview ANSI terminal thumbnail, this shows the actual pixels.
//
// On Windows it is a Win32 GDI window running on its own message-pump thread, so it
// stays responsive between updates regardless of how long the renderer blocks; the
// render thread just hands it a fresh tone-mapped RGB frame via update(). No
// third-party dependency. On non-Windows builds every method is a no-op stub, so
// callers need no platform guards.
#pragma once
#include <vector>
#include <string>
#include <cstdint>

// Interactive FLY-CAMERA input, accumulated since the last drainNav(). The window
// reports raw device input only (it doesn't know the scene/camera); the render loop
// integrates it into camera motion. The navigation model is a single unified flycam —
// you always travel where you look (or the exact opposite when reversing), so there is
// no separate "aim the target" mode and no crosshair.
//
//   * mouse-look STEERS: horizontal motion yaws, vertical motion pitches. While look is
//     captured the OS cursor is hidden and re-centred every frame, so you can turn
//     without limit. `lookDx/lookDy` are raw client-pixel deltas (+dx = cursor right,
//     +dy = cursor down).
//   * `fwd` / `back` are the CURRENT held state of the throttle keys (Space or '+' fly
//     forward; Shift or '-' fly backward) — the render loop advances you ONE fixed step
//     per RENDERED frame while one is held (feedback-locked: motion scales with render
//     speed, so you never skip past geometry you didn't see).
//   * `wheel` (plain-wheel notches, + = up) DOLLIES the camera: each notch is one bounded
//     fly-step forward (+) / back (-), fully rendered — precise, overshoot-proof nudging.
//   * `wheelSpeed` (Ctrl+wheel notches, + = up) adjusts the STEP SIZE (up = bigger steps).
//   * `reset` / `print` / `cycleCollide` are one-shot edge flags ('0'/Home reset the
//     camera; 'P' prints a paste-ready camera block; 'C' cycles the collision mode
//     slide -> stop -> noclip). `looking` reports whether mouse-look is currently
//     captured (Esc releases the cursor so the window can be resized/closed; a click
//     re-captures).
struct NavInput {
    double lookDx = 0.0, lookDy = 0.0;   // mouse-look motion, client pixels
    double wheel  = 0.0;                  // plain-wheel notches (+ = up = dolly forward)
    double wheelSpeed = 0.0;             // Ctrl+wheel notches (+ = up = bigger step size)
    bool   fwd    = false;               // Space / '+' held  -> fly forward
    bool   back   = false;               // Shift / '-' held  -> fly backward
    bool   reset  = false;               // '0' / Home pressed since last drain
    bool   print  = false;               // 'P' pressed since last drain
    bool   cycleCollide = false;         // 'C' pressed since last drain (cycle collision mode)
    bool   looking = false;              // mouse-look captured (cursor hidden)
    bool   any() const { return lookDx || lookDy || wheel || wheelSpeed || fwd || back || reset || print || cycleCollide; }
};

class LiveWindow {
public:
    // Create and show a window sized to (w,h) (clamped to the screen, aspect kept).
    LiveWindow(int w, int h, const char* title);
    ~LiveWindow();

    // Push a fresh frame: rgb is w*h*3 bytes, RGB order, row 0 = image top (matches
    // writeFilm's output). Copied internally, so the caller's buffer may be reused
    // immediately. w/h may differ from the ctor size (the window stretches to fit,
    // preserving aspect with letterboxing).
    void update(int w, int h, const std::vector<uint8_t>& rgb);

    // Replace the title-bar text (UTF-8). Safe to call from the render thread; the
    // change is marshalled to the window's own message-pump thread. Used to show the
    // live render status (scene/output + spp/noise) as the frame converges.
    void setTitle(const std::string& utf8);

    // True once the user has closed the window — lets the render stop early.
    bool closed() const;

    // Return (and clear) the accumulated fly-camera input since the last call: mouse-look
    // deltas, wheel-throttle notches, the current held state of the forward/back throttle
    // keys, and the one-shot reset/print edges. Thread-safe. See NavInput for units.
    NavInput drainNav();

    // Current client-area size in pixels (what the image is letterboxed into). Lets the
    // interactive render loop match its raster resolution to the live window, so shrinking
    // the window renders fewer pixels (faster) and growing it renders more (crisper).
    // Returns false (and leaves w/h untouched) on headless/stub builds or before the
    // window exists. Thread-safe.
    bool clientSize(int& w, int& h) const;

    LiveWindow(const LiveWindow&) = delete;
    LiveWindow& operator=(const LiveWindow&) = delete;

private:
    struct Impl;
    Impl* impl_;
};
