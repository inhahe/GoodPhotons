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

// One interactive control command queued from a key press in the live window. The
// window only reports *which* control was pressed; the render loop (which knows the
// scene scale) drains these and turns each into a camera nudge. Eye/target moves are
// along the WORLD axes so the resulting numbers drop straight into a `.ftsl` camera.
enum class NudgeCmd {
    EyeXNeg, EyeXPos, EyeYNeg, EyeYPos, EyeZNeg, EyeZPos,   // move the camera eye
    TgtXNeg, TgtXPos, TgtYNeg, TgtYPos, TgtZNeg, TgtZPos,   // move the look-at target (world axes)
    TgtNear, TgtFar,                                         // move the target along the view axis
    StepDown, StepUp,                                        // finer / coarser move step
    Reset,                                                   // back to the authored camera
    Print,                                                   // dump a paste-ready camera block
};

// Accumulated pointer (mouse) input since the last drainPointer(). The window reports
// raw motion only — it doesn't know the scene/camera — and the render loop maps it onto
// the look-at target. `dragDx/dragDy` are in IMAGE PIXELS (the window divides the
// client-space drag by the current letterbox scale so one image pixel dragged = one
// image pixel of target motion), with +dragDx = cursor right and +dragDy = cursor down.
// `wheel` is in notches (+ = wheel forward / push the target farther).
struct PointerInput {
    double dragDx = 0.0, dragDy = 0.0;   // left-drag, image-pixel space
    double wheel  = 0.0;                  // wheel notches (+ = away/farther)
    bool   any() const { return dragDx != 0.0 || dragDy != 0.0 || wheel != 0.0; }
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

    // Return (and clear) the interactive control commands queued from key presses
    // since the last call. Empty when nothing was pressed. Thread-safe.
    std::vector<NudgeCmd> drainNudges();

    // Return (and clear) the accumulated mouse drag / wheel motion since the last call.
    // Thread-safe. See PointerInput for units.
    PointerInput drainPointer();

    LiveWindow(const LiveWindow&) = delete;
    LiveWindow& operator=(const LiveWindow&) = delete;

private:
    struct Impl;
    Impl* impl_;
};
