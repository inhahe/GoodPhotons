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
//   * mouse-look is HOVER-look with RATE (joystick) steering: the cursor's offset from the
//     window centre sets a TURN RATE. Near the centre is a neutral dead zone (the view holds
//     still, so you can see the scene); pushing the pointer toward an edge turns the view that
//     way and KEEPS turning while you hold it there, so you can look a full circle without the
//     cursor ever leaving the window. The cursor stays VISIBLE and free — never hidden, clipped,
//     or warped — and steering stops the instant the pointer leaves the client area (so you can
//     reach the title bar or other apps without turning). `lookX/lookY` are the dead-zoned offset,
//     each axis in [-1,+1] (+x = pointer right of centre, +y = pointer below centre); they are
//     PERSISTENT STATE (the current offset), not per-drain accumulators — drainNav reads but does
//     not clear them, so the view keeps turning between drains while the pointer is held off-centre.
//   * `fwd` / `back` are the CURRENT held state of the throttle keys (Space or '+' fly
//     forward; Shift or '-' fly backward) — the render loop advances you ONE fixed step
//     per RENDERED frame while one is held (feedback-locked: motion scales with render
//     speed, so you never skip past geometry you didn't see).
//   * `wheel` (plain-wheel notches, + = up) DOLLIES the camera: each notch is one bounded
//     fly-step forward (+) / back (-), fully rendered — precise, overshoot-proof nudging.
//   * `wheelSpeed` (Ctrl+wheel notches, + = up) adjusts the STEP SIZE (up = bigger steps).
//   * `reset` / `print` / `cycleCollide` are one-shot edge flags ('0'/Home reset the
//     camera; 'P' prints a paste-ready camera block; 'C' cycles the collision mode
//     slide -> stop -> noclip). `looking` reports whether the cursor is currently inside the
//     client area (steering live); it goes false the instant the pointer leaves the window or
//     focus is lost, and the cursor is always free to resize/close the window.
//
// The window can also host an optional CONTROL PANEL below the image (see enablePanel):
// buttons for collision + reset, and — when a multi-frame camera path is present — a
// timeline (scrub/play/pause), a "lock to path" toggle, and two traversal-speed inputs
// with a switch between them. Those controls also feed back through NavInput:
//   * `togglePath` / `togglePlay` are one-shot edges (the Path / Play-Pause buttons).
//   * `scrubTo` is the camera index the user dragged/jumped the timeline to (>=0), else -1.
//   * `stride` (cameras advanced per RENDERED frame) and `camPerSec` (cameras per WALL-CLOCK
//     second, defaulting to the scene fps) are the two mutually-exclusive traversal speeds;
//     `rateMode` is the switch (true = use camPerSec / wall clock, false = use stride / per
//     update). These are current values (0 = "unchanged"), not one-shot edges.
struct NavInput {
    double lookX  = 0.0, lookY  = 0.0;   // hover-look turn RATE from cursor offset, dead-zoned, -1..+1 per axis (persistent state)
    double wheel  = 0.0;                  // plain-wheel notches (+ = up = dolly forward)
    double wheelSpeed = 0.0;             // Ctrl+wheel notches (+ = up = bigger step size)
    bool   fwd    = false;               // Space / '+' held  -> fly forward
    bool   back   = false;               // Shift / '-' held  -> fly backward
    bool   reset  = false;               // '0' / Home / Reset button since last drain
    bool   print  = false;               // 'P' pressed since last drain
    bool   cycleCollide = false;         // 'C' / Clip button since last drain (cycle collision mode)
    bool   looking = false;              // cursor currently inside the client area (steering live)
    // ---- Control-panel outputs (drained alongside the fly input) ----
    bool   togglePath = false;           // "Path" (lock-to-path) button pressed since last drain (one-shot)
    bool   togglePlay = false;           // "Play/Pause" button pressed since last drain (one-shot)
    int    scrubTo    = -1;              // timeline dragged/jumped to this camera index (>=0), else -1
    int    stride     = 0;               // "cameras / screen update" input (current value; 0 = unchanged)
    double camPerSec  = 0.0;             // "cameras / second" input (current value; 0 = unchanged)
    bool   rateMode   = false;           // speed switch: true = cam/sec (wall clock), false = stride (per update)
    // ---- Curve-EDITOR outputs (the interactive camera_curve authoring panel) ----
    // One-shot button edges (drainNav clears them) plus two persistent input values. The
    // render loop owns the actual CameraTrack; the window only reports button presses and
    // the two authoring parameters (simplify tolerance + raw/simplified choice).
    bool   recToggle  = false;           // "Rec" button: start/stop recording a flythrough (one-shot toggle)
    bool   addPoint   = false;           // "+Pt" button: append current pose as a control point (one-shot)
    bool   insPoint   = false;           // "Ins" button: insert a control point at the current scrub position (one-shot)
    bool   delPoint   = false;           // "Del" button: delete the selected/nearest control point (one-shot)
    bool   saveCurve  = false;           // "Save" button: write the authored camera_curve block (one-shot)
    double simplifyTol = -1.0;           // recording simplify tolerance in world units (current value; <0 = unchanged)
    bool   rawRecord  = false;           // "raw" checkbox: keep every recorded sample (true) vs. simplify (false)
    // ---- Paint-mode outputs (speed + orientation painting along the path) ----
    // `paintMode` is the persistent "Paint" checkbox state (not an edge): while it is on and the
    // view is locked to the path, the plain wheel PAINTS local traversal speed (additive brush,
    // clamped) at the current scrub position and mouse-look STEERS the nearest control points'
    // orientation, instead of nudging/being suspended. `speedReset` is the one-shot "Flat" button
    // (reset the painted speed track to uniform).
    bool   paintMode  = false;           // "Paint" checkbox: wheel=speed, mouse=orientation on the path (persistent)
    bool   speedReset = false;           // "Flat" button: reset painted speed to uniform (one-shot)
    bool   any() const { return lookX || lookY || wheel || wheelSpeed || fwd || back || reset || print
                                || cycleCollide || togglePath || togglePlay || scrubTo >= 0
                                || recToggle || addPoint || insPoint || delPoint || saveCurve || speedReset; }
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

    // Show the control panel strip below the image (marshalled to the UI thread; safe to call
    // once from the render thread). The window grows by the panel height so the image area is
    // unchanged. `pathCount` is the number of cameras on the timeline: >=2 shows the timeline,
    // Play/Pause, the Path (lock-to-path) toggle, the two speed inputs and their switch; <2
    // shows only the Clip and Reset buttons (no path controls). `defFps` seeds the cam/sec box
    // and `collideLabel` the initial Clip-button text. An EDITOR row (Rec / +Pt / Ins / Del /
    // Save + a simplify-tolerance box and raw toggle) is always built so a camera_curve can be
    // authored even from a lone camera. No-op on non-Windows / stub builds.
    void enablePanel(int pathCount, double defFps, const char* collideLabel);

    // Reconfigure the timeline at runtime after the user authors/edits a curve (the editor
    // regenerates the played path from its control points). `pathCount` is the new number of
    // cameras: >=2 shows/retunes the timeline + Play/Pause + Path controls (creating them if
    // the panel was built without a path), <2 hides them. Marshalled to the UI thread; no-op if
    // the panel isn't enabled. Never re-emits a NavInput edge.
    void setPathCount(int pathCount);

    // Mirror the editor's state onto the panel: `recording` sets the Rec button label
    // (Rec/Stop) and `pointCount` updates the control-point readout. Marshalled to the UI
    // thread; no feedback edge. No-op if the panel isn't enabled.
    void setEditState(bool recording, int pointCount);

    // Update the panel's painted-speed readout (the "Paint" mode shows the local traversal-speed
    // multiplier at the current scrub position, e.g. "1.35x"). Marshalled to the UI thread; no
    // feedback edge. No-op if the panel isn't enabled / no path controls exist.
    void setSpeedLabel(double speedX);

    // Push live viewer state so the panel mirrors reality (call from the render loop whenever
    // it changes): `idx` moves the timeline slider (e.g. during playback), `playing` sets the
    // Play/Pause label, `pathMode` sets the Path toggle, `collideLabel` sets the Clip button
    // text. Marshalled to the UI thread; setting these never re-emits the corresponding
    // NavInput edge (no feedback loop). No-op if the panel isn't enabled.
    void setPanelState(int idx, bool playing, bool pathMode, const char* collideLabel);

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
