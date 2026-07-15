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

    LiveWindow(const LiveWindow&) = delete;
    LiveWindow& operator=(const LiveWindow&) = delete;

private:
    struct Impl;
    Impl* impl_;
};
