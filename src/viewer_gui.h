// -viewer: the loom native viewer host (F2+).
//
// A Dear ImGui + Direct3D 11 window that reads a loom scene-introspection JSON
// *sidecar* (produced by loom.viewer.ViewerModel.save_sidecar / the §F1 contract)
// and presents the scene structurally: an object list, the dataset table, a
// camera/lights summary, and a 3-D pane that draws the N-D curves' sampled
// polylines.
//
// F7 primary path: when the sidecar carries a "source" key (the `.ftsl` loom emits
// beside it), the viewer parses it with ftrace's own ftsl::load and adds a "Render"
// tab that raymarches the real isosurface field in-process via renderIsoPreviewCuda
// (the -raster-gpu preview kernel — sphere-traces the field bytecode, no
// tessellation), blitting each frame into a D3D11 texture. An orbit camera drives
// it. This replaces the static marching-cubes sidecar mesh with the actual field.
//
// F4 item 2: given a loom scene file to talk to, the viewer also opens the **live
// re-introspection channel** — it spawns `python -m loom.viewer <scene.py>` and asks
// it to re-derive the scene whenever the user moves along a *parameter* dimension (a
// declared build param, or the clock). Orbiting the three shown spatial dims stays a
// free view-only re-projection; a parameter move changes the geometry itself, which
// only loom can re-tessellate. Requests go through a latest-wins off-thread job
// queue, so a fast drag never builds a backlog of stale bakes.
//
// This is a thin GUI shell — it owns no renderer state. On non-Windows builds it
// is a stub that reports the viewer is unavailable (mirrors livewindow.cpp).
#pragma once
#include <string>

// Open the native viewer on a sidecar JSON file. Blocks running the UI loop until
// the window is closed. Returns a process exit code (0 = ok, non-zero = error).
//
// `loomScene` is the loom `build()` file to open the live channel against. Empty
// means "use the sidecar's own `build` key if it has one"; a sidecar with neither
// still opens, just frozen — the Live panel says why.
//
// `startPlaying` opens with the transport already running (the `-play` flag), so a
// loop can be watched — or its per-frame cost traced to stdout — without anyone
// having to click into the window first.
//
// `startPrebake` kicks the §F8(b) prebake walk on open (the `-prebake` flag): the
// clock is baked once end to end into memory and play then runs out of the cache at
// a wall-clock rate instead of at loom's bake rate. Combined with `-play` it is the
// scriptable way to measure what cached playback actually achieves.
//
// `prebakeCapMB` overrides the cache's memory budget (`-prebake-cap`); 0 keeps the
// panel's default. A walk that hits the cap stops there and the prefix it did fill
// is still played from memory, with the tail falling back to baking on demand.
int runViewerGui(const std::string& sidecarPath, const std::string& loomScene = "",
                 bool startPlaying = false, bool startPrebake = false,
                 int prebakeCapMB = 0);
