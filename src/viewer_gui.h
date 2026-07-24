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
// This is a thin GUI shell — it owns no renderer state. On non-Windows builds it
// is a stub that reports the viewer is unavailable (mirrors livewindow.cpp).
#pragma once
#include <string>

// Open the native viewer on a sidecar JSON file. Blocks running the UI loop until
// the window is closed. Returns a process exit code (0 = ok, non-zero = error).
int runViewerGui(const std::string& sidecarPath);
