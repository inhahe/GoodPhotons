// viewerhelp.h — the live viewer's Help button: what every control actually does.
//
// The viewer's control strip is a row of one-word buttons and, with `-nd`, a bank of
// unlabelled sliders and drop-downs. Nothing on screen says what any of it means, and the
// N-D fills in particular are not guessable from their names — "emboss radius" does not
// announce that it does nothing at all on a sphere. This writes a self-contained HTML page
// to the temp directory and hands it to the default browser.
//
// Self-contained on purpose: no network, no assets, no install step, and it works from a
// double-clicked exe with no repo checked out next to it.
#pragma once

#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  // windows.h defines min/max as MACROS, which turns every later std::min(a, b) into a
  // syntax error. This header can be included before anything else, so it must not be the
  // one that lets them in.
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  #include <shellapi.h>
#endif

namespace ftviewerhelp {

inline const char* pageHtml();   // defined below; one long string literal

// Write the page somewhere stable and open it. Returns false if either step fails, so the
// caller can fall back to printing the path rather than silently doing nothing.
inline bool openHelp() {
#ifdef _WIN32
    const char* tmp = std::getenv("TEMP");
    if (!tmp) tmp = std::getenv("TMP");
    if (!tmp) tmp = ".";
    std::string path = std::string(tmp) + "\\ftrace_viewer_help.html";
    // Rewritten every time: the binary is the source of truth for its own help, so a stale
    // file from an older build must never win.
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const char* h = pageHtml();
    std::fwrite(h, 1, std::strlen(h), f);
    std::fclose(f);
    const HINSTANCE r = ShellExecuteA(nullptr, "open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    if ((INT_PTR)r <= 32) {
        std::printf("[viewer] help written to %s (could not open a browser for it)\n", path.c_str());
        std::fflush(stdout);
        return false;
    }
    std::printf("[viewer] help opened: %s\n", path.c_str());
    std::fflush(stdout);
    return true;
#else
    return false;
#endif
}

inline const char* pageHtml() {
    return R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><title>ftrace viewer — controls</title>
<style>
 body{background:#1c1f26;color:#d8dde6;font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;
      max-width:900px;margin:0 auto;padding:32px 24px 80px}
 h1{font-size:26px;margin:0 0 4px;color:#fff}
 h2{font-size:19px;margin:34px 0 10px;color:#fff;border-bottom:1px solid #333a46;padding-bottom:5px}
 h3{font-size:15px;margin:22px 0 6px;color:#9fd3ff}
 .sub{color:#8b95a5;margin:0 0 26px}
 table{border-collapse:collapse;width:100%;margin:10px 0 18px}
 td,th{border-top:1px solid #2c333f;padding:7px 10px;vertical-align:top;text-align:left}
 th{color:#8b95a5;font-weight:600;font-size:13px;text-transform:uppercase;letter-spacing:.4px}
 td:first-child{white-space:nowrap;color:#ffd479;font-weight:600;width:150px}
 code{background:#252b35;padding:1px 5px;border-radius:3px;color:#9fe6b0;font-size:13px}
 .note{background:#232833;border-left:3px solid #5b8dd6;padding:10px 14px;margin:14px 0;border-radius:0 4px 4px 0}
 .warn{background:#2b2622;border-left:3px solid #d6a35b}
 em{color:#c5cede}
</style></head><body>

<h1>ftrace live viewer</h1>
<p class="sub">What every button, slider and field does. Written by the build you are running.</p>

<h2>Flying the camera</h2>
<p>The mouse steers by <em>offset from the centre</em> of the image: the further off-centre the
pointer, the faster you turn, and holding it dead centre holds still. The cursor stays visible,
and leaving the window stops the turn.</p>
<table>
<tr><th>Key</th><th>Does</th></tr>
<tr><td>Space / +</td><td>Fly forward — you travel where you look.</td></tr>
<tr><td>Shift / &minus;</td><td>Fly backward.</td></tr>
<tr><td>Mouse wheel</td><td>Dolly one notch forward or back. Each notch renders, so you cannot overshoot.</td></tr>
<tr><td>Ctrl + wheel</td><td>Bias the step size. Travel normally auto-scales to whatever is ahead — a notch is about 16% of the way to what you are aiming at.</td></tr>
<tr><td>C</td><td>Cycle wall collision: slide along walls → stop dead → noclip.</td></tr>
<tr><td>T</td><td>Cycle the lit preview: raster → mode W (deterministic, CPU, any scene) → path-traced (fast RGB, GPU).</td></tr>
<tr><td>0</td><td>Reset the view.</td></tr>
<tr><td>P</td><td>Print the current camera block to the console, ready to paste into a scene.</td></tr>
</table>
<p>Resizing the window changes the render resolution — the image fills the window with no bars.
Smaller is faster on a heavy scene, larger is crisper. The horizontal field of view widens with
the window; <code>fov_y</code> stays fixed.</p>

<h2>The control strip</h2>
<table>
<tr><th>Control</th><th>Does</th></tr>
<tr><td>Clip</td><td>Cycles wall collision, same as the <code>C</code> key. The label shows the current mode.</td></tr>
<tr><td>Reset</td><td>Returns the camera to its starting pose.</td></tr>
<tr><td>Help</td><td>This page.</td></tr>
<tr><td>Color</td><td>Toggles material colour. Off shades everything as neutral clay, which is the honest way to judge <em>form</em> — albedo and skins hide shape. Same as <code>-flat</code> / <code>-no-color</code>.</td></tr>
<tr><td>See-through</td><td>Draws clear dielectrics as see-through rather than opaque. A preview of <em>look</em> only: there is no refraction and no reflection. <code>-glass-clarity</code> sets how much they dim; <code>-glass-haze</code> caps the frost so a deep pile of glass stays readable.</td></tr>
<tr><td>Flat</td><td>In path mode, resets painted speed along the camera curve to uniform.</td></tr>
</table>

<h3>Camera-path editing</h3>
<table>
<tr><td>Rec</td><td>Records your flight into a <code>camera_curve</code>.</td></tr>
<tr><td>+Pt</td><td>Appends the current pose as a control point.</td></tr>
<tr><td>Ins</td><td>Inserts a point at the scrub position.</td></tr>
<tr><td>Del</td><td>Removes the point nearest the scrub position.</td></tr>
<tr><td>Save</td><td>Writes a <code>camera_curve</code> block you can paste into a scene.</td></tr>
<tr><td>Paint</td><td>Path mode: the wheel paints local speed (point density) at the scrub point while the mouse steers orientation.</td></tr>
<tr><td>Play / timeline</td><td>Runs the camera along the curve. The slider scrubs it.</td></tr>
<tr><td>Path lock</td><td>Locks the camera to the curve — orientation and travel both follow it.</td></tr>
<tr><td>cams/upd, cams/s</td><td>Playback rate. <em>cams/s</em> advances by wall-clock time; <em>per upd</em> advances a fixed number of cameras per rendered frame, which is steadier when the renderer is bursty.</td></tr>
<tr><td>raw / tol</td><td>Curve fitting: <em>raw</em> keeps every recorded sample, <em>tol</em> sets how far the fitted curve may stray from them.</td></tr>
<tr><td>Bind / Unbind / chans</td><td>loom sidecar binding: attach a channel of an external CurveDrive to this view.</td></tr>
</table>

<h2>N-dimensional rotation (<code>-nd</code>)</h2>
<p>The model is lifted into <em>n</em> dimensions, rotated there, and projected orthographically
back to three. The panel grows one slider per rotation <em>plane</em> — n(n&minus;1)/2 of them, so
6 at n=4, 10 at n=5, 45 at n=10 — labelled with the plane and its current angle
(<code>xw +25</code>).</p>

<div class="note"><strong>The one fact that explains everything else.</strong> With every extra
dimension left at <code>zero</code>, the whole slider bank is provably just a 3&times;3 matrix. It can
rotate, shear, squash, and past 180° in a mixed plane produce the <em>mirror</em> image — but it
cannot add structure. If it feels like the sliders only ever squash your model, that is not a bug:
you have not filled a dimension yet. The status line says so, and says what to do about it.</div>

<h3>Fill: what each extra dimension contains</h3>
<p>One row per extra dimension — a drop-down and an amount slider. <strong>The amount is the knob
that decides whether you see anything.</strong> The default 0.25 is subtle enough to read as
"nothing happened"; 1.0 is unmistakable. Amounts are fractions of the model's radius, so the same
number means the same thing on a 2&nbsp;cm ring and a 40&nbsp;m building.</p>
<table>
<tr><th>Fill</th><th>Sets that dimension to…</th></tr>
<tr><td>zero</td><td>Nothing. The classic lift — and the affine case above.</td></tr>
<tr><td>emboss&nbsp;curvature</td><td>How sharply the surface bends at each vertex. Detail and creases push out; flat areas stay put. On a rippled or ornamented model this erupts blades out of the ornament.</td></tr>
<tr><td>emboss&nbsp;radius</td><td>Distance from the model's centre. Smooth and global — it sweeps the whole shape into a wing or blade. <em>Does nothing on a sphere</em>, where every vertex is the same distance from the centre; that is correct, not a failure.</td></tr>
<tr><td>emboss&nbsp;height</td><td>The vertex's own <em>y</em>. Note this is a <em>linear</em> function of position, so rotating it in gives an affine <strong>shear</strong> — very dramatic, but not a genuine 4-D morph.</td></tr>
<tr><td>emboss&nbsp;noise</td><td>Solid noise sampled in model-radius units, so <code>freq</code> reads as cycles across the model. Crumples and folds it.</td></tr>
<tr><td>emboss&nbsp;u / v</td><td>The vertex's texture coordinate. Smooth and predictable on a model with continuous UVs; on an <em>atlased</em> model the UV islands are discontinuous, so the displacement tears along every seam and the result looks shredded. That is the atlas showing through, not a bug — use it on parametric surfaces.</td></tr>
<tr><td>extrude</td><td>Sweeps the mesh into a real N-D prism along that axis, depth in model radii. See below.</td></tr>
</table>

<p>Two things the fill cells do for you. Picking a fill while its amount is still 0 starts it at a
visible default, since an emboss at zero is indistinguishable from <code>zero</code>. And switching one
on while its axis is completely <em>edge-on</em> turns the z-axis plane to 30° so you can see what
you just asked for — the slider visibly moves, and the console says why.</p>

<div class="warn note"><strong>An extruded axis is invisible until you turn it into view.</strong>
An extra dimension reaches the image only through its column of the rotation's first three rows.
Until some plane containing that axis is turned, the column is zero, the sweep projects to zero
area, and it looks like nothing happened. Turn <code>zw</code>, <code>xw</code> or similar.</div>

<h3>What <code>extrude</code> builds</h3>
<p>The boundary of the swept solid: the original triangles at one end, their copy at the other,
and each <em>boundary</em> edge swept into a quad. For a <strong>closed</strong> mesh there are no
boundary edges, so it simply doubles — you see the two copies separate as you rotate, with nothing
between them, because for a closed surface there is genuinely nothing there. An <em>open</em> mesh
(a plane, a shell with a hole) grows walls along its rim, which is what closes its prism.</p>

<h3>Other N-D controls</h3>
<table>
<tr><td>N-D dims</td><td>Changes <em>n</em> on the fly. The slider bank is rebuilt; angles you have set are kept where the planes still exist.</td></tr>
<tr><td>Reset</td><td>Returns every rotation angle to zero. Fills and amounts are left alone.</td></tr>
<tr><td>Save model</td><td>Writes the projected 3-D model out — <code>.obj</code> or <code>.ftmesh</code> — to the <code>-nd-export</code> path if you gave one, else a <code>_nd</code> sibling of the source file.</td></tr>
</table>
<p>The status line under the sliders reads
<code>&lt;n&gt; tris | w=… v=… | …</code>: the triangle count after projection, what each extra
dimension is filled with, and — when it applies — that the whole bank is currently equivalent to a
3&times;3 matrix.</p>

</body></html>
)HTML";
}

}  // namespace ftviewerhelp

// Small free function so livewindow.cpp's message loop can call it without pulling the
// namespace into scope at the call site.
inline void ftViewerHelpOpen() { ftviewerhelp::openHelp(); }
