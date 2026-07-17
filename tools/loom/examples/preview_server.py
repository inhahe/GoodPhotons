"""
Loom example: the **resident preview server** (M12).

``render_range`` spawns a fresh ftrace process *per frame* — each one re-pays
process start-up, live-window creation, CUDA-context init and spectral-table
build before it renders a single pixel.  For an interactive preview loop that
fixed cost dominates.

:func:`loom.preview.preview_range` (this demo) keeps **one** ftrace resident
(``ftrace -serve``) and streams it one scene path per frame over stdin, so all of
that global state is paid for once and the live window updates *in place* frame to
frame — a smooth preview, no window flashing.

Honest scope: this is the resident-*process* win only.  Every frame is still a
full independent render — no incremental delta rendering, no static-geometry / BVH
caching between frames, no reduced preview LOD (yet).

Because ftrace's live GDI window needs the interactive Console session, run this
from a real console (Claude: ``dangerouslyDisableSandbox: true``).

Run:
  python examples/preview_server.py            # print frame-0 .ftsl to stdout
  python examples/preview_server.py --preview   # stream a bobbing sphere loop live
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loom import (  # noqa: E402
    Scene, Camera, Sphere, Material, Light, Sine, Cosine, vec, Const, Raw,
)


def build_scene(res=(360, 360)) -> Scene:
    # A sphere that bobs on a seamless vertical loop and drifts side to side.
    y = Sine(cycles=1, amp=0.28)
    x = Cosine(cycles=1, amp=0.22)
    scene = Scene(Camera(
        eye=(0.0, 0.7, 2.3), look_at=(0.0, 0.0, 0.0), up=(0, 1, 0),
        fov_y=42, mode="R", res=res))
    scene.add(
        Material("ball", "diffuse", reflect=0.82),
        Material("floor", "diffuse", reflect=0.6),
        Sphere(center=vec(x, y, Const(0.0)), radius=0.4, material="ball"),
        Raw('quad { origin -2 -0.45 -2  u 4 0 0  v 0 0 4  material "floor" }'),
        Light("area", origin="-0.4 1.7 0.6", u="0.9 0 0", v="0 0 0.9",
              normal="0 -1 0", spd="preset:bb6500"),
    )
    return scene


def main() -> int:
    scene = build_scene()
    if "--preview" not in sys.argv:
        from loom import Clock, Cache
        print(scene.emit(Clock(t=0.0), Cache()))
        return 0

    from loom.preview import preview_range
    frames = 48
    paths = preview_range(scene, frames, name="preview_server", res=360, noise=4.0)
    print(f"[preview] streamed {len(paths)} frames through one resident ftrace")
    return 0


if __name__ == "__main__":
    sys.exit(main())
