"""Regression check: settle.drop() rests an object on the first surface beneath it, wherever it starts.

    python -u tools/check_settle_drop.py

A unit ball is dropped onto a dome (a sphere of radius 2 centred at the origin, so its top is at
y = 2), 1.2 to the side of the apex, where the dome's surface is at y = sqrt(4 - 1.2^2) = 1.6. Started
above the dome, and started 5 m BELOW it (as tools/settle_scene.py's seat_on_stand can leave a piece:
it turns it about the world origin), the ball must come to rest touching the dome, its lowest point at
most 1 mm above the surface under it and nowhere inside the dome: with nothing under it, drop() lifts
it clear of the surface first. And started between a floor and a canopy over it, the ball must land on
the floor, the first surface below where it starts -- not on the canopy above it. Exits 1 on failure.
"""
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settle import drop  # noqa: E402


def main():
    dome = trimesh.creation.icosphere(subdivisions=5, radius=2.0)
    ok = True
    for start_y in (4.0, -5.0):
        ball = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        ball.apply_translation([1.2, start_y, 0.0])
        t, _ = drop(ball, ("mesh", dome), 1e-5)
        ball.apply_translation(t)
        # The gap between the dropped ball and the dome: the closest approach of their surfaces.
        gap = float(np.linalg.norm(ball.centroid) - 3.0)            # centres 3 apart when touching
        inside = int(dome.contains(ball.vertices).sum())
        good = -2e-3 <= gap <= 1e-3 and inside == 0
        ok &= good
        print("%s: a ball dropped from y = %+.1f onto the dome's flank: its surface %.4f from the "
              "dome's, %d vertices inside it" % ("ok" if good else "FAIL", start_y, gap, inside))
    # A floor slab (top at y = 0) and a canopy slab (from y = 5 to 5.2) over it, one surface mesh.
    floor = trimesh.creation.box(extents=[10.0, 0.2, 10.0])
    floor.apply_translation([0.0, -0.1, 0.0])
    canopy = trimesh.creation.box(extents=[10.0, 0.2, 10.0])
    canopy.apply_translation([0.0, 5.1, 0.0])
    room = trimesh.util.concatenate([floor, canopy])
    ball = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    ball.apply_translation([0.0, 2.5, 0.0])
    t, _ = drop(ball, ("mesh", room), 1e-5)
    bottom = float(ball.vertices[:, 1].min() + t[1])
    good = abs(bottom) <= 1e-3
    ok &= good
    print("%s: a ball between a floor and a canopy lands on the floor: its bottom at y = %.4f"
          % ("ok" if good else "FAIL", bottom))
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
