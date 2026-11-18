"""Regression check: settle.drop() rests an object on the surface beneath it, wherever it starts.

    python -u tools/check_settle_drop.py

A unit ball is dropped onto a dome (a sphere of radius 2 centred at the origin, so its top is at
y = 2), 1.2 to the side of the apex, where the dome's surface is at y = sqrt(4 - 1.2^2) = 1.6. Started
above the dome, and started 5 m BELOW it (as tools/settle_scene.py's seat_on_stand can leave a piece:
it turns it about the world origin), the ball must come to rest touching the dome, its lowest point at
most 1 mm above the surface under it and nowhere inside the dome. Down-rays cast from below the
surface found nothing, and drop() fell back to the dome's bounding top, leaving the ball hovering
over its flank. Exits 1 on failure.
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
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
