#!/usr/bin/env python3
"""Generate scenes/lichen.ppm — the source image for the stochastic-tiling demo.

The demo (scenes/stochtile.ftsl) puts plain repeat tiling next to histogram-preserving
stochastic tiling, so the input has to be chosen to make that comparison *fair* and
*hard*:

  * Exactly periodic (toroidal noise everywhere, no cropping). A non-tileable photo
    would make plain tiling lose on visible SEAMS, which is a different and much easier
    complaint. Here plain tiling is perfectly seamless and still loses — on visible
    REPETITION — which is the artifact the operator actually removes.
  * Blobby, with a feature size a good fraction of the tile. Repetition is only
    conspicuous when the eye has landmarks to match up; fine sand would hide it.
  * A strongly non-Gaussian histogram (a fourth power of the mottle, plus a hard
    ceiling). This is what separates the paper's rank transform from a naive
    cross-fade: the cross-fade regresses everything toward the mean and turns a
    high-contrast lichen into grey soup, while the rank transform reproduces this
    exact histogram. If the input were already Gaussian, both would look the same
    and the demo would prove nothing.

Deterministic (fixed seed) — re-running reproduces the checked-in asset byte for byte.

    python tools/make_stochtex.py            # writes scenes/lichen.ppm
"""

import os
import sys

import numpy as np

SIZE = 192          # texture resolution (square)
SEED = 20260810


def _periodic_value_noise(size, period, rng):
    """Value noise on a `period` x `period` toroidal lattice, smoothstep-interpolated."""
    lat = rng.random((period, period))
    t = np.arange(size) * (period / size)
    i0 = np.floor(t).astype(int) % period
    i1 = (i0 + 1) % period
    f = t - np.floor(t)
    w = f * f * (3.0 - 2.0 * f)                      # smoothstep

    # Separable bilinear blend of the four wrapped lattice corners.
    a = lat[np.ix_(i0, i0)]
    b = lat[np.ix_(i1, i0)]
    c = lat[np.ix_(i0, i1)]
    d = lat[np.ix_(i1, i1)]
    wx = w[:, None]
    wy = w[None, :]
    return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy


def _fbm(size, periods, rng):
    """Sum of periodic value-noise octaves, normalized to [0,1]."""
    out = np.zeros((size, size))
    amp, total = 1.0, 0.0
    for p in periods:
        out += amp * _periodic_value_noise(size, p, rng)
        total += amp
        amp *= 0.5
    return out / total


def _warp(field, dx, dy):
    """Bilinear toroidal resample of `field` by a per-pixel offset, in pixels.

    Warping keeps the result periodic because both the field and the offsets are, and
    every lookup wraps. This is what turns smooth blobs into crinkly lichen edges —
    the irregularity a viewer uses as a landmark, and therefore the thing that makes
    plain tiling's repetition obvious.
    """
    n = field.shape[0]
    gx, gy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    fx, fy = gx + dx, gy + dy
    x0 = np.floor(fx).astype(int); y0 = np.floor(fy).astype(int)
    tx = fx - x0; ty = fy - y0
    x0 %= n; y0 %= n
    x1 = (x0 + 1) % n; y1 = (y0 + 1) % n
    return ((field[x0, y0] * (1 - tx) + field[x1, y0] * tx) * (1 - ty) +
            (field[x0, y1] * (1 - tx) + field[x1, y1] * tx) * ty)


def _periodic_worley(size, cells, rng):
    """Toroidal Worley F1, normalized to roughly [0,1]."""
    pts = (rng.random((cells * cells, 2)) + np.stack(
        np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij"),
        axis=-1).reshape(-1, 2)) / cells

    g = (np.arange(size) + 0.5) / size
    px, py = np.meshgrid(g, g, indexing="ij")
    best = np.full((size, size), 1e9)
    for cx, cy in pts:
        dx = np.abs(px - cx); dx = np.minimum(dx, 1.0 - dx)
        dy = np.abs(py - cy); dy = np.minimum(dy, 1.0 - dy)
        best = np.minimum(best, dx * dx + dy * dy)
    d = np.sqrt(best) * cells
    return np.clip(d / 0.8, 0.0, 1.0)


def build():
    rng = np.random.default_rng(SEED)

    # --- the rock substrate: fine, high-frequency speckle, dark and low-contrast ----
    grain = _fbm(SIZE, (24, 48, 96), rng)
    grit = _periodic_worley(SIZE, 26, rng)           # tight cells = mineral speckle

    # --- the lichen patches: big blobs, warped into crinkly rosettes ----------------
    patch = _fbm(SIZE, (3, 6, 12), rng)
    wx = (_fbm(SIZE, (6, 12, 24), rng) - 0.5) * 14.0
    wy = (_fbm(SIZE, (6, 12, 24), rng) - 0.5) * 14.0
    patch = _warp(patch, wx, wy)
    # A hard-ish threshold: the resulting histogram is BIMODAL (rock mode + lichen
    # mode) with almost nothing between. That gap is the demo. A cross-fade of three
    # crops lands squarely in it and invents a grey that occurs nowhere in the image;
    # the rank transform cannot, because it only ever emits values the image contains.
    mask = np.clip((patch - 0.47) / 0.085, 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)
    # Bloom: brighter, chalkier toward each patch's interior.
    core = np.clip((patch - 0.55) / 0.12, 0.0, 1.0)

    rock = np.stack([
        0.085 + 0.115 * grain + 0.055 * grit,
        0.080 + 0.105 * grain + 0.050 * grit,
        0.078 + 0.095 * grain + 0.045 * grit,
    ], axis=-1)

    inner = _fbm(SIZE, (12, 24, 48), rng)
    micro = 0.82 + 0.36 * _fbm(SIZE, (48, 96), rng)   # granular bloom, kills plastic flatness
    lich = np.stack([
        0.20 + 0.26 * inner + 0.30 * core,
        0.30 + 0.34 * inner + 0.34 * core,
        0.16 + 0.20 * inner + 0.26 * core,
    ], axis=-1) * micro[..., None]
    # A rust-ochre rim where the lichen thins out, so patch borders read as edges.
    rim = np.clip(mask * (1.0 - core) * 1.6, 0.0, 1.0)[..., None]
    lich = lich * (1.0 - 0.35 * rim) + np.array([0.40, 0.24, 0.07]) * (0.35 * rim)

    rgb = np.clip(rock * (1.0 - mask[..., None]) + lich * mask[..., None], 0.0, 1.0)
    return np.rint(rgb * 255.0).astype(np.uint8)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "scenes", "lichen.ppm")
    img = build()
    # Row-major, top row first — the PPM convention Texture::loadPPM expects.
    with open(out, "wb") as f:
        f.write(b"P6\n# generated by tools/make_stochtex.py (seed %d)\n%d %d\n255\n"
                % (SEED, SIZE, SIZE))
        f.write(np.transpose(img, (1, 0, 2)).tobytes())
    print("wrote %s (%dx%d)" % (os.path.normpath(out), SIZE, SIZE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
