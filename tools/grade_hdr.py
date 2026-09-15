#!/usr/bin/env python3
"""Develop a rendered PFM sequence to PNG with a keyframed exposure ramp.

WHY THIS EXISTS
---------------
`-hdr` writes a scene-linear PFM beside every frame: no exposure, no transfer curve, no
clamp. The PNG beside it is ONE development of that data, and for a flythrough it is
deliberately a rigid one -- `exposure_lock` shares frame 0's anchor across every frame so the
sequence cannot flicker.

That is the right default and the wrong final answer for a shot with range in it. Measured on
`gallery_rain`: a front-lit gallery frame and a sun-facing frame are **4.7 stops** apart, and
against a frame-0 anchor 22.2 % of the backlit frame clips. No single exposure serves both.
The PFMs hold it all -- the brightest pixel in that frame is ~50000x the anchor -- so the fix
is to choose exposure AFTER the render, per frame, and ramp it smoothly. That is what a real
pipeline does: render linear, grade later. No re-render, and `exposure_lock` keeps doing its
job for the preview PNGs.

IT MATCHES THE RENDERER'S DEVELOPMENT EXACTLY, which is the whole point of it living in
`tools/` rather than being a scratch script. main.cpp's film->image path is:

    v   = linear * exposure                       # exposure = gain * expComp
    out = clamp(srgbGamma(v) * 255 + 0.5, 0, 255) # TRUE sRGB, not a 2.2 power

with `srgbGamma` from color.h (12.92*c below 0.0031308, else 1.055*c^(1/2.4) - 0.055), and
two exposure modes:

  * absolute  -- exposure = ABS_EXPOSURE_GAIN (6.0) * expComp. Scene power flows straight
                 through; the anchor and the path lock are bypassed. `gallery_rain` is this.
  * auto      -- eAuto = 0.9 / p99, where p99 is the 99th percentile of max(R,G,B) over the
                 linear frame. `--anchor` reuses one frame's value for all of them, which is
                 what `exposure_lock` does inside the renderer.

An approximation here (a 2.2 power, or a different percentile) would make graded frames
subtly mismatch the renderer's own PNGs, which is worse than not having the tool.

USAGE
-----
    # gallery_rain: absolute mode, its film block's exposure 0.25, dim the sun-facing section
    python tools/grade_hdr.py png/rain_fly --absolute --exp-comp 0.25 \
           --ev 0:0 240:0 280:-4.5 330:-4.5 370:0 --out png/rain_graded

    # flat development, no ramp (reproduces the renderer's PNG)
    python tools/grade_hdr.py png/rain_fly --absolute --exp-comp 0.25 --out png/check

`--ev` takes `frame:stops` pairs; stops are EV offsets applied on top of the base exposure
(negative = darker). Between keys the ramp is SMOOTHSTEP, not linear: a linear ramp has a
corner in its derivative at each key, and a visible exposure kink is exactly what grading is
supposed to avoid. Before the first key and after the last, the nearest key is held.
"""
import argparse
import os
import re
import struct
import sys
import zlib

import numpy as np

ABS_EXPOSURE_GAIN = 6.0          # main.cpp: constexpr double ABS_EXPOSURE_GAIN


def read_pfm(path):
    """Read a binary PFM. Returns float64 [H,W,3], row 0 = image top."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic not in (b"PF", b"Pf"):
            raise ValueError("%s is not a binary PFM (magic %r)" % (path, magic))
        ch = 3 if magic == b"PF" else 1
        dims = f.readline().split()
        w, h = int(dims[0]), int(dims[1])
        scale = float(f.readline())
        data = np.frombuffer(f.read(w * h * ch * 4),
                             dtype="<f4" if scale < 0 else ">f4")
    img = data.reshape(h, w, ch).astype(np.float64)
    if ch == 1:
        img = np.repeat(img, 3, axis=2)
    return img[::-1]                      # PFM rows run bottom-to-top


def srgb_gamma(c):
    """color.h srgbGamma, vectorised. NOT a 2.2 power."""
    c = np.maximum(c, 0.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def write_png(path, rgb8):
    h, w, _ = rgb8.shape
    raw = b"".join(b"\x00" + rgb8[j].tobytes() for j in range(h))

    def chunk(tag, data):
        head = struct.pack(">I", len(data)) + tag + data
        return head + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


def auto_anchor(lin):
    """main.cpp: eAuto = 0.9 / p99, p99 over max(R,G,B)."""
    lum = np.maximum(lin.max(axis=2), 0.0).ravel()
    k = int(0.99 * (lum.size - 1))
    p99 = np.partition(lum, k)[k]
    return (0.9 / p99) if p99 > 0 else 1.0


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ev_at(frame, keys):
    """Smoothstep-interpolated EV offset at `frame`; nearest key held outside the range."""
    if not keys:
        return 0.0
    if frame <= keys[0][0]:
        return keys[0][1]
    if frame >= keys[-1][0]:
        return keys[-1][1]
    for (f0, v0), (f1, v1) in zip(keys, keys[1:]):
        if f0 <= frame <= f1:
            if f1 == f0:
                return v1
            return v0 + (v1 - v0) * smoothstep((frame - f0) / float(f1 - f0))
    return keys[-1][1]


def frame_index(path):
    m = re.search(r"(\d+)(?!.*\d)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="directory of .pfm frames, or a single .pfm")
    ap.add_argument("--out", required=True, help="output directory for PNGs")
    ap.add_argument("--absolute", action="store_true",
                    help="absolute EV: exposure = 6.0 * --exp-comp (bypasses any anchor)")
    ap.add_argument("--exp-comp", type=float, default=1.0,
                    help="the camera's exposure compensation (film block `exposure`)")
    ap.add_argument("--anchor", type=float, default=0.0,
                    help="auto mode: reuse this anchor for every frame (what exposure_lock "
                         "does). 0 = derive it from the FIRST frame and hold it.")
    ap.add_argument("--per-frame-auto", action="store_true",
                    help="auto mode: re-derive the anchor per frame. Flickers; for stills.")
    ap.add_argument("--ev", nargs="*", default=[], metavar="FRAME:STOPS",
                    help="keyframed EV offsets, e.g. 0:0 280:-4.5 370:0")
    args = ap.parse_args(argv)

    keys = []
    for spec in args.ev:
        try:
            f, v = spec.split(":")
            keys.append((int(f), float(v)))
        except ValueError:
            ap.error("--ev takes FRAME:STOPS pairs, got %r" % spec)
    keys.sort()

    if os.path.isdir(args.src):
        files = sorted(os.path.join(args.src, n) for n in os.listdir(args.src)
                       if n.lower().endswith(".pfm"))
    else:
        files = [args.src]
    if not files:
        print("no .pfm files in %s -- did the render pass -hdr?" % args.src, file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)

    held = args.anchor if args.anchor > 0 else 0.0
    for path in files:
        lin = read_pfm(path)
        idx = frame_index(path)
        if args.absolute:
            base = ABS_EXPOSURE_GAIN * (args.exp_comp if args.exp_comp > 0 else 1.0)
        else:
            if args.per_frame_auto:
                e_auto = auto_anchor(lin)
            else:
                if held <= 0.0:
                    held = auto_anchor(lin)     # first frame sets it, as exposure_lock does
                e_auto = held
            base = e_auto * (args.exp_comp if args.exp_comp > 0 else 1.0)
        ev = ev_at(idx, keys)
        exposure = base * (2.0 ** ev)
        rgb8 = np.clip(srgb_gamma(lin * exposure) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out = os.path.join(args.out,
                           os.path.splitext(os.path.basename(path))[0] + ".png")
        write_png(out, rgb8)
        clipped = 100.0 * float((lin * exposure > 1.0).mean())
        print("%s  frame %5d  EV %+.2f  exposure %.4g  clipped %.2f%%"
              % (os.path.basename(out), idx, ev, exposure, clipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
