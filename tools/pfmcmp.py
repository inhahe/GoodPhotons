"""HDR compare two .pfm renders: overall energy bias + relative RMS after an NxN
box downsample (so per-pixel Monte-Carlo noise averages out and what is left is
structure). Usage: rcstat.py ref.pfm test.pfm [box=8]"""
import sys
from pfm import read_pfm_flat  # noqa: E402


def lum(w, h, c, d):
    return [(d[i * c] + d[i * c + 1] + d[i * c + 2]) / 3.0 if c == 3 else d[i]
            for i in range(w * h)]


def down(w, h, a, B):
    W, H = w // B, h // B
    o = [0.0] * (W * H)
    for y in range(H):
        for x in range(W):
            s = 0.0
            for j in range(B):
                for i in range(B):
                    s += a[(y * B + j) * w + (x * B + i)]
            o[y * W + x] = s / (B * B)
    return W, H, o


B = int(sys.argv[3]) if len(sys.argv) > 3 else 8
w, h, c, a = read_pfm_flat(sys.argv[1])
_, _, c2, b = read_pfm_flat(sys.argv[2])
A, Bl = lum(w, h, c, a), lum(w, h, c2, b)
mA = sum(A) / len(A)
mB = sum(Bl) / len(Bl)
W, H, dA = down(w, h, A, B)
_, _, dB = down(w, h, Bl, B)
n = 0
s2 = 0.0
mx = 0.0
mxat = None
for i in range(len(dA)):
    if dA[i] <= 1e-30:
        continue
    r = dB[i] / dA[i] - 1.0
    s2 += r * r
    n += 1
    if abs(r) > mx:
        mx, mxat = abs(r), (i % W, i // W)
rms = (s2 / max(1, n)) ** 0.5
print(f"ref={sys.argv[1]}")
print(f"test={sys.argv[2]}")
print(f"  energy bias  {100.0 * (mB / mA - 1.0):+.2f}%   (mean {mA:.5g} -> {mB:.5g})")
print(f"  rel-RMS @{B}x{B}  {100.0 * rms:.2f}%   max|rel| {100.0 * mx:.1f}% at {mxat}")
