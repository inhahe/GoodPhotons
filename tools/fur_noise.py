# Does a FORWARD tracer pay a noise penalty on fur that a BACKWARD tracer does not?
#
# The claim being tested (TODO.md P2): sub-pixel fiber variance is a *point-sampling*
# pathology of backward tracing — a camera ray is a binary hit/miss on something 1/50 of a
# pixel wide. A forward tracer splats accumulated flux into the film, so a pixel is an area
# average by construction and its relative error should be set by photons-per-pixel (i.e.
# brightness), not by how much geometry sits behind the pixel.
#
# So the discriminator is not "which mode is noisier" — it is the RATIO
#
#       relative error on FUR  /  relative error on the bare WALL
#
# measured within each mode. If fur is intrinsically bad for forward tracing, mode B's ratio
# should be much worse than mode R's. If the claim is right, B's ratio should be no worse.
#
# METHOD. Both modes are unbiased and converge to the same image, so each short run is
# scored against a long run of the OTHER mode. That cross-pairing is deliberate: ftrace
# seeds deterministically, so a long run of the SAME mode contains the short run's own
# samples and the two would be correlated, flattering it.
#
# The reference is not noise-free either, and Monte-Carlo error falls as 1/sqrt(t) *per
# region*, so the reference's own per-region error is estimated from the short run of the
# same mode, scaled by sqrt(t_short/t_ref), and removed in quadrature.
#
#   python tools/fur_noise.py
#
# Reproducing the inputs (they land in png/furnoise/, which is git-ignored):
#   ftrace -in scenes/fur_creature.ftsl -mode B -time   30 -o png/furnoise/probe_B.png -window
#   ftrace -in scenes/fur_creature.ftsl -mode R -time   30 -o png/furnoise/probe_R.png -window
#   ftrace -in scenes/fur_creature.ftsl -mode B -time 2400 -o png/furnoise/ref_B.png   -window
#   ftrace -in scenes/fur_creature.ftsl -mode R -time  900 -o png/furnoise/ref_R.png   -window
import math, os, struct, sys, zlib


def readpng(p):
    """Minimal 8-bit PNG decoder — inlined so this tool has no dependencies at all."""
    d = open(p, 'rb').read()
    assert d[:8] == b'\x89PNG\r\n\x1a\n', p
    i = 8; w = h = bd = ct = None; idat = b''
    while i < len(d):
        ln = struct.unpack('>I', d[i:i + 4])[0]
        typ = d[i + 4:i + 8]; data = d[i + 8:i + 8 + ln]; i += 12 + ln
        if typ == b'IHDR':   w, h, bd, ct = struct.unpack('>IIBB', data[:10])
        elif typ == b'IDAT': idat += data
        elif typ == b'IEND': break
    raw = zlib.decompress(idat)
    nch = {0: 1, 2: 3, 4: 2, 6: 4}[ct]; bpp = nch * (bd // 8); stride = w * bpp
    out = bytearray(); prev = bytearray(stride); j = 0
    for y in range(h):
        f = raw[j]; j += 1; line = bytearray(raw[j:j + stride]); j += stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if f == 1:   line[x] = (line[x] + a) & 0xff
            elif f == 2: line[x] = (line[x] + b) & 0xff
            elif f == 3: line[x] = (line[x] + (a + b) // 2) & 0xff
            elif f == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x] + pr) & 0xff
        out += line; prev = line
    return w, h, nch, bd, bytes(out)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D    = os.path.join(ROOT, "png", "furnoise")

T_SHORT_B, T_REF_B = 30.0, 2400.0
T_SHORT_R, T_REF_R = 30.0,  900.0


def load(name):
    w, h, nch, bd, raw = readpng(os.path.join(D, name))
    assert bd == 8 and nch >= 3, (name, bd, nch)
    # sRGB -> linear. Relative error has to be measured on the linear signal; the sRGB
    # curve compresses highlights and would understate noise in the bright wall.
    lut = []
    for i in range(256):
        c = i / 255.0
        lut.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    px = [[0.0] * w for _ in range(h)]
    sat = [[0.0] * w for _ in range(h)]
    for y in range(h):
        row = raw[y * w * nch:(y + 1) * w * nch]
        for x in range(w):
            r, g, b = row[x * nch], row[x * nch + 1], row[x * nch + 2]
            px[y][x] = 0.2126 * lut[r] + 0.7152 * lut[g] + 0.0722 * lut[b]
            mx, mn = max(r, g, b), min(r, g, b)
            sat[y][x] = 0.0 if mx == 0 else (mx - mn) / mx
    return w, h, px, sat


def masks(w, h, ref, sat):
    """Fur vs bare-room, segmented by SATURATION, not by a hand-drawn box.

    The coat is warm brown (`rgb 0.40 0.25 0.13`); every surface in the room is neutral
    grey. So chroma separates them with no hand-placed rectangles to argue about, and it
    stays correct however the camera moves. Dark pixels are dropped from both masks —
    at 8 bits the quantisation floor would dominate the relative error there.
    """
    fur = [[False] * w for _ in range(h)]
    room = [[False] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if ref[y][x] < 0.02:
                continue
            if sat[y][x] > 0.18:
                fur[y][x] = True
            elif sat[y][x] < 0.05:
                room[y][x] = True
    return fur, room


def relerr(test, ref, mask, w, h):
    n = 0
    se = 0.0
    mean = 0.0
    for y in range(h):
        for x in range(w):
            if not mask[y][x]:
                continue
            d = test[y][x] - ref[y][x]
            se += d * d
            mean += ref[y][x]
            n += 1
    if n == 0:
        return float("nan"), 0
    return math.sqrt(se / n) / (mean / n), n


def main():
    w, h, refR, satR = load("ref_R.png")            # least noisy image -> drives the masks
    _, _, refB, _ = load("ref_B.png")
    _, _, shortB, _ = load("probe_B.png")
    _, _, shortR, _ = load("probe_R.png")

    fur, room = masks(w, h, refR, satR)
    nf = sum(sum(1 for v in r if v) for r in fur)
    nr = sum(sum(1 for v in r if v) for r in room)
    print("masks: %d fur px, %d bare-room px (of %d)" % (nf, nr, w * h))

    # Raw (reference noise still included).
    rows = []
    for label, short, ref, ts, tr in (
            ("B (forward)",  shortB, refR, T_SHORT_B, T_REF_R),
            ("R (backward)", shortR, refB, T_SHORT_R, T_REF_B)):
        ef, _ = relerr(short, ref, fur, w, h)
        er, _ = relerr(short, ref, room, w, h)
        rows.append((label, ef, er, ts, tr))

    # Remove the reference's own per-region noise in quadrature. The reference is a long run
    # of the OTHER mode, so its per-region error is that mode's own short-run error scaled by
    # sqrt(t_short / t_ref) -- and that short-run error is the *other* row of this table.
    print("\n%-14s %10s %10s %8s   %10s %10s %8s" %
          ("mode", "fur", "room", "ratio", "fur(corr)", "room(corr)", "ratio"))
    corr = {}
    for i, (label, ef, er, ts, tr) in enumerate(rows):
        other = rows[1 - i]
        k = math.sqrt(other[3] / tr)          # reference error = other mode's short error * k
        rf = max(ef * ef - (other[1] * k) ** 2, 0.0) ** 0.5
        rr = max(er * er - (other[2] * k) ** 2, 0.0) ** 0.5
        corr[label] = (rf, rr)
        print("%-14s %9.2f%% %9.2f%% %8.2fx   %9.2f%% %9.2f%% %8.2fx" %
              (label, 100 * ef, 100 * er, ef / er, 100 * rf, 100 * rr,
               rf / rr if rr > 0 else float("nan")))

    b = corr["B (forward)"]
    r = corr["R (backward)"]
    print("\nfur-vs-room error ratio:  forward %.2fx,  backward %.2fx" %
          (b[0] / b[1], r[0] / r[1]))
    print("The claim under test is that the FORWARD ratio is not the worse of the two.")

    # --- and now the MECHANISM, which is the part actually worth knowing ---------------
    # In a forward tracer a pixel's relative error is 1/sqrt(photons landing in it), and the
    # photons landing in it are proportional to its flux. So the forward fur/room error ratio
    # is PREDICTED, with no free parameters, by the brightness ratio alone:
    #
    #        err_fur / err_room  =  sqrt( L_room / L_fur )
    #
    # If that prediction lands, the fur penalty in forward mode is *entirely* "fur is a dark
    # brown surface in shadow", and nothing to do with strands being sub-pixel. A backward
    # tracer has no such law — its per-pixel variance is set by the spread of path
    # throughputs, which is exactly where binary hit/miss on a 1/50-pixel fiber shows up.
    def meanlum(img, mask):
        s = n = 0
        for y in range(h):
            for x in range(w):
                if mask[y][x]:
                    s += img[y][x]; n += 1
        return s / n
    lf, lr = meanlum(refR, fur), meanlum(refR, room)
    pred = math.sqrt(lr / lf)
    print("\nmean linear luminance: fur %.4f, room %.4f  -> flux-only prediction %.2fx"
          % (lf, lr, pred))
    print("  forward  observed %.2fx  vs predicted %.2fx   (%+.0f%%)"
          % (b[0] / b[1], pred, 100 * ((b[0] / b[1]) / pred - 1)))
    print("  backward observed %.2fx  vs the same %.2fx    (%+.0f%%)  <- no reason it should fit"
          % (r[0] / r[1], pred, 100 * ((r[0] / r[1]) / pred - 1)))


if __name__ == "__main__":
    main()
