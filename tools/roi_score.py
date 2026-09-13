"""Score a render A/B per ROI, with the four traps that have actually bitten this project
built in as checks rather than as advice.

    python tools/roi_score.py <dir> --arms base,other[,other2] --seeds 16 \
        [--bands "name=y0:y1:x0:x1,..." | --rois <file.rois> [--only a,b]] \
        [--null <band>] [--cost <ratio>] [--list]

Files are expected as <dir>/<arm>_s<seed>.pfm.

WHY THIS EXISTS. Every trap below was written down in known-issues.md after it cost an
investigation, and the first two were then walked into AGAIN by hand-rolled scorers. A rule you
have to remember is not a control; a rule the tool applies for you is. Traps 3 and 4 live further
down, beside the code that enforces them: a rectangle is not a population (`maskbands`), and the
arms may be the same image (`main`).

TRAP 1 -- BIAS TESTED WITH A ROBUST STATISTIC.
A trimmed mean is not an unbiased estimator of a skewed distribution's mean. Two arms that
differ in their TAIL -- which is exactly what a variance change is -- therefore have different
trimmed means even when both are perfectly unbiased. Measured instance: an estimator change read
+0.181 % at 4.5 sigma on trimmed means and -0.089 % +/- 0.158 % on raw ones. The first reads as
a broken weight; the second is the truth.
  So: bias is reported on RAW means, always. The trimmed number is shown alongside, and if the
  two disagree by more than the raw standard error the tool says so and tells you which to
  believe. Variance, by contrast, IS scored on a trimmed statistic -- of the per-pixel variances,
  not of radiance -- because there a handful of firefly pixels really would decide the number.

TRAP 2 -- A NULL CONTROL THAT IS NOT 1.00.
Every A/B should contain a region the change provably cannot touch. If it moves, the rig is
measuring something other than its label and every other column is suspect. Measured instance:
a glossy-only estimator change read 0.905x on a diffuse band it could not affect -- the tell that
the arms had been compared at equal TIME with the sample counts thrown away. Pass --null <band>
and the tool checks it and refuses to be quiet about a failure.

COST. A per-sample variance win is not a win. Pass --cost <t_other/t_base> (measure it
INTERLEAVED by seed, never blocked by arm -- blocked runs of one such comparison drifted from
+10.1 % to +26.7 % on thermal/turbo alone) and the equal-cost column is variance x cost.
"""
import argparse, io, os, re, sys
import numpy as np


def readpfm(p):
    with open(p, 'rb') as f:
        hdr = f.readline().strip()
        nc = 3 if hdr == b'PF' else 1
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = map(int, line.split())
        sc = float(f.readline())
        d = np.frombuffer(f.read(w * h * nc * 4), dtype='<f4' if sc < 0 else '>f4')
        return d.reshape(h, w, nc)[::-1].astype(np.float64)


def lum(a):
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2] if a.shape[-1] == 3 \
        else a[..., 0]


def tmean(v, t=0.05):
    v = np.sort(np.asarray(v).ravel())
    k = int(len(v) * t)
    return v[k:len(v) - k].mean() if len(v) - 2 * k > 0 else v.mean()


def boxmask(H, W, ys, ye, xs, xe):
    """A rectangle, as the boolean mask every region is represented by internally."""
    m = np.zeros((H, W), dtype=bool)
    m[ys:ye, xs:xe] = True
    return m


def maskbands(path, want, H, W):
    """Per-material masks from an ftrace `-roi-mask` .pfm plus its .materials.txt legend.

    TRAP 3 -- A RECTANGLE IS NOT A POPULATION, AND FOR SOME MATERIALS NO RECTANGLE IS.
    Fur, foliage and any thin structure are sub-pixel and interleaved with whatever is
    behind them, so a box over them is mostly not them. Measured: on `fur_creature` the
    coat scores purity 0.44 over 28 disjoint regions, and `gallery_rain`'s hand-placed
    `creature` ROI -- the one every FURDIM number came from, labelled "fur coat" -- is
    75 % `cr_belly` skin and 25 % `cr_coat`. No amount of care in placing the box fixes
    that; the material simply is not rectangular.
    So the honest region for such a material is the set of pixels that actually show it,
    which `ftrace -roi-mask` writes and this reads. It is also the only ROI definition
    that transfers across scenes unchanged, which is what makes a cross-scene comparison
    mean anything: the population is "the pixels showing material X" in both, rather than
    a box here and the whole frame there.
    """
    mid = lum(readpfm(path))
    if mid.shape != (H, W):
        sys.exit(f'mask is {mid.shape[1]}x{mid.shape[0]} but the renders are {W}x{H} -- '
                 f'regenerate it with the same -r as the renders')
    names = {}
    try:
        for line in io.open(path + '.materials.txt', encoding='utf-8'):
            if line.startswith('#'):
                continue
            p = line.rstrip('\n').split('\t')
            if len(p) == 2:
                names[int(p[0])] = p[1]
    except OSError:
        sys.exit(f'no legend beside {path} -- expected {path}.materials.txt')
    out = {}
    for i, nm in sorted(names.items()):
        if want is not None and nm not in want:
            continue
        m = (np.rint(mid) == i)
        if m.sum() > 0:
            out[nm] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--arms', required=True, help='comma-separated; the FIRST is the baseline')
    ap.add_argument('--seeds', type=int, default=16)
    ap.add_argument('--bands', default='', help='name=y0:y1:x0:x1,... in [0,1] fractions')
    ap.add_argument('--rois', default='', help='a .rois file (format: name x0 y0 x1 y1)')
    ap.add_argument('--only', default='', help='comma-separated ROI/material names to score')
    ap.add_argument('--mask', default='', help='an ftrace -roi-mask .pfm: score each material '
                                               'on the pixels that actually show it')
    ap.add_argument('--list', action='store_true', help='print the parsed boxes and exit')
    ap.add_argument('--null', default='', help='band the change provably cannot affect')
    ap.add_argument('--cost', type=float, default=1.0, help='t_other / t_base, interleaved')
    a = ap.parse_args()

    arms = a.arms.split(',')
    ims = {}
    for arm in arms:
        fs = [f'{a.dir}/{arm}_s{s}.pfm' for s in range(1, a.seeds + 1)]
        fs = [f for f in fs if os.path.exists(f)]
        if not fs:
            sys.exit(f'no renders for arm {arm} in {a.dir}')
        ims[arm] = np.stack([lum(readpfm(f)) for f in fs])
    H, W = ims[arms[0]].shape[1:]

    # TRAP 4 -- THE ARMS ARE THE SAME IMAGE.
    # A flag that the renderer declined to honour produces a table of identical columns, which
    # reads as "the change is harmless" and is in fact "there was no change". Measured instance:
    # a radiance-cache cell-size sweep over a 32x range, plus a validation-off control, every
    # column identical to the digit -- because ftrace had printed
    #   [radcache] IGNORED: the GPU backward megakernel has no cache -- pass -device cpu
    # and the sweep ran on the GPU. The renderer said so plainly and the sweep was still run,
    # scored and nearly believed. A warning you have to read is not a control; this is.
    for k in range(1, len(arms)):
        a0, ak = ims[arms[0]], ims[arms[k]]
        if a0.shape == ak.shape and np.array_equal(a0, ak):
            sys.exit(f'!! ARMS "{arms[0]}" and "{arms[k]}" are BYTE-IDENTICAL. They are the same '
                     f'image, so every column below would be a comparison of a thing with itself. '
                     f'Check the render logs -- a flag the renderer declined to honour (wrong '
                     f'-device, an unsupported mode) is the usual cause. Refusing to score.')

    ns = {k: v.shape[0] for k, v in ims.items()}
    if len(set(ns.values())) > 1:
        print(f'! arms have different seed counts {ns} -- variance columns are not matched')

    want = set(f for f in a.only.split(',') if f) if a.only else None
    boxinfo = {}
    if a.mask:
        bands = maskbands(a.mask, want, H, W)
        if not bands:
            sys.exit(f'no materials from {a.mask} are visible (or --only matched none)')
    elif a.rois:
        # FIELD ORDER IS `name x0 y0 x1 y1` -- fractions of width/height, x FIRST. Written here
        # once because getting it wrong silently yields one-pixel boxes rather than an error:
        # neighbouring ROIs then report identical numbers and a bad rig looks like a clean result.
        bands = {}
        for line in io.open(a.rois, encoding='utf-8'):
            m = re.match(r'\s*([A-Za-z_][\w]*)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if not m or line.lstrip().startswith('#'): continue
            name = m.group(1)
            if want is not None and name not in want: continue
            x0, y0, x1, y1 = (float(m.group(i)) for i in (2, 3, 4, 5))
            ys, ye = int(y0 * H), int(y1 * H)
            xs, xe = int(x0 * W), int(x1 * W)
            if ye - ys < 1 or xe - xs < 1:
                print('! ROI "%s" is degenerate at %dx%d (%d x %d px) -- check the field order'
                      % (name, W, H, xe - xs, ye - ys))
                continue
            bands[name] = boxmask(H, W, ys, ye, xs, xe)
            boxinfo[name] = (ys, ye, xs, xe)
        if a.list:
            for n, (ys, ye, xs, xe) in boxinfo.items():
                print('%-16s rows %4d..%-4d cols %4d..%-4d  (%d x %d px)'
                      % (n, ys, ye, xs, xe, xe - xs, ye - ys))
            return
    elif a.bands:
        bands = {}
        for spec in a.bands.split(','):
            name, box = spec.split('=')
            y0, y1, x0, x1 = (float(v) for v in box.split(':'))
            bands[name] = boxmask(H, W, int(y0 * H), int(y1 * H), int(x0 * W), int(x1 * W))
    else:
        # Thirds of a frame are slabs, not regions: nothing in them corresponds to
        # anything in the scene, so a difference in one localises to nothing. They are a
        # last resort for "I have no region at all", and saying so is cheaper than a
        # reader assuming the rows mean more than they do.
        print('! no --mask, --rois or --bands: falling back to horizontal thirds, which are\n'
              '! slabs of the image and not regions of the SCENE. Prefer `ftrace -roi-mask`.')
        bands = {'top third': boxmask(H, W, 0, H // 3, 0, W),
                 'middle': boxmask(H, W, H // 3, 2 * H // 3, 0, W),
                 'bottom third': boxmask(H, W, 2 * H // 3, H, 0, W)}
    bands['WHOLE FRAME'] = np.ones((H, W), dtype=bool)

    base = arms[0]
    print(f'\nbaseline: {base}   seeds: {ns[base]}   cost ratio: {a.cost:.3f}\n')
    hdr = '%-20s%8s' % ('band', 'px')
    for arm in arms[1:]:
        hdr += '%13s%13s%13s' % (f'{arm} var', 'equal-cost', f'{arm} bias')
    print(hdr)
    print('-' * len(hdr))

    warnings = []
    for name, sl in bands.items():
        npx = int(sl.sum())
        # ROI SIZE IS PART OF THE RESULT. gallery_rain's `creature` is 6x6 px at 320x180 and
        # `alice_hair` 9x3 -- a trimmed mean over ~30 pixels is a far weaker number than the
        # column width suggests, and nothing else on the line says so.
        # An ROI with no signal in the BASELINE is a misapplied .rois file (wrong scene, wrong
        # camera) far more often than it is a legitimately black region -- and every ratio below
        # would come out NaN and be read as "no change". Say so instead.
        if not (ims[base][:, sl].mean() > 0.0):
            print('%-20s%8d   ! no signal in the baseline -- wrong scene or camera for this .rois?'
                  % (name, npx))
            continue
        row = '%-20s%8d' % (name, npx)
        for arm in arms[1:]:
            vb = tmean(ims[base][:, sl].var(axis=0, ddof=1))
            vo = tmean(ims[arm][:, sl].var(axis=0, ddof=1))
            r = vo / vb if vb > 0 else float('nan')
            # BIAS ON RAW MEANS (trap 1). Per-seed image means, so the standard error comes
            # from the seed spread, which is the only honest error bar available here.
            mb = np.array([im[sl].mean() for im in ims[base]])
            mo = np.array([im[sl].mean() for im in ims[arm]])
            d = mo.mean() / mb.mean() - 1
            se = np.sqrt(mb.var(ddof=1) / len(mb) + mo.var(ddof=1) / len(mo)) / abs(mb.mean())
            # ...and the trimmed one, only to detect the trap.
            tb = np.array([tmean(im[sl]) for im in ims[base]])
            to = np.array([tmean(im[sl]) for im in ims[arm]])
            dt = to.mean() / tb.mean() - 1
            if se > 0 and abs(dt - d) > se:
                warnings.append(
                    f'{name} / {arm}: trimmed bias {100*dt:+.3f}% vs RAW {100*d:+.3f}% '
                    f'(+/-{100*se:.3f}%). The arms differ in the TAIL, so the trimmed mean is '
                    f'not measuring bias -- believe the raw number.')
            row += '%12.3fx%12.3fx%9.3f%%+-%s' % (r, r * a.cost, 100 * d, ('%.3f' % (100 * se)))
        print(row)

    if a.null:
        print()
        if a.null not in bands:
            print(f'! --null names "{a.null}", which is not one of the bands: {list(bands)}')
        else:
            # THE NULL HAS TWO HALVES AND THEY FAIL FOR DIFFERENT REASONS. Variance and bias
            # are separate claims, and which one the null can speak to depends on what the
            # arms differ BY. A parameter that legitimately changes variance everywhere in
            # the frame -- sample count, gather radius, filter width -- makes the variance
            # null inapplicable by construction, not failed: measured instance, a gather
            # radius sweep read 0.278x / 0.074x / 0.013x / 0.004x on a floor strip 4-8 m from
            # the only wall, purely because a wider gather averages more photons. Reporting
            # that as "the rig is measuring something other than its label" is a false alarm
            # that would discredit a correct rig, so both halves are printed and the variance
            # half says plainly when it cannot be read.
            sl = bands[a.null]
            for arm in arms[1:]:
                vb = tmean(ims[base][:, sl].var(axis=0, ddof=1))
                vo = tmean(ims[arm][:, sl].var(axis=0, ddof=1))
                r = vo / vb if vb > 0 else float('nan')
                mb = np.array([im[sl].mean() for im in ims[base]])
                mo = np.array([im[sl].mean() for im in ims[arm]])
                d = mo.mean() / mb.mean() - 1
                se = np.sqrt(mb.var(ddof=1) / len(mb) + mo.var(ddof=1) / len(mo)) / abs(mb.mean())
                bias_ok = abs(d) <= max(2.0 * se, 0.005)
                var_ok = abs(r - 1.0) < 0.05
                print('%s NULL "%s" for %s: bias %+.3f%%+-%.3f%s' %
                      ('  ' if bias_ok else '!!', a.null, arm, 100 * d, 100 * se,
                       '' if bias_ok else '  -- this band CANNOT move, so the rig is measuring '
                                          'something other than its label; every other column '
                                          'is suspect.'))
                print('   %s   variance %.3fx%s' % (' ' if var_ok else '?', r,
                      '' if var_ok else '  -- only meaningful if the arms do NOT differ by '
                                        'something that changes noise everywhere (sample count, '
                                        'gather radius, filter width). If they do, read the bias '
                                        'line and ignore this one.'))
    for w in warnings:
        print('!! ' + w)
    if not warnings and not a.null:
        print('\n(no --null band given: consider adding one, it is the cheapest check there is)')


if __name__ == '__main__':
    main()
