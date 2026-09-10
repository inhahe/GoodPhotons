"""Score a render A/B per ROI, with the two traps that have actually bitten this project
built in as checks rather than as advice.

    python tools/roi_score.py <dir> --arms base,other[,other2] --seeds 16 \
        [--bands "name=y0:y1:x0:x1,..."] [--null <band>] [--cost <ratio>]

Files are expected as <dir>/<arm>_s<seed>.pfm.

WHY THIS EXISTS. Both traps below were written down in known-issues.md after they cost an
investigation, and both were then walked into AGAIN by hand-rolled scorers. A rule you have to
remember is not a control; a rule the tool applies for you is.

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
import argparse, os, sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--arms', required=True, help='comma-separated; the FIRST is the baseline')
    ap.add_argument('--seeds', type=int, default=16)
    ap.add_argument('--bands', default='', help='name=y0:y1:x0:x1,... in [0,1] fractions')
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
    ns = {k: v.shape[0] for k, v in ims.items()}
    if len(set(ns.values())) > 1:
        print(f'! arms have different seed counts {ns} -- variance columns are not matched')

    if a.bands:
        bands = {}
        for spec in a.bands.split(','):
            name, box = spec.split('=')
            y0, y1, x0, x1 = (float(v) for v in box.split(':'))
            bands[name] = (slice(int(y0 * H), int(y1 * H)), slice(int(x0 * W), int(x1 * W)))
    else:
        bands = {'top third': (slice(0, H // 3), slice(None)),
                 'middle': (slice(H // 3, 2 * H // 3), slice(None)),
                 'bottom third': (slice(2 * H // 3, H), slice(None))}
    bands['WHOLE FRAME'] = (slice(None), slice(None))

    base = arms[0]
    print(f'\nbaseline: {base}   seeds: {ns[base]}   cost ratio: {a.cost:.3f}\n')
    hdr = '%-20s' % 'band'
    for arm in arms[1:]:
        hdr += '%13s%13s%13s' % (f'{arm} var', 'equal-cost', f'{arm} bias')
    print(hdr)
    print('-' * len(hdr))

    warnings = []
    for name, sl in bands.items():
        row = '%-20s' % name
        for arm in arms[1:]:
            vb = tmean(ims[base][(slice(None),) + sl].var(axis=0, ddof=1))
            vo = tmean(ims[arm][(slice(None),) + sl].var(axis=0, ddof=1))
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
            sl = bands[a.null]
            for arm in arms[1:]:
                vb = tmean(ims[base][(slice(None),) + sl].var(axis=0, ddof=1))
                vo = tmean(ims[arm][(slice(None),) + sl].var(axis=0, ddof=1))
                r = vo / vb
                ok = abs(r - 1.0) < 0.05
                print('%s NULL CONTROL "%s" for %s: %.3fx%s' %
                      ('  ' if ok else '!!', a.null, arm, r,
                       '' if ok else '  -- the change cannot affect this band, so the rig is '
                                     'measuring something other than its label. Check that the '
                                     'arms are matched on sample count, not just on wall time.'))
    for w in warnings:
        print('!! ' + w)
    if not warnings and not a.null:
        print('\n(no --null band given: consider adding one, it is the cheapest check there is)')


if __name__ == '__main__':
    main()
