"""Regression check: every mode that images through a physical lens agrees with mode R.

    python tools/check_physlens.py [--exe ftrace.exe] [--cpu-only]

A camera's `lens { }` is imaged two ways. Mode R traces camera rays out through the glass
(genLensRay, aimed at the exit pupil by the pupil table; R is also checked against itself with
the aiming off, FTRACE_LENS_FRONT, and across devices). The forward modes trace light in: A and B splat every vertex through a sampled
point of the front element (one estimator under a physical lens, so their rows match to the
digit; B's is there to catch it falling back to the pinhole), C catches what flies through
it, and D/J add that light image
to BDPT's camera paths under a MIS weight built from the lens's own camera density. Each
estimator carries its own normalisation (a cell's irradiance; BDPT's light image divided by
W*H against its spp normaliser; the catch's flux per cell area), and the device versions are
hand-ported twins in double precision. A slip in any of them shows up only as a region a
few per cent too bright or too dark, which no single render reveals. This renders the
same scenes in all of them and compares region means with mode R:

  * scenes/realcam.ftsl (a 50 mm f/2.8 achromat: planar rear stop) and a singlet copy of
    it (curved rear element, the case genLensRay samples by its clear cap), in A, B and D;
  * pbrt's double Gauss (scenes/realcam_dgauss.ftsl) stopped to f/8, in A and D: its front
    element is several times the beam the stop admits, where the pupil table's aiming, and
    BDPT's weighting of it, carry the image;
  * the achromat scene in a homogeneous fog, in D against R, and in J against J with the
    aiming turned off (mode J's merges on the camera's first segment weigh the aimed light
    image too, and J itself reads this fog a few per cent off R, lens or none);
  * the achromat scene under a visible D65 sky, in A (the sky imaged through the lens);
  * a uniform emitter panel in C (C is too photon-starved for the sphere scene at a
    check's budget, but a panel it can resolve);

each on the CPU, and on the GPU when one is present. The directly seen chrome ball is left
out of the A/B comparison: no forward splat can image a mirror through a lens or a pinhole.
`-checklens` pins the lens functions themselves; this pins what the modes do with them.
Exits 0 when every region agrees within its tolerance, 1 otherwise.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'scraps', '_physlens')
W, H = 128, 86
# Regions of realcam.ftsl's frame at 128x86 (rows, cols): floor, three diffuse spheres.
REGIONS = {'floor': (75, 86, 0, 128), 'green': (30, 55, 50, 75), 'red': (28, 60, 102, 125),
           'blue': (29, 45, 15, 35)}
# Region tolerance for A, B, D and the sky. A correct build stays within about 0.5% of R at
# these budgets. D's light image carries only part of each pixel under MIS, so an error in
# its scale is diluted: a light image 10% too bright moves D's regions by just 1-2.6%, which
# a looser bound would let through. The catch (C) is heavy-tailed and keeps 3%.
TOL = 0.01


def read_pfm(p):
    with open(p, 'rb') as f:
        hdr = f.readline().strip()
        ch = 3 if hdr == b'PF' else 1
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        a = np.fromfile(f, '<f4' if scale < 0 else '>f4').reshape(h, w, ch)
    a = np.flipud(a).astype(np.float64)
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722 if ch == 3 else a[..., 0]


def render(exe, scene, name, args, res=(W, H), env=None):
    out = os.path.join(OUT, name + '.png')
    cmd = [exe, '-in', scene, '-r', str(res[0]), str(res[1]), '-hdr', '-no-meter', '-seed', '3',
           '-o', out] + args
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                       env=dict(os.environ, **env) if env else None)
    pfm = out[:-4] + '.pfm'
    if r.returncode != 0 or not os.path.exists(pfm):
        print(f'  render failed: {" ".join(cmd)}\n{r.stdout[-800:]}{r.stderr[-800:]}')
        return None
    return read_pfm(pfm)


def write_scenes():
    src = open(os.path.join(ROOT, 'scenes', 'realcam.ftsl'), encoding='utf-8').read()
    singlet = src.replace('preset achromat', 'preset singlet').replace('fstop 2.8', 'fstop 4')
    sky = src.replace('look_at 0.0 0.35 2.4', 'look_at 0.0 0.9 2.4') + '\nlight env { spd preset:d65 }\n'
    panel = ('scene { units meters  spectral 360 830 5 }\n'
             'light area { origin -0.6 0.0 2.0   u 0.9 0 0   v 0 0.8 0   normal 0 0 -1   spd preset:bb6500 }\n'
             'camera "real" { eye 0.0 0.4 -1.6   look_at 0.0 0.4 2.4   up 0 1 0   focus 2.4\n'
             '    film { res 64 43   format full-frame }\n'
             '    lens { preset achromat  focal 50  fstop 2.8  glass BK7 } }\n')
    # pbrt's double Gauss stopped to f/8: its front element is ~6x the beam the stop admits,
    # the case the pupil table aims for (and the one a uniform front-element sampler wastes
    # ~95% of its samples on).
    dgauss = open(os.path.join(ROOT, 'scenes', 'realcam_dgauss.ftsl'), encoding='utf-8').read()
    assert '8.55   stop' in dgauss
    dg8 = dgauss.replace('8.55   stop', '2.14   stop')
    # Fog through the lens: mode J's merges along the camera's first segment weigh the light
    # image's pick of the camera vertex from the merge point (lensAimRatio), which nothing
    # surface-only exercises.
    fog = src + '\nmedium { sigma_t 0.15   albedo 0.8   g 0.0 }\n'
    paths = {}
    for name, text in (('singlet', singlet), ('sky', sky), ('panel', panel), ('dg8', dg8), ('fog', fog)):
        p = os.path.join(OUT, name + '.ftsl')
        open(p, 'w', encoding='utf-8', newline='').write(text)
        paths[name] = p
    paths['achromat'] = os.path.join(ROOT, 'scenes', 'realcam.ftsl')
    return paths


def compare(label, ref, img, regions, tol):
    ok = True
    parts = []
    for k, (y0, y1, x0, x1) in regions.items():
        q = img[y0:y1, x0:x1].mean() / max(ref[y0:y1, x0:x1].mean(), 1e-300)
        good = abs(q - 1.0) <= tol
        ok = ok and good
        parts.append(f'{k} {q:.4f}{"" if good else " !"}')
    print(f'  {label:28s} {"ok  " if ok else "FAIL"}  ' + '  '.join(parts))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default=os.path.join(ROOT, 'ftrace.exe'))
    ap.add_argument('--cpu-only', action='store_true')
    a = ap.parse_args()
    exe = os.path.abspath(a.exe)
    os.makedirs(OUT, exist_ok=True)
    scenes = write_scenes()
    devices = ['cpu'] if a.cpu_only else ['cpu', 'gpu']
    allok = True
    for lens in ('achromat', 'singlet'):
        print(f'{lens}:')
        refdev = 'gpu' if 'gpu' in devices else 'cpu'
        ref = render(exe, scenes[lens], f'{lens}_R', ['-mode', 'R', '-spp', '4096', '-device', refdev])
        if ref is None:
            return 1
        # R itself aims its camera rays at the exit pupil: the other device's R, and R with the
        # aiming off (FTRACE_LENS_FRONT), must agree with the reference.
        for dev, env, label in ((('cpu' if refdev == 'gpu' else None), None, 'R on the CPU'),
                                (refdev, {'FTRACE_LENS_FRONT': '1'}, f'R on the {refdev.upper()}, unaimed')):
            if dev is None:
                continue
            img = render(exe, scenes[lens], f'{lens}_R_{dev}{"_front" if env else ""}',
                         ['-mode', 'R', '-spp', '4096', '-device', dev], env=env)
            allok = (img is not None and compare(label, ref, img, REGIONS, TOL)) and allok
        for dev in devices:
            for mode, args in (('A', ['-n', '200000000']), ('B', ['-n', '200000000']),
                               ('D', ['-spp', '4096'])):
                img = render(exe, scenes[lens], f'{lens}_{mode}_{dev}', ['-mode', mode, '-device', dev] + args)
                if img is None:
                    allok = False
                    continue
                allok = compare(f'{mode} on the {dev.upper()}', ref, img, REGIONS, TOL) and allok
    print('double Gauss at f/8 (front element ~6x the beam):')
    refdev = 'gpu' if 'gpu' in devices else 'cpu'
    ref = render(exe, scenes['dg8'], 'dg8_R', ['-mode', 'R', '-spp', '16384', '-device', refdev])
    if ref is None:
        return 1
    # Unaimed, R's camera rays use ~5% of the rear element at f/8, hence 16x the samples.
    if refdev == 'gpu':
        img = render(exe, scenes['dg8'], 'dg8_R_cpu', ['-mode', 'R', '-spp', '4096', '-device', 'cpu'])
        allok = (img is not None and compare('R on the CPU', ref, img, REGIONS, TOL)) and allok
    img = render(exe, scenes['dg8'], f'dg8_R_{refdev}_front', ['-mode', 'R', '-spp', '65536', '-device', refdev],
                 env={'FTRACE_LENS_FRONT': '1'})
    allok = (img is not None and compare(f'R on the {refdev.upper()}, unaimed', ref, img, REGIONS, TOL)) and allok
    for dev in devices:
        for mode, args in (('A', ['-n', '200000000']), ('D', ['-spp', '4096'])):
            img = render(exe, scenes['dg8'], f'dg8_{mode}_{dev}', ['-mode', mode, '-device', dev] + args)
            allok = (img is not None and compare(f'{mode} on the {dev.upper()}', ref, img, REGIONS, TOL)) and allok
    # Mode J is held to itself rather than to R: its merges read this fog 3-4% low with no lens
    # at all (known-issues.md J-FOG-REALCAM), which would hide anything the lens adds. With the
    # pupil table turned off (FTRACE_LENS_FRONT) the light image samples the whole front
    # element and lensAimRatio is 1 everywhere, so the two renders differ only by what aiming
    # changes -- in J, the merges' weighing of it on the camera's first segment.
    print('fog through the lens (every mode at -max-bounce 8):')
    deep = ['-max-bounce', '8']
    ref = render(exe, scenes['fog'], 'fog_R', ['-mode', 'R', '-spp', '8192', '-device',
                                              'gpu' if 'gpu' in devices else 'cpu'] + deep)
    if ref is None:
        return 1
    for dev in devices:
        img = render(exe, scenes['fog'], f'fog_D_{dev}', ['-mode', 'D', '-spp', '4096', '-device', dev] + deep)
        allok = (img is not None and compare(f'D on the {dev.upper()}', ref, img, REGIONS, TOL)) and allok
        jargs = ['-mode', 'J', '-spp', '1024', '-device', dev] + deep
        front = render(exe, scenes['fog'], f'fog_Jfront_{dev}', jargs, env={'FTRACE_LENS_FRONT': '1'})
        img = render(exe, scenes['fog'], f'fog_J_{dev}', jargs)
        allok = (img is not None and front is not None and
                 compare(f'J on the {dev.upper()}, vs unaimed', front, img, REGIONS, TOL)) and allok
    print('sky through the lens:')
    sky_regions = {'sky': (3, 15, 38, 90), 'corner': (0, 8, 0, 15)}
    ref = render(exe, scenes['sky'], 'sky_R', ['-mode', 'R', '-spp', '4096', '-device',
                                              'gpu' if 'gpu' in devices else 'cpu'])
    if ref is None:
        return 1
    for dev in devices:
        img = render(exe, scenes['sky'], f'sky_A_{dev}', ['-mode', 'A', '-n', '200000000', '-device', dev])
        allok = (img is not None and compare(f'A on the {dev.upper()}', ref, img, sky_regions, TOL)) and allok
    print('catch (mode C) on a uniform panel:')
    panel_regions = {'centre': (14, 29, 18, 46)}
    ref = render(exe, scenes['panel'], 'panel_R', ['-mode', 'R', '-spp', '4096', '-device', 'cpu'], (64, 43))
    if ref is None:
        return 1
    for dev in devices:
        img = render(exe, scenes['panel'], f'panel_C_{dev}',
                     ['-mode', 'C', '-n', '3000000000' if dev == 'gpu' else '1000000000', '-device', dev], (64, 43))
        allok = (img is not None and compare(f'C on the {dev.upper()}', ref, img, panel_regions, 0.03)) and allok
    print('PASS' if allok else 'FAIL')
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
