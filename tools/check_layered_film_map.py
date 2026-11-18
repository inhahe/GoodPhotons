"""Regression check: a layered coat's thin film takes a pattern for its thickness map, and a scalar map
that names neither a pattern nor a texture is refused.

    python -u tools/check_layered_film_map.py [--out <dir>] [--exe ftrace.exe]

FTSL.md: `film_thickness_map` takes `pattern:<name>` or `texture:<name>`, a 0..1 profile of
`film_thickness`. A `layered` material's `coat { reflectance thinfilm ... }` bound only a texture
there, and dropped a pattern without a word (LAYERED-FILM-PATTERN). It requires:

  * a layered ball whose coat is 600 nm thick under a map that is everywhere 0.25 renders as the
    same ball with a 150 nm coat and no map, every pixel within 2e-3 -- a mapped coat's body reads
    its escape (T-bar) from rows across the map's range, interpolated at 0.25, where the unmapped
    one has a row at 150 nm exactly -- and not as the unmapped 600 nm one (some pixel 10 % off): in
    mode R on the CPU; and in mode R and mode D on the GPU (its connections read the film through
    the rebuilt vertex, DVertHitState) within 1e-2, the interpolated rows parting the device's paths
    a little more (4.3e-3 and 2.7e-3 measured), where a map left out reads 1.8 off. Each render is
    of one seed, whose coat coins the equal reflectances keep in step;
  * a `thinfilm` material's mapped film byte for byte as its 150 nm one in mode R on the CPU (it
    always bound a pattern, and reads no rows: the control);
  * `film_thickness_map` and a mix's `weight_map` given a value that is neither a pattern nor a
    texture (a misspelt `patern:`) stop the load, naming the key, and so do `reflect_map`,
    `transmit_map` and `emit_map`, which take a pattern only;
  * a map that names another material's property holding no map (`film_thickness_map
    other.film_thickness`, a constant there) loads, with no map: one that follows a source with
    none is none.

Exits 1 on any failure.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from pfm import image_diff, read_pfm  # noqa: E402


def scene(material, extra=""):
    return NL.join([
        "scene { units meters }",
        'pattern "quarter" { expr "0.25" }',
        'material "white" { type diffuse reflect 0.7 }',
        'material "base" { type diffuse reflect 0.6 }',
        extra,
        material,
        "quad { origin -2 0 -2  u 0 0 4  v 4 0 0  material white }",
        "quad { origin -2 0 -2  u 4 0 0  v 0 3 0  material white }",
        "light area { origin -0.4 2.9 -0.4  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  spd preset:bb6500  power 60 }",
        "sphere { center 0 0.6 0  radius 0.6  material film }",
        'camera "c" { eye 0 1.2 3.2  look_at 0 0.6 0  up 0 1 0  fov_y 40  film { res 64 48 } }',
    ]) + NL


def layered(thickness, map_word=None):
    m = (" film_thickness_map %s" % map_word) if map_word else ""
    return ('material "film" { type layered  ior 1.5  coat { reflectance thinfilm  film_ior 1.3'
            '  film_thickness %g%s }  layer "base" 1.0 }' % (thickness, m))


def thinfilm(thickness, map_word=None):
    m = (" film_thickness_map %s" % map_word) if map_word else ""
    return ('material "film" { type thinfilm  ior 1.5  film_ior 1.3  film_thickness %g%s  substrate_k 2 }'
            % (thickness, m))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_layered_film_map"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True

    def run(tag, text, mode="R", device="cpu"):
        path = os.path.join(out, tag + ".ftsl")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        png = os.path.join(out, tag + ".png")
        pfm = png[:-4] + ".pfm"
        for f in (png, pfm):
            if os.path.exists(f):
                os.remove(f)
        p = subprocess.run([exe, "-in", path, "-mode", mode, "-device", device, "-spp", "4", "-seed", "1", "-hdr",
                            "-no-meter", "-window-min", "-o", png], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        log = p.stdout + p.stderr
        ran = p.returncode == 0 and os.path.exists(pfm) and (device == "cpu" or "-> GPU" in log)
        return p.returncode, log, (read_pfm(pfm, top_down=True) if ran else None)

    for kind, make, tol, mode, device in (("layered", layered, 2e-3, "R", "cpu"), ("layered", layered, 1e-2, "R", "gpu"),
                                          ("layered", layered, 1e-2, "D", "gpu"), ("thinfilm", thinfilm, 0.0, "R", "cpu")):
        tag = "%s_%s_%s" % (kind, mode, device)
        _, _, mapped = run(tag + "_mapped", scene(make(600, "pattern:quarter")), mode, device)
        _, _, thin = run(tag + "_150", scene(make(150)), mode, device)
        _, _, thick = run(tag + "_600", scene(make(600)), mode, device)
        if mapped is None or thin is None or thick is None:
            ok = False
            print("FAIL %s, mode %s on the %s: a render failed or left the device" % (kind, mode, device.upper()),
                  flush=True)
            continue
        r150, _ = image_diff(mapped, thin)
        r600, _ = image_diff(mapped, thick)
        good = r150 <= tol and r600 > 0.1
        ok &= good
        print("%-4s %s, mode %s on the %s: a 600 nm film mapped to 0.25 renders as a 150 nm one   %.2g of the "
              "pixel off it (%.2g off the unmapped 600 nm)" % ("ok" if good else "FAIL", kind, mode, device.upper(),
                                                              r150, r600), flush=True)

    bad = [("film_thickness_map", scene(layered(600, "patern:quarter")), True),
           ("film_thickness_map", scene(thinfilm(600, "patern:quarter")), True),
           ("weight_map", scene('material "film" { type mix  layer "base" 0.5  layer "white" 0.5'
                                '  weight_map patern:quarter }'), True),
           ("reflect_map", scene('material "film" { type diffuse reflect 0.6  reflect_map patern:quarter }'), False),
           ("transmit_map", scene('material "film" { type translucent reflect 0.3  transmit 0.4'
                                  '  transmit_map patern:quarter }'), False),
           ("emit_map", scene('material "film" { type diffuse reflect 0.1  emit blackbody 5200'
                              '  emit_map patern:quarter }'), False)]
    for i, (key, text, tex) in enumerate(bad):
        rc, log, _ = run("bad_%d" % i, text)
        says = "pattern:<name> or texture:<name>" if tex else "`%s` takes pattern:<name>" % key
        good = rc != 0 and key in log and says in log
        ok &= good
        print("%-4s `%s patern:quarter` stops the load, naming the key%s"
              % ("ok" if good else "FAIL", key, "" if good else " (exit %d)" % rc), flush=True)
    other = 'material "other" { type thinfilm  ior 1.5  film_ior 1.3  film_thickness 150  substrate_k 2 }'
    rc, log, img = run("propref", scene(layered(600, "other.film_thickness"), extra=other))
    _, _, plain = run("propref_none", scene(layered(600), extra=other))
    good = rc == 0 and img is not None and plain is not None and image_diff(img, plain)[0] == 0.0
    ok &= good
    print("%-4s a film map naming a constant property loads, and renders as no map%s"
          % ("ok" if good else "FAIL", "" if good else " (exit %d): %s" % (rc, log[-300:])), flush=True)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
