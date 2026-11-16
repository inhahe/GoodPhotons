"""Regression check: the solids each light is born inside and each camera starts inside, as ftrace reports them.

    python tools/check_enclosures.py [--out png/check_enclosures] [--exe ftrace.exe]

BIRTH-EMPTY-STACK and CAMERA-EMPTY-STACK seed a walk's dielectric stack from the closed solids around
its start (Scene::enclosureAt), and ftrace says at load which lights are born inside a solid and which
cameras start inside one. Each case below is a small scene loaded for a one-sample render, and the
check reads those lines:
  * a tube light in air beside a glass ball that holds the world origin is born in air, and one
    inside a glass ball away from the origin is born inside it (a tube was probed at the origin);
  * with `-no-env`, an environment declared before the lamps leaves each lamp its own solid (the
    stacks were not renumbered when the environment was removed);
  * a window of one dielectric quad encloses neither the lamp nor the camera behind it (an open sheet
    bounds no inside);
  * a lamp inside a ball of a mix whose part is glass is born inside the glass;
  * a camera under water starts inside it, and one in air beside it does not;
  * a lamp inside a glowing ball inside glass (a glow's probes are its surface, not its box) is born
    inside the glass, wholly;
  * a lamp in the air cavity of a hollow glass box starts in air, and one in an inward-wound closed
    glass box starts outside the glass, as the walks read a face's side off its winding -- and so does
    one just under that box (a ray leaving it at its bottom enters it again at its far side);
  * a lamp in a box of water whose bottom shares its edges with another quad starts in the water (a
    solid's faces are a closed piece of their own), as does one in water whose surface, one quad, is
    closed by white walls (a solid's faces that close nothing alone are pooled with the rest), near its
    surface and deep down by its floor (the walls bound the water as its surface does), and one just
    under a wide shallow pool's floor does not;
  * a lamp in a glass ball made as an isosurface starts inside it (build()'s marking sets an
    isosurface's side);
  * a two-sided lamp lying on a glass slab is reported for its back face, born inside the glass, and
    not for its front.
Exits 1 on any failure.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)

HEAD = """scene { units meters }
material "white" { type diffuse reflect 0.7 }
material "glass" { type dielectric  ior 1.5  absorb 0.5 }
material "water" { type dielectric  ior 1.33  absorb 0.3 }
quad { origin -4 0 -4  u 0 0 8  v 8 0 0  material white }
camera "c" { eye 0 1 5  look_at 0 1 0  up 0 1 0  fov_y 50  film { res 16 12 } }
"""
LAMP = "light area { origin %s  u 0.1 0 0  v 0 0 0.1  normal 0 -1 0  spd preset:bb6500  power 10 }"
# Six quads closing a box of glass, every one wound to face in (a quad's normal is u x v).
INWARD_BOX = [
    "quad { origin -0.5 0.5 -0.5  u 0 0 1  v 1 0 0  material glass }",
    "quad { origin -0.5 1.5 -0.5  u 1 0 0  v 0 0 1  material glass }",
    "quad { origin -0.5 0.5 -0.5  u 0 1 0  v 0 0 1  material glass }",
    "quad { origin 0.5 0.5 -0.5  u 0 0 1  v 0 1 0  material glass }",
    "quad { origin -0.5 0.5 -0.5  u 1 0 0  v 0 1 0  material glass }",
    "quad { origin -0.5 0.5 0.5  u 0 1 0  v 1 0 0  material glass }",
]
# A slab of glass, 0.5 thick with its top at y = 1, its six quads wound to face out.
GLASS_SLAB = [
    "quad { origin -1 0.5 -1  u 2 0 0  v 0 0 2  material glass }",
    "quad { origin -1 1.0 -1  u 0 0 2  v 2 0 0  material glass }",
    "quad { origin -1 0.5 -1  u 0 0 2  v 0 0.5 0  material glass }",
    "quad { origin 1 0.5 -1  u 0 0.5 0  v 0 0 2  material glass }",
    "quad { origin -1 0.5 -1  u 0 0.5 0  v 2 0 0  material glass }",
    "quad { origin -1 0.5 1  u 2 0 0  v 0 0.5 0  material glass }",
]

# name: (scene body, extra args, {pattern: should it appear})
CASES = {
    "tube_beside_origin_ball": (
        "sphere { center 0 0 0  radius 0.5  material glass }" + NL +
        "light cylinder { center 0 2 0  axis 1 0 0  length 0.6  radius 0.05  spd preset:bb6500  power 10 }",
        [], {r"\[light\] light 0 \(cylinder[^)]*\) is born inside": False}),
    "tube_in_ball": (
        "sphere { center 1.5 2 0  radius 0.6  material glass }" + NL +
        "light cylinder { center 1.5 2 0  axis 1 0 0  length 0.5  radius 0.05  spd preset:bb6500  power 10 }",
        [], {r"\[light\] light 0 \(cylinder[^)]*\) is born inside 'glass'": True}),
    "noenv_order": (
        "light env { spd preset:d65 }" + NL +
        "sphere { center 1 1 0  radius 0.4  material glass }" + NL +
        LAMP % "0.95 1.0 -0.05" + NL + LAMP % "-1.05 2.0 -0.05",
        ["-no-env"], {r"\[light\] light 0 \(area at 0\.95 1 -0\.05\) is born inside 'glass'": True,
                      r"\[light\] light 1 \([^)]*\) is born inside": False}),
    "open_pane": (
        "quad { origin -3 0 -1  u 6 0 0  v 0 4 0  material glass }" + NL + LAMP % "-0.05 2.0 2.0",
        [], {r"\[light\] light 0 \([^)]*\) is born inside": False, r"\[camera\]": False}),
    "mix_ball": (
        "material \"m\" { type mix  layer \"glass\" 0.9  layer \"white\" 0.1 }" + NL +
        "sphere { center 0 1 0  radius 0.5  material m }" + NL + LAMP % "-0.05 1.0 -0.05",
        [], {r"\[light\] light 0 \([^)]*\) is born inside 'glass'": True}),
    "camera_in_water": (
        "sphere { center 0 1 5  radius 1.0  material water }" + NL + LAMP % "-0.05 3.0 -0.05",
        [], {r"\[camera\] 'c' starts inside 'water'": True}),
    "camera_beside_water": (
        "sphere { center 0 1 1  radius 1.0  material water }" + NL + LAMP % "-0.05 3.0 -0.05",
        [], {r"\[camera\]": False}),
    "glow_in_glass": (
        # A patterned glowing ball is a Glow, its box's corners outside the glass around it.
        "pattern \"bands\" { expr \"0.5+0.5*sin(20*x)\" }" + NL +
        "material \"ember\" { type diffuse  reflect 0.1  emit preset:bb6500  emit_map pattern:bands }" + NL +
        "sphere { center 0 1 0  radius 0.6  material glass }" + NL +
        "sphere { center 0 1 0  radius 0.45  material ember }" + NL + LAMP % "-0.05 3.0 -0.05",
        [], {r"\[light\] light \d+ \(glow[^)]*\) is born inside 'glass'": True,
             r"straddles": False}),
    "hollow_box": (
        # A glass box with an air cavity: an inner box wound to face the cavity. A lamp in the cavity is
        # outside the glass, as the walks read it.
        NL.join(['quad { origin -0.6 0.4 -0.6  u 1.2 0 0  v 0 0 1.2  material glass }', 'quad { origin -0.6 1.6 -0.6  u 0 0 1.2  v 1.2 0 0  material glass }', 'quad { origin -0.6 0.4 -0.6  u 0 0 1.2  v 0 1.2 0  material glass }', 'quad { origin 0.6 0.4 -0.6  u 0 1.2 0  v 0 0 1.2  material glass }', 'quad { origin -0.6 0.4 -0.6  u 0 1.2 0  v 1.2 0 0  material glass }', 'quad { origin -0.6 0.4 0.6  u 1.2 0 0  v 0 1.2 0  material glass }', 'quad { origin -0.4 0.6 -0.4  u 0 0 0.8  v 0.8 0 0  material glass }', 'quad { origin -0.4 1.4 -0.4  u 0.8 0 0  v 0 0 0.8  material glass }', 'quad { origin -0.4 0.6 -0.4  u 0 0.8 0  v 0 0 0.8  material glass }', 'quad { origin 0.4 0.6 -0.4  u 0 0 0.8  v 0 0.8 0  material glass }', 'quad { origin -0.4 0.6 -0.4  u 0.8 0 0  v 0 0.8 0  material glass }', 'quad { origin -0.4 0.6 0.4  u 0 0.8 0  v 0.8 0 0  material glass }']) + NL + LAMP % "-0.05 1.0 -0.05",
        [], {r"\[light\] light 0 \([^)]*\) is born inside": False}),
    "water_on_a_quad": (
        # A box of water whose bottom shares its edges with a white quad: the water's faces are a closed
        # piece of their own whatever they touch.
        NL.join(['quad { origin -1 0 -1  u 2 0 0  v 0 0 2  material water }', 'quad { origin -1 2 -1  u 0 0 2  v 2 0 0  material water }', 'quad { origin -1 0 -1  u 0 0 2  v 0 2 0  material water }', 'quad { origin 1 0 -1  u 0 2 0  v 0 0 2  material water }', 'quad { origin -1 0 -1  u 0 2 0  v 2 0 0  material water }', 'quad { origin -1 0 1  u 2 0 0  v 0 2 0  material water }', 'quad { origin -1 0 -1  u 2 0 0  v 0 0 2  material white }']) + NL + LAMP % "-0.05 1.5 -0.05",
        [], {r"\[light\] light 0 \([^)]*\) is born inside 'water'": True}),
    "inward_box": (
        # The walks read a face's side off its winding (interfaceStep), so to them the lamp inside a box
        # wound inside out is outside the glass, and it starts so.
        NL.join(INWARD_BOX) + NL + LAMP % "-0.05 1.0 -0.05",
        [], {r"\[light\] light 0 \([^)]*\) is born inside": False}),
    "below_inward_box": (
        # The same box, the lamp just under it, outside: two of the probe's rays meet the box's bottom
        # from behind, leaving it there, and enter it again at its far side, so it holds the lamp no more
        # than the walks do.
        NL.join(INWARD_BOX) + NL + LAMP % "0.05 0.45 -0.05",
        [], {r"\[light\] light 0 \([^)]*\) is born inside": False}),
    "iso_ball": (
        # A glass ball made as an isosurface (an SDF): a closed solid, which build()'s marking says.
        "isosurface { material glass  sphere { center 0 1 0  radius 0.5 } }" + NL + LAMP % "-0.05 1.0 -0.05",
        [], {r"\[light\] light 0 \([^)]*\) is born inside 'glass'": True}),
    "pool": (
        # Water whose surface is one quad, closed by five white quads (all wound to face out of it): the
        # water's quad closes no piece alone, but does with the walls.
        NL.join(['quad { origin -1 0.2 -1  u 2 0 0  v 0 0 2  material white }',
                 'quad { origin -1 1.2 -1  u 0 0 2  v 2 0 0  material water }',
                 'quad { origin -1 0.2 -1  u 0 0 2  v 0 1 0  material white }',
                 'quad { origin 1 0.2 -1  u 0 1 0  v 0 0 2  material white }',
                 'quad { origin -1 0.2 -1  u 0 1 0  v 2 0 0  material white }',
                 'quad { origin -1 0.2 1  u 2 0 0  v 0 1 0  material white }']) + NL + LAMP % "-0.05 1.1 -0.05",
        [], {r"\[light\] light 0 \([^)]*\) is born inside 'water'": True}),
    "pool_deep": (
        # The same pool, the lamp near its floor: the rays that leave it leave through its walls and floor,
        # which bound the water as its surface does.
        NL.join(['quad { origin -1 0.2 -1  u 2 0 0  v 0 0 2  material white }', 'quad { origin -1 1.2 -1  u 0 0 2  v 2 0 0  material water }', 'quad { origin -1 0.2 -1  u 0 0 2  v 0 1 0  material white }', 'quad { origin 1 0.2 -1  u 0 1 0  v 0 0 2  material white }', 'quad { origin -1 0.2 -1  u 0 1 0  v 2 0 0  material white }', 'quad { origin -1 0.2 1  u 2 0 0  v 0 1 0  material white }']) + NL + LAMP % "-0.05 0.45 -0.05",
        [], {r"\[light\] light 0 \([^)]*\) is born inside 'water'": True}),
    "under_wide_pool": (
        # A wide shallow pool, the lamp just under its floor, outside the water: the rays enter the water
        # through the floor and leave it through the surface.
        NL.join(['quad { origin -10 0.5 -10  u 20 0 0  v 0 0 20  material white }', 'quad { origin -10 0.9 -10  u 0 0 20  v 20 0 0  material water }', 'quad { origin -10 0.5 -10  u 0 0 20  v 0 0.4 0  material white }', 'quad { origin 10 0.5 -10  u 0 0.4 0  v 0 0 20  material white }', 'quad { origin -10 0.5 -10  u 0 0.4 0  v 20 0 0  material white }', 'quad { origin -10 0.5 10  u 20 0 0  v 0 0.4 0  material white }']) + NL + LAMP % "-0.05 0.35 -0.05",
        [], {r"\[light\] light 0 \([^)]*\) is born inside": False}),
    "back_face_on_glass": (
        # A two-sided lamp lying on a glass slab, facing up: its front face's births start in air, its
        # back face's inside the glass, which the report says of the back face alone.
        NL.join(GLASS_SLAB) + NL +
        "light area { origin -0.25 1.000001 -0.25  u 0.5 0 0  v 0 0 0.5  normal 0 1 0  two_sided on  "
        "spd preset:bb6500  power 10 }",
        [], {r"\[light\] light 1 \([^)]*\), light 0's back face, is born inside 'glass'": True,
             r"\[light\] light 0 \(": False}),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_enclosures"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, out = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    ok = True
    for name, (body, extra, want) in CASES.items():
        scn = os.path.join(out, name + ".ftsl")
        with open(scn, "w", encoding="utf-8", newline="") as f:
            f.write(HEAD + body + NL)
        p = subprocess.run([exe, "-in", scn, "-mode", "R", "-device", "cpu", "-spp", "1", "-no-meter", "-window-min",
                            "-o", os.path.join(out, name + ".png")] + extra,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        said = [l for l in (p.stdout + p.stderr).splitlines() if l.startswith(("[light] light", "[light] fire", "[camera]"))]
        good = p.returncode == 0
        for pat, should in want.items():
            found = any(re.search(pat, l) for l in said)
            good = good and found == should
        ok = ok and good
        print("%s: %s -- %s" % ("ok" if good else "FAIL", name, " | ".join(said) if said else "(nothing said)"),
              flush=True)
        if p.returncode != 0:
            print((p.stdout + p.stderr)[-800:])
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
