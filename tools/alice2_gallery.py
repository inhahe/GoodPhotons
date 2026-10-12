"""Put the finished Alice2 doll and her v3 hair into scenes/gallery_rain.ftsl's hall.

    D:/python314/python.exe -u tools/alice2_gallery.py [--doll <dir>] [--hair <dir>]

Reads the doll (`base_basic_pbr_flat_sparkle_sym_eyes.glb`: the sequinned recolour, the face made
symmetric, the crisp eyes) and her hair (alice2_hyper3d/hair_opus5.5_v3: the guides, the fur blocks,
the flyaways, the scalps and the headband, all in the doll's own model frame), and writes:

  meshes/alice2_doll.glb                      the doll, copied
  meshes/alice2_scalp_{fr,fl,bk}.obj          the hair's root surfaces, copied
  meshes/alice2_headband.obj                  the headband, copied
  meshes/alice2_hair_gallery_curves.ftsl      the guides and the flyaways, moved into the hall
  scenes/alice2_hair_gallery.ftsl             the hair: scalps, headband, fibre, fur blocks

(meshes/ holds generated assets, kept out of git; the scene file is small and checked in.)

THE PLACEMENT is the old doll's spot and turn -- the far-right court's cap, top at y 0.90, at
(9.15, 3.55), `rotate 0 20 0` -- at the hall's display scale for exhibits, 3x. The doll herself,
feet to the top of her hair, is a 10-inch doll (the size the Meshy doll was given), so the model's
1.64 m of doll scales by 0.254 / 1.64 x 3. Her stand (the model's root.2, a base and a rod up to
her waist) stands on the cap: the model's y 0 at y 0.90.

Curves and fur cannot ride a `group` transform (a group bakes meshes, primitives and lights, not
a fur's metric settings), so the guides' and flyaways' points are moved here, and every length a
fur block or curve carries -- fibre radius, clump size, root offset -- is multiplied by the scale.
"""
import argparse
import os
import re
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import hairgeo as hg            # noqa: E402

ALICE2 = "D:/youtube/philosophy/3d objects/alice2/alice2_hyper3d"
DOLL_GLB = "base_basic_pbr_flat_sparkle_sym_eyes.glb"
PLACE = (9.15, 0.90, 3.55)      # the cap's centre, at its top
ROT_Y = 20.0                    # degrees; +z (her front) swings toward +x
DISPLAY = 3.0                   # the hall's scale for exhibits
DOLL_HEIGHT = 0.254             # m: a 10-inch doll, feet to the top of her hair
STAND = "root.2"                # the model's stand: not part of the doll's height
NL = chr(10)


def doll_scale(glb):
    """The mesh scale that makes the doll (everything but the stand) DOLL_HEIGHT x DISPLAY tall."""
    g = hg.Glb(glb)
    ys = []
    for m in g.json["materials"]:
        if m["name"] == STAND:
            continue
        try:
            P = g.mesh(g.mesh_by_material(m["name"]))[0]
        except Exception:
            continue
        ys.append(P[:, 1])
    ys = np.concatenate(ys)
    return DISPLAY * DOLL_HEIGHT / float(np.ptp(ys)), float(ys.min()), float(ys.max())


def place(P, s):
    """Model frame -> hall: world = translate + Ry(s * p), as a mesh block's transform does it."""
    t = np.radians(ROT_Y)
    c, sn = np.cos(t), np.sin(t)
    q = np.asarray(P, np.float64) * s
    return np.c_[c * q[:, 0] + sn * q[:, 2] + PLACE[0], q[:, 1] + PLACE[1], -sn * q[:, 0] + c * q[:, 2] + PLACE[2]]


def move_curves(text, s):
    """Every `point x y z` line moved into the hall; `radius` / `radius_tip` on a curve line scaled."""
    pt = re.compile(r"^(\s*)point\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
    rad = re.compile(r"(radius(?:_tip)?)\s+([0-9.eE+-]+)")
    lines = text.split(NL)
    pts, idx = [], []
    for i, line in enumerate(lines):
        m = pt.match(line)
        if m:
            pts.append([float(m.group(2)), float(m.group(3)), float(m.group(4))]); idx.append(i)
    W = place(np.array(pts), s)
    for k, i in enumerate(idx):
        ind = pt.match(lines[i]).group(1)
        lines[i] = "%spoint %.6f %.6f %.6f" % (ind, *W[k])
    for i, line in enumerate(lines):
        if line.lstrip().startswith("curve") and "radius" in line:
            lines[i] = rad.sub(lambda m: "%s %.7f" % (m.group(1), float(m.group(2)) * s), line)
    return NL.join(lines), len(idx)


def scale_fur(block, s):
    """A fur block's lengths (fibre radius, clump size, root offset) times s; counts and ratios kept."""
    def sc(m):
        return "%s %.7f" % (m.group(1), float(m.group(2)) * s)
    return re.sub(r"\b(radius|radius_tip|clump_size|root_offset)\s+([0-9.eE+-]+)", sc, block)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--doll", default=ALICE2 + "/model")
    ap.add_argument("--hair", default=ALICE2 + "/hair_opus5.5_v3")
    a = ap.parse_args()
    glb = os.path.join(a.doll, DOLL_GLB)
    s, y0, y1 = doll_scale(glb)
    print("doll: model y %.4f..%.4f without the stand -> scale %.6f (%.3f m tall in the hall, %.3f m with the stand)"
          % (y0, y1, s, s * (y1 - y0), s * y1))
    meshes = os.path.join(ROOT, "meshes")
    shutil.copyfile(glb, os.path.join(meshes, "alice2_doll.glb"))
    for name in ("alice2_scalp_fr.obj", "alice2_scalp_fl.obj", "alice2_scalp_bk.obj", "alice2_headband.obj"):
        shutil.copyfile(os.path.join(a.hair, name), os.path.join(meshes, name))
    groom = open(os.path.join(a.hair, "alice2_hair_groom.ftsl"), encoding="utf-8").read()
    guides = open(os.path.join(a.hair, "alice2_hair_guides.ftsl"), encoding="utf-8").read()
    # the guides, and the groom's flyaway curves (everything from its "# flyaways" comment on)
    fly = groom[groom.index("# flyaways"):]
    gtext, ng = move_curves(guides, s)
    ftext, nf = move_curves(fly, s)
    with open(os.path.join(meshes, "alice2_hair_gallery_curves.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write("# GENERATED by tools/alice2_gallery.py -- Alice2's v3 hair guides and flyaways, moved into" + NL +
                "# gallery_rain's hall (translate %.2f %.2f %.2f, rotate 0 %g 0, scale %.6f). Do not edit:" % (*PLACE, ROT_Y, s) + NL +
                "# edit the hair in alice2_hyper3d/hair_opus5.5_v3 and rerun the tool." + NL + NL + gtext + NL + ftext + NL)
    furs = re.findall(r"fur \"[^\"]+\" \{.*?\n\}", groom, re.S)
    mat_hair = re.search(r"material \"alice2_hair\" \{[^\n]*\}", groom).group(0)
    xf = "translate %.6f %.6f %.6f   rotate 0 %g 0   scale %.6f" % (*PLACE, ROT_Y, s)
    L = ["# Alice2's hair FOR gallery_rain -- GENERATED by tools/alice2_gallery.py from the v3 groom",
         "# (alice2_hyper3d/hair_opus5.5_v3), carried into the hall's frame. Do not edit: rerun the tool.",
         "#",
         "#     include \"alice2_hair_gallery.ftsl\"     (gallery_rain.ftsl, right after its `mesh \"alice2\"` block)",
         "#",
         "# Same placement as the doll's mesh block: the scalps and the headband carry it as their own",
         "# transform; the guides and flyaways (meshes/alice2_hair_gallery_curves.ftsl) were moved by the",
         "# tool; every length the fur blocks carry -- fibre radius, clump size, root offset -- is times",
         "# the scale %.6f. `count` is a count and does not scale." % s,
         "",
         "material \"alice2_scalp_mat\" { type diffuse reflect 0.5 }"]
    for part in ("fr", "fl", "bk"):
        L.append("mesh \"alice2_scalp_%s\" { file \"meshes/alice2_scalp_%s.obj\"  material alice2_scalp_mat  shape_only yes" % (part, part))
        L.append("    %s }" % xf)
    L += ["", "# the black satin headband and bow",
          "material \"alice2_ribbon_body\" { type diffuse reflect 0.022 }",
          "material \"alice2_ribbon\" { type layered  ior 1.5  coat { roughness 0.32 }  layer \"alice2_ribbon_body\" 1.0 }",
          "mesh \"alice2_headband\" { file \"meshes/alice2_headband.obj\"  material alice2_ribbon",
          "    %s }" % xf, "",
          "# synthetic doll fibre: eta 1.55, no cuticle tilt, no medulla; `reflect` calibrated against the photos",
          mat_hair, "",
          "# the guides (before the fur that follows them) and the flyaways (after the fibre they wear)",
          "include \"../meshes/alice2_hair_gallery_curves.ftsl\"", ""]
    for fb in furs:
        L += [scale_fur(fb, s), ""]
    with open(os.path.join(ROOT, "scenes", "alice2_hair_gallery.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(NL.join(L))
    print("wrote meshes/alice2_doll.glb, the scalps and headband, %d guide points + %d flyaway points moved, "
          "scenes/alice2_hair_gallery.ftsl (%d fur blocks)" % (ng, nf, len(furs)))
    print("the doll's mesh block: mesh \"alice2\" { file \"meshes/alice2_doll.glb\"  material alice_unpainted  "
          "skip_material root.1" + NL + "    " + xf + " }")


if __name__ == "__main__":
    main()
