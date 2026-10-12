"""Put the finished Alice2 doll and her v3 hair into scenes/gallery_rain.ftsl's hall.

    D:/python314/python.exe -u tools/alice2_gallery.py [--doll <dir>] [--hair <dir>]

Reads the doll (`base_basic_pbr_flat_sparkle_sym_eyes.glb`: the sequinned recolour, the face made
symmetric, the crisp eyes) and her hair rig (alice2_hyper3d/hair_opus5.5_v3: the groom with its
guides, scalps, fibre, fur and flyaways, and the scene that adds the headband), all in the doll's
own model frame, and writes:

  scenes/alice2_gallery.ftsl          the doll's mesh block, the scalps, fibre, fur and headband,
                                      placed in the hall (gallery_rain includes it; checked in)
  meshes/alice2_gallery_guides.ftsl   the guides, verbatim, inside the placement `group`
  meshes/alice2_gallery_flyaways.ftsl the flyaways, verbatim, inside the same `group`
  meshes/alice2_doll.glb, and every mesh file the rig names, copied

(meshes/ holds generated and copied assets, kept out of git.)

THE PLACEMENT is a fixed spot and turn -- the far-right court's cap, top at y 0.90, at (9.15, 3.55),
`rotate 0 20 0` -- at the hall's display scale for exhibits, 3x. The doll herself, feet to the top
of her hair and without her stand (the model's root.2), is a 10-inch doll, so the scale is
0.254 x 3 / her height in the model. Her stand stands on the cap: the model's y 0 at y 0.90.

HOW EACH PART TAKES IT:
  - meshes (the doll, the scalps, the headband): the mesh block's own transform;
  - curves (the guides, the flyaways): a `group` around the rig's text, so the loader transforms
    the points and scales the fibre radii itself;
  - fur blocks: their lengths (fibre radius, clump size, root offset, ...) are multiplied here,
    since a fur grows in the world and no transform reaches its settings. Every setting must be
    one this tool knows how to carry (FUR_KEPT / FUR_LENGTH / FUR_AREA): anything else stops it
    rather than being copied unscaled.

Rerun it whenever the doll or the hair is regenerated: everything placed comes from here.
"""
import argparse
import json
import os
import re
import shutil
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALICE2 = "D:/youtube/philosophy/3d objects/alice2/alice2_hyper3d"
DOLL_GLB = "base_basic_pbr_flat_sparkle_sym_eyes.glb"
GROOM = "alice2_hair_groom.ftsl"        # in the hair rig: guides include, scalps, fibre, fur, flyaways
RIG_SCENE = "alice2_real_hair_eyes.ftsl"  # in the hair rig: where the headband is
PLACE = (9.15, 0.90, 3.55)      # the cap's centre, at its top
ROT_Y = 20.0                    # degrees; +z (her front) swings toward +x
DISPLAY = 3.0                   # the hall's scale for exhibits
DOLL_HEIGHT = 0.254             # m: a 10-inch doll, feet to the top of her hair
STAND = "root.2"                # the model's stand: not part of the doll's height
NAME_PREFIX = "alice2_"         # rig names that do not start with it get it, to stay unique in the hall
NL = chr(10)

# Fur settings, by how a uniform scale s reaches them. A setting in none of these sets is refused.
FUR_KEPT = {"on", "material", "count", "guides", "guide_blend", "points", "segments", "spline",
            "length_jitter", "jitter", "clump", "seed"}
FUR_LENGTH = {"radius", "radius_tip", "clump_size", "root_offset", "guide_falloff", "length"}
FUR_AREA = {"density"}          # per unit area: / s^2
# The fur loader's DEFAULTS for the lengths it does not set to 0 (ftsl.h addFur). A block that
# leaves one out would get the absolute default in the hall, unscaled, so the tool writes it in,
# scaled. (radius_tip left out is derived from radius, so it scales with it.)
FUR_LENGTH_DEFAULTS = {"radius": 0.0008, "clump_size": 0.02, "length": 0.05}
STAND_BASE_TOL = 1e-3           # model units: the stand's base must sit at y 0


def glb_json(path):
    """The JSON chunk of a .glb."""
    with open(path, "rb") as f:
        head = f.read(20)
        magic, _ver, _total, clen, ctype = struct.unpack("<4sIIII", head)
        if magic != b"glTF" or ctype != 0x4E4F534A:
            raise SystemExit("%s: not a GLB with a JSON chunk first" % path)
        return json.loads(f.read(clen))


def doll_extent(glb):
    """(y min, y max, primitive count) of every primitive but the stand's, from the accessors' own
    bounds.

    Refuses node transforms (ftrace's loader would apply them and this measurement would not), and
    a stand whose base is not at y 0 (the placement puts the model's y 0 on the cap's top)."""
    j = glb_json(glb)
    for n in j.get("nodes", []):
        if any(k in n for k in ("matrix", "translation", "rotation", "scale")):
            raise SystemExit("%s: node %r carries a transform; this tool measures raw positions"
                             % (glb, n.get("name")))
    mats = [m.get("name", "") for m in j.get("materials", [])]
    lo, hi, parts, stand = [], [], 0, []
    for mesh in j["meshes"]:
        for p in mesh["primitives"]:
            acc = j["accessors"][p["attributes"]["POSITION"]]
            if "material" in p and mats[p["material"]] == STAND:
                stand.append(acc["min"][1])
                continue
            lo.append(acc["min"][1]); hi.append(acc["max"][1]); parts += 1
    if not parts:
        raise SystemExit("%s: no primitives besides the stand" % glb)
    if not stand:
        raise SystemExit("%s: no %s primitive (the stand), so nothing says where the cap's top is"
                         % (glb, STAND))
    if abs(min(stand)) > STAND_BASE_TOL:
        raise SystemExit("%s: the stand's base is at model y %.4f, not 0; the placement puts y 0 on "
                         "the cap's top, so she would %s it" % (glb, min(stand),
                                                               "sink into" if min(stand) < 0 else "float above"))
    return min(lo), max(hi), parts


def num(v):
    """A number as FTSL writes it: fixed point, no exponent, no trailing zeros."""
    return ("%.10f" % v).rstrip("0").rstrip(".")


def tokens(line):
    """Whitespace tokens, a quoted string kept whole (quotes included)."""
    return re.findall(r'"[^"]*"|\S+', line)


def scale_fur(block, s):
    """A fur block with every length times s and every area density over s^2; refuses settings it
    cannot classify (a `bald` zone or a direction would need the placement, not a factor)."""
    lines = block.split(NL)
    if not re.match(r'^fur "[^"]+" \{\s*$', lines[0]):
        raise SystemExit("fur block: settings on its opening line are not classified: %r" % lines[0])
    out = [lines[0]]
    seen = set()
    for line in lines[1:-1]:
        body, _, comment = line.partition("#")
        tk = tokens(body)
        i = 0
        new = []
        while i < len(tk):
            key = tk[i]
            if i + 1 >= len(tk):
                raise SystemExit("fur block: %r has no value in %r" % (key, line.strip()))
            val = tk[i + 1]
            seen.add(key)
            if key in FUR_KEPT:
                new += [key, val]
            elif key in FUR_LENGTH:
                new += [key, num(float(val) * s)]
            elif key in FUR_AREA:
                new += [key, num(float(val) / (s * s))]
            else:
                raise SystemExit("fur block: no rule for %r (in %r) -- add it to FUR_KEPT, FUR_LENGTH "
                                 "or FUR_AREA, or teach the tool to place it" % (key, line.strip()))
            i += 2
        indent = re.match(r"\s*", line).group(0)
        out.append(indent + "   ".join(" ".join(new[k:k + 2]) for k in range(0, len(new), 2))
                   + (("   #" + comment) if comment else ""))
    missing = [k for k in sorted(FUR_LENGTH_DEFAULTS) if k not in seen]
    if missing:
        out.append("    " + "   ".join("%s %s" % (k, num(FUR_LENGTH_DEFAULTS[k] * s)) for k in missing)
                   + "   # the loader's defaults, scaled")
    out.append(lines[-1])
    return NL.join(out)


def placed_group(name, text, s):
    """`text` (curves, in the model frame) inside a group that places it in the hall."""
    return ("group \"%s\" { translate %.6f %.6f %.6f   rotate 0 %g 0   scale %.6f" % (name, *PLACE, ROT_Y, s)
            + NL + text.rstrip(NL) + NL + "}" + NL)


def place_mesh(line, s, rig_dir, meshes_dir, copied):
    """A one-line `mesh "n" { file "f" ... }` with the file copied into meshes/ and the placement added."""
    m = re.match(r'^mesh "([^"]+)" \{ file "([^"]+)"(.*)\}\s*$', line)
    if not m:
        raise SystemExit("rig: expected a one-line mesh block with a file, got %r" % line)
    name, f, rest = m.groups()
    src = os.path.join(rig_dir, f)
    dst = os.path.basename(f)
    if not dst.startswith(NAME_PREFIX):
        dst = NAME_PREFIX + dst
    shutil.copyfile(src, os.path.join(meshes_dir, dst))
    copied.append(dst)
    return ('mesh "%s" { file "meshes/%s"%s' % (name, dst, rest.rstrip())
            + NL + "    translate %.6f %.6f %.6f   rotate 0 %g 0   scale %.6f }" % (*PLACE, ROT_Y, s))


def rename(text, names):
    """Every rig name in `names` that lacks the prefix, prefixed (whole words only)."""
    for n in names:
        if not n.startswith(NAME_PREFIX):
            text = re.sub(r'(?<![\w"])("?)%s\1(?![\w])' % re.escape(n), lambda m: m.group(1) + NAME_PREFIX + n + m.group(1), text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--doll", default=ALICE2 + "/model")
    ap.add_argument("--hair", default=ALICE2 + "/hair_opus5.5_v3")
    a = ap.parse_args()
    meshes = os.path.join(ROOT, "meshes")

    glb = os.path.join(a.doll, DOLL_GLB)
    y0, y1, nparts = doll_extent(glb)
    s = DISPLAY * DOLL_HEIGHT / (y1 - y0)
    print("doll: %d primitives besides the stand span model y %.4f..%.4f -> scale %.6f "
          "(%.3f m tall in the hall, %.3f m with the stand)" % (nparts, y0, y1, s, s * (y1 - y0), s * y1))
    shutil.copyfile(glb, os.path.join(meshes, "alice2_doll.glb"))
    xf = "translate %.6f %.6f %.6f   rotate 0 %g 0   scale %.6f" % (*PLACE, ROT_Y, s)

    groom = open(os.path.join(a.hair, GROOM), encoding="utf-8").read()
    rig = open(os.path.join(a.hair, RIG_SCENE), encoding="utf-8").read()
    copied = []
    out = ["# Alice2 FOR gallery_rain -- GENERATED by tools/alice2_gallery.py from the doll and the v3 hair",
           "# rig (alice2_hyper3d/hair_opus5.5_v3/%s, and the headband from %s)," % (GROOM, RIG_SCENE),
           "# placed in the hall. Do not edit: rerun the tool.",
           "#",
           "# Meshes carry the placement as their own transform, the curves inside a `group`; the fur",
           "# blocks' lengths are times the scale %.6f." % s,
           "",
           "# the doll, her sculpted hair (glTF material root.1) left out for the strands below; the",
           "# fallback material is gallery_rain's `alice_unpainted`",
           'mesh "alice2" { file "meshes/alice2_doll.glb"  material alice_unpainted  skip_material root.1',
           "    " + xf + " }",
           ""]

    # The groom, part by part, in its own order -- less its own header comment, which names the
    # tool that generated the RIG and would send a reader of this file to the wrong one.
    flyat = groom.find("# flyaways")
    if flyat < 0:
        raise SystemExit("%s: no '# flyaways' section" % GROOM)
    head, fly = groom[:flyat], groom[flyat:]
    hl = head.split(NL)
    k = 0
    while k < len(hl) and (not hl[k].strip() or hl[k].lstrip().startswith("#")):
        k += 1
    head = NL.join(hl[k:])
    pos = 0
    included = False
    for m in re.finditer(r'(?ms)^fur "[^"]+" \{.*?^\}', head):
        for line in head[pos:m.start()].split(NL):
            st = line.strip()
            if not st or st.startswith("#"):
                out.append(line)
            elif st.startswith("include "):
                if included:
                    raise SystemExit("%s: a second include (%r); the tool places one guides file"
                                     % (GROOM, st))
                included = True
                inc = re.match(r'include "([^"]+)"', st).group(1)
                text = open(os.path.join(a.hair, inc), encoding="utf-8").read()
                with open(os.path.join(meshes, "alice2_gallery_guides.ftsl"), "w", encoding="utf-8", newline="") as f:
                    f.write("# GENERATED by tools/alice2_gallery.py: %s, verbatim, placed in gallery_rain's hall." % inc
                            + NL + NL + placed_group("alice2_guides_place", text, s))
                out.append('include "../meshes/alice2_gallery_guides.ftsl"     # %s, placed' % inc)
            elif st.startswith("material "):
                out.append(line)
            elif st.startswith("mesh "):
                out.append(place_mesh(st, s, a.hair, meshes, copied))
            else:
                raise SystemExit("%s: no rule for %r" % (GROOM, st))
        out.append(scale_fur(m.group(0), s))
        pos = m.end()
    if head[pos:].strip():
        raise SystemExit("%s: unexpected text after the last fur block, before the flyaways" % GROOM)
    with open(os.path.join(meshes, "alice2_gallery_flyaways.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write("# GENERATED by tools/alice2_gallery.py: %s's flyaways, verbatim, placed in gallery_rain's hall."
                % GROOM + NL + NL + placed_group("alice2_flyaways_place", fly, s))
    out += ["", "# the flyaways (they wear the fibre above)", 'include "../meshes/alice2_gallery_flyaways.ftsl"', ""]

    # The headband, from the rig's scene: its comment, its materials and its mesh.
    hb = re.search(r'(?ms)^# the black satin headband.*?^mesh "alice2_headband" \{[^\n]*\}', rig)
    if not hb:
        raise SystemExit("%s: no headband section" % RIG_SCENE)
    hbtext = rename(hb.group(0), re.findall(r'^material "([^"]+)"', hb.group(0), re.M))
    for line in hbtext.split(NL):
        st = line.strip()
        if not st or st.startswith("#") or st.startswith("material "):
            out.append(line)
        elif st.startswith("mesh "):
            out.append(place_mesh(st, s, a.hair, meshes, copied))
        else:       # any other geometry would land in the hall unplaced
            raise SystemExit("%s: no rule for %r in the headband section" % (RIG_SCENE, st))
    out.append("")

    with open(os.path.join(ROOT, "scenes", "alice2_gallery.ftsl"), "w", encoding="utf-8", newline="") as f:
        f.write(re.sub(NL + "{3,}", NL + NL, NL.join(out)))
    print("wrote scenes/alice2_gallery.ftsl, meshes/alice2_gallery_guides.ftsl, "
          "meshes/alice2_gallery_flyaways.ftsl; copied alice2_doll.glb, %s" % ", ".join(copied))


if __name__ == "__main__":
    main()
