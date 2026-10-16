"""Regression check: a mesh block's `import_metal mix|mean` reads its glTF's metalness map as asked.

    D:/python314/python.exe -u tools/check_import_metal.py [--out <dir>] [--exe ftrace.exe]

Writes a one-quad GLB whose metallicRoughness map is a clean checkerboard -- half the texels
fully metal, half not, the case per-texel import exists for (a sequin mask) -- and renders it
(mode R, CPU, fixed seed) with every pairing of the keyword and the `-import-metal` flag:

  - `import_metal mix` renders byte-identically to the flag `-import-metal mix`;
  - the keyword wins over the flag: `import_metal mean` with `-import-metal mix` is the plain
    import, and `import_metal mix` with `-import-metal mean` is the per-texel one;
  - the per-texel import differs from the plain one, so the rig can see the effect;
  - the plain import prints the part-metal hint naming the keyword; a mesh that chose `mean`
    does not get it;
  - `import_metal` with another value, or on an OBJ, is refused with a message;
  - the key is never reported unknown.

Exits 1 on any failure.
"""
import argparse
import io
import json
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)


def png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def write_glb(path):
    """A 1 m quad facing +z, with a gold base colour and a checkerboard metalness mask."""
    from PIL import Image
    base = Image.new("RGB", (64, 64), (210, 165, 70))
    mr = Image.new("RGB", (64, 64))
    for y in range(64):                      # glTF: G = roughness, B = metalness
        for x in range(64):
            mr.putpixel((x, y), (0, 60, 255 if (x // 8 + y // 8) % 2 == 0 else 0))
    parts = [
        struct.pack("<12f", -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0),   # POSITION
        struct.pack("<12f", *([0, 0, 1] * 4)),                                         # NORMAL
        struct.pack("<8f", 0, 1, 1, 1, 1, 0, 0, 0),                                    # TEXCOORD_0
        struct.pack("<6H", 0, 1, 2, 0, 2, 3),                                          # indices
        png(base), png(mr)]
    views, blob = [], b""
    for p in parts:
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(p)})
        blob += p + b"\0" * (-len(p) % 4)
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                                    "indices": 3, "material": 0}]}],
        "materials": [{"name": "sequins", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicRoughnessTexture": {"index": 1},
            "metallicFactor": 1.0, "roughnessFactor": 1.0}}],
        "textures": [{"source": 0}, {"source": 1}],
        "images": [{"bufferView": 4, "mimeType": "image/png"}, {"bufferView": 5, "mimeType": "image/png"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
             "min": [-0.5, -0.5, 0], "max": [0.5, 0.5, 0]},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"}],
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(doc).encode("utf-8")
    js += b" " * (-len(js) % 4)
    body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(blob), b"BIN\0") + blob
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def scene(mesh_file, keyword):
    return NL.join([
        'camera "c" { eye 0 0 1.6  look_at 0 0 0  up 0 1 0  fov_y 40  film { res 48 48 } }',
        'material "grey" { type diffuse  reflect 0.5 }',
        'light area { origin -0.4 0.6 0.2  u 0.8 0 0  v 0 0 0.8  normal 0 -1 0  power 50  '
        'spd preset:d65 }',
        'mesh "t" { file "%s"  material grey%s }' % (mesh_file, ("  import_metal " + keyword) if keyword else ""),
    ]) + NL


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_import_metal"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)
    write_glb(os.path.join(d, "t.glb"))
    with open(os.path.join(d, "t.obj"), "w", encoding="utf-8", newline="") as f:
        f.write("v 0 0 0" + NL + "v 1 0 0" + NL + "v 0 1 0" + NL + "f 1 2 3" + NL)

    def render(tag, mesh_file, keyword, flag):
        with open(os.path.join(d, tag + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(scene(mesh_file, keyword))
        pfm = os.path.join(d, tag + ".pfm")
        if os.path.exists(pfm):
            os.remove(pfm)
        cmd = [exe, "-in", tag + ".ftsl", "-mode", "R", "-device", "cpu", "-spp", "4", "-seed", "1",
               "-hdr", "-no-meter", "-o", tag + ".png"] + (["-import-metal", flag] if flag else [])
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8", errors="replace")
        img = open(pfm, "rb").read() if p.returncode == 0 and os.path.exists(pfm) else None
        return p.returncode, p.stdout + p.stderr, img

    fails = []

    def check(ok, what):
        print("%s: %s" % ("ok" if ok else "FAIL", what), flush=True)
        if not ok:
            fails.append(what)

    runs = {
        "kw_mix": render("kw_mix", "t.glb", "mix", None),
        "flag_mix": render("flag_mix", "t.glb", None, "mix"),
        "plain": render("plain", "t.glb", None, None),
        "kw_mean_flag_mix": render("kw_mean_flag_mix", "t.glb", "mean", "mix"),
        "kw_mix_flag_mean": render("kw_mix_flag_mean", "t.glb", "mix", "mean"),
    }
    for tag, (rc, log, img) in runs.items():
        check(rc == 0 and img is not None, "%s renders (exit %d)" % (tag, rc))
        check("unknown key 'import_metal'" not in log, "%s: import_metal is a known key" % tag)
    img = {t: r[2] for t, r in runs.items()}
    check(img["kw_mix"] is not None and img["kw_mix"] == img["flag_mix"],
          "import_metal mix == -import-metal mix, byte for byte")
    check(img["kw_mean_flag_mix"] is not None and img["kw_mean_flag_mix"] == img["plain"],
          "import_metal mean wins over -import-metal mix")
    check(img["kw_mix_flag_mean"] is not None and img["kw_mix_flag_mean"] == img["kw_mix"],
          "import_metal mix wins over -import-metal mean")
    check(img["kw_mix"] is not None and img["kw_mix"] != img["plain"],
          "the per-texel import differs from the plain one (the rig sees the effect)")
    check("import_metal mix` on the mesh block" in runs["plain"][1],
          "the plain import's part-metal hint names the keyword")
    check("is part metal" not in runs["kw_mean_flag_mix"][1],
          "a mesh that chose `mean` gets no hint")
    rc, log, _ = render("bad", "t.glb", "shiny", None)
    check(rc != 0 and "import_metal must be" in log, "import_metal shiny is refused (exit %d)" % rc)
    rc, log, _ = render("obj", "t.obj", "mix", None)
    check(rc != 0 and "not a .gltf/.glb" in log, "import_metal on an OBJ is refused (exit %d)" % rc)
    print("PASS" if not fails else "FAILED (%d)" % len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
