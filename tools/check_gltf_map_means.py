"""Regression check: a glTF's mean metalness and roughness count only the texels its surface samples.

    D:/python314/python.exe -u tools/check_gltf_map_means.py [--out <dir>] [--exe ftrace.exe]

The importer types a material metal or not by its metallicRoughness map's mean metalness, and gives
it the map's mean roughness as its constant (src/gltf.h). An atlas is not all surface: gutters and
background that no hit samples used to weigh in those means as much as the charts (known-issues
GLTF-MAP-MEAN-GUTTERS). A quad here samples only the left 30 % of its map, metalness 0.2 there; the
rest is a gutter of metalness 1, so the map's all-texel mean is 0.76 and would type the quad a
metal. Rendered (mode R, CPU, fixed seed) against the same quad on a map of 0.2 throughout:

  - byte-identical: the gutter weighs nothing, so both import as the same dielectric;
  - and a quad mapped over the whole gutter map differs, so the rig can see the typing at all;
  - the importer's part-metal note (it prints one for a map that is part metal by its weighted
    texels) names neither the gutter map nor the uniform one.

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
CHART = 0.3          # the share of the map's width the quad samples
METAL, GUTTER = 51, 255   # 8-bit metalness: 0.2 in the chart, 1.0 beyond it


def png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def write_glb(path, gutter, u_max):
    """A 1 m quad facing +z, UVs u in [0, u_max]; its metallicRoughness map 100 px wide is
    metalness 0.2 and roughness 0.5 over the chart (u < CHART), metalness `gutter` beyond."""
    from PIL import Image
    mr = Image.new("RGB", (100, 100))
    for y in range(100):
        for x in range(100):
            mr.putpixel((x, y), (0, 128, METAL if x < CHART * 100 else gutter))
    base = Image.new("RGB", (8, 8), (200, 160, 110))
    parts = [
        struct.pack("<12f", -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0),   # POSITION
        struct.pack("<12f", *([0, 0, 1] * 4)),                                         # NORMAL
        struct.pack("<8f", 0, 1, u_max, 1, u_max, 0, 0, 0),                            # TEXCOORD_0
        struct.pack("<6H", 0, 1, 2, 0, 2, 3),                                          # indices
        png(base), png(mr)]
    views, blob = [], b""
    for p in parts:
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(p)})
        blob += p + b"\0" * (-len(p) % 4)
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                                    "indices": 3, "material": 0}]}],
        "materials": [{"name": "plate", "pbrMetallicRoughness": {
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
        "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(doc).encode("utf-8")
    js += b" " * (-len(js) % 4)
    body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(blob), b"BIN\0") + blob
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def scene(mesh_file):
    return NL.join([
        'camera "c" { eye 0 -0.9 1.2  look_at 0 0 0  up 0 0 1  fov_y 45  film { res 48 48 } }',
        'material "grey" { type diffuse  reflect 0.5 }',
        'light area { origin -0.05 0.85 0.95  u 0.1 0 0  v 0 -0.1 0  normal 0 0 -1  power 40  '
        'spd preset:d65 }',
        'light area { origin -1 -1 2  u 2 0 0  v 0 2 0  normal 0 0 -1  power 20  spd preset:d65 }',
        'mesh "t" { file "%s"  material grey }' % mesh_file]) + NL


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_gltf_map_means"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe, d = os.path.abspath(a.exe), os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)

    def render(tag):
        with open(os.path.join(d, tag + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(scene(tag + ".glb"))
        pfm = os.path.join(d, tag + ".pfm")
        if os.path.exists(pfm):
            os.remove(pfm)
        cmd = [exe, "-in", tag + ".ftsl", "-mode", "R", "-device", "cpu", "-spp", "16", "-seed", "1",
               "-hdr", "-no-meter", "-o", tag + ".png"]
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0 or not os.path.exists(pfm):
            print((p.stdout + p.stderr)[-2000:])
            raise SystemExit("render failed: " + " ".join(cmd))
        return open(pfm, "rb").read(), p.stdout + p.stderr

    write_glb(os.path.join(d, "gutter.glb"), GUTTER, CHART)
    write_glb(os.path.join(d, "uniform.glb"), METAL, CHART)
    write_glb(os.path.join(d, "whole.glb"), GUTTER, 1.0)
    g, glog = render("gutter")
    u, ulog = render("uniform")
    w, _ = render("whole")
    fails = []

    def check(name, ok, detail=""):
        print("%-64s %s  %s" % (name, "ok  " if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    check("the chart of a gutter map imports as the uniform map does", g == u,
          "%d of %d bytes differ" % (sum(x != y for x, y in zip(g, u)), len(g)))
    check("  ... where the whole of it, sampled, differs", w != u)
    check("  ... and no part-metal note for either", "part metal" not in glog and "part metal" not in ulog,
          glog.strip()[-300:] if "part metal" in glog else "")
    if fails:
        print("FAILED: " + ", ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
