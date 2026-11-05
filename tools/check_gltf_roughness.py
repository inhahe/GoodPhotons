"""Regression check: a glTF roughness imports as the ftrace roughness of the same lobe width.

    D:/python314/python.exe -u tools/check_gltf_roughness.py [--out <dir>] [--exe ftrace.exe]

glTF's roughness `r` is perceptual (GGX alpha = r^2); ftrace's is a Phong lobe, as wide as a GGX
alpha of 0.646 r_t, so the importer takes r_t = min(1, r^2 / 0.646) (gltfRoughness, src/gltf.h).
Writes one-quad GLBs of a metal (no coat, so the material is one glossy lobe) and renders each
(mode R, CPU, fixed seed) against the same quad with `import_materials no` and the FTSL glossy
material the import should produce:

  - roughnessFactor 0.5, no map: byte-identical to `roughness 0.25/0.646`; and 0.05 to
    `roughness 0.0025/0.646`, the importer's floor being the exponent's own clamp (1e-3);
  - roughnessFactor 1, a constant map of G 128: within float noise of `roughness (128/255)^2/0.646`
    (the map is sampled bilinearly, so the texel need not come back to the last bit);
  - each against the raw roughness the importer used to copy, which must differ, so the rig can
    see the conversion at all.

Exits 1 on any failure.
"""
import argparse
import io
import json
import os
import struct
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
BASE = (0.9, 0.7, 0.5)          # glTF baseColorFactor, linear: the metal's tint


def png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def write_glb(path, rough_factor, map_g=None):
    """A 1 m quad facing +z, a whole metal of base colour BASE; `map_g` (0-255) adds a constant
    metallicRoughness map whose green (roughness) is that and whose blue (metalness) is 255."""
    parts = [
        struct.pack("<12f", -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0),   # POSITION
        struct.pack("<12f", *([0, 0, 1] * 4)),                                         # NORMAL
        struct.pack("<8f", 0, 1, 1, 1, 1, 0, 0, 0),                                    # TEXCOORD_0
        struct.pack("<6H", 0, 1, 2, 0, 2, 3)]                                          # indices
    pbr = {"baseColorFactor": list(BASE) + [1.0], "metallicFactor": 1.0, "roughnessFactor": rough_factor}
    extra = {}
    if map_g is not None:
        from PIL import Image
        parts.append(png(Image.new("RGB", (16, 16), (0, map_g, 255))))
        pbr["metallicRoughnessTexture"] = {"index": 0}
        extra = {"textures": [{"source": 0}], "images": [{"bufferView": 4, "mimeType": "image/png"}]}
    views, blob = [], b""
    for p in parts:
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(p)})
        blob += p + b"\0" * (-len(p) % 4)
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                                    "indices": 3, "material": 0}]}],
        "materials": [{"name": "metal", "pbrMetallicRoughness": pbr}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
             "min": [-0.5, -0.5, 0], "max": [0.5, 0.5, 0]},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"}],
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}]}
    doc.update(extra)
    js = json.dumps(doc).encode("utf-8")
    js += b" " * (-len(js) % 4)
    body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(blob), b"BIN\0") + blob
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def scene(mesh_file, ftsl_rough=None):
    """The quad under a small light, seen near its mirror direction so the highlight's width shows;
    `ftsl_rough` replaces the import's material with an FTSL glossy of that roughness."""
    if ftsl_rough is None:
        mat, mesh = 'material "grey" { type diffuse  reflect 0.5 }', 'mesh "t" { file "%s"  material grey }'
    else:
        mat = ('material "m" { type glossy  roughness %.17g  reflect rgb %g %g %g }'
               % ((ftsl_rough,) + BASE))
        mesh = 'mesh "t" { file "%s"  material m  import_materials no }'
    return NL.join([
        'camera "c" { eye 0 -0.9 1.2  look_at 0 0 0  up 0 0 1  fov_y 45  film { res 48 48 } }',
        mat,
        'light area { origin -0.05 0.85 0.95  u 0.1 0 0  v 0 -0.1 0  normal 0 0 -1  power 40  '
        'spd preset:d65 }',
        mesh % mesh_file]) + NL


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "png", "check_gltf_roughness"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "ftrace.exe"))
    a = ap.parse_args()
    exe = os.path.abspath(a.exe)
    d = os.path.abspath(a.out)
    os.makedirs(d, exist_ok=True)

    def render(tag, text):
        with open(os.path.join(d, tag + ".ftsl"), "w", encoding="utf-8", newline="") as f:
            f.write(text)
        pfm = os.path.join(d, tag + ".pfm")
        if os.path.exists(pfm):
            os.remove(pfm)
        cmd = [exe, "-in", tag + ".ftsl", "-mode", "R", "-device", "cpu", "-spp", "16", "-seed", "1",
               "-hdr", "-no-meter", "-o", tag + ".png"]
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0 or not os.path.exists(pfm):
            print((p.stdout + p.stderr)[-2000:])
            raise SystemExit("render failed: " + " ".join(cmd))
        raw = open(pfm, "rb").read()
        h = raw.split(b"\n", 3)
        img = np.frombuffer(h[3], dtype="<f4" if float(h[2]) < 0 else ">f4")
        return raw, img.astype(np.float64)

    fails = []

    def check(name, ok, detail):
        print("%-58s %s  %s" % (name, "ok  " if ok else "FAIL", detail))
        if not ok:
            fails.append(name)

    # 1. The factor alone.
    write_glb(os.path.join(d, "factor.glb"), 0.5)
    imp, imp_img = render("factor_import", scene("factor.glb"))
    conv, _ = render("factor_ftsl", scene("factor.glb", 0.25 / 0.646))
    raw_, raw_img = render("factor_raw", scene("factor.glb", 0.5))
    check("factor 0.5 imports as roughness 0.25/0.646 (byte-identical)", imp == conv,
          "%d of %d bytes differ" % (sum(x != y for x, y in zip(imp, conv)), len(imp)))
    rel = np.abs(imp_img - raw_img).sum() / max(1e-30, raw_img.sum())
    check("  ... and not as the raw 0.5", rel > 0.02, "relative L1 %.3f" % rel)

    # 1b. A low factor, below where a floor in ftrace's units would reach: as sharp as glTF means.
    write_glb(os.path.join(d, "low.glb"), 0.05)
    imp, _ = render("low_import", scene("low.glb"))
    conv, _ = render("low_ftsl", scene("low.glb", 0.05 * 0.05 / 0.646))
    check("factor 0.05 imports as roughness 0.0025/0.646 (byte-identical)", imp == conv,
          "%d of %d bytes differ" % (sum(x != y for x, y in zip(imp, conv)), len(imp)))

    # 2. A constant map under factor 1.
    g = 128
    r = g / 255.0
    write_glb(os.path.join(d, "map.glb"), 1.0, map_g=g)
    _, imp_img = render("map_import", scene("map.glb"))
    _, conv_img = render("map_ftsl", scene("map.glb", r * r / 0.646))
    _, raw_img = render("map_raw", scene("map.glb", r))
    rel_c = np.abs(imp_img - conv_img).sum() / max(1e-30, conv_img.sum())
    rel_r = np.abs(imp_img - raw_img).sum() / max(1e-30, raw_img.sum())
    check("map G 128 imports as roughness (128/255)^2/0.646", rel_c < 1e-3, "relative L1 %.2e" % rel_c)
    check("  ... and not as the raw 128/255", rel_r > 0.02, "relative L1 %.3f" % rel_r)

    if fails:
        print("FAILED: " + ", ".join(fails))
        sys.exit(1)
    print("all ok")


if __name__ == "__main__":
    main()
