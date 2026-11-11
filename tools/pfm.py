"""PFM images: the one reader tools/ shares (known-issues TOOLS-PFM-READERS).

    from pfm import read_pfm          # tools/ is on sys.path for a script run from it
    img = read_pfm(path)              # (h, w, c) float64: c 3 for "PF", 1 for "Pf"
    img = read_pfm(path, top_down=True)   # row 0 the image's top (a PFM stores rows bottom-up)

ftrace's -hdr output is scene-linear XYZ, before any tone map. The reader skips comment lines and
refuses a file that is not a PFM. Import the names, not the module: many scripts call a path
`pfm`.
"""
import numpy as np

# Two images that differ only in the order of the film's atomic sums agree to about float32's
# 1.2e-7 of a pixel: a sum in another order can round the other way.
ORDER_TOL = 1e-5


def read_pfm(path, top_down=False):
    """The image at `path` as an (h, w, channels) float64 array, its rows in the file's order
    (bottom-up) unless `top_down`."""
    with open(path, "rb") as f:
        kind = f.readline().strip()
        if kind not in (b"PF", b"Pf"):
            raise ValueError("%s: not a PFM (%r)" % (path, kind))
        nc = 3 if kind == b"PF" else 1
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(x) for x in line.split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(w * h * nc * 4), dtype="<f4" if scale < 0 else ">f4")
    img = data.reshape(h, w, nc).astype(np.float64)
    return img[::-1] if top_down else img


def read_pfm_flat(path):
    """(w, h, channels, values): the values in the file's order, a pixel's channels together."""
    img = read_pfm(path)
    h, w, nc = img.shape
    return w, h, nc, img.ravel()


def luma(img):
    """0.2126, 0.7152 and 0.0722 of an image's three channels, or its one channel. (ftrace's PFM
    holds XYZ, whose Y alone is luminance; the checks that weigh it so keep their numbers.)"""
    if img.shape[-1] == 1:
        return img[..., 0]
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def image_diff(x, y):
    """The largest difference between two images as a share of the pixel, and how many values
    differ. A NaN in either makes the first NaN, which fails any `<` test."""
    rel = np.abs(x - y) / np.maximum(np.maximum(np.abs(x), np.abs(y)), 1e-12)
    return float(rel.max()), int((x != y).sum())
