"""Build check: no function the register-capped GPU units compile out of line can be bound, at the
device link, to another unit's copy of it.

    python tools/check_device_linkage.py [--objdir build_cuda3/ftrace.dir/Release | --objects OBJ...]
                                         [--warn-only] [--cuobjdump PATH]

render_cuda_bdpt.cu (kBdptT, -maxrregcount=168) and render_cuda_jsurf.cu (kJSurfLightT, 128) each
compile their own copies of the device code they call, so that every function ptxas leaves out of
line there fits their kernels' launch bounds (render_cuda_dev.cuh). That holds for a function with
internal linkage. One with EXTERNAL linkage -- a non-static inline function, a template, a class's
member function -- that is out of line in a capped unit AND in another unit becomes a weak
definition in both objects, and nvlink keeps one of them for every caller: the other unit's,
compiled without the cap or to another one, may be the one, and then the link refuses kBdptT once
that copy -- or a function it calls, that unit's own (DVertexLobe::pdf, at 76 registers, bound
kJSurfLightT to render_cuda.cu's dMatRoughness at 252) -- needs more registers than the bounds
allow: after an edit anywhere in that unit.

So it lists, from `cuobjdump -res-usage` of the GPU objects, every externally linked function other
than the units' own kernels that a capped unit keeps out of line and another object does too --
render_cuda.obj and raster_cuda.obj, uncapped, or the other capped unit, whose copy is held to ITS
cap -- with both register counts: every one a hazard, whatever its count, for what it calls. The
fix is internal linkage: `static` (FT_UNIT_INLINE in a shared header, unit_linkage.h) for a
function, an unnamed namespace for a struct's members. Exits 1 when there is one (or when an object
or cuobjdump is missing), or with --warn-only prints the same as a warning and exits 0. The build
runs it after every link (CMakeLists.txt, POST_BUILD, given the target's objects by --objects); a
link that fails leaves no post-build step, so run it by hand then -- the objects are there -- to
name the function behind an nvlink refusal.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS = [("render_cuda_bdpt", 168, "kBdptT"), ("render_cuda_jsurf", 128, "kJSurfLightT")]
UNCAPPED = ["render_cuda", "raster_cuda"]
FUNC = re.compile(r"Function ([^:\s]+):")
REG = re.compile(r"REG:(\d+)")


def cuobjdump():
    path = shutil.which("cuobjdump")
    if path:
        return path
    cuda = os.environ.get("CUDA_PATH", "")
    cand = os.path.join(cuda, "bin", "cuobjdump.exe")
    return cand if cuda and os.path.exists(cand) else None


def external_functions(tool, obj):
    """{mangled name: registers} of the externally linked functions `obj`'s device code holds."""
    p = subprocess.run([tool, "-res-usage", obj], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if p.returncode != 0:
        raise SystemExit("cuobjdump failed on %s: %s" % (obj, (p.stdout + p.stderr)[-300:]))
    out, name = {}, None
    for line in p.stdout.splitlines():
        m = FUNC.search(line)
        if m:
            name = m.group(1)
            continue
        r = REG.search(line)
        if name and r:
            # CUDA's own math library's slow paths (__internal_*, __cuda_*) are not ours to give
            # internal linkage, and are fixed code far under either cap (44-52 registers).
            if "_INTERNAL_" not in name and not name.startswith(("__cuda_", "__internal_")):
                out[name] = int(r.group(1))
            name = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--objdir", default=os.path.join(ROOT, "build_cuda3", "ftrace.dir", "Release"),
                    help="the directory holding <unit>.obj (the Visual Studio generator's layout)")
    ap.add_argument("--objects", nargs="*", default=None,
                    help="the target's object files, any generator's names (<unit>.obj, <unit>.cu.obj...)")
    ap.add_argument("--warn-only", action="store_true", help="a hit is a warning: exit 0")
    ap.add_argument("--cuobjdump", default=None, help="the tool's path (default: PATH, then CUDA_PATH)")
    a = ap.parse_args()
    fail = 0 if a.warn_only else 1
    tool = a.cuobjdump if a.cuobjdump and os.path.exists(a.cuobjdump) else cuobjdump()
    if not tool:
        print("cuobjdump not found (PATH, CUDA_PATH)")
        return fail
    capped = [u[0] for u in UNITS]
    units = UNCAPPED + capped
    if a.objects is not None:
        stem = lambda f: os.path.basename(f).split(".")[0]
        objs = {stem(f): f for f in a.objects if stem(f) in units}
    else:
        objs = {n: os.path.join(a.objdir, n + ".obj") for n in units}
    missing = [n for n in units if n not in objs or not os.path.exists(objs[n])]
    if missing:
        print("missing object: " + ", ".join(missing))
        return fail
    fns = {n: external_functions(tool, o) for n, o in objs.items()}
    kernels = [u[2] for u in UNITS]
    bad = 0
    # Each capped unit against every other object: the uncapped ones, and the other capped unit,
    # whose copy is held to ITS cap -- kBdptT's 168 binding kJSurfLightT's calls at 128 refuses the
    # link as surely as an uncapped one. Each pair of capped units once.
    for unit, cap, kernel in UNITS:
        for other in objs:
            if other == unit or (other in capped and other < unit):
                continue
            shared = sorted((n for n in fns[unit] if n in fns[other] and not any(k in n for k in kernels)),
                            key=lambda n: -fns[other][n])
            for n in shared:
                bad += 1
                print("%s: %s is out of line with external linkage here (%d registers) and in %s.cu (%d)"
                      % (unit, n, fns[unit][n], other, fns[other][n]))
    if bad:
        print("%s%d function(s) the device link may bind to another unit's copy, held to another cap or none: "
              "give them internal linkage (FT_UNIT_INLINE or static; a struct's members, an unnamed namespace)"
              % ("WARNING: " if a.warn_only else "", bad))
        return fail
    print("ok: every function the capped units keep out of line is their own")
    return 0


if __name__ == "__main__":
    sys.exit(main())
