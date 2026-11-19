"""Build check: no function the register-capped GPU units compile out of line can be bound, at the
device link, to render_cuda.cu's copy of it.

    python tools/check_device_linkage.py [--objdir build_cuda3/ftrace.dir/Release]

render_cuda_bdpt.cu (kBdptT, -maxrregcount=168) and render_cuda_jsurf.cu (kJSurfLightT, 128) each
compile their own copies of the device code they call, so that every function ptxas leaves out of
line there fits their kernels' launch bounds (render_cuda_dev.cuh). That holds for a function with
internal linkage. One with EXTERNAL linkage -- a non-static inline function, a template, a class's
member function -- that is out of line in a capped unit AND in render_cuda.cu becomes a weak
definition in both objects, and nvlink keeps one of them for every caller: render_cuda.cu's,
compiled without the cap, may be the one, and then the link refuses kBdptT once that copy -- or a
function it calls, render_cuda.cu's own (DVertexLobe::pdf, at 76 registers, bound kJSurfLightT to
render_cuda.cu's dMatRoughness at 252) -- needs more registers than the bounds allow: after an edit
anywhere in render_cuda.cu.

So it lists, from `cuobjdump -res-usage` of the three objects, every externally linked function
other than the unit's own kernels that a capped unit keeps out of line and render_cuda.obj does
too, with both register counts -- every one a hazard, whatever its count, for what it calls. The fix
is internal linkage: `static` for a function, an unnamed namespace for a struct's members. Exits 1 when there is one (or when an object or cuobjdump is missing); build.bat runs it
after every build.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS = [("render_cuda_bdpt", 168, "kBdptT"), ("render_cuda_jsurf", 128, "kJSurfLightT")]
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
    ap.add_argument("--objdir", default=os.path.join(ROOT, "build_cuda3", "ftrace.dir", "Release"))
    a = ap.parse_args()
    tool = cuobjdump()
    if not tool:
        print("cuobjdump not found (PATH, CUDA_PATH)")
        return 1
    objs = {n: os.path.join(a.objdir, n + ".obj") for n in ["render_cuda"] + [u[0] for u in UNITS]}
    missing = [o for o in objs.values() if not os.path.exists(o)]
    if missing:
        print("missing object: " + ", ".join(missing))
        return 1
    main_fns = external_functions(tool, objs["render_cuda"])
    bad = 0
    for unit, cap, kernel in UNITS:
        fns = external_functions(tool, objs[unit])
        shared = sorted((n for n in fns if n in main_fns and kernel not in n), key=lambda n: -main_fns[n])
        for n in shared:
            bad += 1
            print("%s: %s is out of line with external linkage here (%d registers) and in render_cuda.cu (%d%s)"
                  % (unit, n, fns[n], main_fns[n], ", over the cap %d" % cap if main_fns[n] > cap else ""))
    if bad:
        print("%d function(s) the device link may bind to render_cuda.cu's uncapped copy: give them internal "
              "linkage (static; a struct's members, an unnamed namespace)" % bad)
        return 1
    print("ok: every function the capped units keep out of line is their own")
    return 0


if __name__ == "__main__":
    sys.exit(main())
