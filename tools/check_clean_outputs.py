"""Regression check: tools/clean_outputs.py deletes what its rules name and nothing else.

    D:/python314/python.exe -u tools/check_clean_outputs.py

Builds a throwaway repository with one file per rule and per exception -- stale and fresh
checkpoints, an unassembled sequence's checkpoints, frame sets with a complete video, a
subsampled gif, a video older than its frames, a video named for another set, photon maps
older, current and newer than the loader, scraps/ data and sources, citations direct, by bare
name, through a cited script, an OBJ's material library, a directory, a glob and braces, from an
ignored note and from a virtual environment, a junction, a copy with an old modification time,
and a live render's directory -- runs the cleaner's finder over it, and compares the result
with the list it should give. Exits 1 on any difference.
"""
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clean_outputs as co  # noqa: E402

DAY = 86400.0


def set_created(path, when):
    """Back-date a file's creation time (Windows; elsewhere the cleaner reads no birth time)."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateFileW.restype = wintypes.HANDLE
    k.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                              wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    k.SetFileTime.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME), wintypes.LPVOID,
                              wintypes.LPVOID)
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    h = k.CreateFileW(path, 0x100, 7, None, 3, 0x02000000, None)   # write attributes, existing
    if h is None or h == wintypes.HANDLE(-1).value:
        raise OSError(ctypes.get_last_error(), "cannot open to set its creation time", path)
    ft = int(when * 10 ** 7) + 116444736000000000
    ok = k.SetFileTime(h, ctypes.byref(wintypes.FILETIME(ft & 0xFFFFFFFF, ft >> 32)), None, None)
    k.CloseHandle(h)
    if not ok:
        raise OSError(ctypes.get_last_error(), "SetFileTime failed", path)


def build(t):
    """Lay out the test tree under `t`; return (files a rule must take, files that must stay)."""
    gone, kept = set(), set()

    def put(rel, data=b"x", age_days=0.0, created_now=False):
        p = os.path.join(t, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
        when = time.time() - age_days * DAY
        os.utime(p, (when, when))
        if not created_now:
            set_created(p, when)
        return p

    def gif(rel, n, age_days=0.0):
        from PIL import Image
        p = os.path.join(t, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        frames = [Image.new("RGB", (4, 4), (k * 20, 0, 0)) for k in range(n)]
        frames[0].save(p, save_all=True, append_images=frames[1:], duration=40, loop=0)
        when = time.time() - age_days * DAY
        os.utime(p, (when, when))
        set_created(p, when)

    def frames(d, stem, n, age_days=0.5, sib=(".png.ftbuf", ".pfm")):
        names = []
        for k in range(n):
            names.append("%s/%s%03d.png" % (d, stem, k))
            put(names[-1], age_days=age_days)
            for s in sib:
                names.append("%s/%s%03d%s" % (d, stem, k, s))
                put(names[-1], age_days=age_days)
        return names

    os.makedirs(os.path.join(t, "src"))
    with open(os.path.join(co.ROOT, "src", "photonmap_io.h"), "rb") as f:
        hdr = f.read()
    with open(os.path.join(t, "src", "photonmap_io.h"), "wb") as f:
        f.write(hdr)

    # ftbuf / pfm: by age -- the later of written and created
    put("png/a/old.png.ftbuf", age_days=10); gone.add("png/a/old.png.ftbuf")
    put("png/a/new.png.ftbuf", age_days=1); kept.add("png/a/new.png.ftbuf")
    put("png/a/copied.png.ftbuf", age_days=10, created_now=True); kept.add("png/a/copied.png.ftbuf")
    put("png/a/old.pfm", age_days=10); gone.add("png/a/old.pfm")
    put("png/a/new.pfm", age_days=1); kept.add("png/a/new.pfm")
    put("png/a/old.png", age_days=10); kept.add("png/a/old.png")            # a still: no rule
    # an unassembled sequence keeps its checkpoints (loom resumes and re-develops by them)
    for f in frames("png/seq", "seq", 9, age_days=10):
        (gone if f.endswith(".pfm") else kept).add(f)

    # frames: a video beside no other set, holding exactly as many, takes the set at any age
    gone.update(frames("png/fly", "fly", 10))
    for k in range(10):
        put("png/fly/fly%03d.ftsl" % k, age_days=0.5); kept.add("png/fly/fly%03d.ftsl" % k)
    put("png/fly/fly_cam.png", age_days=0.5); kept.add("png/fly/fly_cam.png")
    gif("png/fly/flyby30.gif", 10); kept.add("png/fly/flyby30.gif")
    # ... but not a video of another length there
    kept.update(frames("png/one", "shot", 9))
    gif("png/one/clip.gif", 20); kept.add("png/one/clip.gif")
    # ... nor a gif holding a third of the set, though named for it
    kept.update(frames("png/thin", "thin", 9))
    gif("png/thin/thin.gif", 3); kept.add("png/thin/thin.gif")
    # ... nor a video older than a frame (the set was re-rendered after it)
    kept.update(frames("png/redo", "redo", 9, age_days=0.0))
    gif("png/redo/redo.gif", 9, age_days=1); kept.add("png/redo/redo.gif")
    # beside other sets, only the video named for a set takes it -- not one sharing a prefix
    gone.update(frames("png/rain", "rain_fly", 9))
    kept.update(frames("png/rain", "rain_axicon_fly", 9))
    gif("png/rain/rain_fly.gif", 9); kept.add("png/rain/rain_fly.gif")
    gif("png/rain/rain.gif", 20); kept.add("png/rain/rain.gif")

    # pmap: only a format older than the oldest the loader reads
    put("gallery/old.ftpmap", b"FTPMP01\n" + b"\0" * 64, age_days=0.5); gone.add("gallery/old.ftpmap")
    put("gallery/cur.ftpmap", b"FTPMP09\n" + b"\0" * 64, age_days=30); kept.add("gallery/cur.ftpmap")
    put("gallery/future.ftpmap", b"FTPMP99\n" + b"\0" * 64, age_days=30); kept.add("gallery/future.ftpmap")

    # scraps: old data goes, sources stay
    put("scraps/arr.npy", age_days=10); gone.add("scraps/arr.npy")
    put("scraps/new.npy", age_days=1); kept.add("scraps/new.npy")
    put("scraps/try.py", age_days=10); kept.add("scraps/try.py")
    put("scraps/dump.txt", b"y" * (co.SCRAPS_DATA_SIZE + 1), age_days=10); gone.add("scraps/dump.txt")
    put("scraps/sub/a.log", age_days=10); gone.add("scraps/sub/a.log")
    put("scraps/.settle_cache/iso_1_ab.obj", age_days=60); kept.add("scraps/.settle_cache/iso_1_ab.obj")

    # citations, and what they protect
    put("scraps/cited.npy", age_days=10); kept.add("scraps/cited.npy")
    put("scraps/repro.py", b"load('scraps/repro_mesh.obj'); open(\"near.vdb\")", age_days=10)
    kept.add("scraps/repro.py")
    put("scraps/repro_mesh.obj", age_days=10); kept.add("scraps/repro_mesh.obj")  # via repro.py
    put("scraps/near.vdb", age_days=10); kept.add("scraps/near.vdb")              # relative name
    put("scraps/bare.vdb", age_days=10); kept.add("scraps/bare.vdb")              # bare, joined on
    put("scraps/model.obj", b"mtllib model.mtl\nv 0 0 0\n", age_days=10); kept.add("scraps/model.obj")
    put("scraps/model.mtl", b"newmtl a\nmap_Kd -s 1 1 1 tex.png\n", age_days=10)
    kept.add("scraps/model.mtl")
    put("scraps/tex.png", age_days=10); kept.add("scraps/tex.png")                # via the .mtl
    put("png/cited_frames/c000.pfm", age_days=10); kept.add("png/cited_frames/c000.pfm")
    put("scraps/rig/run.ftsl", b'env "env.exr"', age_days=10); kept.add("scraps/rig/run.ftsl")
    put("scraps/rig/env.exr", age_days=10); kept.add("scraps/rig/env.exr")        # dir -> scene
    put("scraps/rig/other.npy", age_days=10); gone.add("scraps/rig/other.npy")    # not loaded
    put("scraps/rig/dump.txt", b"y" * (co.SCRAPS_DATA_SIZE + 1), age_days=10)    # a dump, not a
    gone.add("scraps/rig/dump.txt")                                                # source of it
    put("scraps/rig/sub/deep.ftsl", b'env "deep.exr"', age_days=10); kept.add("scraps/rig/sub/deep.ftsl")
    put("scraps/rig/sub/deep.exr", age_days=10); gone.add("scraps/rig/sub/deep.exr")  # not the rig's
    put("scraps/g/grey_a.ftsl", b'env "grey_a.exr"', age_days=10); kept.add("scraps/g/grey_a.ftsl")
    put("scraps/g/grey_a.exr", age_days=10); kept.add("scraps/g/grey_a.exr")      # glob -> scene
    for k in (1, 2, 3):
        put("png/b/groom%d_x.pfm" % k, age_days=10)
    kept.update({"png/b/groom1_x.pfm", "png/b/groom2_x.pfm"}); gone.add("png/b/groom3_x.pfm")
    put("scraps/big_cited.npy", age_days=10); kept.add("scraps/big_cited.npy")
    put("scraps/ign_cited.npy", age_days=10); kept.add("scraps/ign_cited.npy")
    put("scraps/venv_cited.npy", age_days=10); gone.add("scraps/venv_cited.npy")

    put("png/tracked.pfm", age_days=10); kept.add("png/tracked.pfm")
    put("notes.md", b"Repro: scraps/repro.py, data scraps/cited.npy, and `png/cited_frames/c000.pfm`."
                    b" Rig: `scraps/rig/`. Greys: scraps/g/grey_*.ftsl. Grooms: png/b/groom{1,2}_*.pfm."
                    b" Model: scraps/model.obj.")
    put("tests/test_x.py", b'SCRAPS = "../scraps"\npath = os.path.join(SCRAPS, "bare.vdb")\n')
    # a source past the dump-size cap still counts (known-issues.md is megabytes)
    put("big.md", b"x " * (co.SCAN_CAP // 2 + 1) + b"see scraps/big_cited.npy")
    put("scenes/untracked.ftsl", b'mesh "../scraps/scene_mesh.glb"')       # untracked source
    put("scraps/scene_mesh.glb", age_days=10); kept.add("scraps/scene_mesh.glb")
    put("local/plan.txt", b"re-check scraps/ign_cited.npy")                 # an ignored note
    put("proj/.venv/lib/mod.py", b"scraps/venv_cited.npy")                  # an environment
    put("png/live/old.png.ftbuf", age_days=10); kept.add("png/live/old.png.ftbuf")

    if os.name == "nt":                             # a junction out of an output root
        import _winapi
        put("outside/old.png.ftbuf", age_days=10)
        _winapi.CreateJunction(os.path.join(t, "outside"), os.path.join(t, "png", "link"))
        kept.add("png/link/old.png.ftbuf")

    with open(os.path.join(t, ".gitignore"), "w", encoding="utf-8", newline="") as f:
        f.write("/scraps/\npng/\n!png/tracked.pfm\ngallery/\nlocal/\n.venv/\noutside/\n")
    git = ["git", "-C", t, "-c", "user.email=t@t", "-c", "user.name=t", "-c", "core.autocrlf=false"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "notes.md", "big.md", "tests/test_x.py", ".gitignore",
                          "src/photonmap_io.h"], check=True)
    subprocess.run(git + ["add", "-f", "png/tracked.pfm"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "t"], check=True)

    chan = os.path.join(t, "_channel")              # a live render writing png/live/
    os.makedirs(chan)
    with open(os.path.join(chan, "%d.run" % os.getpid()), "w", encoding="utf-8") as f:
        f.write("x.ftsl -> png/live/out.png\n")
    with open(os.path.join(chan, "999999.run"), "w", encoding="utf-8") as f:   # long gone
        f.write("x.ftsl -> png/a/out.png\n")
    return gone, kept, chan


def main():
    real_root, real_chan = co.ROOT, co.stop_channel_dir
    with tempfile.TemporaryDirectory(prefix="clean_outputs_") as t:
        try:
            gone, kept, chan = build(t)
            co.ROOT = t
            co.stop_channel_dir = lambda: chan
            found = {co.rel(full) for _, full, _ in co.find(set(co.RULES), 7.0)}
        finally:
            co.ROOT, co.stop_channel_dir = real_root, real_chan
            link = os.path.join(t, "png", "link")
            if os.path.isdir(link):
                os.rmdir(link)                      # the junction itself, never its target's files

    ok = True
    for f in sorted(gone - found):
        print("FAIL: kept %s, which a rule names" % f)
        ok = False
    for f in sorted(found & kept):
        print("FAIL: would delete %s, which must stay" % f)
        ok = False
    for f in sorted(found - gone - kept):
        print("FAIL: would delete %s, which the check did not expect" % f)
        ok = False
    print("PASS (%d deleted, %d kept)" % (len(gone), len(kept)) if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
