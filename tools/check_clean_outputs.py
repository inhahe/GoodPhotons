"""Regression check: tools/clean_outputs.py deletes what its rules name and nothing else.

    D:/python314/python.exe -u tools/check_clean_outputs.py

Builds a throwaway repository with one file per rule and per exception -- a stale and a fresh
checkpoint, a frame set with a complete video, one whose gif keeps every third frame, one
re-rendered after its video, photon maps the loader does and does not read, scraps/ data and
sources, files a document cites directly and through a cited script, a tracked image -- runs
the cleaner's finder over it, and compares the result with the list it should give. Exits 1
on any difference.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clean_outputs as co  # noqa: E402

DAY = 86400.0


def main():
    real_root = co.ROOT
    t = tempfile.mkdtemp(prefix="clean_outputs_")
    try:
        def put(rel, data=b"x", age_days=0.0):
            p = os.path.join(t, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(data)
            when = time.time() - age_days * DAY
            os.utime(p, (when, when))
            return p

        def gif(rel, n, age_days=0.0):
            from PIL import Image
            p = os.path.join(t, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            frames = [Image.new("RGB", (4, 4), (k * 20, 0, 0)) for k in range(n)]
            frames[0].save(p, save_all=True, append_images=frames[1:], duration=40, loop=0)
            when = time.time() - age_days * DAY
            os.utime(p, (when, when))

        os.makedirs(os.path.join(t, "src"))
        shutil.copy(os.path.join(real_root, "src", "photonmap_io.h"), os.path.join(t, "src"))
        gone, kept = set(), set()

        # ftbuf / pfm: by age
        put("png/a/old.png.ftbuf", age_days=10); gone.add("png/a/old.png.ftbuf")
        put("png/a/new.png.ftbuf", age_days=1); kept.add("png/a/new.png.ftbuf")
        put("png/a/old.pfm", age_days=10); gone.add("png/a/old.pfm")
        put("png/a/new.pfm", age_days=1); kept.add("png/a/new.pfm")
        put("png/a/old.png", age_days=10); kept.add("png/a/old.png")        # a still: no rule

        # frames: a complete video takes the set and its siblings, at any age
        for k in range(10):
            put("png/fly/fly%03d.png" % k, age_days=0.5); gone.add("png/fly/fly%03d.png" % k)
            put("png/fly/fly%03d.png.ftbuf" % k, age_days=0.5); gone.add("png/fly/fly%03d.png.ftbuf" % k)
            put("png/fly/fly%03d.pfm" % k, age_days=0.5); gone.add("png/fly/fly%03d.pfm" % k)
            put("png/fly/fly%03d.ftsl" % k, age_days=0.5); kept.add("png/fly/fly%03d.ftsl" % k)
        put("png/fly/fly_cam.png", age_days=0.5); kept.add("png/fly/fly_cam.png")
        gif("png/fly/flyby30.gif", 10); kept.add("png/fly/flyby30.gif")    # sole set: any name
        # ... a gif holding a third of them does not
        for k in range(9):
            put("png/thin/thin%03d.png" % k, age_days=0.5); kept.add("png/thin/thin%03d.png" % k)
        gif("png/thin/thin.gif", 3); kept.add("png/thin/thin.gif")
        # ... nor a video older than a frame (the set was re-rendered after it)
        for k in range(9):
            put("png/redo/redo%03d.png" % k, age_days=0.0); kept.add("png/redo/redo%03d.png" % k)
        gif("png/redo/redo.gif", 9, age_days=1); kept.add("png/redo/redo.gif")
        # ... nor, beside other sets, a video named for another one
        for k in range(9):
            put("png/two/aaa%03d.png" % k, age_days=0.5); gone.add("png/two/aaa%03d.png" % k)
            put("png/two/bbb%03d.png" % k, age_days=0.5); kept.add("png/two/bbb%03d.png" % k)
        gif("png/two/aaa_loop.gif", 9); kept.add("png/two/aaa_loop.gif")

        # pmap: the loader's magic decides
        put("gallery/old.ftpmap", b"FTPMP01\n" + b"\0" * 64, age_days=0.5); gone.add("gallery/old.ftpmap")
        put("gallery/new.ftpmap", b"FTPMP09\n" + b"\0" * 64, age_days=30); kept.add("gallery/new.ftpmap")

        # scraps: old data goes, sources and anything cited stay
        put("scraps/arr.npy", age_days=10); gone.add("scraps/arr.npy")
        put("scraps/new.npy", age_days=1); kept.add("scraps/new.npy")
        put("scraps/try.py", age_days=10); kept.add("scraps/try.py")
        put("scraps/dump.txt", b"y" * (co.SCRAPS_DATA_SIZE + 1), age_days=10); gone.add("scraps/dump.txt")
        put("scraps/sub/a.log", age_days=10); gone.add("scraps/sub/a.log")
        put("scraps/cited.npy", age_days=10); kept.add("scraps/cited.npy")
        put("scraps/repro.py", b"load('scraps/repro_mesh.obj'); open(\"near.vdb\")", age_days=10)
        kept.add("scraps/repro.py")
        put("scraps/repro_mesh.obj", age_days=10); kept.add("scraps/repro_mesh.obj")  # via repro.py
        put("scraps/near.vdb", age_days=10); kept.add("scraps/near.vdb")              # relative name
        put("png/cited_frames/c%03d.pfm" % 0, age_days=10); kept.add("png/cited_frames/c000.pfm")
        put("scraps/.settle_cache/iso_1_ab.obj", age_days=60); kept.add("scraps/.settle_cache/iso_1_ab.obj")

        # tracked, and a tracked doc citing the above
        put("png/tracked.pfm", age_days=10); kept.add("png/tracked.pfm")
        put("notes.md", b"Repro: scraps/repro.py, data scraps/cited.npy, and "
                        b"`png/cited_frames/c000.pfm`.")
        put("scenes/untracked.ftsl", b'mesh "../scraps/scene_mesh.glb"')       # untracked source
        put("scraps/scene_mesh.glb", age_days=10); kept.add("scraps/scene_mesh.glb")
        with open(os.path.join(t, ".gitignore"), "w", encoding="utf-8") as f:
            f.write("/scraps/\npng/\n!png/tracked.pfm\ngallery/\n")
        git = ["git", "-C", t, "-c", "user.email=t@t", "-c", "user.name=t"]
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "notes.md", ".gitignore", "src/photonmap_io.h"], check=True)
        subprocess.run(git + ["add", "-f", "png/tracked.pfm"], check=True)
        subprocess.run(git + ["commit", "-q", "-m", "t"], check=True)

        co.ROOT = t
        found = {co.rel(full) for _, full, _ in co.find(set(co.RULES), 7.0)}
    finally:
        co.ROOT = real_root
        shutil.rmtree(t, ignore_errors=True)

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
