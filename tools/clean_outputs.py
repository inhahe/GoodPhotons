"""Prune the regenerable render output that nothing will read again.

    D:/python314/python.exe tools/clean_outputs.py            # dry run: what would go, and why
    D:/python314/python.exe tools/clean_outputs.py --delete   # delete it
        [--days N]        never touch a file modified in the last N days (default 7)
        [--only R[,R..]]  run only these rules: ftbuf, pfm, frames, pmap, scraps
        [--list]          name every file, not only each directory's total

Renders and checks leave files whose use ends soon after they are written, and nothing removed
them: by 2026-10 png/ held 12 GB of checkpoints and 5 GB of HDR sidecars, scraps/ 10 GB, and
gallery/ a 4.7 GB photon map in a format the loader no longer reads. The rules, each reported
on its own:

  ftbuf   A `-checkpoint` sidecar under png/ or gallery/ older than --days. It resumes an
          interrupted render, extends a finished one, or re-develops it (`-topng`, loom's
          stabilize_exposure) -- all within hours of the render, not weeks.
  pfm     A `-hdr` sidecar under png/ or gallery/ older than --days: written for a comparison
          that has since been made.
  frames  The numbered frames under png/ (`<name>000.png` ..., at least 8 of them) of a set
          already assembled into a video beside them -- a .gif/.mp4 newer than every frame,
          holding at least as many frames (counted: PIL for a gif, ffprobe for the rest), and
          either named with the set's prefix or the only set in its directory -- with each
          frame's .ftbuf, .pfm, .ppm and .jpg. At any age. The video stays, and so do the
          per-frame scenes, the logs and every unnumbered still (the `_cam` view, the `_meter`
          frame). A set re-rendered or re-developed after its video stays too.
  pmap    A photon map (`-savemap`) whose magic the loader in src/photonmap_io.h refuses.
  scraps  A file in scraps/ older than --days that is data rather than a script, scene or note:
          anything not source text, and any file over 1 MB whatever its name. scraps/.settle_cache
          is left to tools/settle_scene.py, which prunes its own.

Never touched, by any rule: a file git tracks; a file a script, scene or document cites by path
-- the reproductions known-issues.md points at in scraps/, the meshes and volumes a scene loads
-- or that such a cited file cites in turn; and anything under the output directory of a live
ftrace process (`ftrace -stop` lists them).
"""
import argparse
import collections
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRUNED = ("png", "scraps", "gallery")          # the output roots the rules delete from
RULES = ("ftbuf", "pfm", "frames", "pmap", "scraps")
MB = 1 << 20

# Text worth scanning for citations, and what counts as source (kept) in scraps/.
TEXT_EXT = {".md", ".py", ".ftsl", ".txt", ".bat", ".cmd", ".ps1", ".sh", ".json", ".c", ".cc",
            ".cpp", ".h", ".hpp", ".cu", ".cuh", ".toml", ".yaml", ".yml", ".ini", ".cfg",
            ".csv", ".html", ".js", ".ts"}
SCAN_CAP = 4 * MB                              # past this a "text" file is a dump, not a source
SCRAPS_DATA_SIZE = 1 * MB
VIDEO_EXT = {".gif", ".mp4", ".webm", ".mov", ".avi", ".mkv"}
FRAME_SIBLINGS = (".png.ftbuf", ".pfm", ".ppm", ".jpg")
MIN_FRAMES = 8
FRAME_RE = re.compile(r"^(?P<stem>.*?)(?P<num>\d{2,})\.png$", re.IGNORECASE)
# A path into an output root, as a script, scene or document writes it (either slash; any
# leading ../ is dropped, which resolves it from the repo root -- where every such path points).
CITE_RE = re.compile(r"(?<![A-Za-z0-9_.])(?:\.\.?[\\/])*((?:%s)[\\/][^\s\"'`<>|*?,;()\[\]{}]+)"
                     % "|".join(PRUNED))
QUOTED_RE = re.compile(r"[\"']([^\"'\r\n]{1,260}\.[A-Za-z0-9]{1,8})[\"']")


def rel(p):
    """`p` as a repo-relative path with forward slashes."""
    return os.path.relpath(p, ROOT).replace("\\", "/")


def git_files(*args):
    r = subprocess.run(["git", "-C", ROOT, "ls-files", "-z", *args], capture_output=True)
    if r.returncode != 0:
        sys.exit("[clean] git ls-files failed: " + r.stderr.decode("utf-8", "replace"))
    return [f for f in r.stdout.decode("utf-8", "replace").split("\0") if f]


def under_pruned(r):
    return r.split("/", 1)[0] in PRUNED


def citations(path, text):
    """The repo-relative files under an output root that the file at `path` names."""
    out = set()
    here = os.path.dirname(path)
    for m in CITE_RE.finditer(text):
        c = m.group(1).replace("\\", "/").rstrip(".:")
        if os.path.isfile(os.path.join(ROOT, c)):
            out.add(c)
    for m in QUOTED_RE.finditer(text):                     # a scene's asset, named relatively
        c = m.group(1).replace("\\", "/")
        for base in (here, ROOT):
            full = os.path.normpath(os.path.join(base, c))
            if full.startswith(ROOT) and os.path.isfile(full):
                r = rel(full)
                if under_pruned(r):
                    out.add(r)
                break
    return out


def protected_files(tracked):
    """Every output-root file that a source cites, closed over the cited files' own citations:
    a doc cites scraps/repro.py, which loads scraps/repro_mesh.obj."""
    seeds = set(tracked) | set(git_files("--others", "--exclude-standard"))
    todo = [f for f in seeds if os.path.splitext(f)[1].lower() in TEXT_EXT]
    seen, cited = set(todo), set()
    while todo:
        f = todo.pop()
        full = os.path.join(ROOT, f)
        try:
            if os.path.getsize(full) > SCAN_CAP:
                continue
            with open(full, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        for c in citations(full, text):
            cited.add(c)
            if c not in seen and os.path.splitext(c)[1].lower() in TEXT_EXT:
                seen.add(c)
                todo.append(c)
    return cited


def live_output_dirs():
    """Output directories of running ftrace processes, from `ftrace -stop`'s listing ("pid N
    <scene> -> <out>"). Exits when there is an ftrace to ask and it cannot be asked."""
    exe = os.path.join(ROOT, "ftrace.exe")
    if not os.path.isfile(exe):
        print("[clean] no ftrace.exe to list live renders; relying on the age rules")
        return []
    r = subprocess.run([exe, "-stop"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        sys.exit("[clean] `ftrace -stop` failed, so live renders cannot be protected:\n" + r.stdout
                 + r.stderr)
    dirs = []
    for line in r.stdout.splitlines():
        m = re.match(r"\s+pid\s+\d+\s+(.*)$", line)
        if not m:
            continue
        out = m.group(1).rsplit(" -> ", 1)[-1].strip()
        if out.startswith("("):                  # "(default output)", "(loom viewer)"
            continue
        full = out if os.path.isabs(out) else os.path.join(ROOT, out)
        dirs.append(os.path.normcase(os.path.dirname(os.path.abspath(full))))
        print("[clean] live render writing %s: its directory is left alone" % out)
    return dirs


def accepted_pmap_magics():
    """The photon-map magics loadPhotonMap accepts, read off its `const bool vN = ...` lines."""
    src = os.path.join(ROOT, "src", "photonmap_io.h")
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    return {m.group(1).encode() + b"\n"
            for m in re.finditer(r'const bool v\d+\s*=.*?"(FTPMP\d\d)\\n"', text)}


def video_frames(path):
    """How many frames the video at `path` holds; 0 when that cannot be told, which keeps the
    frames it might hold."""
    if path.lower().endswith(".gif"):
        try:
            from PIL import Image
            with Image.open(path) as im:
                return im.n_frames
        except Exception:
            return 0
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                            "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
                           capture_output=True, text=True)
        return int(r.stdout.strip()) if r.returncode == 0 else 0
    except (OSError, ValueError):
        return 0


def walk(top):
    for d, subs, files in os.walk(os.path.join(ROOT, top)):
        subs[:] = [s for s in subs if s != ".settle_cache"]
        yield d, files


def find(rules, days):
    """[(rule, full path, size)] of everything the rules would delete."""
    tracked = set(git_files())
    keep = protected_files(tracked)
    live = live_output_dirs()
    cutoff = time.time() - days * 86400
    out, taken = [], set()

    def take(rule, full):
        r = rel(full)
        if r in tracked or r in keep or r in taken:
            return
        nd = os.path.normcase(os.path.abspath(full))
        if any(nd.startswith(d + os.sep) for d in live):
            return
        taken.add(r)
        out.append((rule, full, os.path.getsize(full)))

    def old(full):
        return os.path.getmtime(full) < cutoff

    if "frames" in rules:
        for d, files in walk("png"):
            vids = [f for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXT]
            if not vids:
                continue
            sets = collections.defaultdict(list)
            for f in files:
                m = FRAME_RE.match(f)
                if m:
                    sets[m.group("stem")].append(f)
            sets = {k: v for k, v in sets.items() if len(v) >= MIN_FRAMES}
            for stem, frames in sets.items():
                key = stem.lower().rstrip("_-. ")
                newest = max(os.path.getmtime(os.path.join(d, f)) for f in frames)
                covered = False
                for v in vids:
                    if os.path.getmtime(os.path.join(d, v)) < newest - 2.0:
                        continue                 # older than a frame: may not show it
                    vk = os.path.splitext(v)[0].lower().rstrip("_-. ")
                    short = min(len(vk), len(key))
                    # Named for the set, or the only set in its directory -- a flyby's own
                    # subdirectory, whose video is often named for its frame rate -- and
                    # holding every frame (a gif often keeps one in three).
                    if (len(sets) == 1 or (short >= 3 and vk[:short] == key[:short])) \
                            and video_frames(os.path.join(d, v)) >= len(frames):
                        covered = True
                        break
                if not covered:
                    continue
                for f in frames:
                    full = os.path.join(d, f)
                    take("frames", full)
                    base = full[:-4]
                    for sfx in FRAME_SIBLINGS:
                        sib = base + sfx
                        if os.path.isfile(sib):
                            take("frames", sib)

    for top in ("png", "gallery"):
        for d, files in walk(top):
            for f in files:
                low = f.lower()
                full = os.path.join(d, f)
                if "ftbuf" in rules and low.endswith(".ftbuf") and old(full):
                    take("ftbuf", full)
                elif "pfm" in rules and low.endswith(".pfm") and old(full):
                    take("pfm", full)

    if "pmap" in rules:
        good = accepted_pmap_magics()
        if not good:
            print("[clean] could not read the accepted photon-map magics; skipping the pmap rule")
        else:
            for top in PRUNED:
                for d, files in walk(top):
                    for f in files:
                        if os.path.splitext(f)[1].lower() not in (".ftpmap", ".ftpm", ".pmap", ".map"):
                            continue
                        full = os.path.join(d, f)
                        with open(full, "rb") as fh:
                            magic = fh.read(8)
                        if magic.startswith(b"FTPMP") and magic not in good:
                            take("pmap", full)

    if "scraps" in rules:
        for d, files in walk("scraps"):
            for f in files:
                full = os.path.join(d, f)
                if not old(full):
                    continue
                if os.path.splitext(f)[1].lower() not in TEXT_EXT \
                        or os.path.getsize(full) > SCRAPS_DATA_SIZE:
                    take("scraps", full)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--delete", action="store_true", help="delete (default: only report)")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--only", default=",".join(RULES))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    rules = set(a.only.split(","))
    if rules - set(RULES):
        ap.error("unknown rule(s): " + ", ".join(sorted(rules - set(RULES))))
    found = find(rules, a.days)

    per = collections.defaultdict(lambda: [0, 0])
    for rule, full, size in found:
        r = rel(full)
        parts = r.split("/")
        group = "/".join(parts[:2]) if len(parts) > 2 else parts[0]
        per[(rule, group)][0] += 1
        per[(rule, group)][1] += size
        if a.list:
            print("  %-6s %9.1f MB  %s" % (rule, size / MB, r))
    for rule in RULES:
        rows = sorted(((k[1], v) for k, v in per.items() if k[0] == rule), key=lambda kv: -kv[1][1])
        if not rows:
            continue
        n = sum(v[0] for _, v in rows)
        s = sum(v[1] for _, v in rows)
        print("%-6s %7d files %10.1f MB" % (rule, n, s / MB))
        for g, (gn, gs) in rows[:12]:
            print("         %6d %10.1f MB  %s" % (gn, gs / MB, g))
        if len(rows) > 12:
            print("         ... and %d more directories" % (len(rows) - 12))
    total = sum(s for _, _, s in found)
    if not a.delete:
        print("would free %.2f GB in %d files (dry run: pass --delete)" % (total / (1 << 30), len(found)))
        return 0
    freed, failed, dirs = 0, [], set()
    for _, full, size in found:
        try:
            os.remove(full)
            freed += size
            dirs.add(os.path.dirname(full))
        except OSError as e:
            failed.append("%s: %s" % (rel(full), e.strerror))
    for d in sorted(dirs, key=len, reverse=True):           # directories the rules emptied,
        while "/" in rel(d) and os.path.isdir(d) and not os.listdir(d):   # never a root itself
            os.rmdir(d)
            d = os.path.dirname(d)
    print("freed %.2f GB in %d files" % (freed / (1 << 30), len(found) - len(failed)))
    for f in failed:
        print("  could not delete " + f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
