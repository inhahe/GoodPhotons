"""Prune the regenerable render output that nothing will read again.

    D:/python314/python.exe tools/clean_outputs.py            # dry run: what would go, and why
    D:/python314/python.exe tools/clean_outputs.py --delete   # delete it
        [--days N]        the age at which the checkpoint, HDR and scraps rules take a file
                          (default 7); the frames and pmap rules do not look at age
        [--only R[,R..]]  run only these rules: ftbuf, pfm, frames, pmap, scraps
        [--list]          name every file, not only each directory's total

Renders and checks leave files whose use ends soon after they are written, and nothing else
removes them. The rules, each reported on its own ("age" is the time since the file was last
written or created, whichever is later, so a file copied in today is new whatever its
modification time says):

  ftbuf   A `-checkpoint` sidecar under png/ or gallery/ older than --days. It resumes an
          interrupted render, extends a finished one, or re-develops it (`-topng`) -- all within
          hours of the render, not weeks. A numbered sequence's checkpoints are the exception:
          loom resumes a sequence by them (`render_range(skip_existing=True)`) and re-develops
          its exposure from them, so they stay until the frames rule takes the sequence.
  pfm     A `-hdr` sidecar under png/ or gallery/ older than --days: written for a comparison
          that has since been made.
  frames  The numbered frames under png/ (`<name>000.png` ..., at least 8 of them) of a set
          already assembled into a video beside them, with each frame's .ftbuf, .pfm, .ppm and
          .jpg -- at any age. The video must be newer than every frame and either be named for
          the set (`<name>.mp4` for `<name>000.png`) and hold at least as many frames, or be
          beside no other set and hold exactly as many (a flyby's own directory, whose video
          is often named for its frame rate). Frames are counted with PIL for a gif, ffprobe
          for the rest, so a gif that keeps one frame in three releases nothing. The video
          stays, and so do the per-frame scenes, the logs and every unnumbered still.
  pmap    A photon map (`-savemap`) whose format is older than the oldest one the loader in
          src/photonmap_io.h still reads. A newer or unrecognised format is left alone.
  scraps  A file in scraps/ older than --days that is data rather than a script, scene or note:
          anything not source text, and any file over 1 MB whatever its name. scraps/.settle_cache
          is left to tools/settle_scene.py, which prunes its own.

Never touched, by any rule:
  - a file git tracks;
  - a file that something outside the output roots cites, or that a cited file cites in turn.
    The citers are every text source in the tree -- tracked, untracked, and ignored local notes
    and scripts (CLAUDE.md, notes/, the root .bat files), but not virtual environments, build
    directories or vendored code. A citation is a path into png/, scraps/ or gallery/ (a file;
    a directory, standing for the scripts, scenes and notes directly in it; a glob, braces
    included), or a quoted file name resolved beside the citing file, from the repo root, or
    from the top of an output root (a test that joins "_sphere.vdb" onto its scraps path).
    Cited sources, OBJ/MTL files and glTF files are read in turn, so a repro script keeps the
    mesh it loads and the mesh its textures;
  - anything under the output directory of a live ftrace process (its stop-channel registration
    in %TEMP%/ftrace, which `ftrace -stop` lists);
  - anything reached through a directory junction or symlink inside an output root.
"""
import argparse
import bisect
import collections
import fnmatch
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRUNED = ("png", "scraps", "gallery")          # the output roots the rules delete from
RULES = ("ftbuf", "pfm", "frames", "pmap", "scraps")
MB = 1 << 20

# Text a person writes: what citations are read from, and (up to SCRAPS_DATA_SIZE, see
# is_source) what scraps/ keeps whatever its age.
SOURCE_EXT = {".md", ".py", ".ftsl", ".txt", ".bat", ".cmd", ".ps1", ".sh", ".json", ".c", ".cc",
              ".cpp", ".h", ".hpp", ".cu", ".cuh", ".toml", ".yaml", ".yml", ".ini", ".cfg",
              ".csv", ".html", ".js", ".ts"}
# Also read when cited, for what they name in turn: an OBJ's material libraries, an MTL's
# textures, a glTF's buffers and images.
ASSET_REF_EXT = {".obj", ".mtl", ".gltf"}
SCAN_CAP = 4 * MB          # how much of an output file or an ignored file is read for citations
SCRAPS_DATA_SIZE = 1 * MB
VIDEO_EXT = {".gif", ".mp4", ".webm", ".mov", ".avi", ".mkv"}
FRAME_SIBLINGS = (".png.ftbuf", ".pfm", ".ppm", ".jpg")
MIN_FRAMES = 8
FRAME_RE = re.compile(r"^(?P<stem>.*?)(?P<num>\d{2,})\.png$", re.IGNORECASE)
# Directories whose files cite nothing of ours: environments, caches, CMake output, vendored code.
NOT_SOURCE_DIRS = {".git", ".venv", "venv", "node_modules", "site-packages", "__pycache__",
                   ".pytest_cache", ".mypy_cache", ".vs", ".claude", "third_party"}
# A path into an output root, as a script, scene or document writes it: either slash, any
# leading ../ dropped (which resolves it from the repo root, where every such path points), and
# glob and brace characters kept.
CITE_RE = re.compile(r"(?<![A-Za-z0-9_.])(?:\.\.?[\\/])*((?:%s)[\\/][^\s\"'`<>|;()\[\]]+)"
                     % "|".join(PRUNED))
QUOTED_RE = re.compile(r"[\"']([^\"'\r\n*?]{1,260}\.[A-Za-z0-9]{1,8})[\"']")
# An OBJ's material libraries and an MTL's maps: the file names follow the keyword unquoted.
OBJ_REF_RE = re.compile(r"^[ \t]*(?:mtllib|map_\w+|bump|disp|decal|norm|refl)[ \t]+(.+?)[ \t]*$",
                        re.MULTILINE)


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


def is_source(name, size):
    """Whether a file under an output root is a script, scene or note rather than data: source
    text no larger than SCRAPS_DATA_SIZE (a bigger "text" file is a dump or a log)."""
    return os.path.splitext(name)[1].lower() in SOURCE_EXT and size <= SCRAPS_DATA_SIZE


def is_link(full):
    """A symlink or an NTFS junction: os.walk descends a junction, so it is pruned by hand."""
    return os.path.islink(full) or (hasattr(os.path, "isjunction") and os.path.isjunction(full))


def age_stamp(st):
    """When a file was last written or created, whichever is later: a copy keeps its source's
    modification time but is created now."""
    return max(st.st_mtime, getattr(st, "st_birthtime", 0.0))


class Tree:
    """Every file under the output roots, walked once: repo-relative path -> os.stat result, the
    files of each directory, and a sorted list for directory and glob lookups."""

    def __init__(self):
        self.stat = {}
        self.dirs = collections.defaultdict(list)          # full directory path -> file names
        for top in PRUNED:
            for d, subs, files in os.walk(os.path.join(ROOT, top)):
                subs[:] = [s for s in subs
                           if s != ".settle_cache" and not is_link(os.path.join(d, s))]
                for f in files:
                    full = os.path.join(d, f)
                    try:
                        self.stat[rel(full)] = os.stat(full)
                    except OSError:
                        continue
                    self.dirs[d].append(f)
        self.sorted = sorted(self.stat)

    def under(self, prefix):
        """The files whose repo-relative path starts with `prefix`."""
        i = bisect.bisect_left(self.sorted, prefix)
        while i < len(self.sorted) and self.sorted[i].startswith(prefix):
            yield self.sorted[i]
            i += 1

    def match(self, r, wide):
        """The files the repo-relative path `r` names: itself; with `wide`, also every match
        when it is a glob, and the scripts, scenes and notes directly in it when it is a
        directory below a root. A cited directory is a rig, and what its scripts and scenes
        load is protected through them; protecting all of it would keep a tool's whole default
        output directory -- every checkpoint and sidecar in it -- for good."""
        if r in self.stat:
            return {r}
        if not wide:
            return set()
        if "*" in r or "?" in r:
            lit = r[:min(i for i in (r.find("*"), r.find("?")) if i >= 0)]
            return {f for f in self.under(lit.rsplit("/", 1)[0] + "/") if fnmatch.fnmatchcase(f, r)}
        d = r.rstrip("/")
        if d in PRUNED or "/" not in d:
            return set()                      # a whole output root is no citation
        return {f for f in self.under(d + "/") if "/" not in f[len(d) + 1:]
                and (is_source(f, self.stat[f].st_size) or f.lower().endswith(".gltf"))}


def expand_braces(s):
    """`groom{1,2}_*.png` -> [`groom1_*.png`, `groom2_*.png`]."""
    m = re.search(r"\{([^{}]*,[^{}]*)\}", s)
    if not m:
        return [s]
    return [x for alt in m.group(1).split(",")
            for x in expand_braces(s[:m.start()] + alt + s[m.end():])]


def citations(path, text, tree):
    """The repo-relative output files that the file at `path` names."""
    here = os.path.dirname(path)
    out = set()
    root = os.path.join(ROOT, "")

    def named(c, wide, bases):
        for base in bases:
            full = os.path.normpath(os.path.join(base, c))
            if full.startswith(root):
                r = full[len(root):].replace(os.sep, "/")
                if under_pruned(r):
                    out.update(tree.match(r, wide))

    if os.path.splitext(path)[1].lower() in (".obj", ".mtl"):
        for m in OBJ_REF_RE.finditer(text):
            for tok in m.group(1).split():
                named(tok, False, (here,))
        return out
    for m in CITE_RE.finditer(text):
        for c in expand_braces(m.group(1).replace("\\", "/").rstrip(".,:;")):
            out.update(tree.match(c, True))
    roots = tuple(os.path.join(ROOT, r) for r in PRUNED)
    for m in QUOTED_RE.finditer(text):
        named(m.group(1).replace("\\", "/"), False, (here, ROOT) + roots)
    return out


def ignored_sources():
    """Ignored text sources outside the output roots: local notes and scripts such as CLAUDE.md,
    notes/ and the root .bat files -- not environments, caches or build directories."""
    def skipped(seg):
        return seg in NOT_SOURCE_DIRS or seg.lower().startswith("build")

    out = []
    for e in git_files("--others", "--ignored", "--exclude-standard", "--directory"):
        segs = e.rstrip("/").split("/")
        if segs[0] in PRUNED or any(skipped(s) for s in (segs if e.endswith("/") else segs[:-1])):
            continue
        if not e.endswith("/"):
            if os.path.splitext(e)[1].lower() in SOURCE_EXT:
                out.append(e)
            continue
        for d, subs, files in os.walk(os.path.join(ROOT, e)):
            subs[:] = [s for s in subs if not skipped(s) and not is_link(os.path.join(d, s))]
            out += [rel(os.path.join(d, f)) for f in files
                    if os.path.splitext(f)[1].lower() in SOURCE_EXT]
    return out


def protected_files(tracked, tree):
    """Every output-root file that a source cites, closed over the cited files' own citations:
    a doc cites scraps/repro.py, which loads scraps/repro_mesh.obj, which names its .mtl."""
    curated = set(tracked) | set(git_files("--others", "--exclude-standard"))
    todo = [(f, None) for f in curated if os.path.splitext(f)[1].lower() in SOURCE_EXT
            and not NOT_SOURCE_DIRS.intersection(f.split("/")[:-1])]
    todo += [(f, SCAN_CAP) for f in ignored_sources()]
    seen, cited = {f for f, _ in todo}, set()
    while todo:
        f, cap = todo.pop()
        full = os.path.join(ROOT, f)
        try:
            with open(full, "rb") as fh:
                text = (fh.read(cap) if cap else fh.read()).decode("utf-8", "replace")
        except OSError:
            continue
        for c in citations(full, text, tree):
            cited.add(c)
            ext = os.path.splitext(c)[1].lower()
            if c not in seen and (ext in SOURCE_EXT or ext in ASSET_REF_EXT):
                seen.add(c)
                todo.append((c, SCAN_CAP))
    return cited


def pid_alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)                      # signal 0 is an existence test on POSIX only
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    h = k.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ctypes.get_last_error() == 5      # access denied: it exists
    code = wintypes.DWORD()
    ok = k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return not ok or code.value == 259           # STILL_ACTIVE


def stop_channel_dir():
    """Where every ftrace process registers itself (stopChannelDir in src/main.cpp)."""
    return os.path.join(tempfile.gettempdir(), "ftrace")


def live_output_dirs():
    """Output directories of the ftrace processes alive now, from their stop-channel
    registrations: <pid>.run, whose first line is "<scene> -> <output>" (stopChannelStart in
    src/main.cpp). A relative output is taken from the repo root."""
    d = stop_channel_dir()
    dirs = []
    for f in (os.listdir(d) if os.path.isdir(d) else []):
        if not f.endswith(".run") or not f[:-4].isdigit() or not pid_alive(int(f[:-4])):
            continue
        try:
            with open(os.path.join(d, f), encoding="utf-8", errors="replace") as fh:
                out = fh.readline().strip().rsplit(" -> ", 1)[-1].strip()
        except OSError:
            continue
        if not out or out.startswith("("):     # "(default output)", "(loom viewer)"
            continue
        full = out if os.path.isabs(out) else os.path.join(ROOT, out)
        dirs.append(os.path.normcase(os.path.dirname(os.path.abspath(full))))
        print("[clean] live render (pid %s) writing %s: its directory is left alone" % (f[:-4], out))
    return dirs


def pmap_oldest_read():
    """The oldest photon-map format loadPhotonMap reads, off its `const bool vN = ...` lines."""
    with open(os.path.join(ROOT, "src", "photonmap_io.h"), encoding="utf-8") as fh:
        text = fh.read()
    vs = [int(m.group(1)) for m in re.finditer(r'const bool v\d+\s*=.*?"FTPMP(\d\d)\\n"', text)]
    return min(vs) if vs else None


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
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        return int(r.stdout.strip()) if r.returncode == 0 else 0
    except (OSError, ValueError):
        return 0


def frame_sets(files):
    """{stem: [file names]} of the numbered sequences among `files` (at least MIN_FRAMES)."""
    sets = collections.defaultdict(list)
    for f in files:
        m = FRAME_RE.match(f)
        if m:
            sets[m.group("stem")].append(f)
    return {k: v for k, v in sets.items() if len(v) >= MIN_FRAMES}


def find(rules, days):
    """[(rule, full path, size)] of everything the rules would delete."""
    tracked = set(git_files())
    tree = Tree()
    keep = protected_files(tracked, tree)
    live = live_output_dirs()
    cutoff = time.time() - days * 86400
    out, taken = [], set()

    def take(rule, full):
        r = rel(full)
        if r in tracked or r in keep or r in taken or r not in tree.stat:
            return
        nd = os.path.normcase(os.path.abspath(full))
        if any(nd.startswith(d + os.sep) for d in live):
            return
        taken.add(r)
        out.append((rule, full, tree.stat[r].st_size))

    def old(full):
        return age_stamp(tree.stat[rel(full)]) < cutoff

    def in_root(d, *tops):
        return rel(d).split("/", 1)[0] in tops

    sequence_frames = set()                       # full paths of every numbered frame
    for d, files in tree.dirs.items():
        if not in_root(d, "png", "gallery"):
            continue
        sets = frame_sets(files)
        for frames in sets.values():
            sequence_frames.update(os.path.join(d, f) for f in frames)
        if "frames" not in rules or not in_root(d, "png"):
            continue
        vids = [f for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXT]
        for stem, frames in sets.items():
            key = stem.lower().rstrip("_-. ")
            newest = max(age_stamp(tree.stat[rel(os.path.join(d, f))]) for f in frames)
            covered = False
            for v in vids:
                vfull = os.path.join(d, v)
                if age_stamp(tree.stat[rel(vfull)]) < newest - 2.0:
                    continue                      # older than a frame: may not show it
                named = os.path.splitext(v)[0].lower().rstrip("_-. ") == key
                if not named and len(sets) != 1:
                    continue
                n = video_frames(vfull)
                if (named and n >= len(frames)) or n == len(frames):
                    covered = True
                    break
            if not covered:
                continue
            for f in frames:
                full = os.path.join(d, f)
                take("frames", full)
                for sfx in FRAME_SIBLINGS:
                    take("frames", full[:-4] + sfx)

    for d, files in tree.dirs.items():
        if not in_root(d, "png", "gallery"):
            continue
        for f in files:
            low = f.lower()
            full = os.path.join(d, f)
            if "ftbuf" in rules and low.endswith(".ftbuf") and old(full) \
                    and full[:-len(".ftbuf")] not in sequence_frames:
                take("ftbuf", full)
            elif "pfm" in rules and low.endswith(".pfm") and old(full):
                take("pfm", full)

    if "pmap" in rules:
        oldest = pmap_oldest_read()
        if oldest is None:
            print("[clean] could not read the photon-map formats the loader accepts; "
                  "skipping the pmap rule")
        else:
            for d, files in tree.dirs.items():
                for f in files:
                    if os.path.splitext(f)[1].lower() not in (".ftpmap", ".ftpm", ".pmap", ".map"):
                        continue
                    full = os.path.join(d, f)
                    with open(full, "rb") as fh:
                        m = re.match(rb"FTPMP(\d\d)\n", fh.read(8))
                    if m and int(m.group(1)) < oldest:
                        take("pmap", full)

    if "scraps" in rules:
        for d, files in tree.dirs.items():
            if not in_root(d, "scraps"):
                continue
            for f in files:
                full = os.path.join(d, f)
                if old(full) and not is_source(f, tree.stat[rel(full)].st_size):
                    take("scraps", full)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--delete", action="store_true", help="delete (default: only report)")
    ap.add_argument("--days", type=float, default=7.0,
                    help="age at which the ftbuf, pfm and scraps rules take a file (default 7)")
    ap.add_argument("--only", default=",".join(RULES), help="comma-separated rules to run")
    ap.add_argument("--list", action="store_true", help="name every file")
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
