# Mutation test for -checkfur: break one thing in src/fur.h, rebuild, and confirm the
# section that is supposed to catch it actually does. Restores the file afterwards.
#
# A guard that cannot fail is not a guard, so every -checkfur section owes a deliberate
# break that it — and ideally only it — catches. This is not ceremony: it has now caught a
# dead test twice, and both times the test looked fine.
#
#   mutation 6  — §3 could NOT detect the strands losing their seed, because it tested seed
#                 sensitivity with clumping on, and the guide rng alone moving was enough.
#   mutation 10 — §8 could NOT detect `bald` culling by ROOT only, because its zone sat ON
#                 the surface and was wider than the coat, so it swallowed the roots of
#                 everything that crossed it and the two implementations agreed exactly.
#                 §8 gained a second zone floating clear of the skin, containing no roots.
#
# The pattern in both: a fixture under which the correct and the broken implementation
# produce the same answer. Nothing but a mutation run finds those.
#
#   python tools/mutate_fur.py        # all of them (slow: one full rebuild each)
#   python tools/mutate_fur.py 6 9    # just these, by 1-based index
import io, os, re, subprocess, sys, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUR  = os.path.join(ROOT, "src", "fur.h")

MUTS = [
    ("no sqrt barycentric warp (roots bunch at one vertex)",
     "const double su = std::sqrt(u);", "const double su = u;", [2]),
    ("triangle picked uniformly, not by area",
     "const double a = 0.5 * length(cross(t.v1 - t.v0, t.v2 - t.v0));",
     "const double a = 1.0;", [2]),
    ("read the un-finalized geometric normal (the shipped bug)",
     "if (dot(sn, sn) < 1e-24) sn = cross(t.v1 - t.v0, t.v2 - t.v0);",
     "if (dot(sn, sn) < 1e-24) sn = t.gn;", [7]),
    ("no minimum-grazing-angle clamp",
     "if (dot(dir, n) < minCos) dir = normalize(dir + n * (minCos - dot(dir, n)));",
     "(void)minCos;", [4]),
    ("clump weight constant instead of proportional to t (moves roots)",
     "const double w = spec.clump * t;", "const double w = spec.clump;", [5]),
    ("strand seed ignores spec.seed",
     "rng.seed(mix64((uint64_t)i * 2 + 1), mix64(spec.seed + (uint64_t)i));",
     "rng.seed(mix64((uint64_t)i * 2 + 1), mix64((uint64_t)i));", [3]),
    ("taper inverted (tip fatter than root)",
     "radii[(size_t)k] = spec.radius + (spec.radiusTip - spec.radius) * t;",
     "radii[(size_t)k] = spec.radius * (1.0 + t);", [6]),
    ("sphere roots float above the surface",
     "p = s.center + n * s.radius;", "p = s.center + n * (s.radius * 0.99);", [1]),
    # The clump GUIDES are the second, independent consumer of spec.seed. The original §3
    # tested seed sensitivity once, with clumping on, which could not tell the two apart —
    # either path moving alone was enough to pass. Both are now mutated separately.
    ("guide seed ignores spec.seed",
     "rng.seed(mix64(spec.seed ^ 0x9E3779B97F4A7C15ULL), mix64((uint64_t)g * 2 + 1));",
     "rng.seed(mix64(0x9E3779B97F4A7C15ULL), mix64((uint64_t)g * 2 + 1));", [3]),
    # `bald` culling by ROOT only is the tempting simplification, and it is exactly the bug
    # the parameter was added to fix: the hairs that show on an eyeball are the ones rooted
    # on the skin AROUND it that then grow across it.
    ("bald tests only the root, not the whole strand",
     "        for (int k = 0; k + 1 < n; ++k) {\n"
     "            const Vec3 a = cp[k], ab = cp[k + 1] - a, ac = z.center - a;",
     "        for (int k = 0; k + 1 < n && false; ++k) {\n"
     "            const Vec3 a = cp[k], ab = cp[k + 1] - a, ac = z.center - a;", [8]),
    # A cull must remove the strand's SEGMENTS, not merely flag it: leaving them tessellated
    # keeps the hair in the scene while the count says it went. §8 catches it as segments
    # sitting inside the zone.
    ("bald flags the strand but still tessellates it",
     "            nseg[i] = 0; baldFlag[i] = 1; return;",
     "            baldFlag[i] = 1;", [8]),
]

# Optional filter: `python mutate_fur.py 6 9` runs only those 1-based mutations, which is
# what you want when re-verifying a single hole rather than paying ~4 min x N for the lot.
if len(sys.argv) > 1:
    pick = {int(a) for a in sys.argv[1:]}
    MUTS = [m for i, m in enumerate(MUTS, 1) if i in pick]

orig = io.open(FUR, encoding="utf-8").read()
shutil.copyfile(FUR, FUR + ".bak")
results = []
try:
    for name, old, new, expect in MUTS:
        if orig.count(old) != 1:
            results.append((name, expect, None, "PATCH SITE NOT UNIQUE (%d)" % orig.count(old)))
            continue
        io.open(FUR, "w", encoding="utf-8").write(orig.replace(old, new))
        b = subprocess.run(["cmd", "/c", os.path.join(ROOT, "build.bat")], cwd=ROOT,
                           capture_output=True, text=True, errors="replace")
        if "ftrace.exe updated" not in b.stdout:
            results.append((name, expect, None, "BUILD FAILED"))
            continue
        r = subprocess.run([os.path.join(ROOT, "ftrace.exe"), "-checkfur"], cwd=ROOT,
                           capture_output=True, text=True, errors="replace")
        failed = sorted(int(m) for m in re.findall(r"\[checkfur\] (\d+)\..*-> FAIL", r.stdout))
        ok = all(e in failed for e in expect)
        results.append((name, expect, failed, "caught" if ok else "MISSED"))
        print("  %-58s expect %s got %s  %s" % (name[:58], expect, failed, results[-1][3]),
              flush=True)
finally:
    io.open(FUR, "w", encoding="utf-8").write(orig)
    os.remove(FUR + ".bak")
    subprocess.run(["cmd", "/c", os.path.join(ROOT, "build.bat")], cwd=ROOT, capture_output=True, text=True)

print("\n=== summary ===")
bad = 0
for name, expect, failed, verdict in results:
    if verdict != "caught":
        bad += 1
    print("%-8s %-58s expect %s got %s" % (verdict, name[:58], expect, failed))
print("\n%d/%d mutations caught" % (len(results) - bad, len(results)))
sys.exit(1 if bad else 0)
