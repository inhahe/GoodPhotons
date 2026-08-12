# Forward Raytracer — project instructions

## File / directory conventions

Keep the repo root clean. All generated and throwaway files live in dedicated
directories (already created, and git-ignored):

- **`ppm/`** — all `.ppm` render outputs and test images. Never write `.ppm` files to
  the repo root; pass `-o ppm/<name>.ppm`.
- **`png/`** — all `.png` render outputs and test images. Pass `-o png/<name>.png`.
  - **Flyby series get their own subdirectory.** Any multi-frame moving-camera
    sequence (a `camera_curve` / `camera_path` / orbit flyby — files like
    `<name>_fly000.png`, `<name>_swoop000.png`, `<name>_dolly0.png`, `<name>_orb0.png`)
    must render into its own dedicated subdir `png/<setname>/`, not loose in `png/`.
    Keep each set's companions (its `_cam` still, `_meter` frame, `.log`, palette,
    `.ftbuf` checkpoints, assembled `.gif`/`.mp4`) in that same subdir so each series
    is self-contained. Never let two different flyby series share a directory, and
    never mix a flyby series in with unrelated one-off images. Point `-o` at the
    subdir, e.g. `-o png/myflyby/myflyby.png`.
- **`scraps/`** — all temporary / throwaway Python scripts (comparison helpers,
  one-off analysis, etc.) and any other scratch files. Don't leave temp `.py` files
  in the repo root or in `tools/` (which is for permanent, checked-in tooling).

When creating validation renders or scratch scripts, always target these directories
so the working tree stays tidy and nothing stray shows up in `git status`.

## Running renders — NEVER use the default (sandboxed) Bash tool

**Do NOT launch renders through the ordinary sandboxed Bash tool.** The Bash tool
sandboxes commands, which on Windows runs them in a **non-interactive window
station**. ftrace's `-window` GDI call then has no desktop to attach to, so the
live-preview window is created invisibly / no-op'd — the render still runs and writes
PNGs, but **no watchable window ever reaches the user's screen**, silently violating
the "always show the window" rule below. (Symptom: the startup line reads
`… (Ctrl-C to stop early) …` without the `— live window` suffix, and the process shows
in a non-`Console` session in `tasklist`.)

To actually get a visible window, launch every render with the Bash tool's
**`dangerouslyDisableSandbox: true`** — that runs it in the interactive **Console
session**, where the GDI window appears on the real desktop (verify with
`Get-Process ftrace | Select MainWindowTitle` → `ftrace 🪟 live preview`). So whenever
you want the live preview to actually be visible (the default — see below), launch with
`dangerouslyDisableSandbox: true` alongside `-window`.

## Running renders — launch with the live preview so the user can watch

**The one rule that matters: whenever you render, put the live preview window up
(`-window`) so the user can see it.** That's the whole point — the user wants to be
able to glance over and watch any image you generate converge, without having to ask.

What this rule is **NOT**: it is *not* a reason to avoid rendering, to hold back work,
or to wait for the user. A preview window harmlessly sitting on the desktop (even
while the user is away or asleep) is exactly what's wanted, not a problem. Rendering
to validate your own changes, and then inspecting the output PNGs yourself, is always
fine and encouraged — the live window doesn't block any of that. Don't invent
"I can't render right now" blockers out of this rule.

It's also **not absolute**: if you have a genuine reason to skip the live preview for a
particular render, that's fine — just prefer showing it by default.

So: default to launching every render with the live visual display (`-window`), which
gives both you and the user a real, watchable view of the image converging and confirms
the render will actually finish. On top of `-window`, add periodic crash-safe output so
progress survives a crash. Concretely, when you start a render:

- **ALWAYS pass `-window`.** This opens the real OS live-preview (Win32 GDI)
  showing the actual tone-mapped image refreshed as it converges — the best way to
  watch. It also auto-chunks a plain fixed-`-n` render so periodic writes/status
  happen (see the gotcha below). This is mandatory on every render invocation.
- **ALWAYS pass `-keepwindow` (instead of plain `-window`) so the window does NOT
  auto-close when the render finishes.** By default ftrace tears the live window
  down at process exit, so a finished image only flashes on screen and vanishes —
  the user never gets to look at the result. `-keepwindow` (alias `-hold`, implies
  `-window`) keeps the final image up and blocks until the user closes the window
  themselves. Use it on every render so the completed image stays watchable. (When
  you launch a render in the background to poll its progress, remember the process
  will now stay alive holding the window after the render completes — stop it
  explicitly with `ftrace -stop <pid>` once you and the user are done inspecting;
  see the section below, and never `taskkill /F` it.)
- **Always add periodic crash-safe output.** Use `-interval <sec>` (default 15) for
  the write/refresh cadence, and `-checkpoint` (forward modes A/B/C) so a resumable
  `.ftbuf` sidecar is written next to `-o` and progress survives a crash/Ctrl-C
  (`-resume` continues it). Without `-checkpoint` a long render that dies loses
  everything.
- **Only if a window genuinely can't open** (truly headless session): fall back to
  `-preview` for a live ANSI thumbnail in the terminal, or at least the periodic
  status line (`[live] … photons, ~N% noise`). On this machine there IS a desktop,
  so this is a rare exception — default to `-window`.
- **Prefer a bounded budget over a giant `-n`.** `-time <sec>`, `-noise <pct>`
  (stop at a target graininess), or `-forever` (Ctrl-C to stop) all render
  *progressively* with periodic writes, so you get a usable image early instead of
  waiting for one huge batch. Pick a photon count that's plausibly completable and
  sanity-check the rate from the first status tick rather than guessing.
- **Run it in the background and check the periodic output**, so you can report
  progress and catch a stall or a runaway count early.

**Critical gotcha (the reason for this rule):** a plain fixed-`-n` render with
*none* of `-window` / `-time` / `-noise` / `-forever` takes the **non-chunked path**
(`chunkFixed = !progressive && g_showWindow`, main.cpp ~2125). It runs the *entire*
photon count as one monolithic batch and writes the image **only at the very end** —
no partial image, no checkpoint, no status line, no way to see progress or estimate
completion. So a bare `-n 4000000000` can run for hours producing nothing and
losing it all on a crash. Adding `-window` (or any budget flag) is what enables the
periodic-output loop.

## Stopping a render — `ftrace -stop <pid>`, NEVER `taskkill /F`

**Force-killing ftrace while CUDA kernels are in flight is a known way to wedge the
NVIDIA driver into a TDR/bugcheck.** So no ftrace process in this repo may ever be
stopped with `taskkill /F` (or `Stop-Process -Force`, or `kill -9`) — not the render
you started, not a stale `-keepwindow` preview holding the exe open and blocking
`build.bat` from copying, not one you didn't start. There is a purpose-built clean
stop, and it is always the answer:

- **`ftrace -stop <pid>`** — asks that render to do exactly what Ctrl-C does: finish
  the current chunk, write the final image **and** the `.ftbuf` checkpoint, release
  the CUDA context through the graceful-shutdown path, then exit. It waits (up to
  120 s) for the process to actually be gone, so you can chain a rebuild straight
  after it. It also releases a window being held open by `-keepwindow`.
- **`ftrace -stop`** (bare) — *lists* every live ftrace process: pid + scene → output.
  Use this to find the pid rather than grepping `tasklist`.
- **`ftrace -stop all`** — stops every live ftrace process.

**It covers the GUIs too, not just renders (since 0.182.0).** A `-viewer` (loom native
viewer) or `-explore` process is listed by a bare `-stop` and shuts down cleanly when
targeted — the GUI loop polls the same flag a render polls, then leaves through its
normal teardown. So a viewer holding `ftrace.exe` open and failing your rebuild is a
`-stop`, never a force-kill.

**Trust the exit code (since 0.182.0).** `-stop` exits **0** only when every target is
genuinely gone (a pid that was already dead counts, and says `nothing to stop`). If a
target is still alive when the 120 s window expires it prints `[stop] FAILED — still
running after 120s: <pids>` and exits **2**. Before 0.182.0 it printed `stopped cleanly`
and exited 0 unconditionally, so a failed stop looked like a successful one — if you see
the FAILED line, investigate; still don't reach for `taskkill /F`.

ftrace prints the exact command you need when it starts holding a window
(`[window] render done — close the preview window to exit (or run: ftrace -stop
<pid>)`), so the pid is normally already on screen in the log you're tailing.

`-stop` also works during **scene load** (since 0.138.2): the loader polls the same flag
between top-level blocks and inside `ft::parallelFor`'s chunk cursor, so a stop aimed at
a process that hasn't reached its render loop aborts the load in well under a second,
prints `[stop] scene load stopped before rendering — nothing was rendered or written.`,
and exits **1**. A non-zero exit there is the expected, correct outcome — no scene was
built, so nothing could be rendered. (Latency is bounded by one chunk / one asset, so a
single very large mesh import still has to finish before the stop lands. Still never a
reason to `taskkill /F`.)

Why this needs saying here and not just in `design.md`: the natural reflex when a
build fails with "the destination is probably IN USE" is to kill the process by name,
and by the time you're reading the build error you are nowhere near the docs that
explain why that's dangerous. `-stop` costs the same one command and is safe.

## Performance

- **Optimize the tight loops / CPU-heavy hot paths as much as possible, until the
  point of diminishing returns.** This is a rendering engine — its inner loops
  (per-photon / per-ray tracing, tessellation, the raster z-buffer + shading passes,
  tone-mapping, spectral evaluation) run billions of times, so shaving work there
  matters. When you touch a hot path, look for the usual wins: hoist redundant work
  out of loops, project/compute once instead of per-thread or per-pixel, use
  incremental stepping instead of recomputing from scratch, keep data contiguous and
  cache-friendly, parallelize across threads/bands, and precompute lookup tables for
  expensive functions. Keep going until further effort buys only marginal speedups —
  then stop (don't sacrifice correctness or readability chasing a fraction of a
  percent). Always verify an optimization is bit-for-bit (or visually) identical to
  the baseline before committing it.

## Docs to keep current

The user-facing docs are split into three files. Know which one a change belongs in:

- **`README.md`** is the **landing page** — demo, highlights, building, quick start, a
  pointer table to the other docs, loom, known issues. Keep it short enough to read end
  to end (it was a 3365-line wall before the split; it is now ~400 lines). New reference
  material does **not** go here — add a line to the pointer table and put the detail in
  `REFERENCE.md`. Do update it when something in *its* scope changes (build steps, quick
  start flags, headline features, the demo).
- **`REFERENCE.md`** is the **manual** — render modes, cameras, materials, spectra,
  lights, geometry, textures, procedural patterns, participating media, the FTSL scene
  language overview, the full command-line reference, output formats. Update it whenever
  an observable feature changes. Mode descriptions in particular must match the actual
  implementation (e.g. mode A is the finite-lens physical camera, not the old
  contact-sensor wall). Its `## Contents` list is **hand-maintained**: if you add or
  rename a heading, update the TOC in the same edit. The anchors are GitHub slugs —
  lowercase, punctuation stripped, and **each** space replaced by a hyphen, so a run of
  spaces becomes a run of hyphens (`## Mode W — the …` → `#mode-w--the-…`). Use
  `npx github-slugger` if you need to check one.
- **`FTSL.md`** stays authoritative for the scene-language grammar itself;
  `REFERENCE.md` only summarises it and links across.
- **`known-issues.md`** tracks unsolved bugs and tech debt — log anything you can't fix
  immediately, and mark entries DONE when resolved.

## Versioning & releases

- **Bump the version in the repo-root `VERSION` file every time you rebuild `ftrace.exe`
  with an observable change.** `VERSION` is a single `MAJOR.MINOR.PATCH` line and is the
  single source of truth for the release tag. Whenever you run `build.bat` to produce a
  new binary that changes anything observable, increment `VERSION` in the *same commit*
  as the change — **patch** for fixes, **minor** for new features, **major** for
  breaking changes — so every built binary maps to a distinct version. (No observable
  change → no bump.)
- **`release.bat` publishes a GitHub release** of the current repo-root `ftrace.exe` to
  `inhahe/goodphotons`, tagged `v<VERSION>`, using the GitHub CLI (`gh`). It **refuses to
  publish if a release for the current version already exists**, so the workflow is:
  make the change → bump `VERSION` → `build.bat` → `release.bat`. Running it without
  bumping `VERSION` is a safe no-op.
