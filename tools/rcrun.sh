#!/bin/sh
# rcrun.sh — serial A/B measurement driver for `-radcache` (and any other accuracy
# comparison that wants an HDR pair plus ray counts).
#
#   tools/rcrun.sh <outbase> <scene> [extra ftrace flags...]
#
# Renders <scene> to <outbase>.png / <outbase>.pfm with a MINIMISED live preview
# (per CLAUDE.md: always show the window, but don't steal the desktop), waits for
# the render to finish, then releases the held window with the clean
# `ftrace -stop <pid>` — never a force-kill, which can wedge the NVIDIA driver.
# Finally it echoes the `[radcache]` / `[raystats]` / `[spp]` lines, which are the
# numbers you actually wanted.
#
# Defaults are the measurement configuration used in known-issues.md: mode R on the
# CPU (the cache exists nowhere else), 200x200, 1024 spp. Override any of them with
# environment variables rather than by appending a second conflicting flag:
#
#   RES=400  SPP=4096  tools/rcrun.sh png/rc_on scenes/cornell.ftsl -radcache
#   BUDGET="-time 120" tools/rcrun.sh png/rc_eq scenes/cornell.ftsl -radcache
#
# BUDGET replaces `-spp $SPP` wholesale, so an equal-TIME run is `BUDGET="-time N"`.
# Ray counts, not wall clock, are the meaningful performance measure on this machine:
# three repeats of one identical render came back 18.5 s / 25.8 s / 27.2 s, so compare
# `[raystats]`.
#
# Why the poll loop instead of a plain `wait`: -keepwindow deliberately keeps the
# process alive holding the finished image on screen, so `wait` alone would block
# forever. The loop waits for the "render done" line, then stops that pid.

set -u
if [ $# -lt 2 ]; then
    sed -n '2,28p' "$0"
    exit 2
fi
out=$1; shift
scene=$1; shift

RES=${RES:-200}
SPP=${SPP:-1024}
BUDGET=${BUDGET:--spp $SPP}
FT=${FT:-./ftrace.exe}
TIMEOUT=${TIMEOUT:-1800}          # seconds to wait for "render done"

log="$out.log"
mkdir -p "$(dirname "$out")" 2>/dev/null

# shellcheck disable=SC2086
"$FT" "$scene" -mode R -device cpu -r "$RES" $BUDGET -hdr -o "$out.png" \
    -window-min -keepwindow -raystats "$@" > "$log" 2>&1 &
pid=$!

i=0
while [ "$i" -lt $((TIMEOUT / 2)) ]; do
    grep -q "render done" "$log" 2>/dev/null && break
    kill -0 "$pid" 2>/dev/null || break
    i=$((i + 1))
    sleep 2
done

# ftrace prints the exact release command when it starts holding the window:
#   [window] render done - close the preview window to exit (or run: ftrace -stop <pid>)
rp=$(sed -n 's/.*ftrace -stop \([0-9]*\).*/\1/p' "$log" | tail -1)
[ -n "$rp" ] && "$FT" -stop "$rp" > /dev/null 2>&1
wait "$pid" 2>/dev/null

grep -E "^\[radcache\]|^\[raystats\]" "$log"
grep -E "^\[spp\]" "$log" | tail -1
