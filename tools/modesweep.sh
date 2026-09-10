#!/usr/bin/env bash
# Cross-mode regression sweep.
#
# v0.270.0 -> v0.270.8 shipped eight versions in one session, touching backward.h, lighttree.h,
# scene.h, photonmap_render.h, sppm_render.h, render_cuda.cu, texture.h, gltf.h, ftsl.h,
# main.cpp and meshvoxel.h -- and every one was validated only against the ROIs it targeted.
# Nothing has checked that the other transport modes still run at all.
#
# Deliberately cheap and broad rather than precise: every mode, both devices, one small scene,
# a handful of samples. It is looking for crashes, hangs, all-black frames and NaNs -- the
# failures that a targeted A/B cannot see because it never renders that mode.
#
# `-window-min` is omitted: this is a batch of ~24 short renders and the live viewer blocks
# (see RASTER-GPU-DIFF's repro note), which would deadlock the sweep.
#
# TWO THINGS THAT LOOK LIKE FAILURES AND ARE NOT (both cost an investigation on 2026-09-10):
#
#   * mode V writes `<out>_forward.pfm` and `<out>_backward.pfm`, never `<out>.pfm` -- it renders
#     both transports and compares them. "NO OUTPUT" for V means the sweep looked for the wrong
#     name, not that V failed.
#   * NEGATIVE channel values are EXPECTED in scene-linear output from a spectral renderer. A
#     saturated firefly converts out of the sRGB gamut, pushing one or two channels below zero
#     while the pixel sum stays positive: e.g. (-3.6e13, +5.4e13, -2.7e12). The check that
#     matters is ALL-NEGATIVE pixels, which would be a transport bug; there were none (0 of
#     12288). Mode W shows no negatives at all because its fixed quadrature produces no
#     fireflies to push out of gamut.
cd "D:/visual studio projects/forward raytracer" || exit 1
SC=${SC:-scenes/cornell.ftsl}
OUT=png/modesweep
mkdir -p $OUT
echo "sweep of $SC   binary $(./ftrace.exe -version)   start $(date +%T)"
for m in A B C D J M P R S U V W; do
  for d in cpu gpu; do
    tag="${m}_${d}"
    t0=$(date +%s%3N)
    timeout 300 ./ftrace.exe -in $SC -mode $m -device $d -r 64 64 -spp 8 -n 200000 \
        -seed 1 -hdr -o $OUT/$tag.png >$OUT/$tag.log 2>&1
    rc=$?
    t1=$(date +%s%3N)
    printf '%-8s rc=%-3s %5dms  ' "$tag" "$rc" $((t1-t0))
    if [ -f "$OUT/$tag.pfm" ]; then python tools/pfmstat.py "$OUT/$tag.pfm"
    elif [ -f "$OUT/${tag}_forward.pfm" ]; then python tools/pfmstat.py "$OUT/${tag}_forward.pfm"   # mode V
    else echo "NO OUTPUT  $(grep -iE 'error|unsupported|not supported' $OUT/$tag.log | head -1 | cut -c1-60)"; fi
  done
done
echo "##### DONE $(date +%T)"
