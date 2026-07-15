#!/usr/bin/env bash
# Feature-coverage gallery for the forward light tracer.
# Renders one image per feature into gallery/*.png. See gallery/GALLERY.md for the
# expected appearance of each file. Run from the project root:
#     bash gallery/render_gallery.sh
# Every render prints its own energy/exposure line; failures are tagged [GALLERY-FAIL].
set -u
FT=./build/bin/ftrace.exe
OUT=gallery
mkdir -p "$OUT"

run() {  # run <label> -- <ftrace args...>
    local label="$1"; shift; [ "$1" = "--" ] && shift
    echo ""
    echo "==================== $label ===================="
    if "$FT" "$@"; then :; else echo "[GALLERY-FAIL] $label"; fi
}

# ============================================================================
# 1. RENDER / CAMERA MEASUREMENT MODES  (same Cornell-class scene, so the only
#    difference is how the camera measures the identical forward physics)
# ============================================================================
run "mode B (pinhole splat, default)" -- -scene cornell -mode B -r 256 -n 60000000 -o $OUT/01_mode_B_pinhole.png
run "mode A (finite-lens physical cam)" -- -scene cornell -mode A -aperture 0.20 -focus 2.2 -r 256 -n 120000000 -o $OUT/02_mode_A_lens.png
run "mode C (finite-aperture catch)"    -- -scene materials -mode C -aperture 0.25 -focus 2.2 -r 192 -n 300000000 -o $OUT/03_mode_C_catch.png
run "mode R (backward reference)"       -- -scene cornell -mode R -r 256 -spp 512 -o $OUT/04_mode_R_reference.png
run "mode P (specular composite)"       -- -scene materials -mode P -r 256 -n 40000000 -o $OUT/05_mode_P_composite.png
run "mode D (BDPT)"                      -- -scene materials -mode D -r 256 -spp 256 -o $OUT/06_mode_D_bdpt.png
# mode V is a numeric forward-vs-backward validation (prints PASS + residual)
run "mode V (validate, numeric)"        -- -scene cornell -mode V -r 192 -n 40000000 -spp 256

# ============================================================================
# 2. CAMERA VARIABLES & PRESETS
# ============================================================================
run "DOF: shallow (wide aperture)"  -- -scene cornell -mode A -aperture 0.30 -focus 2.2 -r 256 -n 120000000 -o $OUT/10_dof_shallow.png
run "DOF: deep (narrow aperture)"   -- -scene cornell -mode A -aperture 0.05 -focus 2.2 -r 256 -n 120000000 -o $OUT/11_dof_deep.png
run "fisheye projection (CPU)"      -- -in scenes/fisheye.ftsl -r 256 -n 30000000 -o $OUT/12_fisheye.png
run "camera_path dolly (5 frames)"  -- -in scenes/dolly.ftsl -r 192 -n 25000000 -o $OUT/13_dolly.png
run "exposure / ISO+shutter"        -- -in scenes/expo.ftsl -r 192 -n 25000000 -o $OUT/14_expo.png
run "multi-camera (twocam)"         -- -in scenes/twocam.ftsl -r 192 -n 25000000 -o $OUT/15_twocam.png

# ============================================================================
# 3. SURFACES / MATERIALS
# ============================================================================
run "materials in B (specular=black)" -- -scene materials -mode B -r 256 -n 60000000 -o $OUT/20_materials_B.png
run "dispersive glass caustic"        -- -scene cornell -mode B -r 256 -n 120000000 -o $OUT/21_glass_caustic.png
run "prism dispersion (rainbow)"      -- -scene prism -mode B -r 256 -n 120000000 -o $OUT/22_prism.png
run "iridescent thin-film sphere"     -- -scene iridescent -mode P -r 256 -n 40000000 -o $OUT/23_iridescent.png
run "material presets (metals/glass)" -- -in scenes/material_presets.ftsl -mode P -r 192 -n 40000000 -o $OUT/24_material_presets.png
run "multilayer (structural colour)"  -- -in scenes/multilayer.ftsl -mode P -r 192 -n 40000000 -o $OUT/25_multilayer.png
run "mix material"                    -- -in scenes/mixmat.ftsl -mode B -r 192 -n 40000000 -o $OUT/26_mixmat.png
run "fluorescence (CPU, mode B)"      -- -scene fluoro -mode B -r 192 -n 30000000 -o $OUT/27_fluoro.png
run "texture: checker quad (CPU)"     -- -in scenes/textured.ftsl -r 192 -n 20000000 -o $OUT/28_textured.png
run "texture: UV mesh (CPU)"          -- -in scenes/uvmesh.ftsl -r 192 -n 20000000 -o $OUT/29_uvmesh.png

# ============================================================================
# 4. DIFFRACTION
# ============================================================================
run "diffraction grating ON"  -- -scene grating -mode B -diffraction 1 -r 256 -n 120000000 -o $OUT/30_grating_on.png
run "diffraction grating OFF" -- -scene grating -mode B -nodiffraction -r 256 -n 120000000 -o $OUT/31_grating_off.png

# ============================================================================
# 5. LIGHT SHAPES
# ============================================================================
run "area light (quad)"       -- -scene cornell -mode B -light d65 -r 256 -n 60000000 -o $OUT/40_light_area.png
run "sphere light"            -- -in scenes/spherelight.ftsl -r 256 -n 60000000 -o $OUT/41_light_sphere.png
run "spot light (penumbra)"   -- -in scenes/spotlight.ftsl -r 256 -n 60000000 -o $OUT/42_light_spot.png
run "constant environment"    -- -in scenes/envlight.ftsl -mode R -r 256 -spp 512 -o $OUT/43_light_env.png
run "HDRI environment"        -- -in scenes/envmap.ftsl -mode R -r 256 -spp 512 -o $OUT/44_light_envmap.png
run "cylinder light"          -- -in scenes/cylinderlight.ftsl -r 256 -n 60000000 -o $OUT/45_light_cylinder.png
run "two lights"              -- -in scenes/twolight.ftsl -r 256 -n 60000000 -o $OUT/46_light_two.png

# ============================================================================
# 6. LIGHT SPD PRESETS  (same Cornell box, only the illuminant changes -> colour cast)
# ============================================================================
spd() { run "SPD: $1" -- -scene cornell -mode B -light "$1" -r 192 -n 25000000 -o "$OUT/50_spd_$2.png"; }
spd sun           sun
spd daylight      daylight_d65
spd incandescent  incandescent_A
spd led           led_neutral
spd led-warm      led_warm
spd cfl           fluorescent_cfl
spd f2            f2_cool_white
spd f7            f7_daylight_fl
spd f11           f11_triphosphor
spd hps           sodium_hps
spd lps           sodium_lps
spd mercury       mercury
spd metal-halide  metal_halide
spd bb3200        blackbody_3200K
spd bb6500        blackbody_6500K
spd led4000k      led_4000K

# ============================================================================
# 7. FOG / PARTICIPATING MEDIA
# ============================================================================
run "fog isotropic"           -- -scene cornell -mode B -fog 1.0 -fogalbedo 0.8 -r 256 -n 80000000 -o $OUT/60_fog_iso.png
run "fog forward-scattering"  -- -scene cornell -mode B -fog 1.0 -fogg 0.7 -fogalbedo 0.8 -r 256 -n 80000000 -o $OUT/61_fog_forward.png
run "fog back-scattering"     -- -scene cornell -mode B -fog 1.0 -fogg -0.5 -fogalbedo 0.8 -r 256 -n 80000000 -o $OUT/62_fog_back.png
run "fog Rayleigh (spectral)" -- -scene cornell -mode B -fog 1.5 -fograyleigh -fogalbedo 0.9 -r 256 -n 80000000 -o $OUT/63_fog_rayleigh.png

echo ""
echo "==================== GALLERY COMPLETE ===================="
ls -1 $OUT/*.png | sort
