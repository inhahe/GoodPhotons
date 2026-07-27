# touch_stale_cu.cmake — force nvcc to see header edits.
#
# Run as a PRE_BUILD step of the ftrace target (see CMakeLists.txt):
#     cmake -DSRC_DIR=<repo>/src -P cmake/touch_stale_cu.cmake
#
# WHY THIS EXISTS
# ---------------
# MSVC's ClCompile records every #include it opened (/showIncludes -> .tlog) and rebuilds a
# .cpp when any of them changes. MSBuild's CudaCompile task does NOT: it compares only the
# .cu's own timestamp, so editing a header leaves render_cuda.obj / raster_cuda.obj stale
# while the build still reports success.
#
# That is much worse than "the GPU path is a build behind", because the .cu TUs include the
# same headers as the C++ ones (photonmap.h, render.h, scene.h ...). Every inline/template
# function in those headers is emitted as a COMDAT in BOTH objects, the linker keeps exactly
# ONE copy, and it may keep the stale one — so a header-only edit can silently yield a binary
# running the OLD body of the function you just changed, pulled from a .cu you never touched.
# Observed 2026-07-26: PhotonMap::buildAuto kept its pre-edit cell-count guard across four
# rebuilds (one of them after deleting the exe to force a relink) until render_cuda.cu was
# touched by hand.
#
# CMake's OBJECT_DEPENDS is not a fix: the Visual Studio generator emits CudaCompile items
# with no metadata at all, so the AdditionalInputs never reach MSBuild. Bumping the .cu's own
# mtime is the one signal CudaCompile is guaranteed to honour.
#
# The check is deliberately coarse — ANY header newer than a .cu re-touches it (~1.5 min of
# nvcc) rather than risk a mixed binary. It is idempotent: the touch sets the mtime to now,
# so the next build sees the .cu as newer than every header and does nothing.

if(NOT DEFINED SRC_DIR)
    message(FATAL_ERROR "touch_stale_cu.cmake: SRC_DIR must be set (-DSRC_DIR=...)")
endif()

file(GLOB _headers "${SRC_DIR}/*.h" "${SRC_DIR}/*.cuh")
file(GLOB _cus     "${SRC_DIR}/*.cu")

# Fixed-width "YYYYMMDDHHMMSS" compares correctly as a string, so no arithmetic is needed.
set(_newest "0")
foreach(_h IN LISTS _headers)
    file(TIMESTAMP "${_h}" _ts "%Y%m%d%H%M%S" UTC)
    if(_ts STRGREATER _newest)
        set(_newest "${_ts}")
        set(_newest_file "${_h}")
    endif()
endforeach()

foreach(_cu IN LISTS _cus)
    file(TIMESTAMP "${_cu}" _cts "%Y%m%d%H%M%S" UTC)
    if(_newest STRGREATER _cts)
        get_filename_component(_cuName "${_cu}" NAME)
        get_filename_component(_hName  "${_newest_file}" NAME)
        message(STATUS "[cuda-deps] ${_hName} is newer than ${_cuName} — touching it so nvcc rebuilds")
        file(TOUCH_NOCREATE "${_cu}")
    endif()
endforeach()
