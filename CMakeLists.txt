cmake_minimum_required(VERSION 3.20)
project(forward_raytracer LANGUAGES CXX C)   # C for the vendored ufbx.c (FBX loader)

# --- Version -----------------------------------------------------------------
# The repo-root VERSION file is the single source of truth (release.bat tags
# v<VERSION>). Bake it into the binary too, so a built ftrace.exe can say which
# version it is (`ftrace -version`) instead of being anonymous. CONFIGURE_DEPENDS
# semantics: touching VERSION re-runs CMake, so the define never goes stale.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" FTRACE_VERSION_RAW)
string(STRIP "${FTRACE_VERSION_RAW}" FTRACE_VERSION)
message(STATUS "ftrace version: ${FTRACE_VERSION}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Put binaries in build/bin regardless of generator/config
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg} "${CMAKE_BINARY_DIR}/bin")
endforeach()

if(MSVC)
    # C++ only: passing these MSVC flags to nvcc directly makes it mis-parse the
    # command line, so scope them to the CXX compile with a generator expression.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>
                        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
                        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
                        $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
    # /bigobj: a COFF object file may hold only 65,279 sections, and main.cpp had reached
    # that ceiling by 0.215.0 -- the mode-J template instantiation of gatherPhotonBeamsW was
    # merely the straw that broke it (`error C1128: number of sections exceeded object file
    # format limit`). Every COMDAT -- each inline function, each template instantiation, each
    # string literal -- costs a section, and main.cpp transitively includes essentially the
    # whole renderer, so the count only ever goes up. /bigobj raises the limit to 2^32 and
    # costs nothing but a slightly larger .obj; it changes no generated code. Applied to all
    # C++ TUs rather than just main.cpp because viewer_gui.cpp pulls in the same headers and
    # is the next one that would hit the wall.
    # Fast math is NOT enabled: physical correctness matters more than speed for now.
endif()

add_executable(ftrace
    src/main.cpp
    src/stb_image_impl.cpp   # compiles vendored stb_image (texture load) + stb_image_write (PNG/JPG output) once
    src/livewindow.cpp       # -window live-preview OS window (Win32 GDI; no-op stub off Windows)
    src/vdbgrid.cpp          # confines vendored NanoVDB.h to one TU; bakes .nvdb -> dense fog grid
    src/vdb_openvdb.cpp      # native OpenVDB .vdb reader (no OpenVDB/NanoVDB dep); bakes .vdb -> dense fog grid
    src/third_party/lz4.c    # vendored single-file LZ4 (BSD) for blosc LZ4 decode in the .vdb reader
    src/fbx_load.cpp         # FBX mesh loader (bridge over vendored ufbx); confines ufbx.h to one TU
    src/third_party/ufbx.c   # vendored single-file FBX parser (MIT / public-domain), compiled as C
    src/gpda/tokenized.cpp        # vendored GPDA parser engine (J3c grammar-validation shim)
    src/gpda/ftsl_scene.gen.cpp   # GENERATED (loom.grammar.emit_cpp): shared .ftsl grammar graph + lexer table
    src/viewer_gui.cpp            # -viewer: loom native viewer host (Dear ImGui + Win32/D3D11)
    # --- vendored Dear ImGui core + Win32/D3D11 backends (the -viewer GUI) ---
    src/third_party/imgui/imgui.cpp
    src/third_party/imgui/imgui_draw.cpp
    src/third_party/imgui/imgui_tables.cpp
    src/third_party/imgui/imgui_widgets.cpp
    src/third_party/imgui/backends/imgui_impl_win32.cpp
    src/third_party/imgui/backends/imgui_impl_dx11.cpp
    # --- vendored ImPlot (F3 strip charts), built on ImGui ---
    src/third_party/implot/implot.cpp
    src/third_party/implot/implot_items.cpp
    # --- vendored imnodes (F5 modulator-DAG panel), built on ImGui ---
    src/third_party/imnodes/imnodes.cpp
)
target_include_directories(ftrace PRIVATE src src/gpda src/third_party/imgui src/third_party/implot src/third_party/imnodes)
# The version string lives in ONE tiny translation unit. It used to be a target-wide
# definition, which made every VERSION bump -- and this project bumps on every observable
# change -- dirty all twenty-odd TUs including the 18k-line render_cuda.cu: a full rebuild
# for a one-line change. Now a bump recompiles version.cpp and relinks.
target_sources(ftrace PRIVATE src/version.cpp)
set_source_files_properties(src/version.cpp PROPERTIES
    COMPILE_DEFINITIONS "FTRACE_VERSION=\"${FTRACE_VERSION}\"")
# Host TUs compile in parallel: MSBuild's CL task runs one cl.exe with /MP over the dirty
# items. Matters on a full rebuild, where the 24k-line main.cpp used to queue behind the
# dozen vendored ImGui/ImPlot/imnodes files on a single core.
if(MSVC)
    target_compile_options(ftrace PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP>)
endif()

# The vendored GPDA engine + generated grammar are third-party/generated code;
# don't hold them to ftrace's /W4 /permissive- bar (keeps the build-warning
# signal about our own code).
if(MSVC)
    set_source_files_properties(
        src/gpda/tokenized.cpp src/gpda/ftsl_scene.gen.cpp src/third_party/lz4.c
        src/third_party/imgui/imgui.cpp src/third_party/imgui/imgui_draw.cpp
        src/third_party/imgui/imgui_tables.cpp src/third_party/imgui/imgui_widgets.cpp
        src/third_party/imgui/backends/imgui_impl_win32.cpp
        src/third_party/imgui/backends/imgui_impl_dx11.cpp
        src/third_party/implot/implot.cpp src/third_party/implot/implot_items.cpp
        src/third_party/imnodes/imnodes.cpp
        PROPERTIES COMPILE_OPTIONS "/W0")
endif()

# The -viewer GUI links Direct3D 11 (+ DXGI). These import libs ship with the
# Windows SDK, so no extra dependency — just name them on Windows.
if(WIN32)
    target_link_libraries(ftrace PRIVATE d3d11 dxgi d3dcompiler dwmapi)
endif()

# --- Optional CUDA backend (GPU forward light tracer, model B) ---------------
# If a CUDA toolkit is present, compile src/render_cuda.cu (by nvcc) into ftrace
# and define HAVE_CUDA so main.cpp exposes the -device gpu path. Without CUDA the
# renderer builds and runs exactly as before (CPU-only), so CUDA is not required.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Which GPU architectures to build device code for. This is what makes the
    # binary portable across NVIDIA GPUs rather than baked to one card:
    #   native     (default) — build only for the GPU present in THIS machine.
    #                Fast local builds; ideal when you run where you build. Requires
    #                a visible CUDA GPU at configure time.
    #   all-major  — a redistributable fat binary: one cubin per major arch PLUS
    #                forward-compatible PTX, so even GPUs newer than this toolkit
    #                knows about JIT-compile at load. Use this to ship to others.
    #   all        — every arch the toolkit supports (largest/slowest build).
    #   "75;86;89" — an explicit semicolon list of compute capabilities.
    # Override at configure time, e.g.  -DFTRACE_CUDA_ARCH=all-major
    set(FTRACE_CUDA_ARCH "native" CACHE STRING
        "CUDA architectures to build (native | all-major | all | semicolon list)")

    # Device transport precision. Consumer GeForce GPUs run FP64 at ~1/64 the FP32
    # rate, so by default the megakernel computes geometry/BRDF/spectral math in
    # float and accumulates the film/energy in double (mixed precision). Set OFF for
    # a full-FP64 device path (bit-closer to the CPU reference, far slower on GeForce;
    # useful on datacenter cards or for precision debugging). The CPU renderer is
    # always double and remains the ground-truth regardless of this switch.
    option(FTRACE_GPU_FP32 "Compute GPU transport in float (mixed precision)" ON)

    target_sources(ftrace PRIVATE src/render_cuda.cu src/raster_cuda.cu)

    # Header dependencies for the .cu TUs — DO NOT REMOVE.
    #
    # MSBuild's CudaCompile task does not scan #includes, so a header edit leaves the .cu
    # objects stale AND (because the .cu TUs share headers with the C++ ones) can leave the
    # linker holding a stale COMDAT of an inline function you just changed. OBJECT_DEPENDS
    # does not help — the VS generator emits CudaCompile items with no metadata. Instead a
    # pre-build step bumps the mtime of any .cu older than the headers, which is the one
    # signal CudaCompile does honour. Full rationale in cmake/touch_stale_cu.cmake.
    add_custom_command(TARGET ftrace PRE_BUILD
        COMMAND "${CMAKE_COMMAND}" "-DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}/src"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/touch_stale_cu.cmake"
        COMMENT "Checking .cu header dependencies (MSBuild CudaCompile does not)")

    target_compile_definitions(ftrace PRIVATE HAVE_CUDA=1)
    # Emit source line-number info in device code (nvcc -lineinfo). This is
    # essentially free at runtime (unlike -G, which disables optimisation) and
    # lets compute-sanitizer / cuda-gdb map GPU faults back to file:line — the
    # difference between "an OOB happened somewhere" and "OOB at render_cuda.cu:1234".
    target_compile_options(ftrace PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    # nvcc's optimiser (cicc) and assembler (ptxas) are single-threaded per translation unit,
    # and render_cuda.cu is one 18k-line TU. --split-compile partitions its device code so
    # those phases use every core. Measured 2026-09-06 (i7-8700K, ~3 cores of other load):
    # ptxas 52.6 s -> 21.1 s, cicc 64.3 s -> 43.2 s, the whole TU 2:49 -> 1:50. NVIDIA
    # documents "minimal (if any)" effect on the generated code; checked here by an A/B of
    # render speed on the same scenes (design.md, Build & release).
    # Constraint this adds: every __device__ function must have INTERNAL linkage (static).
    # A plain `inline` template in a header (dPatternEval, before 0.259.2) can be partitioned
    # away from a __launch_bounds__-capped kernel, and nvlink then refuses the link when the
    # callee's own register allocation exceeds the cap (169 vs kBdptT's 168). A global
    # -maxrregcount is NOT the answer: 35 kernels legitimately use up to 255 registers.
    target_compile_options(ftrace PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--split-compile=0>)
    # CCCL 3.x (the thrust/cub shipped with CUDA 13) requires MSVC's standard-conforming
    # preprocessor; without it the thrust headers #error out. Scoped to the CUDA host
    # pass-through so plain C++ TUs keep their default flags.
    if(MSVC)
        target_compile_options(ftrace PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    endif()
    if(FTRACE_GPU_FP32)
        target_compile_definitions(ftrace PRIVATE FTRACE_GPU_FP32=1)
    else()
        target_compile_definitions(ftrace PRIVATE FTRACE_GPU_FP32=0)
    endif()
    set_target_properties(ftrace PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${FTRACE_CUDA_ARCH}")
    # cudart_static is built against the static CRT (LIBCMT) while our C++ uses the
    # dynamic CRT (/MD). Drop the static CRT from the default lib set so the linker
    # resolves the CRT once (dynamic) — clears the benign LNK4098 conflict warning.
    if(MSVC)
        target_link_options(ftrace PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()
    # A host compiler newer than the CUDA toolkit supports is NOT a benign warning here.
    # Measured 2026-08-16 with CUDA 13.3 + MSVC 14.51 (the VS 2026 BuildTools preview):
    # nvcc accepted it, every kernel launched and ran without a single CUDA error, and
    # every GPU render came out BLACK (mode B: absorbed=1.0000, escaped=0.0000, "0.0%
    # noise"). The same source built with MSVC 14.44 is correct at both sm_75 and sm_89,
    # so it is the host pass, not the device arch. Nothing at runtime can catch that —
    # so shout at configure time. (build.bat pins the VS 2022 generator for this reason.)
    if(MSVC AND MSVC_VERSION GREATER_EQUAL 1950)
        message(WARNING
            "MSVC ${MSVC_VERSION} is newer than this CUDA toolkit "
            "(${CMAKE_CUDA_COMPILER_VERSION}) officially supports. That combination has "
            "been observed to MISCOMPILE the device code silently — GPU renders come out "
            "black with no CUDA error. Configure with -G \"Visual Studio 17 2022\" (what "
            "build.bat does) or another supported host compiler.")
    endif()
    message(STATUS "CUDA backend enabled (FTRACE_CUDA_ARCH=${FTRACE_CUDA_ARCH}; "
                   "use -DFTRACE_CUDA_ARCH=all-major for a portable fat binary; "
                   "FTRACE_GPU_FP32=${FTRACE_GPU_FP32})")
else()
    message(STATUS "CUDA not found; building CPU-only (-device gpu unavailable)")
endif()
