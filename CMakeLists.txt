cmake_minimum_required(VERSION 3.20)
project(forward_raytracer LANGUAGES CXX)

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
                        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
    # Fast math is NOT enabled: physical correctness matters more than speed for now.
endif()

add_executable(ftrace
    src/main.cpp
)
target_include_directories(ftrace PRIVATE src)

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

    target_sources(ftrace PRIVATE src/render_cuda.cu)
    target_compile_definitions(ftrace PRIVATE HAVE_CUDA=1)
    set_target_properties(ftrace PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${FTRACE_CUDA_ARCH}")
    # cudart_static is built against the static CRT (LIBCMT) while our C++ uses the
    # dynamic CRT (/MD). Drop the static CRT from the default lib set so the linker
    # resolves the CRT once (dynamic) — clears the benign LNK4098 conflict warning.
    if(MSVC)
        target_link_options(ftrace PRIVATE /NODEFAULTLIB:LIBCMT)
    endif()
    message(STATUS "CUDA backend enabled (FTRACE_CUDA_ARCH=${FTRACE_CUDA_ARCH}; "
                   "use -DFTRACE_CUDA_ARCH=all-major for a portable fat binary)")
else()
    message(STATUS "CUDA not found; building CPU-only (-device gpu unavailable)")
endif()
