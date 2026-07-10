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
    add_compile_options(/W4 /permissive- /Zc:preprocessor)
    # Fast math is NOT enabled: physical correctness matters more than speed for now.
endif()

add_executable(ftrace
    src/main.cpp
)
target_include_directories(ftrace PRIVATE src)
