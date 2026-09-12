#include "version.h"
// FTRACE_VERSION is a compile definition on this file alone (CMakeLists.txt). The fallback
// only fires for a hand-rolled compile outside the CMake build.
#ifndef FTRACE_VERSION
#define FTRACE_VERSION "unknown"
#endif
const char* ftraceVersion() { return FTRACE_VERSION; }
