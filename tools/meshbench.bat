@echo off
REM Build tools/meshbench.cpp -- splits loadObj into text-parse vs crease-smoothing.
REM /std:c++20 is REQUIRED, not cosmetic: src/scene.h uses parenthesized aggregate
REM initialization (C++20 P0960) on Pcg32, which is an aggregate with default member
REM initializers. Under /std:c++17 that is "no overloaded function takes 2 arguments".
REM The main build (CMakeLists) is C++20, so this just matches it.
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0.."
if not exist scraps\meshbench mkdir scraps\meshbench
cl /nologo /std:c++20 /O2 /EHsc /I src tools\meshbench.cpp ^
   /Fe:tools\meshbench.exe /Fo:scraps\meshbench\
exit /b %errorlevel%
