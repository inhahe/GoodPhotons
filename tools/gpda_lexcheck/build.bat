@echo off
REM Builds tools\gpda_lexcheck\lexcheck.exe — the differential validator for
REM src\gpda\gpda_lexer.hpp's first-byte prefilter and literal fast path.
REM See lexcheck.cpp for what it proves and why it has to exist.
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 1>NUL
if not exist obj mkdir obj
cl /nologo /std:c++20 /EHsc /O2 /W3 /I"..\..\src\gpda" /Fe:lexcheck.exe /Fo:obj\ ^
   lexcheck.cpp "..\..\src\gpda\ftsl_scene.gen.cpp" "..\..\src\gpda\tokenized.cpp" ^
   || exit /b 1
echo built lexcheck.exe
