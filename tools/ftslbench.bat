@echo off
REM Build tools/ftslbench.cpp -- splits a .ftsl load into lex vs graph-walk.
REM Intermediates go to scraps\ so the tree stays clean; the exe is gitignored.
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0.."
cl /nologo /std:c++17 /O2 /EHsc /I src\gpda ^
   tools\ftslbench.cpp src\gpda\ftsl_scene.gen.cpp src\gpda\tokenized.cpp ^
   /Fe:tools\ftslbench.exe /Fo:scraps\ftslbench\
exit /b %errorlevel%
