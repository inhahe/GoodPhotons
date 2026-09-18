@echo off
REM Build + run the minijson parse benchmark. See jsonbench.cpp for why it exists.
REM Intermediates go to scraps/ (gitignored); the exe sits next to this script.
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0.."
cl /nologo /std:c++17 /O2 /EHsc tools\jsonbench.cpp /Fe:tools\jsonbench.exe /Fo:scraps\ >nul
if errorlevel 1 exit /b 1
tools\jsonbench.exe %*
