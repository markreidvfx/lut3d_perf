@echo off
setlocal

where /q cl || (
  echo ERROR: "cl" not found - please run this from the MSVC x64 native tools command prompt.
  exit /b 1
)

if "%Platform%" neq "x64" (
    echo ERROR: Platform is not "x64" - please run this from the MSVC x64 native tools command prompt.
    exit /b 1
)

set ROOT=%~dp0
set BUILD_DIR="%ROOT%\build"
set SRC_DIR="%ROOT%\src"
set RESULT_DIR="%ROOT%\results_msvc"

set SAMPLES_DIR="%ROOT%\samples"

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%RESULT_DIR%" mkdir "%RESULT_DIR%"

pushd %BUILD_DIR%

@REM set BASE_FILES=lut3d.c tinyexr.obj
set CFLAGS=/nologo /W3 /Z7 /GS- /Gs999999 -D_CRT_SECURE_NO_WARNINGS -fp:fast -fp:except-
set LDFLAGS=/incremental:no /opt:icf /opt:ref

set ASM_CFLAGS=-DHAVE_ALIGNED_STACK=1 -DARCH_X86_64=1 -DHAVE_CPUNOP=1 -DHAVE_AVX2_EXTERNAL=1 -DHAVE_AVX_EXTERNAL=1
set ASM_CFLAGS=--parser=nasm --oformat=win64  %ASM_CFLAGS%

%ROOT%\tools\yasm.exe -g cv8 %ASM_CFLAGS% -o lut3d_asm_debug.obj %SRC_DIR%\lut3d.asm || goto :error
%ROOT%\tools\yasm.exe %ASM_CFLAGS% -o lut3d_asm_release.obj   %SRC_DIR%\lut3d.asm || goto :error

call cl -D_DEBUG -Od /Z7 /MDd /c %SRC_DIR%\tetrahedral_ocio.cpp /Fotetrahedral_ocio_debug.obj %CFLAGS%  || goto :error
call cl -O2 /c %SRC_DIR%\tetrahedral_ocio.cpp /Fotetrahedral_ocio_release.obj -I%SRC_DIR%\deps\miniz %CFLAGS% || goto :error

call cl -D_DEBUG -Od /Z7 /MDd /c %SRC_DIR%\deps/tinyexr.cpp -I%SRC_DIR%\deps\miniz /Fotinyexr_debug.obj %CFLAGS% || goto :error
call cl -O2 /c %SRC_DIR%\deps/tinyexr.cpp -I%SRC_DIR%\deps\miniz %CFLAGS% || goto :error

call cl -D_DEBUG -Od /Z7 /MDd  -Felut3d_pref_debug_msvc.exe %CFLAGS%  %SRC_DIR%\lut3d_perf.c  /link %LDFLAGS% tinyexr_debug.obj lut3d_asm_debug.obj tetrahedral_ocio_debug.obj /subsystem:console || goto :error
call cl -O2 -Fe..\lut3d_pref_release_msvc.exe %CFLAGS% %SRC_DIR%\lut3d_perf.c /link %LDFLAGS% tinyexr.obj lut3d_asm_release.obj tetrahedral_ocio_release.obj /subsystem:console || goto :error

set IMAGE=2A5A2701.0001.exr
set LUT=ACES2065-1_to_Rec709.csp
@REM set IMAGE=2A5A2701_SLog3.0001.exr
@REM set LUT=SLog3_to_ACESRec709.cube

pushd %RESULT_DIR%
call ..\lut3d_pref_release_msvc.exe %SAMPLES_DIR%\images\%IMAGE% %SAMPLES_DIR%\luts\%LUT%  || goto :error
popd
popd

goto :EOF

:error
popd
popd
echo Failed with error #%errorlevel%.
exit /b %errorlevel%W