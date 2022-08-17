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
set BUILD_DIR="%ROOT%\build\msvc"
set SRC_DIR="%ROOT%\src"
set RESULT_DIR="%ROOT%\results\msvc"

IF "%1"=="clean" (
  rmdir /s /q %BUILD_DIR%
  rmdir /s /q %RESULT_DIR%
  goto :EOF
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

pushd %BUILD_DIR%

@REM set BASE_FILES=lut3d.c tinyexr.obj
set CFLAGS=/nologo /W3 /Z7 /GS- /Gs999999 -D_CRT_SECURE_NO_WARNINGS -fp:fast -fp:except-
set LDFLAGS=/incremental:no /opt:icf /opt:ref

set ASM_CFLAGS=-DHAVE_ALIGNED_STACK=1 -DARCH_X86_64=1 -DHAVE_CPUNOP=1 -DHAVE_AVX2_EXTERNAL=1 -DHAVE_AVX_EXTERNAL=1
set ASM_CFLAGS=--parser=nasm --oformat=win64  %ASM_CFLAGS%

%ROOT%\tools\yasm.exe -g cv8 %ASM_CFLAGS% -o lut3d_asm_debug.obj %SRC_DIR%\lut3d.asm || goto :error
%ROOT%\tools\yasm.exe %ASM_CFLAGS% -o lut3d_asm_release.obj   %SRC_DIR%\lut3d.asm || goto :error

cl -D_DEBUG -Od /Z7 /MDd /c %SRC_DIR%\tetrahedral_ocio.cpp /Fotetrahedral_ocio_debug.obj %CFLAGS%  || goto :error
cl -O2 /c %SRC_DIR%\tetrahedral_ocio.cpp /Fotetrahedral_ocio_release.obj -I%SRC_DIR%\deps\miniz %CFLAGS% || goto :error

cl -D_DEBUG -Od /Z7 /MDd /c %SRC_DIR%\deps/tinyexr.cpp -I%SRC_DIR%\deps\miniz /Fotinyexr_debug.obj %CFLAGS% || goto :error
cl -O2 /c %SRC_DIR%\deps/tinyexr.cpp -I%SRC_DIR%\deps\miniz %CFLAGS% || goto :error

cl -D_DEBUG -Od /Z7 /MDd /c %SRC_DIR%\tetrahedral_sse2.c /Fotetrahedral_sse2_debug.obj %CFLAGS%  || goto :error
cl -O2  /c %SRC_DIR%\tetrahedral_sse2.c /Fotetrahedral_sse2_release.obj -I%SRC_DIR%\deps\miniz %CFLAGS% || goto :error

cl -D_DEBUG -Od /Z7 /MDd /c %SRC_DIR%\tetrahedral_avx.c /arch:AVX /Fotetrahedral_avx_debug.obj %CFLAGS%  || goto :error
cl -O2  /c %SRC_DIR%\tetrahedral_avx.c /arch:AVX /Fotetrahedral_avx_release.obj -I%SRC_DIR%\deps\miniz %CFLAGS% || goto :error

cl -D_DEBUG -Od /Z7 /MDd /c %SRC_DIR%\tetrahedral_avx2.c /arch:AVX2 /Fotetrahedral_avx2_debug.obj %CFLAGS%  || goto :error
cl -O2  /c %SRC_DIR%\tetrahedral_avx2.c /arch:AVX2 /Fotetrahedral_avx2_release.obj -I%SRC_DIR%\deps\miniz %CFLAGS% || goto :error

cl -D_DEBUG -Od /Z7 /MDd  -Felut3d_pref_debug_msvc.exe %CFLAGS% %SRC_DIR%\lut3d_perf.c /link %LDFLAGS% tinyexr_debug.obj lut3d_asm_debug.obj tetrahedral_ocio_debug.obj tetrahedral_sse2_debug.obj tetrahedral_avx_debug.obj tetrahedral_avx2_debug.obj /subsystem:console || goto :error
cl -O2 -Fe..\..\lut3d_pref_release_msvc.exe %CFLAGS% %SRC_DIR%\lut3d_perf.c /link %LDFLAGS% tinyexr.obj lut3d_asm_release.obj tetrahedral_ocio_release.obj tetrahedral_sse2_release.obj tetrahedral_avx_release.obj tetrahedral_avx2_release.obj /subsystem:console || goto :error

IF "%1"=="test_rand" (
  if not exist "%RESULT_DIR%\test_rand" mkdir "%RESULT_DIR%\test_rand"
  pushd "%RESULT_DIR%\test_rand"
  ..\..\..\lut3d_pref_release_msvc.exe || goto :error
  popd
)

IF "%1"=="test_lut1" (
  if not exist "%RESULT_DIR%\test_lut1" mkdir "%RESULT_DIR%\test_lut1"
  pushd "%RESULT_DIR%\test_lut1"
  ..\..\..\lut3d_pref_release_msvc.exe ..\..\..\samples\images\2A5A2701.0001.exr ..\..\..\samples\luts\ACES2065-1_to_Rec709.csp || goto :error
  popd
)

IF "%1"=="test_lut2" (
  if not exist "%RESULT_DIR%\test_lut2" mkdir "%RESULT_DIR%\test_lut2"
  pushd "%RESULT_DIR%\test_lut2"
  ..\..\..\lut3d_pref_release_msvc.exe ..\..\..\samples\images\2A5A2701_SLog3.0001.exr ..\..\..\samples\luts\SLog3_to_ACESRec709.cube  || goto :error
  popd
)

popd
goto :EOF

:error
popd
popd
echo Failed with error #%errorlevel%.
exit /b %errorlevel%W