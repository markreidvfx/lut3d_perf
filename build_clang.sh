#!/usr/bin/env bash
# NOTE: you might need to run this with: bash build_clang.sh

set -e
ROOT="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"


BUILD_DIR="${ROOT}/build"
SRC_DIR="${ROOT}/src"
RESULT_DIR="${ROOT}/results_clang"

mkdir -p "$BUILD_DIR"
mkdir -p "$RESULT_DIR"

cd "$BUILD_DIR"

ASM_CFLAGS=""
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
ASM_FORMAT=elf64
ASM_CFLAGS="-DPIC"
elif [[ "$OSTYPE" == "darwin"* ]]; then
ASM_FORMAT=macho64
ASM_CFLAGS="-DPIC -DPREFIX"
elif [[ "$OSTYPE" == "msys" ]]; then
ASM_FORMAT=win64
fi

ASM_CFLAGS="$ASM_CFLAGS -DHAVE_ALIGNED_STACK=1 -DARCH_X86_64=1 -DHAVE_CPUNOP=1 -DHAVE_AVX2_EXTERNAL=1 -DHAVE_AVX_EXTERNAL=1"
ASM_CFLAGS="--parser=nasm --oformat=${ASM_FORMAT}  $ASM_CFLAGS"

# yasm -g cv8 ${ASM_CFLAGS -o lut3d_asm_debug.obj lut3d.asm
yasm -f elf ${ASM_CFLAGS} -o lut3d_asm_release.o  "${SRC_DIR}/lut3d.asm"

clang++ -O2 -std=c++11 -c "${SRC_DIR}/deps/tinyexr.cpp" -I"${SRC_DIR}/deps/miniz" -o tinyexr.o
clang++ -O2 -std=c++11 -c "${SRC_DIR}/tetrahedral_ocio.cpp" -o ocio.o
clang   -O2 -msse2 -mavx2 -mfma -c "${SRC_DIR}/lut3d_perf.c" -o lut3d_perf.o
clang++ -O2 -std=c++11 lut3d_perf.o ocio.o tinyexr.o lut3d_asm_release.o -o ../lut3d_perf_clang

SAMPLE=2A5A2701.0001.exr
LUT=ACES2065-1_to_Rec709.csp
# IMAGE=2A5A2701_SLog3.0001.exr
# LUT=SLog3_to_ACESRec709.cube

cd "${RESULT_DIR}"
../lut3d_perf_clang ../samples/images/$SAMPLE ../samples/luts/$LUT