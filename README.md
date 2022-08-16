# Performance Tests For Various Tetrahedral 3D Lut Implementations

This is a small project to test performance between different
tetrahedral 3D Lut implementations and compilers.
I've only focused x86_64 and SSE2, AVX and AVX2 instruction sets for now.
It would be nice to have a ARM NEON implementation at some point.

## Test Implementations

- FFmpeg c
- OpenColorIO c++
- OpenColorIO SSE2
- AVX2 intrinsics
- AVX intrinsics
- SSE2 intrinsics
- FFmpeg AVX2 assembly
- FFmpeg AVX assembly
- FFmpeg SSE2 assembly

## Requirements
- yasm
- msvc, gcc or clang

## Test Results
![Random_1024x1024](./images/Random_lut_1024x1024.png)
![ACES2065-1_to_Rec709](./images/ACES2065-1_to_Rec709_results.png)
![SLog3_to_ACESRec709](./images/SLog3_to_ACESRec709_results.png)
![Random_1024x1024_macos](./images/Random_lut_1024x1024_macos.png)

AVX intrinsic version can produce slower/similar speed code to SSE2 on some compilers when compared to the AVX assembly version.
Still investigating why.
