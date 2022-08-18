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
- [yasm](https://yasm.tortall.net)
- msvc, gcc or clang

## Test Results
![Random_1024x1024_windows](./images/Random_lut_1024x1024_windows.png)
![Random_1024x1024_linux](./images/Random_lut_1024x1024_linux.png)
![Random_1024x1024_macos](./images/Random_lut_1024x1024_macos.png)
![ACES2065-1_to_Rec709_windows](./images/ACES2065-1_to_Rec709_windows.png)
![SLog3_to_ACESRec709_windows](./images/SLog3_to_ACESRec709_windows.png)

AVX intrinsic version can produce slower/similar speed code to SSE2 on some compilers when compared to the AVX assembly version.
Still investigating why.

## Testing

### Linux / macOS / MSYS2

From a terminal run

```bash
make test_rand
make test_lut1
make test_lut2
```

### Windows MSVC

From the **MSVC x64 native tools** command prompt run

```cmd
build_msvc.bat test_rand
build_msvc.bat test_lut1
build_msvc.bat test_lut2
```
