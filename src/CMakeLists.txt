add_library(tinyexr deps/tinyexr.cpp)
target_include_directories(tinyexr
    PRIVATE
    deps/miniz)

set(ASM_ARGS
    -DHAVE_ALIGNED_STACK=1
    -DARCH_X86_64=1
    -DHAVE_CPUNOP=1
    -DHAVE_AVX2_EXTERNAL=1
    -DHAVE_AVX_EXTERNAL=1
    -DPIC
)

if(APPLE)
    list(APPEND ASM_ARGS -DPREFIX)
endif()

if (MSVC)
    set(AVX_ARGS /arch:AVX)
    set(AVX2_ARGS /arch:AVX2)
else()
    set(AVX_ARGS -mavx)
    set(AVX2_ARGS -mavx2 -mfma)
endif()

set_property(SOURCE lut3d.asm APPEND PROPERTY COMPILE_OPTIONS ${ASM_ARGS})
add_library(tetrahedral_asm lut3d.asm)

set_property(SOURCE tetrahedral_avx.c APPEND PROPERTY COMPILE_OPTIONS ${AVX_ARGS})
set_property(SOURCE tetrahedral_avx2.c APPEND PROPERTY COMPILE_OPTIONS ${AVX2_ARGS})

add_executable(lut3d_perf
    lut3d_perf.c
    tetrahedral_sse2.c
    tetrahedral_avx.c
    tetrahedral_avx2.c
    tetrahedral_ocio.cpp
)

target_link_libraries(lut3d_perf
    tinyexr
    tetrahedral_asm
)
