cmake_minimum_required(VERSION 3.10)

project(lut3d_perf
    VERSION 1.0.0
    DESCRIPTION "Performance Tests For Various Tetrahedral 3D Lut Implementations"
    HOMEPAGE_URL https://github.com/markreidvfx/lut3d_perf
    LANGUAGES C CXX ASM_NASM)

add_subdirectory(src)

enable_testing()
add_test(NAME test_rand COMMAND lut3d_perf)
add_test(NAME test_lut1 COMMAND lut3d_perf ${CMAKE_SOURCE_DIR}/samples/images/2A5A2701.0001.exr ${CMAKE_SOURCE_DIR}/samples/luts/ACES2065-1_to_Rec709.csp)
add_test(NAME test_lut2 COMMAND lut3d_perf ${CMAKE_SOURCE_DIR}/samples/images/2A5A2701_SLog3.0001.exr ${CMAKE_SOURCE_DIR}/samples/luts/SLog3_to_ACESRec709.cube)