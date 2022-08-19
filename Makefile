all: lut3d_perf

CC ?= gcc
CXX ?= g++
YASM ?= yasm
# CC = clang
# CXX = clang++

EXE=
ASM_CFLAGS=

ifeq ($(OS),Windows_NT)
	ASM_FORMAT=win64
	EXE=.exe
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		ASM_CFLAGS += -DPIC
		ASM_FORMAT = elf64
	endif
	ifeq ($(UNAME_S),Darwin)
		ASM_FORMAT = macho64
		ASM_CFLAGS += -DPIC -DPREFIX
		CC = clang
		CXX = clang++
	endif
endif

PROG=lut3d_perf_$(CC)$(EXE)
BUILD_DIR=build/$(CC)
TEST_DIR=results/$(CC)
SRC_DIR=src

SAMPLES_DIR=samples

TEST_IMAGE1=$(SAMPLES_DIR)/images/2A5A2701.0001.exr
TEST_LUT1=$(SAMPLES_DIR)/luts/ACES2065-1_to_Rec709.csp

TEST_IMAGE2=$(SAMPLES_DIR)/images/2A5A2701_SLog3.0001.exr
TEST_LUT2=$(SAMPLES_DIR)/luts/SLog3_to_ACESRec709.cube


ASM_CFLAGS += -DHAVE_ALIGNED_STACK=1 -DARCH_X86_64=1 -DHAVE_CPUNOP=1 -DHAVE_AVX2_EXTERNAL=1 -DHAVE_AVX_EXTERNAL=1
ASM_CFLAGS += --parser=nasm --oformat=$(ASM_FORMAT)

objects=$(BUILD_DIR)/lut3d_perf.o $(BUILD_DIR)/lut3d_asm.o $(BUILD_DIR)/ocio.o $(BUILD_DIR)/tinyexr.o $(BUILD_DIR)/tetrahedral_sse2.o $(BUILD_DIR)/tetrahedral_avx.o $(BUILD_DIR)/tetrahedral_avx2.o

$(BUILD_DIR):
	mkdir -vp ${BUILD_DIR}

$(BUILD_DIR)/lut3d_perf.o: $(SRC_DIR)/lut3d_perf.c $(SRC_DIR)/*.h $(SRC_DIR)/*.c
	$(CC) -O2 -msse2 -c ${SRC_DIR}/lut3d_perf.c -o ${BUILD_DIR}/lut3d_perf.o

$(BUILD_DIR)/lut3d_asm.o: $(SRC_DIR)/lut3d.asm
	$(YASM) $(ASM_CFLAGS) -o ${BUILD_DIR}/lut3d_asm.o  ${SRC_DIR}/lut3d.asm

$(BUILD_DIR)/tinyexr.o: src/deps/tinyexr.cpp
	$(CXX) -O2 -std=c++11 -c ${SRC_DIR}/deps/tinyexr.cpp -I${SRC_DIR}/deps/miniz -o $(BUILD_DIR)/tinyexr.o

$(BUILD_DIR)/ocio.o: src/tetrahedral_ocio.cpp
	$(CXX) -O2 -std=c++11 -c ${SRC_DIR}/tetrahedral_ocio.cpp -o $(BUILD_DIR)/ocio.o

$(BUILD_DIR)/tetrahedral_sse2.o: src/tetrahedral_sse2.c
	$(CC) -O2 -msse2 -c ${SRC_DIR}/tetrahedral_sse2.c -o $(BUILD_DIR)/tetrahedral_sse2.o

$(BUILD_DIR)/tetrahedral_avx.o: src/tetrahedral_avx.c
	$(CC) -O2 -mavx -c ${SRC_DIR}/tetrahedral_avx.c -o $(BUILD_DIR)/tetrahedral_avx.o

$(BUILD_DIR)/tetrahedral_avx2.o: src/tetrahedral_avx2.c
	$(CC) -O2 -mavx2 -mfma  -c ${SRC_DIR}/tetrahedral_avx2.c -o $(BUILD_DIR)/tetrahedral_avx2.o

$(PROG): $(BUILD_DIR) $(objects)
	$(CXX) -O2 -std=c++11 $(objects) -o $(PROG)

lut3d_perf: $(PROG)

test_rand: lut3d_perf
	mkdir -p $(TEST_DIR)/test_rand
	cd $(TEST_DIR)/test_rand && ../../../$(PROG)

test_lut1: lut3d_perf
	mkdir -p $(TEST_DIR)/test_lut1
	cd $(TEST_DIR)/test_lut1 && ../../../$(PROG) ../../../$(TEST_IMAGE1) ../../../$(TEST_LUT1)

test_lut2: lut3d_perf
	mkdir -p $(TEST_DIR)/test_lut2
	cd $(TEST_DIR)/test_lut2 && ../../../$(PROG) ../../../$(TEST_IMAGE2) ../../../$(TEST_LUT2)

clean:
	rm -Rvf $(BUILD_DIR)
	rm -Rvf $(TEST_DIR)
	rm -vf $(PROG)