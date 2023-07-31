#ifndef LUT3D_H
#define LUT3D_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "platform_info.h"

#define MAX_LEVEL 256
#define PRELUT_SIZE 65536

typedef struct rgbvec {
    float r, g, b;
} rgbvec;

typedef struct rgbavec {
    float r, g, b, a;
} rgbavec;

typedef struct Lut3DPreLut {
    int size;
    float min[3];
    float max[3];
    float scale[3];
    float* lut[3];
} Lut3DPreLut;

typedef struct LUT3DContext {
    void *av_class; // just for ffmpeg asm compat
    struct rgbvec *lut;
    int lutsize;
    int lutsize2;
    struct rgbvec scale;
    Lut3DPreLut prelut;
    struct rgbavec *rgba_lut;
} LUT3DContext;

typedef struct FloatImage {
    int width;
    int height;
    int channels;
    float *data[4];
    uint8_t *mem;
    int stride;
} FloatImage;

typedef struct FloatImageRGBA {
    int width;
    int height;
    float *data;
}FloatImageRGBA;

#endif /* LUT3D_H */