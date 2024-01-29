// The MIT License (MIT)

// Copyright (c) 2022 Mark Reid

// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TETRAHEDRAL_AVX512_H
#define TETRAHEDRAL_AVX512_H

#include "lut3d_perf.h"

int apply_lut_planer_intrinsics_avx512(const LUT3DContext *lut3d, const FloatImage *src_image, FloatImage *dst_image);
int apply_lut_rgba_intrinsics_avx512(const LUT3DContext *lut3d, const FloatImageRGBA *src_image, FloatImageRGBA *dst_image);

#endif /* TETRAHEDRAL_AVX512_H */