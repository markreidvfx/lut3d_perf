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

#include "tetrahedral_avx2.h"
#include <immintrin.h>

// Macros for alignment declarations
#define AVX2_SIMD_BYTES 16
#if defined( _MSC_VER )
#define AVX2_ALIGN(decl) __declspec(align(AVX2_SIMD_BYTES)) decl
#elif ( __APPLE__ )

#define AVX2_ALIGN(decl) decl
#else
#define AVX2_ALIGN(decl) decl __attribute__((aligned(AVX2_SIMD_BYTES)))
#endif

typedef struct {
    float *lut;
    __m256 lutmax;
    __m256 lutsize;
    __m256 lutsize2;

    float *prelut[3];
    __m256 prelut_max;
    __m256 prelut_min[3];
    __m256 prelut_scale[3];

} Lut3DContextAVX2;

typedef struct rgbvec_avx2 {
    __m256 r, g, b;
} rgbvec_avx2;

typedef struct rgbavec_avx2 {
    __m256 r, g, b, a;
} rgbavec_avx2;


static inline __m256 apply_prelut_avx2(const Lut3DContextAVX2 *ctx, __m256 v, int idx)
{
    __m256 zero         = _mm256_setzero_ps();
    __m256 one_f        = _mm256_set1_ps(1);

    __m256 prelut_max   = ctx->prelut_max;
    __m256 prelut_min   = ctx->prelut_min[idx];
    __m256 prelut_scale = ctx->prelut_scale[idx];

    __m256 scaled = _mm256_mul_ps(_mm256_sub_ps(v, prelut_min), prelut_scale);

    // clamp, max first, NAN set to zero
    __m256 x      = _mm256_min_ps(prelut_max, _mm256_max_ps(zero, scaled));
    __m256 prev_f = _mm256_floor_ps(x);
    __m256 d      = _mm256_sub_ps(x, prev_f);
    __m256 next_f = _mm256_min_ps(_mm256_add_ps(prev_f, one_f), prelut_max);

    __m256i prev_i = _mm256_cvtps_epi32(prev_f);
    __m256i next_i = _mm256_cvtps_epi32(next_f);

    __m256 p = _mm256_i32gather_ps(ctx->prelut[idx], prev_i, sizeof(float));
    __m256 n = _mm256_i32gather_ps(ctx->prelut[idx], next_i, sizeof(float));

    // lerp: a + (b - a) * t;
    v = _mm256_fmadd_ps(_mm256_sub_ps(n, p), d, p);

    return v;
}

static inline rgbvec_avx2 interp_tetrahedral_avx2(const Lut3DContextAVX2 *ctx, __m256 r, __m256 g, __m256 b)
{
    __m256 x0;
    __m256 x1;
    __m256 x2;

    __m256 cxxxa;
    __m256 cxxxb;
    __m256 mask;

    __m256 sample_r;
    __m256 sample_g;
    __m256 sample_b;

    rgbvec_avx2 result;

    __m256 lut_max  = ctx->lutmax;
    __m256 lutsize  = ctx->lutsize;
    __m256 lutsize2 = ctx->lutsize2;

    __m256 one_f    = _mm256_set1_ps(1.0f);
    __m256 three_f  = _mm256_set1_ps(3.0f);

    __m256 prev_r = _mm256_floor_ps(r);
    __m256 prev_g = _mm256_floor_ps(g);
    __m256 prev_b = _mm256_floor_ps(b);

    // rgb delta values
    __m256 d_r = _mm256_sub_ps(r, prev_r);
    __m256 d_g = _mm256_sub_ps(g, prev_g);
    __m256 d_b = _mm256_sub_ps(b, prev_b);

    __m256 next_r = _mm256_min_ps(lut_max, _mm256_add_ps(prev_r, one_f));
    __m256 next_g = _mm256_min_ps(lut_max, _mm256_add_ps(prev_g, one_f));
    __m256 next_b = _mm256_min_ps(lut_max, _mm256_add_ps(prev_b, one_f));

    // prescale indices
    prev_r = _mm256_mul_ps(prev_r, lutsize2);
    next_r = _mm256_mul_ps(next_r, lutsize2);

    prev_g = _mm256_mul_ps(prev_g, lutsize);
    next_g = _mm256_mul_ps(next_g, lutsize);

    prev_b = _mm256_mul_ps(prev_b, three_f);
    next_b = _mm256_mul_ps(next_b, three_f);

    // This is the tetrahedral blend equation
    // red = (1-x0) * c000.r + (x0-x1) * cxxxa.r + (x1-x2) * cxxxb.r + x2 * c111.r;
    // The x values are the rgb delta values sorted, x0 >= x1 >= x2
    // c### are samples from the lut, which are indices made with prev_(r,g,b) and next_(r,g,b) values
    // 0 = use prev, 1 = use next
    // c### = (prev_r or next_r) * (lutsize * lutsize) + (prev_g or next_g) * lutsize + (prev_b or next_b)

    // cxxxa
    // always uses 1 next and 2 prev and next is largest delta
    // r> == c100 == (r>g && r>b) == (!b>r && r>g)
    // g> == c010 == (g>r && g>b) == (!r>g && g>b)
    // b> == c001 == (b>r && b>g) == (!g>b && b>r)

    // cxxxb
    // always uses 2 next and 1 prev and prev is smallest delta
    // r< == c011 == (r<=g && r<=b) == (!r>g && b>r)
    // g< == c101 == (g<=r && g<=b) == (!g>b && r>g)
    // b< == c110 == (b<=r && b<=g) == (!b>r && g>b)

    // c000 and c111 are const (prev,prev,prev) and (next,next,next)

    __m256 gt_r = _mm256_cmp_ps(d_r, d_g, _CMP_GT_OQ); // r>g
    __m256 gt_g = _mm256_cmp_ps(d_g, d_b, _CMP_GT_OQ); // g>b
    __m256 gt_b = _mm256_cmp_ps(d_b, d_r, _CMP_GT_OQ); // b>r

    // r> !b>r && r>g
    mask = _mm256_andnot_ps(gt_b, gt_r);
    cxxxa = _mm256_blendv_ps(prev_r, next_r, mask);

    // r< !r>g && b>r
    mask = _mm256_andnot_ps(gt_r, gt_b);
    cxxxb = _mm256_blendv_ps(next_r, prev_r, mask);

    // g> !r>g && g>b
    mask = _mm256_andnot_ps(gt_r, gt_g);
    cxxxa = _mm256_add_ps(cxxxa, _mm256_blendv_ps(prev_g, next_g, mask));

    // g< !g>b && r>g
    mask = _mm256_andnot_ps(gt_g, gt_r);
    cxxxb = _mm256_add_ps(cxxxb, _mm256_blendv_ps(next_g, prev_g, mask));

    // b> !g>b && b>r
    mask = _mm256_andnot_ps(gt_g, gt_b);
    cxxxa = _mm256_add_ps(cxxxa, _mm256_blendv_ps(prev_b, next_b, mask));

    // b< !b>r && g>b
    mask = _mm256_andnot_ps(gt_b, gt_g);
    cxxxb = _mm256_add_ps(cxxxb, _mm256_blendv_ps(next_b, prev_b, mask));

    __m256 c000 = _mm256_add_ps(_mm256_add_ps(prev_r, prev_g), prev_b);
    __m256 c111 = _mm256_add_ps(_mm256_add_ps(next_r, next_g), next_b);

    // sort delta r,g,b x0 >= x1 >= x2
    __m256 rg_min = _mm256_min_ps(d_r, d_g);
    __m256 rg_max = _mm256_max_ps(d_r, d_g);

    x2         = _mm256_min_ps(rg_min, d_b);
    __m256 mid = _mm256_max_ps(rg_min, d_b);

    x0 = _mm256_max_ps(rg_max, d_b);
    x1 = _mm256_min_ps(rg_max, mid);

    // convert indices to int
    __m256i c000_idx  = _mm256_cvttps_epi32(c000);
    __m256i cxxxa_idx = _mm256_cvttps_epi32(cxxxa);
    __m256i cxxxb_idx = _mm256_cvttps_epi32(cxxxb);
    __m256i c111_idx  = _mm256_cvttps_epi32(c111);

    sample_r = _mm256_i32gather_ps((float*)ctx->lut+0, c000_idx, 4);
    sample_g = _mm256_i32gather_ps((float*)ctx->lut+1, c000_idx, 4);
    sample_b = _mm256_i32gather_ps((float*)ctx->lut+2, c000_idx, 4);

    // (1-x0) * c000
    __m256 v = _mm256_sub_ps(one_f, x0);
    result.r = _mm256_mul_ps(sample_r, v);
    result.g = _mm256_mul_ps(sample_g, v);
    result.b = _mm256_mul_ps(sample_b, v);

    sample_r = _mm256_i32gather_ps((float*)ctx->lut+0, cxxxa_idx, 4);
    sample_g = _mm256_i32gather_ps((float*)ctx->lut+1, cxxxa_idx, 4);
    sample_b = _mm256_i32gather_ps((float*)ctx->lut+2, cxxxa_idx, 4);

    // (x0-x1) * cxxxa
    v = _mm256_sub_ps(x0, x1);
    result.r = _mm256_fmadd_ps(v, sample_r, result.r);
    result.g = _mm256_fmadd_ps(v, sample_g, result.g);
    result.b = _mm256_fmadd_ps(v, sample_b, result.b);

    sample_r = _mm256_i32gather_ps((float*)ctx->lut+0, cxxxb_idx, 4);
    sample_g = _mm256_i32gather_ps((float*)ctx->lut+1, cxxxb_idx, 4);
    sample_b = _mm256_i32gather_ps((float*)ctx->lut+2, cxxxb_idx, 4);

    // (x1-x2) * cxxxb
    v = _mm256_sub_ps(x1, x2);
    result.r = _mm256_fmadd_ps(v, sample_r, result.r);
    result.g = _mm256_fmadd_ps(v, sample_g, result.g);
    result.b = _mm256_fmadd_ps(v, sample_b, result.b);

    sample_r = _mm256_i32gather_ps((float*)ctx->lut+0, c111_idx, 4);
    sample_g = _mm256_i32gather_ps((float*)ctx->lut+1, c111_idx, 4);
    sample_b = _mm256_i32gather_ps((float*)ctx->lut+2, c111_idx, 4);

    // x2 * c111
    result.r = _mm256_fmadd_ps(x2, sample_r, result.r);
    result.g = _mm256_fmadd_ps(x2, sample_g, result.g);
    result.b = _mm256_fmadd_ps(x2, sample_b, result.b);

    return result;
}


int apply_lut_planer_intrinsics_avx2(const LUT3DContext *lut3d, const FloatImage *src_image, FloatImage *dst_image)
{
    rgbvec_avx2 c;
    Lut3DContextAVX2 ctx;
    const Lut3DPreLut *prelut = &lut3d->prelut;

    float lutmax = (float)lut3d->lutsize - 1;
    __m256 scale_r = _mm256_set1_ps(lut3d->scale.r * lutmax);
    __m256 scale_g = _mm256_set1_ps(lut3d->scale.g * lutmax);
    __m256 scale_b = _mm256_set1_ps(lut3d->scale.b * lutmax);
    __m256 zero    = _mm256_setzero_ps();

    ctx.lut      = (float*)lut3d->lut;
    ctx.lutmax   = _mm256_set1_ps(lutmax);
    ctx.lutsize  = _mm256_set1_ps((float)lut3d->lutsize * 3);
    ctx.lutsize2 = _mm256_set1_ps((float)lut3d->lutsize2 * 3);

    ctx.prelut[0] = prelut->lut[0];
    ctx.prelut[1] = prelut->lut[1];
    ctx.prelut[2] = prelut->lut[2];

    ctx.prelut_max    = _mm256_set1_ps((float)prelut->size - 1);
    ctx.prelut_min[0] = _mm256_set1_ps(prelut->min[0]);
    ctx.prelut_min[1] = _mm256_set1_ps(prelut->min[1]);
    ctx.prelut_min[2] = _mm256_set1_ps(prelut->min[2]);

    ctx.prelut_scale[0] = _mm256_set1_ps(prelut->scale[0]);
    ctx.prelut_scale[1] = _mm256_set1_ps(prelut->scale[1]);
    ctx.prelut_scale[2] = _mm256_set1_ps(prelut->scale[2]);

    for (int y = 0; y < src_image->height; y++) {

        float *srcr = src_image->data[0] + y * src_image->stride;
        float *srcg = src_image->data[1] + y * src_image->stride;
        float *srcb = src_image->data[2] + y * src_image->stride;

        float *dstr = dst_image->data[0] + y * dst_image->stride;
        float *dstg = dst_image->data[1] + y * dst_image->stride;
        float *dstb = dst_image->data[2] + y * dst_image->stride;

        for (int x = 0; x < src_image->width; x+=8) {
            __m256  r = _mm256_loadu_ps(srcr + x);
            __m256  g = _mm256_loadu_ps(srcg + x);
            __m256  b = _mm256_loadu_ps(srcb + x);

            if (prelut->size) {
                r = apply_prelut_avx2(&ctx, r, 0);
                g = apply_prelut_avx2(&ctx, g, 1);
                b = apply_prelut_avx2(&ctx, b, 2);
            }

            // scale and clamp values
            r = _mm256_min_ps(ctx.lutmax, _mm256_max_ps(zero, _mm256_mul_ps(r, scale_r)));
            g = _mm256_min_ps(ctx.lutmax, _mm256_max_ps(zero, _mm256_mul_ps(g, scale_g)));
            b = _mm256_min_ps(ctx.lutmax, _mm256_max_ps(zero, _mm256_mul_ps(b, scale_b)));

            c = interp_tetrahedral_avx2(&ctx, r, g, b);

            _mm256_storeu_ps(dstr + x, c.r);
            _mm256_storeu_ps(dstg + x, c.g);
            _mm256_storeu_ps(dstb + x, c.b);
        }
    }
    return 0;
}

inline __m256 movelh_ps_avx2(__m256 a, __m256 b)
{
    return _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
}

inline __m256 movehl_ps_avx2(__m256 a, __m256 b)
{
    // NOTE: this is a and b are reversed to match sse2 movhlps which is different than unpckhpd
    return _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(b), _mm256_castps_pd(a)));
}

inline rgbavec_avx2 rgba_transpose_4x4_4x4_avx2(__m256 row0, __m256 row1, __m256 row2, __m256 row3)
{
    rgbavec_avx2 result;
    __m256 tmp0 = _mm256_unpacklo_ps(row0, row1);
    __m256 tmp2 = _mm256_unpacklo_ps(row2, row3);
    __m256 tmp1 = _mm256_unpackhi_ps(row0, row1);
    __m256 tmp3 = _mm256_unpackhi_ps(row2, row3);
    result.r    = movelh_ps_avx2(tmp0, tmp2);
    result.g    = movehl_ps_avx2(tmp2, tmp0);
    result.b    = movelh_ps_avx2(tmp1, tmp3);
    result.a    = movehl_ps_avx2(tmp3, tmp1);

    // the rgba transpose result will look this
    //
    //  0   1   2   3    0   1   2   3         0   1   2   3    0   1   2   3
    // r0, g0, b0, a0 | r1, g1, b1, a1        r0, r2, r4, r6 | r1, r3, r5, r7
    // r2, g2, b2, a2 | r3, g3, b3, a3  <==>  g0, g2, g4, g6 | g1, g3, g5, g7
    // r4, g4, b4, a4 | r5, g5, b5, a5  <==>  b0, b2, b4, b6 | b1, b3, b5, b7
    // r6, g6, b6, a5 | r7, g7, b7, a5        a0, a2, a4, a6 | a1, a4, a5, a7

    // each 128 lane is transpose independently,
    // the channel values end up with a even/odd shuffled order because of this.
    // The exact order is not important for the lut to work.

    return result;
}

int apply_lut_rgba_intrinsics_avx2(const LUT3DContext *lut3d, const FloatImageRGBA *src_image, FloatImageRGBA *dst_image)
{
    rgbavec_avx2 c0;
    rgbvec_avx2  c1;

    Lut3DContextAVX2 ctx;
    const Lut3DPreLut *prelut = &lut3d->prelut;

    float lutmax = (float)lut3d->lutsize - 1;
    __m256 scale_r = _mm256_set1_ps(lut3d->scale.r * lutmax);
    __m256 scale_g = _mm256_set1_ps(lut3d->scale.g * lutmax);
    __m256 scale_b = _mm256_set1_ps(lut3d->scale.b * lutmax);
    __m256 zero    = _mm256_setzero_ps();

    ctx.lut      = (float*)lut3d->lut;
    ctx.lutmax   = _mm256_set1_ps(lutmax);
    ctx.lutsize  = _mm256_set1_ps((float)lut3d->lutsize * 3);
    ctx.lutsize2 = _mm256_set1_ps((float)lut3d->lutsize2 * 3);

    ctx.prelut[0] = prelut->lut[0];
    ctx.prelut[1] = prelut->lut[1];
    ctx.prelut[2] = prelut->lut[2];

    ctx.prelut_max    = _mm256_set1_ps((float)prelut->size - 1);
    ctx.prelut_min[0] = _mm256_set1_ps(prelut->min[0]);
    ctx.prelut_min[1] = _mm256_set1_ps(prelut->min[1]);
    ctx.prelut_min[2] = _mm256_set1_ps(prelut->min[2]);

    ctx.prelut_scale[0] = _mm256_set1_ps(prelut->scale[0]);
    ctx.prelut_scale[1] = _mm256_set1_ps(prelut->scale[1]);
    ctx.prelut_scale[2] = _mm256_set1_ps(prelut->scale[2]);

    int total_pixel_count = src_image->width * src_image->height;
    int pixel_count = total_pixel_count / 8 * 8;
    int remainder = total_pixel_count - pixel_count;
    // printf("total: %d count %d remainder: %d\n",total_pixel_count, pixel_count, remainder);

    float *src = src_image->data;
    float *dst = dst_image->data;

    // NOTE: this interleaving order is the same transposing produces
    __m256i rgba_idx = _mm256_setr_epi32(0, 8, 16, 24, 4, 12, 20, 28);

    for (int i = 0; i < pixel_count; i += 8 ) {
#if 0
        __m256 rgba0 = _mm256_loadu_ps(src +  0);
        __m256 rgba1 = _mm256_loadu_ps(src +  8);
        __m256 rgba2 = _mm256_loadu_ps(src + 16);
        __m256 rgba3 = _mm256_loadu_ps(src + 24);
        c0 = rgba_transpose_4x4_4x4_avx(rgba0, rgba1, rgba2, rgba3);
#else
        c0.r = _mm256_i32gather_ps(src + 0, rgba_idx, 4);
        c0.g = _mm256_i32gather_ps(src + 1, rgba_idx, 4);
        c0.b = _mm256_i32gather_ps(src + 2, rgba_idx, 4);
        c0.a = _mm256_i32gather_ps(src + 3, rgba_idx, 4);
#endif
        if (prelut->size) {
            c0.r = apply_prelut_avx2(&ctx, c0.r, 0);
            c0.g = apply_prelut_avx2(&ctx, c0.g, 1);
            c0.b = apply_prelut_avx2(&ctx, c0.b, 2);
        }

        // scale and clamp values
        c0.r = _mm256_min_ps(ctx.lutmax, _mm256_max_ps(zero,  _mm256_mul_ps(c0.r, scale_r)));
        c0.g = _mm256_min_ps(ctx.lutmax, _mm256_max_ps(zero,  _mm256_mul_ps(c0.g, scale_g)));
        c0.b = _mm256_min_ps(ctx.lutmax, _mm256_max_ps(zero,  _mm256_mul_ps(c0.b, scale_b)));

        c1 = interp_tetrahedral_avx2(&ctx, c0.r, c0.g, c0.b);
        c0 = rgba_transpose_4x4_4x4_avx2(c1.r, c1.g, c1.b, c0.a);

        _mm256_storeu_ps(dst +  0, c0.r);
        _mm256_storeu_ps(dst +  8, c0.g);
        _mm256_storeu_ps(dst + 16, c0.b);
        _mm256_storeu_ps(dst + 24, c0.a);

        src += 32;
        dst += 32;
    }

     // handler leftovers pixels
    if (remainder) {
        AVX2_ALIGN(float r[8]);
        AVX2_ALIGN(float g[8]);
        AVX2_ALIGN(float b[8]);
        AVX2_ALIGN(float a[8]);

        for (int i = 0; i < remainder; i++) {
            r[i] = src[0];
            g[i] = src[1];
            b[i] = src[2];
            a[i] = src[3];
            src += 4;
        }

        c1.r = _mm256_load_ps(r);
        c1.g = _mm256_load_ps(g);
        c1.b = _mm256_load_ps(b);

        if (prelut->size) {
            c1.r = apply_prelut_avx2(&ctx, c1.r, 0);
            c1.g = apply_prelut_avx2(&ctx, c1.g, 1);
            c1.b = apply_prelut_avx2(&ctx, c1.b, 2);
        }

        // scale and clamp values
        c1.r = _mm256_min_ps(ctx.lutmax, _mm256_max_ps(zero,  _mm256_mul_ps(c1.r, scale_r)));
        c1.g = _mm256_min_ps(ctx.lutmax, _mm256_max_ps(zero,  _mm256_mul_ps(c1.g, scale_g)));
        c1.b = _mm256_min_ps(ctx.lutmax, _mm256_max_ps(zero,  _mm256_mul_ps(c1.b, scale_b)));

        c1 = interp_tetrahedral_avx2(&ctx, c1.r, c1.g, c1.b);

        _mm256_store_ps(r, c1.r);
        _mm256_store_ps(g, c1.g);
        _mm256_store_ps(b, c1.b);

        for (int i = 0; i < remainder; i++) {
            dst[0] = r[i];
            dst[1] = g[i];
            dst[2] = b[i];
            dst[3] = a[i];
            dst += 4;
        }
    }

    return 0;
}