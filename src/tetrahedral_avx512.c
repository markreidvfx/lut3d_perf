// The MIT License (MIT)

// Copyright (c) 2024 Mark Reid

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
#define AVX512_SIMD_BYTES 64
#if defined( _MSC_VER )
#define AVX512_ALIGN(decl) __declspec(align(AVX512_SIMD_BYTES)) decl
#elif ( __APPLE__ )

#define AVX512_ALIGN(decl) decl
#else
#define AVX512_ALIGN(decl) decl __attribute__((aligned(AVX512_SIMD_BYTES)))
#endif


typedef struct {
    float *lut;
    __m512 lutmax;
    __m512 lutsize;
    __m512 lutsize2;

    float *prelut[3];
    __m512 prelut_max;
    __m512 prelut_min[3];
    __m512 prelut_scale[3];

} Lut3DContextAVX512;

typedef struct {
    __m512 r, g, b;
} rgbvec_avx512;

typedef struct {
    __m512 r, g, b, a;
} rgbavec_avx512;

#define gather_rgb_avx512(src, idx)                            \
    sample_r = _mm512_i32gather_ps(idx, (void * )(src+0), 4);  \
    sample_g = _mm512_i32gather_ps(idx, (void * )(src+1), 4);  \
    sample_b = _mm512_i32gather_ps(idx, (void * )(src+2), 4)


static inline __m512 apply_prelut_avx512(const Lut3DContextAVX512 *ctx, __m512 v, int idx)
{
    __m512 zero   = _mm512_setzero_ps();
    __m512 one_f  = _mm512_set1_ps(1);

    __m512 prelut_max   = ctx->prelut_max;
    __m512 prelut_min   = ctx->prelut_min[idx];
    __m512 prelut_scale = ctx->prelut_scale[idx];

    __m512 scaled = _mm512_mul_ps(_mm512_sub_ps(v, prelut_min), prelut_scale);

    // clamp, max first, NAN set to zero
    __m512 x      = _mm512_min_ps(_mm512_max_ps(scaled, zero), prelut_max);
    __m512 prev_f = _mm512_floor_ps(x);
    __m512 d      = _mm512_sub_ps(x, prev_f);
    __m512 next_f = _mm512_min_ps(_mm512_add_ps(prev_f, one_f), prelut_max);

    __m512i prev_i = _mm512_cvttps_epi32(prev_f);
    __m512i next_i = _mm512_cvttps_epi32(next_f);

    __m512 p = _mm512_i32gather_ps(prev_i, ctx->prelut[idx], sizeof(float));
    __m512 n = _mm512_i32gather_ps(next_i, ctx->prelut[idx], sizeof(float));

    // lerp: a + (b - a) * t;
    v = _mm512_fmadd_ps(_mm512_sub_ps(n, p), d, p);

    return v;
}

static inline rgbvec_avx512 interp_tetrahedral_avx512(const Lut3DContextAVX512 *ctx, __m512 r, __m512 g, __m512 b)
{
    __m512 x0, x1, x2;
    __m512 cxxxa;
    __m512 cxxxb;
    __mmask16  mask;
    __m512 sample_r, sample_g, sample_b;

    rgbvec_avx512 result;

    __m512 lut_max  = ctx->lutmax;
    __m512 lutsize  = ctx->lutsize;
    __m512 lutsize2 = ctx->lutsize2;

    __m512 one_f   = _mm512_set1_ps(1.0f);
    __m512 four_f  = _mm512_set1_ps(4.0f);

    __m512 prev_r = _mm512_floor_ps(r);
    __m512 prev_g = _mm512_floor_ps(g);
    __m512 prev_b = _mm512_floor_ps(b);

    // rgb delta values
    __m512 d_r = _mm512_sub_ps(r, prev_r);
    __m512 d_g = _mm512_sub_ps(g, prev_g);
    __m512 d_b = _mm512_sub_ps(b, prev_b);

    __m512 next_r = _mm512_min_ps(lut_max, _mm512_add_ps(prev_r, one_f));
    __m512 next_g = _mm512_min_ps(lut_max, _mm512_add_ps(prev_g, one_f));
    __m512 next_b = _mm512_min_ps(lut_max, _mm512_add_ps(prev_b, one_f));

    // prescale indices
    prev_r = _mm512_mul_ps(prev_r, lutsize2);
    next_r = _mm512_mul_ps(next_r, lutsize2);

    prev_g = _mm512_mul_ps(prev_g, lutsize);
    next_g = _mm512_mul_ps(next_g, lutsize);

    prev_b = _mm512_mul_ps(prev_b, four_f);
    next_b = _mm512_mul_ps(next_b, four_f);

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

    __mmask16 gt_r = _mm512_cmp_ps_mask(d_r, d_g, _CMP_GT_OQ); // r>g
    __mmask16 gt_g = _mm512_cmp_ps_mask(d_g, d_b, _CMP_GT_OQ); // g>b
    __mmask16 gt_b = _mm512_cmp_ps_mask(d_b, d_r, _CMP_GT_OQ); // b>r

    // r> !b>r && r>g
    mask = _mm512_kandn(gt_b, gt_r);
    cxxxa = _mm512_mask_blend_ps(mask, prev_r, next_r);

    // r< !r>g && b>r
    mask = _mm512_kandn(gt_r, gt_b);
    cxxxb = _mm512_mask_blend_ps(mask, next_r, prev_r);

    // g> !r>g && g>b
    mask = _mm512_kandn(gt_r, gt_g);
    cxxxa = _mm512_add_ps(cxxxa, _mm512_mask_blend_ps(mask, prev_g, next_g));

    // g< !g>b && r>g
    mask = _mm512_kandn(gt_g, gt_r);
    cxxxb = _mm512_add_ps(cxxxb, _mm512_mask_blend_ps(mask, next_g, prev_g));

    // b> !g>b && b>r
    mask = _mm512_kandn(gt_g, gt_b);
    cxxxa = _mm512_add_ps(cxxxa, _mm512_mask_blend_ps(mask, prev_b, next_b));

    // b< !b>r && g>b
    mask = _mm512_kandn(gt_b, gt_g);
    cxxxb = _mm512_add_ps(cxxxb, _mm512_mask_blend_ps(mask, next_b, prev_b));

    __m512 c000 = _mm512_add_ps(_mm512_add_ps(prev_r, prev_g), prev_b);
    __m512 c111 = _mm512_add_ps(_mm512_add_ps(next_r, next_g), next_b);

    // sort delta r,g,b x0 >= x1 >= x2
    __m512 rg_min = _mm512_min_ps(d_r, d_g);
    __m512 rg_max = _mm512_max_ps(d_r, d_g);

    x2         = _mm512_min_ps(rg_min, d_b);
    __m512 mid = _mm512_max_ps(rg_min, d_b);

    x0 = _mm512_max_ps(rg_max, d_b);
    x1 = _mm512_min_ps(rg_max, mid);

    // convert indices to int
    __m512i c000_idx  = _mm512_cvttps_epi32(c000);
    __m512i cxxxa_idx = _mm512_cvttps_epi32(cxxxa);
    __m512i cxxxb_idx = _mm512_cvttps_epi32(cxxxb);
    __m512i c111_idx  = _mm512_cvttps_epi32(c111);

    gather_rgb_avx512(ctx->lut, c000_idx);

    // (1-x0) * c000
    __m512 v = _mm512_sub_ps(one_f, x0);
    result.r = _mm512_mul_ps(sample_r, v);
    result.g = _mm512_mul_ps(sample_g, v);
    result.b = _mm512_mul_ps(sample_b, v);

    gather_rgb_avx512(ctx->lut, cxxxa_idx);

    // (x0-x1) * cxxxa
    v = _mm512_sub_ps(x0, x1);
    result.r = _mm512_fmadd_ps(v, sample_r, result.r);
    result.g = _mm512_fmadd_ps(v, sample_g, result.g);
    result.b = _mm512_fmadd_ps(v, sample_b, result.b);

    gather_rgb_avx512(ctx->lut, cxxxb_idx);

    // (x1-x2) * cxxxb
    v = _mm512_sub_ps(x1, x2);
    result.r = _mm512_fmadd_ps(v, sample_r, result.r);
    result.g = _mm512_fmadd_ps(v, sample_g, result.g);
    result.b = _mm512_fmadd_ps(v, sample_b, result.b);

    gather_rgb_avx512(ctx->lut, c111_idx);

    // x2 * c111
    result.r = _mm512_fmadd_ps(x2, sample_r, result.r);
    result.g = _mm512_fmadd_ps(x2, sample_g, result.g);
    result.b = _mm512_fmadd_ps(x2, sample_b, result.b);

    return result;
}

int apply_lut_planer_intrinsics_avx512(const LUT3DContext *lut3d, const FloatImage *src_image, FloatImage *dst_image)
{
    rgbvec_avx512 c;
    Lut3DContextAVX512 ctx;
    const Lut3DPreLut *prelut = &lut3d->prelut;

    float lutmax = (float)lut3d->lutsize - 1;
    __m512 scale_r = _mm512_set1_ps(lut3d->scale.r * lutmax);
    __m512 scale_g = _mm512_set1_ps(lut3d->scale.g * lutmax);
    __m512 scale_b = _mm512_set1_ps(lut3d->scale.b * lutmax);
    __m512 zero    = _mm512_setzero_ps();

    ctx.lut      = (float*)lut3d->rgba_lut;
    ctx.lutmax   = _mm512_set1_ps(lutmax);
    ctx.lutsize  = _mm512_set1_ps((float)lut3d->lutsize * 4);
    ctx.lutsize2 = _mm512_set1_ps((float)lut3d->lutsize2 * 4);

    ctx.prelut[0] = prelut->lut[0];
    ctx.prelut[1] = prelut->lut[1];
    ctx.prelut[2] = prelut->lut[2];

    ctx.prelut_max    = _mm512_set1_ps((float)prelut->size - 1);
    ctx.prelut_min[0] = _mm512_set1_ps(prelut->min[0]);
    ctx.prelut_min[1] = _mm512_set1_ps(prelut->min[1]);
    ctx.prelut_min[2] = _mm512_set1_ps(prelut->min[2]);

    ctx.prelut_scale[0] = _mm512_set1_ps(prelut->scale[0]);
    ctx.prelut_scale[1] = _mm512_set1_ps(prelut->scale[1]);
    ctx.prelut_scale[2] = _mm512_set1_ps(prelut->scale[2]);

    for (int y = 0; y < src_image->height; y++) {

        float *srcr = src_image->data[0] + y * src_image->stride;
        float *srcg = src_image->data[1] + y * src_image->stride;
        float *srcb = src_image->data[2] + y * src_image->stride;
        float *srca = src_image->data[3] + y * src_image->stride;

        float *dstr = dst_image->data[0] + y * dst_image->stride;
        float *dstg = dst_image->data[1] + y * dst_image->stride;
        float *dstb = dst_image->data[2] + y * dst_image->stride;
        float *dsta = dst_image->data[3] + y * dst_image->stride;

        for (int x = 0; x < src_image->width; x+=16) {
            __m512  r = _mm512_loadu_ps(srcr + x);
            __m512  g = _mm512_loadu_ps(srcg + x);
            __m512  b = _mm512_loadu_ps(srcb + x);
            __m512  a = _mm512_loadu_ps(srca + x);

            if (prelut->size) {
                r = apply_prelut_avx512(&ctx, r, 0);
                g = apply_prelut_avx512(&ctx, g, 1);
                b = apply_prelut_avx512(&ctx, b, 2);
            }

            // scale and clamp values
            r = _mm512_min_ps(ctx.lutmax, _mm512_max_ps(_mm512_mul_ps(r, scale_r), zero));
            g = _mm512_min_ps(ctx.lutmax, _mm512_max_ps(_mm512_mul_ps(g, scale_g), zero));
            b = _mm512_min_ps(ctx.lutmax, _mm512_max_ps(_mm512_mul_ps(b, scale_b), zero));

            c = interp_tetrahedral_avx512(&ctx, r, g, b);

            _mm512_storeu_ps(dstr + x, c.r);
            _mm512_storeu_ps(dstg + x, c.g);
            _mm512_storeu_ps(dstb + x, c.b);
            _mm512_storeu_ps(dsta + x,   a);
        }
    }
    return 0;
}

inline __m512 avx512_movelh_ps(__m512 a, __m512 b)
{
    return _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(a), _mm512_castps_pd(b)));
}

inline __m512 avx512_movehl_ps(__m512 a, __m512 b)
{
    // NOTE: this is a and b are reversed to match sse2 movhlps which is different than unpckhpd
    return _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(b), _mm512_castps_pd(a)));
}


static inline rgbavec_avx512 avx512RGBATranspose_4x4_4x4_4x4_4x4(__m512 row0,   __m512 row1,   __m512 row2,   __m512 row3)
{
    rgbavec_avx512 result;
    // the rgba transpose result will look this
    //
    //   0    1    2    3 |   4    5    6    7     8    9   10   11    12   13   14   15
    //  r0,  g0,  b0,  a0 |  r1,  g1,  b1,  a1 |  r2,  g2,  b2,  a2 |  r3,  g3,  b3,  a3
    //  r4,  g4,  b4,  a4 |  r5,  g5,  b5,  a5 |  r6,  g6,  b6,  a6 |  r7,  g7,  b7,  a7
    //  r8   g8,  b8,  a8 |  r9,  g9,  b9,  a9 | r10, g10, b10, a10 | r11, g11, b11, a11
    // r12, g12, b12, a12 | r13, g13, b13, a13 | r14, g14, b14, a14 | r15, g15, b15, a15
    //                    |                    |                    |
    //         |          |          |         |          |         |          |
    //         V          |          V         |          V         |          V
    //                    |                    |                    |
    //  r0,  r4,  r8, r12 |  r1,  r5,  r9, r13 |  r2,  r6, r10, r14 |  r3,  r7, r11, r15
    //  g0,  g4,  g8, g12 |  g1,  g5,  g9, g13 |  g2,  g6, g10, g14 |  g3,  g7, g11, g15
    //  b0,  b4,  b9, b12 |  b1,  b5,  b9, b13 |  b2,  b6, b10, b14 |  b3,  b7, b11, b15
    //  a0,  a4,  a8, a12 |  a1,  a5,  a9, a13 |  a2,  a6, a10, a14 |  a3,  a7, a11, a15


    // each 128 lane is transposed independently,
    // the channel values end up with a even/odd shuffled order because of this.
    // if exact order is important more cross lane shuffling is needed

    __m512 tmp0 = _mm512_unpacklo_ps(row0, row1);
    __m512 tmp2 = _mm512_unpacklo_ps(row2, row3);
    __m512 tmp1 = _mm512_unpackhi_ps(row0, row1);
    __m512 tmp3 = _mm512_unpackhi_ps(row2, row3);

    result.r = avx512_movelh_ps(tmp0, tmp2);
    result.g = avx512_movehl_ps(tmp2, tmp0);
    result.b = avx512_movelh_ps(tmp1, tmp3);
    result.a = avx512_movehl_ps(tmp3, tmp1);

    return result;
}


int apply_lut_rgba_intrinsics_avx512(const LUT3DContext *lut3d, const FloatImageRGBA *src_image, FloatImageRGBA *dst_image)
{
    rgbavec_avx512 c0;
    rgbvec_avx512  c1;

    Lut3DContextAVX512 ctx;
    const Lut3DPreLut *prelut = &lut3d->prelut;

    float lutmax = (float)lut3d->lutsize - 1;
    __m512 scale_r = _mm512_set1_ps(lut3d->scale.r * lutmax);
    __m512 scale_g = _mm512_set1_ps(lut3d->scale.g * lutmax);
    __m512 scale_b = _mm512_set1_ps(lut3d->scale.b * lutmax);
    __m512 zero    = _mm512_setzero_ps();

    ctx.lut      = (float*)lut3d->rgba_lut;
    ctx.lutmax   = _mm512_set1_ps(lutmax);
    ctx.lutsize  = _mm512_set1_ps((float)lut3d->lutsize * 4);
    ctx.lutsize2 = _mm512_set1_ps((float)lut3d->lutsize2 * 4);

    ctx.prelut[0] = prelut->lut[0];
    ctx.prelut[1] = prelut->lut[1];
    ctx.prelut[2] = prelut->lut[2];

    ctx.prelut_max    = _mm512_set1_ps((float)prelut->size - 1);
    ctx.prelut_min[0] = _mm512_set1_ps(prelut->min[0]);
    ctx.prelut_min[1] = _mm512_set1_ps(prelut->min[1]);
    ctx.prelut_min[2] = _mm512_set1_ps(prelut->min[2]);

    ctx.prelut_scale[0] = _mm512_set1_ps(prelut->scale[0]);
    ctx.prelut_scale[1] = _mm512_set1_ps(prelut->scale[1]);
    ctx.prelut_scale[2] = _mm512_set1_ps(prelut->scale[2]);

    int total_pixel_count = src_image->width * src_image->height;
    int pixel_count = total_pixel_count / 16 * 16;
    int remainder = total_pixel_count - pixel_count;
    // printf("total: %d count %d remainder: %d\n",total_pixel_count, pixel_count, remainder);

    float *src = src_image->data;
    float *dst = dst_image->data;

    // NOTE: this interleaving order is the same transposing produces
    // __m256i rgba_idx = _mm512_setr_epi32(0, 8, 16, 24, 4, 12, 20, 28);

    for (int i = 0; i < pixel_count; i += 16 ) {
#if 1
        __m512 rgba0 = _mm512_loadu_ps(src +  0);
        __m512 rgba1 = _mm512_loadu_ps(src + 16);
        __m512 rgba2 = _mm512_loadu_ps(src + 32);
        __m512 rgba3 = _mm512_loadu_ps(src + 48);
        c0 = avx512RGBATranspose_4x4_4x4_4x4_4x4(rgba0, rgba1, rgba2, rgba3);
#else
        c0.r = _mm256_i32gather_ps(src + 0, rgba_idx, 4);
        c0.g = _mm256_i32gather_ps(src + 1, rgba_idx, 4);
        c0.b = _mm256_i32gather_ps(src + 2, rgba_idx, 4);
        c0.a = _mm256_i32gather_ps(src + 3, rgba_idx, 4);
#endif
        if (prelut->size) {
            c0.r = apply_prelut_avx512(&ctx, c0.r, 0);
            c0.g = apply_prelut_avx512(&ctx, c0.g, 1);
            c0.b = apply_prelut_avx512(&ctx, c0.b, 2);
        }

        // scale and clamp values
        c0.r = _mm512_min_ps(ctx.lutmax, _mm512_max_ps(zero,  _mm512_mul_ps(c0.r, scale_r)));
        c0.g = _mm512_min_ps(ctx.lutmax, _mm512_max_ps(zero,  _mm512_mul_ps(c0.g, scale_g)));
        c0.b = _mm512_min_ps(ctx.lutmax, _mm512_max_ps(zero,  _mm512_mul_ps(c0.b, scale_b)));

        c1 = interp_tetrahedral_avx512(&ctx, c0.r, c0.g, c0.b);
        c0 = avx512RGBATranspose_4x4_4x4_4x4_4x4(c1.r, c1.g, c1.b, c0.a);

        _mm512_storeu_ps(dst +  0, c0.r);
        _mm512_storeu_ps(dst + 16, c0.g);
        _mm512_storeu_ps(dst + 32, c0.b);
        _mm512_storeu_ps(dst + 48, c0.a);

        src += 64;
        dst += 64;
    }

     // handler leftovers pixels
    if (remainder) {
        AVX512_ALIGN(float r[16]);
        AVX512_ALIGN(float g[16]);
        AVX512_ALIGN(float b[16]);
        AVX512_ALIGN(float a[16]);

        for (int i = 0; i < remainder; i++) {
            r[i] = src[0];
            g[i] = src[1];
            b[i] = src[2];
            a[i] = src[3];
            src += 4;
        }

        c1.r = _mm512_load_ps(r);
        c1.g = _mm512_load_ps(g);
        c1.b = _mm512_load_ps(b);

        if (prelut->size) {
            c1.r = apply_prelut_avx512(&ctx, c1.r, 0);
            c1.g = apply_prelut_avx512(&ctx, c1.g, 1);
            c1.b = apply_prelut_avx512(&ctx, c1.b, 2);
        }

        // scale and clamp values
        c1.r = _mm512_min_ps(ctx.lutmax, _mm512_max_ps(_mm512_mul_ps(c1.r, scale_r), zero));
        c1.g = _mm512_min_ps(ctx.lutmax, _mm512_max_ps(_mm512_mul_ps(c1.g, scale_g), zero));
        c1.b = _mm512_min_ps(ctx.lutmax, _mm512_max_ps(_mm512_mul_ps(c1.b, scale_b), zero));

        c1 = interp_tetrahedral_avx512(&ctx, c1.r, c1.g, c1.b);

        _mm512_store_ps(r, c1.r);
        _mm512_store_ps(g, c1.g);
        _mm512_store_ps(b, c1.b);

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