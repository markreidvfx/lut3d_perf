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

#include "tetrahedral_sse2.h"
#include <immintrin.h>

// Macros for alignment declarations
#define SSE2_SIMD_BYTES 16
#if defined( _MSC_VER )
#define SSE2_ALIGN(decl) __declspec(align(SSE2_SIMD_BYTES)) decl
#elif ( __APPLE__ )

#define SSE2_ALIGN(decl) decl
#else
#define SSE2_ALIGN(decl) decl __attribute__((aligned(SSE2_SIMD_BYTES)))
#endif

typedef struct {
    float *lut;
    __m128 lutmax;
    __m128 lutsize;
    __m128 lutsize2;

    float *prelut[3];
    __m128 prelut_max;
    __m128 prelut_min[3];
    __m128 prelut_scale[3];

} Lut3DContextSSE2;

typedef struct rgbvec_sse2 {
    __m128 r, g, b;
} rgbvec_sse2;

typedef struct rgabvec_sse2 {
    __m128 r, g, b, a;
} m128_rgbavec;


#define i32gather_ps_sse2(src, dst, idx, indices, buffer)  \
    _mm_store_si128((__m128i *)indices, idx);              \
    buffer[0] = (src)[indices[0]];                         \
    buffer[1] = (src)[indices[1]];                         \
    buffer[2] = (src)[indices[2]];                         \
    buffer[3] = (src)[indices[3]];                         \
    dst = _mm_load_ps(buffer)

#define gather_rgb_sse2(src, idx)             \
    _mm_store_si128((__m128i *)indices, idx); \
    buffer_r[0] = (src)[indices[0] + 0];      \
    buffer_g[0] = (src)[indices[0] + 1];      \
    buffer_b[0] = (src)[indices[0] + 2];      \
    buffer_r[1] = (src)[indices[1] + 0];      \
    buffer_g[1] = (src)[indices[1] + 1];      \
    buffer_b[1] = (src)[indices[1] + 2];      \
    buffer_r[2] = (src)[indices[2] + 0];      \
    buffer_g[2] = (src)[indices[2] + 1];      \
    buffer_b[2] = (src)[indices[2] + 2];      \
    buffer_r[3] = (src)[indices[3] + 0];      \
    buffer_g[3] = (src)[indices[3] + 1];      \
    buffer_b[3] = (src)[indices[3] + 2];      \
    sample_r = _mm_load_ps(buffer_r);         \
    sample_g = _mm_load_ps(buffer_g);         \
    sample_b = _mm_load_ps(buffer_b)

inline __m128 floor_ps_sse2(__m128 v)
{
    return _mm_cvtepi32_ps(_mm_cvttps_epi32(v));
}

inline __m128 blendv_ps_sse2(__m128 a, __m128 b, __m128 mask)
{
    return _mm_xor_ps(_mm_and_ps(_mm_xor_ps(a, b), mask), a);
}

inline __m128 fmadd_ps_sse2(__m128 a, __m128 b, __m128 c)
{
    return  _mm_add_ps(_mm_mul_ps(a, b), c);
}

inline __m128 apply_prelut_sse2(const Lut3DContextSSE2 *ctx, __m128 v, int idx)
{
    SSE2_ALIGN(uint32_t indices_p[4]);
    SSE2_ALIGN(uint32_t indices_n[4]);
    SSE2_ALIGN(float buffer_p[4]);
    SSE2_ALIGN(float buffer_n[4]);

    __m128 zero         = _mm_setzero_ps();
    __m128 one_f        = _mm_set1_ps(1.0);

    __m128 prelut_max   = ctx->prelut_max;
    __m128 prelut_min   = ctx->prelut_min[idx];
    __m128 prelut_scale = ctx->prelut_scale[idx];

    __m128 scaled = _mm_mul_ps(_mm_sub_ps(v, prelut_min), prelut_scale);

    // clamp, max first, NAN set to zero
    __m128 x      = _mm_min_ps(prelut_max, _mm_max_ps(zero, scaled));
    __m128 prev_f = floor_ps_sse2(x);
    __m128 d      = _mm_sub_ps(x, prev_f);
    __m128 next_f = _mm_min_ps(_mm_add_ps(prev_f, one_f), prelut_max);

    __m128i prev_i = _mm_cvtps_epi32(prev_f);
    __m128i next_i = _mm_cvtps_epi32(next_f);

    __m128 p,n;
    i32gather_ps_sse2(ctx->prelut[idx], p, prev_i, indices_p, buffer_p);
    i32gather_ps_sse2(ctx->prelut[idx], n, next_i, indices_n, buffer_n);

    // lerp: a + (b - a) * t;
    v = fmadd_ps_sse2(_mm_sub_ps(n, p), d, p);

    return v;
}

static inline rgbvec_sse2 interp_tetrahedral_sse2(const Lut3DContextSSE2 *ctx, __m128 r, __m128 g, __m128 b)
{
    SSE2_ALIGN(uint32_t indices[4]);
    SSE2_ALIGN(float buffer_r[4]);
    SSE2_ALIGN(float buffer_g[4]);
    SSE2_ALIGN(float buffer_b[4]);

    __m128 x0;
    __m128 x1;
    __m128 x2;

    __m128 cxxxa;
    __m128 cxxxb;
    __m128 mask;

    __m128 sample_r;
    __m128 sample_g;
    __m128 sample_b;

    rgbvec_sse2 result;

    __m128 lut_max  = ctx->lutmax;
    __m128 lutsize  = ctx->lutsize;
    __m128 lutsize2 = ctx->lutsize2;

    __m128 one_f    = _mm_set1_ps(1.0f);
    __m128 three_f  = _mm_set1_ps(3.0f);

    __m128 prev_r = floor_ps_sse2(r);
    __m128 prev_g = floor_ps_sse2(g);
    __m128 prev_b = floor_ps_sse2(b);

    // rgb delta values
    __m128 d_r = _mm_sub_ps(r, prev_r);
    __m128 d_g = _mm_sub_ps(g, prev_g);
    __m128 d_b = _mm_sub_ps(b, prev_b);

    __m128 next_r = _mm_min_ps(lut_max, _mm_add_ps(prev_r, one_f));
    __m128 next_g = _mm_min_ps(lut_max, _mm_add_ps(prev_g, one_f));
    __m128 next_b = _mm_min_ps(lut_max, _mm_add_ps(prev_b, one_f));

    // prescale indices
    prev_r = _mm_mul_ps(prev_r, lutsize2);
    next_r = _mm_mul_ps(next_r, lutsize2);

    prev_g = _mm_mul_ps(prev_g, lutsize);
    next_g = _mm_mul_ps(next_g, lutsize);

    prev_b = _mm_mul_ps(prev_b, three_f);
    next_b = _mm_mul_ps(next_b, three_f);

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

    __m128 gt_r = _mm_cmpgt_ps(d_r, d_g); // r>g
    __m128 gt_g = _mm_cmpgt_ps(d_g, d_b); // g>b
    __m128 gt_b = _mm_cmpgt_ps(d_b, d_r); // b>r

    // r> !b>r && r>g
    mask  = _mm_andnot_ps(gt_b, gt_r);
    cxxxa = blendv_ps_sse2(prev_r, next_r, mask);

    // r< !r>g && b>r
    mask  = _mm_andnot_ps(gt_r, gt_b);
    cxxxb = blendv_ps_sse2(next_r, prev_r, mask);

    // g> !r>g && g>b
    mask  = _mm_andnot_ps(gt_r, gt_g);
    cxxxa = _mm_add_ps(cxxxa, blendv_ps_sse2(prev_g, next_g, mask));

    // g< !g>b && r>g
    mask  = _mm_andnot_ps(gt_g, gt_r);
    cxxxb = _mm_add_ps(cxxxb, blendv_ps_sse2(next_g, prev_g, mask));

    // b> !g>b && b>r
    mask  = _mm_andnot_ps(gt_g, gt_b);
    cxxxa = _mm_add_ps(cxxxa, blendv_ps_sse2(prev_b, next_b, mask));

    // b< !b>r && g>b
    mask  = _mm_andnot_ps(gt_b, gt_g);
    cxxxb = _mm_add_ps(cxxxb, blendv_ps_sse2(next_b, prev_b, mask));

    __m128 c000 = _mm_add_ps(_mm_add_ps(prev_r, prev_g), prev_b);
    __m128 c111 = _mm_add_ps(_mm_add_ps(next_r, next_g), next_b);

    // sort delta r,g,b x0 >= x1 >= x2
    __m128 rg_min = _mm_min_ps(d_r, d_g);
    __m128 rg_max = _mm_max_ps(d_r, d_g);

    x2         = _mm_min_ps(rg_min, d_b);
    __m128 mid = _mm_max_ps(rg_min, d_b);

    x0 = _mm_max_ps(rg_max, d_b);
    x1 = _mm_min_ps(rg_max, mid);

    // convert indices to int
    __m128i c000_idx  = _mm_cvttps_epi32(c000);
    __m128i cxxxa_idx = _mm_cvttps_epi32(cxxxa);
    __m128i cxxxb_idx = _mm_cvttps_epi32(cxxxb);
    __m128i c111_idx  = _mm_cvttps_epi32(c111);

    gather_rgb_sse2((float*)ctx->lut, c000_idx);

    // (1-x0) * c000
    __m128 v = _mm_sub_ps(one_f, x0);
    result.r = _mm_mul_ps(sample_r, v);
    result.g = _mm_mul_ps(sample_g, v);
    result.b = _mm_mul_ps(sample_b, v);

    gather_rgb_sse2((float*)ctx->lut, cxxxa_idx);

    // (x0-x1) * cxxxa
    v = _mm_sub_ps(x0, x1);
    result.r = fmadd_ps_sse2(v, sample_r, result.r);
    result.g = fmadd_ps_sse2(v, sample_g, result.g);
    result.b = fmadd_ps_sse2(v, sample_b, result.b);

    gather_rgb_sse2((float*)ctx->lut, cxxxb_idx);

    // (x1-x2) * cxxxb
    v = _mm_sub_ps(x1, x2);
    result.r = fmadd_ps_sse2(v, sample_r, result.r);
    result.g = fmadd_ps_sse2(v, sample_g, result.g);
    result.b = fmadd_ps_sse2(v, sample_b, result.b);

    gather_rgb_sse2((float*)ctx->lut, c111_idx);

    // x2 * c111
    result.r = fmadd_ps_sse2(x2, sample_r, result.r);
    result.g = fmadd_ps_sse2(x2, sample_g, result.g);
    result.b = fmadd_ps_sse2(x2, sample_b, result.b);

    return result;
}

int apply_lut_planer_intrinsics_sse2(const LUT3DContext *lut3d, const FloatImage *src_image, FloatImage *dst_image)
{
    rgbvec_sse2 c;
    Lut3DContextSSE2 ctx;
    const Lut3DPreLut *prelut = &lut3d->prelut;

    float lutmax = (float)lut3d->lutsize - 1;
    __m128 scale_r = _mm_set1_ps(lut3d->scale.r * lutmax);
    __m128 scale_g = _mm_set1_ps(lut3d->scale.g * lutmax);
    __m128 scale_b = _mm_set1_ps(lut3d->scale.b * lutmax);
    __m128 zero    = _mm_setzero_ps();

    ctx.lut      = (float*)lut3d->lut;
    ctx.lutmax   = _mm_set1_ps(lutmax);
    ctx.lutsize  = _mm_set1_ps((float)lut3d->lutsize * 3);
    ctx.lutsize2 = _mm_set1_ps((float)lut3d->lutsize2 * 3);

    ctx.prelut[0] = prelut->lut[0];
    ctx.prelut[1] = prelut->lut[1];
    ctx.prelut[2] = prelut->lut[2];

    ctx.prelut_max    = _mm_set1_ps((float)prelut->size - 1);
    ctx.prelut_min[0] = _mm_set1_ps(prelut->min[0]);
    ctx.prelut_min[1] = _mm_set1_ps(prelut->min[1]);
    ctx.prelut_min[2] = _mm_set1_ps(prelut->min[2]);

    ctx.prelut_scale[0] = _mm_set1_ps(prelut->scale[0]);
    ctx.prelut_scale[1] = _mm_set1_ps(prelut->scale[1]);
    ctx.prelut_scale[2] = _mm_set1_ps(prelut->scale[2]);

    for (int y = 0; y < src_image->height; y++) {
        float *srcr = src_image->data[0] + y * src_image->stride;
        float *srcg = src_image->data[1] + y * src_image->stride;
        float *srcb = src_image->data[2] + y * src_image->stride;
        float *srca = src_image->data[3] + y * src_image->stride;

        float *dstr = dst_image->data[0] + y * dst_image->stride;
        float *dstg = dst_image->data[1] + y * dst_image->stride;
        float *dstb = dst_image->data[2] + y * dst_image->stride;
        float *dsta = dst_image->data[3] + y * dst_image->stride;

        for (int x = 0; x < src_image->width; x+=4) {
            __m128  r = _mm_loadu_ps(srcr + x);
            __m128  g = _mm_loadu_ps(srcg + x);
            __m128  b = _mm_loadu_ps(srcb + x);
            __m128  a = _mm_loadu_ps(srca + x);

            if (prelut->size) {
                r = apply_prelut_sse2(&ctx, r, 0);
                g = apply_prelut_sse2(&ctx, g, 1);
                b = apply_prelut_sse2(&ctx, b, 2);
            }

            // scale and clamp values
            r = _mm_min_ps(ctx.lutmax, _mm_max_ps(zero, _mm_mul_ps(r, scale_r)));
            g = _mm_min_ps(ctx.lutmax, _mm_max_ps(zero, _mm_mul_ps(g, scale_g)));
            b = _mm_min_ps(ctx.lutmax, _mm_max_ps(zero, _mm_mul_ps(b, scale_b)));

            c = interp_tetrahedral_sse2(&ctx, r, g, b);

            _mm_storeu_ps(dstr + x, c.r);
            _mm_storeu_ps(dstg + x, c.g);
            _mm_storeu_ps(dstb + x, c.b);
            _mm_storeu_ps(dsta + x, a);
        }
    }

    return 0;
}

inline m128_rgbavec rgba_transpose_4x4_sse2(__m128 row0, __m128 row1, __m128 row2, __m128 row3)
{
    m128_rgbavec result;
    __m128 tmp0 = _mm_unpacklo_ps(row0, row1);
    __m128 tmp2 = _mm_unpacklo_ps(row2, row3);
    __m128 tmp1 = _mm_unpackhi_ps(row0, row1);
    __m128 tmp3 = _mm_unpackhi_ps(row2, row3);
    result.r    = _mm_movelh_ps(tmp0, tmp2);
    result.g    = _mm_movehl_ps(tmp2, tmp0); // Note movhlps swaps b with a which is different than unpckhpd
    result.b    = _mm_movelh_ps(tmp1, tmp3);
    result.a    = _mm_movehl_ps(tmp3, tmp1);
    return result;
}


int apply_lut_rgba_intrinsics_sse2(const LUT3DContext *lut3d, const FloatImageRGBA *src_image, FloatImageRGBA *dst_image)
{
    m128_rgbavec c;
    rgbvec_sse2 c2;

    Lut3DContextSSE2 ctx;
    const Lut3DPreLut *prelut = &lut3d->prelut;

    float lutmax = (float)lut3d->lutsize - 1;
    __m128 scale_r = _mm_set1_ps(lut3d->scale.r * lutmax);
    __m128 scale_g = _mm_set1_ps(lut3d->scale.g * lutmax);
    __m128 scale_b = _mm_set1_ps(lut3d->scale.b * lutmax);
    __m128 zero    = _mm_setzero_ps();

    ctx.lut      = (float*)lut3d->lut;
    ctx.lutmax   = _mm_set1_ps(lutmax);
    ctx.lutsize  = _mm_set1_ps((float)lut3d->lutsize * 3);
    ctx.lutsize2 = _mm_set1_ps((float)lut3d->lutsize2 * 3);

    ctx.prelut[0] = prelut->lut[0];
    ctx.prelut[1] = prelut->lut[1];
    ctx.prelut[2] = prelut->lut[2];

    ctx.prelut_max    = _mm_set1_ps((float)prelut->size - 1);
    ctx.prelut_min[0] = _mm_set1_ps(prelut->min[0]);
    ctx.prelut_min[1] = _mm_set1_ps(prelut->min[1]);
    ctx.prelut_min[2] = _mm_set1_ps(prelut->min[2]);

    ctx.prelut_scale[0] = _mm_set1_ps(prelut->scale[0]);
    ctx.prelut_scale[1] = _mm_set1_ps(prelut->scale[1]);
    ctx.prelut_scale[2] = _mm_set1_ps(prelut->scale[2]);

    int total_pixel_count = src_image->width * src_image->height;
    int pixel_count = total_pixel_count / 4 * 4;
    int remainder = total_pixel_count - pixel_count;
    // printf("total: %d count %d remainder: %d\n",total_pixel_count, pixel_count, remainder);

    float *src = src_image->data;
    float *dst = dst_image->data;

    for (int i = 0; i < pixel_count; i += 4 ) {
        __m128 rgba0 = _mm_loadu_ps(src +  0);
        __m128 rgba1 = _mm_loadu_ps(src +  4);
        __m128 rgba2 = _mm_loadu_ps(src +  8);
        __m128 rgba3 = _mm_loadu_ps(src + 12);
        c = rgba_transpose_4x4_sse2(rgba0, rgba1, rgba2, rgba3);

        if (prelut->size) {
            c.r = apply_prelut_sse2(&ctx, c.r, 0);
            c.g = apply_prelut_sse2(&ctx, c.g, 1);
            c.b = apply_prelut_sse2(&ctx, c.b, 2);
        }

         // scale and clamp values
        c.r = _mm_min_ps(ctx.lutmax, _mm_max_ps(zero, _mm_mul_ps(c.r, scale_r)));
        c.g = _mm_min_ps(ctx.lutmax, _mm_max_ps(zero, _mm_mul_ps(c.g, scale_g)));
        c.b = _mm_min_ps(ctx.lutmax, _mm_max_ps(zero, _mm_mul_ps(c.b, scale_b)));

        c2 = interp_tetrahedral_sse2(&ctx, c.r, c.g, c.b);

        c = rgba_transpose_4x4_sse2(c2.r, c2.g, c2.b, c.a);
        _mm_storeu_ps(dst +  0, c.r);
        _mm_storeu_ps(dst +  4, c.g);
        _mm_storeu_ps(dst +  8, c.b);
        _mm_storeu_ps(dst + 12, c.a);

        src += 16;
        dst += 16;
    }

    // handler leftovers pixels
    if (remainder) {
        SSE2_ALIGN(float r[4]);
        SSE2_ALIGN(float g[4]);
        SSE2_ALIGN(float b[4]);
        SSE2_ALIGN(float a[4]);

        for (int i = 0; i < remainder; i++) {
            r[i] = src[0];
            g[i] = src[1];
            b[i] = src[2];
            a[i] = src[3];
            src += 4;
        }

        c.r = _mm_load_ps(r);
        c.g = _mm_load_ps(g);
        c.b = _mm_load_ps(b);

        if (prelut->size) {
            c.r = apply_prelut_sse2(&ctx, c.r, 0);
            c.g = apply_prelut_sse2(&ctx, c.g, 1);
            c.b = apply_prelut_sse2(&ctx, c.b, 2);
        }

        // scale and clamp values
        c.r = _mm_min_ps(ctx.lutmax, _mm_max_ps(zero, _mm_mul_ps(c.r, scale_r)));
        c.g = _mm_min_ps(ctx.lutmax, _mm_max_ps(zero, _mm_mul_ps(c.g, scale_g)));
        c.b = _mm_min_ps(ctx.lutmax, _mm_max_ps(zero, _mm_mul_ps(c.b, scale_b)));

        c2 = interp_tetrahedral_sse2(&ctx, c.r, c.g, c.b);

        _mm_store_ps(r, c2.r);
        _mm_store_ps(g, c2.g);
        _mm_store_ps(b, c2.b);

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


