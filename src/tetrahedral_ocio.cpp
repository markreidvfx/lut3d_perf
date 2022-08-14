// this code is extracted from OpenColorIO
// https://github.com/AcademySoftwareFoundation/OpenColorIO
// from Lut1DOpCPU.cpp and Lut3DOpCPU.cpp

#include <vector>
#include <cmath>
#include <algorithm>

#include <memory>
#include <stdint.h>
#include "lut3d_perf.h"

#include <emmintrin.h>

extern "C" {
    int apply_lut_ocio_rgba(const LUT3DContext *lut3d, const FloatImageRGBA *src_image, FloatImageRGBA *dst_image);
    int apply_lut_ocio_sse2_rgba(const LUT3DContext *lut3d, const FloatImageRGBA *src_image, FloatImageRGBA *dst_image);
}

// Macros for alignment declarations
#define OCIO_SIMD_BYTES 16
#if defined( _MSC_VER )
#define OCIO_ALIGN(decl) __declspec(align(OCIO_SIMD_BYTES)) decl
#elif ( __APPLE__ )

#define OCIO_ALIGN(decl) decl
#else
#define OCIO_ALIGN(decl) decl __attribute__((aligned(OCIO_SIMD_BYTES)))
#endif


// Clamp value a to[min, max]
// First compare with max, then with min.
//
// Note: Does not validate max >= min.
// Note: NaN values become 0.
template<typename T>
inline T Clamp(T a, T min, T max)
{
    return std::min(std::max(min, a), max);
}

inline float lerpf(float a, float b, float z)
{
    return (b - a) * z + a;
}


int GetLut3DIndexBlueFast(int indexR, int indexG, int indexB, long dim)
{
    return 3 * (indexB + (int)dim * (indexG + (int)dim * indexR));
}

int apply_lut_ocio_rgba(const LUT3DContext *lut3d, const FloatImageRGBA *src_image, FloatImageRGBA *dst_image)
{
    const Lut3DPreLut *prelut = &lut3d->prelut;

    int numPixels = src_image->width * src_image->height;

    float *in  = src_image->data;
    float *out = dst_image->data;

    const float prelut_dimMinusOne = (float)(prelut->size - 1);

    const float *preLutR = prelut->lut[0];
    const float *preLutG = prelut->lut[1];
    const float *preLutB = prelut->lut[2];

    float dimMinusOne = (float)lut3d->lutsize - 1;
    float scale_r = lut3d->scale.r * dimMinusOne;
    float scale_g = lut3d->scale.g * dimMinusOne;
    float scale_b = lut3d->scale.b * dimMinusOne;

    long m_dim = lut3d->lutsize;
    float *m_optLut = (float*)lut3d->lut;

    for(long i=0; i<numPixels; ++i)
    {
        float newAlpha = (float)in[3];
        float c[3];
        c[0] = in[0];
        c[1] = in[1];
        c[2] = in[2];

        if (prelut->size) {
            float idx[3];

            idx[0] = (c[0] - prelut->min[0]) * prelut->scale[0];
            idx[1] = (c[1] - prelut->min[1]) * prelut->scale[1];
            idx[2] = (c[2] - prelut->min[2]) * prelut->scale[2];


            // NaNs become 0
            idx[0] = std::min(std::max(0.f, idx[0]), prelut_dimMinusOne);
            idx[1] = std::min(std::max(0.f, idx[1]), prelut_dimMinusOne);
            idx[2] = std::min(std::max(0.f, idx[2]), prelut_dimMinusOne);

            unsigned int lowIdx[3];
            lowIdx[0] = static_cast<unsigned int>(std::floor(idx[0]));
            lowIdx[1] = static_cast<unsigned int>(std::floor(idx[1]));
            lowIdx[2] = static_cast<unsigned int>(std::floor(idx[2]));

            // When the idx is exactly equal to an index (e.g. 0,1,2...)
            // then the computation of highIdx is wrong. However,
            // the delta is then equal to zero (e.g. lowIdx-idx),
            // so the highIdx has no impact.
            unsigned int highIdx[3];
            highIdx[0] = static_cast<unsigned int>(std::ceil(idx[0]));
            highIdx[1] = static_cast<unsigned int>(std::ceil(idx[1]));
            highIdx[2] = static_cast<unsigned int>(std::ceil(idx[2]));

            // Computing delta relative to high rather than lowIdx
            // to save computing (1-delta) below.
            float delta[3];
            delta[0] = (float)highIdx[0] - idx[0];
            delta[1] = (float)highIdx[1] - idx[1];
            delta[2] = (float)highIdx[2] - idx[2];

            c[0] = lerpf(preLutR[(unsigned int)highIdx[0]],
                         preLutR[(unsigned int)lowIdx[0]],
                                delta[0]);
            c[1] = lerpf(preLutG[(unsigned int)highIdx[1]],
                         preLutG[(unsigned int)lowIdx[1]],
                                delta[1]);
            c[2] = lerpf(preLutB[(unsigned int)highIdx[2]],
                         preLutB[(unsigned int)lowIdx[2]],
                                delta[2]);
        }
        {
            float idx[3];
            idx[0] = c[0] * scale_r;
            idx[1] = c[1] * scale_g;
            idx[2] = c[2] * scale_b;

             // NaNs become 0.
            idx[0] = Clamp(idx[0], 0.f, dimMinusOne);
            idx[1] = Clamp(idx[1], 0.f, dimMinusOne);
            idx[2] = Clamp(idx[2], 0.f, dimMinusOne);

            int indexLow[3];
            indexLow[0] = static_cast<int>(std::floor(idx[0]));
            indexLow[1] = static_cast<int>(std::floor(idx[1]));
            indexLow[2] = static_cast<int>(std::floor(idx[2]));

            int indexHigh[3];
            // When the idx is exactly equal to an index (e.g. 0,1,2...)
            // then the computation of highIdx is wrong. However,
            // the delta is then equal to zero (e.g. idx-lowIdx),
            // so the highIdx has no impact.
            indexHigh[0] = static_cast<int>(std::ceil(idx[0]));
            indexHigh[1] = static_cast<int>(std::ceil(idx[1]));
            indexHigh[2] = static_cast<int>(std::ceil(idx[2]));

            float fx = idx[0] - static_cast<float>(indexLow[0]);
            float fy = idx[1] - static_cast<float>(indexLow[1]);
            float fz = idx[2] - static_cast<float>(indexLow[2]);

            // Compute index into LUT for surrounding corners
            const int n000 =
                GetLut3DIndexBlueFast(indexLow[0], indexLow[1], indexLow[2],
                                    m_dim);
            const int n100 =
                GetLut3DIndexBlueFast(indexHigh[0], indexLow[1], indexLow[2],
                                    m_dim);
            const int n010 =
                GetLut3DIndexBlueFast(indexLow[0], indexHigh[1], indexLow[2],
                                    m_dim);
            const int n001 =
                GetLut3DIndexBlueFast(indexLow[0], indexLow[1], indexHigh[2],
                                    m_dim);
            const int n110 =
                GetLut3DIndexBlueFast(indexHigh[0], indexHigh[1], indexLow[2],
                                    m_dim);
            const int n101 =
                GetLut3DIndexBlueFast(indexHigh[0], indexLow[1], indexHigh[2],
                                    m_dim);
            const int n011 =
                GetLut3DIndexBlueFast(indexLow[0], indexHigh[1], indexHigh[2],
                                    m_dim);
            const int n111 =
                GetLut3DIndexBlueFast(indexHigh[0], indexHigh[1], indexHigh[2],
                                    m_dim);
            if (fx > fy) {
                if (fy > fz) {
                    out[0] =
                        (1 - fx)  * m_optLut[n000] +
                        (fx - fy) * m_optLut[n100] +
                        (fy - fz) * m_optLut[n110] +
                        (fz)      * m_optLut[n111];

                    out[1] =
                        (1 - fx)  * m_optLut[n000 + 1] +
                        (fx - fy) * m_optLut[n100 + 1] +
                        (fy - fz) * m_optLut[n110 + 1] +
                        (fz)      * m_optLut[n111 + 1];

                    out[2] =
                        (1 - fx)  * m_optLut[n000 + 2] +
                        (fx - fy) * m_optLut[n100 + 2] +
                        (fy - fz) * m_optLut[n110 + 2] +
                        (fz)      * m_optLut[n111 + 2];
                }
                else if (fx > fz)
                {
                    out[0] =
                        (1 - fx)  * m_optLut[n000] +
                        (fx - fz) * m_optLut[n100] +
                        (fz - fy) * m_optLut[n101] +
                        (fy)      * m_optLut[n111];

                    out[1] =
                        (1 - fx)  * m_optLut[n000 + 1] +
                        (fx - fz) * m_optLut[n100 + 1] +
                        (fz - fy) * m_optLut[n101 + 1] +
                        (fy)      * m_optLut[n111 + 1];

                    out[2] =
                        (1 - fx)  * m_optLut[n000 + 2] +
                        (fx - fz) * m_optLut[n100 + 2] +
                        (fz - fy) * m_optLut[n101 + 2] +
                        (fy)      * m_optLut[n111 + 2];
                }
                else
                {
                    out[0] =
                        (1 - fz)  * m_optLut[n000] +
                        (fz - fx) * m_optLut[n001] +
                        (fx - fy) * m_optLut[n101] +
                        (fy)      * m_optLut[n111];

                    out[1] =
                        (1 - fz)  * m_optLut[n000 + 1] +
                        (fz - fx) * m_optLut[n001 + 1] +
                        (fx - fy) * m_optLut[n101 + 1] +
                        (fy)      * m_optLut[n111 + 1];

                    out[2] =
                        (1 - fz)  * m_optLut[n000 + 2] +
                        (fz - fx) * m_optLut[n001 + 2] +
                        (fx - fy) * m_optLut[n101 + 2] +
                        (fy)      * m_optLut[n111 + 2];
                }
            }
            else
            {
                if (fz > fy)
                {
                    out[0] =
                        (1 - fz)  * m_optLut[n000] +
                        (fz - fy) * m_optLut[n001] +
                        (fy - fx) * m_optLut[n011] +
                        (fx)      * m_optLut[n111];

                    out[1] =
                        (1 - fz)  * m_optLut[n000 + 1] +
                        (fz - fy) * m_optLut[n001 + 1] +
                        (fy - fx) * m_optLut[n011 + 1] +
                        (fx)      * m_optLut[n111 + 1];

                    out[2] =
                        (1 - fz)  * m_optLut[n000 + 2] +
                        (fz - fy) * m_optLut[n001 + 2] +
                        (fy - fx) * m_optLut[n011 + 2] +
                        (fx)      * m_optLut[n111 + 2];
                }
                else if (fz > fx)
                {
                    out[0] =
                        (1 - fy)  * m_optLut[n000] +
                        (fy - fz) * m_optLut[n010] +
                        (fz - fx) * m_optLut[n011] +
                        (fx)      * m_optLut[n111];

                    out[1] =
                        (1 - fy)  * m_optLut[n000 + 1] +
                        (fy - fz) * m_optLut[n010 + 1] +
                        (fz - fx) * m_optLut[n011 + 1] +
                        (fx)      * m_optLut[n111 + 1];

                    out[2] =
                        (1 - fy)  * m_optLut[n000 + 2] +
                        (fy - fz) * m_optLut[n010 + 2] +
                        (fz - fx) * m_optLut[n011 + 2] +
                        (fx)      * m_optLut[n111 + 2];
                }
                else
                {
                    out[0] =
                        (1 - fy)  * m_optLut[n000] +
                        (fy - fx) * m_optLut[n010] +
                        (fx - fz) * m_optLut[n110] +
                        (fz)      * m_optLut[n111];

                    out[1] =
                        (1 - fy)  * m_optLut[n000 + 1] +
                        (fy - fx) * m_optLut[n010 + 1] +
                        (fx - fz) * m_optLut[n110 + 1] +
                        (fz)      * m_optLut[n111 + 1];

                    out[2] =
                        (1 - fy)  * m_optLut[n000 + 2] +
                        (fy - fx) * m_optLut[n010 + 2] +
                        (fx - fz) * m_optLut[n110 + 2] +
                        (fz)      * m_optLut[n111 + 2];
                }
            }
        }
        out[3] = newAlpha;
        in  += 4;
        out += 4;
    }

    return 0;
}

//----------------------------------------------------------------------------
// RGB channel ordering.
// Pixels ordered in such a way that the blue coordinate changes fastest,
// then the green coordinate, and finally, the red coordinate changes slowest
//
inline __m128i GetLut3DIndices(const __m128i &idxR,
                               const __m128i &idxG,
                               const __m128i &idxB,
                               const __m128i /*&sizesR*/,
                               const __m128i &sizesG,
                               const __m128i &sizesB)
{
    // SSE2 doesn't have 4-way multiplication for integer registers, so we need
    // split them into two register and multiply-add them separately, and then
    // combine the results.

    // r02 = { sizesG * idxR0, -, sizesG * idxR2, - }
    // r13 = { sizesG * idxR1, -, sizesG * idxR3, - }
    __m128i r02 = _mm_mul_epu32(sizesG, idxR);
    __m128i r13 = _mm_mul_epu32(sizesG, _mm_srli_si128(idxR,4));

    // r02 = { idxG0 + sizesG * idxR0, -, idxG2 + sizesG * idxR2, - }
    // r13 = { idxG1 + sizesG * idxR1, -, idxG3 + sizesG * idxR3, - }
    r02 = _mm_add_epi32(idxG, r02);
    r13 = _mm_add_epi32(_mm_srli_si128(idxG,4), r13);

    // r02 = { sizesB * (idxG0 + sizesG * idxR0), -, sizesB * (idxG2 + sizesG * idxR2), - }
    // r13 = { sizesB * (idxG1 + sizesG * idxR1), -, sizesB * (idxG3 + sizesG * idxR3), - }
    r02 = _mm_mul_epu32(sizesB, r02);
    r13 = _mm_mul_epu32(sizesB, r13);

    // r02 = { idxB0 + sizesB * (idxG0 + sizesG * idxR0), -, idxB2 + sizesB * (idxG2 + sizesG * idxR2), - }
    // r13 = { idxB1 + sizesB * (idxG1 + sizesG * idxR1), -, idxB3 + sizesB * (idxG3 + sizesG * idxR3), - }
    r02 = _mm_add_epi32(idxB, r02);
    r13 = _mm_add_epi32(_mm_srli_si128(idxB,4), r13);

    // r = { idxB0 + sizesB * (idxG0 + sizesG * idxR0),
    //       idxB1 + sizesB * (idxG1 + sizesG * idxR1),
    //       idxB2 + sizesB * (idxG2 + sizesG * idxR2),
    //       idxB3 + sizesB * (idxG3 + sizesG * idxR3) }
    __m128i r = _mm_unpacklo_epi32(_mm_shuffle_epi32(r02, _MM_SHUFFLE(0,0,2,0)),
                                   _mm_shuffle_epi32(r13, _MM_SHUFFLE(0,0,2,0)));

    // return { 4 * (idxB0 + sizesB * (idxG0 + sizesG * idxR0)),
    //          4 * (idxB1 + sizesB * (idxG1 + sizesG * idxR1)),
    //          4 * (idxB2 + sizesB * (idxG2 + sizesG * idxR2)),
    //          4 * (idxB3 + sizesB * (idxG3 + sizesG * idxR3)) }
    return _mm_slli_epi32(r, 2);
}

inline void LookupNearest4(float* optLut,
                           const __m128i &rIndices,
                           const __m128i &gIndices,
                           const __m128i &bIndices,
                           const __m128i &dim,
                           __m128 res[4])
{
    OCIO_ALIGN(int offsetInt[4]);
    __m128i offsets = GetLut3DIndices(rIndices, gIndices, bIndices, dim, dim, dim);
    _mm_store_si128((__m128i *)offsetInt, offsets);
    // int* offsetInt = (int*)&offsets;

    res[0] = _mm_loadu_ps(optLut + offsetInt[0]);
    res[1] = _mm_loadu_ps(optLut + offsetInt[1]);
    res[2] = _mm_loadu_ps(optLut + offsetInt[2]);
    res[3] = _mm_loadu_ps(optLut + offsetInt[3]);
}

int apply_lut_ocio_sse2_rgba(const LUT3DContext *lut3d, const FloatImageRGBA *src_image, FloatImageRGBA *dst_image)
{
    const Lut3DPreLut *prelut = &lut3d->prelut;

    int numPixels = src_image->width * src_image->height;

    float *in  = src_image->data;
    float *out = dst_image->data;

    const float prelut_dimMinusOne = (float)(prelut->size - 1);

    const float *preLutR = prelut->lut[0];
    const float *preLutG = prelut->lut[1];
    const float *preLutB = prelut->lut[2];

    float dimMinusOne = (float)(lut3d->lutsize - 1);
    float scale_r = lut3d->scale.r * dimMinusOne;
    float scale_g = lut3d->scale.g * dimMinusOne;
    float scale_b = lut3d->scale.b * dimMinusOne;

    long m_dim  = lut3d->lutsize;

    float *m_optLut = (float*)lut3d->rgba_lut;

    // __m128 step = _mm_set_ps(1.0f, this->m_step, this->m_step, this->m_step);
    __m128 pre_dimMinusOne = _mm_set1_ps(prelut_dimMinusOne);

    __m128 prelut_scale = _mm_set_ps(1.0f, prelut->scale[2], prelut->scale[1], prelut->scale[0]);
    __m128 prelut_min   = _mm_set_ps(0.0f,   prelut->min[2],   prelut->min[1],   prelut->min[0]);
    __m128 EZERO = _mm_setzero_ps();
    __m128 EONE  = _mm_set1_ps(1.0f);

    __m128 step   = _mm_set_ps(1.0f, scale_b, scale_g, scale_r);
    __m128 maxIdx = _mm_set1_ps((float)(m_dim - 1));
    __m128i dim   = _mm_set1_epi32(m_dim);

    __m128 v[4];
    OCIO_ALIGN(int cmpDelta[4]);
    OCIO_ALIGN(float c[4]);

    for(long i=0; i<numPixels; ++i)
    {
        c[0] = in[0];
        c[1] = in[1];
        c[2] = in[2];
        c[3] = in[3];
        if (prelut->size) {
            __m128 color = _mm_set_ps(c[3], c[2], c[1], c[0]);

            __m128 idx = _mm_mul_ps(_mm_sub_ps(color, prelut_min), prelut_scale);
            // __m128 idx
            //     = _mm_mul_ps(_mm_set_ps(c[3], c[2], c[1], c[0]), step);

            // _mm_max_ps => NaNs become 0
            idx = _mm_min_ps(_mm_max_ps(idx, EZERO), pre_dimMinusOne);

            // zero < std::floor(idx) < maxIdx
            // SSE => zero < truncate(idx) < maxIdx
            //
            __m128 lIdx = _mm_cvtepi32_ps(_mm_cvttps_epi32(idx));

            // zero < std::ceil(idx) < maxIdx
            // SSE => (lowIdx (already truncated) + 1) < maxIdx
            // then clamp to prevent hIdx from falling off the end
            // of the LUT
            __m128 hIdx = _mm_min_ps(_mm_add_ps(lIdx, EONE), pre_dimMinusOne);

            // Computing delta relative to high rather than lowIdx
            // to save computing (1-delta) below.
            __m128 d = _mm_sub_ps(hIdx, idx);

            OCIO_ALIGN(float delta[4]);   _mm_store_ps(delta, d);
            OCIO_ALIGN(float lowIdx[4]);  _mm_store_ps(lowIdx, lIdx);
            OCIO_ALIGN(float highIdx[4]); _mm_store_ps(highIdx, hIdx);

            c[0] = lerpf(preLutR[(unsigned int)highIdx[0]],
                         preLutR[(unsigned int)lowIdx[0]],
                                delta[0]);
            c[1] = lerpf(preLutG[(unsigned int)highIdx[1]],
                         preLutG[(unsigned int)lowIdx[1]],
                                delta[1]);
            c[2] = lerpf(preLutB[(unsigned int)highIdx[2]],
                         preLutB[(unsigned int)lowIdx[2]],
                                delta[2]);
        }

        {
            __m128 data = _mm_set_ps(c[3], c[2], c[1], c[0]);

            __m128 idx = _mm_mul_ps(data, step);

            idx = _mm_max_ps(idx, EZERO);  // NaNs become 0
            idx = _mm_min_ps(idx, maxIdx);

            // lowIdxInt32 = floor(idx),
            // with lowIdx in [0, maxIdx]
            __m128i lowIdxInt32 = _mm_cvttps_epi32(idx);
            __m128 lowIdx = _mm_cvtepi32_ps(lowIdxInt32);

            // highIdxInt32 = ceil(idx), with highIdx in [1, maxIdx]

            __m128i highIdxInt32 = _mm_sub_epi32(lowIdxInt32,
                _mm_castps_si128(_mm_cmplt_ps(lowIdx, maxIdx)));

            __m128 delta = _mm_sub_ps(idx, lowIdx); // d_r, d_g, d_b, d_a
            __m128 delta0 = _mm_shuffle_ps(delta, delta, _MM_SHUFFLE(0, 0, 0, 0)); // d_r
            __m128 delta1 = _mm_shuffle_ps(delta, delta, _MM_SHUFFLE(1, 1, 1, 1)); // d_g
            __m128 delta2 = _mm_shuffle_ps(delta, delta, _MM_SHUFFLE(2, 2, 2, 2)); // d_b

            // lh01 = {L0, H0, L1, H1}
            // lh23 = {L2, H2, L3, H3}, L3 and H3 are not used
            __m128i lh01 = _mm_unpacklo_epi32(lowIdxInt32, highIdxInt32); // prev_r, next_r, prev_g, next_g
            __m128i lh23 = _mm_unpackhi_epi32(lowIdxInt32, highIdxInt32); // prev_b, next_b, prev_a, next_a

            // Since the cube is split along the main diagonal, the lowest corner
            // and highest corner are always used.
            // v[0] = { L0, L1, L2 }
            // v[3] = { H0, H1, H2 }

            __m128i idxR, idxG, idxB;
            // Store vertices transposed on idxR, idxG and idxB:
            // idxR = { v0r, v1r, v2r, v3r }
            // idxG = { v0g, v1g, v2g, v3g }
            // idxB = { v0b, v1b, v2b, v3b }

            // Vertices differences (vi-vj) to be multiplied by the delta factors
            __m128 dv0, dv1, dv2;

            // In tetrahedral interpolation, the cube is divided along the main
            // diagonal into 6 tetrahedra.  We compare the relative fractional
            // position within the cube (deltaArray) to know which tetrahedron
            // we are in and therefore which four vertices of the cube we need.
            //                     r >= g                g >= b                 b >= r
            // cmpDelta = { delta[0] >= delta[1], delta[1] >= delta[2], delta[2] >= delta[0], - }

            _mm_store_ps((float*)cmpDelta,
                        _mm_cmpgt_ps(delta,
                                    _mm_shuffle_ps(delta,
                                                   delta,
                                                   _MM_SHUFFLE(0, 0, 2, 1))));

            if (cmpDelta[0])  // delta[0] > delta[1] r > g
            {
                if (cmpDelta[1])  // delta[1] > delta[2] g > b
                {
                    // R > G > B

                    // v[1] = { H0, L1, L2 } c100
                    // v[2] = { H0, H1, L2 } c110

                    // idxR = { L0, H0, H0, H0 }
                    // idxG = { L1, L1, H1, H1 }
                    // idxB = { L2, L2, L2, H2 }
                    idxR = _mm_shuffle_epi32(lh01, _MM_SHUFFLE(1, 1, 1, 0));
                    idxG = _mm_shuffle_epi32(lh01, _MM_SHUFFLE(3, 3, 2, 2));
                    idxB = _mm_shuffle_epi32(lh23, _MM_SHUFFLE(1, 0, 0, 0));

                    LookupNearest4(m_optLut, idxR, idxG, idxB, dim, v);

                    // Order: R G B => 0 1 2
                    dv0 = _mm_sub_ps(v[1], v[0]);
                    dv1 = _mm_sub_ps(v[2], v[1]);
                    dv2 = _mm_sub_ps(v[3], v[2]);
                }
                else if (!cmpDelta[2])  // delta[0] > delta[2] r > b  || !b > r
                {
                    // R > B > G

                    // v[1] = { H0, L1, L2 } c100
                    // v[2] = { H0, L1, H2 } c101

                    // idxR = { L0, H0, H0, H0 }
                    // idxG = { L1, L1, L1, H1 }
                    // idxB = { L2, L2, H2, H2 }
                    idxR = _mm_shuffle_epi32(lh01, _MM_SHUFFLE(1, 1, 1, 0));
                    idxG = _mm_shuffle_epi32(lh01, _MM_SHUFFLE(3, 2, 2, 2));
                    idxB = _mm_shuffle_epi32(lh23, _MM_SHUFFLE(1, 1, 0, 0));

                    LookupNearest4(m_optLut, idxR, idxG, idxB, dim, v);

                    // Order: R B G => 0 2 1
                    dv0 = _mm_sub_ps(v[1], v[0]);
                    dv2 = _mm_sub_ps(v[2], v[1]);
                    dv1 = _mm_sub_ps(v[3], v[2]);
                }
                else
                {
                    // B > R > G

                    // v[1] = { L0, L1, H2 } c001
                    // v[2] = { H0, L1, H2 } c101

                    // idxR = { L0, L0, H0, H0 }
                    // idxG = { L1, L1, L1, H1 }
                    // idxB = { L2, H2, H2, H2 }
                    idxR = _mm_shuffle_epi32(lh01, _MM_SHUFFLE(1, 1, 0, 0));
                    idxG = _mm_shuffle_epi32(lh01, _MM_SHUFFLE(3, 2, 2, 2));
                    idxB = _mm_shuffle_epi32(lh23, _MM_SHUFFLE(1, 1, 1, 0));

                    LookupNearest4(m_optLut, idxR, idxG, idxB, dim, v);

                    // Order: B R G => 2 0 1
                    dv2 = _mm_sub_ps(v[1], v[0]);
                    dv0 = _mm_sub_ps(v[2], v[1]);
                    dv1 = _mm_sub_ps(v[3], v[2]);
                }
            }
            else
            {
                if (!cmpDelta[1])  // delta[2] > delta[1]  b > g ||  !g > b
                {
                    // B > G > R

                    // v[1] = { L0, L1, H2 }
                    // v[2] = { L0, H1, H2 }

                    // idxR = { L0, L0, L0, H0 }
                    // idxG = { L1, L1, H1, H1 }
                    // idxB = { L2, H2, H2, H2 }
                    idxR = _mm_shuffle_epi32(lh01, _MM_SHUFFLE(1, 0, 0, 0));
                    idxG = _mm_shuffle_epi32(lh01, _MM_SHUFFLE(3, 3, 2, 2));
                    idxB = _mm_shuffle_epi32(lh23, _MM_SHUFFLE(1, 1, 1, 0));

                    LookupNearest4(m_optLut, idxR, idxG, idxB, dim, v);

                    // Order: B G R => 2 1 0
                    dv2 = _mm_sub_ps(v[1], v[0]);
                    dv1 = _mm_sub_ps(v[2], v[1]);
                    dv0 = _mm_sub_ps(v[3], v[2]);
                }
                else if (!cmpDelta[2])  // delta[0] > delta[2] r > b || !b > r
                {
                    // G > R > B

                    // v[1] = { L0, H1, L2 }
                    // v[2] = { H0, H1, L2 }

                    // idxR = { L0, L0, H0, H0 }
                    // idxG = { L1, H1, H1, H1 }
                    // idxB = { L2, L2, L2, H2 }
                    idxR = _mm_shuffle_epi32(lh01, _MM_SHUFFLE(1, 1, 0, 0));
                    idxG = _mm_shuffle_epi32(lh01, _MM_SHUFFLE(3, 3, 3, 2));
                    idxB = _mm_shuffle_epi32(lh23, _MM_SHUFFLE(1, 0, 0, 0));

                    LookupNearest4(m_optLut, idxR, idxG, idxB, dim, v);

                    // Order: G R B => 1 0 2
                    dv1 = _mm_sub_ps(v[1], v[0]);
                    dv0 = _mm_sub_ps(v[2], v[1]);
                    dv2 = _mm_sub_ps(v[3], v[2]);
                }
                else
                {
                    // G > B > R

                    // v[1] = { L0, H1, L2 }
                    // v[2] = { L0, H1, H2 }

                    // idxR = { L0, L0, L0, H0 }
                    // idxG = { L1, H1, H1, H1 }
                    // idxB = { L2, L2, H2, H2 }
                    idxR = _mm_shuffle_epi32(lh01, _MM_SHUFFLE(1, 0, 0, 0));
                    idxG = _mm_shuffle_epi32(lh01, _MM_SHUFFLE(3, 3, 3, 2));
                    idxB = _mm_shuffle_epi32(lh23, _MM_SHUFFLE(1, 1, 0, 0));

                    LookupNearest4(m_optLut, idxR, idxG, idxB, dim, v);

                    // Order: G B R => 1 2 0
                    dv1 = _mm_sub_ps(v[1], v[0]);
                    dv2 = _mm_sub_ps(v[2], v[1]);
                    dv0 = _mm_sub_ps(v[3], v[2]);
                }
            }

            __m128 result = _mm_add_ps(_mm_add_ps(v[0], _mm_mul_ps(delta0, dv0)),
                _mm_add_ps(_mm_mul_ps(delta1, dv1), _mm_mul_ps(delta2, dv2)));
            _mm_storeu_ps(out, result);

        }

        out[3] = in[3];
        in  += 4;
        out += 4;

    }
    return 0;
}