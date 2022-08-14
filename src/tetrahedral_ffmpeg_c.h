// this code is extracted from ffmpeg lut3d.c
#ifndef TETRAHEDRAL_FFMPEG_C_H
#define TETRAHEDRAL_FFMPEG_C_H

#include "libavutil_common.h"

#define PREV(x) ((int)(x))
#define NEXT(x) (FFMIN((int)(x) + 1, lut3d->lutsize - 1))

static inline float prelut_interp_1d_linear(const Lut3DPreLut *prelut,
                                            int idx, const float s)
{
    const int lut_max = prelut->size - 1;
    const float scaled = (s - prelut->min[idx]) * prelut->scale[idx];
    const float x = av_clipf(scaled, 0.0f, (float)lut_max);
    const int prev = PREV(x);
    const int next = FFMIN((int)(x) + 1, lut_max);
    const float p = prelut->lut[idx][prev];
    const float n = prelut->lut[idx][next];
    const float d = x - (float)prev;
    return lerpf(p, n, d);
}

static inline struct rgbvec apply_prelut(const Lut3DPreLut *prelut,
                                         const struct rgbvec *s)
{
    struct rgbvec c;

    if (prelut->size <= 0)
        return *s;

    c.r = prelut_interp_1d_linear(prelut, 0, s->r);
    c.g = prelut_interp_1d_linear(prelut, 1, s->g);
    c.b = prelut_interp_1d_linear(prelut, 2, s->b);
    return c;
}

/**
 * Tetrahedral interpolation. Based on code found in Truelight Software Library paper.
 * @see http://www.filmlight.ltd.uk/pdf/whitepapers/FL-TL-TN-0057-SoftwareLib.pdf
 */
static inline struct rgbvec interp_tetrahedral(const LUT3DContext *lut3d,
                                               const struct rgbvec *s)
{
    const int lutsize2 = lut3d->lutsize2;
    const int lutsize  = lut3d->lutsize;
    const int prev[] = {PREV(s->r), PREV(s->g), PREV(s->b)};
    const int next[] = {NEXT(s->r), NEXT(s->g), NEXT(s->b)};
    const struct rgbvec d = {s->r - prev[0], s->g - prev[1], s->b - prev[2]};
    const struct rgbvec c000 = lut3d->lut[prev[0] * lutsize2 + prev[1] * lutsize + prev[2]];
    const struct rgbvec c111 = lut3d->lut[next[0] * lutsize2 + next[1] * lutsize + next[2]];
    struct rgbvec c;
    if (d.r > d.g) {
        if (d.g > d.b) {
            const struct rgbvec c100 = lut3d->lut[next[0] * lutsize2 + prev[1] * lutsize + prev[2]];
            const struct rgbvec c110 = lut3d->lut[next[0] * lutsize2 + next[1] * lutsize + prev[2]];
            c.r = (1-d.r) * c000.r + (d.r-d.g) * c100.r + (d.g-d.b) * c110.r + (d.b) * c111.r;
            c.g = (1-d.r) * c000.g + (d.r-d.g) * c100.g + (d.g-d.b) * c110.g + (d.b) * c111.g;
            c.b = (1-d.r) * c000.b + (d.r-d.g) * c100.b + (d.g-d.b) * c110.b + (d.b) * c111.b;
        } else if (d.r > d.b) {
            const struct rgbvec c100 = lut3d->lut[next[0] * lutsize2 + prev[1] * lutsize + prev[2]];
            const struct rgbvec c101 = lut3d->lut[next[0] * lutsize2 + prev[1] * lutsize + next[2]];
            c.r = (1-d.r) * c000.r + (d.r-d.b) * c100.r + (d.b-d.g) * c101.r + (d.g) * c111.r;
            c.g = (1-d.r) * c000.g + (d.r-d.b) * c100.g + (d.b-d.g) * c101.g + (d.g) * c111.g;
            c.b = (1-d.r) * c000.b + (d.r-d.b) * c100.b + (d.b-d.g) * c101.b + (d.g) * c111.b;
        } else {
            const struct rgbvec c001 = lut3d->lut[prev[0] * lutsize2 + prev[1] * lutsize + next[2]];
            const struct rgbvec c101 = lut3d->lut[next[0] * lutsize2 + prev[1] * lutsize + next[2]];
            c.r = (1-d.b) * c000.r + (d.b-d.r) * c001.r + (d.r-d.g) * c101.r + (d.g) * c111.r;
            c.g = (1-d.b) * c000.g + (d.b-d.r) * c001.g + (d.r-d.g) * c101.g + (d.g) * c111.g;
            c.b = (1-d.b) * c000.b + (d.b-d.r) * c001.b + (d.r-d.g) * c101.b + (d.g) * c111.b;
        }
    } else {
        if (d.b > d.g) {
            const struct rgbvec c001 = lut3d->lut[prev[0] * lutsize2 + prev[1] * lutsize + next[2]];
            const struct rgbvec c011 = lut3d->lut[prev[0] * lutsize2 + next[1] * lutsize + next[2]];
            c.r = (1-d.b) * c000.r + (d.b-d.g) * c001.r + (d.g-d.r) * c011.r + (d.r) * c111.r;
            c.g = (1-d.b) * c000.g + (d.b-d.g) * c001.g + (d.g-d.r) * c011.g + (d.r) * c111.g;
            c.b = (1-d.b) * c000.b + (d.b-d.g) * c001.b + (d.g-d.r) * c011.b + (d.r) * c111.b;
        } else if (d.b > d.r) {
            const struct rgbvec c010 = lut3d->lut[prev[0] * lutsize2 + next[1] * lutsize + prev[2]];
            const struct rgbvec c011 = lut3d->lut[prev[0] * lutsize2 + next[1] * lutsize + next[2]];
            c.r = (1-d.g) * c000.r + (d.g-d.b) * c010.r + (d.b-d.r) * c011.r + (d.r) * c111.r;
            c.g = (1-d.g) * c000.g + (d.g-d.b) * c010.g + (d.b-d.r) * c011.g + (d.r) * c111.g;
            c.b = (1-d.g) * c000.b + (d.g-d.b) * c010.b + (d.b-d.r) * c011.b + (d.r) * c111.b;
        } else {
            const struct rgbvec c010 = lut3d->lut[prev[0] * lutsize2 + next[1] * lutsize + prev[2]];
            const struct rgbvec c110 = lut3d->lut[next[0] * lutsize2 + next[1] * lutsize + prev[2]];
            c.r = (1-d.g) * c000.r + (d.g-d.r) * c010.r + (d.r-d.b) * c110.r + (d.b) * c111.r;
            c.g = (1-d.g) * c000.g + (d.g-d.r) * c010.g + (d.r-d.b) * c110.g + (d.b) * c111.g;
            c.b = (1-d.g) * c000.b + (d.g-d.r) * c010.b + (d.r-d.b) * c110.b + (d.b) * c111.b;
        }
    }
    return c;
}

static int apply_lut_c(const LUT3DContext *lut3d, const FloatImage *src_image, FloatImage *dst_image)
{
    const Lut3DPreLut *prelut = &lut3d->prelut;
    float lut_max = (float)lut3d->lutsize - 1;
    float scale_r = lut3d->scale.r * lut_max;
    float scale_g = lut3d->scale.g * lut_max;
    float scale_b = lut3d->scale.b * lut_max;

    for (int y = 0; y < src_image->height; y++) {

        float *srcr = src_image->data[0] + y * src_image->stride;
        float *srcg = src_image->data[1] + y * src_image->stride;
        float *srcb = src_image->data[2] + y * src_image->stride;

        float *dstr = dst_image->data[0] + y * dst_image->stride;
        float *dstg = dst_image->data[1] + y * dst_image->stride;
        float *dstb = dst_image->data[2] + y * dst_image->stride;

        for (int x = 0; x < src_image->width; x++) {



            rgbvec rgb = {sanitizef(srcr[x]),
                          sanitizef(srcg[x]),
                          sanitizef(srcb[x])};


            rgbvec prelut_rgb = apply_prelut(prelut, &rgb);

            rgbvec scaled_rgb = {av_clipf(prelut_rgb.r * scale_r, 0, lut_max),
                                 av_clipf(prelut_rgb.g * scale_g, 0, lut_max),
                                 av_clipf(prelut_rgb.b * scale_b, 0, lut_max)};

            rgbvec result = interp_tetrahedral(lut3d, &scaled_rgb);
            dstr[x] = result.r;
            dstg[x] = result.g;
            dstb[x] = result.b;
        }
    }

    return 0;
}

#endif /* TETRAHEDRAL_FFMPEG_C_H */
