// this code is adapted from ffmpeg lut3d.c
#ifndef TETRAHEDRAL_FFMPEG_ASM_H
#define TETRAHEDRAL_FFMPEG_ASM_H

#include "libavutil_common.h"

#define DEFINE_INTERP_FUNC(name, format, opt)                                                                                                                             \
void ff_interp_##name##_##format##_##opt(const LUT3DContext *lut3d, const Lut3DPreLut *prelut, AVFrame *src, AVFrame *dst, int slice_start, int slice_end, int has_alpha); \
static int apply_lut_##opt##_asm(const LUT3DContext *lut3d, const FloatImage *src_image, FloatImage *dst_image)        \
{                                                                                                                      \
    const Lut3DPreLut *prelut = &lut3d->prelut;                                                                        \
    if (!prelut->size)                                                                                                 \
        prelut = NULL;                                                                                                 \
    AVFrame in = {0};                                                                                                  \
    AVFrame out = {0};                                                                                                 \
    /* gbra pixel order */                                                                                             \
    in.data[0] = (uint8_t*)src_image->data[1];                                                                         \
    in.data[1] = (uint8_t*)src_image->data[2];                                                                         \
    in.data[2] = (uint8_t*)src_image->data[0];                                                                         \
    in.data[3] = (uint8_t*)src_image->data[3];                                                                         \
    in.linesize[0] = src_image->stride * 4;                                                                            \
    in.linesize[1] = src_image->stride * 4;                                                                            \
    in.linesize[2] = src_image->stride * 4;                                                                            \
    in.linesize[3] = src_image->stride * 4;                                                                            \
    in.width  = src_image->width;                                                                                      \
    in.height = src_image->height;                                                                                     \
    /* gbra pixel order */                                                                                             \
    out.data[0] = (uint8_t*)dst_image->data[1];                                                                        \
    out.data[1] = (uint8_t*)dst_image->data[2];                                                                        \
    out.data[2] = (uint8_t*)dst_image->data[0];                                                                        \
    out.data[3] = (uint8_t*)dst_image->data[3];                                                                        \
    out.linesize[0] = dst_image->stride * 4;                                                                           \
    out.linesize[1] = dst_image->stride * 4;                                                                           \
    out.linesize[2] = dst_image->stride * 4;                                                                           \
    out.linesize[3] = dst_image->stride * 4;                                                                           \
    out.width  = dst_image->width;                                                                                     \
    out.height = dst_image->height;                                                                                    \
    int slice_start = 0;                                                                                               \
    int slice_end = in.height;                                                                                         \
    int has_alpha = 0;                                                                                                 \
    ff_interp_##name##_##format##_##opt(lut3d, prelut, &in, &out, slice_start, slice_end, has_alpha);                  \
    return 0;                                                                                                          \
}

DEFINE_INTERP_FUNC(tetrahedral, pf32, avx2);
DEFINE_INTERP_FUNC(tetrahedral, pf32, avx)
DEFINE_INTERP_FUNC(tetrahedral, pf32, sse2)

#endif /* TETRAHEDRAL_FFMPEG_ASM_H */