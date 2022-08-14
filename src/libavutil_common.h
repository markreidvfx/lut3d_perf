// utilities need by ffmpeg code
#ifndef LIBAVUTIL_COMMON_H
#define LIBAVUTIL_COMMON_H

#define FLT_MAX          3.402823466e+38F        // max value
#define FFMAX(a,b) ((a) > (b) ? (a) : (b))
#define FFMIN(a,b) ((a) > (b) ? (b) : (a))

#define AV_LOG_ERROR 1
#define AVERROR_PATCHWELCOME -1
#define AVERROR_INVALIDDATA -1

#define AVERROR(value) -1

#define av_sscanf sscanf
#define av_malloc malloc

#define av_assert0 assert

#define av_log(ctx, level, fmt, ...) printf(fmt, ##__VA_ARGS__ )


#define EXPONENT_MASK 0x7F800000
#define MANTISSA_MASK 0x007FFFFF
#define SIGN_MASK     0x80000000

typedef struct AVFrame {
    #define AV_NUM_DATA_POINTERS 8
    uint8_t *data[AV_NUM_DATA_POINTERS];
    int linesize[AV_NUM_DATA_POINTERS];
    uint8_t **extended_data;
    int width, height;
} AVFrame;

union av_intfloat32 {
    uint32_t i;
    float    f;
};

static inline float lerpf(float v0, float v1, float f)
{
    return v0 + (v1 - v0) * f;
}

static inline float sanitizef(float f)
{
    union av_intfloat32 t;
    t.f = f;

    if ((t.i & EXPONENT_MASK) == EXPONENT_MASK) {
        if ((t.i & MANTISSA_MASK) != 0) {
            // NAN
            return 0.0f;
        } else if (t.i & SIGN_MASK) {
            // -INF
            return -FLT_MAX;
        } else {
            // +INF
            return FLT_MAX;
        }
    }
    return f;
}


static float av_clipf(float a, float amin, float amax)
{
    if      (a < amin) return amin;
    else if (a > amax) return amax;
    else               return a;
}

#endif /* LIBAVUTIL_COMMON_H */