// this lut reading code is from ffmpeg lut3d.c
#include "lut3d_perf.h"
#include "libavutil_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void inline av_freep(void * arg)
{
    void *val;

    memcpy(&val, arg, sizeof(val));
    memcpy(arg, &(void *){ NULL }, sizeof(val));
    free(val);
}

void inline *av_malloc_array(size_t nmemb, size_t size)
{
    size_t result = nmemb * size;
    return av_malloc(result);
}

static inline int allocate_3dlut(LUT3DContext *lut3d, int lutsize, int prelut)
{
    void *ctx = lut3d;
    int i;
    if (lutsize < 2 || lutsize > MAX_LEVEL) {
        av_log(ctx, AV_LOG_ERROR, "Too large or invalid 3D LUT size\n");
        return AVERROR(EINVAL);
    }

    av_freep(&lut3d->lut);
    lut3d->lut = (rgbvec*)av_malloc_array(lutsize * lutsize * lutsize, sizeof(*lut3d->lut));
    if (!lut3d->lut)
        return AVERROR(ENOMEM);

    // for ocio SSE2
    lut3d->rgba_lut = (rgbavec*)av_malloc_array(lutsize * lutsize * lutsize, sizeof(*lut3d->rgba_lut));
    if (!lut3d->rgba_lut)
        return AVERROR(ENOMEM);

    if (prelut) {
        lut3d->prelut.size = PRELUT_SIZE;
        for (i = 0; i < 3; i++) {
            av_freep(&lut3d->prelut.lut[i]);
            lut3d->prelut.lut[i] = (float*)av_malloc_array(PRELUT_SIZE, sizeof(*lut3d->prelut.lut[0]));
            if (!lut3d->prelut.lut[i])
                return AVERROR(ENOMEM);
        }
    } else {
        lut3d->prelut.size = 0;
        for (i = 0; i < 3; i++) {
            av_freep(&lut3d->prelut.lut[i]);
        }
    }
    lut3d->lutsize = lutsize;
    lut3d->lutsize2 = lutsize * lutsize;
    return 0;
}

#define NEXT_LINE(loop_cond) do {                           \
    if (!fgets(line, sizeof(line), f)) {                    \
        av_log(ctx, AV_LOG_ERROR, "Unexpected EOF\n");      \
        return AVERROR_INVALIDDATA;                         \
    }                                                       \
} while (loop_cond)

#define NEXT_LINE_OR_GOTO(loop_cond, label) do {            \
    if (!fgets(line, sizeof(line), f)) {                    \
        av_log(ctx, AV_LOG_ERROR, "Unexpected EOF\n");      \
        ret = AVERROR_INVALIDDATA;                          \
        goto label;                                         \
    }                                                       \
} while (loop_cond)


#define NEXT_FLOAT_OR_GOTO(value, label)                    \
    if (!fget_next_word(line, sizeof(line) ,f)) {           \
        ret = AVERROR_INVALIDDATA;                          \
        goto label;                                         \
    }                                                       \
    if (av_sscanf(line, "%f", &value) != 1) {               \
        ret = AVERROR_INVALIDDATA;                          \
        goto label;                                         \
    }

#define MAX_LINE_SIZE 512

static int av_isspace(int c)
{
    return c == ' ' || c == '\f' || c == '\n' || c == '\r' || c == '\t' ||
           c == '\v';
}

static int skip_line(const char *p)
{
    while (*p && av_isspace(*p))
        p++;
    return !*p || *p == '#';
}

static char* fget_next_word(char* dst, int max, FILE* f)
{
    int c;
    char *p = dst;

    /* for null */
    max--;
    /* skip until next non whitespace char */
    while ((c = fgetc(f)) != EOF) {
        if (av_isspace(c))
            continue;

        *p++ = c;
        max--;
        break;
    }

    /* get max bytes or up until next whitespace char */
    for (; max > 0; max--) {
        if ((c = fgetc(f)) == EOF)
            break;

        if (av_isspace(c))
            break;

        *p++ = c;
    }

    *p = 0;
    if (p == dst)
        return NULL;
    return p;
}

static int nearest_sample_index(float *data, float x, int low, int hi)
{
    int mid;
    if (x < data[low])
        return low;

    if (x > data[hi])
        return hi;

    for (;;) {
        av_assert0(x >= data[low]);
        av_assert0(x <= data[hi]);
        av_assert0((hi-low) > 0);

        if (hi - low == 1)
            return low;

        mid = (low + hi) / 2;

        if (x < data[mid])
            hi = mid;
        else
            low = mid;
    }

    return 0;
}

static int parse_cinespace(LUT3DContext *lut3d, FILE *f)
{
    char line[MAX_LINE_SIZE];

    void *ctx = lut3d;
    float in_min[3]  = {0.0, 0.0, 0.0};
    float in_max[3]  = {1.0, 1.0, 1.0};
    float out_min[3] = {0.0, 0.0, 0.0};
    float out_max[3] = {1.0, 1.0, 1.0};
    int inside_metadata = 0, size, size2;
    int prelut = 0;
    int ret = 0;

    int prelut_sizes[3] = {0, 0, 0};
    float *in_prelut[3]  = {NULL, NULL, NULL};
    float *out_prelut[3] = {NULL, NULL, NULL};

    NEXT_LINE_OR_GOTO(skip_line(line), end);
    if (strncmp(line, "CSPLUTV100", 10)) {
        av_log(ctx, AV_LOG_ERROR, "Not cineSpace LUT format\n");
        ret = AVERROR(EINVAL);
        goto end;
    }

    NEXT_LINE_OR_GOTO(skip_line(line), end);
    if (strncmp(line, "3D", 2)) {
        av_log(ctx, AV_LOG_ERROR, "Not 3D LUT format\n");
        ret = AVERROR(EINVAL);
        goto end;
    }

    while (1) {
        NEXT_LINE_OR_GOTO(skip_line(line), end);

        if (!strncmp(line, "BEGIN METADATA", 14)) {
            inside_metadata = 1;
            continue;
        }
        if (!strncmp(line, "END METADATA", 12)) {
            inside_metadata = 0;
            continue;
        }        if (inside_metadata == 0) {
            int size_r, size_g, size_b;

            for (int i = 0; i < 3; i++) {
                int npoints = strtol(line, NULL, 0);

                if (npoints > 2) {
                    float v,last;

                    if (npoints > PRELUT_SIZE) {
                        av_log(ctx, AV_LOG_ERROR, "Prelut size too large.\n");
                        ret = AVERROR_INVALIDDATA;
                        goto end;
                    }

                    if (in_prelut[i] || out_prelut[i]) {
                        av_log(ctx, AV_LOG_ERROR, "Invalid file has multiple preluts.\n");
                        ret = AVERROR_INVALIDDATA;
                        goto end;
                    }

                    in_prelut[i]  = (float*)malloc(npoints * sizeof(float));
                    out_prelut[i] = (float*)malloc(npoints * sizeof(float));
                    if (!in_prelut[i] || !out_prelut[i]) {
                        ret = AVERROR(ENOMEM);
                        goto end;
                    }

                    prelut_sizes[i] = npoints;
                    in_min[i] = FLT_MAX;
                    in_max[i] = -FLT_MAX;
                    out_min[i] = FLT_MAX;
                    out_max[i] = -FLT_MAX;

                    for (int j = 0; j < npoints; j++) {
                        NEXT_FLOAT_OR_GOTO(v, end)
                        in_min[i] = FFMIN(in_min[i], v);
                        in_max[i] = FFMAX(in_max[i], v);
                        in_prelut[i][j] = v;
                        if (j > 0 && v < last) {
                            av_log(ctx, AV_LOG_ERROR, "Invalid file, non increasing prelut.\n");
                            ret = AVERROR(ENOMEM);
                            goto end;
                        }
                        last = v;
                    }

                    for (int j = 0; j < npoints; j++) {
                        NEXT_FLOAT_OR_GOTO(v, end)
                        out_min[i] = FFMIN(out_min[i], v);
                        out_max[i] = FFMAX(out_max[i], v);
                        out_prelut[i][j] = v;
                    }

                } else if (npoints == 2)  {
                    NEXT_LINE_OR_GOTO(skip_line(line), end);
                    if (av_sscanf(line, "%f %f", &in_min[i], &in_max[i]) != 2) {
                        ret = AVERROR_INVALIDDATA;
                        goto end;
                    }
                    NEXT_LINE_OR_GOTO(skip_line(line), end);
                    if (av_sscanf(line, "%f %f", &out_min[i], &out_max[i]) != 2) {
                        ret = AVERROR_INVALIDDATA;
                        goto end;
                    }

                } else {
                    av_log(ctx, AV_LOG_ERROR, "Unsupported number of pre-lut points.\n");
                    ret = AVERROR_PATCHWELCOME;
                    goto end;
                }

                NEXT_LINE_OR_GOTO(skip_line(line), end);
            }

            if (av_sscanf(line, "%d %d %d", &size_r, &size_g, &size_b) != 3) {
                ret = AVERROR(EINVAL);
                goto end;
            }
            if (size_r != size_g || size_r != size_b) {
                av_log(ctx, AV_LOG_ERROR, "Unsupported size combination: %dx%dx%d.\n", size_r, size_g, size_b);
                ret = AVERROR_PATCHWELCOME;
                goto end;
            }

            size = size_r;
            size2 = size * size;

            if (prelut_sizes[0] && prelut_sizes[1] && prelut_sizes[2])
                prelut = 1;

            ret = allocate_3dlut(lut3d, size, prelut);
            if (ret < 0)
                return ret;

            for (int k = 0; k < size; k++) {
                for (int j = 0; j < size; j++) {
                    for (int i = 0; i < size; i++) {
                        struct rgbvec *vec = &lut3d->lut[i * size2 + j * size + k];

                        NEXT_LINE_OR_GOTO(skip_line(line), end);
                        if (av_sscanf(line, "%f %f %f", &vec->r, &vec->g, &vec->b) != 3) {
                            ret = AVERROR_INVALIDDATA;
                            goto end;
                        }

                        vec->r *= out_max[0] - out_min[0];
                        vec->g *= out_max[1] - out_min[1];
                        vec->b *= out_max[2] - out_min[2];
                    }
                }
            }

            break;
        }
    }

    if (prelut) {
        for (int c = 0; c < 3; c++) {

            lut3d->prelut.min[c] = in_min[c];
            lut3d->prelut.max[c] = in_max[c];
            lut3d->prelut.scale[c] =  (1.0f / (float)(in_max[c] - in_min[c])) * (lut3d->prelut.size - 1);

            for (int i = 0; i < lut3d->prelut.size; ++i) {
                float mix = (float) i / (float)(lut3d->prelut.size - 1);
                float x = lerpf(in_min[c], in_max[c], mix), a, b;

                int idx = nearest_sample_index(in_prelut[c], x, 0, prelut_sizes[c]-1);
                av_assert0(idx + 1 < prelut_sizes[c]);

                a   = out_prelut[c][idx + 0];
                b   = out_prelut[c][idx + 1];
                mix = x - in_prelut[c][idx];

                lut3d->prelut.lut[c][i] = sanitizef(lerpf(a, b, mix));
            }
        }
        lut3d->scale.r = 1.00f;
        lut3d->scale.g = 1.00f;
        lut3d->scale.b = 1.00f;

    } else {
        lut3d->scale.r = av_clipf(1.f / (in_max[0] - in_min[0]), 0.f, 1.f);
        lut3d->scale.g = av_clipf(1.f / (in_max[1] - in_min[1]), 0.f, 1.f);
        lut3d->scale.b = av_clipf(1.f / (in_max[2] - in_min[2]), 0.f, 1.f);
    }

end:
    for (int c = 0; c < 3; c++) {
        av_freep(&in_prelut[c]);
        av_freep(&out_prelut[c]);
    }
    return ret;
}

/* Iridas format */
static int parse_cube(LUT3DContext *lut3d, FILE *f)
{
    void *ctx = lut3d;
    char line[MAX_LINE_SIZE];
    float min[3] = {0.0, 0.0, 0.0};
    float max[3] = {1.0, 1.0, 1.0};

    while (fgets(line, sizeof(line), f)) {
        if (!strncmp(line, "LUT_3D_SIZE", 11)) {
            int ret, i, j, k;
            const int size = strtol(line + 12, NULL, 0);
            const int size2 = size * size;

            ret = allocate_3dlut(ctx, size, 0);
            if (ret < 0)
                return ret;

            for (k = 0; k < size; k++) {
                for (j = 0; j < size; j++) {
                    for (i = 0; i < size; i++) {
                        struct rgbvec *vec = &lut3d->lut[i * size2 + j * size + k];

                        do {
try_again:
                            NEXT_LINE(0);
                            if (!strncmp(line, "DOMAIN_", 7)) {
                                float *vals = NULL;
                                if      (!strncmp(line + 7, "MIN ", 4)) vals = min;
                                else if (!strncmp(line + 7, "MAX ", 4)) vals = max;
                                if (!vals)
                                    return AVERROR_INVALIDDATA;
                                av_sscanf(line + 11, "%f %f %f", vals, vals + 1, vals + 2);
                                av_log(ctx, AV_LOG_DEBUG, "min: %f %f %f | max: %f %f %f\n",
                                       min[0], min[1], min[2], max[0], max[1], max[2]);
                                goto try_again;
                            } else if (!strncmp(line, "TITLE", 5)) {
                                goto try_again;
                            }
                        } while (skip_line(line));
                        if (av_sscanf(line, "%f %f %f", &vec->r, &vec->g, &vec->b) != 3)
                            return AVERROR_INVALIDDATA;
                    }
                }
            }
            break;
        }
    }

    lut3d->scale.r = av_clipf(1. / (max[0] - min[0]), 0.f, 1.f);
    lut3d->scale.g = av_clipf(1. / (max[1] - min[1]), 0.f, 1.f);
    lut3d->scale.b = av_clipf(1. / (max[2] - min[2]), 0.f, 1.f);

    return 0;
}


static int set_identity_matrix(LUT3DContext *lut3d, int size)
{
    void *ctx = lut3d;
    int ret, i, j, k;
    const int size2 = size * size;
    const float c = 1.f / (size - 1);

    ret = allocate_3dlut(lut3d, size, 0);
    if (ret < 0)
        return ret;

    for (k = 0; k < size; k++) {
        for (j = 0; j < size; j++) {
            for (i = 0; i < size; i++) {
                struct rgbvec *vec = &lut3d->lut[k * size2 + j * size + i];
                vec->r = k * c;
                vec->g = j * c;
                vec->b = i * c;
            }
        }
    }

    return 0;
}