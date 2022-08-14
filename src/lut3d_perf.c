#include "lut3d_perf.h"
#include "parse_lut.c"
#include "deps/tinyexr.h"

#include <immintrin.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef int (apply_lut_func)(const LUT3DContext *lut3d, const FloatImage *src_image,  FloatImage *dst_image);
typedef int (apply_lut_rgba_func)(const LUT3DContext *lut3d, const FloatImageRGBA *src_image,  FloatImageRGBA *dst_image);


typedef struct EXR {
    EXRImage image;
    EXRHeader header;
    EXRVersion version;
} EXR;

#if _WIN32
#include <windows.h>
#include <intrin.h>
static uint64_t get_timer_frequency()
{
    LARGE_INTEGER Result;
    QueryPerformanceFrequency(&Result);
    return Result.QuadPart;
}
static uint64_t get_timer(void)
{
    LARGE_INTEGER Result;
    QueryPerformanceCounter(&Result);
    return Result.QuadPart;
}
#else
#include <time.h>
#include <unistd.h>
#include <cpuid.h>
static uint64_t get_timer_frequency()
{
    uint64_t Result = 1000000000ull;
    return Result;
}
static uint64_t get_timer(void)
{
    struct timespec Spec;
    clock_gettime(CLOCK_MONOTONIC, &Spec);
    uint64_t Result = ((uint64_t)Spec.tv_sec * 1000000000ull) + (uint64_t)Spec.tv_nsec;
    return Result;
}
#endif

#include "tetrahedral_ffmpeg_c.h"
#include "tetrahedral_ffmpeg_asm.h"
#include "tetrahedral_avx2.h"
#include "tetrahedral_avx.h"

#include "tetrahedral_sse2.h"

static int read_exr(char *filename, EXR *exr)
{
    const char* err;
    int ret;
    InitEXRHeader(&exr->header);
    InitEXRImage(&exr->image);

    ret = ParseEXRVersionFromFile(&exr->version, filename);

    if (ret) {
        printf("error reading version\n");
        return ret;
    }

    ret = ParseEXRHeaderFromFile(&exr->header, &exr->version, filename, &err);

    if (ret) {
        printf("%s\n", err);
        return ret;
    }

    if (exr->header.tiled || exr->header.multipart){
        printf("tiled or multipart not supported\n");
        return -1;
    }

    // Read HALF channel as FLOAT.
    for (int i = 0; i < exr->header.num_channels; i++) {
        exr->header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }

    ret  = LoadEXRImageFromFile(&exr->image, &exr->header, filename, &err);

    if (ret) {
        printf("%s\n", err);
        return ret;
    }

    return 0;
}

int alloc_image(FloatImage *image, int width, int height)
{

    int stride = ((width + 7) / 8) * 8;

    size_t plane_size = height * stride * sizeof(float);
    size_t size = plane_size * 4;

    if (image->mem)
        free(image->mem);

    image->mem = malloc(size);
    memset(image->mem, 0, size);

    uint8_t *p = image->mem;
    image->data[0] = (float*)p;
    p += plane_size;

    image->data[1] = (float*)p;
    p += plane_size;

    image->data[2] = (float*)p;
    p += plane_size;

    image->data[3] = (float*)p;

    // fill alpha with 1;

    for (int y =0; y < height; y++) {
        float *a = image->data[3] + (y * stride);
        for (int x =0; x < width; x++) {
            *a++ = 1.0f;
        }
    }

    image->width = width;
    image->height = height;
    image->stride = stride;

    return 0;
}

int alloc_image_rgba(FloatImageRGBA *image, int width, int height)
{
    if (image->data) {
        free(image->data);
    }

    size_t size = width*height * sizeof(float) * 4;

    image->data = (float*)malloc(size);
    image->width = width;;
    image->height = height;

    memset(image->data, 0, size);

    return 0;
}

int fill_plane(float *src, float *dst, int width, int height, int dst_stride)
{
    for (int y =0; y < height; y++) {
        for (int x =0; x < width; x++) {
            dst[x] = src[x];
        }

        src += width;
        dst += dst_stride;
    }

    return 0;
}

int find_rgba(EXR *exr, FloatImage *image)
{
    // RGBA
    int R = -1;
    int G = -1;
    int B = -1;
    int A = -1;

    int found = 0;

    int w = exr->image.width;
    int h = exr->image.height;

    alloc_image(image, w, h);

    int s = image->stride;

    for (int i = 0; i < exr->header.num_channels; i++) {
        char *name = exr->header.channels[i].name;
        // printf("%d name %s\n", i, exr->header.channels[i].name);
        if (strcmp(name, "R") == 0) {
            fill_plane((float*)exr->image.images[i], image->data[0], w, h, s);
            R = i;
        } else if (strcmp(name, "G") == 0) {
            fill_plane((float*)exr->image.images[i], image->data[1], w, h, s);
            G = i;
        } else if (strcmp(name, "B") == 0) {
            fill_plane((float*)exr->image.images[i], image->data[2], w, h, s);
            B = i;
        } else if (strcmp(name, "A") == 0) {
            fill_plane((float*)exr->image.images[i], image->data[3], w, h, s);
            A = i;
        }
    }

    image->height = exr->image.height;
    image->width = exr->image.width;

    if (R < 0 || G < 0 || B < 0) {
        return -1;
    }
    return 0;
}

static inline int strcmp_ignore_case(const char* str1, const char* str2)
{
    int d;
    while ((d = toupper(*str2) - toupper(*str1)) == 0 && *str1) { str1++; str2++; }
    return d;
}



int read_lut(LUT3DContext *lut3d, char *path)
{
    FILE *f = fopen(path, "rb");
    int ret = 0;
    if (!f) {
        printf("error opening: %s\n", path);
        return -1;
    }

    const char *ext = strrchr(path, '.');
    if (!ext) {
        printf("cannot determine lut format from extension: %s\n", path);
        return -1;
    }

    if (!strcmp_ignore_case(ext, ".csp")) {
        ret = parse_cinespace(lut3d, f);
    } else if (!strcmp_ignore_case(ext, ".cube")) {
        ret = parse_cube(lut3d, f);
    } else {
        printf("unrecognized lut format '.%s' \n", ext);
        return -1;
    }

    if(ret)
        printf("error parsing lut\n");

    return ret;
}

int write_png(FloatImage *image, char *path)
{
    int channels = 3;
    if (image->data[3])
        channels = 4;

    uint8_t *data = malloc(image->width * image->height * channels);

    int stride = image->width * channels;


    for (int y = 0; y < image->height; y++) {
        int offset =  y * image->stride;
        uint8_t *p = data + (y * stride);

        for (int x = 0; x < image->width; x++) {

            float r = image->data[0][offset +x];
            float g = image->data[1][offset +x];
            float b = image->data[2][offset +x];

            r = av_clipf(r * 255.0f, 0.0f, 255.0f);
            g = av_clipf(g * 255.0f, 0.0f, 255.0f);
            b = av_clipf(b * 255.0f, 0.0f, 255.0f);

            *p++ = (uint8_t)r;
            *p++ = (uint8_t)g;
            *p++ = (uint8_t)b;
            if (image->data[3]) {
                float a = image->data[3][offset +x];
                a = av_clipf(a * 255.0f, 0.0f, 255.0f);
                *p++ = (uint8_t)a;

            }
        }
    }

    int ret = stbi_write_png(path, image->width, image->height, channels, data, stride);
    free(data);
    return ret;
}

void write_exr(FloatImage *image, char *path)
{
    int width = image->width;
    int height = image->height;

    float *data = (float*)malloc(width*height * sizeof(float) * 4);
    int plane_size = width * height;

    for (int y = 0; y < height; y++) {
        size_t offset =  y * image->stride;
        float* ptrf = data + (y * width * 4);
        for(int x = 0; x < width; x++) {

            float r = image->data[0][offset + x];
            float g = image->data[1][offset + x];
            float b = image->data[2][offset + x];
            float a = image->data[3][offset + x];

            *ptrf++ = r;
            *ptrf++ = g;
            *ptrf++ = b;
            *ptrf++ = a;
        }
    }

    int save_as_fp16 = 1;
    int components = 4;
    const char *error;
    SaveEXR(data, width, height, components, save_as_fp16, path, &error);
    free(data);
}

static int cmp_images(FloatImage *a, FloatImage *b)
{

    int ret = 0;

    double err = 0;
    double max_error = 0.0f;
    for (int y = 0; y < a->height; y++) {

        for (int c = 0; c < 3; c++) {

            float *a_v = a->data[c] + (y * a->stride);
            float *b_v = b->data[c] + (y * b->stride);

            for (int x =0; x < a->width; x++) {

                double pixel_err =  fabs(*a_v - *b_v);
                if (!ret && *a_v != *b_v) {
                    // printf("%d %dx%d %.12f != %.12f  abs err: %.12f\n",c, x, y, *a_v, *b_v, pixel_err);
                    ret = 1;
                    // return ret;
                }

                if (pixel_err > max_error) {
                    max_error = pixel_err;
                }

                err += pixel_err;
                a_v++;
                b_v++;

            }
        }
    }

    printf("  avg_error: %.12f\n", err / (a->width * a->height * 3));
    printf("  max_error: %.12f\n", max_error);
    return ret;
}

static int planer_to_rgba(FloatImage *src, FloatImageRGBA *dst)
{
    int width  = src->width;
    int height = src->height;

    alloc_image_rgba(dst, width, height);

    for (int y = 0; y < height; y++) {
        size_t offset =  y * src->stride;
        float* ptrf = dst->data + (y * width * 4);
        for(int x = 0; x < width; x++) {

            float r = src->data[0][offset + x];
            float g = src->data[1][offset + x];
            float b = src->data[2][offset + x];
            float a = src->data[3][offset + x];

            *ptrf++ = r;
            *ptrf++ = g;
            *ptrf++ = b;
            *ptrf++ = a;
        }
    }

    return 0;
}


static int rgba_to_planer(FloatImageRGBA *src, FloatImage *dst)
{
    int width  = src->width;
    int height = src->height;

    alloc_image(dst, width, height);

    for (int y = 0; y < height; y++) {
        size_t offset =  y * dst->stride;
        float* ptrf = src->data + (y * width * 4);
        for(int x = 0; x < width; x++) {
            dst->data[0][offset + x] = *ptrf++;
            dst->data[1][offset + x] = *ptrf++;
            dst->data[2][offset + x] = *ptrf++;
            dst->data[3][offset + x] = *ptrf++;
        }
    }

    return 0;
}



static uint32_t float_test_table[] = {
    0x7F800000, // +inf
    0xFF800000, // -inf
    0x00000000, // 0
    0x80000000, // -0
    0x3F800000, // +1
    0xBF800000, // -1
    0xFFFFFFFF, // -nan
    0x7FFFFFFF, // nan
    0x7FBFFFFF, // nan
    0x7F800001, // nan
    0xFF800001, // -nan
    0x7F801000, // nan
    0xFF801000, // -nan
};

typedef struct {
    char *name;
    apply_lut_func *apply_lut;
    apply_lut_rgba_func *apply_lut_rgba;
} LutTestItem;

int apply_lut_ocio_rgba(const LUT3DContext *lut3d, const FloatImageRGBA *src_image, FloatImageRGBA *dst_image);
int apply_lut_ocio_sse2_rgba(const LUT3DContext *lut3d, const FloatImageRGBA *src_image, FloatImageRGBA *dst_image);

static LutTestItem LUTS[] = {
    {"ffmpeg_c",                                    apply_lut_c,                           NULL},
    {"ocio_c++",                                           NULL,            apply_lut_ocio_rgba},
    {"ocio_sse2",                                          NULL,       apply_lut_ocio_sse2_rgba},
    {"avx2_planer_intrinsics", apply_lut_planer_intrinsics_avx2,                           NULL},
    {"avx2_rgba_intrinsics",                               NULL, apply_lut_rgba_intrinsics_avx2},
    {"avx_planer_intrinsics",   apply_lut_planer_intrinsics_avx,                           NULL},
    {"avx_rgba_intrinsics",                                NULL,  apply_lut_rgba_intrinsics_avx},
    {"sse2_planer_intrinsics", apply_lut_planer_intrinsics_sse2,                           NULL},
    {"sse2_rgba_intrinsics",                               NULL, apply_lut_rgba_intrinsics_sse2},
    {"ffmpeg_avx2_asm",                      apply_lut_avx2_asm,                           NULL},
    {"ffmpeg_avx_asm",                        apply_lut_avx_asm,                           NULL},
    {"ffmpeg_sse2_asm",                      apply_lut_sse2_asm,                           NULL},
};

#define ARRAY_SIZE(x)  (sizeof(x) / sizeof((x)[0]))


void nan_nonesense()
{
    union intfloat {
        uint32_t i;
        float    f;
    };

    union intfloat v;

    v.i = 0xFFFFFFFF; // -nan
    if (v.f)
        printf("if nan is true\n");
    else
        printf("if nan is false\n");

    if (!v.f)
        printf("if !nan is true\n");
    else
        printf("if !nan is false\n");

}

void print_cpu()
{
    char CPU[65] = {0};
    for(int index = 0; index < 3; index++)
    {
#if _WIN32
        __cpuid((int *)(CPU + 16*index), 0x80000002 + index);
#else
        __get_cpuid(0x80000002 + index,
                    (int unsigned *)(CPU + 16*index),
                    (int unsigned *)(CPU + 16*index + 4),
                    (int unsigned *)(CPU + 16*index + 8),
                    (int unsigned *)(CPU + 16*index + 12));
#endif
    }

    printf("CPU: %s\n", CPU);
}

typedef struct  {
    char *name;
    double per_frame;
    double elapse;
} TestResult;

int main(int argc, char *argv[])
{
    if(argc < 3) {
        printf("not enough args\n");
        return -1;
    }

    int ret;

    EXR exr = {0};
    FloatImage src_image = {0};
    FloatImage dst_image = {0};

    FloatImageRGBA src_image_rgba = {0};
    FloatImageRGBA dst_image_rgba = {0};

    FloatImage cmp_image = {0};
    LUT3DContext lut3d = {0};
    char result_name_png[1024] = {0};
    char result_name_exr[1024] = {0};

    char *exr_path = argv[1];
    char *lut_path = argv[2];

    uint64_t freq = get_timer_frequency();
    uint64_t start = 0;
    uint64_t end = 0;

    ret = read_exr(exr_path, &exr);
    if (ret)
        return -1;

    ret = find_rgba(&exr, &src_image);
    if (ret) {
        printf("unable to find RGB channels\n");
        return -1;
    }

    ret = alloc_image(&dst_image, src_image.width, src_image.height);
    if (ret) {
        printf("unable to alloc dest image\n");
        return -1;
    }

    ret = alloc_image_rgba(&dst_image_rgba, src_image.width, src_image.height);
    if (ret) {
        printf("unable to alloc dest rgba image\n");
        return -1;
    }

    ret = alloc_image(&cmp_image, src_image.width, src_image.height);
    if (ret) {
        printf("unable to alloc cmp image\n");
        return -1;
    }

    // prep rgb test image too
    planer_to_rgba(&src_image, &src_image_rgba);

    ret = read_lut(&lut3d, lut_path);
    if (ret)
        return -1;

    // sse2 ocio algorithm requires the lut to be rgba
    int lutsize3 = lut3d.lutsize * lut3d.lutsize * lut3d.lutsize;
    for (int i = 0; i < lutsize3; i++) {
        lut3d.rgba_lut[i].r = lut3d.lut[i].r;
        lut3d.rgba_lut[i].g = lut3d.lut[i].g;
        lut3d.rgba_lut[i].b = lut3d.lut[i].b;
        lut3d.rgba_lut[i].a = 0.0f;
    }

    write_png(&src_image, "original.png");

    int runs = 100;
    apply_lut_c(&lut3d, &src_image, &cmp_image);

    // rgba_to_planer(&src_image_rgba, &src_image);

    print_cpu();
    printf("lut: %s\n", lut_path);
    printf("exr: %s\n", exr_path);

    TestResult *test_results = (TestResult*)malloc(  ARRAY_SIZE(LUTS) * sizeof(TestResult));

    for (int i = 0; i < ARRAY_SIZE(LUTS); i++) {

        uint64_t dur = 0;
        LutTestItem *test = &LUTS[i];

        printf("%s : %d runs %dx%d\n", test->name, runs, src_image.width, src_image.height);
        fflush(stdout);
        for (int j=0; j < runs; j++) {
            if (test->apply_lut) {
                start = get_timer();
                test->apply_lut(&lut3d, &src_image, &dst_image);
                end = get_timer();
                dur += end - start;
            } else {
                start = get_timer();
                test->apply_lut_rgba(&lut3d, &src_image_rgba, &dst_image_rgba);
                end = get_timer();
                dur += end - start;
            }
        }

        double elapse    = (double)dur / (double)freq;
        double per_frame = elapse / (double)runs;

        printf("  elapse   : %f secs\n", elapse);
        printf("  perframe : %0.016f secs\n", per_frame);
        if (test->apply_lut) {
            test->apply_lut(&lut3d, &src_image, &dst_image);
        } else {
            test->apply_lut_rgba(&lut3d, &src_image_rgba, &dst_image_rgba);
            rgba_to_planer(&dst_image_rgba, &dst_image);
        }
        cmp_images(&dst_image, &cmp_image);
        printf("\n");
        fflush(stdout);

        snprintf(result_name_png, 1024, "%s_result.png", test->name);
        snprintf(result_name_exr, 1024, "%s_result.exr", test->name);
        write_png(&dst_image, result_name_png);
        write_exr(&dst_image, result_name_exr);

        test_results[i].name      = test->name;
        test_results[i].per_frame = per_frame;
        test_results[i].elapse    = elapse;
    }

    FILE *f = fopen("results.csv", "wb");
    if (!f) {
        printf("error opening: %s\n", "results.csv");
        return -1;
    }
    char buffer[1024] = {0};
    for (int i = 0; i < ARRAY_SIZE(LUTS); i++) {
        int length = snprintf(buffer, 1024, "%s,%g\n", test_results[i].name, test_results[i].per_frame);
        fwrite(buffer, 1, length, f);
    }
    fclose(f);

    return 0;
}