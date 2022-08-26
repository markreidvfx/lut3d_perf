#include "lut3d_perf.h"
#include "parse_lut.c"
#include "deps/tinyexr.h"
#include <stdio.h>
#include <ctype.h>

#include <immintrin.h>

#ifdef _MSC_VER
#define get_extended_control_reg _xgetbv
#else

#define xgetbv_asm(index, eax, edx)                                        \
    __asm__ (".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c" (index))

static inline int64_t get_extended_control_reg(int index)
{
    int low = 0;
    int hi = 0;
    xgetbv_asm(0, low, hi);
    return  (int64_t)hi << 32 | (int64_t)low;
}
#undef xgetbv_asm

// Inline cpuid instruction.  In PIC compilations, %ebx contains the address
// of the global offset table.  To avoid breaking such executables, this code
// must preserve that register's value across cpuid instructions.
#define __cpuid__(index, eax, ebx, ecx, edx)                        \
    __asm__ volatile (                                          \
        "mov    %%rbx, %%rsi \n\t"                              \
        "cpuid               \n\t"                              \
        "xchg   %%rbx, %%rsi"                                   \
        : "=a" (eax), "=S" (ebx), "=c" (ecx), "=d" (edx)        \
        : "0" (index), "2"(0))


#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef int (apply_lut_func)(const LUT3DContext *lut3d, const FloatImage *src_image,  FloatImage *dst_image);
typedef int (apply_lut_rgba_func)(const LUT3DContext *lut3d, const FloatImageRGBA *src_image,  FloatImageRGBA *dst_image);

typedef struct EXR {
    EXRImage image;
    EXRHeader header;
    EXRVersion version;
} EXR;

typedef struct  {
    char *name;
    double per_frame;
    double elapse;
} TestResult;


#if _WIN32
#include <windows.h>
#include <intrin.h>
#define strdup _strdup

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

static void cpuid(int index, int *data)
{
#ifdef _MSC_VER
    __cpuid(data, index);
#else
    int eax, ebx, ecx, edx;
    __cpuid__(index, eax, ebx, ecx, edx);
    data[0] = eax;
    data[1] = ebx;
    data[2] = ecx;
    data[3] = edx;
#endif
}

union cpuid_data {
    int i[4];
    char c[16];
    struct {
        int eax;
        int ebx;
        int ecx;
        int edx;
    } reg;
};

typedef struct {
    char name[65];
    int has_sse2;
    int has_avx;
    int has_avx2;
}CPUFeatures;


static CPUFeatures get_cpu_features()
{
    CPUFeatures features = {0};
    for(int index = 0; index < 3; index++) {
      cpuid(0x80000002 + index, (int *)(features.name + 16*index));
    }

    union cpuid_data info;
    cpuid(0, info.i);
    int max_std_level = info.i[0];

    if (max_std_level >= 1) {
        cpuid(1, info.i);
        if (info.reg.edx & (1 << 26)) {
            features.has_sse2 = 1;
        }
        /* Check OXSAVE and AVX bits */
        if ((info.reg.ecx & 0x18000000) == 0x18000000) {
            int64_t xcr = get_extended_control_reg(0);
            if(xcr & 0x6) {
                features.has_avx = 1;
            }
        }
    }

    if (max_std_level >= 7) {
        cpuid(7, info.i);
        if (features.has_avx  && info.reg.ebx & 0x00000020) {
            features.has_avx2 = 1;
        }
    }

    // disable avx/avx2
    // features.has_avx = 0;
    // features.has_avx2 = 0;

    printf("CPU: %s ", features.name);
    if (features.has_sse2)
        printf("+sse2");
    if (features.has_avx)
        printf("+avx");
    if (features.has_avx2)
        printf("+avx2");
    printf("\n");
    return features;
}



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

static int alloc_image(FloatImage *image, int width, int height)
{

    int stride = ((width + 7) / 8) * 8;

    size_t plane_size = height * stride * sizeof(float);
    size_t size = plane_size * 4;

    if (image->width != width || image->height != height || image->stride != stride) {

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
    }

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

static int alloc_image_rgba(FloatImageRGBA *image, int width, int height)
{
    size_t size = width*height * sizeof(float) * 4;

    if (image->width != width || image->height != height) {

        if (image->data)
            free(image->data);

        image->data = (float*)malloc(size);
    }

    image->width = width;;
    image->height = height;
    memset(image->data, 0, size);

    return 0;
}

static int fill_plane(float *src, float *dst, int width, int height, int dst_stride)
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

static int find_rgba(EXR *exr, FloatImage *image)
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



static int read_lut(LUT3DContext *lut3d, char *path)
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

static int write_png(FloatImage *image, char *path)
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

static void write_exr(FloatImage *image, char *path)
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

static int write_test_results(char *csv_path, TestResult *test_results, int count)
{
    FILE *f = fopen("results.csv", "wb");
    if (!f) {
        printf("error opening: %s\n", "results.csv");
        return -1;
    }
    char buffer[1024] = {0};
    for (int i = 0; i < count; i++) {
        int length = snprintf(buffer, 1024, "%s,%g\n", test_results[i].name, test_results[i].per_frame);
        fwrite(buffer, 1, length, f);
    }
    fclose(f);

    return 0;
}

typedef struct CMPResult {
    double avg_error;
    double max_error;
} CMPResult;

static inline CMPResult cmp_images(FloatImage *a, FloatImage *b)
{
    int ret = 0;
    CMPResult result = {0};

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

    result.avg_error = err / (a->width * a->height * 3);
    result.max_error = max_error;
    return result;
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
    int avx;
    int avx2;
} LutTestItem;

int apply_lut_ocio_rgba(const LUT3DContext *lut3d, const FloatImageRGBA *src_image, FloatImageRGBA *dst_image);
int apply_lut_ocio_sse2_rgba(const LUT3DContext *lut3d, const FloatImageRGBA *src_image, FloatImageRGBA *dst_image);

int LUT_SIZES[] = {32, 64};

static LutTestItem LUTS[] = {
    {"ffmpeg_c",                                    apply_lut_c,                           NULL, 0, 0},
    {"ocio_c++",                                           NULL,            apply_lut_ocio_rgba, 0, 0},
    {"ocio_sse2",                                          NULL,       apply_lut_ocio_sse2_rgba, 0, 0},
    {"avx2_planer_intrinsics", apply_lut_planer_intrinsics_avx2,                           NULL, 1, 1},
    {"avx2_rgba_intrinsics",                               NULL, apply_lut_rgba_intrinsics_avx2, 1, 1},
    {"avx_planer_intrinsics",   apply_lut_planer_intrinsics_avx,                           NULL, 1, 0},
    {"avx_rgba_intrinsics",                                NULL,  apply_lut_rgba_intrinsics_avx, 1, 0},
    {"sse2_planer_intrinsics", apply_lut_planer_intrinsics_sse2,                           NULL, 0, 0},
    {"sse2_rgba_intrinsics",                               NULL, apply_lut_rgba_intrinsics_sse2, 0, 0},
    {"ffmpeg_avx2_asm",                      apply_lut_avx2_asm,                           NULL, 1, 1},
    {"ffmpeg_avx_asm",                        apply_lut_avx_asm,                           NULL, 1, 0},
    {"ffmpeg_sse2_asm",                      apply_lut_sse2_asm,                           NULL, 0, 0},
};

#define ARRAY_SIZE(x)  (sizeof(x) / sizeof((x)[0]))


static void nan_nonesense()
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

static inline float rand_float_range(float a, float b)
{
    assert(a < b);
    float v = (float)rand()/(float)(RAND_MAX);
    v *= (b - a);
    v -= a;
    return v;
}


static inline void rand_float_image(FloatImage *img)
{
    int width = img->width;
    int height = img->height;

    for (int y = 0; y < height; y++) {
        size_t offset =  y * img->stride;
        for(int x = 0; x < width; x++) {

#if 0
            union av_intfloat32 v;
            v.i = rand();
            img->data[0][offset + x] = v.f;
            v.i = rand();
            img->data[1][offset + x] = v.f;
            v.i = rand();
            img->data[2][offset + x] = v.f;
            v.i = rand();
            img->data[3][offset + x] = v.f;
#else
            img->data[0][offset + x] = rand_float_range(-0.05, 2.0f);
            img->data[1][offset + x] = rand_float_range(-0.05, 2.0f);
            img->data[2][offset + x] = rand_float_range(-0.05, 2.0f);
            img->data[3][offset + x] = rand_float_range(-0.05, 2.0f);
#endif
        }
    }
}


static void rand_lut(LUT3DContext *lut3d)
{
    int lutsize = lut3d->lutsize;
    for (int i = 0; i < lutsize*lutsize*lutsize; i++) {
        lut3d->lut[i].r = rand_float_range(0.0f, 1.0f);
        lut3d->lut[i].g = rand_float_range(0.0f, 1.0f);
        lut3d->lut[i].b = rand_float_range(0.0f, 1.0f);

        lut3d->rgba_lut[i].r = lut3d->lut[i].r;
        lut3d->rgba_lut[i].g = lut3d->lut[i].g;
        lut3d->rgba_lut[i].b = lut3d->lut[i].b;
        lut3d->rgba_lut[i].a = 0.0f;
    }
}

static void rand_prelut(LUT3DContext *lut3d)
{
    Lut3DPreLut *prelut = &lut3d->prelut;

    for (int c = 0; c < 3; c++) {

        prelut->min[c] = INFINITY;
        prelut->max[c] = -INFINITY;

        for (int i = 0; i < prelut->size; i++) {
            float v =  rand_float_range(-0.05f, 2.0f);
            prelut->lut[c][i] = v;
            prelut->min[c] = FFMIN(v, prelut->min[c]);
            prelut->max[c] = FFMAX(v, prelut->max[c]);

        }

        prelut->scale[c] =  (1.0f / (prelut->max[c] - prelut->min[c])) * (float)(prelut->size - 1);
        // diff between min and max cannot exceed FLT_MAX
        assert(!isinf(prelut->scale[c]));
    }

    lut3d->scale.r = 1.00f;
    lut3d->scale.g = 1.00f;
    lut3d->scale.b = 1.00f;
}

static int random_lut_test()
{
    uint64_t freq = get_timer_frequency();
    LUT3DContext lut3d = {0};

    FloatImage src_image = {0};
    FloatImage dst_image = {0};
    FloatImage ref_image = {0};

    FloatImageRGBA src_image_rgba = {0};
    FloatImageRGBA dst_image_rgba = {0};

    char test_name[1024] = {0};

    char result_name_png[1024] = {0};
    char result_name_exr[1024] = {0};


    int test_count = ARRAY_SIZE(LUT_SIZES) * ARRAY_SIZE(LUTS);
    TestResult *test_results = (TestResult*)malloc(test_count * sizeof(TestResult));
    TestResult *test_result = test_results;
    test_count = 0;

    CPUFeatures features = get_cpu_features();
    uint64_t test_start = get_timer();

    for (int l = 0; l < ARRAY_SIZE(LUT_SIZES); l++) {
        int lutsize = LUT_SIZES[l];
        int image_width = 1024;
        int image_height = 1024;

        allocate_3dlut(&lut3d, lutsize, 1);

        for (int i = 0; i < ARRAY_SIZE(LUTS); i++) {
            uint64_t dur = 0;
            uint64_t start = 0;
            uint64_t end = 0;

            LutTestItem *test = &LUTS[i];

            if (test->avx2 && !features.has_avx2) {
                printf("skipping %s cpu does not have avx2\n", test->name);
                continue;
            }

            if (test->avx && !features.has_avx) {
                printf("skipping %s cpu does not have avx\n", test->name);
                continue;
            }

            srand(0);

            int random_lut_count = 5;
            int runs = 50;

            snprintf(test_name, 1024, "%s_%dx%d_%dx%dx%d",test->name, image_width, image_height, lutsize,lutsize,lutsize);

            printf("%s random luts: %d runs: %d\n", test_name, random_lut_count, runs);
            fflush(stdout);

            CMPResult cmp = {0};

            for (int j=0; j < random_lut_count; j++) {
                alloc_image(&src_image, image_width, image_height);
                alloc_image(&ref_image, image_width, image_height);

                rand_float_image(&src_image);
                planer_to_rgba(&src_image, &src_image_rgba);

                alloc_image(&dst_image,           src_image.width, src_image.height);
                alloc_image_rgba(&dst_image_rgba, src_image.width, src_image.height);

                rand_lut(&lut3d);
                rand_prelut(&lut3d);

                if (test->apply_lut) {
                    start = get_timer();
                    for (int k = 0; k < runs; k++){
                        test->apply_lut(&lut3d, &src_image, &dst_image);
                    }
                    end = get_timer();
                } else {
                    start = get_timer();
                    for (int k = 0; k < runs; k++){
                        test->apply_lut_rgba(&lut3d, &src_image_rgba, &dst_image_rgba);
                    }
                    end = get_timer();

                    rgba_to_planer(&dst_image_rgba, &dst_image);
                }
                dur += (end - start);

                apply_lut_c(&lut3d, &src_image, &ref_image);
                CMPResult r = cmp_images(&ref_image, &dst_image);
                cmp.max_error = FFMAX(r.max_error, cmp.max_error);
                cmp.avg_error += r.avg_error;
#if 0
                snprintf(result_name_png, 1024, "%s.%04d.png", test_name, j);
                snprintf(result_name_exr, 1024, "%s.%04d.exr", test_name, j);
                write_png(&dst_image, result_name_png);
                write_exr(&dst_image, result_name_exr);
#endif
            }

            cmp.avg_error /= runs;

            double elapse    = (double)dur / (double)freq;
            double per_frame = elapse / ((double)random_lut_count * (double)runs);

            printf("  elapse   : %f secs\n", elapse);
            printf("  per_frame: %0.016f secs\n", per_frame);
            printf("  avg_error: %.12f\n", cmp.avg_error);
            printf("  max_error: %.12f\n", cmp.max_error);
            printf("\n");
            fflush(stdout);

            test_result->name = strdup(test_name);
            test_result->per_frame = per_frame;
            test_result->elapse = elapse;
            test_result++;
            test_count++;
        }
    }

    write_test_results("rand_results.csv", test_results, test_count);

    double test_elapse = (double)(get_timer() - test_start)/ (double)freq;
    printf("tests ran in %f secs\n", test_elapse);

    return 0;
}

static int exr_image_test(int argc, char *argv[])
{
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
    if (ret) {
        printf("unable to read EXR: %s\n", exr_path);
        return -1;
    }
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

    CPUFeatures features = get_cpu_features();
    printf("lut: %s\n", lut_path);
    printf("exr: %s\n", exr_path);

    TestResult *test_results = (TestResult*)malloc(ARRAY_SIZE(LUTS) * sizeof(TestResult));
    TestResult *test_result = test_results;
    int test_count = 0;
    uint64_t test_start = get_timer();

    for (int i = 0; i < ARRAY_SIZE(LUTS); i++) {

        uint64_t dur = 0;
        LutTestItem *test = &LUTS[i];

        if (test->avx2 && !features.has_avx2) {
            printf("skipping %s cpu does not have avx2\n", test->name);
            continue;
        }

        if (test->avx && !features.has_avx) {
            printf("skipping %s cpu does not have avx\n", test->name);
            continue;
        }

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

        if (test->apply_lut) {
            test->apply_lut(&lut3d, &src_image, &dst_image);
        } else {
            test->apply_lut_rgba(&lut3d, &src_image_rgba, &dst_image_rgba);
            rgba_to_planer(&dst_image_rgba, &dst_image);
        }

        CMPResult cmp = cmp_images(&dst_image, &cmp_image);
        printf("  elapse   : %f secs\n", elapse);
        printf("  per_frame: %0.016f secs\n", per_frame);
        printf("  avg_error: %.12f\n", cmp.avg_error);
        printf("  max_error: %.12f\n", cmp.max_error);
        printf("\n");
        fflush(stdout);

        snprintf(result_name_png, 1024, "%s_result.png", test->name);
        snprintf(result_name_exr, 1024, "%s_result.exr", test->name);
        write_png(&dst_image, result_name_png);
        write_exr(&dst_image, result_name_exr);

        test_result->name      = test->name;
        test_result->per_frame = per_frame;
        test_result->elapse    = elapse;
        test_result++;
        test_count++;
    }

    write_test_results("results.csv", test_results, test_count);
    double test_elapse = (double)(get_timer() - test_start)/ (double)freq;
    printf("tests ran in %f secs\n", test_elapse);

    return 0;
}


static void print_usage()
{
    printf("Usage: lut3d_perf [EXR] [LUT]\n"
           "Performance tests for various tetrahedral 3D lut implementations\n"
           "If run with zero arguments will run test with randomly generated \n"
           "images and randomly generated luts\n");
}

int main(int argc, char *argv[])
{
    if(argc == 1) {
        return random_lut_test();
    }

    if(argc != 3) {
        printf("not enough args %d != 3\n", argc);
        print_usage();
        return -1;
    }

    return exr_image_test(argc, argv);
}