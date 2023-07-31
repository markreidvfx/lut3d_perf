
#include "x86_cpu_info.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>

typedef union
{
    int i[4];
    char c[16];
    struct {
        uint32_t eax;
        uint32_t ebx;
        uint32_t ecx;
        uint32_t edx;
    } reg;
} CPUIDResult;

static inline int64_t xgetbv()
{
    int index = 0;
#if _MSC_VER
    return _xgetbv(index);
#else
    int eax = 0;
    int edx = 0;
    __asm__ volatile (".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c" (index));
    return  (int64_t)edx << 32 | (int64_t)eax;
#endif
}

static inline void cpuid(int index, int *data)
{
#if _MSC_VER
    __cpuid(data, index);
#else
    __asm__ volatile (
        "mov    %%rbx, %%rsi \n\t"
        "cpuid               \n\t"
        "xchg   %%rbx, %%rsi"
        : "=a" (data[0]), "=S" (data[1]), "=c" (data[2]), "=d" (data[3])
        : "0" (index), "2"(0));
#endif
}


#define ADD_FLAG_STR(ext_flag, name) \
if (out->flags & ext_flag) {         \
    offset = sprintf(p, "%s", name);     \
    p+= offset;                          \
}

static void make_extensions_string(CPUInfo *out)
{
    char *p = &out->extensions[0];
    int offset = 0;

    ADD_FLAG_STR(X86_CPU_FLAG_SSE2,  "+sse2")
    ADD_FLAG_STR(X86_CPU_FLAG_SSE3,  "+sse3")
    ADD_FLAG_STR(X86_CPU_FLAG_SSSE3, "+ssse3")
    ADD_FLAG_STR(X86_CPU_FLAG_SSE4,  "+sse4")
    ADD_FLAG_STR(X86_CPU_FLAG_SSE42, "+sse42")
    ADD_FLAG_STR(X86_CPU_FLAG_AVX,   "+avx")
    ADD_FLAG_STR(X86_CPU_FLAG_AVX2,  "+avx2")
    ADD_FLAG_STR(X86_CPU_FLAG_AVX512, "+avx512")
    ADD_FLAG_STR(X86_CPU_FLAG_F16C,  "+f16c")
}

void get_cpu_info(CPUInfo *out)
{
    uint32_t flags = 0;
    out->family = 0;
    out->model = 0;
    memset(out->name, 0, sizeof(out->name));
    memset(out->vendor, 0, sizeof(out->vendor));

    CPUIDResult info;
    uint32_t max_std_level, max_ext_level;
    int64_t xcr = 0;

    cpuid(0, info.i);
    max_std_level = info.i[0];
    memcpy(out->vendor + 0, &info.i[1], 4);
    memcpy(out->vendor + 4, &info.i[3], 4);
    memcpy(out->vendor + 8, &info.i[2], 4);

    if (max_std_level >= 1)
    {
        cpuid(1, info.i);
        out->family = ((info.reg.eax >> 8) & 0xf) + ((info.reg.eax >> 20) & 0xff);
        out->model  = ((info.reg.eax >> 4) & 0xf) + ((info.reg.eax >> 12) & 0xf0);

        if (info.reg.edx & (1 << 26))
            flags |= X86_CPU_FLAG_SSE2;

        if (info.reg.ecx & 1)
            flags |= X86_CPU_FLAG_SSE3;

        if (info.reg.ecx & 0x00000200)
            flags |= X86_CPU_FLAG_SSSE3;

        if (info.reg.ecx & 0x00080000)
            flags |= X86_CPU_FLAG_SSE4;

        if (info.reg.ecx & 0x00100000)
            flags |= X86_CPU_FLAG_SSE42;

        /* Check OSXSAVE and AVX bits */
        if (info.reg.ecx & 0x18000000)
        {
            xcr = xgetbv();
            if(xcr & 0x6) {
                flags |= X86_CPU_FLAG_AVX;

                if(info.reg.ecx & 0x20000000) {
                    flags |= X86_CPU_FLAG_F16C;
                }
            }
        }
    }

    if (max_std_level >= 7)
    {
        cpuid(7, info.i);

        if ((flags & X86_CPU_FLAG_AVX) && (info.reg.ebx & 0x00000020))
            flags |= X86_CPU_FLAG_AVX2;

        /* OPMASK/ZMM state */
        if ((xcr & 0xe0) == 0xe0) {
            if ((flags & X86_CPU_FLAG_AVX2) && (info.reg.ebx & 0xd0030000))
                flags |= X86_CPU_FLAG_AVX512;
        }
    }

    cpuid(0x80000000, info.i);
    max_ext_level = info.i[0];

    if (max_ext_level >= 0x80000001)
    {
        cpuid(0x80000001, info.i);
        if (!strncmp(out->vendor, "AuthenticAMD", 12)) {

            /* Athlon64, some Opteron, and some Sempron processors */
            if (flags & X86_CPU_FLAG_SSE2 && !(info.reg.ecx & 0x00000040))
                flags |= X86_CPU_FLAG_SSE2_SLOW;

            /* Bulldozer and Jaguar based CPUs */
            if ((out->family == 0x15 || out->family == 0x16) && (flags & X86_CPU_FLAG_AVX))
                flags |= X86_CPU_FLAG_AVX_SLOW;

            /* Zen 3 and earlier have slow gather */
            if ((out->family <= 0x19) && (flags & X86_CPU_FLAG_AVX2))
                flags |= X86_CPU_FLAG_AVX2_SLOWGATHER;
        }
    }

    if (!strncmp(out->vendor, "GenuineIntel", 12))
    {
        if (out->family == 6 && (out->model == 9 || out->model == 13 || out->model == 14))
        {
            if (flags & X86_CPU_FLAG_SSE2)
                flags |= X86_CPU_FLAG_SSE2_SLOW;

            if (flags & X86_CPU_FLAG_SSE3)
                flags |= X86_CPU_FLAG_SSE3_SLOW;
        }

        /* Conroe has a slow shuffle unit */
        if ((flags & X86_CPU_FLAG_SSSE3) && !(flags & X86_CPU_FLAG_SSE4) && out->family == 6 && out->model < 23)
            flags |= X86_CPU_FLAG_SSSE3_SLOW;

        /* Haswell has slow gather */
        if ((flags & X86_CPU_FLAG_AVX2) && out->family == 6 && out->model < 70)
            flags |= X86_CPU_FLAG_AVX2_SLOWGATHER;
    }

    // get cpu brand string
    for(int index = 0; index < 3; index++)
    {
      cpuid(0x80000002 + index, (int *)(out->name + 16*index));
    }

    out->flags = flags;
    make_extensions_string(out);
}
