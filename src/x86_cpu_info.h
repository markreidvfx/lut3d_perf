#ifndef X86_CPU_INFO_H
#define X86_CPU_INFO_H

#define X86_CPU_FLAG_SSE2             (1 << 0) // SSE2 functions
#define X86_CPU_FLAG_SSE2_SLOW        (1 << 1) // SSE2 supported, but usually not faster than regular MMX/SSE (e.g. Core1)

#define X86_CPU_FLAG_SSE3             (1 << 2) // Prescott SSE3 functions
#define X86_CPU_FLAG_SSE3_SLOW        (1 << 3) // SSE3 supported, but usually not faster than regular MMX/SSE (e.g. Core1)

#define X86_CPU_FLAG_SSSE3            (1 << 4) // Conroe SSSE3 functions
#define X86_CPU_FLAG_SSSE3_SLOW       (1 << 5) // SSSE3 supported, but usually not faster than SSE2

#define X86_CPU_FLAG_SSE4             (1 << 6) // Penryn SSE4.1 functions
#define X86_CPU_FLAG_SSE42            (1 << 7) // Nehalem SSE4.2 functions

#define X86_CPU_FLAG_AVX              (1 << 8) // AVX functions: requires OS support even if YMM registers aren't used
#define X86_CPU_FLAG_AVX_SLOW         (1 << 9) // AVX supported, but slow when using YMM registers (e.g. Bulldozer)

#define X86_CPU_FLAG_AVX2            (1 << 10) // AVX2 functions: requires OS support even if YMM registers aren't used
#define X86_CPU_FLAG_AVX2_SLOWGATHER (1 << 11) // CPU has slow gathers.

#define X86_CPU_FLAG_AVX512          (1 << 12) // AVX-512 functions: requires OS support even if YMM/ZMM registers aren't used

#define X86_CPU_FLAG_F16C            (1 << 13) // CPU Has FP16C half float, AVX2 should always have this??

typedef struct CPUInfo
{
    unsigned int flags;
    int family;
    int model;
    char name[65];
    char vendor[13];
    char extensions[128];
} CPUInfo;

void get_cpu_info(CPUInfo *out);

#endif // X86_CPU_INFO_H
