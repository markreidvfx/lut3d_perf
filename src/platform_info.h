#ifndef PLATFORM_INFO_H
#define PLATFORM_INFO_H

#include <stdint.h>

#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

#if defined(__x86_64__) || defined(_M_X64) || defined(i386) || defined(_M_IX86)
#define ARCH_X86
    #if defined(__x86_64__) || defined(_M_X64)
        #define CPU_ARCH "x86_64"
    #else
        #define CPU_ARCH "x86"
    #endif
#elif defined(__aarch64__)
    #define CPU_ARCH "ARM 64-bit"
#elif defined(__arm__)
    #define CPU_ARCH "ARM"
#elif defined(__PPC64__)
    #define CPU_ARCH "PowerPC 64-bit"
#elif defined(__PPC__)
    #define CPU_ARCH "PowerPC"
#else
    #define CPU_ARCH "Unknown CPU architecture"
#endif

#if defined(__clang_major__) && defined(__clang_minor__) && defined(__clang_patchlevel__)
#define COMPILER_NAME "clang " \
    STRINGIFY(__clang_major__) "." STRINGIFY(__clang_minor__) "." STRINGIFY(__clang_patchlevel__)
#elif defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__)
#define COMPILER_NAME "gnu gcc " \
    STRINGIFY(__GNUC__) "." STRINGIFY(__GNUC_MINOR__) "." STRINGIFY(__GNUC_PATCHLEVEL__)
#elif defined(_MSC_VER)
#define COMPILER_NAME "MSVC " STRINGIFY(_MSC_VER)
#else
#define COMPILER_NAME "Unknown Compiler"
#endif

#if defined(_WIN32)
  #define PLATFORM_NAME "Windows"
#elif defined(__APPLE__) && defined(__MACH__)
  #include <TargetConditionals.h>
  #if TARGET_IPHONE_SIMULATOR == 1
    #define PLATFORM_NAME "iOS Simulator"
  #elif TARGET_OS_IPHONE == 1
    #define PLATFORM_NAME "iOS"
  #elif TARGET_OS_MAC == 1
    #define PLATFORM_NAME "macOS"
  #else
    #define PLATFORM_NAME "Unknown Apple platform"
  #endif
#elif defined(__ANDROID__)
  #define PLATFORM_NAME "Android"
#elif defined(__linux__)
  #define PLATFORM_NAME "Linux"
#elif defined(__FreeBSD__)
  #define PLATFORM_NAME "FreeBSD"
#elif defined(__unix__) || defined(__unix)
  #define PLATFORM_NAME "Unix"
#else
  #define PLATFORM_NAME "Unknown platform"
#endif

#if _WIN32
#include <windows.h>
#include <intrin.h>
#define strdup _strdup

static inline uint64_t get_timer_frequency()
{
    LARGE_INTEGER Result;
    QueryPerformanceFrequency(&Result);
    return Result.QuadPart;
}
static inline uint64_t get_timer(void)
{
    LARGE_INTEGER Result;
    QueryPerformanceCounter(&Result);
    return Result.QuadPart;
}
#else
#include <time.h>
#include <unistd.h>
#include <sys/utsname.h>
#if __APPLE__
#include <sys/sysctl.h>
#endif

static inline uint64_t get_timer_frequency()
{
    uint64_t Result = 1000000000ull;
    return Result;
}
static inline uint64_t get_timer(void)
{
    struct timespec Spec;
    clock_gettime(CLOCK_MONOTONIC, &Spec);
    uint64_t Result = ((uint64_t)Spec.tv_sec * 1000000000ull) + (uint64_t)Spec.tv_nsec;
    return Result;
}

#endif

const char * get_cpu_model_name();
const char * get_platform_name();


#endif // PLATFORM_INFO_H