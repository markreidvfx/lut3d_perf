#include "platform_info.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

#if defined(ARCH_X86)
#include "x86_cpu_info.h"
#endif

#define MAX_BUF 1024
static char PLATFORM_NAME_BUFFER[MAX_BUF];
static char CPU_MODEL_NAME[MAX_BUF];

#if _WIN32

typedef BOOL (WINAPI * RtlGetVersion_FUNC) (OSVERSIONINFOEXW *);

const char * get_platform_name()
{
    HMODULE hMod;
    RtlGetVersion_FUNC RtlGetVersion;
    OSVERSIONINFOEXW osw;

    sprintf(PLATFORM_NAME_BUFFER, "%s", PLATFORM_NAME);

    hMod = LoadLibrary(TEXT("ntdll.dll"));
    if (hMod) {
        RtlGetVersion = (RtlGetVersion_FUNC)GetProcAddress(hMod, "RtlGetVersion");
        if (RtlGetVersion) {
            ZeroMemory(&osw, sizeof(osw));
            osw.dwOSVersionInfoSize = sizeof(osw);
            if (RtlGetVersion(&osw) == 0) {
                sprintf(PLATFORM_NAME_BUFFER, "Windows " CPU_ARCH " %d.%d", osw.dwMajorVersion, osw.dwMinorVersion);
            }
        }
        FreeLibrary(hMod);
    }

    return PLATFORM_NAME_BUFFER;
}

#else
const char * get_platform_name()
{
#if __APPLE__
    size_t size = MAX_BUF;
    char product_version[MAX_BUF] = {0};
    if (sysctlbyname("kern.osproductversion", product_version, &size, NULL, 0) < 0) {
        return PLATFORM_NAME;
    }
    sprintf(PLATFORM_NAME_BUFFER, "%s %s", PLATFORM_NAME, product_version);
#else
    struct utsname info;
    if (uname(&info) != 0) {
        perror("Failed to get system information");
        return PLATFORM_NAME;
    }
    sprintf(PLATFORM_NAME_BUFFER, "%s %s %s %s", info.sysname, info.release, info.version, info.machine);
#endif
    return PLATFORM_NAME_BUFFER;
}

#endif

const char * get_cpu_model_name()
{
#if defined(ARCH_X86)
    CPUInfo info = {0};
    get_cpu_info(&info);
    sprintf(CPU_MODEL_NAME, "%.*s %.*s", 65, info.name, 128, info.extensions);
#elif __APPLE__
    size_t size = MAX_BUF;
    if (sysctlbyname("machdep.cpu.brand_string", CPU_MODEL_NAME, &size, NULL, 0) < 0) {
        return CPU_ARCH;
    }
#else
    static char BUFFER[MAX_BUF];
    sprintf(CPU_MODEL_NAME, "%s", CPU_ARCH);
    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        while (fgets(BUFFER, MAX_BUF, fp) != NULL) {

            if (sscanf(BUFFER, "model name : %[^\n]", CPU_MODEL_NAME) == 1) {
                break;
            }

            if (sscanf(BUFFER, "Model : %[^\n]", CPU_MODEL_NAME) == 1) {
                break;
            }
        }
    fclose(fp);
    }
#endif
    return CPU_MODEL_NAME;
}