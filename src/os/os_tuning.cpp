#include "os_tuning.h"
#include <iostream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <timeapi.h>
#include <avrt.h>

#elif defined(__linux__)
#include <sched.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <unistd.h>

#elif defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

namespace diffusion {
namespace os {

#if defined(_WIN32)
static bool g_timer_tuned = false;
static HANDLE g_mmcss_handle = nullptr;

bool tune_process_for_inference() {
    // 1. High-resolution 1ms timer period
    if (timeBeginPeriod(1) == TIMERR_NOERROR) {
        g_timer_tuned = true;
    }

    // 2. High Process & Thread Priority Class
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

    // 3. Multimedia Pro Audio / Low Latency Scheduler via Avrt
    DWORD task_index = 0;
    HMODULE avrt = LoadLibraryW(L"avrt.dll");
    if (avrt) {
        typedef HANDLE(WINAPI* PAvSetMmThreadCharacteristicsW)(LPCWSTR, LPDWORD);
        PAvSetMmThreadCharacteristicsW pAvSet =
            (PAvSetMmThreadCharacteristicsW)GetProcAddress(avrt, "AvSetMmThreadCharacteristicsW");
        if (pAvSet) {
            g_mmcss_handle = pAvSet(L"Pro Audio", &task_index);
        }
    }

    // 4. Pin thread to Performance (P) cores if hybrid architecture exists
    DWORD_PTR process_affinity = 0;
    DWORD_PTR system_affinity = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_affinity, &system_affinity)) {
        // Favor even-numbered logical processors (first physical thread per core)
        DWORD_PTR p_core_mask = process_affinity & 0x5555555555555555ULL;
        if (p_core_mask != 0) {
            SetThreadAffinityMask(GetCurrentThread(), p_core_mask);
        }
    }

    return true;
}

void restore_process_tuning() {
    if (g_mmcss_handle) {
        HMODULE avrt = GetModuleHandleW(L"avrt.dll");
        if (avrt) {
            typedef BOOL(WINAPI* PAvRevertMmThreadCharacteristics)(HANDLE);
            PAvRevertMmThreadCharacteristics pAvRevert =
                (PAvRevertMmThreadCharacteristics)GetProcAddress(avrt, "AvRevertMmThreadCharacteristics");
            if (pAvRevert) {
                pAvRevert(g_mmcss_handle);
            }
        }
        g_mmcss_handle = nullptr;
    }
    if (g_timer_tuned) {
        timeEndPeriod(1);
        g_timer_tuned = false;
    }
    SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
}

#elif defined(__linux__)

bool tune_process_for_inference() {
    // 1. Attempt Realtime FIFO scheduling; fallback to max nice priority (-20)
    struct sched_param sp;
    sp.sched_priority = 20;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
        setpriority(PRIO_PROCESS, 0, -20);
    }

    // 2. Advise huge pages on heap
#ifdef MADV_HUGEPAGE
    // Suggest kernel allocate 2MB transparent huge pages
    madvise(nullptr, 0, MADV_HUGEPAGE);
#endif

    return true;
}

void restore_process_tuning() {
    struct sched_param sp;
    sp.sched_priority = 0;
    sched_setscheduler(0, SCHED_OTHER, &sp);
    setpriority(PRIO_PROCESS, 0, 0);
}

#elif defined(__APPLE__)

bool tune_process_for_inference() {
    // Elevate macOS thread quality-of-service to user interactive
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    return true;
}

void restore_process_tuning() {
    pthread_set_qos_class_self_np(QOS_CLASS_DEFAULT, 0);
}

#else

bool tune_process_for_inference() {
    return false;
}

void restore_process_tuning() {}

#endif

} // namespace os
} // namespace diffusion
