/* 
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#include "ago_platform.h"

static void agoCpuid(int out[4], int leaf, int subleaf)
{
#if _WIN32
	__cpuidex(out, leaf, subleaf);
#else
	// `volatile` + "memory" clobber: cpuid is an ordered hardware query
	// (its outputs depend on global CPU state that other code paths can
	// touch via wrmsr/cpuid serialization), so we must prevent the
	// compiler from reordering it across surrounding ops or merging
	// adjacent identical-input invocations.
	asm volatile("cpuid"
		: "=a" (out[0]), "=b" (out[1]), "=c" (out[2]), "=d" (out[3])
		: "a" (leaf), "c" (subleaf)
		: "memory");
#endif
}

static uint64_t agoXgetbv(uint32_t index)
{
#if _WIN32
	return _xgetbv(index);
#else
	uint32_t eax, edx;
	// `volatile` + "memory" clobber: xgetbv reads the OS-controlled XCR0
	// (AVX/AVX2/AVX-512 enablement bits). Treat it like cpuid above —
	// the value is hardware/OS state we don't want the optimizer to
	// reorder around any nearby feature-detection logic.
	asm volatile("xgetbv"
		: "=a" (eax), "=d" (edx)
		: "c" (index)
		: "memory");
	return ((uint64_t)edx << 32) | eax;
#endif
}

#if _WIN32 && ENABLE_OPENCL
#pragma comment(lib, "OpenCL.lib")
#endif

// Mirror the USE_AVX / USE_FMA / USE_BMI2 compile-time switches from ago_internal.h
// so the hardware-support check below can reflect the *actual* minimum ISA
// emitted by this binary. Both ago_internal.h and the block here use the
// same `#ifndef … #define … 1 … #endif` guard, so a single
// `-DUSE_AVX=0` / `-DUSE_BMI2=0` on the CMake line propagates consistently
// to every translation unit (this file is compiled before ago_internal.h
// is included down below in the !_WIN32 section).
#ifndef USE_AVX
#define USE_AVX 1
#endif
#ifndef USE_FMA
#define USE_FMA 1
#endif
#ifndef USE_BMI2
#define USE_BMI2 1
#endif

bool agoIsCpuHardwareSupported()
{
	// Refuse to come up on a CPU/OS that can't actually execute the
	// instruction set the binary was compiled to emit. Previously this
	// only checked SSE4.2, so on a SSE4.2-only Nehalem-class CPU we
	// would happily create a vx_context and then SIGILL on the first
	// AVX2 kernel invocation. With USE_AVX/USE_FMA/USE_BMI2 enabled at
	// compile time, the corresponding runtime feature bits are now hard
	// preconditions. f.avx2 already AND-folds the OSXSAVE/XCR0 check
	// in agoGetCpuFeatures() so an OS that disabled AVX state save also
	// correctly fails this gate.
	const ago_cpu_features_t & f = agoGetCpuFeatures();
	if (!f.sse42) return false;
#if USE_AVX
	if (!f.avx2) return false;
#endif
#if USE_FMA
	if (!f.fma) return false;
#endif
#if USE_BMI2
	if (!f.bmi2) return false;
#endif
	return true;
}

const ago_cpu_features_t & agoGetCpuFeatures()
{
	// C++17 guarantees thread-safe initialization of function-local statics,
	// so the cpuid probe runs exactly once even under concurrent first calls.
	static const ago_cpu_features_t features = []() {
		ago_cpu_features_t f = {};
		int CPUInfo[4] = { 0 };
		agoCpuid(CPUInfo, 0, 0);
		int maxLeaf = CPUInfo[0];

		bool osAvx = false;
		if (maxLeaf >= 1) {
			agoCpuid(CPUInfo, 1, 0);
			f.sse42 = (CPUInfo[2] & (1 << 20)) != 0;
			bool cpuAvx = (CPUInfo[2] & (1 << 28)) != 0;
			bool cpuFma = (CPUInfo[2] & (1 << 12)) != 0;
			bool osxsave = (CPUInfo[2] & (1 << 27)) != 0;
			if (cpuAvx && osxsave) {
				uint64_t xcr0 = agoXgetbv(0);
				osAvx = (xcr0 & 0x6) == 0x6;
				f.avx = osAvx;
				// FMA3 operates on YMM registers, so it requires the same
				// OS AVX-state-save support as AVX itself.
				f.fma = osAvx && cpuFma;
			}
		}

		if (maxLeaf >= 7) {
			agoCpuid(CPUInfo, 7, 0);
			f.avx2 = osAvx && ((CPUInfo[1] & (1 << 5)) != 0);
			f.bmi2 = (CPUInfo[1] & (1 << 8)) != 0;

			uint64_t xcr0 = osAvx ? agoXgetbv(0) : 0;
			bool osAvx512 = osAvx && ((xcr0 & 0xe0) == 0xe0);
			f.avx512f = osAvx512 && ((CPUInfo[1] & (1 << 16)) != 0);
			f.avx512dq = osAvx512 && ((CPUInfo[1] & (1 << 17)) != 0);
			f.avx512bw = osAvx512 && ((CPUInfo[1] & (1 << 30)) != 0);
			f.avx512vl = osAvx512 && ((CPUInfo[1] & (1 << 31)) != 0);
		}
		return f;
	}();
	return features;
}

uint32_t agoControlFpSetRoundEven()
{
	uint32_t state;
#if _WIN32
	state = _controlfp(0, 0);
	_controlfp(_RC_NEAR, _MCW_RC); // round to nearest even: RC_CHOP gives matching output with sample code
	return state;
#else
	state = fegetround();
	fesetround(FE_TONEAREST);
#endif
	return state;
}

void agoControlFpReset(uint32_t state)
{
#if _WIN32
	_controlfp(state, _MCW_RC);
#else
	fesetround(state);
#endif
}

bool agoGetEnvironmentVariable(const char * name, char * value, size_t valueSize)
{
#if _WIN32
	DWORD len = GetEnvironmentVariableA(name, value, (DWORD)valueSize);
	value[valueSize-1] = 0;
	return (len > 0) ? true : false;
#else
	const char * v = getenv(name);
	if (v) {
		strncpy(value, v, valueSize);
		value[valueSize-1] = 0;
	}
	return v ? true : false;
#endif
}

bool agoSetEnvironmentVariable(const char * name, const char * value)
{
#if _WIN32
    return SetEnvironmentVariableA(name, value);
#else
    return !(setenv(name, value, 1));
#endif
}

bool agoUnsetEnvironmentVariable(const char * name)
{
#if _WIN32
    return SetEnvironmentVariableA(name, NULL);
#else
    return !(unsetenv(name));
#endif
}

ago_module agoOpenModule(const char * libFileName)
{
#if _WIN32
	return (ago_module)LoadLibraryA(libFileName);
#else
	return (ago_module) dlopen(libFileName, RTLD_NOW | RTLD_LOCAL);
#endif
}

void * agoGetFunctionAddress(ago_module module, const char * functionName)
{
#if _WIN32
	return GetProcAddress((HMODULE)module, functionName);
#else
	return dlsym(module, functionName);
#endif
}

void agoCloseModule(ago_module module)
{
#if _WIN32
	FreeLibrary((HMODULE)module);
#else
	dlclose(module);
#endif
}

int64_t agoGetClockCounter()
{
#if _WIN32
	LARGE_INTEGER v;
	QueryPerformanceCounter(&v);
	return v.QuadPart;
#else
	return chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
}

int64_t agoGetClockFrequency()
{
#if _WIN32
	LARGE_INTEGER v;
	QueryPerformanceFrequency(&v);
	return v.QuadPart;
#else
	return chrono::high_resolution_clock::period::den / chrono::high_resolution_clock::period::num;
#endif
}

#if !_WIN32
#include "ago_internal.h"

#include <mutex>
#include <condition_variable>
#include <fenv.h>
#include <dlfcn.h>

#define VX_SEMAPHORE    1
#define VX_THREAD       2
#define VX_CRITICAL_SECTION       3

typedef struct {
	int type; // should be VX_SEMAPHORE
	int count;
	mutex mtx;
	condition_variable cv;
} vx_semaphore;

typedef struct {
    int type;   // should be VX_THREAD
    thread thread_obj;
    void* thread_param;
} vx_thread;

typedef struct {
    int type;   // should be VX_CRITICAL_SECTION
    mutex mtx;
} vx_critical_section;


// Emulates EnterCriticalSection for non_windows platform
void EnterCriticalSection(CRITICAL_SECTION* cs)
{
    vx_critical_section * crit_sec = (vx_critical_section *)*cs;
    std::lock_guard<std::mutex> lock(crit_sec->mtx);
}

// Emulates LeaveCriticalSection for non_windows platform
void LeaveCriticalSection(CRITICAL_SECTION* cs)
{
    vx_critical_section * crit_sec = (vx_critical_section *)*cs;
    crit_sec->mtx.unlock();
}

// Emulates InitializeCriticalSection for non_windows platform
void InitializeCriticalSection(CRITICAL_SECTION* cs)
{
    vx_critical_section *crit_sec = new vx_critical_section;
    crit_sec->type = VX_CRITICAL_SECTION;
    *cs = crit_sec;
}

// Emulates DeleteCriticalSection for non_windows platform
void DeleteCriticalSection(CRITICAL_SECTION* cs)
{
    vx_critical_section * crit_sec = (vx_critical_section *)*cs;
    crit_sec->type = 0;
    delete crit_sec;
}

HANDLE CreateSemaphore(void *, LONG, LONG, void *)
{
	vx_semaphore * sem = new vx_semaphore;
	sem->type = VX_SEMAPHORE;
	sem->count = 0;
	return sem;
}

HANDLE CreateThread(void *, size_t dwStackSize, LPTHREAD_START_ROUTINE lpStartAddress, LPVOID lpParameter, DWORD dwCreationFlags, void *)
{
    vx_thread *thd = new vx_thread;
    thd->type = VX_THREAD;
    thd->thread_obj = thread(lpStartAddress, lpParameter);
    return thd;
}

void CloseHandle(HANDLE h)
{
	if(h) {
		if(*(int*)h == VX_SEMAPHORE) {
			vx_semaphore * sem = (vx_semaphore *)h;
			sem->type = 0;
			delete sem;
		}
		else if(*(int*)h == VX_THREAD) {
            vx_thread * th = (vx_thread *)h;
            th->type = 0;
            th->thread_obj.join();
            delete th;
        }
	}
}
DWORD WaitForSingleObject(HANDLE h, DWORD dwMilliseconds)
{
	if(h) {
		if(*(int*)h == VX_SEMAPHORE) {
			vx_semaphore * sem = (vx_semaphore *)h;
			{
				unique_lock<mutex> lk(sem->mtx);
				// Wait only if the semaphore count is currently zero; otherwise
				// a notification that arrived before this wait would be lost.
				sem->cv.wait(lk, [&sem]() { return sem->count > 0; });
				sem->count--;
			}
		}
    } else
    {
        printf("Invalid Handle for WaitObject\n");
        return -1;
    }
	return 0;
}

BOOL ReleaseSemaphore(HANDLE h, LONG lReleaseCount, LPLONG lpPreviousCount)
{
	if(h) {
		if(*(int*)h == VX_SEMAPHORE) {
			vx_semaphore * sem = (vx_semaphore *)h;
			{
				lock_guard<mutex> lk(sem->mtx);
				if(lpPreviousCount) *lpPreviousCount = sem->count;
				sem->count += lReleaseCount;
			}
			for(LONG i = 0; i < lReleaseCount; i++) {
				sem->cv.notify_one();
			}
		}
    } else
    {
        printf("Invalid Handle for Semaphore\n");
        return 0;
    }
    return 1;
}

#endif
