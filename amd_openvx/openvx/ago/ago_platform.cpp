/* 
Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
 
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

// macro to port VisualStudio __cpuid to g++
#if !_WIN32
#define __cpuid(out, infoType) asm("cpuid": "=a" (out[0]), "=b" (out[1]), "=c" (out[2]), "=d" (out[3]): "a" (infoType));
#endif

#if _WIN32 && ENABLE_OPENCL
#pragma comment(lib, "OpenCL.lib")
#endif

bool agoIsCpuHardwareSupported()
{
	bool isHardwareSupported = false;
	int CPUInfo[4] = { -1 };
	__cpuid(CPUInfo, 0);
	if (CPUInfo[0] > 1) {
		__cpuid(CPUInfo, 1);
		// check for SSE4.2 support
		if (CPUInfo[2] & 0x100000)
			isHardwareSupported = true;
	}
	return isHardwareSupported;
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

typedef struct {
	int type; // should be VX_SEMAPHORE
	int count;
	mutex mtx;
	condition_variable cv;
} vx_semaphore;

// TBD
void EnterCriticalSection(CRITICAL_SECTION cs)
{
}
void LeaveCriticalSection(CRITICAL_SECTION cs)
{
}
void InitializeCriticalSection(CRITICAL_SECTION cs)
{
}
void DeleteCriticalSection(CRITICAL_SECTION cs)
{
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
	return nullptr;
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
			// TBD
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
				sem->cv.wait(lk); // TBD: implement with timeout
			}
			{
				lock_guard<mutex> lk(sem->mtx);
				sem->count--;
			}
		}
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
	}
	return 0;
}

#endif
