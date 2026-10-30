/*
Copyright (c) 2015 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "runvx_plugin_loader.h"
#include "runvx_plugin_image.h"
#include "runvx_plugin_hash.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#if _WIN32
#include <Windows.h>
#define RUNVX_PLUGIN_IMAGE  "runvx_image_opencv.dll"
#define RUNVX_PLUGIN_HASH   "runvx_hash_openssl.dll"
#else
#include <dlfcn.h>
#include <linux/limits.h>
#include <unistd.h>
#define RUNVX_PLUGIN_IMAGE  "runvx_image_opencv.so"
#define RUNVX_PLUGIN_HASH   "runvx_hash_openssl.so"
#endif

static IRunVxImagePlugin * g_imagePlugin = nullptr;
static IRunVxHashPlugin  * g_hashPlugin  = nullptr;
static void * g_imageHandle = nullptr;
static void * g_hashHandle = nullptr;
static bool g_pluginsLoaded = false;

#if _WIN32
static void * RunVxDlOpen(const char * path) { return (void *)LoadLibraryA(path); }
static void   RunVxDlClose(void * handle) { FreeLibrary((HMODULE)handle); }
static void * RunVxDlSym(void * handle, const char * name) {
    return (void *)GetProcAddress((HMODULE)handle, name);
}
static void RunVxLogError() {
    DWORD err = GetLastError();
    char msg[256];
    DWORD len = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                 msg, sizeof(msg), nullptr);
    if (len) printf("runvx plugin loader: %s\n", msg);
    else printf("runvx plugin loader: Windows error %lu\n", err);
}
#else
static void * RunVxDlOpen(const char * path) { return dlopen(path, RTLD_LAZY); }
static void   RunVxDlClose(void * handle) { dlclose(handle); }
static void * RunVxDlSym(void * handle, const char * name) { return dlsym(handle, name); }
static void RunVxLogError() { const char * err = dlerror(); if (err) printf("runvx plugin loader: %s\n", err); }
#endif

char * RunVxGetPluginLibDirectory()
{
#if _WIN32
    HMODULE hModule = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                            (LPCSTR)&RunVxGetPluginLibDirectory, &hModule)) {
        return nullptr;
    }
    char exePath[MAX_PATH];
    DWORD len = GetModuleFileNameA(hModule, exePath, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) return nullptr;
    char * lastSlash = strrchr(exePath, '\\');
    if (lastSlash) *lastSlash = '\0';
    size_t pathLen = strlen(exePath) + 1 + strlen("..\\lib") + 1;
    char * dir = (char *)malloc(pathLen);
    if (dir) snprintf(dir, pathLen, "%s\\..\\lib", exePath);
    return dir;
#else
    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, PATH_MAX - 1);
    if (len <= 0 || len >= PATH_MAX) return nullptr;
    exePath[len] = '\0';
    char * lastSlash = strrchr(exePath, '/');
    if (lastSlash) *lastSlash = '\0';
    size_t pathLen = strlen(exePath) + 1 + strlen("../lib") + 1;
    char * dir = (char *)malloc(pathLen);
    if (dir) snprintf(dir, pathLen, "%s/../lib", exePath);
    return dir;
#endif
}

static bool RunVxTryLoadPlugin(const char * dir, const char * name,
                               const char * factoryName,
                               void ** outHandle, void ** outFactory)
{
#if _WIN32
    size_t pathLen = strlen(dir) + 1 + strlen(name) + 1;
    char * path = (char *)malloc(pathLen);
    if (!path) return false;
    snprintf(path, pathLen, "%s\\%s", dir, name);
#else
    size_t pathLen = strlen(dir) + 1 + strlen(name) + 1;
    char * path = (char *)malloc(pathLen);
    if (!path) return false;
    snprintf(path, pathLen, "%s/%s", dir, name);
#endif

    void * handle = RunVxDlOpen(path);
    if (!handle) {
        free(path);
        return false;
    }

    void * factory = RunVxDlSym(handle, factoryName);
    free(path);
    if (!factory) {
        RunVxDlClose(handle);
        RunVxLogError();
        return false;
    }

    *outHandle = handle;
    *outFactory = factory;
    return true;
}

static bool RunVxSearchPlugin(const char * pluginName, const char * factoryName, void ** outHandle, void ** outFactory)
{
    *outHandle = nullptr;
    *outFactory = nullptr;

    // 1. Library directory next to the runvx executable: ../lib
    //    This is where the install step places the plugin .so files when
    //    the MIVisionX runtime package is installed.
    char * libDir = RunVxGetPluginLibDirectory();
    if (libDir) {
        if (RunVxTryLoadPlugin(libDir, pluginName, factoryName, outHandle, outFactory)) {
            free(libDir);
            return true;
        }
        free(libDir);
    }

    // 2. Standard library search: dlopen by filename alone. This picks up
    //    LD_LIBRARY_PATH, /etc/ld.so.conf, and the runvx RPATH automatically.
    {
        void * handle = RunVxDlOpen(pluginName);
        if (handle) {
            void * factory = RunVxDlSym(handle, factoryName);
            if (factory) {
                *outHandle = handle;
                *outFactory = factory;
                return true;
            }
            RunVxDlClose(handle);
        }
    }

    // 3. Known install prefixes (in case the executable is moved or run from
    //    a build tree that does not match the install layout).
#if !_WIN32
    static const char * const s_knownLibDirs[] = {
        "/opt/rocm/lib",
        "/usr/local/lib",
        "/usr/lib",
        "/usr/lib64",
        nullptr
    };
    for (const char * const * dir = s_knownLibDirs; *dir; ++dir) {
        if (RunVxTryLoadPlugin(*dir, pluginName, factoryName, outHandle, outFactory)) {
            return true;
        }
    }
#endif

    // 4. Build-tree / developer fallback: the runvx_plugins directory next to
    //    the directory containing the runvx executable (common in build trees).
#if _WIN32
    {
        HMODULE hModule = nullptr;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                               (LPCSTR)&RunVxSearchPlugin, &hModule)) {
            char exePath[MAX_PATH];
            DWORD len = GetModuleFileNameA(hModule, exePath, MAX_PATH);
            if (len > 0 && len < MAX_PATH) {
                char * lastSlash = strrchr(exePath, '\\');
                if (lastSlash) *lastSlash = '\0';
                // try sibling of the binary directory: <exe_dir>\..\runvx_plugins
                size_t buildDirLen = strlen(exePath) + 1 + strlen("..\\runvx_plugins") + 1;
                char * buildDir = (char *)malloc(buildDirLen);
                if (buildDir) {
                    snprintf(buildDir, buildDirLen, "%s\\..\\runvx_plugins", exePath);
                    if (RunVxTryLoadPlugin(buildDir, pluginName, factoryName, outHandle, outFactory)) {
                        free(buildDir);
                        return true;
                    }
                    free(buildDir);
                }
                // also try the binary directory itself
                if (RunVxTryLoadPlugin(exePath, pluginName, factoryName, outHandle, outFactory)) {
                    return true;
                }
            }
        }
    }
#else
    {
        char exePath[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", exePath, PATH_MAX - 1);
        if (len > 0 && len < PATH_MAX) {
            exePath[len] = '\0';
            char * lastSlash = strrchr(exePath, '/');
            if (lastSlash) *lastSlash = '\0';
            // try sibling of the binary directory: <exe_dir>/../runvx_plugins
            size_t buildDirLen = strlen(exePath) + 1 + strlen("../runvx_plugins") + 1;
            char * buildDir = (char *)malloc(buildDirLen);
            if (buildDir) {
                snprintf(buildDir, buildDirLen, "%s/../runvx_plugins", exePath);
                if (RunVxTryLoadPlugin(buildDir, pluginName, factoryName, outHandle, outFactory)) {
                    free(buildDir);
                    return true;
                }
                free(buildDir);
            }
            // also try the binary directory itself
            if (RunVxTryLoadPlugin(exePath, pluginName, factoryName, outHandle, outFactory)) {
                return true;
            }
        }
    }
#endif

    return false;
}

void RunVxLoadPlugins()
{
    if (g_pluginsLoaded) return;
    g_pluginsLoaded = true;

    {
        void * factory = nullptr;
        if (RunVxSearchPlugin(RUNVX_PLUGIN_IMAGE, RUNVX_IMAGE_PLUGIN_FACTORY,
                              &g_imageHandle, &factory)) {
            g_imagePlugin = ((RunVxCreateImagePluginFn)factory)();
            if (g_imagePlugin)
                printf("OK: loaded runvx image plugin\n");
            else {
                RunVxDlClose(g_imageHandle);
                g_imageHandle = nullptr;
            }
        }
    }

    {
        void * factory = nullptr;
        if (RunVxSearchPlugin(RUNVX_PLUGIN_HASH, RUNVX_HASH_PLUGIN_FACTORY,
                              &g_hashHandle, &factory)) {
            g_hashPlugin = ((RunVxCreateHashPluginFn)factory)();
            if (g_hashPlugin)
                printf("OK: loaded runvx hash plugin\n");
            else {
                RunVxDlClose(g_hashHandle);
                g_hashHandle = nullptr;
            }
        }
    }
}

IRunVxImagePlugin * RunVxGetImagePlugin() { return g_imagePlugin; }
IRunVxHashPlugin  * RunVxGetHashPlugin()  { return g_hashPlugin; }

void RunVxUnloadPlugins()
{
    if (g_imagePlugin) {
        delete g_imagePlugin;
        g_imagePlugin = nullptr;
    }
    if (g_hashPlugin) {
        delete g_hashPlugin;
        g_hashPlugin = nullptr;
    }
    if (g_imageHandle) {
        RunVxDlClose(g_imageHandle);
        g_imageHandle = nullptr;
    }
    if (g_hashHandle) {
        RunVxDlClose(g_hashHandle);
        g_hashHandle = nullptr;
    }
    g_pluginsLoaded = false;
}
