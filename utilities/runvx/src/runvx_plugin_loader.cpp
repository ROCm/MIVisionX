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
#include <string>
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
static bool g_pluginsLoaded = false;

#if _WIN32
static void * RunVxDlOpen(const char * path) { return (void *)LoadLibraryA(path); }
static void   RunVxDlClose(void * handle) { FreeLibrary((HMODULE)handle); }
static void * RunVxDlSym(void * handle, const char * name) {
    return (void *)GetProcAddress((HMODULE)handle, name);
}
static void RunVxLogError() { /* Windows error omitted for brevity */ }
#else
static void * RunVxDlOpen(const char * path) { return dlopen(path, RTLD_LAZY); }
static void   RunVxDlClose(void * handle) { dlclose(handle); }
static void * RunVxDlSym(void * handle, const char * name) { return dlsym(handle, name); }
static void RunVxLogError() { const char * err = dlerror(); if (err) printf("runvx plugin loader: %s\n", err); }
#endif

char * RunVxGetPluginDirectory()
{
#if _WIN32
    HMODULE hModule = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                            (LPCSTR)&RunVxGetPluginDirectory, &hModule)) {
        return nullptr;
    }
    char exePath[MAX_PATH];
    DWORD len = GetModuleFileNameA(hModule, exePath, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) return nullptr;
    char * lastSlash = strrchr(exePath, '\\');
    if (lastSlash) *lastSlash = '\0';
    size_t pathLen = strlen(exePath) + 1 + strlen("..\\share\\mivisionx\\runvx_plugins") + 1;
    char * dir = (char *)malloc(pathLen);
    if (dir) snprintf(dir, pathLen, "%s\\..\\share\\mivisionx\\runvx_plugins", exePath);
    return dir;
#else
    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, PATH_MAX - 1);
    if (len <= 0 || len >= PATH_MAX) return nullptr;
    exePath[len] = '\0';
    char * lastSlash = strrchr(exePath, '/');
    if (lastSlash) *lastSlash = '\0';
    size_t pathLen = strlen(exePath) + 1 + strlen("../share/mivisionx/runvx_plugins") + 1;
    char * dir = (char *)malloc(pathLen);
    if (dir) snprintf(dir, pathLen, "%s/../share/mivisionx/runvx_plugins", exePath);
    return dir;
#endif
}

static bool RunVxTryLoadPlugin(const char * dir, const char * name,
                               void ** outHandle, void ** outFactory)
{
    size_t pathLen = strlen(dir) + 1 + strlen(name) + 1;
    char * path = (char *)malloc(pathLen);
    if (!path) return false;
    snprintf(path, pathLen, "%s/%s", dir, name);

    void * handle = RunVxDlOpen(path);
    if (!handle) {
        free(path);
        return false;
    }

    void * factory = RunVxDlSym(handle, RUNVX_IMAGE_PLUGIN_FACTORY);
    if (!factory) factory = RunVxDlSym(handle, RUNVX_HASH_PLUGIN_FACTORY);
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

static bool RunVxSearchPlugin(const char * pluginName, void ** outFactory)
{
    *outFactory = nullptr;

    // 1. RUNVX_PLUGIN_PATH environment variable (colon-separated on Linux)
    const char * envPath = getenv("RUNVX_PLUGIN_PATH");
    if (envPath && envPath[0]) {
        char * paths = strdup(envPath);
        char * savePtr = nullptr;
#if _WIN32
        const char * sep = ";";
#else
        const char * sep = ":";
#endif
        for (char * dir = strtok_r(paths, sep, &savePtr); dir; dir = strtok_r(nullptr, sep, &savePtr)) {
            void * handle = nullptr;
            if (RunVxTryLoadPlugin(dir, pluginName, &handle, outFactory)) {
                free(paths);
                return true;
            }
        }
        free(paths);
    }

    // 2. Directory next to the runvx executable: ../share/mivisionx/runvx_plugins
    char * pluginDir = RunVxGetPluginDirectory();
    if (pluginDir) {
        void * handle = nullptr;
        if (RunVxTryLoadPlugin(pluginDir, pluginName, &handle, outFactory)) {
            free(pluginDir);
            return true;
        }
        free(pluginDir);
    }

    // 3. Build-tree / developer fallback: the runvx_plugins directory next to
    //    the directory containing the runvx executable (common in build trees).
#if !_WIN32
    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, PATH_MAX - 1);
    if (len > 0 && len < PATH_MAX) {
        exePath[len] = '\0';
        char * lastSlash = strrchr(exePath, '/');
        if (lastSlash) *lastSlash = '\0';
        // try sibling of the binary directory
        size_t buildDirLen = strlen(exePath) + 1 + strlen("../runvx_plugins") + 1;
        char * buildDir = (char *)malloc(buildDirLen);
        if (buildDir) {
            snprintf(buildDir, buildDirLen, "%s/../runvx_plugins", exePath);
            void * handle = nullptr;
            if (RunVxTryLoadPlugin(buildDir, pluginName, &handle, outFactory)) {
                free(buildDir);
                return true;
            }
            free(buildDir);
        }
        // also try the binary directory itself
        void * handle = nullptr;
        if (RunVxTryLoadPlugin(exePath, pluginName, &handle, outFactory)) {
            return true;
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
        if (RunVxSearchPlugin(RUNVX_PLUGIN_IMAGE, &factory)) {
            g_imagePlugin = ((RunVxCreateImagePluginFn)factory)();
            if (g_imagePlugin)
                printf("OK: loaded runvx image plugin\n");
        }
    }

    {
        void * factory = nullptr;
        if (RunVxSearchPlugin(RUNVX_PLUGIN_HASH, &factory)) {
            g_hashPlugin = ((RunVxCreateHashPluginFn)factory)();
            if (g_hashPlugin)
                printf("OK: loaded runvx hash plugin\n");
        }
    }
}

IRunVxImagePlugin * RunVxGetImagePlugin() { return g_imagePlugin; }
IRunVxHashPlugin  * RunVxGetHashPlugin()  { return g_hashPlugin; }
