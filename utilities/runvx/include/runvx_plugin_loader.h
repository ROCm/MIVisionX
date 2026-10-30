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

#ifndef __RUNVX_PLUGIN_LOADER_H__
#define __RUNVX_PLUGIN_LOADER_H__

class IRunVxImagePlugin;
class IRunVxHashPlugin;

// Initialize the plugin loader and discover available plugins.
// Safe to call multiple times; plugins are loaded on first call.
void RunVxLoadPlugins();

// Clean up loaded plugins and close shared library handles.
// Safe to call multiple times; handles are closed on first call.
void RunVxUnloadPlugins();

// Get the loaded image plugin, or nullptr if not available.
IRunVxImagePlugin * RunVxGetImagePlugin();

// Get the loaded hash plugin, or nullptr if not available.
IRunVxHashPlugin * RunVxGetHashPlugin();

// Formatted library directory path next to the runvx executable.
// The returned buffer must be freed by the caller.
char * RunVxGetPluginLibDirectory();

#endif /* __RUNVX_PLUGIN_LOADER_H__ */
