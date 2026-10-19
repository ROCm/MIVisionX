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

#ifndef __RUNVX_PLUGIN_IMAGE_H__
#define __RUNVX_PLUGIN_IMAGE_H__

#include <cstdint>
#include <stddef.h>
#include <vector>

// Image/Video I/O, display, and overlay plugin interface for runvx.
// The core runvx binary loads the plugin at runtime using dlopen/dlsym.
// Implementations live in separate shared libraries (e.g., runvx_image_opencv.so).

// Type codes passed through the interface are VX_DF_IMAGE_* fourcc values,
// interpreted by the plugin implementation.

enum RunVxOverlayKind {
    RUNVX_OVERLAY_TEXT,
    RUNVX_OVERLAY_RECT,
    RUNVX_OVERLAY_FILLED_RECT,
    RUNVX_OVERLAY_HISTOGRAM,
    RUNVX_OVERLAY_CIRCLE
};

struct RunVxOverlayItem {
    RunVxOverlayKind kind;
    int x;
    int y;
    int w;
    int h;
    uint32_t color;              // 0xRRGGBB
    const char * text;           // valid for TEXT
    const uint32_t * histValues; // valid for HISTOGRAM
    int histNumBins;             // valid for HISTOGRAM
    uint32_t histMaxValue;       // valid for HISTOGRAM
};

class IRunVxImagePlugin {
public:
    virtual ~IRunVxImagePlugin() = default;

    // Read a single image file into a contiguous buffer.
    // 'type' is a VX_DF_IMAGE_* value the plugin can produce.
    // Returns true on success.
    virtual bool ReadImageFile(const char * path, int & width, int & height, int & type,
                               std::vector<uint8_t> & data) = 0;

    // Open a camera device or video file for multi-frame capture.
    // Returns an opaque capture handle on success, nullptr on failure.
    virtual void * OpenCapture(int cameraDevice) = 0;
    virtual void * OpenCapture(const char * fileName) = 0;

    // Read the next frame from an opened capture handle.
    // Returns true while frames are available; false on EOF or error.
    // The returned buffer is in the requested/requested-compatible format:
    // U8 (grayscale) or RGB. The plugin performs BGR->RGB conversion.
    virtual bool CaptureFrame(void * handle, int & width, int & height, int & type,
                              std::vector<uint8_t> & data) = 0;

    // Close a capture handle previously returned by OpenCapture.
    virtual void CloseCapture(void * handle) = 0;

    // Open a video writer. Returns an opaque writer handle on success.
    virtual void * OpenVideoWriter(const char * fileName, int width, int height, int type,
                                   int fourcc, double fps) = 0;

    // Write a frame to an opened video writer.
    virtual bool WriteVideoFrame(void * handle, int width, int height, int type,
                                 const uint8_t * data, size_t stride) = 0;

    // Close a video writer handle.
    virtual void CloseVideoWriter(void * handle) = 0;

    // Write a single compressed image (jpg/png/bmp/etc.).
    virtual bool WriteCompressed(const char * fileName, int width, int height, int type,
                                 const uint8_t * data, size_t stride) = 0;

    // Display / GUI.
    // Create/destroy a named window. Returns an opaque window handle.
    virtual void * CreateWindow(const char * name) = 0;
    virtual void   DestroyWindow(void * handle) = 0;

    // Show an image in the window identified by the handle, plus optional overlays.
    virtual bool ShowView(void * handle, const char * name, int width, int height, int type,
                          const uint8_t * data, size_t stride,
                          const RunVxOverlayItem * items, size_t itemCount) = 0;

    // Wait for a key press for 'delayMs' milliseconds. Returns the key code or -1.
    virtual int WaitKey(int delayMs) = 0;

    // Draw text on an image buffer.
    virtual void PutText(uint8_t * data, int width, int height, int type, size_t stride,
                         const char * text, int x, int y, uint32_t color) = 0;

    // Resize an image buffer. Simple implementation; used for capture resize.
    // 'inData' is width*height of 'inType'. 'outData' receives outW*outH.
    virtual bool Resize(const uint8_t * inData, int inWidth, int inHeight, int inType, size_t inStride,
                        uint8_t * outData, int outWidth, int outHeight, int outType, size_t outStride) = 0;
};

// Plugin factory exported by the shared library.
extern "C" {
    typedef IRunVxImagePlugin * (*RunVxCreateImagePluginFn)();
#define RUNVX_IMAGE_PLUGIN_FACTORY "RunVxCreateImagePlugin"
}

#endif /* __RUNVX_PLUGIN_IMAGE_H__ */
