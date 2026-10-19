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

#ifndef __PLUGIN_IMAGE_OPENCV_H__
#define __PLUGIN_IMAGE_OPENCV_H__

#include "runvx_plugin_image.h"

class RunVxImagePluginOpenCV : public IRunVxImagePlugin {
public:
    virtual ~RunVxImagePluginOpenCV() = default;

    bool ReadImageFile(const char * path, int & width, int & height, int & type,
                       std::vector<uint8_t> & data) override;

    void * OpenCapture(int cameraDevice) override;
    void * OpenCapture(const char * fileName) override;
    bool CaptureFrame(void * handle, int & width, int & height, int & type,
                      std::vector<uint8_t> & data) override;
    void CloseCapture(void * handle) override;

    void * OpenVideoWriter(const char * fileName, int width, int height, int type,
                           int fourcc, double fps) override;
    bool WriteVideoFrame(void * handle, int width, int height, int type,
                         const uint8_t * data, size_t stride) override;
    void CloseVideoWriter(void * handle) override;

    bool WriteCompressed(const char * fileName, int width, int height, int type,
                         const uint8_t * data, size_t stride) override;

    void * CreateWindow(const char * name) override;
    void   DestroyWindow(void * handle) override;
    bool   ShowView(void * handle, const char * name, int width, int height, int type,
                    const uint8_t * data, size_t stride,
                    const RunVxOverlayItem * items, size_t itemCount) override;
    int    WaitKey(int delayMs) override;

    void PutText(uint8_t * data, int width, int height, int type, size_t stride,
                 const char * text, int x, int y, uint32_t color) override;

    bool Resize(const uint8_t * inData, int inWidth, int inHeight, int inType, size_t inStride,
                uint8_t * outData, int outWidth, int outHeight, int outType, size_t outStride) override;
};

#endif /* __PLUGIN_IMAGE_OPENCV_H__ */
