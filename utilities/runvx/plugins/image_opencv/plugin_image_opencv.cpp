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

#include "plugin_image_opencv.h"

#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include "vx_ext_amd.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>

#if CV_MAJOR_VERSION >= 4
#define CV_BGR2GRAY   cv::COLOR_BGR2GRAY
#define CV_GRAY2RGB   cv::COLOR_GRAY2RGB
#define CV_RGB2BGR    cv::COLOR_RGB2BGR
#define CV_FONT       cv::FONT_HERSHEY_SIMPLEX
#define CV_FILLED     cv::FILLED
#else
#define CV_FONT       CV_FONT_HERSHEY_SIMPLEX
#endif

using namespace cv;

// Map a VX_DF_IMAGE fourcc to an OpenCV Mat type.
// Returns -1 if the format is not supported by the plugin.
static int VxFormatToCvType(int vxFormat)
{
    switch (vxFormat) {
    case VX_DF_IMAGE_U8:
    case VX_DF_IMAGE_U1_AMD:
        return CV_8UC1;
    case VX_DF_IMAGE_U16:
    case VX_DF_IMAGE_S16:
        return CV_16UC1;
    case VX_DF_IMAGE_RGB:
        return CV_8UC3;
    case VX_DF_IMAGE_RGBX:
        return CV_8UC4;
    case VX_DF_IMAGE_F32_AMD:
        return CV_32FC1;
    default:
        return -1;
    }
}

static int VxFormatChannels(int vxFormat)
{
    switch (vxFormat) {
    case VX_DF_IMAGE_U8:
    case VX_DF_IMAGE_U1_AMD:
    case VX_DF_IMAGE_U16:
    case VX_DF_IMAGE_S16:
    case VX_DF_IMAGE_F32_AMD:
        return 1;
    case VX_DF_IMAGE_RGB:
        return 3;
    case VX_DF_IMAGE_RGBX:
        return 4;
    default:
        return 0;
    }
}

// Map 0xRRGGBB to a cv::Scalar in BGR order.
static Scalar RunVxColorToScalar(uint32_t c)
{
    uint8_t r = (c >> 16) & 0xFF;
    uint8_t g = (c >> 8)  & 0xFF;
    uint8_t b = (c >> 0)  & 0xFF;
    return Scalar(b, g, r, 0);
}

// ---------------------------------------------------------------------------
// Image file read
// ---------------------------------------------------------------------------
bool RunVxImagePluginOpenCV::ReadImageFile(const char * path, int & width, int & height, int & type,
                                           std::vector<uint8_t> & data)
{
    Mat img = imread(path, IMREAD_UNCHANGED);
    if (img.empty()) {
        printf("ERROR: OpenCV imread(%s) failed\n", path);
        return false;
    }

    // Convert single-channel to U8 and 3-channel BGR to RGB.
    if (img.channels() == 3) {
        cvtColor(img, img, CV_RGB2BGR);
        type = VX_DF_IMAGE_RGB;
    }
    else if (img.channels() == 1) {
        type = VX_DF_IMAGE_U8;
    }
    else if (img.channels() == 4) {
        type = VX_DF_IMAGE_RGBX;
    }
    else {
        printf("ERROR: OpenCV imread(%s) returned unsupported channel count %d\n", path, img.channels());
        return false;
    }

    width  = img.cols;
    height = img.rows;

    size_t bytes = img.total() * img.elemSize();
    data.resize(bytes);
    memcpy(data.data(), img.data, bytes);
    return true;
}

// ---------------------------------------------------------------------------
// Capture (camera / video)
// ---------------------------------------------------------------------------
void * RunVxImagePluginOpenCV::OpenCapture(int cameraDevice)
{
    VideoCapture * cap = new VideoCapture(cameraDevice);
    if (!cap->isOpened()) {
        printf("ERROR: OpenCV camera capture(%d) failed\n", cameraDevice);
        delete cap;
        return nullptr;
    }
    return cap;
}

void * RunVxImagePluginOpenCV::OpenCapture(const char * fileName)
{
    VideoCapture * cap = new VideoCapture(fileName);
    if (!cap->isOpened()) {
        printf("ERROR: OpenCV video capture(%s) failed\n", fileName);
        delete cap;
        return nullptr;
    }
    return cap;
}

bool RunVxImagePluginOpenCV::CaptureFrame(void * handle, int & width, int & height, int & type,
                                          std::vector<uint8_t> & data)
{
    VideoCapture * cap = static_cast<VideoCapture *>(handle);
    if (!cap) return false;

    Mat frame;
    *cap >> frame;
    if (frame.empty()) return false;

    if (frame.channels() == 3) {
        cvtColor(frame, frame, CV_RGB2BGR);
        type = VX_DF_IMAGE_RGB;
    }
    else if (frame.channels() == 1) {
        type = VX_DF_IMAGE_U8;
    }
    else if (frame.channels() == 4) {
        type = VX_DF_IMAGE_RGBX;
    }
    else {
        type = -1;
        return false;
    }

    width  = frame.cols;
    height = frame.rows;

    size_t bytes = frame.total() * frame.elemSize();
    data.resize(bytes);
    memcpy(data.data(), frame.data, bytes);
    return true;
}

void RunVxImagePluginOpenCV::CloseCapture(void * handle)
{
    VideoCapture * cap = static_cast<VideoCapture *>(handle);
    delete cap;
}

// ---------------------------------------------------------------------------
// Video writer
// ---------------------------------------------------------------------------
void * RunVxImagePluginOpenCV::OpenVideoWriter(const char * fileName, int width, int height, int type,
                                             int fourcc, double fps)
{
    VideoWriter * writer = new VideoWriter();
    Size size(width, height);
    if (fourcc < 0) fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');
    writer->open(fileName, fourcc, fps, size, VxFormatChannels(type) > 1);
    if (!writer->isOpened()) {
        printf("ERROR: OpenCV VideoWriter(%s) failed\n", fileName);
        delete writer;
        return nullptr;
    }
    return writer;
}

bool RunVxImagePluginOpenCV::WriteVideoFrame(void * handle, int width, int height, int type,
                                             const uint8_t * data, size_t stride)
{
    VideoWriter * writer = static_cast<VideoWriter *>(handle);
    if (!writer) return false;

    int cvType = VxFormatToCvType(type);
    if (cvType < 0) return false;

    Mat img(height, width, cvType, const_cast<uint8_t *>(data), stride);
    Mat out;
    if (cvType == CV_8UC3) {
        cvtColor(img, out, CV_RGB2BGR);
    } else {
        out = img;
    }
    writer->write(out);
    return true;
}

void RunVxImagePluginOpenCV::CloseVideoWriter(void * handle)
{
    VideoWriter * writer = static_cast<VideoWriter *>(handle);
    delete writer;
}

// ---------------------------------------------------------------------------
// Compressed write
// ---------------------------------------------------------------------------
bool RunVxImagePluginOpenCV::WriteCompressed(const char * fileName, int width, int height, int type,
                                             const uint8_t * data, size_t stride)
{
    int cvType = VxFormatToCvType(type);
    if (cvType < 0) {
        printf("ERROR: image type (%4.4s) is not supported for compressed write. Exiting.\n", (const char *)&type);
        return false;
    }

    Mat img(height, width, cvType, const_cast<uint8_t *>(data), stride);

    Mat out;
    if (cvType == CV_8UC3) {
        cvtColor(img, out, CV_RGB2BGR);
    } else {
        out = img;
    }

    if (!imwrite(fileName, out)) {
        printf("ERROR: OpenCV imwrite(%s) failed\n", fileName);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Display / GUI
// ---------------------------------------------------------------------------
void * RunVxImagePluginOpenCV::CreateWindow(const char * name)
{
    namedWindow(name, WINDOW_AUTOSIZE);
    // Return a heap-allocated copy of the name as the opaque handle.
    return strdup(name);
}

void RunVxImagePluginOpenCV::DestroyWindow(void * handle)
{
    char * name = static_cast<char *>(handle);
    destroyWindow(name);
    free(name);
}

bool RunVxImagePluginOpenCV::ShowView(void * handle, const char * name, int width, int height, int type,
                                      const uint8_t * data, size_t stride,
                                      const RunVxOverlayItem * items, size_t itemCount)
{
    (void)handle; // OpenCV windows are tracked by name
    int cvType = VxFormatToCvType(type);
    if (cvType < 0) return false;

    Mat img(height, width, cvType, const_cast<uint8_t *>(data), stride);

    // For display, convert to BGR so colors/overlay work as in original code.
    Mat display;
    if (cvType == CV_8UC1) {
        cvtColor(img, display, CV_GRAY2RGB);
    } else if (cvType == CV_8UC3) {
        cvtColor(img, display, CV_RGB2BGR);
    } else {
        display = img;
    }

    // Render overlays.
    for (size_t i = 0; i < itemCount; i++) {
        const RunVxOverlayItem & it = items[i];
        switch (it.kind) {
        case RUNVX_OVERLAY_TEXT:
            if (it.text) {
                putText(display, it.text, Point(it.x, it.y), CV_FONT, 0.5,
                        RunVxColorToScalar(it.color), 2, 8, false);
            }
            break;
        case RUNVX_OVERLAY_RECT:
            rectangle(display, Rect(it.x, it.y, it.w, it.h),
                      RunVxColorToScalar(it.color), 1, 8);
            break;
        case RUNVX_OVERLAY_FILLED_RECT:
            rectangle(display, Rect(it.x, it.y, it.w, it.h),
                      RunVxColorToScalar(it.color), CV_FILLED, 8);
            break;
        case RUNVX_OVERLAY_HISTOGRAM: {
            Rect box(it.x, it.y, it.w, it.h);
            rectangle(display, Rect(box.x - 2, box.y - 2, box.width + 4, box.height + 4),
                      Scalar(0, 0, 255), 1, 8);
            rectangle(display, Rect(box.x - 1, box.y - 1, box.width + 2, box.height + 2),
                      Scalar(255, 0, 0), 1, 8);
            if (it.histMaxValue > 0 && it.histValues && it.histNumBins > 0) {
                int barWidth = box.width / it.histNumBins;
                for (int bin = 0; bin < it.histNumBins; bin++) {
                    int barHeight = (int)(box.height * (uint64_t)it.histValues[bin] / it.histMaxValue);
                    if (barHeight < 0) barHeight = 0;
                    if (barHeight > box.height) barHeight = box.height;
                    Rect bar(box.x + bin * barWidth, box.y + box.height - barHeight,
                             barWidth, barHeight);
                    rectangle(display, bar, RunVxColorToScalar(it.color), CV_FILLED, 8);
                }
            }
            break;
        }
        case RUNVX_OVERLAY_CIRCLE: {
            Point center(it.x, it.y);
            circle(display, center, it.w, Scalar(0, 0, 0), 1, 8);
            circle(display, center, it.w + 1, RunVxColorToScalar(it.color), 1, 8);
            break;
        }
        default:
            break;
        }
    }

    imshow(name, display);
    return true;
}

int RunVxImagePluginOpenCV::WaitKey(int delayMs)
{
    return waitKey(delayMs);
}

// ---------------------------------------------------------------------------
// Text overlay on an image buffer
// ---------------------------------------------------------------------------
void RunVxImagePluginOpenCV::PutText(uint8_t * data, int width, int height, int type, size_t stride,
                                     const char * text, int x, int y, uint32_t color)
{
    int cvType = VxFormatToCvType(type);
    if (cvType < 0) return;

    Mat img(height, width, cvType, data, stride);
    putText(img, text, Point(x, y), CV_FONT, 0.5, RunVxColorToScalar(color), 2, 8, false);
}

// ---------------------------------------------------------------------------
// Resize using OpenCV
// ---------------------------------------------------------------------------
bool RunVxImagePluginOpenCV::Resize(const uint8_t * inData, int inWidth, int inHeight, int inType, size_t inStride,
                                    uint8_t * outData, int outWidth, int outHeight, int outType, size_t outStride)
{
    int inCvType = VxFormatToCvType(inType);
    int outCvType = VxFormatToCvType(outType);
    if (inCvType < 0 || outCvType < 0) return false;

    Mat inMat(inHeight, inWidth, inCvType, const_cast<uint8_t *>(inData), inStride);
    Mat outMat(outHeight, outWidth, outCvType, outData, outStride);

    if (inCvType == outCvType) {
        resize(inMat, outMat, Size(outWidth, outHeight), 0, 0, INTER_CUBIC);
    }
    else {
        // Convert through a temporary if formats differ.
        Mat tmp;
        resize(inMat, tmp, Size(outWidth, outHeight), 0, 0, INTER_CUBIC);
        tmp.convertTo(outMat, outCvType);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
extern "C" IRunVxImagePlugin * RunVxCreateImagePlugin()
{
    return new RunVxImagePluginOpenCV();
}
