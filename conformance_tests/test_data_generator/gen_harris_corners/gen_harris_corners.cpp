/*
 * Copyright (c) 2012-2014 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

struct Params
{
    double minDistance;
    double k;
    int ksize;
    int blockSize;
};

static const Params g_cparams[] =
{
    {3.0, 0.04, 3, 3}, {3.0, 0.04, 3, 5}, {3.0, 0.04, 3, 7},
    {3.0, 0.04, 5, 3}, {3.0, 0.04, 5, 5}, {3.0, 0.04, 5, 7},
    {3.0, 0.04, 7, 3}, {3.0, 0.04, 7, 5}, {3.0, 0.04, 7, 7},

    {3.0, 0.10, 3, 3}, {3.0, 0.10, 3, 5}, {3.0, 0.10, 3, 7},
    {3.0, 0.10, 5, 3}, {3.0, 0.10, 5, 5}, {3.0, 0.10, 5, 7},
    {3.0, 0.10, 7, 3}, {3.0, 0.10, 7, 5}, {3.0, 0.10, 7, 7},

    {3.0, 0.15, 3, 3}, {3.0, 0.15, 3, 5}, {3.0, 0.15, 3, 7},
    {3.0, 0.15, 5, 3}, {3.0, 0.15, 5, 5}, {3.0, 0.15, 5, 7},
    {3.0, 0.15, 7, 3}, {3.0, 0.15, 7, 5}, {3.0, 0.15, 7, 7},


    {5.0, 0.04, 3, 3}, {5.0, 0.04, 3, 5}, {5.0, 0.04, 3, 7},
    {5.0, 0.04, 5, 3}, {5.0, 0.04, 5, 5}, {5.0, 0.04, 5, 7},
    {5.0, 0.04, 7, 3}, {5.0, 0.04, 7, 5}, {5.0, 0.04, 7, 7},

    {5.0, 0.10, 3, 3}, {5.0, 0.10, 3, 5}, {5.0, 0.10, 3, 7},
    {5.0, 0.10, 5, 3}, {5.0, 0.10, 5, 5}, {5.0, 0.10, 5, 7},
    {5.0, 0.10, 7, 3}, {5.0, 0.10, 7, 5}, {5.0, 0.10, 7, 7},

    {5.0, 0.15, 3, 3}, {5.0, 0.15, 3, 5}, {5.0, 0.15, 3, 7},
    {5.0, 0.15, 5, 3}, {5.0, 0.15, 5, 5}, {5.0, 0.15, 5, 7},
    {5.0, 0.15, 7, 3}, {5.0, 0.15, 7, 5}, {5.0, 0.15, 7, 7},
};

static void generateHarrisCornerDataSingle(const char * filepath, const char *outprefix, double minDistance, double k, int ksize, int blockSize)
{
    const double qualityLevel = 0.05;
    cv::Mat image = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        printf("failed to open %s\n", filepath);
        exit(-1);
    }

    char outfilename[2048];
    sprintf(outfilename, "%s_%0.2f_%0.2f_%d_%d.txt", outprefix, minDistance, k, ksize, blockSize);
    cv::Mat corners;
    std::ofstream stream(outfilename);
    cv::goodFeaturesToTrack(image, corners, image.cols * image.rows, qualityLevel, minDistance, cv::noArray(), blockSize, true, k, ksize);
    float scale = (1 << (ksize - 1)) * blockSize * 255.f;
    scale = scale * scale * scale * scale;
    stream << corners.rows << std::endl;
    for (int i = 0; i < corners.rows; i++)
    {
        cv::Point3f *pt = (cv::Point3f *)corners.ptr(i);
        if ((0 <= pt->x) && (pt->x < image.cols) && (0 <= pt->y) && (pt->y < image.rows))
            stream << pt->x << " " << pt->y << " " << pt->z * scale << std::endl;
    }
}

static void generateHarrisCornerDataSuite(const char * filepath, const char *outprefix)
{
    size_t params_count = sizeof(g_cparams) / sizeof(Params);
    for (size_t i = 0; i < params_count; i++)
    {
        generateHarrisCornerDataSingle(filepath, outprefix, g_cparams[i].minDistance, g_cparams[i].k, g_cparams[i].ksize, g_cparams[i].blockSize);
    }
}

int main(int argc, char* argv[])
{
    generateHarrisCornerDataSuite("harriscorners/hc_fsc.bmp", "harriscorners/hc_fsc");
    generateHarrisCornerDataSuite("harriscorners/hc_msc.bmp", "harriscorners/hc_msc");
    return 0;
}
