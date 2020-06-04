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

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

void generate_reference_result(const char* src_name, int winsz, int low_thresh, int high_thresh, bool use_l2)
{
    Mat src = imread(src_name, IMREAD_GRAYSCALE);
    Mat dst;
    if (src.empty())
    {
        printf("failed to open %s\n", src_name);
        exit(-1);
    }

    Canny(src, dst, low_thresh, high_thresh, winsz, use_l2);

    //2-pixel borders:
    rectangle(dst, Point(0,0), Point(dst.cols-1, dst.rows-1), 255, 2, 4, 0);

    char buff[1024];
    sprintf(buff, "canny_%dx%d_%d_%d_%s_%s", winsz, winsz, low_thresh, high_thresh, use_l2 ? "L2" : "L1", src_name);
    imwrite(buff, dst);
}

int main(int, char**)
{
    generate_reference_result("lena_gray.bmp", 3, 70,   71, false);
    generate_reference_result("lena_gray.bmp", 3, 70,   71, true);
    generate_reference_result("lena_gray.bmp", 3, 90,  130, false);
    generate_reference_result("lena_gray.bmp", 3, 90,  130, true);
    generate_reference_result("lena_gray.bmp", 3, 100, 120, false);
    generate_reference_result("lena_gray.bmp", 3, 100, 120, true);
    generate_reference_result("lena_gray.bmp", 3, 120, 120, false);
    generate_reference_result("lena_gray.bmp", 3, 150, 220, false);
    generate_reference_result("lena_gray.bmp", 3, 150, 220, true);
    generate_reference_result("lena_gray.bmp", 5, 100, 100, false);
    generate_reference_result("lena_gray.bmp", 5, 100, 120, false);
    generate_reference_result("lena_gray.bmp", 5, 100, 120, true);
    generate_reference_result("lena_gray.bmp", 7, 80,   80, false);
    generate_reference_result("lena_gray.bmp", 7, 100, 120, false);
    generate_reference_result("lena_gray.bmp", 7, 100, 120, true);
    generate_reference_result("blurred_lena_gray.bmp", 7, 100, 120, true);
    generate_reference_result("blurred_lena_gray.bmp", 5, 100, 120, false);
    generate_reference_result("blurred_lena_gray.bmp", 3, 150, 220, false);
    generate_reference_result("blurred_lena_gray.bmp", 3, 70,   71, false);
    generate_reference_result("blurred_lena_gray.bmp", 3, 70,   71, true);
    generate_reference_result("blurred_lena_gray.bmp", 3, 90,  125, false);
    generate_reference_result("blurred_lena_gray.bmp", 3, 90,  130, true);
    generate_reference_result("blurred_lena_gray.bmp", 3, 100, 120, false);
    generate_reference_result("blurred_lena_gray.bmp", 3, 100, 120, true);
    generate_reference_result("blurred_lena_gray.bmp", 3, 150, 220, true);
    generate_reference_result("blurred_lena_gray.bmp", 5, 100, 120, true);
    generate_reference_result("blurred_lena_gray.bmp", 7, 100, 120, false);
    return 0;
}
