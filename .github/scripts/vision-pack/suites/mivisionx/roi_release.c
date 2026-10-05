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

/* H12 reproducer: release a parent image before its ROI child.
 *   roi_release <regular|uniform> <parent-first|child-first>
 * Exits 0 when the release sequence completes; the bug is a SIGSEGV. */
#include <VX/vx.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <regular|uniform> <parent-first|child-first>\n", argv[0]);
        return 2;
    }
    vx_context ctx = vxCreateContext();
    vx_rectangle_t rect = {0, 0, 128, 128};
    vx_image img;
    if (strcmp(argv[1], "uniform") == 0) {
        vx_pixel_value_t val;
        memset(&val, 0, sizeof(val));
        val.U8 = 1;
        img = vxCreateUniformImage(ctx, 320, 240, VX_DF_IMAGE_U8, &val);
    } else {
        img = vxCreateImage(ctx, 320, 240, VX_DF_IMAGE_U8);
    }
    vx_image roi = vxCreateImageFromROI(img, &rect);
    if (vxGetStatus((vx_reference)roi) != VX_SUCCESS) {
        printf("vxCreateImageFromROI failed\n");
        return 1;
    }
    vx_status s1, s2;
    if (strcmp(argv[2], "parent-first") == 0) {
        s1 = vxReleaseImage(&img);
        s2 = vxReleaseImage(&roi);
    } else {
        s1 = vxReleaseImage(&roi);
        s2 = vxReleaseImage(&img);
    }
    printf("release statuses %d %d\n", s1, s2);
    vxReleaseContext(&ctx);
    return (s1 == VX_SUCCESS && s2 == VX_SUCCESS) ? 0 : 1;
}
