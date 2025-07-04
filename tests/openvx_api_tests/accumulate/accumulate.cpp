/*
Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include <cstring>
#include <iostream>
#include <chrono>

#include <VX/vx.h>
#include <VX/vx_compatibility.h>

using namespace std;

#define ERROR_CHECK_STATUS(status)                                                              \
    {                                                                                           \
        vx_status status_ = (status);                                                           \
        if (status_ != VX_SUCCESS)                                                              \
        {                                                                                       \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    }

#define ERROR_CHECK_OBJECT(obj)                                                                 \
    {                                                                                           \
        vx_status status_ = vxGetStatus((vx_reference)(obj));                                   \
        if (status_ != VX_SUCCESS)                                                              \
        {                                                                                       \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    }

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strlen(string);
    if (len > 0)
    {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}

int main(int argc, char **argv)
{

    int width = 480, height = 360;

    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);

    vx_image input_U8_image = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(input_U8_image);
    vx_image output_S16_image = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
    ERROR_CHECK_OBJECT(output_S16_image);
    vx_status check_val_1 = vxuAccumulateImage(context, input_U8_image, output_S16_image);

    vx_float32 scaleFloat = 1.0f;
    vx_scalar scale = vxCreateScalar(context, VX_TYPE_FLOAT32, &scaleFloat);
    ERROR_CHECK_OBJECT(scale);
    vx_image output_U8_image = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(output_U8_image);
    vx_status check_val_2 = vxuAccumulateWeightedImage(context, input_U8_image, scale, output_U8_image);

    vx_uint32 shiftInt = 1;
    vx_scalar shift = vxCreateScalar(context, VX_TYPE_UINT32, &shiftInt);
    ERROR_CHECK_OBJECT(shift);
    vx_image output_S16_image_2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
    ERROR_CHECK_OBJECT(output_S16_image_2);
    vx_status check_val_3 = vxuAccumulateSquareImage(context, input_U8_image, shift, output_S16_image_2);

    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph)
    vx_image image_1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(image_1);
    vx_image image_2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(image_2);
    vx_node copyNode = vxCopyNode(graph, (vx_reference)image_1, (vx_reference)image_2);

    ERROR_CHECK_STATUS(vxReleaseImage(&input_U8_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&output_S16_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&output_U8_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&output_S16_image_2));
    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    ERROR_CHECK_STATUS(vxReleaseContext(&context));

    return 0;
}
