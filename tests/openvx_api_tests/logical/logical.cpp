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
#include <VX/vx_khr_icd.h>
#include <vx_ext_amd.h>


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

enum vx_df_image_amd_sample {
	VX_DF_IMAGE_Y210_AMD = VX_DF_IMAGE('Y', '2', '1', '0'),  // AGO image with YUV 4:2:2 10-bit (Y210)
	VX_DF_IMAGE_Y212_AMD = VX_DF_IMAGE('Y', '2', '1', '2'),  // AGO image with YUV 4:2:2 12-bit (Y212)
	VX_DF_IMAGE_Y216_AMD = VX_DF_IMAGE('Y', '2', '1', '6'),  // AGO image with YUV 4:2:2 16-bit (Y216)
	VX_DF_IMAGE_RGB4_AMD = VX_DF_IMAGE('R', 'G', 'B', '4'),  // AGO image with RGB-48 16bit per channel (RGB4)
};

int main(int argc, char **argv)
{
    std::cout << VX_VERSION << VX_CONTEXT_NONLINEAR_MAX_DIMENSION << "\n";;
    int width = 5, height = 3;

    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);

    ERROR_CHECK_STATUS(vxRegisterKernelLibrary(context, nullptr, nullptr, nullptr));

    // register image formats
	AgoImageFormatDescription desc = { 3, 1, 32, VX_COLOR_SPACE_DEFAULT, VX_CHANNEL_RANGE_FULL };
	vxSetContextImageFormatDescription(context, VX_DF_IMAGE_Y210_AMD, &desc);
	vxSetContextImageFormatDescription(context, VX_DF_IMAGE_Y212_AMD, &desc);
	vxSetContextImageFormatDescription(context, VX_DF_IMAGE_Y216_AMD, &desc);
	desc = { 3, 1, 48, VX_COLOR_SPACE_DEFAULT, VX_CHANNEL_RANGE_FULL };
	vxSetContextImageFormatDescription(context, VX_DF_IMAGE_RGB4_AMD, &desc);
    char word[VX_MAX_KERNEL_NAME];
    vx_df_image format = VX_DF_IMAGE(word[0], word[1], word[2], word[3]);

    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);
    
    int kernel_size = VX_MAX_KERNEL_NAME;

    vx_image input_U8_image_1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    vx_image input_U8_image_2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    vx_image output_U8_image = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    vx_image output_not_image = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(input_U8_image_1);
    ERROR_CHECK_OBJECT(input_U8_image_2);
    ERROR_CHECK_OBJECT(output_U8_image);
    ERROR_CHECK_OBJECT(output_not_image);

    vx_rectangle_t rectFull = { 0, 0, 100, 100 };
    vx_size size_value = vxComputeImagePatchSize(input_U8_image_1, &rectFull, 0);

    vx_char name[100];
    ERROR_CHECK_STATUS(vxGetReferenceName((vx_reference)input_U8_image_1, name, sizeof(name)));

    vx_status failure_1 = vxGetModuleInternalData(context, nullptr, nullptr, nullptr);
    vx_status failure_2 = vxSetModuleInternalData(context, nullptr, nullptr, size_value);
    vx_context f_context = vxCreateContextFromPlatform(nullptr);

    vx_size m_dims[1] = {100};
    vx_tensor m_tensor = vxCreateTensorFromHandle(context, 1, m_dims, VX_TYPE_UINT8, 0, 0, nullptr, 0);
    vx_threshold v_threshold_1 = vxCreateVirtualThresholdForImage(graph, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U1, VX_DF_IMAGE_U1);
    vx_threshold v_threshold_s16 = vxCreateVirtualThresholdForImage(graph, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8);
    vx_threshold v_threshold_u16 = vxCreateVirtualThresholdForImage(graph, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U16, VX_DF_IMAGE_U8);
    vx_threshold v_threshold_s32 = vxCreateVirtualThresholdForImage(graph, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_S32, VX_DF_IMAGE_U8);
    vx_threshold v_threshold_u32 = vxCreateVirtualThresholdForImage(graph, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U32, VX_DF_IMAGE_U8);
    vx_threshold threshold_1 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U1, VX_DF_IMAGE_U1);
    vx_threshold threshold_s16 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8);
    vx_threshold threshold_u16 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U16, VX_DF_IMAGE_U8);
    vx_threshold threshold_s32 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_S32, VX_DF_IMAGE_U8);
    vx_threshold threshold_u32 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U32, VX_DF_IMAGE_U8);
    vx_threshold threshold_rgb = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_RGB, VX_DF_IMAGE_U8);
    vx_threshold threshold_rgbx = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_RGBX, VX_DF_IMAGE_U8);
    vx_threshold threshold_nv12 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_NV12, VX_DF_IMAGE_U8);
    vx_threshold threshold_R = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, 0 );

    vx_uint16 device = -1;
	ERROR_CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_VENDOR_ID, &device, sizeof(device)));
    ERROR_CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_VERSION, &device, sizeof(device)));
    vx_size device_1 = -1;
    ERROR_CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_EXTENSIONS_SIZE, &device_1, sizeof(device_1)));
    ERROR_CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_EXTENSIONS, &device_1, sizeof(device_1)));
    ERROR_CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_NONLINEAR_MAX_DIMENSION, &device_1, sizeof(device_1)));
    ERROR_CHECK_STATUS(vxQueryContext(context, VX_CONTEXT_MAX_TENSOR_DIMS, &device_1, sizeof(device_1)));

    vx_node nodes[] =
        {
            vxAndNode(graph, input_U8_image_1, input_U8_image_2, output_U8_image),
            vxNotNode(graph, output_U8_image, output_not_image)
        };

    for (vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
    {
        ERROR_CHECK_OBJECT(nodes[i]);
        ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
    }
    ERROR_CHECK_STATUS(vxHint((vx_reference)graph, VX_HINT_SERIALIZE, nullptr, 0));
    ERROR_CHECK_STATUS(vxVerifyGraph(graph));

    auto start = std::chrono::steady_clock::now();
    ERROR_CHECK_STATUS(vxProcessGraph(graph));
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "STATUS: vxProcessGraph() took " << (elapsed_seconds.count()*1000.0f) << "msec (1st iteration)\n";

    start = std::chrono::steady_clock::now();
    for(int i = 0; i < 100; i++){
        ERROR_CHECK_STATUS(vxProcessGraph(graph));
    }
    end = std::chrono::steady_clock::now();
    elapsed_seconds = (end-start)/100;
    std::cout << "STATUS: vxProcessGraph() took " << (elapsed_seconds.count()*1000.0f) << "msec (AVG)\n";

    ERROR_CHECK_STATUS(vxReleaseImage(&input_U8_image_1));
    ERROR_CHECK_STATUS(vxReleaseImage(&input_U8_image_2));
    ERROR_CHECK_STATUS(vxReleaseImage(&output_U8_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&output_not_image));
    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    ERROR_CHECK_STATUS(vxReleaseContext(&context));

    return 0;
}
