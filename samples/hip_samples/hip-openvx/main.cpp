#include <iostream>

//#define __HIP_PLATFORM_HCC__

// hip header file
#include <xmmintrin.h>
#define __HIP_PLATFORM_HCC__
#include "hip/hip_runtime.h"
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <VX/vx_khr_nn.h>
#include <VX/vxu.h>
#include <vx_ext_amd.h>
#include <string>
#include <chrono>
#define DUMP_IMAGE  0

#define ERROR_CHECK_OBJECT(obj) { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status     , "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; } }
#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS) { printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return -1; } }

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strlen(string);
    if (len > 0) {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}


inline int64_t clockCounter()
{
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline int64_t clockFrequency()
{
    return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}

vx_status makeInputImage1(vx_context context, vx_image img, int width, int height, int mem_type)
{
    ERROR_CHECK_OBJECT((vx_reference)img);

    if (mem_type == VX_MEMORY_TYPE_HOST) {
        vx_rectangle_t rect = { 0, 0, (vx_uint32)width, (vx_uint32)height };
        vx_map_id map_id;
        vx_imagepatch_addressing_t addrId;
        vx_uint8 * ptr;
        ERROR_CHECK_STATUS(vxMapImagePatch(img, &rect, 0, &map_id, &addrId, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
        vx_size stride = addrId.stride_y;
        for (int i= 0; i< height; i++ ){
            for (int j=0; j < width; j++) {
                ptr[i*stride + j] = 240;
            }
        }
        ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
    } else {
        vx_rectangle_t rect = { 0, 0, (vx_uint32)width, (vx_uint32)height };
        vx_map_id map_id;
        vx_imagepatch_addressing_t addrId;
        vx_uint8 * ptr;
        ERROR_CHECK_STATUS(vxMapImagePatch(img, &rect, 0, &map_id, &addrId, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
        vx_size stride = addrId.stride_y;
        for (int i= 0; i< height; i++ ){
            for (int j=0; j < width; j++) {
                ptr[i*stride + j] = 240;
            }
        }
        ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#if DUMP_IMAGE
        FILE *fp = fopen("input1.bin", "wb");
        ERROR_CHECK_STATUS(vxMapImagePatch(img, &rect, 0, &map_id, &addrId, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
        if (fp) fwrite(ptr, 1, width*height, fp);
        ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#endif
    }
    vxReleaseImage(&img);
    return VX_SUCCESS;
}

vx_status makeInputImage2(vx_context context, vx_image img, vx_uint32 width, vx_uint32 height, int mem_type)
{
    ERROR_CHECK_OBJECT((vx_reference)img);

    if (mem_type == VX_MEMORY_TYPE_HOST) {
        vx_rectangle_t rect = { 0, 0, (vx_uint32)width, (vx_uint32)height };
        vx_map_id map_id;
        vx_imagepatch_addressing_t addrId;
        vx_uint8 * ptr;
        ERROR_CHECK_STATUS(vxMapImagePatch(img, &rect, 0, &map_id, &addrId, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
        vx_size stride = addrId.stride_y;
        for (int i= 0; i< height; i++ ){
            for (int j=0; j < width; j++) {
                //if ( i>=40 && i < 60 && j >= 20 && j<80 )
                    ptr[i*stride + j] = 120;
                //else
                //    ptr[i*stride + j] = 255;
            }
        }
        ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
    } else {
        vx_rectangle_t rect = { 0, 0, (vx_uint32)width, (vx_uint32)height };
        vx_map_id map_id;
        vx_imagepatch_addressing_t addrId;
        vx_uint8 * ptr;
        ERROR_CHECK_STATUS(vxMapImagePatch(img, &rect, 0, &map_id, &addrId, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
        vx_size stride = addrId.stride_y;
        for (int i= 0; i< height; i++ ){
            for (int j=0; j < width; j++) {
              //  if ( i>=40 && i < 60 && j >= 20 && j<80 )
                    ptr[i*stride + j] = 120;
               // else
               //     ptr[i*width + j] = 255;
            }
        }
        ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#if DUMP_IMAGE
        FILE *fp = fopen("input2.bin", "wb");
        ERROR_CHECK_STATUS(vxMapImagePatch(img, &rect, 0, &map_id, &addrId, (void **)&ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
        if (fp) fwrite(ptr, 1, width*height, fp);
        ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#endif
    }
    vxReleaseImage(&img);
    return VX_SUCCESS;
}


int main(int argc, const char ** argv) {

    void *ptr[3] = {nullptr, nullptr, nullptr};     // hip
    int64_t freq = clockFrequency(), t0, t1;
    vx_image img1, img2, img_out;
    vx_uint32 width = atoi(argv[1]);
    vx_uint32 height = atoi(argv[2]);
    int affinity = atoi(argv[3]);           // 0 for CPU and 1 for GPU
    if (width <= 0) width = 100;
    if (height <= 0) height = 100;
    if (affinity <= 0) affinity = 0;
    char* outImgBuffer = new char[width*height];

    // create context, input, output, and graph
    vxRegisterLogCallback(NULL, log_callback, vx_false_e);
    vx_context context = vxCreateContext();
    vx_status status = vxGetStatus((vx_reference)context);
    if(status) {
        printf("ERROR: vxCreateContext() failed\n");
        return -1;
    }
    vxRegisterLogCallback(context, log_callback, vx_false_e);
    vx_graph graph = vxCreateGraph(context);
    // create graph and set affinity
    if (!affinity) {
        if (graph)
        {
          img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
          img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
          img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
          ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
          AgoTargetAffinityInfo affinity;
          affinity.device_type = AGO_TARGET_AFFINITY_CPU;
          affinity.device_info = 0;
          ERROR_CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity)));
          vx_node node = vxAbsDiffNode(graph, img1, img2, img_out);
          if (node)
          {
              status = vxVerifyGraph(graph);
              ERROR_CHECK_STATUS(makeInputImage1(context, img1, width, height, VX_MEMORY_TYPE_HOST));
              ERROR_CHECK_STATUS(makeInputImage2(context, img2, width, height, VX_MEMORY_TYPE_HOST));
              printf("After makeInputImage\n");
	      if (status == VX_SUCCESS)
              {
                  status = vxProcessGraph(graph);
              }
              printf("After ProcessGraph\n");
              vxReleaseNode(&node);
          }
          vxReleaseGraph(&graph);
        }
    }else
    {
        vx_imagepatch_addressing_t addr = { 0};
        addr.dim_x = width;
        addr.dim_y = height;
        addr.stride_x = 1;
        addr.stride_y = (width+3)&~3;   // stride has to be a multiple of 4 since we process 4 pixels at a time
        hipMalloc((void**)&ptr[0], width*addr.stride_y);
        hipMalloc((void**)&ptr[1], width*addr.stride_y);
        hipMalloc((void**)&ptr[2], width*addr.stride_y);
        hipMemset(ptr[2], 0, width*addr.stride_y);
  //      printf("Main: dst: %p src1: %p src2: %p <%dx%d>\n", ptr[2], ptr[0], ptr[1], width, height);

        ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &addr, &ptr[0], VX_MEMORY_TYPE_HIP));
        ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &addr, &ptr[1], VX_MEMORY_TYPE_HIP));
        ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &addr, &ptr[2], VX_MEMORY_TYPE_HIP));
        if (graph)
        {
            AgoTargetAffinityInfo affinity;
            affinity.device_type = AGO_TARGET_AFFINITY_GPU;
            affinity.device_info = 0;

            ERROR_CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity)));
            vx_node node = vxAbsDiffNode(graph, img1, img2, img_out);
            if (node)
            {
                status = vxVerifyGraph(graph);
                ERROR_CHECK_STATUS(makeInputImage1(context, img1, width, height, VX_MEMORY_TYPE_HIP));
                ERROR_CHECK_STATUS(makeInputImage2(context, img2, width, height, VX_MEMORY_TYPE_HIP));
                if (status == VX_SUCCESS)
                {
                    //for (int i=0; i<100; i++)
                        status = vxProcessGraph(graph);
                }
                vxReleaseNode(&node);
            }
            vxReleaseGraph(&graph);
        }
    }

    // check output values
    vx_rectangle_t rect = { 0, 0, width, height };
    vx_map_id  map_id;
    vx_imagepatch_addressing_t addr = {0};
    vx_uint8 *out_buf;

    ERROR_CHECK_STATUS(vxMapImagePatch(img_out, &rect, 0, &map_id, &addr, (void **)&out_buf, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));

    // verify the results
    int expected = width*height*120;
    int sum = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            sum += out_buf[i * addr.stride_y + j];
    //        printf("sum<%x,%d>: %d\t", i, j, out_buf[i * addr.stride_y + j]);
        }
     //   printf("\n");
    }
    if (sum != expected) {
        printf("FAILED: sum = %d expected: %d \n", sum, expected);
    } else {
        printf("PASSED!\n");
    }
#if DUMP_IMAGE
    FILE *fp = fopen("output.bin", "wb");
    if (fp) fwrite(out_buf, 1, width*height, fp);
#endif
    ERROR_CHECK_STATUS( vxUnmapImagePatch( img_out, map_id ) );

    // free the resources on host side
    if (ptr[0]) hipFree(ptr[0]);
    if (ptr[1]) hipFree(ptr[1]);
    if (ptr[1]) hipFree(ptr[1]);
    free(outImgBuffer);
    vxReleaseContext(&context);

    return 0;
}
