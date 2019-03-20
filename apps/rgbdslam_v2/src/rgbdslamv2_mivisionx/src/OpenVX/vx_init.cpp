#include "vx_init.h"
#include <iostream>

// OpenVX Macros

#define ERROR_CHECK_STATUS( status ) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }

#define ERROR_CHECK_OBJECT( obj ) { \
        vx_status status_ = vxGetStatus((vx_reference)(obj)); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }

//OpenCV Draw Point Function

void DrawPoint(cv::Mat image, int x, int y ){
        cv::Point  center( x, y );
        cv::circle( image, center, 1, cv::Scalar( 0, 0, 255 ), 2 );
}

vx_uint32 image_width;
vx_uint32 image_height;
vx_context context_vx;
vx_kernel kernel_vx;
vx_graph graph_vx;

vx_image input_rgb_image;
vx_array currentKeypoints;
vx_delay keypointsDelay;

int counter;

void print_image(){
    std::cout << "image_width is: " << image_width << std::endl;
}

vx_status init_openvx(){
    // Setup image parameters in vx graph
    vx_uint32 width = image_width;
    vx_uint32 height = image_height;

    counter = 0;

    vx_status status = VX_SUCCESS;
    // Create OpenVX Context
    context_vx = vxCreateContext();

    if(context_vx){
        // Create OpenVX Kernel
        status = vxLoadKernels(context_vx, "vx_opencv");
        if(status == VX_SUCCESS){
            kernel_vx = vxGetKernelByName(context_vx, "org.opencv.orb_detect");
            if(kernel_vx){
                // Setup parameters for OpenVX nodes
                input_rgb_image = vxCreateImage(context_vx, width, height, VX_DF_IMAGE_RGB);
                vx_image output_image = vxCreateImage(context_vx, width, height, VX_DF_IMAGE_U8);
                
                vx_array keypoints = vxCreateArray(context_vx, VX_TYPE_KEYPOINT, 600);
                ERROR_CHECK_OBJECT(keypoints);
                keypointsDelay = vxCreateDelay(context_vx, (vx_reference)keypoints, 2);
                
                currentKeypoints = (vx_array)vxGetReferenceFromDelay(keypointsDelay, 0);
                ERROR_CHECK_OBJECT(currentKeypoints);
                ERROR_CHECK_STATUS(vxReleaseArray(&keypoints));
                
                // Create OpenVX graph
                graph_vx = vxCreateGraph(context_vx);
                vx_image yuv_image = vxCreateVirtualImage(graph_vx, width, height, VX_DF_IMAGE_IYUV);
                vx_image luma_image = vxCreateVirtualImage(graph_vx, width, height, VX_DF_IMAGE_U8);

                // Parameters for ORB
                vx_int32 features = 600;
                vx_float32 scaleFactor = 1.2;
                vx_int32 nlevels = 1;
                vx_int32 edgeThreshold = 31;
                vx_int32 firstLevel = 0;
                vx_int32 WTA_K = 2;
                vx_int32 scoreType = 0;
                vx_int32 patchSize = 31;

                // Determine which OpenVX nodes to call
                vx_node nodes[] = {
                    vxColorConvertNode(graph_vx, input_rgb_image, yuv_image),
                    vxChannelExtractNode(graph_vx, yuv_image, VX_CHANNEL_Y, luma_image),
                    vxExtCvNode_orbDetect(graph_vx, luma_image, luma_image, currentKeypoints, features, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize)
                };
                
                for(vx_size i = 0; i < sizeof(nodes)/sizeof(nodes[0]); i++){
                    ERROR_CHECK_OBJECT(nodes[i]);
                    ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
                }
                // Verify Graph, ready to use
                ERROR_CHECK_STATUS(vxReleaseImage(&yuv_image));
                ERROR_CHECK_STATUS(vxReleaseImage(&luma_image));
                ERROR_CHECK_STATUS(vxVerifyGraph(graph_vx));
            }
        }
    }

    return status;
}

