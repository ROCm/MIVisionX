#include <stdio.h>
#include <VX/vx.h>
// #include "vx_opencv/OpenCV_Tunnel.h"
// #include "vx_opencv/OpenCV_VX_Functions.h"
// #include "vx_opencv/publishkernels.h"
// #include "vx_opencv/vx_opencv.h"
#include "opencv2/imgproc.hpp"
#include <vx_ext_opencv.h>

extern vx_uint32 image_width;
extern vx_uint32 image_height;
extern vx_context context_vx;
extern vx_kernel kernel_vx;
extern vx_graph graph_vx;

extern vx_image input_rgb_image;
extern vx_array currentKeypoints;
extern vx_delay keypointsDelay;

extern int counter;

void set_image_width(const cv::Mat& a);
void set_image_height(const cv::Mat& a);
void print_image();

vx_status init_openvx();

