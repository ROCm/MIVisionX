// TEST SUITE FOR AMD OPENVX HOST AND AMD OPENVX HIP

// case functionality list for reference

// case 1 - agoKernel_AbsDiff_U8_U8U8
// case 2 - agoKernel_AbsDiff_S16_S16S16_Sat
// case 3 - agoKernel_Add_U8_U8U8_Wrap
// case 4 - agoKernel_Add_U8_U8U8_Sat
// case 5 - agoKernel_Add_S16_U8U8
// case 6 - agoKernel_Add_S16_S16U8_Wrap
// case 7 - agoKernel_Add_S16_S16U8_Sat
// case 8 - agoKernel_Add_S16_S16S16_Wrap
// case 9 - agoKernel_Add_S16_S16S16_Sat
// case 10 - agoKernel_Sub_U8_U8U8_Wrap
// case 11 - agoKernel_Sub_U8_U8U8_Sat
// case 12 - agoKernel_Sub_S16_U8U8
// case 13 - agoKernel_Sub_S16_S16U8_Wrap
// case 14 - agoKernel_Sub_S16_S16U8_Sat
// case 15 - agoKernel_Sub_S16_U8S16_Wrap
// case 16 - agoKernel_Sub_S16_U8S16_Sat
// case 17 - agoKernel_Sub_S16_S16S16_Wrap
// case 18 - agoKernel_Sub_S16_S16S16_Sat
// case 19 - agoKernel_Mul_U8_U8U8_Wrap_Trunc
// case 20 - agoKernel_Mul_U8_U8U8_Wrap_Round
// case 21 - agoKernel_Mul_U8_U8U8_Sat_Trunc
// case 22 - agoKernel_Mul_U8_U8U8_Sat_Round
// case 23 - agoKernel_Mul_S16_U8U8_Wrap_Trunc
// case 24 - agoKernel_Mul_S16_U8U8_Wrap_Round
// case 25 - agoKernel_Mul_S16_U8U8_Sat_Trunc
// case 26 - agoKernel_Mul_S16_U8U8_Sat_Round
// case 27 - agoKernel_Mul_S16_S16U8_Wrap_Trunc
// case 28 - agoKernel_Mul_S16_S16U8_Wrap_Round
// case 29 - agoKernel_Mul_S16_S16U8_Sat_Trunc
// case 30 - agoKernel_Mul_S16_S16U8_Sat_Round
// case 31 - agoKernel_Mul_S16_S16S16_Wrap_Trunc
// case 32 - agoKernel_Mul_S16_S16S16_Wrap_Round
// case 33 - agoKernel_Mul_S16_S16S16_Sat_Trunc
// case 34 - agoKernel_Mul_S16_S16S16_Sat_Round
// case 35 - agoKernel_Mul_U24_U24U8_Sat_Round
// case 36 - agoKernel_Mul_U32_U32U8_Sat_Round
// case 37 - agoKernel_And_U8_U8U8
// case 38 - agoKernel_And_U8_U8U1
// case 39 - agoKernel_And_U8_U1U8
// case 40 - agoKernel_And_U8_U1U1
// case 41 - agoKernel_And_U1_U8U8
// case 42 - agoKernel_And_U1_U8U1
// case 43 - agoKernel_And_U1_U1U8
// case 44 - agoKernel_And_U1_U1U1
// case 45 - agoKernel_Not_U8_U8
// case 46 - agoKernel_Not_U8_U1
// case 47 - agoKernel_Not_U1_U8
// case 48 - agoKernel_Not_U1_U1
// case 49 - agoKernel_Or_U8_U8U8
// case 50 - agoKernel_Or_U8_U8U1
// case 51 - agoKernel_Or_U8_U1U8
// case 52 - agoKernel_Or_U8_U1U1
// case 53 - agoKernel_Or_U1_U8U8
// case 54 - agoKernel_Or_U1_U8U1
// case 55 - agoKernel_Or_U1_U1U8
// case 56 - agoKernel_Or_U1_U1U1
// case 57 - agoKernel_Xor_U8_U8U8
// case 58 - agoKernel_Xor_U8_U8U1
// case 59 - agoKernel_Xor_U8_U1U8
// case 60 - agoKernel_Xor_U8_U1U1
// case 61 - agoKernel_Xor_U1_U8U8
// case 62 - agoKernel_Xor_U1_U8U1
// case 63 - agoKernel_Xor_U1_U1U8
// case 64 - agoKernel_Xor_U1_U1U1
// case 65 - agoKernel_Magnitude_S16_S16S16
// case 66 - agoKernel_Phase_U8_S16S16
// case 67 - agoKernel_ChannelCopy_U8_U8
// case 68 - agoKernel_ChannelCopy_U8_U1
// case 69 - agoKernel_ChannelCopy_U1_U8
// case 70 - agoKernel_ChannelCopy_U1_U1
// case 71 - agoKernel_ChannelExtract_U8_U16_Pos0
// case 72 - agoKernel_ChannelExtract_U8_U16_Pos1
// case 73 - agoKernel_ChannelExtract_U8_U24_Pos0
// case 74 - agoKernel_ChannelExtract_U8_U24_Pos1
// case 75 - agoKernel_ChannelExtract_U8_U24_Pos2
// case 76 - agoKernel_ChannelExtract_U8_U32_Pos0
// case 77 - agoKernel_ChannelExtract_U8_U32_Pos1
// case 78 - agoKernel_ChannelExtract_U8_U32_Pos2
// case 79 - agoKernel_ChannelExtract_U8_U32_Pos3
// case 80 - agoKernel_ChannelExtract_U8U8U8_U24
// case 81 - agoKernel_ChannelExtract_U8U8U8_U32
// case 82 - agoKernel_ChannelExtract_U8U8U8U8_U32
// case 83 - agoKernel_ChannelCombine_U16_U8U8
// case 84 - agoKernel_ChannelCombine_U24_U8U8U8_RGB
// case 85 - agoKernel_ChannelCombine_U32_U8U8U8_UYVY
// case 86 - agoKernel_ChannelCombine_U32_U8U8U8_YUYV
// case 87 - agoKernel_ChannelCombine_U32_U8U8U8U8_RGBX
// case 88 - agoKernel_Lut_U8_U8
// case 89 - agoKernel_Threshold_U8_U8_Binary
// case 90 - agoKernel_Threshold_U8_U8_Range
// case 91 - agoKernel_Threshold_U1_U8_Binary
// case 92 - agoKernel_Threshold_U1_U8_Range
// case 93 - agoKernel_ThresholdNot_U8_U8_Binary
// case 94 - agoKernel_ThresholdNot_U8_U8_Range
// case 95 - agoKernel_ThresholdNot_U1_U8_Binary
// case 96 - agoKernel_ThresholdNot_U1_U8_Range
// case 97 - agoKernel_Max_U8_U8
// case 98 - agoKernel_Min_U8_U8
// case 99 - agoKernel_WeightedAverage_U8_U8


#define __HIP_PLATFORM_HCC__
#include "hip/hip_runtime.h"
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <VX/vx_khr_nn.h>
#include <VX/vxu.h>
#include <vx_ext_amd.h>
#include <xmmintrin.h>
#include <string.h>
#include <iostream>

using namespace std;

// ------------------------------------------------------------
// Enable/Disable INPUT/OUTPUT parameters printing
#define PRINT_INPUT
#define PRINT_OUTPUT
// ------------------------------------------------------------

#define ERROR_CHECK_OBJECT(obj) { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status     , "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; } }
#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS) { printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return -1; } }

#define PIXELCHECKU1(pixel) (pixel == (vx_int32)0) ? ((vx_uint8)0) : ((vx_uint8)1)
#define PIXELCHECKU8(pixel) (pixel < (vx_int32)0) ? ((vx_uint8)0) : ((pixel < (vx_int32)UINT8_MAX) ? (vx_uint8)pixel : ((vx_uint8)UINT8_MAX))
#define PIXELCHECKU16(pixel) (pixel < (vx_int32)0) ? ((vx_uint16)0) : ((pixel < (vx_int32)UINT16_MAX) ? (vx_uint16)pixel : ((vx_uint16)UINT16_MAX))
#define PIXELCHECKU32(pixel) (pixel < (vx_int32)0) ? ((vx_uint32)0) : ((pixel < (vx_uint32)UINT32_MAX) ? (vx_uint32)pixel : ((vx_uint32)UINT32_MAX))
#define PIXELCHECKS16(pixel) (pixel < (vx_int32)INT16_MIN) ? ((vx_int16)INT16_MIN) : ((pixel < (vx_int32)INT16_MAX) ? (vx_int16)pixel : ((vx_int16)INT16_MAX))
#define PIXELROUNDF32(value) ((value - (int)(value)) >= 0.5 ? (value + 1) : (value))

#define DBL_EPSILON      __DBL_EPSILON__
#define atan2_p0        (0.273*0.3183098862f)
#define atan2_p1		(0.9997878412794807f*57.29577951308232f)
#define atan2_p3		(-0.3258083974640975f*57.29577951308232f)
#define atan2_p5		(0.1555786518463281f*57.29577951308232f)
#define atan2_p7		(-0.04432655554792128f*57.29577951308232f)

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

int generic_mod(int a, int b)
{
	int val = a % b < 0 ? a % b + b : a % b;
	return val;
}

vx_uint8 Norm_Atan2_deg(vx_int16 Gx, vx_int16 Gy)
{
	float scale = (float)128 / 180.f;
	vx_uint16 ax, ay;
	ax = std::abs(Gx), ay = std::abs(Gy);
	float a, c, c2;
	if (ax >= ay)
	{
		c = (float)ay / ((float)ax + (float)DBL_EPSILON);
		c2 = c*c;
		a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
	}
	else
	{
		c = (float)ax / ((float)ay + (float)DBL_EPSILON);
		c2 = c*c;
		a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
	}
	if (Gx < 0)
		a = 180.f - a;
	if (Gy < 0)
		a = 360.f - a;
	// normalize and copy to dst
	vx_uint8 norm_arct = (vx_uint8)((vx_uint32)(a*scale + 0.5) & 0xFF);
	return norm_arct;
}

template <typename T>
vx_status printImage(T *buffer, vx_uint32 stride_x, vx_uint32 stride_y, vx_uint32 width, vx_uint32 height)
{
	for (int i = 0; i < height; i++, printf("\n"))
		for(int j = 0; j < width; j++)
			printf("<%d,%d>: %d\t",i, j, buffer[i * stride_y + j * stride_x]);
	
	return VX_SUCCESS;
}

template <typename T>
vx_status printBuffer(T *buffer, vx_uint32 width, vx_uint32 height)
{
	T *bufferTemp;
	bufferTemp = buffer;
	for (int i = 0; i < height * width * 2; i++)
		printf("%d ", bufferTemp[i]);
	printf(".....\n");
	
	return VX_SUCCESS;
}

template <typename T>
vx_status printBufferBits(T *buffer, vx_uint32 bufferLength)
{
    T *bufferTemp;
	bufferTemp = buffer;
	unsigned int size = sizeof(T);
    unsigned int maxPow = 1<<(size*8-1);
	for (int i = 0; i < bufferLength; i++)
	{
		T packedVal = *bufferTemp;
		for (int j = 0; j < size * 8; ++j)
		{
			// print last bit and shift left.
			printf("%u ",packedVal&maxPow ? 1 : 0);
			packedVal = packedVal<<1;
		}
		bufferTemp++;
	}
	printf("\n");
	return VX_SUCCESS;
}

template <typename T>
vx_status makeInputImage(vx_context context, vx_image img, vx_uint32 width, vx_uint32 height, vx_enum mem_type, T pix_val)
{
	ERROR_CHECK_OBJECT((vx_reference)img);
	vx_rectangle_t rect = {0, 0, width, height};
	vx_map_id map_id;
	vx_imagepatch_addressing_t addrId;
	T *ptr;
	printf("pix_val = %d",pix_val);
	vx_uint32 stride_x_bytes, stride_x_pixels, stride_y_bytes, stride_y_pixels;
	ERROR_CHECK_STATUS(vxMapImagePatch(img, &rect, 0, &map_id, &addrId, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
	stride_x_bytes = addrId.stride_x;
	stride_x_pixels = stride_x_bytes / sizeof(T);
	stride_y_bytes = addrId.stride_y;
	stride_y_pixels = stride_y_bytes / sizeof(T);
	for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				ptr[i * stride_y_pixels + j * stride_x_pixels] = pix_val;
	ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#ifdef PRINT_INPUT
	printf("\nInput Image: ");
	printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", width, height, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
	printImage(ptr, stride_x_pixels, stride_y_pixels, width, height);
	printf("Input Buffer: ");
	printBuffer(ptr, width, height);
#endif
	vxReleaseImage(&img);
	return VX_SUCCESS;
}


template <typename T>
vx_status makeInputImageYUYV(vx_context context, vx_image img, vx_uint32 width, vx_uint32 height, vx_enum mem_type, T pix_val)
{
	ERROR_CHECK_OBJECT((vx_reference)img);
	vx_rectangle_t rect = {0, 0, width, height};
	vx_map_id map_id;
	vx_imagepatch_addressing_t addrId;
	T *ptr;
	printf("pix_val = %d",pix_val);
	vx_uint32 stride_x_bytes, stride_x_pixels, stride_y_bytes, stride_y_pixels;
	ERROR_CHECK_STATUS(vxMapImagePatch(img, &rect, 0, &map_id, &addrId, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
	// addrId.stride_x = 1;
	// stride_x_bytes = 1;
	stride_x_bytes = addrId.stride_x;
	stride_x_pixels = stride_x_bytes / sizeof(T);
	stride_y_bytes = addrId.stride_y;
	stride_y_pixels = stride_y_bytes / sizeof(T);
	for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				ptr[i * stride_y_pixels + j * stride_x_pixels] = i * stride_y_pixels + j * stride_x_pixels;
	ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#ifdef PRINT_INPUT
	printf("\nInput Image: ");
	printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", width, height, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
	printImage(ptr, stride_x_pixels, stride_y_pixels, width, height);
	printf("Input Buffer: ");
	printBuffer(ptr, width, height);
#endif
	vxReleaseImage(&img);
	return VX_SUCCESS;
}



template <typename T>
vx_status makeInputImageRGB(vx_context context, vx_image img, vx_uint32 width, vx_uint32 height, vx_enum mem_type, T pix_val)
{
	ERROR_CHECK_OBJECT((vx_reference)img);
	vx_rectangle_t rect = {0, 0, width, height};
	vx_map_id map_id;
	vx_imagepatch_addressing_t addrId;
	T *ptr;
	printf("pix_val = %d",pix_val);
	vx_uint32 stride_x_bytes, stride_x_pixels, stride_y_bytes, stride_y_pixels;
	ERROR_CHECK_STATUS(vxMapImagePatch(img, &rect, 0, &map_id, &addrId, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
	stride_x_bytes = addrId.stride_x ;
	stride_x_pixels = stride_x_bytes / sizeof(T);
	stride_y_bytes = addrId.stride_y;
	stride_y_pixels = stride_y_bytes / sizeof(T);
	for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				ptr[i * stride_y_pixels + j * stride_x_pixels] = i * stride_y_pixels + j * stride_x_pixels;
	ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#ifdef PRINT_INPUT
	printf("\nInput Image: ");
	printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", width, height, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
	printImage(ptr, stride_x_pixels, stride_y_pixels, width, height);
	printf("Input Buffer: ");
	printBuffer(ptr, width, height);
#endif
	vxReleaseImage(&img);
	return VX_SUCCESS;
}
int main(int argc, const char ** argv)
{
	// check command-line usage
    const size_t MIN_ARG_COUNT = 5;
    if(argc < MIN_ARG_COUNT)
	{
		printf("\nUsage: ./hipvx_sample <case number (1:99)> <width> <height> <gpu=1/cpu=0> <image1 constant pixel value (optional)> <image2 constant pixel value (optional)>\n");
		return -1;
    }
	
	// setup void ptr for HIP
	void *ptr[3] = {nullptr, nullptr, nullptr};

	// input and output images
	vx_image img1, img2, img_out;

	// setup argument reads and defaults
	vx_uint32 case_number = atoi(argv[1]);
	vx_uint32 width = atoi(argv[2]);
	vx_uint32 height = atoi(argv[3]);
	vx_uint32 device_affinity = atoi(argv[4]);
	vx_int32 pix_img1 = (argc < 6) ?  125 : atoi(argv[5]);
	vx_int32 pix_img2 =  (argc < 7) ?  132 : atoi(argv[6]);

	// required variables and initializations
	vx_int32 missing_function_flag = 0;
	vx_int32 return_value = 0;
	vx_int32 pix_img1_u1 = (vx_int32) PIXELCHECKU1(pix_img1);
	vx_int32 pix_img2_u1 = (vx_int32) PIXELCHECKU1(pix_img2);
	vx_int32 pix_img1_u8 = (vx_int32) PIXELCHECKU8(pix_img1);
	vx_int32 pix_img2_u8 = (vx_int32) PIXELCHECKU8(pix_img2);
	vx_uint32 pix_img1_u16 = (vx_uint16) PIXELCHECKU16(pix_img1);
	vx_uint32 pix_img1_u32 = (vx_uint32) PIXELCHECKU32(pix_img1);
	vx_int32 pix_img1_s16 = (vx_int32) PIXELCHECKS16(pix_img1);
	vx_int32 pix_img2_s16 = (vx_int32) PIXELCHECKS16(pix_img2);
	vx_uint8 *out_buf_uint8;
	vx_int16 *out_buf_int16;
	vx_uint32 out_buf_type;  // (0 - Output type U8 / 1 - Output type S16)
	vx_int32 expected_image_sum, returned_image_sum;
	vx_uint32 stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels;

	if (width <= 0) width = (vx_uint32) 5;
	if (height <= 0) height = (vx_uint32) 5;
	if (device_affinity <= 0) device_affinity = 0;

	// create context, create graph, set affinity, run graph, retrieve output
	vxRegisterLogCallback(NULL, log_callback, vx_false_e);
	vx_context context = vxCreateContext();
	vx_status status = vxGetStatus((vx_reference)context);
	if(status)
	{
		printf("ERROR: vxCreateContext() failed\n");
		return -1;
	}
	vxRegisterLogCallback(context, log_callback, vx_false_e);
	vx_graph graph = vxCreateGraph(context);
	vx_node node;
	vx_rectangle_t out_rect = {0, 0, width, height};
	vx_rectangle_t out_rect_half = {0, 0, width/2, height};
	vx_map_id  out_map_id;
	vx_imagepatch_addressing_t out_addr = {0};
	AgoTargetAffinityInfo affinity;
	affinity.device_info = 0;

	// arguments for specific functionalities	
	vx_float32 Mul_scale_float = (vx_float32) (1.0 / 16.0);
	vx_scalar Mul_scale_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*) &Mul_scale_float);
	// vx_lut LUT = vxCreateLUT(context, VX_TYPE_UINT8, 256);
	vx_int32 Threshold_thresholdValue_int32 = (vx_int32) 100;
	vx_int32 Threshold_thresholdLower_int32 = (vx_int32) 100;
	vx_int32 Threshold_thresholdUpper_int32 = (vx_int32) 200;
	vx_threshold Threshold_thresholdObjectBinary_threshold = vxCreateThreshold(context, VX_THRESHOLD_TYPE_BINARY, VX_TYPE_UINT8);
	vx_threshold Threshold_thresholdObjectRange_threshold = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_UINT8);
	ERROR_CHECK_STATUS(vxSetThresholdAttribute(Threshold_thresholdObjectBinary_threshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE, (void*) &Threshold_thresholdValue_int32, (vx_size)sizeof(vx_int32)));
	ERROR_CHECK_STATUS(vxSetThresholdAttribute(Threshold_thresholdObjectRange_threshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, (void*) &Threshold_thresholdLower_int32, (vx_size)sizeof(vx_int32)));
	ERROR_CHECK_STATUS(vxSetThresholdAttribute(Threshold_thresholdObjectRange_threshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, (void*) &Threshold_thresholdUpper_int32, (vx_size)sizeof(vx_int32)));
	vx_float32 WeightedAverage_alpha_float = (vx_float32) (0.25);
	vx_scalar WeightedAverage_alpha_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*) &WeightedAverage_alpha_float);
	vx_uint8 Lut_lutPtr_uint8[256];
	for (int i = 0; i < 256; i++)
		Lut_lutPtr_uint8[i] = (vx_uint8)(255 - i);
	vx_lut Lut_lutObject_lut = vxCreateLUT(context, VX_TYPE_UINT8, (vx_size)256);
	ERROR_CHECK_STATUS(vxCopyLUT(Lut_lutObject_lut, (void*)Lut_lutPtr_uint8, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

	if (!device_affinity)
	{
		affinity.device_type = AGO_TARGET_AFFINITY_CPU;

		if (graph)
		{
			ERROR_CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity)));

			switch(case_number)
			{
				case 1:
				{					
					// test_case_name = "agoKernel_AbsDiff_U8_U8U8"; 
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAbsDiffNode(graph, img1, img2, img_out);
					expected_image_sum = abs(pix_img1_u8 - pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 2:
				{
					// test_case_name = "agoKernel_AbsDiff_S16_S16S16_Sat";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAbsDiffNode(graph, img1, img2, img_out);
					expected_image_sum = ((vx_int32) PIXELCHECKS16(abs(pix_img1_s16 - pix_img2_s16))) * width * height;
					out_buf_type = 1;
					break;
				}
				case 3:
				{
					// test_case_name = "agoKernel_Add_U8_U8U8_Wrap";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					// expected_image_sum = generic_mod(pix_img1_u8 + pix_img2_u8, 256) * width * height;
					expected_image_sum = (vx_uint8)(pix_img1_u8 + pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 4:
				{
					// test_case_name = "agoKernel_Add_U8_U8U8_Sat";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
					expected_image_sum = ((vx_int32) PIXELCHECKU8(pix_img1_u8 + pix_img2_u8)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 5:
				{
					// test_case_name = "agoKernel_Add_S16_U8U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					expected_image_sum = (pix_img1_u8 + pix_img2_u8) * width * height;
					out_buf_type = 1;
					break;
				}
				case 6:
				{
					// test_case_name = "agoKernel_Add_S16_S16U8_Wrap";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					expected_image_sum = (vx_int16)(pix_img1_s16 + pix_img2_u8) * width * height;
					out_buf_type = 1;
					break;
				}
				case 7:
				{
					// test_case_name = "agoKernel_Add_S16_S16U8_Sat";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
					expected_image_sum = ((vx_int32) PIXELCHECKS16(pix_img1_s16 + pix_img2_u8)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 8:
				{
					// test_case_name = "agoKernel_Add_S16_S16S16_Wrap";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					expected_image_sum = ((vx_int16)(pix_img1_s16 + pix_img2_s16)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 9:
				{
					// test_case_name = "agoKernel_Add_S16_S16S16_Sat";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
					expected_image_sum = ((vx_int32)(PIXELCHECKS16(pix_img1_s16 + pix_img2_s16))) * width * height;
					out_buf_type = 1;
					break;
				}
				case 10:
				{
					// test_case_name = "agoKernel_Sub_U8_U8U8_Wrap";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					// expected_image_sum = generic_mod(pix_img1_u8 - pix_img2_u8, 256) * width * height;
					expected_image_sum = ((vx_uint8)(pix_img1_u8 - pix_img2_u8)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 11:
				{
					// test_case_name = "agoKernel_Sub_U8_U8U8_Sat";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
					expected_image_sum = ((vx_int32) PIXELCHECKU8(pix_img1_u8 - pix_img2_u8)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 12:
				{
					// test_case_name = "agoKernel_Sub_S16_U8U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					expected_image_sum = (pix_img1_u8 - pix_img2_u8) * width * height;
					out_buf_type = 1;
					break;
				}
				case 13:
				{
					// test_case_name = "agoKernel_Sub_S16_S16U8_Wrap";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					expected_image_sum = ((vx_int16)(pix_img1_s16 - pix_img2_u8)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 14:
				{
					// test_case_name = "agoKernel_Sub_S16_S16U8_Sat";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
					expected_image_sum = ((vx_int32) PIXELCHECKS16(pix_img1_s16 - pix_img2_u8)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 15:
				{
					// test_case_name = "HipExec_Sub_S16_U8S16_Wrap";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					expected_image_sum = ((vx_int16)(pix_img1_u8 - pix_img2_s16)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 16:
				{
					// test_case_name = "HipExec_Sub_S16_U8S16_Wrap";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
					expected_image_sum = ((vx_int16) PIXELCHECKS16(pix_img1_u8 - pix_img2_s16)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 17:
				{
					// test_case_name = "agoKernel_Sub_S16_S16S16_Wrap";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					expected_image_sum = ((vx_int16)(pix_img1_s16 - pix_img2_s16)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 18:
				{
					// test_case_name = "agoKernel_Sub_S16_S16S16_Sat";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
					expected_image_sum = ((vx_int32)(PIXELCHECKS16(pix_img1_s16 - pix_img2_s16))) * width * height;
					out_buf_type = 1;
					break;
				}
				case 19:
				{
					// test_case_name = "agoKernel_Mul_U8_U8U8_Wrap_Trunc";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = generic_mod((vx_int32)(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float), 256) * width * height;
					out_buf_type = 0;
					break;
				}
				case 20:
				{
					// test_case_name = "agoKernel_Mul_U8_U8U8_Wrap_Round";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = generic_mod((vx_int32)PIXELROUNDF32(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float), 256) * width * height;
					out_buf_type = 0;
					break;
				}
				case 21:
				{
					// test_case_name = "agoKernel_Mul_U8_U8U8_Sat_Trunc";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = ((vx_int32)PIXELCHECKU8((vx_int32)(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float))) * width * height;
					out_buf_type = 0;
					break;
				}
				case 22:
				{
					// test_case_name = "agoKernel_Mul_U8_U8U8_Sat_Round";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = ((vx_int32)PIXELCHECKU8((vx_int32)PIXELROUNDF32(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float))) * width * height;
					out_buf_type = 0;
					break;
				}
				case 23:
				{
					// test_case_name = "agoKernel_Mul_S16_U8U8_Wrap_Trunc";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = ((vx_int16)(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 24:
				{
					// test_case_name = "agoKernel_Mul_S16_U8U8_Wrap_Round";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = ((vx_int16)PIXELROUNDF32(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 25:
				{
					// test_case_name = "agoKernel_Mul_S16_U8U8_Sat_Trunc";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = ((vx_int16)(PIXELCHECKS16((vx_int32)(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float)))) * width * height;
					out_buf_type = 1;
					break;
				}
				case 26:
				{
					// test_case_name = "agoKernel_Mul_S16_U8U8_Sat_Round";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = ((vx_int16)(PIXELCHECKS16((vx_int32)PIXELROUNDF32(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float)))) * width * height;
					out_buf_type = 1;
					break;
				}
				case 27:
				{
					// test_case_name = "agoKernel_Mul_S16_S16U8_Wrap_Trunc";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = ((vx_int32)(((vx_float32)(pix_img1_s16 * pix_img2_u8)) * Mul_scale_float)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 28:
				{
					// test_case_name = "agoKernel_Mul_S16_S16U8_Wrap_Round";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = ((vx_int16)PIXELROUNDF32(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float)) * width * height;
					// expected_image_sum = generic_mod((vx_int32)(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float), 256) * width * height;
					out_buf_type = 1;
					break;
				}
				case 29:
				{
					// test_case_name = "agoKernel_Mul_S16_S16U8_Sat_Trunc";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = ((vx_int16)(PIXELCHECKS16((vx_int32)(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float)))) * width * height;
					// expected_image_sum = generic_mod((vx_int32)(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float), 256) * width * height;
					out_buf_type = 1;
					break;
				}
				case 30:
				{
					// test_case_name = "agoKernel_Mul_S16_S16U8_Sat_Round";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = ((vx_int32)PIXELCHECKU8((vx_int32)PIXELROUNDF32(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float))) * width * height;
					// expected_image_sum = generic_mod((vx_int32)(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float), 256) * width * height;
					out_buf_type = 1;
					break;
				}
				case 31:
				{
					// test_case_name = "agoKernel_Mul_S16_S16S16_Wrap_Trunc";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = (vx_int32)(((vx_float32)(pix_img1_s16 * pix_img2_s16)) * Mul_scale_float) * width * height;
					out_buf_type = 1;
					break;
				}
				case 32:
				{
					// test_case_name = "agoKernel_Mul_S16_S16S16_Wrap_Round";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = (vx_int32)PIXELROUNDF32(((vx_float32)(pix_img1_s16 * pix_img2_s16)) * Mul_scale_float) * width * height;
					out_buf_type = 1;
					break;
				}
				case 33:
				{
					// test_case_name = "agoKernel_Mul_S16_S16S16_Sat_Trunc";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = ((vx_int32)PIXELCHECKS16((vx_int32)(((vx_float32)(pix_img1_s16 * pix_img2_s16)) * Mul_scale_float))) * width * height;
					out_buf_type = 1;
					break;
				}
				case 34:
				{
					// test_case_name = "agoKernel_Mul_S16_S16S16_Sat_Round";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = ((vx_int32)PIXELCHECKS16((vx_int32)PIXELROUNDF32(((vx_float32)(pix_img1_s16 * pix_img2_s16)) * Mul_scale_float))) * width * height;
					out_buf_type = 1;
					break;
				}
				case 37:
				{
					// test_case_name = "agoKernel_And_U8_U8U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 & pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 38:
				{
					// test_case_name = "agoKernel_And_U8_U8U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 & pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 39:
				{
					// test_case_name = "agoKernel_And_U8_U1U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 & pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 40:
				{
					// test_case_name = "agoKernel_And_U8_U1U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 & pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 41:
				{
					// test_case_name = "agoKernel_And_U1_U8U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 & pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 42:
				{
					// test_case_name = "agoKernel_And_U1_U8U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 & pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 43:
				{
					// test_case_name = "agoKernel_And_U1_U1U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 & pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 44:
				{
					// test_case_name = "agoKernel_And_U1_U1U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (vx_int32)(pix_img1_u1 & pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 45:
				{
					// test_case_name = "agoKernel_Not_U8_U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxNotNode(graph, img1, img_out);
					expected_image_sum = (255 - pix_img1_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 46:
				{
					// test_case_name = "agoKernel_Not_U8_U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxNotNode(graph, img1, img_out);
					expected_image_sum = (255 - pix_img1_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 47:
				{
					// test_case_name = "agoKernel_Not_U1_U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxNotNode(graph, img1, img_out);
					expected_image_sum = (255 - pix_img1_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 48:
				{
					// test_case_name = "agoKernel_Not_U1_U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxNotNode(graph, img1, img_out);
					expected_image_sum = (255 - pix_img1_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 49:
				{
					// test_case_name = "agoKernel_Or_U8_U8U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 | pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 50:
				{
					// test_case_name = "agoKernel_Or_U8_U8U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 | pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 51:
				{
					// test_case_name = "agoKernel_Or_U8_U1U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 | pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 52:
				{
					// test_case_name = "agoKernel_Or_U8_U1U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 | pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 53:
				{
					// test_case_name = "agoKernel_Or_U1_U8U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1_AMD);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 | pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 54:
				{
					// test_case_name = "agoKernel_Or_U1_U8U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 | pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 55:
				{
					// test_case_name = "agoKernel_Or_U1_U1U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 | pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 56:
				{
					// test_case_name = "agoKernel_Or_U1_U1U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 | pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 57:
				{
					// test_case_name = "agoKernel_Xor_U8_U8U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 ^ pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 58:
				{
					// test_case_name = "agoKernel_Xor_U8_U8U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 ^ pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 59:
				{
					// test_case_name = "agoKernel_Xor_U8_U1U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 ^ pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 60:
				{
					// test_case_name = "agoKernel_Xor_U8_U1U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 ^ pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 61:
				{
					// test_case_name = "agoKernel_Xor_U1_U8U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 ^ pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 62:
				{
					// test_case_name = "agoKernel_Xor_U1_U8U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 ^ pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 63:
				{
					// test_case_name = "agoKernel_Xor_U1_U1U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 ^ pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 64:
				{
					// test_case_name = "agoKernel_Xor_U1_U1U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 ^ pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 65:
				{
					// test_case_name = "agoKernel_Magnitude_S16_S16S16";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxMagnitudeNode(graph, img1, img2, img_out);
					vx_int16 z = ((vx_int16) (sqrt ((double)((vx_uint32)((vx_int32)pix_img1_s16 * (vx_int32)pix_img1_s16) + (vx_uint32)((vx_int32)pix_img2_s16 * (vx_int32)pix_img2_s16))))); 
					expected_image_sum = (z > INT16_MAX ? INT16_MAX : z) * width * height;
					out_buf_type = 1;
					break;
				}
				case 66:
				{
					// test_case_name = "agoKernel_Phase_U8_S16S16";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxPhaseNode(graph, img1, img2, img_out); 
					expected_image_sum = Norm_Atan2_deg(pix_img1_s16, pix_img2_s16) * width * height;
					out_buf_type = 0;
					break;
				}
				case 67:
				{
					// test_case_name = "agoKernel_ChannelCopy_U8_U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_NV12);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_Y, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 71:
				{
					// test_case_name = "agoKernel_ChannelExtract_U8_U16_Pos0";
					/* Doubtfull
					if(width%2 != 0)
						width = (width + 1)&~1;
					if(height%2 != 0)
						height = (height + 1)&~1;
						*/
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_YUYV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, (vx_df_image)VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, (vx_enum)VX_CHANNEL_Y, img_out);
					expected_image_sum = pix_img1_u8 * width* height; //Needs Change
					out_buf_type = 0;
					break;
				}
				case 72:
				{
					// test_case_name = "agoKernel_ChannelExtract_U8_U16_Pos1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_UYVY);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_Y, img_out);
					expected_image_sum = pix_img1_u8 * width * height; //Needs Change
					out_buf_type = 0;
					break;
				}
				case 73:
				{
				// case 73 - agoKernel_ChannelExtract_U8_U24_Pos0
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_R, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 74:
				{
				// case 74 - agoKernel_ChannelExtract_U8_U24_Pos1
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_G, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 75:
				{
				// case 75 - agoKernel_ChannelExtract_U8_U24_Pos2
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_B, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 76:
				{
				// case 76 - agoKernel_ChannelExtract_U8_U32_Pos0
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_UYVY);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width/2, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_U, img_out);
					expected_image_sum = pix_img1_u8 * width/2 * height;
					out_buf_type = 2;
					break;
				}
				case 77:
				{
				// case 77 - agoKernel_ChannelExtract_U8_U32_Pos1
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_YUYV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width/2, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_U, img_out);
					expected_image_sum = pix_img1_u8 * width/2 * height;
					out_buf_type = 2;
					break;
				}
				case 78:
				{
				// case 78 - agoKernel_ChannelExtract_U8_U32_Pos2
				img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_UYVY);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width/2, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_V, img_out);
					expected_image_sum = pix_img1_u8 * width/2 * height;
					out_buf_type = 2;
					break;
				}
				case 79:
				{
				// case 79 - agoKernel_ChannelExtract_U8_U32_Pos3
				img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_YUYV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width/2, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_V, img_out);
					expected_image_sum = pix_img1_u8 * width/2 * height;
					out_buf_type = 2;
					break;
				}				 
				case 88:
				{
					// test_case_name = "agoKernel_Lut_U8_U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxTableLookupNode(graph, img1, Lut_lutObject_lut, img_out);
					expected_image_sum = (vx_int32) Lut_lutPtr_uint8[pix_img1_u8] * width * height;
					out_buf_type = 0;
					break;
				}
				case 89:
				{
					// test_case_name = "agoKernel_Threshold_U8_U8_Binary";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 255 : 0) * width * height;
					out_buf_type = 0;
					break;
				}
				case 90:
				{
					// test_case_name = "agoKernel_Threshold_U8_U8_Range";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 0 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 0 : 255)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 91:
				{
					// test_case_name = "agoKernel_Threshold_U1_U8_Binary";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 1 : 0) * width * height;
					out_buf_type = 0;
					break;
				}
				case 92:
				{
					// test_case_name = "agoKernel_Threshold_U1_U8_Range";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1_AMD);
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 0 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 0 : 1)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 93:
				{
					// test_case_name = "agoKernel_ThresholdNot_U8_U8_Binary";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 0 : 255) * width * height;
					out_buf_type = 0;
					break;
				}
				case 94:
				{
					// test_case_name = "agoKernel_ThresholdNot_U8_U8_Range";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 255 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 95:
				{
					// test_case_name = "agoKernel_ThresholdNot_U1_U8_Binary";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 0 : 1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 96:
				{
					// test_case_name = "agoKernel_ThresholdNot_U1_U8_Range";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 1 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 1 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 99:
				{
					// test_case_name = "agoKernel_WeightedAverage_U8_U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					// node = vxWeightedAverageNode(graph, img1, WeightedAverage_alpha_scalar, img2, img_out); // vxWeightedAverageNode not implemented
					expected_image_sum = (((vx_float32)pix_img1_u8 * ((vx_float32)1 - WeightedAverage_alpha_float)) + ((vx_float32)pix_img1_u8 * WeightedAverage_alpha_float)) * width * height;
					out_buf_type = 0;
					break;
				}
				default:
				{
					missing_function_flag = 1;
					break;
				}
			}
		
			if (node && !missing_function_flag)
			{
				status = vxVerifyGraph(graph);
				// U8U8 inputs
				if (
					(case_number == 1) || (case_number == 3) || (case_number == 4) || (case_number == 5) || 
					(case_number == 10) || (case_number == 11) || (case_number == 12) || (case_number == 19) || 
					(case_number == 20) || (case_number == 21) || (case_number == 22) || (case_number == 23) || 
					(case_number == 24) || (case_number == 25) || (case_number == 26) || (case_number == 37) ||
					(case_number == 41) || (case_number == 49) || (case_number == 53) || (case_number == 57) ||
					(case_number == 61) || (case_number == 99)
					)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img2_u8));
				}
				// S16U8 inputs
				else if (
					(case_number == 6) || (case_number == 7) || (case_number == 13) || (case_number == 14) || 
					(case_number == 27) || (case_number == 28) || (case_number == 29) || (case_number == 30)
					)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_int16) pix_img1_s16));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img2_u8));
				}
				// S16S16 inputs
				else if (
					(case_number == 2) || (case_number == 8) || (case_number == 9) || (case_number == 17) || 
					(case_number == 18) || (case_number == 31) || (case_number == 32) || (case_number == 33) || 
					(case_number == 34) || (case_number == 65) || (case_number == 66)
					)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_int16) pix_img1_s16));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HOST, (vx_int16) pix_img2_s16));
				}
				// U8S16 inputs
				else if(
					(case_number == 15) || (case_number == 16)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HOST, (vx_int16) pix_img2_s16));
				}
				// U8 input
				else if(
					(case_number == 45) || (case_number == 47) || (case_number == 67) || (case_number == 88) || 
					(case_number == 89) || (case_number == 90) || (case_number == 91) || (case_number == 92) || 
					(case_number == 93) || (case_number == 94) || (case_number == 95) || (case_number == 96)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u8));
				}
				// U1 input
				else if(
					(case_number == 46) || (case_number == 48)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u1));
				}
				// U1U1 inputs
				else if(
					(case_number == 40) || (case_number == 44) || (case_number == 52) || (case_number == 56) || 
					(case_number == 60) || (case_number == 64)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u1));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img2_u1));
				}
				// U8U1 inputs
				else if(
					(case_number == 38) || (case_number == 42) || (case_number == 50) || (case_number == 54) || 
					(case_number == 58) || (case_number == 62)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img2_u1));
				}
				// U1U8 inputs
				else if(
					(case_number == 39) || (case_number == 43) || (case_number == 51) || (case_number == 55)|| 
					(case_number == 59) || (case_number == 63)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u1));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img2_u8));
				}
				// U16  & U32 inputs
				else if(
					(case_number == 71)  || (case_number == 72) ||
					(case_number == 76)  || (case_number == 77) ||
					(case_number == 78)  || (case_number == 79)
				)
				{
					ERROR_CHECK_STATUS(makeInputImageYUYV(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u8))	
								}
				// U24 inputs
				else if(
					(case_number == 73) || (case_number == 74) || (case_number == 75)
				)
				{
					ERROR_CHECK_STATUS(makeInputImageRGB(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u8))	
								}
				if (status == VX_SUCCESS)
					status = vxProcessGraph(graph);
				vxReleaseNode(&node);
			}
			vxReleaseGraph(&graph);
		}
	}
	else
	{
		vx_imagepatch_addressing_t hip_addr_uint8 = {0};
		hip_addr_uint8.dim_x = width;
		hip_addr_uint8.dim_y = height;
		hip_addr_uint8.stride_x = 1;
		hip_addr_uint8.stride_y = (width+3)&~3;
		hipMalloc((void**)&ptr[0], height * hip_addr_uint8.stride_y);
		hipMalloc((void**)&ptr[1], height * hip_addr_uint8.stride_y);
		hipMalloc((void**)&ptr[2], height * hip_addr_uint8.stride_y);
		hipMemset(ptr[2], 0, height * hip_addr_uint8.stride_y);

		vx_imagepatch_addressing_t hip_addr_int16 = {0};
		hip_addr_int16.dim_x = width;
		hip_addr_int16.dim_y = height;
		hip_addr_int16.stride_x = 2;
		hip_addr_int16.stride_y = ((width+3)&~3)*2;
		hipMalloc((void**)&ptr[0], height * hip_addr_int16.stride_y);
		hipMalloc((void**)&ptr[1], height * hip_addr_int16.stride_y);
		hipMalloc((void**)&ptr[2], height * hip_addr_int16.stride_y);
		hipMemset(ptr[2], 0, height * hip_addr_int16.stride_y);

		vx_imagepatch_addressing_t hip_addr_uint8_yuyv_uyuv_in = {0};
		hip_addr_uint8_yuyv_uyuv_in.dim_x = width;
		hip_addr_uint8_yuyv_uyuv_in.dim_y = height;
		hip_addr_uint8_yuyv_uyuv_in.stride_x = 2;
		hip_addr_uint8_yuyv_uyuv_in.stride_y = (width+15)&~15;
		hipMalloc((void**)&ptr[0], height * hip_addr_uint8_yuyv_uyuv_in.stride_y);
		hipMalloc((void**)&ptr[1], height * hip_addr_uint8_yuyv_uyuv_in.stride_y);
		hipMalloc((void**)&ptr[2], height * hip_addr_uint8_yuyv_uyuv_in.stride_y);
		hipMemset(ptr[2], 0, height * hip_addr_uint8_yuyv_uyuv_in.stride_y);


		vx_imagepatch_addressing_t hip_addr_uint8_yuyv_uyuv_out = {0};
		hip_addr_uint8_yuyv_uyuv_out.dim_x = width/2;
		hip_addr_uint8_yuyv_uyuv_out.dim_y = height;
		hip_addr_uint8_yuyv_uyuv_out.stride_x = 1;
		hip_addr_uint8_yuyv_uyuv_out.stride_y = ((width+15)&~15);
		hipMalloc((void**)&ptr[0], height * hip_addr_uint8_yuyv_uyuv_out.stride_y);
		hipMalloc((void**)&ptr[1], height * hip_addr_uint8_yuyv_uyuv_out.stride_y);
		hipMalloc((void**)&ptr[2], height * hip_addr_uint8_yuyv_uyuv_out.stride_y);
		hipMemset(ptr[2], 0, height * hip_addr_uint8_yuyv_uyuv_out.stride_y);

		vx_imagepatch_addressing_t hip_addr_uint8_rgb_in = {0};
		hip_addr_uint8_rgb_in.dim_x = width;
		hip_addr_uint8_rgb_in.dim_y = height;
		hip_addr_uint8_rgb_in.stride_x = 3;
		hip_addr_uint8_rgb_in.stride_y = (width+15)&~15;
		hipMalloc((void**)&ptr[0], height * hip_addr_uint8_rgb_in.stride_y);
		hipMalloc((void**)&ptr[1], height * hip_addr_uint8_rgb_in.stride_y);
		hipMalloc((void**)&ptr[2], height * hip_addr_uint8_rgb_in.stride_y);
		hipMemset(ptr[2], 0, height * hip_addr_uint8_rgb_in.stride_y);

		vx_imagepatch_addressing_t hip_addr_uint8_rgb_out = {0};
		hip_addr_uint8_rgb_out.dim_x = width;
		hip_addr_uint8_rgb_out.dim_y = height;
		hip_addr_uint8_rgb_out.stride_x = 1;
		hip_addr_uint8_rgb_out.stride_y = (width+15)&~15;
		hipMalloc((void**)&ptr[0], height * hip_addr_uint8_rgb_out.stride_y);
		hipMalloc((void**)&ptr[1], height * hip_addr_uint8_rgb_out.stride_y);
		hipMalloc((void**)&ptr[2], height * hip_addr_uint8_rgb_out.stride_y);
		hipMemset(ptr[2], 0, height * hip_addr_uint8_rgb_out.stride_y);

		affinity.device_type = AGO_TARGET_AFFINITY_GPU;
		
		if (graph)
		{
			ERROR_CHECK_STATUS(vxSetGraphAttribute(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity)));
			
			switch(case_number)
			{
				case 1:
				{
					// test_case_name = "agoKernel_AbsDiff_U8_U8U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAbsDiffNode(graph, img1, img2, img_out);
					expected_image_sum = abs(pix_img1_u8 - pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 2:
				{
					// test_case_name = "agoKernel_AbsDiff_S16_S16S16_Sat";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAbsDiffNode(graph, img1, img2, img_out);
					expected_image_sum = ((vx_int32) PIXELCHECKS16(abs(pix_img1_s16 - pix_img2_s16))) * width * height;
					out_buf_type = 1;
					break;				
				}
				case 3:
				{
					// test_case_name = "agoKernel_Add_U8_U8U8_Wrap";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					// expected_image_sum = generic_mod(pix_img1_u8 + pix_img2_u8, 256) * width * height;
					expected_image_sum = (vx_uint8)(pix_img1_u8 + pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 4:
				{
					// test_case_name = "agoKernel_Add_U8_U8U8_Sat";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
					expected_image_sum = ((vx_int32) PIXELCHECKU8(pix_img1_u8 + pix_img2_u8)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 5:
				{
					// test_case_name = "agoKernel_Add_S16_U8U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					expected_image_sum = (pix_img1_u8 + pix_img2_u8) * width * height;
					out_buf_type = 1;
					break;
				}
				case 6:
				{
					// test_case_name = "agoKernel_Add_S16_S16U8_Wrap";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					expected_image_sum = (vx_int16)(pix_img1_s16 + pix_img2_u8) * width * height;
					out_buf_type = 1;
					break;
				}
				case 7:
				{
					// test_case_name = "agoKernel_Add_S16_S16U8_Sat";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
					expected_image_sum = ((vx_int32) PIXELCHECKS16(pix_img1_s16 + pix_img2_u8)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 8:
				{
					// test_case_name = "agoKernel_Add_S16_S16S16_Wrap";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					expected_image_sum = (vx_int16)(pix_img1_s16 + pix_img2_s16) * width * height;
					out_buf_type = 1;
					break;
				}
				case 9:
				{
					// test_case_name = "agoKernel_Add_S16_S16S16_Sat";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAddNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
					expected_image_sum = ((vx_int32)(PIXELCHECKS16(pix_img1_s16 + pix_img2_s16))) * width * height;
					out_buf_type = 1;
					break;
				}
				case 10:
				{
					// test_case_name = "agoKernel_Sub_U8_U8U8_Wrap";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					// expected_image_sum = generic_mod(pix_img1_u8 - pix_img2_u8, 256) * width * height;
					expected_image_sum = (vx_uint8)(pix_img1_u8 - pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 11:
				{
					// test_case_name = "agoKernel_Sub_U8_U8U8_Sat";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
					expected_image_sum = ((vx_int32) PIXELCHECKU8(pix_img1_u8 - pix_img2_u8)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 12:
				{
					// test_case_name = "agoKernel_Sub_S16_U8U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					expected_image_sum = (pix_img1_u8 - pix_img2_u8) * width * height;
					out_buf_type = 1;
					break;
				}
				case 13:
				{
					// test_case_name = "agoKernel_Sub_S16_S16U8_Wrap";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					expected_image_sum = ((vx_int16)(pix_img1_s16 - pix_img2_u8)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 14:
				{
					// test_case_name = "agoKernel_Sub_S16_S16U8_Sat";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
					expected_image_sum = ((vx_int32) PIXELCHECKS16(pix_img1_s16 - pix_img2_u8)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 15:
				{
					// test_case_name = "agoKernel_Sub_S16_U8S16_Wrap";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					expected_image_sum = ((vx_int16)(pix_img1_u8 - pix_img2_s16)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 16:
				{
					// test_case_name = "agoKernel_Sub_S16_U8S16_Sat";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
					expected_image_sum = ((vx_int16)(PIXELCHECKS16(pix_img1_u8 - pix_img2_s16))) * width * height;
					out_buf_type = 1;
					break;
				}
				case 17:
				{
					// test_case_name = "agoKernel_Sub_S16_S16S16_Wrap";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_WRAP, img_out);
					expected_image_sum = (vx_int16)(pix_img1_s16 - pix_img2_s16) * width * height;
					out_buf_type = 1;
					break;
				}
				case 18:
				{
					// test_case_name = "agoKernel_Sub_S16_S16S16_Sat";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxSubtractNode(graph, img1, img2, VX_CONVERT_POLICY_SATURATE, img_out);
					expected_image_sum = ((vx_int32)(PIXELCHECKS16(pix_img1_s16 - pix_img2_s16))) * width * height;
					out_buf_type = 1;
					break;
				}
				case 19:
				{
					// test_case_name = "agoKernel_Mul_U8_U8U8_Wrap_Trunc";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = generic_mod((vx_int32)(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float), 256) * width * height;
					out_buf_type = 0;
					break;
				}
				case 20:
				{
					// test_case_name = "agoKernel_Mul_U8_U8U8_Wrap_Round";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = generic_mod((vx_int32)PIXELROUNDF32(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float), 256) * width * height;
					out_buf_type = 0;
					break;
				}
				case 21:
				{
					// test_case_name = "agoKernel_Mul_U8_U8U8_Sat_Trunc";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = ((vx_int32)PIXELCHECKU8((vx_int32)(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float))) * width * height;
					out_buf_type = 0;
					break;
				}
				case 22:
				{
					// test_case_name = "agoKernel_Mul_U8_U8U8_Sat_Round";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = ((vx_int32)PIXELCHECKU8((vx_int32)PIXELROUNDF32(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float))) * width * height;
					out_buf_type = 0;
					break;
				}
				case 23:
				{
					// test_case_name = "agoKernel_Mul_S16_U8U8_Wrap_Trunc";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = ((vx_int32)(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 24:
				{
					// test_case_name = "agoKernel_Mul_S16_U8U8_Wrap_Round";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = ((vx_int32)PIXELROUNDF32(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 25:
				{
					// test_case_name = "agoKernel_Mul_S16_U8U8_Sat_Trunc";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = ((vx_int32)(PIXELCHECKS16((vx_int32)(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float)))) * width * height;
					out_buf_type = 1;
					break;
				}
				case 26:
				{
					// test_case_name = "agoKernel_Mul_S16_U8U8_Sat_Round";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = ((vx_int16)(PIXELCHECKS16((vx_int32)PIXELROUNDF32(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float)))) * width * height;
					out_buf_type = 1;
					break;
				}
				case 27:
				{
					// test_case_name = "agoKernel_Mul_S16_S16U8_Wrap_Trunc";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = ((vx_int32)(((vx_float32)(pix_img1_s16 * pix_img2_u8)) * Mul_scale_float)) * width * height;
					out_buf_type = 1;
					break;
				}
				case 28:
				{
					// test_case_name = "agoKernel_Mul_S16_S16U8_Wrap_Round";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = ((vx_int32)PIXELROUNDF32(((vx_float32)(pix_img1_u8 * pix_img2_u8)) * Mul_scale_float)) * width * height;

					out_buf_type = 1;
					break;
				}
				case 29:
				{
					// test_case_name = "agoKernel_Mul_S16_S16U8_Sat_Trunc";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = ((vx_int32)(PIXELCHECKS16((vx_int32)(((vx_float32)(pix_img1_s16 * pix_img2_u8)) * Mul_scale_float)))) * width * height;

					out_buf_type = 1;
					break;
				}
				case 30:
				{
					// test_case_name = "agoKernel_Mul_S16_S16U8_Sat_Round";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = ((vx_int16)(PIXELCHECKS16((vx_int32)PIXELROUNDF32(((vx_float32)(pix_img1_s16 * pix_img2_u8)) * Mul_scale_float)))) * width * height;

					out_buf_type = 1;
					break;
				}
				case 31:
				{
					// test_case_name = "agoKernel_Mul_S16_S16S16_Wrap_Trunc";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = (vx_int32)(((vx_float32)(pix_img1_s16 * pix_img2_s16)) * Mul_scale_float) * width * height;
					out_buf_type = 1;
					break;
				}
				case 32:
				{
					// test_case_name = "agoKernel_Mul_S16_S16S16_Wrap_Round";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_WRAP, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = (vx_int32)PIXELROUNDF32(((vx_float32)(pix_img1_s16 * pix_img2_s16)) * Mul_scale_float) * width * height;
					out_buf_type = 1;
					break;
				}
				case 33:
				{
					// test_case_name = "agoKernel_Mul_S16_S16S16_Sat_Trunc";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_ZERO, img_out);
					expected_image_sum = ((vx_int32)PIXELCHECKS16((vx_int32)(((vx_float32)(pix_img1_s16 * pix_img2_s16)) * Mul_scale_float))) * width * height;
					out_buf_type = 1;
					break;
				}
				case 34:
				{
					// test_case_name = "agoKernel_Mul_S16_S16S16_Sat_Round";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMultiplyNode(graph, img1, img2, Mul_scale_scalar, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN, img_out);
					expected_image_sum = ((vx_int32)PIXELCHECKS16((vx_int32)PIXELROUNDF32(((vx_float32)(pix_img1_s16 * pix_img2_s16)) * Mul_scale_float))) * width * height;
					out_buf_type = 1;
					break;
				}
				case 37:
				{
					// test_case_name = "agoKernel_And_U8_U8U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 & pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 38:
				{
					// test_case_name = "agoKernel_And_U8_U8U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 & pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 39:
				{
					// test_case_name = "agoKernel_And_U8_U1U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 & pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 40:
				{
					// test_case_name = "agoKernel_And_U8_U1U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 & pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 41:
				{
					// test_case_name = "agoKernel_And_U1_U8U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 & pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 42:
				{
					// test_case_name = "agoKernel_And_U1_U8U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 & pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 43:
				{
					// test_case_name = "agoKernel_And_U1_U1U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 & pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 44:
				{
					// test_case_name = "agoKernel_And_U1_U1U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 & pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 45:
				{
					// test_case_name = "agoKernel_Not_U8_U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxNotNode(graph, img1, img_out);
					expected_image_sum = (255 - pix_img1_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 46:
				{
					// test_case_name = "agoKernel_Not_U8_U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxNotNode(graph, img1, img_out);
					expected_image_sum = (255 - pix_img1_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 47:
				{
					// test_case_name = "agoKernel_Not_U1_U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxNotNode(graph, img1, img_out);
					expected_image_sum = (255 - pix_img1_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 48:
				{
					// test_case_name = "agoKernel_Not_U1_U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxNotNode(graph, img1, img_out);
					expected_image_sum = (255 - pix_img1_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 49:
				{
					// test_case_name = "agoKernel_Or_U8_U8U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 | pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 50:
				{
					// test_case_name = "agoKernel_Or_U8_U8U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 | pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 51:
				{
					// test_case_name = "agoKernel_Or_U8_U1U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 | pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 52:
				{
					// test_case_name = "agoKernel_Or_U8_U1U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 | pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 53:
				{
					// test_case_name = "agoKernel_Or_U1_U8U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 | pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 54:
				{
					// test_case_name = "agoKernel_Or_U1_U8U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 | pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 55:
				{
					// test_case_name = "agoKernel_Or_U1_U1U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 | pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 56:
				{
					// test_case_name = "agoKernel_Or_U1_U1U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 | pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 57:
				{
					// test_case_name = "agoKernel_Xor_U8_U8U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 ^ pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 58:
				{
					// test_case_name = "agoKernel_Xor_U8_U8U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 ^ pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 59:
				{
					// test_case_name = "agoKernel_Xor_U8_U1U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 ^ pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 60:
				{
					// test_case_name = "agoKernel_Xor_U8_U1U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 ^ pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 61:
				{
					// test_case_name = "agoKernel_Xor_U1_U8U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 ^ pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 62:
				{
					// test_case_name = "agoKernel_Xor_U1_U8U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 ^ pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 63:
				{
					// test_case_name = "agoKernel_Xor_U1_U1U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 ^ pix_img2_u8) * width * height;
					out_buf_type = 0;
					break;
				}
				case 64:
				{
					// test_case_name = "agoKernel_Xor_U1_U1U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u1 ^ pix_img2_u1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 65:
				{
					// test_case_name = "agoKernel_Magnitude_S16_S16S16";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMagnitudeNode(graph, img1, img2, img_out);
					vx_int16 z = ((vx_int16) (sqrt ((double)((vx_uint32)((vx_int32)pix_img1_s16 * (vx_int32)pix_img1_s16) + (vx_uint32)((vx_int32)pix_img2_s16 * (vx_int32)pix_img2_s16))))); 
					expected_image_sum = (z > INT16_MAX ? INT16_MAX : z) * width * height;
					out_buf_type = 1;
					break;
				}
				case 66:
				{
					// test_case_name = "agoKernel_Phase_U8_S16S16";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxPhaseNode(graph, img1, img2, img_out);
					expected_image_sum = Norm_Atan2_deg(pix_img1_s16, pix_img2_s16) * width * height;
					out_buf_type = 0;
					break;
				}
				case 67:
				{
					// test_case_name = "agoKernel_ChannelCopy_U8_U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, 0, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 71:
				{
					// test_case_name = "agoKernel_ChannelExtract_U8_U16_Pos0";
					/* Doubtfull
					if(width%2 != 0)
						width = (width + 1)&~1;
					if(height%2 != 0)
						height = (height + 1)&~1;
						*/
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUYV, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, (vx_enum)VX_CHANNEL_Y, img_out);
					expected_image_sum = pix_img1_u8 * width* height; //Needs Change
					out_buf_type = 0;
					break;
				}
				case 72:
				{
					// test_case_name = "agoKernel_ChannelExtract_U8_U16_Pos1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_UYVY, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, (vx_enum)VX_CHANNEL_Y, img_out);
					expected_image_sum = pix_img1_u8 * width * height; //Needs Change
					out_buf_type = 0;
					break;
				}
				case 73:
				{
				// case 73 - agoKernel_ChannelExtract_U8_U24_Pos0
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_rgb_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, (vx_enum)VX_CHANNEL_R, img_out);
					expected_image_sum = pix_img1_u8 * width * height; //Needs Change
					out_buf_type = 0;
					break;
				}
				case 74:
				{
				// case 74 - agoKernel_ChannelExtract_U8_U24_Pos1
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_rgb_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, (vx_enum)VX_CHANNEL_G, img_out);
					expected_image_sum = pix_img1_u8 * width * height; //Needs Change
					out_buf_type = 0;
					break;
				}
				case 75:
				{
				// case 75 - agoKernel_ChannelExtract_U8_U24_Pos2
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_rgb_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, (vx_enum)VX_CHANNEL_B, img_out);
					expected_image_sum = pix_img1_u8 * width * height; //Needs Change
					out_buf_type = 0;
					break;
				}
				case 76:
				{
				// case 76 - agoKernel_ChannelExtract_U8_U32_Pos0
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_UYVY, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_yuyv_uyuv_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_U, img_out);
					expected_image_sum = pix_img1_u8 * width/2 * height;
					out_buf_type = 2;
					break;
				}
				case 77:
				{
				// case 77 - agoKernel_ChannelExtract_U8_U32_Pos1
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUYV, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_yuyv_uyuv_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_U, img_out);
					expected_image_sum = pix_img1_u8 * width/2 * height;
					out_buf_type = 2;
					break;
				}
				case 78:
				{
				// case 78 - agoKernel_ChannelExtract_U8_U32_Pos2
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_UYVY, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_yuyv_uyuv_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_V, img_out);
					expected_image_sum = pix_img1_u8 * width/2 * height;
					out_buf_type = 2;
					break;
				}
				case 79:
				{
				// case 79 - agoKernel_ChannelExtract_U8_U32_Pos3
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUYV, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_yuyv_uyuv_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_V, img_out);
					expected_image_sum = pix_img1_u8 * width/2 * height;
					out_buf_type = 2;
					break;
				}	
				case 88:
				{
					// test_case_name = "agoKernel_Lut_U8_U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxTableLookupNode(graph, img1, Lut_lutObject_lut, img_out);
					expected_image_sum = (vx_int32) Lut_lutPtr_uint8[pix_img1_u8] * width * height;
					out_buf_type = 0;
					break;
				}
				case 89:
				{
					// test_case_name = "agoKernel_Threshold_U8_U8_Binary";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 255 : 0) * width * height;
					out_buf_type = 0;
					break;
				}
				case 90:
				{
					// test_case_name = "agoKernel_Threshold_U8_U8_Range";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 0 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 0 : 255)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 91:
				{
					// test_case_name = "agoKernel_Threshold_U1_U8_Binary";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1_AMD, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 1 : 0) * width * height;
					out_buf_type = 0;
					break;
				}
				case 92:
				{
					// test_case_name = "agoKernel_Threshold_U1_U8_Range";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 0 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 0 : 1)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 93:
				{
					// test_case_name = "agoKernel_ThresholdNot_U8_U8_Binary";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 0 : 255) * width * height;
					out_buf_type = 0;
					break;
				}
				case 94:
				{
					// test_case_name = "agoKernel_ThresholdNot_U8_U8_Range";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 255 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 95:
				{
					// test_case_name = "agoKernel_ThresholdNot_U1_U8_Binary";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 0 : 1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 96:
				{
					// test_case_name = "agoKernel_ThresholdNot_U1_U8_Range";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
					expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 1 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 1 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 99:
				{
					// test_case_name = "agoKernel_WeightedAverage_U8_U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					// node = vxWeightedAverageNode(graph, img1, WeightedAverage_alpha_scalar, img2, img_out); // vxWeightedAverageNode not implemented
					expected_image_sum = (((vx_float32)pix_img1_u8 * ((vx_float32)1 - WeightedAverage_alpha_float)) + ((vx_float32)pix_img1_u8 * WeightedAverage_alpha_float)) * width * height;
					out_buf_type = 0;
					break;
				}
				default:
				{
					missing_function_flag = 1;
					break;
				}
			}
			
			if (node && !missing_function_flag)
			{
				status = vxVerifyGraph(graph);
				// U8U8 inputs
				if (
					(case_number == 1) || (case_number == 3) || (case_number == 4) || (case_number == 5) || 
					(case_number == 10) || (case_number == 11) || (case_number == 12) || (case_number == 19) || 
					(case_number == 20) || (case_number == 21) || (case_number == 22) || (case_number == 23) || 
					(case_number == 24) || (case_number == 25) || (case_number == 26) || (case_number == 37) || 
					(case_number == 41) || (case_number == 49) || (case_number == 53) || (case_number == 57) ||
					(case_number == 61) || (case_number == 99)
					)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img1_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img2_u8));
				}
				// S16U8 inputs
				else if (
					(case_number == 6) || (case_number == 7) || (case_number == 13) || (case_number == 14) || 
					(case_number == 27) || (case_number == 28) || (case_number == 29) || (case_number == 30)
					)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_int16) pix_img1_s16));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img2_u8));
				}
				// S16S16 inputs
				else if (
					(case_number == 2) || (case_number == 8) || (case_number == 9) || (case_number == 17) || 
					(case_number == 18) || (case_number == 31) || (case_number == 32) || (case_number == 33) || 
					(case_number == 34) || (case_number == 65) || (case_number == 66)
					)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_int16) pix_img1_s16));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HIP, (vx_int16) pix_img2_s16));
				}
				// U8S16 inputs
				else if (
					(case_number == 15) || (case_number == 16) 
					)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img1_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HIP, (vx_int16) pix_img2_s16));
				}
				// U8 input
				else if (
					(case_number == 45) || (case_number == 47) || (case_number == 67) || (case_number == 88) || 
					(case_number == 89) || (case_number == 90) || (case_number == 91) || (case_number == 92) || 
					(case_number == 93) || (case_number == 94) || (case_number == 95) || (case_number == 96)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img1_u8));
				}
				// U1 input
				else if (
					(case_number == 46) || (case_number == 48) 
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img1_u1));
				}
				// U1U1 inputs
				else if(
					(case_number == 40) || (case_number == 44) || (case_number == 52) || (case_number == 56) || 
					(case_number == 60) || (case_number == 64)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img1_u1));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img2_u1));
				}
				// U8U1 inputs
				else if(
					(case_number == 38) || (case_number == 42) || (case_number == 50) || (case_number == 54) || 
					(case_number == 58) || (case_number == 62)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img1_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img2_u1));
				}
				// U1U8 inputs
				else if(
					(case_number == 39) || (case_number == 43) || (case_number == 51) || (case_number == 55) || 
					(case_number == 59) || (case_number == 63)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img1_u1));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img2_u8));
				}
				// U16  & U32 inputs
				else if(
					(case_number == 71)  || (case_number == 72) ||
					(case_number == 76)  || (case_number == 77) ||
					(case_number == 78)  || (case_number == 79)
				)
				{
					ERROR_CHECK_STATUS(makeInputImageYUYV(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u8))	
								}
				// U24 inputs
				else if(
					(case_number == 73) || (case_number == 74) || (case_number == 75)
				)
				{
					ERROR_CHECK_STATUS(makeInputImageRGB(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u8))	
								}
				if (status == VX_SUCCESS)
					status = vxProcessGraph(graph);
				vxReleaseNode(&node);
			}
			vxReleaseGraph(&graph);
		}
	}

	if (missing_function_flag == 1)
	{
		printf("\n\nThe functionality at case %d doesn't exist!\n", case_number);
		return 0;
	}

	// print output and compute image sum according to output buffer type
	returned_image_sum = 0;

	// for uint8 outputs
	if (out_buf_type == 0)
	{
		ERROR_CHECK_STATUS(vxMapImagePatch(img_out, &out_rect, 0, &out_map_id, &out_addr, (void **)&out_buf_uint8, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		stride_x_bytes = out_addr.stride_x;
		stride_x_pixels = stride_x_bytes / sizeof(vx_uint8);
		stride_y_bytes = out_addr.stride_y;
		stride_y_pixels = stride_y_bytes / sizeof(vx_uint8);
#ifdef PRINT_OUTPUT
		printf("\nOutput Image: ");
		printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", width, height, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
		printImage(out_buf_uint8, stride_x_pixels, stride_y_pixels, width, height);
		printf("Output Buffer: ");
		printBuffer(out_buf_uint8, width, height);
		// printBufferBits(out_buf_uint8, width * height); // To print output interms of bits
#endif
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				returned_image_sum += out_buf_uint8[i * stride_y_pixels + j * stride_x_pixels];
	}
	// for uint8 outputs of half the dimensions of width of input image
	if (out_buf_type == 2)
	{
		ERROR_CHECK_STATUS(vxMapImagePatch(img_out, &out_rect_half, 0, &out_map_id, &out_addr, (void **)&out_buf_uint8, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		stride_x_bytes = out_addr.stride_x;
		stride_x_pixels = stride_x_bytes / sizeof(vx_uint8);
		stride_y_bytes = out_addr.stride_y;
		stride_y_pixels = stride_y_bytes / sizeof(vx_uint8);
#ifdef PRINT_OUTPUT
		printf("\nOutput Image: ");
		printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", width, height, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
		printImage(out_buf_uint8, stride_x_pixels, stride_y_pixels, width, height);
		printf("Output Buffer: ");
		printBuffer(out_buf_uint8, width, height);
		// printBufferBits(out_buf_uint8, width * height); // To print output interms of bits
#endif
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				returned_image_sum += out_buf_uint8[i * stride_y_pixels + j * stride_x_pixels];
	}

	// for int16 outputs
	else if (out_buf_type == 1)
	{
		ERROR_CHECK_STATUS(vxMapImagePatch(img_out, &out_rect, 0, &out_map_id, &out_addr, (void **)&out_buf_int16, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		stride_x_bytes = out_addr.stride_x;
		stride_x_pixels = stride_x_bytes / sizeof(vx_int16);
		stride_y_bytes = out_addr.stride_y;
		stride_y_pixels = stride_y_bytes / sizeof(vx_int16);
#ifdef PRINT_OUTPUT
		printf("\nOutput Image: ");
		printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", width, height, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
		printImage(out_buf_int16, stride_x_pixels, stride_y_pixels, width, height);
		printf("Output Buffer: ");
		printBufferBits(out_buf_int16, width* height);
#endif
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				returned_image_sum += out_buf_int16[i * stride_y_pixels + j * stride_x_pixels];
	}
	
	if (returned_image_sum != expected_image_sum)
	{
		printf("\nTEST FAILED: returned_image_sum = %d expected_image_sum = %d\n", returned_image_sum, expected_image_sum);
		return_value = -1;
	}
	else
	{
		printf("\nTEST PASSED: returned_image_sum = %d expected_image_sum = %d\n", returned_image_sum, expected_image_sum);
		return_value = 1;
	}

	ERROR_CHECK_STATUS(vxUnmapImagePatch(img_out, out_map_id));

	// free resources

	vxReleaseScalar(&Mul_scale_scalar);
	vxReleaseScalar(&WeightedAverage_alpha_scalar);
	vxReleaseThreshold(&Threshold_thresholdObjectBinary_threshold);
	vxReleaseThreshold(&Threshold_thresholdObjectRange_threshold);
	vxReleaseLUT(&Lut_lutObject_lut);
	if (ptr[0]) hipFree(ptr[0]);
	if (ptr[1]) hipFree(ptr[1]);
	if (ptr[2]) hipFree(ptr[2]);
	vxReleaseContext(&context);

	return return_value;
}
	
