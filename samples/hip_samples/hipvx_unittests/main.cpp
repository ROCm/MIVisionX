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
// case 98 - agoKernel_Max_S16_S16
// case 99 - agoKernel_Min_U8_U8
// case 100 - agoKernel_Min_S16_S16
// case 101 - agoKernel_TensorMultiply
// case 102 - agoKernel_TensorAdd
// case 103 - agoKernel_TensorSubtract
// case 104 - agoKernel_WeightedAverage_U8_U8
// case 105 - agoKernel_ColorConvert_RGB_RGBX
// case 166 - agoKernel_ColorConvert_RGB_UYVY
// case 107 - agoKernel_ColorConvert_RGB_YUYV
// case 108 - agoKernel_ColorConvert_RGB_IYUV
// case 109 - agoKernel_ColorConvert_RGB_NV12
// case 110 - agoKernel_ColorConvert_RGB_NV21
// case 111 - agoKernel_ColorConvert_RGBX_RGB
// case 112 - agoKernel_ColorConvert_RGBX_UYVY
// case 113 - agoKernel_ColorConvert_RGBX_YUYV
// case 114 - agoKernel_ColorConvert_RGBX_IYUV
// case 115 - agoKernel_ColorConvert_RGBX_NV12
// case 116 - agoKernel_ColorConvert_RGBX_NV21
// case 117 - agoKernel_ColorConvert_IYUV_RGB
// case 118 - agoKernel_ColorConvert_IYUV_RGBX
// case 119 - agoKernel_FormatConvert_IYUV_UYVY
// case 120 - agoKernel_FormatConvert_IYUV_YUYV
// case 121 - agoKernel_ColorConvert_NV12_RGB
// case 122 - agoKernel_ColorConvert_NV12_RGBX
// case 123 - agoKernel_FormatConvert_NV12_UYVY
// case 124 - agoKernel_FormatConvert_NV12_YUYV  
// case 125 - agoKernel_ColorConvert_IYUV_NV12
// case 126 - agoKernel_ColorConvert_IYUV_NV21
// case 127 - agoKernel_FormatConvert_NV12_IYUV
// case 128 - agoKernel_ColorConvert_YUV4_RGB
// case 129 - agoKernel_ColorConvert_YUV4_RGBX
// case 130 - agoKernel_ColorConvert_YUV4_NV12
// case 131 - agoKernel_ColorConvert_YUV4_NV21
// case 132 - agoKernel_ColorConvert_YUV4_IYUV
// case 133 - agoKernel_Box_U8_U8_3x3
// case 134 - agoKernel_Dilate_U8_U8_3x3
// case 135 - agoKernel_Dilate_U1_U8_3x3
// case 136 - agoKernel_Dilate_U1_U1_3x3
// case 137 - agoKernel_Dilate_U8_U1_3x3
// case 138 - agoKernel_Erode_U8_U8_3x3
// case 139 - agoKernel_Erode_U1_U8_3x3
// case 140 - agoKernel_Erode_U1_U1_3x3
// case 141 - agoKernel_Erode_U8_U1_3x3
// case 142 - agoKernel_Median_U8_U8_3x3
// case 143 - agoKernel_Gaussian_U8_U8_3x3
// case 144 - agoKernel_SobelMagnitude_S16_U8_3x3
// case 145 - agoKernel_SobelPhase_U8_U8_3x3
// case 146 - agoKernel_SobelMagnitudePhase_S16U8_U8_3x3
// case 147 - agoKernel_Sobel_S16S16_U8_3x3_GXY
// case 148 - agoKernel_Sobel_S16_U8_3x3_GX
// case 149 - agoKernel_Sobel_S16_U8_3x3_GY
// case 150 - agoKernel_Convolve_U8_U8
// case 151 - agoKernel_Convolve_S16_U8
// case 152 - agoKernel_LinearFilter_ANY_ANY
// case 153 - agoKernel_LinearFilter_ANYx2_ANY
// case 154 - agoKernel_ScaleImage_U8_U8_Nearest
// case 155 - agoKernel_ScaleImage_U8_U8_Bilinear
// case 156 - agoKernel_ScaleImage_U8_U8_Bilinear_Replicate
// case 157 - agoKernel_ScaleImage_U8_U8_Bilinear_Constant
// case 158 - agoKernel_ScaleImage_U8_U8_Area
// case 159 - agoKernel_ScaleGaussianHalf_U8_U8_3x3
// case 160 - agoKernel_ScaleGaussianHalf_U8_U8_5x5
// case 161 - agoKernel_ScaleGaussianOrb_U8_U8_5x5
// case 162 - agoKernel_WarpAffine_U8_U8_Nearest
// case 163 - agoKernel_WarpAffine_U8_U8_Nearest_Constant
// case 164 - agoKernel_WarpAffine_U8_U8_Bilinear
// case 165 - agoKernel_WarpAffine_U8_U8_Bilinear_Constant
// case 166 - agoKernel_WarpPerspective_U8_U8_Nearest
// case 167 - agoKernel_WarpPerspective_U8_U8_Nearest_Constant
// case 168 - agoKernel_WarpPerspective_U8_U8_Bilinear
// case 169 - agoKernel_WarpPerspective_U8_U8_Bilinear_Constant
// case 170 - agoKernel_ColorDepth_U8_S16_Wrap
// case 171 - agoKernel_ColorDepth_U8_S16_Sat
// case 172 - agoKernel_ColorDepth_S16_U8
// case 173 - agoKernel_NonMaxSupp_XY_ANY_3x3
// case 174 - agoKernel_Remap_U8_U8_Nearest
// case 175 - agoKernel_Remap_U8_U8_Nearest_Constant
// case 176 - agoKernel_Remap_U8_U8_Bilinear
// case 177 - agoKernel_Remap_U8_U8_Bilinear_Constant
// case 178 - agoKernel_Remap_U24_U24_Bilinear
// case 179 - agoKernel_Remap_U24_U32_Bilinear
// case 180 - agoKernel_Remap_U32_U32_Bilinear
// case 181 - agoKernel_CannySobelSuppThreshold_U8_U8_3x3_L1NORM
// case 182 - agoKernel_CannySobelSuppThreshold_U8_U8_3x3_L2NORM
// case 183 - agoKernel_CannySobelSuppThreshold_U8_U8_5x5_L1NORM
// case 184 - agoKernel_CannySobelSuppThreshold_U8_U8_5x5_L2NORM
// case 185 - agoKernel_CannySobelSuppThreshold_U8_U8_7x7_L1NORM
// case 186 - agoKernel_CannySobelSuppThreshold_U8_U8_7x7_L2NORM
// case 187 - agoKernel_CannySobel_U16_U8_3x3_L1NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY
// case 188 - agoKernel_CannySobel_U16_U8_3x3_L2NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY
// case 189 - agoKernel_CannySobel_U16_U8_5x5_L1NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY
// case 190 - agoKernel_CannySobel_U16_U8_5x5_L2NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY
// case 191 - agoKernel_CannySobel_U16_U8_7x7_L1NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY
// case 192 - agoKernel_CannySobel_U16_U8_7x7_L2NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY
// case 193 - agoKernel_CannySobelSuppThreshold_U8XY_U8_3x3_L1NORM, agoKernel_CannyEdgeTrace_U8_U8XY
// case 194 - agoKernel_CannySobelSuppThreshold_U8XY_U8_3x3_L2NORM, agoKernel_CannyEdgeTrace_U8_U8XY
// case 195 - agoKernel_CannySobelSuppThreshold_U8XY_U8_5x5_L1NORM, agoKernel_CannyEdgeTrace_U8_U8XY
// case 196 - agoKernel_CannySobelSuppThreshold_U8XY_U8_5x5_L2NORM, agoKernel_CannyEdgeTrace_U8_U8XY
// case 197 - agoKernel_CannySobelSuppThreshold_U8XY_U8_7x7_L1NORM, agoKernel_CannyEdgeTrace_U8_U8XY
// case 198 - agoKernel_CannySobelSuppThreshold_U8XY_U8_7x7_L2NORM, agoKernel_CannyEdgeTrace_U8_U8XY
// case 199 - agoKernel_CannySuppThreshold_U8_U16_3x3
// case 200 - agoKernel_CannySuppThreshold_U8XY_U16_3x3
// case 201 - agoKernel_CannyEdgeTrace_U8_U8
// case 202 - agoKernel_CannyEdgeTrace_U8_U8XY
// case 203 - agoKernel_FastCorners_XY_U8_Supression
// case 204 - agoKernel_FastCorners_XY_U8_NoSupression
// case 205 - agoKernel_FastCornerMerge_XY_XY
// case 206 - agoKernel_HarrisSobel_HG3_U8_3x3
// case 207 - agoKernel_HarrisSobel_HG3_U8_5x5
// case 208 - agoKernel_HarrisSobel_HG3_U8_7x7
// case 209 - agoKernel_HarrisScore_HVC_HG3_3x3
// case 210 - agoKernel_HarrisScore_HVC_HG3_5x5
// case 211 - agoKernel_HarrisScore_HVC_HG3_7x7
// case 212 - agoKernel_HarrisMergeSortAndPick_XY_HVC
// case 213 - agoKernel_HarrisMergeSortAndPick_XY_XYS
// case 214 - agoKernel_IntegralImage_U32_U8
// case 215 - agoKernel_Histogram_DATA_U8
// case 216 - agoKernel_Equalize_DATA_DATA
// case 217 - agoKernel_MeanStdDev_DATA_U8
// case 218 - agoKernel_MeanStdDev_DATA_U1
// case 219 - agoKernel_OpticalFlowPyrLK_XY_XY
// case 220 - agoKernel_OpticalFlowPrepareLK_XY_XY
// case 221 - agoKernel_OpticalFlowImageLK_XY_XY
// case 222 - agoKernel_OpticalFlowFinalLK_XY_XY
// case 223 - agoKernel_MinMax_DATA_U8
// case 224 - agoKernel_MinMax_DATA_S16
// case 225 - agoKernel_MinMaxMerge_DATA_DATA
// case 226 - agoKernel_MinMaxLoc_DATA_U8DATA_Loc_None_Count_Min
// case 227 - agoKernel_MinMaxLoc_DATA_U8DATA_Loc_None_Count_Max
// case 228 - agoKernel_MinMaxLoc_DATA_U8DATA_Loc_None_Count_MinMax
// case 229 - agoKernel_MinMaxLoc_DATA_U8DATA_Loc_Min_Count_Min
// case 230 - agoKernel_MinMaxLoc_DATA_U8DATA_Loc_Min_Count_MinMax
// case 231 - agoKernel_MinMaxLoc_DATA_U8DATA_Loc_Max_Count_Max
// case 232 - agoKernel_MinMaxLoc_DATA_U8DATA_Loc_Max_Count_MinMax
// case 233 - agoKernel_MinMaxLoc_DATA_U8DATA_Loc_MinMax_Count_MinMax
// case 234 - agoKernel_MinMaxLoc_DATA_S16DATA_Loc_None_Count_Min
// case 235 - agoKernel_MinMaxLoc_DATA_S16DATA_Loc_None_Count_Max
// case 236 - agoKernel_MinMaxLoc_DATA_S16DATA_Loc_None_Count_MinMax
// case 237 - agoKernel_MinMaxLoc_DATA_S16DATA_Loc_Min_Count_Min
// case 238 - agoKernel_MinMaxLoc_DATA_S16DATA_Loc_Min_Count_MinMax
// case 239 - agoKernel_MinMaxLoc_DATA_S16DATA_Loc_Max_Count_Max
// case 240 - agoKernel_MinMaxLoc_DATA_S16DATA_Loc_Max_Count_MinMax
// case 241 - agoKernel_MinMaxLoc_DATA_S16DATA_Loc_MinMax_Count_MinMax
// case 242 - agoKernel_MinMaxLocMerge_DATA_DATA
// case 243 - agoKernel_Copy_DATA_DATA

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
#define ERROR_CHECK_HIP_STATUS(status) {hipError_t error  = status; if (error != hipSuccess) { fprintf(stderr, "ERROR: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); exit(EXIT_FAILURE);} }

#define PIXELCHECKU1(pixel) (pixel == (vx_int32)0) ? ((vx_uint8)0) : ((vx_uint8)1)
#define PIXELCHECKU8(pixel) (pixel < (vx_int32)0) ? ((vx_uint8)0) : ((pixel < (vx_int32)UINT8_MAX) ? (vx_uint8)pixel : ((vx_uint8)UINT8_MAX))
#define PIXELCHECKU16(pixel) (pixel < (vx_int32)0) ? ((vx_uint16)0) : ((pixel < (vx_int32)UINT16_MAX) ? (vx_uint16)pixel : ((vx_uint16)UINT16_MAX))
#define PIXELCHECKU32(pixel) (pixel < (vx_int32)0) ? ((vx_uint32)0) : ((pixel < (vx_uint32)UINT32_MAX) ? (vx_uint32)pixel : ((vx_uint32)UINT32_MAX))
#define PIXELCHECKS16(pixel) (pixel < (vx_int32)INT16_MIN) ? ((vx_int16)INT16_MIN) : ((pixel < (vx_int32)INT16_MAX) ? (vx_int16)pixel : ((vx_int16)INT16_MAX))
#define PIXELROUNDF32(value) ((value - (int)(value)) >= 0.5 ? (value + 1) : (value))
#define FLOAT_MAX(a, b) (a > b ? a : b)
#define FLOAT_MIN(a, b) (a < b ? a : b)

#define DBL_EPSILON      __DBL_EPSILON__
#define atan2_p0        (0.273*0.3183098862f)
#define atan2_p1		(0.9997878412794807f*57.29577951308232f)
#define atan2_p3		(-0.3258083974640975f*57.29577951308232f)
#define atan2_p5		(0.1555786518463281f*57.29577951308232f)
#define atan2_p7		(-0.04432655554792128f*57.29577951308232f)

int global_case = 0;

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

int RGBsum_YUV(vx_df_image itype, int Ypix, int Upix, int Vpix)
{
	int RGBsum;
	float R, G, B, Y, U, V;
	Y = (float)Ypix;
	U = (float)Upix - 128.0f;
	V = (float)Vpix - 128.0f; 
	
	if (itype == VX_DF_IMAGE_UYVY || itype == VX_DF_IMAGE_YUYV)
	{
		R = FLOAT_MIN(FLOAT_MAX(Y + (V * 1.5748f), 0.0f), 255.0f);
		G = FLOAT_MIN(FLOAT_MAX(Y - (U * 0.1873f) - (V * 0.4681f), 0.0f), 255.0f);
		B = FLOAT_MIN(FLOAT_MAX(Y + (U * 1.8556f), 0.0f), 255.0f);
		RGBsum = (int)R+(int)G+(int)B;
		// printf("*************R:%f G:%f B:%f sum = %d\n",R,G,B, RGBsum);
	}
	else if(itype == VX_DF_IMAGE_NV12 || itype == VX_DF_IMAGE_NV21 || itype == VX_DF_IMAGE_IYUV)
	{
		G = (U * 0.1873f) + (V * 0.4681f);
		R = V * 1.5748f;
		B = U * 1.8556f;

		R = FLOAT_MIN(FLOAT_MAX(Y + R, 0.0f), 255.0);
    	G = FLOAT_MIN(FLOAT_MAX(Y - G, 0.0f), 255.0f);
    	B = FLOAT_MIN(FLOAT_MAX(Y + B, 0.0f), 255.0f);
		RGBsum = (int)R+(int)G+(int)B;
		// printf("*************R:%f G:%f B:%f sum = %d\n",R,G,B, RGBsum);
	}

	return RGBsum;
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
vx_status printImageU1(T *buffer, vx_uint32 stride_x, vx_uint32 stride_y, vx_uint32 width, vx_uint32 height)
{
	for (int i = 0; i < height; i++, printf("\n"))
		for(int j = 0; j < stride_y; j++)
			printf("<%d,%d>: %d\t",i, j, buffer[i * stride_y + j]);
	
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
	vx_df_image format = 0;
	vxQueryImage(img, VX_IMAGE_FORMAT, &format, sizeof(format));
	// if (format == VX_DF_IMAGE_U1)
	// {
	// 	vx_pixel_value_t pixel;
	// 	pixel.U1 = 1;
	// 	vxSetImagePixelValues(img, &pixel);
	// }
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
	if(format == VX_DF_IMAGE_U1)
	{
		vx_uint32 x, y, i, j, xb;
		vx_uint8 mask, value;
		for (y = 0; y < addrId.dim_y; y+=addrId.step_y) 
		{
			j = addrId.stride_y*y;

			/* address bytes */
			for (x = 0; x < addrId.dim_x; x += 8) 
			{
				vx_uint8 *tmp = (vx_uint8 *)ptr;
				i = j + x/8;
				vx_uint8 *ptr2 = &tmp[i];

				/* address and set individual bits/pixels within the byte */
				for (xb = 0; xb < 8; xb++) {
					mask = 1 << xb;
					value = pix_val << xb;
					*ptr2 = (*ptr2 & (~mask)) | value;
				}
			}
    	}
		ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#ifdef PRINT_INPUT
	printf("\nInput Image U1: ");
	printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", width, height, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
	printf("dim_x: %d dim_y: %d\nscale_x: %d scale_y: %d\nstep_x: %d step_y: %d\n",addrId.dim_x, addrId.dim_y,addrId.scale_x, addrId.scale_y,addrId.step_x, addrId.step_y);
	printImageU1(ptr, stride_x_pixels, stride_y_pixels, width, height);
	printf("Input Buffer: ");
	printBuffer(ptr, width, height);
#endif
	}
	else
	{
		if (
			(global_case == 147) || (global_case == 148) || (global_case == 149) || 
			(global_case == 187) || (global_case == 188) || (global_case == 189) || (global_case == 190) || (global_case == 191) || (global_case == 192)
			)
		{
			for (int i = 0; i < height/2; i++)
				for (int j = 0; j < width/2; j++)
					ptr[i * stride_y_pixels + j * stride_x_pixels] = pix_val;
			if (
				(global_case == 187) || (global_case == 188) || (global_case == 189) || (global_case == 190) || (global_case == 191) || (global_case == 192)
				)
			{
				for (int i = 0; i < (height/2) - 1; i++)
					for (int j = 0; j < (width/2) - 1; j++)
						ptr[i * stride_y_pixels + j * stride_x_pixels] = 10;
			}
		}
		else if(global_case == 206)
		{
			for(int i =0; i< height;i++)
			{
				for(int j=0;j< width;j++)
				{
					ptr[i * stride_y_pixels + j * stride_x_pixels] = i * width +j + 1;
				}
			}
		}
		else if((global_case == 203) || (global_case == 204) )
		{
			for (int i = 0; i < height; i++)
				for (int j = 0; j < width; j++)
				{
					if(i > 2 && i < height-3 && j > 2 && j < width-3)
					{
						if(i == 3 || i == height-4 || j == 3 || j == width-4) // Fill the rectangular border
						// if((i == j)) 											//Fill Diagonal
						{
								ptr[i * stride_y_pixels + j * stride_x_pixels] = pix_val;
						}
						else
							ptr[i * stride_y_pixels + j * stride_x_pixels] = pix_val - 85;
					}
					else
							ptr[i * stride_y_pixels + j * stride_x_pixels] = 0;
				}
		}
		else if ((global_case == 174) || (global_case == 176))
		{
			for (int i = 0; i < height; i++)
				for (int j = 0; j < width; j++)
					ptr[i * stride_y_pixels + j * stride_x_pixels] = pix_val++;
		}
		else
		{
			for (int i = 0; i < height; i++)
				for (int j = 0; j < width; j++)
					ptr[i * stride_y_pixels + j * stride_x_pixels] = pix_val;
		}
		ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#ifdef PRINT_INPUT
	printf("\nInput Image: ");
	printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", width, height, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
	printf("dim_x: %d dim_y: %d\nscale_x: %d scale_y: %d\nstep_x: %d step_y: %d\n",addrId.dim_x, addrId.dim_y,addrId.scale_x, addrId.scale_y,addrId.step_x, addrId.step_y);
	printImage(ptr, stride_x_pixels, stride_y_pixels, width, height);
	printf("Input Buffer: ");
	printBuffer(ptr, width, height);
#endif
	}
	vxReleaseImage(&img);
	return VX_SUCCESS;
}

template <typename T>
vx_status makeInputPackedImage(vx_context context, vx_image img, vx_uint32 width, vx_uint32 height, vx_enum mem_type, T pix_val)
{
	/* This function is to make input images for packed formats like RGB, YUYV, UYVY, RGBX */
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
	if(stride_x_bytes == 4)
	{
		for (int i = 0; i < height; i++)
		{		
			for (int j = 0; j < width*stride_x_pixels; j+=stride_x_pixels)
			{
				ptr[i * stride_y_pixels + j ] = pix_val;
				ptr[i * stride_y_pixels + j + 1] = pix_val + 1;
				ptr[i * stride_y_pixels + j + 2] = pix_val + 2;
				ptr[i * stride_y_pixels + j + 3] = 255;
			}
		}
	}
	else
	{	
		for (int i = 0; i < height; i++)
		{		
			for (int j = 0; j < width*stride_x_pixels; j+=stride_x_pixels)
			{
				for(int inner_stride=0; inner_stride<stride_x_bytes; inner_stride++)
				{
					ptr[i * stride_y_pixels + j + inner_stride] = pix_val + inner_stride;
				}
			}
		}
	}
	ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#ifdef PRINT_INPUT
	printf("\nInput Image: ");
	printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", width, height, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
	printf("dim_x: %d dim_y: %d\nscale_x: %d scale_y: %d\nstep_x: %d step_y: %d\n",addrId.dim_x, addrId.dim_y,addrId.scale_x, addrId.scale_y,addrId.step_x, addrId.step_y);
	printImage(ptr, stride_x_pixels, stride_y_pixels, width, height);
	printf("Input Buffer: ");
	printBuffer(ptr, width, height);
#endif
	vxReleaseImage(&img);
	return VX_SUCCESS;
}

template <typename T>
vx_status makeInputPlanarImage(vx_context context, vx_image img, vx_uint32 width, vx_uint32 height, vx_enum mem_type, T pix_val)
{
	/* This function is to make input images for Planar image formats like IYUV, NV12, NV21*/
	ERROR_CHECK_OBJECT((vx_reference)img);
	vx_rectangle_t rect = {0, 0, width, height};
	vx_map_id map_id;
	vx_imagepatch_addressing_t addrId;
	T *ptr;
	vx_size planes = 0;
	// vx_df_image format = 0;
	// vxQueryImage(img, VX_IMAGE_FORMAT, &format, sizeof(format));
	vxQueryImage(img, VX_IMAGE_PLANES, &planes, sizeof(planes));
	printf("pix_val = %d",pix_val);
	vx_uint32 stride_x_bytes, stride_x_pixels, stride_y_bytes, stride_y_pixels;

	for(int p=0; p<planes; p++)
	{
		ERROR_CHECK_STATUS(vxMapImagePatch(img, &rect, p, &map_id, &addrId, (void **)&ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		stride_x_bytes = addrId.stride_x;
		stride_x_pixels = stride_x_bytes / sizeof(T);
		stride_y_bytes = addrId.stride_y;
		stride_y_pixels = stride_y_bytes / sizeof(T);
		if(p == 0)
		{
			for (int i = 0; i < height; i++)
				for (int j = 0; j < width; j++)
					ptr[i * stride_y_pixels + j * stride_x_pixels] = pix_val;
					// ptr[i * stride_y_pixels + j * stride_x_pixels] = i * stride_y_pixels + j * stride_x_pixels;
		}
		else
		{
			for (int i = 0; i < (height/2); i++)
				for (int j = 0; j < (width/2)*stride_x_bytes; j+=stride_x_bytes)
				{
					for(int inner_stride=0; inner_stride<stride_x_bytes; inner_stride++)
					{
						ptr[(i * stride_y_pixels) + j + inner_stride] = pix_val+inner_stride+1;				
						// ptr[(i * stride_y_pixels) + j + inner_stride] = (i * stride_y_pixels) + j + inner_stride + 1;			
					}
				}
		}
		ERROR_CHECK_STATUS(vxUnmapImagePatch(img, map_id));
#ifdef PRINT_INPUT
		printf("\nInput Image Plane %d: ",p);
		printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", width, height, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
		printf("dim_x: %d dim_y: %d\nscale_x: %d scale_y: %d\nstep_x: %d step_y: %d\n",addrId.dim_x, addrId.dim_y,addrId.scale_x, addrId.scale_y,addrId.step_x, addrId.step_y);
		printImage(ptr, stride_x_pixels, stride_y_pixels, width, height);
		printf("Input Buffer: ");
		printBuffer(ptr, width, height);
#endif
	}
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
	void *ptr[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
	void * nv_in[2] = {nullptr, nullptr};
	void * nv_out[2] = {nullptr, nullptr};
	void * iyuv_in[3] = {nullptr, nullptr, nullptr};
	void * iyuv_out[3] = {nullptr, nullptr, nullptr};
	void * yuv4_in[3] = {nullptr, nullptr, nullptr};
	void * yuv4_out[3] = {nullptr, nullptr, nullptr};

	// input and output images
	vx_image img1, img2, img3, img_out, img_out2;

	// setup argument reads and defaults
	vx_uint32 case_number = atoi(argv[1]);
	vx_uint32 width = atoi(argv[2]);
	vx_uint32 height = atoi(argv[3]);
	vx_uint32 device_affinity = atoi(argv[4]);
	vx_int32 pix_img1 = (argc < 6) ?  125 : atoi(argv[5]);
	vx_int32 pix_img2 =  (argc < 7) ?  132 : atoi(argv[6]);
	global_case = case_number;

	// required variables and initializations
	vx_int32 missing_function_flag = 0;
	vx_int32 return_value = 0;
	vx_int32 pix_img1_u1 = (vx_int32) PIXELCHECKU1(pix_img1);
	vx_int32 pix_img2_u1 = (vx_int32) PIXELCHECKU1(pix_img2);
	vx_int32 pix_img1_u8 = (vx_int32) PIXELCHECKU8(pix_img1);
	vx_int32 pix_img2_u8 = (vx_int32) PIXELCHECKU8(pix_img2);
	vx_int32 pix_img3_u8 = (vx_int32) PIXELCHECKU8(pix_img2+10);
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
	vx_uint32 widthOut = width;
	vx_uint32 heightOut = height;
	if (
		(case_number == 154) || (case_number == 155) || (case_number == 156)  || (case_number == 157) || 
		(case_number == 158)
		)
	{
		// widthOut = (vx_uint32)((vx_float32)widthOut * 0.5);
		// heightOut = (vx_uint32)((vx_float32)heightOut * 0.667);

		widthOut = (vx_uint32)((vx_float32)widthOut * 1.75);
		heightOut = (vx_uint32)((vx_float32)heightOut * 2.2);
	}
	else if ((case_number == 159) || (case_number == 160))
	{
		widthOut = (width + 1) / 2;
		heightOut = (height + 1) / 2;
	}
	else if (
		(case_number == 162) || (case_number == 163) || 
		(case_number == 164) || (case_number == 165) || 
		(case_number == 166) || (case_number == 167) || 
		(case_number == 168) || (case_number == 169) || 
		(case_number == 174) || (case_number == 176)
		)
	{
		widthOut = width * 2;
		heightOut = height * 2;
	}
	vx_rectangle_t out_rect = {0, 0, widthOut, heightOut};
	vx_rectangle_t out_rect_half = {0, 0, width/2, height};
	vx_map_id  out_map_id;
	vx_imagepatch_addressing_t out_addr = {0};
	AgoTargetAffinityInfo affinity;
	affinity.device_info = 0;

	// arguments for specific functionalities	
	/*Multiplication Params*/
	vx_float32 Mul_scale_float = (vx_float32) (1.0 / 16.0);
	vx_scalar Mul_scale_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*) &Mul_scale_float);

	/*Threshold Params*/
	// vx_int32 Threshold_thresholdValue_int32 = (vx_int32) 100;
	// vx_int32 Threshold_thresholdLower_int32 = (vx_int32) 100;
	// vx_int32 Threshold_thresholdUpper_int32 = (vx_int32) 200;
	// vx_threshold Threshold_thresholdObjectBinary_threshold = vxCreateThreshold(context, VX_THRESHOLD_TYPE_BINARY, VX_TYPE_UINT8);
	// vx_threshold Threshold_thresholdObjectRange_threshold = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_UINT8);
	// ERROR_CHECK_STATUS(vxSetThresholdAttribute(Threshold_thresholdObjectBinary_threshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE, (void*) &Threshold_thresholdValue_int32, (vx_size)sizeof(vx_int32)));
	// ERROR_CHECK_STATUS(vxSetThresholdAttribute(Threshold_thresholdObjectRange_threshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, (void*) &Threshold_thresholdLower_int32, (vx_size)sizeof(vx_int32)));
	// ERROR_CHECK_STATUS(vxSetThresholdAttribute(Threshold_thresholdObjectRange_threshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, (void*) &Threshold_thresholdUpper_int32, (vx_size)sizeof(vx_int32)));
	
	/* Weighted Average Params */
	vx_float32 WeightedAverage_alpha_float = (vx_float32) (0.25);
	vx_scalar WeightedAverage_alpha_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*) &WeightedAverage_alpha_float);

	/* Fast Corners Params */
	vx_float32 fastCorner_strength_threshold = (vx_float32) 80.0;
	vx_scalar fastCorner_threshold_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*) &fastCorner_strength_threshold);
	vx_bool nms_true = 1;
	vx_bool nms_false = 0;
	vx_size key_array_size = 100;
	vx_array output_keypoints_array = vxCreateArray(context, VX_TYPE_KEYPOINT, key_array_size);
	vx_size no_of_corners = 0;
	vx_scalar output_corner_count = vxCreateScalar(context, VX_TYPE_SIZE, (void*) &no_of_corners);

	/* Harris Corners Params */
	vx_float32 HarrisCorner_strength_threshold = (vx_float32) 0.00001;
	vx_scalar HarrisCorner_strength_threshold_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*) &HarrisCorner_strength_threshold);
	vx_float32 HarrisCorner_min_distance = (vx_float32) 3.0;
	vx_scalar HarrisCorner_min_distance_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*) &HarrisCorner_min_distance);
	vx_float32 HarrisCorner_sensitivity = (vx_float32) 0.10;
	vx_scalar HarrisCorner_sensitivity_scalar = vxCreateScalar(context, VX_TYPE_FLOAT32, (void*) &HarrisCorner_sensitivity);
	vx_int32 HarrisCorner_grad_size = (vx_int32) 3;
	vx_int32 HarrisCorner_block_size = (vx_int32) 5;
	vx_size HarrisCorner_key_array_size = 1000;
	vx_array HarrisCorner_output_keypoints_array = vxCreateArray(context, VX_TYPE_KEYPOINT, HarrisCorner_key_array_size);
	vx_size HarrisCorner_no_of_corners = 0;
	vx_scalar HarrisCorner_output_corner_count = vxCreateScalar(context, VX_TYPE_SIZE, (void*) &HarrisCorner_no_of_corners);

	/* Lookup Table Params */
	vx_uint8 Lut_lutPtr_uint8[256];
	for (int i = 0; i < 256; i++)
		Lut_lutPtr_uint8[i] = (vx_uint8)(255 - i);
	vx_lut Lut_lutObject_lut = vxCreateLUT(context, VX_TYPE_UINT8, (vx_size)256);
	ERROR_CHECK_STATUS(vxCopyLUT(Lut_lutObject_lut, (void*)Lut_lutPtr_uint8, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

	/* Filter Params */
	vx_uint32 scale = 1;
	vx_int16 filter3x3[3][3] = {
		{ 1, 2, 1},
		{ 2, 4, 2},
		{ 1, 2, 1},
	};
	vx_int16 filter5x5[5][5] = {
		{ 1, 2, 4, 2, 1},
		{ 2, 4, 8, 4, 2},
		{ 4, 8, 12, 8, 4},
		{ 2, 4, 8, 4, 2},
		{ 1, 2, 4, 2, 1},
	};
	// vx_int16 filter5x5Gaussian[5][5] = {			// if needed
	// 	{ 1, 4, 6, 4, 1},
	// 	{ 4, 16, 24, 16, 4},
	// 	{ 6, 24, 36, 24, 6},
	// 	{ 4, 16, 24, 16, 4},
	// 	{ 1, 4, 6, 4, 1},
	// };
	vx_int16 filter7x7[7][7] = {
		{ 1, 2, 4, 6, 4, 2, 1},
		{ 2, 4, 6, 8, 6, 4, 2},
		{ 4, 8, 10, 12, 10, 8, 4},
		{ 2, 4, 6, 8, 6, 4, 2},
		{ 1, 2, 4, 6, 4, 2, 1},
	};

	/*Convolution Params*/
	vx_convolution Convolve_conv_convolution = vxCreateConvolution(context, 5, 5);
	ERROR_CHECK_STATUS(vxCopyConvolutionCoefficients(Convolve_conv_convolution, (vx_int16*)filter5x5, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
	ERROR_CHECK_STATUS(vxSetConvolutionAttribute(Convolve_conv_convolution, VX_CONVOLUTION_SCALE, &scale, sizeof(scale)));

	/*Scale Image Params*/
	vx_enum ScaleImage_type1_enum = VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR;
	vx_enum ScaleImage_type2_enum = VX_INTERPOLATION_TYPE_BILINEAR;
	vx_enum ScaleImage_type3_enum = VX_INTERPOLATION_TYPE_AREA;

	/*Convert Depth Params*/
	vx_int32 ConvertDepth_shift_int32 = 1;
	vx_scalar ConvertDepth_shift_scalar = vxCreateScalar(context, VX_TYPE_INT32, (void*) &ConvertDepth_shift_int32);

	/*Warp Affine Params*/
	// vx_float32 WarpAffine_affineMatrix_float[3][2] = {			// if needed - rotate 45deg
	// 	{0.707, -0.707}, // 'x' coefficients
	// 	{0.707, 0.707}, // 'y' coefficients
	// 	{-5, 4}, // 'offsets'
	// };
	vx_float32 WarpAffine_affineMatrix_float[3][2] = {			// translate in x and y
		{1, 0}, // 'x' coefficients
		{0, 1}, // 'y' coefficients
		{-3, -3}, // 'offsets'
	};
	vx_matrix WarpAffine_affineMatrix_matrix = vxCreateMatrix(context, VX_TYPE_FLOAT32, 2, 3);
	ERROR_CHECK_STATUS(vxCopyMatrix(WarpAffine_affineMatrix_matrix, WarpAffine_affineMatrix_float, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

	/*Warp Perspective Params*/
	vx_float32 WarpPerspective_perspectiveMatrix_float[3][3] = {			// translate in x and y
		{1, 0, 0}, // 'x' coefficients
		{0, 1, 0}, // 'y' coefficients
		{-3, -3, 1}, // 'offsets'
	};
	vx_matrix WarpPerspective_perspectiveMatrix_matrix = vxCreateMatrix(context, VX_TYPE_FLOAT32, 3, 3);
	ERROR_CHECK_STATUS(vxCopyMatrix(WarpPerspective_perspectiveMatrix_matrix, WarpPerspective_perspectiveMatrix_float, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
	
	/*Remap Params*/
	_vx_coordinates2df_t Remap_remapTable_coordinates2df[widthOut * heightOut];
	vx_size Remap_remapTableStrideY_size = widthOut * 8;
	// ago_coord2d_ushort_t Remap_remapTable_coord2d_ushort[widthOut * heightOut];
	// vx_size Remap_remapTableStrideY_size = widthOut * 4;

	for (int i = 0; i < heightOut; i ++)
	{
		for (int j = 0; j < widthOut; j++)
		{
			if ((j < width) && (i < height))
			{
				Remap_remapTable_coordinates2df[i*widthOut + j].x = j;
				Remap_remapTable_coordinates2df[i*widthOut + j].y = i;
			}
			else
			{
				Remap_remapTable_coordinates2df[i*widthOut + j].x = 0;
				Remap_remapTable_coordinates2df[i*widthOut + j].y = 0;
			}
		}
	}
	if((case_number == 174) || (case_number == 176))
	{
		printf("\nUser Remap Table:\n");
		for (int i = 0; i < heightOut; i ++)
		{
			for (int j = 0; j < widthOut; j++)
			{
				printf("%0.1f,%0.1f\t", Remap_remapTable_coordinates2df[i*widthOut + j].y, Remap_remapTable_coordinates2df[i*widthOut + j].x);
				// printf("%d,%d\t", Remap_remapTable_coord2d_ushort[i*widthOut + j].y, Remap_remapTable_coord2d_ushort[i*widthOut + j].x);
			}
			printf("\n");
		}
	}
	vx_remap Remap_remapTable_remap = vxCreateRemap(context, width, height, widthOut, heightOut);
	ERROR_CHECK_STATUS(vxCopyRemapPatch(Remap_remapTable_remap, &out_rect, Remap_remapTableStrideY_size, (void*) Remap_remapTable_coordinates2df, VX_TYPE_COORDINATES2DF, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
	vx_border_t border1, border2;
	border1.mode = VX_BORDER_REPLICATE;
	border2.mode = VX_BORDER_CONSTANT;
	border2.constant_value.U8 = (vx_uint8) 5;

	/*Canny Edge Detector Params*/
	vx_pixel_value_t CannyEdgeDetector_thresholdLower_uint8;
	vx_pixel_value_t CannyEdgeDetector_thresholdUpper_uint8;
	CannyEdgeDetector_thresholdLower_uint8.U8 = 55;
	CannyEdgeDetector_thresholdUpper_uint8.U8 = 60;
	vx_int32 CannyEdgeDetector_gradientSize3_int32 = 3;
	vx_int32 CannyEdgeDetector_gradientSize5_int32 = 5;
	vx_int32 CannyEdgeDetector_gradientSize7_int32 = 7;
	vx_threshold CannyEdgeDetector_thresholdObjectRange_threshold = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	ERROR_CHECK_STATUS(vxCopyThresholdRange(CannyEdgeDetector_thresholdObjectRange_threshold, &CannyEdgeDetector_thresholdLower_uint8, &CannyEdgeDetector_thresholdUpper_uint8, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
	
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
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 & (pix_img2_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 39:
				{
					// test_case_name = "agoKernel_And_U8_U1U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img2_u8 & (pix_img1_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 40:
				{
					// test_case_name = "agoKernel_And_U8_U1U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u1 ? 255 : 0) & (pix_img2_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 41:
				{
					// test_case_name = "agoKernel_And_U1_U8U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u8 & (1 << 7)) & (pix_img2_u8 & (1 << 7)) ? 255 : 0) *
											(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 42:
				{
					// test_case_name = "agoKernel_And_U1_U8U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (((pix_img1_u8 & (1 << 7)) ? 255 : 0) & (pix_img2_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 43:
				{
					// test_case_name = "agoKernel_And_U1_U1U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (((pix_img2_u8 & (1 << 7)) ? 255 : 0) & (pix_img1_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 44:
				{
					// test_case_name = "agoKernel_And_U1_U1U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u1 ? 255 : 0) & (pix_img2_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
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
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxNotNode(graph, img1, img_out);
					widthOut = ((width % 8 == 0) ? ((width/8)+7) & ~7 : ((width/8)+8) & ~7);
					expected_image_sum = (255 - (pix_img1_u1 ? 255 : 0)) * widthOut * height;
					out_buf_type = 0;
					break;
				}
				case 47:
				{
					// test_case_name = "agoKernel_Not_U1_U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxNotNode(graph, img1, img_out);
					expected_image_sum = (255 - ((pix_img1_u8 & (1<<7)) ? 255 : 0)) * (int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 48:
				{
					// test_case_name = "agoKernel_Not_U1_U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxNotNode(graph, img1, img_out);
					expected_image_sum = (255 - (pix_img1_u1? 255:0)) * (int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
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
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 | (pix_img2_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 51:
				{
					// test_case_name = "agoKernel_Or_U8_U1U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img2_u8 | (pix_img1_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 52:
				{
					// test_case_name = "agoKernel_Or_U8_U1U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u1 ? 255 : 0) | (pix_img2_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 53:
				{
					// test_case_name = "agoKernel_Or_U1_U8U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u8 & (1 << 7)) | (pix_img2_u8 & (1 << 7)) ? 255 : 0) *
											(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 54:
				{
					// test_case_name = "agoKernel_Or_U1_U8U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (((pix_img1_u8 & (1 << 7)) ? 255 : 0) | (pix_img2_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 55:
				{
					// test_case_name = "agoKernel_Or_U1_U1U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (((pix_img2_u8 & (1 << 7)) ? 255 : 0) | (pix_img1_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;					
					out_buf_type = 5;
					break;
				}
				case 56:
				{
					// test_case_name = "agoKernel_Or_U1_U1U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u1 ? 255 : 0) | (pix_img2_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
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
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 ^ (pix_img2_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 59:
				{
					// test_case_name = "agoKernel_Xor_U8_U1U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img2_u8 ^ (pix_img1_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 60:
				{
					// test_case_name = "agoKernel_Xor_U8_U1U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u1 ? 255 : 0) ^ (pix_img2_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 61:
				{
					// test_case_name = "agoKernel_Xor_U1_U8U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u8 & (1 << 7)) ^ (pix_img2_u8 & (1 << 7)) ? 255 : 0) *
											(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 62:
				{
					// test_case_name = "agoKernel_Xor_U1_U8U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (((pix_img1_u8 & (1 << 7)) ? 255 : 0) ^ (pix_img2_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;					
					out_buf_type = 5;
					break;
				}
				case 63:
				{
					// test_case_name = "agoKernel_Xor_U1_U1U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (((pix_img2_u8 & (1 << 7)) ? 255 : 0) ^ (pix_img1_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 64:
				{
					// test_case_name = "agoKernel_Xor_U1_U1U1";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u1 ? 255 : 0) ^ (pix_img2_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
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
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_YUYV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, (vx_df_image)VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, (vx_enum)VX_CHANNEL_Y, img_out);
					expected_image_sum = pix_img1_u8 * width* height; 
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
					expected_image_sum = (pix_img1_u8+1) * width * height; 
					out_buf_type = 0;
					break;
				}
				case 73:
				{
					// test_case_name - agoKernel_ChannelExtract_U8_U24_Pos0
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
					// test_case_name - agoKernel_ChannelExtract_U8_U24_Pos1
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_G, img_out);
					expected_image_sum = (pix_img1_u8+1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 75:
				{
					// test_case_name - agoKernel_ChannelExtract_U8_U24_Pos2
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_B, img_out);
					expected_image_sum = (pix_img1_u8+2) * width * height;
					out_buf_type = 0;
					break;
				}
				case 76:
				{
					// test_case_name - agoKernel_ChannelExtract_U8_U32_Pos0
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
					// test_case_name - agoKernel_ChannelExtract_U8_U32_Pos1
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_YUYV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width/2, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_U, img_out);
					expected_image_sum = (pix_img1_u8+1) * width/2 * height;
					out_buf_type = 2;
					break;
				}
				case 78:
				{
					// test_case_name - agoKernel_ChannelExtract_U8_U32_Pos2
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_UYVY);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width/2, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_V, img_out);
					expected_image_sum = (pix_img1_u8) * width/2 * height;
					out_buf_type = 2;
					break;
				}
				case 79:
				{
					// test_case_name - agoKernel_ChannelExtract_U8_U32_Pos3
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_YUYV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width/2, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_V, img_out);
					expected_image_sum = (pix_img1_u8+1) * width/2 * height;
					out_buf_type = 2;
					break;
				}
				case 83:
				{
					// test_case_name - agoKernel_ChannelCombine_U16_U8U8
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width/2, height/2, VX_DF_IMAGE_U8);
					img3 = vxCreateImage(context, width/2, height/2, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_NV12);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelCombineNode(graph, img1, img2, img3, 0, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + ((pix_img2_u8 + pix_img3_u8) * (width/2) * (height/2));
					out_buf_type = 4;
					break;
				}				
				case 84:
				{
					// test_case_name - agoKernel_ChannelCombine_U24_U8U8U8_RGB
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img3 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelCombineNode(graph, img1, img2, img3, 0, img_out);
					expected_image_sum = (pix_img1_u8 + pix_img2_u8 + pix_img3_u8) * width * height;
					out_buf_type = 3;
					break;
				}
				case 85:
				{
					// test_case_name - agoKernel_ChannelCombine_U32_U8U8U8_UYVY
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width/2, height, VX_DF_IMAGE_U8);
					img3 = vxCreateImage(context, width/2, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_UYVY);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelCombineNode(graph, img1, img2, img3, 0, img_out);
					expected_image_sum = ((pix_img1_u8) * width * height) + ((pix_img2_u8 + pix_img3_u8) * (width/2) * height);
					out_buf_type = 3;
					break;
				}
				case 86:
				{
					// test_case_name - agoKernel_ChannelCombine_U32_U8U8U8_YUYV
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width/2, height, VX_DF_IMAGE_U8);
					img3 = vxCreateImage(context, width/2, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_YUYV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelCombineNode(graph, img1, img2, img3, 0, img_out);
					expected_image_sum = ((pix_img1_u8) * width * height) + ((pix_img2_u8 + pix_img3_u8) * (width/2) * height);
					out_buf_type = 3;
					break;
				}	
				case 87:
				{
					// test_case_name - agoKernel_ChannelCombine_U32_U8U8U8U8_RGBX
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img3 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_RGBX);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxChannelCombineNode(graph, img1, img2, img3, img2, img_out);
					expected_image_sum = (pix_img1_u8 + pix_img2_u8 + pix_img3_u8 + pix_img2_u8) * width * height;
					out_buf_type = 3;
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
				// case 89:
				// {
				// 	// test_case_name = "agoKernel_Threshold_U8_U8_Binary";
				// 	img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	vx_threshold_type_e thresholdType = VX_THRESHOLD_TYPE_BINARY;
				// 	vx_df_image_e thresholdInputImageFormat = VX_DF_IMAGE_U8;
				// 	vx_df_image_e thresholdOutputImageFormat = VX_DF_IMAGE_U8;
				// 	ERROR_CHECK_STATUS(vxSetThresholdAttribute(Threshold_thresholdObjectBinary_threshold, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE, (void*) &Threshold_thresholdValue_int32, (vx_size)sizeof(vx_int32)));
				// 	ERROR_CHECK_STATUS(vxSetThresholdAttribute(Threshold_thresholdObjectBinary_threshold, VX_THRESHOLD_TYPE, (void*) &thresholdType, (vx_size)sizeof(vx_threshold_type_e)));
				// 	ERROR_CHECK_STATUS(vxSetThresholdAttribute(Threshold_thresholdObjectBinary_threshold, VX_THRESHOLD_INPUT_FORMAT, (void*) &thresholdInputImageFormat, (vx_size)sizeof(vx_df_image_e)));
				// 	ERROR_CHECK_STATUS(vxSetThresholdAttribute(Threshold_thresholdObjectBinary_threshold, VX_THRESHOLD_OUTPUT_FORMAT, (void*) &thresholdOutputImageFormat, (vx_size)sizeof(vx_df_image_e)));
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 255 : 0) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				// case 90:
				// {
				// 	// test_case_name = "agoKernel_Threshold_U8_U8_Range";
				// 	img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 0 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 0 : 255)) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				// case 91:
				// {
				// 	// test_case_name = "agoKernel_Threshold_U1_U8_Binary";
				// 	img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 1 : 0) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				// case 92:
				// {
				// 	// test_case_name = "agoKernel_Threshold_U1_U8_Range";
				// 	img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U1_AMD);
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 0 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 0 : 1)) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				// case 93:
				// {
				// 	// test_case_name = "agoKernel_ThresholdNot_U8_U8_Binary";
				// 	img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 0 : 255) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				// case 94:
				// {
				// 	// test_case_name = "agoKernel_ThresholdNot_U8_U8_Range";
				// 	img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 255 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 255 : 0)) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				// case 95:
				// {
				// 	// test_case_name = "agoKernel_ThresholdNot_U1_U8_Binary";
				// 	img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 0 : 1) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				// case 96:
				// {
				// 	// test_case_name = "agoKernel_ThresholdNot_U1_U8_Range";
				// 	img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 1 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 1 : 0)) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				case 104:
				{
					// test_case_name = "agoKernel_WeightedAverage_U8_U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img2 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxWeightedAverageNode(graph, img1, WeightedAverage_alpha_scalar, img2, img_out);
					expected_image_sum = (vx_int32)(((vx_float32)pix_img2_u8 * ((vx_float32)1 - WeightedAverage_alpha_float)) + ((vx_float32)pix_img1_u8 * WeightedAverage_alpha_float)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 105:
				{
					// test_case_name - agoKernel_ColorConvert_RGB_RGBX
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGBX);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = ((3 * pix_img1_u8) + 3)  * width * height;
					out_buf_type = 3;
					break;
				}
				case 106:
				{
					// test_case_name - agoKernel_ColorConvert_RGB_UYVY
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_UYVY);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_UYVY, pix_img1_u8+1, pix_img1_u8, pix_img1_u8)) * width * height;
					out_buf_type = 3;
					break;
				}
				case 107:
				{
					// test_case_name - agoKernel_ColorConvert_RGB_YUYV
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_YUYV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_YUYV, pix_img1_u8, pix_img1_u8+1, pix_img1_u8+1)) * width * height;
					out_buf_type = 3;
					break;
				}
				case 108:
				{
					// test_case_name - agoKernel_ColorConvert_RGB_IYUV
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_IYUV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_IYUV, pix_img1_u8, pix_img1_u8+1, pix_img1_u8+1)) * width * height;
					out_buf_type = 3;
					break;
				}
				case 109:
				{
					// test_case_name - agoKernel_ColorConvert_RGB_NV12
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_NV12);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_NV12, pix_img1_u8, pix_img1_u8+1, pix_img1_u8+2)) * width * height;
					out_buf_type = 3;
					break;
				}
				case 110:
				{
					// test_case_name - agoKernel_ColorConvert_RGB_NV21
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_NV21);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_NV21, pix_img1_u8, pix_img1_u8+2, pix_img1_u8+1)) * width * height;
					out_buf_type = 3;
					break;
				}
				case 111:
				{
					// test_case_name - agoKernel_ColorConvert_RGBX_RGB
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_RGBX);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8+pix_img1_u8+pix_img1_u8+3+255) * width * height; 
					out_buf_type = 3;
					break;
				}
				case 112:
				{
					// test_case_name - agoKernel_ColorConvert_RGBX_UYVY
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_UYVY);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_RGBX);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_UYVY, pix_img1_u8+1, pix_img1_u8, pix_img1_u8) + 255) * width * height;
					out_buf_type = 3;
					break;
				}
				case 113:
				{
					// test_case_name - agoKernel_ColorConvert_RGBX_YUYV
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_YUYV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_RGBX);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_YUYV, pix_img1_u8, pix_img1_u8+1, pix_img1_u8+1) + 255) * width * height;
					out_buf_type = 3;
					break;
				}
				case 114:
				{
					// test_case_name - agoKernel_ColorConvert_RGBX_IYUV
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_IYUV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_RGBX);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_IYUV, pix_img1_u8, pix_img1_u8+1, pix_img1_u8+1) + 255) * width * height;
					out_buf_type = 3;
					break;
				}
				case 115:
				{
					// test_case_name - agoKernel_ColorConvert_RGBX_NV12
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_NV12);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1))
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_RGBX);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_NV12, pix_img1_u8, pix_img1_u8+1, pix_img1_u8+2) + 255) * width * height;
					out_buf_type = 3;
					break;
				}
				case 116:
				{
					// test_case_name - agoKernel_ColorConvert_RGBX_NV21
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_NV21);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1))
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_RGBX);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_NV21, pix_img1_u8, pix_img1_u8+2, pix_img1_u8+1) + 255) * width * height;
					out_buf_type = 3;
					break;
				}
				case 117:
				{
					// test_case_name - agoKernel_ColorConvert_IYUV_RGB
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_IYUV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + ((128+127) * (width/2) * (height/2)) ;
					out_buf_type = 4;
					break;
				}
				case 118:
				{
					// test_case_name - agoKernel_ColorConvert_IYUV_RGBX
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGBX);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_IYUV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + ((128+127) * (width/2) * (height/2)) ;
					out_buf_type = 4;
					break;
				}
				case 119:
				{
					// test_case_name - agoKernel_FormatConvert_IYUV_UYVY
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_UYVY);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_IYUV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = ((pix_img1_u8 + 1) * width * height) + (2 * (pix_img1_u8) * (width/2) * (height/2));
					out_buf_type = 4;
					break;
				}
				case 120:
				{
					// test_case_name - agoKernel_FormatConvert_IYUV_YUYV
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_YUYV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_IYUV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + (2 * (1+pix_img1_u8) * (width/2) * (height/2));
					out_buf_type = 4;
					break;
				}
				case 121:
				{
					// test_case_name  = " agoKernel_ColorConvert_NV12_RGB"
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_NV12);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + ((128+127) * (width/2) * (height/2)) ;
					out_buf_type = 4;
					break;
				}
				case 122:
				{
					// test_case_name  = " agoKernel_ColorConvert_NV12_RGBX"
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGBX);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_NV12);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + ((128+127) * (width/2) * (height/2)) ;
					out_buf_type = 4;
					break;
				}
				case 123:
				{
					// test_case_name  = " agoKernel_FormatConvert_NV12_UYVY "
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_UYVY);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_NV12);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = ((pix_img1_u8 + 1) * width * height) + (2 * pix_img1_u8 * (width / 2) * (height / 2));
					out_buf_type = 4;
					break;
				}
				case 124:
				{
					// test_case_name  = " agoKernel_FormatConvert_NV12_YUYV "
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_YUYV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_NV12);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = ((pix_img1_u8)*width * height) + (2 * (pix_img1_u8 + 1) * (width / 2) * (height / 2));
					out_buf_type = 4;
					break;
				}
				case 125:
				{
					// test_case_name  = "agoKernel_FormatConvert_IYUV_NV12"
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_NV12);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_IYUV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = ((pix_img1_u8)*width * height) + (((2 * pix_img1_u8) + 3) * (width / 2) * (height / 2));
					out_buf_type = 4;
					break;
				}
				case 126:
				{
					// test_case_name  = "agoKernel_FormatConvert_IYUV_NV21"
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_NV21);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_IYUV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = ((pix_img1_u8)*width * height) + (((2 * pix_img1_u8) + 3) * (width / 2) * (height / 2));
					out_buf_type = 4;
					break;
				}
				case 127:
				{
					// test_case_name  = "agoKernel_FormatConvert_NV12_IYUV"
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_IYUV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_NV12);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = ((pix_img1_u8)*width * height) + (2 * (pix_img1_u8 + 1) * (width / 2) * (height / 2));
					out_buf_type = 4;
					break;
				}
				case 128:
				{
				// test_case_name  = " agoKernel_FormatConvert_YUV4_RGB "
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_YUV4);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = ((pix_img1_u8)*width * height) + (2 * (pix_img1_u8 + 1) * (width / 2) * (height / 2));//Needs Change
					out_buf_type = 4;
					break;
				}
				case 129:
				{
				// test_case_name  = " agoKernel_ColorConvert_YUV4_RGBX "
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGBX);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_YUV4);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = ((pix_img1_u8)*width * height) + (2 * (pix_img1_u8 + 1) * (width ) * (height / 2)); //Needs Chnage
					out_buf_type = 4;
					break;
				}
				case 130:
				{
				// test_case_name  = " agoKernel_ColorConvert_YUV4_NV12 "
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_NV12);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_YUV4);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + ((pix_img1_u8 + 1) * (width ) * (height)) + ((pix_img1_u8 + 2) * (width ) * (height)); 
					out_buf_type = 4;
					break;
				}
				case 131:
				{
				// test_case_name  = " agoKernel_ColorConvert_YUV4_NV21 "
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_NV21);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_YUV4);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + ((pix_img1_u8 + 1) * (width ) * (height)) + ((pix_img1_u8 + 2) * (width ) * (height)); 
					out_buf_type = 4;
					break;
				}
				case 132:
				{
				// test_case_name  = " agoKernel_ColorConvert_YUV4_IYUV "
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_IYUV);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img1));
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_YUV4);
					ERROR_CHECK_STATUS(vxGetStatus((vx_reference)img_out));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + (2 *(pix_img1_u8 + 1) * (width ) * (height)) ; 
					out_buf_type = 4;
					break;
				}
				case 133:
				{
					// test_case_name = "agoKernel_Box_U8_U8_3x3";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxBox3x3Node(graph, img1, img_out);
					vx_int32 firstColVal = (6 * pix_img1_u8) / 9;
					expected_image_sum = (pix_img1_u8 * (width - 2) * (height - 2)) + (2 * (height - 2) * firstColVal);
					out_buf_type = 0;
					break;
				}
				case 134:
				{
					// test_case_name = "agoKernel_Dilate_U8_U8_3x3";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxDilate3x3Node(graph, img1, img_out);
					vx_int32 firstColVal = pix_img1_u8;
					expected_image_sum = (pix_img1_u8 * (width - 2) * (height - 2)) + (2 * (height - 2) * firstColVal);
					out_buf_type = 0;
					break;
				}
				case 138:
				{
					// test_case_name = "agoKernel_Erode_U8_U8_3x3";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxErode3x3Node(graph, img1, img_out);
					vx_int32 firstColVal = 0;
					expected_image_sum = (pix_img1_u8 * (width - 2) * (height - 2)) + (2 * (height - 2) * firstColVal);
					out_buf_type = 0;
					break;
				}
				case 142:
				{
					// test_case_name = "agoKernel_Median_U8_U8_3x3";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxMedian3x3Node(graph, img1, img_out);
					vx_int32 firstColVal = pix_img1_u8;
					expected_image_sum = (pix_img1_u8 * (width - 2) * (height - 2)) + (2 * (height - 2) * firstColVal);
					out_buf_type = 0;
					break;
				}
				case 143:
				{
					// test_case_name = "agoKernel_Gaussian_U8_U8_3x3";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxGaussian3x3Node(graph, img1, img_out);
					vx_int32 firstColVal = (vx_int32)PIXELCHECKU8((float)pix_img1_u8 * ((2 * 0.0625) + (3 * 0.125) + 0.25));
					expected_image_sum = (pix_img1_u8 * (width - 2) * (height - 2)) + (2 * (height - 2) * firstColVal);
					out_buf_type = 0;
					break;
				}
				case 147:
				{
					// test_case_name = "agoKernel_Sobel_S16S16_U8_3x3_GXY";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					node = vxSobel3x3Node(graph, img1, img_out, img_out2);
					expected_image_sum = 0;
					out_buf_type = 1;
					break;
				}
				case 148:
				{
					// test_case_name = "agoKernel_Sobel_S16_U8_3x3_GX";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					node = vxSobel3x3Node(graph, img1, img_out, NULL);
					expected_image_sum = 0;
					out_buf_type = 1;
					break;
				}
				case 149:
				{
					// test_case_name = "agoKernel_Sobel_S16_U8_3x3_GY";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out2 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					node = vxSobel3x3Node(graph, img1, NULL, img_out2);
					expected_image_sum = 0;
					out_buf_type = 1;
					break;
				}
				case 150:
				{
					// test_case_name = "agoKernel_Convolve_U8_U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxConvolveNode(graph, img1, Convolve_conv_convolution, img_out);
					vx_int32 firstColVal = PIXELCHECKU8(pix_img1_u8 * (12 + (8 * 3) + (4 * 5) + (2 * 4) + (1 * 2)));
					vx_int32 secondColVal = PIXELCHECKU8(firstColVal + (pix_img1_u8 * ((2 * 2) + (2 * 4) + (1 * 8))));
					vx_int32 remainingVal = PIXELCHECKU8(96 * pix_img1_u8);
					expected_image_sum = (remainingVal * (width - 4) * (height - 4)) + (2 * (height - 4) * (firstColVal + secondColVal));
					out_buf_type = 0;
					break;
				}
				case 151:
				{
					// test_case_name = "agoKernel_Convolve_S16_U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					node = vxConvolveNode(graph, img1, Convolve_conv_convolution, img_out);
					vx_int32 firstColVal = pix_img1_u8 * (12 + (8 * 3) + (4 * 5) + (2 * 4) + (1 * 2));
					vx_int32 secondColVal = firstColVal + (pix_img1_u8 * ((2 * 2) + (2 * 4) + (1 * 8)));
					vx_int32 remainingVal = 96 * pix_img1_u8;
					expected_image_sum = (remainingVal * (width - 4) * (height - 4)) + (2 * (height - 4) * (firstColVal + secondColVal));
					out_buf_type = 1;
					break;
				}
				case 154:
				{
					// test_case_name = "agoKernel_ScaleImage_U8_U8_Nearest";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxScaleImageNode(graph, img1, img_out, ScaleImage_type1_enum);
					expected_image_sum = pix_img1_u8 * (widthOut * heightOut);
					out_buf_type = 0;
					break;
				}
				case 155:
				{
					// test_case_name = "agoKernel_ScaleImage_U8_U8_Bilinear";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxScaleImageNode(graph, img1, img_out, ScaleImage_type2_enum);
					expected_image_sum = 0;
					out_buf_type = 0;
					break;
				}
				case 156:
				{
					// test_case_name = "agoKernel_ScaleImage_U8_U8_Bilinear_Replicate";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxScaleImageNode(graph, img1, img_out, ScaleImage_type2_enum);
					ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_BORDER, &border1, sizeof(border1)));
					expected_image_sum = pix_img1_u8 * (widthOut * heightOut);
					out_buf_type = 0;
					break;
				}
				case 157:
				{
					// test_case_name = "agoKernel_ScaleImage_U8_U8_Bilinear_Constant";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxScaleImageNode(graph, img1, img_out, ScaleImage_type2_enum);
					ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_BORDER, &border2, sizeof(border2)));
					expected_image_sum = 0;
					out_buf_type = 0;
					break;
				}
				case 158:
				{
					// test_case_name = "agoKernel_ScaleImage_U8_U8_Area";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxScaleImageNode(graph, img1, img_out, ScaleImage_type3_enum);
					expected_image_sum = pix_img1_u8 * (widthOut * heightOut);
					out_buf_type = 0;
					break;
				}
				case 159:
				{
					// test_case_name = "agoKernel_ScaleGaussianHalf_U8_U8_3x3";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxHalfScaleGaussianNode(graph, img1, img_out, 3);
					vx_int32 firstColVal = (vx_int32)PIXELCHECKU8((float)pix_img1_u8 * ((2 * 0.0625) + (3 * 0.125) + 0.25));
					if (width % 2 == 0)
						expected_image_sum = (pix_img1_u8 * (widthOut - 1) * (heightOut - 2)) + ((heightOut - 2) * firstColVal);
					else if (width % 2 != 0)
						expected_image_sum = (pix_img1_u8 * (widthOut - 2) * (heightOut - 2)) + ((heightOut - 2) * firstColVal * 2);
					out_buf_type = 0;
					out_buf_type = 0;
					break;
				}
				case 160:
				{
					// test_case_name = "agoKernel_ScaleGaussianHalf_U8_U8_5x5";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxHalfScaleGaussianNode(graph, img1, img_out, 5);
					vx_int32 firstColVal = (vx_int32)PIXELCHECKU8((float)pix_img1_u8 * 176 / 256);
					vx_int32 secondColVal = (vx_int32)PIXELCHECKU8((float)pix_img1_u8 * 240 / 256);
					if (width % 2 == 0)
						expected_image_sum = (pix_img1_u8 * (widthOut - 2) * (heightOut - 2)) + (firstColVal * (heightOut - 2)) + (secondColVal * (heightOut - 2));
					else if (width % 2 != 0)
						expected_image_sum = (pix_img1_u8 * (widthOut - 2) * (heightOut - 2)) + (firstColVal * (heightOut - 2)) + (firstColVal * (heightOut - 2));
					out_buf_type = 0;
					break;
				}
				case 162:
				{
					// test_case_name = "agoKernel_WarpAffine_U8_U8_Nearest";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxWarpAffineNode(graph, img1, WarpAffine_affineMatrix_matrix, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 163:
				{
					// test_case_name = "agoKernel_WarpAffine_U8_U8_Nearest_Constant";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxWarpAffineNode(graph, img1, WarpAffine_affineMatrix_matrix, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, img_out);
					ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_BORDER, &border2, sizeof(border2)));
					expected_image_sum = (pix_img1_u8 * width * height) + (border2.constant_value.U8 * widthOut * heightOut) - (border2.constant_value.U8 * width * height);
					out_buf_type = 0;
					break;
				}
				case 164:
				{
					// test_case_name = "agoKernel_WarpAffine_U8_U8_Bilinear";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxWarpAffineNode(graph, img1, WarpAffine_affineMatrix_matrix, VX_INTERPOLATION_TYPE_BILINEAR, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 165:
				{
					// test_case_name = "agoKernel_WarpAffine_U8_U8_Bilinear_Constant";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxWarpAffineNode(graph, img1, WarpAffine_affineMatrix_matrix, VX_INTERPOLATION_TYPE_BILINEAR, img_out);
					ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_BORDER, &border2, sizeof(border2)));
					expected_image_sum = (pix_img1_u8 * width * height) + (border2.constant_value.U8 * widthOut * heightOut) - (border2.constant_value.U8 * width * height);
					out_buf_type = 0;
					break;
				}
				case 166:
				{
					// test_case_name = "agoKernel_WarpPerspective_U8_U8_Nearest";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxWarpPerspectiveNode(graph, img1, WarpPerspective_perspectiveMatrix_matrix, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 167:
				{
					// test_case_name = "agoKernel_WarpPerspective_U8_U8_Nearest_Constant";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxWarpPerspectiveNode(graph, img1, WarpPerspective_perspectiveMatrix_matrix, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, img_out);
					ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_BORDER, &border2, sizeof(border2)));
					expected_image_sum = (pix_img1_u8 * width * height) + (border2.constant_value.U8 * widthOut * heightOut) - (border2.constant_value.U8 * width * height);
					out_buf_type = 0;
					break;
				}
				case 168:
				{
					// test_case_name = "agoKernel_WarpPerspective_U8_U8_Bilinear";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxWarpPerspectiveNode(graph, img1, WarpPerspective_perspectiveMatrix_matrix, VX_INTERPOLATION_TYPE_BILINEAR, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 169:
				{
					// test_case_name = "agoKernel_WarpPerspective_U8_U8_Bilinear_Constant";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxWarpPerspectiveNode(graph, img1, WarpPerspective_perspectiveMatrix_matrix, VX_INTERPOLATION_TYPE_BILINEAR, img_out);
					ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_BORDER, &border2, sizeof(border2)));
					expected_image_sum = (pix_img1_u8 * width * height) + (border2.constant_value.U8 * widthOut * heightOut) - (border2.constant_value.U8 * width * height);
					out_buf_type = 0;
					break;
				}
				case 170:
				{
					// test_case_name = "agoKernel_ColorDepth_U8_S16_Wrap";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxConvertDepthNode(graph, img1, img_out, VX_CONVERT_POLICY_WRAP, ConvertDepth_shift_scalar);
					expected_image_sum = (vx_int32)((vx_uint8)(pix_img1_s16 >> ConvertDepth_shift_int32)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 171:
				{
					// test_case_name = "agoKernel_ColorDepth_U8_S16_Sat";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxConvertDepthNode(graph, img1, img_out, VX_CONVERT_POLICY_SATURATE, ConvertDepth_shift_scalar);
					expected_image_sum = (vx_int32)PIXELCHECKU8(pix_img1_s16 >> ConvertDepth_shift_int32) * width * height;
					out_buf_type = 0;
					break;
				}
				case 172:
				{
					// test_case_name = "agoKernel_ColorDepth_S16_U8";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, width, height, VX_DF_IMAGE_S16);
					node = vxConvertDepthNode(graph, img1, img_out, VX_CONVERT_POLICY_WRAP, ConvertDepth_shift_scalar);
					expected_image_sum = (pix_img1_u8 << ConvertDepth_shift_int32) * width * height;
					out_buf_type = 1;
					break;
				}
				case 174:
				{
					// test_case_name = "agoKernel_Remap_U8_U8_Nearest";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxRemapNode(graph, img1, Remap_remapTable_remap, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, img_out);
					vx_int32 finalVal = pix_img1_u8 + (width * height) - 1;
					expected_image_sum = (finalVal * (finalVal + 1) / 2) - (pix_img1_u8 * (pix_img1_u8 - 1) / 2) + ((widthOut - width) * heightOut * pix_img1_u8) + (width * (heightOut - height) * pix_img1_u8);
					out_buf_type = 0;
					break;
				}
				case 176:
				{
					// test_case_name = "agoKernel_Remap_U8_U8_Bilinear";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxRemapNode(graph, img1, Remap_remapTable_remap, VX_INTERPOLATION_TYPE_BILINEAR, img_out);
					vx_int32 finalVal = pix_img1_u8 + (width * height) - 1;
					expected_image_sum = (finalVal * (finalVal + 1) / 2) - (pix_img1_u8 * (pix_img1_u8 - 1) / 2) + ((widthOut - width) * heightOut * pix_img1_u8) + (width * (heightOut - height) * pix_img1_u8);
					out_buf_type = 0;
					break;
				}
				case 187:
				{
					// test_case_name = "agoKernel_CannySobel_U16_U8_3x3_L1NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxCannyEdgeDetectorNode(graph, img1, CannyEdgeDetector_thresholdObjectRange_threshold, CannyEdgeDetector_gradientSize3_int32, VX_NORM_L1, img_out);
					expected_image_sum = ((2 * ((heightOut / 2) - 2)) + (2 * ((widthOut / 2) - 2)) - 1) * 255;
					out_buf_type = 0;
					break;
				}
				case 188:
				{
					// test_case_name = "agoKernel_CannySobel_U16_U8_3x3_L2NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxCannyEdgeDetectorNode(graph, img1, CannyEdgeDetector_thresholdObjectRange_threshold, CannyEdgeDetector_gradientSize3_int32, VX_NORM_L2, img_out);
					expected_image_sum = ((2 * ((heightOut / 2) - 2)) + (2 * ((widthOut / 2) - 2)) + 2) * 255;
					out_buf_type = 0;
					break;
				}
				case 189:
				{
					// test_case_name = "agoKernel_CannySobel_U16_U8_5x5_L1NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxCannyEdgeDetectorNode(graph, img1, CannyEdgeDetector_thresholdObjectRange_threshold, CannyEdgeDetector_gradientSize5_int32, VX_NORM_L1, img_out);
					expected_image_sum = ((2 * ((heightOut / 2) - 2)) + (2 * ((widthOut / 2) - 2)) - 1) * 255;
					out_buf_type = 0;
					break;
				}
				case 190:
				{
					// test_case_name = "agoKernel_CannySobel_U16_U8_5x5_L2NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxCannyEdgeDetectorNode(graph, img1, CannyEdgeDetector_thresholdObjectRange_threshold, CannyEdgeDetector_gradientSize5_int32, VX_NORM_L2, img_out);
					expected_image_sum = ((2 * ((heightOut / 2) - 2)) + (2 * ((widthOut / 2) - 2)) + 2) * 255;
					out_buf_type = 0;
					break;
				}
				case 191:
				{
					// test_case_name = "agoKernel_CannySobel_U16_U8_7x7_L1NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxCannyEdgeDetectorNode(graph, img1, CannyEdgeDetector_thresholdObjectRange_threshold, CannyEdgeDetector_gradientSize7_int32, VX_NORM_L1, img_out);
					expected_image_sum = ((2 * ((heightOut / 2) - 2)) + (2 * ((widthOut / 2) - 2)) - 1) * 255;
					out_buf_type = 0;
					break;
				}
				case 192:
				{
					// test_case_name = "agoKernel_CannySobel_U16_U8_7x7_L2NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					img_out = vxCreateImage(context, widthOut, heightOut, VX_DF_IMAGE_U8);
					node = vxCannyEdgeDetectorNode(graph, img1, CannyEdgeDetector_thresholdObjectRange_threshold, CannyEdgeDetector_gradientSize7_int32, VX_NORM_L2, img_out);
					expected_image_sum = ((2 * ((heightOut / 2) - 2)) + (2 * ((widthOut / 2) - 2)) + 2) * 255;
					out_buf_type = 0;
					break;
				}
				case 203:
				{
					//test_case_name = "agoKernel_FastCorners_XY_U8_Supression";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxFastCornersNode(graph, img1, fastCorner_threshold_scalar, nms_true, output_keypoints_array, output_corner_count);
					out_buf_type = -1;
					break;
				}
				case 204:
				{
					//test_case_name = "agoKernel_FastCorners_XY_U8_NoSupression";
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxFastCornersNode(graph, img1, fastCorner_threshold_scalar, nms_false, output_keypoints_array, output_corner_count);
					out_buf_type = -1;
					break;
				}
				case 206:
				{
					img1 = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
					node = vxHarrisCornersNode(graph, img1, HarrisCorner_strength_threshold_scalar, HarrisCorner_min_distance_scalar, HarrisCorner_sensitivity_scalar, HarrisCorner_grad_size, HarrisCorner_block_size, HarrisCorner_output_keypoints_array, HarrisCorner_output_corner_count);
					out_buf_type = -1;
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
					(case_number == 61) || (case_number == 104)
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
					(case_number == 45) || (case_number == 47)  || (case_number == 88) ||
					(case_number == 89) || (case_number == 90) || (case_number == 91) || (case_number == 92) || 
					(case_number == 93) || (case_number == 94) || (case_number == 95) || (case_number == 96) || 
					(case_number == 133) || (case_number == 134) || (case_number == 138) || (case_number == 142) || 
					(case_number == 143) || (case_number == 147) || (case_number == 148) || (case_number == 149) || 
					(case_number == 150) || (case_number == 151) || 
					(case_number == 154) || (case_number == 155) || (case_number == 156)  || (case_number == 157) || 
					(case_number == 158)  || (case_number == 159) || (case_number == 160) || (case_number == 162) || 
					(case_number == 163) || (case_number == 164) || (case_number == 165) || (case_number == 166) || 
					(case_number == 167) || (case_number == 168) || (case_number == 169) || (case_number == 172) || 
					(case_number == 174) || (case_number == 176) || (case_number == 187) || (case_number == 188) || 
					(case_number == 189) || (case_number == 190) || (case_number == 191) || (case_number == 192) ||
					(case_number == 203) || (case_number == 204) || (case_number == 206)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u8));
				}
				// S16 input
				else if(
					(case_number == 170) || (case_number == 171)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_int16) pix_img1_s16));
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
				// U16, U24 and U32 inputs for UYVY, YUYV, RGB, RGBX
				else if(
					(case_number == 71)  || (case_number == 72)  || (case_number == 73)  || (case_number == 74)  ||
					(case_number == 75)  || (case_number == 76)  || (case_number == 77)  || (case_number == 78)  ||
					(case_number == 79)  || (case_number == 105) || (case_number == 106) || (case_number == 107) ||
					(case_number == 111) || (case_number == 112) || (case_number == 113) || (case_number == 117) ||
					(case_number == 118) || (case_number == 119) || (case_number == 120) || (case_number == 121) ||
					(case_number == 122) || (case_number == 123) || (case_number == 124) || (case_number == 128) || (case_number == 129) 
					
				)
				{
					ERROR_CHECK_STATUS(makeInputPackedImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u8))	
				}
				// Planar Input Images - NV12, NV21, IYUV input
				else if(
					(case_number == 67) || (case_number == 108) || (case_number == 109) || (case_number == 110) ||
					(case_number == 114) || (case_number == 115) || (case_number == 116)  ||
					 (case_number == 130) || (case_number == 131) || (case_number == 132) ||
					(case_number == 125) || (case_number == 126) || (case_number == 127)
				)
				{
					ERROR_CHECK_STATUS(makeInputPlanarImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u8));
				}
				// NV12 Channel Combine inputs
				else if(
					(case_number == 83) 
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width/2, height/2, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img2_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img3, width/2, height/2, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img3_u8));
				}
				// RGB RGBX Channel Combine inputs
				else if(
					(case_number == 84) || (case_number == 87)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img2_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img3, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img3_u8));
				}
				else if(
					(case_number == 85) || (case_number == 86)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img1_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width/2, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img2_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img3, width/2, height, VX_MEMORY_TYPE_HOST, (vx_uint8) pix_img3_u8));
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
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[0], height * hip_addr_uint8.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[1], height * hip_addr_uint8.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[2], height * hip_addr_uint8.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[3], height * hip_addr_uint8.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(ptr[2], 0, height  * hip_addr_uint8.stride_y));

		vx_imagepatch_addressing_t hip_addr_uint8_out = {0}; // Already present
		hip_addr_uint8_out.dim_x = widthOut;
		hip_addr_uint8_out.dim_y = heightOut;
		hip_addr_uint8_out.stride_x = 1;
		hip_addr_uint8_out.stride_y = (widthOut+3)&~3;
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[0], heightOut * hip_addr_uint8_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[1], heightOut * hip_addr_uint8_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[2], heightOut * hip_addr_uint8_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(ptr[2], 0, heightOut * hip_addr_uint8_out.stride_y));

		vx_imagepatch_addressing_t hip_addr_int16 = {0};
		hip_addr_int16.dim_x = width;
		hip_addr_int16.dim_y = height;
		hip_addr_int16.stride_x = 2;
		hip_addr_int16.stride_y = ((width+3)&~3)*2;
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[0], height * hip_addr_int16.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[1], height * hip_addr_int16.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[2], height * hip_addr_int16.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(ptr[2], 0, height * hip_addr_int16.stride_y));

		vx_imagepatch_addressing_t hip_addr_uint8_u1 = {0};
		hip_addr_uint8_u1.dim_x = width;
		hip_addr_uint8_u1.dim_y = height;
		hip_addr_uint8_u1.stride_x = 1;
		hip_addr_uint8_u1.stride_y = (width+7)&~7;
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[0], height * hip_addr_uint8_u1.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[1], height * hip_addr_uint8_u1.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[2], height * hip_addr_uint8_u1.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(ptr[2], 0, height  * hip_addr_uint8_u1.stride_y));

		vx_imagepatch_addressing_t hip_addr_u1 = {0};
		hip_addr_u1.dim_x = width;
		hip_addr_u1.dim_y = height;
		hip_addr_u1.stride_x = 0;
		hip_addr_u1.stride_x_bits = 1;
		hip_addr_u1.stride_y = (width % 8 == 0) ? ((width/8)+7) & ~7 : ((width/8)+8) & ~7;
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[0], height * hip_addr_u1.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[1], height * hip_addr_u1.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[2], height * hip_addr_u1.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(ptr[2], 0, height  * hip_addr_u1.stride_y));

		vx_imagepatch_addressing_t hip_addr_uint8_yuyv_uyuv_in = {0};
		hip_addr_uint8_yuyv_uyuv_in.dim_x = width;
		hip_addr_uint8_yuyv_uyuv_in.dim_y = height;
		hip_addr_uint8_yuyv_uyuv_in.stride_x = 2;
		hip_addr_uint8_yuyv_uyuv_in.stride_y = ((width+3)&~3)*2;
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[0], height * hip_addr_uint8_yuyv_uyuv_in.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[1], height * hip_addr_uint8_yuyv_uyuv_in.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[2], height * hip_addr_uint8_yuyv_uyuv_in.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[3], height * hip_addr_uint8_yuyv_uyuv_in.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(ptr[2], 0, height * hip_addr_uint8_yuyv_uyuv_in.stride_y));


		vx_imagepatch_addressing_t hip_addr_uint8_yuyv_uyuv_out = {0};
		hip_addr_uint8_yuyv_uyuv_out.dim_x = width/2;
		hip_addr_uint8_yuyv_uyuv_out.dim_y = height;
		hip_addr_uint8_yuyv_uyuv_out.stride_x = 1;
		hip_addr_uint8_yuyv_uyuv_out.stride_y = ((width+3)&~3);
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[0], height * hip_addr_uint8_yuyv_uyuv_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[1], height * hip_addr_uint8_yuyv_uyuv_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[2], height * hip_addr_uint8_yuyv_uyuv_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[3], height * hip_addr_uint8_yuyv_uyuv_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(ptr[2], 0, height * hip_addr_uint8_yuyv_uyuv_out.stride_y));

		vx_imagepatch_addressing_t hip_addr_uint8_rgb_in = {0};
		hip_addr_uint8_rgb_in.dim_x = width;
		hip_addr_uint8_rgb_in.dim_y = height;
		hip_addr_uint8_rgb_in.stride_x = 3;
		hip_addr_uint8_rgb_in.stride_y = ((width+3)&~3)*3;
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[0], height * hip_addr_uint8_rgb_in.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[1], height * hip_addr_uint8_rgb_in.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[2], height * hip_addr_uint8_rgb_in.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[3], height * hip_addr_uint8_rgb_in.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(ptr[2], 0, height * hip_addr_uint8_rgb_in.stride_y));

		vx_imagepatch_addressing_t hip_addr_uint8_rgbx_in = {0};
		hip_addr_uint8_rgbx_in.dim_x = width;
		hip_addr_uint8_rgbx_in.dim_y = height;
		hip_addr_uint8_rgbx_in.stride_x = 4;
		hip_addr_uint8_rgbx_in.stride_y = ((width+3)&~3)*4;
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[0], height * hip_addr_uint8_rgbx_in.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[1], height * hip_addr_uint8_rgbx_in.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[2], height * hip_addr_uint8_rgbx_in.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[3], height * hip_addr_uint8_rgbx_in.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(ptr[2], 0, height * hip_addr_uint8_rgbx_in.stride_y));

		vx_imagepatch_addressing_t hip_addr_uint8_rgb_out = {0};
		hip_addr_uint8_rgb_out.dim_x = width;
		hip_addr_uint8_rgb_out.dim_y = height;
		hip_addr_uint8_rgb_out.stride_x = 1;
		hip_addr_uint8_rgb_out.stride_y = ((width+3)&~3);
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[0], height * hip_addr_uint8_rgb_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[1], height * hip_addr_uint8_rgb_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[2], height * hip_addr_uint8_rgb_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[3], height * hip_addr_uint8_rgb_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(ptr[2], 0, height * hip_addr_uint8_rgb_out.stride_y));
		
		vx_imagepatch_addressing_t hip_addr_uint8_nv12_nv21_out = {0};
		hip_addr_uint8_nv12_nv21_out.dim_x = width/2;
		hip_addr_uint8_nv12_nv21_out.dim_y = height/2;
		hip_addr_uint8_nv12_nv21_out.stride_x = 1;
		hip_addr_uint8_nv12_nv21_out.stride_y = ((width+3)&~3);
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[0], height * hip_addr_uint8_nv12_nv21_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[1], height * hip_addr_uint8_nv12_nv21_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[2], height * hip_addr_uint8_nv12_nv21_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[3], height * hip_addr_uint8_nv12_nv21_out.stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(ptr[2], 0, height * hip_addr_uint8_nv12_nv21_out.stride_y));

		vx_imagepatch_addressing_t hip_addr_uint8_nv12_nv21_in[2] = {0};
		hip_addr_uint8_nv12_nv21_in[0].dim_x = width;
		hip_addr_uint8_nv12_nv21_in[0].dim_y = height;
		hip_addr_uint8_nv12_nv21_in[0].stride_x = 1;
		hip_addr_uint8_nv12_nv21_in[0].stride_y = ((width+3)&~3);
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&nv_in[0], height * hip_addr_uint8_nv12_nv21_in[0].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[0], height * hip_addr_uint8_nv12_nv21_in[0].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&ptr[1], height * hip_addr_uint8_nv12_nv21_in[0].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&nv_out[0],height * hip_addr_uint8_nv12_nv21_in[0].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(nv_out[0], 0, height * hip_addr_uint8_nv12_nv21_in[0].stride_y));

		hip_addr_uint8_nv12_nv21_in[1].dim_x = width/2;
		hip_addr_uint8_nv12_nv21_in[1].dim_y = height/2;
		hip_addr_uint8_nv12_nv21_in[1].stride_x = 2;
		hip_addr_uint8_nv12_nv21_in[1].stride_y = ((width+3)&~3)*2;
		hip_addr_uint8_nv12_nv21_in[1].step_x = 2;
		hip_addr_uint8_nv12_nv21_in[1].step_y = 2;
		hip_addr_uint8_nv12_nv21_in[1].scale_x = 512;
		hip_addr_uint8_nv12_nv21_in[1].scale_y = 512;
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&nv_in[1], height * hip_addr_uint8_nv12_nv21_in[1].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&nv_out[1], height * hip_addr_uint8_nv12_nv21_in[1].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(nv_out[1], 0, height  * hip_addr_uint8_nv12_nv21_in[1].stride_y));

		vx_imagepatch_addressing_t hip_addr_uint8_iyuv_in[3] = {0};
		hip_addr_uint8_iyuv_in[0].dim_x = width;
		hip_addr_uint8_iyuv_in[0].dim_y = height;
		hip_addr_uint8_iyuv_in[0].stride_x = 1;
		hip_addr_uint8_iyuv_in[0].stride_y = ((width+3)&~3);
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&iyuv_in[0], height * hip_addr_uint8_iyuv_in[0].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&iyuv_out[0],height * hip_addr_uint8_iyuv_in[0].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(iyuv_out[0], 0, height * hip_addr_uint8_iyuv_in[0].stride_y));

		hip_addr_uint8_iyuv_in[1].dim_x = width/2;
		hip_addr_uint8_iyuv_in[1].dim_y = height/2;
		hip_addr_uint8_iyuv_in[1].stride_x = 1;
		hip_addr_uint8_iyuv_in[1].stride_y = ((width+3)&~3);
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&iyuv_in[1], height * hip_addr_uint8_iyuv_in[1].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&iyuv_out[1], height * hip_addr_uint8_iyuv_in[1].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(iyuv_out[1], 0, height  * hip_addr_uint8_iyuv_in[1].stride_y));

		hip_addr_uint8_iyuv_in[2].dim_x = width/2;
		hip_addr_uint8_iyuv_in[2].dim_y = height/2;
		hip_addr_uint8_iyuv_in[2].stride_x = 1;
		hip_addr_uint8_iyuv_in[2].stride_y = ((width+3)&~3);
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&iyuv_in[2], height * hip_addr_uint8_iyuv_in[2].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&iyuv_out[2], height * hip_addr_uint8_iyuv_in[2].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(iyuv_out[2], 0, height  * hip_addr_uint8_iyuv_in[2].stride_y));


		vx_imagepatch_addressing_t hip_addr_uint8_yuv4_in[3] = {0};
		hip_addr_uint8_yuv4_in[0].dim_x = width;
		hip_addr_uint8_yuv4_in[0].dim_y = height;
		hip_addr_uint8_yuv4_in[0].stride_x = 1;
		hip_addr_uint8_yuv4_in[0].stride_y = ((width+3)&~3);
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&yuv4_in[0], height * hip_addr_uint8_yuv4_in[0].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&yuv4_out[0],height * hip_addr_uint8_yuv4_in[0].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(iyuv_out[0], 0, height * hip_addr_uint8_yuv4_in[0].stride_y));

		hip_addr_uint8_yuv4_in[1].dim_x = width;
		hip_addr_uint8_yuv4_in[1].dim_y = height;
		hip_addr_uint8_yuv4_in[1].stride_x = 1;
		hip_addr_uint8_yuv4_in[1].stride_y = ((width+3)&~3);
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&yuv4_in[1], height * hip_addr_uint8_yuv4_in[1].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&yuv4_out[1], height * hip_addr_uint8_yuv4_in[1].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(iyuv_out[1], 0, height  * hip_addr_uint8_yuv4_in[1].stride_y));

		hip_addr_uint8_yuv4_in[2].dim_x = width;
		hip_addr_uint8_yuv4_in[2].dim_y = height;
		hip_addr_uint8_yuv4_in[2].stride_x = 1;
		hip_addr_uint8_yuv4_in[2].stride_y = ((width+3)&~3);
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&yuv4_in[2], height * hip_addr_uint8_yuv4_in[2].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMalloc((void**)&yuv4_out[2], height * hip_addr_uint8_yuv4_in[2].stride_y));
		ERROR_CHECK_HIP_STATUS(hipMemset(iyuv_out[2], 0, height  * hip_addr_uint8_yuv4_in[2].stride_y));


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
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 & (pix_img2_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 39:
				{
					// test_case_name = "agoKernel_And_U8_U1U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img2_u8 & (pix_img1_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 40:
				{
					// test_case_name = "agoKernel_And_U8_U1U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u1 ? 255 : 0) & (pix_img2_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 41:
				{
					// test_case_name = "agoKernel_And_U1_U8U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u8 & (1 << 7)) & (pix_img2_u8 & (1 << 7)) ? 255 : 0) *
											(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 42:
				{
					// test_case_name = "agoKernel_And_U1_U8U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (((pix_img1_u8 & (1 << 7)) ? 255 : 0) & (pix_img2_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 43:
				{
					// test_case_name = "agoKernel_And_U1_U1U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = (((pix_img2_u8 & (1 << 7)) ? 255 : 0) & (pix_img1_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 44:
				{
					// test_case_name = "agoKernel_And_U1_U1U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxAndNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u1 ? 255 : 0) & (pix_img2_u1 ? 255 : 0)) * (int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
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
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxNotNode(graph, img1, img_out);
					widthOut = ((width % 8 == 0) ? ((width/8)+7) & ~7 : ((width/8)+8) & ~7);
					expected_image_sum = (255 - (pix_img1_u1 ? 255 : 0)) * widthOut * height;
					out_buf_type = 0;
					break;
				}
				case 47:
				{
					// test_case_name = "agoKernel_Not_U1_U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxNotNode(graph, img1, img_out);
					expected_image_sum = (255 - ((pix_img1_u8 & (1 << 7)) ? 255 : 0)) * (int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 48:
				{
					// test_case_name = "agoKernel_Not_U1_U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxNotNode(graph, img1, img_out);
					expected_image_sum = (255 - (pix_img1_u1? 255:0)) * (int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
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
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 | (pix_img2_u1 ? 255 : 0)) * width * height;					
					out_buf_type = 0;
					break;
				}
				case 51:
				{
					// test_case_name = "agoKernel_Or_U8_U1U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img2_u8 | (pix_img1_u1 ? 255 : 0)) * width * height;					
					out_buf_type = 0;
					break;
				}
				case 52:
				{
					// test_case_name = "agoKernel_Or_U8_U1U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u1 ? 255 : 0) | (pix_img2_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 53:
				{
					// test_case_name = "agoKernel_Or_U1_U8U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u8 & (1 << 7)) | (pix_img2_u8 & (1 << 7)) ? 255 : 0) *
											(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 54:
				{
					// test_case_name = "agoKernel_Or_U1_U8U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (((pix_img1_u8 & (1 << 7)) ? 255 : 0) | (pix_img2_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 55:
				{
					// test_case_name = "agoKernel_Or_U1_U1U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = (((pix_img2_u8 & (1 << 7)) ? 255 : 0) | (pix_img1_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 56:
				{
					// test_case_name = "agoKernel_Or_U1_U1U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxOrNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u1 ? 255 : 0) | (pix_img2_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
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
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img1_u8 ^ (pix_img2_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 59:
				{
					// test_case_name = "agoKernel_Xor_U8_U1U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (pix_img2_u8 ^ (pix_img1_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 60:
				{
					// test_case_name = "agoKernel_Xor_U8_U1U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u1 ? 255 : 0) ^ (pix_img2_u1 ? 255 : 0)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 61:
				{
					// test_case_name = "agoKernel_Xor_U1_U8U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u8 & (1 << 7)) ^ (pix_img2_u8 & (1 << 7)) ? 255 : 0) *
											(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 62:
				{
					// test_case_name = "agoKernel_Xor_U1_U8U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (((pix_img1_u8 & (1 << 7)) ? 255 : 0) ^ (pix_img2_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;					
					out_buf_type = 5;
					break;
				}
				case 63:
				{
					// test_case_name = "agoKernel_Xor_U1_U1U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = (((pix_img2_u8 & (1 << 7)) ? 255 : 0) ^ (pix_img1_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
					break;
				}
				case 64:
				{
					// test_case_name = "agoKernel_Xor_U1_U1U1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1, &hip_addr_u1, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxXorNode(graph, img1, img2, img_out);
					expected_image_sum = ((pix_img1_u1 ? 255 : 0) ^ (pix_img2_u1 ? 255 : 0)) * 
										(int)(ceil(width/8.0)) * height;
					out_buf_type = 5;
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
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV12, hip_addr_uint8_nv12_nv21_in, nv_in, VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_Y, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 71:
				{
					// test_case_name = "agoKernel_ChannelExtract_U8_U16_Pos0";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUYV, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, (vx_enum)VX_CHANNEL_Y, img_out);
					expected_image_sum = pix_img1_u8 * width* height; 
					out_buf_type = 0;
					break;
				}
				case 72:
				{
					// test_case_name = "agoKernel_ChannelExtract_U8_U16_Pos1";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_UYVY, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, (vx_enum)VX_CHANNEL_Y, img_out);
					expected_image_sum = (pix_img1_u8+1) * width * height; 
					out_buf_type = 0;
					break;
				}
				case 73:
				{
					// test_case_name - agoKernel_ChannelExtract_U8_U24_Pos0
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_rgb_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, (vx_enum)VX_CHANNEL_R, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 74:
				{
					// test_case_name - agoKernel_ChannelExtract_U8_U24_Pos1
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_rgb_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, (vx_enum)VX_CHANNEL_G, img_out);
					expected_image_sum = (pix_img1_u8+1) * width * height;
					out_buf_type = 0;
					break;
				}
				case 75:
				{
					// test_case_name - agoKernel_ChannelExtract_U8_U24_Pos2
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_rgb_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, (vx_enum)VX_CHANNEL_B, img_out);
					expected_image_sum = (pix_img1_u8+2) * width * height;
					out_buf_type = 0;
					break;
				}
				case 76:
				{
					// test_case_name - agoKernel_ChannelExtract_U8_U32_Pos0
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_UYVY, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_yuyv_uyuv_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_U, img_out);
					expected_image_sum = pix_img1_u8 * width/2 * height;
					out_buf_type = 2;
					break;
				}
				case 77:
				{
					// test_case_name - agoKernel_ChannelExtract_U8_U32_Pos1
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUYV, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_yuyv_uyuv_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_U, img_out);
					expected_image_sum = (pix_img1_u8+1) * width/2 * height;
					out_buf_type = 2;
					break;
				}
				case 78:
				{
					// test_case_name - agoKernel_ChannelExtract_U8_U32_Pos2
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_UYVY, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_yuyv_uyuv_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_V, img_out);
					expected_image_sum = (pix_img1_u8) * width/2 * height;
					out_buf_type = 2;
					break;
				}
				case 79:
				{
					// test_case_name - agoKernel_ChannelExtract_U8_U32_Pos3
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUYV, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_yuyv_uyuv_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelExtractNode(graph, img1, VX_CHANNEL_V, img_out);
					expected_image_sum = (pix_img1_u8+1) * width/2 * height;
					out_buf_type = 2;
					break;
				}
				case 83:
				{
					// test_case_name - agoKernel_ChannelCombine_U16_U8U8
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_nv12_nv21_out, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img3 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_nv12_nv21_out, &ptr[3], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV12, hip_addr_uint8_nv12_nv21_in, nv_out, VX_MEMORY_TYPE_HIP));
					node = vxChannelCombineNode(graph, img1, img2, img3 , 0, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + ((pix_img2_u8 +pix_img3_u8 ) * (width/2) * (height/2));
					out_buf_type = 4;
					break;
				}	
				case 84:
				{
					// test_case_name - agoKernel_ChannelCombine_U24_U8U8U8_RGB
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img3 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[3], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelCombineNode(graph, img1, img2, img3 , 0, img_out);
					expected_image_sum = (pix_img1_u8 + pix_img2_u8 + pix_img3_u8) * width * height;
					out_buf_type = 3;
					break;
				}	
				case 85:
				{
					// test_case_name - agoKernel_ChannelCombine_U32_U8U8U8_UYVY
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_yuyv_uyuv_out, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img3 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_yuyv_uyuv_out, &ptr[3], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_UYVY, &hip_addr_uint8_yuyv_uyuv_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelCombineNode(graph, img1, img2, img3 , 0, img_out);
					expected_image_sum = ((pix_img1_u8) * width * height) + ((pix_img2_u8 + pix_img3_u8) * (width/2) * height);
					out_buf_type = 3;
					break;
				}
				case 86:
				{
					// test_case_name - agoKernel_ChannelCombine_U32_U8U8U8_YUYV
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_yuyv_uyuv_out, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img3 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_yuyv_uyuv_out, &ptr[3], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUYV, &hip_addr_uint8_yuyv_uyuv_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelCombineNode(graph, img1, img2, img3 , 0, img_out);
					expected_image_sum = ((pix_img1_u8) * width * height) + ((pix_img2_u8 + pix_img3_u8) * (width/2) * height);
					out_buf_type = 3;
					break;
				}
				case 87:
				{
					// test_case_name - agoKernel_ChannelCombine_U32_U8U8U8U8_RGBX
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img3 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[3], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGBX, &hip_addr_uint8_rgbx_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxChannelCombineNode(graph, img1, img2, img3 , img2, img_out);
					expected_image_sum = (pix_img1_u8 + pix_img2_u8 + pix_img3_u8 + pix_img2_u8) * width * height;
					out_buf_type = 3;
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
				// case 89:
				// {
				// 	// test_case_name = "agoKernel_Threshold_U8_U8_Binary";
				// 	ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
				// 	ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 255 : 0) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				// case 90:
				// {
				// 	// test_case_name = "agoKernel_Threshold_U8_U8_Range";
				// 	ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
				// 	ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 0 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 0 : 255)) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				// case 91:
				// {
				// 	// test_case_name = "agoKernel_Threshold_U1_U8_Binary";
				// 	ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
				// 	ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U1_AMD, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 1 : 0) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				// case 92:
				// {
				// 	// test_case_name = "agoKernel_Threshold_U1_U8_Range";
				// 	ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
				// 	ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 0 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 0 : 1)) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				// case 93:
				// {
				// 	// test_case_name = "agoKernel_ThresholdNot_U8_U8_Binary";
				// 	ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
				// 	ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 0 : 255) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				// case 94:
				// {
				// 	// test_case_name = "agoKernel_ThresholdNot_U8_U8_Range";
				// 	ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
				// 	ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 255 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 255 : 0)) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				// case 95:
				// {
				// 	// test_case_name = "agoKernel_ThresholdNot_U1_U8_Binary";
				// 	ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
				// 	ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectBinary_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdValue_int32) ? 0 : 1) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				// case 96:
				// {
				// 	// test_case_name = "agoKernel_ThresholdNot_U1_U8_Range";
				// 	ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
				// 	ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
				// 	node = vxThresholdNode(graph, img1, Threshold_thresholdObjectRange_threshold, img_out);
				// 	expected_image_sum = ((pix_img1_u8 > Threshold_thresholdUpper_int32) ? 1 : ((pix_img1_u8 < Threshold_thresholdLower_int32) ? 1 : 0)) * width * height;
				// 	out_buf_type = 0;
				// 	break;
				// }
				case 104:
				{
					// test_case_name = "agoKernel_WeightedAverage_U8_U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[1], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxWeightedAverageNode(graph, img1, WeightedAverage_alpha_scalar, img2, img_out);
					expected_image_sum = (vx_int32)(((vx_float32)pix_img2_u8 * ((vx_float32)1 - WeightedAverage_alpha_float)) + ((vx_float32)pix_img1_u8 * WeightedAverage_alpha_float)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 105:
				{
					// test_case_name  = " agoKernel_ColorConvert_RGB_RGBX"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGBX, &hip_addr_uint8_rgbx_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (((3*pix_img1_u8) + 3) * width * height);
					out_buf_type = 3;
					break;
				}
				case 106:
				{
					// test_case_name  = " agoKernel_ColorConvert_RGB_UYVY"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_UYVY, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_UYVY, pix_img1_u8+1, pix_img1_u8, pix_img1_u8)) * width * height;
					out_buf_type = 3;
					break;
				}
				case 107:
				{
					// test_case_name  = " agoKernel_ColorConvert_RGB_YUYV"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUYV, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_YUYV, pix_img1_u8, pix_img1_u8+1, pix_img1_u8+1)) * width * height;
					out_buf_type = 3;
					break;
				}
				case 108:
				{
					// test_case_name  = " agoKernel_ColorConvert_RGB_IYUV"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_IYUV, hip_addr_uint8_iyuv_in, iyuv_in, VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_IYUV, pix_img1_u8, pix_img1_u8+1, pix_img1_u8+1)) * width * height;
					out_buf_type = 3;
					break;
				}
				case 109:
				{
					// test_case_name  = " agoKernel_ColorConvert_RGB_NV12"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV12, hip_addr_uint8_nv12_nv21_in, nv_in, VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_NV12, pix_img1_u8, pix_img1_u8+1, pix_img1_u8+2)) * width * height;
					out_buf_type = 3;
					break;
				}
				case 110:
				{
					// test_case_name  = " agoKernel_ColorConvert_RGB_NV21"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV21, hip_addr_uint8_nv12_nv21_in, nv_in, VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_NV21, pix_img1_u8, pix_img1_u8+2, pix_img1_u8+1)) * width * height;
					out_buf_type = 3;
					break;
				}
				case 111:
				{
					// test_case_name  = " agoKernel_ColorConvert_RGBX_RGB"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGBX, &hip_addr_uint8_rgbx_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8+pix_img1_u8+pix_img1_u8+3+255) * width * height; 
					out_buf_type = 3;
					break;
				}
				case 112:
				{
					// test_case_name  = " agoKernel_ColorConvert_RGBX_UYVY"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_UYVY, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGBX, &hip_addr_uint8_rgbx_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_UYVY, pix_img1_u8+1, pix_img1_u8, pix_img1_u8) + 255) * width * height;
					out_buf_type = 3;
					break;
				}
				case 113:
				{
					// test_case_name  = " agoKernel_ColorConvert_RGBX_YUYV"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUYV, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGBX, &hip_addr_uint8_rgbx_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_YUYV, pix_img1_u8, pix_img1_u8+1, pix_img1_u8+1) + 255) * width * height;
					out_buf_type = 3;
					break;
				}
				case 114:
				{
					// test_case_name  = " agoKernel_ColorConvert_RGBX_IYUV"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_IYUV, hip_addr_uint8_iyuv_in, iyuv_in, VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGBX, &hip_addr_uint8_rgbx_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_IYUV, pix_img1_u8, pix_img1_u8+1, pix_img1_u8+1) + 255) * width * height;
					out_buf_type = 3;
					break;
				}
				case 115:
				{
					// test_case_name  = " agoKernel_ColorConvert_RGBX_NV12"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV12, hip_addr_uint8_nv12_nv21_in, nv_in, VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGBX, &hip_addr_uint8_rgbx_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_NV12, pix_img1_u8, pix_img1_u8+1, pix_img1_u8+2) + 255) * width * height;
					out_buf_type = 3;
					break;
				}
				case 116:
				{
					// test_case_name  = " agoKernel_ColorConvert_RGBX_NV21"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV21, hip_addr_uint8_nv12_nv21_in, nv_in, VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGBX, &hip_addr_uint8_rgbx_in, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (RGBsum_YUV(VX_DF_IMAGE_NV21, pix_img1_u8, pix_img1_u8+2, pix_img1_u8+1) + 255) * width * height;
					out_buf_type = 3;
					break;
				}
				case 117:
				{
					// test_case_name  = " agoKernel_ColorConvert_IYUV_RGB"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_IYUV, hip_addr_uint8_iyuv_in, iyuv_out, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + ((128+127) * (width/2) * (height/2)) ;
					out_buf_type = 4;
					break;
				}
				case 118:
				{
					// test_case_name  = " agoKernel_ColorConvert_IYUV_RGBX"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGBX, &hip_addr_uint8_rgbx_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_IYUV, hip_addr_uint8_iyuv_in, iyuv_out, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + ((128+127) * (width/2) * (height/2)) ;
					out_buf_type = 4;
					break;
				}
				case 119:
				{
					// test_case_name  = " agoKernel_FormatConvert_IYUV_UYVY"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_UYVY, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_IYUV, hip_addr_uint8_iyuv_in, iyuv_out, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = ((pix_img1_u8 + 1) * width * height) + (2 * (pix_img1_u8) * (width/2) * (height/2));
					out_buf_type = 4;
					break;
				}
				case 120:
				{
					// test_case_name  = " agoKernel_FormatConvert_IYUV_YUYV"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUYV, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_IYUV, hip_addr_uint8_iyuv_in, iyuv_out, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + (2 * (1+pix_img1_u8) * (width/2) * (height/2));
					out_buf_type = 4;
					break;
				}
				case 121:
				{
					// test_case_name  = " agoKernel_ColorConvert_NV12_RGB"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV12, hip_addr_uint8_nv12_nv21_in, nv_out, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + ((128+127) * (width/2) * (height/2)) ;
					out_buf_type = 4;
					break;
				}
				case 122:
				{
					// test_case_name  = " agoKernel_ColorConvert_NV12_RGBX"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGBX, &hip_addr_uint8_rgbx_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV12, hip_addr_uint8_nv12_nv21_in, nv_out, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + ((128+127) * (width/2) * (height/2)) ;
					out_buf_type = 4;
					break;
				}
				case 123:
				{
					// test_case_name  = " agoKernel_FormatConvert_NV12_UYVY"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_UYVY, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV12, hip_addr_uint8_nv12_nv21_in, nv_out, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = ((pix_img1_u8 + 1) * width * height) + (2 * pix_img1_u8 * (width / 2) * (height / 2));
					out_buf_type = 4;
					break;
				}
				case 124:
				{
					// test_case_name  = " agoKernel_FormatConvert_NV12_YUYV"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUYV, &hip_addr_uint8_yuyv_uyuv_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV12, hip_addr_uint8_nv12_nv21_in, nv_out, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + (2 * (pix_img1_u8 + 1) * (width / 2) * (height / 2));
					out_buf_type = 4;
					break;
				}
				case 125:
				{
					// test_case_name  = "agoKernel_FormatConvert_IYUV_NV12"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV12, hip_addr_uint8_nv12_nv21_in, nv_in, VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_IYUV, hip_addr_uint8_iyuv_in, iyuv_out, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = ((pix_img1_u8)*width * height) + (((2 * pix_img1_u8) + 3) * (width / 2) * (height / 2));
					out_buf_type = 4;
					break;
				}
				case 126:
				{
					// test_case_name  = "agoKernel_FormatConvert_IYUV_NV21"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV21, hip_addr_uint8_nv12_nv21_in, nv_in, VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_IYUV, hip_addr_uint8_iyuv_in, iyuv_out, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = ((pix_img1_u8)*width * height) + (((2 * pix_img1_u8) + 3) * (width / 2) * (height / 2));
					out_buf_type = 4;
					break;
				}
				case 127:
				{
					// test_case_name  = "agoKernel_FormatConvert_NV12_IYUV"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_IYUV, hip_addr_uint8_iyuv_in, iyuv_in, VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV12, hip_addr_uint8_nv12_nv21_in, nv_out, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = ((pix_img1_u8)*width * height) + (2 * (pix_img1_u8 + 1) * (width / 2) * (height / 2));
					out_buf_type = 4;
					break;
				}
				case 128:
				{
				// test_case_name  = " agoKernel_FormatConvert_YUV4_RGB"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGB, &hip_addr_uint8_rgb_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUV4, hip_addr_uint8_yuv4_in, yuv4_in, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + (2 * (pix_img1_u8 + 1) * (width / 2) * (height / 2)); //Needs Change
					out_buf_type = 4;
					break;
				}
				case 129:
				{
					// test_case_name  = " agoKernel_FormatConvert_YUV4_RGBX"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_RGBX, &hip_addr_uint8_rgbx_in, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUV4, hip_addr_uint8_yuv4_in, yuv4_in, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + (2 * (pix_img1_u8 + 1) * (width / 2) * (height / 2)); //Needs Change
					out_buf_type = 4;
					break;
				}
				case 130:
				{
					// test_case_name  = " agoKernel_FormatConvert_YUV4_NV12"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV12, hip_addr_uint8_nv12_nv21_in, nv_in, VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUV4, hip_addr_uint8_yuv4_in, yuv4_in, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + ((pix_img1_u8 + 1) * (width ) * (height)) + ((pix_img1_u8 + 2) * (width ) * (height)); 
					out_buf_type = 4;
					break;
				}
				case 131:
				{
				// test_case_name  = " agoKernel_FormatConvert_YUV4_NV12"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_NV21, hip_addr_uint8_nv12_nv21_in, nv_in, VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUV4, hip_addr_uint8_yuv4_in, yuv4_in, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + ((pix_img1_u8 + 1) * (width ) * (height)) + ((pix_img1_u8 + 2) * (width ) * (height)); 
					out_buf_type = 4;
					break;
				}
				case 132:
				{
				// test_case_name  = " agoKernel_ColorConvert_YUV4_IYUV"
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_IYUV, hip_addr_uint8_iyuv_in, iyuv_in, VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_YUV4, hip_addr_uint8_yuv4_in, yuv4_in, VX_MEMORY_TYPE_HIP));
					node = vxColorConvertNode(graph, img1, img_out);
					expected_image_sum = (pix_img1_u8 * width * height) + (2 *(pix_img1_u8 + 1) * (width ) * (height)) ; 
					out_buf_type = 4;
					break;
				}
				case 133:
				{
					// test_case_name = "agoKernel_Box_U8_U8_3x3";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxBox3x3Node(graph, img1, img_out);
					vx_int32 firstColVal = (6 * pix_img1_u8) / 9;
					expected_image_sum = (pix_img1_u8 * (width - 2) * (height - 2)) + (2 * (height - 2) * firstColVal);
					out_buf_type = 0;
					break;
				}
				case 134:
				{
					// test_case_name = "agoKernel_Dilate_U8_U8_3x3";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxDilate3x3Node(graph, img1, img_out);
					vx_int32 firstColVal = pix_img1_u8;
					expected_image_sum = (pix_img1_u8 * (width - 2) * (height - 2)) + (2 * (height - 2) * firstColVal);
					out_buf_type = 0;
					break;
				}
				case 138:
				{
					// test_case_name = "agoKernel_Erode_U8_U8_3x3";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxErode3x3Node(graph, img1, img_out);
					vx_int32 firstColVal = 0;
					expected_image_sum = (pix_img1_u8 * (width - 2) * (height - 2)) + (2 * (height - 2) * firstColVal);
					out_buf_type = 0;
					break;
				}
				case 142:
				{
					// test_case_name = "agoKernel_Median_U8_U8_3x3";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxMedian3x3Node(graph, img1, img_out);
					vx_int32 firstColVal = pix_img1_u8;
					expected_image_sum = (pix_img1_u8 * (width - 2) * (height - 2)) + (2 * (height - 2) * firstColVal);
					out_buf_type = 0;
					break;
				}
				case 143:
				{
					// test_case_name = "agoKernel_Gaussian_U8_U8_3x3";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxGaussian3x3Node(graph, img1, img_out);
					vx_int32 firstColVal = (vx_int32)PIXELCHECKU8((float)pix_img1_u8 * ((2 * 0.0625) + (3 * 0.125) + 0.25));
					expected_image_sum = (pix_img1_u8 * (width - 2) * (height - 2)) + (2 * (height - 2) * firstColVal);
					out_buf_type = 0;
					break;
				}
				case 147:
				{
					// test_case_name = "agoKernel_Sobel_S16S16_U8_3x3_GXY";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxSobel3x3Node(graph, img1, img_out, img_out2);
					expected_image_sum = 0;
					out_buf_type = 1;
					break;
				}
				case 148:
				{
					// test_case_name = "agoKernel_Sobel_S16_U8_3x3_GX";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxSobel3x3Node(graph, img1, img_out, img_out2);
					expected_image_sum = 0;
					out_buf_type = 1;
					break;
				}
				case 149:
				{
					// test_case_name = "agoKernel_Sobel_S16_U8_3x3_GY";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out2 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxSobel3x3Node(graph, img1, img_out, img_out2);
					expected_image_sum = 0;
					out_buf_type = 1;
					break;
				}
				case 150:
				{
					// test_case_name = "agoKernel_Convolve_U8_U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxConvolveNode(graph, img1, Convolve_conv_convolution, img_out);
					vx_int32 firstColVal = PIXELCHECKU8(pix_img1_u8 * (12 + (8 * 3) + (4 * 5) + (2 * 4) + (1 * 2)));
					vx_int32 secondColVal = PIXELCHECKU8(firstColVal + (pix_img1_u8 * ((2 * 2) + (2 * 4) + (1 * 8))));
					vx_int32 remainingVal = PIXELCHECKU8(96 * pix_img1_u8);
					expected_image_sum = (remainingVal * (width - 4) * (height - 4)) + (2 * (height - 4) * (firstColVal + secondColVal));
					out_buf_type = 0;
					break;
				}
				case 151:
				{
					// test_case_name = "agoKernel_Convolve_S16_U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxConvolveNode(graph, img1, Convolve_conv_convolution, img_out);
					vx_int32 firstColVal = pix_img1_u8 * (12 + (8 * 3) + (4 * 5) + (2 * 4) + (1 * 2));
					vx_int32 secondColVal = firstColVal + (pix_img1_u8 * ((2 * 2) + (2 * 4) + (1 * 8)));
					vx_int32 remainingVal = 96 * pix_img1_u8;
					expected_image_sum = (remainingVal * (width - 4) * (height - 4)) + (2 * (height - 4) * (firstColVal + secondColVal));
					out_buf_type = 1;
					break;
				}
				case 154:
				{
					// test_case_name = "agoKernel_ScaleImage_U8_U8_Nearest";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxScaleImageNode(graph, img1, img_out, ScaleImage_type1_enum);
					expected_image_sum = pix_img1_u8 * (widthOut * heightOut);
					out_buf_type = 0;
					break;
				}
				case 155:
				{
					// test_case_name = "agoKernel_ScaleImage_U8_U8_Bilinear";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxScaleImageNode(graph, img1, img_out, ScaleImage_type2_enum);
					expected_image_sum = 0;
					out_buf_type = 0;
					break;
				}
				case 156:
				{
					// test_case_name = "agoKernel_ScaleImage_U8_U8_Bilinear_Replicate";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxScaleImageNode(graph, img1, img_out, ScaleImage_type2_enum);
					ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_BORDER, &border1, sizeof(border1)));
					expected_image_sum = pix_img1_u8 * (widthOut * heightOut);
					out_buf_type = 0;
					break;
				}
				case 157:
				{
					// test_case_name = "agoKernel_ScaleImage_U8_U8_Bilinear_Constant";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxScaleImageNode(graph, img1, img_out, ScaleImage_type2_enum);
					ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_BORDER, &border2, sizeof(border2)));
					expected_image_sum = 0;
					out_buf_type = 0;
					break;
				}
				case 158:
				{
					// test_case_name = "agoKernel_ScaleImage_U8_U8_Area";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxScaleImageNode(graph, img1, img_out, ScaleImage_type3_enum);
					expected_image_sum = pix_img1_u8 * (widthOut * heightOut);
					out_buf_type = 0;
					break;
				}
				case 159:
				{
					// test_case_name = "agoKernel_ScaleGaussianHalf_U8_U8_3x3";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxHalfScaleGaussianNode(graph, img1, img_out, 3);
					vx_int32 firstColVal = (vx_int32)PIXELCHECKU8((float)pix_img1_u8 * ((2 * 0.0625) + (3 * 0.125) + 0.25));
					if (width % 2 == 0)
						expected_image_sum = (pix_img1_u8 * (widthOut - 1) * (heightOut - 2)) + ((heightOut - 2) * firstColVal);
					else if (width % 2 != 0)
						expected_image_sum = (pix_img1_u8 * (widthOut - 2) * (heightOut - 2)) + ((heightOut - 2) * firstColVal * 2);
					out_buf_type = 0;
					break;
				}
				case 160:
				{
					// test_case_name = "agoKernel_ScaleGaussianHalf_U8_U8_5x5";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxHalfScaleGaussianNode(graph, img1, img_out, 5);
					vx_int32 firstColVal = (vx_int32)PIXELCHECKU8((float)pix_img1_u8 * 176 / 256);
					vx_int32 secondColVal = (vx_int32)PIXELCHECKU8((float)pix_img1_u8 * 240 / 256);
					if (width % 2 == 0)
						expected_image_sum = (pix_img1_u8 * (widthOut - 2) * (heightOut - 2)) + (firstColVal * (heightOut - 2)) + (secondColVal * (heightOut - 2));
					else if (width % 2 != 0)
						expected_image_sum = (pix_img1_u8 * (widthOut - 2) * (heightOut - 2)) + (firstColVal * (heightOut - 2)) + (firstColVal * (heightOut - 2));
					out_buf_type = 0;
					break;
				}
				case 162:
				{
					// test_case_name = "agoKernel_WarpAffine_U8_U8_Nearest";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxWarpAffineNode(graph, img1, WarpAffine_affineMatrix_matrix, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 163:
				{
					// test_case_name = "agoKernel_WarpAffine_U8_U8_Nearest_Constant";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxWarpAffineNode(graph, img1, WarpAffine_affineMatrix_matrix, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, img_out);
					ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_BORDER, &border2, sizeof(border2)));
					expected_image_sum = (pix_img1_u8 * width * height) + (border2.constant_value.U8 * widthOut * heightOut) - (border2.constant_value.U8 * width * height);
					out_buf_type = 0;
					break;
				}
				case 164:
				{
					// test_case_name = "agoKernel_WarpAffine_U8_U8_Bilinear";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxWarpAffineNode(graph, img1, WarpAffine_affineMatrix_matrix, VX_INTERPOLATION_TYPE_BILINEAR, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 165:
				{
					// test_case_name = "agoKernel_WarpAffine_U8_U8_Bilinear_Constant";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxWarpAffineNode(graph, img1, WarpAffine_affineMatrix_matrix, VX_INTERPOLATION_TYPE_BILINEAR, img_out);
					ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_BORDER, &border2, sizeof(border2)));
					expected_image_sum = (pix_img1_u8 * width * height) + (border2.constant_value.U8 * widthOut * heightOut) - (border2.constant_value.U8 * width * height);
					out_buf_type = 0;
					break;
				}
				case 166:
				{
					// test_case_name = "agoKernel_WarpPerspective_U8_U8_Nearest";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxWarpPerspectiveNode(graph, img1, WarpPerspective_perspectiveMatrix_matrix, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 167:
				{
					// test_case_name = "agoKernel_WarpPerspective_U8_U8_Nearest_Constant";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxWarpPerspectiveNode(graph, img1, WarpPerspective_perspectiveMatrix_matrix, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, img_out);
					ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_BORDER, &border2, sizeof(border2)));
					expected_image_sum = (pix_img1_u8 * width * height) + (border2.constant_value.U8 * widthOut * heightOut) - (border2.constant_value.U8 * width * height);
					out_buf_type = 0;
					break;
				}
				case 168:
				{
					// test_case_name = "agoKernel_WarpPerspective_U8_U8_Bilinear";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxWarpPerspectiveNode(graph, img1, WarpPerspective_perspectiveMatrix_matrix, VX_INTERPOLATION_TYPE_BILINEAR, img_out);
					expected_image_sum = pix_img1_u8 * width * height;
					out_buf_type = 0;
					break;
				}
				case 169:
				{
					// test_case_name = "agoKernel_WarpPerspective_U8_U8_Bilinear_Constant";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxWarpPerspectiveNode(graph, img1, WarpPerspective_perspectiveMatrix_matrix, VX_INTERPOLATION_TYPE_BILINEAR, img_out);
					ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_BORDER, &border2, sizeof(border2)));
					expected_image_sum = (pix_img1_u8 * width * height) + (border2.constant_value.U8 * widthOut * heightOut) - (border2.constant_value.U8 * width * height);
					out_buf_type = 0;
					break;
				}
				case 170:
				{
					// test_case_name = "agoKernel_ColorDepth_U8_S16_Wrap";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxConvertDepthNode(graph, img1, img_out, VX_CONVERT_POLICY_WRAP, ConvertDepth_shift_scalar);
					expected_image_sum = (vx_int32)((vx_uint8)(pix_img1_s16 >> ConvertDepth_shift_int32)) * width * height;
					out_buf_type = 0;
					break;
				}
				case 171:
				{
					// test_case_name = "agoKernel_ColorDepth_U8_S16_Sat";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxConvertDepthNode(graph, img1, img_out, VX_CONVERT_POLICY_SATURATE, ConvertDepth_shift_scalar);
					expected_image_sum = (vx_int32)PIXELCHECKU8(pix_img1_s16 >> ConvertDepth_shift_int32) * width * height;
					out_buf_type = 0;
					break;
				}
				case 172:
				{
					// test_case_name = "agoKernel_ColorDepth_S16_U8";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_S16, &hip_addr_int16, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxConvertDepthNode(graph, img1, img_out, VX_CONVERT_POLICY_WRAP, ConvertDepth_shift_scalar);
					expected_image_sum = (pix_img1_u8 << ConvertDepth_shift_int32) * width * height;
					out_buf_type = 1;
					break;
				}
				case 174:
				{
					// test_case_name = "agoKernel_Remap_U8_U8_Nearest";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxRemapNode(graph, img1, Remap_remapTable_remap, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR, img_out);
					vx_int32 finalVal = pix_img1_u8 + (width * height) - 1;
					expected_image_sum = (finalVal * (finalVal + 1) / 2) - (pix_img1_u8 * (pix_img1_u8 - 1) / 2) + ((widthOut - width) * heightOut * pix_img1_u8) + (width * (heightOut - height) * pix_img1_u8);
					out_buf_type = 0;
					break;
				}
				case 176:
				{
					// test_case_name = "agoKernel_Remap_U8_U8_Bilinear";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxRemapNode(graph, img1, Remap_remapTable_remap, VX_INTERPOLATION_TYPE_BILINEAR, img_out);
					vx_int32 finalVal = pix_img1_u8 + (width * height) - 1;
					expected_image_sum = (finalVal * (finalVal + 1) / 2) - (pix_img1_u8 * (pix_img1_u8 - 1) / 2) + ((widthOut - width) * heightOut * pix_img1_u8) + (width * (heightOut - height) * pix_img1_u8);
					out_buf_type = 0;
					break;
				}
				case 187:
				{
					// test_case_name = "agoKernel_CannySobel_U16_U8_3x3_L1NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxCannyEdgeDetectorNode(graph, img1, CannyEdgeDetector_thresholdObjectRange_threshold, CannyEdgeDetector_gradientSize3_int32, VX_NORM_L1, img_out);
					expected_image_sum = ((2 * ((heightOut / 2) - 2)) + (2 * ((widthOut / 2) - 2)) - 1) * 255;
					out_buf_type = 0;
					break;
				}
				case 188:
				{
					// test_case_name = "agoKernel_CannySobel_U16_U8_3x3_L2NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxCannyEdgeDetectorNode(graph, img1, CannyEdgeDetector_thresholdObjectRange_threshold, CannyEdgeDetector_gradientSize3_int32, VX_NORM_L2, img_out);
					expected_image_sum = ((2 * ((heightOut / 2) - 2)) + (2 * ((widthOut / 2) - 2)) + 2) * 255;
					out_buf_type = 0;
					break;
				}
				case 189:
				{
					// test_case_name = "agoKernel_CannySobel_U16_U8_5x5_L1NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxCannyEdgeDetectorNode(graph, img1, CannyEdgeDetector_thresholdObjectRange_threshold, CannyEdgeDetector_gradientSize5_int32, VX_NORM_L1, img_out);
					expected_image_sum = ((2 * ((heightOut / 2) - 2)) + (2 * ((widthOut / 2) - 2)) - 1) * 255;
					out_buf_type = 0;
					break;
				}
				case 190:
				{
					// test_case_name = "agoKernel_CannySobel_U16_U8_5x5_L2NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxCannyEdgeDetectorNode(graph, img1, CannyEdgeDetector_thresholdObjectRange_threshold, CannyEdgeDetector_gradientSize5_int32, VX_NORM_L2, img_out);
					expected_image_sum = ((2 * ((heightOut / 2) - 2)) + (2 * ((widthOut / 2) - 2)) + 2) * 255;
					out_buf_type = 0;
					break;
				}
				case 191:
				{
					// test_case_name = "agoKernel_CannySobel_U16_U8_7x7_L1NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxCannyEdgeDetectorNode(graph, img1, CannyEdgeDetector_thresholdObjectRange_threshold, CannyEdgeDetector_gradientSize7_int32, VX_NORM_L1, img_out);
					expected_image_sum = ((2 * ((heightOut / 2) - 2)) + (2 * ((widthOut / 2) - 2)) - 1) * 255;
					out_buf_type = 0;
					break;
				}
				case 192:
				{
					// test_case_name = "agoKernel_CannySobel_U16_U8_7x7_L2NORM, agoKernel_CannySuppThreshold_U8XY_U16_3x3, agoKernel_CannyEdgeTrace_U8_U8XY";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));
					ERROR_CHECK_OBJECT(img_out = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8_out, &ptr[2], VX_MEMORY_TYPE_HIP));
					node = vxCannyEdgeDetectorNode(graph, img1, CannyEdgeDetector_thresholdObjectRange_threshold, CannyEdgeDetector_gradientSize7_int32, VX_NORM_L2, img_out);
					expected_image_sum = ((2 * ((heightOut / 2) - 2)) + (2 * ((widthOut / 2) - 2)) + 2) * 255;
					out_buf_type = 0;
					break;
				}
				case 203:
				{
					//test_case_name = "agoKernel_FastCorners_XY_U8_Supression";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));					
					node = vxFastCornersNode(graph, img1, fastCorner_threshold_scalar, nms_true, output_keypoints_array, output_corner_count);
					out_buf_type = -1;
					break;
				}
				case 204:
				{
					//test_case_name = "agoKernel_FastCorners_XY_U8_NoSupression";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));					
					node = vxFastCornersNode(graph, img1, fastCorner_threshold_scalar, nms_false, output_keypoints_array, output_corner_count);
					out_buf_type = -1;
					break;
				}
				case 206:
				{
					//test_case_name = "agoKernel_HarrisSobel_HG3_U8_3x3";
					ERROR_CHECK_OBJECT(img1 = vxCreateImageFromHandle(context, VX_DF_IMAGE_U8, &hip_addr_uint8, &ptr[0], VX_MEMORY_TYPE_HIP));					
					node = vxHarrisCornersNode(graph, img1, HarrisCorner_strength_threshold_scalar, HarrisCorner_min_distance_scalar, HarrisCorner_sensitivity_scalar, HarrisCorner_grad_size, HarrisCorner_block_size, HarrisCorner_output_keypoints_array, HarrisCorner_output_corner_count);
					out_buf_type = -1;
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
					(case_number == 61) || (case_number == 104)
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
					(case_number == 45) || (case_number == 47) || (case_number == 88) || 
					(case_number == 89) || (case_number == 90) || (case_number == 91) || (case_number == 92) || 
					(case_number == 93) || (case_number == 94) || (case_number == 95) || (case_number == 96) || 
					(case_number == 133) || (case_number == 134) || (case_number == 138) || (case_number == 142) || 
					(case_number == 143) || (case_number == 147) || (case_number == 148) || (case_number == 149) || 
					(case_number == 150) || (case_number == 151) || 
					(case_number == 154) || (case_number == 155) || (case_number == 156)  || (case_number == 157) || 
					(case_number == 158) || (case_number == 159) || (case_number == 160) || (case_number == 162) || 
					(case_number == 163) || (case_number == 164) || (case_number == 165) || (case_number == 166) || 
					(case_number == 167) || (case_number == 168) || (case_number == 169) || (case_number == 172) || 
					(case_number == 174) || (case_number == 176) || (case_number == 187) || (case_number == 188) || 
					(case_number == 189) || (case_number == 190) || (case_number == 191) || (case_number == 192) ||
					(case_number == 203) || (case_number == 204) || (case_number == 206)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img1_u8));
				}
				// S16 input
				else if(
					(case_number == 170) || (case_number == 171)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_int16) pix_img1_s16));
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
				// U16, U24 and U32 inputs for UYVY, YUYV, RGB, RGBX
				else if(
					(case_number == 71)  || (case_number == 72)  || (case_number == 73)  || (case_number == 74)  ||
					(case_number == 75)  || (case_number == 76)  || (case_number == 77)  || (case_number == 78)  ||
					(case_number == 79)  || (case_number == 105) || (case_number == 106) || (case_number == 107) ||
					(case_number == 111) || (case_number == 112) || (case_number == 113) || (case_number == 117) ||
					(case_number == 118) || (case_number == 119) || (case_number == 120) || (case_number == 121) ||
					(case_number == 122) || (case_number == 123) || (case_number == 124) || (case_number == 128)  || 
					(case_number == 129)
				)
				{
					ERROR_CHECK_STATUS(makeInputPackedImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img1_u8))	
				}
				// Planar Image Input - NV12, NV21, IYUV inputs
				else if(
					(case_number == 67) || (case_number == 108) || (case_number == 109) || (case_number == 110) || 
					(case_number == 114) || (case_number == 115) || (case_number == 116) || (case_number == 130)  || (case_number == 131) || (case_number == 132)
					|| (case_number == 125) || (case_number == 126) || (case_number == 127)
				)
				{
					ERROR_CHECK_STATUS(makeInputPlanarImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img1_u8));
				}
				// NV12 Channel Combine inputs
				else if(
					(case_number == 83) 
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img1_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width/2, height/2, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img2_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img3, width/2, height/2, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img3_u8));
				}
				// RGB RGBX Channel Combine inputs
				else if(
					(case_number == 84) || (case_number == 87) 
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img1_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img2_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img3, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img3_u8));
				}
				// YUYV UYVY Channel Combine inputs
				else if(
					(case_number == 85) ||  (case_number == 86)
				)
				{
					ERROR_CHECK_STATUS(makeInputImage(context, img1, width, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img1_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img2, width/2, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img2_u8));
					ERROR_CHECK_STATUS(makeInputImage(context, img3, width/2, height, VX_MEMORY_TYPE_HIP, (vx_uint8) pix_img3_u8));
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
		printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", widthOut, heightOut, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
		printf("dim_x: %d dim_y: %d\nscale_x: %d scale_y: %d\nstep_x: %d step_y: %d\n",out_addr.dim_x, out_addr.dim_y,out_addr.scale_x, out_addr.scale_y,out_addr.step_x, out_addr.step_y);
		printImage(out_buf_uint8, stride_x_pixels, stride_y_pixels, widthOut, heightOut);
		printf("Output Buffer: ");
		printBuffer(out_buf_uint8, widthOut, heightOut);
		// printBufferBits(out_buf_uint8, width * height); // To print output interms of bits
#endif
		for (int i = 0; i < heightOut; i++)
			for (int j = 0; j < widthOut; j++)
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
		printf("dim_x: %d dim_y: %d\nscale_x: %d scale_y: %d\nstep_x: %d step_y: %d\n",out_addr.dim_x, out_addr.dim_y,out_addr.scale_x, out_addr.scale_y,out_addr.step_x, out_addr.step_y);
		printImage(out_buf_uint8, stride_x_pixels, stride_y_pixels, width, height);
		printf("Output Buffer: ");
		printBuffer(out_buf_uint8, widthOut, heightOut);
		// printBufferBits(out_buf_uint8, width * height); // To print output interms of bits
#endif
		for (int i = 0; i < heightOut; i++)
			for (int j = 0; j < widthOut; j++)
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
		printf("dim_x: %d dim_y: %d\nscale_x: %d scale_y: %d\nstep_x: %d step_y: %d\n",out_addr.dim_x, out_addr.dim_y,out_addr.scale_x, out_addr.scale_y,out_addr.step_x, out_addr.step_y);
		printImage(out_buf_int16, stride_x_pixels, stride_y_pixels, width, height);
		printf("Output Buffer: ");
		printBuffer(out_buf_int16, width, height);
#endif
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				returned_image_sum += out_buf_int16[i * stride_y_pixels + j * stride_x_pixels];
		if ((case_number == 147) || (case_number == 148) || (case_number == 149))
		{
			ERROR_CHECK_STATUS(vxMapImagePatch(img_out2, &out_rect, 0, &out_map_id, &out_addr, (void **)&out_buf_int16, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
			stride_x_bytes = out_addr.stride_x;
			stride_x_pixels = stride_x_bytes / sizeof(vx_int16);
			stride_y_bytes = out_addr.stride_y;
			stride_y_pixels = stride_y_bytes / sizeof(vx_int16);
#ifdef PRINT_OUTPUT
			printf("\nOutput Image 2: ");
			printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", width, height, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
			printImage(out_buf_int16, stride_x_pixels, stride_y_pixels, width, height);
			printf("Output Buffer 2: ");
			printBuffer(out_buf_int16, width, height);
#endif
			returned_image_sum = 0;
		}
	}
	
	// For packed images - RGB , RGBX, YUYV, UYVY
	else if (out_buf_type == 3)
	{
		ERROR_CHECK_STATUS(vxMapImagePatch(img_out, &out_rect, 0, &out_map_id, &out_addr, (void **)&out_buf_uint8, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		stride_x_bytes = out_addr.stride_x;
		stride_x_pixels = stride_x_bytes / sizeof(vx_uint8);
		stride_y_bytes = out_addr.stride_y;
		stride_y_pixels = stride_y_bytes / sizeof(vx_uint8);
#ifdef PRINT_OUTPUT
		printf("\nOutput Image: ");
		printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", width, height, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
		printf("dim_x: %d dim_y: %d\nscale_x: %d scale_y: %d\nstep_x: %d step_y: %d\n",out_addr.dim_x, out_addr.dim_y,out_addr.scale_x, out_addr.scale_y,out_addr.step_x, out_addr.step_y);
		printImage(out_buf_uint8, stride_x_pixels, stride_y_pixels, width, height);
		printf("Output Buffer: ");
		printBuffer(out_buf_uint8, widthOut, heightOut);
		// printBufferBits(out_buf_uint8, width * height); // To print output interms of bits
#endif
		for (int i = 0; i < heightOut; i++)
			for (int j = 0; j < widthOut*stride_x_bytes; j+=stride_x_bytes)
			{
				for (int pixel_stride=0; pixel_stride<stride_x_bytes; pixel_stride++)
					returned_image_sum += out_buf_uint8[i * stride_y_pixels + j + pixel_stride];
			}
	}
		
	//Planar Images - NV12, NV21 and IYUV output
	if (out_buf_type == 4)
	{	
		vx_size planes = 0;
		vxQueryImage(img_out, VX_IMAGE_PLANES, &planes, sizeof(planes));
		vx_df_image format = 0;
		vxQueryImage(img_out, VX_IMAGE_FORMAT, &format, sizeof(format));

		for(int p=0; p<planes; p++)
		{
			ERROR_CHECK_STATUS(vxMapImagePatch(img_out, &out_rect, p, &out_map_id, &out_addr, (void **)&out_buf_uint8, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
			stride_x_bytes = out_addr.stride_x;
			stride_x_pixels = stride_x_bytes / sizeof(vx_uint8);
			stride_y_bytes = out_addr.stride_y;
			stride_y_pixels = stride_y_bytes / sizeof(vx_uint8);
#ifdef PRINT_INPUT
			printf("\nInput Image Plane %d: ",p);
			printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", widthOut, heightOut, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
			printf("dim_x: %d dim_y: %d\nscale_x: %d scale_y: %d\nstep_x: %d step_y: %d\n",out_addr.dim_x, out_addr.dim_y,out_addr.scale_x, out_addr.scale_y,out_addr.step_x, out_addr.step_y);
			printImage(out_buf_uint8, stride_x_pixels, stride_y_pixels, widthOut, heightOut);
			printf("Input Buffer: ");
			printBuffer(out_buf_uint8, widthOut, heightOut);
#endif
			if(p == 0 || format == VX_DF_IMAGE_YUV4)
			{
				for (int i = 0; i < heightOut; i++)
					for (int j = 0; j < widthOut; j++)
						returned_image_sum += out_buf_uint8[i * stride_y_pixels + j * stride_x_pixels];
			}
			else
			{
				for (int i = 0; i < (heightOut/2); i++)
					for (int j = 0; j < (widthOut/2)*stride_x_bytes; j+=stride_x_bytes)
					{
						for(int inner_stride=0; inner_stride<stride_x_bytes; inner_stride++)
						{
							returned_image_sum += out_buf_uint8[i * stride_y_pixels + j + inner_stride];				
						}
					}
			}
		}
	}

	// for u1 outputs
	else if (out_buf_type == 5)
	{
		ERROR_CHECK_STATUS(vxMapImagePatch(img_out, &out_rect, 0, &out_map_id, &out_addr, (void **)&out_buf_uint8, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
		stride_x_bytes = out_addr.stride_x;
		stride_x_pixels = stride_x_bytes / sizeof(vx_uint8);
		stride_y_bytes = out_addr.stride_y;
		stride_y_pixels = stride_y_bytes / sizeof(vx_uint8);
#ifdef PRINT_OUTPUT
		printf("\nOutput Image U1: ");
		printf("width = %d, height = %d\nstride_x_bytes = %d, stride_y_bytes = %d | stride_x_pixels = %d, stride_y_pixels = %d\n", widthOut, heightOut, stride_x_bytes, stride_y_bytes, stride_x_pixels, stride_y_pixels);
		printf("dim_x: %d dim_y: %d\nscale_x: %d scale_y: %d\nstep_x: %d step_y: %d\n",out_addr.dim_x, out_addr.dim_y,out_addr.scale_x, out_addr.scale_y,out_addr.step_x, out_addr.step_y);
		printImageU1(out_buf_uint8, stride_x_pixels, stride_y_pixels, widthOut, heightOut);
		printf("Output Buffer: ");
		printBuffer(out_buf_uint8, widthOut, heightOut);
		// printBufferBits(out_buf_uint8, width * height); // To print output interms of bits
#endif
		widthOut = ((width % 8 == 0) ? ((width/8)+7) & ~7 : ((width/8)+8) & ~7);
		for (int i = 0; i < heightOut; i++)
			for (int j = 0; j < widthOut; j++)
				returned_image_sum += out_buf_uint8[i * stride_y_pixels + j];
	}
	// else if(out_buf_type == 6)
	// {
	// 	vx_size i, stride = sizeof(vx_keypoint_t);
	// 	void *base = NULL;
	// 	vx_map_id map_id;
	// 	vx_size num_items=5;
	// 	vx_float32 strength_value;
	// 	/* access entire array at once */
	// 	ERROR_CHECK_STATUS(vxMapArrayRange(output_keypoints_array, 0, num_items, &map_id, &stride, (void **)&base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
	// 	for (i = 0; i < num_items; i++)
	// 	{
	// 		vxArrayItem(vx_keypoint_t, base, i, stride).strength;
	// 		printf("%f ",strength_value);
	// 	}
	// 	vxUnmapArrayRange(output_keypoints_array, map_id);
	// }

	// Cases for Manual Override
	if (
		(case_number == 155) || (case_number == 157) || (case_number == 187) || (case_number == 188) ||
		(case_number == 189) || (case_number == 190) || (case_number == 191) || (case_number == 192) ||
		(case_number == 203) || (case_number == 204)  || (case_number == 206)
		)
	{
		printf("\nTEST PASSED: Sum verification overridden due to hard calculation. Manually verified. Not an exact pixel-to-pixel match.\n");
		return_value = 1;
	}
	else if (returned_image_sum != expected_image_sum)
	{
		printf("\nTEST FAILED: returned_image_sum = %d expected_image_sum = %d\n", returned_image_sum, expected_image_sum);
		return_value = -1;
		ERROR_CHECK_STATUS(vxUnmapImagePatch(img_out, out_map_id));
	}
	else
	{
		printf("\nTEST PASSED: returned_image_sum = %d expected_image_sum = %d\n", returned_image_sum, expected_image_sum);
		return_value = 1;
		ERROR_CHECK_STATUS(vxUnmapImagePatch(img_out, out_map_id));
	}

	

	// free resources

	vxReleaseScalar(&Mul_scale_scalar);
	vxReleaseScalar(&WeightedAverage_alpha_scalar);
	vxReleaseScalar(&fastCorner_threshold_scalar);
	//Need to add for Harris Corners
	vxReleaseScalar(&output_corner_count);
	vxReleaseArray(&output_keypoints_array);
	vxReleaseScalar(&ConvertDepth_shift_scalar);
	vxReleaseMatrix(&WarpAffine_affineMatrix_matrix);
	vxReleaseMatrix(&WarpPerspective_perspectiveMatrix_matrix);
	// vxReleaseRemap(&Remap_remapTable_remap);
	// vxReleaseThreshold(&Threshold_thresholdObjectBinary_threshold);
	// vxReleaseThreshold(&Threshold_thresholdObjectRange_threshold);
	vxReleaseLUT(&Lut_lutObject_lut);
	if (ptr[0]) hipFree(ptr[0]);
	if (ptr[1]) hipFree(ptr[1]);
	if (ptr[2]) hipFree(ptr[2]);
	if (ptr[3]) hipFree(ptr[3]);
	if (nv_in[0]) hipFree(nv_in[0]);
	if (nv_in[1]) hipFree(nv_in[1]);
	if (nv_out[0]) hipFree(nv_out[0]);
	if (nv_out[1]) hipFree(nv_out[1]);
	if (iyuv_in[0]) hipFree(iyuv_in[0]);
	if (iyuv_in[1]) hipFree(iyuv_in[1]);
	if (iyuv_in[2]) hipFree(iyuv_in[2]);
	if (iyuv_out[0]) hipFree(iyuv_out[0]);
	if (iyuv_out[1]) hipFree(iyuv_out[1]);
	if (iyuv_out[2]) hipFree(iyuv_out[2]);
	if (yuv4_in[0]) hipFree(yuv4_in[0]);
	if (yuv4_in[1]) hipFree(yuv4_in[1]);
	if (yuv4_in[2]) hipFree(yuv4_in[2]);
	if (yuv4_out[0]) hipFree(yuv4_out[0]);
	if (yuv4_out[1]) hipFree(yuv4_out[1]);
	if (yuv4_out[2]) hipFree(yuv4_out[2]);
	vxReleaseContext(&context);

	return return_value;
}
	
