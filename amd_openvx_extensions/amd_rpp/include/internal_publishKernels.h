/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef _PUBLISH_KERNELS_H_
#define _PUBLISH_KERNELS_H_

#if _WIN32
#define SHARED_PUBLIC __declspec(dllexport)
#else
#define SHARED_PUBLIC __attribute__ ((visibility ("default")))
#endif

#include "internal_rpp.h"

extern "C" SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context);
vx_status ADD_KERENEL(std::function<vx_status(vx_context)>);
vx_status get_kernels_to_publish();

vx_status AbsoluteDifferencebatchPD_Register(vx_context);
vx_status AccumulatebatchPD_Register(vx_context);
vx_status AccumulateSquaredbatchPD_Register(vx_context);
vx_status AccumulateWeightedbatchPD_Register(vx_context);
vx_status AddbatchPD_Register(vx_context);
vx_status BilateralFilterbatchPD_Register(vx_context);
vx_status BitwiseANDbatchPD_Register(vx_context);
vx_status BitwiseNOTbatchPD_Register(vx_context);
vx_status BlendbatchPD_Register(vx_context);
vx_status BlurbatchPD_Register(vx_context);
vx_status BoxFilterbatchPD_Register(vx_context);
vx_status BrightnessbatchPD_Register(vx_context);
vx_status CannyEdgeDetector_Register(vx_context);
vx_status ChannelCombinebatchPD_Register(vx_context);
vx_status ChannelExtractbatchPD_Register(vx_context);
vx_status ColorTemperaturebatchPD_Register(vx_context);
vx_status ColorTwistbatchPD_Register(vx_context);
vx_status ContrastbatchPD_Register(vx_context);
vx_status Copy_Register(vx_context);
vx_status CropMirrorNormalizePD_Register(vx_context);
vx_status CropPD_Register(vx_context);
vx_status CustomConvolutionbatchPD_Register(vx_context);
vx_status DataObjectCopybatchPD_Register(vx_context);
vx_status DilatebatchPD_Register(vx_context);
vx_status ErodebatchPD_Register(vx_context);
vx_status ExclusiveORbatchPD_Register(vx_context);
vx_status ExposurebatchPD_Register(vx_context);
vx_status FastCornerDetector_Register(vx_context);
vx_status FisheyebatchPD_Register(vx_context);
vx_status FlipbatchPD_Register(vx_context);
vx_status FogbatchPD_Register(vx_context);
vx_status GammaCorrectionbatchPD_Register(vx_context);
vx_status GaussianFilterbatchPD_Register(vx_context);
vx_status GaussianImagePyramidbatchPD_Register(vx_context);
vx_status HarrisCornerDetector_Register(vx_context);
vx_status Histogram_Register(vx_context);
vx_status HistogramBalancebatchPD_Register(vx_context);
vx_status HistogramEqualizebatchPD_Register(vx_context);
vx_status HuebatchPD_Register(vx_context);
vx_status InclusiveORbatchPD_Register(vx_context);
vx_status JitterbatchPD_Register(vx_context);
vx_status LaplacianImagePyramid_Register(vx_context);
vx_status LensCorrectionbatchPD_Register(vx_context);
vx_status LocalBinaryPatternbatchPD_Register(vx_context);
vx_status LookUpTablebatchPD_Register(vx_context);
vx_status MagnitudebatchPD_Register(vx_context);
vx_status MaxbatchPD_Register(vx_context);
vx_status MeanStddev_Register(vx_context);
vx_status MedianFilterbatchPD_Register(vx_context);
vx_status MinbatchPD_Register(vx_context);
vx_status MinMaxLoc_Register(vx_context);
vx_status MultiplybatchPD_Register(vx_context);
vx_status NoisebatchPD_Register(vx_context);
vx_status NonLinearFilterbatchPD_Register(vx_context);
vx_status NonMaxSupressionbatchPD_Register(vx_context);
vx_status Nop_Register(vx_context);
vx_status PhasebatchPD_Register(vx_context);
vx_status PixelatebatchPD_Register(vx_context);
vx_status RainbatchPD_Register(vx_context);
vx_status RandomCropLetterBoxbatchPD_Register(vx_context);
vx_status RandomShadowbatchPD_Register(vx_context);
vx_status remap_Register(vx_context);
vx_status ResizebatchPD_Register(vx_context);
vx_status ResizeCropbatchPD_Register(vx_context);
vx_status ResizeCropMirrorPD_Register(vx_context);
vx_status RotatebatchPD_Register(vx_context);
vx_status SaturationbatchPD_Register(vx_context);
vx_status ScalebatchPD_Register(vx_context);
vx_status SnowbatchPD_Register(vx_context);
vx_status SobelbatchPD_Register(vx_context);
vx_status SubtractbatchPD_Register(vx_context);
vx_status TensorAdd_Register(vx_context);
vx_status TensorLookup_Register(vx_context);
vx_status TensorMatrixMultiply_Register(vx_context);
vx_status TensorMultiply_Register(vx_context);
vx_status TensorSubtract_Register(vx_context);
vx_status ThresholdingbatchPD_Register(vx_context);
vx_status VignettebatchPD_Register(vx_context);
vx_status WarpAffinebatchPD_Register(vx_context);
vx_status WarpPerspectivebatchPD_Register(vx_context);

// kernel names
#define VX_KERNEL_RPP_NOP_NAME                          "org.rpp.Nop"
#define VX_KERNEL_RPP_COPY_NAME                         "org.rpp.Copy"
#define VX_KERNEL_RPP_BRIGHTNESSBATCHPD_NAME      		"org.rpp.BrightnessbatchPD"
#define VX_KERNEL_RPP_GAMMACORRECTIONBATCHPD_NAME      	"org.rpp.GammaCorrectionbatchPD"
#define VX_KERNEL_RPP_BLENDBATCHPD_NAME      			"org.rpp.BlendbatchPD"
#define VX_KERNEL_RPP_BLURBATCHPD_NAME      			"org.rpp.BlurbatchPD"
#define VX_KERNEL_RPP_CONTRASTBATCHPD_NAME      		"org.rpp.ContrastbatchPD"
#define VX_KERNEL_RPP_PIXELATEBATCHPD_NAME      		"org.rpp.PixelatebatchPD"
#define VX_KERNEL_RPP_JITTERBATCHPD_NAME      			"org.rpp.JitterbatchPD"
#define VX_KERNEL_RPP_SNOWBATCHPD_NAME      			"org.rpp.SnowbatchPD"
#define VX_KERNEL_RPP_NOISEBATCHPD_NAME      			"org.rpp.NoisebatchPD"
#define VX_KERNEL_RPP_RANDOMSHADOWBATCHPD_NAME      	"org.rpp.RandomShadowbatchPD"
#define VX_KERNEL_RPP_FOGBATCHPD_NAME      				"org.rpp.FogbatchPD"
#define VX_KERNEL_RPP_RAINBATCHPD_NAME      			"org.rpp.RainbatchPD"
#define VX_KERNEL_RPP_RANDOMCROPLETTERBOXBATCHPD_NAME      "org.rpp.RandomCropLetterBoxbatchPD"
#define VX_KERNEL_RPP_EXPOSUREBATCHPD_NAME      		"org.rpp.ExposurebatchPD"
#define VX_KERNEL_RPP_HISTOGRAMBALANCEBATCHPD_NAME      "org.rpp.HistogramBalancebatchPD"
#define VX_KERNEL_RPP_ABSOLUTEDIFFERENCEBATCHPD_NAME    "org.rpp.AbsoluteDifferencebatchPD"
#define VX_KERNEL_RPP_ACCUMULATEWEIGHTEDBATCHPD_NAME    "org.rpp.AccumulateWeightedbatchPD"
#define VX_KERNEL_RPP_ACCUMULATEBATCHPD_NAME      		"org.rpp.AccumulatebatchPD"
#define VX_KERNEL_RPP_ADDBATCHPD_NAME      				"org.rpp.AddbatchPD"
#define VX_KERNEL_RPP_SUBTRACTBATCHPD_NAME      		"org.rpp.SubtractbatchPD"
#define VX_KERNEL_RPP_MAGNITUDEBATCHPD_NAME      				"org.rpp.MagnitudebatchPD"
#define VX_KERNEL_RPP_MULTIPLYBATCHPD_NAME      				"org.rpp.MultiplybatchPD"
#define VX_KERNEL_RPP_PHASEBATCHPD_NAME      					"org.rpp.PhasebatchPD"
#define VX_KERNEL_RPP_ACCUMULATESQUAREDBATCHPD_NAME      		"org.rpp.AccumulateSquaredbatchPD"
#define VX_KERNEL_RPP_BITWISEANDBATCHPD_NAME      				"org.rpp.BitwiseANDbatchPD"
#define VX_KERNEL_RPP_BITWISENOTBATCHPD_NAME      				"org.rpp.BitwiseNOTbatchPD"
#define VX_KERNEL_RPP_EXCLUSIVEORBATCHPD_NAME      				"org.rpp.ExclusiveORbatchPD"
#define VX_KERNEL_RPP_INCLUSIVEORBATCHPD_NAME      				"org.rpp.InclusiveORbatchPD"
#define VX_KERNEL_RPP_HISTOGRAM_NAME      						"org.rpp.Histogram"
#define VX_KERNEL_RPP_THRESHOLDINGBATCHPD_NAME      			"org.rpp.ThresholdingbatchPD"
#define VX_KERNEL_RPP_MAXBATCHPD_NAME      						"org.rpp.MaxbatchPD"
#define VX_KERNEL_RPP_MINBATCHPD_NAME      						"org.rpp.MinbatchPD"
#define VX_KERNEL_RPP_MINMAXLOC_NAME      						"org.rpp.MinMaxLoc"
#define VX_KERNEL_RPP_HISTOGRAMEQUALIZEBATCHPD_NAME      		"org.rpp.HistogramEqualizebatchPD"
#define VX_KERNEL_RPP_MEANSTDDEV_NAME     	 					"org.rpp.MeanStddev"
#define VX_KERNEL_RPP_FLIPBATCHPD_NAME      					"org.rpp.FlipbatchPD"
#define VX_KERNEL_RPP_RESIZEBATCHPD_NAME      					"org.rpp.ResizebatchPD"
#define VX_KERNEL_RPP_RESIZECROPBATCHPD_NAME      				"org.rpp.ResizeCropbatchPD"
#define VX_KERNEL_RPP_ROTATEBATCHPD_NAME      					"org.rpp.RotatebatchPD"
#define VX_KERNEL_RPP_WARPAFFINEBATCHPD_NAME      				"org.rpp.WarpAffinebatchPD"
#define VX_KERNEL_RPP_FISHEYEBATCHPD_NAME      					"org.rpp.FisheyebatchPD"
#define VX_KERNEL_RPP_LENSCORRECTIONBATCHPD_NAME      			"org.rpp.LensCorrectionbatchPD"
#define VX_KERNEL_RPP_SCALEBATCHPD_NAME      					"org.rpp.ScalebatchPD"
#define VX_KERNEL_RPP_WARPPERSPECTIVEBATCHPD_NAME      			"org.rpp.WarpPerspectivebatchPD"
#define VX_KERNEL_RPP_DILATEBATCHPD_NAME      					"org.rpp.DilatebatchPD"
#define VX_KERNEL_RPP_ERODEBATCHPD_NAME      					"org.rpp.ErodebatchPD"
#define VX_KERNEL_RPP_HUEBATCHPD_NAME      						"org.rpp.HuebatchPD"
#define VX_KERNEL_RPP_SATURATIONBATCHPD_NAME      				"org.rpp.SaturationbatchPD"
#define VX_KERNEL_RPP_COLORTEMPERATUREBATCHPD_NAME      		"org.rpp.ColorTemperaturebatchPD"
#define VX_KERNEL_RPP_VIGNETTEBATCHPD_NAME      				"org.rpp.VignettebatchPD"
#define VX_KERNEL_RPP_CHANNELEXTRACTBATCHPD_NAME      			"org.rpp.ChannelExtractbatchPD"
#define VX_KERNEL_RPP_CHANNELCOMBINEBATCHPD_NAME      			"org.rpp.ChannelCombinebatchPD"
#define VX_KERNEL_RPP_LOOKUPTABLEBATCHPD_NAME      				"org.rpp.LookUpTablebatchPD"
#define VX_KERNEL_RPP_BILATERALFILTERBATCHPD_NAME      			"org.rpp.BilateralFilterbatchPD"
#define VX_KERNEL_RPP_BOXFILTERBATCHPD_NAME      				"org.rpp.BoxFilterbatchPD"
#define VX_KERNEL_RPP_SOBELBATCHPD_NAME      					"org.rpp.SobelbatchPD"
#define VX_KERNEL_RPP_MEDIANFILTERBATCHPD_NAME      			"org.rpp.MedianFilterbatchPD"
#define VX_KERNEL_RPP_CUSTOMCONVOLUTIONBATCHPD_NAME      		"org.rpp.CustomConvolutionbatchPD"
#define VX_KERNEL_RPP_NONMAXSUPRESSIONBATCHPD_NAME      		"org.rpp.NonMaxSupressionbatchPD"
#define VX_KERNEL_RPP_GAUSSIANFILTERBATCHPD_NAME      			"org.rpp.GaussianFilterbatchPD"
#define VX_KERNEL_RPP_NONLINEARFILTERBATCHPD_NAME      			"org.rpp.NonLinearFilterbatchPD"
#define VX_KERNEL_RPP_LOCALBINARYPATTERNBATCHPD_NAME      		"org.rpp.LocalBinaryPatternbatchPD"
#define VX_KERNEL_RPP_DATAOBJECTCOPYBATCHPD_NAME      			"org.rpp.DataObjectCopybatchPD"
#define VX_KERNEL_RPP_GAUSSIANIMAGEPYRAMIDBATCHPD_NAME      	"org.rpp.GaussianImagePyramidbatchPD"
#define VX_KERNEL_RPP_LAPLACIANIMAGEPYRAMID_NAME      			"org.rpp.LaplacianImagePyramid"
#define VX_KERNEL_RPP_CANNYEDGEDETECTOR_NAME      				"org.rpp.CannyEdgeDetector"
#define VX_KERNEL_RPP_HARRISCORNERDETECTOR_NAME      			"org.rpp.HarrisCornerDetector"
#define VX_KERNEL_RPP_FASTCORNERDETECTOR_NAME      				"org.rpp.FastCornerDetector"
#define VX_KERNEL_RPP_REMAP_NAME      							"org.rpp.remap"
#define VX_KERNEL_RPP_TENSORADD_NAME      						"org.rpp.TensorAdd"
#define VX_KERNEL_RPP_TENSORSUBTRACT_NAME      					"org.rpp.TensorSubtract"
#define VX_KERNEL_RPP_TENSORMULTIPLY_NAME      					"org.rpp.TensorMultiply"
#define VX_KERNEL_RPP_TENSORMATRIXMULTIPLY_NAME      			"org.rpp.TensorMatrixMultiply"
#define VX_KERNEL_RPP_TENSORLOOKUP_NAME      					"org.rpp.TensorLookup"
#define VX_KERNEL_RPP_COLORTWISTBATCHPD_NAME        			"org.rpp.ColorTwistbatchPD"
#define VX_KERNEL_RPP_CROPMIRRORNORMALIZEBATCHPD_NAME        	"org.rpp.CropMirrorNormalizebatchPD"
#define VX_KERNEL_RPP_CROPPD_NAME   							"org.rpp.CropPD"
#define VX_KERNEL_RPP_RESIZECROPMIRRORPD_NAME      				"org.rpp.ResizeCropMirrorPD"

#endif //_AMDVX_EXT__PUBLISH_KERNELS_H_
