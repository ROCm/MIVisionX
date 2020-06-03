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

vx_status AbsoluteDifference_Register(vx_context);
vx_status AbsoluteDifferencebatchPD_Register(vx_context);
vx_status AbsoluteDifferencebatchPDROID_Register(vx_context);
vx_status AbsoluteDifferencebatchPS_Register(vx_context);
vx_status Accumulate_Register(vx_context);
vx_status AccumulatebatchPD_Register(vx_context);
vx_status AccumulatebatchPDROID_Register(vx_context);
vx_status AccumulatebatchPS_Register(vx_context);
vx_status AccumulateSquared_Register(vx_context);
vx_status AccumulateSquaredbatchPD_Register(vx_context);
vx_status AccumulateSquaredbatchPDROID_Register(vx_context);
vx_status AccumulateSquaredbatchPS_Register(vx_context);
vx_status AccumulateWeighted_Register(vx_context);
vx_status AccumulateWeightedbatchPD_Register(vx_context);
vx_status AccumulateWeightedbatchPDROID_Register(vx_context);
vx_status AccumulateWeightedbatchPS_Register(vx_context);
vx_status Add_Register(vx_context);
vx_status AddbatchPD_Register(vx_context);
vx_status AddbatchPDROID_Register(vx_context);
vx_status AddbatchPS_Register(vx_context);
vx_status BilateralFilter_Register(vx_context);
vx_status BilateralFilterbatchPD_Register(vx_context);
vx_status BilateralFilterbatchPDROID_Register(vx_context);
vx_status BilateralFilterbatchPS_Register(vx_context);
vx_status BitwiseAND_Register(vx_context);
vx_status BitwiseANDbatchPD_Register(vx_context);
vx_status BitwiseANDbatchPDROID_Register(vx_context);
vx_status BitwiseANDbatchPS_Register(vx_context);
vx_status BitwiseNOT_Register(vx_context);
vx_status BitwiseNOTbatchPD_Register(vx_context);
vx_status BitwiseNOTbatchPDROID_Register(vx_context);
vx_status BitwiseNOTbatchPS_Register(vx_context);
vx_status Blend_Register(vx_context);
vx_status BlendbatchPD_Register(vx_context);
vx_status BlendbatchPDROID_Register(vx_context);
vx_status BlendbatchPS_Register(vx_context);
vx_status Blur_Register(vx_context);
vx_status BlurbatchPD_Register(vx_context);
vx_status BlurbatchPDROID_Register(vx_context);
vx_status BlurbatchPS_Register(vx_context);
vx_status BoxFilter_Register(vx_context);
vx_status BoxFilterbatchPD_Register(vx_context);
vx_status BoxFilterbatchPDROID_Register(vx_context);
vx_status BoxFilterbatchPS_Register(vx_context);
vx_status Brightness_Register(vx_context);
vx_status BrightnessbatchPD_Register(vx_context);
vx_status BrightnessbatchPDROID_Register(vx_context);
vx_status BrightnessbatchPS_Register(vx_context);
vx_status CannyEdgeDetector_Register(vx_context);
vx_status ChannelCombine_Register(vx_context);
vx_status ChannelCombinebatchPD_Register(vx_context);
vx_status ChannelCombinebatchPS_Register(vx_context);
vx_status ChannelExtract_Register(vx_context);
vx_status ChannelExtractbatchPD_Register(vx_context);
vx_status ChannelExtractbatchPS_Register(vx_context);
vx_status ColorTemperature_Register(vx_context);
vx_status ColorTemperaturebatchPD_Register(vx_context);
vx_status ColorTemperaturebatchPDROID_Register(vx_context);
vx_status ColorTemperaturebatchPS_Register(vx_context);
vx_status ColorTwist_Register(vx_context);
vx_status ColorTwistbatchPD_Register(vx_context);
vx_status Contrast_Register(vx_context);
vx_status ContrastbatchPD_Register(vx_context);
vx_status ContrastbatchPDROID_Register(vx_context);
vx_status ContrastbatchPS_Register(vx_context);
vx_status ControlFlow_Register(vx_context);
vx_status ControlFlowbatchPD_Register(vx_context);
vx_status ControlFlowbatchPDROID_Register(vx_context);
vx_status ControlFlowbatchPS_Register(vx_context);
vx_status Copy_Register(vx_context);
vx_status CropMirrorNormalizePD_Register(vx_context);
vx_status CropPD_Register(vx_context);
vx_status CustomConvolution_Register(vx_context);
vx_status CustomConvolutionbatchPD_Register(vx_context);
vx_status CustomConvolutionbatchPDROID_Register(vx_context);
vx_status CustomConvolutionbatchPS_Register(vx_context);
vx_status DataObjectCopy_Register(vx_context);
vx_status DataObjectCopybatchPD_Register(vx_context);
vx_status DataObjectCopybatchPDROID_Register(vx_context);
vx_status DataObjectCopybatchPS_Register(vx_context);
vx_status Dilate_Register(vx_context);
vx_status DilatebatchPD_Register(vx_context);
vx_status DilatebatchPDROID_Register(vx_context);
vx_status DilatebatchPS_Register(vx_context);
vx_status Erode_Register(vx_context);
vx_status ErodebatchPD_Register(vx_context);
vx_status ErodebatchPDROID_Register(vx_context);
vx_status ErodebatchPS_Register(vx_context);
vx_status ExclusiveOR_Register(vx_context);
vx_status ExclusiveORbatchPD_Register(vx_context);
vx_status ExclusiveORbatchPDROID_Register(vx_context);
vx_status ExclusiveORbatchPS_Register(vx_context);
vx_status Exposure_Register(vx_context);
vx_status ExposurebatchPD_Register(vx_context);
vx_status ExposurebatchPDROID_Register(vx_context);
vx_status ExposurebatchPS_Register(vx_context);
vx_status FastCornerDetector_Register(vx_context);
vx_status Fisheye_Register(vx_context);
vx_status FisheyebatchPD_Register(vx_context);
vx_status FisheyebatchPDROID_Register(vx_context);
vx_status FisheyebatchPS_Register(vx_context);
vx_status Flip_Register(vx_context);
vx_status FlipbatchPD_Register(vx_context);
vx_status FlipbatchPDROID_Register(vx_context);
vx_status FlipbatchPS_Register(vx_context);
vx_status Fog_Register(vx_context);
vx_status FogbatchPD_Register(vx_context);
vx_status FogbatchPDROID_Register(vx_context);
vx_status FogbatchPS_Register(vx_context);
vx_status GammaCorrection_Register(vx_context);
vx_status GammaCorrectionbatchPD_Register(vx_context);
vx_status GammaCorrectionbatchPDROID_Register(vx_context);
vx_status GammaCorrectionbatchPS_Register(vx_context);
vx_status GaussianFilter_Register(vx_context);
vx_status GaussianFilterbatchPD_Register(vx_context);
vx_status GaussianFilterbatchPDROID_Register(vx_context);
vx_status GaussianFilterbatchPS_Register(vx_context);
vx_status GaussianImagePyramid_Register(vx_context);
vx_status GaussianImagePyramidbatchPD_Register(vx_context);
vx_status GaussianImagePyramidbatchPS_Register(vx_context);
vx_status HarrisCornerDetector_Register(vx_context);
vx_status Histogram_Register(vx_context);
vx_status HistogramBalance_Register(vx_context);
vx_status HistogramBalancebatchPD_Register(vx_context);
vx_status HistogramBalancebatchPDROID_Register(vx_context);
vx_status HistogramBalancebatchPS_Register(vx_context);
vx_status HistogramEqualize_Register(vx_context);
vx_status HistogramEqualizebatchPD_Register(vx_context);
vx_status HistogramEqualizebatchPDROID_Register(vx_context);
vx_status HistogramEqualizebatchPS_Register(vx_context);
vx_status Hue_Register(vx_context);
vx_status HuebatchPD_Register(vx_context);
vx_status HuebatchPDROID_Register(vx_context);
vx_status HuebatchPS_Register(vx_context);
vx_status InclusiveOR_Register(vx_context);
vx_status InclusiveORbatchPD_Register(vx_context);
vx_status InclusiveORbatchPDROID_Register(vx_context);
vx_status InclusiveORbatchPS_Register(vx_context);
vx_status Jitter_Register(vx_context);
vx_status JitterbatchPD_Register(vx_context);
vx_status JitterbatchPDROID_Register(vx_context);
vx_status JitterbatchPS_Register(vx_context);
vx_status LaplacianImagePyramid_Register(vx_context);
vx_status LensCorrection_Register(vx_context);
vx_status LensCorrectionbatchPD_Register(vx_context);
vx_status LensCorrectionbatchPDROID_Register(vx_context);
vx_status LensCorrectionbatchPS_Register(vx_context);
vx_status LocalBinaryPattern_Register(vx_context);
vx_status LocalBinaryPatternbatchPD_Register(vx_context);
vx_status LocalBinaryPatternbatchPDROID_Register(vx_context);
vx_status LocalBinaryPatternbatchPS_Register(vx_context);
vx_status LookUpTable_Register(vx_context);
vx_status LookUpTablebatchPD_Register(vx_context);
vx_status LookUpTablebatchPDROID_Register(vx_context);
vx_status LookUpTablebatchPS_Register(vx_context);
vx_status Magnitude_Register(vx_context);
vx_status MagnitudebatchPD_Register(vx_context);
vx_status MagnitudebatchPDROID_Register(vx_context);
vx_status MagnitudebatchPS_Register(vx_context);
vx_status Max_Register(vx_context);
vx_status MaxbatchPD_Register(vx_context);
vx_status MaxbatchPDROID_Register(vx_context);
vx_status MaxbatchPS_Register(vx_context);
vx_status MeanStddev_Register(vx_context);
vx_status MedianFilter_Register(vx_context);
vx_status MedianFilterbatchPD_Register(vx_context);
vx_status MedianFilterbatchPDROID_Register(vx_context);
vx_status MedianFilterbatchPS_Register(vx_context);
vx_status Min_Register(vx_context);
vx_status MinbatchPD_Register(vx_context);
vx_status MinbatchPDROID_Register(vx_context);
vx_status MinbatchPS_Register(vx_context);
vx_status MinMaxLoc_Register(vx_context);
vx_status Multiply_Register(vx_context);
vx_status MultiplybatchPD_Register(vx_context);
vx_status MultiplybatchPDROID_Register(vx_context);
vx_status MultiplybatchPS_Register(vx_context);
vx_status Noise_Register(vx_context);
vx_status NoisebatchPD_Register(vx_context);
vx_status NoisebatchPDROID_Register(vx_context);
vx_status NoisebatchPS_Register(vx_context);
vx_status NonLinearFilter_Register(vx_context);
vx_status NonLinearFilterbatchPD_Register(vx_context);
vx_status NonLinearFilterbatchPDROID_Register(vx_context);
vx_status NonLinearFilterbatchPS_Register(vx_context);
vx_status NonMaxSupression_Register(vx_context);
vx_status NonMaxSupressionbatchPD_Register(vx_context);
vx_status NonMaxSupressionbatchPDROID_Register(vx_context);
vx_status NonMaxSupressionbatchPS_Register(vx_context);
vx_status Nop_Register(vx_context);
vx_status Occlusion_Register(vx_context);
vx_status OcclusionbatchPD_Register(vx_context);
vx_status OcclusionbatchPDROID_Register(vx_context);
vx_status OcclusionbatchPS_Register(vx_context);
vx_status Phase_Register(vx_context);
vx_status PhasebatchPD_Register(vx_context);
vx_status PhasebatchPDROID_Register(vx_context);
vx_status PhasebatchPS_Register(vx_context);
vx_status Pixelate_Register(vx_context);
vx_status PixelatebatchPD_Register(vx_context);
vx_status PixelatebatchPDROID_Register(vx_context);
vx_status PixelatebatchPS_Register(vx_context);
vx_status Rain_Register(vx_context);
vx_status RainbatchPD_Register(vx_context);
vx_status RainbatchPDROID_Register(vx_context);
vx_status RainbatchPS_Register(vx_context);
vx_status RandomCropLetterBox_Register(vx_context);
vx_status RandomCropLetterBoxbatchPD_Register(vx_context);
vx_status RandomCropLetterBoxbatchPDROID_Register(vx_context);
vx_status RandomCropLetterBoxbatchPS_Register(vx_context);
vx_status RandomShadow_Register(vx_context);
vx_status RandomShadowbatchPD_Register(vx_context);
vx_status RandomShadowbatchPDROID_Register(vx_context);
vx_status RandomShadowbatchPS_Register(vx_context);
vx_status remap_Register(vx_context);
vx_status Resize_Register(vx_context);
vx_status ResizebatchPD_Register(vx_context);
vx_status ResizebatchPDROID_Register(vx_context);
vx_status ResizebatchPS_Register(vx_context);
vx_status ResizeCrop_Register(vx_context);
vx_status ResizeCropbatchPD_Register(vx_context);
vx_status ResizeCropbatchPDROID_Register(vx_context);
vx_status ResizeCropbatchPS_Register(vx_context);
vx_status ResizeCropMirrorPD_Register(vx_context);
vx_status ResizeCropMirrorPD(vx_context);
vx_status Rotate_Register(vx_context);
vx_status RotatebatchPD_Register(vx_context);
vx_status RotatebatchPDROID_Register(vx_context);
vx_status RotatebatchPS_Register(vx_context);
vx_status Saturation_Register(vx_context);
vx_status SaturationbatchPD_Register(vx_context);
vx_status SaturationbatchPDROID_Register(vx_context);
vx_status SaturationbatchPS_Register(vx_context);
vx_status Scale_Register(vx_context);
vx_status ScalebatchPD_Register(vx_context);
vx_status ScalebatchPDROID_Register(vx_context);
vx_status ScalebatchPS_Register(vx_context);
vx_status Snow_Register(vx_context);
vx_status SnowbatchPD_Register(vx_context);
vx_status SnowbatchPDROID_Register(vx_context);
vx_status SnowbatchPS_Register(vx_context);
vx_status Sobel_Register(vx_context);
vx_status SobelbatchPD_Register(vx_context);
vx_status SobelbatchPDROID_Register(vx_context);
vx_status SobelbatchPS_Register(vx_context);
vx_status Subtract_Register(vx_context);
vx_status SubtractbatchPD_Register(vx_context);
vx_status SubtractbatchPDROID_Register(vx_context);
vx_status SubtractbatchPS_Register(vx_context);
vx_status TensorAdd_Register(vx_context);
vx_status TensorLookup_Register(vx_context);
vx_status TensorMatrixMultiply_Register(vx_context);
vx_status TensorMultiply_Register(vx_context);
vx_status TensorSubtract_Register(vx_context);
vx_status Thresholding_Register(vx_context);
vx_status ThresholdingbatchPD_Register(vx_context);
vx_status ThresholdingbatchPDROID_Register(vx_context);
vx_status ThresholdingbatchPS_Register(vx_context);
vx_status Vignette_Register(vx_context);
vx_status VignettebatchPD_Register(vx_context);
vx_status VignettebatchPDROID_Register(vx_context);
vx_status VignettebatchPS_Register(vx_context);
vx_status WarpAffine_Register(vx_context);
vx_status WarpAffinebatchPD_Register(vx_context);
vx_status WarpAffinebatchPDROID_Register(vx_context);
vx_status WarpAffinebatchPS_Register(vx_context);
vx_status WarpPerspective_Register(vx_context);
vx_status WarpPerspectivebatchPD_Register(vx_context);
vx_status WarpPerspectivebatchPDROID_Register(vx_context);
vx_status WarpPerspectivebatchPS_Register(vx_context);

// kernel names
#define VX_KERNEL_RPP_NOP_NAME                          "org.rpp.Nop"
#define VX_KERNEL_RPP_COPY_NAME                         "org.rpp.Copy"
#define VX_KERNEL_RPP_BRIGHTNESS_NAME      				"org.rpp.Brightness"
#define VX_KERNEL_RPP_BRIGHTNESSBATCHPS_NAME      		"org.rpp.BrightnessbatchPS"
#define VX_KERNEL_RPP_BRIGHTNESSBATCHPD_NAME      		"org.rpp.BrightnessbatchPD"
#define VX_KERNEL_RPP_BRIGHTNESSBATCHPDROID_NAME      	"org.rpp.BrightnessbatchPDROID"
#define VX_KERNEL_RPP_GAMMACORRECTION_NAME      		"org.rpp.GammaCorrection"
#define VX_KERNEL_RPP_GAMMACORRECTIONBATCHPS_NAME      	"org.rpp.GammaCorrectionbatchPS"
#define VX_KERNEL_RPP_GAMMACORRECTIONBATCHPD_NAME      	"org.rpp.GammaCorrectionbatchPD"
#define VX_KERNEL_RPP_GAMMACORRECTIONBATCHPDROID_NAME   "org.rpp.GammaCorrectionbatchPDROID"
#define VX_KERNEL_RPP_BLEND_NAME      					"org.rpp.Blend"
#define VX_KERNEL_RPP_BLENDBATCHPS_NAME      			"org.rpp.BlendbatchPS"
#define VX_KERNEL_RPP_BLENDBATCHPD_NAME      			"org.rpp.BlendbatchPD"
#define VX_KERNEL_RPP_BLENDBATCHPDROID_NAME      		"org.rpp.BlendbatchPDROID"
#define VX_KERNEL_RPP_BLUR_NAME      					"org.rpp.Blur"
#define VX_KERNEL_RPP_BLURBATCHPS_NAME      			"org.rpp.BlurbatchPS"
#define VX_KERNEL_RPP_BLURBATCHPD_NAME      			"org.rpp.BlurbatchPD"
#define VX_KERNEL_RPP_BLURBATCHPDROID_NAME      		"org.rpp.BlurbatchPDROID"
#define VX_KERNEL_RPP_CONTRAST_NAME      				"org.rpp.Contrast"
#define VX_KERNEL_RPP_CONTRASTBATCHPS_NAME      		"org.rpp.ContrastbatchPS"
#define VX_KERNEL_RPP_CONTRASTBATCHPD_NAME      		"org.rpp.ContrastbatchPD"
#define VX_KERNEL_RPP_CONTRASTBATCHPDROID_NAME      	"org.rpp.ContrastbatchPDROID"
#define VX_KERNEL_RPP_PIXELATE_NAME      				"org.rpp.Pixelate"
#define VX_KERNEL_RPP_PIXELATEBATCHPS_NAME      		"org.rpp.PixelatebatchPS"
#define VX_KERNEL_RPP_PIXELATEBATCHPD_NAME      		"org.rpp.PixelatebatchPD"
#define VX_KERNEL_RPP_PIXELATEBATCHPDROID_NAME      	"org.rpp.PixelatebatchPDROID"
#define VX_KERNEL_RPP_JITTER_NAME      					"org.rpp.Jitter"
#define VX_KERNEL_RPP_JITTERBATCHPS_NAME      			"org.rpp.JitterbatchPS"
#define VX_KERNEL_RPP_JITTERBATCHPD_NAME      			"org.rpp.JitterbatchPD"
#define VX_KERNEL_RPP_JITTERBATCHPDROID_NAME      		"org.rpp.JitterbatchPDROID"
#define VX_KERNEL_RPP_OCCLUSION_NAME      				"org.rpp.Occlusion"
#define VX_KERNEL_RPP_OCCLUSIONBATCHPS_NAME      		"org.rpp.OcclusionbatchPS"
#define VX_KERNEL_RPP_OCCLUSIONBATCHPD_NAME      		"org.rpp.OcclusionbatchPD"
#define VX_KERNEL_RPP_OCCLUSIONBATCHPDROID_NAME      	"org.rpp.OcclusionbatchPDROID"
#define VX_KERNEL_RPP_SNOW_NAME      					"org.rpp.Snow"
#define VX_KERNEL_RPP_SNOWBATCHPS_NAME      			"org.rpp.SnowbatchPS"
#define VX_KERNEL_RPP_SNOWBATCHPD_NAME      			"org.rpp.SnowbatchPD"
#define VX_KERNEL_RPP_SNOWBATCHPDROID_NAME      		"org.rpp.SnowbatchPDROID"
#define VX_KERNEL_RPP_NOISE_NAME      					"org.rpp.Noise"
#define VX_KERNEL_RPP_NOISEBATCHPS_NAME      			"org.rpp.NoisebatchPS"
#define VX_KERNEL_RPP_NOISEBATCHPD_NAME      			"org.rpp.NoisebatchPD"
#define VX_KERNEL_RPP_NOISEBATCHPDROID_NAME      		"org.rpp.NoisebatchPDROID"
#define VX_KERNEL_RPP_RANDOMSHADOW_NAME      			"org.rpp.RandomShadow"
#define VX_KERNEL_RPP_RANDOMSHADOWBATCHPS_NAME      	"org.rpp.RandomShadowbatchPS"
#define VX_KERNEL_RPP_RANDOMSHADOWBATCHPD_NAME      	"org.rpp.RandomShadowbatchPD"
#define VX_KERNEL_RPP_RANDOMSHADOWBATCHPDROID_NAME      "org.rpp.RandomShadowbatchPDROID"
#define VX_KERNEL_RPP_FOG_NAME      					"org.rpp.Fog"
#define VX_KERNEL_RPP_FOGBATCHPS_NAME      				"org.rpp.FogbatchPS"
#define VX_KERNEL_RPP_FOGBATCHPD_NAME      				"org.rpp.FogbatchPD"
#define VX_KERNEL_RPP_FOGBATCHPDROID_NAME      			"org.rpp.FogbatchPDROID"
#define VX_KERNEL_RPP_RAIN_NAME      					"org.rpp.Rain"
#define VX_KERNEL_RPP_RAINBATCHPS_NAME      			"org.rpp.RainbatchPS"
#define VX_KERNEL_RPP_RAINBATCHPD_NAME      			"org.rpp.RainbatchPD"
#define VX_KERNEL_RPP_RAINBATCHPDROID_NAME      		"org.rpp.RainbatchPDROID"
#define VX_KERNEL_RPP_RANDOMCROPLETTERBOX_NAME      	"org.rpp.RandomCropLetterBox"
#define VX_KERNEL_RPP_RANDOMCROPLETTERBOXBATCHPS_NAME      "org.rpp.RandomCropLetterBoxbatchPS"
#define VX_KERNEL_RPP_RANDOMCROPLETTERBOXBATCHPD_NAME      "org.rpp.RandomCropLetterBoxbatchPD"
#define VX_KERNEL_RPP_RANDOMCROPLETTERBOXBATCHPDROID_NAME  "org.rpp.RandomCropLetterBoxbatchPDROID"
#define VX_KERNEL_RPP_EXPOSURE_NAME      				"org.rpp.Exposure"
#define VX_KERNEL_RPP_EXPOSUREBATCHPS_NAME      		"org.rpp.ExposurebatchPS"
#define VX_KERNEL_RPP_EXPOSUREBATCHPD_NAME      		"org.rpp.ExposurebatchPD"
#define VX_KERNEL_RPP_EXPOSUREBATCHPDROID_NAME      	"org.rpp.ExposurebatchPDROID"
#define VX_KERNEL_RPP_HISTOGRAMBALANCE_NAME      		"org.rpp.HistogramBalance"
#define VX_KERNEL_RPP_HISTOGRAMBALANCEBATCHPS_NAME      "org.rpp.HistogramBalancebatchPS"
#define VX_KERNEL_RPP_HISTOGRAMBALANCEBATCHPD_NAME      "org.rpp.HistogramBalancebatchPD"
#define VX_KERNEL_RPP_HISTOGRAMBALANCEBATCHPDROID_NAME  "org.rpp.HistogramBalancebatchPDROID"
#define VX_KERNEL_RPP_ABSOLUTEDIFFERENCE_NAME      		"org.rpp.AbsoluteDifference"
#define VX_KERNEL_RPP_ABSOLUTEDIFFERENCEBATCHPS_NAME    "org.rpp.AbsoluteDifferencebatchPS"
#define VX_KERNEL_RPP_ABSOLUTEDIFFERENCEBATCHPD_NAME    "org.rpp.AbsoluteDifferencebatchPD"
#define VX_KERNEL_RPP_ABSOLUTEDIFFERENCEBATCHPDROID_NAME	"org.rpp.AbsoluteDifferencebatchPDROID"
#define VX_KERNEL_RPP_ACCUMULATEWEIGHTED_NAME      		"org.rpp.AccumulateWeighted"
#define VX_KERNEL_RPP_ACCUMULATEWEIGHTEDBATCHPS_NAME    "org.rpp.AccumulateWeightedbatchPS"
#define VX_KERNEL_RPP_ACCUMULATEWEIGHTEDBATCHPD_NAME    "org.rpp.AccumulateWeightedbatchPD"
#define VX_KERNEL_RPP_ACCUMULATEWEIGHTEDBATCHPDROID_NAME  	"org.rpp.AccumulateWeightedbatchPDROID"
#define VX_KERNEL_RPP_ACCUMULATE_NAME      				"org.rpp.Accumulate"
#define VX_KERNEL_RPP_ACCUMULATEBATCHPS_NAME      		"org.rpp.AccumulatebatchPS"
#define VX_KERNEL_RPP_ACCUMULATEBATCHPD_NAME      		"org.rpp.AccumulatebatchPD"
#define VX_KERNEL_RPP_ACCUMULATEBATCHPDROID_NAME      	"org.rpp.AccumulatebatchPDROID"
#define VX_KERNEL_RPP_ADD_NAME     						"org.rpp.Add"
#define VX_KERNEL_RPP_ADDBATCHPS_NAME      				"org.rpp.AddbatchPS"
#define VX_KERNEL_RPP_ADDBATCHPD_NAME      				"org.rpp.AddbatchPD"
#define VX_KERNEL_RPP_ADDBATCHPDROID_NAME      			"org.rpp.AddbatchPDROID"
#define VX_KERNEL_RPP_SUBTRACT_NAME      				"org.rpp.Subtract"
#define VX_KERNEL_RPP_SUBTRACTBATCHPS_NAME      		"org.rpp.SubtractbatchPS"
#define VX_KERNEL_RPP_SUBTRACTBATCHPD_NAME      		"org.rpp.SubtractbatchPD"
#define VX_KERNEL_RPP_SUBTRACTBATCHPDROID_NAME      	"org.rpp.SubtractbatchPDROID"
#define VX_KERNEL_RPP_MAGNITUDE_NAME      						"org.rpp.Magnitude"
#define VX_KERNEL_RPP_MAGNITUDEBATCHPS_NAME      				"org.rpp.MagnitudebatchPS"
#define VX_KERNEL_RPP_MAGNITUDEBATCHPD_NAME      				"org.rpp.MagnitudebatchPD"
#define VX_KERNEL_RPP_MAGNITUDEBATCHPDROID_NAME      			"org.rpp.MagnitudebatchPDROID"
#define VX_KERNEL_RPP_MULTIPLY_NAME      						"org.rpp.Multiply"
#define VX_KERNEL_RPP_MULTIPLYBATCHPS_NAME      				"org.rpp.MultiplybatchPS"
#define VX_KERNEL_RPP_MULTIPLYBATCHPD_NAME      				"org.rpp.MultiplybatchPD"
#define VX_KERNEL_RPP_MULTIPLYBATCHPDROID_NAME      			"org.rpp.MultiplybatchPDROID"
#define VX_KERNEL_RPP_PHASE_NAME      							"org.rpp.Phase"
#define VX_KERNEL_RPP_PHASEBATCHPS_NAME      					"org.rpp.PhasebatchPS"
#define VX_KERNEL_RPP_PHASEBATCHPD_NAME      					"org.rpp.PhasebatchPD"
#define VX_KERNEL_RPP_PHASEBATCHPDROID_NAME      				"org.rpp.PhasebatchPDROID"
#define VX_KERNEL_RPP_ACCUMULATESQUARED_NAME      				"org.rpp.AccumulateSquared"
#define VX_KERNEL_RPP_ACCUMULATESQUAREDBATCHPS_NAME      		"org.rpp.AccumulateSquaredbatchPS"
#define VX_KERNEL_RPP_ACCUMULATESQUAREDBATCHPD_NAME      		"org.rpp.AccumulateSquaredbatchPD"
#define VX_KERNEL_RPP_ACCUMULATESQUAREDBATCHPDROID_NAME      	"org.rpp.AccumulateSquaredbatchPDROID"
#define VX_KERNEL_RPP_BITWISEAND_NAME      						"org.rpp.BitwiseAND"
#define VX_KERNEL_RPP_BITWISEANDBATCHPS_NAME      				"org.rpp.BitwiseANDbatchPS"
#define VX_KERNEL_RPP_BITWISEANDBATCHPD_NAME      				"org.rpp.BitwiseANDbatchPD"
#define VX_KERNEL_RPP_BITWISEANDBATCHPDROID_NAME      			"org.rpp.BitwiseANDbatchPDROID"
#define VX_KERNEL_RPP_BITWISENOT_NAME      						"org.rpp.BitwiseNOT"
#define VX_KERNEL_RPP_BITWISENOTBATCHPS_NAME      				"org.rpp.BitwiseNOTbatchPS"
#define VX_KERNEL_RPP_BITWISENOTBATCHPD_NAME      				"org.rpp.BitwiseNOTbatchPD"
#define VX_KERNEL_RPP_BITWISENOTBATCHPDROID_NAME      			"org.rpp.BitwiseNOTbatchPDROID"
#define VX_KERNEL_RPP_EXCLUSIVEOR_NAME      					"org.rpp.ExclusiveOR"
#define VX_KERNEL_RPP_EXCLUSIVEORBATCHPS_NAME      				"org.rpp.ExclusiveORbatchPS"
#define VX_KERNEL_RPP_EXCLUSIVEORBATCHPD_NAME      				"org.rpp.ExclusiveORbatchPD"
#define VX_KERNEL_RPP_EXCLUSIVEORBATCHPDROID_NAME      			"org.rpp.ExclusiveORbatchPDROID"
#define VX_KERNEL_RPP_INCLUSIVEOR_NAME      					"org.rpp.InclusiveOR"
#define VX_KERNEL_RPP_INCLUSIVEORBATCHPS_NAME      				"org.rpp.InclusiveORbatchPS"
#define VX_KERNEL_RPP_INCLUSIVEORBATCHPD_NAME      				"org.rpp.InclusiveORbatchPD"
#define VX_KERNEL_RPP_INCLUSIVEORBATCHPDROID_NAME      			"org.rpp.InclusiveORbatchPDROID"
#define VX_KERNEL_RPP_HISTOGRAM_NAME      						"org.rpp.Histogram"
#define VX_KERNEL_RPP_THRESHOLDING_NAME      					"org.rpp.Thresholding"
#define VX_KERNEL_RPP_THRESHOLDINGBATCHPS_NAME      			"org.rpp.ThresholdingbatchPS"
#define VX_KERNEL_RPP_THRESHOLDINGBATCHPD_NAME      			"org.rpp.ThresholdingbatchPD"
#define VX_KERNEL_RPP_THRESHOLDINGBATCHPDROID_NAME      		"org.rpp.ThresholdingbatchPDROID"
#define VX_KERNEL_RPP_MAX_NAME      							"org.rpp.Max"
#define VX_KERNEL_RPP_MAXBATCHPS_NAME      						"org.rpp.MaxbatchPS"
#define VX_KERNEL_RPP_MAXBATCHPD_NAME      						"org.rpp.MaxbatchPD"
#define VX_KERNEL_RPP_MAXBATCHPDROID_NAME      					"org.rpp.MaxbatchPDROID"
#define VX_KERNEL_RPP_MIN_NAME      							"org.rpp.Min"
#define VX_KERNEL_RPP_MINBATCHPS_NAME      						"org.rpp.MinbatchPS"
#define VX_KERNEL_RPP_MINBATCHPD_NAME      						"org.rpp.MinbatchPD"
#define VX_KERNEL_RPP_MINBATCHPDROID_NAME      					"org.rpp.MinbatchPDROID"
#define VX_KERNEL_RPP_MINMAXLOC_NAME      						"org.rpp.MinMaxLoc"
#define VX_KERNEL_RPP_HISTOGRAMEQUALIZE_NAME      				"org.rpp.HistogramEqualize"
#define VX_KERNEL_RPP_HISTOGRAMEQUALIZEBATCHPS_NAME      		"org.rpp.HistogramEqualizebatchPS"
#define VX_KERNEL_RPP_HISTOGRAMEQUALIZEBATCHPD_NAME      		"org.rpp.HistogramEqualizebatchPD"
#define VX_KERNEL_RPP_HISTOGRAMEQUALIZEBATCHPDROID_NAME      	"org.rpp.HistogramEqualizebatchPDROID"
#define VX_KERNEL_RPP_MEANSTDDEV_NAME     	 					"org.rpp.MeanStddev"
#define VX_KERNEL_RPP_FLIP_NAME      							"org.rpp.Flip"
#define VX_KERNEL_RPP_FLIPBATCHPS_NAME     	 					"org.rpp.FlipbatchPS"
#define VX_KERNEL_RPP_FLIPBATCHPD_NAME      					"org.rpp.FlipbatchPD"
#define VX_KERNEL_RPP_FLIPBATCHPDROID_NAME      				"org.rpp.FlipbatchPDROID"
#define VX_KERNEL_RPP_RESIZE_NAME      							"org.rpp.Resize"
#define VX_KERNEL_RPP_RESIZEBATCHPS_NAME      					"org.rpp.ResizebatchPS"
#define VX_KERNEL_RPP_RESIZEBATCHPD_NAME      					"org.rpp.ResizebatchPD"
#define VX_KERNEL_RPP_RESIZEBATCHPDROID_NAME      				"org.rpp.ResizebatchPDROID"
#define VX_KERNEL_RPP_RESIZECROP_NAME      						"org.rpp.ResizeCrop"
#define VX_KERNEL_RPP_RESIZECROPBATCHPS_NAME      				"org.rpp.ResizeCropbatchPS"
#define VX_KERNEL_RPP_RESIZECROPBATCHPD_NAME      				"org.rpp.ResizeCropbatchPD"
#define VX_KERNEL_RPP_RESIZECROPBATCHPDROID_NAME      			"org.rpp.ResizeCropbatchPDROID"
#define VX_KERNEL_RPP_ROTATE_NAME      							"org.rpp.Rotate"
#define VX_KERNEL_RPP_ROTATEBATCHPS_NAME      					"org.rpp.RotatebatchPS"
#define VX_KERNEL_RPP_ROTATEBATCHPD_NAME      					"org.rpp.RotatebatchPD"
#define VX_KERNEL_RPP_ROTATEBATCHPDROID_NAME      				"org.rpp.RotatebatchPDROID"
#define VX_KERNEL_RPP_WARPAFFINE_NAME      						"org.rpp.WarpAffine"
#define VX_KERNEL_RPP_WARPAFFINEBATCHPS_NAME      				"org.rpp.WarpAffinebatchPS"
#define VX_KERNEL_RPP_WARPAFFINEBATCHPD_NAME      				"org.rpp.WarpAffinebatchPD"
#define VX_KERNEL_RPP_WARPAFFINEBATCHPDROID_NAME      			"org.rpp.WarpAffinebatchPDROID"
#define VX_KERNEL_RPP_FISHEYE_NAME      						"org.rpp.Fisheye"
#define VX_KERNEL_RPP_FISHEYEBATCHPS_NAME      					"org.rpp.FisheyebatchPS"
#define VX_KERNEL_RPP_FISHEYEBATCHPD_NAME      					"org.rpp.FisheyebatchPD"
#define VX_KERNEL_RPP_FISHEYEBATCHPDROID_NAME      				"org.rpp.FisheyebatchPDROID"
#define VX_KERNEL_RPP_LENSCORRECTION_NAME      					"org.rpp.LensCorrection"
#define VX_KERNEL_RPP_LENSCORRECTIONBATCHPS_NAME      			"org.rpp.LensCorrectionbatchPS"
#define VX_KERNEL_RPP_LENSCORRECTIONBATCHPD_NAME      			"org.rpp.LensCorrectionbatchPD"
#define VX_KERNEL_RPP_LENSCORRECTIONBATCHPDROID_NAME      		"org.rpp.LensCorrectionbatchPDROID"
#define VX_KERNEL_RPP_SCALE_NAME      							"org.rpp.Scale"
#define VX_KERNEL_RPP_SCALEBATCHPS_NAME      					"org.rpp.ScalebatchPS"
#define VX_KERNEL_RPP_SCALEBATCHPD_NAME      					"org.rpp.ScalebatchPD"
#define VX_KERNEL_RPP_SCALEBATCHPDROID_NAME      				"org.rpp.ScalebatchPDROID"
#define VX_KERNEL_RPP_WARPPERSPECTIVE_NAME      				"org.rpp.WarpPerspective"
#define VX_KERNEL_RPP_WARPPERSPECTIVEBATCHPS_NAME      			"org.rpp.WarpPerspectivebatchPS"
#define VX_KERNEL_RPP_WARPPERSPECTIVEBATCHPD_NAME      			"org.rpp.WarpPerspectivebatchPD"
#define VX_KERNEL_RPP_WARPPERSPECTIVEBATCHPDROID_NAME      		"org.rpp.WarpPerspectivebatchPDROID"
#define VX_KERNEL_RPP_DILATE_NAME      							"org.rpp.Dilate"
#define VX_KERNEL_RPP_DILATEBATCHPS_NAME      					"org.rpp.DilatebatchPS"
#define VX_KERNEL_RPP_DILATEBATCHPD_NAME      					"org.rpp.DilatebatchPD"
#define VX_KERNEL_RPP_DILATEBATCHPDROID_NAME      				"org.rpp.DilatebatchPDROID"
#define VX_KERNEL_RPP_ERODE_NAME      							"org.rpp.Erode"
#define VX_KERNEL_RPP_ERODEBATCHPS_NAME      					"org.rpp.ErodebatchPS"
#define VX_KERNEL_RPP_ERODEBATCHPD_NAME      					"org.rpp.ErodebatchPD"
#define VX_KERNEL_RPP_ERODEBATCHPDROID_NAME      				"org.rpp.ErodebatchPDROID"
#define VX_KERNEL_RPP_HUE_NAME      							"org.rpp.Hue"
#define VX_KERNEL_RPP_HUEBATCHPS_NAME      						"org.rpp.HuebatchPS"
#define VX_KERNEL_RPP_HUEBATCHPD_NAME      						"org.rpp.HuebatchPD"
#define VX_KERNEL_RPP_HUEBATCHPDROID_NAME      					"org.rpp.HuebatchPDROID"
#define VX_KERNEL_RPP_SATURATION_NAME      						"org.rpp.Saturation"
#define VX_KERNEL_RPP_SATURATIONBATCHPS_NAME      				"org.rpp.SaturationbatchPS"
#define VX_KERNEL_RPP_SATURATIONBATCHPD_NAME      				"org.rpp.SaturationbatchPD"
#define VX_KERNEL_RPP_SATURATIONBATCHPDROID_NAME      			"org.rpp.SaturationbatchPDROID"
#define VX_KERNEL_RPP_COLORTEMPERATURE_NAME      				"org.rpp.ColorTemperature"
#define VX_KERNEL_RPP_COLORTEMPERATUREBATCHPS_NAME      		"org.rpp.ColorTemperaturebatchPS"
#define VX_KERNEL_RPP_COLORTEMPERATUREBATCHPD_NAME      		"org.rpp.ColorTemperaturebatchPD"
#define VX_KERNEL_RPP_COLORTEMPERATUREBATCHPDROID_NAME      	"org.rpp.ColorTemperaturebatchPDROID"
#define VX_KERNEL_RPP_VIGNETTE_NAME      						"org.rpp.Vignette"
#define VX_KERNEL_RPP_VIGNETTEBATCHPS_NAME      				"org.rpp.VignettebatchPS"
#define VX_KERNEL_RPP_VIGNETTEBATCHPD_NAME      				"org.rpp.VignettebatchPD"
#define VX_KERNEL_RPP_VIGNETTEBATCHPDROID_NAME      			"org.rpp.VignettebatchPDROID"
#define VX_KERNEL_RPP_CHANNELEXTRACT_NAME      					"org.rpp.ChannelExtract"
#define VX_KERNEL_RPP_CHANNELEXTRACTBATCHPS_NAME      			"org.rpp.ChannelExtractbatchPS"
#define VX_KERNEL_RPP_CHANNELEXTRACTBATCHPD_NAME      			"org.rpp.ChannelExtractbatchPD"
#define VX_KERNEL_RPP_CHANNELCOMBINE_NAME      					"org.rpp.ChannelCombine"
#define VX_KERNEL_RPP_CHANNELCOMBINEBATCHPS_NAME      			"org.rpp.ChannelCombinebatchPS"
#define VX_KERNEL_RPP_CHANNELCOMBINEBATCHPD_NAME      			"org.rpp.ChannelCombinebatchPD"
#define VX_KERNEL_RPP_LOOKUPTABLE_NAME      					"org.rpp.LookUpTable"
#define VX_KERNEL_RPP_LOOKUPTABLEBATCHPS_NAME      				"org.rpp.LookUpTablebatchPS"
#define VX_KERNEL_RPP_LOOKUPTABLEBATCHPD_NAME      				"org.rpp.LookUpTablebatchPD"
#define VX_KERNEL_RPP_LOOKUPTABLEBATCHPDROID_NAME      			"org.rpp.LookUpTablebatchPDROID"
#define VX_KERNEL_RPP_BILATERALFILTER_NAME      				"org.rpp.BilateralFilter"
#define VX_KERNEL_RPP_BILATERALFILTERBATCHPS_NAME      			"org.rpp.BilateralFilterbatchPS"
#define VX_KERNEL_RPP_BILATERALFILTERBATCHPD_NAME      			"org.rpp.BilateralFilterbatchPD"
#define VX_KERNEL_RPP_BILATERALFILTERBATCHPDROID_NAME      		"org.rpp.BilateralFilterbatchPDROID"
#define VX_KERNEL_RPP_BOXFILTER_NAME      						"org.rpp.BoxFilter"
#define VX_KERNEL_RPP_BOXFILTERBATCHPS_NAME      				"org.rpp.BoxFilterbatchPS"
#define VX_KERNEL_RPP_BOXFILTERBATCHPD_NAME      				"org.rpp.BoxFilterbatchPD"
#define VX_KERNEL_RPP_BOXFILTERBATCHPDROID_NAME      			"org.rpp.BoxFilterbatchPDROID"
#define VX_KERNEL_RPP_SOBEL_NAME      							"org.rpp.Sobel"
#define VX_KERNEL_RPP_SOBELBATCHPS_NAME      					"org.rpp.SobelbatchPS"
#define VX_KERNEL_RPP_SOBELBATCHPD_NAME      					"org.rpp.SobelbatchPD"
#define VX_KERNEL_RPP_SOBELBATCHPDROID_NAME      				"org.rpp.SobelbatchPDROID"
#define VX_KERNEL_RPP_MEDIANFILTER_NAME      					"org.rpp.MedianFilter"
#define VX_KERNEL_RPP_MEDIANFILTERBATCHPS_NAME      			"org.rpp.MedianFilterbatchPS"
#define VX_KERNEL_RPP_MEDIANFILTERBATCHPD_NAME      			"org.rpp.MedianFilterbatchPD"
#define VX_KERNEL_RPP_MEDIANFILTERBATCHPDROID_NAME      		"org.rpp.MedianFilterbatchPDROID"
#define VX_KERNEL_RPP_CUSTOMCONVOLUTION_NAME      				"org.rpp.CustomConvolution"
#define VX_KERNEL_RPP_CUSTOMCONVOLUTIONBATCHPS_NAME      		"org.rpp.CustomConvolutionbatchPS"
#define VX_KERNEL_RPP_CUSTOMCONVOLUTIONBATCHPD_NAME      		"org.rpp.CustomConvolutionbatchPD"
#define VX_KERNEL_RPP_CUSTOMCONVOLUTIONBATCHPDROID_NAME      	"org.rpp.CustomConvolutionbatchPDROID"
#define VX_KERNEL_RPP_NONMAXSUPRESSION_NAME      				"org.rpp.NonMaxSupression"
#define VX_KERNEL_RPP_NONMAXSUPRESSIONBATCHPS_NAME      		"org.rpp.NonMaxSupressionbatchPS"
#define VX_KERNEL_RPP_NONMAXSUPRESSIONBATCHPD_NAME      		"org.rpp.NonMaxSupressionbatchPD"
#define VX_KERNEL_RPP_NONMAXSUPRESSIONBATCHPDROID_NAME      	"org.rpp.NonMaxSupressionbatchPDROID"
#define VX_KERNEL_RPP_GAUSSIANFILTER_NAME      					"org.rpp.GaussianFilter"
#define VX_KERNEL_RPP_GAUSSIANFILTERBATCHPS_NAME      			"org.rpp.GaussianFilterbatchPS"
#define VX_KERNEL_RPP_GAUSSIANFILTERBATCHPD_NAME      			"org.rpp.GaussianFilterbatchPD"
#define VX_KERNEL_RPP_GAUSSIANFILTERBATCHPDROID_NAME     	 	"org.rpp.GaussianFilterbatchPDROID"
#define VX_KERNEL_RPP_NONLINEARFILTER_NAME      				"org.rpp.NonLinearFilter"
#define VX_KERNEL_RPP_NONLINEARFILTERBATCHPS_NAME      			"org.rpp.NonLinearFilterbatchPS"
#define VX_KERNEL_RPP_NONLINEARFILTERBATCHPD_NAME      			"org.rpp.NonLinearFilterbatchPD"
#define VX_KERNEL_RPP_NONLINEARFILTERBATCHPDROID_NAME      		"org.rpp.NonLinearFilterbatchPDROID"
#define VX_KERNEL_RPP_LOCALBINARYPATTERN_NAME      				"org.rpp.LocalBinaryPattern"
#define VX_KERNEL_RPP_LOCALBINARYPATTERNBATCHPS_NAME      		"org.rpp.LocalBinaryPatternbatchPS"
#define VX_KERNEL_RPP_LOCALBINARYPATTERNBATCHPD_NAME      		"org.rpp.LocalBinaryPatternbatchPD"
#define VX_KERNEL_RPP_LOCALBINARYPATTERNBATCHPDROID_NAME      	"org.rpp.LocalBinaryPatternbatchPDROID"
#define VX_KERNEL_RPP_DATAOBJECTCOPY_NAME      					"org.rpp.DataObjectCopy"
#define VX_KERNEL_RPP_DATAOBJECTCOPYBATCHPS_NAME      			"org.rpp.DataObjectCopybatchPS"
#define VX_KERNEL_RPP_DATAOBJECTCOPYBATCHPD_NAME      			"org.rpp.DataObjectCopybatchPD"
#define VX_KERNEL_RPP_DATAOBJECTCOPYBATCHPDROID_NAME      		"org.rpp.DataObjectCopybatchPDROID"
#define VX_KERNEL_RPP_GAUSSIANIMAGEPYRAMID_NAME      			"org.rpp.GaussianImagePyramid"
#define VX_KERNEL_RPP_GAUSSIANIMAGEPYRAMIDBATCHPS_NAME      	"org.rpp.GaussianImagePyramidbatchPS"
#define VX_KERNEL_RPP_GAUSSIANIMAGEPYRAMIDBATCHPD_NAME      	"org.rpp.GaussianImagePyramidbatchPD"
#define VX_KERNEL_RPP_LAPLACIANIMAGEPYRAMID_NAME      			"org.rpp.LaplacianImagePyramid"
#define VX_KERNEL_RPP_CANNYEDGEDETECTOR_NAME      				"org.rpp.CannyEdgeDetector"
#define VX_KERNEL_RPP_HARRISCORNERDETECTOR_NAME      			"org.rpp.HarrisCornerDetector"
#define VX_KERNEL_RPP_FASTCORNERDETECTOR_NAME      				"org.rpp.FastCornerDetector"
#define VX_KERNEL_RPP_CONTROLFLOW_NAME      					"org.rpp.ControlFlow"
#define VX_KERNEL_RPP_CONTROLFLOWBATCHPS_NAME     				"org.rpp.ControlFlowbatchPS"
#define VX_KERNEL_RPP_CONTROLFLOWBATCHPD_NAME      				"org.rpp.ControlFlowbatchPD"
#define VX_KERNEL_RPP_CONTROLFLOWBATCHPDROID_NAME      			"org.rpp.ControlFlowbatchPDROID"
#define VX_KERNEL_RPP_REMAP_NAME      							"org.rpp.remap"
#define VX_KERNEL_RPP_TENSORADD_NAME      						"org.rpp.TensorAdd"
#define VX_KERNEL_RPP_TENSORSUBTRACT_NAME      					"org.rpp.TensorSubtract"
#define VX_KERNEL_RPP_TENSORMULTIPLY_NAME      					"org.rpp.TensorMultiply"
#define VX_KERNEL_RPP_TENSORMATRIXMULTIPLY_NAME      			"org.rpp.TensorMatrixMultiply"
#define VX_KERNEL_RPP_TENSORLOOKUP_NAME      					"org.rpp.TensorLookup"
#define VX_KERNEL_RPP_COLORTWIST_NAME               			"org.rpp.ColorTwist"
#define VX_KERNEL_RPP_COLORTWISTBATCHPD_NAME        			"org.rpp.ColorTwistbatchPD"
#define VX_KERNEL_RPP_CROPMIRRORNORMALIZEBATCHPD_NAME        	"org.rpp.CropMirrorNormalizebatchPD"
#define VX_KERNEL_RPP_CROPPD_NAME   							"org.rpp.CropPD"
#define VX_KERNEL_RPP_RESIZECROPMIRRORPD_NAME      				"org.rpp.ResizeCropMirrorPD"
// #define VX_KERNEL_RPP_COLORTWISTBATCHPS_NAME        "org.rpp.ColorTwistPS"
// #define VX_KERNEL_RPP_COLORTWISTBATCHPDROID_NAME    "org.rpp.ColorTwistPDROID"

#endif //_AMDVX_EXT__PUBLISH_KERNELS_H_
