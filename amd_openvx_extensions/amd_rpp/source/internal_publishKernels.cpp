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

#include "internal_publishKernels.h"
#include "vx_ext_rpp.h"

/**********************************************************************
  PUBLIC FUNCTION for OpenVX user defined functions
**********************************************************************/
extern "C"  SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context)
{
	vx_status status = VX_SUCCESS;

	STATUS_ERROR_CHECK(get_kernels_to_publish());
	STATUS_ERROR_CHECK(Kernel_List->PUBLISH(context));

	return status;
}

/************************************************************************************************************
Add All Kernels to the Kernel List
*************************************************************************************************************/
vx_status get_kernels_to_publish()
{
	vx_status status = VX_SUCCESS;

	Kernel_List = new Kernellist(MAX_KERNELS);
	STATUS_ERROR_CHECK(ADD_KERENEL(Brightness_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BrightnessbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BrightnessbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BrightnessbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(GammaCorrection_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(GammaCorrectionbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(GammaCorrectionbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(GammaCorrectionbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Blend_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BlendbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BlendbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BlendbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Blur_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BlurbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BlurbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BlurbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Contrast_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ContrastbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ContrastbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ContrastbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Pixelate_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(PixelatebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(PixelatebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(PixelatebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Jitter_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(JitterbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(JitterbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(JitterbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Snow_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(SnowbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(SnowbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(SnowbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Noise_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(NoisebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(NoisebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(NoisebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(RandomShadow_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(RandomShadowbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(RandomShadowbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(RandomShadowbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Fog_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(FogbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(FogbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(FogbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Rain_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(RainbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(RainbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(RainbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(RandomCropLetterBox_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(RandomCropLetterBoxbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(RandomCropLetterBoxbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(RandomCropLetterBoxbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Exposure_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ExposurebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ExposurebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ExposurebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(HistogramBalance_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(HistogramBalancebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(HistogramBalancebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(HistogramBalancebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AbsoluteDifference_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AbsoluteDifferencebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AbsoluteDifferencebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AbsoluteDifferencebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AccumulateWeighted_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AccumulateWeightedbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AccumulateWeightedbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AccumulateWeightedbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Accumulate_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AccumulatebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AccumulatebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AccumulatebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Add_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AddbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AddbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AddbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Subtract_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(SubtractbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(SubtractbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(SubtractbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Magnitude_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MagnitudebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MagnitudebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MagnitudebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Multiply_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MultiplybatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MultiplybatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MultiplybatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Phase_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(PhasebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(PhasebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(PhasebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AccumulateSquared_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AccumulateSquaredbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AccumulateSquaredbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(AccumulateSquaredbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BitwiseAND_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BitwiseANDbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BitwiseANDbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BitwiseANDbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BitwiseNOT_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BitwiseNOTbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BitwiseNOTbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BitwiseNOTbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ExclusiveOR_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ExclusiveORbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ExclusiveORbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ExclusiveORbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(InclusiveOR_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(InclusiveORbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(InclusiveORbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(InclusiveORbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Histogram_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Thresholding_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ThresholdingbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ThresholdingbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ThresholdingbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Max_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MaxbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MaxbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MaxbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Min_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MinbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MinbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MinbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MinMaxLoc_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(HistogramEqualize_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(HistogramEqualizebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(HistogramEqualizebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(HistogramEqualizebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MeanStddev_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Flip_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(FlipbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(FlipbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(FlipbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Resize_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ResizebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ResizebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ResizebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ResizeCrop_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ResizeCropbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ResizeCropbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ResizeCropbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Rotate_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(RotatebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(RotatebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(RotatebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(WarpAffine_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(WarpAffinebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(WarpAffinebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(WarpAffinebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Fisheye_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(FisheyebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(FisheyebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(FisheyebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(LensCorrection_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(LensCorrectionbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(LensCorrectionbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(LensCorrectionbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Scale_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ScalebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ScalebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ScalebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(WarpPerspective_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(WarpPerspectivebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(WarpPerspectivebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(WarpPerspectivebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Dilate_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(DilatebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(DilatebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(DilatebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Erode_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ErodebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ErodebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ErodebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Hue_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(HuebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(HuebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(HuebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Saturation_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(SaturationbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(SaturationbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(SaturationbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ColorTemperature_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ColorTemperaturebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ColorTemperaturebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ColorTemperaturebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Vignette_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(VignettebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(VignettebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(VignettebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ChannelExtract_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ChannelExtractbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ChannelExtractbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ChannelCombine_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ChannelCombinebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ChannelCombinebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(LookUpTable_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(LookUpTablebatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(LookUpTablebatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(LookUpTablebatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BilateralFilter_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BilateralFilterbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BilateralFilterbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BilateralFilterbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BoxFilter_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BoxFilterbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BoxFilterbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(BoxFilterbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Sobel_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(SobelbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(SobelbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(SobelbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MedianFilter_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MedianFilterbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MedianFilterbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(MedianFilterbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(CustomConvolution_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(CustomConvolutionbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(CustomConvolutionbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(CustomConvolutionbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(NonMaxSupression_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(NonMaxSupressionbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(NonMaxSupressionbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(NonMaxSupressionbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(GaussianFilter_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(GaussianFilterbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(GaussianFilterbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(GaussianFilterbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(NonLinearFilter_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(NonLinearFilterbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(NonLinearFilterbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(NonLinearFilterbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(LocalBinaryPattern_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(LocalBinaryPatternbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(LocalBinaryPatternbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(LocalBinaryPatternbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(DataObjectCopy_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(DataObjectCopybatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(DataObjectCopybatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(DataObjectCopybatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(GaussianImagePyramid_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(GaussianImagePyramidbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(GaussianImagePyramidbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(LaplacianImagePyramid_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(CannyEdgeDetector_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(HarrisCornerDetector_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(FastCornerDetector_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ControlFlow_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ControlFlowbatchPS_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ControlFlowbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ControlFlowbatchPDROID_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(remap_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(TensorAdd_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(TensorSubtract_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(TensorMultiply_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(TensorMatrixMultiply_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(TensorLookup_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ColorTwist_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ColorTwistbatchPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(CropMirrorNormalizePD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(CropPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(ResizeCropMirrorPD_Register));
	STATUS_ERROR_CHECK(ADD_KERENEL(Copy_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(Nop_Register));
    return status;
}

/************************************************************************************************************
Add Kernels to the Kernel List
*************************************************************************************************************/
vx_status ADD_KERENEL(std::function<vx_status(vx_context)> func)
{
	vx_status status = VX_SUCCESS;
	STATUS_ERROR_CHECK(Kernel_List->ADD(func));
	return status;
}
