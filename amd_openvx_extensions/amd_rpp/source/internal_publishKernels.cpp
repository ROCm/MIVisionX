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
extern "C" SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context)
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
    STATUS_ERROR_CHECK(ADD_KERENEL(BrightnessbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(GammaCorrectionbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(BlendbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(BlurbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(ContrastbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(PixelatebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(JitterbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(SnowbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(NoisebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(RandomShadowbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(FogbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(RainbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(RandomCropLetterBoxbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(ExposurebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(HistogramBalancebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(AbsoluteDifferencebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(AccumulateWeightedbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(AccumulatebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(AddbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(SubtractbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(MagnitudebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(MultiplybatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(PhasebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(AccumulateSquaredbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(BitwiseANDbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(BitwiseNOTbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(ExclusiveORbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(InclusiveORbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(Histogram_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(ThresholdingbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(MaxbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(MinbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(MinMaxLoc_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(HistogramEqualizebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(MeanStddev_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(FlipbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(ResizebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(ResizeCropbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(RotatebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(WarpAffinebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(FisheyebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(LensCorrectionbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(ScalebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(WarpPerspectivebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(DilatebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(ErodebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(HuebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(SaturationbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(ColorTemperaturebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(VignettebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(ChannelExtractbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(ChannelCombinebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(LookUpTablebatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(BilateralFilterbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(BoxFilterbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(SobelbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(MedianFilterbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CustomConvolutionbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(NonMaxSupressionbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(GaussianFilterbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(NonLinearFilterbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(LocalBinaryPatternbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(DataObjectCopybatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(GaussianImagePyramidbatchPD_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(LaplacianImagePyramid_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(CannyEdgeDetector_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(HarrisCornerDetector_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(FastCornerDetector_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(remap_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(TensorAdd_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(TensorSubtract_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(TensorMultiply_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(TensorMatrixMultiply_Register));
    STATUS_ERROR_CHECK(ADD_KERENEL(TensorLookup_Register));
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
