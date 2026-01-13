/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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
    //tensor 
    STATUS_ERROR_CHECK(ADD_KERNEL(Blend_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Blur_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Brightness_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ColorTemperature_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ColorTwist_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Contrast_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Copy_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Crop_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(CropMirrorNormalize_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Exposure_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(FishEye_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Flip_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Fog_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(GammaCorrection_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Glitch_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Hue_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Jitter_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(LensCorrection_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Noise_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Nop_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Pixelate_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(PreemphasisFilter_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Rain_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Resize_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ResizeCrop_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ResizeCropMirror_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ResizeMirrorNormalize_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Rotate_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Saturation_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(SequenceRearrange_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Snow_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Vignette_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(WarpAffine_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Spectrogram_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Downmix_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ToDecibels_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Resample_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(TensorMulScalar_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(TensorAddTensor_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(NonSilentRegionDetection_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Slice_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Normalize_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(MelFilterBank_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Transpose_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Log1p_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(PythonFunction_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(ColorCast_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(GaussianFilter_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(GridMask_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(MedianFilter_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(NonLinearBlend_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Dilate_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Erode_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Magnitude_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Phase_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Threshold_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(WarpPerspective_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Erase_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(CropAndPatch_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Remap_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(Ricap_Register));
    STATUS_ERROR_CHECK(ADD_KERNEL(BitwiseOps_Register));

    return status;
}

/************************************************************************************************************
Add Kernels to the Kernel List
*************************************************************************************************************/
vx_status ADD_KERNEL(std::function<vx_status(vx_context)> func)
{
    vx_status status = VX_SUCCESS;
    STATUS_ERROR_CHECK(Kernel_List->ADD(func));
    return status;
}
