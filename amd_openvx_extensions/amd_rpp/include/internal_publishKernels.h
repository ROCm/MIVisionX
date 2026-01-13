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

#ifndef _PUBLISH_KERNELS_H_
#define _PUBLISH_KERNELS_H_

#include "internal_rpp.h"

#ifndef SHARED_PUBLIC
#if _WIN32
#define SHARED_PUBLIC __declspec(dllexport)
#else
#define SHARED_PUBLIC __attribute__ ((visibility ("default")))
#endif
#endif

#define RPP_MAX_TENSOR_DIMS 5

extern "C" SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context);
vx_status ADD_KERNEL(std::function<vx_status(vx_context)>);
vx_status get_kernels_to_publish();

vx_status Blend_Register(vx_context);
vx_status Blur_Register(vx_context);
vx_status Brightness_Register(vx_context);
vx_status ColorTemperature_Register(vx_context);
vx_status ColorTwist_Register(vx_context);
vx_status Contrast_Register(vx_context);
vx_status Copy_Register(vx_context);
vx_status Crop_Register(vx_context);
vx_status CropMirrorNormalize_Register(vx_context);
vx_status Exposure_Register(vx_context);
vx_status FishEye_Register(vx_context);
vx_status Flip_Register(vx_context);
vx_status Fog_Register(vx_context);
vx_status GammaCorrection_Register(vx_context);
vx_status Glitch_Register(vx_context);
vx_status Hue_Register(vx_context);
vx_status Jitter_Register(vx_context);
vx_status LensCorrection_Register(vx_context);
vx_status Noise_Register(vx_context);
vx_status Nop_Register(vx_context);
vx_status Pixelate_Register(vx_context);
vx_status PreemphasisFilter_Register(vx_context);
vx_status Rain_Register(vx_context);
vx_status Resize_Register(vx_context);
vx_status ResizeCrop_Register(vx_context);
vx_status ResizeCropMirror_Register(vx_context);
vx_status ResizeMirrorNormalize_Register(vx_context);
vx_status Rotate_Register(vx_context);
vx_status Saturation_Register(vx_context);
vx_status SequenceRearrange_Register(vx_context);
vx_status Snow_Register(vx_context);
vx_status Vignette_Register(vx_context);
vx_status WarpAffine_Register(vx_context);
vx_status SequenceRearrange_Register(vx_context);
vx_status Spectrogram_Register(vx_context);
vx_status Downmix_Register(vx_context);
vx_status ToDecibels_Register(vx_context);
vx_status Resample_Register(vx_context);
vx_status TensorMulScalar_Register(vx_context);
vx_status TensorAddTensor_Register(vx_context);
vx_status NonSilentRegionDetection_Register(vx_context);
vx_status Slice_Register(vx_context);
vx_status Normalize_Register(vx_context);
vx_status MelFilterBank_Register(vx_context);
vx_status Transpose_Register(vx_context);
vx_status Log1p_Register(vx_context);
vx_status PythonFunction_Register(vx_context);
vx_status ColorCast_Register(vx_context);
vx_status GaussianFilter_Register(vx_context);
vx_status GridMask_Register(vx_context);
vx_status MedianFilter_Register(vx_context);
vx_status NonLinearBlend_Register(vx_context);
vx_status Dilate_Register(vx_context);
vx_status Erode_Register(vx_context);
vx_status Magnitude_Register(vx_context);
vx_status Phase_Register(vx_context);
vx_status Threshold_Register(vx_context);
vx_status WarpPerspective_Register(vx_context);
vx_status Erase_Register(vx_context);
vx_status CropAndPatch_Register(vx_context);
vx_status Remap_Register(vx_context);
vx_status Ricap_Register(vx_context);
vx_status BitwiseOps_Register(vx_context);

//tensor
#define VX_KERNEL_RPP_BLEND_NAME                                "org.rpp.Blend"
#define VX_KERNEL_RPP_BLUR_NAME                                 "org.rpp.Blur"
#define VX_KERNEL_RPP_BRIGHTNESS_NAME                           "org.rpp.Brightness"
#define VX_KERNEL_RPP_COLORTEMPERATURE_NAME                     "org.rpp.ColorTemperature"
#define VX_KERNEL_RPP_COLORTWIST_NAME                           "org.rpp.ColorTwist"
#define VX_KERNEL_RPP_CONTRAST_NAME                             "org.rpp.Contrast"
#define VX_KERNEL_RPP_COPY_NAME                                 "org.rpp.Copy"
#define VX_KERNEL_RPP_CROP_NAME                                 "org.rpp.Crop"
#define VX_KERNEL_RPP_CROPMIRRORNORMALIZE_NAME                  "org.rpp.CropMirrorNormalize"
#define VX_KERNEL_RPP_EXPOSURE_NAME                             "org.rpp.Exposure"
#define VX_KERNEL_RPP_FISHEYE_NAME                              "org.rpp.FishEye"
#define VX_KERNEL_RPP_FLIP_NAME                                 "org.rpp.Flip"
#define VX_KERNEL_RPP_FOG_NAME                                  "org.rpp.Fog"
#define VX_KERNEL_RPP_GAMMACORRECTION_NAME                      "org.rpp.GammaCorrection"
#define VX_KERNEL_RPP_GLITCH_NAME                               "org.rpp.Glitch"
#define VX_KERNEL_RPP_HUE_NAME                                  "org.rpp.Hue"
#define VX_KERNEL_RPP_JITTER_NAME                               "org.rpp.Jitter"
#define VX_KERNEL_RPP_LENSCORRECTION_NAME                       "org.rpp.LensCorrection"
#define VX_KERNEL_RPP_NOISE_NAME                                "org.rpp.Noise"
#define VX_KERNEL_RPP_NOP_NAME                                  "org.rpp.Nop"
#define VX_KERNEL_RPP_RAIN_NAME                                 "org.rpp.Rain"
#define VX_KERNEL_RPP_RESIZE_NAME                               "org.rpp.Resize"
#define VX_KERNEL_RPP_RESIZECROP_NAME                           "org.rpp.ResizeCrop"
#define VX_KERNEL_RPP_RESIZECROPMIRROR_NAME                     "org.rpp.ResizeCropMirror"
#define VX_KERNEL_RPP_RESIZEMIRRORNORMALIZE_NAME                "org.rpp.ResizeMirrorNormalize"
#define VX_KERNEL_RPP_ROTATE_NAME                               "org.rpp.Rotate"
#define VX_KERNEL_RPP_SATURATION_NAME                           "org.rpp.Saturation"
#define VX_KERNEL_RPP_SEQUENCEREARRANGE_NAME                    "org.rpp.SequenceRearrange"
#define VX_KERNEL_RPP_SNOW_NAME                                 "org.rpp.Snow"
#define VX_KERNEL_RPP_PIXELATE_NAME                             "org.rpp.Pixelate"
#define VX_KERNEL_RPP_VIGNETTE_NAME                             "org.rpp.Vignette"
#define VX_KERNEL_RPP_WARPAFFINE_NAME                           "org.rpp.WarpAffine"
#define VX_KERNEL_RPP_BRIGHTNESS_NAME                           "org.rpp.Brightness"
#define VX_KERNEL_RPP_COPY_NAME                                 "org.rpp.Copy"
#define VX_KERNEL_RPP_CROPMIRRORNORMALIZE_NAME                  "org.rpp.CropMirrorNormalize"
#define VX_KERNEL_RPP_NOP_NAME                                  "org.rpp.Nop"
#define VX_KERNEL_RPP_RESIZE_NAME                               "org.rpp.Resize"
#define VX_KERNEL_RPP_SEQUENCEREARRANGE_NAME                    "org.rpp.SequenceRearrange"
#define VX_KERNEL_RPP_PREEMPHASISFILTER_NAME                    "org.rpp.PreemphasisFilter"
#define VX_KERNEL_RPP_SPECTROGRAM_NAME                          "org.rpp.Spectrogram"
#define VX_KERNEL_RPP_DOWNMIX_NAME                              "org.rpp.Downmix"
#define VX_KERNEL_RPP_TODECIBELS_NAME                           "org.rpp.ToDecibels"
#define VX_KERNEL_RPP_RESAMPLE_NAME                             "org.rpp.Resample"
#define VX_KERNEL_RPP_TENSORMULSCALAR_NAME                      "org.rpp.TensorMulScalar"
#define VX_KERNEL_RPP_TENSORADDTENSOR_NAME                      "org.rpp.TensorAddTensor"
#define VX_KERNEL_RPP_NONSILENTREGIONDETECTION_NAME             "org.rpp.NonSilentRegionDetection"
#define VX_KERNEL_RPP_SLICE_NAME                                "org.rpp.Slice"
#define VX_KERNEL_RPP_NORMALIZE_NAME                            "org.rpp.Normalize"
#define VX_KERNEL_RPP_MELFILTERBANK_NAME                        "org.rpp.MelFilterBank"
#define VX_KERNEL_RPP_TRANSPOSE_NAME                            "org.rpp.Transpose"
#define VX_KERNEL_RPP_LOG1P_NAME                                "org.rpp.Log1p"
#define VX_KERNEL_RPP_PYTHON_FUNCTION_NAME                      "org.rpp.PythonFunction"
#define VX_KERNEL_RPP_COLORCAST_NAME                            "org.rpp.ColorCast"
#define VX_KERNEL_RPP_GAUSSIAN_FILTER_NAME                      "org.rpp.GaussianFilter"
#define VX_KERNEL_RPP_GRIDMASK_NAME                             "org.rpp.GridMask"
#define VX_KERNEL_RPP_MEDIAN_FILTER_NAME                        "org.rpp.MedianFilter"
#define VX_KERNEL_RPP_NON_LINEAR_BLEND_NAME                     "org.rpp.NonLinearBlend"
#define VX_KERNEL_RPP_DILATE_NAME                               "org.rpp.Dilate"
#define VX_KERNEL_RPP_ERODE_NAME                                "org.rpp.Erode"
#define VX_KERNEL_RPP_MAGNITUDE_NAME                            "org.rpp.Magnitude"
#define VX_KERNEL_RPP_PHASE_NAME                                "org.rpp.Phase"
#define VX_KERNEL_RPP_THRESHOLD_NAME                            "org.rpp.Threshold"
#define VX_KERNEL_RPP_WARP_PERSPECTIVE_NAME                     "org.rpp.WarpPerspective"
#define VX_KERNEL_RPP_ERASE_NAME                                "org.rpp.Erase"
#define VX_KERNEL_RPP_CROP_AND_PATCH_NAME                       "org.rpp.CropAndPatch"
#define VX_KERNEL_RPP_REMAP_NAME                         "org.rpp.Remap"
#define VX_KERNEL_RPP_RICAP_NAME                                "org.rpp.Ricap"
#define VX_KERNEL_RPP_BITWISE_OPS_NAME                          "org.rpp.BitwiseOps"

#endif //_AMDVX_EXT__PUBLISH_KERNELS_H_
