/*
Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.

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

extern "C" SHARED_PUBLIC vx_status VX_API_CALL vxPublishKernels(vx_context context);
vx_status ADD_KERENEL(std::function<vx_status(vx_context)>);
vx_status get_kernels_to_publish();
vx_status Copy_Register(vx_context);
vx_status Brightness_Register(vx_context);
vx_status Contrast_Register(vx_context);
vx_status Blur_Register(vx_context);
vx_status Flip_Register(vx_context);
vx_status GammaCorrection_Register(vx_context);
vx_status Resize_Register(vx_context);
vx_status ResizeCrop_Register(vx_context);
vx_status Rotate_Register(vx_context);
vx_status WarpAffine_Register(vx_context);
vx_status Blend_Register(vx_context);
vx_status Exposure_Register(vx_context);
vx_status Fisheye_Register(vx_context);
vx_status Snow_Register(vx_context);
vx_status Vignette_Register(vx_context);
vx_status LensCorrection_Register(vx_context);
vx_status Pixelate_Register(vx_context);
vx_status Jitter_Register(vx_context);
vx_status ColorTemperature_Register(vx_context);
vx_status Rain_Register(vx_context);
vx_status Fog_Register(vx_context);
vx_status NoiseSnp_Register(vx_context);

#define VX_KERNEL_RPP_COPY_NAME                         "org.rpp.Copy"
#define VX_KERNEL_RPP_BRIGHTNESS_NAME                   "org.rpp.Brightness"
#define VX_KERNEL_RPP_CONTRAST_NAME                     "org.rpp.Contrast"
#define VX_KERNEL_RPP_BLUR_NAME                         "org.rpp.Blur"
#define VX_KERNEL_RPP_FLIP_NAME                         "org.rpp.Flip"
#define VX_KERNEL_RPP_GAMMA_CORRECTION_NAME             "org.rpp.GammaCorrection"
#define VX_KERNEL_RPP_RESIZE_NAME                       "org.rpp.Resize"
#define VX_KERNEL_RPP_RESIZE_CROP_NAME                  "org.rpp.ResizeCrop"
#define VX_KERNEL_RPP_ROTATE_NAME                       "org.rpp.Rotate"
#define VX_KERNEL_RPP_WARP_AFFINE_NAME                  "org.rpp.WarpAffine"
#define VX_KERNEL_RPP_BLEND_NAME                        "org.rpp.Blend"
#define VX_KERNEL_RPP_EXPOSURE_NAME                     "org.rpp.Exposure"
#define VX_KERNEL_RPP_FISHEYE_NAME                      "org.rpp.Fisheye"
#define VX_KERNEL_RPP_SNOW_NAME                         "org.rpp.Snow"
#define VX_KERNEL_RPP_VIGNETTE_NAME                     "org.rpp.Vignette"
#define VX_KERNEL_RPP_LENSCORRECTION_NAME               "org.rpp.LensCorrection"
#define VX_KERNEL_RPP_PIXELATE_NAME                     "org.rpp.Pixelate"
#define VX_KERNEL_RPP_JITTER_NAME                       "org.rpp.Jitter"
#define VX_KERNEL_RPP_COLORTEMPERATURE_NAME             "org.rpp.ColorTemperature"
#define VX_KERNEL_RPP_RAIN_NAME                         "org.rpp.Rain"
#define VX_KERNEL_RPP_FOG_NAME                          "org.rpp.Fog"
#define VX_KERNEL_RPP_NOISESNP_NAME                     "org.rpp.NoiseSnp"

#endif //_AMDVX_EXT__PUBLISH_KERNELS_H_
