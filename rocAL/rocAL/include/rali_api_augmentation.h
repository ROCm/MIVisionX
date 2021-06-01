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

#ifndef MIVISIONX_RALI_API_AUGMENTATION_H
#define MIVISIONX_RALI_API_AUGMENTATION_H
#include "rali_api_types.h"

/// Accepts U8 and RGB24 input.
/// \param context
/// \param input
/// \param dest_width
/// \param dest_height
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliResize(RaliContext context, RaliImage input,
                                                unsigned dest_width, unsigned dest_height,
                                                bool is_output );

/// Accepts U8 and RGB24 input.
/// \param context
/// \param input
/// \param dest_width
/// \param dest_height
/// \param is_output
/// \param area
/// \param x_center_drift
/// \param y_center_drift
/// \return
extern "C"  RaliImage  RALI_API_CALL raliCropResize(RaliContext context, RaliImage input, unsigned dest_width,
                                                    unsigned dest_height, bool is_output,
                                                    RaliFloatParam area = NULL,
                                                    RaliFloatParam aspect_ratio = NULL,
                                                    RaliFloatParam x_center_drift = NULL,
                                                    RaliFloatParam y_center_drift = NULL);
/// Accepts U8 and RGB24 input. Crops the input image to a new area and same aspect ratio.
/// \param context
/// \param input
/// \param dest_width
/// \param dest_height
/// \param is_output
/// \param area
/// \param x_center_drift
/// \param y_center_drift
/// \return
extern "C"  RaliImage  RALI_API_CALL raliCropResizeFixed(RaliContext context, RaliImage input, unsigned dest_width,
                                                            unsigned dest_height, bool is_output, float area, float aspect_ratio,
                                                            float x_center_drift, float y_center_drift);
/// Accepts U8 and RGB24 input. The output image dimension can be set to new values allowing the rotated image to fit,
/// otherwise; the image is cropped to fit the result.
/// \param context Rali context
/// \param input Input Rali Image
/// \param is_output True: the output image is needed by user and will be copied to output buffers using the data
/// transfer API calls. False: the output image is just an intermediate image, user is not interested in
/// using it directly. This option allows certain optimizations to be achieved.
/// \param angle Rali parameter defining the rotation angle value in degrees.
/// \param dest_width The output width
/// \param dest_height The output height
/// \return Returns a new image that keeps the result.
extern "C"  RaliImage  RALI_API_CALL raliRotate(RaliContext context, RaliImage input, bool is_output,
                                                RaliFloatParam angle = NULL,  unsigned dest_width = 0,
                                                unsigned dest_height = 0);

/// Accepts U8 and RGB24 input. The output image dimension can be set to new values allowing the rotated image to fit,
/// otherwise; the image is cropped to fit the result.
/// \param context Rali context
/// \param input Input Rali Image
/// \param dest_width The output width
/// \param dest_height The output height
/// \param is_output Is the output image part of the graph output
/// \param angle The rotation angle value in degrees.
/// \return Returns a new image that keeps the result.
extern "C"  RaliImage  RALI_API_CALL raliRotateFixed(RaliContext context, RaliImage input, float angle,
                                                    bool is_output, unsigned dest_width = 0, unsigned dest_height = 0);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param alpha
/// \param beta
/// \return
extern "C"  RaliImage  RALI_API_CALL raliBrightness(RaliContext context, RaliImage input, bool is_output ,
                                                    RaliFloatParam alpha = NULL, RaliFloatParam beta = NULL);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param shift
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliBrightnessFixed(RaliContext context, RaliImage input,
                                                            float alpha, float beta,
                                                            bool is_output );

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param alpha
/// \return
extern "C"  RaliImage  RALI_API_CALL raliGamma(RaliContext context, RaliImage input,
                                                bool is_output,
                                                RaliFloatParam alpha = NULL);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param alpha
/// \param is_output
/// \return

extern "C"  RaliImage  RALI_API_CALL raliGammaFixed(RaliContext context, RaliImage input, float alpha, bool is_output );

/// Accepts U8 and RGB24 inputs.
/// \param context
/// \param input
/// \param is_output
/// \param min
/// \param max
/// \return
extern "C"  RaliImage  RALI_API_CALL raliContrast(RaliContext context, RaliImage input, bool is_output,
                                                    RaliIntParam min = NULL, RaliIntParam max = NULL);

/// Accepts U8 and RGB24 inputs.
/// \param context
/// \param input
/// \param min
/// \param max
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliContrastFixed(RaliContext context, RaliImage input,
                                                        unsigned min, unsigned max,
                                                        bool is_output);


///
/// \param context
/// \param input
/// \param axis
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliFlip(RaliContext context, RaliImage input, bool is_output,
                                                RaliIntParam flip_axis = NULL);

///
/// \param context
/// \param input
/// \param axis
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliFlipFixed(RaliContext context, RaliImage input, int flip_axis, bool is_output );

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param sdev
/// \return
extern "C"  RaliImage  RALI_API_CALL raliBlur(RaliContext context, RaliImage input, bool is_output,
                                                RaliIntParam sdev = NULL);

///
/// \param context
/// \param input
/// \param sdev
/// \param is_output
/// \return

extern "C"  RaliImage  RALI_API_CALL raliBlurFixed(RaliContext context, RaliImage input, int sdev, bool is_output );

/// Blends two input images given the ratio: output = input1*ratio + input2*(1-ratio)
/// \param context
/// \param input1
/// \param input2
/// \param is_output
/// \param ratio Rali parameter defining the blending ratio, should be between 0.0 and 1.0.
/// \return
extern "C"  RaliImage  RALI_API_CALL raliBlend(RaliContext context, RaliImage input1, RaliImage input2, bool is_output,
                                               RaliFloatParam ratio = NULL);

/// Blends two input images given the ratio: output = input1*ratio + input2*(1-ratio)
/// \param context
/// \param input1
/// \param input2
/// \param ratio Float value defining the blending ratio, should be between 0.0 and 1.0.
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliBlendFixed(RaliContext context,RaliImage input1, RaliImage input2,
                                                    float ratio,
                                                    bool is_output );

/// \param context
/// \param input
/// \param is_output
/// \param x0
/// \param x1
/// \param y0
/// \param y1
/// \param o0
/// \param o1
/// \param dest_height
/// \param dest_width
/// \return
extern "C"  RaliImage  RALI_API_CALL raliWarpAffine(RaliContext context, RaliImage input, bool is_output,
                                                     unsigned dest_height = 0, unsigned dest_width = 0,
                                                     RaliFloatParam x0 = NULL, RaliFloatParam x1 = NULL,
                                                     RaliFloatParam y0= NULL, RaliFloatParam y1 = NULL,
                                                     RaliFloatParam o0 = NULL, RaliFloatParam o1 = NULL);

///
/// \param context
/// \param input
/// \param x0
/// \param x1
/// \param y0
/// \param y1
/// \param o0
/// \param o1
/// \param is_output
/// \param dest_height
/// \param dest_width
/// \return
extern "C"  RaliImage  RALI_API_CALL raliWarpAffineFixed(RaliContext context, RaliImage input, float x0, float x1,
                                                         float y0, float y1, float o0, float o1, bool is_output,
                                                         unsigned int dest_height = 0, unsigned int dest_width = 0);

/// \param context
/// \param input
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliFishEye(RaliContext context, RaliImage input, bool is_output );

///
/// \param context
/// \param input
/// \param is_output
/// \param sdev
/// \return
extern "C"  RaliImage  RALI_API_CALL raliVignette(RaliContext context, RaliImage input, bool is_output,
                                                    RaliFloatParam sdev = NULL);

/// \param context
/// \param input
/// \param sdev
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliVignetteFixed(RaliContext context, RaliImage input,float sdev, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \param min
/// \param max
/// \return
extern "C"  RaliImage  RALI_API_CALL raliJitter(RaliContext context, RaliImage input, bool is_output,
                                                RaliIntParam kernel_size = NULL);

///
/// \param context
/// \param input
/// \param min
/// \param max
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliJitterFixed(RaliContext context, RaliImage input,
                                                        int kernel_size, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \param sdev
/// \return
extern "C"  RaliImage  RALI_API_CALL raliSnPNoise(RaliContext context, RaliImage input, bool is_output,
                                                        RaliFloatParam sdev = NULL);

///
/// \param context
/// \param input
/// \param sdev
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliSnPNoiseFixed(RaliContext context, RaliImage input, float sdev, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \param sdev
/// \return
extern "C"  RaliImage  RALI_API_CALL raliSnow(RaliContext context, RaliImage input, bool is_output,
                                                RaliFloatParam shift = NULL);

/// \param context
/// \param input
/// \param sdev
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliSnowFixed(RaliContext context, RaliImage input, float shift, bool is_output );

///
/// \param context
/// \param input
/// \param is_output
/// \param rain_value
/// \param rain_width
/// \param rain_heigth
/// \param rain_transparency
/// \return
extern "C"  RaliImage  RALI_API_CALL raliRain(RaliContext context, RaliImage input, bool is_output,
                                                RaliFloatParam rain_value = NULL,
                                                RaliIntParam rain_width = NULL,
                                                RaliIntParam rain_height = NULL,
                                                RaliFloatParam rain_transparency = NULL);

/// \param context
/// \param input
/// \param is_output
/// \param rain_value
/// \param rain_width
/// \param rain_heigth
/// \param rain_transparency
/// \return
extern "C"  RaliImage  RALI_API_CALL raliRainFixed(RaliContext context, RaliImage input,
                                                        float rain_value,
                                                        int rain_width,
                                                        int rain_height,
                                                        float rain_transparency,
                                                        bool is_output);

/// \param context
/// \param input
/// \param is_output
/// \param adjustment
/// \return
extern "C"  RaliImage  RALI_API_CALL raliColorTemp(RaliContext context, RaliImage input, bool is_output,
                                                        RaliIntParam adjustment = NULL);

/// \param context
/// \param input
/// \param adjustment
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliColorTempFixed(RaliContext context, RaliImage input, int adjustment, bool is_output);

/// \param context
/// \param input
/// \param is_output
/// \param fog_value
/// \return
extern "C"  RaliImage  RALI_API_CALL raliFog(RaliContext context, RaliImage input, bool is_output,
                                                RaliFloatParam fog_value = NULL);

/// \param context
/// \param input
/// \param fog_value
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliFogFixed(RaliContext context, RaliImage input, float fog_value, bool is_output);

/// \param context
/// \param input
/// \param is_output
/// \param strength
/// \param zoom
/// \return
extern "C"  RaliImage  RALI_API_CALL raliLensCorrection(RaliContext context, RaliImage input, bool is_output,
                                                        RaliFloatParam strength = NULL,
                                                        RaliFloatParam zoom = NULL);

/// \param context
/// \param input
/// \param strength
/// \param zoom
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliLensCorrectionFixed(RaliContext context, RaliImage input,
                                                                float strength, float zoom, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliPixelate(RaliContext context, RaliImage input, bool is_output );

///
/// \param context
/// \param input
/// \param is_output
/// \param shift
/// \return
extern "C"  RaliImage  RALI_API_CALL raliExposure(RaliContext context, RaliImage input, bool is_output,
                                                        RaliFloatParam shift = NULL);

/// \param context
/// \param input
/// \param is_output
/// \param shift
/// \return
extern "C"  RaliImage  RALI_API_CALL raliExposureFixed(RaliContext context, RaliImage input, float shift, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \return

extern "C"  RaliImage  RALI_API_CALL raliHue(RaliContext context, RaliImage input,
                                                bool is_output,
                                                RaliFloatParam hue = NULL);

///
/// \param context
/// \param input
/// \param is_output
/// \param hue
/// \return

extern "C"  RaliImage  RALI_API_CALL raliHueFixed(RaliContext context, RaliImage input,
                                                float hue,
                                                bool is_output);

/// Accepts U8 and RGB24 inputs.
/// \param context
/// \param input
/// \param is_output
/// \param min
/// \param max
/// \return

extern "C" RaliImage RALI_API_CALL raliSaturation(RaliContext context,
                                                RaliImage input,
                                                bool is_output,
                                                RaliFloatParam sat = NULL);

extern "C"  RaliImage  RALI_API_CALL raliSaturationFixed(RaliContext context, RaliImage input, float sat,
                                                bool is_output);

/// Accepts U8 and RGB24 inputs.
/// \param context
/// \param input
/// \param is_output
/// \param min
/// \param max
/// \return

extern "C"  RaliImage  RALI_API_CALL raliCopy(RaliContext context, RaliImage input, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliNop(RaliContext context, RaliImage input, bool is_output);


/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliColorTwist(RaliContext context, RaliImage input, bool is_output,
                                                        RaliFloatParam alpha = NULL,
                                                        RaliFloatParam beta = NULL,
                                                        RaliFloatParam hue = NULL,
                                                        RaliFloatParam sat = NULL);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \return
extern "C"  RaliImage  RALI_API_CALL raliColorTwistFixed(RaliContext context, RaliImage input,
                                                        float alpha,
                                                        float beta,
                                                        float hue,
                                                        float sat,
                                                        bool is_output);

extern "C"  RaliImage  RALI_API_CALL raliCropMirrorNormalize(RaliContext context, RaliImage input,
                                                            unsigned crop_depth,
                                                            unsigned crop_height,
                                                            unsigned crop_width,
                                                            float start_x,
                                                            float start_y,
                                                            float start_z,
                                                            std::vector<float> &mean,
                                                            std::vector<float> &std_dev,
                                                            bool is_output ,
                                                            RaliIntParam mirror = NULL);

extern "C" RaliImage  RALI_API_CALL raliCrop(RaliContext context, RaliImage input, bool is_output,
                                             RaliFloatParam crop_width = NULL,
                                             RaliFloatParam crop_height = NULL,
                                             RaliFloatParam crop_depth = NULL,
                                             RaliFloatParam crop_pox_x = NULL,
                                             RaliFloatParam crop_pos_y = NULL,
                                             RaliFloatParam crop_pos_z = NULL);

extern "C"  RaliImage  RALI_API_CALL raliCropFixed(RaliContext context, RaliImage  input,
                                                   unsigned crop_width,
                                                   unsigned crop_height,
                                                   unsigned crop_depth,
                                                   bool is_output,
                                                   float crop_pox_x,
                                                   float crop_pos_y,
                                                   float crop_pos_z);
// //// \param crop_width


extern "C" RaliImage  RALI_API_CALL raliCropCenterFixed(RaliContext context, RaliImage input,
                                                        unsigned crop_width,
                                                        unsigned crop_height,
                                                        unsigned crop_depth,
                                                        bool output);

extern "C"  RaliImage  RALI_API_CALL raliResizeCropMirrorFixed( RaliContext context, RaliImage input,
                                                           unsigned dest_width, unsigned dest_height,
                                                            bool is_output,
                                                            unsigned crop_h,
                                                            unsigned crop_w,
                                                            RaliIntParam mirror
                                                            );
extern "C"  RaliImage  RALI_API_CALL raliResizeCropMirror( RaliContext context, RaliImage input,
                                                           unsigned dest_width, unsigned dest_height,
                                                            bool is_output, RaliFloatParam crop_height = NULL,
                                                            RaliFloatParam crop_width = NULL, RaliIntParam mirror = NULL
                                                            );

/// Accepts U8 and RGB24 inputs and Ouptus Cropped Images, valid bounding boxes and labels
/// \param context
/// \param input
/// \param num_of_attmpts
/// \return
extern "C" RaliImage RALI_API_CALL raliRandomCrop(  RaliContext context, RaliImage input,
                                                    bool is_output,
                                                    RaliFloatParam crop_area_factor  = NULL,
                                                    RaliFloatParam crop_aspect_ratio = NULL,
                                                    RaliFloatParam crop_pos_x = NULL,
                                                    RaliFloatParam crop_pos_y = NULL,
                                                    int num_of_attempts = 20);

/// Accepts U8 and RGB24 inputs and Ouptus Cropped Images, valid bounding boxes and labels
/// \param context
/// \param input
/// \param IOU_threshold
/// \param num_of_attmpts
/// \return
extern "C" RaliImage RALI_API_CALL raliSSDRandomCrop(  RaliContext context, RaliImage input,
                                                    bool is_output,
                                                    RaliFloatParam threshold = NULL,
                                                    RaliFloatParam crop_area_factor  = NULL,
                                                    RaliFloatParam crop_aspect_ratio = NULL,
                                                    RaliFloatParam crop_pos_x = NULL,
                                                    RaliFloatParam crop_pos_y = NULL,
                                                    int num_of_attempts = 20);
// /// Accepts U8 and RGB24 input. The output image dimension can be set to new values allowing the rotated image to fit,
// /// otherwise; the image is cropped to fit the result.
// /// \param context Rali context
// /// \param input Input Rali Image
// /// \param is_output True: the output image is needed by user and will be copied to output buffers using the data
// /// transfer API calls. False: the output image is just an intermediate image, user is not interested in
// /// using it directly. This option allows certain optimizations to be achieved.
// /// \param angle Rali parameter defining the rotation angle value in degrees.
// /// \param dest_width The output width
// /// \param dest_height The output height
// /// \return Returns a new image that keeps the result.

#endif //MIVISIONX_RALI_API_AUGMENTATION_H
