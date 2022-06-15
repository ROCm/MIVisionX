/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef MIVISIONX_ROCAL_API_AUGMENTATION_H
#define MIVISIONX_ROCAL_API_AUGMENTATION_H
#include "rocal_api_types.h"

/// Accepts U8 and RGB24 input.
// Rearranges the order of the frames in the sequences with respect to new_order.
// new_order can have values in the range [0, sequence_length).
// Frames can be repeated or dropped in the new_order.
/// \param context
/// \param input
/// \param new_order
/// \param new_sequence_length
/// \param sequence_length
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalSequenceRearrange(RocalContext context, RocalImage input,
                                                unsigned int* new_order, unsigned int  new_sequence_length,
                                                unsigned int sequence_length, bool is_output );

/// Accepts U8 and RGB24 input.
/// \param context
/// \param input
/// \param dest_width
/// \param dest_height
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalResize(RocalContext context, RocalImage input,
                                                unsigned dest_width, unsigned dest_height,
                                                bool is_output );

/// Accepts U8 and RGB24 input.
/// \param context
/// \param input
/// \param dest_width
/// \param dest_height
/// \param mean
/// \param std_dev
/// \param is_output
/// \param p_mirror
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalResizeMirrorNormalize(RocalContext p_context, RocalImage p_input,
                                                            unsigned dest_width, unsigned dest_height,
                                                            std::vector<float> &mean, std::vector<float> &std_dev,
                                                            bool is_output, RocalIntParam p_mirror = NULL);

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
extern "C"  RocalImage  ROCAL_API_CALL rocalCropResize(RocalContext context, RocalImage input, unsigned dest_width,
                                                    unsigned dest_height, bool is_output,
                                                    RocalFloatParam area = NULL,
                                                    RocalFloatParam aspect_ratio = NULL,
                                                    RocalFloatParam x_center_drift = NULL,
                                                    RocalFloatParam y_center_drift = NULL);
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
extern "C"  RocalImage  ROCAL_API_CALL rocalCropResizeFixed(RocalContext context, RocalImage input, unsigned dest_width,
                                                            unsigned dest_height, bool is_output, float area, float aspect_ratio,
                                                            float x_center_drift, float y_center_drift);
/// Accepts U8 and RGB24 input. The output image dimension can be set to new values allowing the rotated image to fit,
/// otherwise; the image is cropped to fit the result.
/// \param context Rocal context
/// \param input Input Rocal Image
/// \param is_output True: the output image is needed by user and will be copied to output buffers using the data
/// transfer API calls. False: the output image is just an intermediate image, user is not interested in
/// using it directly. This option allows certain optimizations to be achieved.
/// \param angle Rocal parameter defining the rotation angle value in degrees.
/// \param dest_width The output width
/// \param dest_height The output height
/// \return Returns a new image that keeps the result.
extern "C"  RocalImage  ROCAL_API_CALL rocalRotate(RocalContext context, RocalImage input, bool is_output,
                                                RocalFloatParam angle = NULL,  unsigned dest_width = 0,
                                                unsigned dest_height = 0);

/// Accepts U8 and RGB24 input. The output image dimension can be set to new values allowing the rotated image to fit,
/// otherwise; the image is cropped to fit the result.
/// \param context Rocal context
/// \param input Input Rocal Image
/// \param dest_width The output width
/// \param dest_height The output height
/// \param is_output Is the output image part of the graph output
/// \param angle The rotation angle value in degrees.
/// \return Returns a new image that keeps the result.
extern "C"  RocalImage  ROCAL_API_CALL rocalRotateFixed(RocalContext context, RocalImage input, float angle,
                                                    bool is_output, unsigned dest_width = 0, unsigned dest_height = 0);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param alpha
/// \param beta
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalBrightness(RocalContext context, RocalImage input, bool is_output ,
                                                    RocalFloatParam alpha = NULL, RocalFloatParam beta = NULL);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param shift
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalBrightnessFixed(RocalContext context, RocalImage input,
                                                            float alpha, float beta,
                                                            bool is_output );

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param alpha
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalGamma(RocalContext context, RocalImage input,
                                                bool is_output,
                                                RocalFloatParam alpha = NULL);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param alpha
/// \param is_output
/// \return

extern "C"  RocalImage  ROCAL_API_CALL rocalGammaFixed(RocalContext context, RocalImage input, float alpha, bool is_output );

/// Accepts U8 and RGB24 inputs.
/// \param context
/// \param input
/// \param is_output
/// \param min
/// \param max
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalContrast(RocalContext context, RocalImage input, bool is_output,
                                                    RocalIntParam min = NULL, RocalIntParam max = NULL);

/// Accepts U8 and RGB24 inputs.
/// \param context
/// \param input
/// \param min
/// \param max
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalContrastFixed(RocalContext context, RocalImage input,
                                                        unsigned min, unsigned max,
                                                        bool is_output);


///
/// \param context
/// \param input
/// \param axis
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalFlip(RocalContext context, RocalImage input, bool is_output,
                                                RocalIntParam flip_axis = NULL);

///
/// \param context
/// \param input
/// \param axis
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalFlipFixed(RocalContext context, RocalImage input, int flip_axis, bool is_output );

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param sdev
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalBlur(RocalContext context, RocalImage input, bool is_output,
                                                RocalIntParam sdev = NULL);

///
/// \param context
/// \param input
/// \param sdev
/// \param is_output
/// \return

extern "C"  RocalImage  ROCAL_API_CALL rocalBlurFixed(RocalContext context, RocalImage input, int sdev, bool is_output );

/// Blends two input images given the ratio: output = input1*ratio + input2*(1-ratio)
/// \param context
/// \param input1
/// \param input2
/// \param is_output
/// \param ratio Rocal parameter defining the blending ratio, should be between 0.0 and 1.0.
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalBlend(RocalContext context, RocalImage input1, RocalImage input2, bool is_output,
                                               RocalFloatParam ratio = NULL);

/// Blends two input images given the ratio: output = input1*ratio + input2*(1-ratio)
/// \param context
/// \param input1
/// \param input2
/// \param ratio Float value defining the blending ratio, should be between 0.0 and 1.0.
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalBlendFixed(RocalContext context,RocalImage input1, RocalImage input2,
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
extern "C"  RocalImage  ROCAL_API_CALL rocalWarpAffine(RocalContext context, RocalImage input, bool is_output,
                                                     unsigned dest_height = 0, unsigned dest_width = 0,
                                                     RocalFloatParam x0 = NULL, RocalFloatParam x1 = NULL,
                                                     RocalFloatParam y0= NULL, RocalFloatParam y1 = NULL,
                                                     RocalFloatParam o0 = NULL, RocalFloatParam o1 = NULL);

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
extern "C"  RocalImage  ROCAL_API_CALL rocalWarpAffineFixed(RocalContext context, RocalImage input, float x0, float x1,
                                                         float y0, float y1, float o0, float o1, bool is_output,
                                                         unsigned int dest_height = 0, unsigned int dest_width = 0);

/// \param context
/// \param input
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalFishEye(RocalContext context, RocalImage input, bool is_output );

///
/// \param context
/// \param input
/// \param is_output
/// \param sdev
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalVignette(RocalContext context, RocalImage input, bool is_output,
                                                    RocalFloatParam sdev = NULL);

/// \param context
/// \param input
/// \param sdev
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalVignetteFixed(RocalContext context, RocalImage input,float sdev, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \param min
/// \param max
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalJitter(RocalContext context, RocalImage input, bool is_output,
                                                RocalIntParam kernel_size = NULL);

///
/// \param context
/// \param input
/// \param min
/// \param max
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalJitterFixed(RocalContext context, RocalImage input,
                                                        int kernel_size, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \param sdev
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalSnPNoise(RocalContext context, RocalImage input, bool is_output,
                                                        RocalFloatParam sdev = NULL);

///
/// \param context
/// \param input
/// \param sdev
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalSnPNoiseFixed(RocalContext context, RocalImage input, float sdev, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \param sdev
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalSnow(RocalContext context, RocalImage input, bool is_output,
                                                RocalFloatParam shift = NULL);

/// \param context
/// \param input
/// \param sdev
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalSnowFixed(RocalContext context, RocalImage input, float shift, bool is_output );

///
/// \param context
/// \param input
/// \param is_output
/// \param rain_value
/// \param rain_width
/// \param rain_heigth
/// \param rain_transparency
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalRain(RocalContext context, RocalImage input, bool is_output,
                                                RocalFloatParam rain_value = NULL,
                                                RocalIntParam rain_width = NULL,
                                                RocalIntParam rain_height = NULL,
                                                RocalFloatParam rain_transparency = NULL);

/// \param context
/// \param input
/// \param is_output
/// \param rain_value
/// \param rain_width
/// \param rain_heigth
/// \param rain_transparency
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalRainFixed(RocalContext context, RocalImage input,
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
extern "C"  RocalImage  ROCAL_API_CALL rocalColorTemp(RocalContext context, RocalImage input, bool is_output,
                                                        RocalIntParam adjustment = NULL);

/// \param context
/// \param input
/// \param adjustment
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalColorTempFixed(RocalContext context, RocalImage input, int adjustment, bool is_output);

/// \param context
/// \param input
/// \param is_output
/// \param fog_value
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalFog(RocalContext context, RocalImage input, bool is_output,
                                                RocalFloatParam fog_value = NULL);

/// \param context
/// \param input
/// \param fog_value
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalFogFixed(RocalContext context, RocalImage input, float fog_value, bool is_output);

/// \param context
/// \param input
/// \param is_output
/// \param strength
/// \param zoom
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalLensCorrection(RocalContext context, RocalImage input, bool is_output,
                                                        RocalFloatParam strength = NULL,
                                                        RocalFloatParam zoom = NULL);

/// \param context
/// \param input
/// \param strength
/// \param zoom
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalLensCorrectionFixed(RocalContext context, RocalImage input,
                                                                float strength, float zoom, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalPixelate(RocalContext context, RocalImage input, bool is_output );

///
/// \param context
/// \param input
/// \param is_output
/// \param shift
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalExposure(RocalContext context, RocalImage input, bool is_output,
                                                        RocalFloatParam shift = NULL);

/// \param context
/// \param input
/// \param is_output
/// \param shift
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalExposureFixed(RocalContext context, RocalImage input, float shift, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \return

extern "C"  RocalImage  ROCAL_API_CALL rocalHue(RocalContext context, RocalImage input,
                                                bool is_output,
                                                RocalFloatParam hue = NULL);

///
/// \param context
/// \param input
/// \param is_output
/// \param hue
/// \return

extern "C"  RocalImage  ROCAL_API_CALL rocalHueFixed(RocalContext context, RocalImage input,
                                                float hue,
                                                bool is_output);

/// Accepts U8 and RGB24 inputs.
/// \param context
/// \param input
/// \param is_output
/// \param min
/// \param max
/// \return

extern "C" RocalImage ROCAL_API_CALL rocalSaturation(RocalContext context,
                                                RocalImage input,
                                                bool is_output,
                                                RocalFloatParam sat = NULL);

extern "C"  RocalImage  ROCAL_API_CALL rocalSaturationFixed(RocalContext context, RocalImage input, float sat,
                                                bool is_output);

/// Accepts U8 and RGB24 inputs.
/// \param context
/// \param input
/// \param is_output
/// \param min
/// \param max
/// \return

extern "C"  RocalImage  ROCAL_API_CALL rocalCopy(RocalContext context, RocalImage input, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalNop(RocalContext context, RocalImage input, bool is_output);


/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalColorTwist(RocalContext context, RocalImage input, bool is_output,
                                                        RocalFloatParam alpha = NULL,
                                                        RocalFloatParam beta = NULL,
                                                        RocalFloatParam hue = NULL,
                                                        RocalFloatParam sat = NULL);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \return
extern "C"  RocalImage  ROCAL_API_CALL rocalColorTwistFixed(RocalContext context, RocalImage input,
                                                        float alpha,
                                                        float beta,
                                                        float hue,
                                                        float sat,
                                                        bool is_output);

extern "C"  RocalImage  ROCAL_API_CALL rocalCropMirrorNormalize(RocalContext context, RocalImage input,
                                                            unsigned crop_depth,
                                                            unsigned crop_height,
                                                            unsigned crop_width,
                                                            float start_x,
                                                            float start_y,
                                                            float start_z,
                                                            std::vector<float> &mean,
                                                            std::vector<float> &std_dev,
                                                            bool is_output ,
                                                            RocalIntParam mirror = NULL);

extern "C" RocalImage  ROCAL_API_CALL rocalCrop(RocalContext context, RocalImage input, bool is_output,
                                             RocalFloatParam crop_width = NULL,
                                             RocalFloatParam crop_height = NULL,
                                             RocalFloatParam crop_depth = NULL,
                                             RocalFloatParam crop_pox_x = NULL,
                                             RocalFloatParam crop_pos_y = NULL,
                                             RocalFloatParam crop_pos_z = NULL);

extern "C"  RocalImage  ROCAL_API_CALL rocalCropFixed(RocalContext context, RocalImage  input,
                                                   unsigned crop_width,
                                                   unsigned crop_height,
                                                   unsigned crop_depth,
                                                   bool is_output,
                                                   float crop_pox_x,
                                                   float crop_pos_y,
                                                   float crop_pos_z);
// //// \param crop_width


extern "C" RocalImage  ROCAL_API_CALL rocalCropCenterFixed(RocalContext context, RocalImage input,
                                                        unsigned crop_width,
                                                        unsigned crop_height,
                                                        unsigned crop_depth,
                                                        bool output);

extern "C"  RocalImage  ROCAL_API_CALL rocalResizeCropMirrorFixed( RocalContext context, RocalImage input,
                                                           unsigned dest_width, unsigned dest_height,
                                                            bool is_output,
                                                            unsigned crop_h,
                                                            unsigned crop_w,
                                                            RocalIntParam mirror
                                                            );
extern "C"  RocalImage  ROCAL_API_CALL rocalResizeCropMirror( RocalContext context, RocalImage input,
                                                           unsigned dest_width, unsigned dest_height,
                                                            bool is_output, RocalFloatParam crop_height = NULL,
                                                            RocalFloatParam crop_width = NULL, RocalIntParam mirror = NULL
                                                            );

/// Accepts U8 and RGB24 inputs and Ouptus Cropped Images, valid bounding boxes and labels
/// \param context
/// \param input
/// \param num_of_attmpts
/// \return
extern "C" RocalImage ROCAL_API_CALL rocalRandomCrop(  RocalContext context, RocalImage input,
                                                    bool is_output,
                                                    RocalFloatParam crop_area_factor  = NULL,
                                                    RocalFloatParam crop_aspect_ratio = NULL,
                                                    RocalFloatParam crop_pos_x = NULL,
                                                    RocalFloatParam crop_pos_y = NULL,
                                                    int num_of_attempts = 20);

/// Accepts U8 and RGB24 inputs and Ouptus Cropped Images, valid bounding boxes and labels
/// \param context
/// \param input
/// \param IOU_threshold
/// \param num_of_attmpts
/// \return
extern "C" RocalImage ROCAL_API_CALL rocalSSDRandomCrop(  RocalContext context, RocalImage input,
                                                    bool is_output,
                                                    RocalFloatParam threshold = NULL,
                                                    RocalFloatParam crop_area_factor  = NULL,
                                                    RocalFloatParam crop_aspect_ratio = NULL,
                                                    RocalFloatParam crop_pos_x = NULL,
                                                    RocalFloatParam crop_pos_y = NULL,
                                                    int num_of_attempts = 20);
// /// Accepts U8 and RGB24 input. The output image dimension can be set to new values allowing the rotated image to fit,
// /// otherwise; the image is cropped to fit the result.
// /// \param context Rocal context
// /// \param input Input Rocal Image
// /// \param is_output True: the output image is needed by user and will be copied to output buffers using the data
// /// transfer API calls. False: the output image is just an intermediate image, user is not interested in
// /// using it directly. This option allows certain optimizations to be achieved.
// /// \param angle Rocal parameter defining the rotation angle value in degrees.
// /// \param dest_width The output width
// /// \param dest_height The output height
// /// \return Returns a new image that keeps the result.

#endif //MIVISIONX_ROCAL_API_AUGMENTATION_H
