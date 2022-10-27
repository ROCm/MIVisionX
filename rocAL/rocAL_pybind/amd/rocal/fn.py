# Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from amd.rocal import readers
from amd.rocal import decoders
from amd.rocal import random
from amd.rocal import noise
from amd.rocal import reductions

import amd.rocal.types as types
import rocal_pybind as b
from amd.rocal.pipeline import Pipeline

def blend(*inputs,ratio=None):
    kwargs_pybind = {"input_image0":inputs[0], "input_image1":inputs[1], "is_output":False ,"ratio":ratio}
    blend_image = b.Blend(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (blend_image)

def snow(*inputs, snow=0.5, device=None):
    # pybind call arguments
    snow = b.CreateFloatParameter(snow) if isinstance(snow, float) else snow
    kwargs_pybind = {"input_image0": inputs[0],"is_output": False, "shift": snow}
    snow_image = b.Snow(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (snow_image)

def exposure(*inputs, exposure=0.5, device=None):
    # pybind call arguments
    exposure = b.CreateFloatParameter(exposure) if isinstance(exposure, float) else exposure
    kwargs_pybind = {"input_image0": inputs[0],"is_output": False, "shift": exposure}
    exposure_image = b.Exposure(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (exposure_image)

def fish_eye(*inputs, device=None):
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],"is_output": False}
    fisheye_image = b.FishEye(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (fisheye_image)

def fog(*inputs, fog=0.5, device=None):
    # pybind call arguments
    fog = b.CreateFloatParameter(fog) if isinstance(fog, float) else fog
    kwargs_pybind = {"input_image0": inputs[0],
                     "is_output": False, "fog_value": fog}
    fog_image = b.Fog(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (fog_image)

def brightness(*inputs, brightness=1.0, bytes_per_sample_hint=0, image_type=0,
               preserve=False, seed=-1, device=None):
    """
    brightness (float, optional, default = 1.0) –

    Brightness change factor. Values >= 0 are accepted. For example:

    0 - black image,

    1 - no change

    2 - increase brightness twice

    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    image_type (int, optional, default = 0) – The color space of input and output image

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)
    """

    kwargs_pybind = {"input_image0": inputs[0], "is_output": False, "alpha": None, "beta": None}
    brightness_image = b.Brightness(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (brightness_image)

def brightness_fixed(*inputs, alpha=None, beta=None, seed=-1, device=None):
    alpha = b.CreateFloatParameter(alpha) if isinstance(alpha, float) else alpha
    beta = b.CreateFloatParameter(beta) if isinstance(beta, float) else beta
    kwargs_pybind = {"input_image0": inputs[0], "is_output": False, "alpha": alpha, "beta": beta}
    brightness_image = b.Brightness(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (brightness_image)

def lens_correction(*inputs, strength =None, zoom = None):
    strength = b.CreateFloatParameter(strength) if isinstance(
        strength, float) else strength
    zoom = b.CreateFloatParameter(zoom) if isinstance(zoom, float) else zoom
    kwargs_pybind = {"input_image0": inputs[0], "is_output": False, "strength": strength, "zoom": zoom}
    len_corrected_image = b.LensCorrection(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (len_corrected_image)

def blur(*inputs, blur=3, device=None):
    """
    BLUR
    """
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],"is_output": False, "sdev": None}
    blur_image = b.Blur(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (blur_image)

def contrast(*inputs, bytes_per_sample_hint=0, contrast=1.0, image_type=0, min_contrast=None, max_contrast=None,
             preserve=False, seed=-1, device=None):
    """
    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    contrast (float, optional, default = 1.0) –

    Contrast change factor. Values >= 0 are accepted. For example:

    0 - gray image,

    1 - no change

    2 - increase contrast twice

    image_type (int, optional, default = 0) – The color space of input and output image

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    """
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],
                     "is_output": False, "min": min_contrast, "max": max_contrast}
    contrast_image = b.Contrast(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (contrast_image)

def flip(*inputs, flip=0, device=None):
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],
                     "is_output": False, "flip_axis": None}
    flip_image = b.Flip(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (flip_image)

def gamma_correction(*inputs, gamma=0.5, device=None):
    # pybind call arguments
    gamma = b.CreateFloatParameter(gamma) if isinstance(gamma, float) else gamma
    kwargs_pybind = {"input_image0": inputs[0],
                     "is_output": False, "gamma": gamma}
    gamma_correction_image = b.GammaCorrection(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (gamma_correction_image)

def hue(*inputs, bytes_per_sample_hint=0,  hue=None, image_type=0,
        preserve=False, seed = -1, device = None):
    """
    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    hue (float, optional, default = 0.0) – Hue change, in degrees.

    image_type (int, optional, default = 0) – The color space of input and output image

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    """
    # pybind call arguments
    hue = b.CreateFloatParameter(hue) if isinstance(hue, float) else hue
    kwargs_pybind = {"input_image0": inputs[0],
                     "is_output": False, "hue": hue}
    hue_image = b.Hue(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    return (hue_image)

def jitter(*inputs, bytes_per_sample_hint=0, fill_value=0.0, interp_type= 0,
        mask = 1, nDegree = 2, kernel_size=None, preserve = False, seed = -1, device = None):
    """
    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    fill_value (float, optional, default = 0.0) – Color value used for padding pixels.

    interp_type (int, optional, default = 0) – Type of interpolation used.

    mask (int, optional, default = 1) –

    Whether to apply this augmentation to the input image.

    0 - do not apply this transformation

    1 - apply this transformation

    nDegree (int, optional, default = 2) – Each pixel is moved by a random amount in range [-nDegree/2, nDegree/2].

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    """

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],
                     "is_output": False, "kernel_size": kernel_size}
    jitter_image = b.Jitter(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (jitter_image)

def pixelate(*inputs, device = None):
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],
                     "is_output": False}
    pixelate_image = b.Pixelate(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (pixelate_image)

def rain(*inputs, rain=None, rain_width = None, rain_height = None, rain_transparency = None, device = None):
    # pybind call arguments
    rain = b.CreateFloatParameter(rain) if isinstance(rain, float) else rain
    rain_width = b.CreateIntParameter(rain_width) if isinstance(rain_width, float) else rain_width
    rain_height = b.CreateIntParameter(rain_height) if isinstance(rain_height, float) else rain_height
    rain_transparency = b.CreateFloatParameter(rain_transparency) if isinstance(rain_transparency, float) else rain_transparency

    kwargs_pybind = {"input_image0": inputs[0],"is_output": False, "rain_value": rain, "rain_width": rain_width, "rain_height": rain_height, "rain_transparency": rain_transparency}
    rain_image = b.Rain(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (rain_image)

def resize(*inputs, bytes_per_sample_hint=0, image_type=0, interp_type=1, mag_filter=1, max_size=[], min_filter=1,
            minibatch_size=32, preserve=False, resize_longer=0, resize_shorter=0, resize_x=0, resize_y=0, 
            scaling_mode=types.SCALING_MODE_DEFAULT, interpolation_type=types.LINEAR_INTERPOLATION,
            save_attrs=False, seed=1, temp_buffer_hint=0, device = None):
    """
    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    image_type (int, optional, default = 0) – The color space of input and output image.

    interp_type (int, optional, default = 1) – Type of interpolation used. Use min_filter and mag_filter to specify different filtering for downscaling and upscaling.

    mag_filter (int, optional, default = 1) – Filter used when scaling up

    max_size (float or list of float, optional, default = [0.0, 0.0]) –

    Maximum size of the longer dimension when resizing with resize_shorter. When set with resize_shorter, the shortest dimension will be resized to resize_shorter iff the longest dimension is smaller or equal to max_size. If not, the shortest dimension is resized to satisfy the constraint longest_dim == max_size. Can be also an array of size 2, where the two elements are maximum size per dimension (H, W).

    Example:

    Original image = 400x1200.

    Resized with:

        resize_shorter = 200 (max_size not set) => 200x600

        resize_shorter = 200, max_size =  400 => 132x400

        resize_shorter = 200, max_size = 1000 => 200x600

    min_filter (int, optional, default = 1) – Filter used when scaling down

    minibatch_size (int, optional, default = 32) – Maximum number of images processed in a single kernel call

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    resize_longer (float, optional, default = 0.0) – The length of the longer dimension of the resized image. This option is mutually exclusive with resize_shorter,`resize_x` and resize_y. The op will keep the aspect ratio of the original image.

    resize_shorter (float, optional, default = 0.0) – The length of the shorter dimension of the resized image. This option is mutually exclusive with resize_longer, resize_x and resize_y. The op will keep the aspect ratio of the original image. The longer dimension can be bounded by setting the max_size argument. See max_size argument doc for more info.

    resize_x (float, optional, default = 0.0) – The length of the X dimension of the resized image. This option is mutually exclusive with resize_shorter. If the resize_y is left at 0, then the op will keep the aspect ratio of the original image.

    resize_y (float, optional, default = 0.0) – The length of the Y dimension of the resized image. This option is mutually exclusive with resize_shorter. If the resize_x is left at 0, then the op will keep the aspect ratio of the original image.

    save_attrs (bool, optional, default = False) – Save reshape attributes for testing.

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    temp_buffer_hint (int, optional, default = 0) – Initial size, in bytes, of a temporary buffer for resampling. Ingored for CPU variant.
    """

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "dest_width": resize_x, "dest_height": resize_y,
                     "is_output": False, "scaling_mode": scaling_mode, "max_size": max_size, "resize_shorter": resize_shorter, 
                     "resize_longer": resize_longer, "interpolation_type": interpolation_type }
    resized_image = b.Resize(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (resized_image)

def random_crop(*inputs, crop_area_factor=[0.08, 1], crop_aspect_ratio=[0.75, 1.333333],
            crop_pox_x=0, crop_pox_y=0, device = None):
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],"is_output": False,
                    "crop_area_factor": None, "crop_aspect_ratio": None, "crop_pos_x": None, "crop_pos_y": None, "num_of_attempts": 20}
    random_cropped_image = b.RandomCrop(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (random_cropped_image)

def rotate(*inputs, angle=None, axis=None, bytes_per_sample_hint= 0, fill_value = 0.0, interp_type = 1, keep_size = False,
            output_dtype = -1, preserve = False, seed = -1, size = None, device = None):
    """
    angle (float) – Angle, in degrees, by which the image is rotated. For 2D data, the rotation is counter-clockwise, assuming top-left corner at (0,0) For 3D data, the angle is a positive rotation around given axis

    axis (float or list of float, optional, default = []) – 3D only: axis around which to rotate. The vector does not need to be normalized, but must have non-zero length. Reversing the vector is equivalent to changing the sign of angle.

    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    fill_value (float, optional, default = 0.0) – Value used to fill areas that are outside source image. If not specified, source coordinates are clamped and the border pixel is repeated.

    interp_type (int, optional, default = 1) – Type of interpolation used.

    keep_size (bool, optional, default = False) – If True, original canvas size is kept. If False (default) and size is not set, then the canvas size is adjusted to acommodate the rotated image with least padding possible

    output_dtype (int, optional, default = -1) – Output data type. By default, same as input type

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    size (float or list of float, optional, default = []) – Output size, in pixels/points. Non-integer sizes are rounded to nearest integer. Channel dimension should be excluded (e.g. for RGB images specify (480,640), not (480,640,3).

    """
    # pybind call arguments
    angle = b.CreateFloatParameter(angle) if isinstance(angle, float) else angle
    kwargs_pybind = {"input_image0": inputs[0],"is_output": False,
                    "angle": angle, "dest_width": 0, "dest_height": 0}
    rotated_image = b.Rotate(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (rotated_image)

def saturation(*inputs, bytes_per_sample_hint=0,  saturation=1.0, image_type=0, preserve=False, seed = -1, device = None):
    """
    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    image_type (int, optional, default = 0) – The color space of input and output image

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    saturation (float, optional, default = 1.0) –

    Saturation change factor. Values >= 0 are supported. For example:

    0 - completely desaturated image

    1 - no change to image’s saturation

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    """
    # pybind call arguments
    saturation = b.CreateFloatParameter(saturation) if isinstance(saturation, float) else saturation
    kwargs_pybind = {"input_image0": inputs[0],
                     "is_output": False, "sat": saturation}
    saturated_image = b.Saturation(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (saturated_image)

def ssd_random_crop(*inputs, bytes_per_sample_hint=0, num_attempts=1.0, preserve=False, seed= -1, device = None):
    """
    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory
    num_attempts (int, optional, default = 1) – Number of attempts.
    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.
    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)
    """
    if(num_attempts == 1):
        _num_attempts = 20
    else:
        _num_attempts = num_attempts
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "is_output": False, "p_threshold": None,
                    "crop_area_factor": None, "crop_aspect_ratio": None, "crop_pos_x": None, "crop_pos_y": None, "num_of_attempts": _num_attempts}
    ssd_random_cropped_image = b.SSDRandomCrop(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (ssd_random_cropped_image)

def warp_affine(*inputs, bytes_per_sample_hint=0, fill_value=0.0, interp_type = 1, matrix = None, output_dtype = -1, preserve = False, seed = -1, size = None, device = None):
    """
    bytes_per_sample_hint (int, optional, default = 0) – Output size hint (bytes), per sample. The memory will be preallocated if it uses GPU or page-locked memory

    fill_value (float, optional, default = 0.0) – Value used to fill areas that are outside source image. If not specified, source coordinates are clamped and the border pixel is repeated.

    interp_type (int, optional, default = 1) – Type of interpolation used.

    matrix (float or list of float, optional, default = []) –

    Transform matrix (dst -> src). Given list of values (M11, M12, M13, M21, M22, M23) this operation will produce a new image using the following formula

    dst(x,y) = src(M11 * x + M12 * y + M13, M21 * x + M22 * y + M23)

    It is equivalent to OpenCV’s warpAffine operation with a flag WARP_INVERSE_MAP set.

    output_dtype (int, optional, default = -1) – Output data type. By default, same as input type

    preserve (bool, optional, default = False) – Do not remove the Op from the graph even if its outputs are unused.

    seed (int, optional, default = -1) – Random seed (If not provided it will be populated based on the global seed of the pipeline)

    size (float or list of float, optional, default = []) – Output size, in pixels/points. Non-integer sizes are rounded to nearest integer. Channel dimension should be excluded (e.g. for RGB images specify (480,640), not (480,640,3).
    """    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],"is_output": False,"dest_width": 0, "dest_height": 0, "x0": None, "x1": None, "y0": None, "y1": None, "o0": None, "o1": None}
    warp_affine_outputcolor_temp_output = b.WarpAffine(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (warp_affine_outputcolor_temp_output)

def vignette(*inputs, vignette=0.5, device=None):
    # pybind call arguments
    vignette = b.CreateFloatParameter(vignette) if isinstance(vignette, float) else vignette
    kwargs_pybind = {"input_image0": inputs[0],"is_output": False, "sdev": vignette}
    vignette_outputcolor_temp_output = b.Vignette(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (vignette_outputcolor_temp_output)

def crop_mirror_normalize(*inputs, bytes_per_sample_hint=0, crop=[0, 0], crop_d=0, crop_h=0, crop_pos_x=0.5, crop_pos_y=0.5, crop_pos_z=0.5,
                          crop_w=0, image_type=0, mean=[0.0], mirror=1, output_dtype=types.FLOAT, output_layout=types.NCHW, pad_output=False,
                          preserve=False, seed=1, std=[1.0], device=None):

    if(len(crop) == 2):
        crop_depth = crop_d
        crop_height = crop[0]
        crop_width = crop[1]
    elif(len(crop) == 3):
        crop_depth = crop[0]
        crop_height = crop[1]
        crop_width = crop[2]
    else:
        crop_depth = crop_d
        crop_height = crop_h
        crop_width = crop_w
    #Set Seed
    b.setSeed(seed)

    if isinstance(mirror,int):
        if(mirror == 0):
            mirror = b.CreateIntParameter(0)
        else:
            mirror = b.CreateIntParameter(1)

    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "crop_depth":crop_depth, "crop_height":crop_height, "crop_width":crop_width, "start_x":1, "start_y":1, "start_z":1, "mean":mean, "std_dev":std,
                     "is_output": False, "mirror": mirror}
    b.setSeed(seed)
    cmn = b.CropMirrorNormalize(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    Pipeline._current_pipeline._tensor_layout = output_layout
    Pipeline._current_pipeline._tensor_dtype = output_dtype
    Pipeline._current_pipeline._multiplier = list(map(lambda x: 1/x ,std))
    Pipeline._current_pipeline._offset = list(map(lambda x,y: -(x/y), mean, std))
    return (cmn)

def centre_crop(*inputs, bytes_per_sample_hint=0, crop=[100, 100], crop_d=1, crop_h= 0, crop_pos_x = 0.5, crop_pos_y = 0.5, crop_pos_z = 0.5,
                 crop_w=0, image_type=0, output_dtype=types.FLOAT, preserve = False, seed = 1, device = None):

    if(len(crop) == 2):
        crop_depth = crop_d
        crop_height = crop[0]
        crop_width = crop[1]
    elif(len(crop) == 3):
        crop_depth = crop[0]
        crop_height = crop[1]
        crop_width = crop[2]
    else:
        crop_depth = crop_d
        crop_height = crop_h
        crop_width = crop_w
    #Set Seed
    b.setSeed(seed)
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "crop_width":crop_width, "crop_height":crop_height, "crop_depth":crop_depth,
                     "is_output": False}
    centre_cropped_image = b.CenterCropFixed(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    return (centre_cropped_image)

def crop(*inputs, bytes_per_sample_hint=0, crop=[0.0, 0.0], crop_d=1, crop_h= 0, crop_pos_x = 0.5, crop_pos_y = 0.5, crop_pos_z = 0.5,
                 crop_w=0, image_type=0, output_dtype=types.FLOAT, preserve = False, seed = 1, device = None):

    if(len(crop) == 2):
        crop_depth = crop_d
        crop_height = crop[0]
        crop_width = crop[1]
    elif(len(crop) == 3):
        crop_depth = crop[0]
        crop_height = crop[1]
        crop_width = crop[2]
    else:
        crop_depth = crop_d
        crop_height = crop_h
        crop_width = crop_w
    #Set Seed
    b.setSeed(seed)
    if ((crop_width == 0) and (crop_height == 0)):
        # pybind call arguments
        kwargs_pybind = {"input_image0": inputs[0],"is_output": False, "crop_width":None, "crop_height":None, "crop_depth":None ,"crop_pos_x": None, "crop_pos_y": None, "crop_pos_z": None }
        cropped_image = b.Crop(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    else:
        # pybind call arguments
        kwargs_pybind = {"input_image0": inputs[0], "crop_width":crop_width, "crop_height":crop_height, "crop_depth":crop_depth ,"is_output": False,"crop_pos_x": crop_pos_x, "crop_pos_y": crop_pos_y, "crop_pos_z": crop_pos_z }
        cropped_image = b.CropFixed(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (cropped_image)

def color_twist(*inputs, brightness=1.0, bytes_per_sample_hint=0, contrast=1.0, hue=0.0, image_type=0,
                preserve=False, saturation=1.0, seed=-1, device=None):
    brightness = b.CreateFloatParameter(brightness) if isinstance(
        brightness, float) else brightness
    contrast = b.CreateFloatParameter(
        contrast) if isinstance(contrast, float) else contrast
    hue = b.CreateFloatParameter(hue) if isinstance(hue, float) else hue
    saturation = b.CreateFloatParameter(saturation) if isinstance(
        saturation, float) else saturation
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "is_output": False,
                     "alpha": brightness, "beta": contrast, "hue": hue, "sat": saturation}
    color_twist_image = b.ColorTwist(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (color_twist_image)

def uniform(*inputs,rng_range=[-1, 1], device=None):
    output_param = b.CreateFloatUniformRand(rng_range[0], rng_range[1])
    return output_param

def random_bbox_crop(*inputs,all_boxes_above_threshold = True, allow_no_crop =True, aspect_ratio = None, bbox_layout = "", bytes_per_sample_hint = 0,
                crop_shape = None, input_shape = None, ltrb = True, num_attempts = 1 ,scaling =  None,  preserve = False, seed = -1, shape_layout = "",
                threshold_type ="iou", thresholds = None, total_num_attempts = 0, device = None, labels = None ):
    aspect_ratio = aspect_ratio if aspect_ratio else [1.0, 1.0]
    crop_shape = [] if crop_shape is None else crop_shape
    scaling = scaling if scaling else [1.0, 1.0]
    if(len(crop_shape) == 0):
        has_shape = False
        crop_width = 0
        crop_height = 0
    else:
        has_shape = True
        crop_width = crop_shape[0]
        crop_height = crop_shape[1]
    scaling = b.CreateFloatUniformRand(scaling[0], scaling[1])
    aspect_ratio = b.CreateFloatUniformRand(aspect_ratio[0], aspect_ratio[1])

    # pybind call arguments
    kwargs_pybind = {"all_boxes_above_threshold":all_boxes_above_threshold, "no_crop": allow_no_crop, "p_aspect_ratio":aspect_ratio, "has_shape":has_shape, "crop_width":crop_width, "crop_height":crop_height, "num_attemps":num_attempts, "p_scaling":scaling, "total_num_attempts":total_num_attempts }
    random_bbox_crop = b.RandomBBoxCrop(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))

    return (random_bbox_crop,[],[],[])

def bb_flip(*inputs, bytes_per_sample_hint = 0, horizontal = 1, ltrb = False, preserve =  False,seed = -1, vertical = 0, device = None):
    # Dummy Node
    # In rocAL , we do not support just a change in the meta data seperatly .It has to be done in accordance with the augmentation nodes
    return []

def one_hot(*inputs, bytes_per_sample_hint=0, dtype=types.FLOAT, num_classes=0, off_value=0.0,
            on_value=1.0, preserve=False, seed=-1,  device=None):
    Pipeline._current_pipeline._numOfClasses = num_classes
    Pipeline._current_pipeline._oneHotEncoding = True
    return ([])

def box_encoder(*inputs, anchors, bytes_per_sample_hint=0, criteria=0.5, means=None, offset=False, preserve=False, scale=1.0, seed=-1, stds=None ,device = None):
    means = means if means else [0.0, 0.0, 0.0, 0.0]
    stds = stds if stds else [1.0, 1.0, 1.0, 1.0]
    kwargs_pybind ={"anchors":anchors, "criteria":criteria, "means":means, "stds":stds, "offset":offset, "scale":scale}
    box_encoder = b.BoxEncoder(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    Pipeline._current_pipeline._BoxEncoder = True
    return (box_encoder , [])

def color_temp(*inputs, adjustment_value=50, device=None, preserve = False):
    # pybind call arguments
    adjustment_value = b.CreateIntParameter(adjustment_value) if isinstance(adjustment_value, int) else adjustment_value
    kwargs_pybind = {"input_image0": inputs[0],"is_output": False, "adjustment_value": adjustment_value}
    color_temp_output = b.ColorTemp(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (color_temp_output)

def nop(*inputs, device=None, preserve = False):
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],"is_output": False }
    nop_output = b.rocalNop(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (nop_output)

def copy(*inputs, device=None):
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0],"is_output": False }
    copied_image = b.rocalCopy(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (copied_image)

def snp_noise(*inputs, snpNoise=None, device=None, preserve = False):
    # pybind call arguments
    snpNoise = b.CreateFloatParameter(snpNoise) if isinstance(snpNoise, float) else snpNoise
    kwargs_pybind = {"input_image0":inputs[0], "is_output":False ,"snpNoise": snpNoise}
    snp_noise_added_image = b.SnPNoise(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (snp_noise_added_image)
