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



# RocalStatus
from rocal_pybind.types import OK
from rocal_pybind.types import CONTEXT_INVALID
from rocal_pybind.types import RUNTIME_ERROR
from rocal_pybind.types import UPDATE_PARAMETER_FAILED
from rocal_pybind.types import INVALID_PARAMETER_TYPE

#  RocalProcessMode
from rocal_pybind.types import GPU
from rocal_pybind.types import CPU

#  RocalTensorOutputType
from rocal_pybind.types import FLOAT
from rocal_pybind.types import FLOAT16

# RocalImageSizeEvaluationPolicy
from rocal_pybind.types import MAX_SIZE
from rocal_pybind.types import USER_GIVEN_SIZE
from rocal_pybind.types import MOST_FREQUENT_SIZE
from rocal_pybind.types import MAX_SIZE_ORIG
from rocal_pybind.types import USER_GIVEN_SIZE_ORIG

#      RocalImageColor
from rocal_pybind.types import RGB
from rocal_pybind.types import BGR
from rocal_pybind.types import GRAY
from rocal_pybind.types import RGB_PLANAR

#     RocalTensorLayout
from rocal_pybind.types import NHWC
from rocal_pybind.types import NCHW

#     RocalDecodeDevice
from rocal_pybind.types import HARDWARE_DECODE
from rocal_pybind.types import SOFTWARE_DECODE

#     RocalDecodeDevice
from rocal_pybind.types import DECODER_TJPEG
from rocal_pybind.types import DECODER_OPENCV
from rocal_pybind.types import DECODER_HW_JEPG
from rocal_pybind.types import DECODER_VIDEO_FFMPEG_SW
from rocal_pybind.types import DECODER_VIDEO_FFMPEG_HW

#     RocalResizeScalingMode
from rocal_pybind.types import SCALING_MODE_DEFAULT
from rocal_pybind.types import SCALING_MODE_STRETCH
from rocal_pybind.types import SCALING_MODE_NOT_SMALLER
from rocal_pybind.types import SCALING_MODE_NOT_LARGER

#     RocalResizeInterpolationType
from rocal_pybind.types import NEAREST_NEIGHBOR_INTERPOLATION
from rocal_pybind.types import LINEAR_INTERPOLATION
from rocal_pybind.types import CUBIC_INTERPOLATION
from rocal_pybind.types import LANCZOS_INTERPOLATION
from rocal_pybind.types import GAUSSIAN_INTERPOLATION
from rocal_pybind.types import TRIANGULAR_INTERPOLATION

_known_types = {

    OK: ("OK", OK),
    CONTEXT_INVALID: ("CONTEXT_INVALID", CONTEXT_INVALID),
    RUNTIME_ERROR: ("RUNTIME_ERROR", RUNTIME_ERROR),
    UPDATE_PARAMETER_FAILED: ("UPDATE_PARAMETER_FAILED", UPDATE_PARAMETER_FAILED),
    INVALID_PARAMETER_TYPE: ("INVALID_PARAMETER_TYPE", INVALID_PARAMETER_TYPE),

    GPU: ("GPU", GPU),
    CPU: ("CPU", CPU),
    FLOAT: ("FLOAT", FLOAT),
    FLOAT16: ("FLOAT16", FLOAT16),

    MAX_SIZE: ("MAX_SIZE", MAX_SIZE),
    USER_GIVEN_SIZE: ("USER_GIVEN_SIZE", USER_GIVEN_SIZE),
    MOST_FREQUENT_SIZE: ("MOST_FREQUENT_SIZE", MOST_FREQUENT_SIZE),
    MAX_SIZE_ORIG: ("MAX_SIZE_ORIG", MAX_SIZE_ORIG),
    USER_GIVEN_SIZE_ORIG: ("USER_GIVEN_SIZE_ORIG", USER_GIVEN_SIZE_ORIG),

    NHWC: ("NHWC", NHWC),
    NCHW: ("NCHW", NCHW),
    BGR: ("BGR", BGR),
    RGB: ("RGB", RGB),
    GRAY: ("GRAY", GRAY),
    RGB_PLANAR: ("RGB_PLANAR", RGB_PLANAR),

    HARDWARE_DECODE: ("HARDWARE_DECODE", HARDWARE_DECODE),
    SOFTWARE_DECODE: ("SOFTWARE_DECODE", SOFTWARE_DECODE),

    DECODER_TJPEG: ("DECODER_TJPEG", DECODER_TJPEG),
    DECODER_OPENCV: ("DECODER_OPENCV", DECODER_OPENCV),
    DECODER_HW_JEPG: ("DECODER_HW_JEPG", DECODER_HW_JEPG),
    DECODER_VIDEO_FFMPEG_SW: ("DECODER_VIDEO_FFMPEG_SW", DECODER_VIDEO_FFMPEG_SW),
    DECODER_VIDEO_FFMPEG_HW: ("DECODER_VIDEO_FFMPEG_HW", DECODER_VIDEO_FFMPEG_HW),

    NEAREST_NEIGHBOR_INTERPOLATION: ("NEAREST_NEIGHBOR_INTERPOLATION", NEAREST_NEIGHBOR_INTERPOLATION),
    LINEAR_INTERPOLATION: ("LINEAR_INTERPOLATION", LINEAR_INTERPOLATION),
    CUBIC_INTERPOLATION: ("CUBIC_INTERPOLATION", CUBIC_INTERPOLATION),
    LANCZOS_INTERPOLATION: ("LANCZOS_INTERPOLATION", LANCZOS_INTERPOLATION),
    GAUSSIAN_INTERPOLATION: ("GAUSSIAN_INTERPOLATION", GAUSSIAN_INTERPOLATION),
    TRIANGULAR_INTERPOLATION: ("TRIANGULAR_INTERPOLATION", TRIANGULAR_INTERPOLATION),

    SCALING_MODE_DEFAULT: ("SCALING_MODE_DEFAULT", SCALING_MODE_DEFAULT),
    SCALING_MODE_STRETCH: ("SCALING_MODE_STRETCH", SCALING_MODE_STRETCH),
    SCALING_MODE_NOT_SMALLER: ("SCALING_MODE_NOT_SMALLER", SCALING_MODE_NOT_SMALLER),
    SCALING_MODE_NOT_LARGER: ("SCALING_MODE_NOT_LARGER", SCALING_MODE_NOT_LARGER),

}

def data_type_function(dtype):
    if dtype in _known_types:
        ret = _known_types[dtype][0]
        return ret
    else:
        raise RuntimeError(str(dtype) + " does not correspond to a known type.")
