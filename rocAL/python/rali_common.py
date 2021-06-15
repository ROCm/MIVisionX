# Copyright (c) 2018 - 2020 Advanced Micro Devices, Inc. All rights reserved.
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

RALI_LIB_NAME = 'librali.so'
from enum import Enum
from enum import IntEnum
class ColorFormat(Enum):
    IMAGE_RGB24 = 0
    IMAGE_BGR24 = 1
    IMAGE_U8 = 2
    IMAGE_RGB_PLANAR = 3

class Affinity(Enum):
    PROCESS_GPU = 0
    PROCESS_CPU = 1

class TensorLayout(Enum):
    NHWC = 0
    NCHW = 1
class TensorDataType(IntEnum):
	FLOAT32 = 0
	FLOAT16 = 1
