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

import ctypes

from rali_common import *
class RaliImage:
    def __init__(self, obj):
        self.sharedlib = RALI_LIB_NAME
        dllabspath = self.sharedlib
        self.lib = ctypes.cdll.LoadLibrary(dllabspath)
        self.get_width = self.lib.raliGetImageWidth
        self.get_width.restype = ctypes.c_int
        self.get_width.argtypes = [ctypes.c_void_p]
        self.get_height = self.lib.raliGetImageHeight
        self.get_height.restype = ctypes.c_int
        self.get_height.argtypes = [ctypes.c_void_p]
        self.get_planes = self.lib.raliGetImagePlanes
        self.get_planes.restype = ctypes.c_int
        self.get_planes.argtypes = [ctypes.c_void_p]

        self.labels = []
        self.obj = obj

    def shape(self):
        return self.get_width(self.obj), self.get_height(self.obj), self.get_planes(self.obj)
