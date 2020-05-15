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

class RaliIntParameter:
    def __init__(self, value):
        self.sharedlib = RALI_LIB_NAME
        dllabspath = self.sharedlib
        self.lib = ctypes.cdll.LoadLibrary(dllabspath)
        self.create_obj = self.lib.raliCreateIntParameter
        self.create_obj.restype = ctypes.c_void_p
        self.create_obj.argtypes = [ctypes.c_int]
        self.update_obj = self.lib.raliUpdateIntParameter
        self.update_obj.restype = ctypes.c_int
        self.update_obj.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self.latest_gen = self.lib.raliGetIntValue
        self.latest_gen.restype = ctypes.c_int
        self.latest_gen.argtypes = [ctypes.c_void_p]
        self.obj = self.create_obj(value)

    def update(self, value):
        ret = self.update_obj(value, self.obj)
        if(ret != 0):
            raise Exception('FAILED updating the random variable')

    def get(self):
        return self.latest_gen(self.obj)


class RaliFloatParameter:
    def __init__(self, value):
        self.sharedlib = RALI_LIB_NAME
        dllabspath = self.sharedlib
        self.lib = ctypes.cdll.LoadLibrary(dllabspath)
        self.create_obj = self.lib.raliCreateFloatParameter
        self.create_obj.restype = ctypes.c_void_p
        self.create_obj.argtypes = [ctypes.c_float]
        self.update_obj = self.lib.raliUpdateFloatParameter
        self.update_obj.restype = ctypes.c_int
        self.update_obj.argtypes = [ctypes.c_float, ctypes.c_void_p]
        self.latest_gen = self.lib.raliGetFloatValue
        self.latest_gen.restype = ctypes.c_float
        self.latest_gen.argtypes = [ctypes.c_void_p]
        self.obj = self.create_obj(value)

    def update(self, value):
        ret = self.update_obj(value, self.obj)
        if(ret != 0):
            raise Exception('FAILED updating the random variable')

    def get(self):
        return self.latest_gen(self.obj)


class RaliIntUniformRand:
    def __init__(self, start, end):
        self.sharedlib = RALI_LIB_NAME
        dllabspath = self.sharedlib
        self.lib = ctypes.cdll.LoadLibrary(dllabspath)
        self.create_obj = self.lib.raliCreateIntUniformRand
        self.create_obj.restype = ctypes.c_void_p
        self.create_obj.argtypes = [ctypes.c_int, ctypes.c_int]
        self.update_obj = self.lib.raliUpdateIntUniformRand
        self.update_obj.restype = ctypes.c_int
        self.update_obj.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.latest_gen = self.lib.raliGetIntValue
        self.latest_gen.restype = ctypes.c_int
        self.latest_gen.argtypes = [ctypes.c_void_p]
        self.obj = self.create_obj(start, end)

    def update(self, start, end):
        ret = self.update_obj(start, end, self.obj)
        if(ret != 0):
            raise Exception('FAILED updating the random variable')

    def get(self):
        return self.latest_gen(self.obj)


class RaliFloatUniformRand:
    def __init__(self, start, end):
        self.sharedlib = RALI_LIB_NAME
        dllabspath = self.sharedlib
        self.lib = ctypes.cdll.LoadLibrary(dllabspath)
        self.create_obj = self.lib.raliCreateFloatUniformRand
        self.create_obj.restype = ctypes.c_void_p
        self.create_obj.argtypes = [ctypes.c_float, ctypes.c_float]
        self.update_obj = self.lib.raliUpdateFloatUniformRand
        self.update_obj.restype = ctypes.c_int
        self.update_obj.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_void_p]
        self.latest_gen = self.lib.raliGetFloatValue
        self.latest_gen.restype = ctypes.c_float
        self.latest_gen.argtypes = [ctypes.c_void_p]
        self.obj = self.create_obj(start, end)

    def update(self, start, end):
        ret = self.update_obj(start, end, self.obj)
        if(ret != 0):
            raise Exception('FAILED updating the random variable')

    def get(self):
        return self.latest_gen(self.obj)
