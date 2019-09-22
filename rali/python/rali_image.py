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

        self.get_name = self.lib.raliGetImageName
        self.get_name.restype = ctypes.c_void_p
        self.get_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]

        self.get_name_size = self.lib.raliGetImageNameLen
        self.get_name_size.restype = ctypes.c_uint
        self.get_name_size.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self.labels = []
        self.obj = obj

    def shape(self):
        return self.get_width(self.obj), self.get_height(self.obj), self.get_planes(self.obj)

    def name(self, idx):
        size = self.get_name_size(self.obj, idx)
        ret = ctypes.create_string_buffer(size)
        self.get_name(self.obj, ret, idx)
        return ret.value

    def get_labels(self):
        return self.labels

    def set_labels(self, labels):
        self.labels = labels
