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
