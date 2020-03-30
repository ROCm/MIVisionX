from rali_pybind.types import *
# from rali_pybind.types import RaliStatus
# from rali_pybind.types import RaliProcessMode
# from rali_pybind.types import RaliTensorOutputType
# from rali_pybind.types import RaliImageSizeEvaluationPolicy
# from rali_pybind.types import RaliImageColor
# from rali_pybind.types import RaliTensorLayout

from enum import IntEnum
class TensorDataType(IntEnum):
	FLOAT = 0
	FLOAT16 = 1