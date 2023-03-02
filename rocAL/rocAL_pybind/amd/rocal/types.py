


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

# Rocal Extrernal Source Mode
from rocal_pybind.types import EXTSOURCE_FNAME
from rocal_pybind.types import EXTSOURCE_RAW_COMPRESSED
from rocal_pybind.types import EXTSOURCE_RAW_UNCOMPRESSED

_known_types={


	OK : ("OK", OK),
    CONTEXT_INVALID : ("CONTEXT_INVALID", CONTEXT_INVALID),
	RUNTIME_ERROR : ("RUNTIME_ERROR", RUNTIME_ERROR),
    UPDATE_PARAMETER_FAILED : ("UPDATE_PARAMETER_FAILED", UPDATE_PARAMETER_FAILED),
	INVALID_PARAMETER_TYPE : ("INVALID_PARAMETER_TYPE", INVALID_PARAMETER_TYPE),

	GPU : ("GPU", GPU),
    CPU : ("CPU", CPU),
	FLOAT : ("FLOAT", FLOAT),
    FLOAT16 : ("FLOAT16", FLOAT16),


	MAX_SIZE : ("MAX_SIZE", MAX_SIZE),
    USER_GIVEN_SIZE : ("USER_GIVEN_SIZE", USER_GIVEN_SIZE),
	MOST_FREQUENT_SIZE : ("MOST_FREQUENT_SIZE", MOST_FREQUENT_SIZE),
    MAX_SIZE_ORIG : ("MAX_SIZE_ORIG", MAX_SIZE_ORIG),
    USER_GIVEN_SIZE_ORIG : ("USER_GIVEN_SIZE_ORIG", USER_GIVEN_SIZE_ORIG),

	NHWC : ("NHWC", NHWC),
    NCHW : ("NCHW", NCHW),
	BGR : ("BGR", BGR),
    RGB : ("RGB", RGB),
	GRAY : ("GRAY", GRAY),
    RGB_PLANAR : ("RGB_PLANAR", RGB_PLANAR),

    HARDWARE_DECODE : ("HARDWARE_DECODE", HARDWARE_DECODE),
    SOFTWARE_DECODE : ("SOFTWARE_DECODE", SOFTWARE_DECODE),

    DECODER_TJPEG : ("DECODER_TJPEG", DECODER_TJPEG),
    DECODER_OPENCV : ("DECODER_OPENCV", DECODER_OPENCV),
    DECODER_HW_JEPG : ("DECODER_HW_JEPG", DECODER_HW_JEPG),
    DECODER_VIDEO_FFMPEG_SW : ("DECODER_VIDEO_FFMPEG_SW", DECODER_VIDEO_FFMPEG_SW),
    DECODER_VIDEO_FFMPEG_HW : ("DECODER_VIDEO_FFMPEG_HW", DECODER_VIDEO_FFMPEG_HW),

    EXTSOURCE_FNAME : ("EXTSOURCE_FNAME", EXTSOURCE_FNAME),
    EXTSOURCE_RAW_COMPRESSED : ("EXTSOURCE_RAW_COMPRESSED", EXTSOURCE_RAW_COMPRESSED),
    EXTSOURCE_RAW_UNCOMPRESSED : ("EXTSOURCE_RAW_UNCOMPRESSED", EXTSOURCE_RAW_UNCOMPRESSED),
}


def data_type_function(dtype):
    if dtype in _known_types:
        ret = _known_types[dtype][0]
        return ret
    else:
        raise RuntimeError(str(dtype) + " does not correspond to a known type.")
