# Copyright (c) 2018 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

import rocal_pybind as b
import amd.rocal.types as types
import numpy as np
import cupy as cp
import ctypes
import functools
import inspect


class Pipeline(object):

    """Pipeline class internally calls RocalCreate which returns context which will have all
    the info set by the user.

    Parameters
    ----------
    `batch_size` : int, optional, default = -1
        Batch size of the pipeline. Negative values for this parameter
        are invalid - the default value may only be used with
        serialized pipeline (the value stored in serialized pipeline
        is used instead).
    `num_threads` : int, optional, default = -1
        Number of CPU threads used by the pipeline.
        Negative values for this parameter are invalid - the default
        value may only be used with serialized pipeline (the value
        stored in serialized pipeline is used instead).
    `device_id` : int, optional, default = -1
        Id of GPU used by the pipeline.
        Negative values for this parameter are invalid - the default
        value may only be used with serialized pipeline (the value
        stored in serialized pipeline is used instead).
    `seed` : int, optional, default = -1
        Seed used for random number generation. Leaving the default value
        for this parameter results in random seed.
    `exec_pipelined` : bool, optional, default = True
        Whether to execute the pipeline in a way that enables
        overlapping CPU and GPU computation, typically resulting
        in faster execution speed, but larger memory consumption.
    `prefetch_queue_depth` : int or {"cpu_size": int, "gpu_size": int}, optional, default = 2
        Depth of the executor pipeline. Deeper pipeline makes ROCAL
        more resistant to uneven execution time of each batch, but it
        also consumes more memory for internal buffers.
        Specifying a dict:
        ``{ "cpu_size": x, "gpu_size": y }``
        instead of an integer will cause the pipeline to use separated
        queues executor, with buffer queue size `x` for cpu stage
        and `y` for mixed and gpu stages. It is not supported when both `exec_async`
        and `exec_pipelined` are set to `False`.
        Executor will buffer cpu and gpu stages separatelly,
        and will fill the buffer queues when the first :meth:`amd.rocal.pipeline.Pipeline.run`
        is issued.
    `exec_async` : bool, optional, default = True
        Whether to execute the pipeline asynchronously.
        This makes :meth:`amd.rocal.pipeline.Pipeline.run` method
        run asynchronously with respect to the calling Python thread.
    `bytes_per_sample` : int, optional, default = 0
        A hint for ROCAL for how much memory to use for its tensors.
    `set_affinity` : bool, optional, default = False
        Whether to set CPU core affinity to the one closest to the
        GPU being used.
    `max_streams` : int, optional, default = -1
        Limit the number of CUDA streams used by the executor.
        Value of -1 does not impose a limit.
        This parameter is currently unused (and behavior of
        unrestricted number of streams is assumed).
    `default_cuda_stream_priority` : int, optional, default = 0
        CUDA stream priority used by ROCAL. See `cudaStreamCreateWithPriority` in CUDA documentation
    """
    '''.
    Args: batch_size
          rocal_cpu
          gpu_id (default 0)
          cpu_threads (default 1)
    This returns a context'''
    _handle = None
    _current_pipeline = None

    def __init__(self, batch_size=-1, num_threads=0, device_id=-1, seed=1,
                 exec_pipelined=True, prefetch_queue_depth=2,
                 exec_async=True, bytes_per_sample=0,
                 rocal_cpu=False, max_streams=-1, default_cuda_stream_priority=0, tensor_layout = types.NCHW, reverse_channels = False, mean = None, std = None, tensor_dtype=types.FLOAT, output_memory_type = types.CPU_MEMORY):
        if(rocal_cpu):
            self._handle = b.rocalCreate(
                batch_size, types.CPU, device_id, num_threads,prefetch_queue_depth,types.FLOAT)
        else:
            self._handle = b.rocalCreate(
                batch_size, types.GPU, device_id, num_threads,prefetch_queue_depth,types.FLOAT)

        if(b.getStatus(self._handle) == types.OK):
            print("Pipeline has been created succesfully")
        else:
            raise Exception("Failed creating the pipeline")
        self._check_ops = ["CropMirrorNormalize"]
        self._check_crop_ops = ["Resize"]
        self._check_ops_decoder = ["ImageDecoder", "ImageDecoderSlice" , "ImageDecoderRandomCrop", "ImageDecoderRaw"]
        self._check_ops_reader = ["labelReader", "TFRecordReaderClassification", "TFRecordReaderDetection",
            "COCOReader", "Caffe2Reader", "Caffe2ReaderDetection", "CaffeReader", "CaffeReaderDetection"]
        self._batch_size = batch_size
        self._num_threads = num_threads
        self._device_id = device_id
        self._output_memory_type = output_memory_type
        self._seed = seed
        self._exec_pipelined = exec_pipelined
        self._prefetch_queue_depth = prefetch_queue_depth
        self._exec_async = exec_async
        self._bytes_per_sample = bytes_per_sample
        self._rocal_cpu = rocal_cpu
        self._max_streams = max_streams
        self._default_cuda_stream_priority = default_cuda_stream_priority
        self._tensor_layout = tensor_layout
        self._tensor_dtype = tensor_dtype
        self._multiplier = list(map(lambda x: 1/x , std)) if std else [1.0,1.0,1.0]
        self._offset = list(map(lambda x, y: -(x/y), mean, std)) if mean and std else [0.0, 0.0, 0.0]
        self._reverse_channels = reverse_channels
        self._img_h = None
        self._img_w = None
        self._shuffle = None
        self._name = None
        self._anchors = None
        self._BoxEncoder = None
        self._encode_tensor = None
        self._numOfClasses = None
        self._oneHotEncoding = False
        self._castLabels = False
        self._current_pipeline = None
        self._reader = None
        self._define_graph_set = False
        self.set_seed(self._seed)

    def build(self):
        """Build the pipeline using rocalVerify call
        """
        status = b.rocalVerify(self._handle)
        if(status != types.OK):
            print("Verify graph failed")
            exit(0)
        return self

    def run(self):
        """ Run the pipeline using rocalRun call
        """
        status = b.rocalRun(self._handle)
        if(status != types.OK):
            print("Rocal Run failed")
        return status

    def define_graph(self):
        """This function is defined by the user to construct the
        graph of operations for their pipeline.
        It returns a list of outputs created by calling ROCAL Operators."""
        print("definegraph is deprecated")
        raise NotImplementedError

    def get_handle(self):
        return self._handle

    def copyImage(self, array):
        out = np.frombuffer(array, dtype=array.dtype)
        b.rocalCopyToOutput(
            self._handle, np.ascontiguousarray(out, dtype=array.dtype))

    def copyToExternalTensor(self, array,  multiplier, offset, reverse_channels, tensor_format, tensor_dtype):

        b.rocalToTensor(self._handle, ctypes.c_void_p(array.data_ptr()), tensor_format, tensor_dtype,
                                    multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0), self._output_memory_type)

    def copyToExternalTensorNHWC(self, array,  multiplier, offset, reverse_channels, tensor_dtype):
        if(self._rocal_cpu):
            out = np.frombuffer(array, dtype=array.dtype)
            if tensor_dtype == types.FLOAT:
                b.rocalToTensor32(self._handle, np.ascontiguousarray(out, dtype=array.dtype), types.NHWC,
                                        multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0), self._output_memory_type)
            elif tensor_dtype == types.FLOAT16:
                b.rocalToTensor16(self._handle, np.ascontiguousarray(out, dtype=array.dtype), types.NHWC,
                                       multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0), self._output_memory_type)
        else:
            if tensor_dtype == types.FLOAT:
                b.rocalCupyToTensor32(self._handle, array.data.ptr, types.NHWC,
                                        multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0), self._output_memory_type)
            elif tensor_dtype == types.FLOAT16:
                b.rocalCupyToTensor16(self._handle, ctypes.c_void_p(array.ctypes.data), types.NHWC,
                                       multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0), self._output_memory_type)
    def copyToExternalTensorNCHW(self, array,  multiplier, offset, reverse_channels, tensor_dtype):
        if(self._rocal_cpu):
            out = np.frombuffer(array, dtype=array.dtype)
            if tensor_dtype == types.FLOAT:
                b.rocalToTensor32(self._handle, np.ascontiguousarray(out, dtype=array.dtype), types.NCHW,
                                        multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0), self._output_memory_type)
            elif tensor_dtype == types.FLOAT16:
                b.rocalToTensor16(self._handle, np.ascontiguousarray(out, dtype=array.dtype), types.NCHW,
                                        multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0), self._output_memory_type)
        else:
            if tensor_dtype == types.FLOAT:
                b.rocalCupyToTensor32(self._handle, array.data.ptr, types.NCHW,
                                        multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0), self._output_memory_type)
            elif tensor_dtype == types.FLOAT16:
                b.rocalCupyToTensor16(self._handle, ctypes.c_void_p(array.ctypes.data), types.NCHW,
                                        multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0), self._output_memory_type)
    def GetOneHotEncodedLabels(self, array, device):
        if device=="cpu":
            if (isinstance(array,np.ndarray)):
                b.getOneHotEncodedLabels(self._handle, array.ctypes.data_as(ctypes.c_void_p), self._numOfClasses, 0)
            else: #torch tensor
                return b.getOneHotEncodedLabels(self._handle, ctypes.c_void_p(array.data_ptr()), self._numOfClasses, 0)
        else:
            if (isinstance(array,cp.ndarray)):
                b.getCupyOneHotEncodedLabels(self._handle, array.data.ptr, self._numOfClasses, 1)
            else: #torch tensor
                return b.getOneHotEncodedLabels(self._handle, ctypes.c_void_p(array.data_ptr()), self._numOfClasses, 1)

    def set_outputs(self, *output_list):
        self._output_list_length = len(output_list)
        b.setOutputImages(self._handle,len(output_list),output_list)

    def __enter__(self):
        Pipeline._current_pipeline = self
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def set_seed(self,seed=0):
        return b.setSeed(seed)

    @classmethod
    def create_int_param(self,value=1):
        return b.CreateIntParameter(value)

    @classmethod
    def create_float_param(self,value=1):
        return b.CreateFloatParameter(value)

    @classmethod
    def update_int_param(self,value=1,param=1):
        b.UpdateIntParameter(value,param)

    @classmethod
    def update_float_param(self,value=1,param=1):
        b.UpdateFloatParameter(value,param)

    @classmethod
    def get_int_value(self,param):
        return b.GetIntValue(param)

    @classmethod
    def get_float_value(self,param):
        return b.GetFloatValue(param)

    def GetImageNameLen(self, array):
        return b.getImageNameLen(self._handle, array)

    def GetImageName(self, array_len):
        return b.getImageName(self._handle,array_len)

    def GetImageId(self, array):
        b.getImageId(self._handle, array)

    def GetBoundingBoxCount(self, array):
        return b.getBoundingBoxCount(self._handle, array)

    def GetBBLabels(self, array):
        return b.getBBLabels(self._handle, array)

    def GetBBCords(self, array):
        return b.getBBCords(self._handle, array)

    def getImageLabels(self, array):
        if (isinstance(array,np.ndarray)):
            b.getImageLabels(self._handle, array.ctypes.data_as(ctypes.c_void_p), self._output_memory_type)
        elif (isinstance(array,cp.ndarray)):
            # Not passing copy_labels_to_device since if cupy arrays are already device memory
            b.getCupyImageLabels(self._handle, array.data.ptr, self._output_memory_type)
        else: #pytorch tensor
            b.getImageLabels(self._handle, ctypes.c_void_p(array.data_ptr()), self._output_memory_type)

    def copyEncodedBoxesAndLables(self, bbox_array, label_array):
        b.rocalCopyEncodedBoxesAndLables(self._handle, bbox_array, label_array)

    def getEncodedBoxesAndLables(self, batch_size, num_anchors):
        return b.rocalGetEncodedBoxesAndLables(self._handle, batch_size, num_anchors)

    def GetImgSizes(self, array):
        return b.getImgSizes(self._handle, array)

    def GetBoundingBox(self,array):
        return array

    def GetImageNameLength(self,idx):
        return b.getImageNameLen(self._handle,idx)

    def getOutputWidth(self):
        return b.getOutputWidth(self._handle)

    def getOutputHeight(self):
        return b.getOutputHeight(self._handle)

    def getOutputImageCount(self):
        return b.getOutputImageCount(self._handle)

    def getOutputColorFormat(self):
        return b.getOutputColorFormat(self._handle)

    def getRemainingImages(self):
        return b.getRemainingImages(self._handle)

    def rocalResetLoaders(self):
        return b.rocalResetLoaders(self._handle)

    def isEmpty(self):
        return b.isEmpty(self._handle)

    def Timing_Info(self):
        return b.getTimingInfo(self._handle)

def _discriminate_args(func, **func_kwargs):
    """Split args on those applicable to Pipeline constructor and the decorated function."""
    func_argspec = inspect.getfullargspec(func)
    ctor_argspec = inspect.getfullargspec(Pipeline.__init__)

    if 'debug' not in func_argspec.args and 'debug' not in func_argspec.kwonlyargs:
        func_kwargs.pop('debug', False)

    ctor_args = {}
    fn_args = {}

    if func_argspec.varkw is not None:
        raise TypeError(
            f"Using variadic keyword argument `**{func_argspec.varkw}` in a  "
            f"graph-defining function is not allowed.")

    for farg in func_kwargs.items():
        is_ctor_arg = farg[0] in ctor_argspec.args or farg[0] in ctor_argspec.kwonlyargs
        is_fn_arg = farg[0] in func_argspec.args or farg[0] in func_argspec.kwonlyargs
        if is_fn_arg:
            fn_args[farg[0]] = farg[1]
            if is_ctor_arg:
                print(
                    "Warning: the argument `{farg[0]}` shadows a Pipeline constructor "
                    "argument of the same name.")
        elif is_ctor_arg:
            ctor_args[farg[0]] = farg[1]
        else:
            assert False, f"This shouldn't happen. Please double-check the `{farg[0]}` argument"

    return ctor_args, fn_args


def pipeline_def(fn=None, **pipeline_kwargs):
    """
    Decorator that converts a graph definition function into a rocAL pipeline factory.

    A graph definition function is a function that returns intended pipeline outputs.
    You can decorate this function with ``@pipeline_def``::

        @pipeline_def
        def my_pipe(flip_vertical, flip_horizontal):
            ''' Creates a rocAL pipeline, which returns flipped and original images '''
            data, _ = fn.readers.file(file_root=images_dir)
            img = fn.decoders.image(data, device="mixed")
            flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
            return flipped, img

    The decorated function returns a rocAL Pipeline object::

        pipe = my_pipe(True, False)
        # pipe.build()  # the pipeline is not configured properly yet

    A pipeline requires additional parameters such as batch size, number of worker threads,
    GPU device id and so on (see :meth:`amd.rocal.Pipeline()` for a
    complete list of pipeline parameters).
    These parameters can be supplied as additional keyword arguments,
    passed to the decorated function::

        pipe = my_pipe(True, False, batch_size=32, num_threads=1, device_id=0)
        pipe.build()  # the pipeline is properly configured, we can build it now

    The outputs from the original function became the outputs of the Pipeline::

        flipped, img = pipe.run()

    When some of the pipeline parameters are fixed, they can be specified by name in the decorator::

        @pipeline_def(batch_size=42, num_threads=3)
        def my_pipe(flip_vertical, flip_horizontal):
            ...

    Any Pipeline constructor parameter passed later when calling the decorated function will
    override the decorator-defined params::

        @pipeline_def(batch_size=32, num_threads=3)
        def my_pipe():
            data = fn.external_source(source=my_generator)
            return data

        pipe = my_pipe(batch_size=128)  # batch_size=128 overrides batch_size=32

    .. warning::

        The arguments of the function being decorated can shadow pipeline constructor arguments -
        in which case there's no way to alter their values.

    .. note::

        Using ``**kwargs`` (variadic keyword arguments) in graph-defining function is not allowed.
        They may result in unwanted, silent hijacking of some arguments of the same name by
        Pipeline constructor. Code written this way would cease to work with future versions of rocAL
        when new parameters are added to the Pipeline constructor.

    To access any pipeline arguments within the body of a ``@pipeline_def`` function, the function
    :meth:`amd.rocal.Pipeline.current()` can be used:: ( note: this is not supported yet)

        @pipeline_def()
        def my_pipe():
            pipe = Pipeline.current()
            batch_size = pipe.batch_size
            num_threads = pipe.num_threads
            ...

        pipe = my_pipe(batch_size=42, num_threads=3)
        ...
    """

    def actual_decorator(func):

        @functools.wraps(func)
        def create_pipeline(*args, **kwargs):
            ctor_args, fn_kwargs = _discriminate_args(func, **kwargs)
            pipe = Pipeline(**{**pipeline_kwargs, **ctor_args})  # Merge and overwrite dict
            with pipe:
                pipe_outputs = func(*args, **fn_kwargs)
                if isinstance(pipe_outputs, tuple):
                    po = pipe_outputs
                elif pipe_outputs is None:
                    po = ()
                else:
                    po = (pipe_outputs, )
                pipe.set_outputs(*po)
            return pipe

        # Add `is_pipeline_def` attribute to the function marked as `@pipeline_def`
        create_pipeline._is_pipeline_def = True
        return create_pipeline

    return actual_decorator(fn) if fn else actual_decorator
