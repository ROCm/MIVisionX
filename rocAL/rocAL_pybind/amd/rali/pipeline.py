from amd.rali.global_cfg import MetaDataNode, Node
from numpy.core.fromnumeric import trace 
import rali_pybind as b
import amd.rali.types as types
import numpy as np
import torch



class Pipeline(object):

    """Pipeline class internally calls RaliCreate which returns context which will have all
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
        Depth of the executor pipeline. Deeper pipeline makes RALI
        more resistant to uneven execution time of each batch, but it
        also consumes more memory for internal buffers.
        Specifying a dict:
        ``{ "cpu_size": x, "gpu_size": y }``
        instead of an integer will cause the pipeline to use separated
        queues executor, with buffer queue size `x` for cpu stage
        and `y` for mixed and gpu stages. It is not supported when both `exec_async`
        and `exec_pipelined` are set to `False`.
        Executor will buffer cpu and gpu stages separatelly,
        and will fill the buffer queues when the first :meth:`nvidia.dali.pipeline.Pipeline.run`
        is issued.
    `exec_async` : bool, optional, default = True
        Whether to execute the pipeline asynchronously.
        This makes :meth:`nvidia.dali.pipeline.Pipeline.run` method
        run asynchronously with respect to the calling Python thread.
        In order to synchronize with the pipeline one needs to call
        :meth:`nvidia.dali.pipeline.Pipeline.outputs` method.
    `bytes_per_sample` : int, optional, default = 0
        A hint for RALI for how much memory to use for its tensors.
    `set_affinity` : bool, optional, default = False
        Whether to set CPU core affinity to the one closest to the
        GPU being used.
    `max_streams` : int, optional, default = -1
        Limit the number of CUDA streams used by the executor.
        Value of -1 does not impose a limit.
        This parameter is currently unused (and behavior of
        unrestricted number of streams is assumed).
    `default_cuda_stream_priority` : int, optional, default = 0
        CUDA stream priority used by RALI. See `cudaStreamCreateWithPriority` in CUDA documentation
    """
    '''.
    Args: batch_size
          rali_cpu
          gpu_id (default 0)
          cpu_threads (default 1)
    This returns a context'''
    _handle = None
    _current_pipeline = None

    def __init__(self, batch_size=-1, num_threads=-1, device_id=-1, seed=-1,
                 exec_pipelined=True, prefetch_queue_depth=2,
                 exec_async=True, bytes_per_sample=0,
                 rali_cpu=False, max_streams=-1, default_cuda_stream_priority=0, tensor_layout = types.NCHW, reverse_channels = False, multiplier = [1.0,1.0,1.0], offset = [0.0, 0.0, 0.0], tensor_dtype=types.FLOAT):
        if(rali_cpu):
            # print("comes to cpu")
            self._handle = b.raliCreate(
                batch_size, types.CPU, device_id, num_threads,prefetch_queue_depth,types.FLOAT)
        else:
            self._handle = b.raliCreate(
                batch_size, types.GPU, device_id, num_threads,prefetch_queue_depth,types.FLOAT)
        if(b.getStatus(self._handle) == types.OK):
            print("Pipeline has been created succesfully")
        else:
            raise Exception("Failed creating the pipeline")
        self._check_ops = ["CropMirrorNormalize"]
        self._check_crop_ops = ["Resize"]
        self._check_ops_decoder = ["ImageDecoder", "ImageDecoderSlice" , "ImageDecoderRandomCrop", "ImageDecoderRaw"]
        self._check_ops_reader = ["FileReader", "TFRecordReaderClassification", "TFRecordReaderDetection",
            "COCOReader", "Caffe2Reader", "Caffe2ReaderDetection", "CaffeReader", "CaffeReaderDetection"]
        self._batch_size = batch_size
        self._num_threads = num_threads
        self._device_id = device_id
        self._seed = seed
        self._exec_pipelined = exec_pipelined
        self._prefetch_queue_depth = prefetch_queue_depth
        self._exec_async = exec_async
        self._bytes_per_sample = bytes_per_sample
        self._rali_cpu = rali_cpu
        self._max_streams = max_streams
        self._default_cuda_stream_priority = default_cuda_stream_priority
        self._tensor_layout = tensor_layout
        self._tensor_dtype = tensor_dtype
        self._multiplier = multiplier
        self._reverse_channels = reverse_channels
        self._offset = offset
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
        self._prev_input_image = None
        self._current_output_image = None
        self._current_pipeline
        self._reader = None


    def store_values(self, operator):
        """
            Check if ops is one of those functionality to determine tensor_layout and tensor_dtype and crop.
            If so preserve it in pipeline to use for dali iterator call.
        """
        if(operator.data in self._check_ops):
            self._tensor_layout = operator._output_layout
            self._tensor_dtype = operator._output_dtype
            self._multiplier = list(map(lambda x: 1/x ,operator._std))
            self._offset = list(map(lambda x,y: -(x/y), operator._mean, operator._std))
            #changing operator std and mean to (1,0) to make sure there is no double normalization
            operator._std = [1.0]
            operator._mean = [0.0]
            if operator._crop_h != 0 and operator._crop_w != 0:
                self._img_w = operator._crop_w
                self._img_h = operator._crop_h
        elif(operator.data in self._check_crop_ops):
            self._img_w = operator._resize_x
            self._img_h = operator._resize_y

    def process_calls(self, output_image):
        last_operator = output_image.prev
        temp = output_image
        while(temp.prev is not None):
            if(temp.data in (self._check_ops + self._check_crop_ops + self._check_ops_reader)):
                self.store_values(temp)
            temp = temp.prev
        file_reader = temp
        file_reader.rali_c_func_call(self._handle)
        self._shuffle = file_reader._random_shuffle
        self._shard_id = file_reader._shard_id
        self._num_shards = file_reader._num_shards
        self._name = file_reader.data
        temp = temp.next
        operator = temp.next
        while(operator.next.next is not None):
            tensor = operator.next
            if(operator.data in self._check_ops_decoder):
                tensor.data = operator.rali_c_func_call(
                    self._handle, operator.prev.data, self._img_w, self._img_h, self._shuffle, self._shard_id, self._num_shards, False)
            else:
                tensor.data = operator.rali_c_func_call(
                    self._handle, operator.prev.data, False)
            operator = operator.next.next
        tensor = last_operator.next
        if(operator.data in self._check_ops_decoder):
            tensor.data = operator.rali_c_func_call(
                self._handle, operator.prev.data, self._img_w, self._img_h, self._shuffle, self._shard_id, self._num_shards, True)
        else:
            tensor.data = operator.rali_c_func_call(
                self._handle, operator.prev.data, True)
        return tensor.data

    def build(self):
        """Build the pipeline using raliVerify call
        """
        # outputs = self.define_graph()
        # self.process_calls(outputs[0])

        # #Checks for Casting "Labels" as last node & Box Encoder as last Prev node
        # if(len(outputs)==3):
        #     if((isinstance(outputs[1],list)== False) & (isinstance(outputs[2],list)== False)):
        #         if((outputs[2].prev is not None) | (outputs[1].prev is not None)):
        #             if(outputs[2].prev.data == "Cast") :
        #                 self._castLabels = True
        #                 if(outputs[2].prev.prev.prev.data is not None):
        #                     if(outputs[2].prev.prev.prev.data == "BoxEncoder"):
        #                         self._BoxEncoder = True
        #                         self._anchors = outputs[2].prev.prev.data
        #                         self._encode_tensor = outputs[2].prev.prev
        #                         self._encode_tensor.prev.rali_c_func_call(self._handle )
        # #Checks for Box Encoding as the Last Node
        # if(len(outputs)==3):
        #     if(isinstance(outputs[1],list)== False):
        #         if(outputs[1].prev is not None):
        #             if(outputs[2].prev is not None):
        #                 if(outputs[2].prev.data == "BoxEncoder"):
        #                     self._BoxEncoder = True
        #                     self._anchors = outputs[2].data
        #                     self._encode_tensor = outputs[2]
        #                     self._encode_tensor.prev.rali_c_func_call(self._handle )

        # #Checks for One Hot Encoding as the last Node
        # if(isinstance(outputs[1],list)== False):
        #     if(len(outputs)==2):
        #         if(outputs[1].prev is not None):
        #             print(type(outputs[1]))
        #             if(outputs[1].prev.data == "OneHotLabel"):
        #                 self._numOfClasses = outputs[1].prev.rali_c_func_call(self._handle)
        #                 self._oneHotEncoding = True
      

        status = b.raliVerify(self._handle)
        if(status != types.OK):
            print("Verify graph failed")
            exit(0)
        return self

    def run(self):
        """ Run the pipeline using raliRun call
        """
        status = b.raliRun(self._handle)
        if(status != types.OK):
            print("Rali Run failed")
        return status

    def define_graph(self):
        """This function is defined by the user to construct the
        graph of operations for their pipeline.
        It returns a list of outputs created by calling RALI Operators."""
        raise NotImplementedError

    def get_handle(self):
        return self._handle

    def copyImage(self, array):
        out = np.frombuffer(array, dtype=array.dtype)
        b.raliCopyToOutput(
            self._handle, np.ascontiguousarray(out, dtype=array.dtype))

    def copyToTensorNHWC(self, array,  multiplier, offset, reverse_channels, tensor_dtype):
        out = np.frombuffer(array, dtype=array.dtype)
        if tensor_dtype == types.FLOAT:
            b.raliCopyToOutputTensor32(self._handle, np.ascontiguousarray(out, dtype=array.dtype), types.NHWC,
                                       multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0))
        elif tensor_dtype == types.FLOAT16:
            b.raliCopyToOutputTensor16(self._handle, np.ascontiguousarray(out, dtype=array.dtype), types.NHWC,
                                       multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0))

    def copyToTensorNCHW(self, array,  multiplier, offset, reverse_channels, tensor_dtype):
        out = np.frombuffer(array, dtype=array.dtype)
        if tensor_dtype == types.FLOAT:
            b.raliCopyToOutputTensor32(self._handle, np.ascontiguousarray(out, dtype=array.dtype), types.NCHW,
                                       multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0))
        elif tensor_dtype == types.FLOAT16:
            b.raliCopyToOutputTensor16(self._handle, np.ascontiguousarray(out, dtype=array.dtype), types.NCHW,
                                       multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0))

    def encode(self, bboxes_in, labels_in):
        bboxes_tensor = torch.tensor(bboxes_in).float()
        labels_tensor=  torch.tensor(labels_in).long()
        return self._encode_tensor.prev.rali_c_func_call(self._handle, bboxes_tensor , labels_tensor )

    def GetOneHotEncodedLabels(self, array):
        return b.getOneHotEncodedLabels(self._handle, array, self._numOfClasses)

    def set_outputs(self, *output_list):
        print("\nLENGTH OF OUTPUT LIST", len(output_list))
        print("\nOUTPUT LIST", output_list)
        print("\nTYPE OUTPUT LIST", type(output_list))
        # exit(0)
        b.setOutputImages(self._handle,len(output_list),output_list)

    def set_outputs_trial(self,*output_list):
        set_output_images = []
        set_output_meta_data = []
        output_traces_list = []
        backtraced_nodes = []
        for output in output_list:
            if(isinstance(output, Node)):
                output.is_output = True
                output.kwargs_pybind["is_output"] = True
                set_output_images.append(output)
            elif(isinstance(output,MetaDataNode)):
                set_output_meta_data.append(output)

        # Create the Output Tracing List [ [ TraceList1 ],[ TraceList2 ],[ TraceList3 ] ..[ TraceListN ]   ]  by Backtracing the Nodes (N is the number of output Nodes set by the user)
        # TraceList1 = [ Current Node, Number of Prev Nodes, PrevNode1 , PrevNode2 ...PrevNodeN ]
        for node in set_output_images:
            output_dict = []
            node = [node]
            while (len(node)!=0 and node[0].submodule_name != "readers"):
                current_nodes = node
                prev_node_list =[]
                for current_node in current_nodes:
                    if current_node.submodule_name != "readers":
                        current_list = []
                        # non_augmentation_node = []
                        current_list.append(current_node)  # Append Node
                        number_of_prev_nodes = len(current_node.prev)
                        # Append number of Previous Node
                        current_list.append(number_of_prev_nodes)
                        for prev_node in current_node.prev:
                            # Append Previous Nodes
                            if(prev_node.CMN == True): # Needs to be changed at later stages
                                self._tensor_layout = prev_node.kwargs['output_layout']
                                self._tensor_dtype = prev_node.kwargs['output_dtype']
                                self._multiplier = list(map(lambda x: 1/x ,prev_node.kwargs['std']))
                                self._offset = list(map(lambda x,y: -(x/y), prev_node.kwargs['mean'], prev_node.kwargs['std']))
                                #changing operator std and mean to (1,0) to make sure there is no double normalization
                                prev_node.kwargs['std'] = [1.0]
                                prev_node.kwargs['mean'] = [0.0]

                            current_list.append(prev_node)
                            if(prev_node.augmentation_node == True):
                                prev_node_list.append(prev_node)

                        node = prev_node_list
                        output_dict.append(current_list)


            #Reader Node
            self._name = node[0].node_name   #Store the name of the reader in a variable for further use     
            current_list=[node[0],0, "NULL"]
            output_dict.append(current_list)
            output_traces_list.append(output_dict)

        # Excecute the rali c func call's
        for trace_list in output_traces_list: #trace_list =  [ [ Current Node , Number of Prev Nodes, PrevNode1 , PrevNode2 ...PrevNodeN ] , [ Current Node, Number of Prev Nodes, PrevNode1 , PrevNode2 ...PrevNodeN ] , .....]
            trace_list.reverse()
            for trace in trace_list: # trace = [ Current Node, Number of Prev Nodes, PrevNode1 , PrevNode2 ...PrevNodeN ] , # trace[0] = Current Node
                if not trace[0].visited :
                    if trace[0].has_output_image :
                        if trace[0].has_input_image :
                            for i in range(trace[1]):
                                name = "input_image" + str(i)
                                trace[0].kwargs_pybind[name] = trace[0].prev[i].output_image if trace[0].prev[i].visited else trace[0].prev[i].rali_c_func_call(self._handle,*(trace[0].prev[i].kwargs_pybind.values()))#Prev nodei output to current node input
                        for i in range(trace[1]):
                            if trace[0].prev[i].visited:
                                pass
                            else:
                                trace[0].prev[i].rali_c_func_call(self._handle,*(trace[0].prev[i].kwargs_pybind.values()))
                                trace[0].prev[i].set_visited(True)
                        l= (trace[0].kwargs_pybind.values())
                        trace[0].set_output_image(trace[0].rali_c_func_call(self._handle,*l))
                        trace[0].set_visited(True)
                    else:
                        if trace[0].submodule_name == "readers" :
                            l= (trace[0].kwargs_pybind.values())
                            trace[0].rali_c_func_call(self._handle,*l)
                            trace[0].set_visited(True)
                        else:
                            l= (trace[0].kwargs_pybind.values())
                            trace[0].rali_c_func_call(self._handle,*l)
                            trace[0].set_visited(True)
                            trace[0] = trace[0].prev[0]  # Need to check this !!
                        
        for meta_node in set_output_meta_data:
            while(meta_node.prev is not None ):
                if(meta_node.node_name == "OneHotLabel"):
                    self._numOfClasses = meta_node.kwargs["num_classes"]
                    self._oneHotEncoding = True
                elif(meta_node.node_name == "BoxEncoder"):
                    l = (meta_node.kwargs_pybind.values())
                    meta_node.rali_c_func_call(self._handle,*l)
                    self._BoxEncoder = True
                meta_node = meta_node.prev
            if(meta_node.node_name == "OneHotLabel"):
                self._numOfClasses = meta_node.kwargs["num_classes"]
                self._oneHotEncoding = True
            elif(meta_node.node_name == "BoxEncoder"):
                    l = (meta_node.kwargs_pybind.values())
                    meta_node.rali_c_func_call(self._handle,*l)
                    self._BoxEncoder = True
                
        # exit(0)
        status = b.raliVerify(self._handle)
        if(status != types.OK):
            print("Verify graph failed")

    def __enter__(self):
        print("\n __enter__ block")
        
        Pipeline._current_pipeline = self
        print("\n Pipeline handle in enter block:", Pipeline._current_pipeline._handle)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def set_seed(self,seed=0):
        return b.setSeed(seed)

    def create_int_param(self,value):
        return b.CreateIntParameter(value)

    def create_float_param(self,value):
        return b.CreateFloatParameter(value)

    def update_int_param(self,value,param):
        b.UpdateIntParameter(value,param)

    def update_float_param(self,value,param):
        b.UpdateFloatParameter(value,param)

    def get_int_value(self,param):
        return b.GetIntValue(param)

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
        b.getImageLabels(self._handle, array)

    def copyEncodedBoxesAndLables(self, bbox_array, label_array):
        b.raliCopyEncodedBoxesAndLables(self._handle, bbox_array, label_array)
        
    def GetImgSizes(self, array):
        return b.getImgSizes(self._handle, array)

    def GetImageLabels(self, array):
        return b.getImageLabels(self._handle, array)


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

    def raliResetLoaders(self):
        return b.raliResetLoaders(self._handle)

    def isEmpty(self):
        return b.isEmpty(self._handle)

    def Timing_Info(self):
        return b.getTimingInfo(self._handle)
