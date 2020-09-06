import numpy as np
import rali_pybind as b
import amd.rali.types as types
class RALIGenericImageIterator(object):
    def __init__(self, pipeline):
        self.loader = pipeline
        self.w = b.getOutputWidth(self.loader._handle)
        self.h = b.getOutputHeight(self.loader._handle)
        self.n = b.getOutputImageCount(self.loader._handle)
        color_format = b.getOutputColorFormat(self.loader._handle)
        self.p = (1 if color_format is types.GRAY else 3)
        height = self.h*self.n
        self.out_tensor = None
        self.out_image = np.zeros((height, self.w, self.p), dtype = "uint8")
        self.bs = pipeline._batch_size

    def next(self):
        return self.__next__()

    def __next__(self):
        if b.getRemainingImages(self.loader._handle) < self.bs:
            raise StopIteration

        if self.loader.run() != 0:
            raise StopIteration

        self.loader.copyImage(self.out_image)
        return self.out_image , self.out_tensor

    def reset(self):
        b.raliResetLoaders(self.loader._handle)

    def __iter__(self):
        b.raliResetLoaders(self.loader._handle)
        return self

class RALIGenericIteratorDetection(object):
    def __init__(self, pipeline, tensor_layout = types.NCHW, reverse_channels = False, multiplier = [1.0,1.0,1.0], offset = [0.0, 0.0, 0.0], tensor_dtype=types.FLOAT):
        self.loader = pipeline
        self.tensor_format =tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.w = b.getOutputWidth(self.loader._handle)
        self.h = b.getOutputHeight(self.loader._handle)
        self.n = b.getOutputImageCount(self.loader._handle)
        self.bs = pipeline._batch_size
        color_format = b.getOutputColorFormat(self.loader._handle)
        self.p = (1 if color_format is types.GRAY else 3)
        if self.tensor_dtype == types.FLOAT:
            self.out = np.zeros(( self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype = "float32")
        elif self.tensor_dtype == types.FLOAT16:
            self.out = np.zeros(( self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype = "float16")
        # self.labels = np.zeros((self.bs),dtype = "int32")

    def next(self):
        return self.__next__()

    def __next__(self):
        if(b.isEmpty(self.loader._handle)):
            timing_info = b.getTimingInfo(self.loader._handle)
            print("Load     time ::",timing_info.load_time)
            print("Decode   time ::",timing_info.decode_time)
            print("Process  time ::",timing_info.process_time)
            print("Transfer time ::",timing_info.transfer_time)
            raise StopIteration

        if self.loader.run() != 0:
            raise StopIteration

        if(types.NCHW == self.tensor_format):
            self.loader.copyToTensorNCHW(self.out, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))
        else:
            self.loader.copyToTensorNHWC(self.out, self.multiplier, self.offset, self.reverse_channels, int(self.tensor_dtype))
        
        if(self.loader._name == "TFRecordReaderDetection"):
            num_bboxes = 0
            self.bbox_list =[]
            self.label_list=[]
            self.num_bboxes_list=[]

            for idx in range(self.bs):
                num_bboxes=self.loader.GetBoundingBoxCount(idx)
                self.labels = np.zeros(num_bboxes,dtype = "int32")
                self.bboxes = np.zeros(num_bboxes*4,dtype = "float32" )
                self.loader.GetBBLabels(self.labels,idx)
                self.loader.GetBBCords(self.bboxes,idx)
                
                self.bb_2d_numpy = np.reshape(self.bboxes, (-1, 4)).tolist()
                self.label_2d_numpy = np.reshape(self.labels, (-1, 1)).tolist()
                
                self.bbox_list.append(self.bb_2d_numpy)
                self.label_list.append(self.label_2d_numpy)
                self.num_bboxes_list.append(num_bboxes)

            self.target = self.bbox_list
            self.target1 = self.label_list

            max_cols = max([len(row) for batch in self.target for row in batch])
            # max_rows = max([len(batch) for batch in self.target])
            max_rows = 100
            bb_padded = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.target]
            bb_padded_1=[row + [0] * (max_cols - len(row)) for batch in bb_padded for row in batch]
            arr = np.asarray(bb_padded_1)
            self.res = np.reshape(arr, (-1, max_rows, max_cols))

            max_cols = max([len(row) for batch in self.target1 for row in batch])
            # max_rows = max([len(batch) for batch in self.target1])
            max_rows = 100
            lab_padded = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.target1]
            lab_padded_1=[row + [0] * (max_cols - len(row)) for batch in lab_padded for row in batch]
            labarr = np.asarray(lab_padded_1)
            self.l = np.reshape(labarr, (-1, max_rows, max_cols))
            
            self.num_bboxes_arr = np.array(self.num_bboxes_list)

            if self.tensor_dtype == types.FLOAT:
                return self.out.astype(np.float32), self.res, self.l, self.num_bboxes_arr
            elif self.tensor_dtype == types.FLOAT16:
                return self.out.astype(np.float16), self.res, self.l, self.num_bboxes_arr
        elif (self.loader._name == "TFRecordReaderClassification"):
            self.labels = np.zeros((self.bs),dtype = "int32")
            self.loader.getImageLabels(self.labels)
        
            if self.tensor_dtype == types.FLOAT:
                return self.out.astype(np.float32), self.labels
            elif self.tensor_dtype == types.TensorDataType.FLOAT16:
                return self.out.astype(np.float16), self.labels
        
    def reset(self):
        b.raliResetLoaders(self.loader._handle)

    def __iter__(self):
        b.raliResetLoaders(self.loader._handle)
        return self


class RALIIterator(RALIGenericIteratorDetection):
    """
    RALI iterator for detection and classification tasks for PyTorch. It returns 2 or 3 outputs
    (data and label) or (data , bbox , labels) in the form of PyTorch's Tensor.
    Calling
    .. code-block:: python
       RALIIterator(pipelines, size)
    is equivalent to calling
    .. code-block:: python
       RALIGenericIteratorDetection(pipelines, ["data", "label"], size)


    """
    def __init__(self,
                 pipelines,
                 size = 0,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        pipe = pipelines
        super(RALIIterator, self).__init__(pipe, tensor_layout = pipe._tensor_layout, tensor_dtype = pipe._tensor_dtype,
                                                            multiplier=pipe._multiplier, offset=pipe._offset)



class RALI_iterator(RALIGenericImageIterator):
    """
    RALI iterator for classification tasks for PyTorch. It returns 2 outputs
    (data and label) in the form of PyTorch's Tensor.
   
    """
    def __init__(self,
                 pipelines,
                 size = 0,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        pipe = pipelines
        super(RALI_iterator, self).__init__(pipe)