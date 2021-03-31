import torch
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
        self.p = (1 if (color_format == int(types.GRAY)) else 3)
        height = self.h*self.n
        self.out_tensor = None
        self.out_bbox = None
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
        if((self.loader._name == "Caffe2ReaderDetection") or (self.loader._name == "CaffeReaderDetection")):
         
            for i in range(self.bs):
                size = b.getImageNameLen(self.loader._handle,i)
                print(size)
                self.array = np.array(["                 "])
                
                self.out=np.frombuffer(self.array, dtype=(self.array).dtype)
            
                b.getImageName(self.loader._handle, self.out ,i)
            return self.out_image ,self.out_bbox, self.out_tensor
        else:
            return self.out_image , self.out_tensor

    def reset(self):
        b.raliResetLoaders(self.loader._handle)

    def __iter__(self):
        b.raliResetLoaders(self.loader._handle)
        return self


class RALIGenericIterator(object):
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
        self.p = (1 if (color_format == int(types.GRAY)) else 3)
        if self.tensor_dtype == types.FLOAT:
            self.out = np.zeros(( self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype = "float32")
        elif self.tensor_dtype == types.FLOAT16:
            self.out = np.zeros(( self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype = "float16")
        # self.labels = np.zeros((self.bs),dtype = "int32")
        if(self.loader._oneHotEncoding == True):
            self.labels = np.zeros((self.bs)*(self.loader._numOfClasses),dtype = "int32")
        else:
            self.labels = np.zeros((self.bs),dtype = "int32")
        if self.bs != 0:
            self.len = b.getRemainingImages(self.loader._handle)//self.bs
        else:
            self.len = b.getRemainingImages(self.loader._handle)

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
        
        if((self.loader._name == "Caffe2ReaderDetection") or (self.loader._name == "CaffeReaderDetection")):
            self.lis = []  # Empty list for bboxes
            self.lis_lab = []  # Empty list of labels
            
            #Count of labels/ bboxes in a batch
            self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
            self.count_batch = self.loader.GetBoundingBoxCount(self.bboxes_label_count)
            # 1D labels array in a batch
            self.labels = np.zeros(self.count_batch, dtype="int32")
            self.loader.GetBBLabels(self.labels)
            # 1D bboxes array in a batch
            self.bboxes = np.zeros((self.count_batch*4), dtype="float32")
            self.loader.GetBBCords(self.bboxes)
            #Image sizes of a batch
            self.img_size = np.zeros((self.bs * 2),dtype = "int32")
            self.loader.GetImgSizes(self.img_size)
            
            count =0
            sum_count=0
            for i in range(self.bs):
                count = self.bboxes_label_count[i]
  
                self.label_2d_numpy = (self.labels[sum_count : sum_count+count])
                self.label_2d_numpy = np.reshape(self.label_2d_numpy, (-1, 1)).tolist()
                self.bb_2d_numpy = (self.bboxes[sum_count*4 : (sum_count+count)*4])
                self.bb_2d_numpy = np.reshape(self.bb_2d_numpy, (-1, 4)).tolist()
                
                self.lis_lab.append(self.label_2d_numpy)
                self.lis.append(self.bb_2d_numpy)
                
                sum_count = sum_count +count

            self.target = self.lis
            self.target1 = self.lis_lab
            max_cols = max([len(row) for batch in self.target for row in batch])
            max_rows = max([len(batch) for batch in self.target])
            self.bb_padded = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.target]
            self.bb_padded = torch.FloatTensor([row + [0] * (max_cols - len(row)) for batch in self.bb_padded for row in batch])
            self.bb_padded = self.bb_padded.view(-1, max_rows, max_cols)

            max_cols1 = max([len(row) for batch in self.target1 for row in batch])
            max_rows1 = max([len(batch) for batch in self.target1])
            self.labels_padded = [batch + [[0] * (max_cols1)] * (max_rows1 - len(batch)) for batch in self.target1]
            self.labels_padded = torch.LongTensor([row + [0] * (max_cols1 - len(row)) for batch in self.labels_padded for row in batch])
            self.labels_padded = self.labels_padded.view(-1, max_rows1, max_cols1)

            if self.tensor_dtype == types.FLOAT:
                return torch.from_numpy(self.out),self.bb_padded, self.labels_padded
            elif self.tensor_dtype == types.FLOAT16:
                return torch.from_numpy(self.out.astype(np.float16)),self.bb_padded, self.labels_padded

        else:
            if(self.loader._oneHotEncoding == True):
                self.loader.GetOneHotEncodedLabels(self.labels)
                self.labels_tensor = torch.from_numpy(self.labels).type(torch.LongTensor)
                self.labels_tensor = self.labels_tensor.view(-1, self.bs, self.loader._numOfClasses)
            else:
                self.loader.getImageLabels(self.labels)
                self.labels_tensor = torch.from_numpy(self.labels).type(torch.LongTensor)

            if self.tensor_dtype == types.FLOAT:
                return torch.from_numpy(self.out), self.labels_tensor
            elif self.tensor_dtype == types.FLOAT16:
                return torch.from_numpy(self.out.astype(np.float16)), self.labels_tensor

    def reset(self):
        b.raliResetLoaders(self.loader._handle)

    def __iter__(self):
        b.raliResetLoaders(self.loader._handle)
        return self

    def __len__(self):
        return self.len

    def __del__(self):
        b.raliRelease(self.loader._handle)


class RALIClassificationIterator(RALIGenericIterator):
    """
    RALI iterator for classification tasks for PyTorch. It returns 2 outputs
    (data and label) in the form of PyTorch's Tensor.

    Calling

    .. code-block:: python

       RALIClassificationIterator(pipelines, size)

    is equivalent to calling

    .. code-block:: python

       RALIGenericIterator(pipelines, ["data", "label"], size)

    Please keep in mind that Tensors returned by the iterator are
    still owned by RALI. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another tensor.

    Parameters
    ----------
    pipelines : list of amd.raliLI.pipeline.Pipeline
                List of pipelines to use
    size : int
           Number of samples in the epoch (Usually the size of the dataset).
    auto_reset : bool, optional, default = False
                 Whether the iterator resets itself for the next epoch
                 or it requires reset() to be called separately.
    fill_last_batch : bool, optional, default = True
                 Whether to fill the last batch with data up to 'self.batch_size'.
                 The iterator would return the first integer multiple
                 of self._num_gpus * self.batch_size entries which exceeds 'size'.
                 Setting this flag to False will cause the iterator to return
                 exactly 'size' entries.
    dynamic_shape: bool, optional, default = False
                 Whether the shape of the output of the RALI pipeline can
                 change during execution. If True, the pytorch tensor will be resized accordingly
                 if the shape of RALI returned tensors changes during execution.
                 If False, the iterator will fail in case of change.
    last_batch_padded : bool, optional, default = False
                 Whether the last batch provided by RALI is padded with the last sample
                 or it just wraps up. In the conjunction with `fill_last_batch` it tells
                 if the iterator returning last batch with data only partially filled with
                 data from the current epoch is dropping padding samples or samples from
                 the next epoch. If set to False next epoch will end sooner as data from
                 it was consumed but dropped. If set to True next epoch would be the
                 same length as the first one.

    Example
    -------
    With the data set [1,2,3,4,5,6,7] and the batch size 2:
    fill_last_batch = False, last_batch_padded = True  -> last batch = [7], next iteration will return [1, 2]
    fill_last_batch = False, last_batch_padded = False -> last batch = [7], next iteration will return [2, 3]
    fill_last_batch = True, last_batch_padded = True   -> last batch = [7, 7], next iteration will return [1, 2]
    fill_last_batch = True, last_batch_padded = False  -> last batch = [7, 1], next iteration will return [2, 3]
    """
    def __init__(self,
                 pipelines,
                 size = 0,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        pipe = pipelines
        super(RALIClassificationIterator, self).__init__(pipe, tensor_layout = pipe._tensor_layout, tensor_dtype = pipe._tensor_dtype,
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
