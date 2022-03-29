
import types
import collections
import numpy as np
from random import shuffle
import torch


class ExternalInputIterator(object):
    def __init__(self, path, sinset_file_name, batch_size, mode, device_id, num_gpus):
        self.images_dir = path
        self.batch_size = batch_size
        self.mode = mode
        with open(self.images_dir + sinset_file_name, 'r') as f:
            self.files = [line.rstrip() for line in f if line is not '']
        # whole data set size
        self.data_set_len = len(self.files) 
        # based on the device_id and total number of GPUs - world size
        # get proper shard
        self.files = self.files[self.data_set_len * device_id // num_gpus:
                                self.data_set_len * (device_id + 1) // num_gpus]
        self.n = len(self.files)

    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            jpeg_filename, label = self.files[self.i % self.n].split(' ')
            batch.append(np.fromfile(self.images_dir + jpeg_filename, dtype = np.uint8))  # we can use numpy
            labels.append(torch.tensor([int(label)], dtype = torch.uint8)) # or PyTorch's native tensors
            self.i += 1
        return (batch, labels)

    def __len__(self):
        return self.data_set_len

    next = __next__