from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import random

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types

import sys
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
import pycocotools.mask as mask_utils

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class SegmentationMask(object):
    """
    This class stores the segmentations for all objects in the image
    """

    def __init__(self, polygons, size, mode=None):
        """
        Arguments:
            polygons: a list of list of lists of numbers. The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                object, and the third level to the polygon coordinates.
        """
        assert isinstance(polygons, list)

        self.polygons = [Polygons(p, size, mode) for p in polygons]
        self.size = size
        self.mode = mode

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped = []
        for polygon in self.polygons:
            flipped.append(polygon.transpose(method))
        return SegmentationMask(flipped, size=self.size, mode=self.mode)

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]
        cropped = []
        for polygon in self.polygons:
            cropped.append(polygon.crop(box))
        return SegmentationMask(cropped, size=(w, h), mode=self.mode)

    def resize(self, size, *args, **kwargs):
        scaled = []
        for polygon in self.polygons:
            scaled.append(polygon.resize(size, *args, **kwargs))
        return SegmentationMask(scaled, size=size, mode=self.mode)

    def to(self, *args, **kwargs):
        return self

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            selected_polygons = [self.polygons[item]]
        else:
            # advanced indexing on a single dimension
            selected_polygons = []
            if isinstance(item, torch.Tensor):
                if item.dtype == torch.uint8 or item.dtype == torch.bool:
                    item = item.nonzero()
                    item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                selected_polygons.append(self.polygons[i])
        return SegmentationMask(selected_polygons, size=self.size, mode=self.mode)

    def __iter__(self):
        return iter(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s

class Polygons(object):
    """
    This class holds a set of polygons that represents a single instance
    of an object mask. The object can be represented as a set of
    polygons
    """

    def __init__(self, polygons, size, mode):
        # assert isinstance(polygons, list), '{}'.format(polygons)
        if isinstance(polygons, list):
            polygons = [torch.as_tensor(p, dtype=torch.float32) for p in polygons]
        elif isinstance(polygons, Polygons):
            polygons = polygons.polygons

        self.polygons = polygons
        self.size = size
        self.mode = mode

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped_polygons = []
        width, height = self.size
        if method == FLIP_LEFT_RIGHT:
            dim = width
            idx = 0
        elif method == FLIP_TOP_BOTTOM:
            dim = height
            idx = 1

        for poly in self.polygons:
            p = poly.clone()
            TO_REMOVE = 1
            p[idx::2] = dim - poly[idx::2] - TO_REMOVE
            flipped_polygons.append(p)

        return Polygons(flipped_polygons, size=self.size, mode=self.mode)

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]

        # TODO chck if necessary
        w = max(w, 1)
        h = max(h, 1)

        cropped_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] = p[0::2] - box[0]  # .clamp(min=0, max=w)
            p[1::2] = p[1::2] - box[1]  # .clamp(min=0, max=h)
            cropped_polygons.append(p)

        return Polygons(cropped_polygons, size=(w, h), mode=self.mode)

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_polys = [p * ratio for p in self.polygons]
            return Polygons(scaled_polys, size, mode=self.mode)

        ratio_w, ratio_h = ratios
        scaled_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h
            scaled_polygons.append(p)

        return Polygons(scaled_polygons, size=size, mode=self.mode)

    def convert(self, mode):
        width, height = self.size
        if mode == "mask":
            rles = mask_utils.frPyObjects(
                [p.numpy() for p in self.polygons], height, width
            )
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle)
            mask = torch.from_numpy(mask)
            # TODO add squeeze?
            return mask

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_polygons={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s

class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = torch.empty_like(self.tensors, device='cuda')
        cast_tensor.copy_(self.tensors, non_blocking=True)
        #cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    def pin_memory(self):
        pinned_tensor = self.tensors.pin_memory()
        return ImageList(pinned_tensor, self.image_sizes)

class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimenion of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")


class ROCALCOCOIterator(object):
    """
    COCO ROCAL iterator for pyTorch.
    Parameters
    ----------
    pipelines : list of amd.rali.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, device="cpu"):

        assert pipelines is not None, "Number of provided pipelines has to be at least 1"

        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.device = device
        self.device_id = self.loader._device_id
        self.bs = self.loader._batch_size
        self.w = self.loader.getOutputWidth()
        self.h = self.loader.getOutputHeight()
        self.n = self.loader.getOutputImageCount()
        self.rim = self.loader.getRemainingImages()
        print("____________REMAINING IMAGES____________:", self.rim)
        color_format = self.loader.getOutputColorFormat()
        self.p = (1 if color_format is types.GRAY else 3)
        if self.device == "cpu":
            if self.tensor_dtype == types.FLOAT:
                self.out = torch.empty((self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype=torch.float32)
            elif self.tensor_dtype == types.FLOAT16:
                self.out = torch.empty((self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype=torch.float16)
        else:
            torch_gpu_device = torch.device('cuda', self.device_id)
            if self.tensor_dtype == types.FLOAT:
                self.out = torch.empty((self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype=torch.float32, device=torch_gpu_device)
            elif self.tensor_dtype == types.FLOAT16:
                self.out = torch.empty((self.bs*self.n, self.p, int(self.h/self.bs), self.w,), dtype=torch.float16, device=torch_gpu_device)

    def next(self):
        return self.__next__()

    def __next__(self):
        print("In the next routine of COCO Iterator")
        if(self.loader.isEmpty()):
            timing_info = self.loader.Timing_Info()
            print("Load     time ::", timing_info.load_time)
            print("Decode   time ::", timing_info.decode_time)
            print("Process  time ::", timing_info.process_time)
            print("Transfer time ::", timing_info.transfer_time)
            raise StopIteration

        if self.loader.run() != 0:
            raise StopIteration
        self.lis = []  # Empty list for bboxes
        self.lis_lab = []  # Empty list of labels

        self.loader.copyToTensor(
            self.out, self.multiplier, self.offset, self.reverse_channels, self.tensor_format, self.tensor_dtype)


        self.img_names_length = np.empty(self.bs, dtype="int32")
        self.img_names_size = self.loader.GetImageNameLen(self.img_names_length)
        # print("Image name length:", self.img_names_size)
# Images names of a batch
        self.Img_name = self.loader.GetImageName(self.img_names_size)
        # print("Image names in a batch ", self.Img_name)
#Count of labels/ bboxes in a batch
        self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
        self.count_batch = self.loader.GetBoundingBoxCount(self.bboxes_label_count)
        # print("Count Batch:", self.count_batch)
# 1D labels array in a batch
        self.labels = np.zeros(self.count_batch, dtype="int32")
        self.loader.GetBBLabels(self.labels)
        # print(self.labels)
# 1D bboxes array in a batch
        self.bboxes = np.zeros((self.count_batch*4), dtype="float64")
        self.loader.GetBBCords(self.bboxes)
        # print("BBoxes (rocAL): ",self.bboxes)
#Image sizes of a batch
        self.img_size = np.zeros((self.bs * 2),dtype = "int32")
        self.loader.GetImgSizes(self.img_size)
        # print("Image sizes:", self.img_size)
#Image ROI width and height
        self.roi_width = np.zeros((self.bs),dtype = "uint32")
        self.roi_height = np.zeros((self.bs),dtype = "uint32")
        self.loader.getOutputROIWidth(self.roi_width)
        self.loader.getOutputROIHeight(self.roi_height)
        self.roi_sizes = np.vstack((self.roi_height,self.roi_width)).T # Old
        self.roi_sizes_wh = np.vstack((self.roi_width,self.roi_height)).T # New
#Mask info of a batch
        self.mask_count = np.zeros(self.count_batch, dtype="int32")
        self.mask_size = self.loader.GetMaskCount(self.mask_count)
        self.polygon_size = np.zeros(self.mask_size, dtype= "int32")
        self.mask_data = np.zeros(100000, dtype = "float32")
        self.loader.GetMaskCoordinates(self.polygon_size, self.mask_data)

        count =0
        sum_count=0
        j = 0
        list_poly = []
        iteration1 = 0
        iteration = 0
        self.target_batch = []
        self.roi_image_size = []
        self.roi_image_size_wh = []
        for i in range(self.bs):
            count = self.bboxes_label_count[i]
            self.img_name = self.Img_name[i*16:(i*16)+12]
            self.img_name=self.img_name.decode('utf_8')
            self.img_name = np.char.lstrip(self.img_name, chars ='0')
            self.img_size_2d_numpy = (self.img_size[i*2:(i*2)+2])
            self.img_roi_size2d_numpy = (self.roi_sizes[i])
            self.img_roi_size2d_numpy_wh = (self.roi_sizes_wh[i])
            # Image Size ROI
            roi_tmp_list = self.img_roi_size2d_numpy.tolist()
            self.roi_image_size.append(torch.Size(roi_tmp_list))
            roi_tmp_list = self.img_roi_size2d_numpy_wh.tolist()
            self.roi_image_size_wh.append(torch.Size(roi_tmp_list))
            self.label_2d_numpy = (self.labels[sum_count : sum_count+count])
            self.bb_2d_numpy = (self.bboxes[sum_count*4 : (sum_count+count)*4])

            # print("\nBefore : self.bb_2d_numpy\n", self.bb_2d_numpy)
            for index, element in enumerate(self.bb_2d_numpy):
                if index % 2 == 0:
                    self.bb_2d_numpy[index] = self.bb_2d_numpy[index] * self.img_roi_size2d_numpy_wh[0]
                elif index % 2 != 0:
                    self.bb_2d_numpy[index] = self.bb_2d_numpy[index] * self.img_roi_size2d_numpy_wh[1]

            # print("\nAfter : self.bb_2d_numpy\n", self.bb_2d_numpy)

            self.bb_2d_numpy = np.reshape(self.bb_2d_numpy, (-1, 4)).tolist()

            self.target = BoxList(self.bb_2d_numpy, (self.img_roi_size2d_numpy_wh[0],self.img_roi_size2d_numpy_wh[1]), mode="xyxy")
            self.target.add_field("labels", self.label_2d_numpy)

            self.count_mask = self.bboxes_label_count[i]
            poly_batch_list = []
            for i in range(self.count_mask):
                poly_list = []
                for k in range(self.mask_count[iteration1]):
                    polygons = []
                    polygon_size_check = self.polygon_size[iteration]
                    iteration = iteration + 1
                    for loop_idx in range(polygon_size_check):
                        polygons.append(self.mask_data[j])
                        j = j + 1
                    poly_list.append(polygons)
                iteration1 = iteration1 + 1
                poly_batch_list.append(poly_list)

            masks = SegmentationMask(poly_batch_list, (self.img_roi_size2d_numpy_wh[0],self.img_roi_size2d_numpy_wh[1]))
            self.target.add_field("masks", masks)

            self.target_batch.append(self.target)
            sum_count = sum_count +count

        self.img_list_obj = ImageList(self.out ,self.roi_image_size)
        for i in range(self.bs):
            img_name = self.Img_name[i*16:(i*16)+12].decode('utf-8')
            image = self.img_list_obj.tensors[i].cpu().numpy()
            PIL_image = np.array(Image.fromarray(((image.transpose(1,2,0))+[102.9801, 115.9465, 122.7717]).astype('uint8'), 'RGB'))
            PIL_image = cv2.cvtColor(PIL_image, cv2.COLOR_RGB2BGR)
            for box in self.target_batch[i].bbox:
                x1, y1, x2, y2 = box.cpu().numpy().astype(np.int)
                cv2.rectangle(PIL_image, (x1,y1), (x2,y2), (255, 0, 0), 2)
            cv2.imwrite(f'{img_name}.jpg', PIL_image)
        return self.img_list_obj, self.target_batch


    def reset(self):
        self.loader.rocalResetLoaders()

    def __iter__(self):
        self.loader.rocalResetLoaders()
        return self

def main():
    if len(sys.argv) < 5:
        print('Please pass the folder image_folder Annotation_file cpu/gpu batch_size')
        exit(0)

    image_path = sys.argv[1]
    ann_path = sys.argv[2]
    if(sys.argv[3] == "cpu"):
        _rali_cpu = True
    else:
        _rali_cpu = False
    bs = int(sys.argv[4])
    nt = 1
    di = 0
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)

    pipe = Pipeline(batch_size=bs, num_threads=nt,device_id=di, seed=random_seed, rocal_cpu=_rali_cpu)

    with pipe:
        jpegs, bboxes, labels = fn.readers.coco(
            file_root=image_path, annotations_file=ann_path, random_shuffle=True, seed=di, masks=True)
        images_decoded = fn.decoders.image(jpegs, output_type=types.RGB, file_root=image_path, annotations_file=ann_path, shard_id=di, num_shards=nt, random_shuffle=True, seed=di)
        coin_flip = fn.random.coin_flip(probability=0.5)
        rmn_images = fn.resize_mirror_normalize(images_decoded,
                                            device="gpu",
                                            output_dtype=types.FLOAT16,
                                            output_layout=types.NCHW,
                                            resize_min = 1344,
                                            resize_max = 1344,
                                            mirror=coin_flip,
                                            mean= [102.9801, 115.9465, 122.7717],
                                            std = [1. , 1., 1.])
        pipe.set_outputs(rmn_images)
    
    pipe.build()
    if(_rali_cpu):
        data_loader = ROCALCOCOIterator(
            pipe, multiplier=pipe._multiplier, offset=pipe._offset, device="cpu")
    else:
        data_loader = ROCALCOCOIterator(
            pipe, multiplier=pipe._multiplier, offset=pipe._offset, device="cuda")
    epochs = 1
    for epoch in range(int(epochs)):
        print("EPOCH:::::",epoch)
        for i, (images, targets) in enumerate(tqdm(data_loader)):
            print("*******************************",i,"************************")
            print("****************IMAGES****************")
            print(images.image_sizes)
            print("\nBBOXES:\n", targets[0].bbox)
            print("\nLABELS:\n", targets[0].extra_fields['labels'])
            print("\nMASK POLYGONS :\n", targets[0].extra_fields['masks'])
        data_loader.reset()


if __name__ == '__main__':
    main()
