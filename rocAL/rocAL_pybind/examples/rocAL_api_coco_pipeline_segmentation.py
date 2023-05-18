from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch

from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types

import os
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
import pycocotools.mask as mask_utils
from parse_config import parse_args

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

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT16, device="cpu", display=False):

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
        self.display = display
        print("____________REMAINING IMAGES____________:", self.rim)
        color_format = self.loader.getOutputColorFormat()
        self.p = (1 if color_format is types.GRAY else 3)
        if tensor_layout == types.NCHW:
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
        else:
            if self.device == "cpu":
                if self.tensor_dtype == types.FLOAT:
                    self.out = torch.empty((self.bs*self.n, int(self.h/self.bs), self.w, self.p), dtype=torch.float32)
                elif self.tensor_dtype == types.FLOAT16:
                    self.out = torch.empty((self.bs*self.n, int(self.h/self.bs), self.w, self.p), dtype=torch.float16)
            else:
                torch_gpu_device = torch.device('cuda', self.device_id)
                if self.tensor_dtype == types.FLOAT:
                    self.out = torch.empty((self.bs*self.n, int(self.h/self.bs), self.w, self.p), dtype=torch.float32, device=torch_gpu_device)
                elif self.tensor_dtype == types.FLOAT16:
                    self.out = torch.empty((self.bs*self.n, int(self.h/self.bs), self.w, self.p), dtype=torch.float16, device=torch_gpu_device)

    def next(self):
        return self.__next__()

    def __next__(self):
        if(self.loader.isEmpty()):
            timing_info = self.loader.Timing_Info()
            print("Load     time ::", timing_info.load_time)
            print("Decode   time ::", timing_info.decode_time)
            print("Process  time ::", timing_info.process_time)
            print("Transfer time ::", timing_info.transfer_time)
            raise StopIteration

        if self.loader.run() != 0:
            raise StopIteration

        self.loader.copyToExternalTensor(
            self.out, self.multiplier, self.offset, self.reverse_channels, self.tensor_format, self.tensor_dtype)


        self.img_names_length = np.empty(self.bs, dtype="int32")
        self.img_names_size = self.loader.GetImageNameLen(self.img_names_length)
# Images names of a batch
        self.Img_name = self.loader.GetImageName(self.img_names_size)
#Count of labels/ bboxes in a batch
        self.bboxes_label_count = np.zeros(self.bs, dtype="int32")
        self.count_batch = self.loader.GetBoundingBoxCount(self.bboxes_label_count)
# 1D labels array in a batch
        self.labels = np.zeros(self.count_batch, dtype="int32")
        self.loader.GetBBLabels(self.labels)
# 1D bboxes array in a batch
        self.bboxes = np.zeros((self.count_batch*4), dtype="float64")
        self.loader.GetBBCords(self.bboxes)
#Image ROI width and height
        self.roi_sizes_wh = np.zeros((self.bs * 2),dtype = "int32")
        self.loader.GetROIImgSizes(self.roi_sizes_wh)
#Mask info of a batch
        self.mask_count = np.zeros(self.count_batch, dtype="int32")
        self.mask_size = self.loader.GetMaskCount(self.mask_count)
        self.polygon_size = np.zeros(self.mask_size, dtype= "int32")
        self.mask_data = np.zeros(100000, dtype = "float32")
        self.loader.GetMaskCoordinates(self.polygon_size, self.mask_data)

        count =0
        sum_count=0
        j = 0
        iteration1 = 0
        iteration = 0
        self.target_batch = []
        for i in range(self.bs):
            count = self.bboxes_label_count[i]
            self.img_name = self.Img_name[i*16:(i*16)+12]
            self.img_name=self.img_name.decode('utf_8')
            self.img_name = np.char.lstrip(self.img_name, chars ='0')
            self.img_roi_size2d_numpy_wh = (self.roi_sizes_wh[i*2:(i*2)+2])

            self.label_2d_numpy = (self.labels[sum_count : sum_count+count])
            self.bb_2d_numpy = (self.bboxes[sum_count*4 : (sum_count+count)*4])

            for index, _ in enumerate(self.bb_2d_numpy):
                if index % 2 == 0:
                    self.bb_2d_numpy[index] = self.bb_2d_numpy[index] * self.img_roi_size2d_numpy_wh[0]
                elif index % 2 != 0:
                    self.bb_2d_numpy[index] = self.bb_2d_numpy[index] * self.img_roi_size2d_numpy_wh[1]


            self.bb_2d_numpy = np.reshape(self.bb_2d_numpy, (-1, 4)).tolist()

            self.target = BoxList(self.bb_2d_numpy, (self.img_roi_size2d_numpy_wh[0],self.img_roi_size2d_numpy_wh[1]), mode="xyxy")
            self.target.add_field("labels", self.label_2d_numpy)

            self.count_mask = self.bboxes_label_count[i]
            poly_batch_list = []
            for i in range(self.count_mask):
                poly_list = []
                for _ in range(self.mask_count[iteration1]):
                    polygons = []
                    polygon_size_check = self.polygon_size[iteration]
                    iteration = iteration + 1
                    for _ in range(polygon_size_check):
                        polygons.append(self.mask_data[j])
                        j = j + 1
                    poly_list.append(polygons)
                iteration1 = iteration1 + 1
                poly_batch_list.append(poly_list)

            masks = SegmentationMask(poly_batch_list, (self.img_roi_size2d_numpy_wh[0],self.img_roi_size2d_numpy_wh[1]))
            self.target.add_field("masks", masks)

            self.target_batch.append(self.target)
            sum_count = sum_count +count

        self.img_list_obj = ImageList(self.out ,[(self.img_roi_size2d_numpy_wh[1],self.img_roi_size2d_numpy_wh[0])])
        if self.display:
            for i in range(self.bs):
                img_name = self.Img_name[i*16:(i*16)+12].decode('utf-8')
                image = self.img_list_obj.tensors[i].cpu().numpy()
                if self.tensor_format == types.NCHW:
                    PIL_image = np.array(Image.fromarray(((image.transpose(1,2,0))+[102.9801, 115.9465, 122.7717]).astype('uint8'), 'RGB'))
                else:
                    PIL_image = np.array(Image.fromarray((image+[102.9801, 115.9465, 122.7717]).astype('uint8'), 'RGB'))
                PIL_image = cv2.cvtColor(PIL_image, cv2.COLOR_RGB2BGR)
                for box in self.target_batch[i].bbox:
                    x1, y1, x2, y2 = box.cpu().numpy().astype(np.int32)
                    cv2.rectangle(PIL_image, (x1,y1), (x2,y2), (255, 0, 0), 2)
                cv2.imwrite(f'OUTPUT_IMAGES_PYTHON/NEW_API/COCO_READER_SEGM/{img_name}.jpg', PIL_image)
        return self.img_list_obj, self.target_batch


    def reset(self):
        self.loader.rocalResetLoaders()

    def __iter__(self):
        self.loader.rocalResetLoaders()
        return self

def main():
    args = parse_args()
    # Args
    image_path = args.image_dataset_path
    ann_path = args.json_path
    rocal_cpu = False if args.rocal_gpu else True
    bs = args.batch_size
    display = args.display
    num_threads = args.num_threads
    local_rank = args.local_rank
    world_size = args.world_size
    random_seed = args.seed
    tensor_format = types.NHWC if args.NHWC else types.NCHW
    tensor_dtype = types.FLOAT16 if args.fp16 else types.FLOAT
    try:
        path = "OUTPUT_IMAGES_PYTHON/NEW_API/COCO_READER_SEGM/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)

    pipe = Pipeline(batch_size=bs, num_threads=num_threads, device_id=local_rank, seed=random_seed, rocal_cpu=rocal_cpu, mean=[102.9801, 115.9465, 122.7717], std=[1. , 1., 1.], tensor_dtype=tensor_dtype)

    with pipe:
        jpegs, bboxes, labels = fn.readers.coco(
            file_root=image_path, annotations_file=ann_path, random_shuffle=True, seed=local_rank, masks=True)
        images_decoded = fn.decoders.image(jpegs, output_type=types.RGB, file_root=image_path, annotations_file=ann_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True, seed=local_rank)
        coin_flip = fn.random.coin_flip(probability=0.5)
        rmn_images = fn.resize_mirror_normalize(images_decoded,
                                            device="gpu",
                                            output_dtype=types.UINT8,
                                            output_layout=types.NHWC,
                                            resize_min = 1344,
                                            resize_max = 1344,
                                            mirror=coin_flip,
                                            mean= [0., 0., 0.],
                                            std = [1. , 1., 1.])
        pipe.set_outputs(rmn_images)

    pipe.build()
    if(rocal_cpu):
        data_loader = ROCALCOCOIterator(
            pipe, multiplier=pipe._multiplier, offset=pipe._offset, display=display, tensor_layout=tensor_format, tensor_dtype=tensor_dtype, device="cpu")
    else:
        data_loader = ROCALCOCOIterator(
            pipe, multiplier=pipe._multiplier, offset=pipe._offset, display=display, tensor_layout=tensor_format, tensor_dtype=tensor_dtype, device="cuda")
    epochs = 1
    for epoch in range(int(epochs)):
        print("EPOCH:::::",epoch)
        for i, (images, targets) in enumerate(tqdm(data_loader)):
            if i == 0:
                print("*******************************",i,"************************")
                print("****************IMAGES****************")
                print(images.image_sizes)
                print("\nBBOXES:\n", targets[0].bbox)
                print("\nLABELS:\n", targets[0].extra_fields['labels'])
                print("\nMASK POLYGONS :\n", targets[0].extra_fields['masks'])
        data_loader.reset()


if __name__ == '__main__':
    main()
