from amd.rali.plugin.tf import RALIIterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types
import sys
import tensorflow as tf
import numpy as np


class HybridPipe(Pipeline):
    def __init__(self, feature_key_map, tfrecordreader_type, batch_size, num_threads, device_id, data_dir, crop, rali_cpu=True):
        super(HybridPipe, self).__init__(batch_size, num_threads,
                                         device_id, seed=12 + device_id, rali_cpu=rali_cpu)
        self.input = ops.TFRecordReader(path=data_dir, index_path="", reader_type=tfrecordreader_type, user_feature_key_map=feature_key_map,
                                        features={
                                            'image/encoded': tf.FixedLenFeature((), tf.string, ""),
                                            'image/class/label': tf.FixedLenFeature([1], tf.int64,  -1),
                                            'image/class/text': tf.FixedLenFeature([], tf.string, ''),
                                            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
                                            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
                                            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
                                            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
                                            'image/filename': tf.FixedLenFeature((), tf.string, "")
                                        }
                                        )
        rali_device = 'cpu' if rali_cpu else 'gpu'
        decoder_device = 'cpu' if rali_cpu else 'mixed'
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        # self.decode = ops.ImageDecoder(user_feature_key_map=feature_key_map,
        # 										device=decoder_device, output_type=types.RGB)
        self.decode = ops.ImageDecoderRandomCrop(user_feature_key_map=feature_key_map,
                                                 device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[
                                                     0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=rali_device, resize_x=crop, resize_y=crop)
        self.cmnp = ops.CropMirrorNormalize(device="cpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 *
                                                  255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('rali "{0}" variant'.format(rali_device))

    def define_graph(self):
        inputs = self.input(name="Reader")
        images = inputs["image/encoded"]
        labels = inputs["image/class/label"]
        images = self.decode(images)
        images = self.res(images)
        rng = self.coin()
        output = self.cmnp(images, mirror=rng)
        return [output, labels]


def get_onehot(image_labels_array, numClasses):
    one_hot_vector_list = []
    for label in image_labels_array:
        one_hot_vector = np.zeros(numClasses)
        if label[0] != 0:
            np.put(one_hot_vector, label[0] - 1, 1)
        one_hot_vector_list.append(one_hot_vector)

    one_hot_vector_array = np.array(one_hot_vector_list)

    return one_hot_vector_array


def get_weights(num_bboxes):
    weights_array = np.zeros(100)
    for pos in list(range(num_bboxes)):
        np.put(weights_array, pos, 1)

    return weights_array


def main():
    if len(sys.argv) < 5:
        print('Please pass <TensorFlow record path> <Number of classes> <"cpu"/"gpu"> <batch_size>')
        exit(0)
    imagePath = sys.argv[1]
    numClasses = int(sys.argv[2])
    if(sys.argv[3] == "cpu"):
        raliCPU = True
    else:
        raliCPU = False
    bs = int(sys.argv[4])
    nt = 1
    di = 0
    cropSize = 320
    TFRecordReaderType = 1
    featureKeyMap = {
        'image/encoded': 'image/encoded',
        'image/class/label': 'image/object/class/label',
        'image/class/text': 'image/object/class/text',
        'image/object/bbox/xmin': 'image/object/bbox/xmin',
        'image/object/bbox/ymin': 'image/object/bbox/ymin',
        'image/object/bbox/xmax': 'image/object/bbox/xmax',
        'image/object/bbox/ymax': 'image/object/bbox/ymax',
        'image/filename': 'image/filename'
    }

    pipe = HybridPipe(feature_key_map=featureKeyMap, tfrecordreader_type=TFRecordReaderType,
                      batch_size=bs, num_threads=nt, device_id=di, data_dir=imagePath, crop=cropSize, rali_cpu=raliCPU)
    pipe.build()

    imageIterator = RALIIterator(pipe)

    for i, (images_array, bboxes_array, labels_array, num_bboxes_array) in enumerate(imageIterator, 0):
        images_array = np.transpose(images_array, [0, 2, 3, 1])
        print("RALI augmentation pipeline - Processing batch %d....." % i)

        for element in list(range(bs)):
            print("Processing image %d....." % element)
            features_dict = {
                "image": images_array[element],
                "true_image_shape": np.array([len(images_array[element]), len(images_array[element, 0]), len(images_array[element, 0, 0])])
            }
            labels_dict = {
                "num_groundtruth_boxes": num_bboxes_array[element],
                "groundtruth_boxes": bboxes_array[element],
                "groundtruth_classes": get_onehot(labels_array[element], numClasses),
                "groundtruth_weights": get_weights(num_bboxes_array[element])
            }
            processed_tensors = (features_dict, labels_dict)
            print("\nPROCESSED_TENSORS:\n", processed_tensors)
        print("\n\nPrinted first batch with", (bs), "images!")
        break


if __name__ == '__main__':
    main()
