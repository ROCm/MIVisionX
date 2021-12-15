from amd.rali.plugin.tf import RALIIterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types
import sys
import tensorflow as tf
import amd.rali.fn as fn
import numpy as np


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


    pipe = Pipeline(batch_size=bs, num_threads=nt,device_id=di, seed=2, rali_cpu=raliCPU)
    # pipe = HybridPipe(feature_key_map=featureKeyMap, tfrecordreader_type=TFRecordReaderType, batch_size=bs, num_threads=nt, device_id=di, data_dir=imagePath, crop=cropSize, oneHotLabels=oneHotLabel, rali_cpu=raliCPU)
    with pipe:
        inputs = fn.readers.tfrecord(path=imagePath, index_path = "", reader_type=TFRecordReaderType, user_feature_key_map=featureKeyMap, 
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
        jpegs = inputs["image/encoded"]
        labels = inputs["image/class/label"]
        images = fn.decoders.image(jpegs, user_feature_key_map=featureKeyMap, output_type=types.RGB)
        resized = fn.resize(images, resize_x=300, resize_y=300)
        pipe.set_outputs(resized)

    imageIterator = RALIIterator(pipe, display=True)

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
    imageIterator.reset()


if __name__ == '__main__':
    main()
