## TensorFlow classification training examples - Demo with rocAL pipeline
The rocAL pipeline for image augmentations can be integrated with a tensorflow image classification training graph. The example below shows this use case. Please follow the following steps to replicate the training:

- Install docker using <https://docs.docker.com/engine/install/ubuntu/> on a host machine running Ubuntu Bionic 18.04 (LTS) or Ubuntu Xenial 16.04 (LTS)
- Install AMD ROCm using <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>
- Pull the docker image containing the example:
```
sudo docker pull abishekr/mlperf_rocm3.5_tf1.15:v0.1.0
```
- Initiate a docker container:
```
sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined abishekr/mlperf_rocm3.5_tf1.15:v0.1.0
```

### rocAL-TensorFlow training on pets dataset
This example uses the rocAL TFRecordReader to read images from the disk, perform the necessary augmentations and run a classification training on the pets dataset.
```
cd /root/tf_petsTrainingExample
```

This example can also be obtained by cloning this repository inside the docker container. This "tf_petsTrainingExample folder is present in the repository under:"
```
cd rocAL/rocAL_pybind/example/tf_petsTrainingExample
```

For first run, to setup the dataset, edit "train_withRALI_withTFRecordReader.py" and set "DATASET_DOWNLOAD_AND_PREPROCESS = True"
For subsequent runs, after the dataset has already been downloaded and preprocessed, set "DATASET_DOWNLOAD_AND_PREPROCESS = False"

```
python3 train_withRALI_withTFRecordReader.py
```

## Using rocAL TFRecordReader
For using rocAL TFRecordReader for image classification or detection purposes, please follow the standard below.

For image classification, rocAL needs the following 3 features in the user's TFRecord to be read, as seen in rocAL/pybind_python/example/tf_classification.py
```
features = 
{
    'image/encoded':tf.FixedLenFeature((), tf.string, ""),
    'image/class/label':tf.FixedLenFeature([1], tf.int64,  -1),
    'image/filename':tf.FixedLenFeature((), tf.string, "")
}
```

For object detection, rocAL needs the following 8 features in the user's TFRecord to be read, as seen in rocAL/pybind_python/example/tf_detection.py
```
features = 
{
    'image/encoded':tf.FixedLenFeature((), tf.string, ""),
    'image/class/label':tf.FixedLenFeature([1], tf.int64,  -1),
    'image/class/text':tf.FixedLenFeature([ ], tf.string, ""),
    'image/object/bbox/xmin':tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymin':tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/xmax':tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymax':tf.VarLenFeature(dtype=tf.float32),
    'image/filename':tf.FixedLenFeature((), tf.string, "")
}
```

Since the key names in your dataset's TFRecords might vary from those expected by rocAL's TFRecordReader, please provide the following 'featureKeyMap' that maps your dataset's TFRecord key names to those expected by the TFRecordReader:

### Format
```
featureKeyMap = 
{
    <Key name in rocAL TFRecordReader (the 3/8 keys above for classification/detection)> : <Corresponding key name in your dataset's TFRecords>,
    <Key name in rocAL TFRecordReader (the 3/8 keys above for classification/detection)> : <Corresponding key name in your dataset's TFRecords>,
}
```

### Possible example for Image Classification
```
featureKeyMap = 
{
    'image/encoded':'encoded',
    'image/class/label':'IDs',
    'image/filename':'filename'
}
```
or
```
featureKeyMap = 
{
    'image/encoded':'image/encoded',
    'image/class/label':'image/class/IDs',
    'image/filename':'image/filename'
}
```

### Possible example for Object Detection
```
featureKeyMap = 
{
    'image/encoded':'image/encoded',
    'image/class/label':'image/class/label',
    'image/class/text':'description',
    'image/object/bbox/xmin':'bounding_box_x_minimum',
    'image/object/bbox/ymin':'bounding_box_y_minimum',
    'image/object/bbox/xmax':'bounding_box_x_maximum',
    'image/object/bbox/ymax':'bounding_box_y_maximum',
    'image/filename':'filename'
}
```

### Other Notes
- All necessary keys for rocAL TFRecordReader are mentioned above.
- Please ignore any additional keys that may be present in your dataset's TFRecords, and do not include them as part of "featureKeyMap".
- The "features" argument passed to ops.TFRecordReader() remains same as described in tf_classification.py and tf_detection.py irrespective of changes in the "featureKeyMap" argument passed by the user from main().
- "TFRecordReaderType" is an argument, that is 0 for classification and 1 for detection as in tf_classification.py and tf_detection.py
