# rocAL pybind Examples

This section provides instructions to run test scripts.

## Testing all the reader pipelines on Python in single shot

Run the test script:

`./TEST_FILE.sh`

## To test a single reader / multiple reader pipeine from TEST_FILE.sh

Make the respective "Pipeline" to test equal to "1"

```bash
rocAL_api_python_unittest=1
rocAL_api_coco_pipeline=0
```

## Install Pybind to run any tests

- [rocAL Pybind Installation](https://github.com/ROCm/MIVisionX/tree/master/README.md)

## Export ROCAL_DATA_PATH

`export ROCAL_DATA_PATH=<Absolute Path of rocal_data/>`

## Testing all the augmentations in a single shot

```bash
./PYTHON_UNITTEST_TEST_FILE.sh #The default value of number of gpu's is "1" & display is "ON" by default

./PYTHON_UNITTEST_TEST_FILE.sh -n <number_of_gpus> -d <true/false> -b backend<cpu/gpu> -p print_tensor<true/false>
./PYTHON_UNITTEST_TEST_FILE.sh -n "1" -d "false" -b "cpu"
```

## To test multiple reader pipeline from READERS_TEST_FILE.sh

```bash
./READERS_TEST_FILE.sh #The default value of number of gpu's is "1" & display is "ON" by default

./READERS_TEST_FILE.sh -n <number_of_gpus> -d <true/false>
./READERS_TEST_FILE.sh -n "1" -d "false" -b "cpu"
```

## To test a single reader pipeline from READERS_TEST_FILE.sh

Example: To run COCO Pipeline

```bash
rocAL_api_python_unittest=0
rocAL_api_coco_pipeline=1
rocAL_api_caffe_reader=0
rocAL_api_caffe2_reader=0
rocAL_api_tf_classification_reader=0
rocAL_api_tf_detection_pipeline=0
```

## To test a single reader pipeline from command line

Example: COCO Pipeline

```bash
    # Mention the number of gpus
    gpus_per_node=4

    # Mention Batch Size
    batch_size=10

    # python version
    ver=$(python -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\.\2/')

    # Mention dataset_path
    data_dir=$ROCAL_DATA_PATH/coco/coco_10_img/val_10images_2017/

    # Mention json path
    json_path=$ROCAL_DATA_PATH/coco/coco_10_img/annotations/instances_val2017.json
```

```bash
    # rocAL_api_coco_pipeline.py
    # By default : cpu backend, NCHW format , fp32
    # Annotation must be a json file

    python$ver rocAL_api_coco_pipeline.py --image-dataset-path $data_dir --json-path $json_path --batch-size $batch_size --display --rocal-gpu --NHWC \
        --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 1 2>&1 | tee -a run.log.rocAL_api_coco_pipeline.txt
```

- NOTE: Refer to `parse_config.py` for more info on other arguments
