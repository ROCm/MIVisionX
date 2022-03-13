### TESTING ALL THE READER PIPELINES ON PYTHON IN SINGLE SHOT

./TEST_FILE.sh

### TO TEST A SINGLE READER / MULTIPLE READER PIPELINE FROM TEST_FILE.sh

####################################################################################################################################
# Make the respective " Pipeline " to test equal to "1"
rocAL_api_python_unittest=1
rocAL_api_coco_pipeline=0
rocAL_api_caffe_reader=0
rocAL_api_caffe2_reader=0
rocAL_api_tf_classification_reader=0
rocAL_api_tf_detection_pipeline=0
####################################################################################################################################

### TO TEST A SINGLE READER PIPELINE FROM CMD LINE

example: COCO Pipeline

    # Mention the number of gpus
    gpus_per_node=4

    # Mention Batch Size
    batch_size=10

    # python version
    ver=$(python -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\.\2/')

    # Mention dataset_path
    data_dir=mini_db/coco/coco_10_img/val_10images_2017/

    # Mention json path
    json_path=mini_db/coco/coco_10_img/annotations/instances_val2017.json

    # rocAL_api_coco_pipeline.py
    # By default : cpu backend, NCHW format , fp32
    # Annotation must be a json file

    python$ver rocAL_api_coco_pipeline.py --image-dataset-path $data_dir --json-path $json_path --batch-size $batch_size --display --rocal-gpu --NHWC \
        --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 1 2>&1 | tee -a run.log.rocAL_api_coco_pipeline.txt

### [ NOTE: REFER parse_config.py for more INFO on other args]
