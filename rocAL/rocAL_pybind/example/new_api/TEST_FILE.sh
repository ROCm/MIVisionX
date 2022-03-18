
CURRENTDATE=`date +"%Y-%m-%d-%T"`

# Mention the number of gpus
gpus_per_node=1

# Mention Batch Size
batch_size=10

# python version
ver=$(python -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\.\2/')


####################################################################################################################################
# USER TO MAKE CHANGES HERE FOR TEST
# Make the respective " Pipeline " to test equal to 1
rocAL_api_python_unittest=1
rocAL_api_coco_pipeline=0
rocAL_api_caffe_reader=0
rocAL_api_caffe2_reader=0
rocAL_api_tf_classification_reader=0
rocAL_api_tf_detection_pipeline=0
rocAL_api_video_pipeline=0
####################################################################################################################################




####################################################################################################################################
if [[ rocAL_api_python_unittest -eq 1 ]]; then

    # Mention dataset_path
    data_dir=$ROCAL_DATA_PATH/images_jpg/labels_folder/


    # rocAL_api_python_unittest.py
    # By default : cpu backend, NCHW format , fp32
    # Please pass image_folder augmentation_nanme in addition to other common args
    # Refer rocAL_api_python_unitest.py for all augmentation names
    python$ver rocAL_api_python_unittest.py \
        --image-dataset-path $data_dir \
        --augmentation-name snow \
        --batch-size $batch_size \
        --display \
        --NHWC \
        --local-rank 0 \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 2
        # 2>&1 | tee -a run.log.rocAL_api_python_unittest.${CURRENTDATE}
fi
####################################################################################################################################


####################################################################################################################################
if [[ rocAL_api_coco_pipeline -eq 1 ]]; then

    # Mention dataset_path
    data_dir=$ROCAL_DATA_PATH/coco/coco_10_img/val_10images_2017/


    # Mention json path
    json_path=$ROCAL_DATA_PATH/coco/coco_10_img/annotations/instances_val2017.json

    # rocAL_api_coco_pipeline.py
    # By default : cpu backend, NCHW format , fp32
    # Please pass annotation path in addition to other common args
    # Annotation must be a json file
    python$ver rocAL_api_coco_pipeline.py \
        --image-dataset-path $data_dir \
        --json-path $json_path \
        --batch-size $batch_size \
        --display \
        --rocal-gpu \
        --NHWC \
        --local-rank 0 \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1
        # 2>&1 | tee -a run.log.rocAL_api_coco_pipeline.${CURRENTDATE}.txt
fi
####################################################################################################################################


####################################################################################################################################
if [[ rocAL_api_caffe_reader -eq 1 ]]; then

    # Mention dataset_path
    # Classification
    data_dir=$ROCAL_DATA_PATH/caffe/classification/ilsvrc12_train_lmdb/

    # rocAL_api_caffe_reader.py
    # By default : cpu backend, NCHW format , fp32
    # use --classification for Classification / --no-classification for Detection

    python$ver rocAL_api_caffe_reader.py \
        --image-dataset-path $data_dir \
        --classification \
        --batch-size $batch_size \
        --display \
        --rocal-gpu \
        --NHWC \
        --local-rank 0 \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1
        # 2>&1 | tee -a run.log.rocAL_api_caffe_reader_classification.${CURRENTDATE}.txt
fi
####################################################################################################################################



####################################################################################################################################
if [[ rocAL_api_caffe_reader -eq 1 ]]; then

    # Mention dataset_path
    # Detection
    data_dir=$ROCAL_DATA_PATH/caffe/detection/lmdb_record/

    # rocAL_api_caffe_reader.py
    # By default : cpu backend, NCHW format , fp32
    # use --classification for Classification / --no-classification for Detection

    python$ver rocAL_api_caffe_reader.py \
        --image-dataset-path $data_dir \
        --no-classification \
        --batch-size $batch_size \
        --display \
        --rocal-gpu \
        --NHWC \
        --local-rank 0 \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1
        # 2>&1 | tee -a run.log.rocAL_api_caffe_reader_detection.${CURRENTDATE}.txt
fi
####################################################################################################################################



####################################################################################################################################
if [[ rocAL_api_caffe2_reader -eq 1 ]]; then

    # Mention dataset_path
    # Classification
    data_dir=$ROCAL_DATA_PATH/caffe2/classfication/imagenet_val5_encode/

    # rocAL_api_caffe2_reader.py
    # By default : cpu backend, NCHW format , fp32
    # use --classification for Classification / --no-classification for Detection

    python$ver rocAL_api_caffe2_reader.py \
        --image-dataset-path $data_dir \
        --classification \
        --batch-size $batch_size \
        --display \
        --NHWC \
        --local-rank 0 \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1
        # 2>&1 | tee -a run.log.rocAL_api_caffe2_reader_classification.${CURRENTDATE}.txt
fi
####################################################################################################################################


####################################################################################################################################
if [[ rocAL_api_caffe2_reader -eq 1 ]]; then

    # Mention dataset_path
    # Detection
    data_dir=$ROCAL_DATA_PATH/caffe2/detection/lmdb_records/

    # rocAL_api_caffe2_reader.py
    # By default : cpu backend, NCHW format , fp32
    # use --classification for Classification / --no-classification for Detection

    python$ver rocAL_api_caffe2_reader.py \
        --image-dataset-path $data_dir \
        --no-classification \
        --batch-size $batch_size \
        --display \
        --NHWC \
        --local-rank 0 \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1
        # 2>&1 | tee -a run.log.rocAL_api_caffe2_reader_detection.${CURRENTDATE}.txt
fi
####################################################################################################################################


####################################################################################################################################
if [[ rocAL_api_tf_classification_reader -eq 1 ]]; then

    # Mention dataset_path
    # Classification
    data_dir=$ROCAL_DATA_PATH/tf/classification/
    # rocAL_api_tf_classification_reader.py
    # By default : cpu backend, NCHW format , fp32
    # use --classification for Classification / --no-classification for Detection

    python$ver rocAL_api_tf_classification_reader.py \
        --image-dataset-path $data_dir \
        --classification \
        --batch-size $batch_size \
        --display \
        --NHWC \
        --local-rank 0 \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1
        # 2>&1 | tee -a run.log.rocAL_api_tf_classification_reader.${CURRENTDATE}.txt
fi
####################################################################################################################################


####################################################################################################################################
if [[ rocAL_api_tf_detection_pipeline -eq 1 ]]; then

    # Mention dataset_path
    # Detection
    data_dir=$ROCAL_DATA_PATH/tf/detection/
    # rocAL_api_tf_detection_pipeline.py
    # By default : cpu backend, NCHW format , fp32
    # use --classification for Classification / --no-classification for Detection

    python$ver rocAL_api_tf_detection_pipeline.py \
        --image-dataset-path $data_dir \
        --no-classification \
        --batch-size 100 \
        --display \
        --NHWC \
        --local-rank 0 \
        --world-size $gpus_per_node \
        --num-threads 1 \
        --num-epochs 1
        # 2>&1 | tee -a run.log.rocAL_api_tf_detection_pipeline.${CURRENTDATE}.txt
fi
####################################################################################################################################


####################################################################################################################################
if [[ rocAL_api_video_pipeline -eq 1 ]]; then

    # Mention dataset_path
    # Detection
    data_dir=$ROCAL_DATA_PATH/../video_and_sequence_samples/labelled_videos/
    # rocAL_api_video_pipeline.py
    # By default : cpu backend, NCHW format , fp32

    python$ver rocAL_api_video_pipeline.py \
        --video-path $data_dir \
        --batch-size 10 \
        --display \
        --sequence-length 3 \
        --num-epochs 1
        # 2>&1 | tee -a run.log.rocAL_api_video_pipeline.${CURRENTDATE}.txt
fi
####################################################################################################################################