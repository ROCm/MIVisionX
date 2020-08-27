#!/bin/bash

DATASET_URL="http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
GROUNDTRUTH_URL="http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

apt-get install wget
printf "\nDownloading Oxford-IIIT-Pet dataset from $DATASET_URL..."
wget $DATASET_URL
printf "\nDownloading Oxford-IIIT-Pet ground truth from $GROUNDTRUTH_URL..."
wget $GROUNDTRUTH_URL
printf "\nExtracting..."
tar xzvf images.tar.gz
tar xzvf annotations.tar.gz
mkdir tfr
cd /root/models/research/object_detection
python3.6 dataset_tools/create_pet_tf_record.py --data_dir=/root/tf_petsTrainingExample --output_dir=/root/tf_petsTrainingExample/tfr
cd /root/tf_petsTrainingExample/tfr
mkdir train
mv pet_faces_train.record-0000* train
mkdir val
mv pet_faces_val.record-0000* val
printf "\nFinished dataset preparation!\n"
