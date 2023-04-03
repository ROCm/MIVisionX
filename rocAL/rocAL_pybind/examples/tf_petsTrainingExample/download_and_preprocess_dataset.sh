#!/bin/bash
cwd=$pwd
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
mkdir tf_pets_records
git clone https://github.com/swetha097/rocALmodels.git
cd rocALmodels
git checkout TF_V2
cd ../
python3 rocALmodels/models/research/object_detection/dataset_tools/create_pet_tf_record.py --data_dir=./ --output_dir=tf_pets_records/ --label_map_path=rocALmodels/models/research/object_detection/data/pet_label_map.pbtxt
cd tf_pets_records
mkdir train
mv pet_faces_train.record-0000* train
mkdir val
mv pet_faces_val.record-0000* val
cd $cwd
printf "\nFinished dataset preparation!\n"

