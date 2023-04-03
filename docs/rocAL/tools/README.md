# Steps to create MXNet RecordIO files using MXNet's im2rec.py script

https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py

## MXNet Installation

pip install mxnet

## Step1 : to create .lst file

python im2rec.py --list test Dataset_path --recursive

test - name of your .lst file

Dataset_path - path to the list of image folders

--recursive - If set recursively walk through subdirs and assign an unique label to images in each folder. Otherwise only include images in the root folder and give them label 0

## Step2 : to create RecordIO files

python im2rec.py lst_file Dataset_path

lst_file - *.lst file created using Step1

Dataset_path - path to the list of image folders
