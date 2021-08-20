#!/bin/bash

sudo rm -rvf build*
mkdir build
cd build || exit
cmake ..
make

################ AVAILABLE READERS ###############
##   DEFAULT : VideoReader                      ##
##   CASE 2  : VideoReaderResize                ##
##   CASE 3  : SequenceReader                   ##
##################################################
INPUT_PATH=$1
CASE=$2

if [ -z "$INPUT_PATH" ]
  then
    echo "No input argument supplied"
    exit
fi

if [ -z "$CASE" ]
  then
    CASE=1
fi

DISPLAY=1     # (diplay:on/off)
RGB=1         # (rgb:1/gray:0)
DEVICE=0      # (cpu:0/gpu:1)
SHUFFLE=0     # (shuffle:on/off) 
BATCH_SIZE=1
SEQUENCE_LENGTH=3
STEP=3
STRIDE=1
DECODE_WIDTH=1280
DECODE_HEIGHT=720
FILELIST_FRAMENUM=1             # enables file number or timestamps parsing for text file input
ENABLE_METADATA=0               # outputs labels and names of the associated frames
ENABLE_FRAME_NUMBER=0           # outputs the starting frame numbers of the sequences in the batch
ENABLE_TIMESTAMPS=0             # outputs timestamps of the frames in the batch
ENABLE_SEQUENCE_REARRANGE=0     # rearranges the frames in the sequence NOTE: The order needs to be set in the rali_video_unittests.cpp

echo "$INPUT_PATH"
echo ./rali_video_unittests "$INPUT_PATH" $CASE $DEVICE $BATCH_SIZE $SEQUENCE_LENGTH $STEP $STRIDE \
$RGB $DISPLAY $SHUFFLE $DECODE_WIDTH $DECODE_HEIGHT $FILELIST_FRAMENUM \
$ENABLE_METADATA $ENABLE_FRAME_NUMBER $ENABLE_TIMESTAMPS $ENABLE_SEQUENCE_REARRANGE

./rali_video_unittests "$INPUT_PATH" $CASE $DEVICE $BATCH_SIZE $SEQUENCE_LENGTH $STEP $STRIDE \
$RGB $DISPLAY $SHUFFLE $DECODE_WIDTH $DECODE_HEIGHT $FILELIST_FRAMENUM \
$ENABLE_METADATA $ENABLE_FRAME_NUMBER $ENABLE_TIMESTAMPS $ENABLE_SEQUENCE_REARRANGE
