#!/bin/bash

############# Edit GDF path and file names #############
GDF_PATH="../../../vision_tests/gdfs"
GDF_FILE_LIST="01_absDiff.gdf 02_accumulate.gdf 03_accumulateSquared.gdf 04_accumulateWeighted.gdf 05_add.gdf 06_and.gdf"
AFFINITY_LIST="CPU GPU"
############# Edit GDF path and file names #############

for AFFINITY in $AFFINITY_LIST;
do
    printf "\n\n---------------------------------------------"
    printf "\nRunning GDF cases on runvx for $AFFINITY..."
    printf "\n---------------------------------------------\n"
    for GDF_FILE in $GDF_FILE_LIST;
    do
        
        printf "\n$GDF_FILE..."
        runvx -affinity:$AFFINITY $GDF_PATH/$GDF_FILE
    done
done