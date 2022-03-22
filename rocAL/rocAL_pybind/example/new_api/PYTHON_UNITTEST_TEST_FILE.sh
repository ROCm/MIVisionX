if [[ $# -gt 0 ]]; then
    helpFunction()
    {
    echo ""
    echo "Usage: $0 [-n number_of_gpus] [-d display_param<true/false>]"
    echo -e "\t-n Description of what is the number of gpus to be used"
    echo -e "\t-d Description of what is the display param"
    exit 1 # Exit script after printing help
    }

    while getopts "n:d:" opt
    do
        echo "In while loop"
        echo $opt
        case "$opt" in
            n ) number_of_gpus="$OPTARG" ;;
            d ) display_param="$OPTARG" ;;
            ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
        esac
    done

    # Print helpFunction in case parameters are empty
    if [ -z "$number_of_gpus" ] || [ -z "$display_param" ]
    then
        echo "Some or all of the parameters are empty";
        helpFunction
    fi

    # Begin script in case all parameters are correct
    echo "$number_of_gpus"
    echo "$display_param"
    gpus_per_node=$number_of_gpus
    if [[ $display_param == "true" || $display_param == "True" ]]; then
        display_arg=display
    elif [[ $display_param == "false" || $display_param == "False" ]]; then
        display_arg=no-display
    fi
    echo $display_arg

else
    #DEFAULT ARGS
    gpus_per_node=1
    display_arg=display
fi

CURRENTDATE=`date +"%Y-%m-%d-%T"`

# Mention the number of gpus
gpus_per_node=1

# Mention Batch Size
batch_size=10

# python version
ver=$(python -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\.\2/')


####################################################################################################################################
rocAL_api_python_unittest=1
####################################################################################################################################


####################################################################################################################################

    # Mention dataset_path
    data_dir=$ROCAL_DATA_PATH/rocal_data/images_jpg/labels_folder/


    # rocAL_api_python_unittest.py
    # By default : cpu backend, NCHW format , fp32
    # Please pass image_folder augmentation_name in addition to other common args
    # Refer rocAL_api_python_unitest.py for all augmentation names

    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name resize --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.resize.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name rotate --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.rotate.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name brightness --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.brightness.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name gamma_correction --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.gamma_correction.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name contrast --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.contrast.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name flip --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.flip.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name blur --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.blur.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name hue_rotate_blend --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.hue_rotate_blend.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name warp_affine --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.warp_affine.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name fish_eye --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.fish_eye.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name vignette --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.vignette.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name jitter --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.jitter.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name snp_noise --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.snp_noise.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name rain --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.rain.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name fog --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.fog.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name pixelate --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.pixelate.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name exposure --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.exposure.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name hue --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.hue.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name saturation --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.saturation.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name color_twist --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.color_twist.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name crop_mirror_normalize --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.crop_mirror_normalize.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name nop --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.nop.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name centre_crop --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.centre_crop.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name color_temp --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.color_temp.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name copy --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.copy.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name rotate_fisheye_fog --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.rotate_fisheye_fog.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name resize_brightness_jitter --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.resize_brightness_jitter.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name vignetter_blur --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.vignetter_blur.${CURRENTDATE}
    python$ver rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name snow --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 2>&1 | tee -a run.log.snow.${CURRENTDATE}




####################################################################################################################################