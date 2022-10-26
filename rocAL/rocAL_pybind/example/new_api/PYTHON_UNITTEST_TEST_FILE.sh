#!/bin/bash

if [[ $# -gt 0 ]]; then
    helpFunction()
    {
    echo ""
    echo "Usage: $0 [-n number_of_gpus] [-d dump_outputs<true/false>] [-b backend<cpu/gpu>] [-p print_tensor<true/false>]"
    echo -e "\t-n Description of what is the number of gpus to be used"
    echo -e "\t-d Description of what is the display param"
    echo -e "\t-p Description of what is the print tensor param"
    exit 1 # Exit script after printing help
    }

    while getopts "n:d:b:p:" opt
    do
        echo "In while loop"
        echo $opt
        case "$opt" in
            n ) number_of_gpus="$OPTARG" ;;
            d ) dump_outputs="$OPTARG" ;;
            b ) backend="$OPTARG" ;;
            p ) print_tensor="$OPTARG" ;;
            ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
        esac
    done

    # Print helpFunction in case parameters are empty

    if [ -z "$backend" ];
    then
        backend_arg=no-rocal-gpu
    else
        if [[ $backend == "cpu" || $backend == "CPU" ]]; then
            backend_arg=no-rocal-gpu
        elif [[ $backend == "gpu" || $backend == "GPU" ]]; then
            backend_arg=rocal-gpu
        fi
    fi

    if [ -z "$number_of_gpus" ];
    then
        gpus_per_node=1
    else
        gpus_per_node=$number_of_gpus
    fi


    if [ -z "$dump_outputs" ];
    then
        display_arg=display #True by default
    else
        if [[ $dump_outputs == "true" || $dump_outputs == "True" ]]; then
            display_arg=display
        elif [[ $dump_outputs == "false" || $dump_outputs == "False" ]]; then
            display_arg=no-display
        fi
    fi

        if [ -z "$print_tensor" ];
    then
        print_tensor_arg=print_tensor #True by default
    else
        if [[ $print_tensor == "true" || $print_tensor == "True" ]]; then
            print_tensor_arg=print_tensor
        elif [[ $print_tensor == "false" || $print_tensor == "False" ]]; then
            print_tensor_arg=no-print_tensor
        fi
    fi

    echo "$number_of_gpus"
    echo "$dump_outputs"
    echo "$backend"
    echo "$print_tensor_arg"

    echo $display_arg
    echo $backend_arg
    echo $print_tensor_arg
    # exit

else
    #DEFAULT ARGS
    gpus_per_node=1
    display_arg=display
    backend_arg=no-rocal-gpu #CPU by default
    print_tensor_arg=no-print_tensor
    echo $display_arg
    echo $backend_arg
    echo $print_tensor_arg

fi

CURRENTDATE=`date +"%Y-%m-%d-%T"`

# Mention Batch Size
batch_size=10

# python version
ver=$(python3 -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";)


####################################################################################################################################
rocAL_api_python_unittest=1
####################################################################################################################################


####################################################################################################################################

    # Mention dataset_path
    data_dir=$ROCAL_DATA_PATH/images_jpg/labels_folder/


    # rocAL_api_python_unittest.py
    # By default : cpu backend, NCHW format , fp32
    # Please pass image_folder augmentation_name in addition to other common args
    # Refer rocAL_api_python_unitest.py for all augmentation names

    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name resize --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name one_hot --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name rotate --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name brightness --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name gamma_correction --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name contrast --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name flip --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name blur --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name hue_rotate_blend --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name warp_affine --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name fish_eye --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name vignette --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name jitter --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name snp_noise --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name rain --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name fog --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name pixelate --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name exposure --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name hue --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name saturation --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name color_twist --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name crop_mirror_normalize --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name nop --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name centre_crop --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name color_temp --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name copy --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name rotate_fisheye_fog --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name resize_brightness_jitter --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name vignetter_blur --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}
    python"$ver" rocAL_api_python_unittest.py --image-dataset-path $data_dir --augmentation-name snow --batch-size $batch_size --display --NHWC --local-rank 0 --world-size $gpus_per_node --num-threads 1 --num-epochs 2 --$backend_arg --$print_tensor_arg 2>&1 | tee -a run.rocAL_api_log.${CURRENTDATE}


####################################################################################################################################
