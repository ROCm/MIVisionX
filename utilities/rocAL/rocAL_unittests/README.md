# rocAL Unit Tests
This application can be used to verify the functionality of the API offered by rocAL.

## Build Instructions

### Pre-requisites
* Ubuntu Linux, [version `16.04` or later](https://www.microsoft.com/software-download/windows10)
* rocAL library (Part of the MIVisionX toolkit)
* [OpenCV 3.4+](https://github.com/opencv/opencv/releases/tag/3.4.0)
* ROCm Performance Primitives (RPP)
* Python
* Pillow

Install Pillow library using `python3 -m pip install Pillow`

### Build
````
mkdir build
cd build
cmake ../
make
````
## Running the application

```
./rocAL_unittests

Usage: ./rocAL_unittests reader-type pipeline-type=1(classification)2(detection)3(keypoints) <image-dataset-folder> output_image_name <width> <height> test_case gpu=1/cpu=0 rgb=1/grayscale=0 one_hot_labels=num_of_classes/0  display_all=0(display_last_only)1(display_all)
```

### Output verification 

The bash script `testAllScript.sh` can be used to run and dump the outputs for all test cases in rocAL and run the python script to verify the correctness of the generated outputs with the golden outputs.

Input data is available in the following link : [MIVisionX-data](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX-data)

`export ROCAL_DATA_PATH=<absolute_path_to_MIVIsionX-data>`

```
./testAllScripts.sh <device_type 0/1/2> <color_format 0/1/2>
```

Device Type
* Option 0 - For only HOST backend
* Option 1 - For only HIP backend
* Option 2 - For both HOST and HIP backend

Color Format
* Option 0 - For only Greyscale inputs
* Option 1 - For only RGB inputs
* Option 2 - For both Greyscale and RGB inputs