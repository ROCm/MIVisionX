# Object detection with video decoding sample

*This project shows how to run video decoding and object detection using yolo:*

### Pre_requisits: build and install MIVisionX-develop
Install MIVisionX-develop from https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX-develop.git

### Step 1. compile model for OPENCL-ROCm-OpenVX (default) backend using the following command

```
mv_compile.exe --model <caffemodel> --install_folder <folder> --input_dims n,c,h,w
For object detection example choose model like yolov2.
For running object detection with 4 video streams: compile model with batchsize = 4
```
There will be a file libmv_deploy.so (under ./lib), weights.bin and mvtestdeploy sample app (under ./bin).
Also there will be mv_extras folder for postprocessing options.

### Step 2. Make sure mvtestdeploy runs
```	
./bin/mvtestdeploy <inputdatafile> <output.bin> --install_folder . -t N
This runs inference for an input file and generate output for N number of iterations and generates output
```
### Step 3. Build mv_objdetect sample

```
copy libmv_deploy.so under lib folder, weights.bin, mv_deploy.h, mv_deploy_api.cpp and mv_extras folder for postprocessing to mv_objdetect folder
OR copy all files from mv_objdetct sample into install_folder where everything is there to build and run the sample
```

### Step 4. cmake and make mvobjdetect
```
mkdir build && cd build && cmake -DUSE_POSTPROC=ON ../
make -j
```

### Step 5. Run object detection with video/image

```
./build/mv_objdetect test.mp4 - --install_folder . --frames 5000 --bb 20 0.2 0.4 --v
OR ./build/mv_objdetect <txt file with 4 streams> - --install_folder . --frames N --bb 20 0.2 0.4 --v
```

# License
This project is licensed under the MIT License - see the LICENSE.md file for details

# Author
rrawther@amd.com
