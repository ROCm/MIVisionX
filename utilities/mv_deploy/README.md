## mv_deploy

*This project has the source code for MIVIsionX model compiler in mv_compile.cpp*

mv_deploy consists of a model-compiler and necessary header/.cpp files which are required to run inference for a specific NeuralNet model

The "mv_compile" will be built as part of MIVisionX package installer
To build and application using mv_compile, the user can use the deployment api from mv_deploy.h.
The entire use of the mv_compile and deployment is shown in [mv_objdetectsample](../samples/inference/mv_objdetect)
The sample demonstrates the use of mv_compile utility to do video decoding and inference.

## Prerequisites

* Ubuntu `16.04`/`18.04` or CentOS `7.5`/`7.6`
* [ROCm supported hardware](https://rocm.github.io/ROCmInstall.html#hardware-support) 
	* AMD Radeon GPU or APU required
* [ROCm](https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories)
* Build & Install [MIVisionX](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX#linux-1)
	* MIVisionX installs model compiler at `/opt/rocm/mivisionx`
  * mv_compile installs at `/opt/rocm/mivisionx/bin` and mvdeploy_api.h installs at `/opt/rocm/mivisionx/include` 


### Usage
The mv_compile utility generates deployment library, header files, and .cpp files required to run inference for the specified model.

* Usage:
```
mv_compile   --model 	        <model_name: name of the trained model with path> 		[required]
	     --install_folder   <install_folder:  the location for compiled model> 		[required]
	     --input_dims 	<input_dims: n,c,h,w - batch size, channels, height, width> 	[required]
	     --backend 	        <backend: name of the backend for compilation> 	  		[optional - default:OpenVX_Rocm_GPU]
	     --fuse_cba 	<fuse_cba: enable or disable Convolution_bias_activation fuse mode (0/1)> [optional - default: 0]
	     --quant_mode       <quant_mode: fp32/fp16 - quantization_mode for the model: if enabled the model and weights would be converted [optional -default: fp32]
```

# License
This project is licensed under the MIT License - see the LICENSE.md file for details

# Author
rrawther@amd.com
