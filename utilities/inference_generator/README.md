# Inference Generator

caffe2openvx: Convert a pre-trained CAFFE model into a C library for use by applications.
* Extract neural network model from `deploy.prototxt`
  + generate C code that instantiates OpenVX kernels from [vx_nn](../../vx_nn/README.md) module
  + generate build scripts that package C code into a library
  + the generated C code or library can be easily integrated into an application for running inference
* Extract weights and biases from `weights.caffemodel` into separates folders for use by the C library during initialization
* Also generate a GDF for quick prototyping and kernel debugging

The generated C code will have two functions in `annmodule.h`:

```
void annGetTensorDimensions(
        vx_size dimInput[4],    // input tensor dimensions
        vx_size dimOutput[4]    // output tensor dimensions
    );

vx_graph annCreateGraph(
        vx_context context,     // OpenVX context
        vx_tensor input,        // input tensor
        vx_tensor output,       // output tensor
        const char * dataFolder // folder with weights and biases
    );
or
vx_graph annCreateGraphWithInputImage(
        vx_context context,     // OpenVX context
        vx_image input,         // input image (RGB or U8)
        vx_tensor output,       // output tensor
        const char * dataFolder // folder with weights and biases
    );
or
vx_graph annCreateGraphWithInputImageWithArgmaxTensor(
        vx_context context,     // OpenVX context
        vx_image input,         // input image (RGB or U8)
        vx_tensor output,       // output tensor
        const char * dataFolder // folder with weights and biases
    );
or
vx_graph annCreateGraphWithInputImageWithArgmaxImage(
        vx_context context,     // OpenVX context
        vx_image input,         // input image (RGB or U8)
        vx_image output,        // output image (U8)
        const char * dataFolder // folder with weights and biases
    );
or
vx_graph annCreateGraphWithInputImageWithArgmaxImageWithLut(
        vx_context context,     // OpenVX context
        vx_image input,         // input image (RGB or U8)
        vx_image output,        // output image (RGB)
        const char * dataFolder // folder with weights and biases
    );
```

* `annGetTensorDimensions`: allows an application to query dimensions of input and output tensors
* `annCreateGraph` (or another variant above): creates and initializes a graph with trained neural network for inference

## Command-line Usage

```
  % caffe2openvx
        [options]
        <net.prototxt|net.caffemodel>
        [n c H W [type fixed-point-position [convert-policy round-policy]]]
```

| option | description |
| ------ | ----------- |
| --(no-)error-messages     | do/don't enable error messages (default: ON) |
| --(no-)virtual-buffers    | do/don't use virtual buffers (default: ON) |
| --(no-)generate-gdf       | do/don't generate RunVX GDF with weight/bias initialization (default: ON) |
| --(no-)generate-vx-code   | do/don't generate OpenVX C Code with weight/bias initialization (default: ON) |
| --output-dir <folder>     | specify output folder for weights/biases, GDF, and OpenVX C Code (default: current) |
| --input-rgb <a> <b> <rev> | convert input from RGB image into tensor using (a*x+b) conversion: rev=(BGR?1:0) |
| --input-u8  <a> <b>       | convert input from U8 image into tensor using (a*x+b) conversion |
| --argmax-tensor u8/u16 k  | return argmax output with specified tensor type and top_k |
| --argmax-image u8/u16     | return argmax output with specified image type |
| --argmax-lut <rgbLut.txt> | argmax color table: one R G B entry per label |
| --flags <int>             | specify custom flags (default: 0) |

## Example

Make sure that all executables and libraries are in `PATH` and `LD_LIBRARY_PATH` environment variables.

```
% export PATH=$PATH:/opt/rocm/mivisionx/bin
% export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx/lib
```

Below log outlines a simple use-case with inference generator.

```
% caffe2openvx weights.caffemodel 1 3 32 32
% caffe2openvx deploy.prototxt 1 3 32 32
% ls
CMakeLists.txt   annmodule.txt   cmake              weights
annmodule.cpp    anntest.cpp     deploy.prototxt    weights.caffemodel
annmodule.h      bias            net.gdf
% mkdir build
% cd build
% cmake ..
% make
% cd ..
% ls build
CMakeCache.txt  Makefile        cmake_install.cmake
CMakeFiles      anntest         libannmodule.so
% ./build/anntest
OK: annGetTensorDimensions() => [input 32x32x3x32] [output 1x1x10x32]
```

The `anntest.cpp` is a simple program to initialize and run neural network using the `annmodule` library.
