# rocAL

The AMD ROCm Augmentation Library (rocAL) is designed to efficiently decode and process images and videos from a variety of storage formats and modify them through a processing graph programmable by the user. rocAL currently provides C API.
For more details, go to [docs](docs) page.

## Supported Operations

rocAL can be currently used to perform the following operations either with randomized or fixed parameters:

* Brightness
* Contrast
* Gamma
* Blend
* Warp Affine
* Resize
* CropResize
* Rotation
* Flip(Horizontal, Vertical and Both)
* Blur (Gaussian 3x3)
* Fisheye lens
* Vignette
* Jitter
* Salt and pepper noise
* Snowflakes
* Raindrops
* Fog
* Color temperature
* Lens correction
* Pixelization
* Exposure modification
* Hue
* Saturation
* ColorTwist
* Crop
* Crop Mirror Normalization
* Resize Crop Mirror
* Random Crop

## Prerequisites

*  Ubuntu `16.04`/`18.04`/`20.04`
*  AMD [RPP](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp)
*  OpenVX (including RPP and Media extension)
*  Boost lib 1.66 or higher 
*  [Turbo JPEG](https://libjpeg-turbo.org/) version 2.0 or higher
*  Half float library
*  jsoncpp library
*  Google protobuf 3.11.1 or higher

## Build instructions

rocAL builds and installs as part of the MIVisonX toolkit. rocAL depends on the AMD's Radeon Performance Primitives ([RPP](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp)) library, and it needs to be installed for rocAL to build. rocAL also needs the Turbo JPEG library to decode input JPEG images.

1. Make sure to have the AMD's RPP library installed. Please refer to the [RPP's page](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp) for instructions on how to install RPP.
2. Make sure you've installed the Turbo JPEG library version 2.0 or later, refer to the section below.
3. Refer to the [MIVisonX](../README.md) page and follow build an installation steps ([Build & Install MIVisionX](../README.md#build--install-mivisionx)).

### Turbo JPEG installation

Turbo JPEG library is a SIMD optimized library which currently rocAL uses to decode input JPEG images. It needs to be built from the source and installed in the default path for libraries and include headers. You can follow the instruction below to download the source, build and install it.
Note: Make sure you have installed nasm Debian package before installation, it's the dependency required by libturbo-jpeg.

``` 
 sudo apt-get install nasm
```

Note: You need wget package to download the tar file.

``` 
 sudo apt-get install wget
```

``` 
git clone -b 2.0.6.1 https://github.com/rrawther/libjpeg-turbo.git
cd libjpeg-turbo
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_BUILD_TYPE=RELEASE  \
      -DENABLE_STATIC=FALSE       \
      -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-2.0.3 \
      -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib  \
      ..
make -j$nproc
sudo make install      
```

### Jsoncpp installation

``` 
sudo apt-get install libjsoncpp-dev
```

### LMDB installation

``` 
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

```

## Sample and test applications

*  [Image augmentation application](../apps/image_augmentation) demonstrates how rocAL's C API can be used to load jpeg images from the disk, decode them and augment the loaded images with a variety of modifications.
*  [Augmentation unit tests](../utilities/rali/rali_unittests) can be used to test rocAL's API individually.
