# rocAL

The AMD ROCm Augmentation Library (rocAL) is designed to efficiently decode and process images and videos from a variety of storage formats and modify them through a processing graph programmable by the user. rocAL currently provides C API.
For more details, go to [docs](docs) page.

## Supported Operations

rocAL can be currently used to perform the following operations either with randomized or fixed parameters:

<table>
  <tr>
    <th>Blend</th>
    <th>Blur (Gaussian 3x3)</th> 
    <th>Brightness</th>
    <th>Color Temperature</th>
  </tr>
  <tr>
    <th>ColorTwist</th>
    <th>Contrast</th>
    <th>Crop</th>
    <th>Crop Mirror Normalization</th>
  </tr>
  <tr>
    <th>CropResize</th>
    <th>Exposure Modification</th> 
    <th>Fisheye Lens</th>
    <th>Flip (Horizontal, Vertical and Both)</th>
  </tr>
  <tr>
    <th>Fog</th>
    <th>Gamma</th> 
    <th>Hue</th>
    <th>Jitter</th>
  </tr>
  <tr>
    <th>Lens Correction</th>
    <th>Pixelization</th> 
    <th>Raindrops</th>
    <th>Random Crop</th>
  </tr>
  <tr>
    <th>Resize</th>
    <th>Resize Crop Mirror</th> 
    <th>Rotation</th>
    <th>Salt And Pepper Noise</th>
  </tr>
  <tr>
    <th>Saturation</th>
    <th>Snowflakes</th> 
    <th>Vignette</th>
    <th>Warp Affine</th>
  </tr>
</table>

## Prerequisites

*  Ubuntu `20.04`/`22.04`
*  [AMD RPP](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp)
*  [AMD OpenVX&trade;](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/amd_openvx) and AMD OpenVX&trade; Extensions: `VX_RPP` and `AMD Media`
*  [Boost library](https://www.boost.org) - Version `1.66` or higher
*  [Turbo JPEG](https://libjpeg-turbo.org/) - Version `2.0` or higher
*  [Half-precision floating-point](https://half.sourceforge.net) library - Version `1.12.0` or higher
*  [Google Protobuf](https://developers.google.com/protocol-buffers) - Version `3.11.1` or higher

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

### LMDB installation

```
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

```

## Sample and test applications

*  [Image augmentation application](../apps/image_augmentation) demonstrates how rocAL's C API can be used to load jpeg images from the disk, decode them and augment the loaded images with a variety of modifications.
*  [Augmentation unit tests](../utilities/rali/rali_unittests) can be used to test rocAL's API individually.
