# AMD RPP Extension
The AMD RPP extension (vx_amdrpp) is an OpenVX module that implements an interface to access RPP functionality as OpenVX kernels. These kernels can be accessed from within OpenVX framework using OpenVX API call [vxLoadKernels](https://www.khronos.org/registry/vx/specs/1.0.1/html/da/d83/group__group__user__kernels.html#gae00b6343fbb0126e3bf0f587b09393a3)(context, "vx_rpp").

## List of RPP functions 
The following is a list of RPP functions that have been included in the vx_rpp module.

    Brightness               org.rpp.Brightness
    Contrast                 org.rpp.Contrast
    Blur                     org.rpp.Blur
    Flip                     org.rpp.Flip
    Gamma Correction         org.rpp.GammaCorrection
    Image Resize             org.rpp.Resize
    Resize Crop              org.rpp.ResizeCrop
    Rotate                   org.rpp.Rotate
    Warp affine              org.rpp.WarpAffine
    Blend                    org.rpp.Blend
    Exposure                 org.rpp.Exposure
    Fisheye lens effect      org.rpp.Fisheye
    Snow                     org.rpp.Snow
    Vignette                 org.rpp.Vignette
    Lens Correction          org.rpp.LensCorrection
    Pixelate                 org.rpp.Pixelate
    Jitter                   org.rpp.Jitter
    Color temperature        org.rpp.ColorTemperature
    Rain overlay             org.rpp.Rain
    Fog effect               org.rpp.Fog
    Salt and Pepper noise    org.rpp.NoiseSnp

**NOTE** - For list of OpenVX API calls for RPP-interop refer include/[vx_ext_rpp.h](include/vx_ext_rpp.h)

## Build Instructions

### Pre-requisites
* AMD OpenVX library
* AMD RPP library [download](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp).
* CMake 2.8 or newer [download](http://cmake.org/download/).

### Build using CMake on Linux
* Use CMake to configure and generate Makefile
