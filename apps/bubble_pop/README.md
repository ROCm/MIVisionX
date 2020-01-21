[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img width="50%" src="https://upload.wikimedia.org/wikipedia/en/thumb/d/dd/OpenVX_logo.svg/1920px-OpenVX_logo.svg.png" /></p>

# OpenVX Samples

<a href="https://www.khronos.org/openvx/" target="_blank">Khronos OpenVX™</a> is an open, royalty-free standard for cross-platform acceleration of computer vision applications. OpenVX enables performance and power-optimized computer vision processing, especially important in embedded and real-time use cases such as face, body, and gesture tracking, smart video surveillance, advanced driver assistance systems (ADAS), object and scene reconstruction, augmented reality, visual inspection, robotics and more.

In this project, we provide OpenVX sample applications to use with any conformant implementation of OpenVX.

 ## VX Bubble Pop Sample

In this sample we will create an OpenVX graph to run VX Bubble Pop on a live camera. This sample application uses <a href="https://en.wikipedia.org/wiki/OpenCV" target="_blank">OpenCV</a> to decode input image, draw bubbles/donuts and display the output.

 <p align="center"><img width="60%" src="../../docs/images/vx-pop-app.gif" /></p>

### Prerequisites

* MIVisionX or Conformant OpenVX Implementation

* [OpenCV](https://github.com/opencv/opencv/releases/tag/3.4.0)

* Camera

### Steps to run the Bubble Pop sample

* **Step - 1:** Build and install [Conformant OpenVX Implementation](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX). In this example we will use the MIVisionX available on [GitHub](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX)

```
Build OpenVX on Linux

* Git Clone project

      git clone --recursive https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX

* Follow Build & Install instructions for MIVisionX 

      https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX#build--install-mivisionx
```

* **Step - 2:** Export OpenVX Directory Path

```
export PATH=$PATH:/opt/rocm/mivisionx/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx/lib
```

* **Step - 3:** Clone the MIVisionX project and build the bubble pop application

```
cd ~/ && mkdir OpenVX-bubble-pop
cd OpenVX-bubble-pop/
git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX
```

* **Step - 4:** CMake and Build the pop application

```
mkdir pop-build && cd pop-build
cmake ../MIVisionX/apps/bubble_pop/
make
```

* **Step - 5:** Run VX Pop application

    * **Bubbles**
    
    ```
    ./vxPop --bubble
    ```
    
    * **Donuts**
    
    ````
    ./vxPop --donut
    ````
