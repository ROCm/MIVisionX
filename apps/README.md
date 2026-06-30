# Applications

MIVisionX includes sample applications built on top of AMD OpenVX&trade;. They use OpenVX and OpenCV to build applications that can be used as prototypes or as references to develop products.

These applications are built separately, against an installed MIVisionX, using `apps/CMakeLists.txt`.

## Prerequisites
* [MIVisionX](https://github.com/ROCm/MIVisionX/blob/develop/README.md#prerequisites) installed
* OpenCV (used for video decode and display)

## Bubble Pop

This sample [application](./bubble_pop) creates bubbles and donuts to pop using OpenVX & OpenCV functionality.

<p align="center"> <img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/docs/data/vx-pop-app.gif"> </p>

## Optical Flow

This sample [application](./optical_flow/README.md) creates an OpenVX graph to run Optical Flow on a video/live stream. It uses <a href="https://en.wikipedia.org/wiki/OpenCV" target="_blank">OpenCV</a> to decode the input video and display the output.

<p align="center"> <img width="60%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/docs/data/optical_flow_video.gif"> </p>
