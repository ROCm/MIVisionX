# Applications

MIVisionX includes sample applications built on top of AMD OpenVX&trade;. They use OpenVX and OpenCV to build applications that can be used as prototypes or as references to develop products.

These applications are built separately, against an installed MIVisionX, using `apps/CMakeLists.txt`.

## Prerequisites
* [MIVisionX](https://github.com/ROCm/MIVisionX/blob/develop/README.md#prerequisites) installed
* OpenCV (used for video decode and display)

## ADAS Surround Pipeline

This sample [application](./adas_pipeline/README.md) builds a multi-camera surround-view lane pipeline as the three stage chain the OpenVX 1.3.2 pipelining extension (`vx_khr_pipelining`) uses as its example: the cameras are undistorted, denoised and projected onto the ground plane on the GPU, the detector runs on the CPU, and the lane markings are filtered and cleaned up on the GPU. The same work runs seven ways -- one stage at a time on either device, split across both, with queued parameters, as one graph per stage, batched, and streaming -- and `--compare` prints them as a single table, so what pipelining is worth can be read off directly. Splitting stages across devices without pipelining turns out to be *slower* than using the GPU alone, which is the point.

## Bubble Pop

This sample [application](./bubble_pop) creates bubbles and donuts to pop using OpenVX & OpenCV functionality.

<p align="center"> <img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/docs/data/vx-pop-app.gif"> </p>

## Optical Flow

This sample [application](./optical_flow/README.md) creates an OpenVX graph to run Optical Flow on a video/live stream. It uses <a href="https://en.wikipedia.org/wiki/OpenCV" target="_blank">OpenCV</a> to decode the input video and display the output.

<p align="center"> <img width="60%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/docs/data/optical_flow_video.gif"> </p>
