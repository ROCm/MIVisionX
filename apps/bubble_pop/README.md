[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# VX Bubble Pop

A sample OpenVX application that creates bubbles and donuts on a live camera feed using OpenVX graph execution and OpenCV for display.

<p align="center"><img width="60%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/vx-pop-app.gif" /></p>

## Prerequisites

* MIVisionX installed (see [installation instructions](../../README.md#installation-instructions))
* OpenCV `3.4` or later
* A connected camera

## Build

These apps are built separately against an installed MIVisionX using the `apps/CMakeLists.txt`:

```shell
mkdir pop-build && cd pop-build
cmake ../MIVisionX/apps/bubble_pop/
make
```

## Run

```shell
# Bubbles mode
./vxPop --bubble

# Donuts mode
./vxPop --donut
```
