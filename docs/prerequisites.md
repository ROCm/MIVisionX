## Prerequisites

### Hardware

* **CPU**: [AMD64](https://docs.amd.com/bundle/Hardware_and_Software_Reference_Guide/page/Hardware_and_Software_Support.html)
* **GPU**: [AMD Radeon&trade; Graphics](https://docs.amd.com/bundle/Hardware_and_Software_Reference_Guide/page/Hardware_and_Software_Support.html) [optional]
* **APU**: [AMD Radeon&trade; `Mobile`/`Embedded`](https://docs.amd.com/bundle/Hardware_and_Software_Reference_Guide/page/Hardware_and_Software_Support.html) [optional]

  **Note:** Some modules in MIVisionX can be built for `CPU ONLY`. To take advantage of `Advanced Features And Modules` we recommend using `AMD GPUs` or `AMD APUs`.

### Operating System

#### Windows

* Windows `10` / `11`
* Windows SDK
* Visual Studio 2019 or later
* Install the latest AMD [drivers](https://www.amd.com/en/support)
* Install [OpenCL SDK](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0)
* Install [OpenCV 4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
  + Set `OpenCV_DIR` environment variable to `OpenCV/build` folder
  + Add `%OpenCV_DIR%\x64\vc14\bin` or `%OpenCV_DIR%\x64\vc15\bin` to your `PATH`

#### macOS

* Install [Homebrew](https://brew.sh)
* Install [CMake](https://cmake.org)
* Install OpenCV `3`/`4`

  **Note:** macOS [build instructions](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/wiki/macOS#macos-build-instructions)

#### Linux

* Linux distribution
  + **Ubuntu** - `20.04` / `22.04`
  + **CentOS** - `7` / `8`
  + **RedHat** - `8` / `9`
  + **SLES** - `15-SP3`
* Install [ROCm](https://docs.amd.com)
* CMake 3.0 or later
* ROCm MIOpen for `Neural Net Extensions` ([vx_nn](amd_openvx_extensions/amd_nn#openvx-neural-network-extension-library-vx_nn))
* Qt Creator for [Cloud Inference Client](apps/cloud_inference/client_app/README.md)
* [Protobuf](https://github.com/google/protobuf) for inference generator & model compiler
  + install `libprotobuf-dev` and `protobuf-compiler` needed for vx_nn
* [OpenCV 4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
  + Set `OpenCV_DIR` environment variable to `OpenCV/build` folder
* [FFMPEG n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
  + FFMPEG is required for amd_media & mv_deploy modules
* [rocAL](rocAL#prerequisites) Prerequisites

##### Prerequisites setup script for Linux - `MIVisionX-setup.py`

For the convenience of the developer, we here provide the setup script which will install all the dependencies required by this project.

  **NOTE:** This script only needs to be executed once.

###### Prerequisites for running the script

* Linux distribution
  + Ubuntu - `20.04` / `22.04`
  + CentOS - `7` / `8`
  + RedHat - `8` / `9`
  + SLES - `15-SP3`
* [ROCm supported hardware](https://docs.amd.com)
* [ROCm](https://docs.amd.com)

  **usage:**

  ```
  python MIVisionX-setup.py --directory [setup directory - optional (default:~/)]
                            --opencv    [OpenCV Version - optional (default:4.6.0)]
                            --protobuf  [ProtoBuf Version - optional (default:3.12.4)]
                            --rpp       [RPP Version - optional (default:0.99)]
                            --ffmpeg    [FFMPEG V4.4.2 Installation - optional (default:ON) [options:ON/OFF]]
                            --rocal     [MIVisionX rocAL Dependency Install - optional (default:ON) [options:ON/OFF]]
                            --neural_net[MIVisionX Neural Net Dependency Install - optional (default:ON) [options:ON/OFF]]
                            --inference [MIVisionX Neural Net Inference Dependency Install - optional (default:ON) [options:ON/OFF]]
                            --reinstall [Remove previous setup and reinstall (default:OFF)[options:ON/OFF]]
                            --backend   [MIVisionX Dependency Backend - optional (default:HIP) [options:HIP/OCL/CPU]]
                            --rocm_path [ROCm Installation Path - optional (default:/opt/rocm) - ROCm Installation Required]
  ```
    **Note:**
    * **ROCm upgrade** with `sudo apt upgrade` requires the setup script **rerun**.
    * use `X Window` / `X11` for [remote GUI app control](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/wiki/X-Window-forwarding)
