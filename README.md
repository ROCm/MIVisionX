[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![doc](https://img.shields.io/badge/doc-readthedocs-blueviolet)](https://rocm.docs.amd.com/projects/MIVisionX/en/latest/)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/ROCm/MIVisionX)](https://github.com/ROCm/MIVisionX/releases)

<p align="center"><img width="70%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/docs/data/MIVisionX.png" /></p>

AMD MIVisionX is a computer vision toolkit built around a highly optimized, conformant open-source implementation of the <a href="https://registry.khronos.org/OpenVX/specs/1.3.2/html/OpenVX_Specification_1_3_2.html" target="_blank">Khronos OpenVX&trade; 1.3.2</a> specification. As of the `4.0.0` release, MIVisionX ships three components: the [AMD OpenVX&trade;](amd_openvx/README.md) engine, the [AMD RPP](amd_openvx_extensions/amd_rpp/README.md) OpenVX extension, and the [RunVX](utilities/runvx/README.md#amd-runvx) graph executor — across `CPU`, `HIP`, and `OpenCL` backends.

## AMD OpenVX&trade; Extensions

The OpenVX framework provides a mechanism for vendors to add new vision functionality. MIVisionX ships the following extension on top of the [AMD OpenVX&trade;](amd_openvx/README.md) core engine:

* [amd_rpp](amd_openvx_extensions/amd_rpp/README.md): Interface to [ROCm Performance Primitives](https://github.com/ROCm/rpp) (RPP) for image/tensor augmentation via [rocAL](https://github.com/ROCm/rocAL)

> [!NOTE]
> `amd_rpp` supports only the `CPU` and `HIP` backends (RPP has dropped OpenCL support). When the core is built with the `OpenCL` backend, `amd_rpp` is built in CPU-only mode.

## Utilities

* [RunVX](utilities/runvx/README.md#amd-runvx): Command-line utility to execute OpenVX graphs described in GDF text files

## Applications

Sample [applications](apps/README.md#applications) built on AMD OpenVX&trade; and OpenCV (built separately against an installed MIVisionX):

* [bubble_pop](apps/bubble_pop): Creates bubbles and donuts to pop using OpenVX & OpenCV
* [optical_flow](apps/optical_flow/README.md#openvx-samples): Runs Optical Flow on a video/live stream using an OpenVX graph

## Prerequisites

### Hardware

* **CPU**: AMD64
* **GPU**: [AMD Radeon&trade; Graphics / AMD Instinct&trade; Accelerators](https://rocm.docs.amd.com/en/7.13.0-preview/compatibility/compatibility-matrix.html) [optional]
* **APU**: [AMD Radeon&trade; Mobile/Embedded](https://rocm.docs.amd.com/en/7.13.0-preview/compatibility/compatibility-matrix.html) [optional]

> [!IMPORTANT]
> MIVisionX can be built for CPU only. For GPU acceleration via the `HIP` or `OpenCL` backend, an AMD GPU or APU is required.

### Operating Systems

| Platform | Supported versions |
|----------|--------------------|
| Linux (Ubuntu) | `22.04` / `24.04` |
| Linux (RedHat) | `8` / `9` |
| Linux (SLES) | `15-SP7` |
| Windows | `10` / `11` |
| macOS | Ventura `13` / Sonoma `14` / Sequoia `15` |

### Libraries

The **ROCm Core SDK** (ROCm `7.13` or later) provides everything MIVisionX needs to build: the HIP and OpenCL runtimes, the `amdclang++` compiler, OpenMP, the `half` (float16) library, and `RPP` (`3.1.0` or later). Additional dependencies:

| Package | Minimum Version | Notes |
|---------|----------------|-------|
| CMake | `3.10` | install from your distribution's package manager |
| OpenCV | `3.X` / `4.X` | optional — only used by `RunVX` for image/video display |
| OpenSSL / libcrypto | any | optional, Linux only — enables MD5 checksums in `RunVX` `compare` commands; a built-in fallback is used when absent (Windows uses `wincrypt.h` automatically) |

> [!IMPORTANT]
> * Required compiler support: `C++17`, `OpenMP`, `Threads`
> * On `Ubuntu 22.04`: `sudo apt install libstdc++-12-dev`

> [!NOTE]
> `RPP` is included with the ROCm Core SDK. On ROCm releases where it is packaged separately, install it from the ROCm repository (for example `amdrpp7.13-*` / `rpp-dev`). Refer to the [ROCm install guide](https://rocm.docs.amd.com/en/7.13.0-preview/install/rocm.html) for exact package names.

## Installation instructions

> [!IMPORTANT]
> Choose your installation method based on your environment:
> 1. **ROCm `7.2.x` or below** — install the prebuilt packages (see [package install](#package-install)).
> 2. **ROCm `7.13` or later** — build from source on top of the ROCm Core SDK (see [source install](#source-install)).
> 3. **CPU-only** — build from source with `-DGPU_SUPPORT=OFF`; no ROCm or GPU required (see [source install](#source-install)).

### Linux

#### Install the ROCm Core SDK

Follow the [ROCm install guide](https://rocm.docs.amd.com/en/7.13.0-preview/install/rocm.html) for your GPU and operating system. Verify your hardware is on the [compatibility matrix](https://rocm.docs.amd.com/en/7.13.0-preview/compatibility/compatibility-matrix.html) before proceeding.

> [!IMPORTANT]
> Use **either** [package install](#package-install) **or** [source install](#source-install).

#### Package install

Available for **ROCm `7.2.x` and below**. Installs runtime libraries, development headers, and a ctest verification module.

| Distro | Command |
|--------|---------|
| Ubuntu | `sudo apt-get install mivisionx mivisionx-dev mivisionx-test` |
| RedHat | `sudo yum install mivisionx mivisionx-devel mivisionx-test` |
| SLES | `sudo zypper install mivisionx mivisionx-devel mivisionx-test` |

> [!IMPORTANT]
> Package install supports the `HIP` backend only. For the `OpenCL` backend or ROCm `7.13`+, use the source install.

#### Source install

```shell
sudo apt install cmake   # or equivalent for your distro

git clone https://github.com/ROCm/MIVisionX.git
cd MIVisionX
mkdir build-hip && cd build-hip
cmake ../                 # HIP backend (default); use -DBACKEND=OCL for OpenCL, -DGPU_SUPPORT=OFF for CPU
make -j$(nproc)
sudo make install
```

> [!NOTE]
> **CPU-only build** — no ROCm or GPU required:
> ```shell
> mkdir build-cpu && cd build-cpu
> cmake ../ -DGPU_SUPPORT=OFF
> make -j$(nproc)
> sudo make install
> ```

### Windows

* Windows SDK + a C++17 toolchain (Visual Studio 2019 or later)
* [CMake](https://cmake.org/download/) `3.10` or later
* Latest AMD [drivers](https://www.amd.com/en/support)
* [OpenCL SDK](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0) (for the OpenCL backend)

The default Windows backend is `OpenCL`. Build with CMake from the command line or open the generated solution in Visual Studio:

```shell
git clone https://github.com/ROCm/MIVisionX.git
cd MIVisionX

# OpenCL backend (default — requires the OpenCL SDK)
cmake -B build
cmake --build build --config Release

# CPU-only (no GPU or OpenCL SDK required)
cmake -B build -DGPU_SUPPORT=OFF
cmake --build build --config Release
```

### macOS

> [!IMPORTANT]
> macOS supports the CPU backend only.

See the [macOS build instructions](https://github.com/ROCm/MIVisionX/wiki/macOS#macos-build-instructions) wiki page.

## Verify installation

### Linux / macOS

The installer copies files to `/opt/rocm`:

| Path | Contents |
|------|----------|
| `/opt/rocm/bin` | Executables (`runvx`) |
| `/opt/rocm/lib` | Libraries (`libopenvx.so`, `libvxu.so`, `libvx_rpp.so`) |
| `/opt/rocm/include/mivisionx` | Header files |
| `/opt/rocm/share/mivisionx` | Samples |
| `/opt/rocm/share/doc/mivisionx` | Documentation |

**Canny Edge Detection sample:**

<p align="center"><img width="60%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/develop/samples/images/canny_image.PNG" /></p>

```shell
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
runvx /opt/rocm/share/mivisionx/samples/gdf/canny.gdf
```

> [!NOTE]
> * More samples: [samples/README.md](samples/README.md#samples)
> * macOS: use `export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/rocm/lib`

**ctest verification** (after installing the `mivisionx-test` package or building from source):

```shell
mkdir mivisionx-test && cd mivisionx-test
cmake /opt/rocm/share/mivisionx/test/
ctest -VV
```

### Windows

Use `RunVX` to verify the build output. From the repo root:

```shell
.\build\bin\Release\runvx.exe samples\gdf\skintonedetect.gdf
```

## Documentation

* [Published documentation](https://rocm.docs.amd.com/projects/MIVisionX/en/latest/)

Build locally:

```shell
cd docs
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

Doxygen API docs:

```shell
cd docs/doxygen && doxygen Doxyfile
```

## Technical support

File a bug on [GitHub issues](https://github.com/ROCm/MIVisionX/issues) or email `mivisionx.support@amd.com`.

## Release notes

Review all notable [changes](CHANGELOG.md#changelog) with the latest release.

### Tested configurations

| Component | Version |
|-----------|---------|
| ROCm | `7.13` or later (ROCm Core SDK) |
| RPP | `3.1.0` |
| OpenCV | `4.5.4` / `4.6` (optional, RunVX display only) |
