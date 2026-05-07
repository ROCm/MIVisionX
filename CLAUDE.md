# MIVisionX

AMD MIVisionX toolkit - version 3.5.0. Computer vision and machine intelligence libraries, utilities, and applications built on AMD OpenVX.

## Project Structure

```
MIVisionX/
├── amd_openvx/                    # Core AMD OpenVX implementation (v1.3.0)
│   └── openvx/
│       ├── ago/                   # AMD Graph Optimizer (AGO) engine - CPU & GPU kernels
│       ├── api/                   # OpenVX API (vx_api.cpp, vx_nodes.cpp, vxu.cpp)
│       ├── hipvx/                 # HIP GPU backend kernels
│       └── include/               # Khronos OpenVX 1.3 standard headers (VX/) + AMD extensions
├── amd_openvx_extensions/         # OpenVX extension modules
│   ├── amd_nn/                    # Neural network extension (MIOpen-based)
│   ├── amd_opencv/                # OpenCV interop extension
│   ├── amd_media/                 # FFmpeg media extension
│   ├── amd_rpp/                   # ROCm Performance Primitives extension
│   ├── amd_migraphx/              # MIGraphX inference extension
│   ├── amd_loomsl/                # Loom stitch library (OpenCL only)
│   ├── amd_custom/                # Custom kernel extension (HIP only)
│   └── amd_winml/                 # Windows ML extension
├── utilities/                     # CLI tools
│   ├── runvx/                     # OpenVX graph executor
│   ├── runcl/                     # OpenCL kernel runner
│   ├── mv_deploy/                 # Model deployment tool
│   ├── loom_shell/                # Loom stitch shell
│   └── loom_io_media/             # Loom I/O media
├── tests/                         # Test suites
│   ├── openvx_conformance_tests/  # Khronos OpenVX CTS runner (runConformanceTests.py)
│   ├── openvx_api_tests/          # Individual API test programs
│   ├── amd_openvx_gdfs/           # GDF (Graph Description Format) test scripts
│   ├── vision_tests/              # Python-driven vision node tests
│   ├── neural_network_tests/      # NN model import tests (caffe, onnx, nnef)
│   └── ...                        # Extension-specific tests
├── apps/                          # Sample applications
├── model_compiler/                # Model compiler
├── toolkit/                       # Additional toolkit utilities
├── docker/                        # Dockerfiles (build, conformance, release)
├── cmake/                         # CMake find modules
├── docs/                          # Documentation (Sphinx-based)
├── .github/workflows/             # GitHub Actions workflows
│   ├── conformance.yml            # OpenVX conformance tests (CPU backend)
│   └── codeql-analysis.yml        # CodeQL static analysis
└── CMakeLists.txt                 # Top-level CMake (project: mivisionx)
```

## Build System

CMake >= 3.10, C++17 required. Default compiler: `amdclang++` (from ROCm), falls back to system compiler.

### Key CMake Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `BACKEND` | `HIP`, `OPENCL`/`OCL`, `CPU`/`host` | `HIP` | GPU backend (macOS forces `CPU`) |
| `GPU_SUPPORT` | `ON`/`OFF` | `ON` | Enable GPU support (macOS forces `OFF`) |
| `NEURAL_NET` | `ON`/`OFF` | `ON` | Neural net extension |
| `LOOM` | `ON`/`OFF` | `ON` | Loom stitch library |
| `MIGRAPHX` | `ON`/`OFF` | `OFF` | MIGraphX support |

### Build Commands

**CPU-only (HOST backend):**
```bash
mkdir build && cd build
cmake .. -DGPU_SUPPORT=OFF
make -j$(nproc)
```

**HIP backend:**
```bash
mkdir build && cd build
cmake .. -DBACKEND=HIP
make -j$(nproc)
```

**OpenCL backend:**
```bash
mkdir build && cd build
cmake .. -DBACKEND=OCL
make -j$(nproc)
```

Default install prefix: `${ROCM_PATH}` (defaults to `/opt/rocm`).

### Build Output

- Libraries go to `build/lib/` (`libopenvx.so`, `libvxu.so`, extensions)
- Binaries go to `build/bin/` (`runvx`, `runcl`, etc.)

## OpenVX Conformance Tests

The Khronos OpenVX CTS is cloned from `https://github.com/KhronosGroup/OpenVX-cts.git` (branch `openvx_1.3`).

### Running Conformance Locally

Use the provided Python script:
```bash
python tests/openvx_conformance_tests/runConformanceTests.py --backend_type HOST
```

Options: `--backend_type` (`ALL`/`HOST`/`HIP`/`OCL`), `--jobs N`, `--skip-mivisionx-build`, `--skip-cts-build`, `--skip-cts-run`, `--directory`, `--cts-repo`, `--cts-branch`.

### CTS Build Configuration

The CTS needs these CMake variables pointing to the pre-built MIVisionX:
- `OPENVX_INCLUDES` → `amd_openvx/openvx/include`
- `OPENVX_LIBRARIES` → `libopenvx.so;libvxu.so;pthread;dl;m;rt`
- `OPENVX_CONFORMANCE_VISION=ON`
- `CMAKE_POLICY_VERSION_MINIMUM=3.5`

### Runtime Environment

- `LD_LIBRARY_PATH` must include the directory containing `libopenvx.so`
- `VX_TEST_DATA_PATH` must point to `OpenVX-cts/test_data/`
- `AGO_DEFAULT_TARGET` can be set to `CPU` or `GPU` (for OCL/HIP backends)

## Git Workflow

- Primary development branch: `develop`
- CI triggers on: `master`, `main`, `develop`
- Conformance workflow runs parallel test suites: baseline (required), graph, data-objects, image-ops, vision-color, vision-filters, vision-arithmetic, vision-geometric, vision-features, vision-statistics, vision-pyramid

## Dependencies

**Core (CPU-only):** CMake >= 3.10, C++17 compiler, SSE4.2 support
**HIP backend:** ROCm (hip::host, hip::device)
**OpenCL backend:** OpenCL libraries/headers
**Extensions:** MIOpen, OpenCV 3.x/4.x, FFmpeg, RPP >= 3.1.0 (each optional)
**Linux link libs:** `dl`, `m` (core); `pthread`, `dl`, `m`, `rt` (conformance tests)
