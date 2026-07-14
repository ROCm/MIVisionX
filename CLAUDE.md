# MIVisionX

AMD MIVisionX toolkit - version 4.0.0. A streamlined computer vision toolkit built on AMD OpenVX. As of 4.0.0, the project is reduced to its core: the AMD OpenVX engine, the AMD RPP OpenVX extension, and the RunVX graph executor, across `CPU`, `HIP`, and `OpenCL` backends.

## Project Structure

```
MIVisionX/
├── amd_openvx/                    # Core AMD OpenVX implementation (v1.3.2)
│   └── openvx/
│       ├── ago/                   # AMD Graph Optimizer (AGO) engine - CPU & GPU kernels
│       ├── api/                   # OpenVX API (vx_api.cpp, vx_nodes.cpp, vxu.cpp)
│       ├── hipvx/                 # HIP GPU backend kernels
│       └── include/               # Khronos OpenVX 1.3.2 standard headers (VX/) + AMD extensions
├── amd_openvx_extensions/         # OpenVX extension modules
│   └── amd_rpp/                   # ROCm Performance Primitives extension (CPU/HIP; CPU-only with OpenCL core)
├── utilities/                     # CLI tools
│   └── runvx/                     # OpenVX graph executor
├── tests/                         # Test suites
│   ├── openvx_conformance_tests/  # Khronos OpenVX CTS runner (runConformanceTests.py)
│   ├── openvx_api_tests/          # Individual API test programs
│   ├── amd_openvx_gdfs/           # GDF (Graph Description Format) test scripts
│   ├── vision_tests/              # Python-driven vision node tests
│   └── vx_rpp_tests/              # AMD RPP extension GDF tests
├── apps/                          # Sample CV apps (bubble_pop, optical_flow) - OpenVX + OpenCV, built separately
├── samples/                       # Sample GDF graphs and c_samples (canny)
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
| `BACKEND` | `HIP`, `OPENCL`/`OCL`, `CPU`/`host` | `HIP` | GPU backend (Windows defaults to `OpenCL`, macOS forces `CPU`) |
| `GPU_SUPPORT` | `ON`/`OFF` | `ON` | Enable GPU support (macOS forces `OFF`) |
| `CODE_COVERAGE` | `ON`/`OFF` | `OFF` | LLVM source-based coverage instrumentation |

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

- Libraries go to `build/lib/` (`libopenvx.so`, `libvxu.so`, `libvx_rpp.so`)
- Binaries go to `build/bin/` (`runvx`)

## OpenVX Conformance Tests

The Khronos OpenVX CTS is cloned from `https://github.com/KhronosGroup/OpenVX-cts.git` (branch `openvx_1.3.2`).

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

MIVisionX 4.0+ is built on the **ROCm Core SDK** and requires **ROCm 7.13 or later** (install via the [ROCm install guide](https://rocm.docs.amd.com/en/latest/install/rocm.html), e.g. `amdrocm-core-sdk7.13-gfx<arch>`). The Core SDK provides HIP, OpenCL, `amdclang++`, OpenMP, `half`, and `RPP`.

**Core (CPU-only):** CMake >= 3.10, C++17 compiler, SSE4.2 support
**HIP backend:** ROCm (hip::host, hip::device) — from the ROCm Core SDK
**OpenCL backend:** OpenCL libraries/headers — from the ROCm Core SDK
**amd_rpp extension:** RPP >= 3.1.0 (CPU/HIP; CPU-only with an OpenCL core) — from the ROCm Core SDK
**Optional:** OpenCV 3.x/4.x (RunVX display only)
**Linux link libs:** `dl`, `m` (core); `pthread`, `dl`, `m`, `rt` (conformance tests)
