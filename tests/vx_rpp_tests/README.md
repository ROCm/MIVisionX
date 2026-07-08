# VX RPP Tests

Tests for the `vx_rpp` OpenVX extension, which wraps [ROCm Performance Primitives](https://github.com/ROCm/rpp) image and tensor augmentation functions as OpenVX kernels.

## Prerequisites

MIVisionX must be installed (or built from source) with the `amd_rpp` extension enabled. The `CPU` and `HIP` backends are supported; the `OpenCL` backend runs `vx_rpp` in CPU-only mode.

## Running the GDF test

```shell
# From the MIVisionX repo root, with runvx on PATH
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

runvx -dump-profile file tests/vx_rpp_tests/gdf/test_vx_rpp.gdf
```

The `-dump-profile` flag prints per-node execution timing after the graph completes, which is useful for benchmarking augmentation performance on CPU vs HIP.
