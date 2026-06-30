# MIVisionX - OpenVX 1.3 Conformance Tests

Runs the [Khronos OpenVX CTS](https://github.com/KhronosGroup/OpenVX-cts) (branch `openvx_1.3`) against the MIVisionX implementation across the `HOST`, `HIP`, and `OCL` backends.

## Usage

```shell
python tests/openvx_conformance_tests/runConformanceTests.py --help
```

```
usage: runConformanceTests.py
         [--directory CTS_Build_Directory]
         [--backend_type MIVisionX_Backend]
         [--jobs N]
         [--skip-mivisionx-build]
         [--skip-cts-build]
         [--skip-cts-run]
         [--cts-repo URL]
         [--cts-branch BRANCH]

Arguments:
  --directory       Build directory for CTS (default: ~/)
  --backend_type    Backend to test: ALL / HOST / HIP / OCL  (default: ALL)
  --jobs            Parallel make jobs (default: system core count)
  --skip-mivisionx-build  Skip rebuilding MIVisionX
  --skip-cts-build        Skip rebuilding the CTS
  --skip-cts-run          Skip running the CTS (build only)
  --cts-repo        CTS git repository URL (default: https://github.com/KhronosGroup/OpenVX-cts.git)
  --cts-branch      CTS git branch (default: openvx_1.3)
```

## Quick start

Run all backends from the repo root (builds MIVisionX, clones & builds the CTS, then runs all suites):

```shell
python tests/openvx_conformance_tests/runConformanceTests.py --backend_type HOST
```

## Runtime requirements

- `LD_LIBRARY_PATH` must include the directory containing `libopenvx.so`
- `VX_TEST_DATA_PATH` must point to `OpenVX-cts/test_data/`
- For GPU backends, `AGO_DEFAULT_TARGET` can be set to `CPU` or `GPU`

## Test suites

The conformance workflow runs the following suites in parallel: `baseline` (required), `graph`, `data-objects`, `image-ops`, `vision-color`, `vision-filters`, `vision-arithmetic`, `vision-geometric`, `vision-features`, `vision-statistics`, `vision-pyramid`.
