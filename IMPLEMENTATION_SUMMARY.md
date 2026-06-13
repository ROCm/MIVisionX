# OpenVX Row-Based Parallelism Implementation Summary

## Branch
`feature/openvx-row-based-parallelism` (from `develop`)

## What Was Implemented

### 1. Parallel Infrastructure (`ago_parallel.h`)
- **AgoParallelForRows()** - Row-based parallelization with guided scheduling
- **AgoShouldUseThreading()** - Auto-disable for small images (<32 rows)
- **OpenMP backend** with fallback to serial execution
- Configurable via CMake: `-DENABLE_OPENMP=ON/OFF`

### 2. CMake Integration
- OpenMP detection and configuration
- Automatic `-fopenmp` flag addition
- Compile definitions: `USE_OPENMP=1`

### 3. Parallel Kernel Implementations

| Kernel | Function | Status | Expected Speedup |
|--------|----------|--------|------------------|
| Add U8 | `HafCpu_Add_U8_U8U8_Wrap_OpenMP()` | ✅ Implemented | 2.5-3.0x |
| Subtract U8 | `HafCpu_Sub_U8_U8U8_Wrap_OpenMP()` | ✅ Implemented | 2.5-3.0x |
| Box3x3 | `HafCpu_Box_U8_U8_3x3_OpenMP()` | ✅ Implemented | 1.4-1.7x |

## Build Instructions

```bash
cd MIVisionX/amd_openvx
mkdir build && cd build
cmake .. -DENABLE_OPENMP=ON
make -j$(nproc) openvx
```

## How It Works

### Row-Based Parallelism Pattern
```cpp
// Serial version
for (int y = 0; y < height; y++) {
    process_row(y);
}

// Parallel version (guided scheduling)
#pragma omp parallel for schedule(guided)
for (int y = 0; y < height; y++) {
    process_row(y);
}
```

### Key Features
1. **Cache-friendly**: Threads process contiguous rows
2. **No false sharing**: Each thread writes to different rows
3. **Auto-threshold**: Disabled for images < 32 rows
4. **AVX preserved**: SIMD optimizations maintained

## Performance Expectations

Based on OpenCV benchmarks on AMD Ryzen:

| Configuration | Add Kernel | Box3x3 |
|---------------|------------|--------|
| Single-threaded | ~30K MP/s | ~2.1K MP/s |
| 4 threads (expected) | ~75-90K MP/s | ~3.0-3.5K MP/s |
| **Speedup** | **2.5-3.0x** | **1.4-1.7x** |

## Next Steps

### To Complete Implementation:

1. **Connect to OpenVX node layer**
   - Modify kernel registration to use parallel versions
   - Add function prototypes to `ago_internal.h`

2. **Port remaining P0 kernels**
   - Gaussian3x3 (filter)
   - ColorConvert (color)
   - Erode/Dilate (filter)

3. **Performance validation**
   - Build with `-DENABLE_OPENMP=ON`
   - Run `openvx-mark` benchmark
   - Compare with OpenCV results

4. **Fine-tuning**
   - Adjust `AGO_PARALLEL_MIN_HEIGHT` threshold
   - Tune `AGO_PARALLEL_ROWS_PER_TASK` for optimal chunking
   - Profile with `OMP_SCHEDULE=guided,4` vs other settings

## Files Modified/Created

```
MIVisionX/
├── amd_openvx/openvx/
│   ├── ago/
│   │   ├── ago_parallel.h                          [NEW]
│   │   └── ago_haf_cpu_arithmetic_parallel.cpp      [NEW]
│   └── CMakeLists.txt                               [MODIFIED]
```

## Testing

### Manual Test
```bash
# Build
mkdir build && cd build
cmake .. -DENABLE_OPENMP=ON
make openvx

# Run with specific thread count
OMP_NUM_THREADS=4 ./your_test_app

# Verify threading is active
OMP_DISPLAY_ENV=VERBOSE ./your_test_app
```

### Benchmark Comparison
```bash
# Run OpenVX benchmark
./openvx-mark --kernel Add --threads 1
./openvx-mark --kernel Add --threads 4

# Compare with OpenCV
./opencv-mark (from MIVisionX/tests/opencv_benchmark)
```

## Technical Notes

### Why Guided Scheduling?
- **OpenCV uses it** - proven optimal for image processing
- **Adaptive chunk sizes** - starts large, decreases as work completes
- **Load balancing** - handles variable row processing times

### Why Row-Based?
- **Memory access**: Rows are contiguous (cache-friendly)
- **No conflicts**: Each row is independent
- **SIMD compatibility**: Existing AVX code preserved
- **Simple**: Easy to understand and maintain

### Fallback Strategy
```cpp
if (!AgoShouldUseThreading(height, width)) {
    // Use original serial implementation
    return HafCpu_Add_U8_U8U8_Wrap(...);
}
// Use parallel version
```

---

*Implementation date: Sat Jun 13 2026*
*Based on OpenCV benchmark analysis showing 1.71x speedup potential*
