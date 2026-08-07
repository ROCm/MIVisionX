# C/C++ Samples for OpenVX

MIVisionX C/C++ sample applications demonstrating the OpenVX API.

## Canny Edge Detector

Runs Canny edge detection on an image file or live camera using an OpenVX graph.

```shell
cd c_samples/canny/
cmake .
make

# On an image file
./cannyDetect --image <path/to/image>

# On a live camera
./cannyDetect --live
```

## Graph Pipelining

Demonstrates the OpenVX `vx_khr_pipelining` extension on a mixed CPU+GPU
workload. The same graph is executed synchronously (`--pipeline 0`) and with
`QUEUE_AUTO` pipelining (`--pipeline 1`) so the speed-up is visible.

```shell
cd c_samples/pipelining/
cmake .
make

# Synchronous (baseline) — 4K heavy default workload
./pipelining --pipeline 0

# Pipelined — 4K heavy default workload
./pipelining --pipeline 1

# Compare both modes in one table
./pipelining --compare

# Tune pipeline depth
./pipelining --pipeline 1 --pipeline-depth 2

# Lower resolution for faster runs
./pipelining --compare --mode light --resolution fhd --frames 60
```

### What the sample does

The graph processes a synthetic RGB video stream frame by frame. Two presets
are provided:

* **Heavy (default)** — sized so the GPU has enough work to outperform the CPU:
  ```
  RGB input -> ColorConvert -> IYUV -> ChannelExtract(Y) -> Gaussian3x3 -> U8 output
  ```
* **Light** — one filter pass for fast runs and bit-exact verification:
  ```
  RGB input -> ColorConvert -> IYUV -> ChannelExtract(Y) -> Box3x3 -> U8 output
  ```

`ColorConvert` and `ChannelExtract` are lightweight color-space operations,
while `Box3x3` is a heavier 2D convolution. MIVisionX automatically schedules
nodes on the best available target (CPU or GPU), so this graph naturally mixes
host-side work with device-side work.

Resolution presets are selectable with `--resolution hd|fhd|qhd|4k`:

* `hd`  = 1280×720
* `fhd` = 1920×1080
* `qhd` = 2560×1440
* `4k`  = 3840×2160 (default)

4K is the default because the sample is designed to give a discrete GPU
enough work to outperform the CPU backend. Smaller resolutions are useful
for quick correctness checks or slower machines.

### How pipelining works

The synchronous path (`--pipeline 0`) follows the classic OpenVX pattern:

1. Copy one frame into the input image.
2. Call `vxProcessGraph(graph)` and wait for it to finish.
3. Read the output image.
4. Repeat.

The host is idle while the GPU runs, and the GPU is idle while the host copies
the next frame. That serialization limits throughput.

The pipelined path (`--pipeline 1`) uses the `vx_khr_pipelining` extension to
keep multiple frames in flight at the same time:

1. Create a **ring of input/output buffer pairs** (default depth = 4).
2. Expose the RGB input and the U8 output as **graph parameters**.
3. Configure the graph for `QUEUE_AUTO` mode with `vxSetGraphScheduleConfig`.
4. Prime the pipeline by enqueuing all input and output buffers with
   `vxGraphParameterEnqueueReadyRef`.
5. The executor schedules graph instances automatically as soon as a full
   input/output set is available.
6. The host dequeues a finished output with `vxGraphParameterDequeueDoneRef`,
   records the result, refills the matching input slot, and enqueues both slots
   again.

Because the host can prepare the next frame while the previous frame is still
running on the GPU, the CPU and GPU work in parallel and overall throughput
increases.

### Where the performance comes from

* **Overlap host ↔ device transfers with compute.** While frame N is processed,
  the host fills frame N+1 and the device writes frame N−1.
* **Keep the GPU command queue full.** Multiple enqueued frames prevent the
  device from waiting between graph executions.
* **Hide CPU preprocessing.** Color conversion and channel extraction run on
  the CPU and can overlap with the previous frame's GPU convolution.

The sample prints a per-mode aggregate checksum and reports fps. Both presets
are designed to produce identical checksums for the synchronous and pipelined
paths, giving a quick correctness check.

### Why the heavy preset matters

On very small inputs, a GPU backend can appear slower than the CPU backend
because the per-frame launch and data-transfer overhead dominates the actual
compute time. The heavy preset raises the default resolution to 3840×2160
and uses a Gaussian 3x3 filter so the GPU has enough work to amortize that
overhead. On a discrete GPU this typically makes the HIP backend faster than
the CPU-only backend, and the pipelined path faster than the synchronous path.

### Tuning tips

* **Resolution** — Use `--resolution hd|fhd|qhd|4k`. 4K is the default for
  throughput demonstration; `fhd` or `hd` are good for quick correctness checks.
* **Pipeline depth** — Use `--pipeline-depth D`. The default is 4. Smaller
  values reduce latency; larger values can raise throughput when the device
  needs more in-flight work.
* **Mode** — Use `--mode light` for bit-exact correctness verification and
  `--mode heavy` (default) for throughput measurements.
* **Frame count** — Use `--frames N` to control run length.
* **Backend** — The sample runs on CPU-only builds (`BACKEND=CPU`) as well as
  on HIP/OpenCL GPU builds. On the CPU backend the gain comes from overlapping
  host-side graph scheduling with data preparation.

### Caveats when using `vx_khr_pipelining`

A few non-obvious behaviors caught while building these samples, borrowed from
lessons learned in the larger ADAS pipeline app (PR #1730):

* **One graph parameter rebinding.** When you enqueue a new reference for a
  graph parameter, MIVisionX rebinds exactly *one* node parameter. If a
  queued image is read by several nodes, only one of those nodes sees the new
  reference on subsequent frames; the others silently keep using the old one.
  In both samples above, the RGB input is consumed only by `ColorConvert`
  and the U8 output is produced only by the final filter node, so the
  rebinding is unambiguous.

* **`VX_NODE_PERFORMANCE` returns graph time.** Asking a node for its own
  `VX_NODE_PERFORMANCE` currently returns the same figure for every node in
  the graph (the whole graph's execution time). That makes it impossible to
  break down where time is going *inside* a single graph. To compare stage
  costs, split the work into separate graphs per stage, or profile the overall
  arrangement instead.

* **Small inputs hide GPU speed-up.** On tiny frames, the per-frame GPU launch
  and data-transfer cost can dominate the compute, making the CPU backend look
  faster than HIP. Both samples default to 3840×2160 and the hybrid sample adds
  extra filter passes to give the GPU enough work to amortize that overhead.

## Hybrid CPU+GPU Graph Pipelining

A second sample that explicitly pins parts of the graph to the CPU and parts
of the graph to the GPU, then pipelines the result. This is useful when you
want to force the heavy compute onto the GPU while keeping lightweight
host-side work on the CPU.

```shell
cd c_samples/pipelining_hybrid/
cmake .
make

# Synchronous hybrid CPU+GPU graph
./pipelining_hybrid --pipeline 0

# Pipelined hybrid CPU+GPU graph
./pipelining_hybrid --pipeline 1
```

### What the hybrid sample does

The graph is deliberately split between targets:

* **CPU nodes:** `ColorConvert`, `ChannelExtract(Y)`
* **GPU nodes:** `Box3x3 -> Box3x3 -> Box3x3`
* **CPU nodes:** `Threshold`, `Box3x3`

The targets are pinned with `vxSetNodeTarget(..., VX_TARGET_STRING, "CPU")` and
`vxSetNodeTarget(..., VX_TARGET_STRING, "GPU")`. On a HIP backend, `"GPU"` maps
to HIP execution, so the sample demonstrates a true CPU+HIP+CPU pipeline.

Without pipelining the three stages run one after another for each frame.
With `QUEUE_AUTO`, the executor can schedule the CPU work for frame N+1 while
the GPU is still finishing frame N, exposing the cross-target parallelism.

This sample runs at 4K by default. Use `--resolution hd|fhd|qhd|4k` to pick a
smaller frame size, `--pipeline-depth` to tune the number of in-flight frames,
and `--compare` to run both modes back-to-back.
