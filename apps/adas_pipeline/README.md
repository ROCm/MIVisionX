# ADAS surround pipeline

A multi-camera surround-view lane pipeline built as the three stage chain the
[OpenVX graph pipelining extension](https://registry.khronos.org/OpenVX/extensions/vx_khr_pipelining/1.1/html/vx_khr_pipelining_1_1_0.html#sec_pipelining)
uses as its example, and runnable several ways so the arrangements can be
compared on identical work.

```
camera luma x N ->  stage 0  -> surround ->  stage 1  -> edges ->  stage 2  -> lane mask
                   undistort                 canny                oriented filter bank
                   denoise                                        morphological cleanup
                   project to ground
                   blend
```

Stage 0 and stage 2 are many pixel-parallel operations per frame, which is what
keeps a GPU busy for a useful length of time. Stage 1 is the detector, which on
a desktop part runs faster on the CPU than on the GPU. The default placement is
therefore GPU, CPU, GPU, and every hand-off between stages crosses a device.

Run one frame at a time and only one compute unit works while the other waits,
so a frame costs the sum of the stages. Pipeline the stages and each one works
on a different frame at the same time, so a frame costs only as much as the
slowest stage.

## Building

The application builds against an installed MIVisionX and needs OpenCV.

```bash
mkdir build && cd build
cmake -DOPENVX_INCLUDES=<mivisionx>/amd_openvx/openvx/include \
      -DOPENVX_LIBRARIES=<install>/lib/libopenvx.so ..
make -j
```

## Seeing what pipelining is worth

`--compare` runs the same frames through every arrangement and prints one
table. Measured on a gfx942 with a 32 core host, 300 frames per mode:

```
$ adasPipeline --video drive.mp4 --compare --size 1920x1080 --cameras 6 --filters 8

  mode    fps      ms/frame  vs cpu    what runs at the same time
  cpu     15.3     65.40     1.00    x nothing, one stage at a time
  gpu     223.5    4.47      14.62   x nothing, one stage at a time
  split   189.0    5.29      12.36   x nothing, the devices take turns
  queued  337.8    2.96      22.09   x capture against the graph
  staged  348.8    2.87      22.81   x capture and all three stages, on different frames

  staged stage times: surround (gpu) 1.35 ms, detect (cpu) 1.60 ms, refine (gpu) 1.25 ms
```

Reading the ladder from the bottom up is the point of the application:

- `split` is *slower* than `gpu` even though it uses both devices. Splitting
  work across devices without pipelining only adds hand-off cost, because the
  devices take turns rather than working at the same time. This is the trap the
  extension exists to solve.
- `queued` gives the graph several frames through queued parameters, so the
  host captures and converts the next frame while the graph works on earlier
  ones. That alone is worth about 1.8x over `split`.
- `staged` gives each stage its own graph, so all three run at once on
  different frames. The stage times show why it helps: they sum to 4.20 ms but
  the slowest is 1.60 ms, so the pipeline can approach 2.6x the unpipelined
  rate before host costs are counted.

Balance matters more than raw device speed. `--cameras` sets how much work
stage 0 carries and `--filters` how much stage 2 carries, so the two devices
can be brought into the same range; a pipeline is only as fast as its slowest
stage, so an unbalanced split wastes whatever headroom the other stage has.
At 6 cameras and 8 orientations the three stages here are within 30% of each
other, which is roughly where the arrangement pays off best.

`--depth` sets how many frames may be in flight. Two is enough to get most of
the benefit and more than three does not help:

```
depth 1 : 203.7 fps      depth 3 : 356.3 fps
depth 2 : 372.1 fps      depth 6 : 353.6 fps
```

## Modes

| `--mode`  | what it does |
|-----------|--------------|
| `cpu`     | one graph, every stage on the CPU, one frame at a time |
| `gpu`     | one graph, every stage on the GPU, one frame at a time |
| `split`   | one graph, stages split across the devices, still one frame at a time |
| `queued`  | the split graph with queued parameters, so capture overlaps the graph |
| `staged`  | one graph per stage, all three overlapped across frames |
| `batch`   | several frames handed over in one enqueue call |
| `stream`  | `vxStartGraphStreaming`, the framework re-runs the graph itself |

Useful options: `--place <d,d,d>` to move stages between devices,
`--schedule manual` to drive the queues with `vxScheduleGraph` instead of
letting the framework do it, `--no-display` for timing runs, `--dump <dir>` to
write each lane mask as a PNG, and `--live <id>` for a camera instead of a file.
`--help` lists the rest.

Streaming mode reports more graph completions than frames supplied. That is
expected: the framework re-runs the graph as fast as the stages allow, whether
or not the host has managed to supply a new frame, which is what an always-on
sensor loop wants.

## Checking that pipelining did not change the answer

Dump the same frames from an unpipelined mode and a pipelined one and compare
them, ignoring a six pixel frame around the edge:

```bash
for m in split staged; do
  ./adasPipeline --video drive.mp4 --mode $m --frames 20 --no-display --dump /tmp/$m
done
python3 -c "
import glob, numpy as np
from PIL import Image
a = [np.array(Image.open(f)) for f in sorted(glob.glob('/tmp/split/*.png'))]
b = [np.array(Image.open(f)) for f in sorted(glob.glob('/tmp/staged/*.png'))]
print(sum(np.array_equal(x[6:-6,6:-6], y[6:-6,6:-6]) for x, y in zip(a, b)), 'of', len(a))"
```

Every frame matches, on either backend and at any queue depth. The six pixels
have to be excluded because the edge detector's border pixels are undefined by
specification and this implementation leaves them unwritten, so they hold
whatever the buffer they landed in held before. The filter bank is 9x9 and sits
between a dilate and an erode, which carries that undefined content six pixels
in. It is not a pipelining artefact: running the *same* mode twice differs in
the same six pixels and nowhere else, because a freshly allocated device buffer
starts out with different contents each run.

## Two things worth knowing before writing one of these

**A queued graph parameter rebinds exactly one node parameter.** A graph
parameter stands for one parameter of one node, so when a queue swaps a buffer
in, only that node sees the new buffer. If several nodes read the same queued
image, every node but the one owning the parameter keeps reading whichever
buffer was bound when the graph was built, and the results are quietly wrong
for every frame after the first. Stage 2 here has one node per filter
orientation, so it feeds the whole bank from a single node that owns the queued
input. Adding one graph parameter per consumer and enqueuing the same reference
to each would work too, at the cost of a queue each.

**Not every node has a GPU implementation, and there is no signal when one
does not.** Node affinity reports back whatever was requested even after
verification, so the only way to tell that a node fell back to the CPU is that
its time does not change. On this hardware `remap`, `gaussian`, `convolve`,
`warp_perspective`, `dilate`, `erode`, `scale`, `integral`, `fast_corners` and
`optical_flow_lk` are genuinely accelerated, while `median`, `harris_corners`
and `equalize_histogram` take the same time on either device. Colour
conversions *from* RGB to luma are not implemented for HIP either, and the
optimizer fuses `color_convert` and `channel_extract` into one such kernel, so
a pipeline fed RGB will always do that part on the CPU. This one takes luma
directly, which is also what automotive cameras deliver.
