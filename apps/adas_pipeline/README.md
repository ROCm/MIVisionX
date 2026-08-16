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
keeps a GPU busy for a useful length of time. Stage 1 is the detector, a single
node. The default placement is GPU, CPU, GPU, so every hand-off between stages
crosses a device; `--place` moves any stage to either device.

Run one frame at a time and only one compute unit works while the other waits,
so a frame costs the sum of the stages. Pipeline the stages and each one works
on a different frame at the same time.

## Building

The application builds against an installed MIVisionX and needs OpenCV.

```bash
mkdir build && cd build
cmake -DOPENVX_INCLUDES=<mivisionx>/amd_openvx/openvx/include \
      -DOPENVX_LIBRARIES=<install>/lib/libopenvx.so ..
make -j
```

## Trying it with real driving data

The application reads any video file or a numbered image sequence, so any
dashcam clip works as input. The public driving datasets below are a good
source; each is downloaded by the user, so the non-commercial ones are fine to
try even though this sample does not redistribute them. Check every dataset's
own licence and attribution terms before using it beyond a local trial.

| Dataset | Licence | Cameras | Why |
|---------|---------|---------|-----|
| [comma10k](https://github.com/commaai/comma10k) | MIT | single road camera | Real comma.ai driving frames, permissive licence, no account. Smallest way to feed the pipeline real road scenes. |
| [comma2k19](https://github.com/commaai/comma2k19) | MIT | single road camera | Real driving video with real camera intrinsics and vehicle pose, so the undistort and ground-plane stages can be made genuine rather than synthetic. |
| [nuScenes mini](https://www.nuscenes.org/nuscenes) | CC BY-NC-SA 4.0 (non-commercial, account required) | 6-camera 360 surround | The only small dataset with a real synchronised multi-camera rig plus full calibration, for a true surround input. |

The `--video` argument accepts a `printf` pattern for an image sequence. To pull
a handful of real comma10k frames into a numbered sequence and run the
comparison on them:

```bash
mkdir -p /tmp/comma10k && cd /tmp/comma10k
i=0
for url in $(curl -s https://api.github.com/repos/commaai/comma10k/contents/imgs \
            | grep -o 'https://raw.githubusercontent.com/[^"]*\.png' | head -n 8); do
  printf -v out "img_%04d.png" $i; curl -sL "$url" -o "$out"; i=$((i+1))
done
cd -

# comma10k frames are 1164x874; the app resizes to --size
./adasPipeline --video "/tmp/comma10k/img_%04d.png" --compare \
    --frames 8 --size 1164x874 --cameras 4 --filters 6
```

There is one real video here, so the app feeds every camera the same frame:
the workload per cycle is what the measurement is about, not a true surround
stitch. A dataset with a real multi-camera rig (nuScenes above) would give each
camera a distinct view.

## Seeing what pipelining is worth

`--compare` runs the same frames through every arrangement and prints one
table. Everything below was measured on a Radeon RX 7900 XT (gfx1100) with a
Ryzen 9 7900X, ROCm 7.13, HIP backend, at 1920x1080 with 6 cameras and 8 filter
orientations, 300 frames per mode, queue depth 3.

Decoding each frame as it goes, which is what a run against a video file does:

```
  mode    fps      ms/frame  capture   vs cpu    what runs at the same time
  cpu     14.6     68.50     1.89      1.00    x nothing, one stage at a time
  gpu     203.8    4.91      1.75      13.96   x nothing, one stage at a time
  split   183.2    5.46      1.51      12.55   x nothing, the devices take turns
  queued  300.7    3.33      1.88      20.60   x capture against the graph
  staged  351.6    2.84      1.77      24.08   x capture and all three stages
```

The `capture` column is the time spent inside the frame source per frame, and
at this resolution it is comparable to the time the graph takes. A run like
this measures the video decoder as much as it measures the pipeline, which is
what `--preload` is for: it reads the frames into memory up front, so what is
left is the pipeline.

```
  mode    fps      ms/frame  capture   vs cpu    what runs at the same time
  cpu     15.6     63.92     0.00      1.00    x nothing, one stage at a time
  gpu     473.1    2.11      0.00      30.24   x nothing, one stage at a time
  split   305.7    3.27      0.00      19.54   x nothing, the devices take turns
  queued  361.6    2.77      0.00      23.11   x capture against the graph
  staged  596.3    1.68      0.00      38.11   x capture and all three stages

  staged stage graphs: surround (gpu) 1.22 ms, detect (cpu) 1.31 ms, refine (gpu) 1.15 ms
  they sum to 3.67 ms of graph time per frame and the pipeline delivered a
  frame every 1.68 ms, so 2.18x of that work was in flight at once
```

Reading the second table is the point of the application:

- `split` is *slower* than `gpu` even though it uses both devices, 3.27 ms
  against 2.11 ms. Splitting work across devices without pipelining only adds
  the hand-off, because the devices take turns rather than working at once.
  This is the trap the extension exists to solve, and it survives taking the
  decoder out of the picture.
- `queued` gives the same graph several frames through queued parameters, which
  buys back most of that loss (2.77 ms) even with nothing to decode, because
  the host still has six camera images to write and one to read per frame and
  can now do it while the graph runs.
- `staged` gives each stage its own graph, so the three overlap on different
  frames: 1.68 ms per frame, faster than putting everything on the GPU. The
  three stage graphs are within 15% of each other, which is what makes the
  arrangement worth it - a pipeline is only as fast as its slowest stage.

Note that the three stages sum to 3.67 ms while a frame comes out every
1.68 ms, and that this is *less* than the 2.37 ms the two GPU stages sum to.
Work from the two stage graphs sharing the GPU overlaps on the device as well,
so a floor worked out per device would be wrong.

`--cameras` sets how much work stage 0 carries and `--filters` how much stage 2
carries, so the two devices can be brought into the same range.

## Modes

| `--mode`  | what it does |
|-----------|--------------|
| `cpu`     | one graph, every stage on the CPU, one frame at a time |
| `gpu`     | one graph, every stage on the GPU, one frame at a time |
| `split`   | one graph, stages split across the devices, still one frame at a time |
| `queued`  | the split graph with queued parameters, so the host overlaps the graph |
| `staged`  | one graph per stage, all three overlapped across frames |
| `batch`   | several frames handed over in one enqueue call |
| `stream`  | `vxStartGraphStreaming`, the framework re-runs the graph itself |

`--compare` runs every mode in the table above, `batch` and `stream` included,
and `--verify` checks that the pipelined modes match the unpipelined reference.

Useful options: `--preload` to take the video decoder out of the measurement,
`--place <d,d,d>` to move stages between devices, `--schedule manual` to drive
the queues with `vxScheduleGraph` instead of letting the framework do it,
`--no-display` for timing runs, `--dump <dir>` to write each lane mask as a
PNG, and `--live <id>` for a camera instead of a file. `--help` lists the rest.

Streaming mode reports more graph completions than frames supplied. That is
expected: the framework re-runs the graph as fast as the stages allow, whether
or not the host has managed to supply a new frame, which is what an always-on
sensor loop wants.

## Checking that pipelining did not change the answer

`--verify` does this in one run: it produces the lane masks from the unpipelined
`split` reference and from each pipelined mode over the same frames, then reports
how many come out identical in the interior (the six pixel border is excluded,
for the reason below).

```bash
./adasPipeline --video drive.mp4 --verify
```

To compare byte for byte instead, dump the same frames from an unpipelined mode
and a pipelined one, ignoring a six pixel frame around the edge:

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

The border has to be excluded because the edge detector's border pixels are
undefined by specification and this implementation leaves them unwritten, so
they hold whatever the buffer they landed in held before. The filter bank is
9x9 and sits between a dilate and an erode, which carries that undefined
content six pixels in. Running the *same* mode twice differs in those pixels
too, so it is not a pipelining artefact.

## Three things worth knowing before writing one of these

**A queued graph parameter rebinds exactly one node parameter.** A graph
parameter stands for one parameter of one node, so when a queue swaps a buffer
in, only that node sees the new buffer. If several nodes read the same queued
image, every node but the one owning the parameter keeps reading whichever
buffer was bound when the graph was built, and the results are quietly wrong
for every frame after the first. Stage 2 here has one node per filter
orientation, so it feeds the whole bank from a single node that owns the queued
input. Adding one graph parameter per consumer and enqueuing the same reference
to each would work too, at the cost of a queue each.

**A node cannot be timed on its own.** Asking a node for
`VX_NODE_PERFORMANCE` returns its *graph's* time in this implementation, for
every node alike: the single Canny node of a CPU run reports 63.742 ms, the
same figure the graph reports, and each of the 24 nodes in stage 0 reports it
as well. So the per-stage times above are only available for the staged
arrangement, where a stage is a graph. There is no way through the API to ask
where the time went inside a single graph.

**Node affinity is reported back as requested, whether or not it was honoured.**
Querying a node's target after verification returns whatever was asked for, so
a node with no implementation for the requested device falls back silently and
the only hint is that its time does not change. Worth checking a new pipeline
stage against both placements before believing it runs where it was put.
