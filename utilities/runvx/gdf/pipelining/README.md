# Proposed pipelining GDF examples

These GDF files show what `vx_khr_pipelining` support in `runvx` could look like.
They use **proposed syntax** that is not yet implemented. Do not run them with
`runvx` until issue #1734 is resolved.

## Proposed commands

```gdf
graph parameter <dataName> node <kernelName> param <index>
graph schedule queue-auto depth <N>
graph enqueue <parameterName> <refName1> [<refName2> ...]
graph launch-pipelined frames <F>
```

## Files

- `pipelining_box3x3.gdf` — light preset: single `Box3x3` filter.
- `pipelining_gaussian3x3.gdf` — heavy preset: single `Gaussian3x3` filter.
- `pipelining_hybrid.gdf` — explicit CPU/GPU/CPU pinning with `Box3x3` chain.

## What they test

Once issue #1734 is implemented, these GDFs should produce deterministic
aggregate checksums across sync and pipelined runs, matching the behavior of
the C++ samples in `samples/c_samples/pipelining/`.
