##  Test
This application briefs how to configure and pass the prefetch_queue_depth pipeline arugument.
This  test  aims to evaluate the prefetch queue depth support and the perfomance boost up.

## Prefetch queue depth
* The rocAL pipeline allows the buffering of one or more batches of data.
* This can be achieved by configuring the prefetch_queue_depth  pipeline argument which sets the depth of buffer in both load and output routine.
* Configuring this paramter controls the buffer that keeps the decoded image batch ready for processing and the processed image batch ready for user.
* The default prefetch depth is 2.
* Depending on the machine configuration decreasing or increasing   prefetch_queue_depth helps in achieving better performance.

## Running the app
`python3 ./prefetch_queue_depth.py  <path to the dataset> <cpu/gpu> <batch_size> <prefetch_queue_depth>`

## Example
* Run with 10 images and batch size 2 on AMD EPYC 7552 48-Core Processor with nproc - 192.

prefetch_queue_depth as 2

```
OK: loaded 82 kernels from libvx_rpp.so
Pipeline has been created succesfully
Time taken (averaged over 10 runs)  10513 milli seconds
```

prefetch_queue_depth as 4

```
OK: loaded 82 kernels from libvx_rpp.so
Pipeline has been created succesfully
Time taken (averaged over 10 runs)  10491 milli seconds
```