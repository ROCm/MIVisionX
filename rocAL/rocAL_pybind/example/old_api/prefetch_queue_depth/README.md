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
python3.6 ./prefetch_queue_depth.py  <path to the dataset> <cpu/gpu> <batch_size> <prefetch_queue_depth>

## Example
* Run with 10 images and batch size 2 on AMD Ryzen 9 3950X 16-Core Processor with nproc - 32.

prefetch_queue_depth as 2

root@rocal:/media/MIVisionX/rocAL/rocAL_pybind# python3.6 ./prefetch_queue_depth.py /media/samples/ cpu 2 2
OK: loaded 80 kernels from libvx_rpp.so
Pipeline has been created succesfully
Time taken (averaged over 10 runs)  10424 milli seconds

prefetch_queue_depth as 4

root@rocal:/media/MIVisionX/rocAL/rocAL_pybind# python3.6 ./prefetch_queue_depth.py /media/samples/ cpu 2 4
OK: loaded 80 kernels from libvx_rpp.so
Pipeline has been created succesfully
Time taken (averaged over 10 runs)  10397 milli seconds
