# annInferenceServer

This Sample Inference Server supports:
* convert and maintain a database of pre-trained CAFFE models using [caffe2openvx](../inference_generator/README.md)
* allow multiple TCP/IP client connections for inference work submissions
* multi-GPU high-throughput live streaming batch scheduler

Command-line usage:
````
% annInferenceServer [-p port]
                     [-b default-batch-size]
                     [-gpu <comma-separated-list-of-GPUs>]
                     [-q <max-pending-batches>]
                     [-w <server-work-folder>]
````

Make sure that all executables and libraries are in `PATH` and `LD_LIBRARY_PATH` environment variables.
````
% export PATH=$PATH:/opt/rocm/mivisionx/bin
% export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/mivisionx/lib
````

The `annInferenceServer` works with [annInferenceApp](../annInferenceApp/README.md).
* Execute `annInferenceServer` on the server machine with Radeon Instinct GPUs
* Execute `annInferenceApp` on one or more workstations: connect to the server and classify images using any pre-trained neural network
