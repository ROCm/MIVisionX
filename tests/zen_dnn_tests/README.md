# Zen DNN - Tests

## AlexNet F32 Sample
This C++ API example demonstrates how to build an AlexNet neural network topology for forward-pass inference.

### Some key take-aways include:

* How tensors are implemented and submitted to primitives.
* How primitives are created.
* How primitives are sequentially submitted to the network, where the output from primitives is passed as input to the next primitive. The latter specifies a dependency between the primitive input and output data.
* Specific 'inference-only' configurations.
* Limiting the number of reorders performed that are detrimental to performance.

**NOTE:** The example implements the AlexNet layers as numbered primitives (for example, conv1, pool1, conv2).

### Build & Run
```
mkdir build-alexnet && cd build-alexnet
cmake ../alexnet && make
ZENDNN_LOG_OPTS=ALL:5 ZENDNN_VERBOSE=1 ./zendnn_alexnet_f32_test cpu
```

## Convolution Layer Test

### Steps:
1. create engin and stream
2. create user memory (source, weights, destination)
3. create memory descriptor
4. create convolution descriptor
5. create convolution primitive descriptor
6. create convolution primitive
7. execute the convlution
8. create ReLU desciptor
9. create ReLU primitive descriptor
10. create ReLU primitive
11. execute ReLU

### Build & Run
```
mkdir build-conv && cd build-conv
cmake ../conv && make
ZENDNN_LOG_OPTS=ALL:5 ZENDNN_VERBOSE=1 ./zendnn_conv_test cpu
```

**NOTE:** use `export ZENDNN_LOG_OPTS=ALL:5 ZENDNN_VERBOSE=1` for developer information