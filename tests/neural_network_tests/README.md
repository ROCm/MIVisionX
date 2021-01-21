# MIVisionX Neural Network Tests

## Script to run neural network tests

```
python3 runNeuralNetworkTests.py --help
```

usage: 

```
runNeuralNetworkTests.py [--profiler_mode PROFILER_MODE]
                         [--profiler_level PROFILER_LEVEL]
                         [--miopen_find MIOPEN_FIND]

Arguments:
  -h, --help            show this help message and exit
  --profiler_mode       NN Profile Mode - optional (default:0 [range:0 - 9])
  --profiler_level      NN Profile Batch Size in powers of 2 - optional (default:7 [range:1 - N])
  --miopen_find         MIOPEN_FIND_ENFORCE mode - optional (default:1 [range:1 - 5])
```

**Note:** Use `Python3` to run the script