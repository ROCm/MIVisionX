## Tensorflow MNIST digit classification example with RALI pipeline

## Prerequisites

* [ROCm supported hardware](https://rocm.github.io/ROCmInstall.html#hardware-support) 
    * AMD Radeon GPU or APU required
* [ROCm](https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories)
* Build & Install [RPP](https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp)
* Build & Install [MIVisionX](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX#linux-1)

### Step 1. Build and install rali_pybind plugin 
```
sudo ./run.sh (MIVisionX/rali/rali_pybind/)
```
### Step 2. Create venv and install tensorflow 1.15 into it & activate venv to run TF. (alternatively use on of the ROCm tensorflow docker container) 
* Install [TF1.15](https://www.tensorflow.org/install/pip)
* Avoid this step if running on a ROCm TensorFlow docker where TF is pre-installed 

### Step 3. download and prepare MNIST tfrecord dataset

```
* python3 mnist_tfrecord.py (creates tfrecord in /tmp/mnist folder)
* OR
* python3 mnist_tfrecord.py --directory <folderName> (creates tfrecord in <folderName>)

```
### Step 4. Run training with RALI on MNIST dataset
```
* python3 tf_mnist_classification_rali_py <mnist_tfrecord_folder, eg./tmp/mnist> cpu <batch_size>

* e.g. python3 tf_mnist_classification_rali_py /tmp/mnist cpu 128
* expected accuracy around 96% for 10 epochs if training was running correctly
```
### Step 5. Deactivate venv if applicable
* Uninstall TF [TF1.15](https://www.tensorflow.org/install/pip)
