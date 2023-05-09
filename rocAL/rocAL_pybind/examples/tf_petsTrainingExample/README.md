## Running tf_petsTrainingExample

### Building the required TF Rocm docker
* Use the instructions in the [docker section](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/docker) to build the required [Tensorflow docker](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/docker/tensorflow)
* Upgrade pip to the latest version.

### Building the required Pytorch Rocm docker
* Use the instructions in the [docker section](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/docker) to build the required [Pytorch docker](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/docker/pytorch)
* Upgrade pip to the latest version.
* Run requirements.sh to install the required packages.

### Running the training

* For first run, to setup dataset, edit "train_withROCAL_withTFRecordReader.py" and set "DATASET_DOWNLOAD_AND_PREPROCESS = True"
* For subsequent runs, after the dataset has already been downloaded and preprocessed, set "DATASET_DOWNLOAD_AND_PREPROCESS = False"

To run this example for the first run or subsequent runs, just execute:
```
python3 train_withROCAL_withTFRecordReader.py
```
