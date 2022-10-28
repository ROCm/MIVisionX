## Running tf_petsTrainingExample

### Building the required TF Rocm docker
Use the instructions in the [docker section](../../../../../docker) to build the required [Tensorflow docker](../../../../../docker/tensorflow)

### Running the training

* For first run, to setup dataset, edit "train_withROCAL_withTFRecordReader.py" and set "DATASET_DOWNLOAD_AND_PREPROCESS = True"
* For subsequent runs, after the dataset has already been downloaded and preprocessed, set "DATASET_DOWNLOAD_AND_PREPROCESS = False"

To run this example for the first run or subsequent runs, just execute:
```
python3.9 train_withROCAL_withTFRecordReader.py
```
