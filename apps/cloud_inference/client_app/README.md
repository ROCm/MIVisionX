# annInferenceApp

This Sample Inference Application supports:
* connect to [annInferenceServer](../annInferenceServer/README.md) 
  * queries number of GPUs on the server
  * queries pre-loaded neural network models on the server
* upload pre-trained CAFFE models (optional)
  * specify input width and height dimensions
  * browse and pick deploy.prototxt and weights.caffemodel files
  * specify input preprocesor values for normalization (if needed)
  * specify input channel order (RGB or BGR)
  * optionally, you can publish your neural network for use by others
    * select "PublishAs"
    * give a name to your model
    * type password (default passowrd of `annInferenceServer` is "**radeon**")
* pick a CNN model, if upload option is not used
* run inference on images from a folder
  * browse and pick labels (i.e., synset words for each output label)
  * browse and pick input images to be classified
  * optionally, pick list of input images from a .txt or .csv file
  * optionally, specify max number of images to be processed from folder or image list

If you want to run a simple test, use [annInferenceApp.py](annInferenceApp.py) (a python script) to simply pick a network model to classify local images.
````
% python annInferenceApp.py [-v] -host:<hostname> -port:<port> -model:<modelName>
                            [-synset:<synset.txt>] [-output:<output.csv>]
                            <folder>|<file(s)>
````
