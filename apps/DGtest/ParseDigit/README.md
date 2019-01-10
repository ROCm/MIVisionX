# AMD ParseDigit

The AMD ParseDigit is a useful tool for preparing image dataset for [MNIST](http://yann.lecun.com/exdb/mnist/). 
The program will automatically detect digits from the input image, crop, and preprocess the images.

### Explanation
The program uses opencv [MSER function](https://docs.opencv.org/3.1.0/d3/d28/classcv_1_1MSER.html) for detecting digits.
After detecting and cropping the images, it preprocesses the images so that it can be used for MNIST dataset.
The detailed explanation of preprocessing the images can be found at http://opensourc.es/blog/tensorflow-mnist


### Pre-requisites
##### 1. OpenCV 3.3 [download](https://opencv.org/opencv-3-3.html) or higher
##### 2. cmake git 
    sudo apt-get install cmake git

### Build using Cmake on Linux (Ubuntu 16.04 64bit)
     mkdir build
     cd build
     cmake ..
     make

### Usage
     Usage: ./ParseDigit [input_image] [directory_name] [image_name] [-a]
     
     1. [input_image]
         The name of the input image (including the directory to the image) to run the detection and crop.
     2. [directory_name]
         The name of the path where the images will be saved.
     3. [image_name]
         The name of the ouput images (including the directory to the image).
         The name of the images will be "[image_name]00[digit]-[digit count].jpg".
         If -a option is set, the names will be "[image_name]001.jpg", "[image_name]002.jpg" ... and so on.
     4. [-a]
         If -a option is set, it will skip the digit verification process and automatically name the images 
         as mentioned above. Otherwise, it will show each detected digits and go through verification process.
        
### Guideline for Image Preparation
    When preparing your own image for the input, make sure that 
     
     1. The paper does not contain any lines. The plain A4 paper is recommended.
     2. Write the digits big and clear enough so that it can be recognized.
     3. When taking the photo of it, make sure there are no other objects other than the paper.
     
    See the Example/image.jpg & Example/image2.jpg for the reference.
     
### Example
    ./ParseDigit ../Examples/image.jpg ../Examples/Cropped/ digits
    
    The output images will be stored in ../Examples/Cropped folder as digits001-1.jpg, digits001-2.jpg, ... digits009-5.jpg.
    Make sure the destination folder is created before running the program.
   
