.. meta::
  :description: MIVisionX supported models
  :keywords: MIVisionX, ROCm, support, models, operators

******************************************
MIVisionX supported models and operators
******************************************

The following tables list the models and operators supported by different frameworks in the current release of MIVisionX. 

Models
------

.. |blue-sq| image:: https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png
    :alt: Blue Square

.. csv-table::
  :widths: 2, 1, 1, 1

    **Networks**, **Caffe**, **ONNX**, **NNEF**
    AlexNet, ,|blue-sq|, |blue-sq|
    Caffenet, , |blue-sq|,  
    DenseNet, , |blue-sq| 						
    Googlenet, |blue-sq| , |blue-sq| , |blue-sq| 		
    Inception-V1, , |blue-sq| , |blue-sq| 		
    Inception-V2, , |blue-sq| , |blue-sq| 			
    Inception-V3, , , 			
    Inception-V4, |blue-sq| , , 			
    MNIST, |blue-sq| , , |blue-sq| 		
    Mobilenet, , |blue-sq| , |blue-sq| 		
    MobilenetV2, , , |blue-sq| 
    ResNet-18, , , |blue-sq| 			
    ResNet-34, , , |blue-sq| 			
    ResNet-50, |blue-sq| , |blue-sq| , |blue-sq| 			
    ResNet-101, |blue-sq| , , |blue-sq| 		
    ResNet-152, |blue-sq| , , |blue-sq| 			
    ResNetV2-18, , , |blue-sq| 			
    ResNetV2-34, , , |blue-sq| 		
    ResNetV2-50, , , |blue-sq| 		
    ResNetV2-101, , , |blue-sq| 			
    Squeezenet, , |blue-sq| , |blue-sq| 			
    Tiny-Yolo-V2, |blue-sq| , , 			
    VGGNet-16, |blue-sq| , , |blue-sq| 			
    VGGNet-19, |blue-sq| , |blue-sq| , |blue-sq| 			
    Yolo-V3, |blue-sq| , , 			
    ZFNet, , |blue-sq| , 

.. note::
    MIVisionX supports `ONNX models <https://github.com/onnx/models>`_ with `release 1.1` and `release 1.3` tags

Operators
---------

.. csv-table::
  :widths: 2, 1, 1, 1

    **Layers**, **Caffe**, **ONNX**, **NNEF**
    Add, ,|blue-sq|, |blue-sq| 
    Argmax, ,|blue-sq|,|blue-sq| 
    AveragePool,,|blue-sq|,|blue-sq| 
    BatchNormalization,|blue-sq|,|blue-sq|,|blue-sq| 
    Cast,,|blue-sq|,
    Clamp,,,|blue-sq| 
    Clip,,|blue-sq|,
    Concat,|blue-sq|,|blue-sq|,|blue-sq| 
    Constant,,|blue-sq|,
    Conv,|blue-sq|,|blue-sq|,|blue-sq| 
    ConvTranspose,|blue-sq|,|blue-sq|,|blue-sq| 
    Copy,,|blue-sq|,|blue-sq| 
    Crop,|blue-sq|,,
    CropAndResize,,,
    Deconv,|blue-sq|,|blue-sq|,|blue-sq| 
    DetectionOutput,|blue-sq|,,
    Div,,|blue-sq|,|blue-sq| 
    Dropout,,,
    Eltwise,|blue-sq|,,
    Exp,,|blue-sq|,|blue-sq| 
    Equal,,|blue-sq|,
    Flatten,|blue-sq|,,
    Gather,,|blue-sq|,
    GEMM,|blue-sq|,|blue-sq|,|blue-sq| 
    GlobalAveragePool,,|blue-sq|,|blue-sq| 
    Greater,,|blue-sq|,
    GreaterOrEqual,,|blue-sq|,
    InnerProduct,|blue-sq|,,
    Interp,|blue-sq|,,
    LeakyRelu,,|blue-sq|,|blue-sq| 
    Less,,|blue-sq|,
    LessOrEqual,,|blue-sq|,
    Linear,,,|blue-sq| 
    Log,,|blue-sq|,|blue-sq| 
    LRN,|blue-sq|,|blue-sq|,|blue-sq| 
    Matmul,,|blue-sq|,|blue-sq| 
    Max,,|blue-sq|,|blue-sq| 
    MaxPool,,|blue-sq|,|blue-sq| 
    MeanReduce,,,|blue-sq| 
    Min,,|blue-sq|,|blue-sq| 
    Mul,,|blue-sq|,|blue-sq| 
    MulAdd,,,
    NonMaxSuppression,,|blue-sq|,
    Permute,|blue-sq|,,|blue-sq| 
    PriorBox,|blue-sq|,,
    ReduceMin,,|blue-sq|,
    Relu,|blue-sq|,|blue-sq|,|blue-sq| 
    Reshape,|blue-sq|,|blue-sq|,|blue-sq| 
    Shape,,|blue-sq|,
    Sigmoid,,|blue-sq|,|blue-sq| 
    Slice,,|blue-sq|,|blue-sq| 
    Split,|blue-sq|,,
    Softmax,|blue-sq|,|blue-sq|,|blue-sq| 
    SoftmaxWithLoss,|blue-sq|,,
    Squeeze,,|blue-sq|,|blue-sq| 
    Sub,,|blue-sq|,|blue-sq| 
    Sum,,|blue-sq|,
    Tile,,|blue-sq|,
    TopK,,|blue-sq|,
    Transpose,,|blue-sq|,|blue-sq| 
    Unsqueeze,,|blue-sq|,|blue-sq| 
    Upsample,|blue-sq|,,|blue-sq| 

