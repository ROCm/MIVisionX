#include "DGtest.h"
#include "Argmax.h"
#include <string>
#include <iostream>

int main(int argc, const char ** argv)
{
   // check command-line usage
    if(argc != 6) {
        printf(
            "\n"
            "Usage: DGtest [weights.bin] [input-tensor] [output-tensor] [labels.txt] [imagetag.txt]\n"
            "\n"
            "   <weights.bin>: is the name of the weights file to be used for the inference\n."
            "\n"
            "   <input-tensor>: is filename to initialize tensor\n"
            "\n"
            "   <output-tensor>: is filename to write out the tensor\n"
            "\n"
            "   <labels.txt>: is the text file containing the labels of each classes\n"
            "\n"
            "   <imagetag.txt>: is the text file contaning each images' directories\n"
            "\n"
        );
        return -1;
    }
    // create and initialize input tensor data
    const char* weights = argv[1];
    std::string inputFile = argv[2];
    std::string outputFile = argv[3];
    std::string labelFile = argv[4];
    std::string tagFile = argv[5];

    Argmax argmax(outputFile.c_str(), labelFile, tagFile);
    
    int batchSize = argmax.getTagSize();

    DGtest dgtest(weights, inputFile, outputFile, batchSize);
    
    dgtest.runInference();

    argmax.run();

    return 0;
}