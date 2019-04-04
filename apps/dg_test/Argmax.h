#pragma once

#include <vector>
#include <fstream>

/**
 *  Utility class for reading the tensor file and argmax it against the label file
 */
class Argmax
{
public:
    Argmax(const char* fileName, const std::string labelName, const std::string tagName);
    ~Argmax();

    /**
     *  Caculate the index number with the maximum probability
     */
    void setIndex (const std::vector<float> &vec);

    /**
     *  Get the current index
     */
    int getIndex ();

    /**
     *  Get the size of the label file
     */
    int getLabelSize();

    /**
     *  Get the size of the label file
     */
    int getTagSize();

    /**
     *  Prints out the result
     */
    void printResult(const std::vector<std::string> &list, const std::vector<std::string> &tag);

    /**
     *  Converts the text file to the vector
     */
    std::vector<std::string> txtToVector(std::ifstream &textFile);

    /**
     *  Run the argmax
     */
    void run();

private:

    /**
     *  The index of the maximum probability will be stored
     */
    int mIndex;

    /**
     *  The count of a current image
     */
    static int mCount;

    /**
     *  File name to open and read the tensor object
     */
    const char* mFileName;

    /**
     *  File stream to open and read the label text file
     */
    std::vector<std::string> mLabel;
    
    /**
     *  File stream to open and read the tag text file
     */
    std::vector<std::string> mTag;
};