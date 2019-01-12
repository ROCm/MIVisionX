#include "Argmax.h"
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <string>


int Argmax::mCount = 0;

Argmax::Argmax(const char* fileName, const std::string labelName, const std::string tagName) : mFileName(fileName) {
    
    std::ifstream label(labelName, std::ios_base::in);
    if (!label) {
        std::cout << "Failed on loading filename: " << labelName << std::endl;
        exit(-1);
    }
    // else {
    //     std::cout << "Label name: " << labelName << std::endl;
    // }

    mLabel = txtToVector(label);
    label.close();

    std::ifstream tag(tagName, std::ios_base::in);
    if (!tag) {
        std::cout << "Failed on loading filename: " << tagName << std::endl;
        exit(-1);
    }
    // else {
    //     std::cout << "Tag name: " << tagName << std::endl;
    // }

    mTag = txtToVector(tag);
    tag.close();
}

Argmax::~Argmax() {
}

void Argmax::setIndex (const std::vector<float> &vec) {
    auto itr = max_element(vec.begin(), vec.end());
    mIndex = distance(vec.begin(), itr);
}

int Argmax::getIndex () {
    return mIndex;
}

void Argmax::printResult(const std::vector<std::string> &label, const std::vector<std::string> &tag) {
    std::cout << "Tag name: " << tag.at(mCount) << std::endl;
    mCount++;
    std::cout << "Classified as: " << label.at(getIndex()) << std::endl << std::endl;
}

int Argmax::getLabelSize() {
    return mLabel.size();
}

int Argmax::getTagSize() {
    return mTag.size();
}

std::vector<std::string> Argmax::txtToVector(std::ifstream &textFile) {
    std::string str;
    std::vector<std::string> vec;
    while (getline(textFile,str)) {
        vec.push_back(str);
    }
    return vec;
}

void Argmax::run() {
    std::vector<float> vec;
    int classSize = getLabelSize();
    int count = 0;
    float c;

    FILE *file = fopen(mFileName, "rb");
    if (!file) {
        std::cout << "Failed on loading filename: " << mFileName << std::endl;
        exit(-1);
    }
    // else {
    //     std::cout << "File name: " << mFileName << std::endl;
    // }
    
    std::cout << std::endl << "Classification Result" << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl << std::endl;

    while (fread(&c, sizeof(float),1, file)) {
        vec.push_back(c);
        count++;
        if (count%classSize == 0) {
            setIndex(vec);
            printResult(mLabel, mTag);
            vec.clear();
        }
    }

    fclose(file);
}