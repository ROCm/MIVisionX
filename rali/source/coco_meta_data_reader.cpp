
/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "coco_meta_data_reader.h"
#include <iostream>
#include <utility>
#include <algorithm>
#include <jsoncpp/json/value.h>
#include <jsoncpp/json/json.h>
#include<fstream>

using namespace std;

void COCOMetaDataReader::init(const MetaDataConfig &cfg) {
    _path = cfg.path();
    _output = new BoundingBoxBatch();
}

bool COCOMetaDataReader::exists(const std::string& image_name)
{
    return _map_content.find(image_name) != _map_content.end();
}

void COCOMetaDataReader::lookup(const std::vector<std::string> &image_names) {

    if(image_names.empty())
    {
        WRN("No image names passed")
        return;
    }
    if(image_names.size() != (unsigned)_output->size())
    {
        _output->resize(image_names.size());
    }

    for(unsigned i = 0; i < image_names.size(); i++)
    {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        /*
         * User should provide the coco train or val folder containing images with respect to json file.
         * If the processed COCO image was not in the map, returns BoundingBox meta data values as zero since
         * those images doesn't have annotations.
         */
        if(_map_content.end() == it) {

            BoundingBoxCords bb_coords;
            BoundingBoxLabels bb_labels;
            BoundingBoxCord box;

            box.x = box.y = box.w = box.h = 0;
            bb_coords.push_back(box);
            bb_labels.push_back(0);
            _output->get_bb_cords_batch()[i] = bb_coords;
            _output->get_bb_labels_batch()[i] = bb_labels;
        }
        else {
            _output->get_bb_cords_batch()[i] = it->second->get_bb_cords();
            _output->get_bb_labels_batch()[i] = it->second->get_bb_labels();
        }
    }
}

void COCOMetaDataReader::add(std::string image_name, BoundingBoxCords bb_coords, BoundingBoxLabels bb_labels)
{
    if(exists(image_name))
    {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_bb_labels().push_back(bb_labels[0]);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(image_name, info));
}

void COCOMetaDataReader::print_map_contents()
{
    BoundingBoxCords bb_coords;
    BoundingBoxLabels bb_labels;

    std::cerr << "\nMap contents: \n";
    for (auto& elem : _map_content) {
        std::cerr << "Name :\t " << elem.first;
        bb_coords = elem.second->get_bb_cords() ;
        bb_labels = elem.second->get_bb_labels();
        std::cerr << "\nsize of the element  : "<< bb_coords.size() << std::endl;
        for(unsigned int i = 0; i < bb_coords.size(); i++) {
            std::cerr << " x : " << bb_coords[i].x << " y: :" << bb_coords[i].y << " width : " << bb_coords[i].w << " height: :" << bb_coords[i].h << std::endl;
            std::cerr  << "Label Id : " << bb_labels[i] << std::endl;
        }
    }
}

void COCOMetaDataReader::read_all(const std::string &path) {

    std::string annotation_file = path;
    std::ifstream fin;
    fin.open(annotation_file, std::ios::in);

    std::string str;
    str.assign(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    BoundingBoxCords bb_coords;
    BoundingBoxLabels bb_labels;

    Json::Reader reader;
    Json::Value root;
    if (reader.parse(str, root) == false) {
        WRN("Failed to parse Json: " + reader.getFormattedErrorMessages());
    }

    Json::Value annotation = root["annotations"];
    Json::Value image = root["images"];

    BoundingBoxCord box;

    for (auto iterator = annotation.begin(); iterator != annotation.end(); iterator++) {
        box.x = (*iterator)["bbox"][0].asFloat();
        box.y = (*iterator)["bbox"][1].asFloat();
        box.w = (*iterator)["bbox"][2].asFloat();
        box.h = (*iterator)["bbox"][3].asFloat();
        int label = (*iterator)["category_id"].asInt();
        int id = (*iterator)["image_id"].asInt();
        char buffer[13];
        sprintf(buffer, "%012d", id);
        string str(buffer);
        std::string file_name = str + ".jpg";
        bb_coords.push_back(box);
        bb_labels.push_back(label);
        add(file_name, bb_coords, bb_labels);
        bb_coords.clear();
        bb_labels.clear();
    }
    //print_map_contents();
}

void COCOMetaDataReader::release(std::string image_name) {
    if(!exists(image_name))
    {
        WRN("ERROR: Given name not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void COCOMetaDataReader::release() {
    _map_content.clear();
}

COCOMetaDataReader::COCOMetaDataReader()
{
}
