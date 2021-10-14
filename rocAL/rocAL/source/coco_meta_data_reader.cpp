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

void COCOMetaDataReader::init(const MetaDataConfig &cfg)
{
    _path = cfg.path();
    _output = new BoundingBoxBatch();
}

bool COCOMetaDataReader::exists(const std::string& image_name)
{
    return _map_content.find(image_name) != _map_content.end();
}

void COCOMetaDataReader::lookup(const std::vector<std::string> &image_names) {

    if (image_names.empty())
    {
        WRN("No image names passed")
        return;
    }
    if (image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());

    for (unsigned i = 0; i < image_names.size(); i++)
    {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if (_map_content.end() == it) 
            THROW("ERROR: Given name not present in the map" + image_name)    
        _output->get_bb_cords_batch()[i] = it->second->get_bb_cords();
        _output->get_bb_labels_batch()[i] = it->second->get_bb_labels();
        _output->get_img_sizes_batch()[i] = it->second->get_img_sizes();
    }
}

void COCOMetaDataReader::add(std::string image_name, BoundingBoxCords bb_coords, BoundingBoxLabels bb_labels, ImgSizes image_size)
{
    if (exists(image_name))
    {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_bb_labels().push_back(bb_labels[0]);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels,image_size);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(image_name, info));
}

void COCOMetaDataReader::print_map_contents()
{
    BoundingBoxCords bb_coords;
    BoundingBoxLabels bb_labels;
    ImgSizes img_sizes;

    std::cout << "\nBBox Annotations List: \n";
    for (auto& elem : _map_content) {
        std::cout << "\nName :\t " << elem.first;
        bb_coords = elem.second->get_bb_cords() ;
        bb_labels = elem.second->get_bb_labels();
        img_sizes = elem.second->get_img_sizes();
        std::cout << "<wxh, num of bboxes>: "<< img_sizes[0].w << " X " << img_sizes[0].h << " , "  << bb_coords.size() << std::endl;
        for(unsigned int i = 0; i < bb_coords.size(); i++){
            std::cout << " l : " << bb_coords[i].l << " t: :" << bb_coords[i].t << " r : " << bb_coords[i].r << " b: :" << bb_coords[i].b << "Label Id : " << bb_labels[i] << std::endl;
        }
    }
}

void COCOMetaDataReader::read_all(const std::string &path) {

    _coco_metadata_read_time.start();// Debug timing
    std::string annotation_file = path;
    std::ifstream fin;
    fin.open(annotation_file, std::ios::in);

    std::string str;
    str.assign(std::istreambuf_iterator<char>(fin), std::istreambuf_iterator<char>());
    BoundingBoxCords bb_coords;
      BoundingBoxLabels bb_labels;
      ImgSizes img_sizes;

    Json::Reader reader;
    Json::Value root;
    if (reader.parse(str, root) == false) {
          WRN("Failed to parse Json: " + reader.getFormattedErrorMessages());
    }

    Json::Value annotation = root["annotations"];
    Json::Value image = root["images"];

    BoundingBoxCord box;
    ImgSize img_size;
    
    for (auto iterator = image.begin(); iterator != image.end(); iterator++)
    {
        // std::map<int, int,int> id_img_sizes;
        img_size.h = (*iterator)["height"].asInt();
        img_size.w = (*iterator)["width"].asInt();
        img_sizes.push_back(img_size);
        string image_name  = (*iterator)["file_name"].asString();
        
        _map_img_sizes.insert(pair<std::string,std::vector<ImgSize> >(image_name, img_sizes));
        img_sizes.clear();

    }
    
    for (auto iterator = annotation.begin(); iterator != annotation.end(); iterator++)
    {
        float box_x = (*iterator)["bbox"][0].asFloat();
        float box_y = (*iterator)["bbox"][1].asFloat();
        float box_w = (*iterator)["bbox"][2].asFloat();
        float box_h = (*iterator)["bbox"][3].asFloat();
        int label = (*iterator)["category_id"].asInt();
        int id = (*iterator)["image_id"].asInt();
        char buffer[13];
        sprintf(buffer, "%012d", id);
        string str(buffer);
        std::string file_name = str + ".jpg";

        auto it = _map_img_sizes.find(file_name);
        ImgSizes image_size = it->second;        

        //Normalizing the co-ordinates & convert to "ltrb" format
        box.l = box_x/ image_size[0].w;
        box.t = box_y/ image_size[0].h;
        box.r = (box_x + box_w) / image_size[0].w;
        box.b = (box_y + box_h) / image_size[0].h;
        
        bb_coords.push_back(box);
        bb_labels.push_back(label);
        add(file_name, bb_coords, bb_labels,image_size);
        bb_coords.clear();
        bb_labels.clear();
    } 
    fin.close();
    _coco_metadata_read_time.end();// Debug timing
    //print_map_contents();
    //std::cout<<"coco read time in sec: " << _coco_metadata_read_time.get_timing()/1000 << std::endl;
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
    _map_img_sizes.clear();
}

COCOMetaDataReader::COCOMetaDataReader():
        _coco_metadata_read_time("coco meta read time", DBG_TIMING)
{
}
