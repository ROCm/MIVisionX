/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include <string.h>
#include <iostream>
#include <utility>
#include <algorithm>
#include <boost/filesystem.hpp>
#include "commons.h"
#include "exception.h"
#include "external_source_label_reader.h"


using namespace std;

namespace filesys = boost::filesystem;

ExternalSourceLabelReader::ExternalSourceLabelReader()
{
}

void ExternalSourceLabelReader::init(const MetaDataConfig& cfg)
{
    _output = new LabelBatch();
}

bool ExternalSourceLabelReader::exists(const std::string& image_name)
{
    return _map_content.find(image_name) != _map_content.end();
}

void ExternalSourceLabelReader::add(std::string image_name, int label)
{
    pMetaData info = std::make_shared<Label>(label);
    if(exists(image_name))
    {
        WRN("Entity with the same name exists")
        return;
    }
    _map_content.insert(pair<std::string, std::shared_ptr<Label>>(image_name, info));
}

void  ExternalSourceLabelReader::add_labels(std::vector<std::string> image_name, std::vector<int> label)
{
    if(image_name.size() != label.size()) { THROW("ERROR: Image name and labels should have same size") }
    for(uint i = 0; i < image_name.size(); i++)
        add(image_name[i], label[i]);
}

void ExternalSourceLabelReader::print_map_contents()
{
    std::cerr << "\nMap contents: \n";
    for (auto& elem : _map_content)
        std::cerr << "Name :\t " << elem.first << "\t ID:  " << elem.second->get_label() << std::endl;
}

void ExternalSourceLabelReader::release()
{
    _map_content.clear();
}

void ExternalSourceLabelReader::release(std::string image_name)
{
    if(!exists(image_name))
    {
        WRN("ERROR: Given not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void ExternalSourceLabelReader::lookup(const std::vector<std::string>& image_names)
{
    if(image_names.empty())
    {
        WRN("No image names passed")
        return;
    }
    if(image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());

    for(unsigned i = 0; i < image_names.size(); i++)
    {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if(_map_content.end() == it)
            THROW("ERROR: Given name not present in the map"+ image_name )
        _output->get_label_batch()[i] = it->second->get_label();
    }
}