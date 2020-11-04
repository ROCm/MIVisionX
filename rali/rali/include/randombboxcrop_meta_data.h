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

#pragma once
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include "commons.h"


typedef  struct { float crop_x; float crop_y; float crop_width; float crop_height; } CropBoxCord;
typedef  std::vector<CropBoxCord> CropBoxCords;
typedef  std::vector<int> CropBoxLabels;

struct RandomBBoxCrop_MetaData
{
    int& get_label() { return _label_id; }
    CropBoxCords& get_bb_cords() { return _bb_cords; }
    CropBoxLabels& get_bb_labels() { return _bb_label_ids; }
protected:
    CropBoxCords _bb_cords = {}; // For bb use
    CropBoxLabels _bb_label_ids = {};// For bb use
    int _label_id = -1; // For label use only
};

struct CropBox : public RandomBBoxCrop_MetaData
{
    CropBox()= default;
    CropBox(CropBoxCords bb_cords,CropBoxLabels bb_label_ids )
    {
        _bb_cords =std::move(bb_cords);
        _bb_label_ids = std::move(bb_label_ids);
    }
    void set_bb_cords(CropBoxCords bb_cords) { _bb_cords =std::move(bb_cords); }
    void set_bb_labels(CropBoxLabels bb_label_ids) {_bb_label_ids = std::move(bb_label_ids); }
};

struct RandomBBoxCrop_MetaDataBatch
{
    virtual ~RandomBBoxCrop_MetaDataBatch() = default;
    virtual void clear() = 0;
    virtual void resize(int batch_size) = 0;
    virtual int size() = 0;
    virtual RandomBBoxCrop_MetaDataBatch&  operator += (RandomBBoxCrop_MetaDataBatch& other) = 0;
    RandomBBoxCrop_MetaDataBatch* concatenate(RandomBBoxCrop_MetaDataBatch* other)
    {
        *this += *other;
        return this;
    }
    virtual std::shared_ptr<RandomBBoxCrop_MetaDataBatch> clone()  = 0;
    std::vector<int>& get_label_batch() { return _label_id; }
    std::vector<CropBoxCords>& get_bb_cords_batch() { return _bb_cords; }
    std::vector<CropBoxLabels>& get_bb_labels_batch() { return _bb_label_ids; }
protected:
    std::vector<int> _label_id = {}; // For label use only
    std::vector<CropBoxCords> _bb_cords = {};
    std::vector<CropBoxLabels> _bb_label_ids = {};
};

struct CropBoxBatch: public RandomBBoxCrop_MetaDataBatch
{
    void clear() override
    {
        _bb_cords.clear();
        _bb_label_ids.clear();
    }
    RandomBBoxCrop_MetaDataBatch&  operator += (RandomBBoxCrop_MetaDataBatch& other) override
    {
        _bb_cords.insert(_bb_cords.end(),other.get_bb_cords_batch().begin(), other.get_bb_cords_batch().end());
        _bb_label_ids.insert(_bb_label_ids.end(), other.get_bb_labels_batch().begin(), other.get_bb_labels_batch().end());
        return *this;
    }
    void resize(int batch_size) override
    {
        _bb_cords.resize(batch_size);
        _bb_label_ids.resize(batch_size);
    }
    int size() override
    {
        return _bb_cords.size();
    }
    std::shared_ptr<RandomBBoxCrop_MetaDataBatch> clone() override
    {
        return std::make_shared<CropBoxBatch>(*this);
    }
};
using ImageNameBatch = std::vector<std::string>;
using pRandomBBoxCrop_MetaDataBox = std::shared_ptr<CropBox>;
using pRandomBBoxCrop_MetaDataBatch = std::shared_ptr<RandomBBoxCrop_MetaDataBatch>;
