#include "caffe_meta_data_reader_detection.h"
#include <iostream>
#include <fstream>
#include <utility>
#include <algorithm>
#include <fstream>
#include <string>
#include <stdint.h>
#include <google/protobuf/message_lite.h>
#include "lmdb.h"
#include "caffe_protos.pb.h"

using namespace std;

void CaffeMetaDataReaderDetection::init(const MetaDataConfig &cfg)
{
    _path = cfg.path();
    _output = new BoundingBoxBatch();
}

bool CaffeMetaDataReaderDetection::exists(const std::string &_image_name)
{
    return _map_content.find(_image_name) != _map_content.end();
}

void CaffeMetaDataReaderDetection::add(std::string image_name, BoundingBoxCords bb_coords, BoundingBoxLabels bb_labels)
{
    if (exists(image_name))
    {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_bb_labels().push_back(bb_labels[0]);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(image_name, info));
}

void CaffeMetaDataReaderDetection::lookup(const std::vector<std::string> &_image_names)
{
    if (_image_names.empty())
    {
        WRN("No image names passed")
        return;
    }
    if (_image_names.size() != (unsigned)_output->size())
        _output->resize(_image_names.size());

    for (unsigned i = 0; i < _image_names.size(); i++)
    {
        auto image_name = _image_names[i];
        auto it = _map_content.find(image_name);
        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map" + image_name)
        _output->get_bb_cords_batch()[i] = it->second->get_bb_cords();
        _output->get_bb_labels_batch()[i] = it->second->get_bb_labels();
    }
}

void CaffeMetaDataReaderDetection::print_map_contents()
{
    BoundingBoxCords bb_coords;
    BoundingBoxLabels bb_labels;

    std::cerr << "\nMap contents: \n";
    for (auto &elem : _map_content)
    {
        std::cerr << "Name :\t " << elem.first;
        bb_coords = elem.second->get_bb_cords();
        bb_labels = elem.second->get_bb_labels();
        std::cerr << "\nsize of the element  : " << bb_coords.size() << std::endl;
        for (unsigned int i = 0; i < bb_coords.size(); i++)
        {
            std::cerr << " x : " << bb_coords[i].x << " y: :" << bb_coords[i].y << " width : " << bb_coords[i].w << " height: :" << bb_coords[i].h << std::endl;
            std::cerr << "Label Id : " << bb_labels[i] << std::endl;
        }
    }
}

void CaffeMetaDataReaderDetection::read_all(const std::string &path)
{
    string tmp1 = path + "/data.mdb";
    string tmp2 = path + "/lock.mdb";
    uint file_size, file_size1, file_bytes;

    ifstream in_file(tmp1, ios::binary);
    in_file.seekg(0, ios::end);
    file_size = in_file.tellg();
    ifstream in_file1(tmp2, ios::binary);
    in_file1.seekg(0, ios::end);
    file_size1 = in_file1.tellg();
    file_bytes = file_size + file_size1;
    read_lmdb_record(path, file_bytes);
    // print_map_contents();
}

void CaffeMetaDataReaderDetection::read_lmdb_record(std::string file_name, uint file_byte_size)
{
    int rc;
    // Creating an LMDB environment handle
    E(mdb_env_create(&mdb_env));
    // Setting the size of the memory map to use for this environment.
    // The size of the memory map is also the maximum size of the database.
    E(mdb_env_set_mapsize(mdb_env, file_byte_size));
    // Opening an environment handle.
    E(mdb_env_open(mdb_env, _path.c_str(), MDB_RDONLY, 0664));
    // Creating a transaction for use with the environment
    E(mdb_txn_begin(mdb_env, NULL, MDB_RDONLY, &mdb_txn));
    // Opening a database in the environment.
    E(mdb_open(mdb_txn, NULL, 0, &mdb_dbi));
    // Creating a cursor handle.
    // A cursor is associated with a specific transaction and database
    E(mdb_cursor_open(mdb_txn, mdb_dbi, &mdb_cursor));

    // Retrieve by cursor. It retrieves key/data pairs from the database
    while ((rc = mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT)) == 0)
    {
        std::string file_name = string((char *)mdb_key.mv_data);

        caffe_protos::AnnotatedDatum annotatedDatum_protos;
        annotatedDatum_protos.ParseFromArray((char *)mdb_value.mv_data, mdb_value.mv_size);
        caffe_protos::AnnotationGroup annotGrp_protos = annotatedDatum_protos.annotation_group(0);

        int boundBox_size = annotGrp_protos.annotation_size();

        BoundingBoxCords bb_coords;
        BoundingBoxLabels bb_labels;
        BoundingBoxCord box;

        if (boundBox_size != 0)
        {
            for (int i = 0; i < boundBox_size; i++)
            {
                caffe_protos::Annotation annot_protos = annotGrp_protos.annotation(i);
                caffe_protos::NormalizedBBox bbox_protos = annot_protos.bbox();

                // Parsing the bounding box points using Iterator
                box.x = bbox_protos.xmin();
                box.y = bbox_protos.ymin();
                box.w = bbox_protos.xmax();
                box.h = bbox_protos.ymax();

                int label = bbox_protos.label();
                
                bb_coords.push_back(box);
                bb_labels.push_back(label);
                add(file_name.c_str(), bb_coords, bb_labels);
                bb_coords.clear();
                bb_labels.clear();
            }
        }
        else
        {
            box.x = box.y = box.w = box.h = 0;
            bb_coords.push_back(box);
            bb_labels.push_back(0);
            add(file_name.c_str(), bb_coords, bb_labels);
        }
    }

    // Closing all the LMDB environment and cursor handles
    mdb_cursor_close(mdb_cursor);
    mdb_close(mdb_env, mdb_dbi);
    mdb_txn_abort(mdb_txn);
    mdb_env_close(mdb_env);
}

void CaffeMetaDataReaderDetection::release(std::string _image_name)
{
    if (!exists(_image_name))
    {
        WRN("ERROR: Given not present in the map" + _image_name);
        return;
    }
    _map_content.erase(_image_name);
}

void CaffeMetaDataReaderDetection::release()
{
    _map_content.clear();
}

CaffeMetaDataReaderDetection::CaffeMetaDataReaderDetection()
{
}
