#pragma once
#include <vector>
#include <string>
#include <memory>
#include <dirent.h>
#include <map>
#include <iterator>
#include <algorithm>
#include "reader.h"
#include <google/protobuf/message_lite.h>
#include "example.pb.h"
#include "feature.pb.h"

class TFRecordReader : public Reader
        {
public:
    //! Reads the TFRecord File, and loads the image ids and other necessary info
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    Reader::Status initialize(ReaderConfig desc) override;
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded images
     \return Size of the loaded resource
    */
    size_t read(unsigned char* buf, size_t max_size) override;
    //! Opens the next file in the folder
    /*!
     \return The size of the next file, 0 if couldn't access it
    */
    size_t open() override;
    //! Resets the object's state to read from the first file in the folder
    void reset() override;

    //! Returns the id of the latest file opened
    std::string id() override { return _last_id;};

    unsigned count() override;

    ~TFRecordReader() override;

    int close() override;

    TFRecordReader();
private:
    //! opens the folder containnig the images
    Reader::Status tf_record_reader();
    Reader::Status folder_reading();
    std::string _folder_path;
    std::string _path;
    DIR *_src_dir;
    DIR *_sub_dir;
    struct dirent *_entity;
    std::vector<std::string> _file_names;
    std::vector<unsigned int> _file_size;
    unsigned  _curr_file_idx;
    unsigned _current_file_size;
    std::string _last_id;
    std::string _last_file_name;
    unsigned int _last_file_size;
    size_t _shard_id = 0;
    size_t _shard_count = 1;// equivalent of batch size
    bool _last_rec;
    //!< _batch_count Defines the quantum count of the images to be read. It's usually equal to the user's batch size.
    /// The loader will repeat images if necessary to be able to have images available in multiples of the load_batch_count,
    /// for instance if there are 10 images in the dataset and _batch_count is 3, the loader repeats 2 images as if there are 12 images available.
    size_t _batch_count = 1;
    size_t _file_id = 0;
    size_t _in_batch_read_count = 0;
    bool _loop;
    bool _shuffle;
    int _read_counter = 0;
    // protobuf message objects
    tensorflow::Example _single_example;
    tensorflow::Features _features;
    tensorflow::Feature _single_feature;
    void incremenet_read_ptr();
    int release();
    size_t get_file_shard_id();
    void incremenet_file_id() { _file_id++; }
    void replicate_last_image_to_fill_last_shard();
    void read_image(unsigned char* buff, std::string record_file_name, uint file_size);
    void read_image_names(std::ifstream &file_contents, uint file_size);
    std::map <std::string, uint> _image_record_starting;
};

