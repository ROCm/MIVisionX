#pragma once
#include <vector>
#include <string>
#include <memory>
#include "reader.h"


class TFRecordReader : public Reader
        {
public:
    //! Reads the TFRecord File, and loads the image ids and other necessary info
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    Reader::Status initialize(ReaderConfig desc) override { return Reader::Status::OK; }
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded images
     \return Size of the loaded resource
    */
    size_t read(unsigned char* buf, size_t max_size) override { return 0; }
    //! Opens the next file in the folder
    /*!
     \return The size of the next file, 0 if couldn't access it
    */
    size_t open() override { return 0; }

    //! Resets the object's state to read from the first file in the folder
    void reset() override { return ; }

    //! Returns the id of the latest file opened
    std::string id() override { return "0";}

    unsigned count() override { return 0; }

    ~TFRecordReader() override = default;

    int close() override{ return 0; };

    TFRecordReader()= default;

};