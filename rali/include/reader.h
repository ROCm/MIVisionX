#pragma once

enum class StorageType
{
    FILE_SYSTEM = 0,
    RECORDIO,
    TFRecord
};

struct ReaderConfig {
    virtual StorageType type() = 0;
};

class Reader {
public:
    enum class Status
    {
        OK = 0
    };

    // TODO: change method names to open_next, read_next , ...


    //! Initializes the resource which it's spec is defined by the desc argument
    /*!
     \param desc description of the resource infor. It's exact fields are defind by the derived class.
     \return status of the being able to locate the resource pointed to by the desc
    */
    virtual Status initialize(ReaderConfig* desc) = 0;
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded items	
     \return Size of the loaded resource
    */
 
       //! Opens the next item and returns it's size
    /*!
     \return Size of the item, if 0 failed to access it
    */
    virtual size_t open() = 0;
    
    //! Copies the data of the opened item to the buf
    virtual size_t read(unsigned char* buf, size_t read_size) = 0;

    //! Closes the opened item 
    virtual int close() = 0;

    //! Starts reading from the first item in the resource
    virtual void reset() = 0;

    //! Returns the name/identifier of the last item opened in this resource
    virtual std::string id() = 0;
    //! Returns the number of items remained in this resource
    virtual unsigned count() = 0;

    virtual ~Reader() = default;

};