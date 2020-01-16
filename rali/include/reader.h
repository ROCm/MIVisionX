#pragma once

enum class StorageType
{
    FILE_SYSTEM = 0,
    RECORDIO,
    TFRecord
};

struct ReaderConfig {
    explicit ReaderConfig(StorageType type, std::string path = "", bool loop = false):_type(type), _path(path), _loop(loop) {}
    virtual StorageType type() { return _type; };
    void set_path(const std::string& path) { _path = path; }
    void set_load_offset(size_t offset) { _read_offset = offset; }
    void set_load_interval(size_t interval) { _read_interval = interval; }
    void set_loop( bool loop) { _loop = loop; }
    bool loop() { return _loop; }
    size_t interval() { return _read_interval; }
    size_t offset() { return _read_offset; }
    std::string path() { return _path; }
private:
    StorageType _type = StorageType::FILE_SYSTEM;
    std::string _path = "";
    size_t _read_interval= 1 ;
    size_t _read_offset = 0;
    bool _loop = false;
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
    virtual Status initialize(ReaderConfig desc) = 0;
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