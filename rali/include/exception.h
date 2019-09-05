#pragma once
#include <stdexcept>
#include <exception>
#include <string>

class RaliException : public std::exception
{
public:

    explicit RaliException(const std::string& message):_message(message)
    {}
    virtual const char* what() const throw() override
    {
        return _message.c_str();
    }
private:
    std::string _message;
};




