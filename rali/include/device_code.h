#pragma once

#include <vector>
#include <string>



class DeviceCode {
public: 
    explicit DeviceCode(const std::string& source_code, const std::string& program_name, const std::vector<std::string>& kernel_list ):
    m_source_code(source_code), m_prog_name(program_name), m_kernel_list(kernel_list) {}
    const std::string& getSourceCode() const { return m_source_code; }
    const std::string& getName() const { return m_prog_name; }
    const std::vector<std::string>& getKernelList() const { return m_kernel_list; }
private:
    const std::string m_source_code;
    const std::string m_prog_name;
    const std::vector<std::string> m_kernel_list; 
};
