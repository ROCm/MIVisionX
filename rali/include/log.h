#pragma once
#define THROW(X) throw RaliException(" { "+std::string(__func__)+" } " + X);
#define TOSTR(X) std::to_string(static_cast<int>(X))
#define STR(X) std::string(X)

#include <iostream>
#define LOG(X) std::clog << "[INF] "  << " {" << __func__ <<"} " << " " << X << std::endl;
#define ERR(X) std::clog << "[ERR] "  << " {" << __func__ <<"} " << " " << X << std::endl;
#define WRN(X) std::clog << "[WRN] "  << " {" << __func__ <<"} " << " " << X << std::endl;