#pragma once

#define TOSTR(X) std::to_string(static_cast<int>(X))
#define STR(X) std::string(X)

#include <iostream>
#define INFO(X) std::clog << "[INF] "  << " {" << __func__ <<"} " << " " << X << std::endl;
#if DBGLOG
#define LOG(X) std::clog << "[LOG] "  << " {" << __func__ <<"} " << " " << X << std::endl;
#else
#define LOG(X) ;
#endif
#define ERR(X) std::cerr << "[ERR] "  << " {" << __func__ <<"} " << " " << X << std::endl;
#define WRN(X) std::clog << "[WRN] "  << " {" << __func__ <<"} " << " " << X << std::endl;