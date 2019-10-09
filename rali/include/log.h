#pragma once
#include <iostream>
#define LOG(X) std::clog << "[INF] "  << " {" << __func__ <<"} " << " " << X << std::endl;
#define ERR(X) std::clog << "[ERR] "  << " {" << __func__ <<"} " << " " << X << std::endl;
#define WRN(X) std::clog << "[WRN] "  << " {" << __func__ <<"} " << " " << X << std::endl;