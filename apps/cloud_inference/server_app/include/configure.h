#ifndef CONFIGURE_H
#define CONFIGURE_H

#include "infcom.h"
#include "arguments.h"
#include <string>

int runConfigure(int sock, Arguments * args, std::string& clientName, InfComCommand * cmd);

#endif
