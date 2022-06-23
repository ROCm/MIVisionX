#ifndef COMPILER_H
#define COMPILER_H

#include "arguments.h"
#include "infcom.h"
#include <string>

int runCompiler(int sock, Arguments * args, std::string& clientName, InfComCommand * cmd);

#endif
