#ifndef SHADOW_H
#define SHADOW_H

#include "infcom.h"
#include "arguments.h"
#include <string>

int runShadow(int sock, Arguments * args, std::string& clientName, InfComCommand * cmd);

#endif // SHADOW_H

