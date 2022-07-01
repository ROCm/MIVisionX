#ifndef NETUTIL_H
#define NETUTIL_H

#include "infcom.h"
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <string>

int sendBuffer(int sock, const void * buf, size_t len, std::string& clientName);
int recvBuffer(int sock,       void * buf, size_t len, std::string& clientName);

int sendCommand(int sock, const InfComCommand& cmd, std::string& clientName);
int recvCommand(int sock,       InfComCommand& cmd, std::string& clientName, int expectedCommand);

void dumpCommand(const char * info, const InfComCommand& cmd);

int error_close(int sock, const char * format, ...);

#endif
