#include "common.h"
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

void info(const char * format, ...)
{
    printf("INFO: ");
    va_list args;
    va_start(args, format);
    int r = vprintf(format, args);
    va_end(args);
    printf("\n");
}

void warning(const char * format, ...)
{
    printf("WARNING: ");
    va_list args;
    va_start(args, format);
    int r = vprintf(format, args);
    va_end(args);
    printf("\n");
}

int error(const char * format, ...)
{
    printf("ERROR: ");
    va_list args;
    va_start(args, format);
    int r = vprintf(format, args);
    va_end(args);
    printf("\n");
    return -1;
}

void fatal(const char * format, ...)
{
    printf("FATAL: ");
    va_list args;
    va_start(args, format);
    int r = vprintf(format, args);
    va_end(args);
    printf("\n");
    exit(1);
}
