#include "arguments.h"
#include "server.h"

int main(int argc, char * argv[])
{
    // get command-line arguments
    Arguments * args = new Arguments();
    if(args->initializeConfig(argc, argv) < 0)
        return -1;

    // run the server
    if(server(args) < 0)
        return -1;

    return 0;
}
