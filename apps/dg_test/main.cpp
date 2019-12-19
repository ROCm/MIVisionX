#include "UserInterface.h"

int main(int argc, const char ** argv)
{
    // check command-line usage
    if(argc != 2) {
        printf(
            "\n"
            "Usage: ./DGtest [weights.bin]\n"
            "\n"
            "   <weights.bin>: name of the weights file to be used for the inference\n."
            "\n"
        );
        return -1;
    }
    
    const char* weights = argv[1];
    UserInterface UI(weights);
    UI.startUI();

    return 0;
}