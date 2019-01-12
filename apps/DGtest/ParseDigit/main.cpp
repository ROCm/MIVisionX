#include "ParseDigit.h"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	if (argc < 3) {
		cout << "Usage: ./ParseDigit [input_image] [directory_name] [image_name] [-a]" << endl;
		return -1;
	}

	string inputName = argv[1];
    string dirName = argv[2];
	string targetName = argv[3];
    bool automate = false;
    if (argc == 5) {
        string option = argv[4];
        if (option == "-a") {
            automate = true;
        }
    }
    ParseDigit parse(inputName, dirName, targetName, automate);
	parse.run();
	return 0;
}
