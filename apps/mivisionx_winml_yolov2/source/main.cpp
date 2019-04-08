#include "AnnieYoloDetect.h"
#include "Visualize.h"

using namespace std;
using namespace cv;

void printUsage() {
	cout << "Usage: " << endl;
	cout << "  * Image" << endl;
	cout << "       .\MIVisionX_winml_YoloV2.exe --image [image - required] --modelLoc [modelLocation - required] --confidence [1-100 - optional, default: 20]" << endl;
	cout << "  * Camera Capture" << endl;
	cout << "       .\MIVisionX_winml_YoloV2.exe --capture [0 - required] --modelLoc [modelLocation - required] --confidence [1-100 - optional, default: 20]" << endl;
	cout << "  * Video" << endl;
	cout << "       .\MIVisionX_winml_YoloV2.exe --video [video - required] --modelLoc [modelLocation - required] --confidence [1-100 - optional, default: 20]" << endl;
}

void convertBackSlash(string &modelLoc) {
	for (int i = 0; i < modelLoc.size();i++)
	{
		if (modelLoc[i] == '\\')
		{
			modelLoc.insert(i, "\\");
			i++;
		}
	}
}

int main(int argc, char ** argv) {
	
	if (argc < 5 || argc > 7) {
		printUsage();
		return -1;
	}
	
	string option = argv[1];
	string input = argv[2];
	if (strcmp(argv[3], "--modelLoc") != 0) {
		printUsage();
		return -1;
	}

	string modelLoc = argv[4];
	convertBackSlash(modelLoc);
	int confidence;
	if (argc > 6) {
		if (strcmp(argv[5], "--confidence") == 0) {
			confidence = atoi(argv[6]);
		}
		else {
			printUsage();
			return -1;
		}
	}
	else
		confidence = 20;

	//option mode
	int mode;

	unique_ptr<AnnieYoloDetect> Annie;

	if (option == "--image") {
		mode = 0;
		Annie = make_unique<AnnieYoloDetect>(input, modelLoc, confidence, mode);
		Annie->detect();
	}
	else if (option == "--capture") {
		mode = 1;
		Annie = make_unique<AnnieYoloDetect>(input, modelLoc, confidence, mode);
		Annie->detect();
	}
	else if (option == "--video") {
		mode = 2;
		Annie = make_unique<AnnieYoloDetect>(input, modelLoc, confidence, mode);
		Annie->detect();
	}
	else {
		printUsage();
		return -1;
	}
	return 0;
}
