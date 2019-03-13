#include "AnnieYoloDetect.h"
#include "Visualize.h"

using namespace std;
using namespace cv;

int main(int argc, char ** argv) {
	
	//usage
	if (argc < 5) {
		cout << "Usage: MIVisionX_winml_YoloV2.exe --image [image] --modelLoc [modelLocation]" << endl;
		cout << "       MIVisionX_winml_YoloV2.exe --capture 0     --modelLoc [modelLocation](Live Capture)" << endl;
		cout << "       MIVisionX_winml_YoloV2.exe --video [video] --modelLoc [modelLocation]" << endl;
		return -1;
	}

	string option = argv[1];
	string input = argv[2];
	string modelLoc = argv[4];
	
	//option mode
	int mode;

	unique_ptr<AnnieYoloDetect> Annie;

	if (option == "--image") {
		mode = 0;
		Annie = make_unique<AnnieYoloDetect>(input, modelLoc, mode);
		Annie->detect();
	}
	else if (option == "--capture") {
		mode = 1;
		Annie = make_unique<AnnieYoloDetect>(input, modelLoc, mode);
		Annie->detect();
	}
	else if (option == "--video") {
		mode = 2;
		Annie = make_unique<AnnieYoloDetect>(input, modelLoc, mode);
		Annie->detect();
	}
	else {
		cout << "Usage: MIVisionX_winml_YoloV2.exe --image [image] --modelLoc [modelLocation]" << endl;
		cout << "       MIVisionX_winml_YoloV2.exe --capture 0     --modelLoc [modelLocation](Live Capture)" << endl;
		cout << "       MIVisionX_winml_YoloV2.exe --video [video] --modelLoc [modelLocation]" << endl;
		return -1;
	}
	return 0;
}
