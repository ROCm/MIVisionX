#include "AnnieObjectWrapper.h"
#include "Visualize.h"

using namespace std;
using namespace cv;

int main(int argc, char ** argv) {
	
	//usage
	if (argc < 5) {
		cout << "Usage: YoloV2NCS.exe --image [image] --modelLoc [modelLocation]" << endl;
		cout << "       YoloV2NCS.exe --capture 0     --modelLoc [modelLocation](Live Capture)" << endl;
		cout << "       YoloV2NCS.exe --video [video] --modelLoc [modelLocation](Live Capture)" << endl;
		return -1;
	}

	string option = argv[1];
	string input = argv[2];
	string modelLoc = argv[4];
	
	//option mode
	int mode;

	unique_ptr<AnnieObjectWrapper> Annie;

	if (option == "--image") {
		mode = 0;
		Annie = make_unique<AnnieObjectWrapper>(input, modelLoc, mode);
		Annie->detect();
	}
	else if (option == "--capture") {
		mode = 1;
		Annie = make_unique<AnnieObjectWrapper>(input, modelLoc, mode);
		Annie->detect();
	}
	else if (option == "--video") {
		mode = 2;
		Annie = make_unique<AnnieObjectWrapper>(input, modelLoc, mode);
		Annie->detect();
	}
	else {
		cout << "Usage: YoloV2NCS.exe --image [image] --modelLoc [modelLocation]" << endl;
		cout << "       YoloV2NCS.exe --capture 0     --modelLoc [modelLocation](Live Capture)" << endl;
		cout << "       YoloV2NCS.exe --video [video] --modelLoc [modelLocation](Live Capture)" << endl;
		return -1;
	}
	return 0;
}
