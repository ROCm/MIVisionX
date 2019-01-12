#include "ParseDigit.h"
#include <iostream>

using namespace std;
using namespace cv;

/*
 * Check if the input is valid number
 */ 
bool isValidNumber (string s) {
	if (!isdigit(s[0])) {
		if (s[0] != '-') return false;
	}
	for (int i = 1; i < s.length(); i++) {
		if (!isdigit(s[i])) return false;
	}

    return true;
}

ParseDigit::ParseDigit(const std::string imageName, const std::string dirName, const std::string targetName, bool automate)
 : mImageName(imageName), mDirName(dirName), mTargetName(targetName), mAutomate(automate) {

	mOrigImage = imread(mImageName);
	if (mOrigImage.empty()) {
		cout << "Invalid image: " << mImageName << endl;
		exit(-1);
	}

	mGrayImage = imread(mImageName, IMREAD_GRAYSCALE);
	mDetector = make_unique<MSERDetector>(30);
}

ParseDigit::~ParseDigit() {
}

vector<Rect> ParseDigit::detectMSER() {
	vector<vector<cv::Point> > regions;
	vector<cv::Rect> mser_bbox;

	cout << "Parsing...";

	//detect Maximally Stable Extremal Region on the given image
	mDetector->detect(mGrayImage, regions, mser_bbox);

    //check for overlapping rectangles and merge them
    for (auto i = mser_bbox.begin(); i != mser_bbox.end(); i++) {
			for (auto j = i+1; j != mser_bbox.end();) {
                if ((*i & *j).area() > 0) {
                    *i = *i | *j;
                    mser_bbox.erase(j);
					j = i+1;
                }
                else {
                    j++;
                }	
            }
	}

	//check for rectangles that are close and merge them 
	int minLength = min(mOrigImage.cols, mOrigImage.rows);
	for (auto i = mser_bbox.begin(); i!= mser_bbox.end(); i++) {
		for (auto j = i+1; j != mser_bbox.end();) {
			Rect rect1 = *i;
			Rect rect2 = *j;

			Point p(rect1.x + rect1.width/2, rect1.y + rect1.height/2);
			Point q(rect2.x + rect2.width/2, rect2.y + rect2.height/2);
			Point diff = p - q;

			if (sqrt(diff.x*diff.x + diff.y*diff.y) < minLength/10) {
				*i = *i | *j;
				mser_bbox.erase(j);
				j = i+1;
			}
			else {
				j++;
			}
		}
	}

	// //show image with rectangles
	// Mat temp = mOrigImage.clone();
    // for (auto& r : mser_bbox) {
    //     rectangle(temp, r, CV_RGB(255, 255, 0));
    // }
	cout << "done." << endl;
	cout << "Total " << mser_bbox.size() << " digits found. " << endl << endl;
	return mser_bbox;
}

unordered_map<Rect, string, CustomHash> ParseDigit::cropImg(vector<Rect> &mser_bbox) {
	unordered_map<cv::Rect, std::string, CustomHash> rectMap;
	unordered_map<int, int> count;

	Mat crop;
	int digit;
	int imgCount = 1;
	string name;

	if (mAutomate) {
		//automatically name images without verification
		for (auto itr = mser_bbox.begin(); itr != mser_bbox.end(); itr++) {
			name = mDirName;
			name += "/";
			name += mTargetName;
			name += "00";
			name += to_string(imgCount++);
			name += ".jpg";
			rectMap[*itr] = name;
		}
	}
	else {
		//verify each images
		string input = "";
		for (auto itr = mser_bbox.begin(); itr != mser_bbox.end(); itr++) {
			namedWindow("Digit");
			
			cout << "[" << imgCount << "/" << mser_bbox.size() << ']' << endl;
			cout << "What digit is it? (Put -1 for non-digits) : ";
			
			crop = mOrigImage(*itr);
			imshow("Digit", crop);
			waitKey(100);
			
			getline(cin, input);
			stringstream inputDigit(input);
			inputDigit >> digit;

			// check for invalid input
			while (!isValidNumber(input) || digit < -1 || digit > 9) {
				cout << "Please enter a valid digit (0-9) : ";
				getline(cin, input);
				stringstream inputDigit(input);
				inputDigit >> digit;
				cout << "input is : " << input << "+" << digit << endl;
			}

			if (digit == -1) {
				destroyWindow("Digit");
			}
			else {
				name = mDirName;
				name += "/";
				name += mTargetName;
				name += "00";
				name += to_string(digit);
				name += "-";
				name += to_string(++count[digit]);
				name += ".jpg";
				cout << " the name is " << name << endl;
				rectMap[*itr] = name;
				cout << "Saved as: " << name << endl;
				destroyWindow("Digit");
			}
			imgCount++;
			cout << endl;
		}
	}
	
	return rectMap;
}

void ParseDigit::postProcessImg(unordered_map<Rect, string, CustomHash> &rectMap) {
	cout << "Post processing...";
	Mat crop;
	Mat m = Mat::ones(2,2,CV_8U); //kernel for dilation
	int count = 0; //count for filename
	
	for (auto itr = rectMap.begin(); itr != rectMap.end(); itr++) {
	
		//crop the image
		crop = mGrayImage(itr->first);
	
		//invert the color and resize it to 20x20
		resize(255-crop, crop, Size(20, 20));
	
		//convert to black/white image
		threshold(crop, crop, 0, 255, THRESH_OTSU);
	
		//dilate image
		dilate(crop, crop, m); 
	
		//add padding to the image so that the digits will be in the center
		copyMakeBorder(crop, crop, 4, 4, 4, 4, BORDER_CONSTANT, Scalar(0,0,0));
		
		imwrite(itr->second, crop); 
		count++;
	}

	cout << "done" << endl;
}

void ParseDigit::run() {
	vector<Rect> mser_bbox = detectMSER();
	unordered_map<Rect, string, CustomHash> rectMap = cropImg(mser_bbox);
	postProcessImg(rectMap);
}
