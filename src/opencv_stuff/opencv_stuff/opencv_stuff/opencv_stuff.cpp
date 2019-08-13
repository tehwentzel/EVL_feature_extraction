#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <iostream>
#include <string>
#include <map>
#include "Constants.h"

using namespace std;

map<cv::String, cv::Mat> createImageMap(cv::String parentPath = "D:\\git_repos\\EVL_feature_extraction\\src\\data\\images\\");
cv::Mat preprocessImage(cv::Mat);
void showImages( map<cv::String,cv::Mat> );
void showImages(vector<cv::Mat>);
cv::Mat resizeImage(cv::Mat);
cv::Mat preprocessImage(cv::Mat);

int main(int argc, char** argv)
{
	cv::String parentPath = "D:\\git_repos\\EVL_feature_extraction\\src\\data\\images\\";
	auto imageMap = createImageMap();
	showImages(imageMap);
}

void showImages(map<cv::String, cv::Mat> imageMap) {
	vector<cv::Mat> images;
	for (auto it = imageMap.begin(); it != imageMap.end(); it++) {
		images.push_back(it->second);
	}
	showImages(images);
}

void onTrack(int position, void* data) {
	vector<cv::Mat>& d = *reinterpret_cast<vector<cv::Mat>*>(data);
	if (position < d.size()) {
		cv::imshow("images", d[position]);
	}
}

void showImages(vector<cv::Mat> images) {
	int position = 0;
	int maxPosition = images.size();
	const cv::String windowName = "images";
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
	cout << windowName << endl;
	cv::createTrackbar(windowName, windowName, &position, 
		maxPosition, onTrack, (void*)&images);
	cv::imshow(windowName, images[0]);
	cv::waitKey(0);
}

map<cv::String, cv::Mat> createImageMap(cv::String parentPath) {
	vector<cv::String> filenames;
	map<cv::String, cv::Mat> images;
	cv::glob(parentPath + "*.jpg",
		filenames, true);
	cv::Mat image;
	for (auto imageFile : filenames) {
		image = cv::imread(imageFile);
		if (!image.empty()) {
			images.emplace(imageFile, preprocessImage(image));
		}
		else {
			cout << imageFile << endl;
		}
	}
	return images;
}

cv::Mat preprocessImage(cv::Mat image) {
	cv::Mat newImg;
	newImg = resizeImage(image);
	cv::fastNlMeansDenoisingColored(newImg, newImg);
	return newImg;
}

cv::Mat resizeImage(cv::Mat image) {
	cv::Mat newImg;
	cv::Size s = image.size();
	double wRatio = (float)imWidth / (float)s.width;
	double hRatio = (float)imHeight / (float)s.height;
	double ratio = (wRatio > hRatio) ? wRatio : hRatio;
	cv::resize(image, newImg, cv::Size(), ratio, ratio, cv::INTER_AREA);
	return newImg;
}