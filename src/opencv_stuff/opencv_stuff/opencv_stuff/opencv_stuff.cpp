#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <string>
#include <map>
#include "Constants.h"
#include "Preprocess.h"
#include "BagOfWords.h"

using namespace std;

map<cv::String, cv::Mat> createImageMap(cv::String parentPath = "D:\\git_repos\\EVL_feature_extraction\\src\\data\\images\\",
	bool denoise = false);
void showImages( map<cv::String,cv::Mat> );
void showImages(vector<cv::Mat>);
vector<cv::Mat> matDictValues(map<cv::String, cv::Mat>);
vector<cv::Mat> getDSIFTs(map<cv::String, cv::Mat>);
vector<cv::Mat> getDSIFTs(vector<cv::Mat>);
cv::Mat getRootDSIFT(cv::Mat);
cv::Mat rootKernel(cv::Mat);

int main(int argc, char** argv)
{
	cv::String parentPath = "D:\\git_repos\\EVL_feature_extraction\\src\\data\\images\\";
	auto imageMap = createImageMap(parentPath);
	//showImages(imageMap);
	auto dSiftFeatures = getDSIFTs(imageMap);
	auto codeBook = BagOfWords(dSiftFeatures);
}

vector<cv::Mat> getDSIFTs(map<cv::String, cv::Mat> imageMap) {
	auto images = matDictValues(imageMap);
	cout << "getting dsifts..." << endl;
	return getDSIFTs(images);
}

vector<cv::Mat> getDSIFTs(vector<cv::Mat> images) {
	vector<cv::Mat> features;
	cv::Mat labels(900*DSIFT_STEP, 1, CV_32F);
	cv::Mat centers(DSIFT_QUANTIZE_SIZE, 128, CV_32F);
	for (auto& image : images) {
		auto descriptors = getRootDSIFT(image);
		cv::kmeans(descriptors, DSIFT_QUANTIZE_SIZE, labels,
			//for this the last term is eps and 2nd term is max_itter
			cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 600, .00001),
			3, cv::KMEANS_PP_CENTERS, centers);
		features.push_back(centers);
	}
	return features;
}

cv::Mat getRootDSIFT(cv::Mat image) {
	cv::Mat features;
	cv::Ptr< cv::xfeatures2d::SiftDescriptorExtractor> sift;
	try {
		sift = cv::xfeatures2d::SiftDescriptorExtractor::create();
	}
	catch (cv::Exception& e) {
		const char* err_msg = e.what();
		cout << err_msg << endl;
	}
	vector<cv::KeyPoint> keypoints;
	//sift->detect(image, keypoints);
	for (int row = DSIFT_STEP; row < image.rows - DSIFT_STEP; row += DSIFT_STEP) {
		for (int col = DSIFT_STEP; col < image.cols - DSIFT_STEP; col += DSIFT_STEP) {
			keypoints.push_back(cv::KeyPoint(float(row), float(col), float(DSIFT_STEP)));
		}
	}
	sift->compute(image, keypoints, features);
	return rootKernel(features);
}

cv::Mat rootKernel(cv::Mat src) {
	//normalizes rows and computes featurewise sqrt, for doing rootSift instead of sift
	for (int r = 0; r < src.rows; ++r) {
		if(cv::norm(src.row(r), cv::NORM_L1) > 0);
		cv::normalize(src.row(r), src.row(r), 1, cv::NORM_L1);
	}
	cv::sqrt(src, src);
	return src;
}

map<cv::String, cv::Mat> createImageMap(cv::String parentPath, bool denoise) {
	vector<cv::String> filenames;
	map<cv::String, cv::Mat> images;
	cv::glob(parentPath + "*.jpg",
		filenames, true);
	cv::Mat image;
	for (auto& imageFile : filenames) {
		image = cv::imread(imageFile);
		if (!image.empty()) {
			images.emplace(imageFile, preprocessImage(image, denoise));
		}
		else {
			cout << imageFile << endl;
		}
	}
	return images;
}

vector<cv::Mat> matDictValues(map<cv::String, cv::Mat> matMap) {
	vector<cv::Mat> mats;
	for (auto it = matMap.begin(); it != matMap.end(); it++) {
		mats.push_back(it->second);
	}
	return mats;
}

void showImages(map<cv::String, cv::Mat> imageMap) {
	auto images = matDictValues(imageMap);
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
