#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include "json.hpp"
#include <map>
#include "Constants.h"
#include "Preprocess.h"
#include "BagOfWords.h"
#include "FuzzyFeatures.h"

using namespace std;
using json = nlohmann::json;

map<cv::String, cv::Mat> createImageMap(cv::String parentPath = "D:\\git_repos\\EVL_feature_extraction\\src\\data\\images\\",
	bool denoise = false);
void showImages( map<cv::String,cv::Mat> );
void showImages(vector<cv::Mat>);
vector<cv::Mat> matDictValues(map<cv::String, cv::Mat>);
vector<string> stringDictKeys(map<cv::String, cv::Mat>);
vector<cv::Mat> getDSIFTs(map<cv::String, cv::Mat>);
vector<cv::Mat> getDSIFTs(vector<cv::Mat>);
cv::Mat getRootDSIFT(cv::Mat);
cv::Mat rootKernel(cv::Mat);
json extractFeatures(map<cv::String, cv::Mat>, BagOfWords&);
json getSingleImageJson(cv::Mat&, BagOfWords&);
cv::Mat getSingleDSIFT(cv::Mat&);
void saveJson(json, std::string = FEATURE_JSON_FILE);

int main(int argc, char** argv)
{
	cv::String parentPath = "D:\\git_repos\\EVL_feature_extraction\\src\\data\\images\\Experimental\\Microscopy\\";
	auto imageMap = createImageMap(parentPath);
	BagOfWords codeBook;
	if (std::ifstream(BOVW_FILE + std::to_string(DSIFT_TOTAL_CLUSTERS) + ".txt")) {
		codeBook = BagOfWords(BOVW_FILE + std::to_string(DSIFT_TOTAL_CLUSTERS) + ".txt");
	}
	else {
		cout << "creating new bagofwords" << endl;
		auto dSIFTs = getDSIFTs(imageMap);
		codeBook = BagOfWords(dSIFTs);
	}
	//showImages(imageMap);
	json data = extractFeatures(imageMap, codeBook);
	cout <<"BOW Done" << endl;
	saveJson(data);
	//cv::FileStorage fs(FEATURE_DICT_FILE + std::to_string(DSIFT_TOTAL_CLUSTERS) + ".txt", cv::FileStorage::WRITE);
	//fs << "fileNames" << stringDictKeys(imageMap) << "siftBOVW" << bowFeatures;
	//fs.release();
}

void saveJson(json toWrite, std::string fileName) {
	ofstream file;
	file.open(fileName);
	file << toWrite.dump(4) << endl;
	file.close();
	cout << "file successfully saved to " << fileName;
}

json extractFeatures(map<cv::String, cv::Mat> imageMap, BagOfWords& codeBook) {
	json imageJson;
	for (auto& item : imageMap) {
		json newImage = getSingleImageJson(item.second, codeBook);
		imageJson[item.first] = newImage;
	}
	return imageJson;
}

json getSingleImageJson(cv::Mat& image, BagOfWords& codeBook) {
	json features;
	auto dsift = getSingleDSIFT(image);
	features["BOVW"] = codeBook.getWordCounts(dsift);
	features["CrispHistograms"] = standardColorSpaceHistograms(image);
	return features;
}

vector<cv::Mat> getDSIFTs(map<cv::String, cv::Mat> imageMap) {
	auto images = matDictValues(imageMap);
	return getDSIFTs(images);
}

cv::Mat getSingleDSIFT(cv::Mat& image) {
	cv::Mat centers(DSIFT_QUANTIZE_SIZE, 128, CV_32F);
	cv::Mat labels(900 * DSIFT_STEP, 1, CV_32F);  //doesnt' do anything
	auto descriptors = getRootDSIFT(image);
	cv::kmeans(descriptors, DSIFT_QUANTIZE_SIZE, labels,
		//for this the last term is eps and 2nd term is max_itter
		cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, .00001),
		3, cv::KMEANS_PP_CENTERS, centers);
	return centers;
}

vector<cv::Mat> getDSIFTs(vector<cv::Mat> images) {
	vector<cv::Mat> features;
	for (auto& image : images) {
		auto centers = getSingleDSIFT(image);
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
			preprocessImage(image, denoise);
			images.emplace(imageFile, image);
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

vector<string> stringDictKeys(map<cv::String, cv::Mat> matMap) {
	vector<string> files;
	for (auto it = matMap.begin(); it != matMap.end(); it++) {
		files.push_back(it->first);
	}
	return files;
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
