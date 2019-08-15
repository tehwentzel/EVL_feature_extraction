#include "BagOfWords.h"
#include "Constants.h"
#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>

using namespace std;

BagOfWords::BagOfWords(std::vector<cv::Mat> points) {
	inputData = BagOfWords::hstackMats(points);
	cout << inputData.size() << endl;
	cvflann::KMeansIndexParams params(32, 11, cvflann::CENTERS_KMEANSPP);
	cv::Mat centers(DSIFT_TOTAL_CLUSTERS, inputData.size().width, CV_32F);
	cout << "clustering..." << endl;
	cv::flann::hierarchicalClustering<cv::flann::L2<float>>(inputData, centers, params);
	cout << "flann clustering done" << endl;
	wordIndex = (centers, params, cvflann::FLANN_DIST_EUCLIDEAN);
};

std::vector<int> BagOfWords::getWordCounts(cv::Mat) {
	return std::vector<int>();
};

cv::Mat BagOfWords::hstackMats(std::vector<cv::Mat> data) {
	cv::Mat imageSet;
	while (data.size() != 0) {
		imageSet.push_back(data.back());
		data.pop_back();
	}
	return imageSet;
}