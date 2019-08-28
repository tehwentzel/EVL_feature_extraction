#include "BagOfWords.h"
#include "Constants.h"
#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
using namespace std;

BagOfWords::BagOfWords(vector<cv::Mat> points) {
	
	cv::Mat inputData = BagOfWords::hstackMats(points);
	cout << inputData.size() << endl;
	cvflann::KMeansIndexParams params(32, 11, cvflann::CENTERS_KMEANSPP);
	cv::Mat centers(DSIFT_TOTAL_CLUSTERS, inputData.size().width, CV_32F);
	cout << "clustering..." << endl;
	cv::flann::hierarchicalClustering<cv::flann::L2<float>>(inputData, centers, params);
	initIndex(centers);

	cout << "flann clustering done" << endl;
	saveData(inputData, centers);
	cout << "data saved" << endl;
};

BagOfWords::BagOfWords(string dataFile) {
	cv::FileStorage fs(dataFile, cv::FileStorage::READ);
	cv::Mat allpoints;
	cv::Mat centers;
	fs["centers"] >> centers;
	initIndex(centers);
	fs.release();
}

void BagOfWords::initIndex(cv::Mat& centers) {
	vector<vector<double>> points;
	vector<double> currentRow(DSIFT_TOTAL_CLUSTERS);
	cout << "num rows" << centers.rows << endl;
	for (int rowIdx = 0; rowIdx < centers.rows; rowIdx++) {
		centers.row(rowIdx).copyTo(currentRow);
		points.push_back(currentRow);
	}
	wordIndex.reset(new KDTree(points));
}

bool BagOfWords::saveData(cv::Mat allpoints, cv::Mat centers, string dataFile) {
	bool success = true;
	try {
		cv::FileStorage fs(dataFile+std::to_string(DSIFT_TOTAL_CLUSTERS)+".txt", cv::FileStorage::WRITE);
		fs << "centers" << centers << "allpoints" << allpoints;
		fs.release();
	}
	catch (exception& e) {
		success = false;
		cout << e.what();
	}
	return success;
}

vector<int> BagOfWords::getWordCounts(cv::Mat& queryFeatures) {
	//cout << "shape " << queryFeatures.size << endl;
	vector<int> wordCounts(DSIFT_TOTAL_CLUSTERS);
	vector<double> currentWord;
	int currentWordIndex;
	for (int i = 0; i < queryFeatures.rows; i++) {
		queryFeatures.row(i).copyTo(currentWord);
		currentWordIndex = wordIndex->nearest_index(currentWord);
		//cout << currentWordIndex << " ";
		wordCounts.at(currentWordIndex)++;
	}
	//cout << endl;
	return wordCounts;
}

cv::Mat BagOfWords::hstackMats(std::vector<cv::Mat> data) {
	cv::Mat imageSet;
	while (data.size() != 0) {
		imageSet.push_back(data.back());
		data.pop_back();
	}
	return imageSet;
}