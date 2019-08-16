#include "BagOfWords.h"
#include "Constants.h"
#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>
#include <Eigen/Dense>

using namespace std;

BagOfWords::BagOfWords(vector<cv::Mat> points) {
	cv::Mat inputData = BagOfWords::hstackMats(points);
	cout << inputData.size() << endl;
	cvflann::KMeansIndexParams params(32, 11, cvflann::CENTERS_KMEANSPP);
	cv::Mat centers(DSIFT_TOTAL_CLUSTERS, inputData.size().width, CV_32F);
	cout << "clustering..." << endl;
	cv::flann::hierarchicalClustering<cv::flann::L2<float>>(inputData, centers, params);
	initIndex(centers);

	//cout << "flann clustering done" << endl;
	//saveData(inputData, centers);
	//cout << "data saved" << endl;
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
	//cv::flann::AutotunedIndexParams indexParams(.98);
	cv::flann::KDTreeIndexParams indexParams(4);
	wordIndex.reset(new cv::flann::Index(centers, indexParams, cvflann::FLANN_DIST_EUCLIDEAN));
}

bool BagOfWords::saveData(cv::Mat allpoints, cv::Mat centers, string dataFile) {
	bool success = true;
	try {
		cv::FileStorage fs(dataFile, cv::FileStorage::WRITE);
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
	vector<double> currentRow(queryFeatures.cols);
	vector<int> currentWord(1);
	vector<float> currentDistance(1);
	vector<int> wordCounts(DSIFT_TOTAL_CLUSTERS);
	cv::flann::SearchParams params(32);
	for (int r = 0; r < queryFeatures.rows; r++) {
		queryFeatures.row(r).copyTo(currentRow);
		wordIndex->knnSearch(currentRow, currentWord, currentDistance, 1, params);
		cout << currentWord[0] << ' ';
		wordCounts[currentWord[0]]++;
	}
	cout << endl;
	return wordCounts;
}

//cv::Mat BagOfWords::getWordCounts(cv::Mat& queryFeatures) {
//	cv::Mat indices(queryFeatures.rows, 1, CV_16U);
//	cv::Mat dists(queryFeatures.rows, 1, CV_32F);
//	cv::Mat wordCounts = cv::Mat::zeros(cv::Size(DSIFT_TOTAL_CLUSTERS, 1), CV_16U);
//	try {
//		cv::flann::SearchParams params;//with autotuned params the argument is ignored
//		wordIndex->knnSearch(queryFeatures, indices, dists, 1, params);
//		cout << indices.at<int>(0, 0) << ' ' << dists.at<int>(0,0) << endl;
//		argsToCounts(indices, wordCounts);
//		cout << wordCounts.at<int>(0,0) << ' ' << wordCounts.at<int>(0, 10) << endl;
//	}
//	catch (cv::Exception& e) {
//		cout << e.what() << endl;
//	}
//	return wordCounts;
//};
//
//void BagOfWords::argsToCounts(cv::Mat& indices, cv::Mat& words) {
//	for (int i = 0; i < indices.rows; i++) {
//		try {
//			auto arg = indices.at<int>(i,0);
//			words.at<int>(0, arg)++;
//		}
//		catch (cv::Exception& e) {
//			cout << e.what() << endl;
//		}
//	}
//}

cv::Mat BagOfWords::hstackMats(std::vector<cv::Mat> data) {
	cv::Mat imageSet;
	while (data.size() != 0) {
		imageSet.push_back(data.back());
		data.pop_back();
	}
	return imageSet;
}