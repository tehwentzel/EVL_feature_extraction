#include "FuzzyFeatures.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;

template <class T>
vector<T> stack(vector<vector<T>> input) {
	vector<T> output;
	for (auto& member : input) {
		for (auto& value : member) {
			output.push_back(value);
		}
	}
	return output;
}


vector<double> getCrispHistogram(cv::Mat image, int histSize) {
	image.convertTo(image, CV_32F);
	double min, max;
	cv::minMaxLoc(image, &min, &max);
	vector<cv::Mat> channels;
	cv::split(image, channels);
	float ranges[] = { 0,1 };
	const float* histRange[] = { ranges };
	vector<double> combinedHist;
	for (auto& channel : channels) {
		vector<double> normalizedHist;
		checkNormalized(channel);
		cv::Mat hist;
		cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, true, false);
		cv::normalize(hist, normalizedHist, 1.0, 0.0, cv::NORM_L1);
		for (auto& value : normalizedHist) {
			//cout << value << ' ';
			combinedHist.push_back(value);
		}
	}
	//cout << endl;
	return combinedHist;
}

void checkNormalized(cv::Mat& imgChannel) {
	double min, max;
	cv::minMaxLoc(imgChannel, &min, &max);
	if (max > 1 || min < 0) {
		cv::normalize(imgChannel, imgChannel, 0, 1, cv::NORM_MINMAX, CV_32F);
	}
}


vector<double> standardColorSpaceHistograms(cv::Mat& image, int histSize) {
	vector<int> colorSpaces = { cv::COLOR_BGR2HSV, cv::COLOR_BGR2HLS, cv::COLOR_BGR2Lab };
	cv::Mat convertedImage;

	vector<vector<double>> allHists;
	//for (auto& colorCode : colorSpaces) {
	//	cv::cvtColor(image, convertedImage, colorCode);
	//	auto colorHist = getCrispHistogram(convertedImage, histSize);
	//	allHists.push_back(colorHist);
	//}
	//allHists.push_back(getCrispHistogram(image));
	allHists.push_back(getCrispHistogram(BGRToOpponentColorSpace(image)));
	return stack<double>(allHists);
}

cv::Mat BGRToOpponentColorSpace(const cv::Mat& bgrImage){
	// based on private opencv method here https://github.com/opencv/opencv/blob/2.4/modules/features2d/src/descriptors.cpp#L126
	cv::Mat targetImage;
	bgrImage.copyTo(targetImage);
	for (int y = 0; y < bgrImage.rows; ++y)
		for (int x = 0; x < bgrImage.cols; ++x)
		{
			cv::Vec3b bgrPixel = bgrImage.at<cv::Vec3b>(y, x);
			uchar& b = bgrPixel[0];
			uchar& g = bgrPixel[1];
			uchar& r = bgrPixel[2];

			cv::Vec3b oPixel = targetImage.at<cv::Vec3b>(y, x);
			oPixel[0] = cv::saturate_cast<uchar>(0.5f * (255 + g - r)); // (R - G)/sqrt(2)
			oPixel[1] = cv::saturate_cast<uchar>(0.25f * (510 + r + g - 2 * b)); // (R + G - 2B)/sqrt(6)
			oPixel[2] = cv::saturate_cast<uchar>(1.f / 3.f * (r + g + b)); // (R + G + B)/sqrt(3)
			cout << bgrPixel << ' ' << oPixel << endl;
		}
	return targetImage;
}