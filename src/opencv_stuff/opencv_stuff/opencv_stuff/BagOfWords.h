#pragma once

#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>

class BagOfWords
{
	public:
		BagOfWords(std::vector<cv::Mat> points);
		std::vector<int> getWordCounts(cv::Mat);
	private:
		cv::flann::Index wordIndex;
		int num_words;
		cv::Mat hstackMats(std::vector<cv::Mat> data);
		cv::Mat inputData;
};

