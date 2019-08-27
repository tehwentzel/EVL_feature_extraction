#pragma once

#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>
#include "Constants.h"
#include "KDTree.h"

class BagOfWords
{
	public:
		BagOfWords(std::vector<cv::Mat>);
		BagOfWords(std::string dataFile = BOVW_FILE);
		std::vector<int> getWordCounts(cv::Mat&);
		//cv::Mat getWordCounts(cv::Mat&);
		bool saveData(cv::Mat, cv::Mat, std::string dataFile = BOVW_FILE);

	private:
		std::unique_ptr<KDTree> wordIndex;
		cv::Mat hstackMats(std::vector<cv::Mat>);
		void initIndex(cv::Mat&);
		//void argsToCounts(cv::Mat&, cv::Mat&);
};

