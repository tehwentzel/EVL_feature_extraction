#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include "Constants.h"
#include <opencv2/core.hpp>

template <class T>
std::vector<T> stack(std::vector<std::vector<T>>);

std::vector<double> getCrispHistogram(cv::Mat image, int histSize = CRISP_HISTSIZE);
std::vector<double> standardColorSpaceHistograms(cv::Mat& image, int histSize= CRISP_HISTSIZE);
cv::Mat BGRToOpponentColorSpace(const cv::Mat&);
std::vector<double> standardHistograms(cv::Mat&, std::function<std::vector<double>(cv::Mat, int)>, int);
void checkNormalized(cv::Mat&);