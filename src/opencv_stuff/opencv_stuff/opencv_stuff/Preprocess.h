#pragma once

#include <opencv2/core.hpp>

//cv::Mat preprocessImage(cv::Mat);
void resizeImage(cv::Mat&);
void preprocessImage(cv::Mat&, bool denoise = false);
void cropImage(cv::Mat&);
cv::Rect deleteBorder(cv::Mat, int);
