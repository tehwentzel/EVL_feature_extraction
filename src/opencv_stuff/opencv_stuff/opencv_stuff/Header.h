#pragma once
#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/core.hpp>

cv::Mat preprocessImage(cv::Mat);
cv::Mat resizeImage(cv::Mat);
cv::Mat preprocessImage(cv::Mat, bool denoise = false);
cv::Mat cropImage(cv::Mat);
cv::Rect deleteBorder(cv::InputArray, int);

#endif