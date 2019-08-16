#include "Preprocess.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <string>
#include <map>
#include "Constants.h"

using namespace std;

void resizeImage(cv::Mat& image) {
	cv::Size s = image.size();
	double wRatio = (float)imWidth / (float)s.width;
	double hRatio = (float)imHeight / (float)s.height;
	double ratio = (wRatio > hRatio) ? wRatio : hRatio;
	cv::resize(image, image, cv::Size(), ratio, ratio, cv::INTER_AREA);
	//return image;
}

void cropImage(cv::Mat& image) {
	cv::Mat grayImg;
	cv::cvtColor(image, grayImg, cv::COLOR_BGR2GRAY);
	cv::dilate(grayImg, grayImg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
	cv::medianBlur(grayImg, grayImg, 3);
	cv::Rect bbox = deleteBorder(grayImg, 5);
	auto imSize = grayImg.size();
	if ((float)(bbox.width * bbox.height) / (float)(imSize.height * imSize.width) > MIN_CROP_RATIO) {
		//cout << image.size() << ' ';
		image = image(bbox);
		//cout << image.size() << endl;
	}
	//return image;
}

cv::Rect deleteBorder(cv::Mat src, int size) {
	//taken from https://answers.opencv.org/question/30082/detect-and-remove-borders-from-framed-photographs/
	//takes an input array and returns a Rect with the bounding box of the border
	cv::Mat sbl_x, sbl_y;
	int ksize = 2 * size + 1;
	cv::Sobel(src, sbl_x, CV_32FC1, 3, 0, ksize);
	cv::Sobel(src, sbl_y, CV_32FC1, 0, 3, ksize);
	cv::Mat sum_img = sbl_x + sbl_y;

	cv::Mat gray;
	cv::normalize(sum_img, gray, 0, 255, 32, CV_8UC1);

	cv::Mat row_proj, col_proj;
	cv::reduce(gray, row_proj, 1, 1, CV_8UC1);
	cv::reduce(gray, col_proj, 0, 1, CV_8UC1);
	cv::Sobel(row_proj, row_proj, CV_8UC1, 0, 2, 3);
	cv::Sobel(col_proj, col_proj, CV_8UC1, 2, 0, 3);

	cv::Point peak_pos;
	int half_pos = row_proj.total() / 2;
	cv::Rect result;

	cv::minMaxLoc(row_proj(cv::Range(0, half_pos), cv::Range(0, 1)), 0, 0, 0, &peak_pos);
	result.y = peak_pos.y;
	cv::minMaxLoc(row_proj(cv::Range(half_pos, row_proj.total()), cv::Range(0, 1)), 0, 0, 0, &peak_pos);
	result.height = (peak_pos.y + half_pos - result.y);

	half_pos = col_proj.total() / 2;
	cv::minMaxLoc(col_proj(cv::Range(0, 1), cv::Range(0, half_pos)), 0, 0, 0, &peak_pos);
	result.x = peak_pos.x;
	cv::minMaxLoc(col_proj(cv::Range(0, 1), cv::Range(half_pos, col_proj.total())), 0, 0, 0, &peak_pos);
	result.width = (peak_pos.x + half_pos - result.x);
	return result;
}

void preprocessImage(cv::Mat& image, bool denoise) {
	if (denoise) { cv::fastNlMeansDenoisingColored(image, image, 3, 3); }
	cropImage(image);
	resizeImage(image);
	//return image;
}