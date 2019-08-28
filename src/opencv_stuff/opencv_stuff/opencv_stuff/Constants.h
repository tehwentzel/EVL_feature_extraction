#pragma once

static const int imWidth = 400;
static const int imHeight = 400;
static const double MIN_CROP_RATIO = .3;
static const int DSIFT_STEP = 8;
static const int DSIFT_QUANTIZE_SIZE = 50;
static const int DSIFT_TOTAL_CLUSTERS = 800;
static const float ANN_RECALL = .95;
static const int CRISP_HISTSIZE = 16;
static const std::string BOVW_FILE = "D:\\git_repos\\EVL_feature_extraction\\bovw_data";
static const std::string FEATURE_DICT_FILE = "D:\\git_repos\\EVL_feature_extraction\\image_features";
static const std::string FEATURE_JSON_FILE = "D:\\git_repos\\EVL_feature_extraction\\features.json";