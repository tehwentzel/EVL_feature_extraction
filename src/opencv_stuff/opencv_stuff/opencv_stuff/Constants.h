#pragma once

static const int imWidth = 400;
static const int imHeight = 400;
static const double MIN_CROP_RATIO = .3;
static const int DSIFT_STEP = 8;
static const int DSIFT_QUANTIZE_SIZE = 50;
static const int DSIFT_TOTAL_CLUSTERS = 800;
static const float ANN_RECALL = .95;
static const int CRISP_HISTSIZE = 16;
static const std::string DATA_FOLDER = "D:\\git_repos\\EVL_feature_extraction\\src\\data\\";
static const std::string BOVW_FILE = DATA_FOLDER + "bovw_data";
static const std::string FEATURE_DICT_FILE = DATA_FOLDER + "image_features";
static const std::string FEATURE_JSON_FILE = DATA_FOLDER + "features.json";