/**
 * @file   lgdh.h
 *
 * @copyright Copyright (c) 2019 Felix Graule
 * @license GPL v2.0
 */

#ifndef LGHD_H_
#define LGHD_H_

// SYSTEM
#include <iostream>
#include <string>
#include <dirent.h>

// OpenCV
#include "opencv2/core/core.hpp"


class LGHD {
  public:
    void generate_descriptor(const cv::Mat& image_in, const std::vector<cv::KeyPoint>& keypoints_in,
	                     std::vector<cv::KeyPoint>* keypoints_out, cv::Mat* descriptors_out);

  private:
    // TODO: Move this into config file
    // Data & Debug
    const std::string data_dir = "/home/graulef/catkin_ws_amo/src/lghd_catkin/data";
    const std::string save_filters_dir = data_dir + "/filters";
    const bool VERBOSE = false;

    // LGHD descriptor
    const int descriptor_length = 384;
    const int patch_size = 100;
    const int num_scales = 4;
    const int num_orientations = 6;
    const int subregion_factor = 4;
};

#endif // LGHD_H_

