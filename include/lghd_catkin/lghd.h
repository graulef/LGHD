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

    /**
     * @brief Constructor method.
     *
     * @param[in] descriptor_length     Number of values per descriptor.
     * @param[in] patch_size            Size of patch extracted around interest point.
     * @param[in] num_scales            Number of filter scales.
     * @param[in] num_orientations      Number of filter orientation angles.
     * @param[in] subregion_factor      Number of subregions in extracted patch when creating histogram.
     * @param[in] cache_filter_dir      Directory where filters are cached.
     * @param[in] verbose               Enable verbose output and debug files.
     * @param[in] save_debug_dir        Directory where debug files are stored.
     */

    LGHD(const std::string &spectrum,
         const unsigned int descriptor_length = 384,
         const unsigned int patch_size = 100,
         const unsigned int num_scales = 4,
         const unsigned int num_orientations = 6,
         const unsigned int subregion_factor = 4,
         const std::string cache_filters_dir = "./filters",
         const bool debug = true,
         const std::string save_debug_dir = "/home/graulef/catkin_ws_amo/src/lghd_catkin/data"); // default: "./debug"

    /**
     * @brief Destructor method.
     */
    ~LGHD();

    /**
     * @brief Descriptor generation method.
     *
     * @param[in] image_in              Image to compute descriptors on.
     * @param[in] keypoints_in          Interest points in input image to build descriptors for.
     * @param[out] keypoints_out         Interest points for which a descriptor was built (some might have been ignored)
     * @param[out] descriptors_out       Resulting descriptor (Matrix containing descriptor vectors for all keypoints)
     */

    void generate_descriptor(const cv::Mat& image_in,
	                         std::vector<cv::KeyPoint>* keypoints_out,
	                         cv::Mat* descriptors_out);

  private:
    void adaptiveNonMaximalSuppresion(std::vector<cv::KeyPoint>& keypoints,
                                      const int num_keep);
    // TODO: Move parameters to config
    // TODO: Add spectrum to create more meaningful debug files

    // Feature detection
    const int detection_threshold = 10;
    const int high_quality_subset_size = 400;

    // Adaptive Non-Maximum Suppression (see paper)
    const float robust_coeff = 1.11;

    // Selection of matches based on percentile
    const int good_points_max = 200; // 50
    const float good_points_portion = 1.0f; // 0.15f

    const bool use_pc_maps_detection_ = true;

    const std::string &spectrum_;
    const unsigned int descriptor_length_;
    const unsigned int patch_size_;
    const unsigned int num_scales_;
    const unsigned int num_orientations_;
    const unsigned int subregion_factor_;
    const std::string cache_filters_dir_;
    const bool debug_;
    const std::string save_debug_dir_;
};

#endif // LGHD_H_

