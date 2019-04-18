/**
 * @file   lgdh.cpp
 * @author Felix Graule
 *
 * @copyright Copyright (c) 2019 Felix Graule
 * @license GPL v2.0
 */

// SYSTEM
#include <iostream>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

// PROJECT
#include "types.h"
#include "image_io.h"
#include "log_gabor_filter_bank.h"
#include "phase_congruency.h"
#include "../include/lghd_catkin/lghd.h"

// ITK
#include <itkOpenCVImageBridge.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

LGHD::LGHD(const unsigned int descriptor_length,
           const unsigned int patch_size,
           const unsigned int num_scales,
           const unsigned int num_orientations,
           const unsigned int subregion_factor,
           const std::string cache_filters_dir,
           const bool debug,
           const std::string save_debug_dir) :
           descriptor_length_(descriptor_length),
           patch_size_(patch_size),
           num_scales_(num_scales),
           num_orientations_(num_orientations),
           subregion_factor_(subregion_factor),
           cache_filters_dir_(cache_filters_dir),
           debug_(debug),
           save_debug_dir_(save_debug_dir){
}

LGHD::~LGHD() {
}

void LGHD::generate_descriptor(const cv::Mat& image_in,
                               const std::vector<cv::KeyPoint>& keypoints_in,
			       		       std::vector<cv::KeyPoint>* keypoints_out,
			       		       cv::Mat* descriptors_out) {
    // LOG-GABOR FILTER COLLECTION
    const unsigned int width = image_in.cols;
    const unsigned int height = image_in.rows;

    // Check if filter directory exists and create it if not
    std::string cache_filters_dir_local = cache_filters_dir_ + "_" + std::to_string(width) + "_" + std::to_string(height) + "/";
    bool generate_new_filters = true;
    struct stat st;
    if(stat(cache_filters_dir_local.c_str(), &st) == 0) {
        if (st.st_mode & S_IFDIR != 0) {
            generate_new_filters = false;
        }
    }

    std::cout << "generate new filers: " << generate_new_filters << std::endl;

    if (generate_new_filters) {
        // Create folder to store filters in
        int mkdir_errno;
        mkdir_errno = mkdir(cache_filters_dir_local.c_str(), 0777);

        std::cout << "dir: " << cache_filters_dir_local << std::endl;
        std::cout << "errno: " << mkdir_errno << std::endl;
        std::cout << "errno: " << errno << std::endl;

        assert(mkdir_errno = 0);
    }

    // Create a bank of 2D log-Gabor filters (you can skip this if the filters already exist in the disk).
    triple<size_t> size = {width, height, 1};

    log_gabor_filter_bank lg_filter(
            cache_filters_dir_local, // Filename prefix.
            size,                   // Filter size (z=1 for 2D).
            num_scales_,             // Scales.
            num_orientations_,       // Azimuths.
            1,                      // Elevations (1 for 2D).
            1./3,                   // Max central frequency.
            1.6,                    // Multiplicative factor.
            0.75,                   // Frequency spread ratio.
            1,                      // Angular spread ratio.
            15,                     // Butterworth order.
            0.45,                   // Butterworth cutoff.
            false                   // Uniform sampling?
    );

    if (generate_new_filters) {
        // Write parameters and compute filters
        log_gabor_filter_bank::write_parameters(lg_filter);
        lg_filter.compute();
    }

    // Convert OpenCV image to ITK image
    itk::Image<float,3>::Pointer itk_image = itk::OpenCVImageBridge::CVMatToITKImage<itk::Image<float,3>>(image_in);

    // Convert ITK image to array
    float *image_array;
    image_array = new float[width*height];
    image_array = image2array<float, 3>(itk_image);

    // Apply the phase congruency technique to detect edges and corners and other features in the 2D input image.
    phase_congruency phase_cong(
            cache_filters_dir_,       // Filename prefix.
            image_array,            // Input image.
            &lg_filter,             // Bank of log-gabor filters.
            size,                   // Image size (z=1 for 2D).
            NULL,                   // Input mask (NULL for no mask).
            -1.0,                   // Noise energy threshold (< 0 for auto estimation).
            1.0,                    // Noise standard deviation.
            3,                      // Sigmoid weighting gain.
            0.5                     // Sigmoid weighting cutoff.
    );

    // Compute Log-Gabor filtered images over all scales and orientations
    std::vector<cv::Mat> eo_collection = phase_cong.compute_eo_collection();

    // Clean up memory
    delete[] image_array;

    // DEBUG: Store phase congruency image
    if (debug_) {
        for (int i = 0; i < num_orientations_*num_scales_; i++){
            int scale = round(i / num_orientations_);
            int orientation = i - scale * num_orientations_;
            char filename[64];
            sprintf(filename, "%s/%01u_orientation_%01u.jpg", save_debug_dir_.c_str(), scale + 1, orientation + 1);
            cv::Mat current_image = eo_collection[scale * num_orientations_ + orientation];
            current_image.convertTo(current_image, CV_8U, 255.0);
            cv::imwrite(filename, current_image);
        }
    }

    // DESCRIPTOR GENERATION

    // Define keypoint and descriptor vectors
    std::vector<cv::KeyPoint> valid_keypoints;
    std::vector<std::vector<float>> valid_descriptors;

    // Count how many keypoints were ignored
    int ignored_kps = 0;

    // Iterate over all keypoints extracting a patch around it and build the LGHD (Log-Gabor histogram descriptor)
    int kp_num = 0;
    const int patch_half = floor(patch_size_/2);

    for (auto const& kp : keypoints_in) {
        // Define vector holding the actual descriptor
        std::vector<float> descriptor;

        // Get patch location
        const int x = round(kp.pt.x);
        const int y = round(kp.pt.y);

        // Get top-left point of patch
        int x_1 = std::max(1, x - patch_half);
        int y_1 = std::max(1, y - patch_half);
        int x_2 = std::min(x + patch_half, static_cast<int>(width));
        int y_2 = std::min(y + patch_half, static_cast<int>(height));

        // Define patch as rectangular region of interest
        cv::Rect patch_roi(x_1, y_1, x_2-x_1, y_2-y_1);

        // ignore patches that are not well-sized
        if (y_2 - y_1 != patch_size_ || x_2 - x_1 != patch_size_) {
            ignored_kps++;
            continue;
        }

        // iterate over all scales building a partial descriptor for each (eo stands for edge orientation)
        for (int s = 0; s < num_scales_; s++) {
            // Allocate memory for each patch (get updated) and overall maximum patch
            cv::Mat eo_patch = cv::Mat::zeros(cv::Size(patch_size_, patch_size_), CV_32F);
            cv::Mat max_eo_patch = cv::Mat::zeros(cv::Size(patch_size_, patch_size_), CV_32F);

            // Init max image to value bigger than number of orientations (0..5 -> 6) to easily detect unchanged values
            cv::Mat max_idx_eo_patch = cv::Mat::ones(cv::Size(patch_size_, patch_size_), CV_8U) * num_orientations_;

            // Iterate over all orientations and find maximize phase congruency
            for (int o = 0; o < num_orientations_; o++) {

                // Load patch
                eo_patch = eo_collection[s * num_orientations_ + o](patch_roi);

                // Iterate over whole patch and compare to current max patch
                for (int i = 0; i < patch_size_; i++) {
                    for (int j = 0; j < patch_size_; j++) {

                        float current_max = max_eo_patch.at<float>(i, j);
                        float current_value = eo_patch.at<float>(i, j);

                        if (current_value > current_max) {
                            max_idx_eo_patch.at<uint8_t>(i, j) = o;
                            max_eo_patch.at<float>(i, j) = current_value;
                        }
                    }
                }
            }

            // Set parameters for histogram generation
            int hist_size = num_orientations_;
            float range[] = {0.0, 1.0 * (num_orientations_-1)};
            const float* hist_range = {range};
            int subregion_size = round(patch_size_ / subregion_factor_);

            // Divide patch into subregions
            for (int i = 0; i < subregion_factor_; i++){
                for (int j = 0; j < subregion_factor_; j++) {

                    // Define subregion as rectangular region of interest
                    cv::Rect subregion_roi(i*subregion_size, j*subregion_size, subregion_size, subregion_size);

                    // Load subregion as subpatch from maximized filtered images
                    const cv::Mat hist_image = max_idx_eo_patch(subregion_roi);

                    // Define histogram result container
                    cv::Mat hist(cv::Size(1, num_orientations_), CV_32F);

                    // Calculate histogram (uniform sampling, no accumulation)
                    cv::calcHist(&hist_image, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range, true, false);

                    // Convert histogram matrix to vector of floats
                    std::vector<float> hist_vec;
                    if (hist.isContinuous()) {
                        hist_vec.assign((float*)hist.datastart, (float*)hist.dataend);
                    } else {
                        for (int i = 0; i < hist.rows; ++i) {
                            hist_vec.insert(hist_vec.end(), hist.ptr<float>(i), hist.ptr<float>(i)+hist.cols);
                        }
                    }

                    // Assign result to descriptor
                    descriptor.insert(std::end(descriptor), std::begin(hist_vec), std::end(hist_vec));
                }
            }
        }

        // Find norm of per keypoint descriptor to then normalize it
        double accum = 0.;
        for (int i = 0; i < descriptor.size(); ++i) {
            accum += descriptor[i] * descriptor[i];
        }
        double norm = sqrt(accum);

        // Only normalize if norm is bigger than zero
        if (norm > std::numeric_limits<float>::epsilon()) {
            std::transform(descriptor.begin(), descriptor.end(), descriptor.begin(), std::bind(std::divides<float>(), std::placeholders::_1, norm));
        } else {
            std::fill(descriptor.begin(), descriptor.end(), 0);
        }

        // Add keypoint and descriptor to overall collection
	    valid_keypoints.push_back(kp);
        valid_descriptors.push_back(descriptor);

        kp_num++;
    }

    std::cout << "Generated descriptor for " << valid_descriptors.size() << " keypoints." << std::endl;
    std::cout << "Ignored " << ignored_kps << " keypoints while building descriptors." << std::endl;

    // Convert descriptors from vec of vec to OpenCV matrix
    cv::Mat descr(valid_descriptors.size(), valid_descriptors.at(0).size(), CV_32F);
    for (int i = 0; i < descr.rows; ++i) {
        for (int j = 0; j < descr.cols; ++j) {
            descr.at<float>(i, j) = valid_descriptors.at(i).at(j);
        }
    }

    // DEBUG: Store descriptors
    if (debug_) {
        char descr_filename[32];
        sprintf(descr_filename, "%s/descriptors.json", save_debug_dir_.c_str());
        cv::FileStorage descr_file(descr_filename, cv::FileStorage::WRITE);
        descr_file << "matName" << descr;
    }

    // Assign results to output pointers
    *keypoints_out = valid_keypoints;
    *descriptors_out = descr;
}



