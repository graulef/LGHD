/**
 * @file   lgdh.cpp
 * @author Felix Graule
 *
 * @addtogroup none
 * @ingroup    none
 *
 * @copyright Copyright (c) 2019 Felix Graule
 * @license GPL v2.0
 */

// SYSTEM
#include <iostream>
#include <string>
#include <dirent.h>

// PROJECT
#include "types.h"
#include "image_io.h"
#include "log_gabor_filter_bank.h"
#include "phase_congruency.h"

// ITK
#include <itkOpenCVImageBridge.h>

// OpenCV
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/features2d.hpp>

// TODO: Move this into config file
// Data & Debug
const std::string data_dir = "../..";
const std::string save_filters_dir = data_dir + "/filters";
const bool VERBOSE = false;

// Feature detection
const int detection_threshold = 10;
const int high_quality_subset_size = 400;

// LGHD descriptor
const int descriptor_length = 384;
const int patch_size = 100;
const int num_scales = 4;
const int num_orientations = 6;
const int subregion_factor = 4;

// Adaptive Non-Maximum Suppression (see paper)
const float robust_coeff = 1.11;

// Selection of matches based on percentile
const int good_points_max = 200; // 50
const float good_points_portion = 1.0f; // 0.15f


// Implementation guided by the paper "Multi-Image Matching using Multi-Scale Oriented Patches" by Brown, Szeliski, and Winder.
void adaptiveNonMaximalSuppresion(std::vector<cv::KeyPoint>& keypoints,
                                   const int num_keep) {
    // Nothing to be maximized
    if(keypoints.size() < num_keep) {
        return;
    }

    // Sort by response to detection filter
    std::sort(keypoints.begin(), keypoints.end(),
               [&](const cv::KeyPoint& lhs, const cv::KeyPoint& rhs) {
                   return lhs.response > rhs.response;
               });

    std::vector<cv::KeyPoint> anms_points;
    std::vector<double> radii;
    radii.resize(keypoints.size());
    std::vector<double> radii_sorted;
    radii_sorted.resize(keypoints.size());

    // Create list of keypoints sorted by maximum-response radius
    for(int i = 0; i < keypoints.size(); ++i) {
        const float response = keypoints[i].response * robust_coeff;
        double radius = std::numeric_limits<double>::max();
        for(int j = 0; j < i && keypoints[j].response > response; ++j) {
            radius = std::min(radius, cv::norm(keypoints[i].pt - keypoints[j].pt));
        }
        radii[i]       = radius;
        radii_sorted[i] = radius;
    }

    std::sort(radii_sorted.begin(), radii_sorted.end(),
               [&](const double& lhs, const double& rhs) {
                   return lhs > rhs;
               } );

    // Only keep keypoints with highest radii
    const double decision_radius = radii_sorted[num_keep];
    for(int i = 0; i < radii.size(); ++i) {
        if(radii[i] >= decision_radius) {
            anms_points.push_back(keypoints[i]);
        }
    }

    // Allocate result to output
    anms_points.swap(keypoints);
}

std::vector<cv::KeyPoint> get_keypoints(cv::Mat image, std::string spectrum) {
    // Find FAST keypoints
    std::vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(detection_threshold);
    detector->detect(image, keypoints);

    // Only use high quality subset of all keypoints based on ANMS
    adaptiveNonMaximalSuppresion(keypoints, high_quality_subset_size);

    std::cout << "Building descriptors for the " << keypoints.size() << " strongest keypoints." << std::endl;

    // DEBUG: store image with good keypoints
    if (VERBOSE) {
        cv::Mat draw;
        cv::drawKeypoints(image, keypoints, draw);
        char keypoint_filename[32];
        sprintf(keypoint_filename, "%s/debug/%s_keypoints.jpg", data_dir.c_str(), spectrum.c_str());
        cv::imwrite(keypoint_filename, draw);
    }

    return keypoints;
}

void generate_lgdh_descriptor(const cv::Mat& image_in, const std::vector<cv::KeyPoint>& keypoints_in,
			                  std::vector<cv::KeyPoint>* keypoints_out, cv::Mat* descriptors_out){

    // LOG-GABOR FILTER COLLECTION
    const unsigned int width = image_in.cols;
    const unsigned int height = image_in.rows;

    // Check if filter directory exists
    std::string save_filters_dir_local = save_filters_dir + "_" + std::to_string(width) + "_" + std::to_string(height) + "/";
    bool generate_new_filters = true;
    struct stat st;
    if(stat(save_filters_dir_local.c_str(), &st) == 0) {
        if (st.st_mode & S_IFDIR != 0) {
            generate_new_filters = false;
        }
    }

    // Create a bank of 2D log-Gabor filters (you can skip this if the filters already exist in the disk).
    bip::triple<size_t> size = {width, height, 1};

    bip::log_gabor_filter_bank lg_filter(
            save_filters_dir_local, // Filename prefix.
            size,                   // Filter size (z=1 for 2D).
            num_scales,             // Scales.
            num_orientations,       // Azimuths.
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
        // Create folder to store filters in
        mkdir(save_filters_dir_local.c_str(), 0777);
        bip::log_gabor_filter_bank::write_parameters(lg_filter);

        // Compute filters
        lg_filter.compute();
    }

    // Convert OpenCV image to ITK image
    itk::Image<float,3>::Pointer itk_image = itk::OpenCVImageBridge::CVMatToITKImage<itk::Image<float,3>>(image_in);

    // Convert ITK image to array
    float *image_array;
    image_array = new float[width*height];
    image_array = bip::io::image2array<float, 3>(itk_image);

    // Apply the phase congruency technique to detect edges and corners and other features in the 2D input image.
    bip::phase_congruency phase_cong(
            save_filters_dir,       // Filename prefix.
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
    if (VERBOSE) {
        for (int i = 0; i < num_orientations*num_scales; i++){
            int scale = round(i / num_orientations);
            int orientation = i - scale * num_orientations;
            char filename[64];
            sprintf(filename, "%s/debug/%01u_orientation_%01u.jpg", data_dir.c_str(), scale + 1, orientation + 1);
            cv::Mat current_image = eo_collection[scale * num_orientations + orientation];
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
    const int patch_half = floor(patch_size/2);

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
        if (y_2 - y_1 != patch_size || x_2 - x_1 != patch_size) {
            ignored_kps++;
            continue;
        }

        // iterate over all scales building a partial descriptor for each (eo stands for edge orientation)
        for (int s = 0; s < num_scales; s++) {
            // Allocate memory for each patch (get updated) and overall maximum patch
            cv::Mat eo_patch = cv::Mat::zeros(cv::Size(patch_size, patch_size), CV_32F);
            cv::Mat max_eo_patch = cv::Mat::zeros(cv::Size(patch_size, patch_size), CV_32F);

            // Init max image to value bigger than number of orientations (0..5 -> 6) to easily detect unchanged values
            cv::Mat max_idx_eo_patch = cv::Mat::ones(cv::Size(patch_size, patch_size), CV_8U) * num_orientations;

            // Iterate over all orientations and find maximize phase congruency
            for (int o = 0; o < num_orientations; o++) {

                // Load patch
                eo_patch = eo_collection[s * num_orientations + o](patch_roi);

                // Iterate over whole patch and compare to current max patch
                for (int i = 0; i < patch_size; i++) {
                    for (int j = 0; j < patch_size; j++) {

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
            int hist_size = num_orientations;
            float range[] = {0.0, 1.0 * (num_orientations-1)};
            const float* hist_range = {range};
            int subregion_size = round(patch_size / subregion_factor);

            // Divide patch into subregions
            for (int i = 0; i < subregion_factor; i++){
                for (int j = 0; j < subregion_factor; j++) {

                    // Define subregion as rectangular region of interest
                    cv::Rect subregion_roi(i*subregion_size, j*subregion_size, subregion_size, subregion_size);

                    // Load subregion as subpatch from maximized filtered images
                    const cv::Mat hist_image = max_idx_eo_patch(subregion_roi);

                    // Define histogram result container
                    cv::Mat hist(cv::Size(1, num_orientations), CV_32F);

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
    if (VERBOSE) {
        char descr_filename[32];
        sprintf(descr_filename, "%s/debug/descriptors.json", data_dir.c_str());
        cv::FileStorage descr_file(descr_filename, cv::FileStorage::WRITE);
        descr_file << "matName" << descr;
    }

    // Assign results to output pointers
    *keypoints_out = valid_keypoints;
    *descriptors_out = descr;
}


int main(int argc, char *argv[]) {
    // Load RGB image & detect keypoints
    const cv::Mat rgb_image = cv::imread(data_dir + "/test_images/9_rgb.jpg", cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> rgb_kps_detected = get_keypoints(rgb_image, "rgb");

    // Generate LGHD descriptor for RGB
    std::vector<cv::KeyPoint> rgb_kps;
    cv::Mat rgb_descr;
    generate_lgdh_descriptor(rgb_image, rgb_kps_detected, &rgb_kps, &rgb_descr);

    // Load infrared image & detect keypoints
    const cv::Mat ir_image = cv::imread(data_dir + "/test_images/9_ir.jpg", cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> ir_kps_detected = get_keypoints(ir_image, "ir");

    // Generate LGHD descriptor for infrared
    std::vector<cv::KeyPoint> ir_kps;
    cv::Mat ir_descr;
    generate_lgdh_descriptor(ir_image, ir_kps_detected, &ir_kps, &ir_descr);

    // Use Brute-Force matcher to match descriptors
    cv::BFMatcher matcher(cv::NORM_L2, true);	
    std::vector<cv::DMatch> matches;
    matcher.match(rgb_descr, ir_descr, matches);

    std::cout << "Number of IR keypoints: " << ir_kps.size() << std::endl;
    std::cout << "Number of IR descriptors: " << ir_descr.size() << std::endl;

    // Sort matches and preserve top 10% matches
    std::sort(matches.begin(), matches.end());
    std::vector<cv::DMatch> good_matches;
    double minDist = matches.front().distance;
    double maxDist = matches.back().distance;

    const int ptsPairs = std::min(good_points_max, (int)(matches.size() * good_points_portion));
    for( int i = 0; i < ptsPairs; i++ )
    {
        good_matches.push_back(matches[i]);
    }
    std::cout << "Max distance: " << maxDist << std::endl;
    std::cout << "Min distance: " << minDist << std::endl;

    // Draw matches onto image
    cv::Mat img_matches;
    drawMatches(rgb_image, rgb_kps, ir_image, ir_kps,
                good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Save detected matches
    cv::imwrite(data_dir + "/results/bf_matches.jpg", img_matches);

    return 0;
}
