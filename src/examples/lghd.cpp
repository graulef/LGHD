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

#include <iostream>
#include <string>
#include "types.h"
#include "image_io.h"
#include "log_gabor_filter_bank.h"
#include "phase_congruency.h"

#include <itkOpenCVImageBridge.h>

#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

const bool VERBOSE = false;
const int detection_threshold = 10;
const int high_quality_subset_size = 200;
const int descriptor_length = 384;
const int patch_size = 80;
const int num_scales = 4;
const int num_orientations = 6;
const int subregion_factor = 4;
const int num_bins = 6;


std::vector<cv::KeyPoint> get_keypoints(cv::Mat image, std::string spectrum) {
    // Find keypoints using FAST detector
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(detection_threshold);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(image, keypoints);
    std::cout << "Found " << keypoints.size() << " keypoints in total." << std::endl;

    // Only use high quality subset of all keypoints
    cv::KeyPointsFilter::retainBest(keypoints, high_quality_subset_size);
    std::cout << "Building descriptors for the " << keypoints.size() << " strongest keypoints." << std::endl;

    // Save image with kept keypoints to disk
    if (VERBOSE) {
        cv::Mat draw;
        cv::drawKeypoints(image, keypoints, draw);
        char keypoint_filename[32];
        sprintf(keypoint_filename, "%s_keypoints.jpg", spectrum.c_str());
        cv::imwrite(keypoint_filename, draw);
    }

    return keypoints;
}

cv::Mat generate_lgdh_descriptor(cv::Mat image, std::vector<cv::KeyPoint> keypoints, std::string spectrum){
    // LOG-GABOR FILTER COLLECTION

    // Create a bank of 2D log-Gabor filters (you can skip this if the
    // filters already exist in the disk).
    int width = image.cols;
    int height = image.rows;
    bip::triple<size_t> size = {width, height, 1};
    bip::log_gabor_filter_bank lgbf(
            "log_gabor",            // Filename prefix.
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
    bip::log_gabor_filter_bank::write_parameters(lgbf);
    lgbf.compute();

    // Convert OpenCV image to ITK image
    itk::Image<float,3>::Pointer itk_image = itk::OpenCVImageBridge::CVMatToITKImage<itk::Image<float,3>>(image);

    // Convert ITK image to array
    float *image_array;
    image_array = new float[width*height];
    image_array = bip::io::image2array<float, 3>(itk_image);

    // Apply the phase congruency technique to detect edges and corners
    // and other features in the 2D input image.
    bip::phase_congruency pc(
            "phase_congruency",          // Filename prefix.
            image_array,            // Input image.
            &lgbf,                  // Bank of log-gabor filters.
            size,                   // Image size (z=1 for 2D).
            NULL,                   // Input mask (NULL for no mask).
            -1.0,                   // Noise energy threshold (< 0 for auto estimation).
            1.0,                    // Noise standard deviation.
            3,                      // Sigmoid weighting gain.
            0.5                     // Sigmoid weighting cutoff.
    );

    // Compute Log-Gabor filtered images over all scales and orientations
    std::vector<cv::Mat> eo_collection = pc.compute_eo_collection();

    // Clean up memory
    delete[] image_array;

    // Write phase congruency of input image to disk
    if (VERBOSE) {
        for (int i = 0; i < num_orientations*num_scales; i++){
            int scale = round(i / num_orientations);
            int orientation = i - scale * num_orientations;
            char filename[64];
            sprintf(filename, "%s_%01u_orientation_%01u.jpg", spectrum.c_str(), scale + 1, orientation + 1);
            cv::Mat current_image = eo_collection[scale * num_orientations + orientation];
            current_image.convertTo(current_image, CV_8U, 255.0);
            cv::imwrite(filename, current_image);
        }
    }

    // DESCRIPTOR GENERATION

    // Define vector of descriptor vectors
    std::vector<std::vector<float>> descriptors;

    // Count how many keypoints were ignored
    int ignored_kps = 0;

    // Iterate over all keypoints extracting a patch around it and build the LGHD (Log-Gabor histogram descriptor)
    int kp_num = 0;
    const int patch_half = floor(patch_size/2);

    for (auto const& kp : keypoints) {
        // Define vector holding the actual descriptor
        std::vector<float> descriptor;

        // Get patch location
        const int x = round(kp.pt.x);
        const int y = round(kp.pt.y);

        // Get top-left point of patch
        int x_1 = std::max(1, x - patch_half);
        int y_1 = std::max(1, y - patch_half);
        int x_2 = std::min(x + patch_half, width);
        int y_2 = std::min(y + patch_half, height);

        // Define patch as rectangular region of interest
        cv::Rect patch_roi(x_1, y_1, x_2-x_1, y_2-y_1);

        // ignore patches that are not well-sized
        if (y_2 - y_1 != patch_size || x_2 - x_1 != patch_size) {
            ignored_kps++;
            continue;
        }

        // iterate over all scales building a partial descriptor for each
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
            int histSize = num_bins;
            float range[] = {0.0, 1.0 * (num_orientations-1)};
            const float* histRange = {range};
            int subregion_size = round(patch_size / subregion_factor);

            // Divide patch into subregions
            for (int i = 0; i < subregion_factor; i++){
                for (int j = 0; j < subregion_factor; j++) {

                    // Define subregion as rectangular region of interest
                    cv::Rect subregion_roi(i*subregion_size, j*subregion_size, subregion_size, subregion_size);

                    // Load subregion as subpatch from maximized filtered images
                    const cv::Mat imageHist = max_idx_eo_patch(subregion_roi);

                    // Define histogram result container
                    cv::Mat hist(cv::Size(1, num_bins), CV_32F);

                    // Calculate histogram (uniform sampling, no accumulation)
                    cv::calcHist(&imageHist, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

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

        // Add descriptor to overall collection
        descriptors.push_back(descriptor);

        kp_num++;
    }

    std::cout << "Generated descriptor for " << descriptors.size() << " keypoints." << std::endl;
    std::cout << "Ignored " << ignored_kps << " keypoints while building descriptors." << std::endl;

    // Convert descriptors from vec of vec to OpenCV matrix
    cv::Mat lghd_descr(descriptors.size(), descriptors.at(0).size(), CV_32F);
    for (int i = 0; i < lghd_descr.rows; ++i) {
        for (int j = 0; j < lghd_descr.cols; ++j) {
            lghd_descr.at<float>(i, j) = descriptors.at(i).at(j);
        }
    }

    // Store descriptors to disk
    if (VERBOSE) {
        char descr_filename[32];
        sprintf(descr_filename, "%s_descriptors.json", spectrum.c_str());
        cv::FileStorage descr_file(descr_filename, cv::FileStorage::WRITE);
        descr_file << "matName" << lghd_descr;
    }

    return lghd_descr;
}


int main(int argc, char *argv[])
{
    // Load RGB image & detect keypoints
    const cv::Mat rgb_image = cv::imread("data/20160719_191350.jpg", cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> rgb_kps = get_keypoints(rgb_image, "rgb");

    // Generate LGHD descriptor for RGB
    cv::Mat rgb_descr = generate_lgdh_descriptor(rgb_image, rgb_kps, "rgb");

    // Load infrared image & detect keypoints
    const cv::Mat ir_image = cv::imread("data/20160719_191349.jpg", cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> ir_kps = get_keypoints(ir_image, "ir");

    // Generate LGHD descriptor for infrared
    cv::Mat ir_descr = generate_lgdh_descriptor(ir_image, ir_kps, "ir");

    // Use Brute-Force matcher to match descriptors
    cv::BFMatcher matcher(cv::NORM_L2, true);
    std::vector<cv::DMatch> matches;
    matcher.match(rgb_descr, ir_descr, matches);

    // Draw matches onto image
    cv::Mat img_matches;
    drawMatches(rgb_image, rgb_kps, ir_image, ir_kps,
                matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Save detected matches
    cv::imwrite("all_bf_matches.jpg", img_matches);

    return 0;
}
