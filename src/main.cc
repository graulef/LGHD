/**
 * @file   main.cpp
 * @author Felix Graule
 *
 * @copyright Copyright (c) 2019 Felix Graule
 * @license GPL v2.0
 */

// SYSTEM
#include <iostream>
#include <string>
#include <dirent.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

// PROJECT
#include "../include/lghd_catkin/lghd.h"

// TODO: Move this into config file
// Data & Debug
const std::string data_dir = "/home/graulef/catkin_ws_amo/src/lghd_catkin/data";
const std::string save_filters_dir = data_dir + "/filters";
const bool VERBOSE = false;

// Feature detection
const int detection_threshold = 10;
const int high_quality_subset_size = 400;

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

int main(int argc, char *argv[]) {
    LGHD descr;

    // Load RGB image & detect keypoints
    const cv::Mat rgb_image = cv::imread(data_dir + "/test_images/9_rgb.jpg", cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> rgb_kps_detected = get_keypoints(rgb_image, "rgb");

    // Generate LGHD descriptor for RGB
    std::vector<cv::KeyPoint> rgb_kps;
    cv::Mat rgb_descr;
    descr.generate_descriptor(rgb_image, rgb_kps_detected, &rgb_kps, &rgb_descr);

    // Load infrared image & detect keypoints
    const cv::Mat ir_image = cv::imread(data_dir + "/test_images/9_ir.jpg", cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> ir_kps_detected = get_keypoints(ir_image, "ir");

    // Generate LGHD descriptor for infrared
    std::vector<cv::KeyPoint> ir_kps;
    cv::Mat ir_descr;
    descr.generate_descriptor(ir_image, ir_kps_detected, &ir_kps, &ir_descr);

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
