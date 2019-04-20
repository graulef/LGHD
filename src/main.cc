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

#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>

// PROJECT
#include "../include/lghd_catkin/lghd.h"


// Selection of matches based on percentile
const int good_points_max = 1000; // 50
const float good_points_portion = 1.0f; // 0.15f


int main(int argc, char *argv[]) {

    const std::string load_data_dir = "/home/graulef/catkin_ws_amo/src/lghd_catkin/data";
    const std::string input_index = "4";

    // Create descriptor object for RGB
    LGHD lghd_descr_rgb_obj(input_index + "/rgb");

    // Load RGB image
    const cv::Mat rgb_image = cv::imread(load_data_dir + "/test_images/" + input_index + "_rgb.jpg", cv::IMREAD_GRAYSCALE);
    const cv::Mat rgb_image_c = cv::imread(load_data_dir + "/test_images/" + input_index + "_rgb.jpg", cv::IMREAD_COLOR);

    // Generate LGHD descriptor for RGB
    std::vector<cv::KeyPoint> rgb_kps;
    cv::Mat rgb_descr;
    lghd_descr_rgb_obj.generate_descriptor(rgb_image, &rgb_kps, &rgb_descr);

    // Create descriptor object for RGB
    LGHD lghd_descr_ir_obj(input_index + "/ir");

    // Load infrared image
    const cv::Mat ir_image = cv::imread(load_data_dir + "/test_images/" + input_index + "_ir.jpg", cv::IMREAD_GRAYSCALE);
    const cv::Mat ir_image_fc = cv::imread(load_data_dir + "/test_images/" + input_index + "_ir.jpg", cv::IMREAD_COLOR);

    // Generate LGHD descriptor for infrared
    std::vector<cv::KeyPoint> ir_kps;
    cv::Mat ir_descr;
    lghd_descr_ir_obj.generate_descriptor(ir_image, &ir_kps, &ir_descr);

    // Use Brute-Force matcher to match descriptors // TODO: Try out knnMatch & Lowe ratio test
    const bool cross_check = true;
    cv::BFMatcher matcher(cv::NORM_L2, cross_check);
    std::vector<cv::DMatch> matches;
    matcher.match(rgb_descr, ir_descr, matches);

    std::cout << "Number of IR keypoints: " << ir_kps.size() << std::endl;
    std::cout << "Number of IR descriptors: " << ir_descr.size() << std::endl;

    std::cout << "Number of matches before filtering: " << matches.size() << std::endl;

    // Sort matches and preserve top percentile of matches (good_points_portion)
    std::sort(matches.begin(), matches.end());
    std::vector<cv::DMatch> good_matches;
    double minDist = matches.front().distance;
    double maxDist = matches.back().distance;

    const int ptsPairs = std::min(good_points_max, (int)(matches.size() * good_points_portion));
    for( int i = 0; i < ptsPairs; i++ ) {
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
    cv::imwrite(load_data_dir + "/" + input_index + "/bf_matches.jpg", img_matches);

    std::cout << "Number of matches after filtering: " << good_matches.size() << std::endl;

    // Extract corresponding patches from both images

    // Get the keypoints from the good matches
    std::vector<cv::Point2f> rgb_kps_good;
    std::vector<cv::Point2f> ir_kps_good;
    for (size_t i = 0; i < good_matches.size(); i++) {
        rgb_kps_good.push_back(rgb_kps[good_matches[i].queryIdx].pt);
        ir_kps_good.push_back(ir_kps[good_matches[i].trainIdx].pt);
    }

    const int width = rgb_image.cols;
    const int height = rgb_image.rows;
    const int patch_size_ = 100;
    const int patch_half = patch_size_ / 2;

    // Find homography between the two images

    // First using RANSAC to filter outliers
    double ransacReprojThreshold = 10;
    const int maxIters = 20000000;
    const double confidence = 0.95;

    cv::Mat inlier_mask(cv::Size(1, good_matches.size()), CV_8U);
    std::vector<cv::Point2f> rgb_inliers;
    std::vector<cv::Point2f> rgb_outliers;
    std::vector<cv::Point2f> ir_inliers;
    std::vector<cv::Point2f> ir_outliers;

    cv::Mat H_init = cv::findHomography(rgb_kps_good, ir_kps_good, cv::RANSAC, ransacReprojThreshold, inlier_mask, maxIters, confidence);

    // Warp IR patch back to RGB space
    cv::Mat ir_image_warped(cv::Size(width, height), CV_8U);
    cv::warpPerspective(ir_image, ir_image_warped, H_init, ir_image_warped.size(), cv::INTER_NEAREST+cv::WARP_INVERSE_MAP);
    cv::imwrite(load_data_dir + "/" + input_index + "/ir_warped_init_H.jpg", ir_image_warped);

    std::cout << "Inliers mask size: " << inlier_mask.size() << std::endl;
    std::cout << "Initial H = " << H_init << std::endl;

    for (int i = 0; i < rgb_kps_good.size(); i++) {
        if (inlier_mask.at<bool>(i)){
            rgb_inliers.push_back(rgb_kps_good.at(i));
            ir_inliers.push_back(ir_kps_good.at(i));
        } else {
            rgb_outliers.push_back(rgb_kps_good.at(i));
            ir_outliers.push_back(ir_kps_good.at(i));
        }
    }

    std::cout << "Inliers: " << rgb_inliers.size() << ", Outliers: " << rgb_outliers.size() << std::endl;

    // TODO: draw inlier & outlier set
//    // Draw matches onto image
//    cv::Mat img_inliers;
//    drawMatches(rgb_image, rgb_inliers, ir_image, ir_inliers,
//                good_matches, img_inliers, cv::Scalar::all(-1), cv::Scalar::all(-1),
//                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//
//    // Save detected matches
//    cv::imwrite(load_data_dir + "/" + input_index + "/ransac_inliers.jpg", img_inliers);


    // Repeat using LMEDS and inliers only
    ransacReprojThreshold = 3;
    cv::Mat H = cv::findHomography(rgb_inliers, ir_inliers, cv::LMEDS, ransacReprojThreshold, inlier_mask);
    std::cout << "Refined H = " << H << std::endl;

    std::cout << "Inliers mask size refined: " << inlier_mask.size() << std::endl;
    std::cout << "Inliers mask refined: " << inlier_mask << std::endl;

    // Warp IR patch back to RGB space
    //cv::warpPerspective(ir_image, ir_image_warped, H, ir_image_warped.size(), cv::INTER_NEAREST+cv::WARP_INVERSE_MAP);
    //cv::imwrite(load_data_dir + "/" + input_index + "/ir_warped.jpg", ir_image_warped);

    cv::FileStorage file(load_data_dir + "/" + input_index + "/homography.txt", cv::FileStorage::WRITE);
    file << "H" << H;

    // Combine both images (using the colored RGB and false-colored IR image for better contrast)
    cv::Mat overlap(cv::Size(width, height), CV_8U);
    cv::Mat ir_image_fc_warped(cv::Size(width, height), CV_8U);
    cv::warpPerspective(ir_image_fc, ir_image_fc_warped, H, ir_image_fc_warped.size(), cv::INTER_NEAREST+cv::WARP_INVERSE_MAP);
    overlap = 0.4 * ir_image_fc_warped + 0.6 * rgb_image_c;
    cv::imwrite(load_data_dir + "/" + input_index + "/overlap.jpg", overlap);

    // Get the patches for each keypoint
    int i = 0;
    for (auto const& p: rgb_kps_good) {
        i++;
        // Get patch location
        const int x = round(p.x);
        const int y = round(p.y);

        // Get top-left point of patch
        int x_1 = std::max(1, x - patch_half);
        int y_1 = std::max(1, y - patch_half);
        int x_2 = std::min(x + patch_half, static_cast<int>(width));
        int y_2 = std::min(y + patch_half, static_cast<int>(height));

        // Define patch as rectangular region of interest
        cv::Rect patch_roi(x_1, y_1, x_2-x_1, y_2-y_1);

        // Extract patches from rgb image and warped ir image
        cv::Mat rgb_patch = rgb_image(patch_roi);
        cv::imwrite(load_data_dir + "/" +input_index + "/patches/rgb_patch_" + std::to_string(i) + ".jpg", rgb_patch);
        cv::Mat ir_patch = ir_image_warped(patch_roi);
        cv::imwrite(load_data_dir + "/" +input_index + "/patches/ir_patch_" + std::to_string(i) + ".jpg", ir_patch);

        // Combine both patches into one image
        cv::Mat combined_patches(cv::Size(rgb_patch.rows, 2 * rgb_patch.cols), CV_8U);
        cv::hconcat(rgb_patch, ir_patch, combined_patches);
        cv::imwrite(load_data_dir + "/" +input_index + "/patches/patches_" + std::to_string(i) + ".jpg", combined_patches);
    }

    return 0;
}
