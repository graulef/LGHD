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
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include </opt/opencv_contrib/modules/xfeatures2d/include/opencv2/xfeatures2d/nonfree.hpp>

// PROJECT
#include "../include/lghd_catkin/lghd.h"

const int high_quality_subset_size = 1000;


void adaptiveNonMaximalSuppresion(std::vector<cv::KeyPoint>& keypoints,
                                  const int num_keep) {
    const float robust_coeff = 1.11;

    // Only use high quality subset of all keypoints based on ANMS
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
        radii[i] = radius;
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

void generate_brisk_descriptor(const cv::Mat& image_in,
                              std::vector<cv::KeyPoint>* keypoints_out,
                              cv::Mat* descriptors_out,
                              std::string spectrum,
                               std::string save_debug_dir) {
    // Find FAST keypoints
    std::vector <cv::KeyPoint> keypoints;
    const int detection_threshold = 5;
    cv::Ptr <cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(detection_threshold);
    detector->detect(image_in, keypoints);

    // ANMS
    adaptiveNonMaximalSuppresion(keypoints, high_quality_subset_size);

    std::cout << "BRISK: Building descriptors for the " << keypoints.size() << " strongest keypoints." << std::endl;

    // DEBUG: store image with good keypoints
    cv::Mat draw;
    cv::drawKeypoints(image_in, keypoints, draw);
    char keypoint_filename[512];
    sprintf(keypoint_filename, "%s/%s_keypoints.jpg", save_debug_dir.c_str(), spectrum.c_str());
    cv::imwrite(keypoint_filename, draw);

    cv::Ptr<cv::Feature2D> f2d = cv::BRISK::create();
    cv::Mat descriptors;

    f2d->compute(image_in, keypoints, descriptors);

    *keypoints_out = keypoints;
    *descriptors_out = descriptors;
}

void generate_sift_descriptor(const cv::Mat& image_in,
                               std::vector<cv::KeyPoint>* keypoints_out,
                               cv::Mat* descriptors_out,
                               std::string spectrum,
                               std::string save_debug_dir) {
    // Find FAST keypoints
    std::vector <cv::KeyPoint> keypoints;
    const int detection_threshold = 5;
    cv::Ptr <cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(detection_threshold);
    detector->detect(image_in, keypoints);

    // ANMS
    adaptiveNonMaximalSuppresion(keypoints, high_quality_subset_size);

    std::cout << "SIFT: Building descriptors for the " << keypoints.size() << " strongest keypoints." << std::endl;

    // DEBUG: store image with good keypoints
    cv::Mat draw;
    cv::drawKeypoints(image_in, keypoints, draw);
    char keypoint_filename[512];
    sprintf(keypoint_filename, "%s/%s_keypoints.jpg", save_debug_dir.c_str(), spectrum.c_str());
    cv::imwrite(keypoint_filename, draw);

    cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create();
    cv::Mat descriptors;

    f2d->compute(image_in, keypoints, descriptors);

    *keypoints_out = keypoints;
    *descriptors_out = descriptors;
}

void run(std::string input_index, std::string method) {

    const std::string load_data_dir = "/home/graulef/catkin_ws_amo/src/aerial_mapper_optimization/dependencies/LGHD/data";
    const std::string save_debug_dir = "/home/graulef/catkin_ws_amo/src/aerial_mapper_optimization/dependencies/LGHD/data" + input_index;

    // Load RGB image
    const cv::Mat rgb_image = cv::imread(load_data_dir + "/test_images/" + input_index + "_rgb.jpg", cv::IMREAD_GRAYSCALE);
    std::cout << load_data_dir + "/test_images/" + input_index + "_rgb.jpg" << std::endl;
    cv::Mat rgb_image_c = cv::imread(load_data_dir + "/test_images/" + input_index + "_rgb.jpg", cv::IMREAD_COLOR);
    cv::applyColorMap(rgb_image_c, rgb_image_c, cv::COLORMAP_BONE);

    // Load infrared image
    const cv::Mat ir_image = cv::imread(load_data_dir + "/test_images/" + input_index + "_ir.jpg", cv::IMREAD_GRAYSCALE);
    const cv::Mat ir_image_fc = cv::imread(load_data_dir + "/test_images/" + input_index + "_ir.jpg", cv::IMREAD_COLOR);
    cv::applyColorMap(ir_image_fc, ir_image_fc, cv::COLORMAP_COOL);

    std::vector<cv::KeyPoint> rgb_kps;
    cv::Mat rgb_descr;
    std::vector<cv::KeyPoint> ir_kps;
    cv::Mat ir_descr;

    if (method == "lghd") {
        // Create descriptor for RGB
        LGHD lghd_descr_rgb_obj("/rgb", input_index);
        lghd_descr_rgb_obj.generate_descriptor(rgb_image, &rgb_kps, &rgb_descr);
        // Generate descriptor for infrared
        LGHD lghd_descr_ir_obj("/ir", input_index);
        lghd_descr_ir_obj.generate_descriptor(ir_image, &ir_kps, &ir_descr);
    } else if (method == "sift") {
        // Create descriptor for RGB
        generate_sift_descriptor(rgb_image, &rgb_kps, &rgb_descr, "/rgb", save_debug_dir);
        // Generate descriptor for infrared
        generate_sift_descriptor(ir_image, &ir_kps, &ir_descr, "/ir", save_debug_dir);
    } else if (method == "brisk") {
        // Create descriptor for RGB
        generate_brisk_descriptor(rgb_image, &rgb_kps, &rgb_descr, "/rgb", save_debug_dir);
        // Generate descriptor for infrared
        generate_brisk_descriptor(ir_image, &ir_kps, &ir_descr, "/ir", save_debug_dir);
    } else {
        std::cout << "ERROR: Feature detection & description method unknown" << std::endl;
        return;
    }

    // Use Brute-Force matcher to match descriptors
    const bool cross_check = true;
    cv::BFMatcher matcher(cv::NORM_L2, cross_check);
    std::vector<cv::DMatch> matches;
    matcher.match(rgb_descr, ir_descr, matches);
    std::cout << "Number of matches: " << matches.size() << std::endl;

    // Sort matches and print min & max distance
    std::sort(matches.begin(), matches.end());
    double minDist = matches.front().distance;
    double maxDist = matches.back().distance;
    std::cout << "Max distance: " << maxDist << std::endl;
    std::cout << "Min distance: " << minDist << std::endl;

    // Draw raw BF matches onto image
    cv::Mat img_matches;
    drawMatches(rgb_image, rgb_kps, ir_image, ir_kps,
                matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite(load_data_dir + "/" + input_index + "/bf_matches.jpg", img_matches);

    // Extract corresponding patches from both images

    // Get the keypoints from the good matches
    std::vector<cv::Point2f> rgb_kps_good;
    std::vector<cv::Point2f> ir_kps_good;
    for (size_t i = 0; i < matches.size(); i++) {
        rgb_kps_good.push_back(rgb_kps[matches[i].queryIdx].pt);
        ir_kps_good.push_back(ir_kps[matches[i].trainIdx].pt);
    }

    const int width = rgb_image.cols;
    const int height = rgb_image.rows;
    const int patch_size_ = 80;
    const int patch_half = patch_size_ / 2;

    // Find homography between the two images

    // FIRST using RANSAC to filter outliers

    double ransacReprojThreshold = 100;
    const int maxIters = 20000000;
    const double confidence = 0.75;

    cv::Mat inlier_mask_init(cv::Size(1, matches.size()), CV_8U);

    cv::Mat H_init = cv::findHomography(rgb_kps_good, ir_kps_good, cv::RANSAC, ransacReprojThreshold, inlier_mask_init, maxIters, confidence);

    // Warp IR patch back to RGB space
    cv::Mat ir_image_warped(cv::Size(width, height), CV_8U);
    cv::warpPerspective(ir_image, ir_image_warped, H_init, ir_image_warped.size(), cv::INTER_NEAREST+cv::WARP_INVERSE_MAP);
    cv::imwrite(load_data_dir + "/" + input_index + "/ir_warped_init_H.jpg", ir_image_warped);

    // Combine both images (using the colored RGB and false-colored IR image for better contrast)
    cv::Mat overlap_init(cv::Size(width, height), CV_8U);
    cv::Mat ir_image_fc_warped_init(cv::Size(width, height), CV_8U);
    cv::warpPerspective(ir_image_fc, ir_image_fc_warped_init, H_init, ir_image_fc_warped_init.size(), cv::INTER_NEAREST+cv::WARP_INVERSE_MAP);
    overlap_init = 0.3 * ir_image_fc_warped_init + 0.7 * rgb_image_c;
    cv::imwrite(load_data_dir + "/" + input_index + "/overlap_init_H.jpg", overlap_init);

    // Save homography transformation to disk
    cv::FileStorage file_init(load_data_dir + "/" + input_index + "/homography_init.txt", cv::FileStorage::WRITE);
    file_init << "H_init" << H_init;

    std::cout << "Initial H = " << H_init << std::endl;

    std::vector<cv::DMatch> inlier_matches;
    std::vector<cv::DMatch> outlier_matches;
    for (int i = 0; i < matches.size(); i++) {
        if (inlier_mask_init.at<bool>(i) == 1){
            inlier_matches.push_back(matches[i]);
        } else {
            outlier_matches.push_back(matches[i]);
        }
    }

    std::vector<cv::Point2f> rgb_inliers;
    std::vector<cv::Point2f> rgb_outliers;
    std::vector<cv::Point2f> ir_inliers;
    std::vector<cv::Point2f> ir_outliers;
    for (int i = 0; i < inlier_matches.size(); i++) {
        rgb_inliers.push_back(rgb_kps[inlier_matches[i].queryIdx].pt);
        ir_inliers.push_back(ir_kps[inlier_matches[i].trainIdx].pt);
    }

    for (int i = 0; i < outlier_matches.size(); i++) {
        rgb_outliers.push_back(rgb_kps[outlier_matches[i].queryIdx].pt);
        ir_outliers.push_back(ir_kps[outlier_matches[i].trainIdx].pt);
    }

    std::cout << "Initial RANSAC gives Inliers: " << inlier_matches.size() << ", Outliers: " << outlier_matches.size() << std::endl;

    // Save inliers
    cv::Mat img_inliers_init;
    drawMatches(rgb_image, rgb_kps, ir_image, ir_kps,
                inlier_matches, img_inliers_init, cv::Scalar::all(-1), cv::Scalar::all(-1),
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite(load_data_dir + "/" + input_index + "/ransac_inliers_init.jpg", img_inliers_init);

    // Save outliers
    cv::Mat img_outliers_init;
    drawMatches(rgb_image, rgb_kps, ir_image, ir_kps,
                outlier_matches, img_outliers_init, cv::Scalar::all(-1), cv::Scalar::all(-1),
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite(load_data_dir + "/" + input_index + "/ransac_outliers_init.jpg", img_outliers_init);



    // SECOND using LMEDS with more constraints and inliers only
    ransacReprojThreshold = 3;
    cv::Mat inlier_mask_refined(cv::Size(1, rgb_inliers.size()), CV_8U);
    cv::Mat H = cv::findHomography(rgb_inliers, ir_inliers, cv::LMEDS, ransacReprojThreshold, inlier_mask_refined);

    // Warp IR patch back to RGB space
    cv::warpPerspective(ir_image, ir_image_warped, H, ir_image_warped.size(), cv::INTER_NEAREST+cv::WARP_INVERSE_MAP);
    cv::imwrite(load_data_dir + "/" + input_index + "/ir_warped_refined_H.jpg", ir_image_warped);

    // Combine both images (using the colored RGB and false-colored IR image for better contrast)
    cv::Mat overlap(cv::Size(width, height), CV_8U);
    cv::Mat ir_image_fc_warped(cv::Size(width, height), CV_8U);
    cv::warpPerspective(ir_image_fc, ir_image_fc_warped, H, ir_image_fc_warped.size(), cv::INTER_NEAREST+cv::WARP_INVERSE_MAP);
    overlap = 0.3 * ir_image_fc_warped + 0.7 * rgb_image_c;
    cv::imwrite(load_data_dir + "/" + input_index + "/overlap_refined_H.jpg", overlap);

    // Save homography transformation to disk
    cv::FileStorage file_refined(load_data_dir + "/" + input_index + "/homography_refined.txt", cv::FileStorage::WRITE);
    file_refined << "H" << H;

    std::cout << "Refined H = " << H << std::endl;

    std::vector<cv::DMatch> inlier_matches_refined;
    std::vector<cv::DMatch> outlier_matches_refined;
    for (int i = 0; i < rgb_inliers.size(); i++) {
        if (inlier_mask_refined.at<bool>(i) == 1){
            inlier_matches_refined.push_back(inlier_matches[i]);
        } else {
            outlier_matches_refined.push_back(inlier_matches[i]);
        }
    }

    rgb_inliers.clear();
    rgb_outliers.clear();
    ir_inliers.clear();
    ir_outliers.clear();
    for (int i = 0; i < inlier_matches_refined.size(); i++) {
        rgb_inliers.push_back(rgb_kps[inlier_matches_refined[i].queryIdx].pt);
        ir_inliers.push_back(ir_kps[inlier_matches_refined[i].trainIdx].pt);
    }

    for (int i = 0; i < outlier_matches_refined.size(); i++) {
        rgb_outliers.push_back(rgb_kps[outlier_matches_refined[i].queryIdx].pt);
        ir_outliers.push_back(ir_kps[outlier_matches_refined[i].trainIdx].pt);
    }

    std::cout << "Refining LMEDS gives Inliers: " << inlier_matches_refined.size() << ", Outliers: " << outlier_matches_refined.size() << std::endl;

    // Save inliers
    cv::Mat img_inliers_refined;
    drawMatches(rgb_image, rgb_kps, ir_image, ir_kps,
                inlier_matches_refined, img_inliers_refined, cv::Scalar::all(-1), cv::Scalar::all(-1),
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite(load_data_dir + "/" + input_index + "/lmeds_inliers_refined.jpg", img_inliers_refined);

    // Save outliers
    cv::Mat img_outliers_refined;
    drawMatches(rgb_image, rgb_kps, ir_image, ir_kps,
                outlier_matches_refined, img_outliers_refined, cv::Scalar::all(-1), cv::Scalar::all(-1),
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite(load_data_dir + "/" + input_index + "/lmeds_outliers_refined.jpg", img_outliers_refined);


    // Get the patches for each inlier
    int i = 0;
    for (auto const& p: rgb_inliers) {
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
        cv::imwrite(load_data_dir + "/" +input_index + "/inlier_patches/rgb_patch_" + std::to_string(i) + ".jpg", rgb_patch);
        cv::Mat ir_patch = ir_image_warped(patch_roi);
        cv::imwrite(load_data_dir + "/" +input_index + "/inlier_patches/ir_patch_" + std::to_string(i) + ".jpg", ir_patch);

        // Combine both patches into one image
        cv::Mat combined_patches(cv::Size(rgb_patch.rows, 2 * rgb_patch.cols), CV_8U);
        cv::hconcat(rgb_patch, ir_patch, combined_patches);
        cv::imwrite(load_data_dir + "/" +input_index + "/inlier_patches/patches_" + std::to_string(i) + ".jpg", combined_patches);
    }

    i = 0;
    for (auto const& p: rgb_outliers) {
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
        cv::imwrite(load_data_dir + "/" +input_index + "/outlier_patches/rgb_patch_" + std::to_string(i) + ".jpg", rgb_patch);
        cv::Mat ir_patch = ir_image_warped(patch_roi);
        cv::imwrite(load_data_dir + "/" +input_index + "/outlier_patches/ir_patch_" + std::to_string(i) + ".jpg", ir_patch);

        // Combine both patches into one image
        cv::Mat combined_patches(cv::Size(rgb_patch.rows, 2 * rgb_patch.cols), CV_8U);
        cv::hconcat(rgb_patch, ir_patch, combined_patches);
        cv::imwrite(load_data_dir + "/" +input_index + "/outlier_patches/patches_" + std::to_string(i) + ".jpg", combined_patches);
    }

}

int main(int argc, char *argv[]) {
    std::vector<std::string> data_index_vec = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"};
    std::string method = "lghd"; // choose lghd, sift or brisk
    ;
    for (auto const& idx : data_index_vec){
        run(idx, method);
    }

    return 0;
}
