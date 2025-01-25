#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "../include/config.h"

namespace DepthVision {
    class CameraDevice {
    public:
        // Constructor with configuration
        explicit CameraDevice(const ApplicationConfig& config);
        
        // Capture stereo frames
        std::pair<cv::Mat, cv::Mat> captureFrames();
        
        // Apply camera calibration/undistortion
        std::pair<cv::Mat, cv::Mat> undistortFrames(
            const cv::Mat& left_frame, 
            const cv::Mat& right_frame
        );
        
        // Getters for key calibration matrices
        cv::Mat getQ() const { return Q_matrix_; }
        
        // Check if camera is operational
        bool isOpen() const;
        
        // Release camera resources
        void release();

    cv::Mat getQMatrix() const { return Q_matrix_; }

    private:
        // Convert JSON matrix to OpenCV matrix
        cv::Mat jsonToMat(const nlohmann::json& json_matrix);

        cv::VideoCapture camera_;
        cv::Mat camera_matrix_left_;
        cv::Mat dist_coeffs_left_;
        cv::Mat camera_matrix_right_;
        cv::Mat dist_coeffs_right_;
        cv::Mat Q_matrix_;
        
        ApplicationConfig config_;
    };
}