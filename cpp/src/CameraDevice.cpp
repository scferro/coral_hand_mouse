#include "CameraDevice.h"

namespace DepthVision {
    CameraDevice::CameraDevice(const ApplicationConfig& config) : config_(config) {
        // First, load calibration
        if (!config_.loadCalibration()) {
            Logger::log(Logger::Level::WARNING, 
                "Using uncalibrated camera settings");
            return;
        }

        // Open camera
        camera_.open(config.camera_index, cv::CAP_V4L2);
        
        if (!isOpen()) {
            Logger::log(Logger::Level::ERROR, 
                "Failed to open camera with index " + std::to_string(config.camera_index));
            throw std::runtime_error("Camera initialization failed");
        }
        
        // Set camera properties
        camera_.set(cv::CAP_PROP_FRAME_WIDTH, config.combined_size.width);
        camera_.set(cv::CAP_PROP_FRAME_HEIGHT, config.combined_size.height);
        
        // Convert JSON matrices to OpenCV matrices
        auto& calib_data = config_.calibration_data;

        // Left camera matrices
        camera_matrix_left_ = jsonToMat(calib_data["left_camera"]["camera_matrix"]);
        dist_coeffs_left_ = jsonToMat(calib_data["left_camera"]["dist_coeffs"]);

        // Right camera matrices
        camera_matrix_right_ = jsonToMat(calib_data["right_camera"]["camera_matrix"]);
        dist_coeffs_right_ = jsonToMat(calib_data["right_camera"]["dist_coeffs"]);

        // Q matrix for depth computation
        Q_matrix_ = jsonToMat(calib_data["stereo"]["Q"]);

        // Log calibration details
        Logger::log(Logger::Level::INFO, "Camera calibration loaded successfully");
    }
    
    cv::Mat CameraDevice::jsonToMat(const nlohmann::json& json_matrix) {
        // Convert JSON array to OpenCV matrix
        cv::Mat matrix(json_matrix.size(), json_matrix[0].size(), CV_64F);
        
        for (size_t i = 0; i < json_matrix.size(); ++i) {
            for (size_t j = 0; j < json_matrix[i].size(); ++j) {
                matrix.at<double>(i, j) = json_matrix[i][j].get<double>();
            }
        }
        
        return matrix;
    }
    
    std::pair<cv::Mat, cv::Mat> CameraDevice::captureFrames() {
        cv::Mat full_frame;
        if (!camera_.read(full_frame)) {
            Logger::log(Logger::Level::ERROR, "Failed to capture frame");
            return {cv::Mat(), cv::Mat()};
        }
        
        // Split frame into left and right
        int mid = full_frame.cols / 2;
        cv::Mat left_frame = full_frame(cv::Rect(0, 0, mid, full_frame.rows));
        cv::Mat right_frame = full_frame(cv::Rect(mid, 0, mid, full_frame.rows));
        
        return {left_frame, right_frame};
    }
    
    std::pair<cv::Mat, cv::Mat> CameraDevice::undistortFrames(
        const cv::Mat& left_frame, 
        const cv::Mat& right_frame
    ) {
        // If no calibration data, return original frames
        if (camera_matrix_left_.empty() || camera_matrix_right_.empty()) {
            return {left_frame, right_frame};
        }
        
        cv::Mat left_undistorted, right_undistorted;
        cv::undistort(left_frame, left_undistorted, 
            camera_matrix_left_, dist_coeffs_left_);
        cv::undistort(right_frame, right_undistorted, 
            camera_matrix_right_, dist_coeffs_right_);
        
        return {left_undistorted, right_undistorted};
    }
    
    bool CameraDevice::isOpen() const {
        return camera_.isOpened();
    }
    
    void CameraDevice::release() {
        camera_.release();
    }
}