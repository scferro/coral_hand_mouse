#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <string>

namespace DepthVision {
    // Forward declaration of logger
    class Logger {
    public:
        enum class Level { INFO, WARNING, ERROR };
        
        static void log(Level level, const std::string& message);
    };

    // Application configuration structure
    struct ApplicationConfig {
        // Configuration parameters with default values
        int camera_index = 0;
        cv::Size combined_size{1280, 480};
        cv::Size single_size{640, 480};
        
        // Depth estimation parameters
        float min_depth = 0.1f;
        float max_depth = 4.0f;
        
        // Visualization flags
        bool show_depth = true;
        bool show_raw = true;
        
        // Calibration file path (added here to resolve scope issue)
        std::string calibration_path = "../data/stereo_calibration.json";
        
        // JSON storage for calibration data
        nlohmann::json calibration_data;

        // Method to load calibration
        bool loadCalibration() {
            try {
                // Open the calibration file
                std::ifstream file(calibration_path);
                
                // Check if file is open
                if (!file.is_open()) {
                    Logger::log(Logger::Level::ERROR, 
                        "Failed to open calibration file: " + calibration_path);
                    return false;
                }

                // Parse JSON file
                calibration_data = nlohmann::json::parse(file);
                
                // Validate key sections
                if (!calibration_data.contains("left_camera") || 
                    !calibration_data.contains("right_camera") ||
                    !calibration_data.contains("stereo")) {
                    Logger::log(Logger::Level::ERROR, 
                        "Invalid calibration file structure");
                    return false;
                }

                return true;
            } catch (const std::exception& e) {
                Logger::log(Logger::Level::ERROR, 
                    "Calibration loading error: " + std::string(e.what()));
                return false;
            }
        }
    };

    // Implement Logger method outside the class declaration
    inline void Logger::log(Level level, const std::string& message) {
        switch (level) {
            case Level::INFO:
                std::cout << "[INFO] " << message << std::endl;
                break;
            case Level::WARNING:
                std::cerr << "[WARNING] " << message << std::endl;
                break;
            case Level::ERROR:
                std::cerr << "[ERROR] " << message << std::endl;
                break;
        }
    }
}