#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "../include/config.h"
#include "CameraDevice.h"
#include "DepthProcessor.h"

// Command-line argument parsing structure
struct ProgramOptions {
    bool show_depth = true;
    bool verbose = false;
};

// Function to parse command-line arguments
ProgramOptions parseArguments(int argc, char* argv[]) {
    ProgramOptions options;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-depth") {
            options.show_depth = false;
        }
        if (arg == "--verbose" || arg == "-v") {
            options.verbose = true;
        }
        if (arg == "--help" || arg == "-h") {
            std::cout << "Stereo Depth Vision Application\n"
                      << "Usage: ./stereo_depth_vision [options]\n"
                      << "Options:\n"
                      << "  --no-depth     Disable depth map display\n"
                      << "  --verbose, -v  Enable verbose logging\n"
                      << "  --help, -h     Show this help message\n";
            exit(0);
        }
    }
    
    return options;
}

int main(int argc, char* argv[]) {
    try {
        // Parse command-line arguments
        ProgramOptions options = parseArguments(argc, argv);
        
        // Create application configuration
        DepthVision::ApplicationConfig config;
        
        // Verbose logging if requested
        if (options.verbose) {
            DepthVision::Logger::log(
                DepthVision::Logger::Level::INFO, 
                "Initializing Stereo Depth Vision Application"
            );
        }
        
        // Initialize camera device
        DepthVision::CameraDevice camera(config);
        
        // Initialize depth processor
        DepthVision::DepthProcessor depth_processor(config);
        
        // Create windows for visualization
        cv::namedWindow("Stereo Feed", cv::WINDOW_NORMAL);
        cv::namedWindow("Depth Map", cv::WINDOW_NORMAL);
        cv::resizeWindow("Stereo Feed", 1280, 480);
        cv::resizeWindow("Depth Map", 640, 480);
        
        // Performance tracking
        int frame_count = 0;
        double start_time = cv::getTickCount();  // Corrected from cv2 to cv
        
        // Main processing loop
        while (true) {
            // Capture stereo frames
            auto [left_frame, right_frame] = camera.captureFrames();
            
            // Validate frame capture
            if (left_frame.empty() || right_frame.empty()) {
                DepthVision::Logger::log(
                    DepthVision::Logger::Level::ERROR, 
                    "Failed to capture frames"
                );
                break;
            }
            
            // Undistort frames
            std::tie(left_frame, right_frame) = camera.undistortFrames(left_frame, right_frame);
            
            // Compute depth using Q matrix from camera calibration
            auto [depth_map, depth_vis, invalid_mask] = 
                depth_processor.computeDepth(
                    left_frame, 
                    right_frame, 
                    camera.getQMatrix()  // New method to retrieve Q matrix
                );
            
            // Create combined stereo visualization
            cv::Mat stereo_feed;
            cv::hconcat(std::vector<cv::Mat>{left_frame, right_frame}, stereo_feed);
            
            // Display results
            cv::imshow("Stereo Feed", stereo_feed);
            
            // Only show depth map if visualization is enabled
            if (options.show_depth) {
                cv::imshow("Depth Map", depth_vis);
            }
            
            // Performance tracking
            frame_count++;
            if (frame_count % 30 == 0) {
                double end_time = cv::getTickCount();  // Corrected from cv2 to cv
                double fps = frame_count / ((end_time - start_time) / cv::getTickFrequency());
                
                if (options.verbose) {
                    DepthVision::Logger::log(
                        DepthVision::Logger::Level::INFO, 
                        "FPS: " + std::to_string(fps)
                    );
                }
                
                // Reset counters
                frame_count = 0;
                start_time = end_time;
            }
            
            // Handle user input
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) {  // 'q' or ESC key
                break;
            }
            else if (key == 'd') {  // Toggle depth map
                options.show_depth = !options.show_depth;
            }
        }
        
        // Clean up resources
        camera.release();
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        DepthVision::Logger::log(
            DepthVision::Logger::Level::ERROR, 
            std::string("Fatal error: ") + e.what()
        );
        return -1;
    }
    
    return 0;
}