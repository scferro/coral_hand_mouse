cmake_minimum_required(VERSION 3.12)
project(StereoDepthVision CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED COMPONENTS 
    core 
    imgproc 
    highgui 
    videoio 
    calib3d 
    ximgproc
)
find_package(nlohmann_json 3.2.0 REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Main executable
add_executable(stereo_depth_vision 
    main.cpp
    src/CameraDevice.cpp
    src/DepthProcessor.cpp
)

# Link libraries
target_link_libraries(stereo_depth_vision 
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)