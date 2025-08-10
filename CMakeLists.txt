cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(photo-slam LANGUAGES CXX CUDA)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CAFFE2_USE_CUDNN on)
set(Torch_DIR /home/jiang/envs/libtorch-cu118/share/cmake/Torch)
set(ONNXRUNTIME_ROOT /home/jiang/envs/onnxruntime)

find_package(Python3  REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Threads REQUIRED)

# set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE INTERNAL "No dev warnings") # 关闭pcl烦人的警告
find_package(PCL REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/submodules/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
    )

##################################################################################
##  toolbox
##################################################################################

add_library(cuda_colorize STATIC
    cuda_colorize/colorize_pointcloud.h
    cuda_colorize/colorize_pointcloud.cu
)

set_target_properties(cuda_colorize PROPERTIES CUDA_ARCHITECTURES "75;86")

target_include_directories(cuda_colorize PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(cuda_colorize
    "${TORCH_LIBRARIES}"
    "${CUDA_LIBRARIES}"
)