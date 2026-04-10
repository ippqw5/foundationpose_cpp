cmake_minimum_required(VERSION 3.17)
project(foundationpose_cpp_bindings LANGUAGES CXX)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate pybind11 (installed via pip)
# ---------------------------------------------------------------------------
find_package(pybind11 REQUIRED)

# ---------------------------------------------------------------------------
# Locate OpenCV
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

# ---------------------------------------------------------------------------
# Paths to pre-built C++ libraries (no CUDA re-compile)
# ---------------------------------------------------------------------------
set(FOUNDATIONPOSE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(PREBUILT_LIB_DIR   "${FOUNDATIONPOSE_ROOT}/build/lib")

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
include_directories(
    ${FOUNDATIONPOSE_ROOT}/detection_6d_foundationpose/include
    ${FOUNDATIONPOSE_ROOT}/easy_deploy_tool/deploy_core/include
    ${FOUNDATIONPOSE_ROOT}/easy_deploy_tool/inference_core/trt_core/include
    /usr/include/eigen3
    ${OpenCV_INCLUDE_DIRS}
)

# ---------------------------------------------------------------------------
# Imported pre-built shared libraries
# ---------------------------------------------------------------------------
add_library(detection_6d_foundationpose SHARED IMPORTED)
set_target_properties(detection_6d_foundationpose PROPERTIES
    IMPORTED_LOCATION "${FOUNDATIONPOSE_ROOT}/build/detection_6d_foundationpose/libdetection_6d_foundationpose.so"
)

add_library(trt_core SHARED IMPORTED)
set_target_properties(trt_core PROPERTIES
    IMPORTED_LOCATION "${PREBUILT_LIB_DIR}/libtrt_core.so"
)

add_library(deploy_core SHARED IMPORTED)
set_target_properties(deploy_core PROPERTIES
    IMPORTED_LOCATION "${PREBUILT_LIB_DIR}/libdeploy_core.so"
)

# ---------------------------------------------------------------------------
# pybind11 extension module
# ---------------------------------------------------------------------------
pybind11_add_module(_foundationpose_cpp
    src/bindings.cpp
)

target_link_libraries(_foundationpose_cpp PRIVATE
    detection_6d_foundationpose
    trt_core
    deploy_core
    ${OpenCV_LIBS}
)

# Make the .so land next to the Python package
set_target_properties(_foundationpose_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/foundationpose_cpp"
)
