# CLAUDE.md

This file provides guidance for AI assistants working with the `foundationpose_cpp` codebase.

## Project Overview

`foundationpose_cpp` is a C++ implementation of the **FoundationPose** 6D object pose estimation algorithm. It is adapted from [nvidia-isaac-pose-estimation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation) with simplified dependencies, enabling TensorRT inference from ONNX models exported by the [Python FoundationPose](https://github.com/NVlabs/FoundationPose) implementation.

The project supports two core operations:
- **Register**: Initialize pose estimation given an RGB image, depth map, and segmentation mask.
- **Track**: Continuously refine pose using a prior pose hypothesis across frames.

Poses are output in **mesh coordinate space** (`Eigen::Matrix4f`).

## Repository Layout

```
foundationpose_cpp/
├── CMakeLists.txt                      # Top-level CMake (adds 3 subdirectories)
├── detection_6d_foundationpose/        # Core FoundationPose algorithm library
│   ├── CMakeLists.txt
│   ├── include/detection_6d_foundationpose/
│   │   ├── foundationpose.hpp          # Public API: Base6DofDetectionModel interface
│   │   └── mesh_loader.hpp             # BaseMeshLoader interface
│   └── src/
│       ├── foundationpose.cpp          # Main algorithm implementation (class FoundationPose)
│       ├── foundationpose_render.cpp/cu # CUDA rendering pipeline (nvdiffrast-based)
│       ├── foundationpose_sampling.cpp/cu # Pose hypothesis sampling
│       ├── foundationpose_utils.cu     # CUDA utility kernels
│       ├── foundationpose_decoder.cu   # Output decoding kernels
│       └── nvdiffrast/                 # Embedded nvdiffrast rasterizer
│           └── common/cudaraster/      # CUDA rasterization implementation
├── easy_deploy_tool/                   # Inference abstraction submodule (EasyDeploy)
│   ├── deploy_core/                    # BaseInferCore and pipeline abstractions
│   ├── inference_core/                 # TensorRT / ONNX-Runtime / RKNN backends
│   │   └── trt_core/                   # TensorRT inference core (CreateTrtInferCore)
│   └── deploy_utils/                   # Shared utilities
├── simple_tests/                       # GTest integration tests
│   └── src/test_foundationpose.cpp     # End-to-end Register + Track test
├── python_bindings/                    # pybind11 Python bindings
│   ├── CMakeLists.txt                  # Builds _foundationpose_cpp.so (links prebuilt .so)
│   ├── src/bindings.cpp               # pybind11 glue (numpy ↔ cv::Mat ↔ Eigen)
│   ├── foundationpose_cpp/
│   │   ├── __init__.py                # Re-exports C++ bindings + FoundationPose class
│   │   └── model.py                   # High-level Pythonic FoundationPose class
│   └── demo_mustard.py                # End-to-end demo on mustard0 dataset
├── models/                             # TensorRT engine files (not in git)
│   ├── refiner_hwc_dynamic_fp16.engine
│   └── scorer_hwc_dynamic_fp16.engine
├── test_data/mustard0/                 # Public mustard dataset (not in git)
│   ├── mesh/textured_simple.obj
│   ├── mesh/texture_map.png
│   └── cam_K.txt
└── tools/cvt_onnx2trt.bash            # ONNX → TensorRT conversion script
```

## Build System

We have already installed the required dependencies in the local environment (not have to use Docker). And we have already converted the ONNX models to TensorRT engines and placed them in `models/`. You can directly build and run the code.

### Build Commands

```bash

cmake -B build -S . -DENABLE_TENSORRT=ON ..
cmake --build build -j32
```

Build output goes to `build/bin/` and `build/lib/`.

### Run Tests

```bash
cd ./build

# Runs the end-to-end Register + Track test on the mustard0 dataset
./bin/simple_tests --gtest_filter=foundationpose_test.test 

# Runs the speed benchmark for the Register func of FoundationPose
./bin/simple_tests --gtest_filter=foundationpose_test.speed_register

# Runs the speed benchmark for the Track func of FoundationPose
./bin/simple_tests --gtest_filter=foundationpose_test.speed_track
```
## Python Bindings

The `python_bindings/` directory exposes the C++/CUDA library to Python via **pybind11**. The extension module (`_foundationpose_cpp.so`) links against the pre-built `.so` files in `build/` — no CUDA recompile is needed after the initial C++ build.

### Build Python Bindings

The package uses a `pyproject.toml` + `setup.py` with a custom `CMakeBuild` that invokes CMake under the hood. The `.so` is written directly into `foundationpose_cpp/` (via `LIBRARY_OUTPUT_DIRECTORY`), so editable installs work without a separate copy step.

```bash
# Step 1: Build the C++ library (if not already done)
cd /home/isaacsim/code/foundationpose_cpp
mkdir -p build && cd build && cmake -DENABLE_TENSORRT=ON .. && make -j32

# Step 2: Install the Python package in `env_isaaclab_fp` conda env.
conda activate env_isaaclab_fp
cd /home/isaacsim/code/foundationpose_cpp/python_bindings
pip install -e . --no-build-isolation

# Step 3: Verify
python3 -c "import foundationpose_cpp; print('OK')"
```

> `--no-build-isolation` is required on this system (pip 22 + setuptools 59);
> avoid creating an isolated env that would need to re-download build deps.

### Run Demo

```bash
cd /home/isaacsim/code/foundationpose_cpp/python_bindings
python3 demo_mustard.py
```

### Python API

```python
from foundationpose_cpp import FoundationPose
import numpy as np

model = FoundationPose(
    refiner_engine="models/refiner_hwc_dynamic_fp16.engine",
    scorer_engine="models/scorer_hwc_dynamic_fp16.engine",
    mesh_path="test_data/mustard0/mesh/textured_simple.obj",
    mesh_name="mustard",
    intrinsic=K,          # 3×3 float32 numpy array
    crop_h=160, crop_w=160,
    max_h=1080, max_w=1920,
)

# Frame 0: register with RGB + depth + mask
pose = model.register(rgb, depth, mask)   # returns 4×4 float32 np.ndarray

# Subsequent frames: track with prior pose
pose = model.track(rgb, depth, pose)      # returns 4×4 float32 np.ndarray
```

All images are numpy arrays:
- `rgb`: `H×W×3 uint8`
- `depth`: `H×W float32` (metres)
- `mask`: `H×W uint8` (non-zero = object)

### Low-Level Bindings

`_foundationpose_cpp` exposes the following symbols directly:

| Symbol | Description |
|---|---|
| `create_trt_infer_core(engine, input_shapes, output_shapes, num_queue)` | Create TensorRT inference core |
| `create_mesh_loader(name, obj_path)` | Create Assimp mesh loader |
| `create_model(refiner, scorer, mesh_loaders, K, max_h, max_w, crop_h, crop_w, min_depth)` | Create FoundationPose model |
| `FoundationPoseHandle.register(rgb, depth, mask, name, refine_itr)` | Register pose |
| `FoundationPoseHandle.track(rgb, depth, prior_pose, name, refine_itr)` | Track pose |

---

## Key API

### Public Interface (`foundationpose.hpp`)

```cpp
namespace detection_6d {

// Factory function
std::shared_ptr<Base6DofDetectionModel> CreateFoundationPoseModel(
    std::shared_ptr<inference_core::BaseInferCore> refiner_core,
    std::shared_ptr<inference_core::BaseInferCore> scorer_core,
    const std::vector<std::shared_ptr<BaseMeshLoader>>& mesh_loaders,
    const Eigen::Matrix3f& intrinsic,
    int max_input_image_H = 1080,
    int max_input_image_W = 1920,
    int crop_window_H = 160,
    int crop_window_W = 160,
    float min_depth = 0.001f);

// Methods
bool Register(const cv::Mat& rgb, const cv::Mat& depth, const cv::Mat& mask,
              const std::string& target_name, Eigen::Matrix4f& out_pose_in_mesh,
              size_t refine_itr = 1);

bool Track(const cv::Mat& rgb, const cv::Mat& depth,
           const Eigen::Matrix4f& hyp_pose_in_mesh,
           const std::string& target_name, Eigen::Matrix4f& out_pose_in_mesh,
           size_t refine_itr = 1);
}
```

### Mesh Loader (`mesh_loader.hpp`)

```cpp
// Factory function for Assimp-based loader
std::shared_ptr<BaseMeshLoader> CreateAssimpMeshLoader(
    const std::string& name,
    const std::string& obj_path);
```

### TensorRT Inference Core (`easy_deploy_tool/inference_core/trt_core/`)

```cpp
auto core = CreateTrtInferCore(engine_path,
    {{"transf_input", {252, 160, 160, 6}}, {"render_input", {252, 160, 160, 6}}},
    {{"trans", {252, 3}}, {"rot", {252, 3}}},
    /*num_queue=*/1);
```

## Model Information

The pipeline uses two TensorRT engine models:

| Model | Input Blobs | Output Blobs | Shape |
|---|---|---|---|
| **Refiner** | `transf_input`, `render_input` | `trans`, `rot` | `{252, 160, 160, 6}` → `{252, 3}` |
| **Scorer** | `transf_input`, `render_input` | `scores` | `{252, 160, 160, 6}` → `{252, 1}` |

ONNX models are converted to TensorRT engines via:
```bash
bash tools/cvt_onnx2trt.bash
```

## Internal Pipeline Architecture

The `FoundationPose` class (internal to `foundationpose.cpp`) implements:

1. **UploadDataToDevice** — copies RGB/depth/mask to GPU
2. **RefinePreProcess** — renders hypothesis poses, prepares input tensors
3. **Refiner inference** — runs TensorRT refiner model (pose delta prediction)
4. **RefinePostProcess** — applies predicted deltas to hypotheses
5. **ScorePreprocess** — renders refined poses for scoring
6. **Scorer inference** — runs TensorRT scorer model (hypothesis ranking)
7. **ScorePostProcess / TrackPostProcess** — selects best hypothesis

The rendering subsystem is based on an embedded **nvdiffrast** rasterizer (`src/nvdiffrast/`), with CUDA kernels for interpolation, texture sampling, and rasterization.

## Coordinate System

- All output poses (`out_pose_in_mesh`) are expressed in **mesh coordinate space** as `Eigen::Matrix4f` (4×4 homogeneous transformation matrix).
- Camera intrinsics are passed as `Eigen::Matrix3f` (standard K matrix).

## Code Style & Conventions

- Namespaces: `detection_6d`, `inference_core`
- Smart pointers: `std::shared_ptr<>` used throughout; factory functions return shared pointers
- Error handling: `CHECK()` macros from glog for fatal assertions; methods return `bool` for recoverable failures
- CUDA files use `.cu` extension; CUDA headers use `.cu.hpp` convention
- Mixed HWC image layout (Height × Width × Channels) as indicated by model names (`hwc`)

