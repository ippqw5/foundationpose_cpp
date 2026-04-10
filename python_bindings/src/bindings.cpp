/**
 * python_bindings/src/bindings.cpp
 *
 * pybind11 bindings for the FoundationPose C++/CUDA library.
 * Exposes:
 *   - create_trt_infer_core(...)      -> opaque BaseInferCore handle
 *   - create_mesh_loader(name, path)  -> opaque BaseMeshLoader handle
 *   - create_model(...)               -> FoundationPoseHandle
 *   - FoundationPoseHandle.register_object(...)
 *   - FoundationPoseHandle.track(...)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>

#include "detection_6d_foundationpose/foundationpose.hpp"
#include "detection_6d_foundationpose/mesh_loader.hpp"
#include "trt_core/trt_core.h"

namespace py = pybind11;
using namespace detection_6d;
using namespace inference_core;

// ---------------------------------------------------------------------------
// Helper: numpy (H×W×C uint8) → cv::Mat CV_8UC(C)
// ---------------------------------------------------------------------------
static cv::Mat numpy_uint8_to_mat(const py::array_t<uint8_t> &arr)
{
    py::buffer_info buf = arr.request();
    if (buf.ndim == 3) {
        int H = (int)buf.shape[0];
        int W = (int)buf.shape[1];
        int C = (int)buf.shape[2];
        int type = (C == 3) ? CV_8UC3 : CV_8UC1;
        // cv::Mat shares memory – clone to own data
        cv::Mat mat(H, W, type, buf.ptr);
        return mat.clone();
    } else if (buf.ndim == 2) {
        int H = (int)buf.shape[0];
        int W = (int)buf.shape[1];
        cv::Mat mat(H, W, CV_8UC1, buf.ptr);
        return mat.clone();
    }
    throw std::runtime_error("numpy_uint8_to_mat: expected 2D or 3D array");
}

// ---------------------------------------------------------------------------
// Helper: numpy (H×W) float32 → cv::Mat CV_32FC1
// ---------------------------------------------------------------------------
static cv::Mat numpy_float32_to_mat(const py::array_t<float> &arr)
{
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error("numpy_float32_to_mat: expected 2D array (H×W)");
    int H = (int)buf.shape[0];
    int W = (int)buf.shape[1];
    cv::Mat mat(H, W, CV_32FC1, buf.ptr);
    return mat.clone();
}

// ---------------------------------------------------------------------------
// Helper: Eigen::Matrix4f → numpy (4×4) float32
// ---------------------------------------------------------------------------
static py::array_t<float> eigen4f_to_numpy(const Eigen::Matrix4f &mat)
{
    // Return row-major 4×4 float32
    auto result = py::array_t<float>({4, 4});
    py::buffer_info buf = result.request();
    float *ptr = static_cast<float *>(buf.ptr);
    // Eigen is column-major; copy row by row
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            ptr[r * 4 + c] = mat(r, c);
    return result;
}

// ---------------------------------------------------------------------------
// Helper: numpy (3×3) float32 → Eigen::Matrix3f
// ---------------------------------------------------------------------------
static Eigen::Matrix3f numpy_to_eigen3f(const py::array_t<float> &arr)
{
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2 || buf.shape[0] != 3 || buf.shape[1] != 3)
        throw std::runtime_error("numpy_to_eigen3f: expected 3×3 float32 array");
    const float *ptr = static_cast<const float *>(buf.ptr);
    Eigen::Matrix3f mat;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            mat(r, c) = ptr[r * 3 + c];
    return mat;
}

// ---------------------------------------------------------------------------
// Helper: numpy (4×4) float32 → Eigen::Matrix4f
// ---------------------------------------------------------------------------
static Eigen::Matrix4f numpy_to_eigen4f(const py::array_t<float> &arr)
{
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2 || buf.shape[0] != 4 || buf.shape[1] != 4)
        throw std::runtime_error("numpy_to_eigen4f: expected 4×4 float32 array");
    const float *ptr = static_cast<const float *>(buf.ptr);
    Eigen::Matrix4f mat;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            mat(r, c) = ptr[r * 4 + c];
    return mat;
}

// ---------------------------------------------------------------------------
// Thin Python-facing wrapper around Base6DofDetectionModel
// ---------------------------------------------------------------------------
struct FoundationPoseHandle {
    std::shared_ptr<Base6DofDetectionModel> model;
    std::string                             mesh_name;

    /**
     * register_object(rgb, depth, mask, refine_itr=1) -> np.ndarray (4×4)
     * rgb:   H×W×3 uint8 (RGB order)
     * depth: H×W float32 (meters)
     * mask:  H×W uint8
     */
    py::array_t<float> register_object(
        const py::array_t<uint8_t> &rgb,
        const py::array_t<float>   &depth,
        const py::array_t<uint8_t> &mask,
        size_t                      refine_itr = 1)
    {
        cv::Mat cv_rgb   = numpy_uint8_to_mat(rgb);
        cv::Mat cv_depth = numpy_float32_to_mat(depth);
        cv::Mat cv_mask  = numpy_uint8_to_mat(mask);

        Eigen::Matrix4f out_pose;
        bool ok = model->Register(cv_rgb, cv_depth, cv_mask, mesh_name, out_pose, refine_itr);
        if (!ok)
            throw std::runtime_error("FoundationPose::Register failed");
        return eigen4f_to_numpy(out_pose);
    }

    /**
     * track(rgb, depth, prior_pose, refine_itr=1) -> np.ndarray (4×4)
     * prior_pose: 4×4 float32 numpy array
     */
    py::array_t<float> track(
        const py::array_t<uint8_t> &rgb,
        const py::array_t<float>   &depth,
        const py::array_t<float>   &prior_pose,
        size_t                      refine_itr = 1)
    {
        cv::Mat          cv_rgb   = numpy_uint8_to_mat(rgb);
        cv::Mat          cv_depth = numpy_float32_to_mat(depth);
        Eigen::Matrix4f  hyp      = numpy_to_eigen4f(prior_pose);

        Eigen::Matrix4f out_pose;
        bool ok = model->Track(cv_rgb, cv_depth, hyp, mesh_name, out_pose, refine_itr);
        if (!ok)
            throw std::runtime_error("FoundationPose::Track failed");
        return eigen4f_to_numpy(out_pose);
    }
};

// ---------------------------------------------------------------------------
// Opaque holders — pybind11 requires a public (or accessible) destructor.
// BaseInferCore has a *protected* destructor, so we wrap the shared_ptr
// inside a plain struct that has a trivially public destructor.
// ---------------------------------------------------------------------------
struct InferCoreHolder {
    std::shared_ptr<BaseInferCore> core;
};

struct MeshLoaderHolder {
    std::shared_ptr<BaseMeshLoader> loader;
};

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_foundationpose_cpp, m)
{
    m.doc() = "Python bindings for FoundationPose C++/CUDA 6D pose estimation";

    // ------------------------------------------------------------------
    // Opaque holder for BaseInferCore
    // ------------------------------------------------------------------
    py::class_<InferCoreHolder>(m, "InferCoreHolder");

    // ------------------------------------------------------------------
    // Opaque holder for BaseMeshLoader
    // ------------------------------------------------------------------
    py::class_<MeshLoaderHolder>(m, "MeshLoaderHolder")
        .def("get_name",
             [](const MeshLoaderHolder &h) { return h.loader->GetName(); });

    // ------------------------------------------------------------------
    // Factory: TensorRT inference core
    //
    // create_trt_infer_core(
    //     engine_path,
    //     input_shapes  = {"blob_name": [d0, d1, ...]},
    //     output_shapes = {"blob_name": [d0, d1, ...]},
    //     mem_buf_size  = 1,
    // ) -> InferCoreHolder
    // ------------------------------------------------------------------
    m.def("create_trt_infer_core",
        [](const std::string &engine_path,
           const std::unordered_map<std::string, std::vector<uint64_t>> &input_shapes,
           const std::unordered_map<std::string, std::vector<uint64_t>> &output_shapes,
           int mem_buf_size) -> InferCoreHolder
        {
            InferCoreHolder h;
            h.core = CreateTrtInferCore(engine_path, input_shapes, output_shapes, mem_buf_size);
            return h;
        },
        py::arg("engine_path"),
        py::arg("input_shapes")  = std::unordered_map<std::string, std::vector<uint64_t>>{},
        py::arg("output_shapes") = std::unordered_map<std::string, std::vector<uint64_t>>{},
        py::arg("mem_buf_size")  = 1,
        "Create a TensorRT inference core from a .engine file.");

    // ------------------------------------------------------------------
    // Factory: Assimp mesh loader
    // ------------------------------------------------------------------
    m.def("create_mesh_loader",
        [](const std::string &name, const std::string &mesh_path) -> MeshLoaderHolder
        {
            MeshLoaderHolder h;
            h.loader = CreateAssimpMeshLoader(name, mesh_path);
            return h;
        },
        py::arg("name"),
        py::arg("mesh_path"),
        "Create a mesh loader from an OBJ/PLY/STL file.");

    // ------------------------------------------------------------------
    // FoundationPoseHandle (wraps Base6DofDetectionModel)
    // ------------------------------------------------------------------
    py::class_<FoundationPoseHandle>(m, "FoundationPoseHandle")
        .def("register_object", &FoundationPoseHandle::register_object,
             py::arg("rgb"),
             py::arg("depth"),
             py::arg("mask"),
             py::arg("refine_itr") = 1,
             "Run pose registration. Returns 4×4 float32 pose in mesh space.")
        .def("track", &FoundationPoseHandle::track,
             py::arg("rgb"),
             py::arg("depth"),
             py::arg("prior_pose"),
             py::arg("refine_itr") = 1,
             "Run pose tracking. Returns 4×4 float32 pose in mesh space.");

    // ------------------------------------------------------------------
    // Factory: FoundationPose model
    //
    // create_model(
    //     refiner_core, scorer_core,
    //     mesh_loaders,
    //     intrinsic,                    # 3×3 float32 numpy
    //     mesh_name,
    //     max_h=1080, max_w=1920,
    //     crop_h=160, crop_w=160,
    // ) -> FoundationPoseHandle
    // ------------------------------------------------------------------
    m.def("create_model",
        [](const InferCoreHolder                  &refiner_holder,
           const InferCoreHolder                  &scorer_holder,
           const std::vector<MeshLoaderHolder>    &mesh_holder_list,
           const py::array_t<float>               &intrinsic,
           const std::string                      &mesh_name,
           int max_h, int max_w,
           int crop_h, int crop_w) -> FoundationPoseHandle
        {
            // Unwrap holders
            std::vector<std::shared_ptr<BaseMeshLoader>> mesh_loaders;
            mesh_loaders.reserve(mesh_holder_list.size());
            for (const auto &mh : mesh_holder_list)
                mesh_loaders.push_back(mh.loader);

            Eigen::Matrix3f K = numpy_to_eigen3f(intrinsic);
            auto model = CreateFoundationPoseModel(
                refiner_holder.core, scorer_holder.core, mesh_loaders, K, max_h, max_w);
            FoundationPoseHandle handle;
            handle.model     = model;
            handle.mesh_name = mesh_name;
            return handle;
        },
        py::arg("refiner_core"),
        py::arg("scorer_core"),
        py::arg("mesh_loaders"),
        py::arg("intrinsic"),
        py::arg("mesh_name"),
        py::arg("max_h")   = 1080,
        py::arg("max_w")   = 1920,
        py::arg("crop_h")  = 160,
        py::arg("crop_w")  = 160,
        "Create a FoundationPose model. Returns a FoundationPoseHandle.");
}
