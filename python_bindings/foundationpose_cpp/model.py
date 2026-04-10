"""
foundationpose_cpp/model.py

High-level Pythonic FoundationPose class.
Wraps the raw C++ extension (_foundationpose_cpp) with a clean interface.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from . import _foundationpose_cpp as _cpp
# _cpp exposes: InferCoreHolder, MeshLoaderHolder, FoundationPoseHandle,
#               create_trt_infer_core, create_mesh_loader, create_model


class FoundationPose:
    """6D object pose estimator backed by FoundationPose C++/CUDA.

    Parameters
    ----------
    refiner_engine:
        Path to the TensorRT refiner ``.engine`` file.
    scorer_engine:
        Path to the TensorRT scorer ``.engine`` file.
    mesh_path:
        Path to the object mesh (OBJ/PLY/STL).
    mesh_name:
        Identifier string for the mesh (used internally for look-ups).
    intrinsic:
        3×3 float32 numpy camera intrinsic matrix K.
    crop_h, crop_w:
        Crop window size expected by the TRT model (default 160×160).
    max_h, max_w:
        Maximum input image size (default 1080×1920).
    """

    # TensorRT blob shapes  (must match the exported ONNX / engine)
    _REFINER_INPUT_SHAPES  = {
        "transf_input": [252, 160, 160, 6],
        "render_input":  [252, 160, 160, 6],
    }
    _REFINER_OUTPUT_SHAPES = {
        "trans": [252, 3],
        "rot":   [252, 3],
    }
    _SCORER_INPUT_SHAPES   = {
        "transf_input": [252, 160, 160, 6],
        "render_input":  [252, 160, 160, 6],
    }
    _SCORER_OUTPUT_SHAPES  = {
        "scores": [252, 1],
    }

    def __init__(
        self,
        refiner_engine: str,
        scorer_engine: str,
        mesh_path: str,
        mesh_name: str,
        intrinsic: np.ndarray,
        crop_h: int = 160,
        crop_w: int = 160,
        max_h: int = 1080,
        max_w: int = 1920,
    ) -> None:
        intrinsic = np.asarray(intrinsic, dtype=np.float32)
        if intrinsic.shape != (3, 3):
            raise ValueError(f"intrinsic must be 3×3, got {intrinsic.shape}")

        # Convert list shapes to the uint64 vectors the C++ API expects
        def _to_uint64(d: dict) -> dict:
            return {k: [int(x) for x in v] for k, v in d.items()}

        refiner_core = _cpp.create_trt_infer_core(
            refiner_engine,
            _to_uint64(self._REFINER_INPUT_SHAPES),
            _to_uint64(self._REFINER_OUTPUT_SHAPES),
            1,
        )
        scorer_core = _cpp.create_trt_infer_core(
            scorer_engine,
            _to_uint64(self._SCORER_INPUT_SHAPES),
            _to_uint64(self._SCORER_OUTPUT_SHAPES),
            1,
        )

        mesh_loader = _cpp.create_mesh_loader(mesh_name, mesh_path)
        if mesh_loader is None:
            raise RuntimeError(f"Failed to load mesh: {mesh_path}")

        self._handle = _cpp.create_model(
            refiner_core,
            scorer_core,
            [mesh_loader],
            intrinsic,
            mesh_name,
            max_h,
            max_w,
            crop_h,
            crop_w,
        )

    def register(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        refine_itr: int = 1,
    ) -> np.ndarray:
        """Initialise pose estimation from the first frame.

        Parameters
        ----------
        rgb:
            H×W×3 uint8 array in **RGB** order.
        depth:
            H×W float32 array of depth values in metres.
        mask:
            H×W uint8 segmentation mask (non-zero = object).
        refine_itr:
            Number of refinement iterations (default 1).

        Returns
        -------
        np.ndarray
            4×4 float32 pose matrix in mesh coordinate space.
        """
        rgb   = np.ascontiguousarray(rgb,   dtype=np.uint8)
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        mask  = np.ascontiguousarray(mask,  dtype=np.uint8)
        return self._handle.register_object(rgb, depth, mask, refine_itr)

    def track(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        prior_pose: np.ndarray,
        refine_itr: int = 1,
    ) -> np.ndarray:
        """Refine pose from a subsequent frame given a prior hypothesis.

        Parameters
        ----------
        rgb:
            H×W×3 uint8 array in **RGB** order.
        depth:
            H×W float32 array of depth values in metres.
        prior_pose:
            4×4 float32 pose matrix (from ``register`` or a previous ``track``).
        refine_itr:
            Number of refinement iterations (default 1).

        Returns
        -------
        np.ndarray
            4×4 float32 pose matrix in mesh coordinate space.
        """
        rgb        = np.ascontiguousarray(rgb,        dtype=np.uint8)
        depth      = np.ascontiguousarray(depth,      dtype=np.float32)
        prior_pose = np.ascontiguousarray(prior_pose, dtype=np.float32)
        if prior_pose.shape != (4, 4):
            raise ValueError(f"prior_pose must be 4×4, got {prior_pose.shape}")
        return self._handle.track(rgb, depth, prior_pose, refine_itr)
