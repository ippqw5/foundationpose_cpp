#!/usr/bin/env python3
"""
demo_mustard.py

End-to-end FoundationPose demo on the mustard0 dataset.

Usage:
    cd python_bindings
    python demo_mustard.py

Expected output:
    [Register] frame=1581120424100262102
    pose =
    [[ ...  ...  ...  ... ]
     ...
    [Track] frame=1581120424148532296
    pose =
    ...
    Demo complete.
"""

import os
import sys
import glob
import numpy as np
import cv2

# Allow running from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REFINER_ENGINE  = os.path.join(REPO_ROOT, "models", "refiner_hwc_dynamic_fp16.engine")
SCORER_ENGINE   = os.path.join(REPO_ROOT, "models", "scorer_hwc_dynamic_fp16.engine")
DATA_DIR        = os.path.join(REPO_ROOT, "test_data", "mustard0")
MESH_PATH       = os.path.join(DATA_DIR,  "mesh", "textured_simple.obj")
CAM_K_PATH      = os.path.join(DATA_DIR,  "cam_K.txt")
MESH_NAME       = "mustard"
REFINE_ITR      = 1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cam_k(path: str) -> np.ndarray:
    """Load 3×3 intrinsic matrix from a space-separated text file."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    K = np.array(rows, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"cam_K.txt should contain a 3×3 matrix, got {K.shape}")
    return K


def load_rgb(path: str) -> np.ndarray:
    """Load image as H×W×3 uint8 RGB."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read RGB image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth(path: str) -> np.ndarray:
    """Load depth PNG (uint16, millimetres) → float32 metres."""
    depth_mm = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise FileNotFoundError(f"Cannot read depth image: {path}")
    return (depth_mm.astype(np.float32)) / 1000.0


def load_mask(path: str) -> np.ndarray:
    """Load mask as uint8. Non-zero pixels indicate the object."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return mask

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------ #
    # Initialise model                                                     #
    # ------------------------------------------------------------------ #
    print("Loading FoundationPose model …")
    from foundationpose_cpp import FoundationPose  # noqa: imported here so errors surface clearly

    K = load_cam_k(CAM_K_PATH)
    model = FoundationPose(
        refiner_engine=REFINER_ENGINE,
        scorer_engine=SCORER_ENGINE,
        mesh_path=MESH_PATH,
        mesh_name=MESH_NAME,
        intrinsic=K,
    )
    print("Model ready.\n")

    # ------------------------------------------------------------------ #
    # Collect frame IDs (sorted by timestamp)                             #
    # ------------------------------------------------------------------ #
    rgb_files = sorted(glob.glob(os.path.join(DATA_DIR, "rgb", "*.png")))
    if not rgb_files:
        print(f"ERROR: No RGB frames found in {DATA_DIR}/rgb/", file=sys.stderr)
        sys.exit(1)

    frame_ids = [os.path.splitext(os.path.basename(f))[0] for f in rgb_files]

    # ------------------------------------------------------------------ #
    # Frame 0: Register                                                   #
    # ------------------------------------------------------------------ #
    fid   = frame_ids[0]
    rgb   = load_rgb  (os.path.join(DATA_DIR, "rgb",   fid + ".png"))
    depth = load_depth(os.path.join(DATA_DIR, "depth", fid + ".png"))
    mask  = load_mask (os.path.join(DATA_DIR, "masks", fid + ".png"))

    pose = model.register(rgb, depth, mask, refine_itr=REFINE_ITR)

    print(f"[Register] frame={fid}")
    print("pose =")
    print(pose)
    print()

    # ------------------------------------------------------------------ #
    # Remaining frames: Track                                             #
    # ------------------------------------------------------------------ #
    for fid in frame_ids[1:]:
        rgb_path   = os.path.join(DATA_DIR, "rgb",   fid + ".png")
        depth_path = os.path.join(DATA_DIR, "depth", fid + ".png")

        rgb   = load_rgb  (rgb_path)
        depth = load_depth(depth_path)

        pose = model.track(rgb, depth, pose, refine_itr=REFINE_ITR)

        print(f"[Track] frame={fid}")
        print("pose =")
        print(pose)
        print()

    print("Demo complete.")


if __name__ == "__main__":
    main()
