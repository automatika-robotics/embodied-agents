"""Motion detection algorithms for images and point clouds.

Point cloud voxelization can optionally run on GPU through torch when available.
"""

import math
import warnings
from typing import Dict, List, Tuple

import cv2
import numpy as np

__all__ = [
    "frame_difference",
    "optical_flow",
    "roi_mask",
    "apply_transform",
    "crop_cloud",
    "transform_cloud",
    "ensure_torch",
    "unique_voxels",
    "voxel_set_difference",
    "cluster_voxels",
    "cluster_centers",
]

# Bit layout for packing a 3D voxel index into a single int64 key:
# 21 bits per axis with a positive offset -> |index| < 2**20 per axis
_KEY_BITS = 21
_KEY_OFFSET = 1 << (_KEY_BITS - 1)
_KEY_MASK = (1 << _KEY_BITS) - 1


# ---------------------------------------------------------------------------
# Image motion estimation
# ---------------------------------------------------------------------------


def frame_difference(
    previous_gray: np.ndarray, current_gray: np.ndarray, threshold: float
) -> bool:
    """Difference between two grayscale frames thresholded to a motion bool.

    :param previous_gray: Previous frame in grayscale
    :type previous_gray: np.ndarray
    :param current_gray: Current frame in grayscale
    :type current_gray: np.ndarray
    :param threshold: Percentage of changed pixels for detecting motion
    :type threshold: float
    :rtype: bool
    """
    # calculate frame difference
    diff = cv2.subtract(current_gray, previous_gray)
    # apply blur to improve thresholding
    diff = cv2.medianBlur(diff, 3)
    # apply adaptive thresholding
    mask = cv2.adaptiveThreshold(
        diff, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return mask.sum() > (threshold * math.prod(current_gray.shape) / 100)


def optical_flow(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    threshold: float,
    flow_kwargs: Dict,
) -> bool:
    """Farneback optical flow between two frames thresholded to a motion bool.

    :param previous_gray: Previous frame in grayscale
    :type previous_gray: np.ndarray
    :param current_gray: Current frame in grayscale
    :type current_gray: np.ndarray
    :param threshold: Percentage of pixels in motion for detecting motion
    :type threshold: float
    :param flow_kwargs: Keyword arguments for cv2.calcOpticalFlowFarneback
    :type flow_kwargs: Dict
    :rtype: bool
    """
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray, current_gray, None, **flow_kwargs
    )
    mask = np.uint8(flow > 1) / 10
    return mask.sum() > (threshold * math.prod(current_gray.shape) / 100)


def roi_mask(
    shape: Tuple[int, int], ignored_polygon: List[Tuple[int, int]]
) -> np.ndarray:
    """Create a mask of ones with an ignored polygon region set to zero.

    :param shape: (height, width) of the target frames
    :type shape: Tuple[int, int]
    :param ignored_polygon: Pixel coordinates of the polygon to ignore
    :type ignored_polygon: List[Tuple[int, int]]
    :return: uint8 mask to multiply grayscale frames with
    :rtype: np.ndarray
    """
    mask = np.ones(shape, dtype=np.uint8)
    polygon = np.array(ignored_polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [polygon], 0)
    return mask


# ---------------------------------------------------------------------------
# Point cloud motion detection
# ---------------------------------------------------------------------------


def crop_cloud(
    points: np.ndarray,
    min_range: float,
    max_range: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Crop a cloud by planar (xy) range and height limits.

    :param points: Nx3 array of cartesian points
    :type points: np.ndarray
    :rtype: np.ndarray
    """
    planar_range_sq = points[:, 0] ** 2 + points[:, 1] ** 2
    mask = (
        (planar_range_sq >= min_range**2)
        & (planar_range_sq <= max_range**2)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    return points[mask]


def apply_transform(
    points: np.ndarray, translation: np.ndarray, quaternion: np.ndarray
) -> np.ndarray:
    """Apply a rigid transform (quaternion rotation + translation) to points.

    Used to take cloud points from the sensor frame to the robot base frame
    with a transform looked up from TF.

    :param points: Nx3 array of cartesian points
    :type points: np.ndarray
    :param translation: Translation as [x, y, z]
    :type translation: np.ndarray
    :param quaternion: Rotation as [x, y, z, w]
    :type quaternion: np.ndarray
    :rtype: np.ndarray
    """
    qx, qy, qz, qw = (float(q) for q in quaternion)
    rotation = np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=points.dtype,
    )
    return points @ rotation.T + np.asarray(translation, dtype=points.dtype)


def transform_cloud(points: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Transform points by a planar pose for ego-motion compensation.

    Applies a rotation by the pose heading around z followed by the pose
    translation, taking sensor-frame points to the odometry frame.

    :param points: Nx3 array of cartesian points
    :type points: np.ndarray
    :param position: Pose as [x, y, z, heading, ...]
    :type position: np.ndarray
    :rtype: np.ndarray
    """
    heading = position[3]
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rotation = np.array(
        [[cos_h, -sin_h, 0.0], [sin_h, cos_h, 0.0], [0.0, 0.0, 1.0]],
        dtype=points.dtype,
    )
    return points @ rotation.T + np.asarray(position[:3], dtype=points.dtype)


def ensure_torch():
    """Import and return torch, with installation instructions if missing.

    Called when GPU point cloud processing is requested (``device="cuda"``).
    """
    try:
        import torch
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            """Point cloud processing with device='cuda' requires torch. You can install it with:
        'pip install torch'
Or set device='cpu' in MotionDetectorConfig to use numpy based processing."""
        ) from e
    return torch


def unique_voxels(
    points: np.ndarray, voxel_size: float, device: str = "cpu"
) -> np.ndarray:
    """Quantize points to a voxel grid and return unique voxel keys.

    Each occupied voxel is packed into a single int64 key (21 bits per
    axis), which makes set operations between consecutive clouds cheap.
    With ``device="cuda"`` the quantization and deduplication run through
    torch (raises with installation instructions when torch is missing,
    and falls back to numpy with a warning when torch has no CUDA device).

    :param points: Nx3 array of cartesian points
    :type points: np.ndarray
    :param voxel_size: Edge length of a voxel in meters
    :type voxel_size: float
    :param device: "cpu" or "cuda"
    :type device: str
    :return: Sorted 1D int64 array of unique voxel keys
    :rtype: np.ndarray
    """
    if points.size == 0:
        return np.empty(0, dtype=np.int64)

    if device == "cuda":
        torch = ensure_torch()
        if torch.cuda.is_available():
            cloud = torch.from_numpy(np.ascontiguousarray(points)).cuda()
            indices = torch.floor(cloud / voxel_size).to(torch.int64) + _KEY_OFFSET
            keys = (
                (indices[:, 0] << (2 * _KEY_BITS))
                | (indices[:, 1] << _KEY_BITS)
                | indices[:, 2]
            )
            return torch.unique(keys).cpu().numpy()
        warnings.warn(
            "device='cuda' requested but torch reports no available CUDA device. Falling back to numpy based processing.",
            stacklevel=2,
        )

    indices = np.floor(points / voxel_size).astype(np.int64) + _KEY_OFFSET
    keys = (
        (indices[:, 0] << (2 * _KEY_BITS))
        | (indices[:, 1] << _KEY_BITS)
        | indices[:, 2]
    )
    return np.unique(keys)


def voxel_set_difference(
    current_keys: np.ndarray, previous_keys: np.ndarray
) -> np.ndarray:
    """Symmetric difference between two sorted voxel key sets.

    Voxels that appeared plus voxels that disappeared between two
    consecutive clouds, the changed set that motion detection thresholds.

    :rtype: np.ndarray
    """
    appeared = current_keys[~np.isin(current_keys, previous_keys, assume_unique=True)]
    disappeared = previous_keys[
        ~np.isin(previous_keys, current_keys, assume_unique=True)
    ]
    return np.concatenate((appeared, disappeared))


def _keys_to_indices(keys: np.ndarray) -> np.ndarray:
    """Unpack int64 voxel keys back to Nx3 integer voxel indices."""
    indices = np.stack(
        (
            (keys >> (2 * _KEY_BITS)) & _KEY_MASK,
            (keys >> _KEY_BITS) & _KEY_MASK,
            keys & _KEY_MASK,
        ),
        axis=1,
    )
    return indices - _KEY_OFFSET


def cluster_voxels(keys: np.ndarray) -> List[np.ndarray]:
    """Cluster voxel keys into 26-connected components.

    Builds neighbor edges vectorized over the 26 neighbor offsets and
    resolves components with a small union-find.

    :param keys: 1D int64 array of unique voxel keys
    :type keys: np.ndarray
    :return: List of arrays of Nx3 voxel indices, one per cluster
    :rtype: List[np.ndarray]
    """
    if keys.size == 0:
        return []

    sorted_keys = np.sort(keys)
    parents = np.arange(sorted_keys.size)

    def find(i: int) -> int:
        root = i
        while parents[root] != root:
            root = parents[root]
        while parents[i] != root:
            parents[i], i = root, parents[i]
        return root

    # Half of the 26 neighbor offsets is enough as the other half produces
    # the same edges in reverse
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) > (0, 0, 0)
    ]
    for dx, dy, dz in offsets:
        key_offset = (dx << (2 * _KEY_BITS)) + (dy << _KEY_BITS) + dz
        neighbor_positions = np.searchsorted(sorted_keys, sorted_keys + key_offset)
        neighbor_positions = neighbor_positions.clip(max=sorted_keys.size - 1)
        connected = np.flatnonzero(
            sorted_keys[neighbor_positions] == sorted_keys + key_offset
        )
        for i in connected:
            root_a, root_b = find(i), find(neighbor_positions[i])
            if root_a != root_b:
                parents[root_b] = root_a

    roots = np.array([find(i) for i in range(sorted_keys.size)])
    clusters = []
    for root in np.unique(roots):
        clusters.append(_keys_to_indices(sorted_keys[roots == root]))
    return clusters


def cluster_centers(
    keys: np.ndarray,
    voxel_size: float,
    min_cluster_size: int,
    max_clusters: int,
) -> List[Tuple[float, float, float]]:
    """Cluster changed voxels and return metric centers of the largest clusters.

    :param keys: 1D int64 array of unique changed voxel keys
    :type keys: np.ndarray
    :param voxel_size: Edge length of a voxel in meters
    :type voxel_size: float
    :param min_cluster_size: Minimum number of voxels for a valid cluster
    :type min_cluster_size: int
    :param max_clusters: Maximum number of centers to return (largest first)
    :type max_clusters: int
    :return: Cluster centers as (x, y, z) in metric coordinates
    :rtype: List[Tuple[float, float, float]]
    """
    clusters = [
        cluster for cluster in cluster_voxels(keys) if len(cluster) >= min_cluster_size
    ]
    clusters.sort(key=len, reverse=True)
    centers = []
    for cluster in clusters[:max_clusters]:
        center = (cluster.mean(axis=0) + 0.5) * voxel_size
        centers.append((float(center[0]), float(center[1]), float(center[2])))
    return centers
