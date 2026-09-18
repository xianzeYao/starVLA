#!/usr/bin/env python3
"""Lift LIBERO episode 260 index 3 to an axes-free RGB point cloud."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib
from PIL import Image
import numpy as np

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt


DATASET = Path(
    "/root/data/yxz/datasets/libero_rerender/"
    "libero_10_no_noops_1.0.0_lerobot"
)
RGB_SOURCE = DATASET / "videos/chunk-000/observation.images.image/episode_000260.mp4"
DEPTH_SOURCE = DATASET / "depth/chunk-000/observation.depth.image_m/episode_000260.npz"
CAMERA_SOURCE = DATASET / "camera/chunk-000/episode_000260.npz"
OUTPUT = Path(
    "/home/yxz/CoT/artifacts/"
    "libero10_episode_000260_index_0003_pointcloud_world_z_minus20_depthlt2m_transparent.png"
)
FRAME_INDEX = 3
WORLD_Z_ROTATION_DEGREES = -20.0
VIEW_ELEVATION_DEGREES = 12.0
MIRROR_OUTPUT_HORIZONTALLY = True
MAX_CAMERA_DEPTH_M = 2.0


def lift_rgbd_to_camera(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift aligned RGB-D pixels into OpenCV camera coordinates."""

    color = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=np.float32)
    camera_k = np.asarray(intrinsics, dtype=np.float32)
    if color.ndim != 3 or color.shape[2] != 3 or color.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8 [H,W,3], got {color.shape}/{color.dtype}")
    if depth.shape != color.shape[:2]:
        raise ValueError(f"depth shape {depth.shape} does not match rgb shape {color.shape[:2]}")
    if camera_k.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape [3,3], got {camera_k.shape}")

    fx, fy = float(camera_k[0, 0]), float(camera_k[1, 1])
    cx, cy = float(camera_k[0, 2]), float(camera_k[1, 2])
    if not np.isfinite(camera_k).all() or abs(fx) <= 1e-8 or abs(fy) <= 1e-8:
        raise ValueError("intrinsics must be finite with non-zero signed focal lengths")

    height, width = depth.shape
    pixel_u, pixel_v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    valid = np.isfinite(depth) & (depth > 0.0)
    z = depth[valid]
    x = (pixel_u[valid] - cx) * z / fx
    y = (pixel_v[valid] - cy) * z / fy
    points = np.column_stack((x, y, z)).astype(np.float32, copy=False)
    colors = color[valid].copy()
    return points, colors


def filter_cloud_by_camera_depth(
    points_camera: np.ndarray,
    colors_rgb: np.ndarray,
    *,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove far-background samples while preserving XYZ/RGB alignment."""

    points = np.asarray(points_camera, dtype=np.float32)
    colors = np.asarray(colors_rgb, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_camera must have shape [N,3], got {points.shape}")
    if colors.shape != (len(points), 3):
        raise ValueError(f"colors_rgb must have shape {(len(points), 3)}, got {colors.shape}")
    if not np.isfinite(max_depth_m) or max_depth_m <= 0.0:
        raise ValueError("max_depth_m must be finite and positive")

    keep = np.isfinite(points[:, 2]) & (points[:, 2] <= max_depth_m)
    return points[keep], colors[keep]


def move_rgb_horizontal_mirror_to_extrinsics(
    intrinsics: np.ndarray,
    world_from_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Move the stored RGB horizontal mirror from signed K into camera pose."""

    camera_k = np.asarray(intrinsics, dtype=np.float32)
    camera_pose = np.asarray(world_from_camera, dtype=np.float32)
    if camera_k.shape != (3, 3) or camera_pose.shape != (4, 4):
        raise ValueError(
            f"expected K [3,3] and pose [4,4], got {camera_k.shape}/{camera_pose.shape}"
        )
    if not np.isfinite(camera_k).all() or not np.isfinite(camera_pose).all():
        raise ValueError("camera intrinsics and pose must be finite")

    rgb_k = camera_k.copy()
    rgb_pose = camera_pose.copy()
    if rgb_k[0, 0] < 0.0:
        rgb_k[0, 0] *= -1.0
        horizontal_reflection = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        rgb_pose = rgb_pose @ horizontal_reflection
    return rgb_k, rgb_pose


def camera_points_to_world(
    points_camera: np.ndarray,
    world_from_camera: np.ndarray,
) -> np.ndarray:
    """Transform camera-frame XYZ points into world coordinates."""

    points = np.asarray(points_camera, dtype=np.float32)
    camera_pose = np.asarray(world_from_camera, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_camera must have shape [N,3], got {points.shape}")
    if camera_pose.shape != (4, 4) or not np.isfinite(camera_pose).all():
        raise ValueError(f"world_from_camera must be finite [4,4], got {camera_pose.shape}")
    homogeneous = np.column_stack(
        (points, np.ones(len(points), dtype=np.float32))
    )
    return (camera_pose @ homogeneous.T).T[:, :3].astype(np.float32, copy=False)

def compute_orbit_view_angles(
    world_from_camera: np.ndarray,
    *,
    world_z_rotation_degrees: float,
    view_elevation_degrees: float,
) -> tuple[float, float]:
    """Return an RGB-relative world-Z orbit with an explicit mild elevation."""

    camera_pose = np.asarray(world_from_camera, dtype=np.float32)
    if camera_pose.shape != (4, 4):
        raise ValueError(f"world_from_camera must have shape [4,4], got {camera_pose.shape}")
    if (
        not np.isfinite(camera_pose).all()
        or not np.isfinite(world_z_rotation_degrees)
        or not np.isfinite(view_elevation_degrees)
    ):
        raise ValueError("camera pose and view angles must be finite")

    view_direction = -camera_pose[:3, 2]
    horizontal_norm = float(np.hypot(view_direction[0], view_direction[1]))
    if horizontal_norm <= 1e-8:
        raise ValueError("camera pose has a degenerate horizontal view direction")
    base_azimuth = np.rad2deg(np.arctan2(view_direction[1], view_direction[0]))
    return (
        float(base_azimuth + world_z_rotation_degrees),
        float(view_elevation_degrees),
    )




def mirror_image_horizontally(pixels: np.ndarray) -> np.ndarray:
    """Mirror rendered pixels so screen handedness matches the stored RGB."""

    image = np.asarray(pixels)
    if image.ndim not in (2, 3):
        raise ValueError(f"pixels must be a 2D or 3D image, got {image.shape}")
    return image[:, ::-1].copy()


def render_pointcloud_png(
    points_world: np.ndarray,
    colors_rgb: np.ndarray,
    output_path: str | Path,
    *,
    world_from_camera: np.ndarray,
    world_z_rotation_degrees: float,
    view_elevation_degrees: float,
    mirror_output_horizontally: bool,
) -> Path:
    """Render an axes-free world point cloud after orbiting about world Z."""

    points = np.asarray(points_world, dtype=np.float32)
    colors = np.asarray(colors_rgb, dtype=np.uint8)
    camera_pose = np.asarray(world_from_camera, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1:
        raise ValueError(f"points_world must be non-empty [N,3], got {points.shape}")
    if colors.shape != (len(points), 3):
        raise ValueError(f"colors_rgb must have shape {(len(points), 3)}, got {colors.shape}")
    if camera_pose.shape != (4, 4):
        raise ValueError(f"world_from_camera must have shape [4,4], got {camera_pose.shape}")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(camera_pose).all()
        or not np.isfinite(world_z_rotation_degrees)
    ):
        raise ValueError("points, camera pose, and world-Z rotation must be finite")

    low = np.percentile(points, 0.5, axis=0)
    high = np.percentile(points, 99.5, axis=0)
    span = np.maximum(high - low, 1e-3)
    padding = 0.04 * span

    azimuth, elevation = compute_orbit_view_angles(
        camera_pose,
        world_z_rotation_degrees=world_z_rotation_degrees,
        view_elevation_degrees=view_elevation_degrees,
    )

    figure = plt.figure(figsize=(8, 6), dpi=200, facecolor="none")
    axes = figure.add_subplot(111, projection="3d", facecolor="none")
    point_size = float(np.clip(90000.0 / len(points), 0.35, 2.0))
    axes.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors.astype(np.float32) / 255.0,
        s=point_size,
        marker=".",
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )
    axes.set_xlim(float(low[0] - padding[0]), float(high[0] + padding[0]))
    axes.set_ylim(float(low[1] - padding[1]), float(high[1] + padding[1]))
    axes.set_zlim(float(low[2] - padding[2]), float(high[2] + padding[2]))
    axes.set_box_aspect(tuple(float(value) for value in span))
    axes.view_init(elev=float(elevation), azim=float(azimuth))
    axes.set_axis_off()
    axes.set_position((0.0, 0.0, 1.0, 1.0))
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, transparent=True)
    plt.close(figure)
    if mirror_output_horizontally:
        with Image.open(output) as rendered:
            mirrored_pixels = mirror_image_horizontally(np.asarray(rendered))
        Image.fromarray(mirrored_pixels).save(output)
    return output


if __name__ == "__main__":
    reader = imageio.get_reader(RGB_SOURCE)
    try:
        rgb_frame = np.asarray(reader.get_data(FRAME_INDEX), dtype=np.uint8)
    finally:
        reader.close()
    with np.load(DEPTH_SOURCE, allow_pickle=False) as payload:
        depth_frame = np.asarray(payload["depth_m"][FRAME_INDEX], dtype=np.float32)
    with np.load(CAMERA_SOURCE, allow_pickle=False) as payload:
        camera_k = np.asarray(payload["agentview_K"][FRAME_INDEX], dtype=np.float32)
        world_from_camera = np.asarray(
            payload["agentview_T_world_camera"][FRAME_INDEX], dtype=np.float32
        )

    rgb_k, rgb_world_from_camera = move_rgb_horizontal_mirror_to_extrinsics(
        camera_k, world_from_camera
    )
    cloud_camera, cloud_rgb = lift_rgbd_to_camera(rgb_frame, depth_frame, rgb_k)
    cloud_camera, cloud_rgb = filter_cloud_by_camera_depth(
        cloud_camera, cloud_rgb, max_depth_m=MAX_CAMERA_DEPTH_M
    )
    cloud_world = camera_points_to_world(cloud_camera, rgb_world_from_camera)
    path = render_pointcloud_png(
        cloud_world,
        cloud_rgb,
        OUTPUT,
        world_from_camera=rgb_world_from_camera,
        world_z_rotation_degrees=WORLD_Z_ROTATION_DEGREES,
        view_elevation_degrees=VIEW_ELEVATION_DEGREES,
        mirror_output_horizontally=MIRROR_OUTPUT_HORIZONTALLY,
    )
    print(
        {
            "output": str(path),
            "view_elevation_degrees": VIEW_ELEVATION_DEGREES,
            "rgb_horizontal_alignment": "mirrored_output",
            "zero_based_frame_index": FRAME_INDEX,
            "world_z_rotation_degrees": WORLD_Z_ROTATION_DEGREES,
            "valid_points": len(cloud_world),
            "max_camera_depth_m": MAX_CAMERA_DEPTH_M,
            "transparent_background": True,
            "axes_visible": False,
        }
    )
