from pathlib import Path

import numpy as np
import open3d as o3d


def generate_colored_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def write_point_cloud(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> bool:
    # Make sure that the output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = generate_colored_pointcloud(points, colors)
    return o3d.io.write_point_cloud(str(path), pcd)
