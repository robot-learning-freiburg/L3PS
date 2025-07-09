import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


def compute_lidar2cam(
    nusc: NuScenes, pcd: LidarPointCloud, lidar_data: dict, cam_data: dict
) -> LidarPointCloud:
    """
    Transforms a LidarPointCloud from the LiDAR coordinate frame to the camera coordinate frame.

    :param nusc: NuScenes object containing the dataset.
    :param pcd: LidarPointCloud object representing the point cloud to be transformed.
    :param lidar_data: Dictionary containing the LiDAR sample data.
    :param cam_data: Dictionary containing the camera sample data.
    :return: Transformed LidarPointCloud in the camera coordinate frame.
    """

    # Get calibrated sensor information
    lidar_calib = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    cam_calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

    # Get ego poses
    lidar_ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    cam_ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])

    pcd.rotate(Quaternion(lidar_calib["rotation"]).rotation_matrix)
    pcd.translate(np.array(lidar_calib["translation"]))
    pcd.rotate(Quaternion(lidar_ego_pose["rotation"]).rotation_matrix)
    pcd.translate(np.array(lidar_ego_pose["translation"]))

    pcd.translate(-np.array(cam_ego_pose["translation"]))
    pcd.rotate(Quaternion(cam_ego_pose["rotation"]).rotation_matrix.T)

    pcd.translate(-np.array(cam_calib["translation"]))
    pcd.rotate(Quaternion(cam_calib["rotation"]).rotation_matrix.T)

    return pcd


def compute_lidar2img(
    nusc: NuScenes,
    pointcloud: LidarPointCloud,
    lidar_token: str,
    cam_token: str,
    H: int = 900,
    W: int = 1600,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the lidar to camera transformation matrix.
    :param nusc: NuScenes object containing the dataset.
    :param pcd: LidarPointCloud object representing the point cloud to be transformed.
    :param lidar_token: String token identifying the LiDAR sample data.
    :param cam_token: String token identifying the camera sample data.
    :param H: Height of the image. Default is 900 (nuScenes image height).
    :param W: Width of the image. Default is 1600 (nuScenes image width).
    :return: Tuple of points and indices of projected points.
    """
    cam_data = nusc.get("sample_data", cam_token)
    lidar_data = nusc.get("sample_data", lidar_token)
    transformed_pcd = compute_lidar2cam(nusc, pointcloud, lidar_data, cam_data)

    cam_calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    intrinsics = np.array(cam_calib["camera_intrinsic"])

    # Project points to 2D
    depths = transformed_pcd.points[2, :]
    points = view_points(
        transformed_pcd.points[:3, :], intrinsics, normalize=True
    )  # 2xN
    mask = (
        (depths > 1.0)
        & (points[0, :] > 1)
        & (points[0, :] < W - 1)
        & (points[1, :] > 1)
        & (points[1, :] < H - 1)
    )
    indices = np.where(mask)[0]
    points = points.T
    points = points[indices][:, :2].astype(np.int32)
    return points, indices
