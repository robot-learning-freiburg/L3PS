from pathlib import Path

import hydra
import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from omegaconf import DictConfig
from pypatchworkpp import Parameters, patchworkpp
from tqdm import tqdm

from utils.dataset import FastDataset


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):

    dataset = FastDataset(cfg.dataset)
    Path(cfg.accumulate.output_path).mkdir(parents=True, exist_ok=True)

    ground_segmenter = patchworkpp(Parameters())
    odometry_estimator = KissICP(
        load_kiss_config(KISSConfig(**cfg.accumulate.kiss_icp))
    )

    with tqdm(total=len(dataset), desc="[ACCUMULATE] Processing") as pbar:
        for idx in range(len(dataset)):
            accumulate(
                idx=idx,
                ground_segmenter=ground_segmenter,
                odometry_estimator=odometry_estimator,
                dataset=dataset,
                cfg=cfg,
            )
            pbar.update(1)


def accumulate(
    idx: int,
    ground_segmenter: patchworkpp,
    odometry_estimator: KissICP,
    dataset: FastDataset,
    cfg: DictConfig,
):
    samples = dataset[idx]
    scene_pointclouds = []

    mapping = dataset.mapper

    for frame_num, sample_token in enumerate(samples):
        sample = samples[sample_token]
        timestamp = sample["timestamp"]
        pointcloud = load_pointcloud(dataset.path, sample["lidar_file"])
        primal_labels = load_primal_labels(
            Path(cfg.accumulate.primal_path), sample_token
        )
        # gt_semantic_labels, gt_instance_labels = load_gt_labels(
        #     dataset.path, sample["panoptic_file"], mapping
        # )

        ground_segmenter.estimateGround(pointcloud)
        odometry_estimator.register_frame(pointcloud[:, :3], np.array(timestamp))
        pointcloud = transform(pointcloud[:, :3], odometry_estimator.last_pose)

        ground_mask = np.zeros((pointcloud.shape[0], 1), dtype=np.int8).ravel()
        ground_mask[ground_segmenter.getGroundIndices()] = 1
        frame_num_arr = np.full(
            (pointcloud.shape[0], 1), frame_num, dtype=np.int8
        ).ravel()

        feats = np.vstack(
            [
                ground_mask,
                frame_num_arr,
                primal_labels,
            ]
        ).T
        scene_pointclouds.append(np.hstack([pointcloud, feats]))

    scene_pointclouds = np.concatenate(scene_pointclouds, axis=0, dtype=np.float32)
    np.save(
        Path(cfg.accumulate.output_path) / f"{dataset.scenes[idx]}.npy",
        scene_pointclouds,
    )


def transform(points: np.ndarray, odometry: np.ndarray) -> np.ndarray:
    return np.dot(odometry[:3, :3], points.T).T + odometry[:3, 3]


def load_primal_labels(rootdir, sample_token):
    return np.load(rootdir / f"{sample_token}.npy")


def load_pointcloud(rootdir, file):
    return np.fromfile(rootdir / file, dtype=np.float32).reshape(-1, 5)[:, :4]


def load_kiss_config(config: KISSConfig) -> KISSConfig:
    if config.data.max_range < config.data.min_range:
        print("[WARNING] max_range is smaller than min_range, settng min_range to 0.0")
        config.data.min_range = 0.0
    if config.mapping.voxel_size is None:
        config.mapping.voxel_size = float(config.data.max_range / 100.0)

    return config


if __name__ == "__main__":
    run()
