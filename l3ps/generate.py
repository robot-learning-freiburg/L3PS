from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import hydra
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from omegaconf import DictConfig

from utils.dataset import Dataset
from utils.projection import compute_lidar2img


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    dataset = Dataset(cfg.dataset)

    print(f"[GENERATE:] Processing {len(dataset)} samples. This may take a while...")

    # Make sure output directory exists.
    Path(cfg.generate.output_path).mkdir(parents=True, exist_ok=True)

    f = partial(generate, dataset=dataset, cfg=cfg)
    if cfg.process_count > 1:
        with Pool(cfg.process_count) as p:
            p.map(f, range(len(dataset.sample_tokens)))
    else:
        for idx in range(len(dataset.sample_tokens)):
            f(idx)

    print(
        f"[GENERATE:] Finished generating primal labels for {cfg.dataset.split} split.\n"
    )


def generate(idx: int, dataset: Dataset, cfg: DictConfig):
    sample_token = dataset.get_sample(idx)
    sample = dataset.nusc.get("sample", sample_token)

    pointcloud = load_pointcloud(dataset, sample)
    labels = np.zeros((pointcloud.points.shape[1],), dtype=np.int16)

    for camera in cfg.generate.cameras:

        pseudo_labels = load_pseudo_labels(
            Path(cfg.generate.pastel_labels_path)
            / dataset.nusc.get("sample_data", sample["data"][camera])["filename"]
            .replace("samples/", "")
            .replace(".jpg", ".npy")
        )
        points, indices = compute_lidar2img(
            dataset.nusc,
            deepcopy(pointcloud),  # To avoid modifying the original pointcloud.
            sample["data"]["LIDAR_TOP"],
            sample["data"][camera],
        )
        # Check if there are any points that are projected into same pixel
        labels[indices] = pseudo_labels[points[:, 1], points[:, 0]]

    # Convert labels to 1000s.
    labels = np.where(labels < 1000, labels * 1000, labels)

    # Save the labeled points.
    labeled_points_path = Path(cfg.generate.output_path) / f"{sample_token}.npy"
    np.save(labeled_points_path, labels)


def load_pointcloud(dataset: Dataset, sample: dict) -> LidarPointCloud:
    points_path = (
        Path(dataset.nusc.dataroot)
        / dataset.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["filename"]
    )
    return LidarPointCloud.from_file(str(points_path))


def load_pseudo_labels(path: Path) -> np.ndarray:
    pseudo_labels = np.load(path).astype(np.int16)
    # Convert any existing sky (17) labels to void (0).
    pseudo_labels[pseudo_labels == 17] = 0
    return pseudo_labels
