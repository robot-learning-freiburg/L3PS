import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import hydra
import numpy as np
from fast_hdbscan import HDBSCAN
from omegaconf import DictConfig
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from utils.dataset import FastDataset


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):

    dataset = FastDataset(cfg.dataset)
    Path(cfg.refine.output_path).mkdir(parents=True, exist_ok=True)

    hdbscan = HDBSCAN(min_cluster_size=cfg.refine.min_cluster_size)
    knn = KNeighborsClassifier(
        n_neighbors=3, n_jobs=cfg.process_count, metric="euclidean"
    )

    with tqdm(total=len(dataset), desc="[REFINE] Processing") as pbar:
        for idx in range(len(dataset)):
            refine(
                idx=idx,
                hdbscan=hdbscan,
                knn=knn,
                dataset=dataset,
                cfg=cfg,
            )
            pbar.update(1)


def refine(
    idx: int,
    hdbscan: HDBSCAN,
    knn: KNeighborsClassifier,
    dataset: FastDataset,
    cfg: DictConfig,
):
    scene = dataset.scenes[idx]

    # Load the accumulated pointcloud
    # [X, Y, Z, GROUND_MASK, FRAME_NUMS, PRIMAL_LABELS]
    accumulated_pointcloud = np.load(f"{cfg.refine.input_path}/{scene}.npy")

    pointcloud = accumulated_pointcloud[:, :3]
    ground_mask = accumulated_pointcloud[:, 3].astype(bool)
    frame_nums = accumulated_pointcloud[:, 4].astype(int)
    primal_labels = accumulated_pointcloud[:, 5]

    semantic_labels = (primal_labels // 1000).astype(np.int8)
    instance_labels = (primal_labels % 1000).astype(np.int8)

    noise_class = cfg.refine.noise_class
    rare_classes = cfg.refine.rare_classes

    ground_clusters = hdbscan.fit_predict(pointcloud[ground_mask])
    cluster_noise_mask = ground_clusters == -1
    knn.fit(
        pointcloud[ground_mask][~cluster_noise_mask],
        ground_clusters[~cluster_noise_mask],
    )
    ground_clusters[cluster_noise_mask] = knn.predict(
        pointcloud[ground_mask][cluster_noise_mask]
    )

    ng_clusters = hdbscan.fit_predict(pointcloud[~ground_mask])
    cluster_noise_mask = ng_clusters == -1
    knn.fit(
        pointcloud[~ground_mask][~cluster_noise_mask], ng_clusters[~cluster_noise_mask]
    )
    ng_clusters[cluster_noise_mask] = knn.predict(
        pointcloud[~ground_mask][cluster_noise_mask]
    )

    refined_labels = np.zeros_like(semantic_labels)
    refined_labels[ground_mask] = semantic_correction(
        ground_clusters, semantic_labels[ground_mask], rare_classes, noise_class
    )
    refined_labels[~ground_mask] = semantic_correction(
        ng_clusters, semantic_labels[~ground_mask], rare_classes, noise_class
    )

    refined_instances = instance_correction(
        pointcloud, frame_nums, semantic_labels, refined_labels, instance_labels
    )

    panoptic_labels = refined_labels * 1000 + refined_instances

    np.save(f"{cfg.refine.output_path}/{scene}.npy", panoptic_labels)


def determine_dominant_class(label_counts, rare_classes, noise_class):
    majority_class = np.argmax(label_counts)

    if majority_class == noise_class:
        # Check if ratio is over 90%
        if label_counts[majority_class] / np.sum(label_counts) > 0.9:
            return majority_class
        else:
            label_counts[noise_class] = 0
            return np.argmax(label_counts)
    else:
        for rare_class in rare_classes:
            if label_counts[rare_class] / np.sum(label_counts) > 0.3:
                return rare_class
        return majority_class


def semantic_correction(clusters, labels, rare_classes, noise_class):
    for cluster in np.unique(clusters):
        if cluster == -1:
            continue

        mask = clusters == cluster
        label_counts = np.bincount(labels[mask], minlength=17)
        dominant_class = determine_dominant_class(
            label_counts, rare_classes, noise_class
        )
        labels[mask] = dominant_class
    return labels


def instance_correction(pointcloud, frame_nums, old_semantic, new_semantic, instance):
    frames = np.unique(frame_nums)
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=4, metric="euclidean")

    accumulate_instance_labels = []
    for frame in frames:
        frame_cloud = pointcloud[frame_nums == frame]
        frame_old_semantic = old_semantic[frame_nums == frame]
        frame_new_semantic = new_semantic[frame_nums == frame]
        frame_instance = instance[frame_nums == frame]

        # Assign instance labels from old frame to new frame
        # if the semantic label is the same
        same_semantic_mask = frame_old_semantic == frame_new_semantic
        new_instance_labels = np.zeros_like(frame_instance)
        new_instance_labels[same_semantic_mask] = frame_instance[same_semantic_mask]

        thing_mask = (
            (frame_new_semantic > 0) & (frame_new_semantic < 11) & same_semantic_mask
        )

        need_thing_mask = (
            (frame_new_semantic > 0) & (frame_new_semantic < 11) & ~same_semantic_mask
        )

        if need_thing_mask.any():
            if thing_mask.any():
                knn.fit(frame_cloud[thing_mask], new_instance_labels[thing_mask])
                new_instance_labels[need_thing_mask] = knn.predict(
                    frame_cloud[need_thing_mask]
                )
            else:
                new_instance_labels[need_thing_mask] = np.max(new_instance_labels) + 1

        accumulate_instance_labels.append(new_instance_labels)

    return np.concatenate(accumulate_instance_labels)


if __name__ == "__main__":
    run()
