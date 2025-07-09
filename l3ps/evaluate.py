from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from prettytable import PrettyTable

from utils.dataset import FastDataset
from utils.pq import PanopticEval
from tqdm import tqdm


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg):
    dataset = FastDataset(cfg.dataset)
    log_path = Path(f"{cfg.evaluate.log_path}.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
        
    print(f"Logging results to {log_path}")    
    
    evaluator_primal = PanopticEval(
        cfg.evaluate.n_classes, ignore=[cfg.evaluate.ignore_label]
    )
    evaluator_refined = PanopticEval(
        cfg.evaluate.n_classes, ignore=[cfg.evaluate.ignore_label]
    )

    with tqdm(total=len(dataset.scenes), desc="[EVALUATE] Processing") as pbar:
        for idx in range(len(dataset.scenes)):
            evaluate(idx, dataset, evaluator_primal, evaluator_refined, cfg)
            pbar.update(1)


def evaluate(
    idx: int,
    dataset: FastDataset,
    evaluator_primal: PanopticEval,
    evaluator_refined: PanopticEval,
    cfg: DictConfig,
):
    samples = dataset[idx]
    scene_name = dataset.scenes[idx]

    accumulated_pointcloud = np.load(
        Path(cfg.evaluate.primal_labels_path) / f"{scene_name}.npy"
    )
    refined_labels = np.load(
        Path(cfg.evaluate.refined_labels_path) / f"{scene_name}.npy"
    ).astype(np.int16)

    primal_labels = accumulated_pointcloud[:, 5].astype(np.int16)
    frame_nums = accumulated_pointcloud[:, 4].astype(int)

    for frame, sample_token in enumerate(samples):
        rootdir = dataset.path
        mapper = dataset.mapper
        panoptic_file = samples[sample_token]["panoptic_file"]
        gt_semantic_labels, gt_instance_labels = load_gt_labels(
            rootdir, panoptic_file, mapper
        )

        frame_primal_labels = primal_labels[frame_nums == frame]
        frame_refined_labels = refined_labels[frame_nums == frame]

        # Seperate semantic and instance labels
        primal_semantic = frame_primal_labels // 1000
        primal_instance = frame_primal_labels

        refined_semantic = frame_refined_labels // 1000
        refined_instance = frame_refined_labels

        # Create an ignore mask to avoid ignore label prediction
        ignore_primal = np.logical_or(gt_semantic_labels == 0, primal_semantic == 0)
        ignore_refined = np.logical_or(gt_semantic_labels == 0, refined_semantic == 0)

        evaluator_primal.addBatch(
            primal_semantic[~ignore_primal],
            primal_instance[~ignore_primal],
            gt_semantic_labels[~ignore_primal],
            gt_instance_labels[~ignore_primal],
        )

        evaluator_refined.addBatch(
            refined_semantic[~ignore_refined],
            refined_instance[~ignore_refined],
            gt_semantic_labels[~ignore_refined],
            gt_instance_labels[~ignore_refined],
        )

    log_path = Path(cfg.evaluate.log_path)
    printer(evaluator_primal, "Primal", idx, log_path)
    printer(evaluator_refined, "Refined", idx, log_path)


def printer(
    evaluator: PanopticEval,
    prefix: str,
    idx: int,
    log_path: Path,
):
    semantic_classes = [
        "void/ignore",
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction_vehicle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "trailer",
        "truck",
        "driveable_surface",
        "other_flat",
        "sidewalk",
        "terrain",
        "manmade",
        "vegetation",
    ]

    class_PQ, class_SQ, class_RQ, class_all_PQ, class_all_SQ, class_all_RQ = (
        evaluator.getPQ()
    )
    class_IoU, class_all_IoU = evaluator.getSemIoU()

    table = PrettyTable()
    table.field_names = ["Class", "PQ", "SQ", "RQ", "IoU"]
    for i in range(len(semantic_classes)):
        table.add_row(
            [
                semantic_classes[i],
                f"{class_all_PQ[i]:.3f}",
                f"{class_all_SQ[i]:.3f}",
                f"{class_all_RQ[i]:.3f}",
                f"{class_all_IoU[i]:.3f}",
            ]
        )

    table.add_row(
        [
            "All",
            f"{class_PQ:.3f}",
            f"{class_SQ:.3f}",
            f"{class_RQ:.3f}",
            f"{class_IoU:.3f}",
        ]
    )
    # Create a log file if it does not exist
    
    with open(f"{log_path}.txt", "a") as f:
        f.write(f"{prefix} Results for {idx}\n")
        f.write(str(table)+'\n')


def load_gt_labels(rootdir, file, mapping: dict):
    mapper = np.vectorize(mapping.get)
    labels = np.load(rootdir / file)["data"]
    semantic_labels = mapper(labels // 1000)
    thing_mask = np.logical_and(semantic_labels <= 10, semantic_labels > 0)
    instance_labels = labels
    instance_labels[~thing_mask] = 0

    return semantic_labels, instance_labels


if __name__ == "__main__":
    run()
