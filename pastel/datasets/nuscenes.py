
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from numpy.typing import ArrayLike
from nuscenes import NuScenes as NuSc
from nuscenes.panoptic.panoptic_utils import generate_panoptic_colors
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
from pyquaternion import Quaternion
from torchvision import transforms
from tqdm import tqdm
from yacs.config import CfgNode as CN


class NuScenes:
    # Colormap for 12 classes in nuImages dataset
    # github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py
    def __init__(self,
                 version: str,
                 dataroot: str,
                 transform: List[Callable],
                 image_size: List[int],
                 verbose: bool = False,
                 indices: List[int] = None,
                 mode: bool = 'train',
                 scene: str = None):

        assert version in ['v1.0-trainval', 'v1.0-mini', 'v1.0-test'],\
            'Error: Invalid version!'
        self.transform = transforms.Compose(transform)
        self.resize = transforms.Resize(image_size,
                                        interpolation=transforms.InterpolationMode.LANCZOS)
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.image_size = image_size

        self.scene = scene

        self.path_base = dataroot
        self.mode = mode
        self.nusc = NuSc(version=version, dataroot=dataroot, verbose=verbose)
        self.sensors = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK',
                        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',]
        self.samples = self.get_available_samples(indices=indices)
        # self.samples.reverse()
        print(f'Number of available samples: {len(self.samples)}')
        if indices is not None:
            assert len(self.samples) == len(indices) * len(self.sensors), \
                'Error: Number of samples does not match number of indices!'

    def get_available_samples(self, indices: List[int] = None):
        available_samples = []
        sample_list = self.nusc.sample
        skip_dir = \
            '/home/canakcia/git/thesis/spino/panoptic_label_generator/test/train_pseudo_labels/no_argmax'
        if indices is not None:
            for idx in tqdm(indices,
                            desc=f'Collecting available nuScenes indices for {self.mode} mode'):
                if idx in range(len(sample_list)):
                    token = sample_list[idx]['token']
                    sample = self.nusc.get('sample', token)
                    for sensor in self.sensors:
                        sample_data = self.nusc.get('sample_data', sample['data'][sensor])
                        name = sample_data['filename'].split('/')[-1].replace('.jpg', '.npy')
                        path = os.path.join(skip_dir, sensor, name)
                        print(path)
                        if os.path.exists(path):
                            print(f'Index {idx} camera {sensor} already exists. Skipping...')
                            continue
                        try:
                            assert sample_data['is_key_frame'], \
                                'Error: Cannot get annotations for non keyframes!'
                        except AssertionError as e:
                            print(e)
                            print(f'Index {idx} is not available. Skipping...')
                            continue
                        available_samples.append(sample_data)
                else:
                    print(f'Index {idx} is not available. Skipping...')
                    continue
        else:
            scene_splits = create_splits_scenes()
            split = 'val' if self.mode == 'test' else 'train'
            val_set = [scene for scene in self.nusc.scene if scene['name'] in scene_splits[split]]
            sli = range(len(val_set))
            for idx in tqdm(sli,
                            desc=f'Collecting available nuScenes indices for {self.mode} mode'):
                scene = val_set[idx]
                if self.scene is not None and scene['name'] != self.scene:
                    continue
                first_sample_token = scene['first_sample_token']
                sample = self.nusc.get('sample', first_sample_token)
                while True:
                    for sensor in self.sensors:
                        sample_data = self.nusc.get('sample_data', sample['data'][sensor])
                        name = sample_data['filename'].split('/')[-1].replace('.jpg', '.npy')
                        path = os.path.join(skip_dir, sensor, name)
                        if os.path.exists(path):
                            # print(f'Index {idx} camera {sensor} already exists. Skipping...')
                            continue
                        try:
                            assert sample_data['is_key_frame'], \
                                'Error: Cannot get annotations for non keyframes!'
                        except AssertionError as e:
                            print(e)
                            print(f'Index {idx} is not available. Skipping...')
                            continue
                        available_samples.append(sample_data)
                    if sample['next'] == '':
                        break
                    sample = self.nusc.get('sample', sample['next'])
        return available_samples

    def compute_panoptic_label_in_gt_format(self, semantic: ArrayLike, pred_instance: ArrayLike
                                            ) -> Tuple[ArrayLike, ArrayLike]:

        instance_mask = pred_instance > 0
        instance_per_sem_class = np.zeros_like(pred_instance, dtype=np.uint16)
        for sem_class_id in np.unique(semantic):
            instance_ids = np.unique(pred_instance[semantic == sem_class_id])
            instance_ids = instance_ids[instance_ids > 0]
            total_instances_sem_class = 1
            for instance_id in instance_ids:
                instance_per_sem_class[pred_instance == instance_id] = total_instances_sem_class
                total_instances_sem_class += 1

        panoptic = semantic.copy().astype(np.uint16)
        panoptic[instance_mask] = semantic[instance_mask] * 1000 + instance_per_sem_class[
            instance_mask]

        return semantic, panoptic

    def class_id_to_color(self) -> ArrayLike:
        colors = generate_panoptic_colors(self.nusc.colormap,
                                          self.nusc.lidarseg_name2idx_mapping)
        return colors

    def transform_pc_lidar_to_camera(self, pc: LidarPointCloud,
                                     camera: Any, height: int, width: int,
                                     lidar: Any, min_dist: float = 1.0) -> ArrayLike:
        """
        Returns the pixel coordinates of the pointcloud transformed into the relative camera
        frame for the timestamp of the image.

        Args:
            pc : lidar pointcloud
            camera : camera data from sample
            lidar : lidar data from sample
            min_dist : minimum distance for points to be considered
        Returns:
            points : pixel coordinates of the pointcloud transformed
                     into the relative camera [N, 3] 3rd dim is 1
            mask : mask of points that are within the image
        """
        # First step is to transform the pointcloud into the ego vehicle frame for the timestamp
        cs_record = self.nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.nusc.get('ego_pose', lidar['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into
        # the ego vehicle frame for the timestamp of the image.
        poserecord = self.nusc.get('ego_pose', camera['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
        depths = pc.points[2, :]
        points = view_points(pc.points[:3, :],
                             np.array(cs_record['camera_intrinsic']), normalize=True)
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < width - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < height - 1)
        coloring = np.ones((height, width), dtype=np.float32) * -1
        # Assign the depth values to the points
        coloring[points[1, mask].astype(int), points[0, mask].astype(int)] = depths[mask]
        return coloring

    @property
    def thing_classes(self):
        # TODO: Change this to proper
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    @property
    def stuff_classes(self):
        # return list(self.mapping.values())[:-3][:1]
        # TODO: Change this to proper
        return [0, 11, 12, 13, 14, 15, 16, 17]
    @property
    def stuff_classes_without_thing_like_classes(self):
        # return list(self.mapping.values())[:-3][:1]
        # TODO: Change this to proper
        return [0, 11, 12, 13, 14, 16, 17]
    @property
    def num_stuff(self) -> int:
        return len(self.stuff_classes)

    @property
    def num_things(self) -> int:
        return len(self.thing_classes)

    @property
    def background_classes(self):
        return []

    def __getitem__(self, index: int, do_transform: bool=True) -> Dict[str, Any]:
        output = {}
        sample_data = self.samples[index]
        filename = sample_data['filename']
        path = os.path.join(self.nusc.dataroot, filename)
        image = Image.open(path).convert('RGB')
        image = self.resize(image)
        width, height = image.size
        sample = self.nusc.get('sample', sample_data['sample_token'])
        lidar_token = sample['data']['LIDAR_TOP']
        lidar = self.nusc.get('sample_data', lidar_token)
        lpath = os.path.join(self.nusc.dataroot, lidar['filename'])
        pc = LidarPointCloud.from_file(lpath)
        depth = self.transform_pc_lidar_to_camera(pc, sample_data, height, width, lidar)

        output.update({'rgb': image, 'index': index, 'depth': depth,
                'token': sample_data['token'], 'filename': path})
        if do_transform:
            output = self.transform(output)


        return output

    def __len__(self) -> int:
        return len(self.samples)


class NuScenesDataModule(pl.LightningDataModule):
    """
    This is the dataloader for nuImages dataset using Lightning
    """
    def __init__(self,
                 cfg_dataset: Dict[str, Any],
                 num_classes: int,
                 batch_size: int,
                 num_workers: int,
                 transform_train: List[Callable],
                 transform_test: List[Callable],
                 train_sample_indices: Optional[List[int]] = None,
                 test_sample_indices: Optional[List[int]] = None):
        super().__init__()
        cfg_dataset = CN(init_dict=cfg_dataset)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.train_sample_indices = train_sample_indices
        self.test_sample_indices = test_sample_indices

        self.version = cfg_dataset.version
        self.dataroot = cfg_dataset.dataroot
        self.image_size = cfg_dataset.image_size
        self.verbose = cfg_dataset.verbose
        self.pseudo_train = cfg_dataset.pseudo_train
        self.scene = cfg_dataset.scene

        self.nuscenes_test: Optional[NuScenes] = None

    def setup(self, stage: Optional[str] = None):
        # Assign train/test datasets for use in dataloaders
        mode = 'train' if self.pseudo_train else 'test'
        if stage == 'test':
            self.nuscenes_test = NuScenes(version=self.version,
                                            dataroot=self.dataroot,
                                            transform=self.transform_test,
                                            image_size=self.image_size,
                                            verbose=self.verbose,
                                            indices=self.test_sample_indices,
                                            mode=mode,
                                            scene=self.scene)
        else:
            pass

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.nuscenes_test,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           pin_memory=False,
                                           drop_last=False)
