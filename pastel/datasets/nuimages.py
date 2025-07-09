
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from nuimages import NuImages as NuIm
from numpy.typing import ArrayLike
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from yacs.config import CfgNode as CN


class NuImages:
    # Colormap for 12 classes in nuImages dataset
    # github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py

    CLASS_COLOR_12 = np.zeros((256, 3), dtype=np.uint8)
    CLASS_COLOR_12[:12, :] = np.array([
        [244, 35, 232], # void
        [112, 128, 144], # barrier
        [220, 20, 60], # bicycle
        [255, 127, 80], # bus
        [255, 158, 0], # car
        [233, 150, 70], # construction vehicle
        [255, 61, 99], # motorcycle
        [0, 0, 230], # person
        [47, 79, 79], # traffic cone
        [255, 140, 0], # trailer
        [255, 99, 71], # truck
        [0, 207, 191], # road
    ])
    def __init__(self,
                 version: str,
                 dataroot: str,
                 transform: List[Callable],
                 image_size: List[int],
                 verbose: bool = False,
                 lazy: bool = False,
                 indices: List[int] = None,
                 mode: bool = 'train',
                 n_thing_class: int = 10,
                 ann_root: str = 'nuim_anns',):

        assert version in ['v1.0-train', 'v1.0-mini', 'v1.0-test', 'v1.0-val'],\
            'Error: Invalid version!'
        self.transform = transforms.Compose(transform)
        self.resize = transforms.Resize(image_size,
                                        interpolation=transforms.InterpolationMode.LANCZOS)
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.image_size = image_size

        self.path_base = dataroot
        self.mode = mode
        self.nuim = NuIm(version=version, dataroot=dataroot, verbose=verbose, lazy=lazy)
        self.indices = self.get_available_indices(indices=indices)
        self.ann_root = ann_root
        self.n_things = n_thing_class

    def get_available_indices(self, indices: List[int] = None):
        available_indices = []
        sample_list = self.nuim.sample
        if indices is not None:
            for idx in tqdm(indices,
                            desc=f'Collecting available nuImages indices for {self.mode} mode'):
                if idx in range(len(sample_list)):
                    token = sample_list[idx]['token']
                    sample = self.nuim.get('sample', token)
                    key_camera_token = sample['key_camera_token']
                    sample_data = self.nuim.get('sample_data', key_camera_token)
                    try:
                        assert sample_data['is_key_frame'], \
                            'Error: Cannot get annotations for non keyframes!'
                        self.nuim.check_sweeps(sample_data['filename'])
                    except AssertionError as e:
                        print(e)
                        print(f'Index {idx} is not available. Skipping...')
                        continue
                    available_indices.append(idx)
                else:
                    print(f'Index {idx} is not available. Skipping...')
                    continue
        else:
            sli = range(len(sample_list))
            for idx in tqdm(sli,
                            desc=f'Collecting available nuImages indices for {self.mode} mode'):
                token = sample_list[idx]['token']
                sample = self.nuim.get('sample', token)
                key_camera_token = sample['key_camera_token']
                sample_data = self.nuim.get('sample_data', key_camera_token)
                try:
                    assert sample_data['is_key_frame'], \
                        'Error: Cannot get annotations for non keyframes!'
                    self.nuim.check_sweeps(sample_data['filename'])
                except AssertionError as e:
                    print(e)
                    print(f'Index {idx} is not available. Skipping...')
                    continue
                available_indices.append(idx)
        return available_indices

    def get_segmentation(self, key_camera_token: str, height: int, width: int):
        # get filename
        sample_data = self.nuim.get('sample_data', key_camera_token)
        filename = sample_data['filename']
        # load segmentation from nuim_anns/semantic and nuim_anns/instance
        last = filename.split('/')[-1].replace('jpg', 'npy')
        semantic_mask = np.load(os.path.join(self.ann_root, 'semantic', last))
        instance_mask = np.load(os.path.join(self.ann_root, 'instance', last))
        return semantic_mask, instance_mask

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

    def class_id_to_color(self):
        return self.CLASS_COLOR_12

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
        index = self.indices[index]
        output = {}
        key_camera_token = self.nuim.sample[index]['key_camera_token']
        sample_data = self.nuim.get('sample_data', key_camera_token)
        filename = sample_data['filename']
        path = os.path.join(self.nuim.dataroot, filename)
        image = Image.open(path).convert('RGB')
        (width, height) = image.size
        image = self.resize(image)
        output.update({'rgb': image, 'index': index,
                       'token': sample_data['token'], 'filename': path})

        semantic_mask, instance_mask = self.get_segmentation(key_camera_token, height, width)
        semantic_mask = cv2.resize(semantic_mask, list(reversed(self.image_size)),
                                   interpolation=cv2.INTER_NEAREST)
        instance_mask = cv2.resize(instance_mask, list(reversed(self.image_size)),
                                   interpolation=cv2.INTER_NEAREST)

        ego_car_mask = semantic_mask == 0
        output.update({'semantic': semantic_mask, 'instance': instance_mask})
        semantic_path = os.path.join(self.nuim.dataroot, 'semantic_'+filename)
        instance_path = os.path.join(self.nuim.dataroot, 'instance_'+filename)
        output.update({'semantic_path': semantic_path, 'instance_path': instance_path,
                       'ego_car_mask': ego_car_mask})

        if do_transform:
            output = self.transform(output)

        return output

    def __len__(self) -> int:
        return len(self.indices)


class NuImagesDataModule(pl.LightningDataModule):
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
        self.lazy = cfg_dataset.lazy
        self.n_thing_class = cfg_dataset.n_thing_class
        self.ann_root = cfg_dataset.ann_root

        self.nuimages_train: Optional[NuImages] = None
        self.nuimages_test: Optional[NuImages] = None

    def setup(self, stage: Optional[str] = None):
        # Assign train/test datasets for use in dataloaders
        if stage == 'fit':
            self.nuimages_train = NuImages(version=self.version,
                                             dataroot=self.dataroot,
                                             transform=self.transform_train,
                                             image_size=self.image_size,
                                             verbose=self.verbose,
                                             lazy=self.lazy,
                                             indices=self.train_sample_indices,
                                             mode='train',
                                             n_thing_class=self.n_thing_class,
                                             ann_root=self.ann_root)
        elif stage in ('validate', 'predict'):
            pass

        elif stage == 'test':
            self.nuimages_test = NuImages(version=self.version,
                                            dataroot=self.dataroot,
                                            transform=self.transform_test,
                                            image_size=self.image_size,
                                            verbose=self.verbose,
                                            lazy=self.lazy,
                                            indices=self.test_sample_indices,
                                            mode='test',
                                            n_thing_class=self.n_thing_class)
        else:
            pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.nuimages_train,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=True,
                                           pin_memory=False)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.nuimages_test,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           pin_memory=False,
                                           drop_last=False)
