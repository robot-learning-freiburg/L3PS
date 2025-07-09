import pickle
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from omegaconf import DictConfig


def build_dataset(cfg: DictConfig) -> NuScenes:
    """
    :param cfg: DictConfig
    :return: nuScenes API instance.
    """
    version = cfg.version
    dataroot = cfg.path
    return NuScenes(version=version, dataroot=dataroot, verbose=True)


class Dataset:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.nusc: NuScenes = build_dataset(cfg)
        self.scene_sample_tokens, self.scene_list = self._preprocess()
        self.sample_tokens = []
        for scene_tokens in self.scene_sample_tokens.values():
            self.sample_tokens += scene_tokens

    def __len__(self):
        return len(self.sample_tokens)

    def _preprocess(self):
        scenes = create_splits_scenes()[self.cfg.split]
        scene_sample_tokens = {}
        scene_list = []
        for scene in self.nusc.scene:
            if scene["name"] in scenes:
                scene_list.append(scene["name"])
                scene_sample_tokens[scene["name"]] = self.nusc.field2token(
                    "sample", "scene_token", scene["token"]
                )
        return scene_sample_tokens, scene_list

    def get_scene(self, idx: int) -> str:
        """
        :param idx: Index of the scene.
        :return: Scene token.
        """
        return self.scene_list[idx]

    def get_scene_samples(self, scene_name: str) -> list:
        """
        :param scene_name: Name of the scene.
        :return: List of sample tokens.
        """
        return self.scene_sample_tokens[scene_name]

    def get_sample(self, idx: int) -> str:
        """
        :param idx: Index of the sample.
        :return: Sample token.
        """
        return self.sample_tokens[idx]


class FastDataset:
    def __init__(self, cfg: DictConfig):
        # Get current file directory
        info_path = Path(__file__).parent / 'nuscenes_info.pkl'
        self.info = pickle.load(open(info_path, "rb"))
        self.mapper = self.info["mapper"]
        self.path = Path(self.info["rootdir"])
        self.info_split = self.info[cfg.split]
        self.data = self.info_split["scenes"]
        self.scenes = list(self.data.keys())

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx: int):
        scene = self.scenes[idx]
        return self.data[scene]
