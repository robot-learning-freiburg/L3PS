import hydra
from omegaconf import DictConfig

from accumulate import run as accumulate_run
from evaluate import run as evaluate_run
from generate import run as generate_run
from refine import run as refine_run


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    generate_run(cfg)
    accumulate_run(cfg)
    refine_run(cfg)
    evaluate_run(cfg)

if __name__ == "__main__":
    main()
