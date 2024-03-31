from trainer.trainer import Trainer
from datasets.libri_dataset import (
    LibriDataset,
    get_train_dataloader,
    # get_eval_dataloader,
    get_test_dataloader,
)
from datasets.config import config_dataloader
from model.spex_plus import SpEx_Plus
from model.dummy_model import Dummy
from logger.visualize import get_visualizer

# from run_configs.train_config import train_config
from metrics.metrics import SiSdr, Pesq

import torch
import numpy as np
import os
import logging
import wandb


def main():
    # dataset_train = LibriDataset(config_dataloader, config_dataloader["path_to_train"])
    # dataset_val = LibriDataset(config_dataloader, config_dataloader["path_to_val"])
    train_loader = get_train_dataloader(config_dataloader)
    test_loader = get_test_dataloader(config_dataloader)
    logger = logging.getLogger("train")
    metrics = {"SI-SDR": SiSdr(), "PesQ": Pesq()}

    # по хорошему, такие параметры, как количество спикеров для кросс-энтропии стоит вынести в отдельный конфиг, или парсить их здесь
    model = SpEx_Plus()

    trainer = Trainer(model, metrics)

    trainer.run(train_loader, test_loader, 1)


if __name__ == "__main__":
    # parse args or use config?
    print("import wandb key")
    key = input()
    wandb.login("never", key)
    main()
