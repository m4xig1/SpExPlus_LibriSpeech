from trainer.trainer import Trainer
from datasets.libri_dataset import (
    LibriDataset,
    get_train_dataloader,
    get_eval_dataloader,
    get_test_dataloader,
)
from datasets.config import config_dataloader
from model.spex_plus import SpEx_Plus
from model.dummy_model import Dummy
from logger.visualize import get_visualizer
from run_configs.train_config import train_config
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

    model = SpEx_Plus()
    # model = Dummy()
    # parallel?
    # n_gpus = torch.cuda.device_count()
    # logger.info(f"running on {n_gpus} gpus")

    trainer = Trainer(model, metrics)

    trainer.run(train_loader, test_loader, 1)


if __name__ == "__main__":
    # parse args or use config?
    main()

    # test the model
    # train_loader = get_train_dataloader(config_dataloader)  
    # for num, batch in enumerate(train_loader):
        # print(batch['mix'].shape, batch['reference'].shape, batch['mix'].dtype, batch['reference'].dtype)
        # result = model(batch["mix"], batch["reference"], batch["ref_len"])
        # print(result)
        # break 


    # model = SpEx_Plus()
    # mix = torch.ones(1, 1000)
    # ref = torch.ones(1, 500)
    # result = model(mix, ref, torch.tensor([500]))
    # print(result)
    # print(result["logits"].shape)
        
