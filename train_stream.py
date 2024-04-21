from loss.loss import SpexPlusLoss
from trainer.stream_trainer import StreamTrainer
from datasets.libri_dataset import (
    LibriDataset,
    get_train_dataloader,
    get_test_dataloader,
)
from datasets.config import config_dataloader
from model.spex_plus import SpEx_Plus
from logger.visualize import get_visualizer
from metrics.metrics import SiSdr, Pesq
from utils.stream import separate_sources

import torch
import numpy as np
import os
import logging
import wandb
from itertools import repeat
from torch.optim.lr_scheduler import ReduceLROnPlateau

from trainer.config import config_stream as config

torch.manual_seed(42)
np.random.seed(42)


def main():

    train_loader, count_speakers = get_train_dataloader(config_dataloader)
    test_loader = get_test_dataloader(config_dataloader)
    dataloaders = {"train": train_loader, "test": test_loader}

    model = SpEx_Plus(num_speakers=count_speakers)

    logger = logging.getLogger("train")

    trainable_model_params = filter(
        lambda param: param.requires_grad, model.parameters()
    )
    loss = SpexPlusLoss()
    optimizer = torch.optim.Adam(
        params=trainable_model_params,
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    lr_scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode=config["lr_scheduler"]["mode"],
        factor=config["lr_scheduler"]["factor"],
        patience=config["lr_scheduler"]["patience"],
        min_lr=config["lr_scheduler"]["min_lr"],
        verbose=config["lr_scheduler"]["verbose"],
    )

    metrics = {"SI-SDR": SiSdr(), "PesQ": Pesq()}

    trainer = StreamTrainer(
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        dataloaders=dataloaders,
        config=config,
        lr_scheduler=lr_scheduler,
        len_epoch=config["epoch_len"],
        skip_oom=True,
        separate_sources=separate_sources,
    )

    trainer.train()


if __name__ == "__main__":
    # parse args or use config?
    main()
