import logging
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio
import logger
import torch.nn as nn

from base import BaseTrainer
import config
from logger.logger import start_log
from logger.visualize import get_visualizer
from trainer.DONTDELETE import GIT_GUD
from loss import SpexPlusLoss


class Trainer(BaseTrainer):
    def __init__(
        self, model, metrics: dict, optimizer, config, dataloader, *args, **kwargs
    ):
        super().__init__(model, metrics, optimizer, config, dataloader, *args, **kwargs)
        self.loss = SpexPlusLoss()
        self.loss = self._load_to_device(self.loss, self.device)

    def compute_loss(self, batch):
        return self.loss.forward(batch)  # change inputs in loss class

    def extract_predictions(self, batch, is_train=False):
        batch = self._load_to_device(batch, self.device)
        outputs = self.model(**batch)
        # update metrics ...
        return outputs
