import logging
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio
import torch.nn as nn

from .base import BaseTrainer
from .config import config_trainer
from logger.logger import start_log
from logger.visualize import get_visualizer
from trainer.DONTDELETE import GIT_GUD
from loss import SpexPlusLoss


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        metrics: dict,
        *args,
        config=config_trainer,
        **kwargs
    ):
        super().__init__(model, metrics, config, *args, **kwargs)
        self.loss = SpexPlusLoss()
        # self.loss = self._load_to_device(self.loss, self.device)

    def compute_loss(self, batch: dict, is_train=True):
        # est = self.model(batch["mix"], batch["reference"], batch["ref_len"])
        est = self.model(batch["mix"], batch["reference"], batch["ref_len"]) # batch size = 1?
        return self.loss.forward(est, batch["target"], batch["speaker_id"], is_train)
    
    # est_short, est_mid, est_long, pred_spk = torch.nn.parallel.data_parallel(self.model, (batch['mix'], batch['ref'], batch['len']), self.cpuid) # > 1 devices?
    # do smth to 

    def extract_predictions(self, batch, is_train=False):
        batch = self._load_to_device(batch, self.device)
        outputs = self.model(batch)
        # update metrics ...
        return outputs
