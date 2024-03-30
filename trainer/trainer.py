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
        self, model: nn.Module, metrics: dict, *args, config=config_trainer, **kwargs
    ):
        super().__init__(model, metrics, config, *args, **kwargs)
        self.loss = SpexPlusLoss()
        # self.loss = self._load_to_device(self.loss, self.device)

    @staticmethod
    def __mask_pred(pred, shape):
        """
        Assume shape > pred.shape, we pad pred with zeroes to shape.
        """
        new_pred = torch.zeros(shape[0], shape[1])
        new_pred[:, : pred.shape[1]] = pred
        return new_pred

    def compute_loss(self, batch: dict, is_train=True):
        # print(batch["mix"].shape,  batch["reference"].shape, batch["ref_len"].shape)

        if batch["mix"].dim() < 2:  # make batch size 1
            batch["mix"] = batch["mix"].unsqueeze(0)
            batch["target"] = batch["target"].unsqueeze(0)
            batch["reference"] = batch["reference"].unsqueeze(0)

        loss = {}
        metrics = {}
        pred = self.model(batch["mix"], batch["reference"], batch["ref_len"])

        if pred["short"].shape != batch["mix"].shape:
            # maby i should pad for all of pred
            pred["short"] = self.__mask_pred(pred["short"], batch["mix"].shape)

        if not is_train:
            # calc according to short
            metr = self.compute_metrics(pred["short"], batch["target"])
            loss = self.loss(pred, batch["target"], batch["speaker_id"], is_train)
            return {**metr, **loss}, pred, batch["mix"]

        return self.loss(pred, batch["target"], batch["speaker_id"], is_train)

    # est_short, est_mid, est_long, pred_spk = torch.nn.parallel.data_parallel(self.model, (batch['mix'], batch['ref'], batch['len']), self.cpuid) # > 1 devices?
    # do smth to

    def extract_predictions(self, batch: dict, is_train=False):
        outputs = self.model(batch["mix"], batch["reference"], batch["ref_len"])
        # update metrics ...
        return outputs
