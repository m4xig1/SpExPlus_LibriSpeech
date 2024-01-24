from collections import deque
import logging
from multiprocessing import process
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio
from zmq import device
import logger
import torch.nn as nn

import config
from logger.logger import start_log
from logger.visualize import get_visualizer
from trainer.DONTDELETE import GIT_GUD

# я обязательно напишу комментарии к коду


class Trainer:
    def __init__(
        self,
        model,
        metrics: dict,
        optimizer,
        config,
        dataloader,
        *args,
        **kwargs,
    ):
        start_log()  # load config to logger
        self.config = config
        self.logger = logging.getLogger("train logger")
        self.logger.setLevel(config["logger"]["level"])
        self.reporter = get_visualizer()  # Wandb monitor

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA is available")
        # load gpuid
        self.device = "gpu"
        self.logger.info("Running trainer on gpu")

        self.model = model.to(self.device)

        self.metrics = self._load_to_device(metrics, self.device)
        self.optimizer = optimizer  # возможно, тут его загрузить надо
        self.lrScheduler = None  # ?????????

        # self.dataloader = dataloader["train"]  # TODO
        # self.epoch_len = len(self.dataloader)

        self.checkpoint_path = config["checkpoint_path"]  # saving, loading here
        if self.checkpoint_path is not None:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            try:
                self.model.load_state_dict(checkpoint["model"])
            except Exception as e:
                self.logger.info(f"Exception while loading checkpoint: {e}")
            self.model = self.model.to(self.device)

        self.checkpoint_queue = deque(maxlen=self.config["nCheckpoints"])

        self.args = args
        self.kwargs = kwargs
        self.cur_epoch = 0
        # self.save_dir = config["save_dir"]  # ??
        self.logger.info(GIT_GUD)

    def __load_batch(self, obj, device):
        return obj.to(device) if isinstance(obj, torch.Tensor) else obj

    def _load_to_device(self, obj, device):
        """
        loads obj to device if obj contains Tensor
        """
        if isinstance(obj, list):
            return [self.__load_batch(i, device) for i in obj]
        if isinstance(obj, dict):
            return {key: self.__load_batch(val, device) for val, key in obj.items()}
        return self.__load_batch(obj, device)

    def _process_checkpoint(self, path):
        if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
            os.remove(self.checkpoint_queue[0])
        self.checkpoint_queue.append(path)

    def save_checkpoint(self, name="checkpoint"):
        """
        Saves checkpoint in config["checkpoint_dir"]
        """
        check = {
            "epoch": self.cur_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lrScheduler.state_dict(),
        }
        path = self.checkpoint_path.join(f"{name}.pt.tar")
        self.logger.info(f"saving checkpoint {path}")
        torch.save(check, path)
        self._process_checkpoint(path)

    def _load_checkpoint(self, id):
        """
        Must be rewrited to load model from checkpoint and continue training
        """
        checkpoint = torch.load(self.checkpoint_queue[id], map_location="cpu")
        try:
            self.model.load_state_dict(checkpoint["model"])
        except Exception as e:
            self.logger.info(f"Exception while loading checkpoint: {e}")
        self.model = self.model.to(self.device)

    def compute_loss(self, chunk):
        """
        chunk -- batch from Dataloader with predictions
        returns: dict[None|torch.Tensor] with loss
        """
        return NotImplementedError()

    def compute_metrics(self, pred, target):
        return {
            key: metric.forward(pred, target)
            for key, metric in self.metrics.items()  # ало че ты светишься
        }

    def train(self, dataloader):
        
        self.logger.info("Train")
        self.model.train()
        batch_size = len(dataloader)

        for step, batch in enumerate(dataloader):
            batch = self._load_to_device(batch, device)
            self.optimizer.zero_grad()
            loss = self.compute_loss(batch)

    def _train_epoch(self):
        pass

    def run(self):
        pass

    def validate(self, dataloader):  # inference
        pass

    # 4 log
    def _log_epoch(self):
        pass
