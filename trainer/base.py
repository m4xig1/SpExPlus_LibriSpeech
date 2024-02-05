from collections import deque
import logging
from multiprocessing import process
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio
import logger
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from logger.logger import start_log
from logger.visualize import get_visualizer
from trainer.DONTDELETE import GIT_GUD


class BaseTrainer:
    def __init__(
        self,
        model,
        metrics: dict,
        config,
        *args,
        **kwargs,
    ):
        start_log()  # load config to logger
        self.config = config
        self.logger = logging.getLogger("train logger")
        self.logger.setLevel(config["logger"]["level"])
        self.reporter = get_visualizer()  # Wandb monitor
        self.log_step = 50  # how many batches between logging

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA is available")
        # load gpuid
        self.device = "gpu"
        self.logger.info("Running trainer on gpu")

        self.model = model.to(self.device)

        self.metrics = self._load_to_device(metrics, self.device)

        # self.optimizer = optimizer  # возможно, тут его загрузить надо
        # self.lrScheduler = None
        """
        Temporary solution for optimizer and scheduler, better to load modules via hydra
        """
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config["optimizer"]["lr"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
        self.lrScheduler = None

        self.checkpoint_path = config["checkpoint_path"]  # saving, loading here
        if self.checkpoint_path is not None:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            try:
                self.model.load_state_dict(checkpoint["model"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])  # ?
                # load scheduler, optimizer?
            except Exception as e:
                self.logger.info(f"Exception while loading checkpoint: {e}")
            self.model = self.model.to(self.device)
        else:
            self.lrScheduler = ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode=config["lrScheduler"]["mode"],
                factor=config["lrScheduler"]["factor"],
                patience=config["lrScheduler"]["patience"],
                min_lr=config["lrScheduler"]["min_lr"],
                verbose=config["lrScheduler"]["verbose"],
            )
            # self.lrScheduler = ReduceLROnPlateau(**config["lrScheduler"])
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
        }
        check["scheduler"] = (
            self.lrScheduler.state_dict() if self.lrScheduler is not None else None
        )

        path = self.checkpoint_path.join(f"{name}.pt.tar")
        self.logger.info(f"saving checkpoint {path}")
        torch.save(check, path)
        self._process_checkpoint(path)

    def _load_checkpoint(self, id):
        """
        Must be rewrited to load model from checkpoint and continue training.
        Now model is loading checkpoint in __init__
        """
        checkpoint = torch.load(self.checkpoint_queue[id], map_location="cpu")
        try:
            self.model.load_state_dict(checkpoint["model"])
        except Exception as e:
            self.logger.info(f"Exception while loading checkpoint: {e}")
        self.model = self.model.to(self.device)

    def compute_loss(self, batch) -> dict:
        """
        chunk -- batch from Dataloader with predictions
        returns: dict[None|torch.Tensor] with loss
        """
        raise NotImplementedError()

    def compute_metrics(self, pred, target):
        return {
            key: metric.forward(pred, target)
            for key, metric in self.metrics.items()  # ало че ты светишься
        }

    @torch.no_grad()
    def calc_grad_norm(self, p_type=2):
        """
        calc L_p norm of the gradient, using to prevent grad explosions
        """
        params = [
            p for p in self.model.parameters() if p.grad is not None and p.requires_grad
        ]
        norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), p_type) for p in params]).cpu(),
            p_type,
        )
        # print(norm)
        return norm.item()

    def train(self, dataloader):
        self.logger.info("Train mode")
        self.model.train()
        batch_size = len(dataloader)
        logs = {}
        for step, batch in enumerate(tqdm(dataloader)):
            batch = self._load_to_device(batch, self.device)
            self.optimizer.zero_grad()
            batch = self.compute_loss(batch)
            loss = batch["loss"]
            self.logger.info("Loss: {loss}")
            loss.backward()
            self.optimizer.step()
            # if self.lrScheduler is not None:
            #     self.lrScheduler. # do smth?
            logs = {
                "loss": batch["loss"].detach().cpu().numpy(),  # copy?
                "step": (step + 1 + self.cur_epoch * batch_size),
                "grad_norm": self.calc_grad_norm(),
                "progress": (step + 1 + self.cur_epoch * batch_size) / batch_size,
            }
            self.reporter.new_step(logs["step"])
            self.logger.debug(
                f"step: {logs['step']}, Loss: {logs['loss']}, Grad norm: {logs['grad_norm']}"
            )
            for name, x in logs.items():
                self.reporter.log_scalar(name, x)
            # self._log_epoch(logs)

        self.logger.info("force report")

    def eval(self, dataloader):
        self.logger.info("Eval mode")
        self.model.eval()
        batch_size = len(dataloader)
        logs = {"loss": 0.0}
        with torch.no_grad():
            for step, batch in tqdm(enumerate(dataloader)):
                batch = self._load_to_device(batch, self.device)
                batch = self.compute_loss(batch)
                logs["loss"] += batch["loss"]
                # logging here
                
            logs["loss"] /= batch_size
        # force logs

        return logs

    def run(self, trainloader, testloader, nEpochs=50):
        self.save_checkpoint("init checkpoint")
        self.reporter.step = 0
        logs = self.eval(testloader)
        best_loss = logs["loss"]
        while self.cur_epoch < nEpochs:
            self.logger.info(f"Epoch {self.cur_epoch}...")
            self.cur_epoch += 1

            self.train(trainloader)
            logs = self.eval(testloader)
            # good time to log something
            if self.lrScheduler is not None:
                self.lrScheduler.step(logs["loss"])
            # if (logs["loss"] < best_loss):
            self.save_checkpoint(f"epoch_{self.cur_epoch}")

            sys.stdout.flush()

        self.logger.info(f"Training for {self.cur_epoch}/{nEpochs} epoches done!")

    # 4 log
    def _log_epoch(self, logs):
        pass
