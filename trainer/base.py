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

# import config
from .config import config_trainer
from logger.logger import start_log
from logger.visualize import get_visualizer
from metrics.base import BaseMetric  # ??
from trainer.DONTDELETE import GIT_GUD
import gc  # garbage collector


def print_cuda_info():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        metrics: "dict[str, BaseMetric]",
        *args,
        config=config_trainer,
        load_checkpoint=None,  # load from
        **kwargs,
    ):
        start_log()  # load config to logger
        self.config = config
        # self.config['logger']
        self.logger = logging.getLogger("train logger")
        self.logger.setLevel(config["logger"]["level"])
        self.reporter = get_visualizer()  # Wandb monitor

        self.log_step = 50  # how many batches between logging

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA is available")
        # load gpuid
        self.device = self.config["device"]
        self.logger.info(f"Running trainer on {self.device}")

        self.model = model.to(self.device)
        self.reporter.watch(model)

        # self.metrics = self._load_to_device(metrics, self.device)
        self.metrics = metrics

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
        if load_checkpoint is not None:
            checkpoint = torch.load(load_checkpoint, map_location="cpu")
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
        self.epoch_len = config["epoch_len"]

        # self.lrScheduler = ReduceLROnPlateau(**config["lrScheduler"])
        self.checkpoint_queue = deque(maxlen=self.config["nCheckpoints"])

        # steps before stopping the training
        self.no_impr = self.config["no_improvment"]
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
        if isinstance(obj, list) or isinstance(obj, tuple):
            return [self.__load_batch(i, device) for i in obj]
        if isinstance(obj, dict):
            for key in obj.keys():
                obj[key] = obj[key].to(device)
            return {key: self.__load_batch(val, device) for key, val in obj.items()}
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

        path = self.checkpoint_path + f"{name}.pt.tar"
        self.logger.info(f"saving checkpoint {path}")
        torch.save(check, path)
        self._process_checkpoint(path)

    # def _load_checkpoint(self, id):
    #     """
    #     Must be rewrited to load model from checkpoint and continue training.
    #     Now model is loading checkpoint in __init__
    #     """
    #     checkpoint = torch.load(self.checkpoint_queue[id], map_location="cpu")
    #     try:
    #         self.model.load_state_dict(checkpoint["model"])
    #     except Exception as e:
    #         self.logger.info(f"Exception while loading checkpoint: {e}")
    #     self.model = self.model.to(self.device)

    def compute_loss(self, batch, is_train=True):
        """
        chunk -- batch from Dataloader with predictions
        returns: dict[None|torch.Tensor] with loss
        """
        raise NotImplementedError()

    def compute_metrics(self, pred, target):
        return {key: metric(pred, target) for key, metric in self.metrics.items()}

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
        return norm.item()

    def train(self, dataloader):
        self.logger.info("Train mode")
        self.model.train()
        logs = {}
        for step, batch in enumerate(tqdm(dataloader, desc="train")):
            try:
                batch = self._load_to_device(batch, self.device)
                self.optimizer.zero_grad()
                batch = self.compute_loss(batch)
                loss = batch["loss"]
                loss.backward()
                self.optimizer.step()

            except RuntimeError as e:  # oom?
                self.logger.info(f"Out of Memory, step: {step}")
                # print_cuda_info()
                for param in self.model.parameters():
                    if param.grad is not None:
                        del param.grad
                torch.cuda.empty_cache()
                gc.collect()
                continue

            # if self.lrScheduler is not None:
            #     self.lrScheduler. # do smth?
            if step % self.log_step == 0:
                self.reporter.new_step(step + self.cur_epoch * self.epoch_len)
                logs = {
                    "loss": batch["loss"].detach().cpu().numpy(),  # copy?
                    "grad_norm": self.calc_grad_norm(),
                }
                self.logger.info(
                    f"step: {step + self.cur_epoch * self.epoch_len}, Loss: {logs['loss']}, Grad norm: {logs['grad_norm']}"
                )
                for name, x in logs.items():
                    self.reporter.log_scalar(name, x)

        self.logger.info("force report")

    def eval(self, dataloader):
        self.logger.info("Eval mode")
        self.model.eval()
        batch_size = len(dataloader)
        logs = {"loss": 0.0, "ce": 0.0, "SI-SDR": 0.0, "PesQ": 0.0}
        with torch.no_grad():
            for step, batch in enumerate(
                tqdm(dataloader, desc="eval", total=len(dataloader))
            ):
                batch = self._load_to_device(batch, self.device)
                metr, result, mixed = self.compute_loss(batch, is_train=False)

                for key in metr:
                    logs[key] += metr[key]

                if step % self.log_step == 0:
                    self.reporter.new_step(step + self.cur_epoch, "eval")
                    self.logger.info(
                        f"Eval step: {step}, Loss: {metr['loss'] / (step + 1)}, CE: {metr['ce'] / (step + 1)}, SI-SDR: {metr['SI-SDR'] / (step + 1)}, PesQ: {metr['PesQ'] / (step + 1)}"  # metrics on batch
                    )
                    for key in logs:
                        self.reporter.log_scalar(key, metr[key])

                    # self.reporter.log_audio("mix", mixed[0])
                    # self.reporter.log_audio("predicted", result[0])
        # force logs
        logs["loss"] /= batch_size
        return logs

    def run(self, trainloader, testloader, nEpochs=50):
        self.save_checkpoint("init checkpoint")
        self.reporter.step = 0
        best_loss = 0

        logs = self.eval(testloader)
        best_loss = logs["loss"]
        self.logger.info(f"Start from epoch {self.cur_epoch}, Loss: {best_loss}")

        no_impr = 0
        while self.cur_epoch < nEpochs:
            self.logger.info(f"Epoch {self.cur_epoch}...")
            self.cur_epoch += 1

            self.train(trainloader)
            logs = self.eval(testloader)

            if self.lrScheduler is not None:
                self.lrScheduler.step(logs["loss"])

            if logs["loss"] < best_loss:
                no_impr += 1
                self.save_checkpoint(f"epoch_{self.cur_epoch}")
                if no_impr >= self.no_impr:
                    self.logger.info(
                        f"Stopping training, no improvment for {no_impr} epochs"
                    )
                    break
            else:
                no_impr = 0
                best_loss = logs["loss"]
                self.save_checkpoint(f"improvment_epoch_{self.cur_epoch}")
            sys.stdout.flush()

        self.logger.info(f"Training for {self.cur_epoch}/{nEpochs} epoches done!")

    # 4 log
    def _log_epoch(self, logs):
        pass
