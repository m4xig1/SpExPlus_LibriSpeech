import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio
import logger
import torch.nn as nn

import config
from logger.logger import start_log  # TODO


class Trainer(nn.Module):
    def __init__(
        self,
        model,
        metrics,
        optimizer,
        config,
        device,
        dataloader,
        crit,
        *args,
        **kwargs,
    ):
        super().__init__(self, Trainer)

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA is available")
        # load gpuid

        self.model = model
        self.metrics = metrics
        self.optimizer = optimizer
        self.config = config  # тут какой-то определенный конфиг должен быть
        self.device = device

        self.dataloader = dataloader["train"]  # TODO
        self.epoch_len = len(self.dataloader)

        self.crit = crit
        self.args = args
        self.kwargs = kwargs
        self.cur_epoch = 0
        self.save_dir = config["save_dir"]  # dir for saving weights, TODO
        start_log()  # load config to logger

    def _save_checkpoint(self, name="checkpoint", best=True):
        check = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
        }
        torch.save(check, self.save_dir.join(f"{name}.pt.tar"))

    def _load_checkpoint(self, id):
        pass

    def train(self, dataloader):
        pass

    def _train_epoch(self):
        pass

    def run(self):
        pass

    def validate(self, dataloader):  # inference
        pass

    # 4 log
    def _log_epoch(self):
        pass
