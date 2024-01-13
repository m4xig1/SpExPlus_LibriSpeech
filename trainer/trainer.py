import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio

import config  # TODO


class Trainer:
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
        **kwargs
    ):
        self.model = model
        self.metrics = metrics
        self.optimizer = optimizer
        self.config = config  # тут какой-то определенный конфиг должен быть
        self.device = device
        self.dataloader = dataloader["train"] # TODO
        self.crit = crit
        self.args = args
        self.kwargs = kwargs
        self.cur_epoch = 0
        self.save_dir = config["save_dir"]  # dir for saving weights, TODO
        self.

    def _train(self, dataloader):
        pass

    def _validate(self, dataloader):
        pass

    # функции для логирования
