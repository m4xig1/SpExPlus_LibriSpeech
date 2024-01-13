from datetime import datetime
from operator import imod
from matplotlib.pyplot import step
import wandb
import numpy as np


class WandbVisualizer:
    def __init__(self, config):
        wandb.login()
        wandb.init(
            project=config["project_name"],
            name=config["run_name"],
            config=config["config"],
        )  # in trainer?
        self.wandb = wandb
        self.step = 0  # step for log
        self.time = datetime.now()
        # print(f"WanDB logger, log: name_step, project: {config['project_name']}")

    def new_step(self, step_number):
        epoch = datetime.now() - self.time
        if (self.step != 0):
            self.log_scalar("steps-per-sec", 1 / (epoch.total_seconds() + 1e-9))  # ?
        self.step = step_number
        self.time = datetime.now()

    def _log_name(self, name):
        return f"{name}_{self.step}"

    def log_scalar(self, name, scalar):
        self.wandb.log({f"{name}_{self.step}": scalar}, step=self.step)

    def log_audio(self, name, audio, sr=None):
        self.wandb.log(
            {f"{name}_{self.step}": self.wandb.Audio(audio, sr)}, step=self.step
        )

    def log_text(self, name, text=None):
        self.wandb.log({f"{name}_{self.step}": self.wandb.Html(name)}, step=self.step)

    def log_hist(self, name, hist=None):
        raise NotImplementedError()

    def log_table(self, name, table=None):
        self.wandb.log(
            {f"{name}_{self.step}": self.wandb.Table(dataframe=table)}, step=self.step
        )


def get_visualizer(config):
    return WandbVisualizer(config)

from config import config_wdb
get_visualizer(config_wdb)