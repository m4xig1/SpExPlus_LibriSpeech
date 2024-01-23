import torch
import numpy as np
from base import BaseDataset
import os
import glob
import soundfile as sf

# PATH = "/home/m4xig1/speaker_extraction_SpEx/"

# print(sorted(os.listdir(PATH + "libri_dataset/mix/train/"))[3:6])


class LibriDataset(BaseDataset):
    def __init__(self, config, path):
        self.path = path
        super().__init__(config, sorted(os.listdir(self.path)))
        self.pos = 0
        self.batch_size = config["batch_size"]  # unused
        # self.index = sorted(os.listdir(self.path))  # tmp solution, better 2 use index file

    def __getitem__(self, id):
        triplet = {}
        for i in range(3):
            if "mixed" in self.index[id * 3 + i]:
                triplet["mix"] = self.load_audio(self.path + self.index[id * 3 + i])
            elif "ref" in self.index[id * 3 + i]:
                triplet["reference"] = self.load_audio(
                    self.path + self.index[id * 3 + i]
                )
            elif "target" in self.index[id * 3 + i]:
                triplet["target"] = self.load_audio(
                    self.path + self.index[id * 3 + i]
                )
        if (
            "mix" not in triplet
            or "reference" not in triplet
            or "target" not in triplet
        ):
            self.logger.info(f"bad triplet starting with: {self.index[self.pos]}")
            return None
        else:
            return triplet

    def __len__(self):
        return len(self.index) / 3

    # def _load_triplet(self, id):
    #     pass


def get_train_dataloader():
    pass


def get_test_dataloader():
    pass


def get_eval_dataloader():
    pass
