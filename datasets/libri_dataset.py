import torch
import numpy as np
from base import BaseDataset
import os
import glob
import soundfile as sf
from torch.utils.data import DataLoader

# PATH = "/home/m4xig1/speaker_extraction_SpEx/"

# print(sorted(os.listdir(PATH + "libri_dataset/mix/train/"))[3:6])


class LibriDataset(BaseDataset):
    def __init__(self, config, path, is_train=True):
        self.path = path
        # индекс создается каждый раз из всех файлов в директории !!
        super().__init__(config, sorted(os.listdir(self.path)))
        self.pos = 0
        self.batch_size = config["batch_size"]  # unused
        self.is_train = is_train  # pretty unuseful feature

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
                triplet["target"] = self.load_audio(self.path + self.index[id * 3 + i])

        if self.is_train: # speaker id for classification
            triplet["speaker_id"] = self.__get_id(self.index[id * 3])

        if (
            "mix" not in triplet
            or "reference" not in triplet
            or "target" not in triplet
            or (self.is_train and "speaker_id" not in triplet)
        ):
            self.logger.info(f"bad triplet starting with: {self.index[self.pos]}")
            return None
        else:
            return triplet

    def __len__(self):
        return len(self.index) / 3

    @staticmethod
    def __get_id(name):
        """
        name of the file must be {target_id}_{noise_id}_{triplet_id}-{mixed/target/ref}.wav
        """
        return int(name.split("_")[0])


def get_train_dataloader(config, dataset):
    return DataLoader(
        dataset=dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        # chunk size?
    )


def get_test_dataloader(config, dataset):
    return DataLoader(
        dataset=dataset,
        batch_size=config["test"]["batch_size"],
        shuffle=False,
        num_workers=config["test"]["num_workers"],
    )


def get_eval_dataloader(config, dataset):
    return DataLoader(
        dataset=dataset,
        batch_size=config["val"]["batch_size"],
        shuffle=False,
        num_workers=config["val"]["num_workers"],
    )
