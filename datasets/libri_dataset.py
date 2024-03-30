from pyparsing import col
import torch
import numpy as np
from .base import BaseDataset
import os
import glob
import soundfile as sf
from torch.utils.data import DataLoader
import json
from pathlib import Path
import logging
from typing import List


class LibriDataset(BaseDataset):
    def __init__(self, config, path, is_train=True, create_index=True):

        self.config = config
        self.path = path
        self.logger = logging.getLogger(config["logging_name"])
        if create_index:  # create index
            index = self._create_index(save=True)
        else:
            name = "train_index.json" if "train" in self.path else "test_index.json"
            with Path(self.config["index_path"] + name).open() as f:  # load index
                index = json.load(f)

        super().__init__(config, index)
        self.pos = 0
        self.is_train = is_train  # pretty unuseful feature

        # self.index = sorted(os.listdir(self.path))  # tmp solution, better 2 use index file

    def _create_index(self, save=False):
        self.logger.info(f"creating index from {self.path}")

        mix = np.array(sorted(glob.glob(self.path + "*-mixed.wav")))
        ref = np.array(sorted(glob.glob(self.path + "*-ref.wav")))
        target = np.array(sorted(glob.glob(self.path + "*-target.wav")))

        ids = np.array([self.__get_id(path) for path in mix])  # already sorted

        for a, b, c in zip(mix, ref, target): # test
            if (a.split('-')[0] != b.split('-')[0] or a.split('-')[0] != c.split('-')[0]):
                print(a,b,c)
                exit(1)

        if mix.shape != ref.shape or ref.shape != target.shape: # test
            self.logger.warning(f"mix.shape != ref.shape || ref.shape != target.shape")
            print(mix.shape, ref.shape, target.shape)
            return None

        ce_ids = dict()  # ids for classification
        free_id = 0
        index = []
        for i in range(mix.size):
            if ids[i] not in ce_ids:
                ce_ids[ids[i]] = free_id
                free_id += 1
            triplet = {
                "mix": mix[i],
                "ref": ref[i],
                "target": target[i],
                "speaker_id": ce_ids[ids[i]],
            }
            index.append(triplet)

        if save:
            name = "train_index.json" if "train" in self.path else "test_index.json"
            with Path(self.config["index_path"] + name).open("w") as f:
                json.dump(index, f, indent=1)

        return index

    def __getitem__(self, id):
        triplet = {
            "mix": self.load_audio(self.index[id]["mix"]).type(torch.float32),
            "reference": self.load_audio(self.index[id]["ref"]).type(torch.float32),
            "target": self.load_audio(self.index[id]["target"]).type(torch.float32),
            "speaker_id": self.index[id]["speaker_id"],
        }
        triplet["ref_len"] = len(triplet["reference"])
        return triplet

    @staticmethod
    def __get_id(name):
        """
        name of the file must look like path/{target_id}_{noise_id}_{triplet_id}-{mixed/target/ref}.wav
        """
        return int(name.split("/")[-1].split("_")[0])


def collate_fn(batch: List[dict]):
    """
    Pad data in batch here.
    """
    pad_batch = {}
    # audio is mono: (1, N) -> (N,) -> (batch_size, N) 
    pad_batch["reference"] = torch.nn.utils.rnn.pad_sequence(
        [elem["reference"].squeeze(0) for elem in batch], batch_first=True
    )

    pad_batch["target"] = torch.nn.utils.rnn.pad_sequence(
        [elem["target"].squeeze(0) for elem in batch], batch_first=True
    )

    pad_batch["mix"] = torch.nn.utils.rnn.pad_sequence(
        [elem["mix"].squeeze(0) for elem in batch], batch_first=True
    )

    pad_batch["speaker_id"] = torch.tensor([elem["speaker_id"] for elem in batch])
    # padded len?
    pad_batch["ref_len"] = torch.tensor([elem["ref_len"] for elem in batch])

    return pad_batch


def get_train_dataloader(config):
    dataset = LibriDataset(
        config["train"], config["path_to_train"], create_index=config["train"]["create_index"]
    )
    return DataLoader(
        dataset=dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        collate_fn=collate_fn,
    )


def get_test_dataloader(config):
    dataset = LibriDataset(
        config["test"], config["path_to_val"], create_index=config["test"]["create_index"]
    )
    return DataLoader(
        dataset=dataset,
        batch_size=config["test"]["batch_size"],
        shuffle=False,
        num_workers=config["test"]["num_workers"],
        collate_fn=collate_fn,
    )


# def get_eval_dataloader(config):
#     dataset = LibriDataset(
#         config["val"], config["path_to_val"], create_index=config["val"]["create_index"]
#     )
#     return DataLoader(
#         dataset=dataset,
#         batch_size=config["val"]["batch_size"],
#         shuffle=False,
#         num_workers=config["val"]["num_workers"],
#         collate_fn=collate_fn,
#     )
