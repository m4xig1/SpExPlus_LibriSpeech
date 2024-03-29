import logging
import torch
import numpy as np
from torch.utils.data import Dataset
import soundfile as sf
import torchaudio

# from config import config_dataloader


class BaseDataset(Dataset):
    def __init__(self, config, index):
        """
        all init data should be writen in config.
        """
        self.config = config
        super(BaseDataset, self).__init__()
        # self.logger = logging.getLogger(config["logging_name"])
        self.index = index

        self.sr4audio = 16000
        if "sample_rate" in self.config:
            self.sr4audio = self.config["sample_rate"]

    def load_audio(self, path):
        audio, sr = sf.read(path)
        if len(audio.shape) > 1:  # 2 mono
            minDim = np.where(np.mean(audio.shape == np.amin(audio.shape)))[0][0]
            audio = np.mean(audio, axis=minDim)

        audio_tensor = torch.from_numpy(audio)

        if not self.sr4audio is None and self.sr4audio != sr:
            torchaudio.functional.resample(audio_tensor, sr, self.sr4audio)

        return audio_tensor

    def _prep(self):
        return NotImplementedError()

    def __getitem__(self, index):
        return NotImplementedError()

    def __len__(self):
        return len(self.index)
