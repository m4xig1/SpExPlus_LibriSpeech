import logging
import pandas as pd
import torch

from .base import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


logger = logging.Logger("metrics")


class SiSdr(BaseMetric):
    def __init__(self, **kwargs):
        super(SiSdr, self).__init__()
        self.si_sdr = ScaleInvariantSignalDistortionRatio()

    def forward(self, pred, target):
        if pred.shape != target.shape:
            logger.info(f"bad shape in Si-Sdr: {pred.shape} != {target.shape}")
            return 0
        self.metric = self.si_sdr.to(pred.device)  # load to device
        return self.metric(pred, target).item()


class Pesq(BaseMetric):
    def __init__(self, sample_rate=16000, mode="wb", **kwargs):
        super(Pesq, self).__init__()
        self.pesq = PerceptualEvaluationSpeechQuality(sample_rate, mode)

    def forward(self, pred, target):
        if pred.shape != target.shape:
            logger.info(f"bad shape in PesQ: {pred.shape} != {target.shape}")
            return 0

        self.metric = self.pesq.to(pred.device)  # load to device
        return self.metric(pred, target).item()


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()


def normalize_audio(extracted_audio, target_loudness=20):
    return (
        target_loudness * extracted_audio / extracted_audio.norm(dim=-1, keepdim=True)
    )
