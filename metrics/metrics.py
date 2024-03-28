from .base import BaseMetric
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


class SiSdr(BaseMetric):
    def __init__(self, **kwargs):
        super(SiSdr, self).__init__()
        self.si_sdr = ScaleInvariantSignalDistortionRatio()

    def forward(self, pred, target):
        self.metric = self.si_sdr.to(pred.device)  # load to device
        return self.metric(pred, target).item()


class Pesq(BaseMetric):
    def __init__(self, sample_rate=16000, mode="wb", **kwargs):
        super(Pesq, self).__init__()
        self.pesq = PerceptualEvaluationSpeechQuality(sample_rate, mode)

    def forward(self, pred, target):
        self.metric = self.pesq.to(pred.device)  # load to device
        return self.metric(pred, target).item()
