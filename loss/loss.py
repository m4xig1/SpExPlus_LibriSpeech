from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from metrics.metrics import SiSdr


class SpexPlusLoss(nn.Module):
    # gamma, beta, alpha according to the paper
    def __init__(self, mid_scale=0.1, long_scale=0.1, cross_ent_scale=0.5):
        """
        alpha, beta, gamma according to the paper
        """
        super().__init__()
        self.cross_ent_scale = cross_ent_scale
        self.mid_scale = mid_scale
        self.long_scale = long_scale

    @staticmethod
    def __vec_l2norm(x):
        return np.linalg.norm(x, 2)

    @staticmethod
    def __normalize(x: Tensor):
        return NotImplementedError()

    def forward(
        self,
        pred: "dict[str, Tensor]",  # predicted vals
        target: Tensor,
        speaker_id: Tensor,
        is_train=True,
    ):
        # we still need to get speaker id from name of the audio... 
        target -= target.mean(dim=-1, keepdim=True)
        x_short = pred["short"] - pred["short"].mean(dim=-1, keepdim=True)
        x_mid = pred["mid"] - pred["mid"].mean(dim=-1, keepdim=True)
        x_long = pred["long"] - pred["long"].mean(dim=-1, keepdim=True)
        metric = SiSdr()
        phi = 1 - self.mid_scale - self.long_scale
        sisdr_short = metric.forward(x_short, target)
        sisdr_mid = metric.forward(x_mid, target)
        sisdr_long = metric.forward(x_long, target)
        loss = (
            -phi * sisdr_short.sum()
            - self.mid_scale * sisdr_mid.sum()
            - self.long_scale * sisdr_long.sum()
        ) / x_short.shape[0]
        # norm?
        ce_loss = 0
        if is_train:
            ce_loss = F.cross_entropy(pred["logits"], speaker_id)
        return loss + self.ce * ce_loss
