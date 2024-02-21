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
        self.metric = SiSdr()

    @staticmethod
    def __vec_l2norm(x):
        return np.linalg.norm(x, 2)

    @staticmethod
    def __normalize(x: Tensor):
        return x - x.mean(dim=-1, keepdim=True)

    def forward(
        self,
        pred: "dict[str, Tensor]",  # predicted vals
        target: Tensor,
        speaker_id: Tensor,
        is_train=True,
    ):
        # we still need to get speaker id from name of the audio...
        target = self.__normalize(target.squeeze(1))  # 1-d
        x_short = self.__normalize(pred["short"].squeeze(1))
        x_mid = self.__normalize(pred["mid"].squeeze(1))
        x_long = self.__normalize(pred["long"].squeeze(1))
        # metric = SiSdr()
        phi = 1 - self.mid_scale - self.long_scale
        sisdr_short = self.metric.forward(x_short, target)
        sisdr_mid = self.metric.forward(x_mid, target)
        sisdr_long = self.metric.forward(x_long, target)
        loss = (
            -phi * sisdr_short.sum()
            - self.mid_scale * sisdr_mid.sum()
            - self.long_scale * sisdr_long.sum()
        ) / x_short.shape[0]
        # norm?
        ce_loss = 0
        if is_train:
            ce_loss = F.cross_entropy(pred["logits"], speaker_id)
        return {"loss": loss + self.cross_ent_scale * ce_loss} # наверное, хочется возвращать какой-нибудь accuracy для предсказаний
