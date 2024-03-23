from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from metrics.metrics import SiSdr


def sisdr(x, s, remove_dc=True):
    """
    Compute SI-SDR
    x: extracted signal
    s: reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm) ** 2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s) ** 2
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))


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
        sisdr_short = self.metric(x_short, target)  # const?
        sisdr_mid = self.metric(x_mid, target)
        sisdr_long = self.metric(x_long, target)
        loss = (
            -phi * sisdr_short
            - self.mid_scale * sisdr_mid
            - self.long_scale * sisdr_long
        ) / x_short.shape[0]
        # norm?

        ce_loss = F.cross_entropy(pred["logits"], speaker_id)
        if is_train:
            loss += self.cross_ent_scale * ce_loss
        return {"loss": loss, "ce": ce_loss}
