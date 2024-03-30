import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

from metrics.metrics import SiSdr


def sisdr(x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisdr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return torch.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)
        t = torch.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


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
        self.metric = SiSdr() # this gives us avg result on batch

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
        target = self.__normalize(target.squeeze(1))  # if 1-d
        x_short = self.__normalize(pred["short"].squeeze(1))
        x_mid = self.__normalize(pred["mid"].squeeze(1))
        x_long = self.__normalize(pred["long"].squeeze(1))

        phi = 1 - self.mid_scale - self.long_scale
        

        sisdr_short = sisdr(x_short, target)
        sisdr_mid = sisdr(x_mid, target)
        sisdr_long = sisdr(x_long, target)
        # sisdr_short = self.metric(x_short, target)
        # sisdr_mid = self.metric(x_mid, target)
        # sisdr_long = self.metric(x_long, target)
        loss = (
            -phi * sisdr_short
            - self.mid_scale * sisdr_mid
            - self.long_scale * sisdr_long
        ) / x_short.shape[0]

        # norm?
        ce_loss = torch.nn.functional.cross_entropy(pred["logits"], speaker_id)
        # print(ce_loss)
        if is_train:
            loss += self.cross_ent_scale * ce_loss
        return {"loss": loss, "ce": ce_loss}
