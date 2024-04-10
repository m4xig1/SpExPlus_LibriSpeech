import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

from metrics.metrics import SiSdr


def sisdr(pred, target, eps=1e-6):
    """
    Arguments:
    x: separated signal, N x S tensor
    s: reference signal, N x S tensor
    Return:
    sisdr: N tensor
    """

    if pred.shape != target.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                pred.shape, pred.shape))
        
    alpha = (target * pred).sum() / (torch.linalg.norm(target) ** 2 + eps)
    return 20 * torch.log10(torch.linalg.norm(alpha * target) / (torch.linalg.norm(alpha * target - pred) + eps) + eps)
    


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
        self.ce = torch.nn.CrossEntropyLoss()

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

        short_scale = 1 - self.mid_scale - self.long_scale
        
        sisdr_short = sisdr(x_short, target)
        sisdr_mid = sisdr(x_mid, target)
        sisdr_long = sisdr(x_long, target)
        loss = (
            -short_scale * sisdr_short.sum()
            - self.mid_scale * sisdr_mid.sum()
            - self.long_scale * sisdr_long.sum()
        ) / x_short.shape[0]
        

        # norm?
        if is_train:
            ce_loss = self.ce(pred["logits"], speaker_id)
            loss += self.cross_ent_scale * ce_loss

        # return {"loss": loss, "ce": ce_loss}
        return loss
