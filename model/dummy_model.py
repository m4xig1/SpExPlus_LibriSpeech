import torch
import torch.nn as nn
import torch.nn.functional as F


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, ref, ref_len):
        return {"logits": torch.rand_like(torch.tensor([1,100])), 'short': x[:256], 'mid':x[:256], 'long':x[:256]}
