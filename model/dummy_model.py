import torch
import torch.nn as nn
import torch.nn.functional as F


class Dummy(nn.Module):
    def __init__(self, n_feats=256, n_classes=251, dim_hid=1024):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_feats, dim_hid), nn.ReLU(), nn.Linear(dim_hid, n_classes)
        )

    def forward(self, x, ref, ref_len):
        # return {"logits": self.model(x[:256]), }
        return {
            "logits": self.model(x[0][:256]),
            "short": x[:][:256],
            "mid": x[:][:256],
            "long": x[:][:256],
        }
        # return {
        #     "logits": torch.rand(1, 250),
        #     "short": ref[:256],
        #     "mid": ref[:256],
        #     "long": ref[:256],
        # }
