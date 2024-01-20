import torch.nn as nn


class BaseMetric(nn.Module):
    def __init__(self):
        super(BaseMetric, self).__init__()

    def forward(self, pred, target):
        raise NotImplementedError()

    # def __call__(self):  # or make it a functor
    #     raise NotImplementedError()
