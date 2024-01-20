if __name__ == "__main__":
    import torch
    from torch import Tensor
    from metrics import SiSdr, Pesq
    
    target = Tensor([3.0, -0.5, 2.0, 7.0])
    preds = Tensor([2.5, 0.0, 2.0, 8.0])
    metric = SiSdr()
    print(metric.forward(target, preds))

    target, preds = torch.randn(10000), torch.randn(10000)
    metric = Pesq(16000)
    print(metric.forward(preds, target))
