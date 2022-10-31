import torch.nn as nn
import torch


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, predict_y, gt_y):
        return torch.mean((predict_y - gt_y) ** 2)

    def __str__(self):
        return "MSE"


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, predict_y, gt_y):
        return torch.mean(torch.abs(predict_y - gt_y))

    def __str__(self):
        return "MAE"


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.huber_loss = nn.HuberLoss(reduction="mean", delta=1.0)

    def forward(self, predict_y, gt_y):
        return self.huber_loss(predict_y, gt_y)

    def __str__(self):
        return "HuberLoss({})".format(self.delta)

