from pdb import set_trace

import torch
import numpy as np
import torch.nn.functional as F

from constant import target_indices, logs


def rmse(output, target):
    with torch.no_grad():
        output *= 100
        target *= 100
        mse = F.mse_loss(output, target)
        rmse = torch.sqrt(mse).item()
    return rmse


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)