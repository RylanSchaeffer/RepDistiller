from __future__ import print_function

from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

from rep_distiller.distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, PretrainedRepresentationDistillation, \
    VIDLoss, RKDLoss
from rep_distiller.distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from rep_distiller.crd.criterion import CRDLoss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Statistics:
    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, batch_size=1, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v, batch_size)

    def averages(self):
        """
        Compute averages from meters. Handle tensors vs floats (always return a
        float)

        Parameters
        ----------
        meters : Dict[str, util.AverageMeter]
            Dict of average meters, whose averages may be of type ``float`` or ``torch.Tensor``

        Returns
        -------
        metrics : Dict[str, float]
            Average value of each metric
        """
        metrics = {m: vs.avg for m, vs in self.meters.items()}
        metrics = {
            m: v if isinstance(v, float) else v.item() for m, v in metrics.items()
        }
        return metrics

    def __str__(self):
        meter_str = ", ".join(f"{k}={v}" for k, v in self.meters.items())
        return f"Statistics({meter_str})"


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def compute_accuracy(output_tensors_by_model: Dict[str, torch.Tensor],
                     target: torch.Tensor,
                     topk: Tuple = (1,),
                     ) -> Dict[str, Dict[str, torch.Tensor]]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():

        topk_acc_by_model = dict()
        maxk = max(topk)
        batch_size = target.size(0)

        for model_name, model_outputs in output_tensors_by_model.items():

            topk_acc_by_model[model_name] = dict()
            _, pred = model_outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for k in topk:
                # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                topk_acc_by_model[model_name][f'top_{k}_acc'] = correct_k.mul_(100.0 / batch_size)

    return topk_acc_by_model


if __name__ == '__main__':
    pass
