import torch
import numpy as np
import math

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):

    sigma_2 = sigma ** 2
    bbox_diff = torch.abs(bbox_pred - bbox_targets)
    bbox_diff = bbox_inside_weights * bbox_diff
    smoothL1_sign = (bbox_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(bbox_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (bbox_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

# combination smoothl1loss and kl-Loss
# if |bbox_pred - bbox_targets| > 1: Lreg = exp(-a)*(|bbox_pred - bbox_targets| - 1/2) + a/2
# else Lreg = (bbox_pred - bbox_targets)**2 / 2sigma**2 + log(sigma**2)/2 + log(2*pi)/2 - C # C is constant number
def calcLreg(bbox_diff, bbox_std_pred, gt_one):
    alpha = torch.log(bbox_std_pred)
    if gt_one:
        return torch.exp(-alpha) * (bbox_diff - 0.5) + alpha / 2
    else:
        return bbox_diff ** 2 / (2 * bbox_std_pred) + torch.log(bbox_std_pred ** 2) / 2 + np.log(2 * math.pi) / 2 - 1.0


def kl_bbox_regression_loss(bbox_pred, bbox_std_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, stage=1.0, dim=[1]):
    bbox_diff = torch.abs(bbox_pred - bbox_targets)
    bbox_diff = bbox_inside_weights * bbox_diff
    cond = bbox_diff < stage
    loss_bbox = torch.where(cond, calcLreg(bbox_diff, bbox_std_pred, True), calcLreg(bbox_diff, bbox_std_pred, False))
    loss = bbox_outside_weights * loss_bbox
    for i in sorted(dim, reverse=True):
        loss = loss.sum(i)
    loss = loss.mean()
    return loss