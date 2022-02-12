import torch
import torch.nn as nn
import numpy as np

from utils.utils import draw_path_from_device_path, target_denormalize_mean_std_cuda

class MeanIoU(nn.Module):
    __name__ = 'mean_iou'

    def __init__(self):
        super(MeanIoU, self).__init__()

    def forward(self, y_pr_norm, y_gt_norm):
        y_pr = target_denormalize_mean_std_cuda(y_pr_norm)
        y_gt = target_denormalize_mean_std_cuda(y_gt_norm)

        eps = 1e-7
        batch_size = y_gt.shape[0]
        
        empty = np.zeros((320, 416))

        y_pr_masks = []
        y_gt_masks = []
        for i in range(batch_size):
            y_pr_masks.append(draw_path_from_device_path(y_pr[i].cpu().detach().numpy(), empty))
            y_gt_masks.append(draw_path_from_device_path(y_gt[i].cpu().detach().numpy(), empty))

        y_pr_mask_tensor = torch.tensor(y_pr_masks, dtype=torch.int32)
        y_gt_mask_tensor = torch.tensor(y_gt_masks, dtype=torch.int32)

        intersection = torch.sum(y_pr_mask_tensor * y_gt_mask_tensor)
        union = torch.sum(y_gt_mask_tensor) + torch.sum(y_pr_mask_tensor) - intersection + eps
        
        return (intersection + eps) / union
        