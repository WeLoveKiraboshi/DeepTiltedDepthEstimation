import torch
import torch.nn as nn
import torch.nn.functional as F


class AbsoluteRelative(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, surface_normal_pred, sample_batched):
        # mae_loss = nn.MSELoss()
        num_img_in_batch = surface_normal_pred['I_g'].shape[0]
        gt_gravity = sample_batched['gravity'].reshape(num_img_in_batch, 3)
        # return mae_loss(surface_normal_pred['I_g'], gt_gravity)
        prediction_error_g = torch.cosine_similarity(surface_normal_pred['I_g'], gt_gravity, dim=1, eps=1e-6)
        return torch.mean(prediction_error_g)


