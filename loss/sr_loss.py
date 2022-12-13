import torch
import torch.nn as nn
import torch.nn.functional as F




class SRonly_Loss(nn.Module):
    def __init__(self, mode='optimize'):
        nn.Module.__init__(self)
        self.mode = mode

    def forward(self, surface_normal_pred, sample_batched):
        prediction_error_g = torch.cosine_similarity(surface_normal_pred['I_g'], sample_batched['gravity'], dim=1, eps=1e-6) #surface_normal_pred['I_g'], sample_batched['gravity']
        # prediction_error_a = torch.cosine_similarity(surface_normal_pred['I_a'], sample_batched['aligned_directions'],
        #                                              dim=1, eps=1e-6)
        acos_mask_g = (prediction_error_g.detach() < 0.9999).float() * (prediction_error_g.detach() > 0.0).float()
        cos_mask_g = (prediction_error_g.detach() <= 0.0).float()
        acos_mask_g = acos_mask_g > 0.0
        cos_mask_g = cos_mask_g > 0.0

        optimize_loss = torch.sum(torch.acos(prediction_error_g[acos_mask_g])) - torch.sum(prediction_error_g[cos_mask_g])
        logging_loss = 1.0 * (1.0 - torch.mean(prediction_error_g))
        if self.mode == 'optimize':
            return optimize_loss
        else:
            return logging_loss




class SRonly_Loss_(nn.Module):
    def __init__(self, mode='optimize'):
        nn.Module.__init__(self)
        self.mode = mode

    def forward(self, surface_normal_pred_g, sample_batched_g):
        prediction_error_g = torch.cosine_similarity(surface_normal_pred_g, sample_batched_g, dim=1, eps=1e-6) #surface_normal_pred['I_g'], sample_batched['gravity']
        # prediction_error_a = torch.cosine_similarity(surface_normal_pred['I_a'], sample_batched['aligned_directions'],
        #                                              dim=1, eps=1e-6)
        acos_mask_g = (prediction_error_g.detach() < 0.9999).float() * (prediction_error_g.detach() > 0.0).float()
        cos_mask_g = (prediction_error_g.detach() <= 0.0).float()
        acos_mask_g = acos_mask_g > 0.0
        cos_mask_g = cos_mask_g > 0.0

        optimize_loss = torch.sum(torch.acos(prediction_error_g[acos_mask_g])) - torch.sum(prediction_error_g[cos_mask_g])
        logging_loss = 1.0 * (1.0 - torch.mean(prediction_error_g))
        if self.mode == 'optimize':
            return optimize_loss
        else:
            return logging_loss
