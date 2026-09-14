import torch
from torch import nn
from models.triplet_loss import HardTripletLoss
import torch.nn.functional as F
import numpy as np
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        batch_size, num_features, feat_dim = features.shape
        features = features.view(batch_size * num_features, feat_dim)
        labels = torch.arange(batch_size * num_features).cuda()
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        similarity_matrix = similarity_matrix / self.temperature
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss



# LOSS
class LossFun(nn.Module):
    def __init__(self, alpha, margin):
        super(LossFun, self).__init__()
        self.mse_loss = nn.MSELoss()
        # self.mse_loss = nn.L1Loss()
        self.triplet_loss = HardTripletLoss(margin=margin, hardest=True)
        self.ce_loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda()
        self.alpha = alpha
        self.alpha_ce = 0.01

    def forward(self, pred, label, feat):
        # feat (b, n, c), x (b, t, c)
        if feat is not None:
            device = feat.device
            b, n, c = feat.shape
            flat_feat = feat.view(-1, c)  # (bn, c)
            la = torch.arange(n, device=device).repeat(b)

            t_loss = self.triplet_loss(flat_feat, la)
            # t_loss = pair_diversity_loss(feat)
        else:
            self.alpha = 0
            t_loss = 0

        mse_loss = self.mse_loss(pred, label)
        # mse_loss = pearson_loss(pred, label)
        return mse_loss + self.alpha * t_loss, mse_loss, t_loss
        # return f_loss, mse_loss, t_loss, f_loss
