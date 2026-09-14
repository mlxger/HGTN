import numpy as np
from scipy.stats import spearmanr
import pdb
import torch
import torch.nn.functional as F
from utils import AverageMeter


def train_epoch(epoch, model, loss_fn, train_loader, optim, logger, device, args):
    model.train()
    preds = np.array([])
    labels = np.array([])

    losses = AverageMeter('loss', logger)
    mse_losses = AverageMeter('mse', logger)
    tri_losses = AverageMeter('tri', logger)

    for i, (video_feat, label) in enumerate(train_loader):
        video_feat = video_feat.to(device)      # (b, t, c)
        label = label.float().to(device)
        out = model(video_feat)
        pred = out['output']
        # print('label:',label)
        # print('pred:',pred)
        loss, mse, tri = loss_fn(pred, label, out['embed'])

        optim.zero_grad()
        loss.backward()
        # pdb.set_trace()
        optim.step()

        losses.update(loss, label.shape[0])
        mse_losses.update(mse, label.shape[0])
        tri_losses.update(tri, label.shape[0])

        if len(preds) == 0:
            preds = pred.cpu().detach().numpy()
            labels = label.cpu().detach().numpy()
        else:
            preds = np.concatenate((preds, pred.cpu().detach().numpy()), axis=0)
            labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)
    # print('train labels:',labels)
    # print('train preds:',preds)

    coef, _ = spearmanr(preds, labels)
        # 反归一化：将[0,1]范围恢复到[0,25]范围
    ORIGINAL_MAX_SCORE = 25.0
    preds_denorm = preds * ORIGINAL_MAX_SCORE
    labels_denorm = labels * ORIGINAL_MAX_SCORE

       # 直接用公式计算整体 MSE
    mse_value = np.mean((preds_denorm - labels_denorm) ** 2)
    if logger is not None:
        logger.add_scalar('train coef', coef, epoch)
    avg_loss = losses.done(epoch)
    mse_losses.done(epoch)
    tri_losses.done(epoch)
    return avg_loss, coef, mse_value  
