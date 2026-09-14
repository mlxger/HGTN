import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from torch import nn
import matplotlib.pyplot as plt

def test_epoch(epoch, model, test_loader, logger, device, args):
    mse_loss = nn.MSELoss().to(device)
    model.eval()

    preds = np.array([])
    labels = np.array([])
    tol_loss, tol_sample = 0, 0

    feats = []

    with torch.no_grad():
        for i, (video_feat, label) in enumerate(test_loader):
            video_feat = video_feat.to(device)
            label = label.float().to(device)
            out = model(video_feat)
            pred = out['output']

            if 'encode' in out.keys() and out['encode'] is not None:
                feats.append(out['encode'].mean(dim=1).cpu().detach().numpy())
                # feats.append(out['embed'].cpu().detach().numpy())

            loss = mse_loss(pred, label)
            tol_loss += (loss.item() * label.shape[0])
            tol_sample += label.shape[0]

            if len(preds) == 0:
                preds = pred.cpu().detach().numpy()
                labels = label.cpu().detach().numpy()
            else:
                preds = np.concatenate((preds, pred.cpu().detach().numpy()), axis=0)
                labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)
    # print(preds)
    # print('test labels:', labels)
    # print('test preds:', preds)
    avg_coef, _ = spearmanr(preds, labels)
    plcc, _ = pearsonr(preds, labels)
    ORIGINAL_MAX_SCORE = 40.0
    preds_denorm = preds * ORIGINAL_MAX_SCORE
    labels_denorm = labels * ORIGINAL_MAX_SCORE
    mse_value = np.mean((preds_denorm - labels_denorm) ** 2)
    # print('srocc', avg_coef)
    # print('plcc',plcc)
    # print(tol_sample)
    avg_loss = float(tol_loss) / float(tol_sample)
    if logger is not None:
        logger.add_scalar('Test coef', avg_coef, epoch)
        logger.add_scalar('Test loss', avg_loss, epoch)
    # print(preds.tolist())
    # print(labels.tolist())
    return avg_loss, avg_coef, out, mse_value




# 新函数：添加打印最接近的分数组
def test_epoch_with_closest(epoch, model, test_loader, logger, device, args):
    mse_loss = nn.MSELoss().to(device)
    model.eval()

    preds = np.array([])
    labels = np.array([])
    tol_loss, tol_sample = 0, 0

    feats = []

    with torch.no_grad():
        for i, (video_feat, label) in enumerate(test_loader):
            video_feat = video_feat.to(device)
            label = label.float().to(device)
            out = model(video_feat)
            pred = out['output']

            if 'encode' in out.keys() and out['encode'] is not None:
                feats.append(out['encode'].mean(dim=1).cpu().detach().numpy())
                # feats.append(out['embed'].cpu().detach().numpy())

            loss = mse_loss(pred, label)
            tol_loss += (loss.item() * label.shape[0])
            tol_sample += label.shape[0]

            if len(preds) == 0:
                preds = pred.cpu().detach().numpy()
                labels = label.cpu().detach().numpy()
            else:
                preds = np.concatenate((preds, pred.cpu().detach().numpy()), axis=0)
                labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)
    print('test labels:', labels)
    print('test preds:', preds)
    avg_coef, _ = spearmanr(preds, labels)
    plcc, _ = pearsonr(preds, labels)
    ORIGINAL_MAX_SCORE = 40.0
    preds_denorm = preds * ORIGINAL_MAX_SCORE
    labels_denorm = labels * ORIGINAL_MAX_SCORE
    print('test labels:', preds_denorm)
    print('test preds:', labels_denorm)
    mse_value = np.mean((preds_denorm - labels_denorm) ** 2)
    avg_loss = float(tol_loss) / float(tol_sample)
    if logger is not None:
        logger.add_scalar('Test coef', avg_coef, epoch)
        logger.add_scalar('Test loss', avg_loss, epoch)

    # 新增：找出最接近的分数组
    differences = np.abs(preds_denorm - labels_denorm)  # 计算绝对差异
    closest_idx = np.argmin(differences)  # 差异最小的索引
    closest_label = labels_denorm[closest_idx]
    closest_pred = preds_denorm[closest_idx]
    closest_diff = differences[closest_idx]
    
    # 打印最接近的组（视频索引从0开始）
    print(f"Closest match: Video index {closest_idx}, Label: {closest_label:.4f}, Pred: {closest_pred:.4f}, Difference: {closest_diff:.4f}")

    return avg_loss, avg_coef, out, mse_value