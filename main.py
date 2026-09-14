import torch
import numpy as np
import options
from datasets import RGDataset
from datasets2 import FisvDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import model, loss
from models.hgformer_aqa import HyperAQAFormer
#from models.enhanced_hgformer_aqa import EnhancedHyperAQAFormer

import os
from torch import nn

import train
from test import test_epoch, test_epoch_with_closest
import matplotlib.pyplot as plt
from thop import profile
from torchsummary import summary
import time
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optim(model, args):
    if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise Exception("Unknown optimizer")
    return optim


def get_scheduler(optim, args):
    if args.lr_decay is not None:
        if args.lr_decay == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=args.epoch - args.warmup, eta_min=args.lr * args.decay_rate)
        elif args.lr_decay == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[args.epoch - 30], gamma=args.decay_rate)
        else:
            raise Exception("Unknown Scheduler")
    else:
        scheduler = None
    return scheduler


if __name__ == '__main__':
    args = options.parser.parse_args()

    args.model_name = 'PCS'
    args.action_type = 'PCS'
    args.score_type = 'PCS'


    #args.model_name = 'Ball'
    #args.action_type = 'Hoop'

    # args.model_name = 'Hoop'
    # args.action_type = 'Hoop'
    # args.score_type = 'Difficulty_Score'
    # args.score_type = 'Execution_Score'1

    args.ckpt = 'PCS_best'

    args.batch = 32
    args.lr = 1e-2
    args.epoch = 400                         #  250/400/500/150    320/400
    # args.epoch = 500
    args.n_decoder = 2
    args.n_query = 4
    # args.alpha = 1.0 #RG
    args.alpha = 0.5 #fivs

    args.margin = 1.0
    args.lr_decay = 'cos'
    args.decay_rate = 0.01
    # args.dropout = 0.3 #RG
    args.dropout = 0.7 #fivs
    args.test = True

    setup_seed(42)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    '''
    1. load data
    '''
    '''
    train data
    '''
    #train_data = RGDataset(args.video_path, args.train_label_path, clip_num=args.clip_num,
    #                       action_type=args.action_type)
    train_data = FisvDataset(args.video_path, args.train_label_path, clip_num=args.clip_num,
                            action_type=args.action_type, score_type=args.score_type)

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=8)
    print(len(train_data))


    '''
    test data
    '''
    #test_data = RGDataset(args.video_path, args.test_label_path, clip_num=args.clip_num,
    #                      action_type=args.action_type, train=False)

    test_data = FisvDataset(args.video_path, args.test_label_path, clip_num=args.clip_num,
                           action_type=args.action_type, score_type=args.score_type, train=False)


    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)
    print('=============Load dataset successfully=============')

    '''
    2. load model
    '''
    #model = model.ASGTN(args.in_dim, args.hidden_dim, args.n_head, args.n_encoder,
    #                   args.n_decoder, args.n_query, args.dropout).to(device)
    """ 
    model = HyperAQAFormer(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        n_head=args.n_head,
        n_encoder=args.n_encoder,
        n_decoder=args.n_decoder,
        n_query=args.n_query,
        dropout=args.dropout,
        windows=[3,5],     # 多尺度滑窗
        use_topk=True,     # 启用Top-K特征超边
        topk=4,
        de_use_inv=True    # 使用 De^{-1}
    ).to(device)
    """

    model = HyperAQAFormer(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        n_head=args.n_head,
        n_encoder=args.n_encoder,
        n_decoder=args.n_decoder,
        n_query=args.n_query,
        dropout=args.dropout,
        
        # 原有参数 - 保持基础多尺度功能
        windows=[5, 7],           # 基础滑动窗口
        use_topk=True,            # 保持特征相似性连接
        topk=3,
        de_use_inv=True,
        
        # 新增层次化超图参数
        use_hierarchical=False,    # 启用层次化结构
        fine_windows=[2, 3, 4],   # 细粒度层：更密集的小窗口捕获局部细节
        coarse_windows=[6, 8, 12], # 粗粒度层：更大窗口捕获全局结构
        


        # 新增动作阶段感知参数
        use_phase_aware=False,     # 启用阶段感知
        num_phases=3,             # 动作分为3个阶段（准备-执行-结束）
        intra_phase_window=4,     # 阶段内连接窗口，稍大以捕获完整子动作
    ).to(device)
    
    loss_fn = loss.LossFun(args.alpha, args.margin)
    #loss_fn = loss.LossFun(args.alpha, args.margin,
    #                       lambda_consistency=args.lambda_consistency,
    #                       lambda_bias=args.lambda_bias,
    #                       lambda_gate_entropy=args.lambda_gate_entropy,
    #                       lambda_rank=args.lambda_rank,
    #                       lambda_delta=args.lambda_delta)
    train_fn = train.train_epoch
    if args.ckpt is not None:
        checkpoint = torch.load('./ckpt/' + args.ckpt + '.pkl')
        model.load_state_dict(checkpoint)
        print('=============Load model successfully=============')
        print(model)

    print(args)
    print(args.test)

    '''
    test mode



    '''
    if args.test:
        model.eval()
        test_loss, coef, out, mse_value = test_epoch(10, model, test_loader, None, device, args)
        print('Test Loss: {:.4f}\tTest Coef: {:.3f}\tTest Mse: {:.4f}'.format(test_loss, coef, mse_value))
        test_loss, coef, out, mse_value = test_epoch_with_closest(10, model, test_loader, None, device, args)
        raise SystemExit

    '''
    3. record
    '''
    if not os.path.exists("./ckpt/"):
        os.makedirs("./ckpt/")
    if not os.path.exists("./logs/" + args.model_name):
        os.makedirs("./logs/" + args.model_name)
    logger = SummaryWriter(os.path.join('./logs/', args.model_name))
    best_coef, best_epoch, best_mse_value, best_mse_epoch = -1, -1, 500, -1
    final_train_loss, final_train_coef, final_test_loss, final_test_coef = 0, 0, 0, 0

    '''
    4. train
    '''
    optim = get_optim(model, args)
    scheduler = get_scheduler(optim, args)
    print('=============Begin training=============')
    if args.warmup:
        warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda t: t / args.warmup)
    else:
        warmup = None

    for epc in range(args.epoch):

        if args.warmup and epc < args.warmup:
            warmup.step()

        avg_loss, train_coef, mse_value = train_fn(epc, model, loss_fn, train_loader, optim, logger, device, args)
        if scheduler is not None and (args.lr_decay != 'cos' or epc >= args.warmup):
            scheduler.step()

        test_loss, test_coef, out, test_mse_value = test_epoch(epc, model, test_loader, logger, device, args)
        if test_coef > best_coef:
            best_coef, best_epoch = test_coef, epc
            torch.save(model.state_dict(), './ckpt/' + args.model_name + '_best.pkl')
        if test_mse_value < best_mse_value:
            best_mse_value, best_mse_epoch = test_mse_value, epc
            torch.save(model.state_dict(), './ckpt/' + args.model_name + '_best_mse.pkl')

        print('Epoch: {}\tLoss: {:.4f}\t MSE: {:.4f}\t Train Coef: {:.3f}\tTest Loss: {:.4f}\tTest Coef: {:.3f}  Test MSE: {:.4f}'
              .format(epc, avg_loss, mse_value, train_coef, test_loss, test_coef, test_mse_value))

        if epc == args.epoch - 1:
            final_train_loss, final_train_coef, final_test_loss, final_test_coef, final_test_mse_value = \
                avg_loss, train_coef, test_loss, test_coef, test_mse_value
    torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
    print('Best Test Coef: {:.3f}\tBest Test Eopch: {}\tBest Test MSE: {:.4f}\tBest Test MSE Eopch: {}'
          .format(best_coef, best_epoch, best_mse_value, best_mse_epoch))
