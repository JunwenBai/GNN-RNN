import math
from time import time
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os
import datetime
from copy import copy, deepcopy
from model import SAGE
import random
from utils import get_X_Y, build_path
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE

import dgl
import scipy.sparse as sp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append('./')

METRICS = {'rmse', 'r2', 'corr'}
        
huber_fn = nn.SmoothL1Loss()
best_test = {'rmse': 1e9, 'r2': -1e9, 'corr':-1e9}
best_val = {'rmse': 1e9, 'r2': -1e9, 'corr':-1e9}

def eval(pred, Y):
    pred, Y = pred.detach().cpu().numpy(), Y.detach().cpu().numpy()
    metrics = {}
    # RMSE
    metrics['rmse'] = np.sqrt(np.mean((pred-Y)**2))
    # R2
    metrics['r2'] = r2_score(Y, pred)
    # corr
    metrics['corr'] = np.corrcoef(Y, pred)[0, 1]
    # MAE
    metrics['mae'] = MAE(Y, pred)
    # MSE
    metrics['mse'] = np.mean((pred-Y)**2)
    # MAPE
    metrics['mape'] = np.mean(np.abs((Y - pred) / Y))

    return metrics

def loss_fn(pred, Y, mode="logcosh"):
    if mode == "huber":
        # huber loss
        loss = huber_fn(pred, Y)
    elif mode == "logcosh":
        # log cosh loss
        err = Y - pred
        loss = torch.mean(torch.log(torch.cosh(err + 1e-12)))
    return loss

def update_metrics(rmse, r2, corr, mode):
    if mode == "Val":
        if rmse < best_val['rmse']:
            best_val['rmse'] = rmse
            best_val['r2'] = r2
            best_val['corr'] = corr
    elif mode == "Test":
        if rmse < best_test['rmse']:
            best_test['rmse'] = rmse
            best_test['r2'] = r2
            best_test['corr'] = corr

def load_subtensor(year_XY, year, in_nodes, out_nodes, device):
    X, Y = year_XY[year]
    batch_inputs = X[in_nodes].float().to(device)
    batch_labels = Y[out_nodes].float().to(device)
    return batch_inputs, batch_labels

def train_epoch(args, model, device, nodeloader, year_XY, cur_step, optimizer, epoch, writer=None):
    print("\n---------------------")
    print("Epoch ", epoch)
    print("---------------------")
    model.train()
    lr = optimizer.param_groups[0]['lr']
    print("lr =", lr)
    tot_loss, tot_rmse, tot_r2, tot_corr, tot_mae, tot_mape = 0., 0., 0., 0., 0., 0.

    n_batch = 0
    for year in year_XY.keys():
        if year == args.test_year or year == args.test_year-1: 
            continue
        for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(nodeloader):
            batch_inputs, batch_labels = load_subtensor(year_XY, year, in_nodes, out_nodes, device)
            #print(batch_inputs.shape, batch_labels.shape) # [675, 431] [64]
            blocks = [block.int().to(device) for block in blocks]
            batch_pred = model(blocks, batch_inputs).squeeze(-1)
            loss = loss_fn(batch_pred, batch_labels)
            optimizer.zero_grad()

            metrics = eval(batch_pred, batch_labels)
            tot_loss += loss.item()
            tot_rmse += metrics['rmse']
            tot_r2 += metrics['r2']
            tot_corr += metrics['corr']
            tot_mae += metrics['mae']
            tot_mape += metrics['mape']

            loss.backward()
            optimizer.step()
            n_batch += 1

            if n_batch % args.check_freq == 0:
                print("### batch ", n_batch-1)
                print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
                    tot_loss/n_batch, tot_rmse/n_batch, tot_r2/n_batch, tot_corr/n_batch, tot_mae/n_batch, tot_mape/n_batch)
                )
        
            if writer is not None:
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("lr", lr, cur_step)
                writer.add_scalar("Train/loss", tot_loss/n_batch, cur_step)
                writer.add_scalar("Train/rmse", tot_rmse/n_batch, cur_step)
                writer.add_scalar("Train/r2", tot_r2/n_batch, cur_step)
                writer.add_scalar("Train/corr", tot_corr/n_batch, cur_step)
                writer.add_scalar("Train/mae", tot_mae/n_batch, cur_step)
                writer.add_scalar("Train/mape", tot_mape/n_batch, cur_step)
            cur_step += 1

    print("\n###### Overall training metrics")
    print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
        tot_loss/n_batch, tot_rmse/n_batch, tot_r2/n_batch, tot_corr/n_batch, tot_mae/n_batch, tot_mape/n_batch)
    )
    return cur_step

def val_epoch(args, model, device, nodeloader, year_XY, epoch, mode="Val", writer=None):
    print("********************")
    print("Epoch", epoch, mode)
    print("********************")
    model.eval()
    tot_loss, tot_rmse, tot_r2, tot_corr, tot_mae, tot_mape = 0., 0., 0., 0., 0., 0.
    if mode == "Val":
        year = args.test_year-1
    elif mode == "Test":
        year = args.test_year

    for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(nodeloader):
        batch_inputs, batch_labels = load_subtensor(year_XY, year, in_nodes, out_nodes, device)
        blocks = [block.int().to(device) for block in blocks]
        batch_pred = model(blocks, batch_inputs).squeeze(-1)
        loss = loss_fn(batch_pred, batch_labels)

        metrics = eval(batch_pred, batch_labels)
        tot_loss += loss.item()
        tot_rmse += metrics['rmse']
        tot_r2 += metrics['r2']
        tot_corr += metrics['corr']
        tot_mae += metrics['mae']
        tot_mape += metrics['mape']

    n_batch = batch_idx+1
    #print("###### Overall Validation metrics")
    print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
        tot_loss/n_batch, tot_rmse/n_batch, tot_r2/n_batch, tot_corr/n_batch, tot_mae/n_batch, tot_mape/n_batch)
    )
    print("********************")
    update_metrics(tot_rmse/n_batch, tot_r2/n_batch, tot_corr/n_batch, mode)

    if writer is not None:
        writer.add_scalar("{}/loss".format(mode), tot_loss/n_batch, epoch)
        writer.add_scalar("{}/rmse".format(mode), tot_rmse/n_batch, epoch)
        writer.add_scalar("{}/r2".format(mode), tot_r2/n_batch, epoch)
        writer.add_scalar("{}/corr".format(mode), tot_corr/n_batch, epoch)
        writer.add_scalar("{}/mae".format(mode), tot_mae/n_batch, epoch)
        writer.add_scalar("{}/mape".format(mode), tot_mape/n_batch, epoch)

    if mode == "Test":
        print("Val | rmse: {}, r2: {}, corr: {}".format(best_val['rmse'], best_val['r2'], best_val['corr']))
        print("Test | rmse: {}, r2: {}, corr: {}".format(best_test['rmse'], best_test['r2'], best_test['corr']))

    return tot_rmse/n_batch

def train(args):
    print('reading npy...')
    np.random.seed(args.seed) # set the random seed of numpy
    torch.manual_seed(args.seed)
    raw_data = np.load(args.data_dir) #load data from the data_dir
    data = raw_data['data']
    
    X_dict, Y_dict, avail_dict, adj, order_map, min_year, max_year, county_set = get_X_Y(data, args)
    sp_adj = sp.coo_matrix(adj)
    g = dgl.from_scipy(sp_adj)
    
    #print(max(np.sum(adj, axis=1))) # 10

    N = len(adj)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 10])
    nodeloader = dgl.dataloading.NodeDataLoader(
        g,
        range(N),
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers = 0,
    )

    year_XY = {}
    for year in range(min_year, max_year+1):
        X = []
        Y = []
        for county in county_set:
            X.append(X_dict[county][year])
            Y.append(Y_dict[county][year])
        X, Y = torch.tensor(X), torch.tensor(Y)
        year_XY[year] = (X, Y)

    param_setting = "bs-{}_lr-{}_maxepoch-{}_sche-{}_T0-{}_testyear-{}_len-{}_seed-{}".format(
        args.batch_size, args.learning_rate, args.max_epoch, args.scheduler, args.T0, args.test_year, args.length, args.seed)

    summary_dir = 'summary/{}/{}'.format(args.dataset, param_setting)
    model_dir = 'model/{}/{}'.format(args.dataset, param_setting)
    build_path(summary_dir)
    build_path(model_dir)
    writer = SummaryWriter(log_dir=summary_dir)

    print('building network...')

    #building the model
    in_dim = X.shape[1]
    out_dim = 1
    model = SAGE(args, in_dim, out_dim).to(device)
    
    #log the learning rate 
    #writer.add_scalar('learning_rate', args.learning_rate)

    #use the Adam optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=args.eta_min, T_0=args.T0, T_mult=args.T_mult)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.max_epoch//4, 0.5)
    elif args.scheduler == "const":
        scheduler = None
    else:
        raise ValueError("scheduler not supported yet")

    '''if args.resume:
        vae.load_state_dict(torch.load(args.checkpoint_path))
        current_step = int(args.checkpoint_path.split('/')[-1].split('-')[-1]) 
        print("loaded model: %s" % args.label_checkpoint_path)
    else:
        current_step = 0'''

    best_val_rmse = 1e9
    cur_step = 0
    for epoch in range(args.max_epoch):
        cur_step = train_epoch(args, model, device, nodeloader, year_XY, cur_step, optimizer, epoch, writer)
        val_rmse = val_epoch(args, model, device, nodeloader, year_XY, epoch, "Val", writer)
        val_epoch(args, model, device, nodeloader, year_XY, epoch, "Test", writer)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), model_dir+'/model-'+str(epoch))
            print('save model to', model_dir)
        if scheduler is not None:
            scheduler.step()
    
