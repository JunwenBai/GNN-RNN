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
from model import CNN_RNN, RNN
import random
from utils import get_X_Y, build_path
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE

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

def train_epoch(args, model, device, train_loader, optimizer, epoch, writer=None):
    print("\n---------------------")
    print("Epoch ", epoch)
    print("---------------------")
    model.train()
    lr = optimizer.param_groups[0]['lr']
    print("lr =", lr)
    tot_loss, tot_rmse, tot_r2, tot_corr, tot_mae, tot_mape = 0., 0., 0., 0., 0., 0.
    tot_batch = len(train_loader)
    for batch_idx, (X, Y) in enumerate(train_loader): # 397
        X, Y = X.to(device), Y.to(device) # [64, 5, 431] [64, 5]
        optimizer.zero_grad()
        pred = model(X, Y)
        loss = loss_fn(pred[:, :args.length-1], Y[:, :args.length-1]) * args.c1 + \
               loss_fn(pred[:, -1], Y[:, -1]) * args.c2

        metrics = eval(pred[:, -1], Y[:, -1]) # [64, 64]
        tot_loss += loss.item()
        tot_rmse += metrics['rmse']
        tot_r2 += metrics['r2']
        tot_corr += metrics['corr']
        tot_mae += metrics['mae']
        tot_mape += metrics['mape']

        loss.backward()
        optimizer.step()

        if batch_idx % args.check_freq == 0:
            n_batch = batch_idx+1
            print("### batch ", batch_idx)
            print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
                tot_loss/n_batch, tot_rmse/n_batch, tot_r2/n_batch, tot_corr/n_batch, tot_mae/n_batch, tot_mape/n_batch)
            )
        
        cur_step = tot_batch * epoch + batch_idx
        n_batch = batch_idx + 1
        if writer is not None:
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("lr", lr, cur_step)
            writer.add_scalar("Train/loss", tot_loss/n_batch, cur_step)
            writer.add_scalar("Train/rmse", tot_rmse/n_batch, cur_step)
            writer.add_scalar("Train/r2", tot_r2/n_batch, cur_step)
            writer.add_scalar("Train/corr", tot_corr/n_batch, cur_step)
            writer.add_scalar("Train/mae", tot_mae/n_batch, cur_step)
            writer.add_scalar("Train/mape", tot_mape/n_batch, cur_step)

    n_batch = batch_idx+1
    print("\n###### Overall training metrics")
    print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
        tot_loss/n_batch, tot_rmse/n_batch, tot_r2/n_batch, tot_corr/n_batch, tot_mae/n_batch, tot_mape/n_batch)
    )

def val_epoch(args, model, device, test_loader, epoch, mode="Val", writer=None):
    print("********************")
    print("Epoch", epoch, mode)
    print("********************")
    model.eval()
    tot_loss, tot_rmse, tot_r2, tot_corr, tot_mae, tot_mape = 0., 0., 0., 0., 0., 0.
    for batch_idx, (X, Y) in enumerate(test_loader):
        X, Y = X.to(device), Y.to(device)
        pred = model(X, Y)
        loss = loss_fn(pred[:, :args.length-1], Y[:, :args.length-1]) * args.c1 + \
               loss_fn(pred[:, -1], Y[:, -1]) * args.c2

        metrics = eval(pred[:, -1], Y[:, -1])
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
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_X_Y(data, args)
    X_train, X_val, X_test = torch.Tensor(X_train), torch.Tensor(X_val), torch.Tensor(X_test)
    Y_train, Y_val, Y_test = torch.Tensor(Y_train), torch.Tensor(Y_val), torch.Tensor(Y_test)
    print("train:", X_train.shape, Y_train.shape)
    print("val:", X_val.shape, Y_val.shape)
    print("test:", X_test.shape, Y_test.shape)

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    n_train = len(X_train)
    param_setting = "{}_bs-{}_lr-{}_maxepoch-{}_sche-{}_T0-{}_testyear-{}_len-{}_seed-{}".format(
        args.model, args.batch_size, args.learning_rate, args.max_epoch, args.scheduler, args.T0, args.test_year, args.length, args.seed)

    summary_dir = 'summary/{}/{}'.format(args.dataset, param_setting)
    model_dir = 'model/{}/{}'.format(args.dataset, param_setting)
    build_path(summary_dir)
    build_path(model_dir)
    writer = SummaryWriter(log_dir=summary_dir)

    print('building network...')

    #building the model 
    if args.model == "cnn_rnn":
        model = CNN_RNN(args).to(device)
    elif args.model == "rnn":
        model = RNN(args).to(device)
    else:
        raise ValueError("model type not supported yet")
    
    #log the learning rate 
    #writer.add_scalar('learning_rate', args.learning_rate)

    #use the Adam optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.max_epoch//4, 0.5)
    else:
        raise ValueError("scheduler not supported yet")

    '''if args.resume:
        vae.load_state_dict(torch.load(args.checkpoint_path))
        current_step = int(args.checkpoint_path.split('/')[-1].split('-')[-1]) 
        print("loaded model: %s" % args.label_checkpoint_path)
    else:
        current_step = 0'''

    best_val_rmse = 1e9
    for epoch in range(args.max_epoch):
        train_epoch(args, model, device, train_loader, optimizer, epoch, writer)
        val_rmse = val_epoch(args, model, device, val_loader, epoch, "Val", writer)
        val_epoch(args, model, device, test_loader, epoch, "Test", writer)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), model_dir+'/model-'+str(epoch))
            print('save model to', model_dir)
        scheduler.step()
    
