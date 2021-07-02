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




def test_epoch(args, model, device, test_loader, mode="Val"):
    print("********************")
    print(mode)
    print("********************")
    model.eval()
    tot_loss, tot_rmse, tot_r2, tot_corr = 0., 0., 0., 0.
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

    n_batch = batch_idx+1
    #print("###### Overall Validation metrics")
    print("loss: {}\nrmse: {}\t r2: {}\t corr: {}".format(
        tot_loss/n_batch, tot_rmse/n_batch, tot_r2/n_batch, tot_corr/n_batch)
    )
    print("********************")
    update_metrics(tot_rmse/n_batch, tot_r2/n_batch, tot_corr/n_batch, mode)

    #print("Test | rmse: {}, r2: {}, corr: {}".format(best_test['rmse'], best_test['r2'], best_test['corr']))

    return tot_rmse/n_batch

def test(args):
    print('reading npy...')
    np.random.seed(args.seed) # set the random seed of numpy
    torch.manual_seed(args.seed)
    raw_data = np.load(args.data_dir) #load data from the data_dir
    data = raw_data['data']
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_X_Y(data, args)
    X_train, X_val, X_test = torch.Tensor(X_train), torch.Tensor(X_val), torch.Tensor(X_test)
    Y_train, Y_val, Y_test = torch.Tensor(Y_train), torch.Tensor(Y_val), torch.Tensor(Y_test)
    print("test:", X_test.shape, Y_test.shape)

    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    n_train = len(X_train)
    param_setting = "bs-{}_lr-{}_maxepoch-{}_testyear-{}_len-{}_seed-{}".format(
        args.batch_size, args.learning_rate, args.max_epoch, args.test_year, args.length, args.seed)

    #building the model 
    if args.model == "cnn_rnn":
        model = CNN_RNN(args).to(device)
    elif args.model == "rnn":
        model = RNN(args).to(device)
    else:
        raise ValueError("model type not supported yet")
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    
    test_epoch(args, model, device, test_loader, "Test")

