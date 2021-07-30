import csv
import math
import pandas as pd
from time import time
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import sys
import os
import datetime
from copy import copy, deepcopy
from model import SAGE_RNN
import random
import matplotlib.pyplot as plt
from utils import get_X_Y, build_path, get_git_revision_hash
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE
sys.path.append('../cnn-rnn')  # HACK
import visualization_utils

import dgl
import scipy.sparse as sp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append('./')

METRICS = {'rmse', 'r2', 'corr'}
        
huber_fn = nn.SmoothL1Loss()
best_test = {'rmse': 1e9, 'r2': -1e9, 'corr':-1e9}
best_val = {'rmse': 1e9, 'r2': -1e9, 'corr':-1e9}


# TODO - currently only supports single label (predictions and Y are flattened)
def eval(pred, Y):
    pred, Y = pred.flatten().detach().cpu().numpy(), Y.flatten().detach().cpu().numpy()

    # Remove entries where Y is NA
    not_na = ~np.isnan(Y)
    pred = pred[not_na]
    Y = Y[not_na]
    if Y.shape[0] < 2:
        print("Not enough valid labels in this batch :O")
        return {'rmse': 0, 'r2': 0, 'corr': 0, 'mae': 0, 'mse': 0, 'mape': 0}
    # if np.any(Y == 0):
    #     print("Y was 0")
    #     print(Y)

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
    metrics['mape'] = np.mean(np.abs((Y - pred) / (Y + 1e-5)))

    return metrics

def loss_fn(pred, Y, mode="logcosh"):
    # Remove entries where Y is NA
    not_na = ~torch.isnan(Y)
    pred = pred[not_na]
    Y = Y[not_na]
    if Y.shape[0] < 1:
        return torch.tensor(0)

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
        # if rmse < best_val['rmse']:
        best_val['rmse'] = rmse
        best_val['r2'] = r2
        best_val['corr'] = corr
    elif mode == "Test":
        # if rmse < best_test['rmse']:
        best_test['rmse'] = rmse
        best_test['r2'] = r2
        best_test['corr'] = corr

def load_subtensor(year_XY, year, in_nodes, out_nodes, device):
    X, Y, counties = year_XY[year]
    batch_inputs = X[in_nodes].float().to(device)
    batch_labels = Y[out_nodes].float().to(device)
    batch_counties = counties[out_nodes].int().to(device)
    return batch_inputs, batch_labels, batch_counties

def train_epoch(args, model, device, nodeloader, year_XY, cur_step, optimizer, epoch, writer=None):
    print("\n---------------------")
    print("Epoch ", epoch)
    print("---------------------")
    model.train()
    lr = optimizer.param_groups[0]['lr']
    print("lr =", lr)
    tot_loss, tot_rmse, tot_r2, tot_corr, tot_mae, tot_mape = 0., 0., 0., 0., 0., 0.
    all_pred = []
    all_Y = []
    result_dfs = []

    n_batch = 0
    for year in list(year_XY.keys()):
        if year == args.test_year or year == args.test_year-1: 
            continue

        # If there are missing labels in the Y sequence that's being passed to
        # the RNN, substitute the average value across the previous 4 years.
        X, Y, counties = year_XY[year]
        Y = Y[:, :-1, :]  # Exclude current year, since the model should not receive info about the current year's labels as input
        year_avg_Y = (Y[~torch.isnan(Y)].mean() - args.means) / args.stds
        # print("Year", year, "avg Y", year_avg_Y)

        for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(nodeloader):
            batch_inputs, batch_labels, batch_counties = load_subtensor(year_XY, year, in_nodes, out_nodes, device)
            batch_labels_std = (batch_labels - args.means) / args.stds
            batch_labels_std[torch.isnan(batch_labels_std)] = year_avg_Y

            #print(batch_inputs.shape, batch_labels.shape) # [711, 5, 431] [64, 5]
            blocks = [block.int().to(device) for block in blocks]
            batch_pred_std = model(blocks, batch_inputs, batch_labels_std[:, :-1]) #.squeeze(-1)
            batch_pred = batch_pred_std * args.stds + args.means

            if torch.isnan(batch_pred).any():
                print("Counties with nan")
                print(counties[np.isnan(batch_pred.squeeze())])
                print("Corresponding input")
                print(year_XY[year][np.isnan(batch_pred.squeeze())])

            loss = loss_fn(batch_pred, batch_labels[:, -1])
            optimizer.zero_grad()

            all_pred.append(batch_pred)
            all_Y.append(batch_labels[:, -1])
            metrics = eval(batch_pred, batch_labels[:, -1])
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

            # Create a dataframe with true vs. predicted yield (so that we can produce maps later)
            result_df_dict = {"fips": batch_counties.detach().cpu().numpy().astype(int).tolist(),
                              "year": [year] * batch_counties.shape[0]}
            for i in range(batch_labels.shape[2]):
                output_name = args.output_names[i]
                result_df_dict["predicted_" + output_name] = batch_pred[:, i].detach().cpu().numpy().tolist()
                result_df_dict["true_" + output_name] = batch_labels[:, -1, i].detach().cpu().numpy().tolist()
            result_dfs.append(pd.DataFrame(result_df_dict))

    results = pd.concat(result_dfs)

    # Calculate stats on all data
    all_pred = torch.cat(all_pred, dim=0)
    all_Y = torch.cat(all_Y, dim=0)
    metrics_all = eval(all_pred, all_Y)

    print("\n###### Overall training metrics")
    print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
        tot_loss/n_batch, tot_rmse/n_batch, tot_r2/n_batch, tot_corr/n_batch, tot_mae/n_batch, tot_mape/n_batch)
    )
    return cur_step, metrics_all, results


def val_epoch(args, model, device, nodeloader, year_XY, epoch, mode="Val", writer=None):
    print("********************")
    print("Epoch", epoch, mode)
    print("********************")
    model.eval()
    tot_loss, tot_rmse, tot_r2, tot_corr, tot_mae, tot_mape = 0., 0., 0., 0., 0., 0.
    all_pred = []
    all_Y = []
    result_dfs = []
    if mode == "Val":
        year = args.test_year-1
    elif mode == "Test":
        year = args.test_year

    # X, Y, counties = year_XY[year]
    # year_avg_Y = (Y[~torch.isnan(Y)].mean() - args.means) / args.stds
    #     # If there are missing labels in the Y sequence that's being passed to
    #     # the RNN, substitute the average value across the previous 4 years.
    #     X, Y, counties = year_XY[year]
    #     Y = Y[:, :-1, :]  # Exclude current year, since the model should not receive info about the current year's labels as input

    # If there are missing labels in the Y sequence that's being passed to
    # the RNN, substitute the average value across the previous 4 years.
    X, Y, counties = year_XY[year]
    Y = Y[:, :-1, :]  # Exclude current year, since the model should not receive info about the current year's labels as input
    year_avg_Y = (Y[~torch.isnan(Y)].mean() - args.means) / args.stds

    for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(nodeloader):
        batch_inputs, batch_labels, batch_counties = load_subtensor(year_XY, year, in_nodes, out_nodes, device)
        batch_labels_std = (batch_labels - args.means) / args.stds
        batch_labels_std[torch.isnan(batch_labels_std)] = year_avg_Y
        blocks = [block.int().to(device) for block in blocks]

        batch_pred_std = model(blocks, batch_inputs, batch_labels_std[:, :-1]) #.squeeze(-1)
        batch_pred = batch_pred_std * args.stds + args.means
        loss = loss_fn(batch_pred, batch_labels[:, -1])

        all_pred.append(batch_pred)
        all_Y.append(batch_labels[:, -1])
        metrics = eval(batch_pred, batch_labels[:, -1])
        tot_loss += loss.item()
        tot_rmse += metrics['rmse']
        tot_r2 += metrics['r2']
        tot_corr += metrics['corr']
        tot_mae += metrics['mae']
        tot_mape += metrics['mape']

        # Create a dataframe with true vs. predicted yield for each county in the validation
        # year (so that we can produce maps later)
        result_df_dict = {"fips": batch_counties.detach().cpu().numpy().astype(int).tolist(),
                          "year": [year] * batch_counties.shape[0]}
        for i in range(batch_labels.shape[2]):
            output_name = args.output_names[i]
            result_df_dict["predicted_" + output_name] = batch_pred[:, i].detach().cpu().numpy().tolist()
            result_df_dict["true_" + output_name] = batch_labels[:, -1, i].detach().cpu().numpy().tolist()
        result_dfs.append(pd.DataFrame(result_df_dict))

    results = pd.concat(result_dfs)

    # Calculate stats on all data
    all_pred = torch.cat(all_pred, dim=0)
    all_Y = torch.cat(all_Y, dim=0)
    metrics_all = eval(all_pred, all_Y)

    n_batch = batch_idx+1
    #print("###### Overall Validation metrics")
    print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
        tot_loss/n_batch, metrics_all['rmse'], metrics_all['r2'], metrics_all['corr'], metrics_all['mae'], metrics_all['mape'])
    )
    print("********************")

    if writer is not None:
        writer.add_scalar("{}/loss".format(mode), tot_loss/n_batch, epoch)
        writer.add_scalar("{}/rmse".format(mode), tot_rmse/n_batch, epoch)
        writer.add_scalar("{}/r2".format(mode), tot_r2/n_batch, epoch)
        writer.add_scalar("{}/corr".format(mode), tot_corr/n_batch, epoch)
        writer.add_scalar("{}/mae".format(mode), tot_mae/n_batch, epoch)
        writer.add_scalar("{}/mape".format(mode), tot_mape/n_batch, epoch)

    return metrics_all, results

def train(args):
    print('reading npy...')
    np.random.seed(args.seed) # set the random seed of numpy
    torch.manual_seed(args.seed)
    raw_data = np.load(args.data_dir) #load data from the data_dir
    data = raw_data['data']
    
    X_dict, Y_dict, avail_dict, adj, order_map, min_year, max_year, county_set = get_X_Y(data, args, device)
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
    l = args.length
    for year in range(min_year+l-1, max_year+1):
        X_seqs = []
        Y_seqs = []
        counties = []
        for county in county_set:
            X_seq = []
            Y_seq = []
            for i in range(year-l+1, year+1):
                X_seq.append(X_dict[county][i])
                Y_seq.append(Y_dict[county][i])
            X_seqs.append(X_seq)
            Y_seqs.append(Y_seq)
            counties.append(county)
        X_seqs, Y_seqs, counties = torch.tensor(X_seqs), torch.tensor(Y_seqs), torch.tensor(counties)
        year_XY[year] = (X_seqs, Y_seqs, counties)

    param_setting = "gnn-rnn_bs-{}_lr-{}_maxepoch-{}_sche-{}_T0-{}_step-{}_gamma-{}_sleep-{}_testyear-{}_len-{}_seed-{}".format(
        args.batch_size, args.learning_rate, args.max_epoch, args.scheduler, args.T0, args.lrsteps, args.gamma, args.sleep, args.test_year, args.length, args.seed)
    if args.no_management:
        param_setting += "_no-management"

    summary_dir = 'summary/{}/{}'.format(args.dataset, param_setting)
    model_dir = 'model/{}/{}'.format(args.dataset, param_setting)
    results_dir = 'results/{}/{}'.format(args.dataset, param_setting)
    build_path(summary_dir)
    build_path(model_dir)
    build_path(results_dir)
    writer = SummaryWriter(log_dir=summary_dir)

    # Summary csv file of all results. Create this if it doesn't exist
    results_summary_file = os.path.join('results/results_summary.csv')
    if not os.path.isfile(results_summary_file):
        with open(results_summary_file, mode='w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['dataset', 'model', 'git_commit', 'command', 'val_year', 'val_rmse', 'val_r2', 'val_corr', 'test_year', 'test_rmse', 'test_r2', 'test_corr'])

    #building the model
    print('building network...')
    in_dim = year_XY[2000][0].shape[-1]
    out_dim = 1
    model = SAGE_RNN(args, in_dim, out_dim).to(device)
    
    #log the learning rate 
    #writer.add_scalar('learning_rate', args.learning_rate)

    #use the Adam optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=args.eta_min, T_0=args.T0, T_mult=args.T_mult)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lrsteps, args.gamma)
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

    # Store RMSE/R2 at each epoch, for plotting later
    train_rmse_list, val_rmse_list, test_rmse_list, train_r2_list, val_r2_list, test_r2_list = [], [], [], [], [], []

    for epoch in range(args.max_epoch):
        cur_step, train_metrics, train_results = train_epoch(args, model, device, nodeloader, year_XY, cur_step, optimizer, epoch, writer)
        val_metrics, val_results = val_epoch(args, model, device, nodeloader, year_XY, epoch, "Val", writer)
        test_metrics, test_results = val_epoch(args, model, device, nodeloader, year_XY, epoch, "Test", writer)

        # Record epoch metrics in list
        val_rmse = val_metrics['rmse']
        train_rmse_list.append(train_metrics['rmse'])
        val_rmse_list.append(val_metrics['rmse'])
        test_rmse_list.append(test_metrics['rmse'])
        train_r2_list.append(train_metrics['r2'])
        val_r2_list.append(val_metrics['r2'])
        test_r2_list.append(test_metrics['r2'])

        if val_rmse < best_val_rmse:
            update_metrics(val_metrics['rmse'], val_metrics['r2'], val_metrics['corr'], "Val")
            update_metrics(test_metrics['rmse'], test_metrics['r2'], test_metrics['corr'], "Test")
            best_val_rmse = val_rmse

            # Save model to file
            torch.save(model.state_dict(), model_dir+'/model-'+str(epoch))
            print('save model to', model_dir)

            # Save raw results (true and predicted labels) to files
            train_results.to_csv(os.path.join(results_dir, "train_results.csv"), index=False)
            val_results.to_csv(os.path.join(results_dir, "val_results.csv"), index=False)
            test_results.to_csv(os.path.join(results_dir, "test_results.csv"), index=False)

            # Plot results
            visualization_utils.plot_true_vs_predicted(train_results[train_results["year"] == 1986], args.output_names, "1986_train", results_dir)
            visualization_utils.plot_true_vs_predicted(val_results, args.output_names, str(args.test_year - 1) + "_val", results_dir)
            visualization_utils.plot_true_vs_predicted(test_results, args.output_names, str(args.test_year) + "_test", results_dir)

        print("BEST Val | rmse: {}, r2: {}, corr: {}".format(best_val['rmse'], best_val['r2'], best_val['corr']))
        print("BEST Test | rmse: {}, r2: {}, corr: {}".format(best_test['rmse'], best_test['r2'], best_test['corr']))
        if scheduler is not None and epoch < args.sleep-1:
            scheduler.step()
    
    # Record final results
    git_commit = get_git_revision_hash()
    command_string = " ".join(sys.argv)
    print("Command used:", command_string)
    print("Model dir:", model_dir)    
    with open(results_summary_file, mode='a+') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([args.dataset, args.model, git_commit, command_string,
                             str(args.test_year - 1), best_val['rmse'], best_val['r2'], best_val['corr'],
                             str(args.test_year), best_test['rmse'], best_test['r2'], best_test['corr']])

    # Record Git commit and command used, along with final metrics
    with open(os.path.join(results_dir, "summary.txt"), 'w') as f:
        f.write("Algorithm: " + args.model + "\n")
        f.write("Dataset: " + args.dataset + "\n")
        f.write("Git commit: " + git_commit + "\n")
        f.write("Command: " + command_string + "\n\n")
        f.write("BEST Val (" + str(args.test_year - 1) + ") | rmse: {}, r2: {}, corr: {}\n".format(best_val['rmse'], best_val['r2'], best_val['corr']))
        f.write("BEST Test (" + str(args.test_year) + ") | rmse: {}, r2: {}, corr: {}\n".format(best_test['rmse'], best_test['r2'], best_test['corr']))
 
    # Plot RMSE over time
    epoch_list = range(len(train_rmse_list))
    plots = []
    train_rmse_plot, = plt.plot(epoch_list, train_rmse_list, color='blue', label='Train RMSE (1981-' + str(args.test_year - 2) + ')')
    val_rmse_plot, = plt.plot(epoch_list, val_rmse_list, color='orange', label='Validation RMSE (' + str(args.test_year - 1) + ')')
    test_rmse_plot, = plt.plot(epoch_list, test_rmse_list, color='red', label='Test RMSE (' + str(args.test_year) + ')')
    plots.append(train_rmse_plot)
    plots.append(val_rmse_plot)
    plots.append(test_rmse_plot)
    plt.legend(handles=plots)
    plt.xlabel('Epoch #')
    plt.ylabel('RMSE')
    plt.savefig(os.path.join(results_dir, "metrics_rmse.png"))
    plt.close()

    # Plot R2 over time
    plots = []
    train_r2_plot, = plt.plot(epoch_list, train_r2_list, color='blue', label='Train R^2 (1981-' + str(args.test_year - 2) + ')')
    val_r2_plot, = plt.plot(epoch_list, val_r2_list, color='orange', label='Validation R^2 (' + str(args.test_year - 1) + ')')
    test_r2_plot, = plt.plot(epoch_list, test_r2_list, color='red', label='Test R^2 (' + str(args.test_year) + ')')
    plots.append(train_r2_plot)
    plots.append(val_r2_plot)
    plots.append(test_r2_plot)
    plt.legend(handles=plots)
    plt.xlabel('Epoch #')
    plt.ylabel('R^2')
    plt.savefig(os.path.join(results_dir, "metrics_r2.png"))
    plt.close()
