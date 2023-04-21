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
import sys
import os
import datetime
from copy import copy, deepcopy
from model import SAGE_RNN
import random
from utils import get_X_Y, build_path, mask_end, get_git_revision_hash
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE
import dgl
import scipy.sparse as sp
sys.path.append('../cnn-rnn')  # HACK
import visualization_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append('./')

METRICS = {'rmse', 'r2', 'corr'}
        
huber_fn = nn.SmoothL1Loss()
best_test = {'rmse': 1e9, 'r2': -1e9, 'corr':-1e9}
best_val = {'rmse': 1e9, 'r2': -1e9, 'corr':-1e9}

# TODO - currently only supports single label (predictions and Y are flattened)
def eval(pred, Y, args):
    Y = (Y - args.means) / args.stds
    pred = (pred - args.means) / args.stds
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
    if np.all(Y == Y[0]) or np.all(pred == pred[0]):  # If all predictions are the same, calculating correlation produces an error, so just set to 0
        metrics['corr'] = 0
    else:
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
    # if Y.shape[0] < 1:
    #     return torch.tensor(0)

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


def test_epoch(args, model, device, nodeloader, year_XY, county_avg, year_avg_Y, mode="Test"):
    print("********************")
    print("Test GNN-RNN")
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

    with torch.no_grad():
        for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(nodeloader):
            batch_inputs, batch_labels, batch_counties = load_subtensor(year_XY, year, in_nodes, out_nodes, device)

            # To simulate early prediction, mask out data starting from the specified "validation_week"
            batch_input_counties = year_XY[year][2][in_nodes].int().to(device)
            # batch_inputs = mask_end(batch_inputs, batch_input_counties, county_avg, args, args.validation_week, args.validation_week, device)
            batch_inputs[:, -1, :] = mask_end(batch_inputs[:, -1, :], batch_input_counties, county_avg, args, args.validation_week, args.validation_week, device)

            batch_labels_std = (batch_labels - args.means) / args.stds

            # If labels are missing in the INPUT to the model (batch_labels_std),
            # replace them with the average for that year
            seq_years = batch_labels_std.shape[1]
            for i in range(seq_years - 1):  # Exclude current year
                for j in range(batch_labels_std.shape[2]):  # Loop through each output variable (crop type)
                    year_i = (year - seq_years + 1) + i
                    missing_indices = torch.isnan(batch_labels_std[:, i, j])
                    batch_labels_std[missing_indices, i, j] = (year_avg_Y[year_i][j] - args.means) / args.stds

            blocks = [block.int().to(device) for block in blocks]

            batch_pred_std = model(blocks, batch_inputs, batch_labels_std[:, :-1]) #.squeeze(-1)
            batch_pred = batch_pred_std * args.stds + args.means
            loss = loss_fn(batch_pred[:, -1], batch_labels[:, -1])

            all_pred.append(batch_pred[:, -1])
            all_Y.append(batch_labels[:, -1])
            metrics = eval(batch_pred[:, -1], batch_labels[:, -1], args)
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
                result_df_dict["predicted_" + output_name] = batch_pred[:, -1, i].detach().cpu().numpy().tolist()
                result_df_dict["true_" + output_name] = batch_labels[:, -1, i].detach().cpu().numpy().tolist()
            result_dfs.append(pd.DataFrame(result_df_dict))

        results = pd.concat(result_dfs)

        # Calculate stats on all data
        all_pred = torch.cat(all_pred, dim=0)
        all_Y = torch.cat(all_Y, dim=0)
        metrics_all = eval(all_pred, all_Y, args)

        n_batch = batch_idx+1
        #print("###### Overall Validation metrics")
        print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
            tot_loss/n_batch, metrics_all['rmse'], metrics_all['r2'], metrics_all['corr'], metrics_all['mae'], metrics_all['mape'])
        )
        print("********************")


    return metrics_all, results


def test(args):
    print('reading npy...')
    np.random.seed(args.seed) # set the random seed of numpy
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Compute results directory
    normalized_checkpoint_path = os.path.normpath(args.checkpoint_path)
    normalized_checkpoint_path = normalized_checkpoint_path.split(os.sep)
    results_dir = os.path.join("results", normalized_checkpoint_path[-4], normalized_checkpoint_path[-3], normalized_checkpoint_path[-2])
    print("RESULTS DIR", results_dir)

    # Load data from data_dir
    raw_data = np.load(args.data_dir)
    data = raw_data['data']
    
    X_dict, Y_dict, avail_dict, adj, order_map, min_year, max_year, county_set, county_avg, year_avg_Y = get_X_Y(data, args, device)
    for year in year_avg_Y:
        year_avg_Y[year] = torch.tensor(year_avg_Y[year], device=device)
    sp_adj = sp.coo_matrix(adj)
    g = dgl.from_scipy(sp_adj)
    N = len(adj)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 10])
    nodeloader = dgl.dataloading.DataLoader(
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

    #building the model
    print('building network...')
    in_dim = year_XY[2000][0].shape[-1]
    out_dim = 1
    model = SAGE_RNN(args, in_dim, out_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    test_metrics, test_results = test_epoch(args, model, device, nodeloader, year_XY, county_avg, year_avg_Y)
    time_str = str(args.test_year) + "_week_" + str(args.validation_week)
    test_results.to_csv(os.path.join(results_dir, "test_results_" + time_str + ".csv"), index=False)
    visualization_utils.plot_true_vs_predicted(test_results, args.output_names, 
                                                time_str + "_test", results_dir)

    # Record Git commit and command used, along with final metrics
    git_commit = get_git_revision_hash()
    command_string = " ".join(sys.argv)
    with open(os.path.join(results_dir, "test_summary_" + time_str + ".txt"), 'w') as f:
        f.write("Algorithm: " + args.model + "\n")
        f.write("Dataset: " + args.dataset + "\n")
        f.write("Git commit: " + git_commit + "\n")
        f.write("Test Command: " + command_string + "\n")
        f.write("Checkpoint: " + args.checkpoint_path + "\n")
        f.write("Test (" + time_str + ") | rmse: {}, r2: {}, corr: {}\n".format(test_metrics['rmse'], test_metrics['r2'], test_metrics['corr']))
 
    # Summary csv file of all TEST results. Create this if it doesn't exist
    results_summary_file = 'results/{}/{}/results_summary_TEST.csv'.format(args.dataset, args.test_year)
    if not os.path.isfile(results_summary_file):
        with open(results_summary_file, mode='w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['dataset', 'model', 'git_commit', 'command', 'test_week', 'mask_value', 'mask_prob', 'test_year', 'test_rmse', 'test_r2', 'test_corr', 'checkpoint'])
    git_commit = get_git_revision_hash()
    command_string = " ".join(sys.argv)
    with open(results_summary_file, mode='a+') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([args.dataset, args.model, git_commit, command_string,
                             args.validation_week, args.mask_value, args.mask_prob,
                             str(args.test_year), test_metrics['rmse'], test_metrics['r2'], test_metrics['corr'], args.checkpoint_path])
