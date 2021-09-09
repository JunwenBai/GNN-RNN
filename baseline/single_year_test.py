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
from single_year_models import SingleYearCNN, SingleYearRNN
import random
from baseline_utils import get_X_Y, build_path, mask_end, get_git_revision_hash
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE
import visualization_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append('./')

METRICS = {'rmse', 'r2', 'corr'}
        
huber_fn = nn.SmoothL1Loss()
best_test = {'rmse': 1e9, 'r2': -1e9, 'corr':-1e9}
best_val = {'rmse': 1e9, 'r2': -1e9, 'corr':-1e9}

# pred, Y assumed to be 2D: [examples x outputs]
def eval(pred, Y, args):
    # Standardize based on mean/std of each output (crop type). NOTE - ignore this for now
    Y = (Y - args.means) / args.stds
    pred = (pred - args.means) / args.stds
    pred, Y = pred.detach().cpu().numpy(), Y.detach().cpu().numpy()
    metric_names = ['rmse', 'r2', 'corr', 'mae', 'mse', 'mape']
    metrics = {metric_name : {} for metric_name in metric_names}

    # Compute metrics for each output variable (crop type)
    for idx in range(Y.shape[-1]): # in enumerate(args.output_names):
        output_name = args.output_names[idx]
        not_na = ~np.isnan(Y[:, idx])
        Y_i = Y[not_na, idx]
        pred_i = pred[not_na, idx]
        if Y_i.shape[0] == 0:
            continue

        # RMSE
        metrics['rmse'][output_name] = np.sqrt(np.mean((pred_i-Y_i)**2))
        # R2
        metrics['r2'][output_name] = r2_score(Y_i, pred_i)
        # corr
        if np.all(Y_i == Y_i[0]) or np.all(pred_i == pred_i[0]):  # If all predictions are the same, calculating correlation produces an error, so just set to 0
            metrics['corr'][output_name] = 0
        else:
            metrics['corr'][output_name] = np.corrcoef(Y_i, pred_i)[0, 1]
        # MAE
        metrics['mae'][output_name] = MAE(Y_i, pred_i)
        # MSE
        metrics['mse'][output_name] = np.mean((pred_i-Y_i)**2)
        # MAPE
        metrics['mape'][output_name] = np.mean(np.abs((Y_i - pred_i) / (Y_i + 1e-5)))

    # For each metric, average over all outputs
    for metric_name in metrics:
        metrics[metric_name]["avg"] = sum(metrics[metric_name].values()) / len(metrics[metric_name])

    return metrics

# pred, Y can be 2D or 3D, but the last dimension is the "output" dimension. We take the average loss across all outputs.
def loss_fn(pred, Y, args, mode="logcosh"):
    loss = 0

    # Y = torch.reshape(Y, (-1, Y.shape[-1]))
    # pred = torch.reshape(pred, (-1, pred.shape[-1]))

    # Standardize based on mean/std of each output (crop type)
    Y = (Y - args.means) / args.stds
    pred = (pred - args.means) / args.stds

    # Compute loss for each output (crop type)
    for i in range(Y.shape[-1]):
        # Remove rows with NA label
        not_na = ~torch.isnan(Y[:, i])
        Y_i = Y[not_na, i]
        pred_i = pred[not_na, i]
        if Y_i.shape[0] == 0:
            print("Entire column is NaN")
            continue

        if mode == "huber":
            # huber loss
            loss = huber_fn(pred_i, Y_i)
        elif mode == "logcosh":
            # log cosh loss
            err = Y_i - pred_i
            loss += torch.mean(torch.log(torch.cosh(err + 1e-12)))
    loss = loss / Y.shape[-1]
    if np.isnan(loss.item()):
        print("Loss was nan :(")
        print("True", Y)
        print("Predicted", pred)
        exit(1)
 
    return loss


# Note: this was changed so that metrics are always updated!
def update_metrics(rmse, r2, corr, mode):
    if mode == "Val":
        best_val['rmse'] = rmse
        best_val['r2'] = r2
        best_val['corr'] = corr
    elif mode == "Test":
        # if rmse < best_test['rmse']:
        best_test['rmse'] = rmse
        best_test['r2'] = r2
        best_test['corr'] = corr


def test_epoch(args, model, device, test_loader, county_avg, mode="Val"):
    print("********************")
    print("Test single-year model")
    print("********************")
    model.eval()
    tot_loss = 0.
    result_dfs = []
    all_pred = []
    all_Y = []
    for batch_idx, (X, Y, counties, years) in enumerate(test_loader):
        X, Y, counties, years = X.to(device), Y.to(device), counties.to(device), years.to(device)

        # To simulate early prediction, mask out data starting from the specified "validation_week"
        X = mask_end(X, counties, county_avg, args, args.validation_week, args.validation_week, device)

        predictions_std = model(X)
        pred = predictions_std * args.stds + args.means
        loss = loss_fn(pred, Y, args)
        tot_loss += loss.item()
        all_pred.append(pred)
        all_Y.append(Y)

        # Create a dataframe with true vs. predicted yield for each county in the validation
        # year (so that we can produce maps later)
        result_df_dict = {"fips": counties.detach().cpu().numpy().astype(int).tolist(),
                          "year": years.detach().cpu().numpy().astype(int).tolist()}
        for i in range(Y.shape[1]):
            output_name = args.output_names[i]
            result_df_dict["predicted_" + output_name] = pred[:, i].detach().cpu().numpy().tolist()
            result_df_dict["true_" + output_name] = Y[:, i].detach().cpu().numpy().tolist()
        result_dfs.append(pd.DataFrame(result_df_dict))

    results = pd.concat(result_dfs)

    # Calculate stats on all data
    all_pred = torch.cat(all_pred, dim=0)
    all_Y = torch.cat(all_Y, dim=0)
    metrics_all = eval(all_pred, all_Y, args)

    n_batch = batch_idx+1

    print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
        tot_loss/n_batch, metrics_all['rmse']['avg'], metrics_all['r2']['avg'], metrics_all['corr']['avg'], metrics_all['mae']['avg'], metrics_all['mape']['avg'])
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
    results_dir = os.path.join("results", normalized_checkpoint_path[-3], normalized_checkpoint_path[-2])
    print("RESULTS DIR", results_dir)

    # Load data from the data_dir
    print("Loading data from", args.data_dir)
    if args.data_dir.endswith(".npz"):
        raw_data = np.load(args.data_dir) 
        data = raw_data['data']
    elif args.data_dir.endswith(".npy"):
        data = np.load(args.data_dir)
    elif args.data_dir.endswith(".csv"):
        data = np.genfromtxt(args.data_dir, dtype=float, delimiter=',')
    else:
        raise ValueError("--data_dir argument must end in .npz, .npy, or .csv")
    print("Raw data shape", data.shape)

    # Extract X/Y matrices from raw data
    X_train, Y_train, counties_train, years_train, X_val, Y_val, counties_val, years_val, X_test, Y_test, counties_test, years_test, county_avg = get_X_Y(data, args, device)

    # Create Tensors, datasets, dataloaders
    X_train, X_val, X_test = torch.Tensor(X_train), torch.Tensor(X_val), torch.Tensor(X_test)
    Y_train, Y_val, Y_test = torch.Tensor(Y_train), torch.Tensor(Y_val), torch.Tensor(Y_test)
    counties_train, counties_val, counties_test = torch.Tensor(counties_train), torch.Tensor(counties_val), torch.Tensor(counties_test)
    years_train, years_val, years_test = torch.Tensor(years_train), torch.Tensor(years_val), torch.Tensor(years_test)
    print("test:", X_test.shape, Y_test.shape)

    test_dataset = TensorDataset(X_test, Y_test, counties_test, years_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    
    #building the model 
    print('building network...')
    if args.model == "cnn":
        model = SingleYearCNN(args).to(device)
    elif args.model == "lstm" or args.model == "gru":
        model = SingleYearRNN(args).to(device)
    else:
        raise ValueError("model type not supported yet")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    
    test_metrics, test_results = test_epoch(args, model, device, test_loader, county_avg, "Test")
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
    results_summary_file = 'results/{}/results_summary_TEST.csv'.format(args.dataset)  #, args.test_year)
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
                             str(args.test_year), test_metrics['rmse']['avg'], test_metrics['r2']['avg'], test_metrics['corr']['avg'], args.checkpoint_path])
