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
from utils import get_X_Y, build_path, get_git_revision_hash, mask_end
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE
sys.path.append('../cnn-rnn')  # HACK
import visualization_utils

import dgl
import scipy.sparse as sp

# device = torch.device("cuda:0")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device", device)
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

# pred, Y can be 2D or 3D, but the last dimension is the "output" dimension. We take the average loss across all outputs.
def loss_fn(pred, Y, args, mode="logcosh"):
    # Remove entries where Y is NA
    # print("Loss fn", pred.shape, Y.shape)
    Y = torch.reshape(Y, (-1, Y.shape[-1]))
    pred = torch.reshape(pred, (-1, pred.shape[-1]))
    not_na = ~torch.isnan(Y)
    pred = pred[not_na]
    Y = Y[not_na]
    # if Y.shape[0] < 1:
    #     return torch.tensor(0)

    # Standardize based on mean/std of each output (crop type)
    Y = (Y - args.means) / args.stds
    pred = (pred - args.means) / args.stds

    # TODO Compute loss for each output
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
    X, Y, counties, in_nodes, out_nodes = X.to(device), Y.to(device), counties.to(device), in_nodes.to(device), out_nodes.to(device)
    batch_inputs = X[in_nodes].float().to(device)
    batch_labels = Y[out_nodes].float().to(device)
    batch_counties = counties[out_nodes].int().to(device)
    return batch_inputs, batch_labels, batch_counties

def train_epoch(args, model, device, nodeloader, year_XY, county_avg, year_avg_Y, cur_step, optimizer, epoch, writer=None):
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

    # Generate random order of years
    all_years = list(year_XY.keys())
    np.random.shuffle(all_years)
    print("All years", all_years)
    for year in all_years:  # list(year_XY.keys()):
        if year == args.test_year or year == args.test_year-1: 
            continue

        for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(nodeloader):
            batch_inputs, batch_labels, batch_counties = load_subtensor(year_XY, year, in_nodes, out_nodes, device)
            if torch.isnan(batch_labels[:, -1]).all():
                continue
            batch_labels_std = (batch_labels - args.means) / args.stds

            # Randomly mask out some data from the end of the last year in the 5-year sequence,
            # to force model to learn how to make early predictions
            # batch_input_counties = year_XY[year][2][in_nodes].int().to(device)
            # batch_inputs[:, -1, :] = mask_end(batch_inputs[:, -1, :], batch_input_counties, county_avg, args, args.train_week_start, args.time_intervals, device)

            # # If labels are missing in the INPUT to the model (batch_labels_std),
            # # replace them with the average for that year
            # seq_years = batch_labels_std.shape[1]
            # for i in range(seq_years - 1):  # Exclude current year
            #     for j in range(batch_labels_std.shape[2]):  # Loop through each output variable (crop type)
            #         year_i = (year - seq_years + 1) + i
            #         missing_indices = torch.isnan(batch_labels_std[:, i, j])
            #         batch_labels_std[missing_indices, i, j] = (year_avg_Y[year_i][j] - args.means) / args.stds

            # # print("Batch inputs", batch_inputs.shape)
            # # print("Batch labels", batch_labels_std.shape)
            # indices_with_nan_labels = torch.isnan(batch_labels_std).any(dim=1).squeeze()
            
            # batch_labels_std[torch.isnan(batch_labels_std)] = args.means[0]  # year_avg_Y



            # # Quick sanity checking
            # visualization_utils.sanity_check_input(batch_inputs, batch_input_counties, year, args, X_mean, X_std)
            # for i in range(5):
            #     print(str(year) + ", county " + str(batch_counties[i]) + ": yield = " + str(batch_labels[i, -1].item()))
            # exit(0)

            #print(batch_inputs.shape, batch_labels.shape) # [711, 5, 431] [64, 5]
            blocks = [block.int().to(device) for block in blocks]
            batch_pred_std = model(blocks, batch_inputs, batch_labels_std[:, :-1]) #.squeeze(-1)
            batch_pred = batch_pred_std * args.stds + args.means

            # if torch.isnan(batch_pred).any():
            #     print("Counties with nan")
            #     print(counties[np.isnan(batch_pred.squeeze())])
            #     print("Corresponding input")
            #     print(year_XY[year][np.isnan(batch_pred.squeeze())])

            # loss = loss_fn(batch_pred, batch_labels[:, -1])

            loss = loss_fn(batch_pred[:, :args.length-1, :], batch_labels[:, :args.length-1, :], args) * args.c1 + \
                   loss_fn(batch_pred[:, -1, :], batch_labels[:, -1, :], args) * args.c2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_batch += 1

            # Record values and metrics
            all_pred.append(batch_pred[:, -1])
            all_Y.append(batch_labels[:, -1])
            metrics_batch = eval(batch_pred[:, -1], batch_labels[:, -1], args)
            tot_loss += loss.item()

            if n_batch % args.check_freq == 0:
                print("### batch ", n_batch-1)
                print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
                    loss.item(), metrics_batch['rmse'], metrics_batch['r2'], metrics_batch['corr'], metrics_batch['mae'], metrics_batch['mape'])
                )
        
            if writer is not None:
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("lr", lr, cur_step)
                writer.add_scalar("Train/loss", loss.item(), cur_step)
                writer.add_scalar("Train/rmse", metrics_batch['rmse'], cur_step)
                writer.add_scalar("Train/r2", metrics_batch['r2'], cur_step)
                writer.add_scalar("Train/corr", metrics_batch['corr'], cur_step)
                writer.add_scalar("Train/mae", metrics_batch['mae'], cur_step)
                writer.add_scalar("Train/mape", metrics_batch['mape'], cur_step)
            cur_step += 1

            # Create a dataframe with true vs. predicted yield (so that we can produce maps later)
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
    print("\n###### Overall training metrics")
    print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
        tot_loss/n_batch, metrics_all['rmse'], metrics_all['r2'], metrics_all['corr'], metrics_all['mae'], metrics_all['mape'])
    )

    return cur_step, metrics_all, results


def val_epoch(args, model, device, nodeloader, year_XY, county_avg, year_avg_Y, epoch, mode="Val", writer=None):
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

    with torch.no_grad():
        for batch_idx, (in_nodes, out_nodes, blocks) in enumerate(nodeloader):
            batch_inputs, batch_labels, batch_counties = load_subtensor(year_XY, year, in_nodes, out_nodes, device)

            # To simulate early prediction, mask out data starting from the specified "validation_week"
            # batch_input_counties = year_XY[year][2][in_nodes].int().to(device)
            # batch_inputs[:, -1, :] = mask_end(batch_inputs[:, -1, :], batch_input_counties, county_avg, args, args.validation_week, args.validation_week, device)

            batch_labels_std = (batch_labels - args.means) / args.stds

            # # If labels are missing in the INPUT to the model (batch_labels_std),
            # # replace them with the average for that year
            # seq_years = batch_labels_std.shape[1]
            # for i in range(seq_years - 1):  # Exclude current year
            #     for j in range(batch_labels_std.shape[2]):  # Loop through each output variable (crop type)
            #         year_i = (year - seq_years + 1) + i
            #         missing_indices = torch.isnan(batch_labels_std[:, i, j])
            #         batch_labels_std[missing_indices, i, j] = (year_avg_Y[year_i][j] - args.means) / args.stds

            blocks = [block.int().to(device) for block in blocks]

            batch_pred_std = model(blocks, batch_inputs, batch_labels_std[:, :-1]) #.squeeze(-1)
            batch_pred = batch_pred_std * args.stds + args.means
            loss = loss_fn(batch_pred[:, -1], batch_labels[:, -1], args)

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

        if writer is not None:
            writer.add_scalar("{}/loss".format(mode), tot_loss/n_batch, epoch)
            writer.add_scalar("{}/rmse".format(mode), metrics_all['rmse'], epoch)
            writer.add_scalar("{}/r2".format(mode), metrics_all['r2'], epoch)
            writer.add_scalar("{}/corr".format(mode), metrics_all['corr'], epoch)
            writer.add_scalar("{}/mae".format(mode), metrics_all['mae'], epoch)
            writer.add_scalar("{}/mape".format(mode), metrics_all['mape'], epoch)

    return metrics_all, results

def train(args):
    print('reading npy...')
    torch.set_num_threads(12)
    random.seed(args.seed)
    np.random.seed(args.seed) # set the random seed of numpy
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dgl.seed(args.seed)
    # dgl.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    raw_data = np.load(args.data_dir) #load data from the data_dir
    data = raw_data['data']
    
    X_dict, Y_dict, avail_dict, adj, order_map, min_year, max_year, county_set, county_avg, year_avg_Y = get_X_Y(data, args, device)
    print("Average Y calculated originally")
    for year in year_avg_Y:
        year_avg_Y[year] = torch.tensor(year_avg_Y[year], device=device)
    print(year_avg_Y)
    print('=============================================')
    sp_adj = sp.coo_matrix(adj)
    g = dgl.from_scipy(sp_adj).to(device)
    
    #print(max(np.sum(adj, axis=1))) # 10

    N = len(adj)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 10])
    nodeloader = dgl.dataloading.DataLoader(
        g,
        torch.arange(N).to(device),
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers = 0,
        device=device
    )

    year_XY = {}
    l = args.length
    num_nan_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # Construct 5-year sequences for each county/year
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
            
            # Count number of NaN in Y
            num_nan = sum(math.isnan(y) for y in Y_seq)
            num_nan_distribution[num_nan] += 1
            X_seqs.append(X_seq)
            Y_seqs.append(Y_seq)
            counties.append(county)
        X_seqs, Y_seqs, counties = torch.tensor(X_seqs), torch.tensor(Y_seqs), torch.tensor(counties)
        year_XY[year] = (X_seqs, Y_seqs, counties)

    # Compute average yield for year
    year_avg_Y_new = {}
    for year in range(min_year, max_year+1):
        Y_values = []
        for county in county_set:
            Y_values.append(Y_dict[county][year])
        Y = np.array(Y_values)
        avg_Y = np.nanmean(Y, axis=0)
        year_avg_Y_new[year] = torch.tensor(avg_Y)
    print("New avg Y")
    print(year_avg_Y_new)

    param_setting = "gnn-rnn_bs-{}_lr-{}_maxepoch-{}_sche-{}_T0-{}_etamin-{}_step-{}_gamma-{}_dropout-{}_sleep-{}_testyear-{}_aggregator-{}_encoder-{}_trainweekstart-{}_len-{}_weightdecay-{}_seed-{}".format(
        args.batch_size, args.learning_rate, args.max_epoch, args.scheduler, args.T0, args.eta_min, args.lrsteps, args.gamma, args.dropout, args.sleep, args.test_year, args.aggregator_type, args.encoder_type, args.train_week_start, args.length, args.weight_decay, args.seed)
    if args.no_management:
        param_setting += "_no-management"
    if args.checkpoint_path != "./ckpt":
        param_setting += ("checkpoint-" + args.checkpoint_path)
        print("Checkpoint:", args.checkpoint_path)

    summary_dir = 'summary/{}/{}/{}'.format(args.dataset, args.test_year, param_setting)
    model_dir = 'model/{}/{}/{}'.format(args.dataset, args.test_year, param_setting)
    results_dir = 'results/{}/{}/{}'.format(args.dataset, args.test_year, param_setting)
    build_path(summary_dir)
    build_path(model_dir)
    build_path(results_dir)
    writer = SummaryWriter(log_dir=summary_dir)

    # Summary csv file of all results. Create this if it doesn't exist
    results_summary_file = 'results/{}/{}/results_summary.csv'.format(args.dataset, args.test_year)
    if not os.path.isfile(results_summary_file):
        with open(results_summary_file, mode='w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['dataset', 'model', 'git_commit', 'command', 'val_year', 'val_rmse', 'val_r2', 'val_corr', 'test_year', 'test_rmse', 'test_r2', 'test_corr', 'path_to_model'])
    git_commit = get_git_revision_hash()
    command_string = " ".join(sys.argv)

    #building the model
    print('building network...')
    in_dim = year_XY[2000][0].shape[-1]
    out_dim = 1
    model = SAGE_RNN(args, in_dim, out_dim).to(device)
    # model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    # print("Loaded state dict")

    # model.load_state_dict(torch.load(os.path.join(model_dir, args.model_filename), map_location=device))

    #log the learning rate 
    #writer.add_scalar('learning_rate', args.learning_rate)

    #use the Adam optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  #1e-5)
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
        epoch_start = time()
        cur_step, train_metrics, train_results = train_epoch(args, model, device, nodeloader, year_XY, county_avg, year_avg_Y, cur_step, optimizer, epoch, writer)
        val_metrics, val_results = val_epoch(args, model, device, nodeloader, year_XY, county_avg, year_avg_Y, epoch, "Val", writer)
        test_metrics, test_results = val_epoch(args, model, device, nodeloader, year_XY, county_avg, year_avg_Y, epoch, "Test", writer)
        epoch_time =  time() - epoch_start
        print("Epoch finished in", epoch_time)

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
            best_model_path = model_dir + '/model-' + str(epoch)
            torch.save(model.state_dict(), best_model_path)
            print('save model to', best_model_path)

            # Save raw results (true and predicted labels) to files
            train_results.to_csv(os.path.join(results_dir, "train_results.csv"), index=False)
            val_results.to_csv(os.path.join(results_dir, "val_results.csv"), index=False)
            test_results.to_csv(os.path.join(results_dir, "test_results.csv"), index=False)

            # Plot results
            # visualization_utils.plot_true_vs_predicted(train_results[train_results["year"] == 1986], args.output_names, "1986_train", results_dir)
            visualization_utils.plot_true_vs_predicted(val_results, args.output_names, str(args.test_year - 1) + "_val", results_dir)
            visualization_utils.plot_true_vs_predicted(test_results, args.output_names, str(args.test_year) + "_test", results_dir)

            # Record Git commit and command used, along with current metrics
            with open(os.path.join(results_dir, "summary_current.txt"), 'w') as f:
                f.write("Algorithm: " + args.model + "\n")
                f.write("Dataset: " + args.dataset + "\n")
                f.write("Git commit: " + git_commit + "\n")
                f.write("Command: " + command_string + "\n")
                f.write("Final model path: " + best_model_path + "\n\n")
                f.write("BEST Val (" + str(args.test_year - 1) + ") | rmse: {}, r2: {}, corr: {}\n".format(best_val['rmse'], best_val['r2'], best_val['corr']))
                f.write("BEST Test (" + str(args.test_year) + ") | rmse: {}, r2: {}, corr: {}\n".format(best_test['rmse'], best_test['r2'], best_test['corr']))

            with open(os.path.join(results_dir, "csv_row.csv"), mode='w') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([args.dataset, args.model, git_commit, command_string,
                                    str(args.test_year - 1), best_val['rmse'], best_val['r2'], best_val['corr'],
                                    str(args.test_year), best_test['rmse'], best_test['r2'], best_test['corr'], best_model_path])       

        if epoch % 5 == 0 or epoch == args.max_epoch - 1:
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
            plt.ylim(0, 1)
            plt.ylabel('R^2')
            plt.savefig(os.path.join(results_dir, "metrics_r2.png"))
            plt.close()

        print("BEST Val | rmse: {}, r2: {}, corr: {}".format(best_val['rmse'], best_val['r2'], best_val['corr']))
        print("BEST Test | rmse: {}, r2: {}, corr: {}".format(best_test['rmse'], best_test['r2'], best_test['corr']))
        print("Model path:", best_model_path)

        if scheduler is not None and epoch < args.sleep-1:
            scheduler.step()
    
    # Record final results
    print("Command used:", command_string)
    print("Model dir:", model_dir)    
    with open(results_summary_file, mode='a+') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([args.dataset, args.model, git_commit, command_string,
                             str(args.test_year - 1), best_val['rmse'], best_val['r2'], best_val['corr'],
                             str(args.test_year), best_test['rmse'], best_test['r2'], best_test['corr'], best_model_path])

    # Record Git commit and command used, along with final metrics
    with open(os.path.join(results_dir, "summary_FINAL.txt"), 'w') as f:
        f.write("Algorithm: " + args.model + "\n")
        f.write("Dataset: " + args.dataset + "\n")
        f.write("Git commit: " + git_commit + "\n")
        f.write("Command: " + command_string + "\n")
        f.write("Final model path: " + best_model_path + "\n\n")
        f.write("BEST Val (" + str(args.test_year - 1) + ") | rmse: {}, r2: {}, corr: {}\n".format(best_val['rmse'], best_val['r2'], best_val['corr']))
        f.write("BEST Test (" + str(args.test_year) + ") | rmse: {}, r2: {}, corr: {}\n".format(best_test['rmse'], best_test['r2'], best_test['corr']))

