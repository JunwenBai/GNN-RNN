import csv
import math
import matplotlib.pyplot as plt
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
from utils import get_X_Y, build_path, get_git_revision_hash, mask_end
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE
import pandas as pd
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

    Y = torch.reshape(Y, (-1, Y.shape[-1]))
    pred = torch.reshape(pred, (-1, pred.shape[-1]))

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


def train_epoch(args, model, device, train_loader, county_avg, optimizer, epoch, writer=None):
    print("\n---------------------")
    print("Epoch ", epoch)
    print("---------------------")
    model.train()
    lr = optimizer.param_groups[0]['lr']
    print("lr =", lr)

    tot_loss = 0.
    tot_batch = len(train_loader)
    all_pred = []
    all_Y = []
    result_dfs = []

    for batch_idx, (X, Y, counties, years) in enumerate(train_loader): # 397
        X, Y, counties, years = X.to(device), Y.to(device), counties.to(device), years.to(device)  # X: [batch size, 5, num features]  Y: [batch size, 5]
        
        # Randomly mask out some data from the end of the last year in the 5-year sequence,
        # to force model to learn how to make early predictions
        X[:, -1, :] = mask_end(X[:, -1, :], counties, county_avg, args, args.train_week_start, args.time_intervals, device)

        # Clear gradients and pass X through model
        optimizer.zero_grad()
        predictions_std = model(X)
        pred = predictions_std * args.stds + args.means
        loss = loss_fn(pred[:, :args.length-1, :], Y[:, :args.length-1, :], args) * args.c1 + \
               loss_fn(pred[:, -1, :], Y[:, -1, :], args) * args.c2

        all_pred.append(pred[:, -1, :])
        all_Y.append(Y[:, -1, :])
        metrics_batch = eval(pred[:, -1, :], Y[:, -1, :], args) # [64, 64]
        # tot_rmse += metrics['rmse']['avg']  # TODO - report individual crop values!
        # tot_r2 += metrics['r2']['avg']
        # if not math.isnan(metrics['corr']['avg']):
        #     tot_corr += metrics['corr']['avg']
        # tot_mae += metrics['mae']['avg']
        # tot_mape += metrics['mape']['avg']

        tot_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % args.check_freq == 0:
            n_batch = batch_idx+1
            print("### batch ", batch_idx)
            # print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
            #     tot_loss/n_batch, tot_rmse/n_batch, tot_r2/n_batch, tot_corr/n_batch, tot_mae/n_batch, tot_mape/n_batch)
            # )
            print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
                loss.item(), metrics_batch['rmse']['avg'], metrics_batch['r2']['avg'], metrics_batch['corr']['avg'], metrics_batch['mae']['avg'], metrics_batch['mape']['avg'])
            )
        
        cur_step = tot_batch * epoch + batch_idx
        n_batch = batch_idx + 1
        if writer is not None:
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("lr", lr, cur_step)
            writer.add_scalar("Train/loss", loss.item(), cur_step)
            writer.add_scalar("Train/rmse", metrics_batch['rmse']['avg'], cur_step)
            writer.add_scalar("Train/r2", metrics_batch['r2']['avg'], cur_step)
            writer.add_scalar("Train/corr", metrics_batch['corr']['avg'], cur_step)
            writer.add_scalar("Train/mae", metrics_batch['mae']['avg'], cur_step)
            writer.add_scalar("Train/mape", metrics_batch['mape']['avg'], cur_step)
            # writer.add_scalar("Train/loss", tot_loss/n_batch, cur_step)
            # writer.add_scalar("Train/rmse", tot_rmse/n_batch, cur_step)
            # writer.add_scalar("Train/r2", tot_r2/n_batch, cur_step)
            # writer.add_scalar("Train/corr", tot_corr/n_batch, cur_step)
            # writer.add_scalar("Train/mae", tot_mae/n_batch, cur_step)
            # writer.add_scalar("Train/mape", tot_mape/n_batch, cur_step)

        # Create a dataframe with true vs. predicted yield (so that we can produce maps later)
        result_df_dict = {"fips": counties.detach().cpu().numpy().astype(int).tolist(),
                          "year": years.detach().cpu().numpy().astype(int).tolist()}
        for i in range(Y.shape[2]):
            output_name = args.output_names[i]
            result_df_dict["predicted_" + output_name] = pred[:, -1, i].detach().cpu().numpy().tolist()
            result_df_dict["true_" + output_name] = Y[:, -1, i].detach().cpu().numpy().tolist()
        result_dfs.append(pd.DataFrame(result_df_dict))

    results = pd.concat(result_dfs)
    n_batch = batch_idx+1
    # print("\n###### Overall training metrics")
    # print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
    #     tot_loss/n_batch, tot_rmse/n_batch, tot_r2/n_batch, tot_corr/n_batch, tot_mae/n_batch, tot_mape/n_batch)
    # )

    # Calculate stats on all data
    all_pred = torch.cat(all_pred, dim=0)
    all_Y = torch.cat(all_Y, dim=0)
    metrics_all = eval(all_pred, all_Y, args)

    print("\n###### Overall training metrics")
    print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
        tot_loss/n_batch, metrics_all['rmse']['avg'], metrics_all['r2']['avg'], metrics_all['corr']['avg'], metrics_all['mae']['avg'], metrics_all['mape']['avg'])
    )
    return metrics_all, results


def val_epoch(args, model, device, test_loader, county_avg, epoch, mode="Val", writer=None):
    print("********************")
    print("Epoch", epoch, mode)
    print("********************")
    model.eval()
    tot_loss = 0.
    result_dfs = []
    all_pred = []
    all_Y = []
    for batch_idx, (X, Y, counties, years) in enumerate(test_loader):
        X, Y, counties, years = X.to(device), Y.to(device), counties.to(device), years.to(device)

        # To simulate early prediction, mask out data starting from the specified "validation_week"
        X[:, -1, :] = mask_end(X[:, -1, :], counties, county_avg, args, args.validation_week, args.validation_week, device)

        predictions_std = model(X)
        pred = predictions_std * args.stds + args.means

        loss = loss_fn(pred[:, :args.length-1, :], Y[:, :args.length-1, :], args) * args.c1 + \
               loss_fn(pred[:, -1, :], Y[:, -1, :], args) * args.c2
        tot_loss += loss.item()
        all_pred.append(pred[:, -1, :])
        all_Y.append(Y[:, -1, :])

        # metrics = eval(pred[:, -1, :], Y[:, -1, :], args)
        # tot_rmse += metrics['rmse']['avg']
        # tot_r2 += metrics['r2']['avg']
        # tot_corr += metrics['corr']['avg']
        # tot_mae += metrics['mae']['avg']
        # tot_mape += metrics['mape']['avg']

        # Create a dataframe with true vs. predicted yield for each county in the validation
        # year (so that we can produce maps later)
        result_df_dict = {"fips": counties.detach().cpu().numpy().astype(int).tolist(),
                          "year": years.detach().cpu().numpy().astype(int).tolist()}
        for i in range(Y.shape[2]):
            output_name = args.output_names[i]
            result_df_dict["predicted_" + output_name] = pred[:, -1, i].detach().cpu().numpy().tolist()
            result_df_dict["true_" + output_name] = Y[:, -1, i].detach().cpu().numpy().tolist()
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
    for i, output_name in enumerate(args.output_names):
        print("{} r2: {}".format(output_name, metrics_all['r2'][output_name]))

    # # print("###### Overall Validation metrics")
    # print("loss: {}\nrmse: {}\t r2: {}\t corr: {}\n mae: {}\t mape: {}".format(
    #     tot_loss/n_batch, tot_rmse/n_batch, tot_r2/n_batch, tot_corr/n_batch, tot_mae/n_batch, tot_mape/n_batch)
    # )
    print("********************")
    if writer is not None:
        writer.add_scalar("{}/loss".format(mode), tot_loss/n_batch, epoch)
        writer.add_scalar("{}/rmse".format(mode), metrics_all['rmse']['avg'], epoch)
        writer.add_scalar("{}/r2".format(mode), metrics_all['r2']['avg'], epoch)
        writer.add_scalar("{}/corr".format(mode), metrics_all['corr']['avg'], epoch)
        writer.add_scalar("{}/mae".format(mode), metrics_all['mae']['avg'], epoch)
        writer.add_scalar("{}/mape".format(mode), metrics_all['mape']['avg'], epoch)

    return metrics_all, results

def train(args):
    print('reading npy...')
    np.random.seed(args.seed) # set the random seed of numpy
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

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

    X_train, Y_train, counties_train, years_train, X_val, Y_val, counties_val, years_val, X_test, Y_test, counties_test, years_test, county_avg = get_X_Y(data, args, device)

    # Create Tensors, datasets, dataloaders
    X_train, X_val, X_test = torch.Tensor(X_train), torch.Tensor(X_val), torch.Tensor(X_test)
    Y_train, Y_val, Y_test = torch.Tensor(Y_train), torch.Tensor(Y_val), torch.Tensor(Y_test)
    counties_train, counties_val, counties_test = torch.Tensor(counties_train), torch.Tensor(counties_val), torch.Tensor(counties_test)
    years_train, years_val, years_test = torch.Tensor(years_train), torch.Tensor(years_val), torch.Tensor(years_test)
    print("train:", X_train.shape, Y_train.shape, counties_train.shape, years_train.shape)
    print("val:", X_val.shape, Y_val.shape)
    print("test:", X_test.shape, Y_test.shape)

    train_dataset = TensorDataset(X_train, Y_train, counties_train, years_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_dataset = TensorDataset(X_val, Y_val, counties_val, years_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_dataset = TensorDataset(X_test, Y_test, counties_test, years_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    n_train = len(X_train)

    param_setting = "{}_bs-{}_lr-{}_maxepoch-{}_sche-{}_T0-{}_testyear-{}_trainweekstart-{}_len-{}_seed-{}".format(
        args.model, args.batch_size, args.learning_rate, args.max_epoch, args.scheduler, args.T0, args.test_year, args.train_week_start, args.length, args.seed)
    if args.share_conv_parameters:
        param_setting += "_share-params"
    if args.no_management:
        param_setting += "_no-management"
    elif args.combine_weather_and_management:
        param_setting += "_combine-w-m"


    # Directories to store TensorBoard summary, model params, and results
    summary_dir = 'summary/{}/{}/{}'.format(args.dataset, args.test_year, param_setting)
    model_dir = 'model/{}/{}/{}'.format(args.dataset, args.test_year, param_setting)
    results_dir = 'results/{}/{}/{}'.format(args.dataset, args.test_year, param_setting)
    build_path(summary_dir)
    build_path(model_dir)
    build_path(results_dir)
    writer = SummaryWriter(log_dir=summary_dir)

    # Summary csv file of all results. Create this if it doesn't exist
    results_summary_file = os.path.join('results/results_summary.csv')
    if not os.path.isfile(results_summary_file):
        with open(results_summary_file, mode='w') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['dataset', 'model', 'git_commit', 'command', 'val_year', 'val_rmse', 'val_r2', 'val_corr', 'test_year', 'test_rmse', 'test_r2', 'test_corr', 'path_to_model'])

    #building the model 
    print('building network...')
    if args.model == "cnn_rnn":
        model = CNN_RNN(args).to(device)
    elif args.model == "lstm" or args.model == "gru":
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
    best_model_path = ""

    # Store RMSE/R2 at each epoch, for plotting later
    train_rmse_list, val_rmse_list, test_rmse_list, train_r2_list, val_r2_list, test_r2_list = [], [], [], [], [], []

    # Train/validate/test
    for epoch in range(args.max_epoch):
        train_metrics, train_results = train_epoch(args, model, device, train_loader, county_avg, optimizer, epoch, writer)
        val_metrics, val_results = val_epoch(args, model, device, val_loader, county_avg, epoch, "Val", writer)
        test_metrics, test_results = val_epoch(args, model, device, test_loader, county_avg, epoch, "Test", writer)

        # Record epoch metrics in list
        val_rmse = val_metrics['rmse']['avg']
        train_rmse_list.append(train_metrics['rmse']['avg'])
        val_rmse_list.append(val_metrics['rmse']['avg'])
        test_rmse_list.append(test_metrics['rmse']['avg'])
        train_r2_list.append(train_metrics['r2']['avg'])
        val_r2_list.append(val_metrics['r2']['avg'])
        test_r2_list.append(test_metrics['r2']['avg'])

        # Only update metrics if validation RMSE improved
        if val_rmse < best_val_rmse:
            update_metrics(val_metrics['rmse']['avg'], val_metrics['r2']['avg'], val_metrics['corr']['avg'], mode="Val")
            update_metrics(test_metrics['rmse']['avg'], test_metrics['r2']['avg'], test_metrics['corr']['avg'], mode="Test")
            best_val_rmse = val_rmse

            # Save model to file
            best_model_path = model_dir + '/model-' + str(epoch)
            torch.save(model.state_dict(), best_model_path)
            print('save model to', best_model_path)
            print('results file', os.path.join(results_dir, "val_results.csv"))
            
            # Save raw results (true and predicted labels) to files
            train_results.to_csv(os.path.join(results_dir, "train_results.csv"), index=False)
            val_results.to_csv(os.path.join(results_dir, "val_results.csv"), index=False)
            test_results.to_csv(os.path.join(results_dir, "test_results.csv"), index=False)

            # Plot results
            visualization_utils.plot_true_vs_predicted(train_results[train_results["year"] == 1993], args.output_names,  "1993_train", results_dir)
            visualization_utils.plot_true_vs_predicted(val_results, args.output_names, str(args.test_year - 1) + "_val", results_dir)
            visualization_utils.plot_true_vs_predicted(test_results, args.output_names, str(args.test_year) + "_test", results_dir)


        print("BEST Val | rmse: {}, r2: {}, corr: {}".format(best_val['rmse'], best_val['r2'], best_val['corr']))
        print("BEST Test | rmse: {}, r2: {}, corr: {}".format(best_test['rmse'], best_test['r2'], best_test['corr']))
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
                             str(args.test_year), best_test['rmse'], best_test['r2'], best_test['corr'], best_model_path])

    # Record Git commit and command used, along with final metrics
    with open(os.path.join(results_dir, "summary.txt"), 'w') as f:
        f.write("Algorithm: " + args.model + "\n")
        f.write("Dataset: " + args.dataset + "\n")
        f.write("Git commit: " + git_commit + "\n")
        f.write("Command: " + command_string + "\n")
        f.write("Final model path: " + best_model_path + "\n\n")
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
