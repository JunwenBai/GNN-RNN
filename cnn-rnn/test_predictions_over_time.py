
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
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append('./')

# Compute and plot predictions over time for a specific row of X_test.
# "county_test_row" and "county_train_avg" should be 1-D vectors for a specific county/year
# "Y_true" is a 1-D vector, of the labels for that specific row.
def compute_and_plot_predictions(args, model, device, county_test_row, county_train_avg, Y_true, visualizations_dir):
    model.eval()
    
    # Generate a batch for every week (time interval)
    assert(county_test_row.shape[0] == 1)
    batch = np.tile(county_test_row, (args.time_intervals, 1, 1))
    print("Batch shape", batch.shape)


    # For each week i, fill in features after week "i" with county averages.
    for i in range(args.time_intervals):
        # Compute list of feature indices that are from after week "i".
        indices_to_replace = []
        for j in range(args.num_weather_vars):
            indices_to_replace.extend(range(j * args.time_intervals + i + 1, (j+1) * args.time_intervals))
        batch[i, -1, indices_to_replace] = county_train_avg[indices_to_replace]
    batch = torch.Tensor(batch).to(device)

    # Pass this batch through model
    pred = model(batch, "asdfjkl;").detach().cpu().numpy()  # The Y parameter is not used
    pred = pred[:, -1, :]

    # Plot each crop
    week_numbers = np.array(range(1, args.time_intervals + 1))
    for idx, output_name in enumerate(args.output_names):
        plt.plot(week_numbers, pred[:, idx])
        plt.xlabel("Week #")
        plt.ylabel("Predicted yield")
        plt.axhline(y = Y_true[idx], color='r', linestyle='--')  # Dashed line for true value for this year
        plt.title("Predictions over time: " + str(output_name) + ", county " + str(args.county_to_plot) + ", year " + str(args.test_year))
        plt.savefig(os.path.join(visualizations_dir, "predictions_over_time_county_" + str(args.county_to_plot) + "_year_" + str(args.test_year) + "_" + output_name + ".png"))
        plt.close()
        


def test_predictions_over_time(args):
    print('reading npy...')
    np.random.seed(args.seed) # set the random seed of numpy
    torch.manual_seed(args.seed)

    # Read in data
    print(args.data_dir)
    if args.data_dir.endswith(".npz"):
        raw_data = np.load(args.data_dir) #load data from the data_dir
        data = raw_data['data']
        args.output_names = ["soybean"]
    elif args.data_dir.endswith(".npy"):
        data = np.load(args.data_dir)  #, dtype=float, delimiter=',')
        args.output_names = ["corn", "cotton", "sorghum", "soybeans", "spring_wheat", "winter_wheat"]
        print("Data shape", data.shape)    
    elif args.data_dir.endswith(".csv"):
        data = np.genfromtxt(args.data_dir, dtype=float, delimiter=',')
        args.output_names = ["corn", "cotton", "sorghum", "soybeans", "spring_wheat", "winter_wheat"]
        print("Data shape", data.shape)
    else:
        raise ValueError("--data_dir argument must end in .npz, .npy, or .csv")

    X_train, Y_train, counties_train, years_train, X_val, Y_val, counties_val, years_val, X_test, Y_test, counties_test, years_test = get_X_Y(data, args)

    # Compute average features for this county (on train years)
    county_train_rows = X_train[counties_train == args.county_to_plot]
    county_train_avg = np.mean(county_train_rows, axis=(0, 1))

    # Get the features for this county for test year
    county_test_row = X_test[(counties_test == args.county_to_plot) & (years_test == args.test_year)]
    county_test_Y = Y_test[(counties_test == args.county_to_plot) & (years_test == args.test_year)]
    print("County train rows shape", county_train_rows.shape)
    print("County train avg shape", county_train_avg.shape)
    print("County test row shape", county_test_row.shape)
    print("County test Y shape", county_test_Y.shape)

    # Build model 
    if args.model == "cnn_rnn":
        model = CNN_RNN(args).to(device)
    elif args.model == "rnn":
        model = RNN(args).to(device)
    else:
        raise ValueError("model type not supported yet")
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()

    # Compute results directory
    normalized_checkpoint_path = os.path.normpath(args.checkpoint_path)
    normalized_checkpoint_path = normalized_checkpoint_path.split(os.sep)
    visualizations_dir = os.path.join("results", normalized_checkpoint_path[-3], normalized_checkpoint_path[-2])
    print("VIS DIR", visualizations_dir)
    exit(1)
    compute_and_plot_predictions(args, model, device, county_test_row, county_train_avg, np.squeeze(county_test_Y)[-1, :], visualizations_dir)


