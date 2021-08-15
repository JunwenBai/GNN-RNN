import os
import pickle
import torch
import numpy as np
import sys
sys.path.append('../cnn-rnn')  # HACK
import subprocess
import visualization_utils
import warnings
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def build_path(path):
    path_levels = path.split('/')
    cur_path = ""
    for path_seg in path_levels:
        if len(cur_path):
            cur_path = cur_path + "/" + path_seg
        else:
            cur_path = path_seg
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)

# Compute metrics for a single variable.
def compute_metrics(pred_i, Y_i):
    metrics = {}
    # RMSE
    metrics['rmse'] = np.sqrt(np.mean((pred_i-Y_i)**2))
    # R2
    metrics['r2'] = r2_score(Y_i, pred_i)
    # corr
    if np.all(Y_i == Y_i[0]) or np.all(pred_i == pred_i[0]):  # If all predictions are the same, calculating correlation produces an error, so just set to 0
        metrics['corr'] = 0
    else:
        metrics['corr'] = np.corrcoef(Y_i, pred_i)[0, 1]
    # MAE
    metrics['mae'] = MAE(Y_i, pred_i)
    # MSE
    metrics['mse']= np.mean((pred_i-Y_i)**2)
    # MAPE
    metrics['mape'] = np.mean(np.abs((Y_i - pred_i) / (Y_i + 1e-5)))
    return metrics


# For each row in X, randomly choose a week between min_week and max_week (inclusive),
# where weeks are indexed starting from 1.
# If we want to fix a week, simply make min_week and max_week equal.
# Zero out all features AFTER this week.
# TODO - to save time, this could be put inside the DataLoader code
def mask_end(X, counties, county_avg, args, min_week, max_week, device):
    # If min_week is time_intervals, we're not masking any data, so
    # just return the original X
    if min_week == args.time_intervals:
        return X

    n_w = args.time_intervals*args.num_weather_vars  # Original: 52*6, new: 52*23
    n_m = args.time_intervals*args.num_management_vars
    num_vars = n_m + n_w
    batch_size = X.shape[0]

    # Random boolean Tensor: True if we should mask the example, False otherwise
    examples_to_mask = (torch.rand((batch_size), device=device, dtype=float) <= args.mask_prob)

    # Create mask which is True (1) for features we want to replace/hide -
    # e.g. features after the current week. It's False (0) for features
    # up to (and including) the current week. Index is 1 based.
    # Initialize the mask to all 0. Then for the examples we want to mask,
    # set weeks after the "current week" to 1.
    mask = torch.zeros((batch_size, num_vars // args.time_intervals, args.time_intervals), dtype=bool, device=device)
    if min_week == max_week:
        mask[examples_to_mask, :, min_week:] = 1
    else:
        weeks = np.random.randint(min_week, max_week+1, size=batch_size)
        for i in range(batch_size):
            if examples_to_mask[i]:
                mask[i, :, weeks[i]:] = 1

    # Get historical average features for each county
    if args.mask_value == "county_avg":
        county_avg_matrix = torch.empty((batch_size, num_vars), device=device)
        for i in range(batch_size):
            county = counties[i].item()
            county_avg_matrix[i] = county_avg[county][:n_w+n_m]  # Only include time-dependent weather and management features

    # "Flatten" mask. Then update all indices where the mask is 1, and replace them with the county average values or 0.
    mask = mask.reshape((batch_size, num_vars))
    if args.mask_value == "zero":
        X[:, :n_w+n_m][mask] = 0
    elif args.mask_value == "county_avg":
        X[:, :n_w+n_m][mask] = county_avg_matrix[mask]

    return X


# # For each row in X, randomly choose a week between min_week and max_week (inclusive),
# # where weeks are indexed starting from 1.
# # If we want to fix a week, simply make min_week and max_week equal.
# # Zero out all features AFTER this week.
# # TODO - to save time, this could be put inside the DataLoader code
# def mask_end(X, args, min_week, max_week):
#     # If min_week is time_intervals, we're not masking any data, so
#     # just return the original X
#     if min_week == args.time_intervals:
#         return X

#     n_w = args.time_intervals*args.num_weather_vars  # Original: 52*6, new: 52*23
#     n_m = args.time_intervals*args.num_management_vars
#     num_vars = n_m + n_w
#     batch_size = X.shape[0]

#     # Create mask which is 1 for features up to (and incluing) the current
#     # week, and 0 afterwards. Index is 1 based
#     mask = np.zeros((batch_size, num_vars // args.time_intervals, args.time_intervals))
#     if min_week == max_week:
#         mask[:, :, :min_week] = 1
#     else:
#         weeks = np.random.randint(min_week, max_week+1, size=batch_size)
#         for i in range(batch_size):
#             mask[i, :, :weeks[i]] = 1

#     # "Flatten" mask, and then multiply each feature vector by the mask. The
#     # effect is to zero out all features after the chosen week.
#     mask = mask.reshape((batch_size, num_vars))
#     X[:, :n_w+n_m] = X[:, :n_w+n_m] * mask
#     return X

# If remove_nan_Y is true, we remove all rows where ANY outputs are NaN
def get_X_Y(data, args, device, remove_nan_Y=False):
    if args.data_dir == "soybean_data_full.npz":
        # Old dataset (given from CNN-RNN paper)
        counties = data[:, 0].astype(int)
        years = data[:, 1].astype(int)
        Y = data[:, 2:3]
        X = data[:, 3:]
    else:
        # Our dataset
        print("Initially data", data.shape)
        counties_all = data[:, 0].astype(int)
        years_all = data[:, 1].astype(int)

        # Only include years up to the test year.
        # Exclude county 25019 (Nantucket County) since it has no NLDAS data.
        data = data[(years_all <= args.test_year) & (counties_all != 25019)]
        print("After filtering", data.shape)
        counties = data[:, 0].astype(int)
        years = data[:, 1].astype(int)
        Y = data[:, [args.output_idx]]
        X = data[:, 8:]

    print("get_X_Y")
    print("X shape", X.shape)
    print("Y shape", Y.shape)

    # Compute the unique years and counties
    min_year = int(min(years))
    max_year = int(max(years))
    county_set = sorted(list(set(counties)))

    # Compute average yield of each year (to detect underlying yearly trends)
    avg_Y = {}
    avg_Y_lst = []
    for year in range(min_year, max_year+1):
        avg_Y[year] = np.nanmean(Y[years == year, :], axis=0)
        avg_Y_lst.append(avg_Y[year])
    avg_Y[min_year-1] = avg_Y[min_year]

    # For each row in X, get the average yield of the previous year, and add this as a column of X
    Ybar = []
    for year in years:
        Ybar.append(avg_Y[year-1])
    Ybar = np.array(Ybar) #.reshape(-1, 1) - removed this because we may have multiple outputs
    X = np.concatenate((X, Ybar), axis=1)

    # Compute the mean and standard deviation of each feature (over non-NaN values), over the train years.
    # We will use these to standardize the features later.
    # If the feature is NaN everywhere, return 0 for mean/std (the "nan_to_num" function does this)
    known_years = data[:, 1] < (args.test_year-1)
    with warnings.catch_warnings():  # Supress warning about columns being NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)
        X_mean = np.nanmean(X[known_years], axis=0, keepdims=True)
        X_std = np.nanstd(X[known_years], axis=0, keepdims=True)

        # HACK: If the standard deviation of a feature on train set is 0 (e.g. all
        # values are the same), then standardizing anything apart from that value
        # will produce a super extreme z-score. So just replace those standard 
        # deviations with 1. Also, if the mean/std are NaN, replace with 0 and 1. 
        X_std[(X_std < 1e-6) | np.isnan(X_std)] = 1
        X_mean[np.isnan(X_mean)] = 0

    # Standardize each feature (column of X)
    X = (X - X_mean) / (X_std + 1e-10)

    # # Check for extreme values in X (after standardization)
    # indices = np.argwhere((X > 100) | (X < -100))
    # for i in range(indices.shape[0]):
    #     row, col = indices[i, 0], indices[i, 1]
    #     print("!!! Extreme value indices", row, col + 7, "- yr", years[row])

    # For now, replace all NA with 0.
    X = np.nan_to_num(X)

    # # Fill in gaps for progress data. TODO - refactor this into function
    # assert ((args.progress_indices[-1] + 1 - args.progress_indices[0]) % args.time_intervals == 0)
    # # Loop through each example
    # for i in range(X.shape[0]):
    #     # Loop through each progress variable (which is itself a range of "args.time_intervals" variables)
    #     for progress_var_start in range(args.progress_indices[0], args.progress_indices[-1] + 1, args.time_intervals):
    #         current_progress = 0
    #         for progress_idx in range(progress_var_start, progress_var_start + args.time_intervals):
    #             if np.isnan(X[i, progress_idx]):
    #                 X[i, progress_idx] = current_progress
    #             else:
    #                 current_progress = X[i, progress_idx]

    # Compute average of each output
    Y_mean = np.nanmean(Y[known_years], axis=0, keepdims=True)
    Y_std = np.nanstd(Y[known_years], axis=0, keepdims=True)
    args.means = torch.tensor(Y_mean, device=device)
    args.stds = torch.tensor(Y_std, device=device)
    print("Y (output) means", args.means, "stds", args.stds)

    # Create dictionaries mapping from county to features
    county_dict = {}  # Features per county, over TRAIN years
    for county in county_set:
        county_dict[county] = []
    for county, year, x, y in zip(counties, years, X, Y):
        if year < args.test_year - 1:
            county_dict[county].append(x)

    # Compute average features per COUNTY (that can be used when we're doing early prediction
    # and don't have complete weather data)
    county_avg = {}
    for county in county_set:
        county_dict[county] = np.array(county_dict[county])
        county_avg[county] = torch.tensor(np.nanmean(county_dict[county], axis=0)).to(device)

    # If requested, remove rows where ANY output is NaN
    if remove_nan_Y:
        not_na = ~(np.isnan(Y).any(axis=1))
        X = X[not_na]
        Y = Y[not_na]
        counties = counties[not_na]
        years = years[not_na]

    # Split into train/val/test
    X_train, X_val, X_test = [], [], []
    Y_train, Y_val, Y_test = [], [], []
    counties_train, counties_val, counties_test = [], [], []
    years_train, years_val, years_test = [], [], []
    for x_seq, y_seq, county, year in zip(X, Y, counties, years):
        if year == args.test_year:
            X_test.append(x_seq)
            Y_test.append(y_seq)
            counties_test.append(county)
            years_test.append(year)
        elif year == args.test_year - 1:
            X_val.append(x_seq)
            Y_val.append(y_seq)
            counties_val.append(county)
            years_val.append(year)
        else:
            X_train.append(x_seq)
            Y_train.append(y_seq)
            counties_train.append(county)
            years_train.append(year)
    X_train, X_val, X_test = np.array(X_train), np.array(X_val), np.array(X_test)
    Y_train, Y_val, Y_test = np.array(Y_train), np.array(Y_val), np.array(Y_test)
    counties_train, counties_val, counties_test = np.array(counties_train), np.array(counties_val), np.array(counties_test)
    years_train, years_val, years_test = np.array(years_train), np.array(years_val), np.array(years_test)

    return X_train, Y_train, counties_train, years_train, X_val, Y_val, counties_val, years_val, X_test, Y_test, counties_test, years_test, county_avg  # X_train, Y_train, X_val, Y_val, X_test, Y_test
