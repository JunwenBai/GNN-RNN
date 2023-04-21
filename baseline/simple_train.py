"""
Trains a simple feature-based (scikit-learn) model to predict crop yield.
Does a grid search over hyperparameters (running each configuration 3 times),
selects the configuration that performs best on the validation set, and reports
results for the 3 runs.
"""

import csv
import math
import numpy as np
import os
import sys
import argparse
sys.path.append('../cnn-rnn')  # HACK
import visualization_utils
from baseline_utils import build_path, get_X_Y, get_git_revision_hash, compute_metrics, mask_end
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor

# Index of the yield variable for each variable
OUTPUT_INDICES = {'corn': 2,
                  'upland_cotton': 3,
                  'sorghum': 4,
                  'soybeans': 5,
                  'spring_wheat': 6,
                  'winter_wheat': 7}

# Indices of the progress variables for each crop type in the X array.
PROGRESS_INDICES_DAILY = {'corn': list(range(8403-8, 13148-8)),
                          'upland_cotton': list(range(13148-8, 17893-8)),
                          'sorghum': list(range(17893-8, 22638-8)),
                          'soybeans': list(range(22638-8, 28113-8)),
                          'spring_wheat': list(range(32858-8, 37603-8)),
                          'winter_wheat': list(range(37603-8, 43443-8))}
PROGRESS_INDICES_WEEKLY = {'corn': list(range(1204-8, 1880-8)),
                          'upland_cotton': list(range(1880-8, 2556-8)),
                          'sorghum': list(range(2556-8, 3232-8)),
                          'soybeans': list(range(3232-8, 4012-8)),
                          'spring_wheat': list(range(4688-8, 5364-8)),
                          'winter_wheat': list(range(5364-8, 6196-8))}
SEEDS = [0, 1, 2]
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', "--dataset", default='soybean', type=str, help='dataset name')
parser.add_argument('-dd', "--data_dir", default='./data/soybean_data.npz', type=str, help='The data directory')
parser.add_argument('-test_year', "--test_year", default=2017, type=int, help='test year')
parser.add_argument('-model', "--model", type=str, choices=['ridge', 'lasso', 'gradient_boosting_regressor', 'mlp'], help="Regression algorithm to use")
# parser.add_argument('-input_scaling', "--input_scaling", default="standard_scaler", choices=['none', 'standard_scaler', 'min_max_scaler'], help="How to scale input features")
parser.add_argument('-standardize_outputs', "--standardize_outputs", default=False, action='store_true', help="whether to standardize the output variables")

# Added: dataset params
parser.add_argument('-crop_type', '--crop_type', choices=["corn", "cotton", "sorghum", "soybeans", "spring_wheat", "winter_wheat"])
parser.add_argument('-num_weather_vars', "--num_weather_vars", default=23, type=int, help='Number of daily weather vars, from PRISM and NLDAS. There were 6 in the CNN-RNN paper, 23 in our new dataset.')
parser.add_argument('-num_management_vars', "--num_management_vars", default=96, type=int, help='Number of weekly management (crop progress) variables. There are 96 in our new dataset.')
parser.add_argument('-num_soil_vars', "--num_soil_vars", default=20, type=int, help='Number of depth-dependent soil vars, from gSSURGO. There were 10 in the CNN-RNN paper, 20 in our new dataset.')
parser.add_argument('-num_extra_vars', "--num_extra_vars", default=6, type=int, help='Number of extra vars, e.g. gSSURGO variables that are not dependent on depth. There were 5 in the CNN-RNN paper, 6 in our new dataset.')
parser.add_argument('-soil_depths', "--soil_depths", default=6, type=int, help='Number of depths in the gSSURGO dataset. There were 10 in the CNN-RNN paper, 10 in our new dataset.')
parser.add_argument('-no_management', "--no_management", default=False, action='store_true', help='Whether to completely ignore management (crop progress/condition) data')
parser.add_argument('-train_week_start', "--train_week_start", default=52, type=int, help="For each train example, pick a random week between this week and the end (inclusive, 1-based indexing), and mask out data after the random week. Set to args.time_intervals for no masking.")
parser.add_argument('-validation_week', "--validation_week", default=52, type=int, help="Mask out data starting from this week during validation. Set to args.time_intervals for no masking.")

args = parser.parse_args()

# Set number of time intervals per year (365 for daily dataset, 52 for weekly dataset)
args.output_idx = OUTPUT_INDICES[args.crop_type]
args.output_names = [args.crop_type]
if "daily" in args.data_dir:
    args.time_intervals = 365
    args.progress_indices = PROGRESS_INDICES_DAILY[args.crop_type]
elif "weekly" in args.data_dir or args.data_dir.endswith(".npy") or args.data_dir == "soybean_data_full.npz":  # A bit of a hack to accomodate the previous paper's CNN-RNN dataset, which is weekly
    args.time_intervals = 52
    args.progress_indices = PROGRESS_INDICES_WEEKLY[args.crop_type]
else:
    raise ValueError("Data file must contain the string 'daily' or 'weekly'")


# Load data
print(args.data_dir)
if args.data_dir.endswith(".npz"):
    raw_data = np.load(args.data_dir) #load data from the data_dir
    data = raw_data['data']
elif args.data_dir.endswith(".npy"):
    data = np.load(args.data_dir)  #, dtype=float, delimiter=',')
elif args.data_dir.endswith(".csv"):
    data = np.genfromtxt(args.data_dir, dtype=float, delimiter=',')
else:
    raise ValueError("--data_dir argument must end in .npz, .npy, or .csv")
print("Raw data shape", data.shape)    
X_train, Y_train, counties_train, years_train, X_val, Y_val, counties_val, years_val, X_test, Y_test, counties_test, years_test, county_avg = get_X_Y(data, args, device="cpu", remove_nan_Y=True)
Y_train, Y_val, Y_test = Y_train.flatten(), Y_val.flatten(), Y_test.flatten()
print("After filtering nan", X_train.shape, Y_train.shape)

# Randomly mask out some data from the end of the year to learn how to make early predictions
X_train = mask_end(X_train, counties_train, county_avg, args, args.train_week_start, args.time_intervals, "cpu")
X_val = mask_end(X_val, counties_val, county_avg, args, args.validation_week, args.validation_week, "cpu")
X_test = mask_end(X_test, counties_test, county_avg, args, args.validation_week, args.validation_week, "cpu")

# Set up results directory
param_setting = "{}_testyear-{}".format(
    args.model, args.test_year)
if args.no_management:
    param_setting += "_no-management"
results_dir = 'results/{}/{}'.format(args.dataset, param_setting)
build_path(results_dir)

# Summary csv file of all results FOR THIS RUN. Create this if it doesn't exist.
results_summary_file = os.path.join(results_dir, "results_summary.csv")
if not os.path.isfile(results_summary_file):
    with open(results_summary_file, mode='w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['dataset', 'model', 'git_commit', 'command', 'params', 'val_year', 'val_rmse', 'val_r2', 'val_corr', 'test_year', 'test_rmse', 'test_r2', 'test_corr'])

# # Scale input data if necessary. Note - get_X_Y already does scaling, perhaps remove this completely?
# if args.input_scaling != "none":
#     if args.input_scaling == "standard_scaler":
#         scaler = StandardScaler()
#     elif args.input_scaling == "min_max_scaler":
#         scaler = MinMaxScaler()
#     else:
#         raise ValueError("Unsupported scaler type")
#     scaler.fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_val = scaler.transform(X_val)
#     X_test = scaler.transform(X_test)

# Scale output data if necessary
args.means = args.means.numpy().flatten()
args.stds = args.stds.numpy().flatten()
if not args.standardize_outputs:  # If we don't standardize, subtract 0 and divide by 1 (e.g. don't change anything)
    args.means = np.zeros(args.means.shape)
    args.stds = np.ones(args.stds.shape)

Y_train_std = (Y_train - args.means) / args.stds
Y_val_std = (Y_val - args.means) / args.stds
Y_test_std = (Y_test - args.means) / args.stds
print("Finished preparing data")

# Fit models (with various hyperparam settings). This is basically a manual implementation of grid
# search to fit with our train/validation/test split. (SciKit-Learn's implementation only supports 
# cross-validation.)
regression_models = dict()
if args.model == 'ridge':
    alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]  #[0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for alpha in alphas:
        models = []
        for random_state in SEEDS:
            regression_model = Ridge(alpha=alpha, random_state=random_state).fit(X_train, Y_train_std) # HuberRegressor(alpha=alpha, max_iter=1000).fit(X_train, Y_train)
            models.append(regression_model)
        param_string = 'alpha=' + str(alpha)
        print("Params", param_string)
        regression_models[param_string] = models
elif args.model == 'lasso':
    alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000]  #[0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for alpha in alphas:
        models = []
        for random_state in SEEDS:
            regression_model = Lasso(alpha=alpha, random_state=random_state).fit(X_train, Y_train_std) # HuberRegressor(alpha=alpha, max_iter=1000).fit(X_train, Y_train)
            models.append(regression_model)
        param_string = 'alpha=' + str(alpha)
        print("Params", param_string)
        regression_models[param_string] = models
elif args.model == 'gradient_boosting_regressor':
    max_iter_values = [100, 300, 1000] #
    max_depth_values = [2, 3, None]
    # n_estimator_values = [700, 1000]
    # learning_rates = [0.01, 0.1, 0.5]
    # max_depths = [1, 10]
    for max_iter in max_iter_values:
        for max_depth in max_depth_values:
            models = []
            for random_state in SEEDS:
                regression_model = HistGradientBoostingRegressor(max_iter=max_iter, max_depth=max_depth, learning_rate=0.1, random_state=random_state).fit(X_train, Y_train_std)
                models.append(regression_model)
            param_string = 'max_iter=' + str(max_iter) + ', max_depth=' + str(max_depth)
            print(param_string)
            regression_models[param_string] = models
elif args.model == 'mlp':
    hidden_layer_sizes = [(100, 100)] # [(100,), (20, 20), (100, 100), (100, 100, 100)] #[(100, 100)] # 
    learning_rate_inits =  [1e-3, 1e-4]  #[1e-2, 1e-3, 1e-4]  # [1e-3] #
    max_iter = 10000
    for hidden_layer_size in hidden_layer_sizes:
        for learning_rate_init in learning_rate_inits:
            models = []
            for random_state in SEEDS:
                regression_model = MLPRegressor(hidden_layer_sizes=hidden_layer_size, learning_rate_init=learning_rate_init, max_iter=max_iter, random_state=random_state).fit(X_train, Y_train_std)
                models.append(regression_model)
            param_string = 'hidden_layer_sizes=' + str(hidden_layer_size) + ', learning_rate_init=' + str(learning_rate_init)
            print(param_string)
            regression_models[param_string] = models
else:
    raise ValueError("Unsupported method")

# print('Coefficients', regression_model.coef_)
best_loss = float('inf')
best_params = 'N/A'

# Loop through all hyperparameter settings we trained models for, and compute
# loss on the validation set
average_losses_val = []
for params, models in regression_models.items():
    losses_val = []
    for model in models:
        predictions_val_std = model.predict(X_val)
        loss_val = math.sqrt(mean_squared_error(Y_val_std, predictions_val_std))
        losses_val.append(loss_val)
    average_loss_val = sum(losses_val) / len(losses_val)
    print(params + ': avg val loss', round(average_loss_val, 4))
    if average_loss_val < best_loss:
        best_loss = average_loss_val
        best_params = params

# For the best hyperparameter setting, loop through the models and compute results again.
# Record to csv.
git_commit = get_git_revision_hash()
command_string = " ".join(sys.argv)
with open(results_summary_file, mode='a+') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for idx, model in enumerate(regression_models[best_params]):
        predictions_val_std = model.predict(X_val)
        predictions_test_std = model.predict(X_test)
        val_metrics = compute_metrics(predictions_val_std, Y_val_std)
        test_metrics = compute_metrics(predictions_test_std, Y_test_std)
        csv_writer.writerow([args.dataset, args.model, git_commit, command_string, best_params + ", seed=" + str(SEEDS[idx]),
                             str(args.test_year - 1), val_metrics['rmse'], val_metrics['r2'], val_metrics['corr'],
                             str(args.test_year), test_metrics['rmse'], test_metrics['r2'], test_metrics['corr']])

        print("=============== Seed", SEEDS[idx], "=================")
        print("VAL: \trmse: {}\t r2: {}\t corr: {}".format(
            val_metrics['rmse'], val_metrics['r2'], val_metrics['corr'])
        )
        print("TEST: \trmse: {}\t r2: {}\t corr: {}".format(
            test_metrics['rmse'], test_metrics['r2'], test_metrics['corr'])
        ) 

        # For the first model, visualize the results.
        if idx == 0:
            output_name = args.output_names[0]
            val_results_dict = {}
            val_results_dict["predicted_" + output_name] = ((predictions_val_std * args.stds[0]) + args.means[0]).tolist()
            val_results_dict["true_" + output_name] = ((Y_val_std * args.stds[0]) + args.means[0]).tolist()
            val_results_dict["fips"] = counties_val.tolist()
            val_results = pd.DataFrame(val_results_dict)
            visualization_utils.plot_true_vs_predicted(val_results, args.output_names, str(args.test_year - 1) + "_val", results_dir)
            test_results_dict = {}
            test_results_dict["predicted_" + output_name] = ((predictions_test_std * args.stds[0]) + args.means[0]).tolist()
            test_results_dict["true_" + output_name] = ((Y_test_std * args.stds[0]) + args.means[0]).tolist()
            test_results_dict["fips"] = counties_test.tolist()
            test_results = pd.DataFrame(test_results_dict)
            visualization_utils.plot_true_vs_predicted(test_results, args.output_names, str(args.test_year) + "_test", results_dir)
