import argparse
from train import train
from test import test
from test_predictions_over_time import test_predictions_over_time


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

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', "--dataset", default='soybean', type=str, help='dataset name')
parser.add_argument('-adj', "--us_adj_file", default='', type=str, help='adjacency file')
parser.add_argument('-fid_map', "--crop_id_to_fid", default='', type=str, help='crop id to fid file')
parser.add_argument('-cp', "--checkpoint_path", default='./ckpt', type=str, help='The path to a checkpoint from which to fine-tune.')

parser.add_argument('-dd', "--data_dir", default='./data/soybean_data.npz', type=str, help='The data directory')

parser.add_argument('-seed', "--seed", default=0, type=int, help='seed')
parser.add_argument('-bs', "--batch_size", default=64, type=int, help='the number of data points in one minibatch')
#parser.add_argument('-tbs', "--test_batch_size", default=64, type=int, help='the number of data points in one testing or validation batch')
parser.add_argument('-lr', "--learning_rate", default=1e-3, type=float, help='initial learning rate')
parser.add_argument('-epoch', "--max_epoch", default=200, type=int, help='max epoch to train')
parser.add_argument('-wd', "--weight_decay", default=1e-5, type=float, help='weight decay rate')
parser.add_argument('-lrdr', "--lr_decay_ratio", default=0.5, type=float, help='The decay ratio of learning rate')

parser.add_argument('-se', "--save_epoch", default=1, type=int, help='epochs to save the checkpoint of the model')
parser.add_argument('-max_keep', "--max_keep", default=3, type=int, help='maximum number of saved model')
parser.add_argument('-check_freq', "--check_freq", default=50, type=int, help='checking frequency')

parser.add_argument('-eta_min', "--eta_min", default=1e-5, type=float, help='minimum lr')
parser.add_argument('-T0', "--T0", default=50, type=int, help='optimizer T0')
parser.add_argument('-T_mult', "--T_mult", default=2, type=int, help='optimizer T_multi')
parser.add_argument('-patience', "--patience", default=1, type=int, help='optimizer patience')
parser.add_argument('-test_year', "--test_year", default=2017, type=int, help='test year')
parser.add_argument('-length', "--length", default=5, type=int, help='length of sequence (# years)')
parser.add_argument('-z_dim', "--z_dim", default=64, type=int, help='hidden units in RNN')

parser.add_argument('-keep_prob', "--keep_prob", default=1.0, type=float, help='1.-drop out rate')
parser.add_argument('-c1', "--c1", default=0.0, type=float, help='c1')
parser.add_argument('-c2', "--c2", default=1.0, type=float, help='c2')
parser.add_argument('-mode', "--mode", type=str, help='training/test mode')
parser.add_argument('-model', "--model", type=str, help='model type', choices=['cnn_rnn', 'lstm', 'gru'])
parser.add_argument('-sche', "--scheduler", default='cosine', choices=['cosine', 'step', 'plateau', 'exp'], help='lr scheduler')
parser.add_argument('-exp_gamma', "--exp_gamma", default=0.98, type=float, help='exp lr decay gamma')

parser.add_argument('-clip_grad', "--clip_grad", default=10.0, type=float, help='clip_grad')

# Added: dataset params
# parser.add_argument('-data_dir', "--data_dir", type=str, default="/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly_1981-2020.csv")
# parser.add_argument('-data_file', "--data_file", type=str, default="combined_dataset_weekly_1981-2020.csv")
# parser.add_argument('-num_outputs', "--num_outputs", default=6, type=int)
parser.add_argument('-crop_type', '--crop_type', choices=["corn", "upland_cotton", "sorghum", "soybeans", "spring_wheat", "winter_wheat"])
parser.add_argument('-num_weather_vars', "--num_weather_vars", default=23, type=int, help='Number of daily weather vars, from PRISM and NLDAS. There were 6 in the CNN-RNN paper, 23 in our new dataset.')
parser.add_argument('-num_management_vars', "--num_management_vars", default=96, type=int, help='Number of weekly management (crop progress) variables. There are 96 in our new dataset.')
parser.add_argument('-num_soil_vars', "--num_soil_vars", default=20, type=int, help='Number of depth-dependent soil vars, from gSSURGO. There were 10 in the CNN-RNN paper, 20 in our new dataset.')
parser.add_argument('-num_extra_vars', "--num_extra_vars", default=6, type=int, help='Number of extra vars, e.g. gSSURGO variables that are not dependent on depth. There were 5 in the CNN-RNN paper, 6 in our new dataset.')
parser.add_argument('-soil_depths', "--soil_depths", default=6, type=int, help='Number of depths in the gSSURGO dataset. There were 10 in the CNN-RNN paper, 10 in our new dataset.')
parser.add_argument('-share_conv_params', "--share_conv_parameters", default=False, action='store_true', help='Whether weather variables should share the same conv parameters or not')
parser.add_argument('-combine_weather_and_management', "--combine_weather_and_management", default=False, action='store_true', help='Whether weather variables should share the same conv parameters or not')
parser.add_argument('-no_management', "--no_management", default=False, action='store_true', help='Whether to completely ignore management (crop progress/condition) data')
parser.add_argument('-train_week_start', "--train_week_start", default=52, type=int, help="For each train example, pick a random week between this week and the end (inclusive, 1-based indexing), and mask out data after the random week. Set to args.time_intervals for no masking.")
parser.add_argument('-validation_week', "--validation_week", default=52, type=int, help="Mask out data starting from this week during validation. Set to args.time_intervals for no masking.")
parser.add_argument('-mask_prob', "--mask_prob", default=1, type=float, help="Probability of masking. 0 means don't mask any data.")
parser.add_argument('-mask_value', "--mask_value", choices=['zero', 'county_avg'], default='zero')

# ONLY used for test_predictions_over_time, if we're plotting predictions over time for a specific county and the test year.
parser.add_argument('-county_to_plot', "--county_to_plot", default=17083, type=int, help='County FIPS to plot (ONLY used for the "test_predictions_over_time" mode).')

args = parser.parse_args()
if args.no_management:
    assert(args.combine_weather_and_management)

if args.crop_type not in args.dataset:
    print("Alert! Did you forget to change the 'crop_type' param? You set 'crop_type' to", args.crop_type, "but 'dataset' to", args.dataset)
    exit(1)

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

print("Time intervals", args.time_intervals)

if __name__ == "__main__":
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'test_predictions_over_time':
        test_predictions_over_time(args)
    else:
        raise ValueError("mode %s is not supported." % args.mode)
    