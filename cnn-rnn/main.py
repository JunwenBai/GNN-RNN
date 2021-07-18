import argparse
from train import train
from test import test
from test_predictions_over_time import test_predictions_over_time


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
parser.add_argument('-length', "--length", default=5, type=int, help='test year')
parser.add_argument('-z_dim', "--z_dim", default=64, type=int, help='hidden units in RNN')

parser.add_argument('-keep_prob', "--keep_prob", default=1.0, type=float, help='1.-drop out rate')
parser.add_argument('-c1', "--c1", default=0.0, type=float, help='c1')
parser.add_argument('-c2', "--c2", default=1.0, type=float, help='c2')
parser.add_argument('-mode', "--mode", type=str, help='training/test mode')
parser.add_argument('-model', "--model", type=str, help='model type')
parser.add_argument('-sche', "--scheduler", default='cosine', choices=['cosine', 'step', 'plateau', 'exp'], help='lr scheduler')
parser.add_argument('-exp_gamma', "--exp_gamma", default=0.98, type=float, help='exp lr decay gamma')

parser.add_argument('-clip_grad', "--clip_grad", default=10.0, type=float, help='clip_grad')

# Added: dataset params
# parser.add_argument('-data_dir', "--data_dir", type=str, default="/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly_1981-2020.csv")
# parser.add_argument('-data_file', "--data_file", type=str, default="combined_dataset_weekly_1981-2020.csv")
# parser.add_argument('-num_outputs', "--num_outputs", default=6, type=int)
parser.add_argument('-crop_type', '--crop_type', choices=["corn", "cotton", "sorghum", "soybeans", "spring_wheat", "winter_wheat"])
parser.add_argument('-num_weather_vars', "--num_weather_vars", default=23, type=int, help='Number of daily weather vars, from PRISM and NLDAS. There were 6 in the CNN-RNN paper, 23 in our new dataset.')
parser.add_argument('-num_management_vars', "--num_management_vars", default=96, type=int, help='Number of management (crop progress) variables. There were 14 in the CNN-RNN paper, ??? in our new dataset.')
parser.add_argument('-num_soil_vars', "--num_soil_vars", default=20, type=int, help='Number of depth-dependent soil vars, from gSSURGO. There were 10 in the CNN-RNN paper, 20 in our new dataset.')
parser.add_argument('-num_extra_vars', "--num_extra_vars", default=5, type=int, help='Number of extra vars, e.g. gSSURGO variables that are not dependent on depth. There were 5 in the CNN-RNN paper, 6 in our new dataset.')
parser.add_argument('-soil_depths', "--soil_depths", default=6, type=int, help='Number of depths in the gSSURGO dataset. There were 10 in the CNN-RNN paper, 10 in our new dataset.')

# ONLY used for test_predictions_over_time, if we're plotting predictions over time for a specific county and the test year.
parser.add_argument('-county_to_plot', "--county_to_plot", default=17083, type=int, help='County FIPS to plot (ONLY used for the "test_predictions_over_time" mode).')


args = parser.parse_args()

# Set number of time intervals per year (365 for daily dataset, 52 for weekly dataset)
if "daily" in args.data_dir:
    args.time_intervals = 365
elif "weekly" in args.data_dir or args.data_dir.endswith(".npy") or args.data_dir.endswith(".npz"):  # A bit of a hack to accomodate the previous paper's CNN-RNN dataset, which is weekly
    args.time_intervals = 52
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
