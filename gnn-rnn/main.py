import argparse
from train import train
from test import test


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', "--dataset", default='soybean', type=str, help='dataset name')
parser.add_argument('-adj', "--us_adj_file", default='', type=str, help='adjacency file')
parser.add_argument('-fid_map', "--crop_id_to_fid", default='', type=str, help='crop id to fid file')
parser.add_argument('-cp', "--checkpoint_path", default='./ckpt', type=str, help='The path to a checkpoint from which to fine-tune.')

parser.add_argument('-dd', "--data_dir", default='./data/soybean_data.npz', type=str, help='The data directory')

parser.add_argument('-seed', "--seed", default=0, type=int, help='seed')
parser.add_argument('-bs', "--batch_size", default=128, type=int, help='the number of data points in one minibatch')
#parser.add_argument('-tbs', "--test_batch_size", default=64, type=int, help='the number of data points in one testing or validation batch')
parser.add_argument('-lr', "--learning_rate", default=1e-3, type=float, help='initial learning rate')
parser.add_argument('-epoch', "--max_epoch", default=200, type=int, help='max epoch to train')
parser.add_argument('-wd', "--weight_decay", default=1e-5, type=float, help='weight decay rate')
parser.add_argument('-lrdr', "--lr_decay_ratio", default=0.5, type=float, help='The decay ratio of learning rate')

parser.add_argument('-se', "--save_epoch", default=1, type=int, help='epochs to save the checkpoint of the model')
parser.add_argument('-max_keep', "--max_keep", default=3, type=int, help='maximum number of saved model')
parser.add_argument('-check_freq', "--check_freq", default=50, type=int, help='checking frequency')

parser.add_argument('-eta_min', "--eta_min", default=1e-5, type=float, help='minimum lr')
parser.add_argument('-gamma', "--gamma", default=0.5, type=float, help='StepLR decay')
parser.add_argument('-T0', "--T0", default=50, type=int, help='optimizer T0')
parser.add_argument('-sleep', "--sleep", default=50, type=int, help='sleep time')
parser.add_argument('-lrsteps', "--lrsteps", default=50, type=int, help='StepLR steps')
parser.add_argument('-T_mult', "--T_mult", default=2, type=int, help='optimizer T_multi')
parser.add_argument('-patience', "--patience", default=1, type=int, help='optimizer patience')
parser.add_argument('-test_year', "--test_year", default=2018, type=int, help='test year')
parser.add_argument('-length', "--length", default=5, type=int, help='test year')
parser.add_argument('-z_dim', "--z_dim", default=64, type=int, help='hidden units in RNN')

parser.add_argument('-keep_prob', "--keep_prob", default=1.0, type=float, help='1.-drop out rate')
parser.add_argument('-c1', "--c1", default=0.0, type=float, help='c1')
parser.add_argument('-c2', "--c2", default=1.0, type=float, help='c2')
parser.add_argument('-mode', "--mode", type=str, help='training/test mode')
parser.add_argument('-sche', "--scheduler", default='cosine', choices=['cosine', 'step', 'plateau', 'exp', 'const'], help='lr scheduler')
parser.add_argument('-exp_gamma', "--exp_gamma", default=0.98, type=float, help='exp lr decay gamma')

parser.add_argument('-clip_grad', "--clip_grad", default=10.0, type=float, help='clip_grad')

# GNN specific
parser.add_argument('-n_layers', "--n_layers", default=2, type=int, help='GraphSage # of layers')
parser.add_argument('-dropout', "--dropout", default=0.5, type=float, help='dropout')

args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise ValueError("mode %s is not supported." % args.mode)
