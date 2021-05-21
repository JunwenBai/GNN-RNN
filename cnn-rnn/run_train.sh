python main.py --dataset soybean --data_dir ../data/soybean_data_full.npz -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode train --length 5 -bs 64 --max_epoch 100 --test_year 2018 --model cnn_rnn\
    -lr 5e-4 --eta_min 1e-5 --check_freq 80 --T0 50 -sche step
