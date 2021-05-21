python main.py --dataset soybean --data_dir ../data/soybean_data_full.npz -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode train --length 5 -bs 64 --max_epoch 100 --sleep 100 \
    --test_year 2018 -lr 3e-4 --check_freq 80 \
    --sche const --eta_min 1e-5 --T0 50 --lrsteps 50 --gamma 0.8

python main.py --dataset soybean --data_dir ../data/soybean_data_full.npz -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode train --length 5 -bs 64 --max_epoch 100 --sleep 100 \
    --test_year 2018 -lr 4e-4 --check_freq 80 \
    --sche const --eta_min 1e-5 --T0 50 --lrsteps 50 --gamma 0.8
