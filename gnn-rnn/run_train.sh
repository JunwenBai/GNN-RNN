echo "Hello, we are in run_train.sh. About to call python."

# python main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
#     -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode train --length 5 -bs 64 --max_epoch 100 --sleep 100 \
#     --test_year 2018 -lr 1e-4 --check_freq 80 \
#     --sche const --eta_min 1e-5 --T0 50 --lrsteps 50 --gamma 0.8 \
#     --crop_type corn --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 --no_management

# python main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
#     -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode train --length 5 -bs 64 --max_epoch 100 --sleep 100 \
#     --test_year 2018 -lr 3e-4 --check_freq 80 \
#     --sche const --eta_min 1e-5 --T0 50 --lrsteps 50 --gamma 0.8 \
#     --crop_type corn --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 --no_management

# python main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
#     -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode train --length 5 -bs 64 --max_epoch 100 --sleep 100 \
#     --test_year 2018 -lr 1e-3 --check_freq 80 \
#     --sche const --eta_min 1e-5 --T0 50 --lrsteps 50 --gamma 0.8 \
#     --crop_type corn --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 --no_management

# python main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
#     -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode train --length 5 -bs 32 --max_epoch 100 --sleep 100 \
#     --test_year 2018 -lr 1e-4 --check_freq 80 \
#     --sche const --eta_min 1e-5 --T0 50 --lrsteps 50 --gamma 0.8 \
#     --crop_type corn --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 --no_management


python main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
    -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode train --length 5 -bs 64 --max_epoch 100 --sleep 100 \
    --test_year 2018 -lr 1e-5 --check_freq 80 \
    --sche const --eta_min 1e-5 --T0 50 --lrsteps 50 --gamma 0.8 \
    --crop_type corn --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 --no_management
