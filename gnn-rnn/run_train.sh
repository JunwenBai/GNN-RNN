echo "Hello, we are in run_train.sh. About to call python (2018)."


# BEST 2018 corn
# python main.py --dataset corn_weekly_no_Y_input_shuffle --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl --crop_type corn --mode train --length 5 -bs 32 --max_epoch 100 --sleep 100 --test_year 2018 -lr 5e-5 --check_freq 80 --sche cosine --eta_min 1e-6 --T0 100 --T_mult 2 --lrsteps 25 --gamma 1 --dropout 0.1 --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 --aggregator_type pool --encoder_type cnn --no_management --train_week_start 52 --validation_week 52 --seed 0 --weight_decay 1e-5 --mask_prob 0.5 --mask_value zero

for lr in 1e-4 5e-5 2e-5
do
    for s in 0 1 2
    do
        python main.py --dataset corn_weekly_no_Y_input_shuffle_CORRECTED --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
            -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl --crop_type corn --mode train --length 5 \
            -bs 32 --max_epoch 100 --sleep 100 --test_year 2018 -lr $lr --check_freq 80 --sche cosine --eta_min 1e-6 \
            --T0 100 --T_mult 2 --lrsteps 25 --gamma 1 --dropout 0.1 \
            --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
            --aggregator_type pool --encoder_type cnn --no_management --train_week_start 52 --validation_week 52 \
            --seed $s --weight_decay 1e-5 --mask_prob 0.5 --mask_value zero
    done
done

# for s in 0
# do
#     python main.py --dataset corn_weekly_Y_zero --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
#         -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#         --mode train --length 5 -bs $1 --max_epoch 200 --sleep 100 \
#         --test_year $2 -lr $3 --check_freq 80 \
#         --sche $4 --eta_min 1e-5 --T0 50 --lrsteps 50 --gamma 0.8 --dropout $5 \
#         --crop_type corn --num_weather_vars 23 --num_management_vars 96 \
#         --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
#         --aggregator_type pool --encoder_type cnn --no_management \
#         --train_week_start $6 --validation_week 52 --seed $s \
#         --mask_prob 0.5 --mask_value zero
# done

# ./run_train.sh 64 2020 1e-4 0.1 52 
# ./run_train.sh 32 2020 3e-5 cosine 0.1 52

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


# python main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
#     -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode train --length 5 -bs 32 --max_epoch 100 --sleep 100 \
#     --test_year 2019 -lr 1e-4 --check_freq 80 \
#     --sche const --eta_min 1e-5 --T0 50 --lrsteps 50 --gamma 0.8 --dropout 0.1 \
#     --crop_type corn --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 --no_management
