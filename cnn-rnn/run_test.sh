
python main.py --dataset soybeans_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
    -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode test --length 5 -bs 128 --test_year 2018 --model cnn_rnn \
    --crop_type soybeans --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
    --combine_weather_and_management --no_management --validation_week 26 --mask_prob 1 --mask_value county_avg \
    -cp model/soybeans_weekly/cnn_rnn_bs-128_lr-0.0005_maxepoch-100_sche-step_T0-150_testyear-2018_trainweekstart-52_len-5_seed-0_no-management/model-19


# python main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
#     -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode test --length 5 -bs 128 --test_year 2018 --model cnn_rnn \
#     --crop_type corn --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
#     --combine_weather_and_management --no_management \
#     --validation_week 22 --mask_prob 1 --mask_value county_avg \
#     -cp model/corn_weekly/cnn_rnn_bs-128_lr-0.0005_maxepoch-100_sche-step_T0-150_testyear-2018_trainweekstart-52_len-5_seed-0_no-management/model-29
    
    



# python main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
#     -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode test --length 5 -bs 32 --test_year 2020 --model cnn_rnn \
#     --crop_type corn --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
#     --combine_weather_and_management --no_management --validation_week 20 --mask_prob 1 --mask_value county_avg \
#     -cp model/corn_weekly/cnn_rnn_bs-128_lr-0.0001_maxepoch-100_sche-step_T0-50_testyear-2020_len-5_seed-0_no-management/model-15


# python main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
#     -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode test --length 5 -bs 32 --test_year 2018 --model cnn-rnn \
#     -cp asdf
    
    #  -lr 6e-4 --eta_min 1e-5 --check_freq 80 --T0 50 \
    # --model rnn \
    # -cp model/soybean/bs-32_lr-0.0006_maxepoch-100_testyear-2018_len-5_seed-0/model-58
