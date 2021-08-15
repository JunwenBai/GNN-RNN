python main.py --dataset corn_weekly_no_Y_input --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
    -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode test --length 5 --test_year 2020 -bs 64 \
    --crop_type corn --num_weather_vars 23 --num_management_vars 96 \
    --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
    --aggregator_type pool --encoder_type cnn --no_management \
    --validation_week 52 --seed 0 \
    --mask_prob 1 --mask_value zero \
    -cp model/corn_weekly_no_Y_input/gnn-rnn_bs-64_lr-0.0001_maxepoch-50_sche-const_T0-50_step-50_gamma-0.8_dropout-0.1_sleep-100_testyear-2020_aggregator-pool_len-5_seed-0_no-management/model-39

python main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
    -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode test --length 5 --test_year 2019 -bs 32 \
    --crop_type corn --num_weather_vars 23 --num_management_vars 96 \
    --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
    --aggregator_type pool --encoder_type cnn --no_management \
    --validation_week 52 --seed 0 \
    --mask_prob 1 --mask_value zero \
    -cp model/corn_weekly/gnn-rnn_bs-32_lr-0.0001_maxepoch-100_sche-const_T0-50_step-50_gamma-0.8_dropout-0.5_sleep-100_testyear-2019_len-5_seed-0_no-management/model-20

python main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
    -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode test --length 5 --test_year 2018 -bs 32 \
    --crop_type corn --num_weather_vars 23 --num_management_vars 96 \
    --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
    --aggregator_type pool --encoder_type cnn --no_management \
    --validation_week 52 --seed 0 \
    --mask_prob 1 --mask_value zero \
    -cp model/corn_weekly/gnn-rnn_bs-32_lr-0.0001_maxepoch-100_sche-const_T0-50_step-50_gamma-0.8_dropout-0.1_sleep-100_testyear-2018_len-5_seed-0_no-management/model-18

    # -cp model/corn_weekly_no_Y_input/gnn-rnn_bs-64_lr-0.001_maxepoch-100_sche-const_T0-50_step-50_gamma-0.8_dropout-0.1_sleep-100_testyear-2020_aggregator-pool_encoder-cnn_trainweekstart-18_len-5_seed-0_no-management/model-37