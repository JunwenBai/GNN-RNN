crop="corn"
year=2018
lr=1e-4
python main.py --dataset ${crop}_weekly --crop_type $crop --test_year $year \
    --data_dir ../data/data_weekly_subset.npz \
    -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode test --length 5 -bs 32 --max_epoch 100 -sleep 100 \
    -lr $lr --sche cosine --T0 100 --eta_min 1e-6 --check_freq 80  \
    --T_mult 2 --lrsteps 25 --gamma 1 \
    --dropout 0.1 --z_dim 64 --weight_decay 1e-5 \
    --no_management --aggregator_type pool --encoder_type cnn \
    --validation_week 52 --mask_prob 1 --mask_value county_avg \
    -cp model/corn_weekly/2018/gnn-rnn_bs-32_lr-0.0001_maxepoch-100_sche-cosine_T0-100_etamin-1e-06_step-25_gamma-1.0_dropout-0.1_sleep-100_testyear-2018_aggregator-pool_encoder-cnn_trainweekstart-52_len-5_weightdecay-1e-05_seed-0_no-management/model-25


# python main.py --dataset ${crop}_weekly --crop_type $crop --test_year $year \
#     --data_dir ../data/data_weekly_subset.npz \
#     -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode test --length 5 -bs 32 --lr $lr \
#     --aggregator_type pool --encoder_type cnn --no_management \
#     --validation_week 26 --seed 0 \
#     --mask_prob 1 --mask_value county_avg \
#     -cp model/soybeans_weekly_no_Y_input_shuffle/2018/gnn-rnn_bs-32_lr-0.0001_maxepoch-100_sche-cosine_T0-34_etamin-1e-05_step-25_gamma-1.0_dropout-0.1_sleep-100_testyear-2018_aggregator-pool_encoder-cnn_trainweekstart-52_len-5_weightdecay-1e-05_seed-0_no-management/model-16

# model/soybeans_weekly_no_Y_input_shuffle/2018/gnn-rnn_bs-32_lr-0.0001_maxepoch-100_sche-cosine_T0-100_etamin-1e-06_step-25_gamma-1.0_dropout-0.1_sleep-100_testyear-2018_aggregator-pool_encoder-cnn_trainweekstart-52_len-5_weightdecay-0.0001_seed-1_no-management/model-32

# python main.py --dataset corn_weekly_no_Y_input_shuffle --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
#     -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode test --length 5 --test_year 2018 -bs 32 \
#     --crop_type corn --num_weather_vars 23 --num_management_vars 96 \
#     --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
#     --aggregator_type pool --encoder_type cnn --no_management \
#     --validation_week 22 --seed 0 \
#     --mask_prob 1 --mask_value county_avg \
#     -cp model/corn_weekly_no_Y_input_shuffle/2018/gnn-rnn_bs-32_lr-5e-05_maxepoch-200_sche-cosine_T0-200_etamin-1e-06_step-25_gamma-1.0_dropout-0.1_sleep-200_testyear-2018_aggregator-pool_encoder-cnn_trainweekstart-52_len-5_weightdecay-1e-05_seed-0_no-management/model-76

    # -cp model/corn_weekly_no_Y_input_shuffle/2018/gnn-rnn_bs-32_lr-5e-05_maxepoch-100_sche-cosine_T0-100_etamin-1e-06_step-25_gamma-1.0_dropout-0.1_sleep-100_testyear-2018_aggregator-pool_encoder-cnn_trainweekstart-52_len-5_weightdecay-1e-05_seed-2_no-management/model-72




# python main.py --dataset corn_weekly_no_Y_input --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
#     -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode test --length 5 --test_year 2020 -bs 64 \
#     --crop_type corn --num_weather_vars 23 --num_management_vars 96 \
#     --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
#     --aggregator_type pool --encoder_type cnn --no_management \
#     --validation_week 52 --seed 0 \
#     --mask_prob 1 --mask_value zero \
#     -cp model/corn_weekly_no_Y_input/gnn-rnn_bs-64_lr-0.0001_maxepoch-50_sche-const_T0-50_step-50_gamma-0.8_dropout-0.1_sleep-100_testyear-2020_aggregator-pool_len-5_seed-0_no-management/model-39

# python main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
#     -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode test --length 5 --test_year 2019 -bs 32 \
#     --crop_type corn --num_weather_vars 23 --num_management_vars 96 \
#     --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
#     --aggregator_type pool --encoder_type cnn --no_management \
#     --validation_week 52 --seed 0 \
#     --mask_prob 1 --mask_value zero \
#     -cp model/corn_weekly/gnn-rnn_bs-32_lr-0.0001_maxepoch-100_sche-const_T0-50_step-50_gamma-0.8_dropout-0.5_sleep-100_testyear-2019_len-5_seed-0_no-management/model-20

# python main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
#     -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode test --length 5 --test_year 2018 -bs 32 \
#     --crop_type corn --num_weather_vars 23 --num_management_vars 96 \
#     --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
#     --aggregator_type pool --encoder_type cnn --no_management \
#     --validation_week 52 --seed 0 \
#     --mask_prob 1 --mask_value zero \
#     -cp model/corn_weekly/gnn-rnn_bs-32_lr-0.0001_maxepoch-100_sche-const_T0-50_step-50_gamma-0.8_dropout-0.1_sleep-100_testyear-2018_len-5_seed-0_no-management/model-18

    # -cp model/corn_weekly_no_Y_input/gnn-rnn_bs-64_lr-0.001_maxepoch-100_sche-const_T0-50_step-50_gamma-0.8_dropout-0.1_sleep-100_testyear-2020_aggregator-pool_encoder-cnn_trainweekstart-18_len-5_seed-0_no-management/model-37