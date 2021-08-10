python main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz  \
    -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode test --length 5 -bs 64 --test_year 2018 --z_dim 64 \
    --crop_type corn --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 \
    --num_extra_vars 6 --soil_depths 6 --no_management --aggregator_type pool \
    --validation_week 52 \
    -cp model/corn_weekly/gnn_bs-64_lr-5e-05_maxepoch-200_sche-cosine_T0-100_testyear-2018_aggregator-pool_trainweekstart-17_len-5_seed-0_no-management/model-115

# python main.py --dataset soybean --data_dir ../data/soybean_data_full.npz \
#     --mode test --length 5 -bs 32 --test_year 2018 -lr 6e-4 --eta_min 1e-5 --check_freq 80 --T0 50 \
#     -cp model/soybean/bs-32_lr-0.0006_maxepoch-100_testyear-2018_len-5_seed-0/model-58
