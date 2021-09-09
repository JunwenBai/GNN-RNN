python single_year_main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
    --mode test -bs 128 --max_epoch 100 --test_year 2018 --model lstm \
    --crop_type corn --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 \
    --soil_depths 6 --no_management --validation_week 22 --mask_prob 1 --mask_value county_avg  \
    -cp model/corn_weekly/lstm_bs-128_lr-0.0001_maxepoch-100_sche-step_T0-50_testyear-2018_seed-0_no-management/model-37


# python single_year_main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
#     --mode test -bs 128 --max_epoch 100 --test_year 2018 --model cnn \
#     --crop_type corn --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 \
#     --soil_depths 6 --no_management --validation_week 22 --mask_prob 1 --mask_value county_avg  \
#     -cp model/corn_weekly/cnn_bs-128_lr-0.001_maxepoch-100_sche-step_T0-50_testyear-2018_seed-0_no-management/model-8