python simple_train.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
    --test_year 2020 --model mlp --crop_type corn --num_weather_vars 23 \
    --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
    --no_management --train_week_start 17 --validation_week 52