python simple_train.py --dataset soybean_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
    --test_year 2019 --model lasso --crop_type soybeans --num_weather_vars 23 \
    --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
    --no_management --train_week_start 52 --validation_week 52