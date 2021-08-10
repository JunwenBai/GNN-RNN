python single_year_main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
    --mode train -bs 128 --max_epoch 100 --test_year 2020 --model cnn \
    -lr 1e-4 --eta_min 1e-5 --check_freq 80 --T0 50 -sche step \
    --crop_type corn --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
    --no_management