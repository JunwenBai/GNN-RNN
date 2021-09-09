
for y in 2018 2019
do
    for l in 1e-3 1e-4 1e-5
    do
        python single_year_main.py --dataset corn_weekly --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
            --mode train -bs 128 --max_epoch 100 --test_year $y --model $1 \
            -lr $l --eta_min 1e-5 --check_freq 80 --T0 50 -sche step \
            --crop_type corn --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
            --no_management
    done
done

# Run these commands:
# ./single_year_run_train.sh cnn
# ./single_year_run_train.sh gru
# ./single_year_run_train.sh lstm
