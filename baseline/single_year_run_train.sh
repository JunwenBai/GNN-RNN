
for y in 2018 # 2019
do
    for s in 0 1 2 #
    do
        python single_year_main.py --dataset corn_weekly --crop_type corn --test_year $y --model $1 \
            --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
            --mode train -bs 128 --max_epoch 100 -lr $2 --eta_min 1e-5 --check_freq 80 --T0 50 -sche step \
            --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
            --no_management --seed $s
    done
done

# Run these commands:
# ./single_year_run_train.sh cnn 1e-3
# ./single_year_run_train.sh gru 1e-3
# ./single_year_run_train.sh lstm 1e-4
