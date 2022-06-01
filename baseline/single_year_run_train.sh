
for y in 2018 # 2019
do
    for lr in 5e-4 2e-4 1e-4
    do
        for s in 0 1 2
        do
            # python single_year_main.py --dataset soybeans_weekly --crop_type soybeans --test_year $y --model $1 \
            #     --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
            #     --mode train -bs 128 --max_epoch 100 -lr $lr --eta_min 1e-5 --check_freq 80 --T0 50 -sche step \
            #     --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 \
            #     --no_management --seed $s
            python single_year_main.py --dataset corn_weekly_CORRECTED --crop_type corn \
                --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
                --mode train -bs 128 --max_epoch 100 --test_year $y --model cnn -lr $lr --eta_min 1e-5 \
                --check_freq 80 --T0 50 -sche step --num_weather_vars 23 --num_management_vars 96 \
                --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 --no_management --seed $s
        done
    done
done

# Run these commands:
# ./single_year_run_train.sh cnn
# ./single_year_run_train.sh gru 
# ./single_year_run_train.sh lstm 
