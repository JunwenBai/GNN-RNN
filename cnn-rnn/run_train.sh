# Old dataset
# python main.py --dataset soybean --data_dir ../data/soybean_data_full.npz -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode train --length 5 -bs 64 --max_epoch 100 --test_year 2018 --model cnn_rnn \
#     -lr 5e-4 --eta_min 1e-5 --check_freq 80 --T0 50 -sche step \
#     --num_outputs 1 --num_weather_vars 6 --num_soil_vars 10 --num_management_vars 14 --num_extra_vars 4 --soil_depths 10

# New dataset
# You can change "crop" to the desired crop ("corn", "upland_cotton", "sorghum", "soybeans", "spring_wheat", "winter_wheat"),
# and adjust "year" (test_year), "lr" (learning rates), etc.
# Add "--share_conv_parameters" if you want each weather variable to share the same conv parameters (same with soil/progress data).
# Add "--combine_weather_and_management" if you want weather and management data to be processed together in the same CNN.
# In "data_dir", you can change the path to the data file.
# Usage:
# ./run_train.sh cnn_rnn
# ./run_train.sh rnn

for crop in corn  # soybeans
do
    for year in 2018  # 2019
    do
        for lr in 1e-4  # 3e-4 1e-3 3e-5
        do
            for seed in 0  # 1 2
            do
                python main.py --dataset ${crop}_weekly --crop_type $crop --test_year $year --model $1 \
                    --data_dir ../data/data_weekly_subset.npz \
                    -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
                    --mode train --length 5 \
                    -bs 128 --max_epoch 100 --test_year 2018 -lr $lr --eta_min 1e-5 --check_freq 80 --T0 50 -sche step \
                    --combine_weather_and_management --no_management --seed $seed
            done
        done
    done
done
#--data_dir ../data/data_weekly_subset.npz \
#--data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
