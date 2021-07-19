# Old dataset
# python main.py --dataset soybean --data_dir ../data/soybean_data_full.npz -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
#     --mode train --length 5 -bs 64 --max_epoch 100 --test_year 2018 --model cnn_rnn \
#     -lr 5e-4 --eta_min 1e-5 --check_freq 80 --T0 50 -sche step \
#     --num_outputs 1 --num_weather_vars 6 --num_soil_vars 10 --num_management_vars 14 --num_extra_vars 4 --soil_depths 10


# New dataset
# You can change "crop_type" to the desired crop ("corn", "cotton", "sorghum", "soybeans", "spring_wheat", "winter_wheat")
# Add the "--share_conv_parameters" if you want each weather variable to share the same conv parameters (same with soil/progress data).
# In "data_dir", you can change "combined_dataset_weekly" to "combined_dataset_daily" to use daily data.
python main.py --dataset corn_weekly_no_param_sharing --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \
    -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode train --length 5 -bs 128 --max_epoch 200 --test_year 2016 --model cnn_rnn \
    -lr 5e-4 --eta_min 1e-5 --check_freq 80 --T0 50 -sche step \
    --crop_type corn --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6
