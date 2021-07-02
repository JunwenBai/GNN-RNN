python main.py -dataset all_crops --data_dir /home/fs01/jyf6/Crop_Yield_Prediction/data/new_soybean_data.npy \
    -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode test_predictions_over_time --length 5 --test_year 1993 --county_to_plot 17083 --model cnn_rnn \
    -cp model/all_crops/cnn_rnn_bs-128_lr-0.0005_maxepoch-100_sche-step_T0-50_testyear-2016_len-5_seed-0/model-99 \
    --num_outputs 6 --num_weather_vars 23 --num_soil_vars 20 --num_management_vars 0 --num_extra_vars 6 --soil_depths 6

