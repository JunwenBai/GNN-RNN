
crop="corn"
year=2018
model="cnn_rnn"
lr=1e-4

python main.py --dataset ${crop}_weekly --crop_type $crop --test_year $year --model $model \
    --data_dir ../data/data_weekly_subset.npz \
    -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode test --length 5 \
    -bs 128 --max_epoch 100 --test_year 2018 -lr $lr --eta_min 1e-5 --check_freq 80 --T0 50 -sche step \
    --combine_weather_and_management --no_management --validation_week 26 --mask_prob 1 --mask_value county_avg \
    -cp model/corn_weekly/2018/cnn_rnn_bs-128_lr-0.0001_maxepoch-100_sche-step_T0-50_testyear-2018_trainweekstart-52_len-5_seed-0_no-management/model-5
