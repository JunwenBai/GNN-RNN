crop="corn"
year=2018
lr=1e-4

python main.py --dataset ${crop}_weekly --crop_type $crop --test_year $year \
    --data_dir ../data/data_weekly_subset.npz \
    -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode test -bs 32 --max_epoch 100 \
    -lr $lr -sche cosine --T0 100 --eta_min 1e-5 --check_freq 80 \
    --dropout 0.1 --z_dim 64 \
    --no_management --aggregator_type pool \
    --validation_week 26 --mask_prob 1 --mask_value county_avg \
    -cp model/corn_weekly/2018/gnn_bs-32_lr-5e-05_maxepoch-100_sche-cosine_T0-100_step-50_gamma-0.5_dropout-0.1_testyear-2018_aggregator-pool_encoder-cnn_trainweekstart-52_len-5_seed-0_no-management/model-29

