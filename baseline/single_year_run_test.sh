crop="corn"
year=2018
model="cnn"
lr=1e-4

python single_year_main.py --dataset ${crop}_weekly --crop_type $crop --test_year $year --model $model \
    --data_dir ../data/data_weekly_subset.npz \
    --mode test -bs 128 --max_epoch 100 -lr $lr --eta_min 1e-5 --check_freq 80 --T0 50 -sche step \
    --no_management --validation_week 26 --mask_prob 1 --mask_value county_avg  \
    -cp model/corn_weekly/cnn_bs-128_lr-0.0001_maxepoch-100_sche-step_T0-50_testyear-2018_seed-0_no-management/model-64
