python main.py --dataset soybean --data_dir ../data/soybean_data_full.npz \
    --mode test --length 5 -bs 32 --test_year 2018 -lr 6e-4 --eta_min 1e-5 --check_freq 80 --T0 50 \
    -cp model/soybean/bs-32_lr-0.0006_maxepoch-100_testyear-2018_len-5_seed-0/model-58
