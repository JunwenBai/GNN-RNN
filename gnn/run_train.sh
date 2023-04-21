for crop in corn  # soybeans
do
    for year in 2018  # 2019
    do
        for lr in 5e-5  #1e-4  # 3e-4 1e-3 3e-5
        do
            for seed in 0  # 1 2
            do
                python main.py --dataset ${crop}_weekly --crop_type $crop --test_year $year \
                    --data_dir ../data/data_weekly_subset.npz \
                    -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
                    --mode train -bs 32 --max_epoch 100 \
                    -lr $lr -sche cosine --T0 100 --eta_min 1e-5 --check_freq 80 \
                    --dropout 0.1 --z_dim 64 \
                    --no_management --aggregator_type pool --seed $seed 
            done
        done
    done
done