# Trains a single-year baseline. Run these commands:
# ./single_year_run_train.sh cnn
# ./single_year_run_train.sh gru 
# ./single_year_run_train.sh lstm 

for crop in corn  # soybeans
do
    for year in 2018  # 2019
    do
        for lr in 1e-4  # 3e-4 1e-3 3e-5
        do
            for seed in 0  # 1 2
            do
                python single_year_main.py --dataset ${crop}_weekly --crop_type $crop --test_year $year --model $1 \
                    --data_dir ../data/data_weekly_subset.npz \
                    --mode train -bs 128 --max_epoch 100 -lr $lr --eta_min 1e-5 --check_freq 80 --T0 50 -sche step \
                    --no_management --seed $seed
            done
        done
    done
done
#--data_dir ../data/data_weekly_subset.npz \
#--data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \


