# Runs a simple scikit-learn method. Does a grid search over possible parameters, and outputs the performance
# of 3 runs with the best parameters.


for model in ridge # lasso gradient_boosting_regressor mlp  # Model type
do
    for crop in corn # soybeans  # Crop type
    do
        for year in 2018 # 2019  # Test year
        do
            python simple_train.py --dataset ${crop}_weekly --crop_type $crop --test_year $year --model $model \
                --data_dir ../data/data_weekly_subset.npz \
                --no_management --standardize_outputs
        done
    done
done
#--data_dir ../data/data_weekly_subset.npz \
#--data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz \

