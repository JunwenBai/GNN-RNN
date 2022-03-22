import os
import pandas as pd
import visualization_utils

# OUTPUT_NAMES = ["corn"]
# TEST_YEAR = 2018
# DIR = "corn_weekly_no_Y_input_shuffle/2018/gnn-rnn_bs-32_lr-5e-05_maxepoch-100_sche-cosine_T0-100_etamin-1e-06_step-25_gamma-1.0_dropout-0.1_sleep-100_testyear-2018_aggregator-pool_encoder-cnn_trainweekstart-52_len-5_weightdecay-1e-05_seed-2_no-management"


# OUTPUT_NAMES = ["corn"]
# TEST_YEAR = 2019
# DIR = "corn_weekly_no_Y_input_shuffle/2019/gnn-rnn_bs-32_lr-5e-05_maxepoch-100_sche-cosine_T0-200_etamin-1e-06_step-25_gamma-1.0_dropout-0.1_sleep-100_testyear-2019_aggregator-pool_encoder-cnn_trainweekstart-52_len-5_weightdecay-1e-05_seed-2_no-management"

# OUTPUT_NAMES = ["soybeans"]
# TEST_YEAR = 2018
# DIR = "soybeans_weekly_no_Y_input_shuffle/2018/gnn-rnn_bs-32_lr-0.0001_maxepoch-100_sche-cosine_T0-100_etamin-1e-06_step-25_gamma-1.0_dropout-0.1_sleep-100_testyear-2018_aggregator-pool_encoder-cnn_trainweekstart-52_len-5_weightdecay-0.0001_seed-2_no-management"

OUTPUT_NAMES = ["soybeans"]
TEST_YEAR = 2019
DIR =  "soybeans_weekly_no_Y_input_shuffle/2019/gnn-rnn_bs-32_lr-5e-05_maxepoch-100_sche-cosine_T0-100_etamin-1e-06_step-25_gamma-1.0_dropout-0.1_sleep-100_testyear-2019_aggregator-pool_encoder-cnn_trainweekstart-52_len-5_weightdecay-1e-05_seed-0_no-management"


RESULTS_DIR = os.path.join("/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/output/gnn-rnn/results", DIR)
OUTPUT_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/exploratory_plots/paper_plots_2"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

test_results = pd.read_csv(os.path.join(RESULTS_DIR, "test_results.csv"))
visualization_utils.plot_true_vs_predicted(test_results, OUTPUT_NAMES, str(TEST_YEAR), OUTPUT_DIR)

