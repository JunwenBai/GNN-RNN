import numpy as np
import pandas as pd

DATASET_CSV_FILE = "/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_daily_1981-2020.csv"  # "/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_1981-2020_gssurgo.csv"
OUTPUT_FILE = '/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_daily'

data_csv = pd.read_csv(DATASET_CSV_FILE, delimiter=',')
data_npz = np.array(data_csv)
np.savez_compressed(OUTPUT_FILE, data=data_npz)