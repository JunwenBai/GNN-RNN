import pandas as pd
import numpy as np

DATASET_CSV_FILE = "/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly_1981-2020.csv"  # "/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_1981-2020_gssurgo.csv"
OUTPUT_FILE = '/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly'

df = pd.read_csv(DATASET_CSV_FILE, delimiter=',')
print(df.head())
df = df.to_numpy()
np.save(OUTPUT_FILE, df)