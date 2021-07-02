import pandas as pd
import numpy as np

CSV_FILE = "/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly_1981-2020.csv"  # "/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_1981-2020_gssurgo.csv"

df = pd.read_csv(CSV_FILE, delimiter=',')
print(df.head())
df = df.to_numpy()
np.save('/home/fs01/jyf6/Crop_Yield_Prediction/data/new_soybean_data', df)