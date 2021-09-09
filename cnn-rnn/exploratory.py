import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import visualization_utils

DATASET_FILE = "/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz"
DATASET_CSV_FILE = "/mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly_1981-2020.csv"
OUTPUT_FILE = "exploratory/num_counties.csv"
PLOT_DIR = 'exploratory'
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
    os.makedirs(os.path.join(PLOT_DIR, "box_plots"))
    os.makedirs(os.path.join(PLOT_DIR, "histograms"))
    os.makedirs(os.path.join(PLOT_DIR, "maps"))

raw_data = np.load(DATASET_FILE) #load data from the data_dir
data = raw_data['data']
column_mins = np.nanmin(data, axis=0)
column_maxs = np.nanmax(data, axis=0)
print("Column mins shape", column_mins.shape)
YEARS = list(range(1981, 2021))
year_column = data[:, 1]
# YIELD_INDICES = {'corn': 2,
#                   'cotton': 3,
#                   'sorghum': 4,
#                   'soybeans': 5,
#                   'wheat': 6}
COLUMN_NAMES = pd.read_csv(DATASET_CSV_FILE, index_col=False, nrows=0).columns.tolist()
PLOT_INDICES = list(range(2, 8))
print(COLUMN_NAMES[0:10])

# exit(1)
# List of lists. Goal is to create a csv file where each feature has a row,
# containing the number of counties with data in each year.
num_counties_with_data = []

# Loop through features
all_states = set()
all_counties = set()
for col_idx in [2, 5, 7]:  # range(2, data.shape[1]):
    col_name = COLUMN_NAMES[col_idx]
    values_by_year = []
    years_with_data = []
    counties_with_col_data = [col_name]
    print("Crop", col_name, "=================================")
    all_year_states = set()
    for year in YEARS:
        rows_to_select = (year_column == year)
        values = data[rows_to_select, col_idx]  # Filter for rows within this year
        counties = data[rows_to_select, 0]


        # Plot map of variable data this year (we include the NaN entries so that
        # every county gets plotted
        if col_idx in PLOT_INDICES:
            visualization_utils.plot_county_data(counties, values, col_name, year,
                                                 os.path.join(PLOT_DIR, "maps"))
            zero_values = np.empty_like(values)
            zero_values[values == 0] = 0
            zero_values[values != 0] = np.nan
            visualization_utils.plot_county_data(counties, zero_values, col_name + "_ZERO_YIELD", year,
                                                 os.path.join(PLOT_DIR, "maps"))
        non_nan_rows = ~np.isnan(values)
        values = values[non_nan_rows]
        counties = counties[non_nan_rows]
        if values.size == 0:
            counties_with_col_data.append(np.nan)
            continue
        counties_with_col_data.append(counties.size)
        years_with_data.append(year)
        values_by_year.append(values)

        states = set([int(county // 1000) for county in counties])
        print("****************")
        print("States in year", year, "-", states)
        print("Num states:", len(states))
        all_year_states = all_year_states.union(states)
        all_counties = all_counties.union(counties)

    print("Total number of states:", len(all_year_states))
    print(all_year_states)

    all_states = all_states.union(all_year_states)
    print("CUMULATIVE states", len(all_states))
    print(all_states)
    print("CUMULATIVE counties", len(all_counties))

    # Append number of counties with data
    num_counties_with_data.append(counties_with_col_data)
    if len(values_by_year) == 0:
        continue

    # Plot box-and-whisker plots
    fig, ax = plt.subplots(1, 1, figsize=(24, 10))
    ax.boxplot(values_by_year, labels=years_with_data)
    ax.set_xlabel("year")
    ax.set_ylabel(col_name)
    ax.set_title(col_name + " over time")
    plt.savefig(os.path.join(PLOT_DIR, "box_plots/box_plot_" + str(col_idx).zfill(4) + "_" + col_name + ".png"))
    plt.close()

data_counts = pd.DataFrame(num_counties_with_data, columns=["column_name"] + YEARS)
print(data_counts.head())
data_counts.to_csv(OUTPUT_FILE)


# ================================== OLD CODE, NOT NEEDED NOW =====================================
# # For each crop, plot yields over time
# for crop, yield_idx in YIELD_INDICES.items():
#     valid_indices = ~np.isnan(data[:, yield_idx])
#     plt.scatter(data[valid_indices, 1], data[valid_indices, yield_idx])
#     plt.savefig(os.path.join(PLOT_DIR, crop + "_yields_over_time.png"))

# # Inspect each variable
# for col_idx in range(2, data.shape[1]):
    
#     num_entries_with_data = np.count_nonzero(~np.isnan(data[:, col_idx]))
#     if num_entries_with_data == data.shape[0]:
#         print("COLUMN " + COLUMN_NAMES[col_idx] + ": all data present!")
#     elif num_entries_with_data == data.shape[0] - 40:
#         print("COLUMN " + COLUMN_NAMES[col_idx] + ": all data present except one county!")
#     else:
#         print("COLUMN " + COLUMN_NAMES[col_idx] + ": " + str(num_entries_with_data) + \
#               " of " + str(data.shape[0]) + " entries present!")
#         if num_entries_with_data > 0:
#             years = data[:, 1]
#             years_with_data = years[~np.isnan(data[:, col_idx])]
#             unique, counts = np.unique(years_with_data, return_counts=True)
#             for i in range(len(unique)):
#                 print("--> " + str(int(unique[i])) + ': ' + str(counts[i]) + " entries present")

# Count number of counties with missing yield data, for each year and variable
# results = {k: [] for k in YIELD_INDICES.keys()}
# results['year'] = YEARS
# for year in YEARS:
#     year_data = data[data[:, 1] == year]
#     counties = year_data[:, 0]
#     num_counties = {'year': year}
#     for crop, yield_idx in YIELD_INDICES.items():
#         counties_with_data = counties[~np.isnan(year_data[:, yield_idx])]
#         assert(len(set(counties_with_data)) == counties_with_data.size)
#         assert(len(set(counties_with_data)) == np.count_nonzero(~np.isnan(year_data[:, yield_idx])))
#         results[crop].append(len(set(counties_with_data)))

#         # Plot map of yield data this year
#         visualization_utils.plot_county_data(counties, year_data[:, yield_idx], crop, year)

#         # Plot histogram of yield data for this year
#         if not np.isnan(column_mins[yield_idx]):
#             histogram_range = (column_mins[yield_idx], column_maxs[yield_idx])
#             visualization_utils.plot_histogram(year_data[:, yield_idx], range=histogram_range, column_name=crop, year=year)


# print("Results_df")
# results_df = pd.DataFrame(results)
# print(results_df.head())
# results_df.to_csv(OUTPUT_FILE)
