import os
import pickle
import torch
import numpy as np
import subprocess
import visualization_utils
import warnings

def build_path(path):
    path_levels = path.split('/')
    cur_path = ""
    for path_seg in path_levels:
        if len(cur_path):
            cur_path = cur_path + "/" + path_seg
        else:
            cur_path = path_seg
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_X_Y(data, args, device):
    if args.data_dir == "soybean_data_full.npz":
        # Old dataset (given from CNN-RNN paper)
        counties = data[:, 0].astype(int)
        years = data[:, 1].astype(int)
        Y = data[:, 2:3]
        X = data[:, 3:]
    else:
        # Our dataset
        print("Initially data", data.shape)
        counties_all = data[:, 0].astype(int)
        years_all = data[:, 1].astype(int)
        # all_nan_rows = np.all(np.isnan(data[:, [args.output_idx]]), axis=1)  # Remove rows where all labels are NaN

        # For now, we only have yield until 2016, so only include years up to 2016.
        # Exclude county 25019 (Nantucket County) since it has no NLDAS data.
        # Exclude rows where all labels are NaN.
        data = data[(years_all <= args.test_year) & (counties_all != 25019)]  # & (~all_nan_rows)]
        print("After filtering", data.shape)
        counties = data[:, 0].astype(int)
        years = data[:, 1].astype(int)
        Y = data[:, [args.output_idx]]
        X = data[:, 7:]

    print("get_X_Y")
    print("X shape", X.shape)
    print("Y shape", Y.shape)

    # Compute the unique years and counties
    min_year = int(min(years))
    max_year = int(max(years))
    county_set = sorted(list(set(counties)))

    # Compute average yield of each year (to detect underlying yearly trends)
    avg_Y = {}
    avg_Y_lst = []
    for year in range(min_year, max_year+1):
        avg_Y[year] = np.nanmean(Y[years == year, :], axis=0)
        avg_Y_lst.append(avg_Y[year])
    '''mean_Y = np.mean(avg_Y_lst)
    std_Y = np.std(avg_Y_lst)
    for year in range(min_year, max_year+1):
        avg_Y[year] = (avg_Y[year] - mean_Y) / std_Y'''
    avg_Y[min_year-1] = avg_Y[min_year]

    # For each row in X, get the average yield of the previous year, and add this as a column of X
    Ybar = []
    for year in years:
        Ybar.append(avg_Y[year-1])
    Ybar = np.array(Ybar)  #.reshape(-1, 1) - removed this because we may have multiple outputs
    X = np.concatenate((X, Ybar), axis=1)

    # Compute the mean and standard deviation of each feature (over non-NaN values), over the train years.
    # We will use these to standardize the features later.
    # If the feature is NaN everywhere, return 0 for mean/std (the "nan_to_num" function does this)
    known_years = data[:, 1] < (args.test_year-1)
    with warnings.catch_warnings():  # Supress warning about columns being NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)
        X_mean = np.nanmean(X[known_years], axis=0, keepdims=True)
        X_std = np.nanstd(X[known_years], axis=0, keepdims=True)

        # HACK: If the standard deviation of a feature on train set is 0 (e.g. all
        # values are the same), then standardizing anything apart from that value
        # will produce a super extreme z-score. So just replace those standard 
        # deviations with 1. Also, if the mean/std are NaN, replace with 0 and 1. 
        X_std[(X_std < 1e-6) | np.isnan(X_std)] = 1
        X_mean[np.isnan(X_mean)] = 0

    # Check stds
    np.set_printoptions(threshold=np.inf)
    # print("X_std")
    # print(np.squeeze(X_std))
    # print('============================== :):):):)')
    # print(np.argwhere(X_std < 1e-7))
    # exit(1)

    # Standardize each feature (column of X)
    X = (X - X_mean) / (X_std + 1e-10)

    # Check for extreme values in X (after standardization)
    indices = np.argwhere((X > 100) | (X < -100))
    if indices.shape[0] == 0:
        print('no extreme values in X :)')
    else:
        for i in range(indices.shape[0]):
            row, col = indices[i, 0], indices[i, 1]
            print("Extreme value indices", row, col + 7, "- yr", years[row])

    # For now, replace all NA with 0.
    X = np.nan_to_num(X)

    # # Fill in gaps for progress data
    # assert ((args.progress_indices[-1] + 1 - args.progress_indices[0]) % args.time_intervals == 0)
    # # Loop through each example
    # for i in range(X.shape[0]):
    #     # Loop through each progress variable (which is itself a range of "args.time_intervals" variables)
    #     for progress_var_start in range(args.progress_indices[0], args.progress_indices[-1] + 1, args.time_intervals):
    #         current_progress = 0
    #         for progress_idx in range(progress_var_start, progress_var_start + args.time_intervals):
    #             if np.isnan(X[i, progress_idx]):
    #                 X[i, progress_idx] = current_progress
    #             else:
    #                 current_progress = X[i, progress_idx]

    # # Sanity check: plot a feature for a given year/week
    # print("Random checks")
    # X_iowa = X[(counties == 19065) & (years == 1993)]
    # print(X_iowa[0, 1456:1508])
    # print(X_iowa[0, 1508:1560])
    # print(X_iowa[0, 1560:1612])

    # YEAR = 1981
    # WEEK = 22
    # COLUMN_IDX = (1464+WEEK-1) - 8
    # COLUMN_NAME = "corn_PROGRESS, MEASURED IN PCT PLANTED_week_" + str(WEEK)
    # # visualization_utils.plot_county_data(counties[years == YEAR], X[years == YEAR, COLUMN_IDX], COLUMN_NAME, YEAR)


    # # TODO Impute missing X values. First compute the mean feature vector for every county; if no data exists for that county, replace it with mean feature for the year (across all counties)
    # county_avgs = dict()
    # for county in county_set:
    #     county_avgs[county] = np.nanmean(X[counties == county, :], axis=0)
    # year_avgs = dict()
    # for year in range(min_year, max_year+1):
    #     year_avgs[year] = np.nanmean(X[years == year, :], axis=0)

    # nan_entry_indices = np.argwhere(np.isnan(X))
    # for nan_entry in nan_entry_indices:
    #     print("Nan entry", nan_entry)
    #     row = nan_entry[0]
    #     col = nan_entry[1]
    #     county = counties[row]
    #     year = years[row]
    #     if not np.isnan(county_avgs[county][col]):
    #         print("Substituting with county avg", county_avgs[county][col])
    #         X[row, col] = county_avgs[county][col]
    #     elif not np.isnan(year_avgs[year][col]):
    #         print("No values for feature", col, "in county", county, "in any year. Replacing with this year's avg across ALL COUNTIES")
    #         X[row, col] = year_avgs[year][col]
    #     else:
    #         X[row, col] = 0

    # Check for nan in X
    # print("Nan here")
    # print(np.argwhere(np.isnan(X)))
    # exit(0)

    # Sanity check: plot map of feature after standardizing
    # visualization_utils.plot_county_data(counties[years == YEAR], X[years == YEAR, COLUMN_IDX], COLUMN_NAME + "_std", YEAR)


    # Compute average of each output
    Y_mean = np.nanmean(Y[known_years], axis=0, keepdims=True)
    Y_std = np.nanstd(Y[known_years], axis=0, keepdims=True)
    args.means = torch.tensor(Y_mean, device=device)
    args.stds = torch.tensor(Y_std, device=device)
    print("Y (output) means", args.means, "stds", args.stds)

    # Create dictionaries mapping from (county + year) to features/labels
    X_dict = {}
    Y_dict = {}
    year_dict = {}
    for county in county_set:
        X_dict[county] = {}
        Y_dict[county] = {}
    for year in range(min_year, max_year+1):
        year_dict[year] = []
    for county, year, x, y in zip(counties, years, X, Y):
        X_dict[county][year] = x
        Y_dict[county][year] = y
        year_dict[year].append(x)

    # Compute average features for each year (to use if there's missing data)
    year_avg = {}
    for year in range(min_year, max_year+1):
        year_dict[year] = np.array(year_dict[year])
        year_avg[year] = np.mean(year_dict[year], axis=0)

    #### Adjacency ####
    Data = pickle.load(open(args.us_adj_file, 'rb'))
    adj = Data['adj']
    ctid_to_order = Data['ctid_to_order']
    crop_data = pickle.load(open(args.crop_id_to_fid, 'rb'))
    id_to_fid = crop_data['fid_dict']
    order_map = {}
    indices = []  # Indices of counties present in dataset (e.g. excluding Alaska/Hawaii) 
    for i, loc in enumerate(county_set):
        order_map[loc] = i
        if args.data_dir == "soybean_data_full.npz":
            fid = id_to_fid[loc]
        else:
            fid = loc
        indices.append(ctid_to_order[fid])
    sub_adj = adj[indices][:, indices]

    for year in range(min_year, max_year+1):
        for i, county in enumerate(county_set):
            # If data isn't present, fill in county features with the average feature values
            # of neighbors, or if no neighbors have data, replace them with the average
            # features of all US counties for the year.
            if year not in X_dict[county]:
                print("No data for county", county, "year", year)
                X_nbs = []
                Y_nbs = []
                for j, nb in enumerate(county_set):
                    if sub_adj[i, j] == 1 and year in X_dict[nb]:
                        print("--> Adding data from neighboring county", nb)
                        X_nbs.append(X_dict[nb][year])
                        Y_nbs.append(Y_dict[nb][year])
                if len(X_nbs):
                    X_dict[county][year] = np.mean(X_nbs, axis=0)
                    Y_dict[county][year] = np.mean(Y_nbs, axis=0)
                else:
                    print("--> Not even neighboring counties have data :O")
                    X_dict[county][year] = year_avg[year]
                    Y_dict[county][year] = avg_Y[year]
    #########

    l = args.length
    X_seqs = []
    Y_seqs = []
    county_seqs = []
    year_seqs = []
    #print(min_year, max_year) # 1980, 2018
    #print(county_set) # n_counties

    # For each county/year, retrieve features from all of the last 5 (or "length") years,
    # including the current year
    mode = 2
    for county in county_set:
        for year in range(min_year+l-1, max_year+1):
            #if year not in X_dict[county]: continue
            if mode == 1:
                is_continuous = True
                for i in range(l):
                    if year-i not in X_dict[county]:
                        is_continuous = False
                        break
                if not is_continuous: continue
                
                seq_X = []
                seq_Y = []
                for i in range(l):
                    seq_year = year - (l-i-1)
                    if seq_year in X_dict[county]:
                        seq_X.append(X_dict[county][seq_year]) # 431
                        seq_Y.append(Y_dict[county][seq_year]) # 1
                    else:
                        seq_X.append(year_avg[seq_year])
                        seq_Y.append(avg_Y[seq_year])
            elif mode == 2:
                seq_X = []
                seq_Y = []
                for i in range(l):
                    seq_year = year - (l-i-1)
                    seq_X.append(X_dict[county][seq_year]) # 431
                    seq_Y.append(Y_dict[county][seq_year]) # 1

            seq_X, seq_Y = np.array(seq_X), np.array(seq_Y)

            # If all yield variables for the last year are NaN, remove this sequence
            if np.all(np.isnan(seq_Y[-1, :])):
                continue

            X_seqs.append(seq_X)
            Y_seqs.append(seq_Y)
            county_seqs.append(county)
            year_seqs.append(year)

    X_seqs = np.array(X_seqs)
    Y_seqs = np.array(Y_seqs)
    county_seqs = np.array(county_seqs)
    year_seqs = np.array(year_seqs)
    print("X_seqs", X_seqs.shape)
    print("Y_seqs", Y_seqs.shape)
    print("county_seqs", county_seqs.shape)
    print("year_seqs", year_seqs.shape)

    #print(X_seqs.shape, Y_seqs.shape) # (26371, 5, 431) (26371, 5)
    X_train, X_val, X_test = [], [], []
    Y_train, Y_val, Y_test = [], [], []
    counties_train, counties_val, counties_test = [], [], []
    years_train, years_val, years_test = [], [], []
    for x_seq, y_seq, county, year in zip(X_seqs, Y_seqs, county_seqs, year_seqs):
        if year == args.test_year:
            X_test.append(x_seq)
            Y_test.append(y_seq)
            counties_test.append(county)
            years_test.append(year)
        elif year == args.test_year - 1:
            X_val.append(x_seq)
            Y_val.append(y_seq)
            counties_val.append(county)
            years_val.append(year)
        else:
            X_train.append(x_seq)
            Y_train.append(y_seq)
            counties_train.append(county)
            years_train.append(year)
    X_train, X_val, X_test = np.array(X_train), np.array(X_val), np.array(X_test)
    Y_train, Y_val, Y_test = np.array(Y_train), np.array(Y_val), np.array(Y_test)
    counties_train, counties_val, counties_test = np.array(counties_train), np.array(counties_val), np.array(counties_test)
    years_train, years_val, years_test = np.array(years_train), np.array(years_val), np.array(years_test)


    '''loc1 = 300
    o1 = order_map[loc1]
    fid1 = id_to_fid[loc1]
    print("###", fid1)
    for year in range(2010, 2015):
        if year in Y_dict[loc1] and year+1 in Y_dict[loc1]:
            print("{:.2f}".format(Y_dict[loc1][year+1] - Y_dict[loc1][year]), end=',')
        else:
            print("-1.00", end=',')
    print()
    for i, loc2 in enumerate(county_set):
        if loc2 == loc1: continue
        o2 = order_map[loc2]
        if sub_adj[o1, o2] == 1:
            fid2 = id_to_fid[loc2]
            print("###", fid2)
            for year in range(2010, 2015):
                if year in Y_dict[loc2] and year+1 in Y_dict[loc2]:
                    print("{:.2f}".format(Y_dict[loc2][year+1] - Y_dict[loc2][year]), end=',')
                else:
                    print("-1.00", end=",")
            print()
    exit()'''
    ######

    return X_train, Y_train, counties_train, years_train, X_val, Y_val, counties_val, years_val, X_test, Y_test, counties_test, years_test  # X_train, Y_train, X_val, Y_val, X_test, Y_test
