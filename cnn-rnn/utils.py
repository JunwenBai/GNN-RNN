import os
import pickle
import torch
import numpy as np

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

def get_X_Y(data, args):
    # TODO Delete this!!!!
    if args.data_dir.endswith(".npz"):
        counties = data[:, 0].astype(int)
        years = data[:, 1].astype(int)
        Y = data[:, 2:3]
        X = data[:, 3:]
    else:
        years_all = data[:, 1].astype(int)
        X_all = data[:, 8:]  # 2+args.num_outputs:]
        data = data[(years_all <= 2017) & (~np.isnan(X_all).any(axis=1))]
        counties = data[:, 0].astype(int)
        years = data[:, 1].astype(int)
        Y = data[:, 2:8]  #2:2+args.num_outputs]
        X = data[:, 8:]  # 2+args.num_outputs:]

    print("Get X Y")
    print("X shape", X.shape)
    print("Y shape", Y.shape)
    print("Years", data[:, 1])

    # Check for nan in X
    # print("Nan here")
    # print(np.argwhere(np.isnan(X)))
    # exit(0)


    # For standardization purposes, only consider train years (e.g. the years before args.test_year)
    known_years = data[:, 1] <= (args.test_year-1)

    # Standardize each feature (column of X)
    X_mean = np.mean(X[known_years], axis=0, keepdims=True)
    X_std = np.std(X[known_years], axis=0, keepdims=True)
    X = (X - X_mean) / (X_std + 1e-10)

    # Compute average yield of each year (to detect underlying yearly trends)
    min_year = int(min(years))
    max_year = int(max(years))
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
    Ybar = np.array(Ybar)  #.reshape(-1, 1)
    X = np.concatenate((X, Ybar), axis=1)

    # Create dictionaries mapping from (county + year) to features/labels
    X_dict = {}
    Y_dict = {}
    county_set = sorted(list(set(counties)))
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
        if args.data_dir.endswith(".npz"):
            fid = id_to_fid[loc]
        else:
            fid = loc
        indices.append(ctid_to_order[fid])
    sub_adj = adj[indices][:, indices]

    for year in range(min_year, max_year+1):
        for i, county in enumerate(county_set):
            # If data isn't present, fill in county features with the average feature values
            # of neighbors, or if no neighbors have data, replace them with the average
            # features of all US counties.
            if year not in X_dict[county]:
                X_nbs = []
                Y_nbs = []
                for j, nb in enumerate(county_set):
                    if sub_adj[i, j] == 1 and year in X_dict[nb]:
                        X_nbs.append(X_dict[nb][year])
                        Y_nbs.append(Y_dict[nb][year])
                if len(X_nbs):
                    X_dict[county][year] = np.mean(X_nbs, axis=0)
                    Y_dict[county][year] = np.mean(Y_nbs, axis=0)
                else:
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
