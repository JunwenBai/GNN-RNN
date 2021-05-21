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
    X = data[:, 3:]
    counties = data[:, 0].astype(int)
    years = data[:, 1]
    Y = data[:, 2]

    known_years = data[:, 1] <= (args.test_year-1)
    X_mean = np.mean(X[known_years], axis=0, keepdims=True)
    X_std = np.std(X[known_years], axis=0, keepdims=True)
    X = (X - X_mean) / (X_std + 1e-10)

    min_year = int(min(years))
    max_year = int(max(years))
    avg_Y = {}
    avg_Y_lst = []
    for year in range(min_year, max_year+1):
        avg_Y[year] = np.mean(Y[years == year])
        avg_Y_lst.append(avg_Y[year])
    '''mean_Y = np.mean(avg_Y_lst)
    std_Y = np.std(avg_Y_lst)
    for year in range(min_year, max_year+1):
        avg_Y[year] = (avg_Y[year] - mean_Y) / std_Y'''
    avg_Y[min_year-1] = avg_Y[min_year]

    Ybar = []
    for year in years:
        Ybar.append(avg_Y[year-1])
    Ybar = np.array(Ybar).reshape(-1, 1)
    X = np.concatenate((X, Ybar), axis=1)

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
    year_avg = {}
    for year in range(min_year, max_year+1):
        year_dict[year] = np.array(year_dict[year])
        year_avg[year] = np.mean(year_dict[year], axis=0)

    #l = args.length
    #print(min_year, max_year) # 1980, 2018
    #print(county_set) # n_counties

    avail_dict = {}
    for year in range(min_year, max_year+1):
        avail_dict[year] = []
        for j, county in enumerate(county_set):
            if year in X_dict[county]:
                avail_dict[year].append(j)

    # Adjacency
    Data = pickle.load(open(args.us_adj_file, 'rb'))
    adj = Data['adj']
    ctid_to_order = Data['ctid_to_order']
    crop_data = pickle.load(open(args.crop_id_to_fid, 'rb'))
    id_to_fid = crop_data['fid_dict']
    order_map = {}
    indices = []
    for i, loc in enumerate(county_set):
        order_map[loc] = i
        fid = id_to_fid[loc]
        indices.append(ctid_to_order[fid])
    sub_adj = adj[indices][:, indices]

    for year in range(min_year, max_year+1):
        for i, county in enumerate(county_set):
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

    return X_dict, Y_dict, avail_dict, sub_adj, order_map, min_year, max_year, county_set
