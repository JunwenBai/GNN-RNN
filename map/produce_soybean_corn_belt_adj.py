import numpy as np
import pickle

Data = pickle.load(open('us_adj.pkl', 'rb'))
adj = Data['adj']
ctid_to_order = Data['ctid_to_order']

soybean_data = pickle.load(open('soybean_fid_dict.pkl', 'rb'))
fid_dict = soybean_data['fid_dict']

fopen = open("Data_Soybean_full.csv", "r")
locs = set()
for i, line in enumerate(fopen):
    if i == 0: continue
    lst = line.split(",")
    loc_id = int(float(lst[0]))
    locs.add(loc_id)
locs = sorted(list(locs))
n_loc = len(locs)
print("n_loc:", n_loc)

order_map = {}
indices = []
for i, loc in enumerate(locs):
    order_map[loc] = i
    fid = fid_dict[loc]
    indices.append(ctid_to_order[fid])

print(adj.shape)
print(adj[indices][:, indices].shape)

sub_adj = adj[indices][:, indices]


