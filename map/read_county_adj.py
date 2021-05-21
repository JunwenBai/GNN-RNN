import pickle
import numpy as np

fopen = open("county_adjacency.txt", "r", encoding="ISO-8859-1")
ct_name_to_id = {}
ct_id_to_name = {}
st_name_to_id = {}
st_id_to_name = {}
st_of_ctid = {}
name_of_ctid = {}
neighbors = {}

def update_dict(name, idx):
    name = name.strip(" \"\t\n")
    name_lst = name.split(',')
    #print(name_lst)
    ct, st = name_lst[0].strip(), name_lst[1].strip()
    #print(st, ct)
    st_id = int(idx[:2])

    st_name_to_id[st] = st_id
    st_id_to_name[st_id] = st

    ct_name_to_id[name] = int(idx)
    ct_id_to_name[int(idx)] = name
    st_of_ctid[int(idx)] = st
    name_of_ctid[int(idx)] = ct

def update_neighbors(node, nb):
    if node not in neighbors:
        neighbors[node] = [nb]
    else:
        neighbors[node].append(nb)

node = -1
for line in fopen:
    county_lst = line.strip().split("\t")
    if len(county_lst) > 2:
        update_dict(county_lst[0], county_lst[1])
        update_dict(county_lst[2], county_lst[3])
        node = int(county_lst[1])
        update_neighbors(node, int(county_lst[3]))
    else:
        update_dict(county_lst[0], county_lst[1])
        update_neighbors(node, int(county_lst[1]))

def is_valid(ct_id):
    st = st_of_ctid[ct_id]
    if st == 'HI' or st == 'PR' or st == 'AS' or st == 'GU' or st == 'MP' or st == 'VI':
        return False
    return True

all_ct_ids = list(ct_id_to_name.keys())
ct_ids = []
for ct in all_ct_ids:
    if is_valid(ct):
        ct_ids.append(ct)
ct_ids = sorted(ct_ids)

ct_to_order = {}
order_to_ct = {}
for i, cid in enumerate(ct_ids):
    ct_to_order[cid] = i
    order_to_ct[i] = cid

cnt_valid = 0
for ct_id in neighbors.keys():
    if is_valid(ct_id):
        cnt_valid += 1

adj = np.zeros((cnt_valid, cnt_valid), dtype=int)
for ct_id in neighbors.keys():
    if not is_valid(ct_id):
        continue
    ct_order = ct_to_order[ct_id]
    adj[ct_order, ct_order] = 1
    for nb in neighbors[ct_id]:
        if not is_valid(nb):
            continue
        nb_order = ct_to_order[nb]
        adj[ct_order, nb_order] = 1
        adj[nb_order, ct_order] = 1

# TEST #

name1 = "Ballard County, KY"
name2 = "Alexander County, IL"
ctid1 = ct_name_to_id[name1]
ctid2 = ct_name_to_id[name2]
o1 = ct_to_order[ctid1]
o2 = ct_to_order[ctid2]
for i, v in enumerate(adj[o1]):
    if v == 1:
        ctid2 = order_to_ct[i]
        name2 = ct_id_to_name[ctid2]
        print(name2)
exit()
###



Data = {'adj': adj,
        'ct_name_to_id': ct_name_to_id,
        'ct_id_to_name': ct_id_to_name,
        'st_name_to_id': st_name_to_id,
        'st_id_to_name': st_id_to_name,
        'st_of_ctid': st_of_ctid,
        'name_of_ctid': name_of_ctid,
        'ctid_to_order': ct_to_order,
        'order_to_ctid': order_to_ct
}

pickle.dump(Data, open('us_adj.pkl', 'wb'))

