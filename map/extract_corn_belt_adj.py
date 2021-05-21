import pickle
import numpy as np

fopen = open("us_county_abbrev.txt", "r")
st_to_abbv = {}
for line in fopen:
    lst = line.strip().split()
    st, abbv = " ".join(lst[:-1]), lst[-1]
    st_to_abbv[st] = abbv
fopen.close()

Data = pickle.load(open('us_adj.pkl', 'rb'))
adj = Data['adj']
ct_name_to_id = Data['ct_name_to_id']
ctid_to_order = Data['ctid_to_order']

def ct_dict(path):
    fopen = open(path, "r")
    crop_dict = {}
    for i, line in enumerate(fopen):
        if i == 0:
            continue
        lst = line.strip().split(',')
        crop_dict[lst[1]+" "+lst[0]] = int(lst[-1])
    fopen.close()
    return crop_dict

#corn_dict = ct_dict("Corn_Loc_ID.csv")
soybean_dict = ct_dict("Soybeans_Loc_ID.csv")

def trans(county):
    if county[:2] == "mc":
        return county[:2].capitalize()+county[2:].capitalize()
    if county == "lagrange":
        return "LaGrange"
    if county == "lake of the woods":
        return "Lake of the Woods"
    if county == "oglala lakota":
        return "Shannon"
    lst = county.strip().split()
    new_name = " ".join([n.capitalize() for n in lst])
    if "St " in new_name:
        new_name = new_name.replace("St ", "St. ")
    if "Ste " in new_name:
        new_name = new_name.replace("Ste ", "Ste. ")
    return new_name

fopen = open("Soybeans_Loc_ID.csv", "r")
fid_dict = {}
for i, line in enumerate(fopen):
    if i == 0:
        continue
    lst = line.strip().split(',')
    ct_name1 = lst[1]+" "+lst[0]
    ct_name2 = trans(lst[1])+" County, "+st_to_abbv[lst[0].upper()]
    '''if ct_name2 not in ct_name_to_id:
        print(ct_name1)
        print(ct_name2)'''

    id1 = soybean_dict[ct_name1]
    ct_id = ct_name_to_id[ct_name2]

    fid_dict[id1] = ct_id

Data = {
    'fid_dict': fid_dict
}

#print(fid_dict[150])
#print(fid_dict[200])

pickle.dump(Data, open('soybean_fid_dict.pkl', 'wb'))
