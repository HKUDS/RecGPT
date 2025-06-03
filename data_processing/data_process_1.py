import numpy as np
import pandas as pd
import re
import os
from datetime import datetime
import pickle
from scipy.sparse import coo_matrix
import json
from datasets import load_dataset

dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Magazine_Subscriptions", trust_remote_code=True) 

records_dict = {}
counter = 0
for i in range(len(dataset["full"])):
    counter += 1
    if dataset["full"][i]['user_id'] not in records_dict:
        records_dict[dataset["full"][i]['user_id']] = []
    records_dict[dataset["full"][i]['user_id']].append([dataset["full"][i]['parent_asin'], dataset["full"][i]['timestamp']])
print(counter)
print(len(records_dict))

np_records_dict = {}
counter = 0
for uid in records_dict:
    np_seq = np.array(records_dict[uid])
    if np.shape(np_seq)[0] >= 5:
        np_records_dict[uid] = np.array(records_dict[uid])
        counter += 1

print(counter)
print(len(np_records_dict))

sorted_seq_dict = {}

for uid in np_records_dict:
    np_seq = np_records_dict[uid]
    sorted_indices = np.argsort(np_seq[:, 1])
    sorted_seq = np_seq[sorted_indices]
    sorted_seq_dict[uid] = list(sorted_seq[:, 0])

print(len(sorted_seq_dict))

iid_map = {}
counter = 0
for uid in sorted_seq_dict:
    item_list = sorted_seq_dict[uid]
    for iid in item_list:
        if iid not in iid_map:
            iid_map[iid] = counter
            counter += 1

print(counter)
print(len(iid_map))

counter = 0
seqs_dict = {}
counter_u = 0
for uid in sorted_seq_dict:
    seqs_dict[counter_u] = list()
    for iid in sorted_seq_dict[uid]:
        seqs_dict[counter_u].append(iid_map[iid])
        counter += 1
    counter_u += 1

print(len(seqs_dict))
print(counter_u)
print(counter)

dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Magazine_Subscriptions", split="full", trust_remote_code=True) 

item_dict = {}
counter = 0
for i in range(len(dataset)):
    counter += 1
    if dataset[i]['parent_asin'] not in item_dict:
        item_dict[dataset[i]['parent_asin']] = {}
    item_dict[dataset[i]['parent_asin']]['main_category'] = dataset[i]['main_category']
    item_dict[dataset[i]['parent_asin']]['title'] = dataset[i]['title']
    item_dict[dataset[i]['parent_asin']]['description'] = dataset[i]['description']
    item_dict[dataset[i]['parent_asin']]['categories']= dataset[i]['categories']
    item_dict[dataset[i]['parent_asin']]['details'] = dataset[i]['details']
print(counter)
print(len(item_dict))

id_item_dict = {}
for iid in item_dict:
    if iid in iid_map:
        id_item_dict[iid_map[iid]] = item_dict[iid]
print(len(id_item_dict))

f = open('./../data/pre_train/magazine/item_text_dict.pkl', 'wb')
pickle.dump(id_item_dict, f)
f.close()

train_seqs_dict = {}
test_seqs_dict = {}
for uid in seqs_dict:
    random_float = np.random.random()
    if random_float < 0.1:
        test_seqs_dict[uid] = seqs_dict[uid]
    else:
        train_seqs_dict[uid] = seqs_dict[uid]

print(len(test_seqs_dict))
print(len(train_seqs_dict))

f = open('./../data/pre_train/magazine/train.pkl', 'wb')
pickle.dump(train_seqs_dict, f)
f.close()

f = open('./../data/pre_train/magazine/test.pkl', 'wb')
pickle.dump(test_seqs_dict, f)
f.close()

counter = 0
max_length = 0
min_length = 99999
for uid in seqs_dict:
    counter += len(seqs_dict[uid])
    max_length = max(len(seqs_dict[uid]), max_length)
    min_length = min(len(seqs_dict[uid]), min_length)
print(counter)
print(counter / len(seqs_dict))
print(max_length)
print(min_length)