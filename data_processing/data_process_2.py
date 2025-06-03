import numpy as np
import pandas as pd
import re
import os
from datetime import datetime
import pickle
from scipy.sparse import coo_matrix
import json
from datasets import load_dataset

all_seqs_dict = {}
all_item_id_map_beauty = {}
all_item_id_map_books = {}
all_item_id_map_clothing = {}
all_item_id_map_electronics = {}
all_item_id_map_health = {}
all_item_id_map_kindle = {}
all_item_id_map_kitchen = {}
all_item_id_map_magazine = {}
all_item_id_map_movies = {}
all_item_id_map_phones = {}
all_item_id_map_sports = {}
record_counter = 0
item_counter = 0

# # 16w
# with open('./../../data/full/baby/train.pkl', 'rb') as f:
#     train_seqs_dict_baby = pickle.load(f)
# print(len(train_seqs_dict_baby))

temp_sum = 0
# 0.1w
with open('./../../data/full/beauty/train.pkl', 'rb') as f:
    train_seqs_dict_beauty = pickle.load(f)
print(len(train_seqs_dict_beauty))
temp_counter = 0
temp_item_set = set()
for uid in train_seqs_dict_beauty.keys():
    temp_counter += len(train_seqs_dict_beauty[uid])
    for iid in train_seqs_dict_beauty[uid]:
        temp_item_set.add(iid)
print(len(temp_item_set))
print(temp_counter)
print(temp_counter / len(train_seqs_dict_beauty))
temp_sum += temp_counter
print("-------")

# 109w
with open('./../../data/full/books/train.pkl', 'rb') as f:
    train_seqs_dict_books = pickle.load(f)
print(len(train_seqs_dict_books))
temp_counter = 0
temp_item_set = set()
for uid in train_seqs_dict_books.keys():
    temp_counter += len(train_seqs_dict_books[uid])
    for iid in train_seqs_dict_books[uid]:
        temp_item_set.add(iid)
print(len(temp_item_set))
print(temp_counter)
print(temp_counter / len(train_seqs_dict_books))
temp_sum += temp_counter
print("-------")

# 308w
with open('./../../data/full/clothing/train.pkl', 'rb') as f:
    train_seqs_dict_clothing = pickle.load(f)
print(len(train_seqs_dict_clothing))
temp_counter = 0
temp_item_set = set()
for uid in train_seqs_dict_clothing.keys():
    temp_counter += len(train_seqs_dict_clothing[uid])
    for iid in train_seqs_dict_clothing[uid]:
        temp_item_set.add(iid)
print(len(temp_item_set))
print(temp_counter)
print(temp_counter / len(train_seqs_dict_clothing))
temp_sum += temp_counter
print("-------")

# 169w
with open('./../../data/full/electronics/train.pkl', 'rb') as f:
    train_seqs_dict_electronics = pickle.load(f)
print(len(train_seqs_dict_electronics))
temp_counter = 0
temp_item_set = set()
for uid in train_seqs_dict_electronics.keys():
    temp_counter += len(train_seqs_dict_electronics[uid])
    for iid in train_seqs_dict_electronics[uid]:
        temp_item_set.add(iid)
print(len(temp_item_set))
print(temp_counter)
print(temp_counter / len(train_seqs_dict_electronics))
temp_sum += temp_counter
print("-------")

# # 1w
# with open('./../../data/full/fashion/train.pkl', 'rb') as f:
#     train_seqs_dict_fashion = pickle.load(f)
# print(len(train_seqs_dict_fashion))

# # 10w
# with open('./../../data/full/games/train.pkl', 'rb') as f:
#     train_seqs_dict_games = pickle.load(f)
# print(len(train_seqs_dict_games))

# 82w
with open('./../../data/full/health/train.pkl', 'rb') as f:
    train_seqs_dict_health = pickle.load(f)
print(len(train_seqs_dict_health))
temp_counter = 0
temp_item_set = set()
for uid in train_seqs_dict_health.keys():
    temp_counter += len(train_seqs_dict_health[uid])
    for iid in train_seqs_dict_health[uid]:
        temp_item_set.add(iid)
print(len(temp_item_set))
print(temp_counter)
print(temp_counter / len(train_seqs_dict_health))
temp_sum += temp_counter
print("-------")

# # 7w
# with open('./../../data/full/instruments/train.pkl', 'rb') as f:
#     train_seqs_dict_instruments = pickle.load(f)
# print(len(train_seqs_dict_instruments))

# 90w
with open('./../../data/full/kindle/train.pkl', 'rb') as f:
    train_seqs_dict_kindle = pickle.load(f)
print(len(train_seqs_dict_kindle))
temp_counter = 0
temp_item_set = set()
for uid in train_seqs_dict_kindle.keys():
    temp_counter += len(train_seqs_dict_kindle[uid])
    for iid in train_seqs_dict_kindle[uid]:
        temp_item_set.add(iid)
print(len(temp_item_set))
print(temp_counter)
print(temp_counter / len(train_seqs_dict_kindle))
temp_sum += temp_counter
print("-------")

# 309w
with open('./../../data/full/kitchen/train.pkl', 'rb') as f:
    train_seqs_dict_kitchen = pickle.load(f)
print(len(train_seqs_dict_kitchen))
temp_counter = 0
temp_item_set = set()
for uid in train_seqs_dict_kitchen.keys():
    temp_counter += len(train_seqs_dict_kitchen[uid])
    for iid in train_seqs_dict_kitchen[uid]:
        temp_item_set.add(iid)
print(len(temp_item_set))
print(temp_counter)
print(temp_counter / len(train_seqs_dict_kitchen))
temp_sum += temp_counter
print("-------")

# 0.03w
with open('./../../data/full/magazine/train.pkl', 'rb') as f:
    train_seqs_dict_magazine = pickle.load(f)
print(len(train_seqs_dict_magazine))
temp_counter = 0
temp_item_set = set()
for uid in train_seqs_dict_magazine.keys():
    temp_counter += len(train_seqs_dict_magazine[uid])
    for iid in train_seqs_dict_magazine[uid]:
        temp_item_set.add(iid)
print(len(temp_item_set))
print(temp_counter)
print(temp_counter / len(train_seqs_dict_magazine))
temp_sum += temp_counter
print("-------")

# 65w
with open('./../../data/full/movies/train.pkl', 'rb') as f:
    train_seqs_dict_movies = pickle.load(f)
print(len(train_seqs_dict_movies))
temp_counter = 0
temp_item_set = set()
for uid in train_seqs_dict_movies.keys():
    temp_counter += len(train_seqs_dict_movies[uid])
    for iid in train_seqs_dict_movies[uid]:
        temp_item_set.add(iid)
print(len(temp_item_set))
print(temp_counter)
print(temp_counter / len(train_seqs_dict_movies))
temp_sum += temp_counter
print("-------")

# # 30w
# with open('./../../data/full/office/train.pkl', 'rb') as f:
#     train_seqs_dict_office = pickle.load(f)
# print(len(train_seqs_dict_office))

# 54w
with open('./../../data/full/phones/train.pkl', 'rb') as f:
    train_seqs_dict_phones = pickle.load(f)
print(len(train_seqs_dict_phones))
temp_counter = 0
temp_item_set = set()
for uid in train_seqs_dict_phones.keys():
    temp_counter += len(train_seqs_dict_phones[uid])
    for iid in train_seqs_dict_phones[uid]:
        temp_item_set.add(iid)
print(len(temp_item_set))
print(temp_counter)
print(temp_counter / len(train_seqs_dict_phones))
temp_sum += temp_counter
print("-------")

# # 9w
# with open('./../../data/full/scientific/train.pkl', 'rb') as f:
#     train_seqs_dict_scientific = pickle.load(f)
# print(len(train_seqs_dict_scientific))

# 56w
with open('./../../data/full/sports/train.pkl', 'rb') as f:
    train_seqs_dict_sports = pickle.load(f)
print(len(train_seqs_dict_sports))
temp_counter = 0
temp_item_set = set()
for uid in train_seqs_dict_sports.keys():
    temp_counter += len(train_seqs_dict_sports[uid])
    for iid in train_seqs_dict_sports[uid]:
        temp_item_set.add(iid)
print(len(temp_item_set))
print(temp_counter)
print(temp_counter / len(train_seqs_dict_sports))
temp_sum += temp_counter
print("-------")

print(temp_sum)
exit()

item_id_set = set()
for _ in train_seqs_dict_beauty:
    for iid in train_seqs_dict_beauty[_]:
        if iid not in item_id_set:
            item_id_set.add(iid)
            all_item_id_map_beauty[iid] = item_counter
            item_counter += 1

for _ in train_seqs_dict_beauty:
    all_seqs_dict[record_counter] = []
    for iid in train_seqs_dict_beauty[_]:
        all_seqs_dict[record_counter].append(all_item_id_map_beauty[iid])
    record_counter += 1

item_id_set = set()
for _ in train_seqs_dict_books:
    for iid in train_seqs_dict_books[_]:
        if iid not in item_id_set:
            item_id_set.add(iid)
            all_item_id_map_books[iid] = item_counter
            item_counter += 1

for _ in train_seqs_dict_books:
    all_seqs_dict[record_counter] = []
    for iid in train_seqs_dict_books[_]:
        all_seqs_dict[record_counter].append(all_item_id_map_books[iid])
    record_counter += 1

item_id_set = set()
for _ in train_seqs_dict_clothing:
    for iid in train_seqs_dict_clothing[_]:
        if iid not in item_id_set:
            item_id_set.add(iid)
            all_item_id_map_clothing[iid] = item_counter
            item_counter += 1

for _ in train_seqs_dict_clothing:
    all_seqs_dict[record_counter] = []
    for iid in train_seqs_dict_clothing[_]:
        all_seqs_dict[record_counter].append(all_item_id_map_clothing[iid])
    record_counter += 1

item_id_set = set()
for _ in train_seqs_dict_electronics:
    for iid in train_seqs_dict_electronics[_]:
        if iid not in item_id_set:
            item_id_set.add(iid)
            all_item_id_map_electronics[iid] = item_counter
            item_counter += 1

for _ in train_seqs_dict_electronics:
    all_seqs_dict[record_counter] = []
    for iid in train_seqs_dict_electronics[_]:
        all_seqs_dict[record_counter].append(all_item_id_map_electronics[iid])
    record_counter += 1

item_id_set = set()
for _ in train_seqs_dict_health:
    for iid in train_seqs_dict_health[_]:
        if iid not in item_id_set:
            item_id_set.add(iid)
            all_item_id_map_health[iid] = item_counter
            item_counter += 1

for _ in train_seqs_dict_health:
    all_seqs_dict[record_counter] = []
    for iid in train_seqs_dict_health[_]:
        all_seqs_dict[record_counter].append(all_item_id_map_health[iid])
    record_counter += 1

item_id_set = set()
for _ in train_seqs_dict_kindle:
    for iid in train_seqs_dict_kindle[_]:
        if iid not in item_id_set:
            item_id_set.add(iid)
            all_item_id_map_kindle[iid] = item_counter
            item_counter += 1

for _ in train_seqs_dict_kindle:
    all_seqs_dict[record_counter] = []
    for iid in train_seqs_dict_kindle[_]:
        all_seqs_dict[record_counter].append(all_item_id_map_kindle[iid])
    record_counter += 1

item_id_set = set()
for _ in train_seqs_dict_kitchen:
    for iid in train_seqs_dict_kitchen[_]:
        if iid not in item_id_set:
            item_id_set.add(iid)
            all_item_id_map_kitchen[iid] = item_counter
            item_counter += 1

for _ in train_seqs_dict_kitchen:
    all_seqs_dict[record_counter] = []
    for iid in train_seqs_dict_kitchen[_]:
        all_seqs_dict[record_counter].append(all_item_id_map_kitchen[iid])
    record_counter += 1

item_id_set = set()
for _ in train_seqs_dict_magazine:
    for iid in train_seqs_dict_magazine[_]:
        if iid not in item_id_set:
            item_id_set.add(iid)
            all_item_id_map_magazine[iid] = item_counter
            item_counter += 1

for _ in train_seqs_dict_magazine:
    all_seqs_dict[record_counter] = []
    for iid in train_seqs_dict_magazine[_]:
        all_seqs_dict[record_counter].append(all_item_id_map_magazine[iid])
    record_counter += 1

item_id_set = set()
for _ in train_seqs_dict_movies:
    for iid in train_seqs_dict_movies[_]:
        if iid not in item_id_set:
            item_id_set.add(iid)
            all_item_id_map_movies[iid] = item_counter
            item_counter += 1

for _ in train_seqs_dict_movies:
    all_seqs_dict[record_counter] = []
    for iid in train_seqs_dict_movies[_]:
        all_seqs_dict[record_counter].append(all_item_id_map_movies[iid])
    record_counter += 1

item_id_set = set()
for _ in train_seqs_dict_phones:
    for iid in train_seqs_dict_phones[_]:
        if iid not in item_id_set:
            item_id_set.add(iid)
            all_item_id_map_phones[iid] = item_counter
            item_counter += 1

for _ in train_seqs_dict_phones:
    all_seqs_dict[record_counter] = []
    for iid in train_seqs_dict_phones[_]:
        all_seqs_dict[record_counter].append(all_item_id_map_phones[iid])
    record_counter += 1

item_id_set = set()
for _ in train_seqs_dict_sports:
    for iid in train_seqs_dict_sports[_]:
        if iid not in item_id_set:
            item_id_set.add(iid)
            all_item_id_map_sports[iid] = item_counter
            item_counter += 1

for _ in train_seqs_dict_sports:
    all_seqs_dict[record_counter] = []
    for iid in train_seqs_dict_sports[_]:
        all_seqs_dict[record_counter].append(all_item_id_map_sports[iid])
    record_counter += 1

print(item_counter)
print(record_counter)

f = open('./../../data/full/beauty/all_item_id_map.pkl', 'wb')
pickle.dump(all_item_id_map_beauty, f)
f.close()

f = open('./../../data/full/books/all_item_id_map.pkl', 'wb')
pickle.dump(all_item_id_map_books, f)
f.close()

f = open('./../../data/full/clothing/all_item_id_map.pkl', 'wb')
pickle.dump(all_item_id_map_clothing, f)
f.close()

f = open('./../../data/full/electronics/all_item_id_map.pkl', 'wb')
pickle.dump(all_item_id_map_electronics, f)
f.close()

f = open('./../../data/full/health/all_item_id_map.pkl', 'wb')
pickle.dump(all_item_id_map_health, f)
f.close()

f = open('./../../data/full/kindle/all_item_id_map.pkl', 'wb')
pickle.dump(all_item_id_map_kindle, f)
f.close()

f = open('./../../data/full/kitchen/all_item_id_map.pkl', 'wb')
pickle.dump(all_item_id_map_kitchen, f)
f.close()

f = open('./../../data/full/magazine/all_item_id_map.pkl', 'wb')
pickle.dump(all_item_id_map_magazine, f)
f.close()

f = open('./../../data/full/movies/all_item_id_map.pkl', 'wb')
pickle.dump(all_item_id_map_movies, f)
f.close()

f = open('./../../data/full/phones/all_item_id_map.pkl', 'wb')
pickle.dump(all_item_id_map_phones, f)
f.close()

f = open('./../../data/full/sports/all_item_id_map.pkl', 'wb')
pickle.dump(all_item_id_map_sports, f)
f.close()



f = open('./../../data/full/train.pkl', 'wb')
pickle.dump(all_seqs_dict, f)
f.close()

# item: 15491643
# records: 12472073
