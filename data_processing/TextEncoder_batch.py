from transformers import AutoTokenizer, AutoModel

from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = SentenceTransformer('all-mpnet-base-v2')

with open('./../data/test/yelp/item_text_dict.pkl', 'rb') as f:
    info_dict = pickle.load(f)

print(len(info_dict))
print(max(info_dict.keys()))
print(min(info_dict.keys()))

embedding_dict = {}

embedding_list = None
iid_list = []
batch_list = []
counter = 0

temp_counter = 0
for iid in info_dict:
    counter += 1
    if counter % 100000 == 0:
        print(counter)
    iid_list.append(iid)
    batch_list.append(str(info_dict[iid]).replace('{','').replace('}',''))
    temp_counter += 1
    if temp_counter == 1024 or counter == len(info_dict):
        embedding_list = model.encode(batch_list)
        for _ in range(len(embedding_list)):
            embedding_dict[iid_list[_]] = embedding_list[_]
        iid_list = []
        batch_list = []
        temp_counter = 0

print(len(embedding_dict))
print(counter)

keys = np.array(list(embedding_dict.keys()))
values = np.array(list(embedding_dict.values()))

print(keys[:10])
print(np.mean(values[5]))
print(np.mean(values[10]))

indexed_array = np.zeros((np.max(keys) + 1, 768))
indexed_array[keys] = values

print(np.mean(indexed_array[keys[5]]))
print(np.mean(indexed_array[keys[10]]))

print(np.shape(indexed_array))

np.save('./../data/test/yelp/item_text_embeddings.npy', indexed_array)