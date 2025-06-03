import argparse
import pickle
import os
import fsspec
import numpy as np
import torch

number_item = 0

with fsspec.open('./../data/full/beauty/item_text_embeddings.npy', "rb") as f:
	item_embed_beauty = np.load(f)
with fsspec.open('./../data/full/beauty/all_item_id_map.pkl', "rb") as f:
	all_item_id_map_beauty = pickle.load(f)
number_item += len(all_item_id_map_beauty)

with fsspec.open('./../data/full/books/item_text_embeddings.npy', "rb") as f:
	item_embed_books = np.load(f)
with fsspec.open('./../data/full/books/all_item_id_map.pkl', "rb") as f:
	all_item_id_map_books = pickle.load(f)
number_item += len(all_item_id_map_books)

with fsspec.open('./../data/full/clothing/item_text_embeddings.npy', "rb") as f:
	item_embed_clothing = np.load(f)
with fsspec.open('./../data/full/clothing/all_item_id_map.pkl', "rb") as f:
	all_item_id_map_clothing = pickle.load(f)
number_item += len(all_item_id_map_clothing)

with fsspec.open('./../data/full/electronics/item_text_embeddings.npy', "rb") as f:
	item_embed_electronics = np.load(f)
with fsspec.open('./../data/full/electronics/all_item_id_map.pkl', "rb") as f:
	all_item_id_map_electronics = pickle.load(f)
number_item += len(all_item_id_map_electronics)

with fsspec.open('./../data/full/health/item_text_embeddings.npy', "rb") as f:
	item_embed_health = np.load(f)
with fsspec.open('./../data/full/health/all_item_id_map.pkl', "rb") as f:
	all_item_id_map_health = pickle.load(f)
number_item += len(all_item_id_map_health)

with fsspec.open('./../data/full/kindle/item_text_embeddings.npy', "rb") as f:
	item_embed_kindle = np.load(f)
with fsspec.open('./../data/full/kindle/all_item_id_map.pkl', "rb") as f:
	all_item_id_map_kindle = pickle.load(f)
number_item += len(all_item_id_map_kindle)

with fsspec.open('./../data/full/kitchen/item_text_embeddings.npy', "rb") as f:
	item_embed_kitchen = np.load(f)
with fsspec.open('./../data/full/kitchen/all_item_id_map.pkl', "rb") as f:
	all_item_id_map_kitchen = pickle.load(f)
number_item += len(all_item_id_map_kitchen)

with fsspec.open('./../data/full/magazine/item_text_embeddings.npy', "rb") as f:
	item_embed_magazine = np.load(f)
with fsspec.open('./../data/full/magazine/all_item_id_map.pkl', "rb") as f:
	all_item_id_map_magazine = pickle.load(f)
number_item += len(all_item_id_map_magazine)

with fsspec.open('./../data/full/movies/item_text_embeddings.npy', "rb") as f:
	item_embed_movies = np.load(f)
with fsspec.open('./../data/full/movies/all_item_id_map.pkl', "rb") as f:
	all_item_id_map_movies = pickle.load(f)
number_item += len(all_item_id_map_movies)

with fsspec.open('./../data/full/phones/item_text_embeddings.npy', "rb") as f:
	item_embed_phones = np.load(f)
with fsspec.open('./../data/full/phones/all_item_id_map.pkl', "rb") as f:
	all_item_id_map_phones = pickle.load(f)
number_item += len(all_item_id_map_phones)

with fsspec.open('./../data/full/sports/item_text_embeddings.npy', "rb") as f:
	item_embed_sports = np.load(f)
with fsspec.open('./../data/full/sports/all_item_id_map.pkl', "rb") as f:
	all_item_id_map_sports = pickle.load(f)
number_item += len(all_item_id_map_sports)

all_array = np.zeros((number_item, 768))

keys = np.array(list(all_item_id_map_beauty.keys()))
values = np.array(list(all_item_id_map_beauty.values()))
all_array[values] = item_embed_beauty[keys]

keys = np.array(list(all_item_id_map_books.keys()))
values = np.array(list(all_item_id_map_books.values()))
all_array[values] = item_embed_books[keys]

keys = np.array(list(all_item_id_map_clothing.keys()))
values = np.array(list(all_item_id_map_clothing.values()))
all_array[values] = item_embed_clothing[keys]

keys = np.array(list(all_item_id_map_electronics.keys()))
values = np.array(list(all_item_id_map_electronics.values()))
all_array[values] = item_embed_electronics[keys]

keys = np.array(list(all_item_id_map_health.keys()))
values = np.array(list(all_item_id_map_health.values()))
all_array[values] = item_embed_health[keys]

keys = np.array(list(all_item_id_map_kindle.keys()))
values = np.array(list(all_item_id_map_kindle.values()))
all_array[values] = item_embed_kindle[keys]

keys = np.array(list(all_item_id_map_kitchen.keys()))
values = np.array(list(all_item_id_map_kitchen.values()))
all_array[values] = item_embed_kitchen[keys]

keys = np.array(list(all_item_id_map_magazine.keys()))
values = np.array(list(all_item_id_map_magazine.values()))
all_array[values] = item_embed_magazine[keys]

keys = np.array(list(all_item_id_map_movies.keys()))
values = np.array(list(all_item_id_map_movies.values()))
all_array[values] = item_embed_movies[keys]

keys = np.array(list(all_item_id_map_phones.keys()))
values = np.array(list(all_item_id_map_phones.values()))
all_array[values] = item_embed_phones[keys]

keys = np.array(list(all_item_id_map_sports.keys()))
values = np.array(list(all_item_id_map_sports.values()))
all_array[values] = item_embed_sports[keys]

item_text_embeddings = all_array

print(np.shape(item_text_embeddings)[0])

np.save('./../data/full/all_item_text_embeddings.npy', item_text_embeddings)
