import argparse
import pickle
import os
import fsspec
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import GPT2Model, GPT2Config

from ae_trainer import *

def main():

	device = torch.device("cuda:0")

	with open('./../data/pre_train/all_item_text_embeddings.npy', "rb") as f:
		item_text_embeddings = np.load(f)
		item_text_embeddings = torch.tensor(item_text_embeddings).float()

	num_items = item_text_embeddings.shape[0]

	print("item number: ", num_items)

	vae_data = VAEData(num_items)
	vae_data_loader = DataLoader(vae_data, 4096, shuffle=False)

	ae = AutoEncoder(seq_len=4)
	ae.load_state_dict(torch.load('./../vae_ckpt/vae_len4_fsq88865_ep90.pt', map_location=device))
	ae.setItemEmbeddings(item_text_embeddings)
	ae.to(device)

	ae.eval()
	token_id_list = []
	with torch.no_grad():
		for batch_ids in vae_data_loader:
			_, quant_indices = ae.encode(batch_ids, device)
			token_id_list.extend(quant_indices.cpu().tolist()) 

	f = open('./../data/pre_train/token_id_list.pkl', 'wb')
	pickle.dump(token_id_list, f)
	f.close()

if __name__ == "__main__":
	main()