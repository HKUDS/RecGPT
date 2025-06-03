import argparse
import pickle
import os
import fsspec
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import GPT2Config
from modeling_gpt2_rec import GPT2Model

from model import GPT4RecModel
from utils.data import *
from ae_trainer import *
from utils.Trie import *

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--test_batch_size", type=int, default=32)
	parser.add_argument("--tf_layer", type=int, default=3)
	parser.add_argument("--gpu", type=str, default='0')
	parser.add_argument("--ckpt_path", type=str, default='./ckpt/recgpt_layer_3_weight.pt')
	args = parser.parse_args()

	device = torch.device("cuda:" + args.gpu)

	args.n_embed = 768
	args.initializer_range = 0.02

	data_path = './data/eval'

	with open(data_path + '/eval.pkl', "rb") as f:
		test_dict = pickle.load(f)

	print("Records Number: " +  str(len(test_dict)))

	with open(data_path + '/all_item_text_embeddings_eval.npy', "rb") as f:
		item_text_embeddings = np.load(f)
		item_text_embeddings = torch.tensor(item_text_embeddings).float()

	args.num_items = item_text_embeddings.shape[0]

	ae = AutoEncoder(seq_len=4)
	ae.load_state_dict(torch.load('./vae_ckpt/vae_len4_fsq88865_ep90.pt', map_location=device))
	ae.setItemEmbeddings(item_text_embeddings)
	ae.to(device)

	ae.eval()
	with open(data_path + '/token_id_list_eval.pkl', 'rb') as f:
		token_id_list = pickle.load(f)

	with open(data_path + '/eval_id_map.pkl', 'rb') as f:
		small_id_map = pickle.load(f)

	gpt2 = GPT2Model(GPT2Config(n_layer=args.tf_layer))

	print(gpt2)

	test_data = GPT2RecBatchEvalAuxData(test_dict, args.num_items)
	test_data.setTokenIdList(token_id_list)
	test_data.setIdMap(small_id_map)

	model = GPT4RecModel(args, gpt2)

	model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
	print("load successfully!")

	model.setItemEmbed(item_text_embeddings)
	model.setAutoEncoder(ae)

	test_data_loader = DataLoader(test_data, args.test_batch_size, shuffle=False)

	model.to(device)
	model.eval()
	
	eval_rec_loss = 0.0

	print("all : " + str(len(test_data_loader)))

	with torch.no_grad():

		for batch_seq, batch_labels, batch_aux in test_data_loader:
			batch_seq = batch_seq.to(device)
			batch_labels = batch_labels.to(device)
			batch_aux = batch_aux.to(device)

			loss_rec = model(batch_seq, batch_labels, batch_aux, ae)

			eval_rec_loss += loss_rec.item()
	
	print(eval_rec_loss / len(test_data_loader))
	
if __name__ == "__main__":
	main()    