import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import fsspec
import pickle
import argparse
import os
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torch.optim as optim

from utils.util import *
from utils.fsq import *
from utils.data import *
from decoder import *
from encoder import *

class AutoEncoder(nn.Module):
	def __init__(self, seq_len=8):
		super(AutoEncoder, self).__init__()

		self.encoder = Encoder(seq_len=seq_len)
		self.decoder = Decoder(seq_len=seq_len)
		self.quantizer = FSQ(dim=(768 // seq_len))

		self.seq_len = seq_len
		self.l1loss = nn.L1Loss()
		self.celoss = nn.CrossEntropyLoss()

	def setItemEmbeddings(self, item_embeddings):
		self.item_embeddings = item_embeddings

	def encode(self, batch_ids, device=None):
		if device != None:
			batch_embeds = self.item_embeddings[batch_ids].to(device)
		else:
			batch_embeds = self.item_embeddings[batch_ids]
		batch_seq_embeds = self.encoder(batch_embeds)
		quant_embeds, quant_indices = self.quantizer(batch_seq_embeds)

		return quant_embeds, quant_indices
	
	def encode_aux(self, batch_ids, num_item, device=None):
		embed_mask = (batch_ids > -1).long()
		embeds_ids = (batch_ids * embed_mask).clamp_(0, num_item-1).cpu()

		if device != None:
			batch_embeds = self.item_embeddings[embeds_ids].to(device)
		else:
			batch_embeds = self.item_embeddings[embeds_ids]

		batch_embeds = self.encoder.forward_aux(batch_embeds).detach()
		embed_mask = embed_mask.unsqueeze(-1).repeat(1, 1, 4)
		embed_mask = embed_mask.view(embed_mask.shape[0], embed_mask.shape[1] * embed_mask.shape[2])

		return batch_embeds, embed_mask.unsqueeze(-1)

	def decode(self, quant_embeds):
		return self.decoder(quant_embeds)
	
	def decode_pred(self, quant_indices):
		return self.decode(self.quantizer.indices_to_codes(quant_indices))
	
	def pred_rec(self, batch_ids):
		quant_embeds, quant_indices = self.encode(batch_ids)
		reconstructions = self.decode_pred(quant_indices)
		logits = torch.matmul(reconstructions, self.item_embeddings.transpose(0, 1))

		return logits

	def forward(self, batch_ids, device=None):
		quant_embeds, quant_indices = self.encode(batch_ids, device)
		dec = self.decode(quant_embeds)

		return dec

	def cal_loss(self, batch_ids, device=None):
		reconstructions = self.forward(batch_ids, device)
		if device != None:
			batch_embeds = self.item_embeddings[batch_ids].to(device)
		else:
			batch_embeds = self.item_embeddings[batch_ids]
		l1loss = self.l1loss(batch_embeds, reconstructions)

		return l1loss
	
	def cal_loss_rec(self, batch_ids):
		self.eval()
		quant_embeds, quant_indices = self.encode(batch_ids)
		self.train()
		reconstructions = self.decode_pred(quant_indices.detach())

		logits = torch.matmul(reconstructions, self.item_embeddings.transpose(0, 1))
		celoss = self.celoss(logits, batch_ids)

		return celoss


def main():
	device = torch.device("cuda:0")

	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--epoch", type=int, default=100)
	args = parser.parse_args()

	args.n_embed = 768

	with open('./data/pre_train/all_item_text_embeddings.npy', 'rb') as f:
		item_text_mebddings = torch.tensor(np.load(f)).float()

	args.num_items = item_text_mebddings.shape[0]

	print("item number: ", args.num_items)

	train_data = VAEData(args.num_items)

	model = AutoEncoder(seq_len=4)
	model.to(device)

	model.setItemEmbeddings(item_text_embeddings)

	train_data_loader = DataLoader(train_data, args.batch_size, shuffle=True)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	for epoch in range(args.epoch):
		model.train()
		train_loss = 0.0

		for batch_ids in train_data_loader:
			optimizer.zero_grad()

			loss = model.cal_loss(batch_ids, device)

			loss.backward()
			optimizer.step()

			train_loss += loss.item()

		if epoch % 10 == 0:
			torch.save(model.state_dict(), './vae_ckpt/vae_len4_fsq88865_ep' + str(epoch) + '.pt')

		print(f"Epoch {epoch} - Loss: {train_loss / len(train_data_loader)}")


if __name__ == "__main__":
	main()







	

